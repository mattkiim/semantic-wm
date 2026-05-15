import torch
import torch.nn.functional as F
import numpy as np
import imageio
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ..models.adapters import IdentityAdapter
from .utils import (
    downsample_actions_temporal,
    downsample_sequence_temporal,
    mask_future_conditioning,
)


def validate_step(
    model,
    ema,
    autoencoder,
    adapter,
    diffusion,
    val_iter,
    val_loader,
    device,
    precision,
    args,
    total_samples_seen,
    checkpoint_dir,
    pixel_decoder=None,
):
    """
    Perform a validation step.

    Args:
        model: The main model
        ema: EMA model
        autoencoder: Autoencoder (VAE/RAE/ScaleRAE) for encoding/decoding
        adapter: Adapter for latent projection (frozen)
        diffusion: Diffusion model
        val_iter: Validation data iterator
        val_loader: Validation data loader
        device: Device to run on
        precision: Precision for autocast
        args: Arguments containing validation settings
        total_samples_seen: Total samples seen during training
        checkpoint_dir: Directory to save outputs

    Returns:
        Updated val_iter
    """
    import einops
    num_views = getattr(args, "num_views", 1)

    model.eval()
    with torch.no_grad():
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)
        val_x, val_actions, val_tactile = _unpack_batch(batch)

        val_x = val_x.to(device)
        with torch.autocast(device_type="cuda", dtype=precision):
            if num_views > 1:
                # val_x: (B, V, T, H, W, C) -> encode each view independently
                B_mv, V = val_x.shape[:2]
                val_x_flat = einops.rearrange(val_x, "b v t h w c -> (b v) t h w c")
                val_latent = autoencoder.encode(val_x_flat)
                is_identity = isinstance(adapter, IdentityAdapter)
                if not is_identity:
                    val_latent_adapted = adapter.encode(val_latent)
                    if isinstance(val_latent_adapted, tuple):
                        val_latent_adapted = val_latent_adapted[0]
                else:
                    val_latent_adapted = val_latent
                val_latent_adapted = einops.rearrange(
                    val_latent_adapted, "(b v) t h w c -> b t h (v w) c", v=V
                )
                # For GT metrics, concatenate views along width
                val_x = einops.rearrange(val_x, "b v t h w c -> b t h (v w) c")
            else:
                val_latent = autoencoder.encode(val_x)
                # Apply adapter encoding (identity adapter is a no-op)
                is_identity = isinstance(adapter, IdentityAdapter)
                if not is_identity:
                    val_latent_adapted = adapter.encode(val_latent)
                    if isinstance(val_latent_adapted, tuple):
                        val_latent_adapted = val_latent_adapted[0]
                else:
                    val_latent_adapted = val_latent
            val_actions = val_actions.to(device)
            if val_tactile is not None:
                val_tactile = val_tactile.to(device)
            ema.eval()

            # Handle temporal downsampling (e.g. Qwen video, V-JEPA 2.1)
            temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)
            context_frames = args.num_history + 1
            if context_frames % temporal_ds != 0:
                raise ValueError(
                    f"num_history + 1={context_frames} must be divisible by "
                    f"temporal_downsample_factor={temporal_ds}"
                )
            if temporal_ds > 1:
                val_actions = downsample_actions_temporal(val_actions, temporal_ds)
                if val_tactile is not None:
                    val_tactile = downsample_sequence_temporal(val_tactile, temporal_ds)
            n_ctx = context_frames // temporal_ds
            effective_skip = n_ctx
            val_tactile = mask_future_conditioning(val_tactile, n_ctx)

            samples_latent = diffusion.generate(
                ema,
                val_latent_adapted,
                val_actions,
                n_context_frames=n_ctx,
                n_frames=val_latent_adapted.shape[1],
                window_len=args.window_len,
                horizon=args.horizon,
                cfg=args.cfg,
                tactile=val_tactile,
            )
            # Calculate MSE in latent space (before decoding)
            mse = F.mse_loss(samples_latent, val_latent_adapted).detach().cpu()

            # Decode: adapter.decode → autoencoder.decode
            # Encoders without a decoder (Qwen, V-JEPA 2.1) must use the pixel decoder.
            has_encoder_decoder = getattr(autoencoder, "has_decoder", True)
            use_pixel_dec = (
                (getattr(args, "use_pixel_decoder_for_val", False) or not has_encoder_decoder)
                and pixel_decoder is not None
            )
            if num_views > 1:
                # Reverse multi-view concatenation for decoding
                # samples_latent: (B, T, h, V*w, c) -> (B*V, T, h, w, c)
                samples_latent_dec = einops.rearrange(
                    samples_latent, "b t h (v w) c -> (b v) t h w c", v=num_views
                )
            else:
                samples_latent_dec = samples_latent

            gt_pixels = None
            if use_pixel_dec:
                samples = pixel_decoder(samples_latent_dec)
                # Decode GT latent through pixel decoder for a proper RGB side-by-side.
                # val_x may be embeddings (precomputed encoder), not raw RGB frames.
                gt_latent_dec = einops.rearrange(
                    val_latent_adapted, "b t h (v w) c -> (b v) t h w c", v=num_views
                ) if num_views > 1 else val_latent_adapted
                gt_pixels = pixel_decoder(gt_latent_dec)
                if num_views > 1:
                    gt_pixels = einops.rearrange(
                        gt_pixels, "(b v) t h w c -> b t h (v w) c", v=num_views
                    )
            elif has_encoder_decoder:
                if not is_identity:
                    samples_high = adapter.decode(samples_latent_dec)
                else:
                    samples_high = samples_latent_dec
                samples = autoencoder.decode(samples_high)
            else:
                # No decoder available — skip visual decoding, log only latent MSE
                wandb.log({"val/mse_latent": float(mse.item())}, step=total_samples_seen)
                print(
                    f"Validation at {total_samples_seen} samples: MSE (latent)={mse:.6f} "
                    "(no decoder available for visual metrics)"
                )
                return val_iter

            # Re-concatenate multi-view decoded samples along width for metrics
            if num_views > 1:
                samples = einops.rearrange(
                    samples, "(b v) t h w c -> b t h (v w) c", v=num_views
                )

    # Use pixel-decoder GT when val_x is embeddings (precomputed encoder),
    # otherwise use the raw input frames directly.
    if gt_pixels is not None:
        gt_np = gt_pixels.float().clamp(0, 1).cpu().numpy()
    else:
        gt_np = val_x.float().clamp(0, 1).cpu().numpy()

    # Log MSE to WandB
    wandb.log({"val/mse_latent": float(mse.item())}, step=total_samples_seen)

    # Resize decoded samples to match GT resolution if needed (pixel decoder
    # output resolution may differ from the original input resolution).
    samples_for_vis = samples.float().clamp(0, 1)
    gt_h, gt_w = gt_np.shape[2], gt_np.shape[3]
    sam_h, sam_w = samples_for_vis.shape[2], samples_for_vis.shape[3]
    if (sam_h, sam_w) != (gt_h, gt_w):
        B_s, T_s = samples_for_vis.shape[:2]
        # (B, T, H, W, C) -> (B*T, C, H, W) for interpolate
        samples_for_vis = samples_for_vis.reshape(B_s * T_s, sam_h, sam_w, -1).permute(0, 3, 1, 2)
        samples_for_vis = F.interpolate(samples_for_vis, size=(gt_h, gt_w), mode="bilinear", align_corners=False)
        samples_for_vis = samples_for_vis.permute(0, 2, 3, 1).reshape(B_s, T_s, gt_h, gt_w, -1)
    samples_np = samples_for_vis.cpu().numpy()

    # Handle temporal dimension mismatch: if the encoder temporally downsampled
    # the latents, the decoded samples may have fewer frames than the GT.
    if samples_np.shape[1] < gt_np.shape[1]:
        # Subsample GT to match decoded temporal resolution
        indices = np.linspace(0, gt_np.shape[1] - 1, samples_np.shape[1]).astype(int)
        gt_np = gt_np[:, indices]

    # Create side-by-side MP4: [actual GT | generated samples]
    sample_str = f"{total_samples_seen:012d}"
    video_path = checkpoint_dir / f"gen_samples_{sample_str}.mp4"

    sidebyside = np.concatenate(
        [gt_np[0], samples_np[0]], axis=1
    )  # (T, H, W*2, C)
    sidebyside_uint8 = (sidebyside * 255).astype(np.uint8)
    imageio.mimsave(str(video_path), sidebyside_uint8, fps=2)

    # Log side-by-side MP4 to WandB (after file is created)
    wandb.log({"val/generation": wandb.Video(str(video_path))}, step=total_samples_seen)

    # Calculate PSNR and SSIM for generated frames, skipping history (context) frames.
    # The first num_history frames are zero-noised copies of the input, so comparing
    # them against GT mostly measures autoencoder fidelity, not the world model.
    pixel_dec_temporal_up = use_pixel_dec and getattr(pixel_decoder, "temporal_upsample", False)
    skip = (args.num_history + 1) if pixel_dec_temporal_up else effective_skip
    metrics = calculate_image_metrics(
        samples_np, gt_np, skip_frames=skip
    )

    # Log metrics to WandB
    wandb.log(
        {
            "val/psnr": metrics["psnr"],
            "val/ssim": metrics["ssim"],
        },
        step=total_samples_seen,
    )
    print(
        f"Validation at {total_samples_seen} samples: MSE (latent)={mse:.6f}, PSNR={metrics['psnr']:.2f}, SSIM={metrics['ssim']:.4f}"
    )

    return val_iter


def _unpack_batch(batch):
    if len(batch) == 2:
        val_x, val_actions = batch
        return val_x, val_actions, None
    if len(batch) == 3:
        return batch
    raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")


def validate_spill(
    model,
    ema,
    autoencoder,
    adapter,
    diffusion,
    spill_loader,
    device,
    precision,
    args,
    total_samples_seen,
    checkpoint_dir,
    pixel_decoder=None,
):
    """Run validation over all spill windows and log aggregate metrics."""
    from ..models.adapters import IdentityAdapter

    model.eval()
    ema.eval()
    all_psnr, all_ssim, all_mse = [], [], []

    temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)
    context_frames = args.num_history + 1
    if context_frames % temporal_ds != 0:
        raise ValueError(
            f"num_history + 1={context_frames} must be divisible by "
            f"temporal_downsample_factor={temporal_ds}"
        )
    n_ctx = context_frames // temporal_ds
    effective_skip = n_ctx
    is_identity = isinstance(adapter, IdentityAdapter)
    has_encoder_decoder = getattr(autoencoder, "has_decoder", True)
    use_pixel_dec = (
        (getattr(args, "use_pixel_decoder_for_val", False) or not has_encoder_decoder)
        and pixel_decoder is not None
    )

    with torch.no_grad():
        for batch in spill_loader:
            val_x, val_actions, val_tactile = _unpack_batch(batch)
            val_x = val_x.to(device)
            val_actions = val_actions.to(device)
            if val_tactile is not None:
                val_tactile = val_tactile.to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                val_latent = autoencoder.encode(val_x)
                if not is_identity:
                    val_latent_adapted = adapter.encode(val_latent)
                    if isinstance(val_latent_adapted, tuple):
                        val_latent_adapted = val_latent_adapted[0]
                else:
                    val_latent_adapted = val_latent

                if temporal_ds > 1:
                    from .utils import downsample_actions_temporal, downsample_sequence_temporal
                    val_actions = downsample_actions_temporal(val_actions, temporal_ds)
                    if val_tactile is not None:
                        val_tactile = downsample_sequence_temporal(val_tactile, temporal_ds)
                val_tactile = mask_future_conditioning(val_tactile, n_ctx)

                samples_latent = diffusion.generate(
                    ema, val_latent_adapted, val_actions,
                    n_context_frames=n_ctx,
                    n_frames=val_latent_adapted.shape[1],
                    window_len=args.window_len,
                    horizon=args.horizon,
                    cfg=args.cfg,
                    tactile=val_tactile,
                )
                mse = F.mse_loss(samples_latent, val_latent_adapted).item()
                all_mse.append(mse)

                if use_pixel_dec:
                    samples = pixel_decoder(samples_latent)
                    gt_pixels = pixel_decoder(val_latent_adapted)
                    gt_np = gt_pixels.float().clamp(0, 1).cpu().numpy()
                elif has_encoder_decoder:
                    if not is_identity:
                        samples = autoencoder.decode(adapter.decode(samples_latent))
                    else:
                        samples = autoencoder.decode(samples_latent)
                    gt_np = val_x.float().clamp(0, 1).cpu().numpy()
                else:
                    continue

                samples_np = samples.float().clamp(0, 1).cpu().numpy()
                if samples_np.shape[1] < gt_np.shape[1]:
                    indices = np.linspace(0, gt_np.shape[1] - 1, samples_np.shape[1]).astype(int)
                    gt_np = gt_np[:, indices]

                metrics = calculate_image_metrics(samples_np, gt_np, skip_frames=effective_skip)
                all_psnr.append(metrics["psnr"])
                all_ssim.append(metrics["ssim"])

    if all_psnr:
        wandb.log(
            {
                "val_spill/psnr": float(np.mean(all_psnr)),
                "val_spill/ssim": float(np.mean(all_ssim)),
                "val_spill/mse_latent": float(np.mean(all_mse)),
            },
            step=total_samples_seen,
        )
        print(
            f"Spill validation at {total_samples_seen} samples: "
            f"MSE={np.mean(all_mse):.6f}, PSNR={np.mean(all_psnr):.2f}, SSIM={np.mean(all_ssim):.4f} "
            f"({len(all_psnr)} spill windows)"
        )


def calculate_image_metrics(generated_frames, gt_frames, skip_frames=0):
    """
    Calculate PSNR and SSIM metrics for generated frames.

    Args:
        generated_frames: Generated frames as numpy array (B, T, H, W, C) or (T, H, W, C) in range [0, 1]
        gt_frames: Ground truth frames as numpy array (B, T, H, W, C) or (T, H, W, C) in range [0, 1]
        skip_frames: Number of leading frames to skip (e.g. history/context frames that
                     are copied verbatim and would produce Inf PSNR).

    Returns:
        Dictionary containing:
            - psnr: Mean PSNR across all frames and batch samples
            - ssim: Mean SSIM across all frames and batch samples
    """
    # Handle both batched (B, T, H, W, C) and single (T, H, W, C) inputs
    if generated_frames.ndim == 4:
        # Single sample, add batch dimension
        generated_frames = generated_frames[np.newaxis, ...]
        gt_frames = gt_frames[np.newaxis, ...]

    batch_size = generated_frames.shape[0]
    all_psnr_values = []
    all_ssim_values = []

    # Calculate metrics for each sample in batch
    for b in range(batch_size):
        # Convert to uint8 (0-255 range)
        generated_uint8 = (generated_frames[b] * 255).astype(np.uint8)
        gt_uint8 = (gt_frames[b] * 255).astype(np.uint8)

        # Skip leading history/context frames — they are zero-noised copies of the ground
        # truth, so MSE == 0 and PSNR == inf, which would skew the metric.
        generated_uint8 = generated_uint8[skip_frames:]
        gt_uint8 = gt_uint8[skip_frames:]

        # Calculate metrics for each frame
        for t in range(generated_uint8.shape[0]):
            gen_frame = generated_uint8[t]
            gt_frame = gt_uint8[t]

            # PSNR — guard against identical frames (MSE=0 → inf) with a large sentinel value
            mse = np.mean(
                (gt_frame.astype(np.float32) - gen_frame.astype(np.float32)) ** 2
            )
            if mse == 0:
                # Identical frames would produce infinite PSNR; use a large sentinel value
                psnr = 100.0
            else:
                psnr = peak_signal_noise_ratio(gt_frame, gen_frame, data_range=255)
            all_psnr_values.append(psnr)

            # SSIM (calculate for each channel if RGB, then average)
            if gen_frame.ndim == 3 and gen_frame.shape[2] == 3:
                # Convert to grayscale for SSIM calculation
                ssim = structural_similarity(
                    np.mean(gt_frame, axis=2),
                    np.mean(gen_frame, axis=2),
                    data_range=255,
                )
            else:
                ssim = structural_similarity(gt_frame, gen_frame, data_range=255)
            all_ssim_values.append(ssim)

    return {
        "psnr": float(np.mean(all_psnr_values)),
        "ssim": float(np.mean(all_ssim_values)),
    }
