"""Main evaluation loop: load model, generate predictions, compute metrics."""

from __future__ import annotations

import json
import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import OpenXMP4VideoDataset
from ..models.adapters import IdentityAdapter, create_adapter
from ..models.base_autoencoder import create_autoencoder, encoder_config_from_args
from ..models.model import DiT
from ..training.diffusion import Diffusion, FlowMatching
from ..training.utils import (
    downsample_actions_temporal,
    downsample_sequence_temporal,
    mask_future_conditioning,
    resolve_adapter_ckpt,
    setup_pixel_decoder_for_val,
    load_frozen_adapter_weights,
    strip_state_dict_prefix,
)
from .fvd import compute_fvd
from .metrics import (
    compute_fid,
    compute_lpips_batch,
    compute_per_step_metrics,
    compute_psnr_ssim,
)

logger = logging.getLogger(__name__)

# Head architecture constants (same as train.py)
_DECODER_DIM = 2048
_DECODER_DEPTH = 2
_DECODER_HEADS = 16


def _load_dit_checkpoint(
    checkpoint_path: str,
    args,
    in_channels: int,
    device: torch.device,
    action_dim_override: int | None = None,
    tactile_dim_override: int | None = None,
) -> torch.nn.Module:
    """Load DiT model from checkpoint, returning EMA weights."""
    model = DiT(
        in_channels=in_channels,
        patch_size=args.patch_size,
        dim=args.model_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        action_dim=action_dim_override if action_dim_override is not None else args.action_dim,
        tactile_dim=tactile_dim_override if tactile_dim_override is not None else getattr(args, "tactile_dim", 0),
        max_frames=args.n_frames,
        action_dropout_prob=0.0,  # no dropout at eval
        wide_head=args.wide_head,
        decoder_dim=_DECODER_DIM,
        decoder_depth=_DECODER_DEPTH,
        decoder_heads=_DECODER_HEADS,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Prefer EMA weights
    if "ema" in ckpt:
        state_dict = strip_state_dict_prefix(ckpt["ema"])
        logger.info("Loading EMA weights from checkpoint")
    elif "model" in ckpt:
        state_dict = strip_state_dict_prefix(ckpt["model"])
        logger.info("Loading model weights from checkpoint (no EMA found)")
    else:
        state_dict = strip_state_dict_prefix(ckpt)
        logger.info("Loading raw state dict from checkpoint")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _decode_latents(
    latents: torch.Tensor,
    adapter,
    autoencoder,
    pixel_decoder,
    use_pixel_decoder: bool,
    is_identity: bool,
) -> torch.Tensor:
    """Decode latent representations to pixel space.

    Returns
    -------
    (B, T, H, W, 3) float tensor in [0, 1]
    """
    if use_pixel_decoder and pixel_decoder is not None:
        return pixel_decoder(latents)

    if not is_identity:
        latents = adapter.decode(latents)
    return autoencoder.decode(latents)


def _unpack_batch(batch):
    if len(batch) == 2:
        x, actions = batch
        return x, actions, None
    if len(batch) == 3:
        return batch
    raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")


def _compute_reconstruction_ceiling(
    autoencoder,
    adapter,
    pixel_decoder,
    use_pixel_decoder: bool,
    is_identity: bool,
    gt_pixels: np.ndarray,
    gt_tensor: torch.Tensor,
    device: torch.device,
    precision: torch.dtype,
    skip_frames: int,
) -> Dict[str, float]:
    """Encode + decode GT through the full pipeline (no diffusion) to measure decoder quality."""
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision):
        z = autoencoder.encode(gt_tensor)
        if not is_identity:
            z_adapted = adapter.encode(z)
            if isinstance(z_adapted, tuple):
                z_adapted = z_adapted[0]
        else:
            z_adapted = z
        recon = _decode_latents(z_adapted, adapter, autoencoder, pixel_decoder, use_pixel_decoder, is_identity)

    recon_np = recon.float().clamp(0, 1).cpu().numpy()
    metrics = compute_psnr_ssim(recon_np, gt_pixels, skip_frames=skip_frames)
    return {
        "recon_psnr": float(np.mean(metrics["psnr"])),
        "recon_ssim": float(np.mean(metrics["ssim"])),
    }


def evaluate_model(args) -> Dict:
    """Run full evaluation: generate predictions and compute all metrics.

    Parameters
    ----------
    args : argparse.Namespace with model/data/eval configuration

    Returns
    -------
    dict with all computed metrics and per-step breakdowns
    """
    assert torch.cuda.is_available(), "CUDA device required for evaluation"
    device = torch.device("cuda")
    precision = torch.bfloat16 if getattr(args, "precision", "bfloat16") == "bfloat16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_compute = set(getattr(args, "metrics", "psnr,ssim,lpips,fvd,fid,recon").split(","))
    logger.info("Metrics to compute: %s", metrics_to_compute)

    # ── Load models ──────────────────────────────────────────────────────
    logger.info("Loading autoencoder: %s", args.encoder_type)
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)

    # Set up pixel decoder
    use_pixel_decoder = getattr(args, "use_pixel_decoder", False)
    pixel_decoder = None
    if use_pixel_decoder:
        # Reuse the same setup logic as training
        # Temporarily set the attr that setup_pixel_decoder_for_val checks
        args.use_pixel_decoder_for_val = True
        pixel_decoder = setup_pixel_decoder_for_val(args, adapter_ckpt_data, device)
        use_pixel_decoder = pixel_decoder is not None

    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, pixel_decoder, adapter_ckpt_data)
    adapter.eval()
    adapter.requires_grad_(False)
    if pixel_decoder is not None:
        pixel_decoder.eval()
        pixel_decoder.requires_grad_(False)

    is_identity = isinstance(adapter, IdentityAdapter)
    in_channels = adapter.latent_dim

    # Match training: temporal encoders (VJEPA, Qwen) concatenate actions pairwise
    temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)
    if getattr(args, "use_tactile", False) and getattr(args, "tactile_dim", 0) <= 0:
        raise ValueError("--tactile_dim must be > 0 when --use_tactile True")
    effective_action_dim = args.action_dim * temporal_ds
    effective_tactile_dim = getattr(args, "tactile_dim", 0) * temporal_ds

    logger.info("Loading DiT checkpoint: %s", args.checkpoint_path)
    model = _load_dit_checkpoint(
        args.checkpoint_path,
        args,
        in_channels,
        device,
        action_dim_override=effective_action_dim,
        tactile_dim_override=effective_tactile_dim,
    )

    # ── Diffusion ────────────────────────────────────────────────────────
    shift = 1.0
    if args.encoder_type != "vae" and getattr(args, "use_shift", True):
        m = (256 / args.patch_size ** 2) * in_channels
        shift = (m / 4096) ** 0.5
        logger.info("Time shift: m=%.0f  shift=%.4f", m, shift)

    if args.objective == "flow_matching":
        diffusion = FlowMatching(
            timesteps=getattr(args, "timesteps", 1000),
            sampling_timesteps=getattr(args, "sampling_timesteps", 10),
            time_dist_type=getattr(args, "time_dist_type", "uniform"),
            time_dist_shift=shift,
            device=device,
        ).to(device)
    else:
        diffusion = Diffusion(
            timesteps=getattr(args, "timesteps", 1000),
            sampling_timesteps=getattr(args, "sampling_timesteps", 10),
            time_dist_shift=shift,
            device=device,
        ).to(device)

    # ── Dataset ──────────────────────────────────────────────────────────
    logger.info("Loading test dataset...")
    test_dataset = OpenXMP4VideoDataset(args, split="test")
    def _seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    loader_generator = torch.Generator()
    eval_seed = getattr(args, "eval_seed", 42)
    loader_generator.manual_seed(eval_seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "num_workers", 4),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )

    num_eval_samples = getattr(args, "num_eval_samples", 2048)
    context_frames = args.num_history + 1
    if args.n_frames <= context_frames:
        raise ValueError(
            f"n_frames={args.n_frames} must be greater than "
            f"num_history + 1={context_frames} to evaluate future predictions"
        )
    if context_frames % temporal_ds != 0:
        raise ValueError(
            f"num_history + 1={context_frames} must be divisible by "
            f"temporal_downsample_factor={temporal_ds}"
        )
    n_ctx = context_frames // temporal_ds
    import ipdb; ipdb.set_trace()
    print(n_ctx)
    print(max(args.num_history, 1))
    cfg = getattr(args, "cfg", 1.0)

    # ── Generation loop (pixel metrics) ─────────────────────────────────
    pixel_metrics = {"psnr", "ssim", "lpips", "fid", "fvd", "recon", "pck"}
    need_pixel_pass = bool(metrics_to_compute & pixel_metrics)

    all_gen_pixels = []
    all_gt_pixels = []
    all_psnr_per_step = []
    all_ssim_per_step = []
    all_lpips_per_step = []
    pck_thresholds = tuple(getattr(args, "pck_thresholds", [5, 10, 20, 40]))
    all_pck_per_step = {k: [] for k in pck_thresholds}
    all_pck_coverage = []
    recon_metrics = None
    recon_computed = False
    samples_processed = 0

    if need_pixel_pass:
        logger.info(
            "Starting pixel evaluation: %d samples, batch_size=%d, n_frames=%d, num_history=%d",
            num_eval_samples, args.batch_size, args.n_frames, args.num_history,
        )

        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if samples_processed >= num_eval_samples:
                break

            x, actions, tactile = _unpack_batch(batch)
            x = x.to(device)
            actions = actions.to(device)
            if tactile is not None:
                tactile = tactile.to(device)
            B = x.shape[0]

            # Keep raw GT pixels for metric computation (before any encoding)
            gt_pixels = x.float().clamp(0, 1).cpu().numpy()

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision):
                # Encode GT
                z = autoencoder.encode(x)
                if not is_identity:
                    z_adapted = adapter.encode(z)
                    if isinstance(z_adapted, tuple):
                        z_adapted = z_adapted[0]
                else:
                    z_adapted = z

                if temporal_ds > 1:
                    actions = downsample_actions_temporal(actions, temporal_ds)
                    if tactile is not None:
                        tactile = downsample_sequence_temporal(tactile, temporal_ds)
                tactile = mask_future_conditioning(tactile, n_ctx)

                # Generate
                samples_latent = diffusion.generate(
                    model,
                    z_adapted,
                    actions,
                    n_context_frames=n_ctx,
                    n_frames=z_adapted.shape[1],
                    window_len=getattr(args, "window_len", None),
                    horizon=getattr(args, "horizon", 1),
                    cfg=cfg,
                    tactile=tactile,
                )

                # Decode generated latents to pixels
                gen_decoded = _decode_latents(
                    samples_latent, adapter, autoencoder, pixel_decoder, use_pixel_decoder, is_identity
                )

            gen_pixels = gen_decoded.float().clamp(0, 1).cpu().numpy()

            # Compute reconstruction ceiling once
            if "recon" in metrics_to_compute and not recon_computed:
                recon_metrics = _compute_reconstruction_ceiling(
                    autoencoder, adapter, pixel_decoder, use_pixel_decoder, is_identity,
                    gt_pixels, x, device, precision, skip_frames=context_frames,
                )
                recon_computed = True
                logger.info(
                    "Reconstruction ceiling: PSNR=%.2f, SSIM=%.4f",
                    recon_metrics["recon_psnr"], recon_metrics["recon_ssim"],
                )

            # Per-frame metrics (compare gen vs raw GT pixels)
            skip = context_frames
            if "psnr" in metrics_to_compute or "ssim" in metrics_to_compute:
                ps = compute_psnr_ssim(gen_pixels, gt_pixels, skip_frames=skip)
                all_psnr_per_step.append(ps["psnr"])
                all_ssim_per_step.append(ps["ssim"])

            if "lpips" in metrics_to_compute:
                lp = compute_lpips_batch(gen_pixels, gt_pixels, device, skip_frames=skip)
                all_lpips_per_step.append(lp)

            # PCK via CoTracker point tracking
            if "pck" in metrics_to_compute:
                from .pck import compute_pck

                pck_result = compute_pck(
                    gen_pixels, gt_pixels, device, skip_frames=skip,
                    thresholds=pck_thresholds,
                    grid_size=getattr(args, "pck_grid_size", 16),
                    point_mode=getattr(args, "pck_point_mode", "grid"),
                )
                for k in pck_thresholds:
                    key = f"pck@{k}_per_step"
                    if key in pck_result:
                        all_pck_per_step[k].append(pck_result[key])
                if "coverage_per_step" in pck_result:
                    all_pck_coverage.append(pck_result["coverage_per_step"])

            # Accumulate frames for FID/FVD (skip context frames)
            if "fid" in metrics_to_compute or "fvd" in metrics_to_compute:
                all_gen_pixels.append(gen_pixels[:, skip:])
                all_gt_pixels.append(gt_pixels[:, skip:])

            samples_processed += B
    else:
        logger.info("Skipping pixel metric generation (not requested)")

    # ── Aggregate results ────────────────────────────────────────────────
    results: Dict = {"model_preset": getattr(args, "model_preset", "custom")}

    # Per-step metrics (average across batches per step)
    T_eval = args.n_frames - context_frames
    if all_psnr_per_step:
        psnr_arr = np.array(all_psnr_per_step)  # (num_batches, T_eval)
        ssim_arr = np.array(all_ssim_per_step)
        results["psnr_per_step"] = np.mean(psnr_arr, axis=0).tolist()
        results["ssim_per_step"] = np.mean(ssim_arr, axis=0).tolist()
        results["psnr_mean"] = float(np.mean(psnr_arr))
        results["ssim_mean"] = float(np.mean(ssim_arr))
        results["psnr_std"] = float(np.std(np.mean(psnr_arr, axis=1)))
        results["ssim_std"] = float(np.std(np.mean(ssim_arr, axis=1)))
        logger.info("PSNR: %.2f ± %.2f", results["psnr_mean"], results["psnr_std"])
        logger.info("SSIM: %.4f ± %.4f", results["ssim_mean"], results["ssim_std"])

    if all_lpips_per_step:
        lpips_arr = np.array(all_lpips_per_step)
        results["lpips_per_step"] = np.mean(lpips_arr, axis=0).tolist()
        results["lpips_mean"] = float(np.mean(lpips_arr))
        results["lpips_std"] = float(np.std(np.mean(lpips_arr, axis=1)))
        logger.info("LPIPS: %.4f ± %.4f", results["lpips_mean"], results["lpips_std"])

    # PCK metrics
    if any(all_pck_per_step[k] for k in pck_thresholds):
        for k in pck_thresholds:
            if not all_pck_per_step[k]:
                continue
            arr = np.array(all_pck_per_step[k])  # (num_batches, T_eval)
            results[f"pck@{k}_per_step"] = np.mean(arr, axis=0).tolist()
            results[f"pck@{k}_mean"] = float(np.mean(arr))
            results[f"pck@{k}_std"] = float(np.std(np.mean(arr, axis=1)))
            logger.info("PCK@%d: %.4f ± %.4f", k, results[f"pck@{k}_mean"], results[f"pck@{k}_std"])
        if all_pck_coverage:
            coverage_arr = np.array(all_pck_coverage)
            results["pck_coverage_per_step"] = np.mean(coverage_arr, axis=0).tolist()
            results["pck_coverage_mean"] = float(np.mean(coverage_arr))
            logger.info("PCK coverage: %.4f", results["pck_coverage_mean"])

    # Distribution metrics
    if all_gen_pixels and "fid" in metrics_to_compute:
        gen_all = np.concatenate(all_gen_pixels, axis=0)  # (N, T, H, W, 3)
        gt_all = np.concatenate(all_gt_pixels, axis=0)
        # Flatten time dim for FID: (N*T, H, W, 3)
        gen_flat = gen_all.reshape(-1, *gen_all.shape[2:])
        gt_flat = gt_all.reshape(-1, *gt_all.shape[2:])
        results["fid"] = compute_fid(gen_flat, gt_flat, device)
        logger.info("FID: %.2f", results["fid"])

    if all_gen_pixels and "fvd" in metrics_to_compute:
        gen_all = np.concatenate(all_gen_pixels, axis=0)
        gt_all = np.concatenate(all_gt_pixels, axis=0)
        results["fvd"] = compute_fvd(gen_all, gt_all, device)
        logger.info("FVD: %.2f", results["fvd"])

    if recon_metrics is not None:
        results.update(recon_metrics)

    # ── Controllability (separate pass, latent-only) ─────────────────────
    if "controllability" in metrics_to_compute:
        from .controllability import compute_controllability

        ctrl_optimizer = getattr(args, "ctrl_optimizer", "cem")
        search_dims = tuple(getattr(args, "search_dims", [0, 1, 2, 3, 4, 5, 6]))

        ctrl_kwargs = {"search_dims": search_dims}
        if ctrl_optimizer == "cem":
            ctrl_kwargs.update({
                "n_candidates": getattr(args, "cem_candidates", 256),
                "n_elite": getattr(args, "cem_elite", 25),
                "n_iterations": getattr(args, "cem_iterations", 5),
                "max_batch_size": getattr(args, "ctrl_batch_size", 64),
            })
        elif ctrl_optimizer == "gradient":
            ctrl_kwargs.update({
                "n_steps": getattr(args, "grad_steps", 50),
                "lr": getattr(args, "grad_lr", 0.01),
            })
        elif ctrl_optimizer == "grid":
            ctrl_kwargs.update({
                "n_points_per_dim": getattr(args, "grid_points_per_dim", 30),
                "max_batch_size": getattr(args, "ctrl_batch_size", 64),
            })

        ctrl_kwargs["action_dim"] = args.action_dim

        ctrl_results = compute_controllability(
            diffusion, model, autoencoder, adapter, is_identity,
            test_loader, device, precision,
            num_eval_samples=num_eval_samples,
            optimizer=ctrl_optimizer,
            save_path=output_dir / f"metrics-{args.ctrl_optimizer}.json",
            **ctrl_kwargs,
        )
        results.update(ctrl_results)

    # ── Save results ─────────────────────────────────────────────────────
    results_path = output_dir / f"metrics-{args.ctrl_optimizer}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved metrics to %s", results_path)

    # ── Horizon plots ────────────────────────────────────────────────────
    _plot_horizon_curves(results, output_dir, T_eval)

    return results


def _plot_horizon_curves(results: Dict, output_dir: Path, T_eval: int) -> None:
    """Generate and save horizon curve plots for per-step metrics."""
    steps = list(range(1, T_eval + 1))

    # Row 1: standard pixel metrics
    pixel_configs = [
        ("psnr_per_step", "PSNR (dB)"),
        ("ssim_per_step", "SSIM"),
        ("lpips_per_step", "LPIPS"),
    ]
    # Row 2: PCK metrics (only if present)
    pck_configs = [
        (key, key.replace("_per_step", "").upper())
        for key in sorted(results.keys())
        if key.startswith("pck@") and key.endswith("_per_step")
    ]

    has_pixel = any(k in results for k, _ in pixel_configs)
    has_pck = bool(pck_configs)

    nrows = (1 if has_pixel else 0) + (1 if has_pck else 0)
    if nrows == 0:
        return

    ncols = max(len(pixel_configs) if has_pixel else 0, len(pck_configs) if has_pck else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    row = 0
    if has_pixel:
        for col, (key, ylabel) in enumerate(pixel_configs):
            ax = axes[row, col]
            if key not in results:
                ax.set_visible(False)
                continue
            values = results[key]
            ax.plot(steps[: len(values)], values, "o-", markersize=4)
            ax.set_xlabel("Prediction Step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs Prediction Horizon")
            ax.grid(True, alpha=0.3)
        # Hide extra columns
        for col in range(len(pixel_configs), ncols):
            axes[row, col].set_visible(False)
        row += 1

    if has_pck:
        for col, (key, ylabel) in enumerate(pck_configs):
            ax = axes[row, col]
            values = results[key]
            ax.plot(steps[: len(values)], values, "s-", markersize=4, color="tab:green")
            ax.set_xlabel("Prediction Step")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} vs Prediction Horizon")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        for col in range(len(pck_configs), ncols):
            axes[row, col].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / "horizon_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved horizon curves to %s", plot_path)
    plt.close(fig)
