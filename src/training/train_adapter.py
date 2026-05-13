"""Training script for adapter layers (S-VAE / PS-VAE / MLP).

The adapter learns to project high-dimensional RAE features (d_h) to a
compact latent space (d_l) and back, with optional KL regularisation.

Usage (from repo root)::

    python -m src.launch_adapter \\
        --encoder_type rae \\
        --adapter_type svae \\
        --adapter_latent_dim 96 \\
        --dataset_dir sample_data \\
        --subset_names bridge_v2
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

try:
    import lpips
except ImportError:
    lpips = None

try:
    from pytorch_msssim import ms_ssim as _ms_ssim_fn
except ImportError:
    _ms_ssim_fn = None

import wandb

from ..data.dataset import H5TrajectoryDataset, OpenXMP4VideoDataset
from ..models.adapters import create_adapter, kl_divergence, adapter_config_from_args
from ..models.base_autoencoder import create_autoencoder, encoder_config_from_args
from ..models.discriminator import (
    PatchGANDiscriminator,
    hinge_loss_disc,
    hinge_loss_gen,
    adaptive_weight,
)
from ..models.pixel_decoder import create_pixel_decoder, pixel_decoder_config_from_args
from .adapter_validation import (
    semantic_reconstruction_loss,
    validate_adapter,
    log_reconstruction_video,
)
from .utils import (
    init_distributed,
    requires_grad,
    init_wandb,
    log_model_param_counts,
    load_adapter_training_checkpoint,
    save_adapter_checkpoint,
)


def _compute_ms_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MS-SSIM between two (B*T, C, H, W) tensors in [0, 1]."""
    if _ms_ssim_fn is not None:
        return _ms_ssim_fn(pred, target, data_range=1.0, size_average=True)
    # Fallback: plain SSIM via F — single-scale structural similarity
    # This is a lightweight substitute when pytorch_msssim is not installed.
    mu_x = F.avg_pool2d(pred, 11, 1, 5)
    mu_y = F.avg_pool2d(target, 11, 1, 5)
    sigma_x2 = F.avg_pool2d(pred * pred, 11, 1, 5) - mu_x * mu_x
    sigma_y2 = F.avg_pool2d(target * target, 11, 1, 5) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 11, 1, 5) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean()


def train_adapter(args) -> None:
    """Train an adapter layer on frozen autoencoder features."""
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    assert torch.cuda.is_available(), "CUDA required"

    precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    local_rank, rank, world_size, distributed = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    if args.checkpoint_dir is None:
        checkpoint_dir = (
            Path("outputs")
            / f"adapter_{args.adapter_type}_{args.adapter_hidden_dim}_{args.adapter_latent_dim}_{args.stage}"
        )
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── WandB ─────────────────────────────────────────────────────────────────
    if rank == 0:
        import datetime
        run_name = (
            f"Adapter-{args.adapter_type.upper()}_{args.encoder_type.upper()}_"
            f"d{args.adapter_latent_dim}_{args.stage}_B{args.batch_size}_"
            f"{datetime.datetime.now().strftime('%dT%H-%M-%S')}"
        )
        init_wandb(args, checkpoint_dir, run_name)

    # ── Data ─────────────────────────────────────────────────────────────────
    if getattr(args, "h5_train_path", None):
        train_dataset = H5TrajectoryDataset(args, split="train")
        val_dataset = H5TrajectoryDataset(args, split="test")
    else:
        train_dataset = OpenXMP4VideoDataset(args, split="train")
        val_dataset = OpenXMP4VideoDataset(args, split="test")

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed else None
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True,
    )
    train_iter = iter(train_loader)

    samples_per_epoch = len(train_dataset)
    steps_per_epoch = max(samples_per_epoch // (args.batch_size * world_size), 1)
    max_train_steps = args.num_epochs * steps_per_epoch

    # ── Models ────────────────────────────────────────────────────────────────
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()
    requires_grad(autoencoder, False)

    temporal_ds = autoencoder.temporal_downsample_factor
    if temporal_ds > 1:
        assert args.n_frames % temporal_ds == 0, (
            f"n_frames={args.n_frames} must be divisible by "
            f"temporal_downsample_factor={temporal_ds}"
        )
    # Decoder is only needed during validation; keep it on CPU during training.
    if hasattr(autoencoder, "decoder"):
        autoencoder.decoder = autoencoder.decoder.to("cpu")

    adapter = create_adapter(adapter_config_from_args(args), input_dim=autoencoder.latent_dim).to(device)
    adapter.train()
    adapter = torch.compile(adapter, mode="default")  # type: ignore[assignment]
    if hasattr(autoencoder, "encode"):
        autoencoder.encode = torch.compile(autoencoder.encode, mode="default")  # type: ignore[assignment]

    pixel_decoder = None
    pixel_decoder_no_ddp = None
    if getattr(args, "use_pixel_decoder", False):
        pd_config = pixel_decoder_config_from_args(args)
        pixel_decoder = create_pixel_decoder(pd_config).to(device)
        pixel_decoder.train()
        pixel_decoder = torch.compile(pixel_decoder, mode="default")  # type: ignore[assignment]
        pixel_decoder_no_ddp = pixel_decoder

    lpips_vgg = None
    if getattr(args, "use_lpips", False) and lpips is not None:
        lpips_vgg = lpips.LPIPS(net="vgg").to(device)
        lpips_vgg.eval()
        requires_grad(lpips_vgg, False)

    # ── Discriminator (optional) ──────────────────────────────────────────────
    discriminator = None
    disc_optimizer = None
    use_discriminator = getattr(args, "use_discriminator", False)
    disc_start_samples = getattr(args, "disc_start_samples", 100_000)
    disc_weight_cfg = getattr(args, "disc_weight", 0.1)
    if use_discriminator:
        discriminator = PatchGANDiscriminator(in_channels=3).to(device)
        discriminator.train()
        disc_optimizer = optim.AdamW(
            discriminator.parameters(),
            lr=getattr(args, "disc_lr", 4e-5),
            weight_decay=0.0,
            betas=(0.0, 0.99),
        )

    if rank == 0:
        log_model_param_counts(autoencoder, adapter=adapter, pixel_decoder=pixel_decoder)

    # ── Optimizer (separate LR for pixel decoder) ─────────────────────────────
    adapter_params = list(adapter.parameters())
    pd_params = list(pixel_decoder.parameters()) if pixel_decoder is not None else []
    pd_lr_mult = getattr(args, "pixel_decoder_lr_multiplier", 1.0)
    if pd_params and pd_lr_mult != 1.0:
        param_groups = [
            {"params": adapter_params, "lr": args.lr},
            {"params": pd_params, "lr": args.lr * pd_lr_mult},
        ]
    else:
        param_groups = [{"params": adapter_params + pd_params, "lr": args.lr}]
    optimizer = optim.AdamW(
        param_groups,
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 1e-4),
        betas=(0.9, 0.99),
    )

    # ── DDP ───────────────────────────────────────────────────────────────────
    if distributed:
        adapter = nn.parallel.DistributedDataParallel(
            adapter, device_ids=[local_rank], output_device=local_rank
        )
        adapter_no_ddp = adapter.module
        if pixel_decoder is not None:
            pixel_decoder = nn.parallel.DistributedDataParallel(
                pixel_decoder, device_ids=[local_rank], output_device=local_rank
            )
            pixel_decoder_no_ddp = pixel_decoder.module
    else:
        adapter_no_ddp = adapter

    # ── LR schedule ───────────────────────────────────────────────────────────
    warmup_steps = getattr(args, "warmup_epochs", 1) * steps_per_epoch
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(
                optimizer,
                T_max=max(max_train_steps - warmup_steps, 1),
                eta_min=args.lr * getattr(args, "min_lr_ratio", 0.1),
            ),
        ],
        milestones=[warmup_steps],
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    train_steps, total_samples_seen = load_adapter_training_checkpoint(
        checkpoint_dir, adapter_no_ddp, pixel_decoder_no_ddp, optimizer, lr_scheduler, device
    )
    current_epoch = train_steps // steps_per_epoch

    # ── Loss hyper-parameters ─────────────────────────────────────────────────
    kl_weight_target = getattr(args, "kl_weight", 1e-6)
    kl_warmup_fraction = getattr(args, "kl_warmup_fraction", 0.0)
    cos_weight = getattr(args, "cos_weight", 1.0)
    pixel_weight = getattr(args, "pixel_weight", 1.0)
    stage = getattr(args, "stage", "svae")
    perceptual_warmup_samples = getattr(args, "perceptual_warmup_samples", 0)
    use_ssim = getattr(args, "use_ssim", False)
    ssim_weight = getattr(args, "ssim_weight", 0.5)
    spectral_weight = getattr(args, "spectral_weight", 0.0)

    log_every = getattr(args, "log_every_samples", 10_000)
    val_every = getattr(args, "validate_every_samples", 100_000)
    vis_every = val_every * 2

    running_loss = running_mse = running_cos = running_pixel = 0.0
    running_ssim_loss = running_disc_loss = running_gen_loss = 0.0
    num_batches = 0
    last_log = (total_samples_seen // log_every) * log_every
    last_val = (total_samples_seen // val_every) * val_every
    last_vis = (total_samples_seen // vis_every) * vis_every

    pbar = tqdm(total=max_train_steps, desc="Adapter training") if rank == 0 else None
    if pbar and train_steps > 0:
        pbar.n = train_steps
        pbar.refresh()

    # ── Training loop ─────────────────────────────────────────────────────────
    while train_steps < max_train_steps:
        try:
            x, _ = next(train_iter)
        except StopIteration:
            current_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(current_epoch)
            train_iter = iter(train_loader)
            x, _ = next(train_iter)

        x = x.to(device)

        # ── KL warmup schedule ────────────────────────────────────────────
        kl_warmup_steps = int(kl_warmup_fraction * max_train_steps) if kl_warmup_fraction > 0 else 0
        if kl_warmup_steps > 0:
            kl_weight = min(kl_weight_target, kl_weight_target * (train_steps / kl_warmup_steps))
        else:
            kl_weight = kl_weight_target

        # ── Forward (autocast for encoder + adapter + pixel MSE) ──────────
        with torch.autocast(device_type="cuda", dtype=precision):
            with torch.no_grad():
                f_h = autoencoder.encode(x)

            enc_out = adapter_no_ddp.encode(f_h)
            if isinstance(enc_out, tuple):
                z_l, mu, logvar = enc_out
            else:
                z_l, mu, logvar = enc_out, None, None

            f_h_rec = adapter_no_ddp.decode(z_l)
            loss_rec, loss_mse, loss_cos = semantic_reconstruction_loss(
                f_h, f_h_rec, cos_weight, spectral_weight=spectral_weight
            )
            loss_kl = (
                kl_divergence(mu, logvar)
                if mu is not None and logvar is not None
                else torch.tensor(0.0, device=device)
            )
            loss = loss_rec + kl_weight * loss_kl

            loss_pixel_direct = torch.tensor(0.0, device=device)
            loss_ssim = torch.tensor(0.0, device=device)
            loss_disc_val = torch.tensor(0.0, device=device)
            loss_gen_val = torch.tensor(0.0, device=device)
            x_rec = None

            if pixel_decoder is not None:
                z_l_pd = z_l.detach() if stage == "svae" else z_l
                x_rec = pixel_decoder(z_l_pd)
                x_target = x.float()
                if x_rec.shape[2:4] != x_target.shape[2:4]:
                    B_, T_, H_o, W_o, C_ = x_rec.shape
                    x_rec = (
                        F.interpolate(
                            x_rec.reshape(B_ * T_, H_o, W_o, C_).permute(0, 3, 1, 2),
                            size=(x_target.shape[2], x_target.shape[3]),
                            mode="bilinear", align_corners=False,
                        )
                        .permute(0, 2, 3, 1)
                        .reshape(B_, T_, x_target.shape[2], x_target.shape[3], C_)
                    )
                loss_pixel_direct = F.mse_loss(x_rec, x_target)

                # ── MS-SSIM loss (inside autocast is fine, it's lightweight) ──
                if use_ssim:
                    b_s, t_s, h_s, w_s, c_s = x_target.shape
                    pred_nchw = x_rec.reshape(b_s * t_s, h_s, w_s, c_s).permute(0, 3, 1, 2).clamp(0, 1)
                    tgt_nchw = x_target.reshape(b_s * t_s, h_s, w_s, c_s).permute(0, 3, 1, 2)
                    loss_ssim = ssim_weight * (1.0 - _compute_ms_ssim(pred_nchw.float(), tgt_nchw.float()))

                loss = loss + pixel_weight * (loss_pixel_direct + loss_ssim)

        # ── LPIPS in float32 OUTSIDE autocast to avoid bf16 instability ───
        if pixel_decoder is not None and lpips_vgg is not None and x_rec is not None:
            lpips_active = total_samples_seen >= perceptual_warmup_samples
            if lpips_active:
                b_l, t_l, h_l, w_l, c_l = x_target.shape
                with torch.no_grad():
                    rgbs_tgt = (x_target * 2 - 1).reshape(b_l * t_l, h_l, w_l, c_l).permute(0, 3, 1, 2).float()
                rgbs_pred = (x_rec * 2 - 1).reshape(b_l * t_l, h_l, w_l, c_l).permute(0, 3, 1, 2).float()
                lpips_loss = getattr(args, "lpips_weight", 0.5) * lpips_vgg(rgbs_tgt, rgbs_pred).mean()
                loss = loss + pixel_weight * lpips_loss

        # ── Generator step (discriminator GAN loss on generator) ──────────
        disc_active = use_discriminator and discriminator is not None and total_samples_seen >= disc_start_samples
        if disc_active and x_rec is not None:
            b_g, t_g, h_g, w_g, c_g = x_rec.shape
            fake_nchw = x_rec.reshape(b_g * t_g, h_g, w_g, c_g).permute(0, 3, 1, 2).float()
            fake_logits = discriminator(fake_nchw)
            g_loss = hinge_loss_gen(fake_logits)

            # Adaptive weight via last pixel decoder layer
            last_layer_w = pixel_decoder_no_ddp.conv_out.weight
            ada_w = adaptive_weight(loss, g_loss, last_layer_w)
            loss_gen_val = disc_weight_cfg * ada_w * g_loss
            loss = loss + loss_gen_val

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
        if pd_params:
            nn.utils.clip_grad_norm_(pd_params, max_norm=1.0)
        optimizer.step()

        # ── Discriminator step ────────────────────────────────────────────
        if disc_active and x_rec is not None and disc_optimizer is not None:
            disc_optimizer.zero_grad()
            with torch.no_grad():
                b_d, t_d, h_d, w_d, c_d = x_rec.shape
                fake_nchw_d = x_rec.reshape(b_d * t_d, h_d, w_d, c_d).permute(0, 3, 1, 2).float()
                real_nchw_d = x_target.reshape(b_d * t_d, h_d, w_d, c_d).permute(0, 3, 1, 2).float()
            real_logits = discriminator(real_nchw_d)
            fake_logits_d = discriminator(fake_nchw_d.detach())
            loss_disc_val = hinge_loss_disc(real_logits, fake_logits_d)
            loss_disc_val.backward()
            disc_optimizer.step()

        lr_scheduler.step()

        running_loss += loss.item()
        running_mse += loss_mse.item()
        running_cos += loss_cos.item()
        running_pixel += loss_pixel_direct.item()
        running_ssim_loss += loss_ssim.item()
        running_disc_loss += loss_disc_val.item()
        running_gen_loss += loss_gen_val.item() if isinstance(loss_gen_val, torch.Tensor) else loss_gen_val
        num_batches += 1
        total_samples_seen += args.batch_size * world_size
        train_steps += 1

        # Logging
        if total_samples_seen - last_log >= log_every:
            n = max(num_batches, 1)
            if rank == 0:
                if pbar:
                    pbar.set_postfix(loss=f"{running_loss / n:.5f}", samples=total_samples_seen)
                    pbar.update(log_every // (args.batch_size * world_size))
                log_dict = {
                    "adapter/loss": running_loss / n,
                    "adapter/loss_rec": loss_rec.item(),
                    "adapter/loss_mse": running_mse / n,
                    "adapter/loss_cos": running_cos / n,
                    "adapter/loss_kl": loss_kl.item(),
                    "adapter/kl_weight": kl_weight,
                    "adapter/loss_pixel_direct": running_pixel / n,
                    "adapter/grad_norm": grad_norm.item(),
                    "adapter/lr": optimizer.param_groups[0]["lr"],
                }
                if use_ssim:
                    log_dict["adapter/loss_ssim"] = running_ssim_loss / n
                if use_discriminator:
                    log_dict["adapter/loss_disc"] = running_disc_loss / n
                    log_dict["adapter/loss_gen"] = running_gen_loss / n
                wandb.log(log_dict, step=total_samples_seen)
            running_loss = running_mse = running_cos = running_pixel = 0.0
            running_ssim_loss = running_disc_loss = running_gen_loss = 0.0
            num_batches = 0
            last_log = total_samples_seen

        # Visualization (2× less frequent than validation)
        if total_samples_seen - last_vis >= vis_every and rank == 0:
            log_reconstruction_video(
                autoencoder, adapter_no_ddp, val_loader, device, precision,
                checkpoint_dir, total_samples_seen, pixel_decoder=pixel_decoder_no_ddp,
            )
            last_vis = total_samples_seen
            adapter.train()
            if pixel_decoder is not None:
                pixel_decoder.train()

        # Validation + checkpoint
        if total_samples_seen - last_val >= val_every and rank == 0:
            save_video = (total_samples_seen - last_vis) >= vis_every
            val_metrics = validate_adapter(
                autoencoder, adapter_no_ddp, val_loader, device, precision,
                cos_weight, kl_weight, checkpoint_dir=checkpoint_dir,
                total_samples_seen=total_samples_seen, save_video=save_video,
                pixel_decoder=pixel_decoder_no_ddp,
            )
            if save_video:
                last_vis = total_samples_seen

            wandb.log(
                {
                    "adapter_val/loss_rec": val_metrics["loss_rec"],
                    "adapter_val/loss_mse": val_metrics["loss_mse"],
                    "adapter_val/loss_cos": val_metrics["loss_cos"],
                    "adapter_val/loss_kl": val_metrics["loss_kl"],
                    "adapter_val/psnr": val_metrics["psnr"],
                    "adapter_val/ssim": val_metrics["ssim"],
                    "adapter_val/psnr_direct": val_metrics["psnr_direct"],
                    "adapter_val/ssim_direct": val_metrics["ssim_direct"],
                },
                step=total_samples_seen,
            )
            logging.info(
                "Adapter val @ %d: rec=%.5f mse=%.5f cos=%.5f kl=%.5f "
                "psnr=%.2f ssim=%.4f | direct psnr=%.2f ssim=%.4f",
                total_samples_seen,
                val_metrics["loss_rec"], val_metrics["loss_mse"],
                val_metrics["loss_cos"], val_metrics["loss_kl"],
                val_metrics["psnr"], val_metrics["ssim"],
                val_metrics["psnr_direct"], val_metrics["ssim_direct"],
            )
            save_adapter_checkpoint(
                checkpoint_dir, adapter_no_ddp, pixel_decoder_no_ddp,
                optimizer, lr_scheduler, train_steps, total_samples_seen,
                adapter_config=adapter_config_from_args(args),
                encoder_config=encoder_config_from_args(args),
                pixel_decoder_config=(
                    pixel_decoder_config_from_args(args)
                    if pixel_decoder_no_ddp is not None else None
                ),
            )
            last_val = total_samples_seen
            adapter.train()
            if pixel_decoder is not None:
                pixel_decoder.train()

    if distributed:
        dist.destroy_process_group()
    if rank == 0 and pbar:
        pbar.close()
