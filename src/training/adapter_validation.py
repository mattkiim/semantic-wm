"""Adapter validation and visualization helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from .validation import calculate_image_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def semantic_reconstruction_loss(
    f_h: torch.Tensor,
    f_h_rec: torch.Tensor,
    cos_weight: float = 1.0,
    spectral_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L_rec = MSE(f_h, f_h_rec) + cos_weight * (1 - cos_sim(f_h, f_h_rec))
            + spectral_weight * spectral_loss(f_h, f_h_rec).

    The spectral loss compares DCT-like frequency content of the feature maps
    via FFT along the spatial token dimension, penalizing loss of high-frequency
    structure through the adapter bottleneck.

    Returns (total_loss, mse_loss, cos_loss).
    """
    mse = F.mse_loss(f_h_rec, f_h)
    cos_sim = F.cosine_similarity(
        f_h.flatten(0, -2), f_h_rec.flatten(0, -2), dim=-1
    ).mean()
    cos_loss = 1.0 - cos_sim
    total = mse + cos_weight * cos_loss

    if spectral_weight > 0.0:
        total = total + spectral_weight * _spectral_loss(f_h, f_h_rec)

    return total, mse, cos_loss


def _spectral_loss(f_h: torch.Tensor, f_h_rec: torch.Tensor) -> torch.Tensor:
    """L1 loss on FFT magnitudes along the spatial/token dimension.

    Operates on (B, T, N, C) or (B, T, H, W, C) features — we flatten spatial
    dims and compute 1-D FFT along the token axis to capture spatial frequency
    structure in the feature space.
    """
    # Flatten to (BT, N, C)
    if f_h.dim() == 5:
        B, T, H, W, C = f_h.shape
        f_h = f_h.reshape(B * T, H * W, C)
        f_h_rec = f_h_rec.reshape(B * T, H * W, C)
    elif f_h.dim() == 4:
        B, T, N, C = f_h.shape
        f_h = f_h.reshape(B * T, N, C)
        f_h_rec = f_h_rec.reshape(B * T, N, C)

    # 1-D FFT along token dimension (dim=1)
    spec_orig = torch.fft.rfft(f_h.float(), dim=1).abs()
    spec_rec = torch.fft.rfft(f_h_rec.float(), dim=1).abs()
    return F.l1_loss(spec_rec, spec_orig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate_adapter(
    autoencoder,
    adapter,
    val_loader: DataLoader,
    device,
    precision,
    cos_weight: float,
    kl_weight: float,
    checkpoint_dir: Path,
    total_samples_seen: int,
    max_batches: int = 50,
    save_video: bool = False,
    pixel_decoder=None,
) -> Dict[str, float]:
    from ..models.adapters import kl_divergence

    adapter.eval()
    if pixel_decoder is not None:
        pixel_decoder.eval()

    total_rec = total_mse = total_cos = total_kl = 0.0
    all_psnr: List[float] = []
    all_ssim: List[float] = []
    all_psnr_direct: List[float] = []
    all_ssim_direct: List[float] = []
    video_logged = False
    n = 0

    has_decoder = hasattr(autoencoder, "decode") and getattr(
        autoencoder, "has_decoder", True
    )
    if has_decoder:
        autoencoder.decoder.to(device)

    for i, (x, _) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device)
        x_orig_px = x_rec_px = x_rec_direct_px = None

        with torch.autocast(device_type="cuda", dtype=precision):
            f_h = autoencoder.encode(x)
            enc_out = adapter.encode(f_h)
            if isinstance(enc_out, tuple):
                z_l, mu, logvar = enc_out
            else:
                z_l, mu, logvar = enc_out, None, None
            f_h_rec = adapter.decode(z_l)

            loss_total, loss_mse, loss_cos = semantic_reconstruction_loss(
                f_h, f_h_rec, cos_weight
            )
            total_rec += loss_total.item()
            total_mse += loss_mse.item()
            total_cos += loss_cos.item()
            if mu is not None and logvar is not None:
                total_kl += kl_divergence(mu, logvar).item()

            if has_decoder:
                x_orig_px = autoencoder.decode(f_h)
                x_rec_px = autoencoder.decode(f_h_rec)

            if pixel_decoder is not None:
                x_rec_direct_px = pixel_decoder(z_l)
                if x_rec_direct_px.shape[2:4] != x.shape[2:4]:
                    x_rec_direct_px = _resize_to(x_rec_direct_px, x.shape[2:4])

        if has_decoder and x_orig_px is not None and x_rec_px is not None:
            m = calculate_image_metrics(
                x_orig_px.float().clamp(0, 1).cpu().numpy(),
                x_rec_px.float().clamp(0, 1).cpu().numpy(),
            )
            all_psnr.append(m["psnr"])
            all_ssim.append(m["ssim"])

        if x_rec_direct_px is not None:
            m = calculate_image_metrics(
                x.float().clamp(0, 1).cpu().numpy(),
                x_rec_direct_px.float().clamp(0, 1).cpu().numpy(),
            )
            all_psnr_direct.append(m["psnr"])
            all_ssim_direct.append(m["ssim"])

        if save_video and not video_logged:
            panels = [x.float().clamp(0, 1).cpu().numpy()]
            if x_rec_px is not None:
                panels.append(x_rec_px.float().clamp(0, 1).cpu().numpy())
            if x_rec_direct_px is not None:
                panels.append(x_rec_direct_px.float().clamp(0, 1).cpu().numpy())
            log_comparison_video(panels, checkpoint_dir, total_samples_seen)
            video_logged = True

        n += 1

    if has_decoder:
        autoencoder.decoder.to("cpu")

    return {
        "loss_rec": total_rec / max(n, 1),
        "loss_mse": total_mse / max(n, 1),
        "loss_cos": total_cos / max(n, 1),
        "loss_kl": total_kl / max(n, 1),
        "psnr": float(np.mean(all_psnr)) if all_psnr else 0.0,
        "ssim": float(np.mean(all_ssim)) if all_ssim else 0.0,
        "psnr_direct": float(np.mean(all_psnr_direct)) if all_psnr_direct else 0.0,
        "ssim_direct": float(np.mean(all_ssim_direct)) if all_ssim_direct else 0.0,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def log_reconstruction_video(
    autoencoder,
    adapter,
    val_loader: DataLoader,
    device,
    precision,
    checkpoint_dir: Path,
    total_samples_seen: int,
    pixel_decoder=None,
) -> None:
    """Encode one validation batch and log a multi-panel MP4 to WandB."""
    has_ae_decoder = hasattr(autoencoder, "decode") and getattr(
        autoencoder, "has_decoder", True
    )
    if not has_ae_decoder and pixel_decoder is None:
        return

    try:
        x, _ = next(iter(val_loader))
        x = x.to(device)
        target_h, target_w = x.shape[2], x.shape[3]

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision):
            f_h = autoencoder.encode(x)
            enc_out = adapter.encode(f_h)
            z_l = enc_out[0] if isinstance(enc_out, tuple) else enc_out

            panels = [x.float().clamp(0, 1).cpu().numpy()]

            if has_ae_decoder:
                autoencoder.decoder.to(device)
                f_h_rec = adapter.decode(z_l)
                x_rec = autoencoder.decode(f_h_rec)
                if x_rec.shape[2:4] != (target_h, target_w):
                    x_rec = _resize_to(x_rec, (target_h, target_w))
                panels.append(x_rec.float().clamp(0, 1).cpu().numpy())
                autoencoder.decoder.to("cpu")

            if pixel_decoder is not None:
                x_direct = pixel_decoder(z_l)
                if x_direct.shape[2:4] != (target_h, target_w):
                    x_direct = _resize_to(x_direct, (target_h, target_w))
                panels.append(x_direct.float().clamp(0, 1).cpu().numpy())

        log_comparison_video(panels, checkpoint_dir, total_samples_seen)
    except Exception as e:
        logger.warning("Skipping vis video: %s", e)


def log_comparison_video(
    panels: list,
    checkpoint_dir: Path,
    total_samples_seen: int,
) -> None:
    """Concatenate panels horizontally and log an MP4 to WandB.

    Each panel is (B, T, H, W, C) in [0, 1]; only the first sample is used.
    """
    try:
        frames = [p[0] if p.ndim == 5 else p for p in panels]  # (T, H, W, C) each
        combined = np.concatenate(frames, axis=2)  # (T, H, W*n, C)
        video_path = checkpoint_dir / f"adapter_recon_{total_samples_seen:012d}.mp4"
        imageio.mimsave(str(video_path), (combined * 255).astype(np.uint8), fps=2)
        wandb.log(
            {"adapter_val/recon_video": wandb.Video(str(video_path))},
            step=total_samples_seen,
        )
    except Exception as e:
        logger.warning("Failed to save comparison video: %s", e)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resize_to(x: torch.Tensor, target_hw: tuple) -> torch.Tensor:
    """Bilinear-resize a (B, T, H, W, C) tensor to target_hw."""
    B, T, H, W, C = x.shape
    th, tw = target_hw
    return (
        F.interpolate(
            x.reshape(B * T, H, W, C).permute(0, 3, 1, 2),
            size=(th, tw),
            mode="bilinear",
            align_corners=False,
        )
        .permute(0, 2, 3, 1)
        .reshape(B, T, th, tw, C)
    )
