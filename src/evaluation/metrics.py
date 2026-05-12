"""Core metric wrappers: PSNR, SSIM, LPIPS, FID with uniform API."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PSNR / SSIM
# ---------------------------------------------------------------------------


def compute_psnr_ssim(
    gen: np.ndarray,
    gt: np.ndarray,
    skip_frames: int = 0,
) -> Dict[str, List[float]]:
    """Compute per-frame PSNR and SSIM between generated and ground-truth frames.

    Parameters
    ----------
    gen : (N, T, H, W, 3) float32 in [0, 1]
    gt  : (N, T, H, W, 3) float32 in [0, 1]
    skip_frames : number of leading context frames to skip

    Returns
    -------
    dict with keys ``"psnr"`` and ``"ssim"``, each a list of per-frame values
    (averaged across the batch for each timestep).
    """
    if gen.ndim == 4:
        gen = gen[np.newaxis]
        gt = gt[np.newaxis]

    N, T, H, W, C = gen.shape
    gen = gen[:, skip_frames:]
    gt = gt[:, skip_frames:]
    T_eval = gen.shape[1]

    psnr_per_step: List[float] = []
    ssim_per_step: List[float] = []

    for t in range(T_eval):
        psnr_vals = []
        ssim_vals = []
        for b in range(N):
            gen_frame = (gen[b, t] * 255).astype(np.uint8)
            gt_frame = (gt[b, t] * 255).astype(np.uint8)

            mse = np.mean((gt_frame.astype(np.float32) - gen_frame.astype(np.float32)) ** 2)
            if mse == 0:
                psnr_vals.append(100.0)
            else:
                psnr_vals.append(
                    peak_signal_noise_ratio(gt_frame, gen_frame, data_range=255)
                )

            # Use channel_axis for multichannel SSIM (not grayscale conversion)
            ssim_vals.append(
                structural_similarity(
                    gt_frame, gen_frame, data_range=255, channel_axis=2
                )
            )

        psnr_per_step.append(float(np.mean(psnr_vals)))
        ssim_per_step.append(float(np.mean(ssim_vals)))

    return {"psnr": psnr_per_step, "ssim": ssim_per_step}


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

_lpips_model = None


def _get_lpips_model(device: torch.device):
    """Lazily load the LPIPS model (singleton)."""
    global _lpips_model
    if _lpips_model is None:
        import lpips

        _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    elif next(_lpips_model.parameters()).device != device:
        _lpips_model = _lpips_model.to(device)
    return _lpips_model


def compute_lpips_batch(
    gen: np.ndarray,
    gt: np.ndarray,
    device: torch.device,
    skip_frames: int = 0,
    batch_size: int = 64,
) -> List[float]:
    """Compute per-frame LPIPS (AlexNet) between generated and GT frames.

    Parameters
    ----------
    gen : (N, T, H, W, 3) float32 in [0, 1]
    gt  : (N, T, H, W, 3) float32 in [0, 1]
    device : torch device
    skip_frames : number of leading context frames to skip
    batch_size : internal batch size for LPIPS forward passes

    Returns
    -------
    List of per-frame LPIPS values (averaged across batch), length T - skip_frames.
    Lower is better.
    """
    if gen.ndim == 4:
        gen = gen[np.newaxis]
        gt = gt[np.newaxis]

    gen = gen[:, skip_frames:]
    gt = gt[:, skip_frames:]
    N, T_eval, H, W, C = gen.shape

    model = _get_lpips_model(device)
    lpips_per_step: List[float] = []

    with torch.no_grad():
        for t in range(T_eval):
            # LPIPS expects (B, 3, H, W) in [-1, 1]
            gen_t = torch.from_numpy(gen[:, t]).permute(0, 3, 1, 2).float().to(device) * 2 - 1
            gt_t = torch.from_numpy(gt[:, t]).permute(0, 3, 1, 2).float().to(device) * 2 - 1

            vals = []
            for i in range(0, N, batch_size):
                d = model(gen_t[i : i + batch_size], gt_t[i : i + batch_size])
                vals.append(d.squeeze().cpu().numpy())
            vals = np.concatenate(vals) if len(vals) > 1 else vals[0].reshape(-1)
            lpips_per_step.append(float(np.mean(vals)))

    return lpips_per_step


# ---------------------------------------------------------------------------
# FID (Frechet distance on InceptionV3 features)
# ---------------------------------------------------------------------------

_inception_model = None


def _get_inception_model(device: torch.device):
    """Lazily load InceptionV3 with pool3 features (singleton)."""
    global _inception_model
    if _inception_model is None:
        from torchvision.models import inception_v3, Inception_V3_Weights

        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = torch.nn.Identity()  # remove classifier, get 2048-dim pool features
        model = model.to(device).eval()
        _inception_model = model
    elif next(_inception_model.parameters()).device != device:
        _inception_model = _inception_model.to(device)
    return _inception_model


def _extract_inception_features(
    frames: np.ndarray, device: torch.device, batch_size: int = 64
) -> np.ndarray:
    """Extract InceptionV3 pool3 features from frames.

    Parameters
    ----------
    frames : (N, H, W, 3) float32 in [0, 1]
    device : torch device
    batch_size : internal batch size

    Returns
    -------
    (N, 2048) float32 features
    """
    from torchvision import transforms

    model = _get_inception_model(device)
    preprocess = transforms.Compose([
        transforms.Resize((299, 299), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = torch.from_numpy(frames[i : i + batch_size]).permute(0, 3, 1, 2).float().to(device)
            batch = preprocess(batch)
            feats = model(batch)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute Frechet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_fid(
    gen_frames: np.ndarray,
    gt_frames: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Compute FID between generated and GT frame distributions.

    Parameters
    ----------
    gen_frames : (N, H, W, 3) float32 in [0, 1]  — flattened across time
    gt_frames  : (M, H, W, 3) float32 in [0, 1]
    device : torch device
    batch_size : internal batch size for feature extraction

    Returns
    -------
    FID score (lower is better).
    """
    logger.info("Extracting InceptionV3 features for FID (gen=%d, gt=%d)...", len(gen_frames), len(gt_frames))
    gen_feats = _extract_inception_features(gen_frames, device, batch_size)
    gt_feats = _extract_inception_features(gt_frames, device, batch_size)

    mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)
    mu_gt, sigma_gt = gt_feats.mean(axis=0), np.cov(gt_feats, rowvar=False)

    return _frechet_distance(mu_gen, sigma_gen, mu_gt, sigma_gt)


# ---------------------------------------------------------------------------
# Horizon aggregation
# ---------------------------------------------------------------------------


def compute_per_step_metrics(
    all_metrics: Dict[str, List[float]],
) -> Dict[str, Dict[str, List[float]]]:
    """Package per-step metrics with mean/std per step.

    Parameters
    ----------
    all_metrics : dict mapping metric name → list of per-step values

    Returns
    -------
    dict mapping metric name → {"mean": [...], "std": [...], "steps": [0, 1, ...]}
    """
    result = {}
    for name, values in all_metrics.items():
        arr = np.array(values)
        result[name] = {
            "mean": arr.tolist(),
            "steps": list(range(len(arr))),
        }
    return result
