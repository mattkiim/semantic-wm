"""PCK (Percentage of Correct Keypoints) metric via CoTracker point tracking."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CoTracker singleton
# ---------------------------------------------------------------------------

_cotracker_model = None


def _get_cotracker_model(device: torch.device):
    """Lazily load CoTracker v2 (singleton)."""
    global _cotracker_model
    if _cotracker_model is None:
        logger.info("Loading CoTracker v2 via torch.hub...")
        _cotracker_model = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker2", skip_validation=True
        )
        _cotracker_model = _cotracker_model.to(device).eval()
    elif next(_cotracker_model.parameters()).device != device:
        _cotracker_model = _cotracker_model.to(device)
    return _cotracker_model


# ---------------------------------------------------------------------------
# Query point creation
# ---------------------------------------------------------------------------


def _create_grid_queries(
    height: int,
    width: int,
    grid_size: int = 16,
    frame_idx: int = 0,
) -> torch.Tensor:
    """Create a regular grid of query points.

    Returns
    -------
    (1, grid_size^2, 3) tensor with columns [frame_idx, x, y].
    """
    margin = max(height, width) * 0.02  # small margin from edges
    ys = torch.linspace(margin, height - margin, grid_size)
    xs = torch.linspace(margin, width - margin, grid_size)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack(
        [
            torch.full_like(grid_x.flatten(), frame_idx),
            grid_x.flatten(),
            grid_y.flatten(),
        ],
        dim=-1,
    )
    return points.unsqueeze(0)  # (1, N, 3)


def _create_salient_queries(
    frame: np.ndarray,
    n_points: int = 256,
    frame_idx: int = 0,
) -> torch.Tensor:
    """Create query points at salient locations using Shi-Tomasi corners.

    Parameters
    ----------
    frame : (H, W, 3) float32 in [0, 1]

    Returns
    -------
    (1, N, 3) tensor with columns [frame_idx, x, y].
    """
    gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(
        gray, maxCorners=n_points, qualityLevel=0.01, minDistance=5
    )

    h, w = frame.shape[:2]
    if corners is None or len(corners) < n_points // 4:
        logger.warning(
            "Too few salient points (%d), falling back to grid",
            0 if corners is None else len(corners),
        )
        grid_size = int(np.sqrt(n_points))
        return _create_grid_queries(h, w, grid_size=grid_size, frame_idx=frame_idx)

    # corners shape: (N, 1, 2) with (x, y)
    xy = corners.squeeze(1)  # (N, 2)
    t_col = np.full((len(xy), 1), frame_idx, dtype=np.float32)
    points = np.concatenate([t_col, xy], axis=1)  # (N, 3): [t, x, y]
    return torch.from_numpy(points).unsqueeze(0)  # (1, N, 3)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def _track_points(
    video: np.ndarray,
    queries: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track query points through a single video using CoTracker.

    Parameters
    ----------
    video : (T, H, W, 3) float32 in [0, 1]
    queries : (1, N, 3) tensor with [frame_idx, x, y]

    Returns
    -------
    tracks : (T, N, 2) float32 — xy positions per frame per point
    visibility : (T, N) bool — whether each point is confidently tracked
    """
    model = _get_cotracker_model(device)

    # CoTracker expects (B, T, 3, H, W) float32 in [0, 255]
    vid_tensor = (
        torch.from_numpy(video)
        .permute(0, 3, 1, 2)  # (T, 3, H, W)
        .unsqueeze(0)  # (1, T, 3, H, W)
        .float()
        .to(device)
        * 255.0
    )

    queries = queries.float().to(device)

    with torch.no_grad():
        pred_tracks, pred_visibility = model(vid_tensor, queries=queries)

    # pred_tracks: (1, T, N, 2), pred_visibility: (1, T, N)
    tracks = pred_tracks[0].cpu().numpy()  # (T, N, 2)
    visibility = pred_visibility[0].cpu().numpy() > 0.5  # (T, N)

    return tracks, visibility


# ---------------------------------------------------------------------------
# PCK computation
# ---------------------------------------------------------------------------


def compute_pck(
    gen_videos: np.ndarray,
    gt_videos: np.ndarray,
    device: torch.device,
    skip_frames: int = 0,
    thresholds: Tuple[int, ...] = (5, 10, 20, 40),
    grid_size: int = 16,
    point_mode: str = "grid",
) -> Dict[str, Any]:
    """Compute PCK at multiple thresholds with per-step breakdown.

    Parameters
    ----------
    gen_videos : (N, T, H, W, 3) float32 in [0, 1]
    gt_videos  : (N, T, H, W, 3) float32 in [0, 1]
    device : torch device
    skip_frames : number of leading context frames to skip in reporting
    thresholds : pixel-distance thresholds for PCK
    grid_size : grid resolution when point_mode="grid" (grid_size^2 points)
    point_mode : "grid" or "salient"

    Returns
    -------
    dict with:
      "pck@{k}_per_step" : List[float] for each threshold k
      "pck@{k}_mean"     : float for each threshold k
      "coverage_per_step": List[float]
    """
    if gen_videos.ndim == 4:
        gen_videos = gen_videos[np.newaxis]
        gt_videos = gt_videos[np.newaxis]

    N, T, H, W, C = gen_videos.shape
    T_eval = T - skip_frames

    # Accumulators: per-threshold, per-step
    pck_accum = {k: np.zeros(T_eval, dtype=np.float64) for k in thresholds}
    coverage_accum = np.zeros(T_eval, dtype=np.float64)
    valid_count = np.zeros(T_eval, dtype=np.float64)  # for nanmean

    for i in range(N):
        gt_vid = gt_videos[i]  # (T, H, W, 3)
        gen_vid = gen_videos[i]

        # Create queries on frame 0 (shared context)
        if point_mode == "salient":
            queries = _create_salient_queries(gt_vid[0], n_points=grid_size ** 2, frame_idx=0)
        else:
            queries = _create_grid_queries(H, W, grid_size=grid_size, frame_idx=0)

        # Track through both videos
        gt_tracks, gt_vis = _track_points(gt_vid, queries, device)
        gen_tracks, gen_vis = _track_points(gen_vid, queries, device)

        # Slice to evaluation window
        gt_tracks = gt_tracks[skip_frames:]  # (T_eval, N_pts, 2)
        gen_tracks = gen_tracks[skip_frames:]
        gt_vis = gt_vis[skip_frames:]  # (T_eval, N_pts)
        gen_vis = gen_vis[skip_frames:]

        # Valid = visible in both
        valid = gt_vis & gen_vis  # (T_eval, N_pts)

        # Euclidean distance
        dist = np.linalg.norm(gt_tracks - gen_tracks, axis=-1)  # (T_eval, N_pts)

        n_pts = queries.shape[1]
        for t in range(T_eval):
            valid_mask = valid[t]
            n_valid = valid_mask.sum()
            if n_valid == 0:
                continue

            valid_count[t] += 1
            coverage_accum[t] += n_valid / n_pts

            d = dist[t, valid_mask]
            for k in thresholds:
                pck_accum[k][t] += (d < k).mean()

    # Average across videos
    result: Dict[str, Any] = {}
    for k in thresholds:
        per_step = np.where(valid_count > 0, pck_accum[k] / valid_count, np.nan)
        result[f"pck@{k}_per_step"] = np.nan_to_num(per_step, nan=0.0).tolist()
        result[f"pck@{k}_mean"] = float(np.nanmean(per_step))

    coverage_per_step = np.where(valid_count > 0, coverage_accum / valid_count, 0.0)
    result["coverage_per_step"] = coverage_per_step.tolist()
    result["coverage_mean"] = float(np.mean(coverage_per_step))

    return result
