"""Trajectory success probes and feature extraction utilities.

Probe architectures operate on patch-level features (B, T, H, W, C)
and perform pooling internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature extraction (online, no caching)
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_features(
    frames: torch.Tensor,
    autoencoder,
    adapter=None,
    feature_space: str = "adapter",
) -> torch.Tensor:
    """Encode pixel frames to latent features.

    Parameters
    ----------
    frames : (B, T, H, W, C) float tensor in [0, 1]
    autoencoder : BaseAutoencoder
    adapter : BaseAdapter or None
    feature_space : "encoder" or "adapter"

    Returns
    -------
    (B, T, h, w, C_feat) latent features (e.g. h=w=16 for 256x256 input)
    """
    z = autoencoder.encode(frames)

    if feature_space == "encoder" or adapter is None:
        return z

    z_adapted = adapter.encode(z)
    if isinstance(z_adapted, tuple):
        z_adapted = z_adapted[0]
    return z_adapted


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------


def _pool_patches(x: torch.Tensor, pool_mode: str) -> torch.Tensor:
    """Pool spatial patch dimensions.

    Parameters
    ----------
    x : (B, T, H, W, C) patch-level features
    pool_mode : "mean" or "super_patch_4x4"

    Returns
    -------
    "mean":            (B, T, C)
    "super_patch_4x4": (B, T, 16, C)  (4x4 grid flattened)
    """
    if pool_mode == "mean":
        return x.mean(dim=(2, 3))  # (B, T, C)

    elif pool_mode == "super_patch_4x4":
        B, T, H, W, C = x.shape
        # Reshape to (B*T, C, H, W) for adaptive_avg_pool2d
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (4, 4))  # (B*T, C, 4, 4)
        x = x.permute(0, 2, 3, 1).reshape(B, T, 16, C)
        return x
    
    elif pool_mode == "super_patch_8x8":
        B, T, H, W, C = x.shape
        x = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, (8, 8))  # (B*T, C, 8, 8)
        x = x.permute(0, 2, 3, 1).reshape(B, T, 64, C)
        return x
    else:
        raise ValueError(f"Unknown pool_mode: {pool_mode}")


# ---------------------------------------------------------------------------
# Probe architectures
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """True linear probe: logistic regression on mean-pooled features.

    Pools (B, T, H, W, C) -> (B, T, C) -> flatten -> Linear(T*C, 1).
    """

    def __init__(self, feature_dim: int, n_frames: int, pool_mode: str = "mean"):
        super().__init__()
        self.pool_mode = pool_mode
        if pool_mode == "mean":
            in_dim = n_frames * feature_dim
        elif pool_mode == "super_patch_4x4":
            in_dim = n_frames * 16 * feature_dim
        elif pool_mode == "super_patch_8x8":
            in_dim = n_frames * 64 * feature_dim
        else:
            raise ValueError(f"Unknown pool_mode: {pool_mode}")
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, H, W, C) patch-level features

        Returns
        -------
        (B,) logits
        """
        x = _pool_patches(x, self.pool_mode)  # (B, T, C) or (B, T, 16, C)
        return self.classifier(x.flatten(1)).squeeze(-1)


class TemporalProbe(nn.Module):
    """Lightweight temporal probe: 1-layer transformer on pooled features.

    Pools (B, T, H, W, C) -> (B, T, C) -> [CLS] + pos_embed ->
    1-layer TransformerEncoder -> head.
    """

    def __init__(
        self,
        feature_dim: int,
        n_frames: int,
        n_heads: int = 8,
        pool_mode: str = "mean",
    ):
        super().__init__()
        self.pool_mode = pool_mode

        if pool_mode in ("super_patch_4x4", "super_patch_8x8"):
            n_patches = 16 if pool_mode == "super_patch_4x4" else 64
            self.spatial_proj = nn.Linear(n_patches * feature_dim, feature_dim)
        else:
            self.spatial_proj = None

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, H, W, C) patch-level features

        Returns
        -------
        (B,) logits
        """
        x = _pool_patches(x, self.pool_mode)  # (B, T, C) or (B, T, 16, C)

        if self.spatial_proj is not None:
            B, T = x.shape[:2]
            x = x.reshape(B, T, -1)
            x = self.spatial_proj(x)  # (B, T, C)

        B = x.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, C)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)  # CLS output


class SpatiotemporalProbe(nn.Module):
    """Spatiotemporal probe: each spatial patch is its own token.

    Pools (B, T, H, W, C) -> (B, T, S, C) via super_patch, then flattens
    to (B, T*S, C) so the Transformer attends over both time and space.
    Uses learned temporal + spatial position embeddings.
    """

    def __init__(
        self,
        feature_dim: int,
        n_frames: int,
        n_heads: int = 8,
        pool_mode: str = "super_patch_8x8",
    ):
        super().__init__()
        if pool_mode == "super_patch_4x4":
            self.n_patches = 16
        elif pool_mode == "super_patch_8x8":
            self.n_patches = 64
        else:
            raise ValueError(
                f"SpatiotemporalProbe requires super_patch pool_mode, got: {pool_mode}"
            )
        self.pool_mode = pool_mode
        self.n_frames = n_frames
        n_tokens = n_frames * self.n_patches  # e.g. 8*64 = 512

        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.temporal_embed = nn.Parameter(torch.randn(1, n_frames, 1, feature_dim))
        self.spatial_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, feature_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, H, W, C) patch-level features

        Returns
        -------
        (B,) logits
        """
        x = _pool_patches(x, self.pool_mode)  # (B, T, S, C)
        B, T, S, C = x.shape
        # Add temporal + spatial position embeddings (broadcast-summed)
        x = x + self.temporal_embed + self.spatial_embed
        x = x.reshape(B, T * S, C)  # (B, T*S, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, T*S+1, C)
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)


class ProgressRegressor(nn.Module):
    """Per-frame regression: predict normalized timestep t/T.

    Pools (B, T, H, W, C) -> (B, T, C) -> Linear(C, 1) per frame.
    """

    def __init__(self, feature_dim: int, pool_mode: str = "mean"):
        super().__init__()
        self.pool_mode = pool_mode
        self.head = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, H, W, C)

        Returns
        -------
        (B, T) predicted normalized timestamps
        """
        x = _pool_patches(x, self.pool_mode)  # (B, T, C)
        if x.ndim == 4:
            # super_patch: (B, T, 16, C) -> mean over patches
            x = x.mean(dim=2)
        return self.head(x).squeeze(-1)  # (B, T)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_probe(
    probe_type: str,
    feature_dim: int,
    n_frames: int,
    pool_mode: str = "mean",
    n_heads: int = 8,
) -> nn.Module:
    """Create a probe model by type name."""
    if probe_type == "linear":
        return LinearProbe(feature_dim, n_frames, pool_mode=pool_mode)
    elif probe_type == "temporal":
        return TemporalProbe(feature_dim, n_frames, n_heads=n_heads, pool_mode=pool_mode)
    elif probe_type == "spatiotemporal":
        return SpatiotemporalProbe(feature_dim, n_frames, n_heads=n_heads, pool_mode=pool_mode)
    elif probe_type == "progress":
        return ProgressRegressor(feature_dim, pool_mode=pool_mode)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
