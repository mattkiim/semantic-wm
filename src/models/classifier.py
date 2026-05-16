"""Spill classifier operating directly on raw patch embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpillClassifier(nn.Module):
    """Binary spill classifier.

    Operates on raw patch embeddings (no adapter). Optionally fuses tactile
    by broadcasting the tactile CLS token to each RGB patch before the MLP.

    Architecture: LayerNorm → Linear → ReLU → Linear(1) → mean over patches → logit per frame.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, D) or (B, T, N, D) → logits (B, T)."""
        if x.dim() == 5:
            B, T, H, W, D = x.shape
            x = x.reshape(B, T, H * W, D)
        B, T, N, D = x.shape
        logits = self.head(x.reshape(B * T, N, D)).squeeze(-1)  # (B*T, N)
        return logits.mean(dim=-1).reshape(B, T)                 # (B, T)

    def forward_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Alias so eval_classifier.py works unchanged."""
        return self.forward(z)


def fuse_tactile(emb: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
    """Broadcast tactile CLS token to each RGB patch and concatenate.

    Args:
        emb:     (B, T, H, W, D_rgb) raw patch embeddings
        tactile: (B, T, D_tact) CLS token

    Returns:
        (B, T, H*W, D_rgb + D_tact)
    """
    B, T, H, W, D = emb.shape
    x = emb.reshape(B, T, H * W, D)
    t = tactile.unsqueeze(2).expand(-1, -1, H * W, -1)  # (B, T, N, D_tact)
    return torch.cat([x, t], dim=-1)
