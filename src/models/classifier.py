"""Spill classifier operating on adapter latents."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpillClassifier(nn.Module):
    """Binary spill classifier.

    Encodes raw patch embeddings through a frozen adapter, then applies a
    per-patch MLP followed by mean-pooling to produce a per-frame logit.

    Can also classify directly from pre-encoded adapter latents (e.g. WM
    outputs) by calling :meth:`forward_from_latent`.
    """

    def __init__(self, adapter: nn.Module, latent_dim: int = 96, hidden_dim: int = 256) -> None:
        super().__init__()
        self.adapter = adapter
        for p in self.adapter.parameters():
            p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, D) → latent (B, T, H, W, latent_dim)."""
        with torch.no_grad():
            z = self.adapter.encode(x)
        if isinstance(z, tuple):
            z = z[0]
        return z

    def forward_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, H, W, latent_dim) or (B, T, N, latent_dim) → logits (B, T)."""
        B, T = z.shape[:2]
        z_flat = z.reshape(B * T, -1, z.shape[-1])     # (B*T, N, latent_dim)
        logits = self.head(z_flat).squeeze(-1)           # (B*T, N)
        return logits.mean(dim=-1).reshape(B, T)         # (B, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, H, W, D) raw patch embeddings → logits (B, T)."""
        return self.forward_from_latent(self._encode(x))
