"""Adapter layers for projecting high-dimensional RAE latents to compact
diffusion-friendly spaces and back.

Hierarchy
---------
BaseAdapter          (ABC)
├── IdentityAdapter   – pass-through, used for VAE or no projection
├── MLPAdapter        – simple Linear → GELU → Linear
└── SVAEAdapter       – Semantic VAE with Transformer blocks + diagonal Gaussian

Factory
-------
``create_adapter(config, input_dim)`` instantiates the right adapter from a
config dict, mirroring ``create_autoencoder`` in ``base_autoencoder.py``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders.rae import ViTMAEConfig, ViTMAELayer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseAdapter(nn.Module, ABC):
    """Projects autoencoder latent → compact space (encode) and back (decode).

    During training the adapter's ``encode`` may return auxiliary outputs
    (e.g. mu, logvar for KL loss).  During eval it returns only the latent.
    """

    @abstractmethod
    def encode(self, z: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Map high-dim features to compact latent.

        Parameters
        ----------
        z : (B, T, H, W, C_high) or (B, T, N, C_high)

        Returns
        -------
        z_l          during eval: (B, T, H, W, C_low) or (B, T, N, C_low)
        (z_l, ...)   during training: may include mu, logvar, etc.
        """
        ...

    @abstractmethod
    def decode(self, z_l: torch.Tensor) -> torch.Tensor:
        """Reconstruct high-dim features from compact latent.

        Parameters
        ----------
        z_l : (B, T, H, W, C_low) or (B, T, N, C_low)

        Returns
        -------
        z_rec : same shape as the original high-dim input.
        """
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Compact latent channel dimensionality (d_l)."""
        ...


# ---------------------------------------------------------------------------
# IdentityAdapter
# ---------------------------------------------------------------------------


class IdentityAdapter(BaseAdapter):
    """Pass-through adapter — no projection.  Used when no dim reduction is
    needed (e.g. for the standard VAE whose latent is already small)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self._latent_dim = input_dim

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def decode(self, z_l: torch.Tensor) -> torch.Tensor:
        return z_l


# ---------------------------------------------------------------------------
# MLPAdapter
# ---------------------------------------------------------------------------


class MLPAdapter(BaseAdapter):
    """Simple MLP projection: d_h → hidden → d_l  (and symmetric decoder).

    Both encoder and decoder are ``Linear → GELU → Linear``.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 96,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or (input_dim + latent_dim) // 2
        self._latent_dim = latent_dim

        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def _reshape_encode(self, z: torch.Tensor):
        """Handle BTHWC → BT(HW)C flattening so MLP sees (*, C)."""
        shape = z.shape
        is_spatial = z.dim() == 5  # (B, T, H, W, C)
        if is_spatial:
            B, T, H, W, C = shape
            z = z.reshape(B * T, H * W, C)
        else:
            B, T, N, C = shape
            z = z.reshape(B * T, N, C)
        return z, shape, is_spatial

    def _reshape_decode(
        self, z: torch.Tensor, orig_shape, is_spatial: bool, new_c: int
    ):
        if is_spatial:
            B, T, H, W, _ = orig_shape
            return z.reshape(B, T, H, W, new_c)
        else:
            B, T, N, _ = orig_shape
            return z.reshape(B, T, N, new_c)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        z_flat, orig_shape, is_spatial = self._reshape_encode(z)
        z_l = self.encoder_mlp(z_flat)
        return self._reshape_decode(z_l, orig_shape, is_spatial, self._latent_dim)

    def decode(self, z_l: torch.Tensor) -> torch.Tensor:
        orig_shape = z_l.shape
        is_spatial = z_l.dim() == 5
        if is_spatial:
            B, T, H, W, C = z_l.shape
            z_flat = z_l.reshape(B * T, H * W, C)
        else:
            B, T, N, C = z_l.shape
            z_flat = z_l.reshape(B * T, N, C)
        z_rec = self.decoder_mlp(z_flat)
        input_dim = z_rec.shape[-1]
        if is_spatial:
            return z_rec.reshape(B, T, H, W, input_dim)
        else:
            return z_rec.reshape(B, T, N, input_dim)


# ---------------------------------------------------------------------------
# DiagonalGaussian
# ---------------------------------------------------------------------------


class DiagonalGaussian(nn.Module):
    """Splits input into mu + logvar, samples via reparameterisation trick.

    Input : (*, 2 * d_l)
    Output: z (*, d_l),  mu (*, d_l),  logvar (*, d_l)

    Note: ``self.training`` is managed by ``nn.Module``.  The explicit
    ``super().__init__()`` call below ensures ``self.training = True`` is set
    at construction time, and ``nn.Module.train()`` / ``nn.Module.eval()``
    propagate the flag recursively to this module as a submodule of
    ``SVAEAdapter``.
    """

    def __init__(self) -> None:
        super().__init__()  # initialises self.training = True via nn.Module

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = x.chunk(2, dim=-1)
        if self.training:  # set/cleared by .train() / .eval() on parent module
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu  # deterministic at eval
        return z, mu, logvar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _best_head_count(dim: int, max_heads: int = 16) -> int:
    """Return the largest number of heads ≤ *max_heads* that divides *dim*."""
    for h in range(min(max_heads, dim), 0, -1):
        if dim % h == 0:
            return h
    return 1


# ---------------------------------------------------------------------------
# SVAEAdapter  (S-VAE / PS-VAE)
# ---------------------------------------------------------------------------


class SVAEAdapter(BaseAdapter):
    """Semantic VAE adapter following the PS-VAE paper.

    **Encoder Es** (d_h → d_l):
        1. Transformer blocks at dim d_h
        2. LayerNorm
        3. Linear d_h → 2 * d_l  (mu + logvar)
        4. DiagonalGaussian → z_l

    With ``progressive=True`` (gradual dimension reduction):
        1. ``enc_layers_high`` Transformer blocks at d_h
        2. Linear d_h → d_mid, LayerNorm
        3. ``enc_layers_mid`` Transformer blocks at d_mid
        4. Linear d_mid → 2 * d_l, DiagonalGaussian

    With ``latent_layers > 0`` (latent-space refinement):
        After Gaussian sampling, ``latent_layers`` lightweight Transformer
        blocks at d_l refine the compressed representation.

    **Decoder Ds** mirrors the encoder structure.

    Parameters
    ----------
    input_dim  : d_h — high-dim feature size (e.g. 1024 for WebSSL)
    latent_dim : d_l — compact latent size (default 96)
    num_heads  : attention heads in Transformer blocks
    num_layers : number of Transformer blocks in encoder / decoder (default 3)
    intermediate_size : FFN hidden dim inside Transformer blocks
    progressive : enable gradual dim reduction via an intermediate stage
    mid_dim : intermediate dimension for progressive mode (default: geometric mean
              of d_h and d_l, rounded to nearest multiple of mid_heads)
    mid_heads : attention heads for mid-dim blocks (default: auto from mid_dim)
    latent_layers : number of refinement Transformer blocks at d_l (default 0)
    latent_heads : attention heads for latent-dim blocks (default: auto)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 96,
        num_heads: int = 12,
        num_layers: int = 3,
        intermediate_size: int = 3072,
        progressive: bool = False,
        mid_dim: Optional[int] = None,
        mid_heads: Optional[int] = None,
        latent_layers: int = 0,
        latent_heads: Optional[int] = None,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._latent_dim = latent_dim
        self._progressive = progressive
        self._latent_layers = latent_layers

        # Build a minimal ViTMAEConfig for the high-dim Transformer blocks
        block_cfg = ViTMAEConfig(
            hidden_size=input_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
        )

        if progressive:
            # ---- Progressive: split layers between high-dim and mid-dim ------
            # At least 1 layer at each scale; extra layers go to high-dim
            enc_layers_high = max(num_layers - 1, 1)
            enc_layers_mid = max(num_layers - enc_layers_high, 1)

            # Intermediate dimension: geometric mean, rounded to be head-divisible
            if mid_dim is None:
                mid_dim = int(sqrt(input_dim * latent_dim))
            if mid_heads is None:
                # Pick largest head count ≤ 16 that divides mid_dim
                mid_heads = _best_head_count(mid_dim, max_heads=16)
            mid_dim = (mid_dim // mid_heads) * mid_heads  # ensure divisibility
            self._mid_dim = mid_dim

            mid_cfg = ViTMAEConfig(
                hidden_size=mid_dim,
                num_hidden_layers=enc_layers_mid,
                num_attention_heads=mid_heads,
                intermediate_size=mid_dim * 4,
            )

            # ---- Encoder (progressive) ------
            self.enc_blocks = nn.ModuleList(
                [ViTMAELayer(block_cfg) for _ in range(enc_layers_high)]
            )
            self.enc_norm_high = nn.LayerNorm(input_dim)
            self.enc_down = nn.Linear(input_dim, mid_dim)
            self.enc_blocks_mid = nn.ModuleList(
                [ViTMAELayer(mid_cfg) for _ in range(enc_layers_mid)]
            )
            self.enc_norm = nn.LayerNorm(mid_dim)
            self.enc_proj = nn.Linear(mid_dim, latent_dim * 2)

            # ---- Decoder (progressive) ------
            self.dec_proj = nn.Linear(latent_dim, mid_dim)
            self.dec_blocks_mid = nn.ModuleList(
                [ViTMAELayer(mid_cfg) for _ in range(enc_layers_mid)]
            )
            self.dec_norm_mid = nn.LayerNorm(mid_dim)
            self.dec_up = nn.Linear(mid_dim, input_dim)
            self.dec_blocks = nn.ModuleList(
                [ViTMAELayer(block_cfg) for _ in range(enc_layers_high)]
            )
            self.dec_norm = nn.LayerNorm(input_dim)

            logger.info(
                "SVAEAdapter progressive: d_h=%d (%d layers) → d_mid=%d (%d layers) → d_l=%d",
                input_dim, enc_layers_high, mid_dim, enc_layers_mid, latent_dim,
            )
        else:
            # ---- Original (flat) architecture --------------------------------
            self.enc_blocks = nn.ModuleList(
                [ViTMAELayer(block_cfg) for _ in range(num_layers)]
            )
            self.enc_norm = nn.LayerNorm(input_dim)
            self.enc_proj = nn.Linear(input_dim, latent_dim * 2)

            self.dec_proj = nn.Linear(latent_dim, input_dim)
            self.dec_blocks = nn.ModuleList(
                [ViTMAELayer(block_cfg) for _ in range(num_layers)]
            )
            self.dec_norm = nn.LayerNorm(input_dim)

        # ---- Gaussian sampler ------------------------------------------------
        self.gaussian = DiagonalGaussian()

        # ---- Latent-space refinement blocks ----------------------------------
        if latent_layers > 0:
            if latent_heads is None:
                latent_heads = _best_head_count(latent_dim, max_heads=8)
            lat_cfg = ViTMAEConfig(
                hidden_size=latent_dim,
                num_hidden_layers=latent_layers,
                num_attention_heads=latent_heads,
                intermediate_size=latent_dim * 4,
            )
            self.latent_enc_blocks = nn.ModuleList(
                [ViTMAELayer(lat_cfg) for _ in range(latent_layers)]
            )
            self.latent_enc_norm = nn.LayerNorm(latent_dim)
            self.latent_dec_blocks = nn.ModuleList(
                [ViTMAELayer(lat_cfg) for _ in range(latent_layers)]
            )
            self.latent_dec_norm = nn.LayerNorm(latent_dim)
            logger.info(
                "SVAEAdapter latent refinement: %d blocks at d_l=%d (heads=%d)",
                latent_layers, latent_dim, latent_heads,
            )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # -- train / eval propagation (explicit override for clarity) ------------

    def train(self, mode: bool = True) -> "SVAEAdapter":
        """Propagate training mode to all sub-modules (including self.gaussian)
        so the reparameterisation trick in DiagonalGaussian is toggled
        correctly.  nn.Module.train() already handles this recursively; the
        override makes the intent explicit."""
        super().train(mode)
        return self

    def eval(self) -> "SVAEAdapter":
        """Switch to eval mode — DiagonalGaussian will return mu instead of
        sampling, and SVAEAdapter.encode returns z_l only (not a tuple)."""
        return self.train(False)

    # -- helpers to flatten / unflatten BTHWC ↔ (BT, N, C) --

    @staticmethod
    def _to_seq(z: torch.Tensor):
        """(B,T,H,W,C) → ((BT), N, C) with metadata to reverse."""
        if z.dim() == 5:
            B, T, H, W, C = z.shape
            return z.reshape(B * T, H * W, C), (B, T, H, W, C), True
        B, T, N, C = z.shape
        return z.reshape(B * T, N, C), (B, T, N, C), False

    @staticmethod
    def _from_seq(z: torch.Tensor, meta, is_spatial: bool, new_c: int):
        if is_spatial:
            B, T, H, W, _ = meta
            return z.reshape(B, T, H, W, new_c)
        B, T, N, _ = meta
        return z.reshape(B, T, N, new_c)

    # -----------------------------------------------------------------

    def encode(
        self, z: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Encode high-dim features → compact latent.

        Returns
        -------
        Training : (z_l, mu, logvar)  — all shaped (B, T, H, W, d_l) or (B, T, N, d_l)
        Eval     : z_l
        """
        x, meta, is_spatial = self._to_seq(z)

        # High-dim blocks
        for blk in self.enc_blocks:
            x = blk(x)[0]

        if self._progressive:
            x = self.enc_norm_high(x)
            x = self.enc_down(x)  # d_h → d_mid
            for blk in self.enc_blocks_mid:
                x = blk(x)[0]

        x = self.enc_norm(x)
        x = self.enc_proj(x)  # → 2*d_l

        z_l, mu, logvar = self.gaussian(x)

        # Latent-space refinement
        if self._latent_layers > 0:
            z_l = self._refine_latent_enc(z_l)

        z_l = self._from_seq(z_l, meta, is_spatial, self._latent_dim)

        if self.training:
            mu = self._from_seq(mu, meta, is_spatial, self._latent_dim)
            logvar = self._from_seq(logvar, meta, is_spatial, self._latent_dim)
            return z_l, mu, logvar

        return z_l

    def _refine_latent_enc(self, z_l: torch.Tensor) -> torch.Tensor:
        """Apply latent-space refinement blocks (encoder side)."""
        for blk in self.latent_enc_blocks:
            z_l = blk(z_l)[0]
        return self.latent_enc_norm(z_l)

    def _refine_latent_dec(self, z_l: torch.Tensor) -> torch.Tensor:
        """Apply latent-space refinement blocks (decoder side)."""
        for blk in self.latent_dec_blocks:
            z_l = blk(z_l)[0]
        return self.latent_dec_norm(z_l)

    def decode(self, z_l: torch.Tensor) -> torch.Tensor:
        """Decode compact latent → reconstructed high-dim features."""
        x, meta, is_spatial = self._to_seq(z_l)

        # Latent-space refinement (decoder side)
        if self._latent_layers > 0:
            x = self._refine_latent_dec(x)

        x = self.dec_proj(x)  # d_l → d_mid (progressive) or d_h (flat)

        if self._progressive:
            for blk in self.dec_blocks_mid:
                x = blk(x)[0]
            x = self.dec_norm_mid(x)
            x = self.dec_up(x)  # d_mid → d_h

        for blk in self.dec_blocks:
            x = blk(x)[0]
        x = self.dec_norm(x)

        return self._from_seq(x, meta, is_spatial, self._input_dim)


# ---------------------------------------------------------------------------
# KL divergence helper
# ---------------------------------------------------------------------------


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(q(z|x) || N(0,I)) for diagonal Gaussian.

    Parameters
    ----------
    mu, logvar : (B, T, *, d_l)

    Returns
    -------
    Scalar mean KL.
    """
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_adapter(config: Dict[str, Any], input_dim: int) -> BaseAdapter:
    """Instantiate an adapter from a config dict.

    Parameters
    ----------
    config : dict
        ``"adapter_type"`` — one of ``"identity"``, ``"mlp"``, ``"svae"``.
        Adapter-specific keys are read from the same dict.
    input_dim : int
        The autoencoder's ``latent_dim`` (d_h).

    Returns
    -------
    BaseAdapter
    """
    adapter_type = config.get("adapter_type", "identity")

    if adapter_type == "identity":
        return IdentityAdapter(input_dim)

    if adapter_type == "mlp":
        return MLPAdapter(
            input_dim=input_dim,
            latent_dim=config.get("adapter_latent_dim", 96),
            hidden_dim=config.get("adapter_hidden_dim"),
        )

    if adapter_type == "svae":
        return SVAEAdapter(
            input_dim=input_dim,
            latent_dim=config.get("adapter_latent_dim", 96),
            num_heads=config.get("adapter_num_heads", 12),
            num_layers=config.get("adapter_num_layers", 3),
            intermediate_size=config.get("adapter_intermediate_size", 3072),
            progressive=config.get("adapter_progressive", False),
            mid_dim=config.get("adapter_mid_dim"),
            mid_heads=config.get("adapter_mid_heads"),
            latent_layers=config.get("adapter_latent_layers", 0),
            latent_heads=config.get("adapter_latent_heads"),
        )

    raise ValueError(f"Unknown adapter type: {adapter_type}")


def adapter_config_from_args(args) -> Dict[str, Any]:
    """Convert CLI args into an adapter config dict."""
    return {
        "adapter_type": getattr(args, "adapter_type", "identity"),
        "adapter_latent_dim": getattr(args, "adapter_latent_dim", 96),
        "adapter_hidden_dim": getattr(args, "adapter_hidden_dim", None),
        "adapter_num_heads": getattr(args, "adapter_num_heads", 12),
        "adapter_num_layers": getattr(args, "adapter_num_layers", 3),
        "adapter_intermediate_size": getattr(args, "adapter_intermediate_size", 3072),
        "adapter_progressive": getattr(args, "adapter_progressive", False),
        "adapter_mid_dim": getattr(args, "adapter_mid_dim", None),
        "adapter_mid_heads": getattr(args, "adapter_mid_heads", None),
        "adapter_latent_layers": getattr(args, "adapter_latent_layers", 0),
        "adapter_latent_heads": getattr(args, "adapter_latent_heads", None),
    }
