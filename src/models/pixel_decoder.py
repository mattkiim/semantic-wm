"""Lightweight LDM-style pixel decoder: compact latent → RGB image.

This module provides the direct pixel reconstruction path used in S-VAE and
PS-VAE evaluation (and PS-VAE training):

    fl  ∈ (B, T, H_l, W_l, d_l)
        ↓  PixelDecoder
    RGB ∈ (B, T, H_l·2^n, W_l·2^n, 3)

where ``n = len(channel_multipliers) - 1`` upsample steps.

Concretely, with the default ``channel_multipliers = (1, 1, 2, 4, 8)`` and an
encoder that produces 16×16 token grids (256×256 input, patch 16), the decoder
upsample chain is:  16 → 32 → 64 → 128 → 256 (four 2× steps).

Architecture (Rombach et al., 2022 LDM decoder):
  proj_in  : Conv2d(d_l, base_ch * ch_mult[-1], 1)
  mid      : ResBlock + AttnBlock + ResBlock
  up[i]    : num_res_blocks × ResBlock  [→ Upsample 2× if i > 0]
  norm_out : GroupNorm(32, base_ch)
  conv_out : Conv2d(base_ch, 3, 3, pad=1)  + Sigmoid (or Tanh rescaled)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ResBlock(nn.Module):
    """2-D residual block: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv + skip."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class _AttnBlock(nn.Module):
    """Multi-head spatial self-attention with residual (GroupNorm pre-norm).

    Uses ``F.scaled_dot_product_attention`` (flash / memory-efficient kernels)
    when available, falling back to manual computation otherwise.
    """

    def __init__(self, ch: int, num_heads: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(32, ch), ch, eps=1e-6, affine=True)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj_out = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        nh = self.num_heads
        head_dim = C // nh

        h = self.norm(x)
        q = self.q(h).reshape(B, nh, head_dim, H * W).permute(0, 1, 3, 2)  # (B, nh, HW, hd)
        k = self.k(h).reshape(B, nh, head_dim, H * W).permute(0, 1, 3, 2)
        v = self.v(h).reshape(B, nh, head_dim, H * W).permute(0, 1, 3, 2)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, nh, HW, hd)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj_out(out)


class _Upsample(nn.Module):
    """Nearest-neighbour 2× upsampling followed by a 3×3 Conv (no bias)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# ---------------------------------------------------------------------------
# PixelDecoder
# ---------------------------------------------------------------------------


class PixelDecoder(nn.Module):
    """LDM-style CNN decoder from compact latent ``fl`` to RGB pixels.

    Parameters
    ----------
    latent_dim : int
        Channel dimensionality ``d_l`` of the compact latent ``fl``.
    base_channels : int
        Base channel width ``C`` of the decoder.  Level ``i`` has
        ``C * channel_multipliers[i]`` channels.
    channel_multipliers : sequence of int
        Channel multipliers from *outer* (output) to *inner* (most compressed)
        level.  The number of 2× upsamplings is ``len(channel_multipliers) - 1``.
        Default ``(1, 1, 2, 4, 8)`` gives 4 upsamplings:
            16×16  →  32×32  →  64×64  →  128×128  →  256×256.
    num_res_blocks : int
        Number of residual blocks per resolution level.
    dropout : float
        Dropout probability inside residual blocks.
    output_activation : str
        ``"sigmoid"`` (output in [0, 1]) or ``"tanh"`` (output in [-1, 1],
        then rescaled to [0, 1] after the decoder forward call).
    """

    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
        temporal_upsample: bool = False,
        attn_resolutions: Sequence[int] = (16,),
        attn_heads: int = 1,
    ) -> None:
        super().__init__()

        self._latent_dim = latent_dim
        self._output_activation = output_activation
        self.temporal_upsample = temporal_upsample
        mults = tuple(channel_multipliers)

        if self.temporal_upsample:
            self.temporal_upsample_layer = nn.ConvTranspose1d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        # Channel widths per level (index 0 = outermost / output side)
        ch_per_level = [base_channels * m for m in mults]
        inner_ch = ch_per_level[-1]  # channel width at the deepest level

        # Number of 2× upsamplings — used to infer starting spatial resolution
        n_ups = len(mults) - 1

        # ---------- proj_in: d_l → inner_ch --------------------------------
        self.proj_in = nn.Conv2d(latent_dim, inner_ch, kernel_size=1)

        # ---------- mid block -----------------------------------------------
        self.mid_block1 = _ResBlock(inner_ch, inner_ch, dropout)
        self.mid_attn = _AttnBlock(inner_ch, num_heads=min(attn_heads, inner_ch))
        self.mid_block2 = _ResBlock(inner_ch, inner_ch, dropout)

        # ---------- upsample levels (built from inner → outer) --------------
        # We track spatial resolution to know when to insert attention blocks.
        # The input token grid is at the smallest resolution; each upsample
        # level doubles it.  We don't know the absolute input resolution at
        # construction time, so we store the *relative* level index and
        # convert ``attn_resolutions`` to a set of level indices in forward()
        # … Actually, it's simpler to store attn_resolutions and compute
        # the current resolution during __init__ assuming a canonical input
        # (e.g. 16×16 tokens → final 256).  The canonical input resolution is
        # ``target_output / 2^n_ups``.  We'll assume the common 16×16 start.
        #
        # Level ordering in the build loop (reversed range):
        #   iteration 0 → innermost (mults[-1]), res = start_res
        #   iteration 1 → next,                  res = start_res * 2  (after upsample)
        #   …
        # We compute resolution *after* upsampling at each level.
        attn_res_set = set(int(r) for r in attn_resolutions)

        self.up_levels = nn.ModuleList()
        current_ch = inner_ch
        # ``cur_res`` tracks spatial resolution *entering* each level.
        # We don't know the true input resolution yet, so we walk relative
        # to a canonical starting point and record which level indices need
        # attention.  However, level indices are deterministic, so we
        # instead compute the resolution each level outputs and check.

        # Resolution entering the first (innermost) level — unknown at init.
        # We store attn_resolutions and resolve at forward time? No — the
        # standard approach is to accept a ``resolution`` parameter that
        # gives the *input* spatial size to proj_in.
        # For backward compat we default to 16 (256-px input / patch 16).
        start_res = 16  # smallest spatial resolution (tokens)
        cur_res = start_res

        for level_idx, i in enumerate(reversed(range(len(mults)))):
            target_ch = ch_per_level[i]
            level = nn.ModuleDict()
            blocks = nn.ModuleList()
            for j in range(num_res_blocks):
                in_ch = current_ch if j == 0 else target_ch
                blocks.append(_ResBlock(in_ch, target_ch, dropout))
            level["blocks"] = blocks

            # Insert attention if current resolution is in attn_resolutions
            if cur_res in attn_res_set:
                nh = min(attn_heads, target_ch)
                level["attn"] = _AttnBlock(target_ch, num_heads=nh)

            current_ch = target_ch
            # Upsample on every level except the outermost
            if i != 0:
                level["upsample"] = _Upsample(current_ch)
                cur_res *= 2
            self.up_levels.append(level)

        # ---------- output head ---------------------------------------------
        self.norm_out = nn.GroupNorm(min(32, current_ch), current_ch, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(current_ch, 3, kernel_size=3, padding=1)

        logger.info(
            "PixelDecoder: d_l=%d  base_ch=%d  mults=%s  n_upsample=%d  attn_res=%s  attn_heads=%d",
            latent_dim, base_channels, mults, n_ups, attn_res_set, attn_heads,
        )

    # ------------------------------------------------------------------

    def forward(self, z_l: torch.Tensor) -> torch.Tensor:
        """Decode compact latent to RGB image.

        Parameters
        ----------
        z_l : (B, T, H_l, W_l, d_l)

        Returns
        -------
        rgb : (B, T, H_l·2^n, W_l·2^n, 3) — values in [0, 1]
        """
        B, T, H_l, W_l, C = z_l.shape

        if self.temporal_upsample:
            # (B, T, H_l, W_l, C) -> (B, H_l, W_l, C, T) -> (B*H_l*W_l, C, T)
            z_l = z_l.permute(0, 2, 3, 4, 1).reshape(B * H_l * W_l, C, T)
            z_l = self.temporal_upsample_layer(z_l)
            T = T * 2
            # back to (B, T, H_l, W_l, C)
            z_l = z_l.view(B, H_l, W_l, C, T).permute(0, 4, 1, 2, 3).contiguous()

        # Reshape to (B*T, d_l, H_l, W_l) for 2-D convolutions
        x = z_l.reshape(B * T, H_l, W_l, C).permute(0, 3, 1, 2).contiguous()

        # proj_in
        x = self.proj_in(x)

        # mid block
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # up levels
        for level in self.up_levels:
            for blk in level["blocks"]:
                x = blk(x)
            if "attn" in level:
                x = level["attn"](x)
            if "upsample" in level:
                x = level["upsample"](x)

        # output head
        x = self.conv_out(F.silu(self.norm_out(x)))

        if self._output_activation == "sigmoid":
            x = torch.sigmoid(x)
        else:  # tanh → rescale to [0, 1]
            x = (torch.tanh(x) + 1.0) / 2.0

        # Reshape back to (B, T, H, W, 3)
        _, _, H_out, W_out = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, T, H_out, W_out, 3)
        return x


# ---------------------------------------------------------------------------
# Factory / config helpers
# ---------------------------------------------------------------------------


def create_pixel_decoder(config: Dict[str, Any]) -> PixelDecoder:
    """Instantiate a PixelDecoder from a config dict.

    Keys consumed (all optional, defaults match PixelDecoder.__init__):
      ``"latent_dim"``, ``"base_channels"``, ``"channel_multipliers"``,
      ``"num_res_blocks"``, ``"dropout"``, ``"output_activation"``.
    """
    has_attn_cfg = "attn_resolutions" in config or "attn_heads" in config
    if has_attn_cfg:
        attn_resolutions = tuple(int(r) for r in config.get("attn_resolutions", (16,)))
        attn_heads = int(config.get("attn_heads", 1))
    else:
        # Backward compatibility: old checkpoints did not include per-level
        # attention config, so build the legacy no-attention architecture.
        attn_resolutions = ()
        attn_heads = 1

    return PixelDecoder(
        latent_dim=config["latent_dim"],
        base_channels=config.get("base_channels", 128),
        channel_multipliers=tuple(config.get("channel_multipliers", (1, 1, 2, 4, 8))),
        num_res_blocks=config.get("num_res_blocks", 2),
        dropout=config.get("dropout", 0.0),
        output_activation=config.get("output_activation", "sigmoid"),
        temporal_upsample=config.get("temporal_upsample", False),
        attn_resolutions=attn_resolutions,
        attn_heads=attn_heads,
    )


def pixel_decoder_config_from_args(args) -> Dict[str, Any]:
    """Extract pixel decoder config from parsed CLI args."""
    mults = getattr(args, "pixel_decoder_channel_multipliers", [1, 1, 2, 4, 8])
    attn_res = getattr(args, "pixel_decoder_attn_resolutions", [16])
    return {
        "latent_dim": getattr(args, "adapter_latent_dim", 96),
        "base_channels": getattr(args, "pixel_decoder_base_channels", 128),
        "channel_multipliers": tuple(mults),
        "num_res_blocks": getattr(args, "pixel_decoder_num_res_blocks", 2),
        "dropout": getattr(args, "pixel_decoder_dropout", 0.0),
        "output_activation": getattr(args, "pixel_decoder_output_activation", "sigmoid"),
        "temporal_upsample": (
            getattr(args, "encoder_type", "") == "qwen" and getattr(args, "qwen_mode", "video") == "video"
        ),
        "attn_resolutions": tuple(int(r) for r in attn_res),
        "attn_heads": getattr(args, "pixel_decoder_attn_heads", 1),
    }
