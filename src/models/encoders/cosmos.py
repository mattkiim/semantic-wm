"""Cosmos Tokenizer CI16x16 encoder/decoder wrapper.

Uses NVIDIA's Cosmos continuous image tokenizer (16x spatial compression,
16-channel latent) via JIT-compiled encoder/decoder checkpoints.

For a 256x256 input image:
    * Spatial compression: 16x → 16x16 latent grid
    * Latent channels: 16
    * Output: (B, T, 16, 16, 16)

Checkpoints are loaded via ``torch.jit.load`` — no external
``cosmos_tokenizer`` package is required.
"""

import logging
import os

import einops
import torch

from ..base_autoencoder import BaseAutoencoder

logger = logging.getLogger(__name__)


class CosmosTokenizerWrapper(BaseAutoencoder):
    """Frozen Cosmos CI16x16 tokenizer for latent diffusion.

    Parameters
    ----------
    checkpoint_dir : str
        Directory containing ``encoder.jit`` and ``decoder.jit``.
    """

    def __init__(self, checkpoint_dir: str):
        super().__init__()
        enc_path = os.path.join(checkpoint_dir, "encoder.jit")
        dec_path = os.path.join(checkpoint_dir, "decoder.jit")

        logger.info("Loading Cosmos CI16x16 encoder from %s", enc_path)
        self.encoder = torch.jit.load(enc_path, map_location="cpu")
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

        logger.info("Loading Cosmos CI16x16 decoder from %s", dec_path)
        self.decoder = torch.jit.load(dec_path, map_location="cpu")
        self.decoder.eval()
        for p in self.decoder.parameters():
            p.requires_grad_(False)

        self.encoder = self.encoder.to(torch.bfloat16)
        self.decoder = self.decoder.to(torch.bfloat16)

        logger.info("Cosmos CI16x16 tokenizer ready: latent_dim=16, spatial_ds=16x")

    @property
    def latent_dim(self) -> int:
        return 16

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to Cosmos latents.

        Parameters
        ----------
        x : (B, T, H, W, C) float tensor in [0, 1].

        Returns
        -------
        z : (B, T, h, w, 16) where h = H/16, w = W/16.
        """
        B, T = x.shape[:2]
        # (B, T, H, W, C) → (B*T, C, H, W), scale [0,1] → [-1,1]
        x_in = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x_in = x_in * 2 - 1

        with torch.no_grad():
            z = self.encoder(x_in)
            if isinstance(z, tuple):
                z = z[0]

        # (B*T, 16, h, w) → (B, T, h, w, 16)
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode Cosmos latents back to pixel space.

        Parameters
        ----------
        z : (B, T, h, w, 16) latent tensor.

        Returns
        -------
        x : (B, T, H, W, C) float tensor approximately in [0, 1].
        """
        B, T = z.shape[:2]
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")

        with torch.no_grad():
            x = self.decoder(z_in)
            if isinstance(x, tuple):
                x = x[0]

        # [-1, 1] → [0, 1]
        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x
