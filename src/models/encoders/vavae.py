"""VA-VAE (Vision Foundation Model Aligned VAE) encoder/decoder wrapper.

Uses the VA-VAE f16d32 variant from LightningDiT (CVPR 2025).
Architecture is an enhanced KL-VAE with 32-channel latent space and
16x spatial compression, trained with DINOv2 feature alignment.

For a 256x256 input image:
    * Spatial compression: 16x → 16x16 latent grid
    * Latent channels: 32
    * Output: (B, T, 16, 16, 32)

The model source is vendored from ``hustvl/LightningDiT`` in
``_vavae_src/autoencoder.py``.
"""

import logging

import einops
import torch

from ..base_autoencoder import BaseAutoencoder

logger = logging.getLogger(__name__)


class VAVAEWrapper(BaseAutoencoder):
    """Frozen VA-VAE f16d32 autoencoder for latent diffusion.

    Parameters
    ----------
    checkpoint_path : str
        Path to the ``.pt`` checkpoint file.
    """

    def __init__(self, checkpoint_path: str):
        super().__init__()
        from ._vavae_src.autoencoder import AutoencoderKL

        logger.info("Loading VA-VAE f16d32 from %s", checkpoint_path)
        self.model = AutoencoderKL(
            embed_dim=32,
            ch_mult=(1, 1, 2, 2, 4),
            ckpt_path=checkpoint_path,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.model = self.model.to(torch.bfloat16)

        logger.info("VA-VAE f16d32 ready: latent_dim=32, spatial_ds=16x")

    @property
    def latent_dim(self) -> int:
        return 32

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to VA-VAE latents.

        Parameters
        ----------
        x : (B, T, H, W, C) float tensor in [0, 1].

        Returns
        -------
        z : (B, T, h, w, 32) where h = H/16, w = W/16.
        """
        B, T = x.shape[:2]
        # (B, T, H, W, C) → (B*T, C, H, W), scale [0,1] → [-1,1]
        x_in = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x_in = x_in * 2 - 1

        with torch.no_grad():
            posterior = self.model.encode(x_in)
            z = posterior.sample()

        # (B*T, 32, h, w) → (B, T, h, w, 32)
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode VA-VAE latents back to pixel space.

        Parameters
        ----------
        z : (B, T, h, w, 32) latent tensor.

        Returns
        -------
        x : (B, T, H, W, C) float tensor approximately in [0, 1].
        """
        B, T = z.shape[:2]
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")

        with torch.no_grad():
            x = self.model.decode(z_in)

        # [-1, 1] → [0, 1]
        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x
