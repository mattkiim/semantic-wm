import einops
import torch
from torch import nn
from diffusers.models import AutoencoderKL

from ..base_autoencoder import BaseAutoencoder

_DEFAULT_VAE_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
_DEFAULT_VAE_SUBFOLDER = "vae"


class VAE(BaseAutoencoder):
    def __init__(
        self,
        model_path: str = _DEFAULT_VAE_MODEL,
        subfolder: str | None = _DEFAULT_VAE_SUBFOLDER,
    ):
        super().__init__()
        kwargs = {"subfolder": subfolder} if subfolder else {}
        self.vae = AutoencoderKL.from_pretrained(model_path, **kwargs)
        self.vae.eval().requires_grad_(False)
        self.vae.to(torch.bfloat16)

    @property
    def latent_dim(self) -> int:
        return self.vae.config.latent_channels

    def _chunked(self, fn, x: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        if x.shape[0] <= chunk:
            return fn(x)
        return torch.cat([fn(x[i:i + chunk]) for i in range(0, x.shape[0], chunk)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x_in = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x_in = x_in * 2 - 1

        with torch.no_grad():
            z = self._chunked(lambda x: self.vae.encode(x).latent_dist.sample(), x_in)

        z = z * self.vae.config.scaling_factor
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = z.shape
        z_in = einops.rearrange(z, "b t h w c -> (b t) c h w")
        z_in = z_in / self.vae.config.scaling_factor

        with torch.no_grad():
            x = self._chunked(lambda x: self.vae.decode(x, return_dict=False)[0], z_in)

        x = (x + 1) / 2
        x = einops.rearrange(x, "(b t) c h w -> b t h w c", b=B, t=T)
        return x
