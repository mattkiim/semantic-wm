"""PatchGAN discriminator with spectral normalisation for adversarial adapter training.

Architecture (~2.8 M params):
    3→64 (k4,s2) → 64→128 (k4,s2,BN) → 128→256 (k4,s2,BN) → 256→1 (k4,s1)

All Conv2d layers use spectral normalisation.  Inner layers use BatchNorm +
LeakyReLU(0.2).  The final layer outputs a spatial map (PatchGAN).

Includes a VQGAN-style adaptive weight function that balances GAN loss vs
reconstruction loss by comparing gradient magnitudes at the last decoder layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """Patch-level discriminator with spectral normalisation."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 1, 4, stride=1, padding=1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (B, C, H, W)  — images in channel-first format.

        Returns
        -------
        (B, 1, H', W') patch-level logits.
        """
        return self.net(x)


def hinge_loss_disc(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discriminator training."""
    return (F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()) * 0.5


def hinge_loss_gen(fake_logits: torch.Tensor) -> torch.Tensor:
    """Hinge loss for generator (minimise -D(fake))."""
    return -fake_logits.mean()


def adaptive_weight(
    loss_rec: torch.Tensor,
    loss_gan: torch.Tensor,
    last_layer_weight: torch.Tensor,
    max_weight: float = 1e4,
) -> torch.Tensor:
    """VQGAN-style adaptive weight: ||∂L_rec/∂w|| / ||∂L_gan/∂w||.

    Parameters
    ----------
    loss_rec : scalar reconstruction loss.
    loss_gan : scalar generator GAN loss.
    last_layer_weight : parameter tensor (e.g. ``pixel_decoder.conv_out.weight``).
    max_weight : upper clamp value.

    Returns
    -------
    Scalar adaptive weight (detached).
    """
    rec_grad = torch.autograd.grad(loss_rec, last_layer_weight, retain_graph=True)[0]
    gan_grad = torch.autograd.grad(loss_gan, last_layer_weight, retain_graph=True)[0]

    ratio = torch.norm(rec_grad) / (torch.norm(gan_grad) + 1e-6)
    return torch.clamp(ratio, 0.0, max_weight).detach()
