"""Self-contained SigLIP / WebSSL encoder–decoder wrapper for latent diffusion.

Follows the same ``encode()`` / ``decode()`` interface as :class:`VAE` and
:class:`RAE` so it can be used as a drop-in replacement in the training loop.

No imports from Scale-RAE – only HuggingFace ``transformers`` models are used.
"""

import logging
from math import sqrt
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoImageProcessor, AutoModel, Dinov2Model

from ..base_autoencoder import BaseAutoencoder
from .rae import GeneralDecoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class SiglipEncoder(nn.Module):
    """Wraps a HuggingFace SigLIP2 vision model and returns patch features.

    For ``google/siglip2-so400m-patch14-224``:
        * 224×224 input, patch_size 14 → 16×16 = 256 patch tokens
        * hidden_size 1152
        * No CLS token (mean-pooling architecture)

    Features are taken from ``hidden_states[-1]`` (last encoder layer,
    *before* ``post_layernorm``), matching Scale-RAE's training convention.
    """

    def __init__(self, model_name: str = "google/siglip2-so400m-patch14-224"):
        super().__init__()
        full_model = AutoModel.from_pretrained(model_name)
        self.vision_tower = full_model.vision_model
        del full_model  # free any multi-modal wrapper

        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.hidden_size: int = self.vision_tower.config.hidden_size  # 1152
        self.patch_size: int = self.vision_tower.config.patch_size  # 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, H, W) normalised images.  Returns (B, N, D)."""
        outputs = self.vision_tower(x, output_hidden_states=True)
        # hidden_states[-1] = last encoder layer output (pre post_layernorm)
        features = outputs.hidden_states[-1]
        return features


class WebSSLEncoder(nn.Module):
    """Wraps a HuggingFace WebSSL / DINOv2 model and returns patch features.

    For ``facebook/webssl-dino300m-full2b-224``:
        * 224×224 input, patch_size 14 → 16×16 = 256 patch tokens
        * hidden_size 1024
        * First token is CLS → dropped

    Features are taken from ``last_hidden_state`` (includes final LayerNorm),
    matching Scale-RAE's training convention.
    """

    def __init__(self, model_name: str = "facebook/webssl-dino300m-full2b-224"):
        super().__init__()
        self.vision_tower = Dinov2Model.from_pretrained(model_name)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()

        self.hidden_size: int = self.vision_tower.config.hidden_size  # 1024
        self.patch_size: int = (
            self.vision_tower.embeddings.patch_embeddings.projection.stride[0]
        )  # 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, C, H, W) normalised images.  Returns (B, N, D)."""
        outputs = self.vision_tower(x)
        # Drop CLS token (index 0)
        features = outputs.last_hidden_state[:, 1:]
        return features


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ENCODER_REGISTRY = {
    "siglip": {
        "class": SiglipEncoder,
        "default_model_name": "google/siglip2-so400m-patch14-224",
    },
    "webssl": {
        "class": WebSSLEncoder,
        "default_model_name": "facebook/webssl-dino300m-full2b-224",
    },
}


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------


class ScaleRAE(BaseAutoencoder):
    """Encoder–decoder wrapper for SigLIP / WebSSL based RAE.

    Provides the same ``encode()`` / ``decode()`` interface as :class:`VAE`
    and :class:`RAE`, making it a drop-in replacement for the diffusion
    training loop.

    Parameters
    ----------
    encoder_name : ``"siglip"`` or ``"webssl"``
    encoder_model_name : HuggingFace model name override (optional).
    decoder_config_path : Path to the ViT-MAE-style decoder ``config.json``.
    pretrained_decoder_path : Path to the pretrained decoder ``model.pt``.
    encoder_input_size : Expected spatial resolution for the encoder (224).
    reshape_to_2d : If True, latents are returned as ``(B, T, h, w, C)``;
        otherwise ``(B, T, N, C)``.
    noise_tau : Noise regularisation strength (0 = disabled).
    normalization_stat_path : Path to per-channel latent normalisation stats.
    eps : Small constant for numerical stability in normalisation.
    """

    def __init__(
        self,
        encoder_name: str = "siglip",
        encoder_model_name: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        pretrained_decoder_path: Optional[str] = None,
        encoder_input_size: int = 224,
        reshape_to_2d: bool = True,
        noise_tau: float = 0.0,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()

        if encoder_name not in _ENCODER_REGISTRY:
            raise ValueError(
                f"Unknown encoder_name '{encoder_name}'. "
                f"Choose from: {list(_ENCODER_REGISTRY.keys())}"
            )

        entry = _ENCODER_REGISTRY[encoder_name]
        model_name = encoder_model_name or entry["default_model_name"]

        # ---- Encoder --------------------------------------------------------
        EncoderCls = entry["class"]
        self.encoder = EncoderCls(model_name)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.to(torch.bfloat16)

        # Fetch normalisation stats from the HF image processor so that
        # ``encode()`` can normalise raw [0, 1] frames with pure torch ops.
        proc = AutoImageProcessor.from_pretrained(model_name)
        self.register_buffer(
            "encoder_mean",
            torch.tensor(proc.image_mean).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor(proc.image_std).view(1, 3, 1, 1),
        )

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size: int = self.encoder.patch_size
        self._latent_dim: int = self.encoder.hidden_size
        self.base_patches: int = (
            self.encoder_input_size // self.encoder_patch_size
        ) ** 2  # 256

        # ---- Decoder --------------------------------------------------------
        if decoder_config_path is None:
            raise ValueError(
                "decoder_config_path is required for ScaleRAE. "
                "Point it to the ViT-MAE config.json that matches the "
                "pretrained decoder."
            )

        decoder_config = AutoConfig.from_pretrained(decoder_config_path)

        # Infer decoder sizes from the checkpoint if available
        state_dict = None
        if pretrained_decoder_path is not None:
            state_dict = torch.load(pretrained_decoder_path, map_location="cpu")
            if "decoder_embed.weight" in state_dict:
                decoder_config.decoder_hidden_size = state_dict[
                    "decoder_embed.weight"
                ].shape[0]
            if "decoder_layers.0.intermediate.dense.weight" in state_dict:
                decoder_config.decoder_intermediate_size = state_dict[
                    "decoder_layers.0.intermediate.dense.weight"
                ].shape[0]

        # Ensure config matches the encoder we are using
        decoder_config.hidden_size = self._latent_dim
        decoder_config.image_size = int(
            self.encoder_patch_size * sqrt(self.base_patches)
        )

        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        if state_dict is not None:
            self.decoder.load_state_dict(state_dict, strict=False)
            logger.info("Loaded pretrained decoder from %s", pretrained_decoder_path)

        self.decoder.eval()
        self.decoder.requires_grad_(False)
        self.decoder.to(torch.bfloat16)

        # Keep decoder on CPU between validation calls to save GPU memory.
        self._decoder_device = torch.device("cpu")
        self.decoder.cpu()

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d

        # Optional per-channel latent normalisation
        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.register_buffer("latent_mean", stats.get("mean", None))
            self.register_buffer("latent_std", stats.get("std", None))
            self.do_normalization = self.latent_std is not None
        else:
            self.do_normalization = False
        self.eps = eps

        if self.do_normalization:
            logger.info("Loaded latent normalization stats from %s", normalization_stat_path)
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (x.dim() - 1), device=x.device
        )
        return x + sigma * torch.randn_like(x)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames into latent tokens.

        Parameters
        ----------
        x : (B, T, H, W, C) float tensor in [0, 1].

        Returns
        -------
        z : (B, T, h, w, C_latent) if ``reshape_to_2d`` else (B, T, N, C_latent).
        """
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> (b t) c h w")

        # Auto-resize to encoder's expected resolution
        if H != self.encoder_input_size or W != self.encoder_input_size:
            x = F.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )

        # Normalise with encoder-specific stats (pure torch, GPU-resident)
        x = (x - self.encoder_mean) / self.encoder_std

        with torch.no_grad():
            z = self.encoder(x)  # (BT, N, D)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)

        if self.reshape_to_2d:
            bt, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(bt, c, h, w)

        if self.do_normalization:
            lm = self.latent_mean if self.latent_mean is not None else 0
            lv = self.latent_std if self.latent_std is not None else 1
            z = (z - lm) / torch.sqrt(lv + self.eps)

        if self.reshape_to_2d:
            z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        else:
            z = einops.rearrange(z, "(b t) n c -> b t n c", b=B, t=T)
        return z

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent tokens back to pixel space.

        Parameters
        ----------
        z : (B, T, H, W, C) latent tensor.

        Returns
        -------
        x_rec : (B, T, H, W, 3) float tensor in approximately [0, 1].
        """
        B, T, H, W, C = z.shape
        if self.reshape_to_2d:
            z = einops.rearrange(z, "b t h w c -> (b t) c h w")
        else:
            z = einops.rearrange(z, "b t n c -> (b t) n c")

        if self.do_normalization:
            lm = self.latent_mean if self.latent_mean is not None else 0
            lv = self.latent_std if self.latent_std is not None else 1
            z = z * torch.sqrt(lv + self.eps) + lm

        if self.reshape_to_2d:
            bt, c, h, w = z.shape
            z = z.view(bt, c, h * w).transpose(1, 2)  # (BT, N, D)

        # Move decoder to GPU on-demand
        target_device = z.device
        if self._decoder_device != target_device:
            self.decoder.to(target_device)
            self._decoder_device = target_device

        with torch.no_grad():
            output = self.decoder(z, drop_cls_token=False).logits
            x_rec = self.decoder.unpatchify(output)
            # Undo encoder normalisation → ~[0, 1]
            x_rec = x_rec * self.encoder_std + self.encoder_mean

        # Move decoder back to CPU to free GPU memory
        self.decoder.cpu()
        self._decoder_device = torch.device("cpu")
        torch.cuda.empty_cache()

        x_rec = einops.rearrange(x_rec, "(b t) c h w -> b t h w c", b=B, t=T)
        return x_rec
