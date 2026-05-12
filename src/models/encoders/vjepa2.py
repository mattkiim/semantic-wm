"""V-JEPA 2.1 ViT-L/16 encoder wrapper for latent diffusion (image mode).

Uses vendored V-JEPA 2.1 source from ``facebookresearch/vjepa2`` (stored
in ``_vjepa2_src/``) to avoid torch.hub downloads and ``sys.path`` conflicts.

The model uses RoPE (rotary positional embeddings) computed from grid
coordinates, so it naturally handles input resolutions different from its
training resolution (384×384).  We feed 256×256 frames to obtain a 16×16
spatial token grid.

Encoding is done in **image mode**: each frame is fed individually as
``(B*T, C, 1, H, W)``, triggering V-JEPA's ``patch_embed_img`` path
(PatchEmbed3D with ``tubelet_size=1``).  This produces one latent per
input frame with no temporal downsampling.  The final output is
normalised by V-JEPA's built-in LayerNorm (``norms_block[-1]``).

For V-JEPA 2.1 ViT-L/16:
    * Trained at 384×384 (24×24 patch grid)
    * With 256×256 input → 16×16 = 256 spatial tokens per frame
    * Image mode → 1 latent per frame (no temporal downsampling)
    * embed_dim 1024
    * Output: (B, T, 16, 16, 1024)
"""

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F

from ..base_autoencoder import BaseAutoencoder

logger = logging.getLogger(__name__)

# Checkpoint URLs on dl.fbaipublicfiles.com
_CHECKPOINT_URLS = {
    "vitl": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
    "vitb": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
}


def _load_vjepa2_1_encoder(
    model_size: str = "vitl",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
):
    """Instantiate and load a V-JEPA 2.1 ViT encoder."""
    from ._vjepa2_src import vision_transformer as vit_module

    vit_factory = {"vitl": "vit_large", "vitb": "vit_base"}
    if model_size not in vit_factory:
        raise ValueError(
            f"Unknown model_size={model_size!r}. Choose from {list(vit_factory)}"
        )

    arch_fn = getattr(vit_module, vit_factory[model_size])
    encoder = arch_fn(
        patch_size=16,
        img_size=(384, 384),  # training resolution (RoPE adapts to actual input)
        num_frames=64,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    # ---- Load pretrained weights ----
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info("Loading V-JEPA 2.1 weights from local path: %s", checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    else:
        url = _CHECKPOINT_URLS.get(model_size)
        if url is None:
            raise ValueError(f"No checkpoint URL for model_size={model_size}")
        logger.info("Downloading V-JEPA 2.1 weights from: %s", url)
        state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)

    # The checkpoint stores encoder under "ema_encoder" key
    encoder_key = "ema_encoder"
    if encoder_key in state_dict:
        encoder_sd = state_dict[encoder_key]
    elif "encoder" in state_dict:
        encoder_sd = state_dict["encoder"]
    else:
        encoder_sd = state_dict

    # Clean prefixes (module., backbone.)
    cleaned = {}
    for k, v in encoder_sd.items():
        k = k.replace("module.", "").replace("backbone.", "")
        cleaned[k] = v

    encoder.load_state_dict(cleaned, strict=True)
    logger.info("Loaded V-JEPA 2.1 encoder weights (strict=True)")

    return encoder


class VJEPA2EncoderWrapper(BaseAutoencoder):
    """Frozen V-JEPA 2.1 vision encoder for latent diffusion (image mode).

    Each frame is encoded independently via V-JEPA's image embedding path
    (``patch_embed_img`` with ``tubelet_size=1``), producing one latent per
    frame.  The output is normalised by the final LayerNorm
    (``norms_block[-1]``).

    The model was trained at 384×384 but uses RoPE, so it naturally adapts
    to 256×256 input, producing a 16×16 spatial token grid.

    Parameters
    ----------
    model_size : str
        ``"vitl"`` (300M, ViT-L/16) or ``"vitb"`` (80M, ViT-B/16).
    checkpoint_path : str or None
        Local path to a ``.pt`` checkpoint.  If ``None``, weights are
        downloaded from ``dl.fbaipublicfiles.com``.
    input_size : int
        Spatial resolution to feed the encoder.  Default 256.
    """

    def __init__(
        self,
        model_size: str = "vitl",
        checkpoint_path: Optional[str] = None,
        input_size: int = 256,
    ):
        super().__init__()
        self.has_decoder = False
        self._input_size = input_size
        self._patch_size = 16

        logger.info("Loading V-JEPA 2.1 %s encoder ...", model_size)
        self.model = _load_vjepa2_1_encoder(
            model_size=model_size,
            checkpoint_path=checkpoint_path,
        )
        self.model.eval()
        self.model.requires_grad_(False)

        self._latent_dim: int = self.model.embed_dim  # 1024 for ViT-L
        self._spatial_tokens = self._input_size // self._patch_size

        # Cast to bfloat16 for memory efficiency
        self.model = self.model.to(torch.bfloat16)

        # ImageNet normalisation stats (registered as buffers for device tracking)
        self.register_buffer(
            "encoder_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 1, 3),
        )
        self.register_buffer(
            "encoder_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 1, 3),
        )

        logger.info(
            "V-JEPA 2.1 encoder ready (image mode): embed_dim=%d, patch=%d, "
            "input=%d, spatial_grid=%dx%d, temporal_ds=1",
            self._latent_dim,
            self._patch_size,
            self._input_size,
            self._spatial_tokens,
            self._spatial_tokens,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def temporal_downsample_factor(self) -> int:
        return 1  # image mode: one latent per frame

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video frames to spatial token grids (image mode).

        Each frame is encoded independently via V-JEPA's image embedding
        path (``patch_embed_img``, ``tubelet_size=1``).  The output is
        normalised by V-JEPA's final LayerNorm (``norms_block[-1]``).

        Parameters
        ----------
        x : (B, T, H, W, C) float tensor in [0, 1].

        Returns
        -------
        z : (B, T, h, w, embed_dim)
            where h = w = input_size // patch_size  (16 for 256px input).
        """
        B, T, H, W, C = x.shape

        # ---- Resize to target resolution if needed ----
        if H != self._input_size or W != self._input_size:
            x_r = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x_r = F.interpolate(
                x_r,
                size=(self._input_size, self._input_size),
                mode="bicubic",
                align_corners=False,
            )
            x = x_r.permute(0, 2, 3, 1).reshape(
                B, T, self._input_size, self._input_size, C
            )

        # ---- Normalise with ImageNet stats ----
        x = x.clamp(0, 1)
        x = (x - self.encoder_mean) / self.encoder_std

        # ---- Reshape to (B*T, C, 1, H, W) for image-mode encoding ----
        x = x.permute(0, 1, 4, 2, 3)                # (B, T, C, H, W)
        x = x.reshape(B * T, C, 1, self._input_size, self._input_size)
        x = x.to(dtype=self.model.patch_embed_img.proj.weight.dtype)

        # ---- Forward pass (image mode) ----
        # Input (B*T, C, 1, H, W) triggers check_temporal_dim → patch_embed_img
        # with tubelet_size=1.  Output is LayerNorm'd by norms_block[-1].
        with torch.no_grad(), torch.autocast(
            device_type=x.device.type, dtype=torch.bfloat16
        ):
            hidden = self.model(x)  # (B*T, num_spatial_patches, embed_dim)

        # ---- Reshape to (B, T, h, w, D) ----
        h = w = self._spatial_tokens
        z = hidden.reshape(B, T, h, w, self._latent_dim)

        return z

    # ------------------------------------------------------------------
    # Decode (not available)
    # ------------------------------------------------------------------

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "V-JEPA 2.1 has no pretrained pixel decoder. "
            "Use a pixel decoder or adapter decoder instead."
        )
