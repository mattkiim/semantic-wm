"""Base autoencoder interface and factory for all encoder/decoder types.

Every concrete autoencoder (VAE, RAE, ScaleRAE, …) inherits from
:class:`BaseAutoencoder` so that the rest of the codebase can treat them
uniformly through ``encode()`` / ``decode()`` / ``latent_dim``.

The :func:`create_autoencoder` factory and :func:`encoder_config_from_args`
helper centralise the instantiation logic that was previously duplicated
across ``train.py`` and ``world_model.py``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseAutoencoder(nn.Module, ABC):
    """Abstract base class for all autoencoders used in the world model.

    Subclasses **must** implement:
      - ``encode(x)``  – maps pixel tensors to latents.
      - ``decode(z)``  – maps latents back to pixel tensors.
      - ``latent_dim`` – the channel/feature dimensionality of the latent space.
    """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to latent representations.

        Parameters
        ----------
        x : (B, T, H, W, C) float tensor in [0, 1].

        Returns
        -------
        z : (B, T, h, w, C_latent) or (B, T, N, C_latent).
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representations back to pixel space.

        Parameters
        ----------
        z : Latent tensor produced by ``encode()``.

        Returns
        -------
        x_rec : (B, T, H, W, C) float tensor approximately in [0, 1].
        """
        ...

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Channel / feature dimensionality of the latent space."""
        ...

    @property
    def temporal_downsample_factor(self) -> int:
        """Factor by which the encoder reduces the temporal dimension.

        Default is 1 (no temporal reduction).  Encoders with 3-D tubelet
        embeddings (e.g. Qwen, V-JEPA 2) override this to 2.
        """
        return 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_autoencoder(config: Dict[str, Any]) -> BaseAutoencoder:
    """Instantiate an autoencoder from a configuration dict.

    Parameters
    ----------
    config : dict
        Must contain ``"encoder_type"`` (one of ``"vae"``, ``"rae"``,
        ``"scale_rae_siglip"``, ``"scale_rae_webssl"``).  Type-specific
        parameters are passed through under the corresponding key
        (e.g. ``"rae_params"``).

    Returns
    -------
    BaseAutoencoder
    """
    encoder_type = config["encoder_type"]

    if encoder_type == "vae":
        from .encoders.vae import VAE

        vae_kwargs = {}
        if config.get("vae_model_path"):
            vae_kwargs["model_path"] = config["vae_model_path"]
        if config.get("vae_subfolder"):
            vae_kwargs["subfolder"] = config["vae_subfolder"]
        return VAE(**vae_kwargs)

    if encoder_type == "rae":
        from .encoders.rae import RAE

        rae_params = config.get("rae_params", {})
        return RAE(**rae_params)

    if encoder_type in ("scale_rae_siglip", "scale_rae_webssl"):
        from .encoders.scale_rae import ScaleRAE

        encoder_name = encoder_type.replace("scale_rae_", "")
        return ScaleRAE(
            encoder_name=encoder_name,
            decoder_config_path=config.get("scale_rae_decoder_config"),
            pretrained_decoder_path=config.get("pretrained_decoder_path"),
            normalization_stat_path=config.get("encoder_normalization_stat_path"),
        )

    if encoder_type == "qwen":
        from .encoders.qwen import QwenEncoderWrapper
        return QwenEncoderWrapper(
            model_path=config.get("qwen_model_path", "Qwen/Qwen2.5-VL-3B-Instruct"),
            mode=config.get("qwen_mode", "video")
        )

    if encoder_type == "vjepa2":
        from .encoders.vjepa2 import VJEPA2EncoderWrapper
        return VJEPA2EncoderWrapper(
            model_size=config.get("vjepa2_model_size", "vitl"),
            checkpoint_path=config.get("vjepa2_checkpoint_path"),
            input_size=config.get("vjepa2_input_size", 256),
        )

    if encoder_type == "cosmos":
        from .encoders.cosmos import CosmosTokenizerWrapper
        return CosmosTokenizerWrapper(
            checkpoint_dir=config.get("cosmos_checkpoint_dir"),
        )

    if encoder_type == "vavae":
        from .encoders.vavae import VAVAEWrapper
        return VAVAEWrapper(
            checkpoint_path=config.get("vavae_checkpoint_path"),
        )

    if encoder_type == "precomputed":
        from .encoders.precomputed import PrecomputedEncoder
        return PrecomputedEncoder(embedding_dim=config.get("embedding_dim", 384))

    raise ValueError(f"Unknown encoder type: {encoder_type}")


def encoder_config_from_args(args) -> Dict[str, Any]:
    """Convert CLI *args* (from ``launch.py``) to an autoencoder config dict.

    This is the single place where argparse attributes are translated into
    the dict format expected by :func:`create_autoencoder`.
    """
    config: Dict[str, Any] = {"encoder_type": args.encoder_type}

    if args.encoder_type == "vae":
        if getattr(args, "vae_model_path", None):
            config["vae_model_path"] = args.vae_model_path
        if getattr(args, "vae_subfolder", None):
            config["vae_subfolder"] = args.vae_subfolder

    if args.encoder_type == "rae":
        rae_params: Dict[str, Any] = {}
        if getattr(args, "rae_config_path", None):
            with open(args.rae_config_path, "r") as f:
                rae_params = json.load(f).get("rae_params", {})
        if getattr(args, "rae_pretrained_decoder_path", None):
            rae_params["pretrained_decoder_path"] = args.rae_pretrained_decoder_path
        config["rae_params"] = rae_params

    elif args.encoder_type in ("scale_rae_siglip", "scale_rae_webssl"):
        config["scale_rae_decoder_config"] = getattr(
            args, "scale_rae_decoder_config", None
        )
        config["pretrained_decoder_path"] = getattr(
            args, "rae_pretrained_decoder_path", None
        )
        config["encoder_normalization_stat_path"] = getattr(
            args, "encoder_normalization_stat_path", None
        )
        
    elif args.encoder_type == "qwen":
        config["qwen_model_path"] = getattr(args, "qwen_model_path", "Qwen/Qwen2.5-VL-3B-Instruct")
        config["qwen_mode"] = getattr(args, "qwen_mode", "video")

    elif args.encoder_type == "vjepa2":
        config["vjepa2_model_size"] = getattr(args, "vjepa2_model_size", "vitl")
        config["vjepa2_checkpoint_path"] = getattr(args, "vjepa2_checkpoint_path", None)
        config["vjepa2_input_size"] = getattr(args, "vjepa2_input_size", 256)

    elif args.encoder_type == "cosmos":
        config["cosmos_checkpoint_dir"] = getattr(args, "cosmos_checkpoint_dir", None)

    elif args.encoder_type == "vavae":
        config["vavae_checkpoint_path"] = getattr(args, "vavae_checkpoint_path", None)

    elif args.encoder_type == "precomputed":
        config["embedding_dim"] = getattr(args, "embedding_dim", 384)

    return config
