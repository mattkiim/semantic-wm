import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union, Protocol

import numpy as np
import torch
from torch import nn
import einops
from math import sqrt

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import ACT2FN
from transformers import AutoConfig, AutoImageProcessor, Dinov2WithRegistersModel

from ..base_autoencoder import BaseAutoencoder


class ViTMAEConfig(PretrainedConfig):
    model_type = "vit_mae"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.75,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss


@dataclass
class ViTMAEDecoderOutput(ModelOutput):
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class ViTMAESelfAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class ViTMAESelfOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTMAEAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAESelfAttention(config)
        self.output = ViTMAESelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class ViTMAEIntermediate(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTMAEOutput(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTMAELayer(nn.Module):
    def __init__(self, config: ViTMAEConfig) -> None:
        super().__init__()
        self.attention = ViTMAEAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class GeneralDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(
            config.hidden_size, config.decoder_hidden_size, bias=True
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                ViTMAELayer(decoder_config)
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size**2 * config.num_channels,
            bias=True,
        )
        self.config = config
        self.num_patches = num_patches
        self.initialize_weights(num_patches)
        self.decoder_config = decoder_config
        self.set_trainable_cls_token()

    def set_trainable_cls_token(self, tensor: Optional[torch.Tensor] = None):
        tensor = (
            torch.zeros(1, 1, self.decoder_config.hidden_size)
            if tensor is None
            else tensor
        )
        self.trainable_cls_token = nn.Parameter(tensor)

    def initialize_weights(self, num_patches):
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

    def interpolate_latent(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if l == self.num_patches:
            return x
        h, w = int(l**0.5), int(l**0.5)
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)
        target_size = (int(self.num_patches**0.5), int(self.num_patches**0.5))
        x = nn.functional.interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )
        x = x.permute(0, 2, 3, 1).contiguous().view(b, self.num_patches, c)
        return x

    def unpatchify(
        self,
        patchified_pixel_values,
        original_image_size: Optional[Tuple[int, int]] = None,
    ):
        patch_size, num_channels = self.config.patch_size, self.config.num_channels
        original_image_size = (
            original_image_size
            if original_image_size is not None
            else (self.config.image_size, self.config.image_size)
        )
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum(
            "nhwpqc->nchpwq", patchified_pixel_values
        )
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        drop_cls_token: bool = False,
    ):
        x = self.decoder_embed(hidden_states)
        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)
        cls_token = self.trainable_cls_token.expand(x_.shape[0], -1, -1)
        x = torch.cat([cls_token, x_], dim=1)

        hidden_states = x + self.decoder_pos_embed

        for layer_module in self.decoder_layers:
            layer_outputs = layer_module(
                hidden_states, head_mask=None, output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]

        if not return_dict:
            return (logits,)
        return ViTMAEDecoderOutput(logits=logits)


class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=False
            )
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features


class Stage1Protocol(Protocol):
    patch_size: int
    hidden_size: int

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class RAE(BaseAutoencoder):
    def __init__(
        self,
        encoder_config_path: str = "facebook/dinov2-with-registers-base",
        encoder_input_size: int = 224,
        decoder_config_path: str = "src/configs/decoder/ViTXL",
        decoder_patch_size: int = 16,
        pretrained_decoder_path: Optional[str] = None,
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Dinov2withNorm(encoder_config_path)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        self.encoder.to(torch.bfloat16)  # frozen encoder can safely use float16

        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.register_buffer(
            "encoder_mean", torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "encoder_std", torch.tensor(proc.image_std).view(1, 3, 1, 1)
        )

        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self._latent_dim = self.encoder.hidden_size
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2

        decoder_config = AutoConfig.from_pretrained(decoder_config_path)
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
        else:
            decoder_config.hidden_size = self._latent_dim
            decoder_config.decoder_hidden_size = self._latent_dim

        decoder_config.patch_size = decoder_patch_size
        decoder_config.image_size = int(decoder_patch_size * sqrt(self.base_patches))
        self.decoder = GeneralDecoder(decoder_config, num_patches=self.base_patches)

        if pretrained_decoder_path is not None:
            self.decoder.load_state_dict(state_dict, strict=False)

        self.decoder.eval()
        self.decoder.requires_grad_(False)
        self.decoder.to(torch.bfloat16)  # frozen decoder can safely use float16
        # Keep decoder on CPU — it is only needed for decode() during validation,
        # not during the training forward pass. decode() moves it to GPU on demand.
        self._decoder_device = torch.device("cpu")
        self.decoder.cpu()

        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d

        if normalization_stat_path is not None:
            stats = torch.load(normalization_stat_path, map_location="cpu")
            self.register_buffer("latent_mean", stats.get("mean", None))
            self.register_buffer("latent_var", stats.get("var", None))
            self.do_normalization = self.latent_var is not None
            self.eps = eps
        else:
            self.do_normalization = False

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device
        )
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> (b t) c h w")

        if H != self.encoder_input_size or W != self.encoder_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )

        x = (x - self.encoder_mean) / self.encoder_std

        with torch.no_grad():
            z = self.encoder(x)

        if self.training and self.noise_tau > 0:
            z = self.noising(z)

        if self.reshape_to_2d:
            bt, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(bt, c, h, w)

        if self.do_normalization:
            latent_mean = self.latent_mean if self.latent_mean is not None else 0
            latent_var = self.latent_var if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)

        if self.reshape_to_2d:
            z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        else:
            z = einops.rearrange(z, "(b t) n c -> b t n c", b=B, t=T)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = z.shape
        if self.reshape_to_2d:
            z = einops.rearrange(z, "b t h w c -> (b t) c h w")
        else:
            z = einops.rearrange(z, "b t n c -> (b t) n c")

        if self.do_normalization:
            latent_mean = self.latent_mean if self.latent_mean is not None else 0
            latent_var = self.latent_var if self.latent_var is not None else 1
            z = z * torch.sqrt(latent_var + self.eps) + latent_mean

        if self.reshape_to_2d:
            bt, c, h, w = z.shape
            z = z.view(bt, c, h * w).transpose(1, 2)

        # Move decoder to the same device as z on demand (it lives on CPU
        # between calls to save GPU memory during training).
        target_device = z.device
        if self._decoder_device != target_device:
            self.decoder.to(target_device)
            self._decoder_device = target_device

        with torch.no_grad():
            output = self.decoder(z, drop_cls_token=False).logits
            x_rec = self.decoder.unpatchify(output)
            x_rec = x_rec * self.encoder_std + self.encoder_mean

        # Move decoder back to CPU to free GPU memory.
        self.decoder.cpu()
        self._decoder_device = torch.device("cpu")
        torch.cuda.empty_cache()

        x_rec = einops.rearrange(x_rec, "(b t) c h w -> b t h w c", b=B, t=T)
        return x_rec
