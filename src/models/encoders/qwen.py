import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_autoencoder import BaseAutoencoder
from transformers import Qwen2_5_VLModel, AutoProcessor
import logging

logger = logging.getLogger(__name__)


class QwenVisionEncoder(nn.Module):
    def __init__(self, model_path=None):
        super().__init__()
        print(
            f"Loading processor from {model_path}... (using slow processor to support video)"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        print(f"Loading vision encoder weights from {model_path} directly...")
        full_model = Qwen2_5_VLModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        self.visual = full_model.visual
        del full_model.language_model  # free up LLM memory
        torch.cuda.empty_cache()
        print("Successfully loaded vision encoder from transformers.")

    def process_and_encode(
        self, images=None, videos=None, project_to_llm=False, as_multidimensional=True
    ):
        """
        Process images/videos via the standalone vision processor and forward them.
        Args:
            images: List of PIL images.
            videos: List of lists of PIL images.
            project_to_llm: If True, returns projected 2048-dim tokens. False returns raw 1280-dim ViT tokens.
            as_multidimensional: If True, reshapes the flat output sequence into [Batch, T, H, W, C].
        """
        dummy_text = (
            "<|image|>"
            if images is not None
            else ("<|video|>" if videos is not None else None)
        )
        inputs = self.processor(
            text=dummy_text, images=images, videos=videos, return_tensors="pt"
        )

        inputs = {
            k: v.to(self.visual.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.visual.dtype)
        if "pixel_values_videos" in inputs:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                self.visual.dtype
            )

        with torch.no_grad():
            pixel_values = (
                inputs.get("pixel_values")
                if images is not None
                else inputs.get("pixel_values_videos")
            )
            # grid_thw holds the raw ViT spatial dimensions [T, H, W]
            grid_thw = (
                inputs.get("image_grid_thw")
                if images is not None
                else inputs.get("video_grid_thw")
            )

            if project_to_llm:
                outputs = self.visual(hidden_states=pixel_values, grid_thw=grid_thw)
                res = outputs[0] if isinstance(outputs, tuple) else outputs

                if as_multidimensional:
                    # Qwen's grid_thw holds the T, H, W for the UNPROJECTED patches.
                    # The PatchMerger downsamples H and W by a factor of 2.
                    t, h, w = grid_thw[0]
                    res = res.view(
                        1, t.item(), h.item() // 2, w.item() // 2, res.shape[-1]
                    )
                return res, grid_thw
            else:
                hidden_states = self.visual.patch_embed(pixel_values)
                rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
                window_index, cu_window_seqlens = self.visual.get_window_index(grid_thw)
                cu_window_seqlens = torch.tensor(
                    cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
                )
                cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

                seq_len, _ = hidden_states.size()
                spatial_unit = self.visual.spatial_merge_unit
                hidden_states = hidden_states.reshape(
                    seq_len // spatial_unit, spatial_unit, -1
                )
                hidden_states = hidden_states[window_index, :, :]
                hidden_states = hidden_states.reshape(seq_len, -1)

                rotary_pos_emb = rotary_pos_emb.reshape(
                    seq_len // spatial_unit, spatial_unit, -1
                )
                rotary_pos_emb = rotary_pos_emb[window_index, :, :]
                rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                position_embeddings = (emb.cos(), emb.sin())

                cu_seqlens = torch.repeat_interleave(
                    grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
                ).cumsum(dim=0, dtype=torch.int32)
                cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

                for layer_num, blk in enumerate(self.visual.blocks):
                    if layer_num in self.visual.fullatt_block_indexes:
                        cu_seqlens_now = cu_seqlens
                    else:
                        cu_seqlens_now = cu_window_seqlens

                    hidden_states = blk(
                        hidden_states,
                        cu_seqlens=cu_seqlens_now,
                        position_embeddings=position_embeddings,
                    )

                raw_vit_grouped = hidden_states.reshape(
                    seq_len // spatial_unit, spatial_unit, -1
                )
                reverse_indices = torch.argsort(window_index)
                canonical_raw_vit = raw_vit_grouped[reverse_indices, :, :].reshape(
                    seq_len, -1
                )

                # Apply the pre-trained Qwen RMSNorm to stabilize feature magnitudes
                canonical_raw_vit = self.visual.merger.ln_q(canonical_raw_vit)

                if as_multidimensional:
                    # grid_thw strictly matches the raw ViT patches without spatial downsampling.
                    t, h, w = grid_thw[0]
                    canonical_raw_vit = canonical_raw_vit.view(
                        1, t.item(), h.item(), w.item(), canonical_raw_vit.shape[-1]
                    )

                return canonical_raw_vit, grid_thw


class QwenEncoderWrapper(BaseAutoencoder):
    def __init__(self, model_path="Qwen/Qwen2.5-VL-3B-Instruct", mode="video"):
        super().__init__()
        self.mode = mode  # "image" or "video"
        self._latent_dim = 1280
        self.has_decoder = False

        self.qwen_enc = QwenVisionEncoder(model_path=model_path)

        # Monkey-patch Qwen's attention to be dynamo-friendly (avoiding list comprehensions)
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_mod

        def dynamo_friendly_attn_forward(
            attn_self,
            hidden_states: torch.Tensor,
            cu_seqlens: torch.Tensor,
            rotary_pos_emb: torch.Tensor = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
            **kwargs,
        ):
            seq_length = hidden_states.shape[0]
            query_states, key_states, value_states = (
                attn_self.qkv(hidden_states)
                .reshape(seq_length, 3, attn_self.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            cos, sin = position_embeddings
            query_states, key_states = qwen_mod.apply_rotary_pos_emb_vision(
                query_states, key_states, cos, sin
            )

            query_states = query_states.transpose(0, 1).unsqueeze(0)
            key_states = key_states.transpose(0, 1).unsqueeze(0)
            value_states = value_states.transpose(0, 1).unsqueeze(0)

            # Uniform sequence lengths guarantee we can reshape perfectly instead of tracking via lists
            num_chunks = cu_seqlens.shape[0] - 1
            chunk_len = seq_length // num_chunks

            q_vec = (
                query_states.view(1, attn_self.num_heads, num_chunks, chunk_len, -1)
                .transpose(1, 2)
                .squeeze(0)
            )
            k_vec = (
                key_states.view(1, attn_self.num_heads, num_chunks, chunk_len, -1)
                .transpose(1, 2)
                .squeeze(0)
            )
            v_vec = (
                value_states.view(1, attn_self.num_heads, num_chunks, chunk_len, -1)
                .transpose(1, 2)
                .squeeze(0)
            )

            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q_vec,
                k_vec,
                v_vec,
                dropout_p=(
                    0.0 if not attn_self.training else attn_self.attention_dropout
                ),
                is_causal=False,
            )

            attn_output = (
                attn_out.unsqueeze(0)
                .transpose(1, 2)
                .reshape(1, attn_self.num_heads, seq_length, -1)
            )
            attn_output = (
                attn_output.transpose(1, 2).reshape(seq_length, -1).contiguous()
            )
            return attn_self.proj(attn_output)

        qwen_mod.Qwen2_5_VLVisionAttention.forward = dynamo_friendly_attn_forward

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    @property
    def temporal_downsample_factor(self) -> int:
        return 2 if self.mode == "video" else 1

    @torch.compiler.disable()
    def _get_qwen_metadata(self, device, T_curr, out_H, out_W, B):
        if self.mode == "video":
            grid_thw = torch.tensor(
                [[T_curr // 2, out_H, out_W]] * B, dtype=torch.int32, device=device
            )
        else:
            # Treat each frame as an independent image
            grid_thw = torch.tensor(
                [[1, out_H, out_W]] * (B * T_curr), dtype=torch.int32, device=device
            )

        rotary_pos_emb = self.qwen_enc.visual.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.qwen_enc.visual.get_window_index(
            grid_thw
        )

        cu_window_seqlens = torch.tensor(
            cu_window_seqlens, device=device, dtype=torch.int32
        ).unique_consecutive()

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        return grid_thw, rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape

        if H != 224 or W != 224:
            x_norm = x.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
            x_norm = F.interpolate(
                x_norm, size=(224, 224), mode="bicubic", align_corners=False
            )
            x_norm = x_norm.permute(0, 2, 3, 1).reshape(B, T, 224, 224, C).contiguous()
        else:
            x_norm = x.contiguous()

        x_norm = x_norm.clamp(0, 1)

        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1, 3)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1, 3)
        x_norm = (x_norm - mean) / std

        patch_size = 14
        out_H = 224 // patch_size
        out_W = 224 // patch_size

        T_curr = T
        if self.mode == "video":
            if T_curr % 2 != 0:
                last_frame = x_norm[:, -1:, :, :, :]
                x_norm = torch.cat([x_norm, last_frame], dim=1)
                T_curr += 1
            pixel_values = x_norm.permute(0, 1, 4, 2, 3).reshape(
                B * T_curr, C, 224, 224
            )
        else:
            # Image mode: duplicate each image exactly how the AutoProcessor does
            # to make sure the Conv3D groups the same image instead of separate images.
            x_norm = (
                x_norm.unsqueeze(2)
                .expand(-1, -1, 2, -1, -1, -1)
                .reshape(B, T_curr * 2, 224, 224, C)
            )
            pixel_values = x_norm.permute(0, 1, 4, 2, 3).reshape(
                B * T_curr * 2, C, 224, 224
            )

        pixel_values = pixel_values.to(self.qwen_enc.visual.dtype).contiguous()

        grid_thw, rotary_pos_emb, window_index, cu_window_seqlens, cu_seqlens = (
            self._get_qwen_metadata(x.device, T_curr, out_H, out_W, B)
        )

        hidden_states = self.qwen_enc.visual.patch_embed(pixel_values)

        seq_len = hidden_states.size(0)
        spatial_unit = self.qwen_enc.visual.spatial_merge_unit

        hidden_states = hidden_states.view(seq_len // spatial_unit, spatial_unit, -1)[
            window_index, :, :
        ].view(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.view(seq_len // spatial_unit, spatial_unit, -1)[
            window_index, :, :
        ].view(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        for layer_num, blk in enumerate(self.qwen_enc.visual.blocks):
            cu_seqlens_now = (
                cu_seqlens
                if layer_num in self.qwen_enc.visual.fullatt_block_indexes
                else cu_window_seqlens
            )
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                position_embeddings=position_embeddings,
            )

        raw_vit_grouped = hidden_states.view(seq_len // spatial_unit, spatial_unit, -1)
        reverse_indices = torch.argsort(window_index)
        canonical_raw_vit = raw_vit_grouped[reverse_indices, :, :].view(seq_len, -1)

        t_per_batch = T_curr // 2 if self.mode == "video" else T_curr
        h_per_batch = out_H
        w_per_batch = out_W
        canonical_raw_vit = canonical_raw_vit.view(
            B, t_per_batch, h_per_batch, w_per_batch, canonical_raw_vit.size(-1)
        )

        # Apply the pre-trained Qwen RMSNorm to stabilize feature magnitudes, replacing the manual layer_norm
        canonical_raw_vit = self.qwen_enc.visual.merger.ln_q(canonical_raw_vit)

        return canonical_raw_vit

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Qwen has no pretrained decoder")
