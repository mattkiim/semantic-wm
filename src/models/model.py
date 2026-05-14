import torch
from torch import nn
import torch.nn.functional as F
import einops
import math
import functools
from typing import Sequence, Optional, Tuple
import sys
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

class AttentionType(StrEnum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    JOINT_TEMPORAL = "joint_temporal"


class RotaryType(StrEnum):
    STANDARD = "standard"
    PIXEL = "pixel"

class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # SwiGLU typically uses 2/3 of the hidden dim compared to GeLU to maintain parameter count
        hidden_features = int(2 * hidden_features / 3)
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


@functools.lru_cache
def rope_nd(
    shape: Sequence[int],
    dim: int = 64,
    base: float = 10_000.0,
    rotary_type: RotaryType = RotaryType.STANDARD,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    D = len(shape)
    assert (
        dim % (2 * D) == 0
    ), f"`dim` must be divisible by 2 × D (got dim={dim}, D={D})"

    dim_per_axis = dim // D
    half = dim_per_axis // 2
    if rotary_type == RotaryType.STANDARD:
        inv_freq = 1.0 / (
            base ** (torch.arange(half, device=device, dtype=dtype) / half)
        )
        coords = [torch.arange(n, device=device, dtype=dtype) for n in shape]
    elif rotary_type == RotaryType.PIXEL:
        inv_freq = (
            torch.linspace(1.0, 256.0 / 2, half, device=device, dtype=dtype) * math.pi
        )
        coords = [
            torch.linspace(-1, +1, steps=n, device=device, dtype=dtype) for n in shape
        ]
    else:
        raise NotImplementedError(f"invalid rotary type: {rotary_type}")

    mesh = torch.meshgrid(*coords, indexing="ij")

    embeddings = []
    for pos in mesh:
        theta = pos.unsqueeze(-1) * inv_freq
        emb_axis = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        embeddings.append(emb_axis)
    return torch.cat(embeddings, dim=-1)


@functools.lru_cache
def rope_nd_multiview(
    spatial_shape: Sequence[int],
    num_views: int,
    dim: int = 64,
    base: float = 10_000.0,
    rotary_type: RotaryType = RotaryType.STANDARD,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate 3-D RoPE for (H, W_per_view, num_views).

    Splits ``dim`` into spatial dims (H, W) and a view dim. The spatial part
    uses the same frequency layout as the 2-D ``rope_nd`` so that pretrained
    single-view weights transfer directly. The view dimension gets a small
    slice of the frequency bands.

    The dimension budget is:
      * view_dim  = dim // 3  (rounded to nearest even)
      * spatial_dim_per_axis = (dim - view_dim) // 2
    For head_dim=72 (XL preset): view=24, h=24, w=24  (72%6==0)
    For head_dim=64 (S/B/L):     view=22, h=21, w=21  (64=22+21+21)
    """
    # Allocate dims: view gets ~1/3, spatial h/w split the rest
    view_dim = (dim // 3)
    if view_dim % 2 != 0:
        view_dim -= 1
    spatial_dim = dim - view_dim
    h_dim = spatial_dim // 2
    if h_dim % 2 != 0:
        h_dim -= 1
    w_dim = spatial_dim - h_dim
    if w_dim % 2 != 0:
        # Steal one from h to keep both even
        w_dim -= 1
        h_dim += 1

    H, W = spatial_shape

    def _make_freq_and_coords(n: int, d: int):
        half = d // 2
        if rotary_type == RotaryType.STANDARD:
            freq = 1.0 / (base ** (torch.arange(half, device=device, dtype=dtype) / half))
            coord = torch.arange(n, device=device, dtype=dtype)
        elif rotary_type == RotaryType.PIXEL:
            freq = torch.linspace(1.0, 256.0 / 2, half, device=device, dtype=dtype) * math.pi
            coord = torch.linspace(-1, +1, steps=n, device=device, dtype=dtype)
        else:
            raise NotImplementedError(f"invalid rotary type: {rotary_type}")
        return freq, coord

    h_freq, h_coord = _make_freq_and_coords(H, h_dim)
    w_freq, w_coord = _make_freq_and_coords(W, w_dim)
    v_freq, v_coord = _make_freq_and_coords(num_views, view_dim)

    mesh_h, mesh_w, mesh_v = torch.meshgrid(h_coord, w_coord, v_coord, indexing="ij")

    embeddings = []
    for pos, freq in [(mesh_h, h_freq), (mesh_w, w_freq), (mesh_v, v_freq)]:
        theta = pos.unsqueeze(-1) * freq
        emb = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        embeddings.append(emb)

    return torch.cat(embeddings, dim=-1)  # (H, W, V, dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_mix(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    return x * cos + rotate_half(x) * sin


def apply_rope_nd(
    q: torch.Tensor,
    k: torch.Tensor,
    shape: tuple[int, ...],
    rotary_type: RotaryType,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = q.shape[-1]
    rope = rope_nd(
        shape, dim, base, rotary_type=rotary_type, dtype=q.dtype, device=q.device
    )
    rope = rope.view(*shape, len(shape), 2, -1)
    cos, sin = rope.unbind(-2)
    cos = cos.reshape(*shape, -1)
    sin = sin.reshape(*shape, -1)

    q_rot = rope_mix(q, cos, sin)
    k_rot = rope_mix(k, cos, sin)
    return q_rot, k_rot


def apply_rope_nd_multiview(
    q: torch.Tensor,
    k: torch.Tensor,
    spatial_shape: tuple[int, int],
    num_views: int,
    rotary_type: RotaryType,
    *,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 3-D view-aware RoPE to q/k in spatial attention.

    q, k are shaped ``(batch, head, H, W_total, d)`` where
    ``W_total = W_per_view * num_views``.  We reshape to
    ``(batch, head, H, W_per_view, V, d)``, apply 3-D RoPE for ``(H, W, V)``,
    then flatten spatial dims back.
    """
    dim = q.shape[-1]
    H, W_per_view = spatial_shape

    q = q.unflatten(-2, (W_per_view, num_views))
    k = k.unflatten(-2, (W_per_view, num_views))

    rope = rope_nd_multiview(
        (H, W_per_view), num_views, dim, base,
        rotary_type=rotary_type, dtype=q.dtype, device=q.device,
    )
    # rope layout per position: [h_cos, h_sin, w_cos, w_sin, v_cos, v_sin]
    # rope_mix expects cos of size dim//2 and sin of size dim//2.
    # Recompute axis dim split (must match rope_nd_multiview).
    view_dim = dim // 3
    if view_dim % 2 != 0:
        view_dim -= 1
    spatial_dim = dim - view_dim
    h_dim = spatial_dim // 2
    if h_dim % 2 != 0:
        h_dim -= 1
    w_dim = spatial_dim - h_dim
    if w_dim % 2 != 0:
        w_dim -= 1
        h_dim += 1

    h_emb, w_emb, v_emb = rope.split([h_dim, w_dim, view_dim], dim=-1)
    h_cos, h_sin = h_emb.chunk(2, dim=-1)
    w_cos, w_sin = w_emb.chunk(2, dim=-1)
    v_cos, v_sin = v_emb.chunk(2, dim=-1)

    cos = torch.cat([h_cos, w_cos, v_cos], dim=-1)
    sin = torch.cat([h_sin, w_sin, v_sin], dim=-1)

    q_rot = rope_mix(q, cos, sin)
    k_rot = rope_mix(k, cos, sin)

    # (B, head, H, W, V, d) -> (B, head, H, W*V, d)
    q_rot = q_rot.flatten(-3, -2)
    k_rot = k_rot.flatten(-3, -2)
    return q_rot, k_rot


_block_causal_mask_cache: dict[tuple, torch.Tensor] = {}


def get_block_causal_mask(
    T_frames: int, S_patches: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Block-causal mask: all patches within a frame attend to each other,
    and to all patches in earlier frames, but not to future frames."""
    cache_key = (T_frames, S_patches, device, dtype)
    if cache_key in _block_causal_mask_cache:
        return _block_causal_mask_cache[cache_key]

    total = T_frames * S_patches
    frame_idx = torch.arange(total, device=device) // S_patches
    # key frame must be <= query frame
    causal = frame_idx.unsqueeze(0) <= frame_idx.unsqueeze(1)
    mask = torch.zeros(total, total, device=device, dtype=dtype)
    mask.masked_fill_(~causal, float("-inf"))

    if len(_block_causal_mask_cache) > 10:
        _block_causal_mask_cache.clear()
    _block_causal_mask_cache[cache_key] = mask
    return mask


class FinalLayer(nn.Module):
    def __init__(self, dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 2, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _, _, H, W, _ = x.shape
        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(2, dim=-1)
        x = self.linear(self.norm(x) * (1 + m[1]) + m[0])
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        is_causal: bool,
        attention_type: AttentionType,
        rotary_type: RotaryType = RotaryType.STANDARD,
        use_qknorm: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.is_causal = is_causal
        self.attention_type = attention_type
        self.rotary_type = rotary_type
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.use_qknorm = use_qknorm

        if use_qknorm:
            self.q_norm = nn.RMSNorm(dim // num_heads, elementwise_affine=False)
            self.k_norm = nn.RMSNorm(dim // num_heads, elementwise_affine=False)

    def forward(self, x: torch.Tensor, num_views: int = 1):
        B, T, H, W, D = x.shape

        if self.attention_type == AttentionType.SPATIAL:
            x = einops.rearrange(x, "b t h w d -> (b t) h w d")
            sequence_shape = x.shape[1:-1]  # (H, W)
        elif self.attention_type == AttentionType.TEMPORAL:
            x = einops.rearrange(x, "b t h w d -> (b h w) t d")
            sequence_shape = x.shape[1:-1]  # (T,)
        elif self.attention_type == AttentionType.JOINT_TEMPORAL:
            x = einops.rearrange(x, "b t h w d -> b (t h w) d")
            sequence_shape = (T, H, W)  # 3D RoPE
        else:
            raise NotImplementedError(f"invalid attention type: {self.attention_type}")

        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = einops.rearrange(q, "B ... (head d) -> B head ... d", head=self.num_heads)
        k = einops.rearrange(k, "B ... (head d) -> B head ... d", head=self.num_heads)
        v = einops.rearrange(v, "B ... (head d) -> B head ... d", head=self.num_heads)

        if self.use_qknorm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if num_views > 1 and self.attention_type == AttentionType.SPATIAL:
            # View-aware 3D RoPE: (H, W_per_view, V) instead of (H, W_total)
            w_per_view = W // num_views
            q, k = apply_rope_nd_multiview(
                q, k, (H, w_per_view), num_views, rotary_type=self.rotary_type,
            )
            # apply_rope_nd_multiview returns (B, head, H, W_total, d) — flatten spatial
            q = einops.rearrange(q, "B head ... d -> B head (...) d")
            k = einops.rearrange(k, "B head ... d -> B head (...) d")
        elif self.attention_type == AttentionType.JOINT_TEMPORAL:
            # 3D RoPE over (T, H, W). head_dim must be divisible by 6 for
            # full 3-axis RoPE; when it isn't, apply RoPE to the largest
            # divisible prefix and pass the remainder through unchanged.
            head_dim = q.shape[-1]
            rope_dim = head_dim - (head_dim % 6)
            q = einops.rearrange(q, "B head (t h w) d -> B head t h w d", t=T, h=H, w=W)
            k = einops.rearrange(k, "B head (t h w) d -> B head t h w d", t=T, h=H, w=W)
            if rope_dim == head_dim:
                q, k = apply_rope_nd(q, k, sequence_shape, rotary_type=self.rotary_type)
            else:
                q_rope, q_pass = q[..., :rope_dim], q[..., rope_dim:]
                k_rope, k_pass = k[..., :rope_dim], k[..., rope_dim:]
                q_rope, k_rope = apply_rope_nd(
                    q_rope, k_rope, sequence_shape, rotary_type=self.rotary_type
                )
                q = torch.cat([q_rope, q_pass], dim=-1)
                k = torch.cat([k_rope, k_pass], dim=-1)
            q = einops.rearrange(q, "B head t h w d -> B head (t h w) d")
            k = einops.rearrange(k, "B head t h w d -> B head (t h w) d")
        else:
            q, k = apply_rope_nd(q, k, sequence_shape, rotary_type=self.rotary_type)
            # Flatten the sequence dimension
            q = einops.rearrange(q, "B head ... d -> B head (...) d")
            k = einops.rearrange(k, "B head ... d -> B head (...) d")
        v = einops.rearrange(v, "B head ... d -> B head (...) d")

        if self.attention_type == AttentionType.JOINT_TEMPORAL and self.is_causal:
            S_patches = H * W
            attn_mask = get_block_causal_mask(T, S_patches, x.device, q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            x = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        x = einops.rearrange(x, "B head seq d -> B seq (head d)")
        x = self.out_proj(x)

        if self.attention_type == AttentionType.SPATIAL:
            x = einops.rearrange(x, "(b t) (h w) d -> b t h w d", t=T, h=H, w=W)
        elif self.attention_type == AttentionType.TEMPORAL:
            x = einops.rearrange(x, "(b h w) t d -> b t h w d", h=H, w=W)
        elif self.attention_type == AttentionType.JOINT_TEMPORAL:
            x = einops.rearrange(x, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_type: AttentionType,
        rotary_type: RotaryType,
        is_causal: bool,
    ) -> None:
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6, bias=True)
        )
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads,
            is_causal=is_causal,
            attention_type=attention_type,
            rotary_type=rotary_type,
            use_qknorm=True,
        )
        self.ffwd = SwiGLU(
            in_features=dim,
            hidden_features=dim * 4,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_views: int = 1) -> torch.Tensor:
        _, _, H, W, _ = x.shape
        m = self.adaLN_modulation(c)
        m = einops.repeat(m, "b t d -> b t h w d", h=H, w=W).chunk(6, dim=-1)
        x = x + self.attn(self.norm1(x) * (1 + m[1]) + m[0], num_views=num_views) * m[2]
        x = x + self.ffwd(self.norm2(x) * (1 + m[4]) + m[3]) * m[5]
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope_config: dict[AttentionType, RotaryType] | None = None,
        temporal_mode: str = "factored",
    ) -> None:
        super().__init__()
        self.temporal_mode = temporal_mode
        self.s_block = DiTBlock(
            dim,
            num_heads,
            is_causal=False,
            attention_type=AttentionType.SPATIAL,
            rotary_type=(
                rope_config[AttentionType.SPATIAL]
                if rope_config
                else RotaryType.STANDARD
            ),
        )
        temporal_attn_type = (
            AttentionType.JOINT_TEMPORAL
            if temporal_mode == "joint"
            else AttentionType.TEMPORAL
        )
        self.t_block = DiTBlock(
            dim,
            num_heads,
            is_causal=True,
            attention_type=temporal_attn_type,
            rotary_type=(
                rope_config[AttentionType.TEMPORAL]
                if rope_config
                else RotaryType.STANDARD
            ),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, num_views: int = 1) -> torch.Tensor:
        x = self.s_block(x, c, num_views=num_views)
        x = self.t_block(x, c)
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        patch_size: int = 2,
        dim: int = 1152,
        num_layers: int = 28,
        num_heads: int = 16,
        action_dim: int = 0,
        max_frames: int = 16,
        rope_config: dict[AttentionType, RotaryType] | None = None,
        action_dropout_prob: float = 0.1,
        wide_head: bool = False,
        decoder_dim: int | None = None,
        decoder_depth: int = 2,
        decoder_heads: int = 16,
        num_views: int = 1,
        temporal_mode: str = "factored",
        tactile_dim: int = 0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.action_dim = action_dim
        self.tactile_dim = tactile_dim
        self.action_dropout_prob = action_dropout_prob
        self.wide_head = wide_head
        self.num_views = num_views

        self.x_proj = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(256, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.action_embedder = nn.Linear(action_dim, dim)
        self.tactile_embedder = nn.Linear(tactile_dim, dim) if tactile_dim > 0 else None
        self.blocks = nn.ModuleList(
            [Block(dim, num_heads, rope_config, temporal_mode=temporal_mode) for _ in range(num_layers)]
        )

        if wide_head:
            decoder_dim = decoder_dim or dim
            self.s_projector = nn.Linear(dim, decoder_dim)
            self.head_final_layer = FinalLayer(decoder_dim, patch_size, in_channels)
        else:
            self.final_layer = FinalLayer(dim, patch_size, in_channels)

        self.max_frames = max_frames
        self.initialize_weights()

    def timestep_embedding(
        self, t: torch.Tensor, dim: int = 256, max_period: int = 10000
    ) -> torch.Tensor:
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.timestep_mlp[0].weight, std=0.02)
        nn.init.normal_(self.timestep_mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        if self.wide_head is False:
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.wide_head is True:
            nn.init.constant_(self.head_final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.head_final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(self.head_final_layer.linear.weight, 0)
            nn.init.constant_(self.head_final_layer.linear.bias, 0)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> (b t) c h w")
        x = self.x_proj(x)
        x = einops.rearrange(x, "(b t) d h w -> b t h w d", t=T)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            x,
            "b h w (p1 p2 c) -> b (h p1) (w p2) c",
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.in_channels,
        )

    def get_null_cond(self, action: torch.Tensor) -> torch.Tensor:
        null_action = torch.zeros_like(action)
        # NOTE: all-zero action is still conditional (meaning "do not move"), so we
        # need to reserve the last component of the action vector to indicate null.
        null_action[..., -1] = 1
        return null_action

    def get_cond(
        self,
        t: torch.Tensor,
        action: torch.Tensor,
        tactile: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        B, T = t.shape
        t = einops.rearrange(t, "b t -> (b t)")
        t_freq = self.timestep_embedding(t)
        time_cond = self.timestep_mlp(t_freq)
        time_cond = einops.rearrange(time_cond, "(b t) d -> b t d", t=T)
        if self.training and self.action_dropout_prob > 0:
            should_drop = (
                torch.rand((B, 1, 1), device=action.device) < self.action_dropout_prob
            )
            null_action = self.get_null_cond(action)
            action = torch.where(should_drop, null_action, action)

        tactile_cond = None
        if self.tactile_embedder is not None:
            if tactile is None:
                raise ValueError("DiT was configured with tactile_dim > 0 but no tactile tensor was provided")
            tactile_cond = self.tactile_embedder(tactile)

        return time_cond, self.action_embedder(action), tactile_cond

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        action: torch.Tensor,
        tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        x = self.patchify(x)
        time_cond, action_cond, tactile_cond = self.get_cond(t, action, tactile)
        c = time_cond + action_cond
        if tactile_cond is not None:
            c = c + tactile_cond
        for block in self.blocks:
            x = block(x, c, num_views=self.num_views)

        if self.wide_head is True:            
            # Combine encoder features with timestep only
            head_cond = x + time_cond.unsqueeze(2).unsqueeze(3)
            head_cond = self.s_projector(F.silu(head_cond))
            head_c = head_cond.mean(dim=(2, 3))
            x = self.head_final_layer(head_cond, head_c)
        else:
            x = self.final_layer(x, c)

        x = einops.rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)
        x = einops.rearrange(x, "(b t) h w c -> b t h w c", t=T)
        return x
