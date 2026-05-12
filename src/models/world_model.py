from tqdm import tqdm
import torch
import numpy as np
import einops
from .model import DiT, AttentionType, RotaryType
from .base_autoencoder import create_autoencoder
from .adapters import create_adapter, IdentityAdapter
from ..training.diffusion import Diffusion


class WorldModel:
    def __init__(
        self,
        checkpoint_path: str,
        use_pixel_rope: bool = False,
        default_cfg: float = 1.0,
        config: dict = None,
    ):
        self.device = "cuda:0"

        # Default config for VAE-based model
        if config is None:
            config = {
                "encoder_type": "vae",
                "dit_params": {
                    "in_channels": 16,
                    "patch_size": 2,
                    "dim": 1024,
                    "num_layers": 16,
                    "num_heads": 16,
                    "action_dim": 10,
                    "max_frames": 20,
                },
            }

        self.config = config
        dit_params = config["dit_params"]

        self.num_views = config.get("num_views", 1)
        self.model = (
            DiT(
                in_channels=dit_params["in_channels"],
                patch_size=dit_params["patch_size"],
                dim=dit_params["dim"],
                num_layers=dit_params["num_layers"],
                num_heads=dit_params["num_heads"],
                action_dim=dit_params["action_dim"],
                max_frames=dit_params["max_frames"],
                rope_config={
                    AttentionType.SPATIAL: (
                        RotaryType.PIXEL if use_pixel_rope else RotaryType.STANDARD
                    ),
                    AttentionType.TEMPORAL: RotaryType.STANDARD,
                },
                num_views=self.num_views,
                temporal_mode=dit_params.get("temporal_mode", "factored"),
            )
            .to(self.device)
            .eval()
        )
        state_dict = torch.load(
            checkpoint_path, weights_only=True, map_location=self.device
        )
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        self.model.load_state_dict(state_dict, strict=True)

        self.autoencoder = create_autoencoder(config).to(self.device).eval()

        # ---- Adapter (frozen) ------------------------------------------------
        adapter_config = config.get("adapter_config", {"adapter_type": "identity"})
        self.adapter = (
            create_adapter(adapter_config, input_dim=self.autoencoder.latent_dim)
            .to(self.device)
            .eval()
        )
        adapter_ckpt = config.get("adapter_checkpoint_path")
        if adapter_ckpt is not None:
            ckpt_data = torch.load(adapter_ckpt, map_location=self.device)
            self.adapter.load_state_dict(ckpt_data["adapter"])
        self._is_identity_adapter = isinstance(self.adapter, IdentityAdapter)

        self.diffusion = Diffusion(
            timesteps=1000, sampling_timesteps=10, device=self.device
        ).to(self.device)
        self.chunk_size = 1
        self.actions = None
        self.curr_frame = 0
        self.cfg = default_cfg  # Feel free to override this after __init__

    def reset(self, x):
        """Reset with initial frame(s).

        For single-view: ``x`` is ``(H, W, C)``.
        For multi-view:  ``x`` is ``(V, H, W, C)`` — one frame per view.
        """
        if self.num_views > 1 and x.dim() == 4:
            # (V, H, W, C) -> encode each view independently then concatenate
            V = x.shape[0]
            x = einops.rearrange(x, "v h w c -> v 1 h w c")  # (V, 1, H, W, C)
            x_enc = self.autoencoder.encode(x)
            if not self._is_identity_adapter:
                x_enc = self.adapter.encode(x_enc)
                if isinstance(x_enc, tuple):
                    x_enc = x_enc[0]
            # (V, 1, h, w, c) -> (1, 1, h, V*w, c)
            self.xs = einops.rearrange(x_enc, "v t h w c -> 1 t h (v w) c")
        else:
            x = einops.repeat(x, "h w c -> b t h w c", b=1, t=1)
            self.xs = self.autoencoder.encode(x)
            if not self._is_identity_adapter:
                self.xs = self.adapter.encode(self.xs)
                if isinstance(self.xs, tuple):
                    self.xs = self.xs[0]
        self.actions = torch.zeros((1, 1, self.model.action_dim), device=self.device)
        self.curr_frame = 1

    @torch.no_grad()
    def generate_chunk(self, action_vec):
        """See Diffusion.generate"""
        action_chunk = torch.zeros(
            (1, self.chunk_size, self.model.action_dim), device=self.device
        )
        assert self.actions.shape[1] == self.curr_frame
        self.actions = torch.cat([self.actions, action_chunk], dim=1)
        self.actions[:, self.curr_frame : self.curr_frame + self.chunk_size, :] = (
            action_vec
        )

        scheduling_matrix = self.diffusion.generate_pyramid_scheduling_matrix(
            self.chunk_size
        )
        chunk = torch.randn(
            (1, self.chunk_size, *self.xs.shape[-3:]), device=self.device
        )
        self.xs = torch.cat([self.xs, chunk], dim=1)

        # Adjust context length
        start_frame = max(0, self.curr_frame + self.chunk_size - self.model.max_frames)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for m in range(scheduling_matrix.shape[0] - 1):
                t, t_next = scheduling_matrix[m], scheduling_matrix[m + 1]
                t, t_next = map(
                    lambda x: einops.repeat(x, "t -> b t", b=1), (t, t_next)
                )
                t, t_next = map(
                    lambda x: torch.cat(
                        (torch.zeros((1, self.curr_frame), dtype=torch.long), x), dim=1
                    ),
                    (t, t_next),
                )

                self.xs[:, start_frame:] = self.diffusion.ddim_sample_step(
                    self.model,
                    self.xs[:, start_frame:],
                    self.actions[:, start_frame : self.curr_frame + self.chunk_size],
                    t[:, start_frame:],
                    t_next[:, start_frame:],
                    cfg=self.cfg,
                )

                latest_clean_idx = (t_next == 0).nonzero()[-1][1]
                if latest_clean_idx >= self.curr_frame:
                    z = self.xs[:, latest_clean_idx : latest_clean_idx + 1]
                    if self.num_views > 1:
                        # (1, 1, h, V*w, c) -> (V, 1, h, w, c) for decoding
                        z = einops.rearrange(
                            z, "b t h (v w) c -> (b v) t h w c", v=self.num_views
                        )
                    if not self._is_identity_adapter:
                        z = self.adapter.decode(z)
                    xs = self.autoencoder.decode(z)
                    if self.num_views > 1:
                        # (V, 1, H, W, C) -> (1, 1, H, V*W, C)
                        xs = einops.rearrange(
                            xs, "(b v) t h w c -> b t h (v w) c", v=self.num_views
                        )
                    yield latest_clean_idx, xs
        self.curr_frame += self.chunk_size
