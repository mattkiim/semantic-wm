import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import einops
import numpy as np


def latest_context_tactile_state(
    tactile: torch.Tensor | None,
    context_frames: int,
) -> torch.Tensor | None:
    """Select tactile from the latest clean context frame, never from future frames."""
    if tactile is None:
        return None
    if context_frames <= 0:
        raise ValueError(
            "context_frames must be > 0 when tactile conditioning is enabled"
        )
    if tactile.ndim != 3:
        raise ValueError("tactile must have shape (B, T, tactile_dim)")
    if tactile.shape[1] < context_frames:
        raise ValueError(
            f"tactile sequence length {tactile.shape[1]} is shorter than context_frames={context_frames}"
        )
    return tactile[:, context_frames - 1 : context_frames]


def truncated_logitnormal_sample(
    shape, mu, sigma, low=0.0, high=1.0, device=None, dtype=None
):
    mu = torch.as_tensor(mu, device=device, dtype=dtype)
    sigma = torch.as_tensor(sigma, device=device, dtype=dtype)
    low = torch.as_tensor(low, device=device, dtype=dtype)
    high = torch.as_tensor(high, device=device, dtype=dtype)

    z_low = torch.logit(low)
    z_high = torch.logit(high)

    base = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    alpha = (z_low - mu) / sigma
    beta = (z_high - mu) / sigma

    cdf_alpha = base.cdf(alpha)
    cdf_beta = base.cdf(beta)

    U = torch.rand(shape, device=device, dtype=dtype)
    U = cdf_alpha + (cdf_beta - cdf_alpha) * U.clamp(0, 1)

    Z = mu + sigma * base.icdf(U)
    X = torch.sigmoid(Z)

    return X.clamp(low, high)


class Diffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1_000,
        sampling_timesteps: int = 10,
        time_dist_shift: float = 1.0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.time_dist_shift = time_dist_shift
        alphas = 1 - self.sigmoid_beta_schedule(self.timesteps)
        # an unfortunate variable name, but it's the standard one
        self.register_buffer("alphas_cumprod", alphas.cumprod(dim=0), persistent=False)
        self.stabilization_level = 15
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

    def time_shift(self, t: torch.Tensor) -> torch.Tensor:
        """Apply resolution-dependent time shift (Esser et al., 2024 / SD3).

        t must be in [0, 1]. Returns shifted t in [0, 1].
        """
        if self.time_dist_shift == 1.0:
            return t
        return (self.time_dist_shift * t) / (1 + (self.time_dist_shift - 1) * t)

    def sample_t(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """Sample discrete timestep indices with optional time shift.

        With shift==1.0 this is equivalent to torch.randint(0, timesteps).
        With shift!=1.0 a continuous uniform is shifted then mapped to discrete.
        """
        if self.time_dist_shift == 1.0:
            return torch.randint(0, self.timesteps, (B, T), device=device, dtype=torch.long)
        u = torch.rand((B, T), device=device)
        u_shifted = self.time_shift(u)
        return (u_shifted * self.timesteps).long().clamp(0, self.timesteps - 1)

    def sigmoid_beta_schedule(
        self, timesteps: int, start: float = -3, end: float = 3, tau: float = 1
    ) -> torch.Tensor:
        # https://arxiv.org/abs/2212.11972
        t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (
            v_end - v_start
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999).float()

    def q_sample(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        B, T = t.shape

        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)

        return alphas_cumprod.sqrt() * x + (1 - alphas_cumprod).sqrt() * noise

    def loss_fn(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        num_history: int = 0,
        tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        B, T, D = actions.shape

        t = self.sample_t(B, T, device=x.device)
        noise = torch.randn_like(x)

        x_t = self.q_sample(x, t, noise)
        tactile_context_frames = num_history if num_history > 0 else T
        tactile_state = (
            latest_context_tactile_state(tactile, tactile_context_frames)
            if tactile is not None
            else None
        )
        pred_v = model(x_t, t, actions, tactile_state=tactile_state)

        # Build target v
        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        target_v = alphas_cumprod.sqrt() * noise - (1 - alphas_cumprod).sqrt() * x

        loss = F.mse_loss(pred_v, target_v)
        return loss

    def ddim_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        t_idx: torch.Tensor,
        t_next_idx: torch.Tensor,
        tactile: torch.Tensor | None = None,
        cfg: float = 1.0,
    ) -> torch.Tensor:
        # Derived from
        # https://github.com/buoyancy99/diffusion-forcing/blob/475e0bcab87545e48b24b39fb46a81fe59d80594/algorithms/diffusion_forcing/models/diffusion.py#L383
        B, T, H, W, C = x.shape
        B, T, D = actions.shape
        B, T = t_idx.shape

        sampling_noise_steps = torch.linspace(
            -1,
            self.timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=x.device,
            dtype=torch.long,
        )
        t = sampling_noise_steps[t_idx]
        t_next = sampling_noise_steps[t_next_idx]

        clipped_t = torch.where(
            t < 0, torch.full_like(t, self.stabilization_level - 1, dtype=torch.long), t
        )
        orig_x = x.clone().detach()
        scaled_context = self.q_sample(x, clipped_t, torch.zeros_like(x))
        x = torch.where(t.reshape(B, T, 1, 1, 1) < 0, scaled_context, x)

        alphas_cumprod = self.alphas_cumprod[t.reshape(-1)].view(B, T, 1, 1, 1)
        alphas_next_cumprod = torch.where(
            t_next < 0,
            torch.ones_like(t_next),
            self.alphas_cumprod[t_next.reshape(-1)].view(B, T),
        ).view(B, T, 1, 1, 1)
        c = (1 - alphas_next_cumprod).sqrt()

        v_pred_cond = model(x, clipped_t, actions, tactile_state=tactile)
        if cfg != 1.0:
            v_pred_null = model(
                x, clipped_t, model.get_null_cond(actions), tactile_state=tactile
            )
            v_pred = (1 - cfg) * v_pred_null + cfg * v_pred_cond
        else:
            v_pred = v_pred_cond

        x_start = alphas_cumprod.sqrt() * x - (1 - alphas_cumprod).sqrt() * v_pred
        pred_noise = ((1 / alphas_cumprod).sqrt() * x - x_start) / (
            (1 / alphas_cumprod) - 1
        ).sqrt()
        x_pred = alphas_next_cumprod.sqrt() * x_start + c * pred_noise
        x_pred = torch.where(
            (t == t_next).view(B, T, 1, 1, 1),
            orig_x,
            x_pred,
        )
        return x_pred

    def generate_pyramid_scheduling_matrix(self, horizon: int) -> torch.Tensor:
        height = self.sampling_timesteps + horizon
        scheduling_matrix = torch.zeros((height, horizon), dtype=torch.long)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + t - m
        return torch.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        n_context_frames: int = 1,
        n_frames: int = 1,
        horizon: int = 1,
        window_len: int | None = None,
        cfg: float = 0.0,
        tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape
        curr_frame = 0
        x_pred = x[:, :n_context_frames]
        curr_frame += n_context_frames
        tactile_state = latest_context_tactile_state(tactile, n_context_frames)

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            horizon = min(n_frames - curr_frame, horizon)
            scheduling_matrix = self.generate_pyramid_scheduling_matrix(horizon)

            chunk = torch.randn((B, horizon, *x.shape[-3:]), device=self.device)
            x_pred = torch.cat([x_pred, chunk], dim=1)

            # Adjust context length
            start_frame = max(
                0, curr_frame + horizon - (window_len or model.max_frames)
            )

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + horizon,
                }
            )

            for m in range(scheduling_matrix.shape[0] - 1):
                t, t_next = scheduling_matrix[m], scheduling_matrix[m + 1]
                t, t_next = map(
                    lambda x: einops.repeat(x, "t -> b t", b=B), (t, t_next)
                )
                t, t_next = map(
                    lambda x: torch.cat(
                        (torch.zeros((B, curr_frame), dtype=torch.long), x), dim=1
                    ),
                    (t, t_next),
                )

                x_pred[:, start_frame:] = self.ddim_sample_step(
                    model,
                    x_pred[:, start_frame:],
                    actions[:, start_frame : curr_frame + horizon],
                    t[:, start_frame:],
                    t_next[:, start_frame:],
                    tactile=tactile_state,
                    cfg=cfg,
                )

            curr_frame += horizon
            pbar.update(horizon)

        return x_pred


class FlowMatching(nn.Module):
    def __init__(
        self,
        timesteps: int = 1_000,
        sampling_timesteps: int = 10,
        time_dist_type: str = "uniform",
        time_dist_shift: float = 1.0,
        logit_mu: float = 0.0,
        logit_sigma: float = 1.0,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.time_dist_type = time_dist_type
        self.time_dist_shift = time_dist_shift
        self.logit_mu = logit_mu
        self.logit_sigma = logit_sigma
        self.stabilization_level = 0.01
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

    def time_shift(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_dist_shift == 1.0:
            return t
        return (self.time_dist_shift * t) / (1 + (self.time_dist_shift - 1) * t)

    def sample_t(
        self, B: int, T: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self.time_dist_type == "uniform":
            t = torch.rand((B, T), dtype=dtype, device=device)
        elif self.time_dist_type == "logit_normal":
            t = truncated_logitnormal_sample(
                (B, T),
                mu=self.logit_mu,
                sigma=self.logit_sigma,
                low=0.0,
                high=1.0,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown time_dist_type: {self.time_dist_type}")
        return self.time_shift(t).to(dtype)

    def loss_fn(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        num_history: int = 0,
        tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, H, W, C = x.shape

        if num_history > 0 and T > num_history:
            # Split into history (clean context) and future (noised prediction target)
            # This matches ctrl-world-rae: history frames get t≈0, only future is noised,
            # loss is computed on future frames only.
            T_future = T - num_history
            history = x[:, :num_history]  # (B, num_history, H, W, C)
            future = x[:, num_history:]  # (B, T_future, H, W, C)

            # Add light noise augmentation to history frames (SVD-style, prevents overfitting)
            if model.training:
                sigma_h = (
                    torch.randn(B, num_history, 1, 1, 1, device=x.device, dtype=x.dtype)
                    * 0.3
                )
                noise_h = torch.randn_like(history)
                norm_factor = torch.sqrt(sigma_h.pow(2) + 1.0)
                history_aug = (history + sigma_h * noise_h) / norm_factor
            else:
                history_aug = history

            # Sample t and noise only for future frames
            t_future = self.sample_t(B, T_future, x.dtype, self.device)
            x_1_future = torch.randn_like(future)
            x_t_future = torch.lerp(
                future, x_1_future, t_future.view(B, T_future, 1, 1, 1)
            )

            # Concatenate: history (clean) + future (noised)
            x_t = torch.cat([history_aug, x_t_future], dim=1)  # (B, T, H, W, C)

            # History frames get t=0 (clean signal), matching ctrl-world-rae
            t_history = torch.zeros(B, num_history, dtype=x.dtype, device=self.device)
            t = torch.cat([t_history, t_future], dim=1)  # (B, T)

            t_input = t
            if not getattr(model, "use_normalized_t", False):
                t_input = t * self.timesteps

            tactile_state = (
                latest_context_tactile_state(tactile, num_history)
                if tactile is not None
                else None
            )
            v_t = model(x_t, t_input, actions, tactile_state=tactile_state)

            # Loss only on future frames (history frames are context, not prediction targets)
            v_future = v_t[:, num_history:]
            target_future = x_1_future - future
            loss = F.mse_loss(v_future, target_future)
        else:
            # Standard mode: all frames are noised and predicted
            t = self.sample_t(B, T, x.dtype, self.device)
            x_1 = torch.randn_like(x)
            x_t = torch.lerp(x, x_1, t.view(B, T, 1, 1, 1))

            t_input = t
            if not getattr(model, "use_normalized_t", False):
                t_input = t * self.timesteps
            tactile_state = (
                latest_context_tactile_state(tactile, T)
                if tactile is not None
                else None
            )
            v_t = model(x_t, t_input, actions, tactile_state=tactile_state)

            loss = F.mse_loss(v_t, x_1 - x)
        return loss

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        n_context_frames: int = 1,
        n_frames: int = 1,
        horizon: int = 1,
        window_len: int | None = None,
        cfg: float = 0.0,
        tactile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert horizon == 1
        assert window_len is None

        B, T, H, W, C = x.shape
        curr_frame = 0
        x_pred = x[:, :n_context_frames]
        curr_frame += n_context_frames
        tactile_state = latest_context_tactile_state(tactile, n_context_frames)

        schedule = torch.linspace(
            1.0, 0.0, self.sampling_timesteps + 1, dtype=x.dtype, device=self.device
        )
        schedule = self.time_shift(schedule)

        pbar = tqdm(total=n_frames, initial=curr_frame, desc="Sampling")
        while curr_frame < n_frames:
            chunk = torch.randn((B, 1, *x.shape[-3:]), device=self.device)
            x_pred = torch.cat([x_pred, chunk], dim=1)
            # Adjust context length
            start_frame = max(0, curr_frame + 1 - model.max_frames)

            pbar.set_postfix(
                {
                    "start": start_frame,
                    "end": curr_frame + 1,
                }
            )

            t = torch.full(
                (B, curr_frame),
                self.stabilization_level,
                dtype=x.dtype,
                device=self.device,
            )
            for i in range(self.sampling_timesteps):
                t_curr, t_next = schedule[i], schedule[i + 1]
                dt = t_curr - t_next
                # predict the velocity
                t_curr = torch.cat([t, einops.repeat(t_curr, " -> b 1", b=B)], dim=1)
                t_input = t_curr[:, start_frame:]
                if not getattr(model, "use_normalized_t", False):
                    t_input = t_input * self.timesteps

                v_cond = model(
                    x_pred[:, start_frame:],
                    t_input,
                    actions[:, start_frame : curr_frame + 1],
                    tactile_state=tactile_state,
                )
                if cfg >= 1.0:
                    # Pure conditional — no need to evaluate the null model
                    v_pred = v_cond
                else:
                    v_null = model(
                        x_pred[:, start_frame:],
                        t_input,
                        model.get_null_cond(actions)[:, start_frame : curr_frame + 1],
                        tactile_state=tactile_state,
                    )
                    v_pred = (1 - cfg) * v_null + cfg * v_cond
                # take a step in the backwards velocity direction
                x_pred[:, -1:] -= dt * v_pred[:, -1:]

            curr_frame += 1
            pbar.update(1)

        return x_pred
