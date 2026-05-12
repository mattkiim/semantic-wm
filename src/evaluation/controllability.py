"""Controllability metric: measures how well the DiT responds to actions in latent space.

For each 2-frame transition (o1, o2) with ground truth action a_gt:
1. Find a' = argmin_a ||DiT(o1, a) - o2||^2
2. Report ||a_gt - a'|| as controllability error (lower = better)

Action convention: action[t] is embedded with frame[t] and drives the transition
from frame[t] to frame[t+1]. So for a pair (frame_t, frame_{t+1}), the action
that controls the transition is action[t] (the context frame's action).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dimension labels for per-dim reporting
_DIM_NAMES = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "gripper"]

# Candidate evaluation function: (N, action_dim) -> (N,) losses
EvalFn = Callable[[torch.Tensor], torch.Tensor]


def _build_full_candidates(
    partial_candidates: torch.Tensor,
    gt_action: torch.Tensor,
    search_dims: List[int],
    action_dim: int,
) -> torch.Tensor:
    """Build full action tensors from partial candidates over search_dims.

    Parameters
    ----------
    partial_candidates : (N, len(search_dims)) values for searched dims
    gt_action : (action_dim,) ground truth to fill non-searched dims
    search_dims : which dims are being searched
    action_dim : total action dimensionality

    Returns
    -------
    (N, action_dim) full action tensors
    """
    N = partial_candidates.shape[0]
    candidates = gt_action.unsqueeze(0).expand(N, -1).clone()
    for i, dim_idx in enumerate(search_dims):
        candidates[:, dim_idx] = partial_candidates[:, i]
    return candidates


def generate_single_step_batched(
    diffusion,
    model,
    o1_latent: torch.Tensor,
    candidate_actions: torch.Tensor,
    target_action: torch.Tensor,
    noise: torch.Tensor | None = None,
    precision: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run batched single-frame denoising for N candidate context actions.

    Parameters
    ----------
    diffusion : FlowMatching instance
    model : DiT model
    o1_latent : (1, 1, H, W, C) context frame latent
    candidate_actions : (N, action_dim) full candidate actions for the context frame
    target_action : (1, action_dim) action paired with target frame (held fixed)
    noise : (1, 1, H, W, C) fixed noise; if None, sampled once
    precision : autocast dtype

    Returns
    -------
    (N, 1, H, W, C) predicted o2 latents
    """
    N = candidate_actions.shape[0]
    device = o1_latent.device

    # Expand o1 to (N, 1, H, W, C) without copying
    o1_exp = o1_latent.expand(N, -1, -1, -1, -1)

    # Sample or expand noise
    if noise is None:
        noise = torch.randn_like(o1_latent)
    noise_exp = noise.expand(N, -1, -1, -1, -1).clone()

    # x_pred = [o1, noise] -> (N, 2, H, W, C)
    x_pred = torch.cat([o1_exp, noise_exp], dim=1)

    # Build action tensor: [candidate (context), target] -> (N, 2, action_dim)
    tgt_exp = target_action.expand(N, -1)
    actions = torch.stack([candidate_actions, tgt_exp], dim=1)

    # Build denoising schedule
    schedule = torch.linspace(1.0, 0.0, diffusion.sampling_timesteps + 1,
                              dtype=o1_latent.dtype, device=device)
    schedule = diffusion.time_shift(schedule)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision):
        for i in range(diffusion.sampling_timesteps):
            t_curr, t_next = schedule[i], schedule[i + 1]
            dt = t_curr - t_next

            t_ctx = torch.full((N, 1), diffusion.stabilization_level,
                               dtype=o1_latent.dtype, device=device)
            t_tgt = einops.repeat(t_curr, " -> b 1", b=N)
            t_input = torch.cat([t_ctx, t_tgt], dim=1)

            if not getattr(model, "use_normalized_t", False):
                t_input = t_input * diffusion.timesteps

            v_pred = model(x_pred, t_input, actions)
            x_pred[:, -1:] -= dt * v_pred[:, -1:]

    return x_pred[:, -1:]


# ---------------------------------------------------------------------------
# Optimizer implementations
# ---------------------------------------------------------------------------

def cem_find_best_action(
    eval_fn: EvalFn,
    gt_action: torch.Tensor,
    action_dim: int,
    search_dims: tuple,
    n_candidates: int = 256,
    n_elite: int = 25,
    n_iterations: int = 5,
    action_low: float = -2.0,
    action_high: float = 2.0,
    max_batch_size: int = 64,
) -> Tuple[torch.Tensor, float]:
    """Find best action via Cross-Entropy Method (CEM).

    Only dims in ``search_dims`` are optimized; the rest are filled from
    ``gt_action``. Returns (best_action, best_loss).

    Parameters
    ----------
    eval_fn : callable (N, action_dim) -> (N,) losses
    gt_action : (action_dim,) ground truth to fill non-searched dims
    action_dim : total action dimensionality
    search_dims : which dims to optimize
    """
    device = gt_action.device
    n_search = len(search_dims)

    # Init distribution over searched dims only
    mu = gt_action[list(search_dims)].clone()
    sigma = torch.full((n_search,), (action_high - action_low) / 4.0, device=device)

    best_action_overall = None
    best_loss_overall = float("inf")

    for _ in range(n_iterations):
        # Sample candidates for searched dims
        partial = mu + sigma * torch.randn(n_candidates, n_search, device=device)
        partial = partial.clamp(action_low, action_high)

        # Build full action tensors
        candidates = _build_full_candidates(partial, gt_action, list(search_dims), action_dim)

        # Evaluate in sub-batches
        all_losses = []
        for start in range(0, n_candidates, max_batch_size):
            end = min(start + max_batch_size, n_candidates)
            loss = eval_fn(candidates[start:end])
            all_losses.append(loss)

        all_losses = torch.cat(all_losses)

        # Track best across all iterations
        batch_best_idx = all_losses.argmin()
        batch_best_loss = all_losses[batch_best_idx].item()
        if batch_best_loss < best_loss_overall:
            best_loss_overall = batch_best_loss
            best_action_overall = candidates[batch_best_idx].clone()

        # Select elite and update distribution
        elite_indices = all_losses.topk(n_elite, largest=False).indices
        elite = partial[elite_indices]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp(min=1e-4)

    return best_action_overall, best_loss_overall


def gradient_find_best_action(
    diffusion,
    model,
    o1_latent: torch.Tensor,
    o2_target: torch.Tensor,
    target_action: torch.Tensor,
    gt_action: torch.Tensor | None = None,
    action_dim: int = 10,
    search_dims: tuple = (0, 1, 2, 3, 4, 5, 6),
    n_steps: int = 50,
    lr: float = 0.01,
    action_low: float = -2.0,
    action_high: float = 2.0,
    precision: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, float]:
    """Find best context action via gradient-based optimization.

    Only dims in ``search_dims`` are optimized; all other dims are filled from
    ``gt_action``. Returns (best_action, best_loss).
    """
    assert gt_action is not None, "Gradient optimizer requires gt_action to fill non-searched dims"
    device = o1_latent.device
    search_dims_list = list(search_dims)
    n_search = len(search_dims_list)

    # Learnable parameter: only the searched dims, initialized from GT
    a_opt = gt_action[search_dims_list].clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([a_opt], lr=lr)

    # Fixed noise
    noise = torch.randn_like(o1_latent)

    # Build denoising schedule once
    schedule = torch.linspace(1.0, 0.0, diffusion.sampling_timesteps + 1,
                              dtype=o1_latent.dtype, device=device)
    schedule = diffusion.time_shift(schedule)

    best_action = None
    best_loss = float("inf")

    for _ in range(n_steps):
        optimizer.zero_grad()

        # Build full action: start from gt, override searched dims
        full_action = gt_action.clone()
        for i, dim_idx in enumerate(search_dims_list):
            full_action[dim_idx] = a_opt[i]

        # Setup: (1, 2, H, W, C) and (1, 2, action_dim)
        x_pred = torch.cat([o1_latent, noise.clone()], dim=1)
        actions = torch.stack([full_action, target_action.squeeze(0)]).unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=precision):
            for i in range(diffusion.sampling_timesteps):
                t_curr, t_next = schedule[i], schedule[i + 1]
                dt = t_curr - t_next

                t_ctx = torch.full((1, 1), diffusion.stabilization_level,
                                   dtype=o1_latent.dtype, device=device)
                t_tgt = einops.repeat(t_curr, " -> b 1", b=1)
                t_input = torch.cat([t_ctx, t_tgt], dim=1)
                if not getattr(model, "use_normalized_t", False):
                    t_input = t_input * diffusion.timesteps

                v_pred = model(x_pred, t_input, actions)
                x_pred = torch.cat([x_pred[:, :1], x_pred[:, -1:] - dt * v_pred[:, -1:]], dim=1)

            pred_o2 = x_pred[:, -1:]
            loss = F.mse_loss(pred_o2, o2_target.unsqueeze(0))

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            a_opt.clamp_(action_low, action_high)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_full = gt_action.clone()
            for i, dim_idx in enumerate(search_dims_list):
                best_full[dim_idx] = a_opt[i].detach()
            best_action = best_full

    return best_action, best_loss


def grid_find_best_action(
    eval_fn: EvalFn,
    gt_action: torch.Tensor,
    action_dim: int,
    search_dims: tuple,
    n_points_per_dim: int = 30,
    action_low: float = -2.0,
    action_high: float = 2.0,
    max_batch_size: int = 64,
) -> Tuple[torch.Tensor, float]:
    """Find best action via grid search over a subset of dims.

    Only ``search_dims`` are swept; all other dims are filled from ``gt_action``.

    Parameters
    ----------
    eval_fn : callable (N, action_dim) -> (N,) losses
    """
    device = gt_action.device
    search_dims_list = list(search_dims)
    n_search = len(search_dims_list)

    # Create grid over searched dims only
    linspaces = [torch.linspace(action_low, action_high, n_points_per_dim, device=device)
                 for _ in range(n_search)]
    grids = torch.meshgrid(*linspaces, indexing="ij")
    partial = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (N_total, n_search)

    n_total = partial.shape[0]
    logger.info("Grid search: %d candidates (%d^%d over dims %s)",
                n_total, n_points_per_dim, n_search, search_dims)

    # Build full action tensors
    candidates = _build_full_candidates(partial, gt_action, search_dims_list, action_dim)

    # Evaluate in sub-batches
    best_action = None
    best_loss = float("inf")

    for start in tqdm(range(0, n_total, max_batch_size), desc="Grid search"):
        end = min(start + max_batch_size, n_total)
        losses = eval_fn(candidates[start:end])

        batch_best_idx = losses.argmin()
        batch_best_loss = losses[batch_best_idx].item()
        if batch_best_loss < best_loss:
            best_loss = batch_best_loss
            best_action = candidates[start + batch_best_idx].clone()

    return best_action, best_loss


# ---------------------------------------------------------------------------
# Aggregation and persistence
# ---------------------------------------------------------------------------

def _aggregate_results(
    all_errors: list,
    all_per_dim_errors: list,
    all_gt_losses: list,
    search_dims_list: list,
    pairs_processed: int,
    num_eval_samples: int,
) -> Dict:
    """Build results dict from accumulated per-pair data."""
    errors_tensor = torch.tensor(all_errors)
    per_dim_tensor = torch.stack(all_per_dim_errors)
    gt_losses_tensor = torch.tensor(all_gt_losses)

    return {
        "pairs_evaluated": pairs_processed,
        "pairs_requested": num_eval_samples,
        "controllability_mean": float(errors_tensor.mean()),
        "controllability_std": float(errors_tensor.std()) if len(errors_tensor) > 1 else 0.0,
        "controllability_median": float(errors_tensor.median()),
        "controllability_per_dim": {
            name: float(per_dim_tensor[:, i].mean())
            for i, name in enumerate(_DIM_NAMES)
        },
        "search_dims": search_dims_list,
        "gt_sanity_loss_mean": float(gt_losses_tensor.mean()),
        "gt_sanity_loss_std": float(gt_losses_tensor.std()) if len(gt_losses_tensor) > 1 else 0.0,
        "gt_sanity_loss_median": float(gt_losses_tensor.median()),
    }


def _save_results(results: Dict, save_path: Path) -> None:
    """Atomically save results to JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    tmp_path.rename(save_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_controllability(
    diffusion,
    model,
    autoencoder,
    adapter,
    is_identity: bool,
    test_loader: DataLoader,
    device: torch.device,
    precision: torch.dtype,
    num_eval_samples: int = 2048,
    optimizer: str = "cem",
    save_path: Optional[Path] = None,
    **optimizer_kwargs,
) -> Dict:
    """Compute controllability metric over the test set.

    For each consecutive pair (frame_t, frame_{t+1}), optimizes action[t]
    (the context frame action that drives the transition) to minimize
    ||DiT(frame_t, a) - frame_{t+1}||^2 in latent space.

    Only dims specified by `search_dims` are optimized; the rest are filled
    from the ground truth action.

    Also runs a GT sanity check: passes the true action[t] through denoising
    to measure baseline latent MSE (should be low if the model is controllable).

    Results are saved incrementally to `save_path` after each pair, so partial
    results survive interruptions.

    Returns dict with mean/std/median L2 error, per-dim breakdown, and GT loss.
    """
    _OPTIMIZERS = {
        "cem": cem_find_best_action,
        "gradient": gradient_find_best_action,
        "grid": grid_find_best_action,
    }
    find_best_action = _OPTIMIZERS[optimizer]
    action_dim = optimizer_kwargs.pop("action_dim", 10)
    optimizer_kwargs.pop("precision", None)

    # Get search_dims from kwargs (all methods now support it)
    search_dims = optimizer_kwargs.get("search_dims", None)
    search_dims_list = list(search_dims) if search_dims is not None else list(range(7))

    if save_path is not None:
        save_path = Path(save_path)

    all_errors = []
    all_per_dim_errors = []
    all_gt_losses = []
    pairs_processed = 0

    dim_names_searched = [_DIM_NAMES[i] for i in search_dims_list if i < len(_DIM_NAMES)]
    logger.info(
        "Computing controllability with optimizer=%s, target pairs=%d, search_dims=%s (%s)",
        optimizer, num_eval_samples, search_dims_list, dim_names_searched,
    )

    pbar = tqdm(total=num_eval_samples, desc="Controllability pairs")

    for _, (x, actions) in enumerate(test_loader):
        if pairs_processed >= num_eval_samples:
            break

        x = x.to(device)
        actions = actions.to(device)
        B, T = x.shape[0], x.shape[1]

        # Encode to latent space (once per batch)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=precision):
            z = autoencoder.encode(x)
            if not is_identity:
                z_adapted = adapter.encode(z)
                if isinstance(z_adapted, tuple):
                    z_adapted = z_adapted[0]
            else:
                z_adapted = z

        # Extract all T-1 consecutive pairs
        for b in range(B):
            for t in range(z_adapted.shape[1] - 1):
                if pairs_processed >= num_eval_samples:
                    break

                o1 = z_adapted[b, t:t+1].unsqueeze(0)       # (1, 1, H, W, C)
                o2 = z_adapted[b, t+1:t+2]                   # (1, H, W, C)
                gt_action = actions[b, t]                     # (action_dim,)
                target_action = actions[b, t+1:t+2]           # (1, action_dim)

                # Fixed noise for fair candidate comparison
                noise = torch.randn_like(o1)

                def _make_eval_fn(o1_l, o2_t, tgt_a, ns):
                    def fn(candidates):
                        pred = generate_single_step_batched(
                            diffusion, model, o1_l, candidates, tgt_a,
                            noise=ns, precision=precision,
                        )
                        return (pred - o2_t.unsqueeze(0)).pow(2).mean(dim=(1, 2, 3, 4))
                    return fn

                eval_fn = _make_eval_fn(o1, o2, target_action, noise)

                # GT sanity check
                gt_loss = eval_fn(gt_action.unsqueeze(0)).item()
                all_gt_losses.append(gt_loss)

                # Find best action (all methods get gt_action for non-searched dims)
                best_action, best_loss = find_best_action(
                    diffusion, model, o1, o2, target_action,
                    gt_action=gt_action, action_dim=action_dim, precision=precision,
                    **optimizer_kwargs,
                ) if optimizer == "gradient" else find_best_action(
                    eval_fn, gt_action,
                    action_dim=action_dim,
                    search_dims=tuple(search_dims_list),
                    **optimizer_kwargs,
                )

                # L2 error between ground truth and best-found action
                error = (gt_action - best_action).pow(2).sqrt()
                l2_error = error.norm().item()

                all_errors.append(l2_error)
                all_per_dim_errors.append(error[:7].detach().cpu())

                pairs_processed += 1
                pbar.update(1)
                pbar.set_postfix(
                    l2_err=f"{l2_error:.4f}",
                    opt_loss=f"{best_loss:.6f}",
                    gt_loss=f"{gt_loss:.6f}",
                )

                # Save incrementally after each pair
                if save_path is not None:
                    results = _aggregate_results(
                        all_errors, all_per_dim_errors, all_gt_losses,
                        search_dims_list, pairs_processed, num_eval_samples,
                    )
                    _save_results(results, save_path)

            if pairs_processed >= num_eval_samples:
                break

    pbar.close()

    # Final aggregation
    results = _aggregate_results(
        all_errors, all_per_dim_errors, all_gt_losses,
        search_dims_list, pairs_processed, num_eval_samples,
    )
    if save_path is not None:
        _save_results(results, save_path)
        logger.info("Final results saved to %s", save_path)

    logger.info(
        "Controllability: mean=%.4f, std=%.4f, median=%.4f (%d/%d pairs)",
        results["controllability_mean"],
        results["controllability_std"],
        results["controllability_median"],
        pairs_processed,
        num_eval_samples,
    )
    logger.info(
        "GT sanity check (latent MSE with true action): mean=%.6f, std=%.6f, median=%.6f",
        results["gt_sanity_loss_mean"],
        results["gt_sanity_loss_std"],
        results["gt_sanity_loss_median"],
    )
    for name, val in results["controllability_per_dim"].items():
        logger.info("  %s: %.4f", name, val)

    return results
