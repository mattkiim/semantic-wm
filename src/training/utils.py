"""Shared training utilities: distributed setup, EMA, checkpointing, WandB."""

from __future__ import annotations

import datetime
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
import wandb


# ---------------------------------------------------------------------------
# Distributed / grad helpers
# ---------------------------------------------------------------------------


def init_distributed() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed if available.

    Returns (local_rank, global_rank, world_size, is_distributed).
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, world_size, True
    return 0, 0, 1, False


def requires_grad(model: nn.Module, flag: bool = True) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def maybe_compile(model: nn.Module, name: str, compile: bool = True) -> nn.Module:
    if compile and hasattr(torch, "compile"):
        logging.info("Compiling %s with torch.compile", name)
        return torch.compile(model)
    return model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


# ---------------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------------


def init_wandb(args, checkpoint_dir: Path, run_name: str) -> None:
    """Initialize WandB on rank 0, resuming a prior run if a saved ID exists."""
    run_id_file = checkpoint_dir / "wandb_run_id.txt"
    wandb_resume_id = run_id_file.read_text().strip() if run_id_file.exists() else None
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        name=run_name if wandb_resume_id is None else None,
        id=wandb_resume_id,
        resume="allow" if wandb_resume_id is not None else None,
        mode=args.wandb_mode,
        config=vars(args) if hasattr(args, "__dict__") else dict(args),
        tags=args.wandb_tags,
    )
    if wandb.run is not None:
        run_id_file.write_text(wandb.run.id)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_model_param_counts(
    autoencoder,
    dit_model: Optional[nn.Module] = None,
    adapter: Optional[nn.Module] = None,
    pixel_decoder: Optional[nn.Module] = None,
) -> None:
    """Log a comprehensive parameter count for all model components."""
    from ..models.adapters import IdentityAdapter

    lines = []
    if hasattr(autoencoder, "encoder"):
        enc = sum(p.numel() for p in autoencoder.encoder.parameters())
        dec = (
            sum(p.numel() for p in autoencoder.decoder.parameters())
            if hasattr(autoencoder, "decoder")
            else 0
        )
        lines.append(
            f"  Autoencoder [{type(autoencoder).__name__}]"
            f"  d_h={autoencoder.latent_dim}"
            f"  encoder={enc:,}  decoder={dec:,}"
            f"  total={enc + dec:,}"
        )
    if adapter is not None and not isinstance(adapter, IdentityAdapter):
        n = sum(p.numel() for p in adapter.parameters())
        lines.append(
            f"  Adapter     [{type(adapter).__name__}]"
            f"  d_l={adapter.latent_dim}  params={n:,}"
        )
    if dit_model is not None:
        n = sum(p.numel() for p in dit_model.parameters())
        lines.append(f"  DiT         [{type(dit_model).__name__}]  params={n:,}")
    if pixel_decoder is not None:
        n = sum(p.numel() for p in pixel_decoder.parameters())
        lines.append(
            f"  PixelDecoder [{type(pixel_decoder).__name__}]  params={n:,}"
        )
    logging.info("Model parameter counts:\n%s", "\n".join(lines))


def log_training_config(
    args,
    dataset_size: int,
    steps_per_epoch: int,
    max_train_steps: int,
    world_size: int,
) -> None:
    grad_accum = max(int(getattr(args, "gradient_accumulation_steps", 1)), 1)
    logging.info(
        "Training config | dataset=%d  batch/gpu=%d  grad_accum=%d  gpus=%d  eff_batch=%d  "
        "updates/epoch=%d  epochs=%d  total_updates=%d  "
        "log_every=%d  val_every=%d  history=%d  lr=%.2e",
        dataset_size,
        args.batch_size,
        grad_accum,
        world_size,
        args.batch_size * world_size * grad_accum,
        steps_per_epoch,
        args.num_epochs,
        max_train_steps,
        args.log_every_samples,
        args.validate_every_samples,
        args.num_history,
        args.lr,
    )


# ---------------------------------------------------------------------------
# DiT training checkpoints
# ---------------------------------------------------------------------------


def load_training_checkpoint(
    checkpoint_dir: Path,
    model_no_ddp: nn.Module,
    ema: nn.Module,
    optimizer,
    lr_scheduler,
    device,
) -> tuple[int, Optional[int]]:
    """Load the latest DiT training checkpoint.

    Returns (train_steps, total_samples_seen). Both are 0/None if no checkpoint.
    """
    ckpts = sorted(checkpoint_dir.glob("ckpt_samples_*.pt"))
    if not ckpts:
        ckpts = sorted(checkpoint_dir.glob("ckpt_*.pt"))
    if not ckpts:
        return 0, None

    if "samples" in ckpts[-1].stem:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    else:
        latest = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

    data = torch.load(latest, map_location=device)
    model_no_ddp.load_state_dict(data["model"])
    optimizer.load_state_dict(data["optimizer"])
    if "ema" in data:
        ema.load_state_dict(data["ema"])
    else:
        update_ema(ema, model_no_ddp, 0.0)
    train_steps = int(data.get("step", 0))
    total_samples_seen = data.get("total_samples_seen", None)

    if "lr_scheduler" in data:
        lr_scheduler.load_state_dict(data["lr_scheduler"])
    elif train_steps > 0:
        for _ in range(train_steps):
            lr_scheduler.step()
        logging.info("Fast-forwarded LR scheduler to step %d", train_steps)

    logging.info("Loaded checkpoint %s (step %d)", latest, train_steps)
    return train_steps, total_samples_seen


def save_training_checkpoint(
    checkpoint_dir: Path,
    model_no_ddp: nn.Module,
    ema: nn.Module,
    optimizer,
    lr_scheduler,
    train_steps: int,
    total_samples_seen: int,
) -> None:
    """Save checkpoint and prune all but the most recent."""
    torch.save(
        {
            "model": model_no_ddp.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": train_steps,
            "total_samples_seen": total_samples_seen,
        },
        checkpoint_dir / f"ckpt_samples_{total_samples_seen:012d}.pt",
    )
    for old in sorted(checkpoint_dir.glob("ckpt_samples_*.pt"))[:-1]:
        os.remove(old)


# ---------------------------------------------------------------------------
# Adapter checkpoint helpers (used in both train.py and train_adapter.py)
# ---------------------------------------------------------------------------


def resolve_adapter_ckpt(args, device) -> tuple[dict, Optional[dict]]:
    """Return (adapter_cfg, ckpt_data_or_none).

    Loads and validates the adapter checkpoint if a path is given.
    The returned adapter_cfg may be overridden by values stored in the checkpoint.
    """
    from ..models.adapters import adapter_config_from_args

    adapter_cfg = adapter_config_from_args(args)
    ckpt_path = getattr(args, "adapter_checkpoint_path", None)
    # VAE uses an identity adapter — ignore any adapter checkpoint path.
    if ckpt_path is None or adapter_cfg.get("adapter_type", "identity") == "identity":
        return adapter_cfg, None

    ckpt_data = torch.load(ckpt_path, map_location=device)
    if "adapter_config" in ckpt_data:
        logging.info("Using adapter_config from checkpoint")
        adapter_cfg = ckpt_data["adapter_config"]

    encoder_cfg_ckpt = ckpt_data.get("encoder_config", {})
    if encoder_cfg_ckpt:
        ckpt_enc_type = encoder_cfg_ckpt.get("encoder_type")
        assert ckpt_enc_type == args.encoder_type, (
            f"Encoder type mismatch! Checkpoint: {ckpt_enc_type}, Args: {args.encoder_type}"
        )
        ckpt_norm = encoder_cfg_ckpt.get("encoder_normalization_stat_path")
        args_norm = getattr(args, "encoder_normalization_stat_path", None)
        if ckpt_norm and args_norm:
            assert os.path.basename(ckpt_norm) == os.path.basename(args_norm), (
                f"Norm stats mismatch! Checkpoint used {os.path.basename(ckpt_norm)}, "
                f"but args specified {os.path.basename(args_norm)}"
            )

    return adapter_cfg, ckpt_data


def load_frozen_adapter_weights(
    adapter: nn.Module,
    pixel_decoder: Optional[nn.Module],
    ckpt_data: dict,
) -> None:
    """Load adapter (and optionally pixel decoder) weights from a checkpoint dict."""
    state_dict = strip_state_dict_prefix(ckpt_data["adapter"])
    adapter.load_state_dict(state_dict)
    logging.info("Loaded pretrained adapter weights")

    if pixel_decoder is not None and ckpt_data.get("pixel_decoder"):
        pd_state = strip_state_dict_prefix(ckpt_data["pixel_decoder"])
        pixel_decoder.load_state_dict(pd_state)
        logging.info("Loaded pretrained pixel decoder weights")


def setup_pixel_decoder_for_val(
    args, ckpt_data: Optional[dict], device
) -> Optional[nn.Module]:
    """Create pixel decoder for validation if configured.

    Sets args.use_pixel_decoder_for_val=False and returns None when the
    checkpoint doesn't contain a pixel decoder config.
    """
    from ..models.pixel_decoder import create_pixel_decoder

    if not getattr(args, "use_pixel_decoder_for_val", False):
        return None
    if ckpt_data is not None and "pixel_decoder_config" in ckpt_data:
        return create_pixel_decoder(ckpt_data["pixel_decoder_config"]).to(device)
    logging.warning(
        "use_pixel_decoder_for_val=True but no pixel_decoder_config in checkpoint; "
        "falling back to standard decoding."
    )
    args.use_pixel_decoder_for_val = False
    return None


# ---------------------------------------------------------------------------
# Adapter training checkpoints
# ---------------------------------------------------------------------------


def load_adapter_training_checkpoint(
    checkpoint_dir: Path,
    adapter_no_ddp: nn.Module,
    pixel_decoder_no_ddp: Optional[nn.Module],
    optimizer,
    lr_scheduler,
    device,
) -> tuple[int, int]:
    """Resume adapter training from the latest checkpoint.

    Returns (train_steps, total_samples_seen).
    """
    ckpts = sorted(checkpoint_dir.glob("adapter_ckpt_*.pt"))
    if not ckpts:
        return 0, 0

    latest = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    data = torch.load(latest, map_location=device)
    adapter_no_ddp.load_state_dict(data["adapter"])
    optimizer.load_state_dict(data["optimizer"])
    train_steps = int(data.get("step", 0))
    total_samples_seen = int(data.get("total_samples_seen", 0))

    if "lr_scheduler" in data:
        lr_scheduler.load_state_dict(data["lr_scheduler"])
    else:
        for _ in range(train_steps):
            lr_scheduler.step()

    if pixel_decoder_no_ddp is not None and "pixel_decoder" in data:
        pixel_decoder_no_ddp.load_state_dict(data["pixel_decoder"])
        logging.info("Resumed pixel_decoder weights from %s", latest)

    logging.info(
        "Resumed from %s (step=%d, samples=%d)", latest, train_steps, total_samples_seen
    )
    return train_steps, total_samples_seen


def save_adapter_checkpoint(
    checkpoint_dir: Path,
    adapter_no_ddp: nn.Module,
    pixel_decoder_no_ddp: Optional[nn.Module],
    optimizer,
    lr_scheduler,
    train_steps: int,
    total_samples_seen: int,
    adapter_config: dict,
    encoder_config: dict,
    pixel_decoder_config: Optional[dict] = None,
) -> None:
    """Save adapter checkpoint and prune all but the most recent."""
    ckpt_data: dict = {
        "adapter": adapter_no_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "step": train_steps,
        "total_samples_seen": total_samples_seen,
        "adapter_config": adapter_config,
        "encoder_config": encoder_config,
    }
    if pixel_decoder_no_ddp is not None and pixel_decoder_config is not None:
        ckpt_data["pixel_decoder"] = pixel_decoder_no_ddp.state_dict()
        ckpt_data["pixel_decoder_config"] = pixel_decoder_config

    torch.save(ckpt_data, checkpoint_dir / f"adapter_ckpt_{total_samples_seen:012d}.pt")
    for old in sorted(checkpoint_dir.glob("adapter_ckpt_*.pt"))[:-1]:
        os.remove(old)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def strip_state_dict_prefix(state_dict: dict) -> dict:
    """Remove 'module.' and '_orig_mod.' prefixes left by DDP / torch.compile."""
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict


def downsample_actions_temporal(actions: torch.Tensor, factor: int) -> torch.Tensor:
    """Concatenate consecutive action frames to match temporal downsampling.

    When an encoder reduces the temporal dimension by ``factor`` (e.g. 2 for
    tubelet embeddings), the per-frame action vectors must be merged so that
    the action sequence length matches the latent temporal dimension.

    Parameters
    ----------
    actions : (B, T, D)
    factor : int  (e.g. 2)

    Returns
    -------
    (B, T // factor, D * factor)
    """
    B, T, D = actions.shape
    assert T % factor == 0, (
        f"Temporal dim T={T} is not divisible by downsample factor={factor}"
    )
    return actions.reshape(B, T // factor, factor * D)


def downsample_sequence_temporal(sequence: torch.Tensor, factor: int) -> torch.Tensor:
    """Concatenate consecutive per-frame vectors after temporal downsampling."""
    B, T, D = sequence.shape
    assert T % factor == 0, (
        f"Temporal dim T={T} is not divisible by downsample factor={factor}"
    )
    return sequence.reshape(B, T // factor, factor * D)
