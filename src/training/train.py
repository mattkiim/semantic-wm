"""World-model DiT training loop."""

import datetime
import logging
from copy import deepcopy
from pathlib import Path

import einops
import torch
import torch.distributed as dist
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import wandb

from ..data.dataset import H5EmbeddingDataset, H5TrajectoryDataset, MultiViewMP4VideoDataset, OpenXMP4VideoDataset
from ..models.model import DiT
from ..models.base_autoencoder import create_autoencoder, encoder_config_from_args
from ..models.adapters import create_adapter, IdentityAdapter
from .diffusion import Diffusion, FlowMatching
from .validation import validate_step
from .utils import (
    init_distributed,
    requires_grad,
    maybe_compile,
    update_ema,
    init_wandb,
    log_model_param_counts,
    log_training_config,
    load_training_checkpoint,
    save_training_checkpoint,
    resolve_adapter_ckpt,
    load_frozen_adapter_weights,
    setup_pixel_decoder_for_val,
    downsample_actions_temporal,
    downsample_sequence_temporal,
    mask_future_conditioning,
)

# Head architecture constants (kept here rather than as magic numbers inline)
_DECODER_DIM = 2048
_DECODER_DEPTH = 2
_DECODER_HEADS = 16


def train_wm(args) -> None:
    assert torch.cuda.is_available(), "CUDA device required for training"

    precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    if args.encoder_type == "vae":
        args.wide_head = False  # VAE latents are already compact

    local_rank, rank, world_size, distributed = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")

    # ── Checkpoint dir & WandB ────────────────────────────────────────────────
    if args.checkpoint_dir is None:
        checkpoint_dir = Path("outputs") / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        checkpoint_dir = Path(args.checkpoint_dir)

    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        obj = "DDPM" if args.objective == "ddpm" else "Flow"
        run_name = (
            f"DiT-{args.dit_size}_{args.encoder_type.upper()}_{obj}_"
            f"B{args.batch_size * world_size}_"
            f"H{args.num_history}F{args.n_frames - args.num_history}_"
            f"{datetime.datetime.now().strftime('%dT%H-%M-%S')}"
        )
        init_wandb(args, checkpoint_dir, run_name)

    # ── Data ─────────────────────────────────────────────────────────────────
    num_views = getattr(args, "num_views", 1)
    if getattr(args, "h5_train_path", None):
        if args.encoder_type == "precomputed":
            train_dataset = H5EmbeddingDataset(args, split="train")
            val_dataset = H5EmbeddingDataset(args, split="test")
        else:
            train_dataset = H5TrajectoryDataset(args, split="train")
            val_dataset = H5TrajectoryDataset(args, split="test")
    elif num_views > 1:
        train_dataset = MultiViewMP4VideoDataset(args, split="train")
        val_dataset = MultiViewMP4VideoDataset(args, split="test")
    else:
        train_dataset = OpenXMP4VideoDataset(args, split="train")
        val_dataset = OpenXMP4VideoDataset(args, split="test")

    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if distributed else None
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=True,
    )

    samples_per_epoch = len(train_dataset)
    grad_accum_steps = max(int(getattr(args, "gradient_accumulation_steps", 1)), 1)
    micro_steps_per_epoch = max(samples_per_epoch // (args.batch_size * world_size), 1)
    steps_per_epoch = max(micro_steps_per_epoch // grad_accum_steps, 1)
    max_train_steps = args.num_epochs * steps_per_epoch

    if rank == 0:
        log_training_config(args, len(train_dataset), steps_per_epoch, max_train_steps, world_size)

    # ── Models ────────────────────────────────────────────────────────────────
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)

    # Temporal downsampling (e.g. Qwen video mode, V-JEPA 2.1)
    # Read before compile to avoid OptimizedModule attribute delegation issues.
    temporal_ds = autoencoder.temporal_downsample_factor
    context_frames = args.num_history + 1
    if args.n_frames <= context_frames:
        raise ValueError(
            f"n_frames={args.n_frames} must be greater than "
            f"num_history + 1={context_frames} to predict future frames"
        )
    autoencoder = maybe_compile(autoencoder, "autoencoder", args.compile_models)
    if temporal_ds > 1:
        assert args.n_frames % temporal_ds == 0, (
            f"n_frames={args.n_frames} must be divisible by "
            f"temporal_downsample_factor={temporal_ds}"
        )
        assert context_frames % temporal_ds == 0, (
            f"num_history + 1={context_frames} must be divisible by "
            f"temporal_downsample_factor={temporal_ds}"
        )
        logging.info(
            "Temporal downsampling active: factor=%d, effective_action_dim=%d, "
            "effective_context_frames=%d",
            temporal_ds, args.action_dim * temporal_ds, context_frames // temporal_ds,
        )
    if getattr(args, "use_tactile", False) and getattr(args, "tactile_dim", 0) <= 0:
        raise ValueError("--tactile_dim must be > 0 when --use_tactile True")

    effective_action_dim = args.action_dim * temporal_ds
    effective_tactile_dim = getattr(args, "tactile_dim", 0) * temporal_ds
    effective_context_frames = context_frames // temporal_ds

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
    pixel_decoder = setup_pixel_decoder_for_val(args, adapter_ckpt_data, device)

    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, pixel_decoder, adapter_ckpt_data)
    adapter.eval()
    adapter.requires_grad_(False)

    if pixel_decoder is not None:
        pixel_decoder.eval()
        pixel_decoder.requires_grad_(False)
        pixel_decoder = maybe_compile(pixel_decoder, "pixel_decoder", args.compile_models)

    in_channels = adapter.latent_dim
    is_identity_adapter = isinstance(adapter, IdentityAdapter)

    model = _build_model(
        args,
        in_channels,
        device,
        action_dim=effective_action_dim,
        tactile_dim=effective_tactile_dim,
    )
    # Transfer learning: --pretrained_checkpoint is an alias for single-view→multi-view
    pretrained_ckpt = getattr(args, "pretrained_checkpoint", None)
    if pretrained_ckpt and not args.dit_pretrained_backbone_path:
        args.dit_pretrained_backbone_path = pretrained_ckpt
    _load_pretrained_backbone(model, args, device)

    if rank == 0:
        log_model_param_counts(autoencoder, dit_model=model, adapter=adapter, pixel_decoder=pixel_decoder)

    # ── Diffusion ─────────────────────────────────────────────────────────────
    shift = 1.0
    if args.encoder_type != "vae" and args.use_shift:
        m = (256 / args.patch_size ** 2) * in_channels
        shift = (m / 4096) ** 0.5
        logging.info("Time shift: m=%.0f  shift=%.4f", m, shift)

    if args.objective == "ddpm":
        diffusion = Diffusion(
            timesteps=args.timesteps, sampling_timesteps=args.sampling_timesteps,
            time_dist_shift=shift, device=device,
        ).to(device)
    else:
        diffusion = FlowMatching(
            timesteps=args.timesteps, sampling_timesteps=args.sampling_timesteps,
            time_dist_type=args.time_dist_type, logit_mu=args.logit_mu,
            logit_sigma=args.logit_sigma, time_dist_shift=shift, device=device,
        ).to(device)

    # ── DDP + EMA ─────────────────────────────────────────────────────────────
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        model = maybe_compile(model, "model", args.compile_models)
        model_no_ddp = model.module
    else:
        model = maybe_compile(model, "model", args.compile_models)
        model_no_ddp = model

    ema = deepcopy(model_no_ddp).to(device)
    requires_grad(ema, False)
    update_ema(ema, model_no_ddp, args.ema_decay)

    # ── Optimizer + LR schedule ───────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.99), eps=1e-8,
    )
    warmup_steps = args.warmup_epochs * steps_per_epoch
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(max_train_steps - warmup_steps, 1), eta_min=args.lr * args.min_lr_ratio),
        ],
        milestones=[warmup_steps],
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    train_steps, resumed_samples = load_training_checkpoint(
        checkpoint_dir, model_no_ddp, ema, optimizer, lr_scheduler, device
    )
    current_epoch = (train_steps * grad_accum_steps) // micro_steps_per_epoch
    total_samples_seen = (
        resumed_samples if resumed_samples is not None
        else train_steps * args.batch_size * world_size * grad_accum_steps
    )

    if train_sampler is not None:
        train_sampler.set_epoch(current_epoch)
    train_iter = iter(train_loader)
    if train_steps > 0:
        batches_to_skip = (train_steps * grad_accum_steps) % micro_steps_per_epoch
        for _ in range(batches_to_skip):
            try:
                next(train_iter)
            except StopIteration:
                break

    val_iter = iter(val_loader)
    last_log_samples = (total_samples_seen // args.log_every_samples) * args.log_every_samples
    last_val_samples = (total_samples_seen // args.validate_every_samples) * args.validate_every_samples

    # Backbone freeze for transfer learning (unfreeze after N epochs)
    freeze_backbone_epochs = getattr(args, "freeze_backbone_epochs", 0)
    freeze_backbone_steps = freeze_backbone_epochs * steps_per_epoch
    backbone_frozen = False
    if freeze_backbone_epochs > 0 and train_steps < freeze_backbone_steps:
        requires_grad(model_no_ddp, False)
        backbone_frozen = True
        if rank == 0:
            logging.info("Freezing backbone for %d epochs (%d steps)", freeze_backbone_epochs, freeze_backbone_steps)

    running_loss = torch.tensor(0.0)
    num_batches = 0
    pbar = tqdm(total=max_train_steps, desc="Training") if rank == 0 else None
    if pbar is not None:
        pbar.n = train_steps
        pbar.refresh()
    optimizer.zero_grad(set_to_none=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    while train_steps < max_train_steps:
        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                current_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(current_epoch)
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x, actions, tactile = _unpack_batch(batch)

            # Unfreeze backbone after freeze period
            if backbone_frozen and train_steps >= freeze_backbone_steps:
                requires_grad(model_no_ddp, True)
                backbone_frozen = False
                if rank == 0:
                    logging.info("Unfreezing backbone at step %d", train_steps)

            x, actions = x.to(device), actions.to(device)
            if tactile is not None:
                tactile = tactile.to(device)
            with torch.autocast(device_type="cuda", dtype=precision):
                if num_views > 1:
                    # x: (B, V, T, H, W, C) -> encode each view independently
                    B_mv, V, T_mv = x.shape[:3]
                    x = einops.rearrange(x, "b v t h w c -> (b v) t h w c")
                    x = autoencoder.encode(x)
                    if not is_identity_adapter:
                        x = adapter.encode(x)
                        if isinstance(x, tuple):
                            x = x[0]
                    x = einops.rearrange(x, "(b v) t h w c -> b t h (v w) c", v=V)
                else:
                    x = autoencoder.encode(x)
                    if not is_identity_adapter:
                        x = adapter.encode(x)
                        if isinstance(x, tuple):
                            x = x[0]
                if temporal_ds > 1:
                    actions = downsample_actions_temporal(actions, temporal_ds)
                    if tactile is not None:
                        tactile = downsample_sequence_temporal(tactile, temporal_ds)
                tactile = mask_future_conditioning(tactile, effective_context_frames)
                loss = diffusion.loss_fn(
                    model,
                    x,
                    actions,
                    num_history=effective_context_frames,
                    tactile=tactile,
                )

            (loss / grad_accum_steps).backward()
            running_loss += loss.detach().cpu()
            num_batches += 1
            total_samples_seen += args.batch_size * world_size

        max_norm = args.max_grad_norm if args.max_grad_norm > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        lr_scheduler.step()
        update_ema(ema, model_no_ddp, args.ema_decay)
        optimizer.zero_grad(set_to_none=True)
        train_steps += 1

        if pbar is not None:
            pbar.update(1)

        # Logging
        if total_samples_seen - last_log_samples >= args.log_every_samples:
            avg_loss = running_loss / num_batches
            if distributed:
                avg_loss = avg_loss.to(device)
                dist.all_reduce(avg_loss)
                avg_loss = (avg_loss / world_size).detach().cpu()
            if rank == 0:
                pbar.set_postfix({"loss": avg_loss.item(), "samples": total_samples_seen})
                wandb.log(
                    {
                        "train/loss": float(avg_loss),
                        "train/grad_norm": float(grad_norm),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=total_samples_seen,
                )
            running_loss.zero_()
            num_batches = 0
            last_log_samples = total_samples_seen

        # Validation + checkpoint
        if total_samples_seen - last_val_samples >= args.validate_every_samples and rank == 0:
            val_iter = validate_step(
                model=model, ema=ema, autoencoder=autoencoder, adapter=adapter,
                diffusion=diffusion, val_iter=val_iter, val_loader=val_loader,
                device=device, precision=precision, args=args,
                total_samples_seen=total_samples_seen, checkpoint_dir=checkpoint_dir,
                pixel_decoder=pixel_decoder,
            )
            if args.save_model:
                save_training_checkpoint(
                    checkpoint_dir, model_no_ddp, ema, optimizer, lr_scheduler,
                    train_steps, total_samples_seen,
                )
            model.train()
            last_val_samples = total_samples_seen

    if distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_model(
    args,
    in_channels: int,
    device,
    action_dim: int | None = None,
    tactile_dim: int | None = None,
) -> torch.nn.Module:
    act_dim = action_dim if action_dim is not None else args.action_dim
    tact_dim = tactile_dim if tactile_dim is not None else getattr(args, "tactile_dim", 0)
    mv = getattr(args, "num_views", 1)
    if args.model_type == "dit":
        return DiT(
            in_channels=in_channels,
            patch_size=args.patch_size,
            dim=args.model_dim,
            num_layers=args.layers,
            num_heads=args.heads,
            action_dim=act_dim,
            max_frames=args.n_frames,
            action_dropout_prob=args.action_dropout_prob,
            wide_head=args.wide_head,
            decoder_dim=_DECODER_DIM,
            decoder_depth=_DECODER_DEPTH,
            decoder_heads=_DECODER_HEADS,
            num_views=mv,
            temporal_mode=getattr(args, "temporal_mode", "factored"),
            tactile_dim=tact_dim,
            tactile_dropout_prob=getattr(args, "tactile_dropout_prob", 0.0),
        ).to(device)
    raise ValueError(f"Unknown model type: {args.model_type}")


def _unpack_batch(batch):
    if len(batch) == 2:
        x, actions = batch
        return x, actions, None
    if len(batch) == 3:
        return batch
    raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")


def _load_pretrained_backbone(model, args, device) -> None:
    if not args.dit_pretrained_backbone_path:
        return
    logging.info("Loading pretrained backbone from %s", args.dit_pretrained_backbone_path)
    pretrained = torch.load(args.dit_pretrained_backbone_path, map_location=device)
    # Unwrap common checkpoint formats
    for key in ("model", "state_dict", "ema"):
        if isinstance(pretrained, dict) and key in pretrained:
            pretrained = pretrained[key]
            break

    model_state = model.state_dict()
    compatible = {
        k: v for k, v in pretrained.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    skipped = [k for k in pretrained if k not in compatible]
    if skipped:
        prefixes = sorted({k.split(".")[0] for k in skipped})
        logging.info("Skipped %d keys from pretrained (prefixes: %s)", len(skipped), prefixes)

    result = model.load_state_dict(compatible, strict=False)
    if result.missing_keys:
        notable = [k for k in result.missing_keys if "temporal" not in k and "time_embed" not in k]
        if notable:
            logging.info("Notable missing keys (randomly initialized): %s", notable[:5])
    logging.info("Loaded %d compatible keys from pretrained backbone", len(compatible))
