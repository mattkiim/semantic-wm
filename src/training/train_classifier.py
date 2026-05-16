"""Training loop for the spill classifier.

The world model (frozen) rolls out future frames from GT context. The
classifier head is trained to detect spills from those WM-generated latents,
supervised by GT labels at the corresponding future timesteps.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.dataset import H5EmbeddingDataset
from ..models.adapters import IdentityAdapter, create_adapter, adapter_config_from_args
from ..models.base_autoencoder import create_autoencoder, encoder_config_from_args
from ..models.classifier import SpillClassifier
from ..models.model import DiT
from .diffusion import Diffusion, FlowMatching
from .utils import (
    load_frozen_adapter_weights,
    resolve_adapter_ckpt,
    setup_pixel_decoder_for_val,
    strip_state_dict_prefix,
    downsample_actions_temporal,
    downsample_sequence_temporal,
)

logger = logging.getLogger(__name__)

_DECODER_DIM = 2048
_DECODER_DEPTH = 2
_DECODER_HEADS = 16


def _unpack_batch(batch):
    # last element is always labels; second element is actions; third (optional) is tactile
    *front, labels = batch
    emb = front[0]
    actions = front[1]
    tactile = front[2] if len(front) > 2 else None
    return emb, actions, tactile, labels


def _rollout(autoencoder, adapter, ema, diffusion, emb, actions, tactile,
             n_ctx, temporal_ds, precision, device):
    """Encode GT embeddings, roll out WM, return WM latents (B, T, H, W, C)."""
    is_identity = isinstance(adapter, IdentityAdapter)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=precision):
            latent = autoencoder.encode(emb)
            if not is_identity:
                latent_adapted = adapter.encode(latent)
                if isinstance(latent_adapted, tuple):
                    latent_adapted = latent_adapted[0]
            else:
                latent_adapted = latent

            if temporal_ds > 1:
                actions = downsample_actions_temporal(actions, temporal_ds)
                if tactile is not None:
                    tactile = downsample_sequence_temporal(tactile, temporal_ds)

            wm_latent = diffusion.generate(
                ema,
                latent_adapted,
                actions,
                n_context_frames=n_ctx,
                n_frames=latent_adapted.shape[1],
                tactile=tactile,
            )
    return wm_latent


def _eval_epoch(classifier, autoencoder, adapter, ema, diffusion,
                loader, device, precision, n_ctx, temporal_ds):
    classifier.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb, actions, tactile, labels = _unpack_batch(batch)
            emb = emb.to(device)
            actions = actions.to(device)
            if tactile is not None:
                tactile = tactile.to(device)

            wm_latent = _rollout(
                autoencoder, adapter, ema, diffusion,
                emb, actions, tactile, n_ctx, temporal_ds, precision, device,
            )
            with torch.autocast(device_type="cuda", dtype=precision):
                logits = classifier.forward_from_latent(wm_latent)   # (B, T)

            logits_future = logits[:, n_ctx:].reshape(-1).float().cpu()
            labels_future = labels[:, n_ctx:].reshape(-1).float()
            all_logits.append(logits_future)
            all_labels.append(labels_future)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    loss = F.binary_cross_entropy_with_logits(all_logits, all_labels).item()
    acc = ((all_logits > 0).float() == all_labels).float().mean().item()
    try:
        auc = roc_auc_score(all_labels.numpy(), torch.sigmoid(all_logits).numpy())
    except ValueError:
        auc = float("nan")
    return {"loss": loss, "accuracy": acc, "auc": auc}


def train_classifier(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    precision = torch.bfloat16 if getattr(args, "precision", "bfloat16") == "bfloat16" else torch.float16

    args.return_labels = True

    train_dataset = H5EmbeddingDataset(args, split="train")
    val_dataset = H5EmbeddingDataset(args, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── Frozen backbone ───────────────────────────────────────────────────
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()
    temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
    pixel_decoder = setup_pixel_decoder_for_val(args, adapter_ckpt_data, device)
    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, pixel_decoder, adapter_ckpt_data)
    adapter.eval()
    adapter.requires_grad_(False)

    effective_action_dim = args.action_dim * temporal_ds
    effective_tactile_dim = getattr(args, "tactile_dim", 0) * temporal_ds

    # ── Frozen world model ────────────────────────────────────────────────
    wm = DiT(
        in_channels=adapter.latent_dim,
        patch_size=args.patch_size,
        dim=args.model_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        action_dim=effective_action_dim,
        tactile_dim=effective_tactile_dim,
        max_frames=args.n_frames,
        action_dropout_prob=0.0,
        tactile_dropout_prob=0.0,
        wide_head=args.wide_head,
        decoder_dim=_DECODER_DIM,
        decoder_depth=_DECODER_DEPTH,
        decoder_heads=_DECODER_HEADS,
    ).to(device)
    ema = deepcopy(wm)

    wm_ckpt = torch.load(args.wm_checkpoint_path, map_location=device)
    if "ema" in wm_ckpt:
        ema.load_state_dict(strip_state_dict_prefix(wm_ckpt["ema"]), strict=False)
    if "model" in wm_ckpt:
        wm.load_state_dict(strip_state_dict_prefix(wm_ckpt["model"]), strict=False)
    wm.eval()
    ema.eval()
    wm.requires_grad_(False)
    ema.requires_grad_(False)
    logger.info("Loaded WM from %s", args.wm_checkpoint_path)

    shift = 1.0
    if args.encoder_type != "vae":
        m = (256 / args.patch_size ** 2) * adapter.latent_dim
        shift = (m / 4096) ** 0.5
    diffusion = (
        FlowMatching(
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            time_dist_type="uniform",
            time_dist_shift=shift,
            device=device,
        )
        if args.objective == "flow_matching"
        else Diffusion(
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            time_dist_shift=shift,
            device=device,
        )
    ).to(device)

    n_ctx = (args.num_history + 1) // max(temporal_ds, 1)

    # ── Trainable classifier head ─────────────────────────────────────────
    classifier = SpillClassifier(
        adapter,
        latent_dim=adapter.latent_dim,
        hidden_dim=args.classifier_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        classifier.head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=args.wandb_project_name,
        entity=getattr(args, "wandb_entity", None),
        mode=getattr(args, "wandb_mode", "online"),
        config=vars(args),
    )

    best_auc = 0.0
    total_samples = 0

    for epoch in range(args.num_epochs):
        classifier.train()
        classifier.adapter.eval()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            emb, actions, tactile, labels = _unpack_batch(batch)
            emb = emb.to(device)
            actions = actions.to(device)
            labels = labels.to(device).float()
            if tactile is not None:
                tactile = tactile.to(device)

            # Roll out WM (no grad through WM or adapter)
            wm_latent = _rollout(
                autoencoder, adapter, ema, diffusion,
                emb, actions, tactile, n_ctx, temporal_ds, precision, device,
            )

            # Train classifier head on WM future latents
            with torch.autocast(device_type="cuda", dtype=precision):
                logits = classifier.forward_from_latent(wm_latent)   # (B, T)
                loss = F.binary_cross_entropy_with_logits(
                    logits[:, n_ctx:], labels[:, n_ctx:]
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            total_samples += emb.shape[0]

        train_loss = running_loss / max(n_batches, 1)
        val_metrics = _eval_epoch(
            classifier, autoencoder, adapter, ema, diffusion,
            val_loader, device, precision, n_ctx, temporal_ds,
        )

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | val_auc=%.3f",
            epoch, train_loss, val_metrics["loss"], val_metrics["accuracy"], val_metrics["auc"],
        )
        wandb.log({
            "train/loss": train_loss,
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "val/auc": val_metrics["auc"],
            "epoch": epoch,
        }, step=total_samples)

        ckpt = {
            "epoch": epoch,
            "head": classifier.head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_auc": val_metrics["auc"],
        }
        torch.save(ckpt, checkpoint_dir / "classifier_last.pt")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(ckpt, checkpoint_dir / "classifier_best.pt")
            logger.info("  → New best AUC: %.4f", best_auc)

    wandb.finish()
