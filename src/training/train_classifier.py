"""Training loop for the spill classifier on GT observations.

Frozen adapter encodes raw patch embeddings → optional tactile fusion → per-patch MLP → BCE loss. 
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from ..data.dataset import H5EmbeddingDataset
from ..models.adapters import IdentityAdapter, create_adapter, adapter_config_from_args
from ..models.base_autoencoder import create_autoencoder, encoder_config_from_args
from ..models.classifier import SpillClassifier, fuse_tactile
from .utils import load_frozen_adapter_weights, resolve_adapter_ckpt, setup_pixel_decoder_for_val

logger = logging.getLogger(__name__)


def _unpack_batch(batch, use_tactile: bool):
    """Returns (emb, tactile_or_None, labels)."""
    if use_tactile:
        emb, _actions, tactile, labels = batch
    else:
        emb, _actions, labels = batch
        tactile = None
    return emb, tactile, labels


def _encode(autoencoder, adapter, emb, device, precision):
    is_identity = isinstance(adapter, IdentityAdapter)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=precision):
            latent = autoencoder.encode(emb)
            if not is_identity:
                z = adapter.encode(latent)
                if isinstance(z, tuple):
                    z = z[0]
            else:
                z = latent
    return z


def _prepare_input(z: torch.Tensor, tactile=None) -> torch.Tensor:
    """z: (B,T,H,W,D_latent) adapter output. Optionally fuse tactile CLS."""
    if tactile is not None:
        return fuse_tactile(z, tactile)   # (B,T,N, D_latent+D_tact)
    B, T, H, W, D = z.shape
    return z.reshape(B, T, H * W, D)


def _eval_epoch(classifier, autoencoder, adapter, loader, device, precision, n_ctx, use_tactile):
    classifier.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb, tactile, labels = _unpack_batch(batch, use_tactile)
            emb = emb.to(device)
            if tactile is not None:
                tactile = tactile.to(device)
            z = _encode(autoencoder, adapter, emb, device, precision)
            with torch.autocast(device_type="cuda", dtype=precision):
                x = _prepare_input(z, tactile)
                logits = classifier(x)
            all_probs.append(torch.sigmoid(logits[:, n_ctx:]).float().cpu().reshape(-1))
            all_labels.append(labels[:, n_ctx:].float().reshape(-1))

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    loss = F.binary_cross_entropy_with_logits(
        torch.logit(all_probs.clamp(1e-6, 1 - 1e-6)), all_labels
    ).item()
    try:
        auc = roc_auc_score(all_labels.numpy(), all_probs.numpy())
    except ValueError:
        auc = float("nan")

    return {"loss": loss, "auc": auc}


def train_classifier(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    precision = torch.bfloat16 if getattr(args, "precision", "bfloat16") == "bfloat16" else torch.float16

    use_tactile = bool(getattr(args, "use_tactile", False))
    args.return_labels = True

    train_dataset = H5EmbeddingDataset(args, split="train")
    val_dataset   = H5EmbeddingDataset(args, split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── Frozen encoder + adapter ──────────────────────────────────────────
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
    pixel_decoder = setup_pixel_decoder_for_val(args, adapter_ckpt_data, device)
    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, pixel_decoder, adapter_ckpt_data)
    adapter.eval()
    adapter.requires_grad_(False)

    n_ctx = (args.num_history + 1) // max(getattr(autoencoder, "temporal_downsample_factor", 1), 1)

    # input_dim: adapter latent + optional tactile CLS
    input_dim = adapter.latent_dim
    if use_tactile:
        input_dim += args.tactile_dim

    # ── Trainable classifier head ─────────────────────────────────────────
    classifier = SpillClassifier(
        input_dim=input_dim, hidden_dim=args.classifier_hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay,
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
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            emb, tactile, labels = _unpack_batch(batch, use_tactile)
            emb    = emb.to(device)
            if tactile is not None:
                tactile = tactile.to(device)
            labels = labels.to(device).float()

            z = _encode(autoencoder, adapter, emb, device, precision)

            with torch.autocast(device_type="cuda", dtype=precision):
                x = _prepare_input(z, tactile)
                logits = classifier(x)
                loss = F.binary_cross_entropy_with_logits(
                    logits[:, n_ctx:], labels[:, n_ctx:]
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss  += loss.item()
            n_batches     += 1
            total_samples += emb.shape[0]

        train_loss  = running_loss / max(n_batches, 1)
        val_metrics = _eval_epoch(
            classifier, autoencoder, adapter, val_loader, device, precision, n_ctx, use_tactile
        )

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_auc=%.3f",
            epoch, train_loss, val_metrics["loss"], val_metrics["auc"],
        )
        wandb.log({
            "train/loss": train_loss,
            "val/loss":   val_metrics["loss"],
            "val/auc":    val_metrics["auc"],
            "epoch":      epoch,
        }, step=total_samples)

        ckpt = {
            "epoch":      epoch,
            "head":       classifier.head.state_dict(),
            "input_dim":  input_dim,
            "hidden_dim": args.classifier_hidden_dim,
            "optimizer":  optimizer.state_dict(),
            "val_auc":    val_metrics["auc"],
        }
        torch.save(ckpt, checkpoint_dir / "classifier_last.pt")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(ckpt, checkpoint_dir / "classifier_best.pt")
            logger.info("  → New best AUC: %.4f", best_auc)

    wandb.finish()
