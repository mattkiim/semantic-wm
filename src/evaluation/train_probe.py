"""Training and evaluation loops for trajectory success probes."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from .probe import create_probe, extract_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_success_probe(args) -> Dict:
    """Train a success classifier probe on frozen encoder features.

    Parameters
    ----------
    args : namespace with probe/model/data configuration

    Returns
    -------
    dict with training metrics and probe checkpoint path
    """
    from ..data.probe_dataset import TrajectoryProbeDataset
    from ..models.base_autoencoder import create_autoencoder, encoder_config_from_args
    from ..models.adapters import create_adapter, IdentityAdapter
    from ..training.utils import resolve_adapter_ckpt, load_frozen_adapter_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = torch.bfloat16 if getattr(args, "precision", "bfloat16") == "bfloat16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Wandb ──────────────────────────────────────────────────────────
    wandb_mode = getattr(args, "wandb_mode", "disabled")
    wandb.init(
        project=getattr(args, "wandb_project_name", "world-model-rae"),
        entity=getattr(args, "wandb_entity", "sarath-chandar"),
        name=f"probe-{getattr(args, 'probe_type', 'temporal')}_{args.encoder_type}",
        mode=wandb_mode,
        config={k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()},
        dir=str(output_dir),
    )

    # ── Load frozen encoder + adapter ────────────────────────���───────────
    logger.info("Loading autoencoder: %s", args.encoder_type)
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    adapter = None
    feature_space = getattr(args, "feature_space", "adapter")
    if feature_space == "adapter":
        adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
        adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
        if adapter_ckpt_data is not None:
            load_frozen_adapter_weights(adapter, None, adapter_ckpt_data)
        adapter.eval()
        adapter.requires_grad_(False)
        feature_dim = adapter.latent_dim
        is_identity = isinstance(adapter, IdentityAdapter)
        if is_identity:
            feature_dim = autoencoder.latent_dim
            adapter = None
            feature_space = "encoder"
    else:
        feature_dim = autoencoder.latent_dim

    logger.info("Feature space: %s, feature_dim: %d", feature_space, feature_dim)

    # ── Datasets ────────────────────────────────────────────────────���────
    n_sample_frames = getattr(args, "n_sample_frames", 8)
    sampling_strategy = getattr(args, "sampling_strategy", "uniform")
    pool_mode = getattr(args, "pool_mode", "mean")

    train_dataset = TrajectoryProbeDataset(
        dataset_dir=args.dataset_dir,
        subset_names=args.subset_names,
        split="train",
        n_sample_frames=n_sample_frames,
        sampling_strategy=sampling_strategy,
        input_h=getattr(args, "input_h", 256),
        input_w=getattr(args, "input_w", 256),
    )
    test_dataset = TrajectoryProbeDataset(
        dataset_dir=args.dataset_dir,
        subset_names=args.subset_names,
        split="test",
        n_sample_frames=n_sample_frames,
        sampling_strategy=sampling_strategy,
        input_h=getattr(args, "input_h", 256),
        input_w=getattr(args, "input_w", 256),
    )

    batch_size = getattr(args, "probe_batch_size", 16)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=getattr(args, "num_workers", 4), pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=getattr(args, "num_workers", 4), pin_memory=True,
    )

    # ── Create probe ────────────────────────────────────────────────────���
    probe_type = getattr(args, "probe_type", "temporal")
    probe = create_probe(
        probe_type=probe_type,
        feature_dim=feature_dim,
        n_frames=n_sample_frames,
        pool_mode=pool_mode,
    ).to(device)

    logger.info("Probe: %s (params: %d)", probe_type,
                sum(p.numel() for p in probe.parameters()))

    # ── Training loop ───────────────────────────────────��────────────────
    lr = getattr(args, "probe_lr", 1e-3)
    epochs = getattr(args, "probe_epochs", 50)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for frames, actions, labels, lengths in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False
        ):
            frames = frames.to(device)
            labels = labels.float().to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                features = extract_features(
                    frames, autoencoder, adapter, feature_space=feature_space
                )
                logits = probe(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = (logits.detach() > 0).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = correct / total
        train_loss = total_loss / total

        # ── Evaluate on test set ─────────────────────────────────���───────
        test_metrics = _evaluate_probe(
            probe, autoencoder, adapter, feature_space, test_loader,
            device, precision, criterion,
        )

        wandb.log({
            "probe/train_loss": train_loss,
            "probe/train_acc": train_acc,
            "probe/test_loss": test_metrics["loss"],
            "probe/test_acc": test_metrics["accuracy"],
            "probe/test_auc": test_metrics["auc"],
            "probe/lr": scheduler.get_last_lr()[0],
            "probe/epoch": epoch + 1,
        })
        logger.info(
            "Epoch %d: train_loss=%.4f train_acc=%.4f | test_loss=%.4f test_acc=%.4f test_auc=%.4f",
            epoch + 1, train_loss, train_acc,
            test_metrics["loss"], test_metrics["accuracy"], test_metrics["auc"],
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            best_epoch = epoch + 1
            ckpt_path = output_dir / "probe_best.pt"
            torch.save({
                "probe": probe.state_dict(),
                "probe_type": probe_type,
                "feature_dim": feature_dim,
                "n_sample_frames": n_sample_frames,
                "pool_mode": pool_mode,
                "feature_space": feature_space,
                "encoder_type": args.encoder_type,
                "epoch": epoch + 1,
                "test_accuracy": best_acc,
            }, ckpt_path)

    # ── Also train progress regressor if requested ───────────────────────
    progress_results = {}
    if getattr(args, "train_progress_regressor", False):
        progress_results = _train_progress_regressor(
            args, autoencoder, adapter, feature_space, feature_dim,
            train_loader, test_loader, device, precision, output_dir,
        )

    results = {
        "best_test_accuracy": best_acc,
        "best_epoch": best_epoch,
        "probe_checkpoint": str(output_dir / "probe_best.pt"),
        **progress_results,
    }

    wandb.log({
        "probe/best_test_acc": best_acc,
        "probe/best_epoch": best_epoch,
    })

    results_path = output_dir / "probe_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return results


def _evaluate_probe(
    probe: nn.Module,
    autoencoder,
    adapter,
    feature_space: str,
    loader: DataLoader,
    device: torch.device,
    precision: torch.dtype,
    criterion: nn.Module,
) -> Dict:
    """Evaluate probe on a dataloader. Returns accuracy, loss, AUC."""
    probe.eval()
    all_logits = []
    all_labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for frames, actions, labels, lengths in loader:
            frames = frames.to(device)
            labels = labels.float().to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                features = extract_features(
                    frames, autoencoder, adapter, feature_space=feature_space
                )
                logits = probe(features)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    preds = (all_logits > 0).long()
    accuracy = (preds == all_labels.long()).float().mean().item()

    # Compute AUC
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.sigmoid(all_logits).numpy()
        auc = roc_auc_score(all_labels.numpy(), probs)
    except Exception:
        auc = 0.0

    return {
        "accuracy": accuracy,
        "loss": total_loss / total,
        "auc": auc,
    }


# ---------------------------------------------------------------------------
# Progress regressor
# ---------------------------------------------------------------------------


def _train_progress_regressor(
    args,
    autoencoder,
    adapter,
    feature_space: str,
    feature_dim: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    precision: torch.dtype,
    output_dir: Path,
) -> Dict:
    """Train a progress-to-goal regressor (normalized timestep t/T)."""
    from .probe import create_probe

    n_sample_frames = getattr(args, "n_sample_frames", 8)
    pool_mode = getattr(args, "pool_mode", "mean")

    regressor = create_probe(
        probe_type="progress",
        feature_dim=feature_dim,
        n_frames=n_sample_frames,
        pool_mode=pool_mode,
    ).to(device)

    optimizer = torch.optim.AdamW(regressor.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = getattr(args, "probe_epochs", 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_mse = float("inf")

    for epoch in range(epochs):
        regressor.train()
        total_loss = 0.0
        total = 0

        for frames, actions, labels, lengths in tqdm(
            train_loader, desc=f"Progress epoch {epoch+1}/{epochs}", leave=False
        ):
            frames = frames.to(device)
            B, T = frames.shape[:2]
            # Target: normalized timestep t/T for each sampled frame
            targets = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                features = extract_features(
                    frames, autoencoder, adapter, feature_space=feature_space
                )
                preds = regressor(features)
                loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total += B

        scheduler.step()

        # Evaluate
        regressor.eval()
        test_loss = 0.0
        test_total = 0
        with torch.no_grad():
            for frames, actions, labels, lengths in test_loader:
                frames = frames.to(device)
                B, T = frames.shape[:2]
                targets = torch.linspace(0, 1, T).unsqueeze(0).expand(B, -1).to(device)

                with torch.autocast(device_type="cuda", dtype=precision):
                    features = extract_features(
                        frames, autoencoder, adapter, feature_space=feature_space
                    )
                    preds = regressor(features)
                    loss = criterion(preds, targets)

                test_loss += loss.item() * B
                test_total += B

        test_mse = test_loss / test_total
        if test_mse < best_mse:
            best_mse = test_mse
            torch.save({
                "probe": regressor.state_dict(),
                "probe_type": "progress",
                "feature_dim": feature_dim,
                "epoch": epoch + 1,
            }, output_dir / "progress_regressor_best.pt")

        wandb.log({
            "progress/train_mse": total_loss / total,
            "progress/test_mse": test_mse,
            "progress/epoch": epoch + 1,
        })
        logger.info("Progress epoch %d: train_mse=%.6f test_mse=%.6f",
                     epoch + 1, total_loss / total, test_mse)

    return {"progress_best_mse": best_mse}


# ---------------------------------------------------------------------------
# WM-generated trajectory evaluation
# ---------------------------------------------------------------------------


def evaluate_probe_on_generated(
    args,
    probe: nn.Module,
    autoencoder,
    adapter,
    feature_space: str,
    diffusion,
    dit_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    precision: torch.dtype,
) -> Dict:
    """Evaluate a trained probe on WM-generated trajectories.

    For each test episode:
    1. Encode initial context frame(s) from real data
    2. Generate future frames via diffusion conditioned on real actions
    3. Resample generated frames to match probe's expected temporal sampling
    4. Run probe on generated features
    5. Compare with ground-truth success labels

    Returns accuracy on generated data and the drop vs real accuracy.
    """
    from ..models.adapters import IdentityAdapter

    probe.eval()
    is_identity = adapter is None or isinstance(adapter, IdentityAdapter)

    n_ctx = max(getattr(args, "num_history", 1), 1)
    cfg = getattr(args, "cfg", 1.0)
    n_sample_frames = getattr(args, "n_sample_frames", 8)

    all_logits_real = []
    all_logits_gen = []
    all_labels = []

    with torch.no_grad():
        for frames, actions, labels, lengths in tqdm(test_loader, desc="Eval on generated"):
            frames = frames.to(device)
            actions = actions.to(device)
            labels_cpu = labels.clone()
            B, T = frames.shape[:2]

            with torch.autocast(device_type="cuda", dtype=precision):
                # Real features for comparison
                features_real = extract_features(
                    frames, autoencoder, adapter, feature_space=feature_space
                )
                logits_real = probe(features_real)

                # Encode all frames to get latents for context
                z = autoencoder.encode(frames)
                if not is_identity and adapter is not None:
                    z_adapted = adapter.encode(z)
                    if isinstance(z_adapted, tuple):
                        z_adapted = z_adapted[0]
                else:
                    z_adapted = z

                # Generate via diffusion: condition on first n_ctx frames
                gen_latent = diffusion.generate(
                    dit_model,
                    z_adapted,
                    actions,
                    n_context_frames=min(n_ctx, T - 1),
                    n_frames=z_adapted.shape[1],
                    cfg=cfg,
                )

                # Run probe on generated latents
                logits_gen = probe(gen_latent)

            all_logits_real.append(logits_real.cpu())
            all_logits_gen.append(logits_gen.cpu())
            all_labels.append(labels_cpu)

    all_logits_real = torch.cat(all_logits_real)
    all_logits_gen = torch.cat(all_logits_gen)
    all_labels = torch.cat(all_labels).float()

    # Compute metrics
    acc_real = ((all_logits_real > 0).long() == all_labels.long()).float().mean().item()
    acc_gen = ((all_logits_gen > 0).long() == all_labels.long()).float().mean().item()
    acc_drop = acc_real - acc_gen

    try:
        from sklearn.metrics import roc_auc_score
        auc_real = roc_auc_score(all_labels.numpy(), torch.sigmoid(all_logits_real).numpy())
        auc_gen = roc_auc_score(all_labels.numpy(), torch.sigmoid(all_logits_gen).numpy())
    except Exception:
        auc_real = auc_gen = 0.0

    results = {
        "probe_accuracy_real": acc_real,
        "probe_accuracy_generated": acc_gen,
        "probe_accuracy_drop": acc_drop,
        "probe_auc_real": auc_real,
        "probe_auc_generated": auc_gen,
    }

    wandb.log({
        "eval/probe_accuracy_real": acc_real,
        "eval/probe_accuracy_generated": acc_gen,
        "eval/probe_accuracy_drop": acc_drop,
        "eval/probe_auc_real": auc_real,
        "eval/probe_auc_generated": auc_gen,
    })
    logger.info("Probe results: acc_real=%.4f acc_gen=%.4f drop=%.4f",
                acc_real, acc_gen, acc_drop)
    return results
