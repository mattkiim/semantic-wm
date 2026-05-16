"""Evaluate the spill classifier.

Two modes:
  1. GT mode (default): classify ground-truth embeddings for all val windows,
     predicting labels for frames num_history+1 … num_history+8.

  2. WM mode (--wm_checkpoint_path): given N history frames, roll out the world
     model for 8 steps, then classify the generated latents.

Usage::

    # GT eval
    python -m src.eval_classifier \\
        --classifier_checkpoint_path outputs/classifier/classifier_best.pt \\
        --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel_bs4_ga4/adapter_ckpt_000000117936.pt \\
        --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \\
        --encoder_type precomputed

    # WM eval
    python -m src.eval_classifier \\
        --classifier_checkpoint_path outputs/classifier/classifier_best.pt \\
        --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel_bs4_ga4/adapter_ckpt_000000117936.pt \\
        --wm_checkpoint_path outputs/dit_dinov3_precomputed_tactile_v1_do_0.2_cls_new_adapter_long/ckpt_samples_000000771232.pt \\
        --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \\
        --encoder_type precomputed --use_tactile True --tactile_dim 512
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import H5EmbeddingDataset
from src.models.adapters import IdentityAdapter, create_adapter, adapter_config_from_args
from src.models.base_autoencoder import create_autoencoder, encoder_config_from_args
from src.models.classifier import SpillClassifier
from src.models.model import DiT
from src.training.diffusion import Diffusion, FlowMatching
from src.training.utils import (
    resolve_adapter_ckpt,
    setup_pixel_decoder_for_val,
    load_frozen_adapter_weights,
    strip_state_dict_prefix,
    downsample_actions_temporal,
    downsample_sequence_temporal,
)

_DECODER_DIM = 2048
_DECODER_DEPTH = 2
_DECODER_HEADS = 16

logger = logging.getLogger(__name__)

PREDICT_STEPS = 8  # how many future frames to evaluate


def _unpack_batch(batch):
    # last element is always labels when return_labels=True
    *front, labels = batch
    emb = front[0]
    actions = front[1]
    tactile = front[2] if len(front) > 2 else None
    return emb, actions, tactile, labels


def _compute_metrics(all_logits, all_labels):
    probs = torch.sigmoid(all_logits).numpy()
    preds = (probs > 0.5).astype(int)
    labels_np = all_labels.numpy().astype(int)
    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "f1": f1, "auc": auc}


def evaluate_classifier(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.return_labels = True
    val_dataset = H5EmbeddingDataset(args, split="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()

    temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)
    effective_action_dim = args.action_dim * temporal_ds
    effective_tactile_dim = getattr(args, "tactile_dim", 0) * temporal_ds

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
    pixel_decoder = setup_pixel_decoder_for_val(args, adapter_ckpt_data, device)
    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, pixel_decoder, adapter_ckpt_data)
    adapter.eval()
    adapter.requires_grad_(False)

    classifier = SpillClassifier(
        adapter, latent_dim=adapter.latent_dim, hidden_dim=args.classifier_hidden_dim
    ).to(device)
    ckpt = torch.load(args.classifier_checkpoint_path, map_location=device)
    classifier.head.load_state_dict(ckpt["head"])
    classifier.eval()
    logger.info("Loaded classifier from %s (epoch %d, val_auc=%.4f)",
                args.classifier_checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_auc", float("nan")))

    use_wm = bool(getattr(args, "wm_checkpoint_path", None))
    wm = ema = diffusion = None
    if use_wm:
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
        wm.eval()
        ema.eval()

        shift = 1.0
        if args.encoder_type != "vae":
            m = (256 / args.patch_size ** 2) * adapter.latent_dim
            shift = (m / 4096) ** 0.5
        diffusion = FlowMatching(
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            time_dist_type="uniform",
            time_dist_shift=shift,
            device=device,
        ).to(device) if args.objective == "flow_matching" else Diffusion(
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            time_dist_shift=shift,
            device=device,
        ).to(device)

    n_ctx = (args.num_history + 1) // max(temporal_ds, 1)
    num_history = args.num_history + 1  # context frames to skip when evaluating

    all_logits_gt, all_logits_wm, all_labels_flat = [], [], []

    logger.info("Running classifier eval (wm_mode=%s)...", use_wm)
    with torch.no_grad():
        for batch in val_loader:
            emb, actions, tactile, labels = _unpack_batch(batch)
            emb = emb.to(device)
            actions = actions.to(device)
            if tactile is not None:
                tactile = tactile.to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                val_latent = autoencoder.encode(emb)
                is_identity = isinstance(adapter, IdentityAdapter)
                if not is_identity:
                    val_latent_adapted = adapter.encode(val_latent)
                    if isinstance(val_latent_adapted, tuple):
                        val_latent_adapted = val_latent_adapted[0]
                else:
                    val_latent_adapted = val_latent

                if temporal_ds > 1:
                    actions = downsample_actions_temporal(actions, temporal_ds)
                    if tactile is not None:
                        tactile = downsample_sequence_temporal(tactile, temporal_ds)

                # GT classification
                gt_logits = classifier.forward_from_latent(val_latent_adapted)  # (B, T)
                gt_future = gt_logits[:, num_history:num_history + PREDICT_STEPS]
                labels_future = labels[:, num_history:num_history + PREDICT_STEPS].float()

                all_logits_gt.append(gt_future.float().cpu().reshape(-1))
                all_labels_flat.append(labels_future.cpu().reshape(-1))

                # WM classification
                if use_wm:
                    n_total = min(n_ctx + PREDICT_STEPS, val_latent_adapted.shape[1])
                    wm_latent = diffusion.generate(
                        ema,
                        val_latent_adapted,
                        actions,
                        n_context_frames=n_ctx,
                        n_frames=n_total,
                        window_len=args.window_len,
                        horizon=args.horizon,
                        cfg=args.cfg,
                        tactile=tactile,
                    )
                    wm_logits = classifier.forward_from_latent(wm_latent)  # (B, n_total)
                    wm_future = wm_logits[:, n_ctx:n_ctx + PREDICT_STEPS]
                    all_logits_wm.append(wm_future.float().cpu().reshape(-1))

    all_logits_gt = torch.cat(all_logits_gt)
    all_labels = torch.cat(all_labels_flat)
    gt_metrics = _compute_metrics(all_logits_gt, all_labels)

    results = {"mode": "gt", **{f"gt/{k}": v for k, v in gt_metrics.items()}}

    print("\n=== Classifier Evaluation (GT frames) ===")
    print(f"  Accuracy : {gt_metrics['accuracy']:.4f}")
    print(f"  F1       : {gt_metrics['f1']:.4f}")
    print(f"  AUC      : {gt_metrics['auc']:.4f}")

    if use_wm and all_logits_wm:
        all_logits_wm = torch.cat(all_logits_wm)
        wm_metrics = _compute_metrics(all_logits_wm, all_labels)
        results.update({f"wm/{k}": v for k, v in wm_metrics.items()})
        print("\n=== Classifier Evaluation (WM-generated frames) ===")
        print(f"  Accuracy : {wm_metrics['accuracy']:.4f}")
        print(f"  F1       : {wm_metrics['f1']:.4f}")
        print(f"  AUC      : {wm_metrics['auc']:.4f}")

    out_path = output_dir / "classifier_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved to %s", out_path)
    return results


def get_configs():
    DIT_SIZES = {
        "S": {"model_dim": 384, "layers": 12, "heads": 6},
        "B": {"model_dim": 768, "layers": 12, "heads": 12},
        "L": {"model_dim": 1024, "layers": 24, "heads": 16},
        "XL": {"model_dim": 1152, "layers": 28, "heads": 16},
    }

    parser = argparse.ArgumentParser(description="Evaluate spill classifier")

    # ── Classifier ───────────────────────────────────────────────────────
    parser.add_argument("--classifier_checkpoint_path", type=str, required=True)
    parser.add_argument("--classifier_hidden_dim", type=int, default=256)

    # ── Adapter (frozen) ─────────────────────────────────────────────────
    parser.add_argument("--adapter_type", type=str, default="svae",
                        choices=["identity", "mlp", "svae"])
    parser.add_argument("--adapter_checkpoint_path", type=str, required=True)
    parser.add_argument("--adapter_latent_dim", type=int, default=96)
    parser.add_argument("--adapter_num_heads", type=int, default=16)
    parser.add_argument("--adapter_num_layers", type=int, default=3)
    parser.add_argument("--adapter_intermediate_size", type=int, default=2048)
    parser.add_argument("--use_pixel_decoder_for_val",
                        type=lambda x: x.lower() == "true", default=False)

    # ── WM (optional) ────────────────────────────────────────────────────
    parser.add_argument("--wm_checkpoint_path", type=str, default=None)
    parser.add_argument("--dit_size", type=str, default="S", choices=list(DIT_SIZES.keys()))
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--wide_head", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--objective", type=str, default="flow_matching",
                        choices=["ddpm", "flow_matching"])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)

    # ── Encoder ──────────────────────────────────────────────────────────
    parser.add_argument("--encoder_type", type=str, default="precomputed",
                        choices=["vae", "rae", "precomputed", "scale_rae_siglip",
                                 "scale_rae_webssl", "qwen", "vjepa2", "cosmos", "vavae"])
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--patch_h", type=int, default=14)
    parser.add_argument("--patch_w", type=int, default=14)
    parser.add_argument("--h5_embedding_key", type=str, default="cam_0_patch_embd")

    # ── Tactile ──────────────────────────────────────────────────────────
    parser.add_argument("--use_tactile", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--tactile_dim", type=int, default=0)
    parser.add_argument("--h5_tactile_key", type=str, default="cam_tactile_cls_embd")

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument("--h5_val_path", type=str, required=True)
    parser.add_argument("--h5_train_path", type=str, default=None)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--variable_history_sampling",
                        type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bfloat16")

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="eval_outputs/classifier")

    args = parser.parse_args()

    size_cfg = DIT_SIZES[args.dit_size]
    if args.model_dim is None:
        args.model_dim = size_cfg["model_dim"]
    if args.layers is None:
        args.layers = size_cfg["layers"]
    if args.heads is None:
        args.heads = size_cfg["heads"]

    return args


if __name__ == "__main__":
    args = get_configs()
    evaluate_classifier(args)
