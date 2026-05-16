"""Evaluate the spill classifier on WM rollouts.

Spill-only mode (--spill_only True):
  - Evaluates only on windows near 0→1 label transitions where the context
    frames are all label=0 (clean) and at least one future frame is label=1.
  - Generates side-by-side videos: GT (left) | WM predicted (right), with
    green/red borders showing GT label vs classifier prediction per frame.

Usage::

    python -m src.eval_classifier \\
        --classifier_checkpoint_path outputs/classifier_v1/classifier_best.pt \\
        --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel_bs4_ga4/adapter_ckpt_000000117936.pt \\
        --wm_checkpoint_path outputs/dit_dinov3_precomputed_tactile_v1_do_0.2_cls_new_adapter_long/ckpt_samples_000000771232.pt \\
        --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \\
        --encoder_type precomputed \\
        --use_tactile True --tactile_dim 512 \\
        --use_pixel_decoder_for_val True \\
        --spill_only True \\
        --output_dir eval_outputs/classifier_v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import H5EmbeddingDataset
from src.models.adapters import IdentityAdapter, create_adapter, adapter_config_from_args
from src.models.base_autoencoder import create_autoencoder, encoder_config_from_args
from src.models.classifier import SpillClassifier, fuse_tactile
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
_GREEN = np.array([0, 220, 0], dtype=np.uint8)
_RED   = np.array([220, 0, 0], dtype=np.uint8)
_GRAY  = np.array([120, 120, 120], dtype=np.uint8)

logger = logging.getLogger(__name__)


def _unpack_batch(batch):
    *front, labels = batch
    emb, actions = front[0], front[1]
    tactile = front[2] if len(front) > 2 else None
    return emb, actions, tactile, labels


def _add_border(img: np.ndarray, color: np.ndarray, thickness: int = 10) -> np.ndarray:
    """img: (H, W, 3) uint8. Returns copy with colored border."""
    img = img.copy()
    img[:thickness, :] = color
    img[-thickness:, :] = color
    img[:, :thickness] = color
    img[:, -thickness:] = color
    return img


def _make_side_by_side(gt_img, pred_img, gt_label, pred_prob, border=10):
    """Both imgs: (H, W, 3) uint8. Returns (H, 2W, 3) uint8."""
    gt_color   = _GREEN if gt_label == 0 else _RED
    pred_color = _GREEN if pred_prob < 0.5 else _RED
    return np.concatenate([
        _add_border(gt_img,   gt_color,   border),
        _add_border(pred_img, pred_color, border),
    ], axis=1)


def _save_video(frames: list[np.ndarray], path: Path, fps: int = 2) -> None:
    with imageio.get_writer(str(path), fps=fps, format="FFMPEG",
                            codec="libx264", quality=7, macro_block_size=1) as writer:
        for f in frames:
            writer.append_data(f)


def _load_wm(args, adapter, device):
    temporal_ds = 1  # resolved earlier
    effective_action_dim  = args.action_dim * temporal_ds
    effective_tactile_dim = getattr(args, "tactile_dim", 0) * temporal_ds

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
    wm.eval(); ema.eval()
    wm.requires_grad_(False); ema.requires_grad_(False)
    logger.info("Loaded WM from %s", args.wm_checkpoint_path)

    shift = 1.0
    if args.encoder_type != "vae":
        m = (256 / args.patch_size ** 2) * adapter.latent_dim
        shift = (m / 4096) ** 0.5
    diffusion = (
        FlowMatching(timesteps=args.timesteps, sampling_timesteps=args.sampling_timesteps,
                     time_dist_type="uniform", time_dist_shift=shift, device=device)
        if args.objective == "flow_matching"
        else Diffusion(timesteps=args.timesteps, sampling_timesteps=args.sampling_timesteps,
                       time_dist_shift=shift, device=device)
    ).to(device)
    return ema, diffusion


def evaluate_classifier(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.return_labels = True
    spill_only = getattr(args, "spill_only", False)

    dataset = H5EmbeddingDataset(
        args, split="test",
        spill_only=spill_only,
        spill_windows_per_transition=getattr(args, "spill_windows_per_transition", 3),
    )
    logger.info("Dataset size: %d windows (spill_only=%s)", len(dataset), spill_only)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── Models ───────────────────────────────────────────────────────────
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()
    temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
    pixel_decoder = setup_pixel_decoder_for_val(args, adapter_ckpt_data, device)
    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, pixel_decoder, adapter_ckpt_data)
    adapter.eval(); adapter.requires_grad_(False)
    if pixel_decoder is not None:
        pixel_decoder.eval(); pixel_decoder.requires_grad_(False)

    ckpt = torch.load(args.classifier_checkpoint_path, map_location=device)
    clf_input_dim = ckpt.get("input_dim", args.embedding_dim)
    clf_hidden_dim = ckpt.get("hidden_dim", args.classifier_hidden_dim)
    classifier = SpillClassifier(input_dim=clf_input_dim, hidden_dim=clf_hidden_dim).to(device)
    classifier.head.load_state_dict(ckpt["head"])
    classifier.eval()
    logger.info("Loaded classifier (epoch %d, val_auc=%.4f, input_dim=%d)",
                ckpt.get("epoch", -1), ckpt.get("val_auc", float("nan")), clf_input_dim)
    use_tactile_cls = bool(getattr(args, "use_tactile", False)) and clf_input_dim > adapter.latent_dim

    assert args.wm_checkpoint_path, "--wm_checkpoint_path is required"
    ema, diffusion = _load_wm(args, adapter, device)

    n_ctx = (args.num_history + 1) // max(temporal_ds, 1)
    use_pixel_dec = getattr(args, "use_pixel_decoder_for_val", False) and pixel_decoder is not None
    make_videos = use_pixel_dec and spill_only

    if make_videos:
        video_dir = output_dir / "videos"
        video_dir.mkdir(exist_ok=True)

    is_identity = isinstance(adapter, IdentityAdapter)
    all_probs, all_gt_probs, all_labels_flat = [], [], []
    n_videos_saved = 0

    logger.info("Running eval...")
    with torch.no_grad():
        for batch in loader:
            emb, actions, tactile, labels = _unpack_batch(batch)
            emb = emb.to(device)
            actions = actions.to(device)
            if tactile is not None:
                tactile = tactile.to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                val_latent = autoencoder.encode(emb)
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

                wm_latent = diffusion.generate(
                    ema, val_latent_adapted, actions,
                    n_context_frames=n_ctx,
                    n_frames=val_latent_adapted.shape[1],
                    tactile=tactile,
                )
                # Both GT and WM are in adapter latent space; optionally fuse tactile CLS.
                tact = tactile if use_tactile_cls else None
                gt_x = fuse_tactile(val_latent_adapted, tact) if tact is not None else val_latent_adapted.reshape(
                    *val_latent_adapted.shape[:2], val_latent_adapted.shape[2] * val_latent_adapted.shape[3], val_latent_adapted.shape[4])
                wm_x = fuse_tactile(wm_latent, tact) if tact is not None else wm_latent.reshape(
                    *wm_latent.shape[:2], wm_latent.shape[2] * wm_latent.shape[3], wm_latent.shape[4])
                gt_logits = classifier(gt_x)
                wm_logits = classifier(wm_x)

                wm_probs  = torch.sigmoid(wm_logits)
                gt_probs  = torch.sigmoid(gt_logits)

                # Decode to pixels for video
                gt_pixels = wm_pixels = None
                if make_videos:
                    gt_pixels = pixel_decoder(val_latent_adapted)        # (B, T, H, W, C)
                    wm_pixels = pixel_decoder(wm_latent)

            # Per-sample processing
            B = emb.shape[0]
            for b in range(B):
                lbl = labels[b]                      # (T,) cpu
                ctx_labels = lbl[:n_ctx]
                fut_labels = lbl[n_ctx:]

                if spill_only:
                    # Skip windows where context is dirty or future has no spill
                    if ctx_labels.any() or not fut_labels.any():
                        continue

                probs_fut    = wm_probs[b, n_ctx:].float().cpu()
                gt_probs_fut = gt_probs[b, n_ctx:].float().cpu()
                all_probs.append(probs_fut)
                all_gt_probs.append(gt_probs_fut)
                all_labels_flat.append(fut_labels.float())

                # Video for this window
                if make_videos and n_videos_saved < args.num_videos:
                    gt_imgs = (gt_pixels[b].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    wm_imgs = (wm_pixels[b].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    T = gt_imgs.shape[0]
                    frames = []
                    for t in range(T):
                        gt_lbl   = lbl[t].item()
                        pred_prob = wm_probs[b, t].item()
                        if t < n_ctx:
                            # context: gray border to distinguish from predictions
                            frame = np.concatenate([
                                _add_border(gt_imgs[t], _GRAY),
                                _add_border(wm_imgs[t], _GRAY),
                            ], axis=1)
                        else:
                            frame = _make_side_by_side(
                                gt_imgs[t], wm_imgs[t], gt_lbl, pred_prob
                            )
                        frames.append(frame)
                    _save_video(frames, video_dir / f"spill_window_{n_videos_saved:03d}.mp4")
                    n_videos_saved += 1

    if not all_probs:
        logger.error("No valid windows found.")
        return {}

    all_probs    = torch.cat(all_probs).numpy()
    all_gt_probs = torch.cat(all_gt_probs).numpy()
    all_labels   = torch.cat(all_labels_flat).numpy()

    try:
        wm_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        wm_auc = float("nan")
    try:
        gt_auc = roc_auc_score(all_labels, all_gt_probs)
    except ValueError:
        gt_auc = float("nan")

    def _confusion(probs, labels, thresh=0.5):
        preds = (probs >= thresh).astype(float)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        return tp, tn, fp, fn

    wm_tp, wm_tn, wm_fp, wm_fn = _confusion(all_probs, all_labels)
    gt_tp, gt_tn, gt_fp, gt_fn = _confusion(all_gt_probs, all_labels)

    n_windows = len(all_probs) // max(1, (args.n_frames - n_ctx))
    results = {
        "wm_auc": wm_auc, "gt_auc": gt_auc,
        "wm_tp": wm_tp, "wm_tn": wm_tn, "wm_fp": wm_fp, "wm_fn": wm_fn,
        "gt_tp": gt_tp, "gt_tn": gt_tn, "gt_fp": gt_fp, "gt_fn": gt_fn,
        "n_windows": n_windows, "spill_only": spill_only, "n_videos": n_videos_saved,
    }

    print(f"\n=== Classifier Eval (spill_only={spill_only}) ===")
    print(f"  GT AUC (classifier ceiling) : {gt_auc:.4f}")
    print(f"  GT  TP={gt_tp}  TN={gt_tn}  FP={gt_fp}  FN={gt_fn}")
    print(f"  WM AUC (end-to-end)         : {wm_auc:.4f}")
    print(f"  WM  TP={wm_tp}  TN={wm_tn}  FP={wm_fp}  FN={wm_fn}")
    print(f"  Gap (GT - WM)               : {gt_auc - wm_auc:.4f}")
    print(f"  Windows                     : {n_windows}")
    if make_videos:
        print(f"  Videos                      : {n_videos_saved} → {video_dir}")

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
                        type=lambda x: x.lower() == "true", default=True)

    # ── World model ───────────────────────────────────────────────────────
    parser.add_argument("--wm_checkpoint_path", type=str, required=True)
    parser.add_argument("--dit_size", type=str, default="S", choices=list(DIT_SIZES.keys()))
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--wide_head", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--objective", type=str, default="flow_matching",
                        choices=["ddpm", "flow_matching"])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=5)
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
    parser.add_argument("--n_frames", type=int, default=11)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--variable_history_sampling",
                        type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="bfloat16")

    # ── Spill / video ────────────────────────────────────────────────────
    parser.add_argument("--spill_only", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--spill_windows_per_transition", type=int, default=3)
    parser.add_argument("--num_videos", type=int, default=20)

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
