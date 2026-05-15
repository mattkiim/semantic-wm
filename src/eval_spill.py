"""Standalone spill-window evaluation script.

Loads a DiT checkpoint and runs validation only on windows that contain at
least one label=1 (spill) frame, reporting PSNR / SSIM / MSE-latent.

Usage::

    python -m src.eval_spill \
        --checkpoint_path outputs/dit_dinov3_precomputed_tactile_v1/ckpt_samples_000100000.pt \
        --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel/adapter_ckpt_000000057344.pt \
        --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
        --encoder_type precomputed \
        --adapter_type svae \
        --adapter_latent_dim 96 \
        --action_dim 7 \
        --use_pixel_decoder_for_val True \
        --output_dir eval_outputs/spill_tactile_v1
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
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import H5EmbeddingDataset
from src.models.adapters import IdentityAdapter, create_adapter, adapter_config_from_args
from src.models.base_autoencoder import create_autoencoder, encoder_config_from_args
from src.models.model import DiT
from src.training.diffusion import Diffusion, FlowMatching
from src.training.validation import calculate_image_metrics
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


def _load_checkpoint(checkpoint_path: str, model: torch.nn.Module, ema: torch.nn.Module, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "ema" in ckpt:
        result = ema.load_state_dict(strip_state_dict_prefix(ckpt["ema"]), strict=False)
        if result.missing_keys:
            logger.warning("EMA missing keys (architecture mismatch?): %s", result.missing_keys)
            logger.warning("Pass --wide_head / --use_tactile / --tactile_dim to match the checkpoint.")
        logger.info("Loaded EMA weights")
    if "model" in ckpt:
        model.load_state_dict(strip_state_dict_prefix(ckpt["model"]), strict=False)
    samples = ckpt.get("total_samples_seen", "unknown")
    logger.info("Checkpoint: %s  (samples seen: %s)", checkpoint_path, samples)
    return samples


def _unpack_batch(batch):
    if len(batch) == 2:
        return batch[0], batch[1], None
    if len(batch) == 3:
        return batch
    raise ValueError(f"Unexpected batch length {len(batch)}")


def evaluate_spill(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset (spill windows only) ──────────────────────────────────────
    logger.info("Indexing spill windows in val set...")
    spill_dataset = H5EmbeddingDataset(
        args, split="test", spill_only=True,
        spill_windows_per_transition=args.spill_windows_per_transition,
    )
    logger.info("Spill windows found: %d", len(spill_dataset))
    if len(spill_dataset) == 0:
        logger.error("No spill windows found — check that labels key exists and label=1 is present.")
        return {}

    spill_loader = DataLoader(
        spill_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── Models ────────────────────────────────────────────────────────────
    logger.info("Loading autoencoder: %s", args.encoder_type)
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
    if pixel_decoder is not None:
        pixel_decoder.eval()
        pixel_decoder.requires_grad_(False)

    is_identity = isinstance(adapter, IdentityAdapter)
    in_channels = adapter.latent_dim
    has_encoder_decoder = getattr(autoencoder, "has_decoder", True)
    use_pixel_dec = (
        (getattr(args, "use_pixel_decoder_for_val", False) or not has_encoder_decoder)
        and pixel_decoder is not None
    )

    logger.info("Building DiT (size=%s, in_channels=%d)", args.dit_size, in_channels)
    model = DiT(
        in_channels=in_channels,
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
    ema = deepcopy(model)

    samples_seen = _load_checkpoint(args.checkpoint_path, model, ema, device)
    model.eval()
    ema.eval()

    # ── Diffusion ─────────────────────────────────────────────────────────
    shift = 1.0
    if args.encoder_type != "vae":
        m = (256 / args.patch_size ** 2) * in_channels
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
    effective_skip = n_ctx

    # ── Evaluation loop ───────────────────────────────────────────────────
    all_psnr, all_ssim, all_mse = [], [], []

    logger.info("Running spill evaluation (%d windows)...", len(spill_dataset))
    with torch.no_grad():
        for batch in spill_loader:
            val_x, val_actions, val_tactile = _unpack_batch(batch)
            val_x = val_x.to(device)
            val_actions = val_actions.to(device)
            if val_tactile is not None:
                val_tactile = val_tactile.to(device)

            with torch.autocast(device_type="cuda", dtype=precision):
                val_latent = autoencoder.encode(val_x)
                if not is_identity:
                    val_latent_adapted = adapter.encode(val_latent)
                    if isinstance(val_latent_adapted, tuple):
                        val_latent_adapted = val_latent_adapted[0]
                else:
                    val_latent_adapted = val_latent

                if temporal_ds > 1:
                    val_actions = downsample_actions_temporal(val_actions, temporal_ds)
                    if val_tactile is not None:
                        val_tactile = downsample_sequence_temporal(val_tactile, temporal_ds)

                samples_latent = diffusion.generate(
                    ema,
                    val_latent_adapted,
                    val_actions,
                    n_context_frames=n_ctx,
                    n_frames=val_latent_adapted.shape[1],
                    window_len=args.window_len,
                    horizon=args.horizon,
                    cfg=args.cfg,
                    tactile=val_tactile,
                )

                mse = F.mse_loss(samples_latent, val_latent_adapted).item()
                all_mse.append(mse)

                if use_pixel_dec:
                    samples = pixel_decoder(samples_latent)
                    gt_pixels = pixel_decoder(val_latent_adapted)
                    gt_np = gt_pixels.float().clamp(0, 1).cpu().numpy()
                elif has_encoder_decoder:
                    decoded = adapter.decode(samples_latent) if not is_identity else samples_latent
                    samples = autoencoder.decode(decoded)
                    gt_np = val_x.float().clamp(0, 1).cpu().numpy()
                else:
                    continue

                samples_np = samples.float().clamp(0, 1).cpu().numpy()
                if samples_np.shape[1] < gt_np.shape[1]:
                    indices = np.linspace(0, gt_np.shape[1] - 1, samples_np.shape[1]).astype(int)
                    gt_np = gt_np[:, indices]

                metrics = calculate_image_metrics(samples_np, gt_np, skip_frames=effective_skip)
                all_psnr.append(metrics["psnr"])
                all_ssim.append(metrics["ssim"])

    results = {
        "checkpoint": str(args.checkpoint_path),
        "samples_seen": samples_seen,
        "spill_windows": len(spill_dataset),
        "psnr": float(np.mean(all_psnr)),
        "ssim": float(np.mean(all_ssim)),
        "mse_latent": float(np.mean(all_mse)),
        "psnr_std": float(np.std(all_psnr)),
        "ssim_std": float(np.std(all_ssim)),
    }

    print("\n=== Spill Evaluation Results ===")
    print(f"  Spill windows : {results['spill_windows']}")
    print(f"  MSE (latent)  : {results['mse_latent']:.6f}")
    print(f"  PSNR          : {results['psnr']:.2f} ± {results['psnr_std']:.2f}")
    print(f"  SSIM          : {results['ssim']:.4f} ± {results['ssim_std']:.4f}")

    out_path = output_dir / "spill_metrics.json"
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

    parser = argparse.ArgumentParser(description="Evaluate on spill windows only")

    # ── Checkpoint ───────────────────────────────────────────────────────
    parser.add_argument("--checkpoint_path", type=str, required=True)

    # ── DiT model ────────────────────────────────────────────────────────
    parser.add_argument("--dit_size", type=str, default="S", choices=list(DIT_SIZES.keys()))
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--wide_head", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--use_tactile", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--tactile_dim", type=int, default=0)
    parser.add_argument("--h5_tactile_key", type=str, default="cam_tactile_patch_embd")

    # ── Encoder ──────────────────────────────────────────────────────────
    parser.add_argument("--encoder_type", type=str, default="precomputed",
                        choices=["vae", "rae", "precomputed", "scale_rae_siglip",
                                 "scale_rae_webssl", "qwen", "vjepa2", "cosmos", "vavae"])
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--patch_h", type=int, default=14)
    parser.add_argument("--patch_w", type=int, default=14)
    parser.add_argument("--h5_embedding_key", type=str, default="cam_0_patch_embd")

    # ── Adapter ──────────────────────────────────────────────────────────
    parser.add_argument("--adapter_type", type=str, default="svae",
                        choices=["identity", "mlp", "svae"])
    parser.add_argument("--adapter_checkpoint_path", type=str, default=None)
    parser.add_argument("--adapter_latent_dim", type=int, default=96)
    parser.add_argument("--adapter_num_heads", type=int, default=16)
    parser.add_argument("--adapter_num_layers", type=int, default=3)
    parser.add_argument("--adapter_intermediate_size", type=int, default=2048)

    # ── Pixel decoder ────────────────────────────────────────────────────
    parser.add_argument("--use_pixel_decoder_for_val",
                        type=lambda x: x.lower() == "true", default=True)

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument("--h5_val_path", type=str, required=True)
    parser.add_argument("--h5_train_path", type=str, default=None)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--variable_history_sampling",
                        type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # ── Generation ───────────────────────────────────────────────────────
    parser.add_argument("--objective", type=str, default="flow_matching",
                        choices=["ddpm", "flow_matching"])
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bfloat16")

    # ── Spill indexing ───────────────────────────────────────────────────
    parser.add_argument("--spill_windows_per_transition", type=int, default=3)

    # ── Output ───────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="eval_outputs/spill")

    args = parser.parse_args()

    # Apply DiT size preset if individual dims not set
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
    evaluate_spill(args)
