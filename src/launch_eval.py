"""CLI entry point for world model evaluation.

Usage:
    # With a preset:
    python -m src.launch_eval --model_preset DiT-L_WEBSSL --dataset_dir /tmp/ --subset_names bridge_v2

    # With manual config:
    python -m src.launch_eval \
        --checkpoint_path trained_dits/DiT-S_VAE/ckpt_samples_000005102016.pt \
        --encoder_type vae --dit_size S --adapter_type identity --patch_size 2 \
        --dataset_dir /tmp/ --subset_names bridge_v2
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluate import evaluate_model

DIT_MODEL_PRESETS = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


def get_configs():
    parser = argparse.ArgumentParser(
        description="Evaluate world model generation quality"
    )

    # ── Model preset ─────────────────────────────────────────────────────
    parser.add_argument(
        "--model_preset",
        type=str,
        choices=list(MODEL_PRESETS.keys()),
        default=None,
        help="Predefined model config. Overrides individual model args.",
    )

    # ── Model config (manual) ────────────────────────────────────────────
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=[
            "vae",
            "rae",
            "scale_rae_siglip",
            "scale_rae_webssl",
            "qwen",
            "vjepa2",
            "cosmos",
            "vavae",
        ],
        default="vae",
    )
    parser.add_argument("--cosmos_checkpoint_dir", type=str, default=None)
    parser.add_argument("--vavae_checkpoint_path", type=str, default=None)
    parser.add_argument("--dit_size", type=str, default="S")
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument(
        "--wide_head", type=lambda x: x.lower() == "true", default=False
    )
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=10)

    # ── Adapter ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--adapter_type",
        type=str,
        choices=["identity", "mlp", "svae"],
        default="identity",
    )
    parser.add_argument("--adapter_checkpoint_path", type=str, default=None)
    parser.add_argument("--adapter_latent_dim", type=int, default=96)
    parser.add_argument("--adapter_hidden_dim", type=int, default=None)
    parser.add_argument("--adapter_num_heads", type=int, default=12)
    parser.add_argument("--adapter_num_layers", type=int, default=3)
    parser.add_argument("--adapter_intermediate_size", type=int, default=3072)

    # ── Encoder paths ────────────────────────────────────────────────────
    parser.add_argument("--scale_rae_decoder_config", type=str, default=None)
    parser.add_argument("--rae_pretrained_decoder_path", type=str, default=None)
    parser.add_argument("--rae_config_path", type=str, default=None)
    parser.add_argument("--encoder_normalization_stat_path", type=str, default=None)
    parser.add_argument("--vjepa2_checkpoint_path", type=str, default=None)
    parser.add_argument("--vjepa2_model_size", type=str, default="vitl")
    parser.add_argument("--vjepa2_input_size", type=int, default=256)
    parser.add_argument(
        "--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--qwen_mode", type=str, default="video")
    parser.add_argument(
        "--use_pixel_decoder", type=lambda x: x.lower() == "true", default=False
    )

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--subset_names", type=str, default="bridge_v2")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=0)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument(
        "--variable_history_sampling", type=lambda x: x.lower() == "true", default=False
    )

    # ── Eval settings ────────────────────────────────────────────────────
    parser.add_argument("--num_eval_samples", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--metrics", type=str, default="psnr,ssim,lpips,fvd,fid,recon")
    parser.add_argument("--precision", type=str, default="bfloat16")

    # ── Controllability metric ────────────────────────────────────────────
    parser.add_argument(
        "--ctrl_optimizer", type=str, choices=["cem", "gradient", "grid"], default="cem"
    )
    parser.add_argument("--cem_candidates", type=int, default=256)
    parser.add_argument("--cem_elite", type=int, default=25)
    parser.add_argument("--cem_iterations", type=int, default=5)
    parser.add_argument("--ctrl_batch_size", type=int, default=64)
    parser.add_argument("--grid_points_per_dim", type=int, default=30)
    parser.add_argument(
        "--search_dims",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5, 6],
        help="Action dims to optimize (default: 0-6 = all effective). "
        "Non-searched dims are filled from ground truth. "
        "E.g. --search_dims 0 1 for pos_x pos_y only.",
    )
    parser.add_argument("--grad_steps", type=int, default=50)
    parser.add_argument("--grad_lr", type=float, default=0.01)

    # ── PCK metric ──────────────────────────────────────────────────────
    parser.add_argument(
        "--pck_grid_size",
        type=int,
        default=16,
        help="Grid resolution for PCK point tracking (grid_size^2 points)",
    )
    parser.add_argument(
        "--pck_thresholds",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40],
        help="Pixel thresholds for PCK computation",
    )
    parser.add_argument(
        "--pck_point_mode",
        type=str,
        choices=["grid", "salient"],
        default="grid",
        help="Point initialization: 'grid' (uniform) or 'salient' (corner-based)",
    )

    # ── Diffusion / generation ───────────────────────────────────────────
    parser.add_argument(
        "--objective",
        type=str,
        choices=["ddpm", "flow_matching"],
        default="flow_matching",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--use_shift", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--time_dist_type", type=str, default="uniform")
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)

    # ── Reproducibility ─────────────────────────────────────────────────
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=42,
        help="RNG seed for deterministic eval sampling across runs",
    )

    return parser.parse_args()


def _apply_preset(args, preset_name: str) -> None:
    """Override args with values from a model preset."""
    preset = MODEL_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(args, key, value)
    args.model_preset = preset_name


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = get_configs()

    # Apply preset if specified
    if args.model_preset is not None:
        _apply_preset(args, args.model_preset)

    # Ensure checkpoint_path is set
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint_path is required (or use --model_preset)")

    # Apply DiT size preset for model dimensions
    dit_preset = DIT_MODEL_PRESETS[args.dit_size]
    if args.model_dim is None:
        args.model_dim = dit_preset["hidden_size"]
    if args.layers is None:
        args.layers = dit_preset["depth"]
    if args.heads is None:
        args.heads = dit_preset["num_heads"]

    # Set output dir with model name
    preset_name = (
        getattr(args, "model_preset", None)
        or f"DiT-{args.dit_size}_{args.encoder_type}"
    )
    if args.output_dir == "eval_outputs":
        args.output_dir = f"eval_outputs/{preset_name}"

    # Seed all RNGs for reproducible eval
    seed = args.eval_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info("Eval seed: %d", seed)

    logging.info("=" * 60)
    logging.info("Evaluating: %s", preset_name)
    logging.info("Checkpoint: %s", args.checkpoint_path)
    logging.info(
        "Encoder: %s | DiT: %s | Wide head: %s",
        args.encoder_type,
        args.dit_size,
        args.wide_head,
    )
    logging.info("Eval samples: %d | Metrics: %s", args.num_eval_samples, args.metrics)
    logging.info("=" * 60)

    results = evaluate_model(args)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Results for {preset_name}")
    print("=" * 60)
    for key in [
        "psnr_mean",
        "ssim_mean",
        "lpips_mean",
        "fid",
        "fvd",
        "recon_psnr",
        "recon_ssim",
        "pck@5_mean",
        "pck@10_mean",
        "pck@20_mean",
        "pck@40_mean",
        "pck_coverage_mean",
        "controllability_mean",
        "controllability_std",
        "controllability_median",
    ]:
        if key in results:
            print(f"  {key}: {results[key]:.4f}")
    print(f"\nFull results saved to: {args.output_dir}/metrics.json")
