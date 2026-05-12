"""CLI entry point for trajectory success probe training and evaluation.

Usage:
    # Train probe only:
    python -m src.launch_probe --mode train \
        --encoder_type scale_rae_webssl \
        --adapter_type svae --adapter_checkpoint_path /path/to/adapter.pt \
        --dataset_dir /path/to/data --subset_names soar \
        --probe_type temporal

    # Train + evaluate on WM-generated trajectories:
    python -m src.launch_probe --mode train_eval \
        --encoder_type scale_rae_webssl \
        --adapter_type svae --adapter_checkpoint_path /path/to/adapter.pt \
        --checkpoint_path /path/to/dit.pt \
        --dataset_dir /path/to/data --subset_names soar \
        --probe_type temporal

    # Evaluate existing probe on WM-generated trajectories:
    python -m src.launch_probe --mode eval \
        --probe_checkpoint_path /path/to/probe_best.pt \
        --checkpoint_path /path/to/dit.pt \
        --encoder_type scale_rae_webssl \
        --adapter_type svae --adapter_checkpoint_path /path/to/adapter.pt \
        --dataset_dir /path/to/data --subset_names soar
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

from src.evaluation.train_probe import train_success_probe, evaluate_probe_on_generated
from src.launch_eval import MODEL_PRESETS, DIT_MODEL_PRESETS


def get_configs():
    parser = argparse.ArgumentParser(description="Trajectory success probe training and evaluation")

    # ── Mode ────────────────────────────────────────────────────────────
    parser.add_argument("--mode", type=str, choices=["train", "eval", "train_eval"], default="train_eval")

    # ── Model preset (reuse from launch_eval) ───────────────────────────
    parser.add_argument("--model_preset", type=str, choices=list(MODEL_PRESETS.keys()), default=None)

    # ── Encoder config ──────────────────────────────────────────────────
    parser.add_argument("--encoder_type", type=str,
                        choices=["vae", "rae", "scale_rae_siglip", "scale_rae_webssl", "qwen", "vjepa2", "cosmos", "vavae"],
                        default="scale_rae_webssl")
    parser.add_argument("--cosmos_checkpoint_dir", type=str, default=None)
    parser.add_argument("--vavae_checkpoint_path", type=str, default=None)
    parser.add_argument("--scale_rae_decoder_config", type=str, default=None)
    parser.add_argument("--rae_pretrained_decoder_path", type=str, default=None)
    parser.add_argument("--rae_config_path", type=str, default=None)
    parser.add_argument("--encoder_normalization_stat_path", type=str, default=None)
    parser.add_argument("--vjepa2_checkpoint_path", type=str, default=None)
    parser.add_argument("--vjepa2_model_size", type=str, default="vitl")
    parser.add_argument("--vjepa2_input_size", type=int, default=256)
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--qwen_mode", type=str, default="video")

    # ── Adapter config ──────────────────────────────────────────────────
    parser.add_argument("--adapter_type", type=str, choices=["identity", "mlp", "svae"], default="svae")
    parser.add_argument("--adapter_checkpoint_path", type=str, default=None)
    parser.add_argument("--adapter_latent_dim", type=int, default=96)
    parser.add_argument("--adapter_hidden_dim", type=int, default=None)
    parser.add_argument("--adapter_num_heads", type=int, default=12)
    parser.add_argument("--adapter_num_layers", type=int, default=3)
    parser.add_argument("--adapter_intermediate_size", type=int, default=3072)

    # ── Probe config ────────────────────────────────────────────────────
    parser.add_argument("--probe_type", type=str, choices=["linear", "temporal", "spatiotemporal"], default="temporal")
    parser.add_argument("--n_sample_frames", type=int, default=8)
    parser.add_argument("--sampling_strategy", type=str, choices=["uniform", "fps_1", "bookend", "final_heavy"], default="uniform")
    parser.add_argument("--pool_mode", type=str, choices=["mean", "super_patch_4x4", "super_patch_8x8"], default="mean")
    parser.add_argument("--feature_space", type=str, choices=["encoder", "adapter"], default="adapter")
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_batch_size", type=int, default=16)
    parser.add_argument("--probe_checkpoint_path", type=str, default=None,
                        help="Path to pre-trained probe (for --mode eval)")
    parser.add_argument("--train_progress_regressor", action="store_true")

    # ── Data ────────────────────────────────────────────────────────────
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--subset_names", type=str, default="soar")
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # ── DiT config (for eval mode) ──────────────────────────────────────
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="DiT checkpoint for WM generation (required for eval mode)")
    parser.add_argument("--dit_size", type=str, default="S")
    parser.add_argument("--model_dim", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--wide_head", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--action_dim", type=int, default=10)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--use_pixel_decoder", type=lambda x: x.lower() == "true", default=False)

    # ── Diffusion (for eval mode) ───────────────────────────────────────
    parser.add_argument("--objective", type=str, choices=["ddpm", "flow_matching"], default="flow_matching")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--use_shift", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--time_dist_type", type=str, default="uniform")

    # ── Wandb ───────────────────────────────────────────────────────────
    parser.add_argument("--wandb_project_name", type=str, default="world-model-rae")
    parser.add_argument("--wandb_entity", type=str, default="sarath-chandar")
    parser.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="disabled")

    # ── Output / misc ───────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="probe_outputs")
    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _apply_preset(args, preset_name: str) -> None:
    """Override args with values from a model preset."""
    preset = MODEL_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(args, key, value)
    args.model_preset = preset_name


def _setup_eval(args, device):
    """Set up DiT model and diffusion for WM-generated evaluation."""
    from src.evaluation.evaluate import _load_dit_checkpoint
    from src.training.diffusion import Diffusion, FlowMatching
    from src.models.base_autoencoder import create_autoencoder, encoder_config_from_args
    from src.models.adapters import create_adapter, IdentityAdapter
    from src.training.utils import resolve_adapter_ckpt, load_frozen_adapter_weights

    precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16

    # Load encoder + adapter
    autoencoder = create_autoencoder(encoder_config_from_args(args)).to(device)
    autoencoder.eval()
    autoencoder.requires_grad_(False)

    adapter_cfg, adapter_ckpt_data = resolve_adapter_ckpt(args, device)
    adapter = create_adapter(adapter_cfg, input_dim=autoencoder.latent_dim).to(device)
    if adapter_ckpt_data is not None:
        load_frozen_adapter_weights(adapter, None, adapter_ckpt_data)
    adapter.eval()
    adapter.requires_grad_(False)

    is_identity = isinstance(adapter, IdentityAdapter)
    feature_space = getattr(args, "feature_space", "adapter")
    if is_identity:
        feature_space = "encoder"
        feature_dim = autoencoder.latent_dim
        adapter = None
    else:
        feature_dim = adapter.latent_dim

    in_channels = adapter.latent_dim if adapter is not None else autoencoder.latent_dim
    temporal_ds = getattr(autoencoder, "temporal_downsample_factor", 1)
    effective_action_dim = args.action_dim * temporal_ds

    # Load DiT
    dit_model = _load_dit_checkpoint(
        args.checkpoint_path, args, in_channels, device,
        action_dim_override=effective_action_dim,
    )

    # Diffusion
    shift = 1.0
    if args.encoder_type != "vae" and getattr(args, "use_shift", True):
        m = (256 / args.patch_size ** 2) * in_channels
        shift = (m / 4096) ** 0.5

    if args.objective == "flow_matching":
        diffusion = FlowMatching(
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            time_dist_type=getattr(args, "time_dist_type", "uniform"),
            time_dist_shift=shift,
            device=device,
        ).to(device)
    else:
        diffusion = Diffusion(
            timesteps=args.timesteps,
            sampling_timesteps=args.sampling_timesteps,
            time_dist_shift=shift,
            device=device,
        ).to(device)

    return autoencoder, adapter, feature_space, feature_dim, dit_model, diffusion


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = get_configs()

    # Apply preset if specified
    if args.model_preset is not None:
        _apply_preset(args, args.model_preset)

    # Apply DiT size preset
    dit_preset = DIT_MODEL_PRESETS.get(args.dit_size, DIT_MODEL_PRESETS["S"])
    if args.model_dim is None:
        args.model_dim = dit_preset["hidden_size"]
    if args.layers is None:
        args.layers = dit_preset["depth"]
    if args.heads is None:
        args.heads = dit_preset["num_heads"]

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("=" * 60)
    logging.info("Mode: %s | Encoder: %s | Probe: %s", args.mode, args.encoder_type, args.probe_type)
    logging.info("Feature space: %s | Pool: %s | Frames: %d (%s)",
                 args.feature_space, args.pool_mode, args.n_sample_frames, args.sampling_strategy)
    logging.info("=" * 60)

    # ── Train ───────────────────────────────────────────────────────────
    if args.mode in ("train", "train_eval"):
        results = train_success_probe(args)
        logging.info("Training complete. Best test accuracy: %.4f", results["best_test_accuracy"])

    # ── Eval on WM-generated ────────────────────────────────────────────
    if args.mode in ("eval", "train_eval"):
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint_path (DiT) is required for eval mode")

        autoencoder, adapter, feature_space, feature_dim, dit_model, diffusion = _setup_eval(args, device)

        # Load probe
        from src.evaluation.probe import create_probe

        if args.mode == "eval":
            assert args.probe_checkpoint_path is not None, "--probe_checkpoint_path required for --mode eval"
            probe_ckpt = torch.load(args.probe_checkpoint_path, map_location=device)
        else:
            probe_ckpt = torch.load(Path(args.output_dir) / "probe_best.pt", map_location=device)

        probe = create_probe(
            probe_type=probe_ckpt["probe_type"],
            feature_dim=probe_ckpt["feature_dim"],
            n_frames=probe_ckpt["n_sample_frames"],
            pool_mode=probe_ckpt["pool_mode"],
        ).to(device)
        probe.load_state_dict(probe_ckpt["probe"])
        probe.eval()

        # Test dataset
        from src.data.probe_dataset import TrajectoryProbeDataset
        from torch.utils.data import DataLoader

        test_dataset = TrajectoryProbeDataset(
            dataset_dir=args.dataset_dir,
            subset_names=args.subset_names,
            split="test",
            n_sample_frames=probe_ckpt["n_sample_frames"],
            sampling_strategy=args.sampling_strategy,
            input_h=args.input_h,
            input_w=args.input_w,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.probe_batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        precision = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
        eval_results = evaluate_probe_on_generated(
            args, probe, autoencoder, adapter, feature_space,
            diffusion, dit_model, test_loader, device, precision,
        )

        print("\n" + "=" * 60)
        print("Probe Evaluation Results (Real vs WM-Generated)")
        print("=" * 60)
        for k, v in eval_results.items():
            print(f"  {k}: {v:.4f}")
        print("=" * 60)
