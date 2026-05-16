"""Launch script for spill classifier training."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_classifier import train_classifier


def get_configs():
    DIT_SIZES = {
        "S": {"model_dim": 384, "layers": 12, "heads": 6},
        "B": {"model_dim": 768, "layers": 12, "heads": 12},
        "L": {"model_dim": 1024, "layers": 24, "heads": 16},
        "XL": {"model_dim": 1152, "layers": 28, "heads": 16},
    }

    parser = argparse.ArgumentParser(description="Train spill classifier on WM rollouts")

    # ── Data ─────────────────────────────────────────────────────────────
    parser.add_argument("--h5_train_path", type=str, required=True)
    parser.add_argument("--h5_val_path", type=str, required=True)
    parser.add_argument("--h5_embedding_key", type=str, default="cam_0_patch_embd")
    parser.add_argument("--embedding_dim", type=int, default=384)
    parser.add_argument("--patch_h", type=int, default=14)
    parser.add_argument("--patch_w", type=int, default=14)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--input_h", type=int, default=224)
    parser.add_argument("--input_w", type=int, default=224)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--variable_history_sampling",
                        type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--use_tactile", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--tactile_dim", type=int, default=0)
    parser.add_argument("--h5_tactile_key", type=str, default="cam_tactile_cls_embd")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # ── Encoder ──────────────────────────────────────────────────────────
    parser.add_argument("--encoder_type", type=str, default="precomputed",
                        choices=["vae", "rae", "precomputed", "scale_rae_siglip",
                                 "scale_rae_webssl", "qwen", "vjepa2", "cosmos", "vavae"])

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

    # ── World model (frozen) ──────────────────────────────────────────────
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

    # ── Classifier head ──────────────────────────────────────────────────
    parser.add_argument("--classifier_hidden_dim", type=int, default=256)

    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/classifier")

    # ── WandB ────────────────────────────────────────────────────────────
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project_name", type=str, default="semantic-wm")

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
    train_classifier(args)
