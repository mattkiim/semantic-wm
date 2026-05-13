# argparse conversion
import argparse
from pathlib import Path
import os
import datetime
import wandb
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training import train_wm
import logging


def get_configs():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("--dataset_dir", type=Path, default=Path("sample_data"))
    parser.add_argument("--h5_train_path", type=str, default=None,
                        help="Path to train HDF5 file (combined_v3 format). Overrides --dataset_dir.")
    parser.add_argument("--h5_val_path", type=str, default=None,
                        help="Path to val HDF5 file (combined_v3 format).")
    parser.add_argument("--h5_camera_key", type=str, default="camera_0",
                        help="Dataset key for camera frames inside each trajectory group.")
    parser.add_argument(
        "--variable_history_sampling", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument("--checkpoint_dir", type=Path, default=None)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--subset_names", type=str, default="bridge_v2")
    parser.add_argument("--action_dim", type=int, default=7,
                        help="Action dimensionality (7 for combined_v3, 10 for bridge_v2 MP4).")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_new", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--action_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--objective",
        type=str,
        choices=["ddpm", "flow_matching"],
        default="ddpm",
    )
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--model_dim", type=int, default=1152)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["vae", "rae", "scale_rae_siglip", "scale_rae_webssl", "qwen", "vjepa2", "cosmos", "vavae"],
        default="vae",
    )
    parser.add_argument("--vae_model_path", type=str, default=None,
                        help="HuggingFace repo or local path for the VAE encoder "
                             "(default: stabilityai/sd-vae-ft-mse). Use "
                             "'stabilityai/stable-diffusion-3-medium-diffusers' + "
                             "--vae_subfolder vae for the SD3 VAE if you have access.")
    parser.add_argument("--vae_subfolder", type=str, default=None,
                        help="Subfolder inside the VAE repo (e.g. 'vae' for SD3).")
    parser.add_argument("--qwen_model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--qwen_mode", type=str, default="video", choices=["video", "image"])
    parser.add_argument("--vjepa2_model_size", type=str, default="vitl", choices=["vitl", "vitb"],
                        help="V-JEPA 2.1 model size: vitl (300M) or vitb (80M)")
    parser.add_argument("--vjepa2_checkpoint_path", type=str, default=None,
                        help="Local path to V-JEPA 2.1 .pt checkpoint (downloads if not provided)")
    parser.add_argument("--vjepa2_input_size", type=int, default=256,
                        help="Input resolution for V-JEPA 2.1 (256 gives 16x16 grid, 384 gives 24x24)")
    parser.add_argument("--cosmos_checkpoint_dir", type=str, default=None,
                        help="Directory containing Cosmos encoder.jit and decoder.jit")
    parser.add_argument("--vavae_checkpoint_path", type=str, default=None,
                        help="Path to VA-VAE f16d32 .pt checkpoint")
    parser.add_argument("--encoder_normalization_stat_path", type=str, default=None)
    parser.add_argument("--rae_config_path", type=str, default=None)
    parser.add_argument("--rae_pretrained_decoder_path", type=str, default=None)
    parser.add_argument(
        "--scale_rae_decoder_config",
        type=str,
        default=None,
        help="Path to Scale-RAE decoder config.json (e.g. configs/decoder/siglip/config.json)",
    )
    # ---- Adapter (frozen during DiT training) --------------------------------
    parser.add_argument(
        "--adapter_type",
        type=str,
        choices=["identity", "mlp", "svae"],
        default="identity",
        help="Adapter type for latent projection (identity = no projection)",
    )
    parser.add_argument(
        "--adapter_checkpoint_path",
        type=str,
        default=None,
        help="Path to pretrained adapter checkpoint",
    )
    parser.add_argument(
        "--use_pixel_decoder_for_val",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Use pixel decoder from adapter checkpoint for validation instead of RAE decoder",
    )
    parser.add_argument(
        "--adapter_latent_dim",
        type=int,
        default=96,
        help="Compact latent dim for mlp/svae adapters",
    )
    parser.add_argument("--adapter_hidden_dim", type=int, default=None)
    parser.add_argument("--adapter_num_heads", type=int, default=12)
    parser.add_argument("--adapter_num_layers", type=int, default=3)
    parser.add_argument("--adapter_intermediate_size", type=int, default=3072)
    parser.add_argument("--dit_pretrained_backbone_path", type=str, default=None)
    parser.add_argument("--dit_size", type=str, default="S")
    parser.add_argument("--validate_every_samples", type=int, default=1_000_000)
    parser.add_argument("--log_every_samples", type=int, default=10_000)
    parser.add_argument("--sampling_timesteps", type=int, default=10)
    parser.add_argument("--window_len", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--precision", type=str, default="bfloat16")
    parser.add_argument(
        "--save_model", type=lambda x: x.lower() == "true", default=False
    )

    # torch compile settings
    parser.add_argument(
        "--compile_models", type=lambda x: x.lower() == "true", default=True
    )
    parser.add_argument("--compile_cache_dir", type=str, default="/tmp")

    parser.add_argument("--model_type", type=str, choices=["dit"], default="dit")
    parser.add_argument("--time_dist_type", type=str, default="logit_normal")
    parser.add_argument("--logit_mu", type=float, default=0.0)
    parser.add_argument("--logit_sigma", type=float, default=1.0)
    parser.add_argument("--use_shift", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=2e-3)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.7)

    # wandb
    parser.add_argument("--wandb_project_name", type=str, default="world-model-rae")
    parser.add_argument("--wandb_entity", type=str, default="sarath-chandar")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
    )
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])
    parser.add_argument("--wide_head", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--temporal_mode", type=str, choices=["factored", "joint"], default="factored",
                        help="Temporal attention mode: 'factored' (per-patch across time) or 'joint' (all patches across time with block-causal mask)")

    # Multi-view transfer learning
    parser.add_argument("--num_views", type=int, default=1,
                        help="Number of camera views (1=single-view, 3=multi-view)")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None,
                        help="Path to single-view pretrained checkpoint for transfer learning")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0,
                        help="Freeze pretrained backbone params for N epochs (transfer learning)")

    return parser.parse_args()


DIT_MODEL_PRESETS = {
    "S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}

if __name__ == "__main__":
    args = get_configs()

    args.model_dim = DIT_MODEL_PRESETS[args.dit_size]["hidden_size"]
    args.layers = DIT_MODEL_PRESETS[args.dit_size]["depth"]
    args.heads = DIT_MODEL_PRESETS[args.dit_size]["num_heads"]

    world_size = int(os.environ.get("WORLD_SIZE", 1))

    now = datetime.datetime.now().strftime("%dT%H-%M-%S")

    Objective = "DDPM" if args.objective == "ddpm" else "Flow"

    logging.basicConfig(level=logging.INFO)

    if args.compile_cache_dir is not None and args.compile_models:
        cache_dir = os.path.abspath(args.compile_cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        # PyTorch uses TORCHINDUCTOR_CACHE_DIR or torch._inductor.config.cache_dir
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        # Also set FX graph cache for cross-run reuse of compiled graphs
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        print(f"Set PyTorch compile cache directory to: {cache_dir}")

    train_wm(args)
