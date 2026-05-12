"""CLI entry point for adapter training.

Usage::

    python -m src.launch_adapter \\
        --encoder_type rae \\
        --adapter_type svae \\
        --adapter_latent_dim 96 \\
        --dataset_dir sample_data \\
        --subset_names bridge_v2

    # PS-VAE second stage (unfreezes representation encoder):
    python -m src.launch_adapter \\
        --stage psvae \\
        --adapter_checkpoint_path outputs/adapter_run/adapter_ckpt_*.pt \\
        ...
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.train_adapter import train_adapter


def get_configs():
    parser = argparse.ArgumentParser(
        description="Train adapter layers for RAE latent projection"
    )

    # ---- Dataset (shared with launch.py) ------------------------------------
    parser.add_argument("--dataset_dir", type=str, default="sample_data")
    parser.add_argument("--subset_names", type=str, default="bridge_v2")
    parser.add_argument("--n_frames", type=int, default=10)
    parser.add_argument("--num_history", type=int, default=2)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=10)
    parser.add_argument("--input_h", type=int, default=256)
    parser.add_argument("--input_w", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--variable_history_sampling", type=lambda x: x.lower() == "true", default=False
    )

    # ---- Encoder (frozen) ---------------------------------------------------
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=["vae", "rae", "scale_rae_siglip", "scale_rae_webssl", "qwen", "vjepa2", "cosmos", "vavae"],
        default="rae",
    )
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
    
    parser.add_argument("--rae_config_path", type=str, default=None)
    parser.add_argument("--rae_pretrained_decoder_path", type=str, default=None)
    parser.add_argument("--scale_rae_decoder_config", type=str, default=None)
    parser.add_argument("--encoder_normalization_stat_path", type=str, default=None)

    # ---- Adapter ------------------------------------------------------------
    parser.add_argument(
        "--adapter_type",
        type=str,
        choices=["identity", "mlp", "svae"],
        default="svae",
    )
    parser.add_argument("--adapter_latent_dim", type=int, default=96)
    parser.add_argument(
        "--adapter_hidden_dim",
        type=int,
        default=None,
        help="MLP hidden dim (default: auto)",
    )
    parser.add_argument("--adapter_num_heads", type=int, default=16)
    parser.add_argument("--adapter_num_layers", type=int, default=3)
    parser.add_argument("--adapter_intermediate_size", type=int, default=2048)
    parser.add_argument("--adapter_progressive", type=lambda x: x.lower() == "true", default=False,
                        help="Enable gradual dim reduction: d_h → d_mid → d_l instead of d_h → d_l")
    parser.add_argument("--adapter_mid_dim", type=int, default=None,
                        help="Intermediate dim for progressive mode (default: geometric mean of d_h and d_l)")
    parser.add_argument("--adapter_mid_heads", type=int, default=None,
                        help="Attention heads for mid-dim blocks (default: auto)")
    parser.add_argument("--adapter_latent_layers", type=int, default=0,
                        help="Number of transformer refinement blocks at d_l (default: 0 = disabled)")
    parser.add_argument("--adapter_latent_heads", type=int, default=None,
                        help="Attention heads for latent-dim blocks (default: auto)")
    parser.add_argument(
        "--adapter_checkpoint_path",
        type=str,
        default=None,
        help="Resume adapter from this checkpoint",
    )

    # ---- Training stage -----------------------------------------------------
    parser.add_argument(
        "--stage",
        type=str,
        choices=["svae", "psvae"],
        default="svae",
        help="svae = frozen encoder, psvae = unfreeze encoder for pixel loss",
    )

    # ---- Loss hyper-parameters ----------------------------------------------
    parser.add_argument("--kl_weight", type=float, default=1e-4)
    parser.add_argument("--kl_warmup_fraction", type=float, default=0.2,
                        help="Fraction of training over which KL weight ramps from 0 to kl_weight (0 = no warmup)")
    parser.add_argument("--cos_weight", type=float, default=1.0)
    parser.add_argument("--spectral_weight", type=float, default=0.0,
                        help="Weight for spectral (FFT) reconstruction loss on adapter features (0 = disabled)")
    parser.add_argument("--pixel_weight", type=float, default=1.0)
    parser.add_argument("--use_lpips", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--lpips_weight", type=float, default=0.1)
    parser.add_argument("--perceptual_warmup_samples", type=int, default=50000,
                        help="Number of samples before enabling LPIPS loss (0 = immediate)")
    parser.add_argument("--use_ssim", type=lambda x: x.lower() == "true", default=False,
                        help="Enable MS-SSIM loss")
    parser.add_argument("--ssim_weight", type=float, default=0.5,
                        help="Weight for MS-SSIM loss term")

    # ---- Discriminator (adversarial training) --------------------------------
    parser.add_argument("--use_discriminator", type=lambda x: x.lower() == "true", default=False,
                        help="Enable PatchGAN adversarial training")
    parser.add_argument("--disc_weight", type=float, default=0.1,
                        help="GAN loss weight (before adaptive scaling)")
    parser.add_argument("--disc_start_samples", type=int, default=100_000,
                        help="Number of samples before starting discriminator")
    parser.add_argument("--disc_lr", type=float, default=4e-5,
                        help="Discriminator learning rate")

    # ---- Pixel decoder (optional direct fl → RGB path) ----------------------
    # When enabled, a lightweight LDM-style CNN is trained alongside the adapter
    # to decode the compact latent fl directly to RGB.
    # S-VAE: trained with fl.detach() (no gradient leaks into adapter/encoder).
    # PS-VAE: trained with fl (gradients propagate back into the encoder).
    parser.add_argument(
        "--use_pixel_decoder",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Train a direct fl → RGB pixel decoder alongside the adapter",
    )
    parser.add_argument(
        "--pixel_decoder_base_channels",
        type=int,
        default=128,
        help="Base channel width of the pixel decoder CNN",
    )
    parser.add_argument(
        "--pixel_decoder_channel_multipliers",
        type=int,
        nargs="+",
        default=[1, 1, 2, 4, 8],
        help=(
            "Channel multipliers (outer→inner). "
            "Upsamplings = len-1. Default [1,1,2,4,8] → 4× (16→256 for 256-px inputs)."
        ),
    )
    parser.add_argument(
        "--pixel_decoder_num_res_blocks",
        type=int,
        default=2,
        help="Residual blocks per level in the pixel decoder",
    )
    parser.add_argument(
        "--pixel_decoder_dropout",
        type=float,
        default=0.0,
        help="Dropout probability inside pixel decoder residual blocks",
    )
    parser.add_argument(
        "--pixel_decoder_output_activation",
        type=str,
        choices=["sigmoid", "tanh"],
        default="sigmoid",
        help="Output activation of the pixel decoder ('sigmoid' outputs [0,1])",
    )
    parser.add_argument(
        "--pixel_decoder_attn_resolutions",
        type=int,
        nargs="+",
        default=[16],
        help="Spatial resolutions at which to insert attention blocks in the pixel decoder",
    )
    parser.add_argument(
        "--pixel_decoder_attn_heads",
        type=int,
        default=4,
        help="Number of attention heads in pixel decoder attention blocks",
    )

    # ---- Optimiser / schedule -----------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pixel_decoder_lr_multiplier", type=float, default=3.0,
                        help="LR multiplier for pixel decoder parameters (relative to --lr)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.9)
    parser.add_argument("--precision", type=str, default="bfloat16")

    # ---- Logging / checkpoints ----------------------------------------------
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--log_every_samples", type=int, default=1000)
    parser.add_argument("--validate_every_samples", type=int, default=10_000)

    # ---- WandB --------------------------------------------------------------
    parser.add_argument("--wandb_project_name", type=str, default="world-model-rae")
    parser.add_argument("--wandb_entity", type=str, default="sarath-chandar")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
    )
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=[])

    return parser.parse_args()


if __name__ == "__main__":
    args = get_configs()

    logging.basicConfig(level=logging.INFO)
    train_adapter(args)
