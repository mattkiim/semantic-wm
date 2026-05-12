import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
import fire
import einops

# Ensure we can import from src
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.dataset import OpenXMP4VideoDataset
from src.models.scale_rae import ScaleRAE

def calculate_stats(
    dataset_dir: str = "sample_data",
    subset_names: str = "bridge_v2",
    n_frames: int = 10,
    num_history: int = 2,
    frame_skip: int = 2,
    action_dim: int = 10,
    input_h: int = 256,
    input_w: int = 256,
    batch_size: int = 8,
    num_workers: int = 8,
    encoder_type: str = "scale_rae_webssl",
    scale_rae_decoder_config: str = None,
    rae_pretrained_decoder_path: str = None,
    output_path: str = "latent_stats.pt",
    reshape_to_2d: bool = True,
    precision: str = "bfloat16",
):
    """
    Calculates the mean and variance of ScaleRAE latents over a dataset.
    
    Args:
        dataset_dir: Root directory of the dataset.
        subset_names: Subsets to include (comma-separated).
        n_frames: Number of frames in each sample.
        num_history: Number of history frames.
        frame_skip: Frame skip factor.
        action_dim: Dimension of actions.
        input_h: Input height for images.
        input_w: Input width for images.
        batch_size: Batch size for dataloader.
        num_workers: Number of workers for dataloader.
        encoder_type: Type of ScaleRAE encoder (scale_rae_siglip or scale_rae_webssl).
        scale_rae_decoder_config: Path to decoder config.json.
        rae_pretrained_decoder_path: Path to pretrained decoder model.pt.
        output_path: Where to save the calculated stats.
        reshape_to_2d: Whether to reshape latents to 2D before calculating stats.
        precision: Precision to use (float32 or bfloat16).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32

    # Create a dummy args object for the dataset
    class Args:
        pass
    args = Args()
    args.dataset_dir = dataset_dir
    args.subset_names = subset_names
    args.n_frames = n_frames
    args.num_history = num_history
    args.frame_skip = frame_skip
    args.action_dim = action_dim
    args.input_h = input_h
    args.input_w = input_w
    args.variable_history_sampling = False

    # Initialize ScaleRAE
    encoder_name = encoder_type.replace("scale_rae_", "")
    print(f"Initializing ScaleRAE with encoder: {encoder_name}")
    
    if scale_rae_decoder_config is None:
        raise ValueError("scale_rae_decoder_config is required.")
    if rae_pretrained_decoder_path is None:
        raise ValueError("rae_pretrained_decoder_path is required.")

    model = ScaleRAE(
        encoder_name=encoder_name,
        decoder_config_path=scale_rae_decoder_config,
        pretrained_decoder_path=rae_pretrained_decoder_path,
        reshape_to_2d=reshape_to_2d,
    ).to(device)
    model.eval()

    # Initialize Dataset
    print(f"Initializing dataset from {dataset_dir} for subsets {subset_names}")
    dataset = OpenXMP4VideoDataset(args, split="train")
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        shuffle=False
    )

    sum_z = None
    sum_z2 = None
    count = 0

    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Calculating latent statistics"):
            # x is (B, T, H, W, C)
            x = x.to(device)
            
            # Encode WITHOUT normalization
            with torch.autocast(device_type=device.type, dtype=dtype):
                # ScaleRAE.encode returns (B, T, h, w, C_lat) if reshape_to_2d=True
                z = model.encode(x) 
            
            if reshape_to_2d:
                # z is (B, T, h, w, C_lat)
                # We want to match (BT, C, H, W) where normalization happens in scale_rae.py
                # This ensures the calculated stats have the right broadcasting shape (1, C, 1, 1)
                z_for_stats = einops.rearrange(z, "b t h w c -> (b t) c h w")
                
                # sum_z should be (1, C, 1, 1)
                curr_sum = z_for_stats.sum(dim=(0, 2, 3), keepdim=True)
                curr_sum2 = (z_for_stats**2).sum(dim=(0, 2, 3), keepdim=True)
                curr_count = z_for_stats.shape[0] * z_for_stats.shape[2] * z_for_stats.shape[3]
            else:
                # z is (B, T, N, C_lat)
                # Normalization happens when it's (BT, N, C)
                # sum_z should be (1, 1, C)
                z_for_stats = einops.rearrange(z, "b t n c -> (b t) n c")
                curr_sum = z_for_stats.sum(dim=(0, 1), keepdim=True)
                curr_sum2 = (z_for_stats**2).sum(dim=(0, 1), keepdim=True)
                curr_count = z_for_stats.shape[0] * z_for_stats.shape[1]
            
            if sum_z is None:
                sum_z = curr_sum.to(torch.float64)
                sum_z2 = curr_sum2.to(torch.float64)
            else:
                sum_z += curr_sum.to(torch.float64)
                sum_z2 += curr_sum2.to(torch.float64)
            
            count += curr_count

    if count == 0:
        print("No data processed. Check dataset path and subsets.")
        return

    mean = sum_z / count
    var = (sum_z2 / count) - (mean ** 2)
    
    # Save as float32 for model weights
    stats = {
        "mean": mean.to(torch.float32).cpu(),
        "var": var.to(torch.float32).cpu()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) if os.path.dirname(output_path) else None
    torch.save(stats, output_path)
    print(f"Statistics calculated over {count} spatial tokens.")
    print(f"Stats saved to: {output_path}")

if __name__ == "__main__":
    fire.Fire(calculate_stats)
