# Reconstruction or Semantics?

Official code for **Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models**.

This repository trains action-conditioned latent diffusion world models for robot video generation and policy evaluation. The paper studies whether robotic world models should operate in reconstruction-aligned latent spaces, such as VAEs and Cosmos, or semantic latent spaces from pretrained vision encoders, such as V-JEPA 2.1, Web-DINO, and SigLIP 2.

**Links:** [Project page](https://hskalin.github.io/semantic-wm/) | [arXiv](https://arxiv.org/abs/2605.06388) | [Hugging Face](https://huggingface.co/Nilaksh404/semantic-wm)

The main finding is that pixel fidelity alone is not enough for choosing a world-model latent space. Reconstruction encoders can score well on visual metrics, but semantic encoders generally preserve action information, task progress, planning utility, and downstream policy behavior better across model scales.

---

## What This Code Includes

- **Multiple encoders**: VAE (SD3), DINOv2-based RAE, SigLIP2/WebSSL ScaleRAE, Qwen2.5-VL, V-JEPA 2.1, Cosmos CI16x16, VA-VAE
- **Semantic adapters (S-VAE)**: Compress high-dimensional encoder features (768–1280-d) to a compact diffusion-friendly space (96-d) via Transformer-based VAE
- **Pixel decoder**: Lightweight CNN that directly maps adapter latents to RGB, bypassing the large Transformer decoder
- **Flow matching & DDPM**: Both objectives supported, with logit-normal time sampling and time-shift scheduling
- **Multi-view support**: Transfer single-view pretrained weights to 3-camera setups via view-aware 3D RoPE
- **Evaluation suite**: PSNR/SSIM/LPIPS/FID/FVD, PCK (keypoint tracking), controllability (action optimization in latent space), and trajectory success probing

---

## Installation

```bash
uv venv
uv pip install -r requirements.txt
```

---

## Quick Start

### 1. Download Data

```bash
pip install tensorflow tensorflow_datasets
python -m src.data.download_data --dataset_name bridge_v2 --output_dir ./data
```

### 2. Train an Adapter

Required for representation encoders (RAE, ScaleRAE, Qwen, V-JEPA 2.1). Not needed for VAE, Cosmos, or VA-VAE.

```bash
python -m src.launch_adapter \
    --encoder_type scale_rae_webssl \
    --adapter_type svae \
    --adapter_latent_dim 96 \
    --dataset_dir ./data \
    --subset_names bridge_v2 \
    --batch_size 16 \
    --num_epochs 50 \
    --use_pixel_decoder true \
    --stage svae
```

### 3. Train the World Model (DiT)

```bash
# Single GPU
python -m src.launch \
    --encoder_type scale_rae_webssl \
    --adapter_type svae \
    --adapter_checkpoint_path outputs/adapter_svae/adapter_ckpt_50.pt \
    --adapter_latent_dim 96 \
    --dit_size XL \
    --objective flow_matching \
    --dataset_dir ./data \
    --subset_names bridge_v2 \
    --batch_size 8

# Multi-GPU
torchrun --nproc_per_node=4 -m src.launch \
    --encoder_type scale_rae_webssl \
    --adapter_type svae \
    --adapter_checkpoint_path outputs/adapter_svae/adapter_ckpt_50.pt \
    --adapter_latent_dim 96 \
    --dit_size XL \
    --objective flow_matching \
    --dataset_dir ./data \
    --batch_size 8
```

Checkpoints and GIF samples are written to `outputs/<timestamp>/`.

### 4. Evaluate

```bash
python -m src.launch_eval \
    --model_preset DiT-S_WEBSSL_WIDE \
    --dataset_dir ./data \
    --subset_names bridge_v2 \
    --metrics "psnr,ssim,lpips,fvd,pck,controllability"
```

---

## Architecture

### Encoders (`src/models/base_autoencoder.py`)

All encoders inherit from `BaseAutoencoder` and expose a uniform `encode(x)` / `decode(z)` / `latent_dim` API. Instantiate via `create_autoencoder(config)`.

| `encoder_type` | Class | Latent Dim | Backbone |
|---|---|---|---|
| `vae` | `VAE` | 16 | Stable Diffusion 3, frozen |
| `rae` | `RAE` | 768 | DINOv2-Base + ViT-MAE decoder |
| `scale_rae_siglip` | `ScaleRAE` | 1152 | SigLIP2 + ViT-XL decoder |
| `scale_rae_webssl` | `ScaleRAE` | 1024 | WebSSL/DINOv2 + ViT-XL decoder |
| `qwen` | `QwenEncoderWrapper` | 1280 | Qwen2.5-VL-3B (3D temporal) |
| `vjepa2` | `VJEPA2EncoderWrapper` | 1024 | V-JEPA 2.1 ViT-L/16 (image mode) |
| `cosmos` | `CosmosTokenizerWrapper` | 16 | Cosmos CI16x16; no adapter needed |
| `vavae` | `VAVAEWrapper` | 32 | VA-VAE f16d32; no adapter needed |

### Adapters (`src/models/adapters.py`)

Project high-dimensional encoder latents (d_h = 768–1280) down to a compact space (d_l, default 96). The adapter is **always frozen** during DiT training.

| `adapter_type` | Description |
|---|---|
| `identity` | Pass-through (use with VAE, Cosmos, VA-VAE) |
| `mlp` | Two-layer MLP: d_h → hidden → d_l |
| `svae` | Transformer blocks + diagonal Gaussian; optional pixel decoder |

### Diffusion Transformer (`src/models/model.py`)

DiT variants with causal attention across time, action conditioning via concatenation, and spatial/temporal rotary embeddings.

| Size | Hidden | Depth | Heads |
|---|---|---|---|
| S | 384 | 12 | 6 |
| B | 768 | 12 | 12 |
| L | 1024 | 24 | 16 |
| XL | 1152 | 28 | 16 |

### Inference API (`src/models/world_model.py`)

```python
from src.models.world_model import WorldModel

model = WorldModel(checkpoint_path="model.pt")
model.reset(initial_frames)                    # encode and cache history
next_frames = model.generate_chunk(action_vector)  # autoregressive generation
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| PSNR / SSIM / LPIPS / FID / FVD | Standard pixel-level video quality |
| **PCK@k** | Percentage of Correct Keypoints within k pixels (via CoTracker); measures spatial structure preservation |
| **Controllability** | Action optimization error in latent space (CEM/gradient/grid); isolates DiT action-following quality |
| **Probe accuracy** | Trajectory success classifier on frozen features; measures semantic fidelity of generated videos |

---

## Citation

If you use this code or build on the paper, please cite:

```bibtex
@article{nilaksh2026reconstruction,
  title={Reconstruction or Semantics? What Makes a Latent Space Useful for Robotic World Models},
  author={Nilaksh and Saurav Jha and Artem Zholus and Sarath Chandar},
  year={2026},
  eprint={2605.06388},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2605.06388}
}
```
