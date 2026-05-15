
# --   adapter using precomputed DinoV3   -- #
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.launch_adapter \
  --h5_train_path /extra_storage/mkim/data/consolidated_train_backbone_labeled_new.h5 \
  --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
  --encoder_type precomputed \
  --h5_embedding_key cam_0_patch_embd \
  --embedding_dim 384 \
  --patch_h 14 \
  --patch_w 14 \
  --precomputed_rgb_target_key camera_0 \
  --adapter_type svae \
  --adapter_latent_dim 96 \
  --use_pixel_decoder True \
  --input_h 224 \
  --input_w 224 \
  --action_dim 7 \
  --batch_size 4 \
  --gradient_accumulation_steps 16 \
  --wandb_mode online \
  --wandb_entity mattkiim-learning \
  --wandb_project_name semantic-wm \
  --log_every_samples 1000 \
  --validate_every_samples 2000 \
  --num_epochs 100 \
  --checkpoint_dir outputs/adapter_dinov3_precomputed_pixel

  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.launch_adapter   --h5_train_path /extra_storage/mkim/data/consolidated_train_backbone_labeled_new.h5   --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5   --encoder_type precomputed   --h5_embedding_key cam_0_patch_embd   --embedding_dim 384   --patch_h 14   --patch_w 14   --precomputed_rgb_target_key camera_0   --adapter_type svae   --adapter_latent_dim 96   --use_pixel_decoder True   --input_h 224   --input_w 224   --action_dim 7   --batch_size 4  --gradient_accumulation_steps 4    --wandb_mode online   --wandb_entity mattkiim-learning   --wandb_project_name semantic-wm   --log_every_samples 1000   --validate_every_samples 1000   --num_epochs 200   --checkpoint_dir outputs/adapter_dinov3_precomputed_pixel_bs4_ga4



# --   DiT on RAE/DINOv2 (RGB only)   -- #
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.launch \
  --h5_train_path /extra_storage/mkim/data/consolidated_train_backbone_labeled_new.h5 \
  --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
  --encoder_type rae \
  --adapter_type svae \
  --adapter_latent_dim 96 \
  --adapter_checkpoint_path outputs/adapter_v2/adapter_ckpt_000000058000.pt \
  --action_dim 7 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --dit_size S \
  --objective flow_matching \
  --wandb_mode online \
  --wandb_entity mattkiim-learning \
  --wandb_project_name semantic-wm \
  --num_epochs 400 \
  --log_every_samples 1000 \
  --validate_every_samples 1000 \
  --save_model True \
  --checkpoint_dir outputs/dit_rae_rgb_v2


# --   DiT on RAE/DINOv2 + tactile   -- #
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.launch \
  --h5_train_path /extra_storage/mkim/data/consolidated_train_backbone_labeled_new.h5 \
  --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
  --encoder_type rae \
  --adapter_type svae \
  --adapter_latent_dim 96 \
  --adapter_checkpoint_path outputs/adapter_v2/adapter_ckpt_000000058000.pt \
  --action_dim 7 \
  --use_tactile True \
  --tactile_dim 512 \
  --h5_tactile_key cam_tactile_cls_embd \
  --tactile_dropout_prob 0.2 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --dit_size S \
  --objective flow_matching \
  --wandb_mode online \
  --wandb_entity mattkiim-learning \
  --wandb_project_name semantic-wm \
  --num_epochs 400 \
  --log_every_samples 1000 \
  --validate_every_samples 1000 \
  --save_model True \
  --checkpoint_dir outputs/dit_rae_tactile_v1


# --   DiT on precomputed DinoV3 (RGB only)   -- #
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.launch \
  --h5_train_path /extra_storage/mkim/data/consolidated_train_backbone_labeled_new.h5 \
  --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
  --encoder_type precomputed \
  --adapter_type svae \
  --adapter_latent_dim 96 \
  --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel/adapter_ckpt_000000057344.pt \
  --action_dim 7 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --dit_size S \
  --objective flow_matching \
  --wandb_mode online \
  --wandb_entity mattkiim-learning \
  --wandb_project_name semantic-wm \
  --num_epochs 400 \
  --log_every_samples 1000 \
  --validate_every_samples 1000 \
  --use_pixel_decoder_for_val True \
  --save_model True \
  --checkpoint_dir outputs/dit_dinov3_precomputed_rgb_v2


# --   DiT on precomputed DinoV3 + tactile   -- #
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.launch \
  --h5_train_path /extra_storage/mkim/data/consolidated_train_backbone_labeled_new.h5 \
  --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
  --encoder_type precomputed \
  --adapter_type svae \
  --adapter_latent_dim 96 \
  --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel_bs4_ga4/adapter_ckpt_000000117936.pt \
  --action_dim 7 \
  --use_tactile True \
  --tactile_dim 512 \
  --h5_tactile_key cam_tactile_cls_embd \
  --tactile_dropout_prob 0.2 \
  --batch_size 16 \
  --gradient_accumulation_steps 1 \
  --dit_size S \
  --objective flow_matching \
  --wandb_mode online \
  --wandb_entity mattkiim-learning \
  --wandb_project_name semantic-wm \
  --num_epochs 400 \
  --log_every_samples 1000 \
  --validate_every_samples 1000 \
  --use_pixel_decoder_for_val True \
  --save_model True \
  --checkpoint_dir outputs/dit_dinov3_precomputed_tactile_v1_do_0.2_cls_new_adapter


# -- eval -- #
CUDA_VISIBLE_DEVICES=0 python -m src.eval_spill \
  --checkpoint_path outputs/dit_dinov3_precomputed_tactile_v1_do_0.2_cls/ckpt_samples_000000235872.pt \
  --adapter_checkpoint_path outputs/adapter_dinov3_precomputed_pixel/adapter_ckpt_000000057344.pt \
  --h5_val_path /extra_storage/mkim/data/consolidated_val_backbone_labeled_new.h5 \
  --encoder_type precomputed \
  --adapter_type svae \
  --adapter_latent_dim 96 \
  --action_dim 7 \
  --use_tactile True \
  --tactile_dim 512 \
  --h5_tactile_key cam_tactile_cls_embd \
  --use_pixel_decoder_for_val True \
  --output_dir eval_outputs/spill_tactile_v1
