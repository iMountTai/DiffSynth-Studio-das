#!/usr/bin/env bash
# Modified by Hygon Information Technology Co., Ltd., 2026.
# MiniMax-H3 Ref2VA LoRA finetune on HCU (8 cards, ZeRO-3).
set -euo pipefail
# Based on examples/minimax_h3/model_training/lora/MiniMax-H3-Ref2VA.sh.

# Weights are read from disk instead of being downloaded. DIFFSYNTH_MODEL_BASE_PATH
# replaces the default "./models" root, so the paths below resolve to
#   path/to/models/MiniMax/MiniMax-H3/Ref2VA/...
# Adjust this to wherever the weights actually live on your node.
export DIFFSYNTH_MODEL_BASE_PATH=path/to/model
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HSA_FORCE_FINE_GRAIN_PCIE=1
export WANDB_MODE=offline

# LoRA recompute experiments; all disabled unless overridden before launch.
export DIFFSYNTH_FA_CPU_CACHE_LAYERS="${DIFFSYNTH_FA_CPU_CACHE_LAYERS:-0}"
export DIFFSYNTH_FA_GPU_CACHE_LAYERS="${DIFFSYNTH_FA_GPU_CACHE_LAYERS:-0}"
export DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS="${DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS:-0}"
export DIFFSYNTH_FA_CACHE_LOG="${DIFFSYNTH_FA_CACHE_LOG:-0}"

modelscope download --dataset DiffSynth-Studio/diffsynth_example_dataset --include "minimax_h3/MiniMax-H3-Ref2VA/*" --local_dir ./data/diffsynth_example_dataset

# stage 1 (data process)
# No --mixed_precision here, matching upstream: the pipeline is loaded with an
# explicit torch_dtype=bfloat16, so Accelerate's mixed_precision setting does not
# affect compute dtype. Accelerate warns that it defaulted to 'no'; that warning
# is cosmetic for this codebase.
accelerate launch examples/minimax_h3/model_training/train.py \
  --dataset_base_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA \
  --dataset_metadata_path data/diffsynth_example_dataset/minimax_h3/MiniMax-H3-Ref2VA/metadata.json \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 1 \
  --seed 42 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/text_encoder/model*.safetensors,MiniMax/MiniMax-H3:Ref2VA/video_vae/source/model.safetensors,MiniMax/MiniMax-H3:Ref2VA/audio_vae/model.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-hcu-split-cache" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --task "sft:data_process"

# stage 2 (train)
# Uses the HCU ZeRO-3 config with overlap_comm=false.
accelerate launch --config_file examples-hcu/MiniMax-H3/configs/accelerate_zero3_hcu.yaml \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path ./models/train/MiniMax-H3-Ref2VA-hcu-split-cache \
  --data_file_keys "video,input_audio,references" \
  --extra_inputs "input_audio,references" \
  --height 480 \
  --width 832 \
  --num_frames 124 \
  --dataset_repeat 100 \
  --seed 42 \
  --model_id_with_origin_paths "MiniMax/MiniMax-H3:Ref2VA/transformer/model*.safetensors" \
  --processor_path "MiniMax/MiniMax-H3:Ref2VA/processor/" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/MiniMax-H3-Ref2VA-hcu" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --find_unused_parameters \
  --enable_compile \
  --compile_mode default \
  --enable_csv_log \
  --task "sft:train"
