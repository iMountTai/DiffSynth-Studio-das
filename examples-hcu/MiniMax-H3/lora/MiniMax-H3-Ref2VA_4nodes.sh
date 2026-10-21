#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_DIR}/models}"
export DIFFSYNTH_SKIP_DOWNLOAD=True
export HSA_FORCE_FINE_GRAIN_PCIE=1
export WANDB_MODE=offline

# LoRA recompute experiments; all disabled unless overridden before launch.
export DIFFSYNTH_FA_CPU_CACHE_LAYERS="${DIFFSYNTH_FA_CPU_CACHE_LAYERS:-0}"
export DIFFSYNTH_FA_GPU_CACHE_LAYERS="${DIFFSYNTH_FA_GPU_CACHE_LAYERS:-0}"
export DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS="${DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS:-0}"
export DIFFSYNTH_FA_CACHE_LOG="${DIFFSYNTH_FA_CACHE_LOG:-0}"

# Supplied explicitly by start.sh; no node-count or rank inference.
# Usage: bash MiniMax-H3-Ref2VA_4nodes.sh RANK NUM_MACHINES NUM_PROCESSES MASTER_ADDR MASTER_PORT
MACHINE_RANK="${1:?Missing MACHINE_RANK}"
NUM_MACHINES="${2:?Missing NUM_MACHINES}"
NUM_PROCESSES="${3:?Missing NUM_PROCESSES}"
MASTER_ADDR="${4:?Missing MASTER_ADDR}"
MASTER_PORT="${5:?Missing MASTER_PORT}"

CONFIG_FILE=examples-hcu/MiniMax-H3/configs/accelerate_zero3_hcu_4nodes.yaml
CACHE_DIR=./models/train/MiniMax-H3-Ref2VA-hcu-split-cache

cd "${PROJECT_DIR}"
source examples-hcu/MiniMax-H3/configs/env_shca_b075.sh

echo "=================================================="
echo "HOST          = $(hostname)"
echo "MACHINE_RANK  = ${MACHINE_RANK}"
echo "MASTER_ADDR   = ${MASTER_ADDR}"
echo "MASTER_PORT   = ${MASTER_PORT}"
echo "NUM_MACHINES  = ${NUM_MACHINES}"
echo "NUM_PROCESSES = ${NUM_PROCESSES}"
echo "=================================================="

# Stage 1 must already be complete. Share this cache or copy the COMPLETE
# cache to every node at the same path before launching this script.
# This existence check does not verify that the copies are identical.
if [[ ! -d "${CACHE_DIR}" ]] || [[ -z "$(find "${CACHE_DIR}" -name '*.pth' -print -quit)" ]]; then
  echo "Missing preprocessing cache: ${PROJECT_DIR}/${CACHE_DIR#./}" >&2
  echo "Prepare stage 1 first and make the complete cache available on all nodes." >&2
  exit 1
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "accelerate not found; activate the training environment in the container before launching." >&2
  exit 1
fi

# stage 2 (train)
# Matches the working single-node ZeRO-3 settings without CPU offload. For LoRA,
# --deepspeed_zero3_lora_single_param_all_reduce already forces overlap_comm=false
# at runtime on HIP and routes single-parameter LoRA fetches through all-reduce.
exec accelerate launch \
  --config_file "${CONFIG_FILE}" \
  --machine_rank "${MACHINE_RANK}" \
  --num_machines "${NUM_MACHINES}" \
  --num_processes "${NUM_PROCESSES}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  examples/minimax_h3/model_training/train.py \
  --dataset_base_path "${CACHE_DIR}" \
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
  --output_path "./models/train/MiniMax-H3-Ref2VA-hcu-${NUM_PROCESSES}gpu" \
  --lora_base_model "dit" \
  --lora_target_modules "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --deepspeed_zero3_lora_single_param_all_reduce \
  --find_unused_parameters \
  --task "sft:train" \
  --enable_csv_log
