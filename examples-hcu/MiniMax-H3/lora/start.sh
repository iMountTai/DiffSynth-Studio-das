#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# 在列表第一个 IP 对应的宿主机执行：bash start.sh
# 通过环境变量 HOSTS 或修改下方列表配置多节点 IP；列表顺序对应 machine_rank（从 0 开始）。
HOSTS=(
  ${HOSTS[@]:-10.0.0.1 10.0.0.2}
)
GPUS_PER_NODE=8
MASTER_PORT="${MASTER_PORT:-29500}"
SSH_USER="${SSH_USER:-$(whoami)}"
CONTAINER="${CONTAINER:-diffsynth-hcu}"

NUM_MACHINES=${#HOSTS[@]}
if (( NUM_MACHINES == 0 )); then
  echo "HOSTS must contain at least one IP" >&2
  exit 2
fi
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_ADDR=${HOSTS[0]}

# 容器内共享目录：各节点写不同的日志文件，任一节点均可查看。
PROJECT_DIR="${PROJECT_DIR:-/data/DiffSynth-Studio-das}"
RUN_SCRIPT=${PROJECT_DIR}/examples-hcu/MiniMax-H3/lora/MiniMax-H3-Ref2VA_4nodes.sh
LOG_DIR=${PROJECT_DIR}/models/train/minimax_h3_${NUM_MACHINES}nodes_logs
docker exec "${CONTAINER}" bash -c 'mkdir -p -- "$1" && test -w "$1"' _ "${LOG_DIR}"

echo "MiniMax-H3: ${NUM_MACHINES} nodes, ${NUM_PROCESSES} processes"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Shared container logs: ${LOG_DIR}"
pids=()
for rank in "${!HOSTS[@]}"; do
  node_ip=${HOSTS[$rank]}
  log_file=${LOG_DIR}/node${rank}-${node_ip}.log
  # 参数和日志重定向均在目标容器内执行。
  printf -v train_command 'mkdir -p -- %q && exec env DIFFSYNTH_FA_CPU_CACHE_LAYERS=%q DIFFSYNTH_FA_GPU_CACHE_LAYERS=%q DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS=%q DIFFSYNTH_FA_CACHE_LOG=%q bash %q %q %q %q %q %q > %q 2>&1' \
    "${LOG_DIR}" "${DIFFSYNTH_FA_CPU_CACHE_LAYERS:-0}" "${DIFFSYNTH_FA_GPU_CACHE_LAYERS:-0}" \
    "${DIFFSYNTH_H3_NO_CHECKPOINT_LAST_LAYERS:-0}" "${DIFFSYNTH_FA_CACHE_LOG:-0}" \
    "${RUN_SCRIPT}" "${rank}" "${NUM_MACHINES}" \
    "${NUM_PROCESSES}" "${MASTER_ADDR}" "${MASTER_PORT}" "${log_file}"
  if (( rank == 0 )); then
    docker exec "${CONTAINER}" /bin/bash -ilc "${train_command}" &
  else
    printf -v remote_command 'docker exec %q /bin/bash -ilc %q' "${CONTAINER}" "${train_command}"
    ssh -o BatchMode=yes -o ConnectTimeout=15 "${SSH_USER}@${node_ip}" "${remote_command}" &
  fi
  pids[$rank]=$!
  echo "已提交启动请求：${node_ip} machine_rank=${rank}"
done

# 保持当前会话直到训练结束；汇总退出状态，不自动停止其他节点。
status=0
for rank in "${!pids[@]}"; do
  if wait "${pids[$rank]}"; then
    :
  else
    node_status=$?
    echo "${HOSTS[$rank]} rank=${rank}: failed (exit ${node_status}); see ${LOG_DIR}" >&2
    status=1
  fi
done
exit "${status}"
