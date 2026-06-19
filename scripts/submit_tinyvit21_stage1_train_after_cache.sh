#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}"
GPU_TYPE="${GPU_TYPE:-h100}"
PACE_PARTITION="${PACE_PARTITION:-gpu-${GPU_TYPE}}"
PACE_GRES="${PACE_GRES:-gpu:${GPU_TYPE}:1}"
PACE_ACCOUNT="${PACE_ACCOUNT:-gts-agarg35}"
PACE_QOS="${PACE_QOS:-embers}"

mkdir -p "${SCRATCH_ROOT}/logs/slurm"

sbatch \
  --job-name=tv21_s1_train \
  --account="${PACE_ACCOUNT}" \
  --qos="${PACE_QOS}" \
  --partition="${PACE_PARTITION}" \
  --gres="${PACE_GRES}" \
  --cpus-per-task="${CPUS_PER_TASK:-8}" \
  --mem="${MEM:-240G}" \
  --time="${TIME_LIMIT:-72:00:00}" \
  --output="${SCRATCH_ROOT}/logs/slurm/tv21_s1_train-%j.out" \
  --export=ALL,REPO_DIR="${REPO_DIR}",SCRATCH_ROOT="${SCRATCH_ROOT}",BATCH_SIZE="${BATCH_SIZE:-32}",GPUS="${GPUS:-1}",USE_WANDB="${USE_WANDB:-1}" \
  --wrap "bash ${REPO_DIR}/scripts/run_tinyvit21_stage1_train_after_cache.sh"
