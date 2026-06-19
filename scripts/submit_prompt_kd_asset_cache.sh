#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}"
GPU_TYPE="${GPU_TYPE:-h100}"
PACE_PARTITION="${PACE_PARTITION:-gpu-${GPU_TYPE}}"
PACE_GRES="${PACE_GRES:-gpu:${GPU_TYPE}:1}"
PACE_ACCOUNT="${PACE_ACCOUNT:-gts-agarg35-ideas_l40s}"
PACE_QOS="${PACE_QOS:-embers}"

mkdir -p "${SCRATCH_ROOT}/logs/slurm"

sbatch \
  --job-name=tv21_kd_cache \
  --account="${PACE_ACCOUNT}" \
  --qos="${PACE_QOS}" \
  --partition="${PACE_PARTITION}" \
  --gres="${PACE_GRES}" \
  --cpus-per-task="${CPUS_PER_TASK:-16}" \
  --mem="${MEM:-240G}" \
  --time="${TIME_LIMIT:-48:00:00}" \
  --output="${SCRATCH_ROOT}/logs/slurm/tv21_kd_cache-%j.out" \
  --export=ALL,REPO_DIR="${REPO_DIR}",SCRATCH_ROOT="${SCRATCH_ROOT}",TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-8}",GPUS="${GPUS:-1}" \
  --wrap "bash ${REPO_DIR}/scripts/prepare_tinyvit21_prompt_kd_assets.sh"
