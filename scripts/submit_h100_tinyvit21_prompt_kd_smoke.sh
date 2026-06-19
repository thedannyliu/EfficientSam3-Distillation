#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}
SCRATCH_ROOT=${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}
PACE_ACCOUNT=${PACE_ACCOUNT:-gts-agarg35-ideas_l40s}
PACE_QOS=${PACE_QOS:-embers}
PACE_PARTITION=${PACE_PARTITION:-gpu-h100}
PACE_GRES=${PACE_GRES:-gpu:h100:1}

mkdir -p "${SCRATCH_ROOT}/logs/slurm"

sbatch \
  --job-name=tv21_prompt_kd \
  --account="${PACE_ACCOUNT}" \
  --qos="${PACE_QOS}" \
  --partition="${PACE_PARTITION}" \
  --gres="${PACE_GRES}" \
  --cpus-per-task="${CPUS_PER_TASK:-8}" \
  --mem="${MEM:-160G}" \
  --time="${TIME_LIMIT:-08:00:00}" \
  --output="${SCRATCH_ROOT}/logs/slurm/tv21_prompt_kd-%j.out" \
  --export=ALL,REPO_DIR="${REPO_DIR}",SCRATCH_ROOT="${SCRATCH_ROOT}",USE_WANDB="${USE_WANDB:-1}",WANDB_PROJECT="${WANDB_PROJECT:-efficientsam3-prompt-kd}" \
  "${REPO_DIR}/scripts/slurm_tinyvit21_prompt_kd_smoke_body.sbatch"
