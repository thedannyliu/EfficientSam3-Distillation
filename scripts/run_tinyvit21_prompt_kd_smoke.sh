#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}
SCRATCH_ROOT=${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}
OUTPUT=${OUTPUT:-${SCRATCH_ROOT}/runs/tinyvit21_prompt_kd_smoke}
DEVICE=${DEVICE:-cuda}
USE_WANDB=${USE_WANDB:-0}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT}"

args=(
  --output "${OUTPUT}"
  --backbone tiny_vit_21m
  --img-size 1008
  --embed-dim 1024
  --embed-size 72
  --batch-size "${BATCH_SIZE:-1}"
  --epochs "${EPOCHS:-1}"
  --steps-per-epoch "${STEPS_PER_EPOCH:-1}"
  --device "${DEVICE}"
  --auto-resume
)

if [[ "${USE_WANDB}" == "1" ]]; then
  args+=(--use-wandb --wandb-project "${WANDB_PROJECT:-efficientsam3-prompt-kd}" --wandb-resume "${WANDB_RESUME:-allow}")
fi

python -m stage_prompt_kd.train_smoke "${args[@]}"
