#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}
SCRATCH_ROOT=${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}
SAM3_CHECKPOINT=${SAM3_CHECKPOINT:-}
DEVICE=${DEVICE:-cuda}
RUN_FORWARD=${RUN_FORWARD:-1}
RUN_TEACHER=${RUN_TEACHER:-0}

cd "${REPO_DIR}"
mkdir -p "${SCRATCH_ROOT}/shape_audit"

args=(
  --backbone tiny_vit_21m
  --img-size 1008
  --embed-dim 1024
  --embed-size 72
  --device "${DEVICE}"
  --output-json "${SCRATCH_ROOT}/shape_audit/tinyvit21_sam3_shape.json"
)

if [[ "${RUN_FORWARD}" == "1" ]]; then
  args+=(--run-forward)
fi
if [[ "${RUN_TEACHER}" == "1" ]]; then
  args+=(--run-teacher --sam3-checkpoint "${SAM3_CHECKPOINT}")
fi

python -m stage_prompt_kd.shape_check "${args[@]}"
