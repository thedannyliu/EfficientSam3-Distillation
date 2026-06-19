#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/pace_py312}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/teacher_cache}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/runs/tinyvit21_stage1_sa1b_1p}"

SA1B_ROOT="${SA1B_ROOT:-${DATA_ROOT}/SA-1B-1P}"
TEACHER_EMB="${TEACHER_EMB:-${CACHE_ROOT}/stage1_sa1b_1p_sam3/embeddings}"
SAM3_CKPT="${SAM3_CKPT:-${SCRATCH_ROOT}/checkpoints/sam3/sam3.pt}"
OUTPUT="${OUTPUT:-${RUN_ROOT}/train}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPUS="${GPUS:-1}"
EPOCHS="${EPOCHS:-50}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-efficientsam3-stage1-tinyvit21}"

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"

mkdir -p "${OUTPUT}" "${SCRATCH_ROOT}/logs/stage1"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ ! -d "${SA1B_ROOT}/images/train" ]; then
  echo "ERROR: missing SA-1B 1% root: ${SA1B_ROOT}" >&2
  echo "Run scripts/prepare_tinyvit21_prompt_kd_assets.sh first." >&2
  exit 1
fi
if [ ! -s "${TEACHER_EMB}/rank0-values.bin" ]; then
  echo "ERROR: missing teacher embeddings: ${TEACHER_EMB}" >&2
  echo "Run scripts/prepare_tinyvit21_prompt_kd_assets.sh first." >&2
  exit 1
fi

extra_args=()
if [ "${USE_WANDB}" = "1" ]; then
  extra_args+=(--use-wandb --wandb-project "${WANDB_PROJECT}" --wandb-resume "${WANDB_RESUME:-allow}")
fi

bash "${REPO_DIR}/stage1/scripts/train_image_student.sh" \
  CFG="${REPO_DIR}/stage1/configs/es_tv_l.yaml" \
  DATA_PATH="${SA1B_ROOT}" \
  OUTPUT="${OUTPUT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GPUS="${GPUS}" \
  "${extra_args[@]}" \
  --opts \
    TRAIN.EPOCHS "${EPOCHS}" \
    TRAIN.WARMUP_EPOCHS "${WARMUP_EPOCHS}" \
    TRAIN.AUTO_RESUME True \
    DATA.NUM_SAMPLES -1 \
    DATA.RANDOM_SAMPLE False \
    DATA.NUM_WORKERS "${NUM_WORKERS}" \
    DISTILL.TEACHER_EMBED_PATH "${TEACHER_EMB}"

final_ckpt="${OUTPUT}/ckpt_epoch_$((EPOCHS - 1)).pth"
if [ -s "${final_ckpt}" ]; then
  "${ENV_DIR}/bin/python" "${REPO_DIR}/stage1/convert_image_encoder_weights_stage1.py" \
    --student-ckpt "${final_ckpt}" \
    --sam3-ckpt "${SAM3_CKPT}" \
    --output "${RUN_ROOT}/efficient_sam3_tinyvit21_stage1_final.pt"
fi
