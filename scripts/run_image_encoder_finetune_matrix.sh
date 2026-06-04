#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-../efficientsam3_distill_runs}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/venv}"

DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
FINETUNE_ROOT="${FINETUNE_ROOT:-${DATA_ROOT}/SA-1B-0.01P-FINETUNE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/sam3_checkpoints}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3.pt}"
TEACHER_OUTPUT="${FINETUNE_TEACHER_OUTPUT:-${OUTPUT_ROOT}/stage1_teacher_sa1b_0.01p_finetune}"
TEACHER_EMB="${FINETUNE_TEACHER_EMB:-${TEACHER_OUTPUT}/embeddings}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
E2E_HEAD_EPOCHS="${E2E_HEAD_EPOCHS:-1}"
RUN_E2E_HEAD_STAGE="${RUN_E2E_HEAD_STAGE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FINETUNE_NUM_SAMPLES="${FINETUNE_NUM_SAMPLES:--1}"
GPUS="${GPUS:-1}"
LOG_DIR="${FINETUNE_LOG_DIR:-${RUN_ROOT}/logs/finetune_matrix}"

export HF_HOME="${DISTILL_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

DEFAULT_FINETUNE_SPECS="\
es_rv_s:stage1_geometry_finetune/configs/es_rv_s.yaml:efficient_sam3_repvit_s.pt:geometry/es_rv_s:efficient_sam3_repvit_s_e2e_ft.pt:2 \
es_rv_m:stage1_geometry_finetune/configs/es_rv_m.yaml:efficient_sam3_repvit_m.pt:geometry/es_rv_m:efficient_sam3_repvit_m_e2e_ft.pt:2 \
es_rv_l:stage1_geometry_finetune/configs/es_rv_l.yaml:efficient_sam3_repvit_l.pt:geometry/es_rv_l:efficient_sam3_repvit_l_e2e_ft.pt:1 \
es_tv_s:stage1_geometry_finetune/configs/es_tv_s.yaml:efficient_sam3_tinyvit_s.pt:geometry/es_tv_s:efficient_sam3_tinyvit_s_e2e_ft.pt:2 \
es_tv_m:stage1_geometry_finetune/configs/es_tv_m.yaml:efficient_sam3_tinyvit_m.pt:geometry/es_tv_m:efficient_sam3_tinyvit_m_e2e_ft.pt:2 \
es_tv_l:stage1_geometry_finetune/configs/es_tv_l.yaml:efficient_sam3_tinyvit_l.pt:geometry/es_tv_l:efficient_sam3_tinyvit_l_e2e_ft.pt:1 \
es_ev_s:stage1_geometry_finetune/configs/es_ev_s.yaml:efficient_sam3_efficientvit_s.pt:geometry/es_ev_s:efficient_sam3_efficientvit_s_e2e_ft.pt:2 \
es_ev_m:stage1_geometry_finetune/configs/es_ev_m.yaml:efficient_sam3_efficientvit_m.pt:geometry/es_ev_m:efficient_sam3_efficientvit_m_e2e_ft.pt:2 \
es_ev_l:stage1_geometry_finetune/configs/es_ev_l.yaml:efficient_sam3_efficientvit_l.pt:geometry/es_ev_l:efficient_sam3_efficientvit_l_e2e_ft.pt:1 \
es_vit_s:stage1_geometry_finetune/configs/es_vit_s.yaml:efficient_sam3_vit_s.pt:geometry/es_vit_s:efficient_sam3_vit_s_e2e_ft.pt:1 \
es_vit_m:stage1_geometry_finetune/configs/es_vit_m.yaml:efficient_sam3_vit_m.pt:geometry/es_vit_m:efficient_sam3_vit_m_e2e_ft.pt:1 \
es_vit_l:stage1_geometry_finetune/configs/es_vit_l.yaml:efficient_sam3_vit_l.pt:geometry/es_vit_l:efficient_sam3_vit_l_e2e_ft.pt:1"
FINETUNE_SPECS="${FINETUNE_SPECS:-${DEFAULT_FINETUNE_SPECS}}"

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" \
  "${HF_HOME}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" "${WANDB_DIR}"

LOG_FILE="${LOG_DIR}/run_image_encoder_finetune_matrix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python)"
fi
export PATH="${ENV_DIR}/bin:${PATH}"

echo "Log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Finetune data: ${FINETUNE_ROOT}"
echo "Teacher embeddings: ${TEACHER_EMB}"
echo "Run E2E head stage: ${RUN_E2E_HEAD_STAGE}"

if [ ! -d "${FINETUNE_ROOT}/images/train" ]; then
  bash "${REPO_DIR}/scripts/prepare_sa1b_fixed_splits.sh"
fi

if [ ! -s "${SAM3_CKPT}" ]; then
  HF_BIN="${HF_BIN:-${ENV_DIR}/bin/hf}"
  if [ ! -x "${HF_BIN}" ] && command -v hf >/dev/null 2>&1; then
    HF_BIN="$(command -v hf)"
  fi
  if [ ! -x "${HF_BIN}" ]; then
    echo "ERROR: hf CLI not found. Set HF_BIN or run preflight first." >&2
    exit 1
  fi
  "${HF_BIN}" download facebook/sam3 sam3.pt --local-dir "${CHECKPOINT_DIR}"
fi

if [ ! -s "${TEACHER_EMB}/rank0-values.bin" ]; then
  echo "Exporting teacher image embeddings for fixed SA-1B finetune split"
  bash "${REPO_DIR}/stage1/scripts/save_image_embeddings.sh" \
    CFG="${REPO_DIR}/stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml" \
    DATA_PATH="${FINETUNE_ROOT}" \
    OUTPUT="${TEACHER_OUTPUT}" \
    BATCH_SIZE="${TEACHER_BATCH_SIZE}" \
    GPUS="${GPUS}" \
    --opts \
      MODEL.RESUME "${SAM3_CKPT}" \
      DATA.NUM_SAMPLES -1 \
      DATA.RANDOM_SAMPLE False \
      DATA.NUM_WORKERS "${NUM_WORKERS}" \
      DISTILL.TEACHER_EMBED_PATH "${TEACHER_EMB}"
fi

FINAL_GEOM_EPOCH=$((FINETUNE_EPOCHS - 1))
FINAL_E2E_EPOCH=$((E2E_HEAD_EPOCHS - 1))
MERGED=()

for spec in ${FINETUNE_SPECS}; do
  IFS=':' read -r NAME CFG PRETRAINED_NAME OUTPUT_SUBDIR FINAL_NAME BATCH_SIZE <<EOF
${spec}
EOF
  PRETRAINED_CKPT="${OUTPUT_ROOT}/${PRETRAINED_NAME}"
  GEOM_OUTPUT="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/image_encoder"
  GEOM_MERGED="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/${NAME}_geometry_ft.pt"
  FINAL_OUTPUT="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}/e2e_heads"
  FINAL_CKPT="${OUTPUT_ROOT}/${FINAL_NAME}"
  BATCH_SIZE="${BATCH_SIZE:-2}"

  if [ ! -s "${PRETRAINED_CKPT}" ]; then
    echo "ERROR: missing distilled checkpoint for ${NAME}: ${PRETRAINED_CKPT}" >&2
    exit 1
  fi

  echo "Geometry fine-tuning ${NAME}"
  bash "${REPO_DIR}/stage1_geometry_finetune/scripts/train_geometry_finetune.sh" \
    CFG="${REPO_DIR}/${CFG}" \
    DATA_PATH="${FINETUNE_ROOT}" \
    OUTPUT="${GEOM_OUTPUT}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    GPUS="${GPUS}" \
    --pretrained "${PRETRAINED_CKPT}" \
    --sam3-checkpoint "${SAM3_CKPT}" \
    --teacher-embed-path "${TEACHER_EMB}" \
    --opts \
      TRAIN.EPOCHS "${FINETUNE_EPOCHS}" \
      TRAIN.WARMUP_EPOCHS 1 \
      TRAIN.AUTO_RESUME False \
      DATA.NUM_SAMPLES "${FINETUNE_NUM_SAMPLES}" \
      DATA.NUM_WORKERS "${NUM_WORKERS}"

  "${PYTHON}" "${REPO_DIR}/stage1_geometry_finetune/convert_geometry_finetune.py" \
    --finetune-ckpt "${GEOM_OUTPUT}/ckpt_epoch_${FINAL_GEOM_EPOCH}.pth" \
    --pretrained "${PRETRAINED_CKPT}" \
    --output "${GEOM_MERGED}"

  if [ "${RUN_E2E_HEAD_STAGE}" = "1" ]; then
    echo "Conservative E2E head fine-tuning ${NAME}"
    bash "${REPO_DIR}/stage1_geometry_finetune/scripts/train_geometry_finetune.sh" \
      CFG="${REPO_DIR}/${CFG}" \
      DATA_PATH="${FINETUNE_ROOT}" \
      OUTPUT="${FINAL_OUTPUT}" \
      BATCH_SIZE="${BATCH_SIZE}" \
      GPUS="${GPUS}" \
      --pretrained "${GEOM_MERGED}" \
      --sam3-checkpoint "${SAM3_CKPT}" \
      --teacher-embed-path "${TEACHER_EMB}" \
      --unfreeze-fpn \
      --unfreeze-geometry-encoder \
      --unfreeze-segmentation-head \
      --opts \
        TRAIN.EPOCHS "${E2E_HEAD_EPOCHS}" \
        TRAIN.WARMUP_EPOCHS 0 \
        TRAIN.BASE_LR 1e-5 \
        TRAIN.MIN_LR 1e-7 \
        TRAIN.AUTO_RESUME False \
        DATA.NUM_SAMPLES "${FINETUNE_NUM_SAMPLES}" \
        DATA.NUM_WORKERS "${NUM_WORKERS}"

    "${PYTHON}" "${REPO_DIR}/stage1_geometry_finetune/convert_geometry_finetune.py" \
      --finetune-ckpt "${FINAL_OUTPUT}/ckpt_epoch_${FINAL_E2E_EPOCH}.pth" \
      --pretrained "${GEOM_MERGED}" \
      --output "${FINAL_CKPT}" \
      --include-e2e-heads
  else
    cp "${GEOM_MERGED}" "${FINAL_CKPT}"
  fi
  MERGED+=("${FINAL_CKPT}")
done

echo "Done. Fine-tuned checkpoints:"
printf '  %s\n' "${MERGED[@]}"
