#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-./efficientsam3_distill_runs}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/venv}"

DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
DISTILL_ROOT="${DISTILL_ROOT:-${DATA_ROOT}/SA-1B-1P}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/output}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/sam3_checkpoints}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3.pt}"
SAM3_DOWNLOAD_BACKEND="${SAM3_DOWNLOAD_BACKEND:-git}"
SAM3_HF_REPO_URL="${SAM3_HF_REPO_URL:-https://huggingface.co/facebook/sam3}"
SAM3_GIT_DIR="${SAM3_GIT_DIR:-${RUN_ROOT}/cache/huggingface_git/facebook_sam3}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${OUTPUT_ROOT}/stage1_teacher_sa1b_1p}"
TEACHER_EMB="${TEACHER_EMB:-${TEACHER_OUTPUT}/embeddings}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-1}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-3}"
STUDENT_WARMUP_EPOCHS="${STUDENT_WARMUP_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_NUM_SAMPLES="${DATA_NUM_SAMPLES:--1}"
GPUS="${GPUS:-1}"
LOG_DIR="${DISTILL_LOG_DIR:-${RUN_ROOT}/logs/distill_matrix}"

export HF_HOME="${DISTILL_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

DEFAULT_STUDENT_SPECS="\
es_rv_s:stage1/configs/es_rv_s.yaml:stage1/es_rv_s:efficient_sam3_repvit_s.pt:4 \
es_rv_m:stage1/configs/es_rv_m.yaml:stage1/es_rv_m:efficient_sam3_repvit_m.pt:4 \
es_rv_l:stage1/configs/es_rv_l.yaml:stage1/es_rv_l:efficient_sam3_repvit_l.pt:2 \
es_tv_s:stage1/configs/es_tv_s.yaml:stage1/es_tv_s:efficient_sam3_tinyvit_s.pt:4 \
es_tv_m:stage1/configs/es_tv_m.yaml:stage1/es_tv_m:efficient_sam3_tinyvit_m.pt:4 \
es_tv_l:stage1/configs/es_tv_l.yaml:stage1/es_tv_l:efficient_sam3_tinyvit_l.pt:2 \
es_ev_s:stage1/configs/es_ev_s.yaml:stage1/es_ev_s:efficient_sam3_efficientvit_s.pt:4 \
es_ev_m:stage1/configs/es_ev_m.yaml:stage1/es_ev_m:efficient_sam3_efficientvit_m.pt:4 \
es_ev_l:stage1/configs/es_ev_l.yaml:stage1/es_ev_l:efficient_sam3_efficientvit_l.pt:2 \
es_vit_s:stage1/configs/es_vit_s.yaml:stage1/es_vit_s:efficient_sam3_vit_s.pt:2 \
es_vit_m:stage1/configs/es_vit_m.yaml:stage1/es_vit_m:efficient_sam3_vit_m.pt:1 \
es_vit_l:stage1/configs/es_vit_l.yaml:stage1/es_vit_l:efficient_sam3_vit_l.pt:1"
STUDENT_SPECS="${STUDENT_SPECS:-${DEFAULT_STUDENT_SPECS}}"

mkdir -p "${RUN_ROOT}" "${OUTPUT_ROOT}" "${CHECKPOINT_DIR}" "${LOG_DIR}" \
  "${HF_HOME}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" "${WANDB_DIR}"

LOG_FILE="${LOG_DIR}/run_image_encoder_distill_matrix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python)"
fi
export PATH="${ENV_DIR}/bin:${PATH}"

echo "Log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Distill data: ${DISTILL_ROOT}"
echo "Teacher embeddings: ${TEACHER_EMB}"
echo "Student specs: ${STUDENT_SPECS}"
echo "SAM3 download backend: ${SAM3_DOWNLOAD_BACKEND}"

if [ ! -d "${DISTILL_ROOT}/images/train" ]; then
  bash "${REPO_DIR}/scripts/prepare_sa1b_fixed_splits.sh"
fi

if [ ! -s "${SAM3_CKPT}" ]; then
  case "${SAM3_DOWNLOAD_BACKEND}" in
    git)
      bash "${REPO_DIR}/scripts/download_hf_file_git.sh" \
        "${SAM3_HF_REPO_URL}" sam3.pt "${CHECKPOINT_DIR}" "${SAM3_GIT_DIR}"
      ;;
    hf)
      HF_BIN="${HF_BIN:-${ENV_DIR}/bin/hf}"
      if [ ! -x "${HF_BIN}" ] && command -v hf >/dev/null 2>&1; then
        HF_BIN="$(command -v hf)"
      fi
      if [ ! -x "${HF_BIN}" ]; then
        echo "ERROR: hf CLI not found. Set HF_BIN or use SAM3_DOWNLOAD_BACKEND=git." >&2
        exit 1
      fi
      "${HF_BIN}" download facebook/sam3 sam3.pt --local-dir "${CHECKPOINT_DIR}"
      ;;
    *)
      echo "ERROR: unsupported SAM3_DOWNLOAD_BACKEND=${SAM3_DOWNLOAD_BACKEND}; use git or hf." >&2
      exit 1
      ;;
  esac
fi

TEACHER_KEYS="${TEACHER_EMB}/rank0-keys.txt"
TEACHER_VALUES="${TEACHER_EMB}/rank0-values.bin"
if [ -s "${TEACHER_VALUES}" ] && [ -f "${TEACHER_KEYS}" ]; then
  echo "Using existing teacher image embeddings at ${TEACHER_EMB}"
else
  echo "Exporting teacher image embeddings for fixed SA-1B 1% split"
  bash "${REPO_DIR}/stage1/scripts/save_image_embeddings.sh" \
    CFG="${REPO_DIR}/stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml" \
    DATA_PATH="${DISTILL_ROOT}" \
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

FINAL_EPOCH=$((STUDENT_EPOCHS - 1))
MERGED=()
for spec in ${STUDENT_SPECS}; do
  IFS=':' read -r STUDENT_NAME STUDENT_CFG STUDENT_OUTPUT_SUBDIR MERGED_NAME BATCH_SIZE <<EOF
${spec}
EOF
  STUDENT_OUTPUT="${OUTPUT_ROOT}/${STUDENT_OUTPUT_SUBDIR}"
  MERGED_CKPT="${OUTPUT_ROOT}/${MERGED_NAME}"
  BATCH_SIZE="${BATCH_SIZE:-4}"

  echo "Training ${STUDENT_NAME}"
  bash "${REPO_DIR}/stage1/scripts/train_image_student.sh" \
    CFG="${REPO_DIR}/${STUDENT_CFG}" \
    DATA_PATH="${DISTILL_ROOT}" \
    OUTPUT="${STUDENT_OUTPUT}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    GPUS="${GPUS}" \
    --opts \
      TRAIN.EPOCHS "${STUDENT_EPOCHS}" \
      TRAIN.WARMUP_EPOCHS "${STUDENT_WARMUP_EPOCHS}" \
      TRAIN.AUTO_RESUME False \
      DATA.NUM_SAMPLES "${DATA_NUM_SAMPLES}" \
      DATA.RANDOM_SAMPLE False \
      DATA.NUM_WORKERS "${NUM_WORKERS}" \
      DISTILL.TEACHER_EMBED_PATH "${TEACHER_EMB}"

  echo "Merging ${STUDENT_NAME}"
  "${PYTHON}" "${REPO_DIR}/stage1/convert_image_encoder_weights_stage1.py" \
    --student-ckpt "${STUDENT_OUTPUT}/ckpt_epoch_${FINAL_EPOCH}.pth" \
    --sam3-ckpt "${SAM3_CKPT}" \
    --output "${MERGED_CKPT}"
  MERGED+=("${MERGED_CKPT}")
done

echo "Done. Merged checkpoints:"
printf '  %s\n' "${MERGED[@]}"
