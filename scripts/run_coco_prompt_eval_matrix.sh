#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-../efficientsam3_distill_runs}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/venv}"

COCO_ROOT="${COCO_ROOT:-${RUN_ROOT}/data/coco}"
COCO_SPLIT="${COCO_SPLIT:-val2017}"
MODEL_SET="${MODEL_SET:-distilled}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/output}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/eval/coco_prompts}"
PROMPT_MODES="${PROMPT_MODES:-point box text}"
MODELS="${MODELS:-rv_s rv_m rv_l tv_s tv_m tv_l ev_s ev_m ev_l vit_s vit_m vit_l}"
NUM_IMAGES="${NUM_IMAGES:--1}"
LOG_DIR="${COCO_EVAL_LOG_DIR:-${RUN_ROOT}/logs/coco_eval}"

case "${MODEL_SET}" in
  distilled) CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-}" ;;
  e2e_ft) CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-_e2e_ft}" ;;
  *) echo "ERROR: MODEL_SET must be distilled or e2e_ft" >&2; exit 1 ;;
esac

export HF_HOME="${DISTILL_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

LOG_FILE="${LOG_DIR}/run_coco_prompt_eval_matrix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python)"
fi

MANIFEST="${MANIFEST:-${RUN_ROOT}/data/manifests/coco_${COCO_SPLIT}_prompts.jsonl}"
if [ ! -s "${MANIFEST}" ]; then
  if [ "${COCO_SPLIT}" = "test2017" ] && [ -z "${ALLOW_TEST_SCAFFOLD:-}" ]; then
    echo "ERROR: test2017 requires a prompt manifest with text/point/box prompts." >&2
    echo "Set MANIFEST=... or ALLOW_TEST_SCAFFOLD=1 for image-only scaffold rows." >&2
    exit 1
  fi
  BUILD_ARGS=(
    --coco-root "${COCO_ROOT}"
    --split "${COCO_SPLIT}"
    --output "${MANIFEST}"
  )
  if [ "${NUM_IMAGES}" != "-1" ]; then
    BUILD_ARGS+=(--max-images "${NUM_IMAGES}")
  fi
  "${PYTHON}" "${REPO_DIR}/tools/build_coco_prompt_manifest.py" "${BUILD_ARGS[@]}"
fi

EVAL_ARGS=(
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --checkpoint-suffix "${CHECKPOINT_SUFFIX}"
  --manifest "${MANIFEST}"
  --coco-root "${COCO_ROOT}"
  --split "${COCO_SPLIT}"
  --output-dir "${OUTPUT_DIR}"
  --models ${MODELS}
  --prompt-modes ${PROMPT_MODES}
)
if [ "${NUM_IMAGES}" != "-1" ]; then
  EVAL_ARGS+=(--max-rows "${NUM_IMAGES}")
fi

"${PYTHON}" "${REPO_DIR}/tools/eval_efficientsam3_coco_prompts.py" "${EVAL_ARGS[@]}" "$@"
