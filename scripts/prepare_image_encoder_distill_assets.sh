#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DEFAULT_RUN_ROOT="${REPO_DIR}/efficientsam3_distill_smoke"
RUN_ROOT="${RUN_ROOT:-${DEFAULT_RUN_ROOT}}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/venv}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
AMBIENT_HF_HOME="${HF_HOME:-}"
HF_HOME="${DISTILL_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HF_TOKEN_PATH:-}" ] && \
   [ -n "${AMBIENT_HF_HOME}" ] && [ "${AMBIENT_HF_HOME}" != "${HF_HOME}" ] && \
   [ -f "${AMBIENT_HF_HOME}/token" ]; then
  HF_TOKEN_PATH="${AMBIENT_HF_HOME}/token"
fi
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
RAW_TAR_DIR="${RAW_TAR_DIR:-${DATA_ROOT}/sa-1b-1p}"
REORG_ROOT="${REORG_ROOT:-${DATA_ROOT}/SA-1B-1P}"
SUBSET_ROOT="${SUBSET_ROOT:-${DATA_ROOT}/SA-1B-0.01P}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/sam3_checkpoints}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3.pt}"
SAM3_DOWNLOAD_BACKEND="${SAM3_DOWNLOAD_BACKEND:-git}"
SAM3_HF_REPO_URL="${SAM3_HF_REPO_URL:-https://huggingface.co/facebook/sam3}"
SAM3_GIT_DIR="${SAM3_GIT_DIR:-${RUN_ROOT}/cache/huggingface_git/facebook_sam3}"
NUM_SAMPLES="${NUM_SAMPLES:-1120}"
SAMPLE_SEED="${SAMPLE_SEED:-5090}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
DOWNLOAD_CONCURRENCY="${DOWNLOAD_CONCURRENCY:-4}"
SA1B_DOWNLOAD_BACKEND="${SA1B_DOWNLOAD_BACKEND:-hf_git}"
SA1B_HF_REPO="${SA1B_HF_REPO:-ssbai/sa1b}"
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"
ASSET_INSTALL_DEPS="${ASSET_INSTALL_DEPS:-1}"
LOG_DIR="${ASSET_LOG_DIR:-${RUN_ROOT}/logs/assets}"

export CONDA_PKGS_DIRS PIP_CACHE_DIR HF_HOME HF_TOKEN_PATH

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}" "${CHECKPOINT_DIR}" \
  "${LOG_DIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}"

LOG_FILE="${LOG_DIR}/prepare_assets_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Asset prep log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Subset root: ${SUBSET_ROOT}"
echo "SAM3 checkpoint: ${SAM3_CKPT}"
echo "SAM3 download backend: ${SAM3_DOWNLOAD_BACKEND}"
echo "Workers: ${NUM_WORKERS}"
echo "Download concurrency: ${DOWNLOAD_CONCURRENCY}"
echo "SA-1B download backend: ${SA1B_DOWNLOAD_BACKEND}"

NEED_HF_CLI=0
if [ "${SAM3_DOWNLOAD_BACKEND}" = "hf" ] || [ "${SA1B_DOWNLOAD_BACKEND}" = "hf" ]; then
  NEED_HF_CLI=1
fi

if [ ! -x "${ENV_DIR}/bin/python" ] || \
   { [ "${NEED_HF_CLI}" = "1" ] && [ ! -x "${ENV_DIR}/bin/hf" ]; }; then
  echo "Scratch environment is missing or incomplete; running preflight first."
  PREFLIGHT_INSTALL_DEPS="${ASSET_INSTALL_DEPS}" \
    bash "${REPO_DIR}/scripts/preflight_image_encoder_distill.sh"
fi

PYTHON="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ ! -s "${SAM3_CKPT}" ]; then
  echo "Downloading SAM3 checkpoint to ${SAM3_CKPT}"
  case "${SAM3_DOWNLOAD_BACKEND}" in
    git)
      bash "${REPO_DIR}/scripts/download_hf_file_git.sh" \
        "${SAM3_HF_REPO_URL}" sam3.pt "${CHECKPOINT_DIR}" "${SAM3_GIT_DIR}"
      ;;
    hf)
      "${ENV_DIR}/bin/hf" download facebook/sam3 sam3.pt \
        --local-dir "${CHECKPOINT_DIR}"
      ;;
    *)
      echo "ERROR: unsupported SAM3_DOWNLOAD_BACKEND=${SAM3_DOWNLOAD_BACKEND}; use git or hf." >&2
      exit 1
      ;;
  esac
else
  echo "Using existing SAM3 checkpoint at ${SAM3_CKPT}"
fi

if [ ! -d "${SUBSET_ROOT}/images/train" ] || \
   [ "$(find "${SUBSET_ROOT}/images/train" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)" -lt "${NUM_SAMPLES}" ]; then
  if [ ! -d "${REORG_ROOT}/images/train" ]; then
    echo "Downloading SA-1B 1% shards to ${RAW_TAR_DIR}"
    case "${SA1B_DOWNLOAD_BACKEND}" in
      hf)
        HF_BIN="${ENV_DIR}/bin/hf" SA1B_HF_REPO="${SA1B_HF_REPO}" \
          bash "${REPO_DIR}/data/download_sa1b_hf.sh" \
            "${REPO_DIR}/data/sa-1b-1p.txt" \
            "${RAW_TAR_DIR}" \
            "${SA1B_HF_REPO}"
        ;;
      hf_git)
        HF_DOWNLOAD_BACKEND=git SA1B_HF_REPO="${SA1B_HF_REPO}" \
          bash "${REPO_DIR}/data/download_sa1b_hf.sh" \
            "${REPO_DIR}/data/sa-1b-1p.txt" \
            "${RAW_TAR_DIR}" \
            "${SA1B_HF_REPO}"
        ;;
      tsv)
        bash "${REPO_DIR}/data/download_sa1b.sh" \
          "${REPO_DIR}/data/sa-1b-1p.txt" \
          "${RAW_TAR_DIR}" \
          "${DOWNLOAD_CONCURRENCY}"
        ;;
      *)
        echo "ERROR: unsupported SA1B_DOWNLOAD_BACKEND=${SA1B_DOWNLOAD_BACKEND}; use hf_git, hf, or tsv." >&2
        exit 1
        ;;
    esac

    echo "Reorganizing SA-1B shards under ${DATA_ROOT}"
    (
      cd "${DATA_ROOT}"
      "${PYTHON}" "${REPO_DIR}/data/reorg_sa1b.py" \
        --source-dir "${RAW_TAR_DIR}" \
        --output-dir "${REORG_ROOT}" \
        --num-workers "${NUM_WORKERS}"
    )
  else
    echo "Using existing reorganized SA-1B data at ${REORG_ROOT}"
  fi

  echo "Creating deterministic ${NUM_SAMPLES}-sample SA-1B subset"
  "${PYTHON}" "${REPO_DIR}/data/create_sa1b_subset.py" \
    --source "${REORG_ROOT}" \
    --output "${SUBSET_ROOT}" \
    --num-samples "${NUM_SAMPLES}" \
    --seed "${SAMPLE_SEED}" \
    --mode hardlink

  if [ "${CLEAN_INTERMEDIATE}" = "1" ]; then
    echo "Cleaning intermediate SA-1B 1% raw/reorganized data inside ${DATA_ROOT}"
    rm -rf "${RAW_TAR_DIR}" "${REORG_ROOT}"
  fi
else
  echo "Using existing subset at ${SUBSET_ROOT}"
fi

echo "Prepared assets:"
echo "  SAM3 checkpoint: ${SAM3_CKPT}"
echo "  SA-1B subset: ${SUBSET_ROOT}"
if [ -f "${SUBSET_ROOT}/subset_manifest.json" ]; then
  echo "  Manifest: ${SUBSET_ROOT}/subset_manifest.json"
fi
