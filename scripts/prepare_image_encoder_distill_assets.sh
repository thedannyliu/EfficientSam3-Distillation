#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
RUN_ROOT="${RUN_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_distill_smoke}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/conda_env}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
HF_HOME="${HF_HOME:-${RUN_ROOT}/cache/huggingface}"
DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
RAW_TAR_DIR="${RAW_TAR_DIR:-${DATA_ROOT}/sa-1b-1p}"
REORG_ROOT="${REORG_ROOT:-${DATA_ROOT}/SA-1B-1P}"
SUBSET_ROOT="${SUBSET_ROOT:-${DATA_ROOT}/SA-1B-0.01P}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/sam3_checkpoints}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-1120}"
SAMPLE_SEED="${SAMPLE_SEED:-5090}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
DOWNLOAD_CONCURRENCY="${DOWNLOAD_CONCURRENCY:-4}"
CLEAN_INTERMEDIATE="${CLEAN_INTERMEDIATE:-1}"
ASSET_INSTALL_DEPS="${ASSET_INSTALL_DEPS:-1}"

export CONDA_PKGS_DIRS PIP_CACHE_DIR HF_HOME

mkdir -p "${RUN_ROOT}" "${DATA_ROOT}" "${CHECKPOINT_DIR}" \
  "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${HF_HOME}"

LOG_FILE="${RUN_ROOT}/prepare_assets_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Asset prep log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Data root: ${DATA_ROOT}"
echo "Subset root: ${SUBSET_ROOT}"
echo "SAM3 checkpoint: ${SAM3_CKPT}"
echo "Workers: ${NUM_WORKERS}"
echo "Download concurrency: ${DOWNLOAD_CONCURRENCY}"

if [ ! -x "${ENV_DIR}/bin/python" ] || [ ! -x "${ENV_DIR}/bin/huggingface-cli" ]; then
  echo "Scratch environment is missing or incomplete; running preflight first."
  PREFLIGHT_INSTALL_DEPS="${ASSET_INSTALL_DEPS}" \
    bash "${REPO_DIR}/scripts/preflight_image_encoder_distill.sh"
fi

PYTHON="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ ! -s "${SAM3_CKPT}" ]; then
  echo "Downloading SAM3 checkpoint to ${SAM3_CKPT}"
  huggingface-cli download facebook/sam3 sam3.pt \
    --local-dir "${CHECKPOINT_DIR}" \
    --local-dir-use-symlinks False
else
  echo "Using existing SAM3 checkpoint at ${SAM3_CKPT}"
fi

if [ ! -d "${SUBSET_ROOT}/images/train" ] || \
   [ "$(find "${SUBSET_ROOT}/images/train" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)" -lt "${NUM_SAMPLES}" ]; then
  if [ ! -d "${REORG_ROOT}/images/train" ]; then
    echo "Downloading SA-1B 1% shards to ${RAW_TAR_DIR}"
    bash "${REPO_DIR}/data/download_sa1b.sh" \
      "${REPO_DIR}/data/sa-1b-1p.txt" \
      "${RAW_TAR_DIR}" \
      "${DOWNLOAD_CONCURRENCY}"

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
