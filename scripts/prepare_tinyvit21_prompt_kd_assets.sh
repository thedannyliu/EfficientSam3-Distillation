#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/pace_py312}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/teacher_cache}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs/assets}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_ROOT}/checkpoints/sam3}"
SAM3_CKPT="${SAM3_CKPT:-${CHECKPOINT_DIR}/sam3.pt}"
SAM3_DOWNLOAD_BACKEND="${SAM3_DOWNLOAD_BACKEND:-hf}"
SAM3_HF_REPO_URL="${SAM3_HF_REPO_URL:-https://huggingface.co/facebook/sam3}"
SAM3_GIT_DIR="${SAM3_GIT_DIR:-${SCRATCH_ROOT}/cache/huggingface_git/facebook_sam3}"

SA1B_ROOT="${SA1B_ROOT:-${DATA_ROOT}/SA-1B-1P}"
SA1B_RAW_DIR="${SA1B_RAW_DIR:-${DATA_ROOT}/sa-1b-1p}"
SA1B_DOWNLOAD_BACKEND="${SA1B_DOWNLOAD_BACKEND:-hf}"
SA1B_HF_REPO="${SA1B_HF_REPO:-ssbai/sa1b}"
DOWNLOAD_CONCURRENCY="${DOWNLOAD_CONCURRENCY:-8}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

DOWNLOAD_SACO="${DOWNLOAD_SACO:-1}"
DOWNLOAD_COCO="${DOWNLOAD_COCO:-1}"
DOWNLOAD_LVIS="${DOWNLOAD_LVIS:-1}"
PREPARE_SA1B="${PREPARE_SA1B:-1}"
EXPORT_STAGE1_EMBEDDINGS="${EXPORT_STAGE1_EMBEDDINGS:-1}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0+cu128}"
TORCHVISION_SPEC="${TORCHVISION_SPEC:-torchvision==0.26.0+cu128}"
SETUPTOOLS_SPEC="${SETUPTOOLS_SPEC:-setuptools==70.2.0}"

SACO_ROOT="${SACO_ROOT:-${DATA_ROOT}/sa-v-text}"
COCO_ROOT="${COCO_ROOT:-${DATA_ROOT}/coco}"
LVIS_ROOT="${LVIS_ROOT:-${DATA_ROOT}/lvis}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${CACHE_ROOT}/stage1_sa1b_1p_sam3}"
TEACHER_EMB="${TEACHER_EMB:-${TEACHER_OUTPUT}/embeddings}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-8}"
GPUS="${GPUS:-1}"

AMBIENT_HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HOME="${DISTILL_HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${SCRATCH_ROOT}/cache/conda_pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"

mkdir -p "${SCRATCH_ROOT}" "${DATA_ROOT}" "${CACHE_ROOT}" "${LOG_DIR}" \
  "${CHECKPOINT_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" \
  "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

if [ ! -f "${HF_HOME}/token" ] && [ -f "${AMBIENT_HF_HOME}/token" ]; then
  cp "${AMBIENT_HF_HOME}/token" "${HF_HOME}/token"
  chmod 600 "${HF_HOME}/token"
fi

LOG_FILE="${LOG_DIR}/prepare_tinyvit21_prompt_kd_assets_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Env: ${ENV_DIR}"
echo "Data root: ${DATA_ROOT}"
echo "Teacher embeddings: ${TEACHER_EMB}"
echo "Stage1 dataset: SA-1B 1% at ${SA1B_ROOT}"

if command -v module >/dev/null 2>&1; then
  module load python/3.12.5 cuda/12.6.1 || true
fi

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  python -m venv "${ENV_DIR}"
fi
PYTHON="${ENV_DIR}/bin/python"
PIP="${ENV_DIR}/bin/pip"
export PATH="${ENV_DIR}/bin:${PATH}"

if [ "${INSTALL_DEPS}" = "1" ]; then
  "${PYTHON}" -m pip install -U pip wheel "${SETUPTOOLS_SPEC}" "huggingface_hub[cli]"
  if ! "${PIP}" install -e "${REPO_DIR}[stage1]" wandb; then
    echo "Full stage1 extra install failed; retrying with cache/export dependency set."
    "${PIP}" install -e "${REPO_DIR}" --no-deps
    "${PIP}" install --index-url "${PYTORCH_INDEX_URL}" --extra-index-url "${PYPI_INDEX_URL}" \
      "${TORCH_SPEC}" "${TORCHVISION_SPEC}"
    "${PIP}" install \
      "timm>=1.0.17" "numpy>=1.26.4" tqdm "ftfy==6.1.1" regex \
      "iopath>=0.1.10" typing_extensions "huggingface_hub[cli]" psutil \
      "decord>=0.6.0" "mmengine>=0.10.4" "pycocotools>=2.0.7" \
      "yacs>=0.1.8" "Pillow>=10.0.0" "opencv-python>=4.9.0.80" \
      "scipy>=1.10.0" "scikit-image>=0.21.0" "scikit-learn>=1.3.0" \
      "tensorboard>=2.12.0" "einops>=0.7.0" "hydra-core>=1.3.2" \
      "submitit>=1.5.1" "fvcore>=0.1.5.post20221221" \
      "fairscale>=0.4.13" pandas pyyaml segment-anything wandb
  fi
fi

"${PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

if [ ! -s "${SAM3_CKPT}" ]; then
  case "${SAM3_DOWNLOAD_BACKEND}" in
    hf)
      "${ENV_DIR}/bin/hf" download facebook/sam3 sam3.pt \
        --local-dir "${CHECKPOINT_DIR}"
      ;;
    git)
      bash "${REPO_DIR}/scripts/download_hf_file_git.sh" \
        "${SAM3_HF_REPO_URL}" sam3.pt "${CHECKPOINT_DIR}" "${SAM3_GIT_DIR}"
      ;;
    *)
      echo "ERROR: unsupported SAM3_DOWNLOAD_BACKEND=${SAM3_DOWNLOAD_BACKEND}; use hf or git." >&2
      exit 1
      ;;
  esac
else
  echo "Using existing SAM3 checkpoint: ${SAM3_CKPT}"
fi

if [ "${PREPARE_SA1B}" = "1" ]; then
  if [ ! -d "${SA1B_ROOT}/images/train" ]; then
    echo "Downloading and reorganizing SA-1B 1% split."
    case "${SA1B_DOWNLOAD_BACKEND}" in
      hf)
        HF_BIN="${ENV_DIR}/bin/hf" SA1B_HF_REPO="${SA1B_HF_REPO}" \
          bash "${REPO_DIR}/data/download_sa1b_hf.sh" \
            "${REPO_DIR}/data/sa-1b-1p.txt" "${SA1B_RAW_DIR}" "${SA1B_HF_REPO}"
        ;;
      hf_git)
        HF_DOWNLOAD_BACKEND=git SA1B_HF_REPO="${SA1B_HF_REPO}" \
          bash "${REPO_DIR}/data/download_sa1b_hf.sh" \
            "${REPO_DIR}/data/sa-1b-1p.txt" "${SA1B_RAW_DIR}" "${SA1B_HF_REPO}"
        ;;
      tsv)
        bash "${REPO_DIR}/data/download_sa1b.sh" \
          "${REPO_DIR}/data/sa-1b-1p.txt" "${SA1B_RAW_DIR}" "${DOWNLOAD_CONCURRENCY}"
        ;;
      *)
        echo "ERROR: unsupported SA1B_DOWNLOAD_BACKEND=${SA1B_DOWNLOAD_BACKEND}" >&2
        exit 1
        ;;
    esac
    "${PYTHON}" "${REPO_DIR}/data/reorg_sa1b.py" \
      --source-dir "${SA1B_RAW_DIR}" \
      --output-dir "${SA1B_ROOT}" \
      --num-workers "${NUM_WORKERS}"
  else
    echo "Using existing SA-1B 1% split: ${SA1B_ROOT}"
  fi
else
  echo "Skipping SA-1B 1% preparation because PREPARE_SA1B=0."
fi

if [ "${DOWNLOAD_SACO}" = "1" ]; then
  mkdir -p "${SACO_ROOT}"
  for repo_name in SACo-Gold SACo-Silver SACo-VEval; do
    local_name="$(printf '%s' "${repo_name}" | tr 'A-Z' 'a-z')"
    out_dir="${SACO_ROOT}/${local_name}"
    if [ -d "${out_dir}" ] && [ "$(find "${out_dir}" -type f | wc -l)" -gt 0 ]; then
      echo "Using existing ${repo_name}: ${out_dir}"
    else
      "${ENV_DIR}/bin/hf" download "facebook/${repo_name}" \
        --repo-type dataset \
        --local-dir "${out_dir}"
    fi
  done
fi

download_and_unzip() {
  local url="$1"
  local dest="$2"
  local filename
  filename="$(basename "${url}")"
  mkdir -p "${dest}"
  if [ ! -s "${dest}/${filename}" ]; then
    wget -nc -O "${dest}/${filename}" "${url}"
  fi
  unzip -n "${dest}/${filename}" -d "${dest}"
}

if [ "${DOWNLOAD_COCO}" = "1" ]; then
  download_and_unzip "http://images.cocodataset.org/zips/train2017.zip" "${COCO_ROOT}/images"
  download_and_unzip "http://images.cocodataset.org/zips/val2017.zip" "${COCO_ROOT}/images"
  download_and_unzip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "${COCO_ROOT}"
fi

if [ "${DOWNLOAD_LVIS}" = "1" ]; then
  download_and_unzip "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip" "${LVIS_ROOT}/annotations"
  download_and_unzip "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" "${LVIS_ROOT}/annotations"
fi

SA1B_COUNT="$(find "${SA1B_ROOT}/images/train" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)"
echo "SA-1B 1% image count: ${SA1B_COUNT}"

if [ "${EXPORT_STAGE1_EMBEDDINGS}" = "1" ]; then
  if [ "${SA1B_COUNT}" -eq 0 ]; then
    echo "ERROR: cannot export Stage1 embeddings without SA-1B images." >&2
    exit 1
  fi
  keys_file="${TEACHER_EMB}/rank0-keys.txt"
  values_file="${TEACHER_EMB}/rank0-values.bin"
  if [ -s "${values_file}" ] && [ -f "${keys_file}" ] && \
     [ "$(wc -l < "${keys_file}")" -eq "${SA1B_COUNT}" ]; then
    echo "Using existing complete Stage1 teacher embeddings: ${TEACHER_EMB}"
  else
    echo "Exporting Stage1 SAM3 teacher embeddings for SA-1B 1%."
    bash "${REPO_DIR}/stage1/scripts/save_image_embeddings.sh" \
      CFG="${REPO_DIR}/stage1/configs/teacher/sam_vit_huge_sa1b.yaml" \
      DATA_PATH="${SA1B_ROOT}" \
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
fi

"${PYTHON}" - "${SCRATCH_ROOT}" "${SA1B_ROOT}" "${TEACHER_EMB}" "${SACO_ROOT}" "${COCO_ROOT}" "${LVIS_ROOT}" <<'PY'
import json, sys
from pathlib import Path
scratch, sa1b, emb, saco, coco, lvis = map(Path, sys.argv[1:])
manifest = {
    "scratch_root": str(scratch),
    "stage1_sa1b_1p": str(sa1b),
    "stage1_teacher_embeddings": str(emb),
    "saco_root": str(saco),
    "coco_root": str(coco),
    "lvis_root": str(lvis),
}
(scratch / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
PY

echo "Asset and Stage1 embedding preparation finished."
