#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:-./efficientsam3_distill_runs}"
ENV_DIR="${ENV_DIR:-${RUN_ROOT}/venv}"

DATA_ROOT="${DATA_ROOT:-${RUN_ROOT}/data}"
DISTILL_RAW_DIR="${DISTILL_RAW_DIR:-${DATA_ROOT}/sa-1b-1p}"
DISTILL_ROOT="${DISTILL_ROOT:-${DATA_ROOT}/SA-1B-1P}"
FINETUNE_RAW_DIR="${FINETUNE_RAW_DIR:-${DATA_ROOT}/sa-1b-finetune-raw}"
FINETUNE_REORG_ROOT="${FINETUNE_REORG_ROOT:-${DATA_ROOT}/SA-1B-FINETUNE-POOL}"
FINETUNE_ROOT="${FINETUNE_ROOT:-${DATA_ROOT}/SA-1B-0.01P-FINETUNE}"

DISTILL_TSV="${DISTILL_TSV:-${REPO_DIR}/data/sa-1b-1p.txt}"
FULL_TSV="${FULL_TSV:-${REPO_DIR}/data/sa-1b.txt}"
SA1B_DOWNLOAD_BACKEND="${SA1B_DOWNLOAD_BACKEND:-hf}"
SA1B_HF_REPO="${SA1B_HF_REPO:-ssbai/sa1b}"
DOWNLOAD_CONCURRENCY="${DOWNLOAD_CONCURRENCY:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FINETUNE_NUM_SAMPLES="${FINETUNE_NUM_SAMPLES:-1120}"
FINETUNE_SEED="${FINETUNE_SEED:-5091}"
FINETUNE_NUM_SHARDS="${FINETUNE_NUM_SHARDS:-1}"
LINK_MODE="${LINK_MODE:-hardlink}"
LOG_DIR="${SA1B_SPLIT_LOG_DIR:-${RUN_ROOT}/logs/data}"

export HF_HOME="${DISTILL_HF_HOME:-${RUN_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/cache/pip}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-${RUN_ROOT}/conda_pkgs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RUN_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${RUN_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

mkdir -p "${DATA_ROOT}" "${LOG_DIR}" "${HF_HOME}" "${PIP_CACHE_DIR}" \
  "${CONDA_PKGS_DIRS}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

LOG_FILE="${LOG_DIR}/prepare_sa1b_fixed_splits_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

PYTHON="${PYTHON:-${ENV_DIR}/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  PYTHON="$(command -v python)"
fi
HF_BIN="${HF_BIN:-${ENV_DIR}/bin/hf}"
if [ ! -x "${HF_BIN}" ] && command -v hf >/dev/null 2>&1; then
  HF_BIN="$(command -v hf)"
fi

download_sa1b() {
  local tsv="$1"
  local out_dir="$2"
  mkdir -p "${out_dir}"
  case "${SA1B_DOWNLOAD_BACKEND}" in
    hf)
      if [ ! -x "${HF_BIN}" ]; then
        echo "ERROR: hf CLI not found. Set HF_BIN or run preflight first." >&2
        exit 1
      fi
      HF_BIN="${HF_BIN}" SA1B_HF_REPO="${SA1B_HF_REPO}" \
        bash "${REPO_DIR}/data/download_sa1b_hf.sh" "${tsv}" "${out_dir}" "${SA1B_HF_REPO}"
      ;;
    tsv)
      bash "${REPO_DIR}/data/download_sa1b.sh" "${tsv}" "${out_dir}" "${DOWNLOAD_CONCURRENCY}"
      ;;
    *)
      echo "ERROR: unsupported SA1B_DOWNLOAD_BACKEND=${SA1B_DOWNLOAD_BACKEND}" >&2
      exit 1
      ;;
  esac
}

echo "Log: ${LOG_FILE}"
echo "Repo: ${REPO_DIR}"
echo "Run root: ${RUN_ROOT}"
echo "Distill root: ${DISTILL_ROOT}"
echo "Finetune root: ${FINETUNE_ROOT}"

if [ ! -d "${DISTILL_ROOT}/images/train" ]; then
  echo "Preparing fixed SA-1B 1% distillation split"
  download_sa1b "${DISTILL_TSV}" "${DISTILL_RAW_DIR}"
  "${PYTHON}" "${REPO_DIR}/data/reorg_sa1b.py" \
    --source-dir "${DISTILL_RAW_DIR}" \
    --output-dir "${DISTILL_ROOT}" \
    --num-workers "${NUM_WORKERS}"
else
  echo "Using existing distillation split: ${DISTILL_ROOT}"
fi

"${PYTHON}" - "${DISTILL_ROOT}" "${DISTILL_TSV}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
tsv = Path(sys.argv[2])
keys = sorted(p.stem for p in (root / "images" / "train").glob("*.jpg"))
manifest = {
    "split": "distill",
    "source_tsv": str(tsv),
    "actual_num_samples": len(keys),
    "keys": keys,
}
(root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Distill keys: {len(keys)}")
PY

FINETUNE_TSV="${DATA_ROOT}/sa-1b-finetune-shards.tsv"
if [ ! -s "${FINETUNE_TSV}" ]; then
  echo "Selecting finetune shard list disjoint from ${DISTILL_TSV}"
  "${PYTHON}" - "${FULL_TSV}" "${DISTILL_TSV}" "${FINETUNE_TSV}" "${FINETUNE_NUM_SHARDS}" "${FINETUNE_SEED}" <<'PY'
import random, sys
from pathlib import Path
full_tsv, distill_tsv, out_tsv = map(Path, sys.argv[1:4])
n = int(sys.argv[4])
seed = int(sys.argv[5])
full = [line for line in full_tsv.read_text().splitlines() if line.strip()]
distill_names = {line.split("\t", 1)[0] for line in distill_tsv.read_text().splitlines() if line.strip()}
candidates = [line for line in full if line.split("\t", 1)[0] not in distill_names]
if not candidates:
    raise SystemExit("No SA-1B shards remain after excluding distillation TSV")
rng = random.Random(seed)
selected = rng.sample(candidates, min(n, len(candidates)))
out_tsv.write_text("\n".join(selected) + "\n")
print(f"Selected {len(selected)} finetune shard(s): {out_tsv}")
PY
fi

if [ ! -d "${FINETUNE_REORG_ROOT}/images/train" ]; then
  echo "Preparing disjoint SA-1B finetune pool"
  download_sa1b "${FINETUNE_TSV}" "${FINETUNE_RAW_DIR}"
  "${PYTHON}" "${REPO_DIR}/data/reorg_sa1b.py" \
    --source-dir "${FINETUNE_RAW_DIR}" \
    --output-dir "${FINETUNE_REORG_ROOT}" \
    --num-workers "${NUM_WORKERS}"
else
  echo "Using existing finetune pool: ${FINETUNE_REORG_ROOT}"
fi

if [ ! -d "${FINETUNE_ROOT}/images/train" ] || \
   [ "$(find "${FINETUNE_ROOT}/images/train" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)" -lt "${FINETUNE_NUM_SAMPLES}" ]; then
  echo "Creating deterministic disjoint finetune subset"
  "${PYTHON}" "${REPO_DIR}/data/create_sa1b_subset.py" \
    --source "${FINETUNE_REORG_ROOT}" \
    --output "${FINETUNE_ROOT}" \
    --num-samples "${FINETUNE_NUM_SAMPLES}" \
    --seed "${FINETUNE_SEED}" \
    --mode "${LINK_MODE}"
else
  echo "Using existing finetune subset: ${FINETUNE_ROOT}"
fi

"${PYTHON}" - "${DISTILL_ROOT}" "${FINETUNE_ROOT}" "${FINETUNE_TSV}" <<'PY'
import json, sys
from pathlib import Path
distill_root, finetune_root, finetune_tsv = map(Path, sys.argv[1:4])
distill_keys = set(json.loads((distill_root / "split_manifest.json").read_text())["keys"])
subset_manifest = json.loads((finetune_root / "subset_manifest.json").read_text())
finetune_keys = set(subset_manifest["keys"])
overlap = sorted(distill_keys & finetune_keys)
manifest = dict(subset_manifest)
manifest.update({
    "split": "finetune",
    "source_tsv": str(finetune_tsv),
    "disjoint_from": str(distill_root / "split_manifest.json"),
    "overlap_with_distill": overlap,
})
(finetune_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if overlap:
    raise SystemExit(f"Finetune split overlaps distill split: {len(overlap)} keys")
print(f"Finetune keys: {len(finetune_keys)}")
print("SA-1B fixed splits are ready.")
PY
