#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}"
ENV_DIR="${ENV_DIR:-${SCRATCH_ROOT}/envs/pace_py312}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/data}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/teacher_cache}"
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs/text_teacher}"

SACO_ROOT="${SACO_ROOT:-${DATA_ROOT}/sa-v-text}"
TEXT_DATA_ROOT="${TEXT_DATA_ROOT:-${DATA_ROOT}/saco_text_annotations}"
SAM3_CKPT="${SAM3_CKPT:-${SCRATCH_ROOT}/checkpoints/sam3/sam3.pt}"
TEACHER_OUTPUT="${TEACHER_OUTPUT:-${CACHE_ROOT}/saco_gold_silver_text_teacher_ctx32}"
TEACHER_EMB="${TEACHER_EMB:-${TEACHER_OUTPUT}/embeddings}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}"
GPUS="${GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SCRATCH_ROOT}/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRATCH_ROOT}/cache/xdg}"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/cache/torch}"
export WANDB_DIR="${WANDB_DIR:-${SCRATCH_ROOT}/wandb}"

mkdir -p "${TEXT_DATA_ROOT}" "${TEACHER_OUTPUT}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/saco_text_teacher_cache_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log: ${LOG_FILE}"
echo "SA-Co root: ${SACO_ROOT}"
echo "Text data root: ${TEXT_DATA_ROOT}"
echo "Teacher embeddings: ${TEACHER_EMB}"

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  echo "ERROR: missing env ${ENV_DIR}. Run prepare_tinyvit21_prompt_kd_assets.sh first." >&2
  exit 1
fi
if [ ! -s "${SAM3_CKPT}" ]; then
  echo "ERROR: missing SAM3 checkpoint ${SAM3_CKPT}. Run asset cache first." >&2
  exit 1
fi
export PATH="${ENV_DIR}/bin:${PATH}"

"${ENV_DIR}/bin/python" - "${SACO_ROOT}" "${TEXT_DATA_ROOT}/text_annotations_combined.json" <<'PY'
import json
import sys
from pathlib import Path

saco_root = Path(sys.argv[1])
output = Path(sys.argv[2])
texts = set()
sources = []
for subset in ("saco-gold", "saco-silver"):
    anno_dir = saco_root / subset / "gt-annotations"
    if not anno_dir.exists():
        print(f"warning: missing {anno_dir}")
        continue
    json_files = sorted(anno_dir.glob("*.json"))
    sources.append({"subset": subset, "files": len(json_files)})
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for image in data.get("images", []):
            text = str(image.get("text_input", "")).strip()
            if text:
                texts.add(text)

if not texts:
    raise SystemExit(f"No SA-Co text_input entries found under {saco_root}")

payload = {
    "info": {
        "description": "SA-Co Gold+Silver noun phrases for SAM3 text teacher cache",
        "num_annotations": len(texts),
        "sources": sources,
    },
    "text_annotations": sorted(texts),
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {len(texts)} text annotations to {output}")
PY

keys_file="${TEACHER_EMB}/rank0-keys.txt"
values_file="${TEACHER_EMB}/rank0-values.bin"
text_count="$("${ENV_DIR}/bin/python" - "${TEXT_DATA_ROOT}/text_annotations_combined.json" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1], encoding="utf-8"))["text_annotations"]))
PY
)"

if [ -s "${values_file}" ] && [ -f "${keys_file}" ] && \
   [ "$(wc -l < "${keys_file}")" -eq "${text_count}" ]; then
  echo "Using existing complete text teacher embeddings: ${TEACHER_EMB}"
else
  bash "${REPO_DIR}/stage1/scripts/save_text_embeddings.sh" \
    CFG="${REPO_DIR}/stage1/configs/teacher/sam_text_teacher_sav.yaml" \
    DATA_PATH="${TEXT_DATA_ROOT}" \
    OUTPUT="${TEACHER_OUTPUT}" \
    BATCH_SIZE="${TEXT_BATCH_SIZE}" \
    GPUS="${GPUS}" \
    --opts \
      MODEL.RESUME "${SAM3_CKPT}" \
      DATA.NUM_WORKERS "${NUM_WORKERS}" \
      DISTILL.TEACHER_EMBED_PATH "${TEACHER_EMB}"
fi

echo "SA-Co text teacher embedding cache finished."
