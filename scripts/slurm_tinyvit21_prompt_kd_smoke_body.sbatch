#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR}"
module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate

python -m pip install -e ".[stage1]"

bash scripts/run_tinyvit21_shape_check.sh
bash scripts/run_tinyvit21_prompt_kd_smoke.sh
