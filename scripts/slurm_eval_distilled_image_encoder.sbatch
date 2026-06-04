#!/usr/bin/env bash
#SBATCH --job-name=sam3_img_eval
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/logs/sam3_img_eval-%j.out

set -euo pipefail

export REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
export RUN_ROOT="${RUN_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_distill_smoke}"
export ENV_DIR="${ENV_DIR:-${RUN_ROOT}/conda_env}"

mkdir -p "${RUN_ROOT}/logs"
export PATH="${ENV_DIR}/bin:${PATH}"

"${ENV_DIR}/bin/python" "${REPO_DIR}/tools/eval_distilled_image_encoder.py" \
  --run-root "${RUN_ROOT}" \
  --prepare-coco10 \
  --sizes s m l \
  --prompt-modes text point box \
  "$@"
