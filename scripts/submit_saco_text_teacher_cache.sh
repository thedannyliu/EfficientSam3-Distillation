#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_prompt_kd}"
GPU_TYPE="${GPU_TYPE:-l40s}"
PACE_PARTITION="${PACE_PARTITION:-gpu-${GPU_TYPE}}"
PACE_GRES="${PACE_GRES:-gpu:${GPU_TYPE}:1}"
PACE_ACCOUNT="${PACE_ACCOUNT:-gts-agarg35-ideas_l40s}"
PACE_QOS="${PACE_QOS:-embers}"
JOB_NAME="${JOB_NAME:-tv21_saco_text_cache}"
DEPENDENCY="${DEPENDENCY:-}"

dependency_args=()
if [ -n "${DEPENDENCY}" ]; then
  dependency_args=(--dependency="${DEPENDENCY}")
fi

mkdir -p "${SCRATCH_ROOT}/logs/slurm"

sbatch \
  --job-name="${JOB_NAME}" \
  --account="${PACE_ACCOUNT}" \
  --qos="${PACE_QOS}" \
  --partition="${PACE_PARTITION}" \
  --gres="${PACE_GRES}" \
  --cpus-per-task="${CPUS_PER_TASK:-8}" \
  --mem="${MEM:-128G}" \
  --time="${TIME_LIMIT:-03:00:00}" \
  --output="${SCRATCH_ROOT}/logs/slurm/${JOB_NAME}-%j.out" \
  "${dependency_args[@]}" \
  --export=ALL,REPO_DIR="${REPO_DIR}",SCRATCH_ROOT="${SCRATCH_ROOT}",TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-256}",GPUS="${GPUS:-1}" \
  --wrap "bash ${REPO_DIR}/scripts/run_saco_text_teacher_embedding_cache.sh"
