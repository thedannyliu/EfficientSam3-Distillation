#!/usr/bin/env bash
#SBATCH --job-name=sam3_img_assets
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/logs/assets/sam3_img_assets-%j.out

set -euo pipefail

export REPO_DIR=/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
export RUN_ROOT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke
export NUM_SAMPLES=1120
export SAMPLE_SEED=5090
export NUM_WORKERS="${SLURM_CPUS_PER_TASK:-4}"
export DOWNLOAD_CONCURRENCY=4
export CLEAN_INTERMEDIATE=1
export ASSET_INSTALL_DEPS=0

bash "${REPO_DIR}/scripts/prepare_image_encoder_distill_assets.sh"
