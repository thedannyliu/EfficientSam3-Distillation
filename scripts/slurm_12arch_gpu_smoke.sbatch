#!/usr/bin/env bash
#SBATCH --job-name=sam3_12arch_smoke
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke/logs/slurm/sam3_12arch_smoke-%j.out

set -euo pipefail

export REPO_DIR="${REPO_DIR:-/storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation}"
export RUN_ROOT="${RUN_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke}"
export ENV_DIR="${ENV_DIR:-/storage/scratch1/9/eliu354/efficientsam3_distill_smoke/conda_env}"
export SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-/storage/scratch1/9/eliu354/efficientsam3_distill_smoke}"

export HF_HOME="${RUN_ROOT}/cache/huggingface"
export PIP_CACHE_DIR="${RUN_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${RUN_ROOT}/conda_pkgs"
export XDG_CACHE_HOME="${RUN_ROOT}/cache/xdg"
export TORCH_HOME="${RUN_ROOT}/cache/torch"
export WANDB_DIR="${RUN_ROOT}/wandb"
export PATH="${ENV_DIR}/bin:${PATH}"

mkdir -p "${RUN_ROOT}/logs/slurm" "${RUN_ROOT}/data/manifests" \
  "${HF_HOME}" "${PIP_CACHE_DIR}" "${CONDA_PKGS_DIRS}" "${XDG_CACHE_HOME}" \
  "${TORCH_HOME}" "${WANDB_DIR}"

cd "${REPO_DIR}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this Slurm job")
print("device", torch.cuda.get_device_name(0))
PY

export DISTILL_ROOT="${SOURCE_RUN_ROOT}/data/SA-1B-0.01P"
export TEACHER_EMB="${SOURCE_RUN_ROOT}/output/stage1_teacher/embeddings"
export SAM3_CKPT="${SOURCE_RUN_ROOT}/sam3_checkpoints/sam3.pt"
export OUTPUT_ROOT="${RUN_ROOT}/output"
export CHECKPOINT_DIR="${RUN_ROOT}/sam3_checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
ln -sf "${SAM3_CKPT}" "${CHECKPOINT_DIR}/sam3.pt"

export STUDENT_EPOCHS=1
export STUDENT_WARMUP_EPOCHS=0
export DATA_NUM_SAMPLES=4
export NUM_WORKERS=4
export GPUS=1
export STUDENT_SPECS="\
es_rv_s:stage1/configs/es_rv_s.yaml:stage1/es_rv_s:efficient_sam3_repvit_s.pt:1 \
es_rv_m:stage1/configs/es_rv_m.yaml:stage1/es_rv_m:efficient_sam3_repvit_m.pt:1 \
es_rv_l:stage1/configs/es_rv_l.yaml:stage1/es_rv_l:efficient_sam3_repvit_l.pt:1 \
es_tv_s:stage1/configs/es_tv_s.yaml:stage1/es_tv_s:efficient_sam3_tinyvit_s.pt:1 \
es_tv_m:stage1/configs/es_tv_m.yaml:stage1/es_tv_m:efficient_sam3_tinyvit_m.pt:1 \
es_tv_l:stage1/configs/es_tv_l.yaml:stage1/es_tv_l:efficient_sam3_tinyvit_l.pt:1 \
es_ev_s:stage1/configs/es_ev_s.yaml:stage1/es_ev_s:efficient_sam3_efficientvit_s.pt:1 \
es_ev_m:stage1/configs/es_ev_m.yaml:stage1/es_ev_m:efficient_sam3_efficientvit_m.pt:1 \
es_ev_l:stage1/configs/es_ev_l.yaml:stage1/es_ev_l:efficient_sam3_efficientvit_l.pt:1 \
es_vit_s:stage1/configs/es_vit_s.yaml:stage1/es_vit_s:efficient_sam3_vit_s.pt:1 \
es_vit_m:stage1/configs/es_vit_m.yaml:stage1/es_vit_m:efficient_sam3_vit_m.pt:1 \
es_vit_l:stage1/configs/es_vit_l.yaml:stage1/es_vit_l:efficient_sam3_vit_l.pt:1"

bash scripts/run_image_encoder_distill_matrix.sh

cat > "${RUN_ROOT}/data/manifests/truck_prompt_smoke.jsonl" <<'JSONL'
{"sample_id":"truck_smoke","split":"asset","image_id":"truck","file_name":"truck.jpg","image_path":"sam3/assets/images/truck.jpg","height":640,"width":960,"text_prompt":"truck","box_xyxy":[250,180,760,520],"point":[500,360],"point_label":1}
JSONL

python tools/eval_efficientsam3_coco_prompts.py \
  --checkpoint-dir "${OUTPUT_ROOT}" \
  --manifest "${RUN_ROOT}/data/manifests/truck_prompt_smoke.jsonl" \
  --output-dir "${RUN_ROOT}/eval/truck_prompt_smoke" \
  --models rv_s rv_m rv_l tv_s tv_m tv_l ev_s ev_m ev_l vit_s vit_m vit_l \
  --prompt-modes point box text \
  --device cuda

export FINETUNE_ROOT="${DISTILL_ROOT}"
export FINETUNE_TEACHER_EMB="${TEACHER_EMB}"
export FINETUNE_TEACHER_OUTPUT="${SOURCE_RUN_ROOT}/output/stage1_teacher"
export FINETUNE_EPOCHS=1
export E2E_HEAD_EPOCHS=1
export RUN_E2E_HEAD_STAGE=0
export FINETUNE_NUM_SAMPLES=4
export FINETUNE_SPECS="es_vit_s:stage1_geometry_finetune/configs/es_vit_s.yaml:efficient_sam3_vit_s.pt:geometry/es_vit_s:efficient_sam3_vit_s_e2e_ft.pt:1"
bash scripts/run_image_encoder_finetune_matrix.sh

echo "12-architecture GPU smoke complete."
echo "Run root: ${RUN_ROOT}"
