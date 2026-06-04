# RTX 5090 Image Encoder Distillation Matrix

This runbook reproduces the local workstation pipeline for EfficientSAM3 image encoder distillation, conservative end-to-end fine-tuning, and COCO prompt evaluation. Run commands from the repository root. Keep generated data, caches, logs, checkpoints, and W&B files outside the repo by setting `RUN_ROOT` to a relative path.

## Summary

- Hardware target: one RTX 5090 class workstation.
- Teacher: official SAM3 image checkpoint.
- Students: RepViT S/M/L, TinyViT S/M/L, EfficientViT S/M/L, ViT S/M/L.
- Distillation data: fixed SA-1B 1% split from `data/sa-1b-1p.txt`.
- Fine-tune data: fixed disjoint SA-1B 0.01% split, default `1120` samples and seed `5091`.
- COCO eval: full `val2017` or `test2017`, with point, box, and text prompt modes.
- Outputs: merged distilled checkpoints and merged conservative E2E fine-tuned checkpoints.

## Paths and Environment

Use relative paths in shell setup:

```bash
export RUN_ROOT="../efficientsam3_distill_runs"
export ENV_DIR="${RUN_ROOT}/venv"
export HF_HOME="${RUN_ROOT}/cache/huggingface"
export PIP_CACHE_DIR="${RUN_ROOT}/cache/pip"
export CONDA_PKGS_DIRS="${RUN_ROOT}/conda_pkgs"
export XDG_CACHE_HOME="${RUN_ROOT}/cache/xdg"
export TORCH_HOME="${RUN_ROOT}/cache/torch"
export WANDB_DIR="${RUN_ROOT}/wandb"
mkdir -p "${RUN_ROOT}"
```

Create and activate the environment:

```bash
python3.12 -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel
pip install -e ".[stage1]"
```

If dependency resolution installs a non-CUDA PyTorch wheel, use the fallback install from `scripts/preflight_image_encoder_distill.sh`.

Authenticate Hugging Face into the run-local cache:

```bash
HF_HOME="${HF_HOME}" hf auth login
HF_HOME="${HF_HOME}" hf auth whoami
```

## Preflight

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/preflight_image_encoder_distill.sh
```

This checks the Stage 1 dependency set and builds all 12 student backbones.

## Fixed SA-1B Splits

Prepare the fixed distillation and fine-tune splits:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/prepare_sa1b_fixed_splits.sh
```

Important outputs:

```text
${RUN_ROOT}/data/SA-1B-1P/split_manifest.json
${RUN_ROOT}/data/SA-1B-0.01P-FINETUNE/split_manifest.json
```

The script verifies that the fine-tune keys do not overlap the distillation keys.

## Distill All 12 Image Encoders

Run the matrix:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  STUDENT_EPOCHS=3 \
  bash scripts/run_image_encoder_distill_matrix.sh
```

The script exports teacher embeddings once and trains/merges all 12 checkpoints:

```text
${RUN_ROOT}/output/efficient_sam3_repvit_s.pt
${RUN_ROOT}/output/efficient_sam3_repvit_m.pt
${RUN_ROOT}/output/efficient_sam3_repvit_l.pt
${RUN_ROOT}/output/efficient_sam3_tinyvit_s.pt
${RUN_ROOT}/output/efficient_sam3_tinyvit_m.pt
${RUN_ROOT}/output/efficient_sam3_tinyvit_l.pt
${RUN_ROOT}/output/efficient_sam3_efficientvit_s.pt
${RUN_ROOT}/output/efficient_sam3_efficientvit_m.pt
${RUN_ROOT}/output/efficient_sam3_efficientvit_l.pt
${RUN_ROOT}/output/efficient_sam3_vit_s.pt
${RUN_ROOT}/output/efficient_sam3_vit_m.pt
${RUN_ROOT}/output/efficient_sam3_vit_l.pt
```

The ViT family uses a ViTDet/SAM-style windowed ViT at 1008 resolution: S=`vit_tiny`, M=`vit_small`, L=`vit_base`.

For a single-family smoke run, override `STUDENT_SPECS` with a subset of the default colon-separated entries printed by the script.

## Conservative E2E Fine-Tune

Run the fine-tune matrix after distilled checkpoints exist:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  FINETUNE_EPOCHS=2 \
  E2E_HEAD_EPOCHS=1 \
  bash scripts/run_image_encoder_finetune_matrix.sh
```

The first stage fine-tunes the image encoder with the existing geometry dual-path distillation. The second stage uses a lower LR and unfreezes FPN, geometry encoder, and segmentation head. Text encoder and video/memory-bank components remain frozen/out of scope for this image-only pipeline.

For GPU smoke validation, run only one tiny model and skip the head-unfreeze stage:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  FINETUNE_EPOCHS=1 \
  FINETUNE_NUM_SAMPLES=4 \
  RUN_E2E_HEAD_STAGE=0 \
  FINETUNE_SPECS="es_vit_s:stage1_geometry_finetune/configs/es_vit_s.yaml:efficient_sam3_vit_s.pt:geometry/es_vit_s:efficient_sam3_vit_s_e2e_ft.pt:1" \
  bash scripts/run_image_encoder_finetune_matrix.sh
```

This smoke confirms loading, training, checkpoint writing, and merge conversion. It is not a mask-quality result.

Expected final checkpoints:

```text
${RUN_ROOT}/output/efficient_sam3_repvit_s_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_repvit_m_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_repvit_l_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_tinyvit_s_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_tinyvit_m_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_tinyvit_l_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_efficientvit_s_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_efficientvit_m_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_efficientvit_l_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_vit_s_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_vit_m_e2e_ft.pt
${RUN_ROOT}/output/efficient_sam3_vit_l_e2e_ft.pt
```

To skip the head-unfreeze stage and only run image-encoder geometry fine-tuning:

```bash
RUN_E2E_HEAD_STAGE=0 bash scripts/run_image_encoder_finetune_matrix.sh
```

## COCO Prompt Evaluation

Point `COCO_ROOT` to a COCO 2017 directory. The evaluator expects this layout:

```text
${COCO_ROOT}/images/val2017/*.jpg
${COCO_ROOT}/images/test2017/*.jpg
${COCO_ROOT}/annotations/instances_val2017.json
```

Example:

```bash
export COCO_ROOT="${RUN_ROOT}/data/coco"
```

Run a 10-image smoke eval on distilled checkpoints:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" COCO_ROOT="${COCO_ROOT}" \
  NUM_IMAGES=10 MODEL_SET=distilled \
  bash scripts/run_coco_prompt_eval_matrix.sh
```

Run full COCO val on E2E fine-tuned checkpoints:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" COCO_ROOT="${COCO_ROOT}" \
  MODEL_SET=e2e_ft COCO_SPLIT=val2017 NUM_IMAGES=-1 \
  bash scripts/run_coco_prompt_eval_matrix.sh
```

The val manifest uses the largest non-crowd object per image:

```text
point prompt: mask centroid
box prompt: COCO annotation bbox
text prompt: COCO category name
```

For `test2017`, provide a prompt manifest with prompts:

```bash
MANIFEST="${RUN_ROOT}/data/manifests/coco_test2017_prompts.jsonl" \
COCO_SPLIT=test2017 \
bash scripts/run_coco_prompt_eval_matrix.sh
```

Test split evaluation records inference/timing only unless ground-truth annotations are supplied.

## Outputs and Logs

Key output locations:

```text
${RUN_ROOT}/logs/data/
${RUN_ROOT}/logs/distill_matrix/
${RUN_ROOT}/logs/finetune_matrix/
${RUN_ROOT}/logs/coco_eval/
${RUN_ROOT}/eval/coco_prompts/<timestamp>/metrics.csv
${RUN_ROOT}/eval/coco_prompts/<timestamp>/summary.json
```

Completion checks:

```bash
wc -l "${RUN_ROOT}/output/stage1_teacher_sa1b_1p/embeddings/rank0-keys.txt"
ls -lh "${RUN_ROOT}"/output/efficient_sam3_*.pt
tail -40 "${RUN_ROOT}"/logs/distill_matrix/*.log
tail -40 "${RUN_ROOT}"/logs/finetune_matrix/*.log
```

Record the task, seed, manifest paths, GPU type, checkpoint/output directory, and W&B project in PR notes or experiment notes.

## PACE Smoke Validation

This repository snapshot was smoke-tested on PACE before handoff. The reusable runbook above keeps paths relative through `${RUN_ROOT}` for local workstation runs; the concrete cluster outputs below are the validation record.

```text
RUN_ROOT=/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke
BASE_SMOKE_ROOT=/storage/scratch1/9/eliu354/efficientsam3_distill_smoke
```

- Date: 2026-06-04.
- Distill/inference/finetune job: Slurm `9432004`, `gpu-rtx6000`, QOS `embers`, exit `0:0`, elapsed `00:22:27`.
- Post-finetune inference job: Slurm `9432295`, `gpu-rtx6000`, QOS `embers`, exit `0`.
- Distillation smoke used `4` SA-1B samples, `1` epoch, and produced all 12 merged checkpoints under `${RUN_ROOT}/output`.
- Inference smoke loaded all 12 distilled checkpoints and ran one-image point, box, and text prompt inference.
- Fine-tune smoke used `es_vit_s`, `4` samples, `1` epoch, `RUN_E2E_HEAD_STAGE=0`, and produced `${RUN_ROOT}/output/efficient_sam3_vit_s_e2e_ft.pt`.
- Post-finetune inference loaded `${RUN_ROOT}/output/efficient_sam3_vit_s_e2e_ft.pt` with suffix `_e2e_ft` and ran point, box, and text prompt inference.

Validation artifacts:

```text
${RUN_ROOT}/logs/slurm/sam3_12arch_smoke-9432004.out
${RUN_ROOT}/logs/distill_matrix/run_image_encoder_distill_matrix_20260604_185923.log
${RUN_ROOT}/logs/finetune_matrix/run_image_encoder_finetune_matrix_20260604_191559.log
${RUN_ROOT}/eval/truck_prompt_smoke/20260604_191158/summary.json
${RUN_ROOT}/eval/finetune_truck_prompt_smoke/vit_s_e2e_ft_20260604_192149/summary.json
```

The smoke manifest was a one-image truck prompt manifest without COCO annotations, so IoU fields are `null`; this validates code paths, checkpoint loading, prompt modes, and timing output rather than mask quality.
