# TinyViT-21M SAM3 Prompt-KD Plan

This plan uses the largest TinyViT student (`tiny_vit_21m`) as the image trunk
student for SAM3-compatible prompt-conditioned distillation.

## Shape Audit

At SAM3 image size 1008:

| Component | Shape |
| --- | --- |
| SAM3 teacher trunk | `[B, 1024, 72, 72]` |
| TinyViT-21M raw final feature | `[B, 576, 32, 32]` |
| Current student projection head | `[B, 1024, 72, 72]` |

The current architecture is shape-compatible with SAM3 because the student head
projects `576 -> 1024` and interpolates `32x32 -> 72x72`. It is not
resolution-native. The main risk is that the 32-to-72 upsample must recover
prompt-local detail through downstream distillation, so prompt-conditioned
geometry/text losses are required after image feature KD.

Run the audit:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
SCRATCH_ROOT=/storage/scratch1/9/eliu354/efficientsam3_prompt_kd \
RUN_FORWARD=1 \
bash scripts/run_tinyvit21_shape_check.sh
```

## Scratch Layout

Use Scratch for all mutable training assets:

```text
/storage/scratch1/9/eliu354/efficientsam3_prompt_kd/
  data/
  teacher_cache/
  runs/
  logs/
  shape_audit/
```

Do not write checkpoints, W&B files, teacher caches, or large intermediate
features into the git repo.

## Full Training Schedule

Use H100 for the full run and W&B for every stage.

| Stage | Epochs | Trainable modules | Data | Main loss |
| --- | ---: | --- | --- | --- |
| 1. Image feature KD | 50 | TinyViT-21M + projection head | SA-1B / SA-Co images | trunk MSE + cosine |
| 2. Point/box geometry KD | 30 | TinyViT-21M + projection head | SA-1B prompt masks | trunk MSE + mask BCE/Dice |
| 3. Text prompt KD | 20 | TinyViT-21M + projection head | SA-Co Silver text prompts | text-conditioned mask/object KD |
| 4. Hard negative/exemplar refine | 10 | TinyViT-21M + projection head | LVIS/COCO/SA-Co hard cases | false-positive suppression + mask KD |

Total: 110 epochs.

SA-Co Gold remains eval-only. Use SA-Co VEval for validation and COCO fixed
prompts for sanity checks.

## Resumable Smoke

The prompt-KD smoke trainer verifies the checkpoint and W&B resume contract. It
saves:

```text
${SCRATCH_ROOT}/runs/tinyvit21_prompt_kd_smoke/
  checkpoints/latest.pt
  checkpoints/epoch_0000.pt
  wandb_run_id.txt
  config.json
```

Run locally on an allocated H100:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
module load python/3.12.5 cuda/12.6.1
source .venv/bin/activate
python -m pip install -e ".[stage1]"

USE_WANDB=1 \
WANDB_PROJECT=efficientsam3-prompt-kd \
bash scripts/run_tinyvit21_prompt_kd_smoke.sh
```

Resume the same run:

```bash
USE_WANDB=1 \
WANDB_PROJECT=efficientsam3-prompt-kd \
WANDB_RESUME=allow \
bash scripts/run_tinyvit21_prompt_kd_smoke.sh
```

`wandb_run_id.txt` is reused automatically. To force a known W&B run id:

```bash
WANDB_RUN_ID=<existing-run-id> USE_WANDB=1 bash scripts/run_tinyvit21_prompt_kd_smoke.sh
```

## Submit Smoke on H100

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation
PACE_PARTITION=gpu-h100 \
PACE_GRES=gpu:h100:1 \
PACE_QOS=embers \
USE_WANDB=1 \
bash scripts/submit_h100_tinyvit21_prompt_kd_smoke.sh
```

Override `PACE_ACCOUNT` if the default account is not valid for H100 on the
current cluster allocation.
