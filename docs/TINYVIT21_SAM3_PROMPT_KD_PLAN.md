# TinyViT-21M SAM3 Prompt-KD Plan

This plan uses the largest TinyViT student (`tiny_vit_21m`) as the image trunk
student for SAM3-compatible prompt-conditioned distillation.

## Executed Implementation Plan

The initial implementation commit is:

```text
1a96b48 Add TinyViT21 prompt KD scaffolding
```

This commit implemented the scaffolding needed to start the TinyViT-21M SAM3
prompt-KD path safely:

1. Added `stage_prompt_kd.shape_check` to make the SAM3/TinyViT dimension
   assumptions executable.
   - Verifies the expected TinyViT-21M raw shape.
   - Verifies the projected student target shape.
   - Can optionally run student and teacher forward passes when the correct
     environment/checkpoints are available.

2. Added `stage_prompt_kd.checkpointing` for resumable training state.
   - Saves `checkpoints/latest.pt`.
   - Saves epoch checkpoints as `checkpoints/epoch_XXXX.pt`.
   - Stores model, optimizer, scheduler, scaler, epoch, global step, and W&B
     run id.
   - Reuses `wandb_run_id.txt` so W&B resume can attach to the same run.

3. Added prompt-KD helper modules.
   - `stage_prompt_kd.losses`: feature MSE, mask BCE/Dice, box L1.
   - `stage_prompt_kd.manifest`: small prompt-record JSONL helpers and
     mask-to-box/point utilities.

4. Added a resumable smoke trainer.
   - Entry point: `python -m stage_prompt_kd.train_smoke`.
   - Script wrapper: `scripts/run_tinyvit21_prompt_kd_smoke.sh`.
   - Default output root:
     `/storage/scratch1/9/eliu354/efficientsam3_prompt_kd/runs/tinyvit21_prompt_kd_smoke`.
   - Supports `--auto-resume`, `--resume`, `--use-wandb`, `--wandb-run-id`,
     and `--wandb-resume`.

5. Added H100 submission wrappers.
   - `scripts/submit_h100_tinyvit21_prompt_kd_smoke.sh`
   - `scripts/slurm_tinyvit21_prompt_kd_smoke_body.sbatch`
   - Defaults to Scratch for logs and outputs.
   - Defaults to `embers` QOS.

6. Registered the new package in `pyproject.toml`.
   - Added `stage_prompt_kd*`.
   - Also included `stage1_geometry_finetune*` because the smoke trainer reuses
     its existing `StudentTrunk`.

7. Added unit tests in `tests/test_prompt_kd_scaffolding.py`.
   - TinyViT-21M shape math.
   - KD loss finiteness.
   - manifest helpers.
   - checkpoint save/load and W&B run id persistence.

Validation completed before the commit:

```bash
python -m py_compile stage_prompt_kd/*.py
bash -n scripts/run_tinyvit21_shape_check.sh \
  scripts/run_tinyvit21_prompt_kd_smoke.sh \
  scripts/submit_h100_tinyvit21_prompt_kd_smoke.sh \
  scripts/slurm_tinyvit21_prompt_kd_smoke_body.sbatch
python -m unittest discover -s tests
python -m stage_prompt_kd.shape_check \
  --device cpu \
  --output-json /tmp/tinyvit21_sam3_shape.json
git diff --check
```

The metadata-only shape check produced:

```text
TinyViT-21M raw:        [1, 576, 32, 32]
Student projected:     [1, 1024, 72, 72]
SAM3 teacher target:   [1, 1024, 72, 72]
```

One limitation was observed during local validation: running the model-importing
smoke trainer from the login-node conda Python failed because that environment
did not have `iopath`. That is expected for this repo; real smoke/full runs
should use the project `.venv` after loading the PACE Python/CUDA modules.

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
  envs/
  data/
    SA-1B-1P/
    sa-v-text/
    coco/
    lvis/
  teacher_cache/
    stage1_sa1b_1p_sam3/
  runs/
  logs/
  shape_audit/
  checkpoints/
  asset_manifest.json
```

Do not write checkpoints, W&B files, teacher caches, or large intermediate
features into the git repo.

## Dataset Preparation Targets

The cache-first pipeline prepares data before any formal training starts.
Stage 1 uses SA-1B 1% as requested.

| Dataset | Stage | Target amount | Scratch location |
| --- | --- | ---: | --- |
| SA-1B 1% | Stage 1 image feature KD | all images in `data/sa-1b-1p.txt` | `${SCRATCH_ROOT}/data/SA-1B-1P` |
| SA-1B prompts | Stage 2 point/box KD | 300k images / about 1M prompt instances | derived from SA-1B or SA-Co masks |
| SA-Co/Silver | Stage 3 text KD | 500k text prompt records | `${SCRATCH_ROOT}/data/sa-v-text/saco-silver` |
| LVIS | Stage 4 hard negatives | all train annotations, val for eval | `${SCRATCH_ROOT}/data/lvis` |
| COCO | sanity and auxiliary hard cases | train2017 + val2017 | `${SCRATCH_ROOT}/data/coco` |
| SA-Co/VEval | validation | all annotation records | `${SCRATCH_ROOT}/data/sa-v-text/saco-veval` |
| SA-Co/Gold | final eval only | all annotation records | `${SCRATCH_ROOT}/data/sa-v-text/saco-gold` |

Gold stays eval-only to avoid benchmark leakage.

## Cache First

Prepare datasets and export SAM3 teacher image embeddings for Stage 1:

```bash
cd /storage/project/r-agarg35-0/eliu354/projects/EfficientSam3-Distillation

SCRATCH_ROOT=/storage/scratch1/9/eliu354/efficientsam3_prompt_kd \
TEACHER_BATCH_SIZE=8 \
GPUS=1 \
bash scripts/prepare_tinyvit21_prompt_kd_assets.sh
```

Submit the same cache job to a GPU node:

```bash
GPU_TYPE=h100 \
TEACHER_BATCH_SIZE=8 \
bash scripts/submit_prompt_kd_asset_cache.sh
```

For A100 or L40S, change only `GPU_TYPE` and batch size:

```bash
GPU_TYPE=a100 TEACHER_BATCH_SIZE=4 bash scripts/submit_prompt_kd_asset_cache.sh
GPU_TYPE=l40s TEACHER_BATCH_SIZE=2 bash scripts/submit_prompt_kd_asset_cache.sh
```

The cache job writes:

```text
${SCRATCH_ROOT}/teacher_cache/stage1_sa1b_1p_sam3/embeddings/
  rank0-keys.txt
  rank0-values.bin
${SCRATCH_ROOT}/asset_manifest.json
```

Only start formal training after `rank0-keys.txt` covers the full SA-1B 1%
image count.

## Stage 1 Training After Cache

After the cache job finishes:

```bash
SCRATCH_ROOT=/storage/scratch1/9/eliu354/efficientsam3_prompt_kd \
BATCH_SIZE=32 \
USE_WANDB=1 \
bash scripts/run_tinyvit21_stage1_train_after_cache.sh
```

Submit to Slurm:

```bash
GPU_TYPE=h100 \
BATCH_SIZE=32 \
USE_WANDB=1 \
bash scripts/submit_tinyvit21_stage1_train_after_cache.sh
```

The default Stage 1 run uses:

```text
Backbone: tiny_vit_21m
Dataset: SA-1B 1%
Epochs: 50
Warmup epochs: 5
Teacher embeddings: ${SCRATCH_ROOT}/teacher_cache/stage1_sa1b_1p_sam3/embeddings
Output: ${SCRATCH_ROOT}/runs/tinyvit21_stage1_sa1b_1p/train
```

## Checkpoint Policy

Current Stage 1 checkpoint behavior:

- Every save writes `ckpt_epoch_N.pth`.
- Every save also updates `ckpt_epoch_latest.pth`.
- Auto-resume still picks the newest `.pth` in the output directory, so
  `ckpt_epoch_latest.pth` is the expected resume target after this update.
- W&B writes `wandb_run_id.txt` in the training output directory and reuses it
  when `--use-wandb --wandb-resume allow` is passed again.

Selection policy:

- `latest`: for resume only.
- `final`: the last planned epoch, used for the first merged SAM3 checkpoint.
- `best`: not selected during Stage 1 because Stage 1 has no validation metric
  loop. Produce `best` after running COCO/SA-Co/LVIS eval and selecting by the
  primary eval metric.

For the Stage 1 run above, the automatic merged final checkpoint is:

```text
${SCRATCH_ROOT}/runs/tinyvit21_stage1_sa1b_1p/efficient_sam3_tinyvit21_stage1_final.pt
```

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
