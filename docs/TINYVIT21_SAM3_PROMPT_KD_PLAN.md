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
