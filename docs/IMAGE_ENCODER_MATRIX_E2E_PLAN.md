# Image Encoder Matrix and Conservative E2E Plan

## Summary

Implement a script-first workflow for comparing image encoder architectures in EfficientSAM3:

- Distill RepViT, TinyViT, EfficientViT, and ViT in S/M/L sizes.
- Add a conservative end-to-end fine-tune pipeline after image encoder replacement.
- Evaluate speed and mask quality on COCO with point, box, and text prompts.
- Keep all generated data and caches grouped under the git-ignored `RUN_ROOT`.

## Implementation

- `scripts/prepare_sa1b_fixed_splits.sh`
  - Prepares the fixed SA-1B 1% distillation split.
  - Samples a disjoint SA-1B 0.01% fine-tune split with seed `5091`.
  - Writes manifest files under `${RUN_ROOT}/data/...`; inspect them with `ls` or `cat`, do not execute the manifest paths.

- `scripts/run_image_encoder_distill_matrix.sh`
  - Exports SAM3 teacher image embeddings once.
  - Before formal distillation, run a one-epoch smoke on the smallest RepViT, TinyViT, and EfficientViT students with a tiny fixed sample count and separate `_smoke.pt` outputs.
  - First formal run should train the smallest RepViT, TinyViT, and EfficientViT students with the original Stage 1 schedule: `50` epochs and `5` warmup epochs.
  - The same script can train all 12 student image encoders after the first formal run is healthy.
  - Merges each image encoder into a full SAM3 checkpoint.

- `scripts/run_image_encoder_finetune_matrix.sh`
  - Reuses the fixed disjoint fine-tune split.
  - Runs geometry prompt-aware image encoder fine-tuning.
  - Optionally runs a low-LR head-unfreeze stage for FPN, geometry encoder, and segmentation head.

- `tools/build_coco_prompt_manifest.py`
  - Builds COCO val prompt manifests from annotations.
  - Uses largest non-crowd object per image, bbox for box prompts, mask centroid for point prompts, and COCO category name for text prompts.

- `tools/eval_efficientsam3_coco_prompts.py`
  - Evaluates all 12 checkpoints with point, box, and text prompts.
  - Records timing and IoU when ground truth is available.

## Fine-Tune Policy

Default fine-tuning is conservative to reduce catastrophic forgetting:

- Train image encoder/projection first while SAM3 heads are frozen.
- Then optionally unfreeze only FPN, geometry encoder, and segmentation head at lower LR.
- Keep text encoder and memory/video components frozen or outside the image-only training path.
- Do not unfreeze the full SAM3 model by default.

## Fine-Tune Execution Plan

- GPU smoke:
  - Run after 12-architecture distillation and inference smoke.
  - Use `es_vit_s` only, `4` SA-1B samples, `1` epoch.
  - Set `RUN_E2E_HEAD_STAGE=0` so the smoke validates geometry fine-tune loading, training, checkpoint save, and merge conversion without spending time on quality tuning.
  - Treat success as a pipeline check only, not a quality result.

- Formal finetune matrix:
  - Use `scripts/run_image_encoder_finetune_matrix.sh`.
  - Input checkpoints are the 12 distilled checkpoints under `${RUN_ROOT}/output`.
  - Training data is the fixed disjoint SA-1B 0.01% split with seed `5091`.
  - Stage 1 runs geometry prompt-aware image encoder fine-tuning with SAM3 heads frozen.
  - Stage 2 runs low-LR conservative E2E tuning with FPN, geometry encoder, and segmentation head unfrozen.
  - Text encoder and memory/video components remain frozen/out of scope.
  - Final outputs are `efficient_sam3_{repvit,tinyvit,efficientvit,vit}_{s,m,l}_e2e_ft.pt`.

## PACE GPU Smoke Record

- Date: 2026-06-04.
- Run root: `/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke`.
- Base smoke data and teacher embeddings reused from `/storage/scratch1/9/eliu354/efficientsam3_distill_smoke`.
- Slurm distill/inference/finetune job: `9432004`, `gpu-rtx6000`, QOS `embers`, state `COMPLETED`, exit `0:0`, elapsed `00:22:27`.
- Post-finetune inference job: `9432295`, `gpu-rtx6000`, QOS `embers`, exit `0`.
- Distillation smoke: all 12 model specs trained for `1` epoch on `4` samples and merged successfully.
- Distilled inference smoke: all 12 merged checkpoints loaded and ran one-image point, box, and text prompt inference.
- Fine-tune smoke: `es_vit_s` ran `1` epoch on `4` samples with `RUN_E2E_HEAD_STAGE=0`, then converted to `output/efficient_sam3_vit_s_e2e_ft.pt`.
- Post-finetune inference smoke: `vit_s` with checkpoint suffix `_e2e_ft` loaded and ran one-image point, box, and text prompt inference.

Important smoke outputs:

```text
/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke/output/efficient_sam3_{repvit,tinyvit,efficientvit,vit}_{s,m,l}.pt
/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke/output/efficient_sam3_vit_s_e2e_ft.pt
/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke/eval/truck_prompt_smoke/20260604_191158/summary.json
/storage/scratch1/9/eliu354/efficientsam3_12arch_smoke/eval/finetune_truck_prompt_smoke/vit_s_e2e_ft_20260604_192149/summary.json
```

The truck prompt smoke manifest does not include COCO annotations, so IoU fields are intentionally `null`; this run validates execution and timing capture only.

## Validation

- Run `bash -n` on all new shell scripts.
- Import both new Python tools.
- Run preflight to instantiate all 12 student backbones.
- Smoke one model per family before full matrix:
  - `es_rv_s`
  - `es_tv_s`
  - `es_ev_s`
  - `es_vit_s`
- Run `NUM_IMAGES=10` COCO val smoke eval before full val.

## Assumptions

- Local RTX 5090 workstation is the primary target for this runbook.
- `RUN_ROOT` is a relative path such as `./efficientsam3_distill_runs`.
- COCO `test2017` has no mask-quality score unless a prompt/annotation manifest is provided.
- SA-1B 0.01% uses `1120` samples by default.
