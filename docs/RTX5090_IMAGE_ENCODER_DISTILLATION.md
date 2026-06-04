# RTX 5090 SAM3 Image Encoder Distillation Smoke Run

This runbook describes how to reproduce the Stage 1 SAM3 image encoder distillation pipeline on a single RTX 5090 workstation. Run commands from the repository root. Keep data, checkpoints, logs, and evaluation output outside the repository by setting `RUN_ROOT`.

## Summary

- Hardware target: 1x RTX 5090 with 32 GB VRAM.
- Teacher: official SAM3 image model checkpoint, using only the frozen image trunk.
- Students: ES-RV-S / RepViT-M0.9, ES-RV-M / RepViT-M1.1, and ES-RV-L / RepViT-M2.3.
- Dataset for the smoke run: 1120 randomly sampled SA-1B training images, approximately 0.01% of full SA-1B.
- Sampling seed: `5090`.
- Loss: masked feature MSE plus masked channel-wise cosine loss.
- Output target: merged EfficientSAM3 checkpoints with distilled image encoders and original SAM3 heads.

The subset manifest at `${RUN_ROOT}/data/SA-1B-0.01P/subset_manifest.json` records the selected keys. The smoke configs use `DATA.RANDOM_SAMPLE=False` when pointed at that materialized subset, so teacher embedding export and student training use exactly the same image keys.

## 1. Clone and Configure Paths

Clone the repo anywhere on the workstation, then run the rest of the commands from the repo root:

```bash
git clone git@github.com:thedannyliu/EfficientSam3-Distillation.git
cd EfficientSam3-Distillation
git checkout image-encoder-distill-pipeline
```

Choose a run root outside the repo. The default below is relative to the repo root and keeps generated output out of the main directory:

```bash
export RUN_ROOT="../efficientsam3_distill_smoke"
export ENV_DIR="${RUN_ROOT}/venv"
mkdir -p "${RUN_ROOT}"/logs/{preflight,assets,distill,eval}
```

Check the workstation before starting:

```bash
nvidia-smi
python3 --version
df -h "$(dirname "${RUN_ROOT}")"
```

The run root needs enough free disk for the SA-1B subset, the teacher embedding file, the Python environment/cache, logs, and three merged checkpoints.

## 2. Create a venv

Create a Python 3.12 virtual environment under `RUN_ROOT`:

```bash
python3.12 -m venv "${ENV_DIR}"
source "${ENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel
```

Install the Stage 1 dependency set:

```bash
pip install -e ".[stage1]"
```

If that fails on optional heavy dependencies or installs a PyTorch wheel that cannot initialize CUDA, use the fallback set:

```bash
pip install -e . --no-deps
pip install --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
pip install \
  "timm>=1.0.17" "numpy>=1.26.4" tqdm "ftfy==6.1.1" regex \
  "iopath>=0.1.10" typing_extensions huggingface_hub psutil \
  "decord>=0.6.0" "mmengine>=0.10.4" "pycocotools>=2.0.7" \
  "yacs>=0.1.8" "Pillow>=10.0.0" "opencv-python>=4.9.0.80" \
  "scipy>=1.10.0" "scikit-image>=0.21.0" "scikit-learn>=1.3.0" \
  "tensorboard>=2.12.0" "einops>=0.7.0" "hydra-core>=1.3.2" \
  "submitit>=1.5.1" "fvcore>=0.1.5.post20221221" \
  "fairscale>=0.4.13" pandas pyyaml segment-anything
```

Verify CUDA before continuing:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
PY
```

Do not continue if `torch.cuda.is_available()` is `False`.

## 3. Hugging Face Login

The SAM3 checkpoint is gated on Hugging Face. Authenticate the run-local cache:

```bash
HF_HOME="${RUN_ROOT}/cache/huggingface" hf auth login
HF_HOME="${RUN_ROOT}/cache/huggingface" hf auth whoami
```

If you already have an approved token file in a path relative to this repo, you can use it instead:

```bash
export HF_TOKEN_PATH="../hf/token"
```

## 4. Preflight

Run the repo preflight from the repository root:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/preflight_image_encoder_distill.sh
```

After dependencies are installed, rerun only the checks with:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" PREFLIGHT_INSTALL_DEPS=0 \
  bash scripts/preflight_image_encoder_distill.sh
```

The preflight writes logs to:

```text
${RUN_ROOT}/logs/preflight/preflight_*.log
```

## 5. One-Command Smoke Run

Run the full local smoke pipeline:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/run_image_encoder_distill_smoke.sh
```

The runner downloads data/checkpoints, builds the deterministic 0.01% subset, exports teacher embeddings, trains ES-RV-S, ES-RV-M, and ES-RV-L for 3 smoke epochs each, and merges one checkpoint per student size.

All generated artifacts stay under `RUN_ROOT`:

```text
${RUN_ROOT}/
├── cache/
├── data/
│   └── SA-1B-0.01P/
├── eval/
│   └── image_encoder_distill/
├── logs/
│   ├── assets/
│   ├── distill/
│   ├── eval/
│   └── preflight/
├── output/
├── sam3_checkpoints/
└── venv/
```

Do not put generated output in the repository root.

## 6. Asset Preparation Only

To prepare the checkpoint, SA-1B subset, and environment without starting GPU training:

```bash
RUN_ROOT="${RUN_ROOT}" ENV_DIR="${ENV_DIR}" \
  bash scripts/prepare_image_encoder_distill_assets.sh
```

Logs are written to:

```text
${RUN_ROOT}/logs/assets/prepare_assets_*.log
```

The full runner reuses these assets and skips checkpoint/data preparation if they already exist.

## 7. Manual Checkpoint and Data Steps

The one-command runner handles this automatically. Use this section only when debugging or preparing pieces manually.

Download `sam3.pt`:

```bash
mkdir -p "${RUN_ROOT}/sam3_checkpoints"
HF_HOME="${RUN_ROOT}/cache/huggingface" hf download facebook/sam3 sam3.pt \
  --local-dir "${RUN_ROOT}/sam3_checkpoints"
```

Expected checkpoint path:

```text
${RUN_ROOT}/sam3_checkpoints/sam3.pt
```

Download the SA-1B 1% shard list from the Hugging Face mirror:

```bash
mkdir -p "${RUN_ROOT}/data"
bash data/download_sa1b_hf.sh \
  data/sa-1b-1p.txt \
  "${RUN_ROOT}/data/sa-1b-1p" \
  ssbai/sa1b
```

Reorganize the downloaded tar files into the Stage 1 dataloader layout:

```bash
python data/reorg_sa1b.py \
  --source-dir "${RUN_ROOT}/data/sa-1b-1p" \
  --output-dir "${RUN_ROOT}/data/SA-1B-1P" \
  --num-workers 4
```

Create the deterministic 0.01% subset:

```bash
python data/create_sa1b_subset.py \
  --source "${RUN_ROOT}/data/SA-1B-1P" \
  --output "${RUN_ROOT}/data/SA-1B-0.01P" \
  --num-samples 1120 \
  --seed 5090 \
  --mode hardlink
```

The training data path must contain:

```text
images/train/*.jpg
annotations/train/*.json
images/val/*.jpg
annotations/val/*.json
```

## 8. Export Teacher Image Embeddings

Start conservatively with `BATCH_SIZE=1`. Increase to `2` only after confirming memory headroom.

```bash
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1_teacher" \
  BATCH_SIZE=1 \
  GPUS=1 \
  --opts \
    MODEL.RESUME "${RUN_ROOT}/sam3_checkpoints/sam3.pt" \
    DATA.RANDOM_SAMPLE False \
    DISTILL.TEACHER_EMBED_PATH "${RUN_ROOT}/output/stage1_teacher/embeddings"
```

Expected output:

```text
${RUN_ROOT}/output/stage1_teacher/
├── config.json
├── log_rank0.txt
└── embeddings/
    ├── rank0-keys.txt
    └── rank0-values.bin
```

Optional integrity check:

```bash
bash stage1/scripts/save_image_embeddings.sh \
  CFG=stage1/configs/teacher/sam_vit_huge_sa1b_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1_teacher_check" \
  BATCH_SIZE=1 \
  GPUS=1 \
  --check-saved-embed
```

## 9. Train Student Image Encoders

The one-command runner trains all three RepViT sizes by default. To run one student manually, use ES-RV-M / RepViT-M1.1 as the balanced first check:

```bash
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1/es_rv_m" \
  BATCH_SIZE=4 \
  GPUS=1 \
  --opts \
    DATA.RANDOM_SAMPLE False \
    DISTILL.TEACHER_EMBED_PATH "${RUN_ROOT}/output/stage1_teacher/embeddings"
```

The smoke config trains for 3 epochs. If the workstation has memory headroom, try `BATCH_SIZE=8` for ES-RV-S/M. If it OOMs, use `BATCH_SIZE=2`. ES-RV-L defaults to `BATCH_SIZE=2`.

Manual settings for the other two sizes:

```text
ES-RV-S: CFG=stage1/configs/es_rv_s_5090_smoke.yaml, OUTPUT=${RUN_ROOT}/output/stage1/es_rv_s, BATCH_SIZE=4
ES-RV-L: CFG=stage1/configs/es_rv_l_5090_smoke.yaml, OUTPUT=${RUN_ROOT}/output/stage1/es_rv_l, BATCH_SIZE=2
```

## 10. Merge Student Encoder with SAM3 Heads

After a student training run finishes:

```bash
python stage1/convert_image_encoder_weights_stage1.py \
  --student-ckpt "${RUN_ROOT}/output/stage1/es_rv_m/ckpt_epoch_2.pth" \
  --sam3-ckpt "${RUN_ROOT}/sam3_checkpoints/sam3.pt" \
  --output "${RUN_ROOT}/output/efficient_sam3_repvit_m_smoke.pt"
```

The one-command runner writes all three merged checkpoints:

```text
${RUN_ROOT}/output/efficient_sam3_repvit_s_smoke.pt
${RUN_ROOT}/output/efficient_sam3_repvit_m_smoke.pt
${RUN_ROOT}/output/efficient_sam3_repvit_l_smoke.pt
```

## 11. Timing, IoU, and Overlay Checks

After the three merged image encoder checkpoints exist, run the fixed COCO-10 timing and accuracy check. The ten-image manifest is tracked in this repo; images are copied or downloaded into `${RUN_ROOT}/data/coco_fixed10`, so full COCO is not kept.

If you have a local `efficientsam3-benchmark` clone with the same COCO-10 images, point to it with a relative path:

```bash
export BENCHMARK_ROOT="../efficientsam3-benchmark"
```

Prepare only the ten COCO images:

```bash
python tools/eval_distilled_image_encoder.py \
  --run-root "${RUN_ROOT}" \
  --benchmark-root "${BENCHMARK_ROOT:-../efficientsam3-benchmark}" \
  --prepare-coco10 \
  --prepare-only
```

Run ES-RV-S/M/L with text, point, and box prompts:

```bash
python tools/eval_distilled_image_encoder.py \
  --run-root "${RUN_ROOT}" \
  --benchmark-root "${BENCHMARK_ROOT:-../efficientsam3-benchmark}" \
  --prepare-coco10 \
  --sizes s m l \
  --prompt-modes text point box
```

The evaluator writes structured output under:

```text
${RUN_ROOT}/eval/image_encoder_distill/<timestamp>/
├── metrics.csv
├── summary.json
├── single_image/
└── coco10/
```

Each COCO overlay shows the selected GT mask in red, the predicted mask in blue, and the active point or box prompt when applicable. The interactive notebook at `notebooks/distilled_image_encoder_interactive_eval.ipynb` lets you choose ES-RV-S/M/L, text/point/box prompt mode, and a fixed COCO sample; point and box sliders update a prompt preview before running inference, then the notebook displays timing, IoU, and the saved overlay image.

## 12. Notebook

Start Jupyter from the same venv:

```bash
source "${ENV_DIR}/bin/activate"
pip install jupyter ipywidgets
jupyter notebook notebooks/distilled_image_encoder_interactive_eval.ipynb
```

In the notebook, set:

```python
RUN_ROOT = Path(os.environ.get("RUN_ROOT", "../efficientsam3_distill_smoke")).resolve()
```

Use the same `RUN_ROOT` that contains the three merged checkpoints.

## 13. Completion Checks

After the one-command RTX 5090 run, verify:

```bash
wc -l "${RUN_ROOT}/output/stage1_teacher/embeddings/rank0-keys.txt"
ls -lh \
  "${RUN_ROOT}/output/stage1_teacher/embeddings/rank0-values.bin" \
  "${RUN_ROOT}/output/efficient_sam3_repvit_s_smoke.pt" \
  "${RUN_ROOT}/output/efficient_sam3_repvit_m_smoke.pt" \
  "${RUN_ROOT}/output/efficient_sam3_repvit_l_smoke.pt"
tail -40 "${RUN_ROOT}"/logs/distill/run_*.log
```

Expected smoke-run signals:

```text
Teacher key count: 1120
Teacher values size: about 11.9 GB
Merged checkpoints: efficient_sam3_repvit_s_smoke.pt, efficient_sam3_repvit_m_smoke.pt, efficient_sam3_repvit_l_smoke.pt
```

The exact checkpoint byte sizes may differ if package versions, PyTorch serialization, or training settings change. The required completion signal is that all three merged checkpoint files exist, each corresponding `stage1/es_rv_*/log_rank0.txt` reaches the final configured epoch, and the latest run log ends with all three merged checkpoint paths and `Done.`

## 14. Scaling Up

Once the smoke run succeeds, keep the same architecture and increase training scale:

```bash
bash stage1/scripts/train_image_student.sh \
  CFG=stage1/configs/es_rv_m_5090_smoke.yaml \
  DATA_PATH="${RUN_ROOT}/data/SA-1B-0.01P" \
  OUTPUT="${RUN_ROOT}/output/stage1/es_rv_m_1120_50ep" \
  BATCH_SIZE=4 \
  GPUS=1 \
  --opts TRAIN.EPOCHS 50 TRAIN.WARMUP_EPOCHS 5
```

For a larger random subset, override both teacher export and student training with the same values:

```bash
--opts DATA.NUM_SAMPLES 10000 DATA.RANDOM_SAMPLE True DATA.SAMPLE_SEED 5090
```

The teacher embedding export must be rerun whenever `DATA.NUM_SAMPLES`, `DATA.RANDOM_SAMPLE`, `DATA.SAMPLE_SEED`, image size, or embedding shape changes.

## 15. Time Estimates

Use the first 50-100 logged steps as the reliable estimate. The code reports `throughput` and `total_eta` directly in:

```text
${RUN_ROOT}/output/stage1_teacher/log_rank0.txt
${RUN_ROOT}/output/stage1/es_rv_s/log_rank0.txt
${RUN_ROOT}/output/stage1/es_rv_m/log_rank0.txt
${RUN_ROOT}/output/stage1/es_rv_l/log_rank0.txt
```
