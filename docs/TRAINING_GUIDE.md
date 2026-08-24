# End-to-End Training & Colab/Kaggle Survival Guide

This guide walks through training the **YOLO11n** primary detector from scratch (`pretrained=False`) on free cloud GPU resources (Google Colab / Kaggle) or a local machine.

---

## 1. Zero-Cost Training Strategy & Architecture

```text
Google Colab / Kaggle (Disposable Compute)
    ├── Local RAM Disk / Scratch (/content)  <-- Staged Training Data & Fast I/O
    └── Google Drive (/content/drive/MyDrive/proctoring_cv) <-- Durable Artifacts
            ├── datasets/
            ├── experiments/
            │   └── yolo11n_scratch_coco_v1_seed42/
            │       ├── config.yaml
            │       ├── environment.json
            │       ├── initialization_proof.json
            │       ├── checkpoints/ (best.pt, last.pt, checksums.sha256)
            │       ├── logs/
            │       └── metrics/
            └── registry/ (experiments.csv, experiments.jsonl)
```

---

## 2. Step-by-Step Training Walkthrough

### Step 2.1: Open Google Colab with GPU Runtime
1. Open [Google Colab](https://colab.research.google.com).
2. Go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **T4 GPU** (or standard free GPU).
3. Mount Google Drive for durable storage:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### Step 2.2: Clone Repository & Install Dependencies
```bash
%cd /content
!git clone https://github.com/Szaman07/Mim.git
%cd /content/Mim
!pip install -e .
```

### Step 2.3: Verify Environment & Hardware
Run the smoke test notebook `notebooks/00_environment_smoke_test.ipynb` or run:
```bash
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))
    print('VRAM (GiB):', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
"
```

### Step 2.4: Prepare Dataset (Filtered COCO 2-Class)
Extract or filter the dataset into project classes (0: `person`, 1: `cellphone`):
```bash
# Validate data integrity
python scripts/validate_dataset.py --config configs/data.yaml --dry-run

# Prepare dataset folder structure and manifests
python scripts/prepare_dataset.py --config configs/data.yaml
```

### Step 2.5: Run Preflight Gate (Tiny Overfit Test)
Before launching an expensive training run, verify that the loss drops and the model converges on a tiny fixed subset:
```bash
python scripts/tiny_overfit.py --config configs/experiments/yolo11n_scratch_coco.yaml --epochs 5 --device 0
```
> [!IMPORTANT]
> If `tiny_overfit.py` fails, stop immediately. Check learning rate, architecture YAML, or dataset paths before launching full training.

### Step 2.6: Launch Full Scratch Training
```bash
python scripts/train.py \
    --config configs/experiments/yolo11n_scratch_coco.yaml \
    --experiment-id yolo11n_scratch_coco_v1_seed42 \
    --drive-root /content/drive/MyDrive/proctoring_cv \
    --mode full
```

During training, the system:
1. Verifies random parameter initialization (`pretrained=False`) and hashes weights.
2. Evaluates validation metrics every epoch.
3. Atomically writes `best.pt`, `last.pt`, and periodic checkpoints with SHA-256 checksums.
4. Synchronizes checkpoints and metrics directly to your Google Drive experiment directory.

---

## 3. Recovery & Resume Workflow

If Colab disconnects or hits an idle timeout:

### Exact Resume (Continuing Interrupted Run)
```bash
python scripts/resume_training.py \
    --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42 \
    --checkpoint last.pt \
    --resume
```

### Branching to a New Experiment from `best.pt`
To test a new hyperparameter, learning rate, or dataset version without overwriting the original run:
```bash
python scripts/resume_training.py \
    --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42 \
    --checkpoint best.pt \
    --branch \
    --new-experiment-id yolo11n_coco_v2_branch
```

---

## 4. Evaluating & Exporting Checkpoints

```bash
# Evaluate checkpoint on validation or test split
python scripts/evaluate.py \
    --checkpoint /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42/checkpoints/best.pt \
    --split val
```

Copy the verified `best.pt` checkpoint to your local application deployment directory for Django integration.
