"""Build the execution-ready Colab notebook without running its cells."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_colab_yolo11n_scratch_coco.ipynb"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.splitlines()]}


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"name": "Mim_YOLO11n_From_Scratch_COCO.ipynb", "provenance": [], "include_colab_link": True},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        markdown("""# Mim — YOLO11n Scratch Training on Filtered COCO

This notebook is an **execution-ready procedure** for a two-class detector: `person` (0) and `cellphone` (1). It downloads the official COCO 2017 annotations, selects a deterministic limited subset containing those classes, constructs `YOLO('yolo11n.yaml')` from random initialization, and mirrors artifacts to Google Drive. It deliberately does **not** load a `.pt` model, download pretrained weights, or use the proctoring-specific evaluation set during training.

> Before execution, choose **Runtime → Change runtime type → T4 GPU**. Run the cells from top to bottom. The full training cell can be rerun with `--resume` after a Colab interruption."""),
        code("""from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/proctoring_cv'
EXPERIMENT_ID = 'yolo11n_scratch_coco_v1_seed42'
DATASET_ROOT = '/content/coco2017_filtered'
LOCAL_ROOT = '/content/runs/experiments'
!nvidia-smi
"""),
        markdown("""## 1. Clone the exact repository revision and install dependencies

This setup installs the repository in editable mode and adds only the COCO parsing requirements used by the companion script."""),
        code("""!rm -rf /content/Mim
!git clone https://github.com/Szaman07/Mim.git /content/Mim
%cd /content/Mim
!pip install -q -e . pycocotools pillow
!python scripts/colab_train_yolo11n_scratch_coco.py --help
"""),
        markdown("""## 2. Prepare the filtered official COCO subset

The default limits are 8,000 train and 1,200 validation images. The selector reserves approximately 40% of each split for images containing a `cell phone`, then fills the remaining slots with eligible `person` images. The script writes YOLO labels, manifests, `dataset.yaml`, and a hard preflight report before training."""),
        code("""%cd /content/Mim
!python scripts/colab_train_yolo11n_scratch_coco.py prepare \\
  --dataset-root "$DATASET_ROOT" \\
  --drive-root "$DRIVE_ROOT" \\
  --experiment-id "$EXPERIMENT_ID" \\
  --train-limit 8000 --val-limit 1200 --phone-fraction 0.40
"""),
        markdown("""## 3. Scratch-only sanity gate

This trains a separate fresh model on a 64-image subset for five epochs. It is a preflight gate only: it writes an initial-parameter SHA-256 proof and is discarded before the full run. Stop and inspect the loss curve if this gate fails to improve."""),
        code("""%cd /content/Mim
!python scripts/colab_train_yolo11n_scratch_coco.py sanity \\
  --dataset-root "$DATASET_ROOT" \\
  --local-root "$LOCAL_ROOT" \\
  --drive-root "$DRIVE_ROOT" \\
  --experiment-id "$EXPERIMENT_ID" \\
  --sanity-images 64 --sanity-epochs 5 --sanity-imgsz 320 --sanity-batch 8
"""),
        markdown("""## 4. Full random-initialization training

The command builds a new `YOLO('yolo11n.yaml')`, records the initial parameter hash, trains locally on the Colab runtime, and mirrors new files to `MyDrive/proctoring_cv/experiments/<experiment-id>` every two minutes. This keeps local I/O fast while preserving `last.pt`, `best.pt`, plots, configuration, and logs in Drive."""),
        code("""%cd /content/Mim
!python scripts/colab_train_yolo11n_scratch_coco.py train \\
  --dataset-root "$DATASET_ROOT" \\
  --local-root "$LOCAL_ROOT" \\
  --drive-root "$DRIVE_ROOT" \\
  --experiment-id "$EXPERIMENT_ID" \\
  --epochs 50 --imgsz 640 --batch 16 --device 0 --workers 2 --save-period 5 --patience 15
"""),
        markdown("""## 5. Resume after an interruption

Rerun the setup and preparation cells if the runtime reset, then run this command. It restores `last.pt` from the Drive mirror if a local copy is unavailable. Do not rerun the full command without `--resume`, because that starts a distinct fresh experiment."""),
        code("""%cd /content/Mim
!python scripts/colab_train_yolo11n_scratch_coco.py train \\
  --dataset-root "$DATASET_ROOT" \\
  --local-root "$LOCAL_ROOT" \\
  --drive-root "$DRIVE_ROOT" \\
  --experiment-id "$EXPERIMENT_ID" --resume
"""),
        markdown("""## 6. Verify `best.pt` and preserve the final report

This loads only the newly produced `best.pt`, runs validation on the held-out filtered COCO validation set, calculates its SHA-256 checksum, and copies `final_verification.json` to Drive."""),
        code("""%cd /content/Mim
!python scripts/colab_train_yolo11n_scratch_coco.py verify \\
  --dataset-root "$DATASET_ROOT" \\
  --local-root "$LOCAL_ROOT" \\
  --drive-root "$DRIVE_ROOT" \\
  --experiment-id "$EXPERIMENT_ID" --imgsz 640 --device 0

!find "$DRIVE_ROOT/experiments/$EXPERIMENT_ID" -maxdepth 4 -type f | sort
"""),
        markdown("""## Expected durable artifacts

The Drive mirror holds `environment.json`, `initialization_proof.json`, `config_snapshot.yaml`, periodic checkpoints, `best.pt`, `last.pt`, Ultralytics plots, `results.csv`, and `final_verification.json`. Keep the custom proctoring webcam evaluation set isolated until the detector architecture and thresholds are frozen."""),
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(OUTPUT)
