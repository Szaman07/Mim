# COLAB_TRAINING_SURVIVAL_GUIDE.md

This guide assumes that you have never managed a long-running cloud training job. The runtime is temporary. **Google Drive is the durable workspace; `/content` is not.** Colab’s free resources, GPU access, idle behavior, and maximum runtime vary; Google documents that free notebooks can run for at most 12 hours depending on availability and usage.[1]

## 1. Start a fresh Colab session

Open a new notebook, select a GPU runtime, and do not begin training until the verification cells succeed. A new runtime is normal after a disconnect or timeout.

```python
from google.colab import drive
drive.mount('/content/drive')
```

Use the project root:

```python
PROJECT = '/content/drive/MyDrive/proctoring_cv'
```

## 2. Clone the repository

Clone code into the temporary runtime for execution, but keep the authoritative repository in GitHub and all important outputs in Drive.

```bash
%cd /content
!rm -rf proctoring-cv
!git clone https://github.com/YOUR_ACCOUNT/proctoring-cv.git
%cd /content/proctoring-cv
```

Replace the placeholder with the actual private or public repository URL. Do not store tokens in the notebook.

## 3. Install exact dependencies

Install the pinned requirements, not an unpinned latest release.

```bash
!python -m pip install --upgrade pip
!python -m pip install -r requirements.txt
```

The notebook must print Python, PyTorch, CUDA, Ultralytics, OpenCV, and MediaPipe versions and save them to the Drive experiment directory.

## 4. Verify the GPU

```python
import os, platform, subprocess, sys, torch
print('Python:', sys.version)
print('Platform:', platform.platform())
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    print('Torch CUDA:', torch.version.cuda)
else:
    print('Training GPU unavailable; use dry-run/inference or reconnect later.')
```

Do not assume the same GPU is available in the next session. The training script chooses or recommends batch size from detected VRAM.

## 5. Locate the current experiment

List the Drive experiments and read the manifest before starting.

```bash
!find /content/drive/MyDrive/proctoring_cv/experiments -maxdepth 2 -type f | sort | tail -50
```

A valid experiment directory contains `config.yaml`, `manifest.json`, `environment.json`, `last.pt` when training has started, and logs/metrics. Never reuse a directory for a different experiment.

## 6. Run a dry check and tiny overfit test

```bash
!python scripts/prepare_dataset.py --config configs/data.yaml --dry-run
!python scripts/train.py --config configs/experiments/scratch_baseline.yaml --mode tiny-overfit
```

Do not launch a full run if labels, images, classes, or checkpoint paths fail validation. The tiny overfit test is a pipeline test, not a scientific result.

## 7. Start training

Use a unique experiment ID and a Drive output directory.

```bash
!python scripts/train.py \
  --config configs/experiments/scratch_baseline.yaml \
  --experiment-id yolo11n_scratch_coco_v1_seed42 \
  --drive-root /content/drive/MyDrive/proctoring_cv \
  --mode full
```

The primary scratch mode must construct the model from YAML/configuration with `pretrained=False`. A `.pt` path is not accepted in scratch mode.

## 8. How checkpoints are saved

Each experiment saves `last.pt` after the latest completed checkpoint interval, `best.pt` when the selected validation metric improves, periodic checkpoints such as `epoch_005.pt`, optimizer/scheduler state when supported, configuration, logs, metrics, plots, environment information, and checksums. The checkpoint manager writes to a temporary file and renames or copies only after integrity verification. It keeps at least the latest two periodic checkpoints and the last known good checkpoint.

## 9. Monitor training

Watch the cell output and inspect the CSV/JSON metrics in Drive. Check that the epoch is advancing, validation metrics are being written, and `last.pt` modification time changes. Do not keep a GPU session running without active work; Kaggle specifically documents interactive idle timeouts and recommends monitoring and stopping unused sessions.[2]

## 10. Safely stop a run

If you need to stop, interrupt only after a checkpoint has completed if possible. Then verify that `last.pt`, its checksum, and the latest metrics exist on Drive. Record the stop reason in the experiment notes. Do not delete the experiment directory.

## 11. If Colab disconnects or times out

Open a new session, mount Drive, clone the repository, install the pinned dependencies, run the environment check, and locate the experiment. Do not start from memory or create a new directory accidentally. Verify the last checkpoint checksum and compare the saved epoch with the metrics file.

## 12. Resume an interrupted run

Use resume only when continuing the same experiment with the same model configuration, dataset manifest, optimizer/scheduler plan, and intended training schedule:

```bash
!python scripts/resume_training.py \
  --experiment-dir /content/drive/MyDrive/proctoring_cv/experiments/yolo11n_scratch_coco_v1_seed42 \
  --checkpoint last.pt \
  --resume
```

Ultralytics documents that resume restores weights, optimizer, scheduler, and epoch state from `last.pt`.[3] The script must preserve the original experiment ID and append to its logs rather than overwrite them.

## 13. Start a new experiment from `best.pt`

Use `best.pt` for an intentional branch: a new augmentation, image size, dataset version, model, optimizer, or fine-tuning comparison. Create a new ID and copy the checkpoint as an input artifact. Do not use `resume=True`, because that would incorrectly imply exact continuation of optimizer and scheduler state.

## 14. Recover from out-of-memory (OOM)

Stop the run, preserve all existing artifacts, and start a new branch with one change at a time: reduce batch size, use AMP, reduce workers, reduce image size, disable RAM caching, or use gradient accumulation if supported. Record the error and changed configuration. Never delete the last working checkpoint to solve OOM.

## 15. Recover from a corrupted runtime or package mismatch

Restart the runtime, mount Drive again, clone the repository at the recorded Git commit, reinstall the pinned requirements, and run the smoke test. Do not upgrade packages inside a recovery session unless creating a named compatibility experiment. If a checkpoint fails to load, try the next periodic checkpoint or the backup copy after checksum verification.

## 16. Recover from Drive mount failure

Retry the mount in a new runtime and confirm that the expected directory exists. Do not train with the only copy of a checkpoint on temporary storage. If Drive remains unavailable, run only a dry check or save temporary diagnostics with an explicit warning; reconnect Drive before continuing full training.

## 17. Back up important results

At the end of a successful run, verify and copy `best.pt`, `last.pt`, metrics, plots, configuration, environment information, manifest, and README notes into a dated Drive backup directory. Commit code and configuration to GitHub, but do not commit datasets, raw webcam data, or large checkpoints blindly.

## 18. Avoid deleting the dataset

Never use recursive deletion against the project root. Keep datasets under `datasets/<version>` with a read-only or documented policy where possible. Delete only a temporary staged copy under `/content`, not the Drive source. Before cleanup, print the exact path and confirm it begins with `/content` rather than `/content/drive/MyDrive/proctoring_cv`.

## 19. Identify the latest experiment

Use the registry and manifest dates, not directory order alone:

```bash
!python scripts/experiment_registry.py --root /content/drive/MyDrive/proctoring_cv --sort latest
```

The latest experiment is the one with a valid manifest, configuration, Git SHA, and checkpoint metadata. A recently modified directory with missing files is not considered complete.

## 20. Reproduce an old experiment

Checkout the recorded Git commit, install the recorded dependency lock, use the recorded dataset manifest/hash, restore the recorded configuration and seed, and run the documented command. Reproduction should create a new output directory unless the goal is an exact read-only evaluation of the existing checkpoint.

## Recovery checklist

```text
New session
  → mount Drive
  → clone recorded Git commit
  → install pinned dependencies
  → verify GPU and versions
  → load experiment manifest
  → verify checkpoint checksum
  → resume same run OR create a new branch
  → verify logs and checkpoint on Drive
```

## References

[1]: https://research.google.com/colaboratory/faq.html "Google Colab FAQ"
[2]: https://www.kaggle.com/docs/efficient-gpu-usage "Kaggle Efficient GPU Usage Tips"
[3]: https://docs.ultralytics.com/modes/train "Model Training with Ultralytics YOLO"
