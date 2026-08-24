# Colab Scratch-Training Package

This package prepares the exact workflow requested for **YOLO11n trained from random initialization** on a deterministic filtered COCO 2017 subset. It has been created and statically validated, but **no dataset download, training run, or model checkpoint has been executed or produced**.

| File | Purpose |
|---|---|
| `01_colab_yolo11n_scratch_coco.ipynb` | The ready-to-run Google Colab notebook, arranged as setup, data preparation, sanity gate, full training, resume, and verification cells. |
| `../scripts/colab_train_yolo11n_scratch_coco.py` | The companion implementation that filters COCO, validates labels, proves scratch initialization, mirrors artifacts to Drive, resumes from `last.pt`, validates `best.pt`, and copies the final checkpoint to the Drive root. |
| `../tools/build_colab_notebook.py` | The deterministic notebook builder. Run this only after changing the notebook template. |

The notebook defaults to `yolo11n_scratch_coco_v1_seed42`, uses class `0=person` and `1=cellphone`, reserves COCO source splits rather than mixing images across source train and validation sets, and constructs the detector from `YOLO('yolo11n.yaml')` with no `.pt` checkpoint input. Its final verification stage writes a checksum and copies the result to `/content/drive/MyDrive/proctoring_cv/best.pt`.

> Run the notebook manually in a T4-backed Colab runtime. The notebook is designed to fail before full training if the data preflight or five-epoch scratch sanity gate fails.
