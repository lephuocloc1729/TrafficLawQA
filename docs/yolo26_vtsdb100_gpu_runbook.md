# Fine-tune YOLO26s and YOLO26m on VTSDB100

This runbook uses the `data/archive.zip` file without extracting it on the
Mac. The preparation script runs on the rented GPU machine, extracts only the
all annotated images (including real alternate-position captures marked
`_aug_`), preserves the provided train/validation/test splits, trains both
models, evaluates the
selected checkpoints once on the held-out test set, and bundles the results.

## Before renting

- Choose a Linux GPU machine with an NVIDIA GPU, at least **16 GB VRAM** if
  possible, and a persistent disk with **100 GB free**. The archive is about
  37 GB; the clean image set is about 13.3 GiB. The archive and extracted set
  coexist during preparation, and training needs room for checkpoints/logs.
- Keep the existing Mac copy of `data/archive.zip`. Upload a copy to the GPU
  machine; do not try to extract it on the Mac.
- Upload or clone this repository on the GPU machine. Keep the ZIP outside the
  `--workdir` directory.
- Make sure the GPU machine has internet access so PyTorch/Ultralytics and the
  official pretrained `yolo26s.pt` / `yolo26m.pt` weights can be downloaded.

## One-command run

On the GPU machine, from the repository root:

```bash
python -m pip install -U ultralytics
python scripts/yolo26_vtsdb100.py \
  --archive /workspace/archive.zip \
  --workdir /workspace/yolo26_vtsdb100 \
  --device 0
```

The script prepares the dataset first, then trains YOLO26s and YOLO26m in
sequence with the same seed, image size, epoch limit, and optimizer policy. It
uses early stopping, selects each model's best checkpoint using validation
metrics, and evaluates each selected checkpoint on the new test split once.
Default settings are 640 px, up to 80 epochs, patience 15, automatic batch size,
and no RAM/disk image cache. You can override them, for example with
`--imgsz 960` or `--epochs 60`.

## Optional: inspect the split before spending GPU time

Run preparation and audit first:

```bash
python scripts/yolo26_vtsdb100.py \
  --archive /workspace/archive.zip \
  --workdir /workspace/yolo26_vtsdb100 \
  --prepare-only
```

Review `/workspace/yolo26_vtsdb100/prepare_report.json`. It should show all 100
classes in train and `groups_are_isolated_across_splits` set to `true`. Check
the `classes_missing` lists for validation/test before training. The current
archive has three classes (Cliff_ahead, Railway_crossing_with_gates, and
Underpass) only in one unprefixed frame sequence. To prevent adjacent frames
from leaking across splits, that sequence stays in train, so those classes
cannot receive an independent validation/test score in this experiment.
Then start training using the prepared data:

```bash
python scripts/yolo26_vtsdb100.py \
  --archive /workspace/archive.zip \
  --workdir /workspace/yolo26_vtsdb100 \
  --skip-prepare \
  --device 0
```

The script refuses to overwrite an existing prepared dataset. If preparation
was interrupted, use a fresh `--workdir` or remove the incomplete work
directory before retrying. If disk space is tight after the report is written,
you can remove the uploaded ZIP from the GPU machine; the prepared dataset no
longer depends on it.

## Download after completion

The script creates:

```text
/workspace/yolo26_vtsdb100/yolo26_vtsdb100_download.zip
```

Download that ZIP to the Mac using the GPU provider's file browser/download
button or its supported file-transfer command. It contains:

- `yolo26s_vtsdb100_best.pt`
- `yolo26m_vtsdb100_best.pt`
- `comparison.csv` and `comparison.json` with test precision, recall, mAP50,
  mAP50–95, per-class metrics, and inference timing
- the data preparation audit and training `results.csv` logs

For a compact visual comparison and full confusion/PR plots, also download the
`runs/` directory from the GPU machine. The final checkpoints and comparison
table are already included in the ZIP.

## Why the script rebuilds the split

The archive contains 77,499 train images, 7,496 validation images, and 7,519
test images. Despite the `_aug_` filename marker, these include real captures
of the same scene from alternate positions. The script retains every annotated
image and preserves the provided split assignments exactly. Some scene IDs
overlap between supplied splits, so metrics use the dataset's official split
and may overstate performance on entirely unseen scenes; the preparation audit
records the overlap.

The project pins an older Ultralytics release for the application. Install or
update Ultralytics only in the rented GPU environment; there is no need to
change the application's existing dependency pin to run these experiments.
