"""Extract every labeled traffic-sign crop from the archived YOLO dataset.

Images are grouped into one directory per exact source class label. The source
archive is read-only; an existing output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def split_members(archive: Path) -> dict[str, str]:
    members: dict[str, str] = {}
    split_root = archive / "split_dataset"
    for split in ("train", "test"):
        file_path = split_root / f"{split}_files.txt"
        if not file_path.is_file():
            continue
        for name in file_path.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if not name:
                continue
            key = Path(name).as_posix()
            if key in members and members[key] != split:
                raise ValueError(f"Image appears in multiple splits: {key}")
            members[key] = split
    return members


def find_label_file(labels_dir: Path, relative_image: Path) -> Path | None:
    candidate = labels_dir / relative_image.with_suffix(".txt")
    if candidate.is_file():
        return candidate
    # Some YOLO exports flatten labels even when images are nested.
    fallback = labels_dir / f"{relative_image.stem}.txt"
    return fallback if fallback.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/archive_sign_crops"))
    parser.add_argument("--padding", type=float, default=0.08, help="Fractional padding around each YOLO box (default: 0.08)")
    parser.add_argument("--include-camera-class", action="store_true", help="Also export the dataset's non-sign Camera class")
    args = parser.parse_args()

    archive = args.archive_dir.resolve()
    output = args.output_dir.resolve()
    images_dir, labels_dir = archive / "images", archive / "labels"
    classes_path = archive / "classes.txt"
    if not images_dir.is_dir() or not labels_dir.is_dir() or not classes_path.is_file():
        parser.error(f"Expected images/, labels/, and classes.txt under {archive}")
    if not 0 <= args.padding <= 0.5:
        parser.error("--padding must be between 0 and 0.5")
    if output.exists():
        parser.error(f"Output directory already exists; refusing to overwrite: {output}")

    class_names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines()]
    splits = split_members(archive)
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in supported_extensions)
    output.mkdir(parents=True)
    manifest_path = output / "manifest.csv"
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_object_counts: Counter[str] = Counter()
    missing_labels: list[str] = []
    skipped_camera = 0
    total_labeled_objects = 0
    manifest_fields = (
        "crop_path", "sign_code", "class_id", "source_class_label", "split", "source_image",
        "source_label", "annotation_line", "bbox_xyxy", "source_width", "source_height",
        "crop_width", "crop_height",
    )

    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_fields)
        writer.writeheader()
        for image_path in image_paths:
            relative_image = image_path.relative_to(images_dir)
            relative_key = relative_image.as_posix()
            label_path = find_label_file(labels_dir, relative_image)
            if label_path is None:
                missing_labels.append(relative_key)
                continue
            split = splits.get(relative_key, splits.get(relative_image.name, "unassigned"))
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                width, height = image.size
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    fields = line.split()
                    if not fields:
                        continue
                    class_id = int(fields[0])
                    if not 0 <= class_id < len(class_names) or len(fields) < 5:
                        raise ValueError(f"Malformed YOLO row in {label_path}: {line}")
                    source_label = class_names[class_id]
                    if source_label.casefold() == "camera" and not args.include_camera_class:
                        skipped_camera += 1
                        continue
                    cx, cy, box_width, box_height = map(float, fields[1:5])
                    if box_width <= 0 or box_height <= 0 or any(v < 0 or v > 1 for v in (cx, cy, box_width, box_height)):
                        raise ValueError(f"Invalid normalized box in {label_path}: {line}")
                    x1, y1 = (cx - box_width / 2) * width, (cy - box_height / 2) * height
                    x2, y2 = (cx + box_width / 2) * width, (cy + box_height / 2) * height
                    pad_x, pad_y = (x2 - x1) * args.padding, (y2 - y1) * args.padding
                    box = (
                        max(0, int(np.floor(x1 - pad_x))), max(0, int(np.floor(y1 - pad_y))),
                        min(width, int(np.ceil(x2 + pad_x))), min(height, int(np.ceil(y2 + pad_y))),
                    )
                    crop = image.crop(box)
                    if crop.width < 2 or crop.height < 2:
                        raise ValueError(f"Degenerate crop in {label_path}: {line}")
                    safe_image_name = re.sub(r"[^A-Za-z0-9._-]+", "_", relative_image.as_posix())
                    crop_name = f"{safe_image_name}__line{line_number:03d}.jpg"
                    class_dir = output / source_label
                    class_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = class_dir / crop_name
                    if crop_path.exists():
                        raise FileExistsError(f"Crop filename collision: {crop_path}")
                    crop.save(crop_path, format="JPEG", quality=95)
                    class_counts[source_label] += 1
                    class_object_counts[source_label] += 1
                    split_counts[split] += 1
                    total_labeled_objects += 1
                    writer.writerow({
                        "crop_path": crop_path.relative_to(output).as_posix(),
                        "sign_code": source_label,
                        "class_id": class_id,
                        "source_class_label": source_label,
                        "split": split,
                        "source_image": str(image_path),
                        "source_label": str(label_path),
                        "annotation_line": line_number,
                        "bbox_xyxy": json.dumps(box),
                        "source_width": width,
                        "source_height": height,
                        "crop_width": crop.width,
                        "crop_height": crop.height,
                    })
            if len(class_counts) and len(image_paths) and (sum(split_counts.values()) % 1000 == 0):
                print(f"Extracted {sum(split_counts.values())} crops", flush=True)

    summary = {
        "archive_dir": str(archive),
        "output_dir": str(output),
        "image_files_scanned": len(image_paths),
        "images_missing_label_files": missing_labels,
        "annotated_objects_exported": total_labeled_objects,
        "camera_objects_skipped": skipped_camera,
        "crop_counts_by_class": dict(sorted(class_counts.items())),
        "crop_counts_by_split": dict(sorted(split_counts.items())),
        "class_folders_created": len(class_counts),
        "padding_fraction": args.padding,
        "split_definition": "train/test from split_dataset lists; images absent from those lists are marked unassigned",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# Archive sign crops\n\n"
        "One folder per exact class label from the source dataset. Each crop is derived from a YOLO bounding box with 8% context padding by default. `manifest.csv` maps each crop to its source image, class, split, and box; `summary.json` records counts. The original archive was not modified.\n\n"
        f"Exported {total_labeled_objects} crops in {len(class_counts)} class folders. Camera annotations skipped: {skipped_camera}.\n",
        encoding="utf-8",
    )
    print(f"Done: {total_labeled_objects} crops in {len(class_counts)} class folders -> {output}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images had no matching label file; see summary.json", flush=True)


if __name__ == "__main__":
    main()
