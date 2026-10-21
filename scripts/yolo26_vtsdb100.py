#!/usr/bin/env python3
"""Prepare the supplied VTSDB100 splits and fine-tune YOLO26s/m.

The input archive is read in place. All annotated images, including real
alternate-position captures marked ``_aug_``, are retained in their original
train/validation/test split.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT_NAME = "VTSDB100"
SPLITS = ("train", "val", "test")
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
AUGMENTED_RE = re.compile(r"_aug_\d+", re.IGNORECASE)
FRAME_RE = re.compile(r"^(?P<prefix>\d*)frame[_-]\d+$", re.IGNORECASE)


def _source_group(filename: str, crc: int, size: int) -> str:
    stem = AUGMENTED_RE.sub("", Path(filename).stem)
    frame_match = FRAME_RE.match(stem)
    if frame_match:
        prefix = frame_match.group("prefix")
        return f"frame-sequence:{prefix or 'unprefixed'}"
    return f"source-base:{stem}"


def _find_group(parent: dict[str, str], group: str) -> str:
    parent.setdefault(group, group)
    if parent[group] != group:
        parent[group] = _find_group(parent, parent[group])
    return parent[group]


def _union_groups(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find_group(parent, left)
    right_root = _find_group(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _read_source_records(archive: zipfile.ZipFile) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    zip_infos = {item.filename: item for item in archive.infolist()}
    records: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] | None = None
    seen_crc: dict[tuple[int, int], dict[str, Any]] = {}
    group_parent: dict[str, str] = {}
    duplicate_rows = 0
    duplicate_label_disagreements = 0
    class_counts: collections.Counter[int] = collections.Counter()
    annotated_counts: collections.Counter[str] = collections.Counter()
    augmented_counts: collections.Counter[str] = collections.Counter()

    for split in SPLITS:
        annotation_path = f"{ROOT_NAME}/annotations/instances_{split}.json"
        dataset = json.loads(archive.read(annotation_path))
        current_categories = sorted(dataset["categories"], key=lambda category: category["id"])
        if categories is None:
            categories = current_categories
        elif [(c["id"], c["name"]) for c in current_categories] != [
            (c["id"], c["name"]) for c in categories
        ]:
            raise ValueError(f"Category definitions differ in {annotation_path}")

        images = {image["id"]: image for image in dataset["images"]}
        image_annotations: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
        for annotation in dataset["annotations"]:
            image_annotations[annotation["image_id"]].append(annotation)

        for image_id, image in images.items():
            filename = Path(image["file_name"]).name
            annotated_counts[split] += 1
            if AUGMENTED_RE.search(filename):
                augmented_counts[split] += 1
            member = f"{ROOT_NAME}/{image['file_name']}"
            info = zip_infos.get(member)
            if info is None:
                raise FileNotFoundError(f"Image is missing from ZIP: {member}")
            key = (info.CRC, info.file_size)
            annotations = image_annotations.get(image_id, [])
            signature = tuple(
                sorted(
                    (
                        int(annotation["category_id"]),
                        tuple(round(float(value), 2) for value in annotation["bbox"]),
                    )
                    for annotation in annotations
                )
            )
            for annotation in annotations:
                class_counts[int(annotation["category_id"])] += 1

            row = {
                "source_split": split,
                "filename": filename,
                "member": member,
                "crc": info.CRC,
                "size": info.file_size,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": annotations,
                "signature": signature,
                "group": _source_group(filename, info.CRC, info.file_size),
            }
            if key in seen_crc:
                duplicate_rows += 1
                existing = seen_crc[key]
                _union_groups(group_parent, existing["group"], row["group"])
                if existing["signature"] != signature:
                    duplicate_label_disagreements += 1
            else:
                seen_crc[key] = row
            records.append(row)

    if categories is None or len(categories) != 100:
        raise ValueError("Expected exactly 100 VTSDB100 categories")
    expected_ids = list(range(1, len(categories) + 1))
    if [category["id"] for category in categories] != expected_ids:
        raise ValueError("Category ids must be contiguous 1..100 for YOLO conversion")

    for row in records:
        row["group"] = _find_group(group_parent, row["group"])
    records.sort(key=lambda row: (row["source_split"], row["filename"], row["crc"]))
    source_counts = collections.Counter(row["source_split"] for row in records)
    audit = {
        "annotated_images_by_original_split": dict(annotated_counts),
        "augmented_images_by_original_split": dict(augmented_counts),
        "images_written_by_original_split": dict(source_counts),
        "unique_image_content_count": len(seen_crc),
        "duplicate_rows_preserved_in_original_splits": duplicate_rows,
        "duplicate_groups_with_annotation_differences": duplicate_label_disagreements,
        "class_object_counts_before_deduplication": {
            str(class_id): count for class_id, count in sorted(class_counts.items())
        },
    }
    return records, [category["name"] for category in categories], audit


def _assign_group_splits(records: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Greedily balance image/object counts while keeping source groups intact."""
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        groups[record["group"]].append(record)

    group_labels: dict[str, collections.Counter[int]] = {}
    label_group_frequency: collections.Counter[int] = collections.Counter()
    total_labels: collections.Counter[int] = collections.Counter()
    for group, group_records in groups.items():
        counts: collections.Counter[int] = collections.Counter()
        present: set[int] = set()
        for record in group_records:
            for annotation in record["annotations"]:
                class_id = int(annotation["category_id"])
                counts[class_id] += 1
                present.add(class_id)
        group_labels[group] = counts
        total_labels.update(counts)
        label_group_frequency.update(present)

    rng = random.Random(seed)
    random_keys = {group: rng.random() for group in groups}
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            min((label_group_frequency[label] for label in group_labels[group]), default=10**9),
            -len(groups[group]),
            random_keys[group],
        ),
    )
    split_names = list(SPLIT_RATIOS)
    assigned: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    split_labels = {split: collections.Counter() for split in split_names}
    split_images = collections.Counter()
    target_images = {split: len(records) * ratio for split, ratio in SPLIT_RATIOS.items()}
    target_labels = {
        split: {class_id: count * ratio for class_id, count in total_labels.items()}
        for split, ratio in SPLIT_RATIOS.items()
    }

    # If a class exists in only one independent source group, keep that group
    # in train. Such classes cannot have an honest held-out score without
    # leaking adjacent frames or copies of the same source into evaluation.
    train_only_groups = {
        group
        for group, counts in group_labels.items()
        if any(label_group_frequency[class_id] == 1 for class_id in counts)
    }

    for group in ordered_groups:
        group_records = groups[group]
        counts = group_labels[group]
        if group in train_only_groups:
            destination = "train"
            assigned[destination].extend(group_records)
            split_images[destination] += len(group_records)
            split_labels[destination].update(counts)
            continue
        scores: dict[str, float] = {}
        for split in split_names:
            score = 0.18 * (target_images[split] - split_images[split]) / max(target_images[split], 1)
            for class_id, count in counts.items():
                target = max(target_labels[split][class_id], 1.0)
                deficit = (target - split_labels[split][class_id]) / target
                score += count * deficit / max(total_labels[class_id], 1)
            scores[split] = score + rng.random() * 1e-8
        destination = max(scores, key=scores.get)
        assigned[destination].extend(group_records)
        split_images[destination] += len(group_records)
        split_labels[destination].update(counts)

    missing = {
        split: [class_id for class_id in range(1, 101) if split_labels[split][class_id] == 0]
        for split in split_names
    }
    if missing["train"]:
        raise RuntimeError(
            "Group-aware stratification left classes absent from train: "
            + json.dumps(missing, ensure_ascii=False)
        )
    return assigned


def _write_dataset(archive_path: Path, workdir: Path, seed: int) -> dict[str, Any]:
    dataset_root = workdir / "dataset"
    if dataset_root.exists():
        raise FileExistsError(
            f"{dataset_root} already exists. Move it aside or choose a fresh --workdir."
        )
    dataset_root.mkdir(parents=True)

    with zipfile.ZipFile(archive_path) as archive:
        records, names, audit = _read_source_records(archive)
        split_records = {
            split: [row for row in records if row["source_split"] == split]
            for split in SPLITS
        }

        # Use generated names so two source folders cannot collide on basename.
        for split, rows in split_records.items():
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)
            rows.sort(key=lambda row: (row["source_split"], row["filename"]))
            for index, row in enumerate(rows):
                stem = f"{row['source_split']}_{Path(row['filename']).stem}_{index:06d}"
                suffix = Path(row["filename"]).suffix.lower() or ".png"
                image_dest = dataset_root / "images" / split / f"{stem}{suffix}"
                label_dest = dataset_root / "labels" / split / f"{stem}.txt"
                with archive.open(row["member"]) as src, image_dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

                label_lines = []
                for annotation in row["annotations"]:
                    class_id = int(annotation["category_id"]) - 1
                    x, y, width, height = map(float, annotation["bbox"])
                    x0 = min(max(x, 0.0), float(row["width"]))
                    y0 = min(max(y, 0.0), float(row["height"]))
                    x1 = min(max(x + width, 0.0), float(row["width"]))
                    y1 = min(max(y + height, 0.0), float(row["height"]))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    xc = ((x0 + x1) / 2) / row["width"]
                    yc = ((y0 + y1) / 2) / row["height"]
                    bw = (x1 - x0) / row["width"]
                    bh = (y1 - y0) / row["height"]
                    label_lines.append(f"{class_id} {xc:.7f} {yc:.7f} {bw:.7f} {bh:.7f}")
                if not label_lines:
                    raise ValueError(f"Image has no valid boxes: {row['member']}")
                label_dest.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    yaml_lines = [f"path: {json.dumps(str(dataset_root.resolve()))}", "train: images/train", "val: images/val", "test: images/test", "nc: 100", "names:"]
    yaml_lines.extend(f"  {index}: {json.dumps(name)}" for index, name in enumerate(names))
    (dataset_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    report: dict[str, Any] = {
        "archive": str(archive_path.resolve()),
        "seed": seed,
        "class_names": {str(index): name for index, name in enumerate(names)},
        "provided_splits_preserved": True,
        "augmented_images_used": True,
        "audit": audit,
        "splits": {},
    }
    group_splits: dict[str, set[str]] = collections.defaultdict(set)
    for split, rows in split_records.items():
        counts: collections.Counter[int] = collections.Counter()
        groups = set()
        for row in rows:
            groups.add(row["group"])
            group_splits[row["group"]].add(split)
            counts.update(int(annotation["category_id"]) for annotation in row["annotations"])
        report["splits"][split] = {
            "images": len(rows),
            "source_groups": len(groups),
            "objects": sum(counts.values()),
            "classes_present": len(counts),
            "classes_missing": [class_id - 1 for class_id in range(1, 101) if counts[class_id] == 0],
            "class_object_counts": {str(class_id - 1): count for class_id, count in sorted(counts.items())},
        }
    report["groups_overlapping_original_splits"] = sum(
        len(splits) > 1 for splits in group_splits.values()
    )
    return report


def prepare(args: argparse.Namespace) -> Path:
    archive_path = args.archive.resolve()
    workdir = args.workdir.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    workdir.mkdir(parents=True, exist_ok=True)
    report = _write_dataset(archive_path, workdir, args.seed)
    report_path = workdir / "prepare_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Prepared dataset: {workdir / 'dataset'}")
    print(f"Preparation audit: {report_path}")
    for split, details in report["splits"].items():
        print(f"  {split}: {details['images']:,} images, {details['objects']:,} boxes, {details['classes_present']} classes")
    return workdir / "dataset" / "data.yaml"


def _metric_dict(metrics: Any) -> dict[str, Any]:
    raw = getattr(metrics, "results_dict", {}) or {}
    result = {str(key): float(value) for key, value in raw.items() if isinstance(value, (int, float))}
    box = getattr(metrics, "box", None)
    if box is not None:
        result.setdefault("precision", float(getattr(box, "mp", 0.0)))
        result.setdefault("recall", float(getattr(box, "mr", 0.0)))
        result.setdefault("mAP50", float(getattr(box, "map50", 0.0)))
        result.setdefault("mAP50-95", float(getattr(box, "map", 0.0)))
        maps = getattr(box, "maps", None)
        if maps is not None:
            result["per_class_mAP50-95"] = [float(value) for value in maps]
    speed = getattr(metrics, "speed", None)
    if isinstance(speed, dict):
        result["speed_ms_per_image"] = {key: float(value) for key, value in speed.items()}
    return result


def train_and_compare(args: argparse.Namespace, data_yaml: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install a current Ultralytics release first: python -m pip install -U ultralytics") from exc

    workdir = args.workdir.resolve()
    runs_dir = workdir / "runs"
    deliver_dir = workdir / "deliverables"
    deliver_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "dataset_yaml": str(data_yaml.resolve()),
        "class_names": json.loads((workdir / "prepare_report.json").read_text(encoding="utf-8"))["class_names"],
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "seed": args.seed,
            "close_mosaic": args.close_mosaic,
        },
        "models": {},
    }

    for model_name in ("yolo26s", "yolo26m"):
        print(f"\n=== Fine-tuning {model_name} ===", flush=True)
        model = YOLO(f"{model_name}.pt")
        train_result = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer="auto",
            close_mosaic=args.close_mosaic,
            seed=args.seed,
            deterministic=True,
            cache=False,
            workers=args.workers,
            device=args.device,
            project=str(runs_dir),
            name=model_name,
            exist_ok=False,
            plots=True,
            save=True,
            verbose=True,
        )
        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", runs_dir / model_name))
        best_checkpoint = save_dir / "weights" / "best.pt"
        if not best_checkpoint.is_file():
            raise FileNotFoundError(f"Expected best checkpoint at {best_checkpoint}")

        print(f"Evaluating {model_name} once on the held-out test split", flush=True)
        evaluator = YOLO(str(best_checkpoint))
        test_metrics = evaluator.val(
            data=str(data_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=True,
            project=str(runs_dir),
            name=f"{model_name}_test",
            exist_ok=False,
            verbose=True,
        )

        delivery_checkpoint = deliver_dir / f"{model_name}_vtsdb100_best.pt"
        shutil.copy2(best_checkpoint, delivery_checkpoint)
        summary["models"][model_name] = {
            "best_checkpoint": str(delivery_checkpoint.name),
            "run_directory": str(save_dir.relative_to(workdir)),
            "test_metrics": _metric_dict(test_metrics),
        }
        # Retain the training result in the record for troubleshooting, if exposed.
        if train_result is not None:
            summary["models"][model_name]["training_result"] = _metric_dict(train_result)

    summary_path = deliver_dir / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = deliver_dir / "comparison.csv"
    fieldnames = ["model", "precision", "recall", "mAP50", "mAP50-95", "checkpoint"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, model_data in summary["models"].items():
            metrics = model_data["test_metrics"]
            writer.writerow(
                {
                    "model": model_name,
                    "precision": metrics.get("metrics/precision(B)", metrics.get("precision")),
                    "recall": metrics.get("metrics/recall(B)", metrics.get("recall")),
                    "mAP50": metrics.get("metrics/mAP50(B)", metrics.get("mAP50")),
                    "mAP50-95": metrics.get("metrics/mAP50-95(B)", metrics.get("mAP50-95")),
                    "checkpoint": model_data["best_checkpoint"],
                }
            )

    archive_output = workdir / "yolo26_vtsdb100_download.zip"
    prep_report = workdir / "prepare_report.json"
    with zipfile.ZipFile(archive_output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(deliver_dir.iterdir()):
            if path.is_file():
                bundle.write(path, arcname=f"deliverables/{path.name}")
        if prep_report.is_file():
            bundle.write(prep_report, arcname="prepare_report.json")
        if data_yaml.is_file():
            bundle.write(data_yaml, arcname="data.yaml")
        for model_name in ("yolo26s", "yolo26m"):
            run_path = runs_dir / model_name / "results.csv"
            if run_path.is_file():
                bundle.write(run_path, arcname=f"training_logs/{model_name}_results.csv")
    print(f"\nFinished. Download this file: {archive_output}")
    print(f"Comparison table: {csv_path}")
    return archive_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Path to data/archive.zip")
    parser.add_argument("--workdir", type=Path, default=Path("/workspace/yolo26_vtsdb100"), help="GPU disk workspace (must have at least 80 GB free)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 8))
    parser.add_argument("--device", default="0")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare and audit data, without training")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse dataset created by an earlier --prepare-only run")
    args = parser.parse_args()
    args.archive = args.archive.expanduser()
    args.workdir = args.workdir.expanduser()
    return args


def main() -> int:
    args = parse_args()
    if args.prepare_only and args.skip_prepare:
        raise ValueError("Use either --prepare-only or --skip-prepare, not both")
    if args.skip_prepare:
        data_yaml = args.workdir.expanduser().resolve() / "dataset" / "data.yaml"
        report = args.workdir.expanduser().resolve() / "prepare_report.json"
        if not data_yaml.is_file() or not report.is_file():
            raise FileNotFoundError("--skip-prepare requires dataset/data.yaml and prepare_report.json in --workdir")
    else:
        data_yaml = prepare(args)
    if not args.prepare_only:
        train_and_compare(args, data_yaml)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Ultralytics saves checkpoints under the work directory; rerun with a fresh workdir to start again.", file=sys.stderr)
        raise SystemExit(130)
