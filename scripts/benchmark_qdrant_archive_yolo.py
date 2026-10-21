"""Evaluate sign retrieval on a held-out split of a YOLO-format dataset.

The archive is read-only. Ground-truth boxes from the test split are cropped
into the output directory and queried against the existing LawDB collection,
then against LawDB plus the separate supplementary gallery.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deps.embeddings import get_embedding_model
from src.settings import settings


TARGETS = {"P.102", "P.103a", "P.104", "P.123a", "P.123b", "P.127", "P.130", "P.131a", "P.131b", "P.131c"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_split(archive: Path, split_file: Path) -> list[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def map_class(label: str) -> tuple[str, str | None]:
    if label.startswith("P.127*"):
        return "P.127", label.removeprefix("P.127*")
    return label, None


def create_crops(archive: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    classes = [line.strip() for line in (archive / "classes.txt").read_text(encoding="utf-8").splitlines()]
    test_names = read_split(archive, archive / "split_dataset" / "test_files.txt")
    crops_dir = output_dir / "query_crops"
    records: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    for name in test_names:
        image_path = archive / "images" / name
        label_path = archive / "labels" / f"{Path(name).stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or YOLO label for test entry {name}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                fields = line.split()
                if not fields:
                    continue
                class_id = int(fields[0])
                if class_id < 0 or class_id >= len(classes):
                    raise ValueError(f"Invalid class ID {class_id} in {label_path}")
                source_label = classes[class_id]
                code, subtype = map_class(source_label)
                if code not in TARGETS:
                    continue
                if len(fields) < 5:
                    raise ValueError(f"Malformed YOLO row in {label_path}: {line}")
                cx, cy, bw, bh = (float(value) for value in fields[1:5])
                if not all(0.0 <= value <= 1.0 for value in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                    raise ValueError(f"Invalid normalized box in {label_path}: {line}")
                x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                # Add 8% context around the annotated sign while keeping the crop in-frame.
                pad_x, pad_y = (x2 - x1) * 0.08, (y2 - y1) * 0.08
                box = (
                    max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                    min(width, int(np.ceil(x2 + pad_x))), min(height, int(np.ceil(y2 + pad_y))),
                )
                crop = image.crop(box)
                if crop.width < 2 or crop.height < 2:
                    raise ValueError(f"Degenerate crop in {label_path}: {line}")
                code_dir = crops_dir / code.replace(".", "_")
                code_dir.mkdir(parents=True, exist_ok=True)
                query_id = f"{Path(name).stem}__{line_number:03d}"
                crop_path = code_dir / f"{query_id}.jpg"
                crop.save(crop_path, quality=95)
                records.append({
                    "query_id": query_id,
                    "source_image": str(image_path),
                    "source_label": source_label,
                    "gold_code": code,
                    "speed_subtype": subtype,
                    "bbox_xyxy": list(box),
                    "image_path": str(crop_path),
                })
                raw_counts[code] += 1
    return records, dict(sorted(raw_counts.items()))


def code_ranking(points: list[Any]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for point in points:
        code = str((point.payload or {}).get("sign_code") or "")
        if code:
            best[code] = max(best.get(code, -1.0), float(point.score))
    return sorted(best.items(), key=lambda item: item[1], reverse=True)


def rank_code(ranking: list[tuple[str, float]], gold: str) -> int:
    return next((i for i, (code, _) in enumerate(ranking, start=1) if code == gold), len(ranking) + 1)


def metrics(ranks: list[int]) -> dict[str, float | int]:
    return {
        "query_count": len(ranks),
        "recall_at_1": float(np.mean([rank <= 1 for rank in ranks])) if ranks else 0.0,
        "recall_at_3": float(np.mean([rank <= 3 for rank in ranks])) if ranks else 0.0,
        "recall_at_5": float(np.mean([rank <= 5 for rank in ranks])) if ranks else 0.0,
        "mrr": float(np.mean([1 / rank for rank in ranks])) if ranks else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/qdrant_archive_test_20261008")
    parser.add_argument("--base-collection", default=settings.collection_name)
    parser.add_argument("--supplement-collection", default="traffic_sign_gallery_20261008_clip_gmp_vit_l_14")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    archive = args.archive_dir.resolve()
    if not (archive / "classes.txt").is_file():
        parser.error(f"Not a YOLO dataset directory: {archive}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, counts = create_crops(archive, args.output_dir)
    if not records:
        parser.error("No target sign instances found in the test split")
    print(f"Prepared {len(records)} held-out target crops: {counts}", flush=True)

    client = QdrantClient(url=settings.qdrant_uri, check_compatibility=False)
    for collection in (args.base_collection, args.supplement_collection):
        if not client.collection_exists(collection):
            parser.error(f"Required Qdrant collection is missing: {collection}")
    base_count = int(client.get_collection(args.base_collection).points_count or 0)
    supplement_count = int(client.get_collection(args.supplement_collection).points_count or 0)
    if not base_count or not supplement_count:
        parser.error("Baseline or supplement collection is empty")

    device, model_name = choose_device(args.device), settings.embedding_name
    embedder, dimension = get_embedding_model(model_name, device=device)
    base_dim = client.get_collection(args.base_collection).config.params.vectors.size
    if dimension != base_dim:
        parser.error(f"Embedding dimension mismatch: model={dimension}, Qdrant={base_dim}")
    rows = []
    embed_seconds = 0.0
    query_seconds = 0.0
    for index, record in enumerate(records, start=1):
        with Image.open(record["image_path"]) as image:
            start = time.perf_counter()
            vector = np.asarray(embedder.embed_image(image.convert("RGB")), dtype=np.float32).reshape(-1)
            embed_seconds += time.perf_counter() - start
        start = time.perf_counter()
        base_points = client.query_points(
            collection_name=args.base_collection, query=vector.tolist(), limit=base_count,
            with_payload=True, with_vectors=False,
        ).points
        supplement_points = client.query_points(
            collection_name=args.supplement_collection, query=vector.tolist(), limit=supplement_count,
            with_payload=True, with_vectors=False,
        ).points
        query_seconds += time.perf_counter() - start
        base_ranking = code_ranking(base_points)
        augmented_ranking = code_ranking([*base_points, *supplement_points])
        gold = record["gold_code"]
        base_rank, augmented_rank = rank_code(base_ranking, gold), rank_code(augmented_ranking, gold)
        rows.append({
            **record,
            "baseline_rank": base_rank,
            "baseline_top5": [{"code": code, "score": round(score, 6)} for code, score in base_ranking[:5]],
            "augmented_rank": augmented_rank,
            "augmented_top5": [{"code": code, "score": round(score, 6)} for code, score in augmented_ranking[:5]],
            "rank_improved": augmented_rank < base_rank,
            "rank_worsened": augmented_rank > base_rank,
        })
        if index % 50 == 0 or index == len(records):
            print(f"Queried {index}/{len(records)} archive crops", flush=True)

    split_metrics = {
        split: metrics([row[f"{split}_rank"] for row in rows])
        for split in ("baseline", "augmented")
    }
    per_code = []
    for code in sorted(TARGETS):
        code_rows = [row for row in rows if row["gold_code"] == code]
        per_code.append({
            "sign_code": code,
            "query_count": len(code_rows),
            "present_in_test_split": bool(code_rows),
            "baseline_recall_at_1": float(np.mean([row["baseline_rank"] == 1 for row in code_rows])) if code_rows else None,
            "augmented_recall_at_1": float(np.mean([row["augmented_rank"] == 1 for row in code_rows])) if code_rows else None,
            "baseline_recall_at_5": float(np.mean([row["baseline_rank"] <= 5 for row in code_rows])) if code_rows else None,
            "augmented_recall_at_5": float(np.mean([row["augmented_rank"] <= 5 for row in code_rows])) if code_rows else None,
        })
    present_codes = sorted(counts)
    macro = {
        split: {
            "recall_at_1": float(np.mean([row[f"{split}_recall_at_1"] for row in per_code if row["sign_code"] in present_codes])),
            "recall_at_5": float(np.mean([row[f"{split}_recall_at_5"] for row in per_code if row["sign_code"] in present_codes])),
        }
        for split in ("baseline", "augmented")
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(archive),
        "split": "split_dataset/test_files.txt (ground-truth YOLO boxes)",
        "query_count": len(rows),
        "target_codes": sorted(TARGETS),
        "test_counts_by_code": counts,
        "missing_target_codes": sorted(TARGETS - set(counts)),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "device": device,
        "qdrant_uri": settings.qdrant_uri,
        "baseline_collection": args.base_collection,
        "baseline_points": base_count,
        "supplement_collection": args.supplement_collection,
        "supplement_points": supplement_count,
        "rank_aggregation": "maximum cosine score per exact sign_code across points; augmented scores merge both collections",
        "metrics_micro": split_metrics,
        "metrics_macro_over_present_target_codes": macro,
        "rank_changes": {
            "rank_improved": sum(row["rank_improved"] for row in rows),
            "rank_worsened": sum(row["rank_worsened"] for row in rows),
            "rank_unchanged": sum(not row["rank_improved"] and not row["rank_worsened"] for row in rows),
            "top1_newly_correct": sum(row["baseline_rank"] > 1 and row["augmented_rank"] == 1 for row in rows),
            "top1_lost": sum(row["baseline_rank"] == 1 and row["augmented_rank"] > 1 for row in rows),
        },
        "runtime": {
            "embedding_seconds": round(embed_seconds, 3),
            "qdrant_query_seconds": round(query_seconds, 3),
            "embedding_ms_per_crop": round(1000 * embed_seconds / len(rows), 3),
            "qdrant_ms_per_crop": round(1000 * query_seconds / len(rows), 3),
        },
        "limitations": [
            "Only the archive's labeled test split was evaluated; train images were not used as queries.",
            "The archive has no P.131b or P.131c labels, so these two target signs cannot be evaluated here.",
            "P.127 speed-limit variants are collapsed to the target family P.127 for retrieval scoring; original subtype is retained per row.",
            "This is a retrieval-only test using ground-truth boxes, not an end-to-end detector evaluation.",
            "Class imbalance in the test split is summarized with both micro- and macro-averaged metrics.",
        ],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "query_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "per_code.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_code[0]))
        writer.writeheader()
        writer.writerows(per_code)
    lines = [
        "# Archive held-out sign retrieval comparison", "",
        f"- Test crops: {len(rows)} across {len(counts)} of 10 target codes.",
        f"- Model: `{model_name}`; baseline collection `{args.base_collection}` ({base_count} points).",
        f"- Augmented gallery: `{args.supplement_collection}` ({supplement_count} points).",
        "- Crops came from ground-truth boxes in `test_files.txt`; no test samples were ingested.", "",
        "| Gallery | Micro R@1 | Micro R@3 | Micro R@5 | MRR | Macro R@1 | Macro R@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, label in (("baseline", "LawDB only"), ("augmented", "LawDB + supplementary gallery")):
        m = split_metrics[split]
        lines.append(f"| {label} | {m['recall_at_1']:.3f} | {m['recall_at_3']:.3f} | {m['recall_at_5']:.3f} | {m['mrr']:.3f} | {macro[split]['recall_at_1']:.3f} | {macro[split]['recall_at_5']:.3f} |")
    lines.extend(["", "## Per-code results", "", "| Code | Test instances | LawDB R@1 | Augmented R@1 | LawDB R@5 | Augmented R@5 |", "|---|---:|---:|---:|---:|---:|"])
    for row in per_code:
        fmt = lambda x: "—" if x is None else f"{x:.3f}"
        lines.append(f"| {row['sign_code']} | {row['query_count']} | {fmt(row['baseline_recall_at_1'])} | {fmt(row['augmented_recall_at_1'])} | {fmt(row['baseline_recall_at_5'])} | {fmt(row['augmented_recall_at_5'])} |")
    lines.extend(["", "## Notes", "", f"- Newly top-1 correct: {report['rank_changes']['top1_newly_correct']}; top-1 lost: {report['rank_changes']['top1_lost']}.", f"- Missing from this dataset test split: {', '.join(report['missing_target_codes']) or 'none'}.", "- Existing LawDB and supplementary Qdrant collections were queried read-only; archive crops were not ingested."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
