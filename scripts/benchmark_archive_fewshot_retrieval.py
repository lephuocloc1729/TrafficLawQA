"""Compare zero-shot LawDB retrieval with few-shot example caches.

Support crops come only from the archive train split; test queries come only
from test_files.txt. No images are uploaded to Qdrant. The experiment compares
max-example similarity and class-prototype similarity as cache strategies.
"""

from __future__ import annotations

import argparse
import json
import random
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
K_VALUES = (1, 2, 4, 8, 16)
SEEDS = (13, 37, 73, 101, 211)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_entries(archive: Path, split: str) -> list[str]:
    return [line.strip() for line in (archive / "split_dataset" / f"{split}_files.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def class_code(label: str) -> tuple[str, str | None]:
    return ("P.127", label.removeprefix("P.127*")) if label.startswith("P.127*") else (label, None)


def extract_split_crops(archive: Path, split: str, out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    class_names = [line.strip() for line in (archive / "classes.txt").read_text(encoding="utf-8").splitlines()]
    records: list[dict[str, Any]] = []
    for image_name in image_entries(archive, split):
        image_path = archive / "images" / image_name
        label_path = archive / "labels" / f"{Path(image_name).stem}.txt"
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing image or label for {split} entry: {image_name}")
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            width, height = image.size
            for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
                values = line.split()
                if not values:
                    continue
                class_id = int(values[0])
                if not 0 <= class_id < len(class_names) or len(values) < 5:
                    raise ValueError(f"Malformed annotation in {label_path}: {line}")
                raw_label = class_names[class_id]
                code, subtype = class_code(raw_label)
                if code not in TARGETS:
                    continue
                cx, cy, bw, bh = map(float, values[1:5])
                if bw <= 0 or bh <= 0 or any(v < 0 or v > 1 for v in (cx, cy, bw, bh)):
                    raise ValueError(f"Invalid YOLO box in {label_path}: {line}")
                x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
                x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
                px, py = (x2 - x1) * 0.08, (y2 - y1) * 0.08
                box = (max(0, int(x1-px)), max(0, int(y1-py)), min(width, int(np.ceil(x2+px))), min(height, int(np.ceil(y2+py))))
                crop = image.crop(box)
                if crop.width < 2 or crop.height < 2:
                    raise ValueError(f"Degenerate crop in {label_path}: {line}")
                query_id = f"{Path(image_name).stem}__{line_no:03d}"
                code_dir = out_dir / f"{split}_crops" / code.replace(".", "_")
                code_dir.mkdir(parents=True, exist_ok=True)
                crop_path = code_dir / f"{query_id}.jpg"
                crop.save(crop_path, quality=95)
                records.append({
                    "query_id": query_id,
                    "source_image": str(image_path.resolve()),
                    "source_label": raw_label,
                    "gold_code": code,
                    "speed_subtype": subtype,
                    "image_path": str(crop_path.resolve()),
                })
    counts = dict(sorted(Counter(row["gold_code"] for row in records).items()))
    return records, counts


def embed_records(embedder: Any, records: list[dict[str, Any]], name: str) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    vectors = []
    for i, row in enumerate(records, 1):
        with Image.open(row["image_path"]) as image:
            vector = np.asarray(embedder.embed_image(image.convert("RGB")), dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Invalid embedding for {row['image_path']}")
        vectors.append(vector / norm)
        if i % 100 == 0 or i == len(records):
            print(f"Embedded {name}: {i}/{len(records)}", flush=True)
    return np.stack(vectors), time.perf_counter() - start


def merge_rankings(*score_maps: dict[str, float]) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for score_map in score_maps:
        for code, score in score_map.items():
            scores[code] = max(scores.get(code, -1.0), float(score))
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def point_scores(points: list[Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for point in points:
        code = str((point.payload or {}).get("sign_code") or "")
        if code:
            scores[code] = max(scores.get(code, -1.0), float(point.score))
    return scores


def ranks_for(ranking: list[tuple[str, float]], gold: str) -> int:
    return next((i for i, (code, _) in enumerate(ranking, 1) if code == gold), len(ranking) + 1)


def summarize_ranks(ranks: list[int]) -> dict[str, float | int]:
    return {
        "query_count": len(ranks),
        "recall_at_1": float(np.mean(np.asarray(ranks) <= 1)),
        "recall_at_3": float(np.mean(np.asarray(ranks) <= 3)),
        "recall_at_5": float(np.mean(np.asarray(ranks) <= 5)),
        "mrr": float(np.mean(1 / np.asarray(ranks, dtype=np.float64))),
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/archive_fewshot_20261008")
    parser.add_argument("--base-collection", default=settings.collection_name)
    parser.add_argument("--supplement-collection", default="traffic_sign_gallery_20261008_clip_gmp_vit_l_14")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()
    archive, out_dir = args.archive_dir.resolve(), args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train, train_counts = extract_split_crops(archive, "train", out_dir)
    test, test_counts = extract_split_crops(archive, "test", out_dir)
    if not train or not test:
        parser.error("No target crops found in archive train/test splits")
    print(f"Train support pool: {len(train)} crops {train_counts}", flush=True)
    print(f"Held-out test: {len(test)} crops {test_counts}", flush=True)

    device = choose_device(args.device)
    model_name = settings.embedding_name
    embedder, dimension = get_embedding_model(model_name, device=device)
    train_vectors, train_embed_seconds = embed_records(embedder, train, "train support pool")
    test_vectors, test_embed_seconds = embed_records(embedder, test, "held-out test")
    del embedder
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    client = QdrantClient(url=settings.qdrant_uri, check_compatibility=False)
    for collection in (args.base_collection, args.supplement_collection):
        if not client.collection_exists(collection):
            parser.error(f"Required Qdrant collection is missing: {collection}")
    base_count = int(client.get_collection(args.base_collection).points_count or 0)
    supplement_count = int(client.get_collection(args.supplement_collection).points_count or 0)
    base_dim = client.get_collection(args.base_collection).config.params.vectors.size
    if dimension != base_dim or not base_count or not supplement_count:
        parser.error("Collection size/dimension is empty or incompatible with the configured embedding model")

    base_maps: list[dict[str, float]] = []
    gallery_maps: list[dict[str, float]] = []
    qdrant_seconds = 0.0
    for i, (row, vector) in enumerate(zip(test, test_vectors), 1):
        start = time.perf_counter()
        base_points = client.query_points(collection_name=args.base_collection, query=vector.tolist(), limit=base_count, with_payload=True, with_vectors=False).points
        gallery_points = client.query_points(collection_name=args.supplement_collection, query=vector.tolist(), limit=supplement_count, with_payload=True, with_vectors=False).points
        qdrant_seconds += time.perf_counter() - start
        base_maps.append(point_scores(base_points))
        gallery_maps.append(point_scores(gallery_points))
        if i % 100 == 0 or i == len(test):
            print(f"Queried Qdrant baseline/gallery: {i}/{len(test)}", flush=True)

    # Cosine similarity from the frozen, L2-normalized image features.
    cache_sims = test_vectors @ train_vectors.T
    train_indices: dict[str, list[int]] = defaultdict(list)
    train_by_image: dict[str, dict[str, int]] = defaultdict(dict)
    for idx, row in enumerate(train):
        # Select at most one support crop per source image and class.
        train_by_image[row["gold_code"]].setdefault(row["source_image"], idx)
    for code, by_image in train_by_image.items():
        train_indices[code] = list(by_image.values())

    # Full-code rankings remain the output; support cache adds only the eight
    # labeled target identities present in the archive.
    baseline_ranks = [ranks_for(merge_rankings(m), row["gold_code"]) for m, row in zip(base_maps, test)]
    gallery_ranks = [ranks_for(merge_rankings(base, gallery), row["gold_code"]) for base, gallery, row in zip(base_maps, gallery_maps, test)]
    shot_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detailed_rows: list[dict[str, Any]] = []
    for shots in K_VALUES:
        eligible = {code: indices for code, indices in train_indices.items() if len(indices) >= shots}
        if not eligible:
            continue
        for seed in SEEDS:
            rng = random.Random(seed)
            chosen: dict[str, list[int]] = {}
            for code, indices in eligible.items():
                chosen[code] = sorted(rng.sample(indices, shots))
            nearest_maps: list[dict[str, float]] = [dict() for _ in test]
            proto_maps: list[dict[str, float]] = [dict() for _ in test]
            for code, indices in chosen.items():
                cache_score = np.max(cache_sims[:, indices], axis=1)
                proto = train_vectors[indices].mean(axis=0)
                proto /= np.linalg.norm(proto)
                proto_score = test_vectors @ proto
                for query_i in range(len(test)):
                    nearest_maps[query_i][code] = float(cache_score[query_i])
                    proto_maps[query_i][code] = float(proto_score[query_i])
            nearest_ranks, proto_ranks, gallery_nearest_ranks = [], [], []
            for i, row in enumerate(test):
                near_rank = ranks_for(merge_rankings(base_maps[i], nearest_maps[i]), row["gold_code"])
                proto_rank = ranks_for(merge_rankings(base_maps[i], proto_maps[i]), row["gold_code"])
                gallery_near_rank = ranks_for(merge_rankings(base_maps[i], gallery_maps[i], nearest_maps[i]), row["gold_code"])
                nearest_ranks.append(near_rank)
                proto_ranks.append(proto_rank)
                gallery_nearest_ranks.append(gallery_near_rank)
                detailed_rows.append({
                    "query_id": row["query_id"], "gold_code": row["gold_code"], "shots_per_class": shots,
                    "seed": seed, "baseline_rank": baseline_ranks[i], "gallery_rank": gallery_ranks[i],
                    "fewshot_nearest_rank": near_rank, "fewshot_prototype_rank": proto_rank,
                    "gallery_plus_fewshot_nearest_rank": gallery_near_rank,
                })
            run = {
                "shots_per_class": shots,
                "seed": seed,
                "classes_with_support": sorted(chosen),
                "supports_per_class": {code: len(indices) for code, indices in chosen.items()},
                "fewshot_nearest": summarize_ranks(nearest_ranks),
                "fewshot_prototype": summarize_ranks(proto_ranks),
                "gallery_plus_fewshot_nearest": summarize_ranks(gallery_nearest_ranks),
            }
            shot_runs[str(shots)].append(run)
            print(f"Completed {shots}-shot seed {seed}: nearest R@1={run['fewshot_nearest']['recall_at_1']:.3f}, prototype R@1={run['fewshot_prototype']['recall_at_1']:.3f}", flush=True)

    baseline_metrics, gallery_metrics = summarize_ranks(baseline_ranks), summarize_ranks(gallery_ranks)
    fewshot_summary = {}
    for shots, runs in shot_runs.items():
        fewshot_summary[shots] = {}
        for method in ("fewshot_nearest", "fewshot_prototype", "gallery_plus_fewshot_nearest"):
            fewshot_summary[shots][method] = {
                metric: mean_std([float(run[method][metric]) for run in runs])
                for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")
            }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(archive),
        "support_split": "split_dataset/train_files.txt",
        "test_split": "split_dataset/test_files.txt",
        "train_support_crop_count": len(train), "train_support_counts_by_code": train_counts,
        "test_query_count": len(test), "test_counts_by_code": test_counts,
        "target_codes": sorted(TARGETS), "missing_from_train": sorted(TARGETS - set(train_counts)),
        "missing_from_test": sorted(TARGETS - set(test_counts)),
        "embedding_model": model_name, "embedding_dimension": dimension, "device": device,
        "baseline_collection": args.base_collection, "baseline_points": base_count,
        "supplement_collection": args.supplement_collection, "supplement_points": supplement_count,
        "comparison_protocol": "Frozen CLIP-GmP embeddings; balanced K supports per available target code sampled from distinct train images; evaluate only ground-truth-box crops from held-out test split; five support seeds; rankings merge by max cosine per exact sign_code.",
        "baseline_lawdb_only": baseline_metrics,
        "existing_gallery_lawdb_plus_100_crops": gallery_metrics,
        "fewshot_seed_mean_std": fewshot_summary,
        "runtime": {
            "train_embedding_seconds": round(train_embed_seconds, 3),
            "test_embedding_seconds": round(test_embed_seconds, 3),
            "qdrant_seconds": round(qdrant_seconds, 3),
            "cache_similarity_matrix_seconds_excluded": "computed by NumPy after embeddings; negligible compared with image encoding",
        },
        "limitations": [
            "P.131b and P.131c are absent from the archive train and test labels and cannot be evaluated.",
            "P.127* speed variants are grouped as P.127 for this target-code comparison; original subtype is retained in crop manifests.",
            "The support cache covers only the target identities represented in this archive; all other sign codes remain baseline-only competitors.",
            "This is retrieval from ground-truth sign boxes, not end-to-end detection.",
            "The simple cache methods use max cosine similarity and no learned calibration; they are an initial few-shot baseline, not a tuned Tip-Adapter implementation.",
        ],
    }
    (out_dir / "fewshot_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "fewshot_query_rows.json").write_text(json.dumps(detailed_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Few-shot retrieval on archive test split", "",
        f"- Model: `{model_name}` on `{device}`; {len(train)} training crops form the support pool.",
        f"- Test: {len(test)} ground-truth crops; each K-shot run samples supports from distinct training images, never from test.",
        f"- Seeds per K: {len(SEEDS)}; baseline LawDB `{args.base_collection}` ({base_count} points); existing gallery `{args.supplement_collection}` ({supplement_count} points).", "",
        "| Retrieval method | R@1 | R@3 | R@5 | MRR |",
        "|---|---:|---:|---:|---:|",
        f"| LawDB only | {baseline_metrics['recall_at_1']:.3f} | {baseline_metrics['recall_at_3']:.3f} | {baseline_metrics['recall_at_5']:.3f} | {baseline_metrics['mrr']:.3f} |",
        f"| LawDB + existing 100-image gallery | {gallery_metrics['recall_at_1']:.3f} | {gallery_metrics['recall_at_3']:.3f} | {gallery_metrics['recall_at_5']:.3f} | {gallery_metrics['mrr']:.3f} |",
    ]
    for shots, methods in fewshot_summary.items():
        for key, label in (("fewshot_nearest", "K-shot nearest-example"), ("fewshot_prototype", "K-shot class prototype"), ("gallery_plus_fewshot_nearest", "Existing gallery + K-shot nearest")):
            fields = [methods[key][m] for m in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")]
            lines.append(f"| {label} ({shots}/class; mean ± SD) | " + " | ".join(f"{v['mean']:.3f} ± {v['std']:.3f}" for v in fields) + " |")
    lines.extend(["", "## Training support counts", "", "| Code | Train supports | Test queries |", "|---|---:|---:|"])
    for code in sorted(TARGETS):
        lines.append(f"| {code} | {train_counts.get(code, 0)} | {test_counts.get(code, 0)} |")
    lines.extend(["", "## Caveats", "", *[f"- {item}" for item in report["limitations"]]])
    (out_dir / "fewshot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written: {out_dir / 'fewshot_report.md'}", flush=True)


if __name__ == "__main__":
    main()
