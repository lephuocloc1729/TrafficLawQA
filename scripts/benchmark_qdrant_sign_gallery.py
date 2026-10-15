"""Ingest a supplementary sign-crop gallery into Qdrant and run an A/B test.

The project's existing LawDB collection remains read-only. The user-provided
folder is ingested into a separate collection using the configured production
embedding model, then retrieval is compared with and without that collection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sign_embeddings import load_folder_dataset
from src.deps.embeddings import get_embedding_model
from src.settings import settings


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def family_for_code(code: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+\.\d+)[a-z]?", code)
    return match.group(1) if match else code


def sample_id_for(row: dict[str, Any]) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, row["image_path"]))


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_images(model_name: str, records: list[dict[str, Any]], device: str) -> tuple[np.ndarray, float, float, int]:
    start = time.perf_counter()
    embedder, dimension = get_embedding_model(model_name, device=device)
    load_seconds = time.perf_counter() - start
    vectors = []
    embed_start = time.perf_counter()
    for index, row in enumerate(records, start=1):
        with Image.open(row["image_path"]) as image:
            vector = np.asarray(embedder.embed_image(image.convert("RGB")), dtype=np.float32).reshape(-1)
        if vector.shape[0] != dimension:
            raise ValueError(f"Expected embedding dimension {dimension}, got {vector.shape[0]}")
        vectors.append(vector)
        if index % 50 == 0 or index == len(records):
            print(f"Embedded {index}/{len(records)} images", flush=True)
    embedding_seconds = time.perf_counter() - embed_start
    parameter_count = int(embedder.parameters())
    del embedder
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.stack(vectors), load_seconds, embedding_seconds, parameter_count


def ensure_supplement_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            on_disk_payload=True,
        )
        return

    info = client.get_collection(collection_name)
    if info.config.params.vectors.size != vector_size or info.config.params.vectors.distance != models.Distance.COSINE:
        raise RuntimeError(f"Existing collection {collection_name} has incompatible vector configuration")
    points, _ = client.scroll(collection_name=collection_name, limit=10000, with_payload=True, with_vectors=False)
    if any((point.payload or {}).get("source_dataset") != "user_sign_crops_20261008" for point in points):
        raise RuntimeError(
            f"Collection {collection_name} already contains unrelated data; refusing to modify it"
        )


def upsert_supplement(
    client: QdrantClient,
    collection_name: str,
    records: list[dict[str, Any]],
    vectors: np.ndarray,
) -> None:
    points = []
    for row, vector in zip(records, vectors):
        path = Path(row["image_path"])
        payload = {
            "sample_id": sample_id_for(row),
            "source_dataset": "user_sign_crops_20261008",
            "source_type": "user_provided_sign_crop",
            "sign_code": row["code"],
            "sign_family": family_for_code(row["code"]),
            "image_path": str(path),
            "file_name": path.name,
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        points.append(models.PointStruct(id=payload["sample_id"], vector=vector.tolist(), payload=payload))
    client.upload_points(collection_name=collection_name, points=points, batch_size=32, wait=True)


def code_ranking(points: list[Any]) -> list[tuple[str, float]]:
    best: dict[str, float] = {}
    for point in points:
        code = str((point.payload or {}).get("sign_code") or "")
        best[code] = max(best.get(code, -1.0), float(point.score))
    return sorted(best.items(), key=lambda item: item[1], reverse=True)


def rank_code(ranking: list[tuple[str, float]], gold: str) -> int:
    return next((i for i, (code, _) in enumerate(ranking, start=1) if code == gold), len(ranking) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", default="outputs/qdrant_sign_gallery_ab_20261008")
    parser.add_argument("--base-collection", default=settings.collection_name)
    parser.add_argument("--supplement-collection", default=None)
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_dir}")
    records = load_folder_dataset(args.dataset_dir)
    if not records:
        parser.error(f"No labeled images found in {args.dataset_dir}")
    counts = defaultdict(int)
    for row in records:
        counts[row["code"]] += 1
    if any(count != 10 for count in counts.values()):
        print("Note: dataset is not balanced at exactly 10 images per code.", flush=True)

    device = choose_device(args.device)
    model_name = settings.embedding_name
    suffix = safe_name(model_name)
    supplement_collection = args.supplement_collection or f"traffic_sign_gallery_20261008_{suffix}"
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(url=settings.qdrant_uri, check_compatibility=False)

    if not client.collection_exists(args.base_collection):
        parser.error(f"Baseline collection does not exist: {args.base_collection}")
    base_info = client.get_collection(args.base_collection)
    base_size = base_info.config.params.vectors.size
    base_count = int(base_info.points_count or 0)
    if base_count <= 0:
        parser.error(f"Baseline collection is empty: {args.base_collection}")

    print(f"Embedding {len(records)} supplemental images with {model_name} on {device}", flush=True)
    vectors, load_seconds, embedding_seconds, parameter_count = embed_images(model_name, records, device)
    if vectors.shape[1] != base_size:
        parser.error(
            f"Model output size {vectors.shape[1]} does not match baseline collection size {base_size}"
        )

    ensure_supplement_collection(client, supplement_collection, int(vectors.shape[1]))
    upsert_supplement(client, supplement_collection, records, vectors)
    supplement_count = int(client.get_collection(supplement_collection).points_count or 0)
    if supplement_count != len(records):
        raise RuntimeError(f"Expected {len(records)} points in supplement, found {supplement_count}")

    query_rows = []
    for index, (record, vector) in enumerate(zip(records, vectors), start=1):
        base_points = client.query_points(
            collection_name=args.base_collection,
            query=vector.tolist(),
            limit=base_count,
            with_payload=True,
            with_vectors=False,
        ).points
        supplement_points = client.query_points(
            collection_name=supplement_collection,
            query=vector.tolist(),
            query_filter=models.Filter(
                must_not=[models.FieldCondition(key="sample_id", match=models.MatchValue(value=sample_id_for(record)))]
            ),
            limit=max(1, supplement_count - 1),
            with_payload=True,
            with_vectors=False,
        ).points
        base_ranking = code_ranking(base_points)
        augmented_ranking = code_ranking([*base_points, *supplement_points])
        gold = record["code"]
        base_rank = rank_code(base_ranking, gold)
        augmented_rank = rank_code(augmented_ranking, gold)
        query_rows.append(
            {
                "query_image": record["image_name"],
                "query_path": record["image_path"],
                "gold_code": gold,
                "baseline_rank": base_rank,
                "baseline_top1": base_ranking[0][0] if base_ranking else None,
                "baseline_top5": [{"code": c, "score": round(s, 6)} for c, s in base_ranking[:5]],
                "augmented_rank": augmented_rank,
                "augmented_top1": augmented_ranking[0][0] if augmented_ranking else None,
                "augmented_top5": [{"code": c, "score": round(s, 6)} for c, s in augmented_ranking[:5]],
                "rank_improved": augmented_rank < base_rank,
                "rank_worsened": augmented_rank > base_rank,
            }
        )
        if index % 20 == 0 or index == len(records):
            print(f"Queried {index}/{len(records)} evaluation images", flush=True)

    metrics = {}
    for split in ("baseline", "augmented"):
        ranks = [row[f"{split}_rank"] for row in query_rows]
        metrics[split] = {
            "query_count": len(ranks),
            "recall_at_1": float(np.mean([rank <= 1 for rank in ranks])),
            "recall_at_3": float(np.mean([rank <= 3 for rank in ranks])),
            "recall_at_5": float(np.mean([rank <= 5 for rank in ranks])),
            "mrr": float(np.mean([1 / rank for rank in ranks])),
        }
    changes = {
        "rank1_improved_queries": sum(row["rank_improved"] for row in query_rows),
        "rank1_worsened_queries": sum(row["rank_worsened"] for row in query_rows),
        "rank_unchanged_queries": sum(not row["rank_improved"] and not row["rank_worsened"] for row in query_rows),
        "top1_correct_before_after": sum(
            row["baseline_rank"] > 1 and row["augmented_rank"] == 1 for row in query_rows
        ),
        "top1_lost_after_augmentation": sum(
            row["baseline_rank"] == 1 and row["augmented_rank"] > 1 for row in query_rows
        ),
    }
    by_code = []
    for code in sorted(counts):
        rows = [row for row in query_rows if row["gold_code"] == code]
        by_code.append(
            {
                "sign_code": code,
                "query_count": len(rows),
                "baseline_recall_at_1": float(np.mean([row["baseline_rank"] == 1 for row in rows])),
                "augmented_recall_at_1": float(np.mean([row["augmented_rank"] == 1 for row in rows])),
                "baseline_recall_at_5": float(np.mean([row["baseline_rank"] <= 5 for row in rows])),
                "augmented_recall_at_5": float(np.mean([row["augmented_rank"] <= 5 for row in rows])),
            }
        )

    report_data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qdrant_uri": settings.qdrant_uri,
        "device": device,
        "embedding_model": model_name,
        "embedding_dimension": int(vectors.shape[1]),
        "baseline_collection": args.base_collection,
        "baseline_points": base_count,
        "supplement_collection": supplement_collection,
        "supplement_points": supplement_count,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_image_count": len(records),
        "dataset_codes": dict(sorted(counts.items())),
        "model_load_seconds": round(load_seconds, 3),
        "embedding_seconds": round(embedding_seconds, 3),
        "embedding_ms_per_image": round(1000 * embedding_seconds / len(records), 3),
        "parameter_count": parameter_count,
        "query_protocol": "100 dataset crops queried; each query point excluded from its supplement collection; baseline LawDB collection unchanged",
        "known_limitations": [
            "Query photos are from the same collection campaign as supplemental gallery images; exact sample IDs are held out but physical sign-instance IDs are unavailable.",
            "Evaluation covers only these 10 sign codes and does not measure detector performance or unknown-sign rejection.",
            "Ranking merges the maximum cosine score per exact sign_code across baseline and supplement collections.",
        ],
        "metrics": metrics,
        "rank_changes": changes,
        "per_code": by_code,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "query_rows.json").write_text(json.dumps(query_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "per_code.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_code[0]))
        writer.writeheader()
        writer.writerows(by_code)

    report_lines = [
        "# Qdrant supplementary sign-gallery A/B test",
        "",
        f"- Embedding model: `{model_name}` ({vectors.shape[1]} dimensions) on `{device}`.",
        f"- Baseline collection: `{args.base_collection}` ({base_count} points; read-only).",
        f"- New supplement collection: `{supplement_collection}` ({supplement_count} points).",
        f"- Queries: {len(query_rows)} labeled crops across {len(counts)} codes; the exact query point is excluded from its supplement gallery.",
        "- Retrieval ranks are aggregated by exact sign code using the maximum cosine similarity across matching points.",
        "",
        "| Gallery | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, label in (("baseline", "LawDB only"), ("augmented", "LawDB + supplement")):
        m = metrics[split]
        report_lines.append(
            f"| {label} | {m['recall_at_1']:.3f} | {m['recall_at_3']:.3f} | {m['recall_at_5']:.3f} | {m['mrr']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            f"- Top-1 correct after adding gallery: {changes['top1_correct_before_after']} newly correct; {changes['top1_lost_after_augmentation']} previously correct queries lost top-1.",
            f"- Query rank improved/worsened/unchanged: {changes['rank1_improved_queries']}/{changes['rank1_worsened_queries']}/{changes['rank_unchanged_queries']}.",
            "- The existing LawDB collection was not modified.",
            "",
            "## Per-code results",
            "",
            "| Code | Queries | LawDB R@1 | Augmented R@1 | LawDB R@5 | Augmented R@5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in by_code:
        report_lines.append(
            f"| {row['sign_code']} | {row['query_count']} | {row['baseline_recall_at_1']:.3f} | "
            f"{row['augmented_recall_at_1']:.3f} | {row['baseline_recall_at_5']:.3f} | {row['augmented_recall_at_5']:.3f} |"
        )
    report_lines.extend(["", "## Limitations", ""])
    report_lines.extend(f"- {limitation}" for limitation in report_data["known_limitations"])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Report written to {output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
