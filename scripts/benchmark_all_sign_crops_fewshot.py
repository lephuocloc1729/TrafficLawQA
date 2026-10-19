"""Leakage-aware few-shot retrieval using the curated all-sign-crops folder.

Support comes from ``data/all-traffic-sign-crops`` and queries from
``data/tested_sign_crops``. Any tested crop from a source frame represented in
the support gallery is excluded. The existing LawDB Qdrant collection is
queried read-only; no images or vectors are added to Qdrant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deps.embeddings import get_embedding_model
from src.settings import settings

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
K_VALUES = (1, 2, 4, 8)
SEEDS = (13, 37, 73, 101, 211)


def canonical_code(label: str) -> str:
    # The tested folder contains P.127 and its numeric speed variants in
    # separate folders; the LawDB uses P.127 as the parent sign identity.
    return "P.127" if label.startswith("P.127*") else label


def source_group(path: Path) -> str:
    # Crops retain their original frame name: 0503.jpg__line001.jpg.
    return re.sub(r"__line\d+.*$", "", path.stem)


def load_folder(folder: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rows.append({
            "image_name": path.name,
            "image_path": str(path.resolve()),
            "raw_label": path.parent.name,
            "code": canonical_code(path.parent.name),
            "source_group": source_group(path),
        })
    return rows


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_records(embedder: Any, rows: list[dict[str, Any]], batch_size: int) -> tuple[np.ndarray, float]:
    """Batch encode images using the configured CLIP model and its processor."""
    vectors: list[np.ndarray] = []
    start = time.perf_counter()
    for start_index in range(0, len(rows), batch_size):
        batch_rows = rows[start_index : start_index + batch_size]
        images = []
        for row in batch_rows:
            with Image.open(row["image_path"]) as im:
                images.append(im.convert("RGB"))
        inputs = embedder.processor(images=images, return_tensors="pt").to(embedder.device)
        with torch.inference_mode():
            features = embedder.model.get_image_features(**inputs)
            features = F.normalize(features, p=2, dim=1)
        vectors.append(features.detach().cpu().numpy().astype(np.float32, copy=False))
        for image in images:
            image.close()
        done = min(start_index + len(batch_rows), len(rows))
        if done % 500 < batch_size or done == len(rows):
            print(f"Embedded {done}/{len(rows)} crops", flush=True)
    return np.concatenate(vectors, axis=0), time.perf_counter() - start


def fetch_lawdb_vectors(client: QdrantClient, collection: str) -> tuple[np.ndarray, list[str]]:
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        points.extend(batch)
        if offset is None:
            break
    vectors, codes = [], []
    for point in points:
        vector = point.vector
        if isinstance(vector, dict):
            vector = vector.get("", next(iter(vector.values())))
        if vector is None:
            continue
        vectors.append(np.asarray(vector, dtype=np.float32))
        codes.append(canonical_code(str((point.payload or {}).get("sign_code") or "")))
    if not vectors:
        raise RuntimeError(f"No vectors found in LawDB collection {collection}")
    matrix = np.stack(vectors)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return matrix, codes


def grouped_scores(similarities: np.ndarray, codes: list[str]) -> list[dict[str, float]]:
    by_code: dict[str, list[int]] = defaultdict(list)
    for index, code in enumerate(codes):
        if code:
            by_code[code].append(index)
    result = []
    for row in similarities:
        result.append({code: float(np.max(row[indices])) for code, indices in by_code.items()})
    return result


def rank(scores: dict[str, float], gold: str) -> tuple[int, str | None]:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    rank_value = next((i for i, (code, _) in enumerate(ranked, 1) if code == gold), len(ranked) + 1)
    return rank_value, ranked[0][0] if ranked else None


def score_metrics(ranks: list[int], codes: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "query_count": len(ranks),
        "recall_at_1": float(np.mean(np.asarray(ranks) <= 1)),
        "recall_at_3": float(np.mean(np.asarray(ranks) <= 3)),
        "recall_at_5": float(np.mean(np.asarray(ranks) <= 5)),
        "mrr": float(np.mean(1.0 / np.asarray(ranks, dtype=np.float64))),
    }
    per_code = []
    for code in sorted(set(codes)):
        selected = [r for r, c in zip(ranks, codes) if c == code]
        per_code.append(float(np.mean(np.asarray(selected) <= 1)))
    result["macro_recall_at_1"] = float(np.mean(per_code)) if per_code else None
    return result


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def embedding_fingerprint(model_name: str, supports: list[dict[str, Any]], tests: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for partition, rows in (("support", supports), ("test", tests)):
        digest.update(partition.encode("utf-8"))
        for row in rows:
            path = Path(row["image_path"])
            stat = path.stat()
            digest.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\0{row['code']}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, default=ROOT / "data/all-traffic-sign-crops")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/tested_sign_crops")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/all_sign_fewshot_20261019")
    parser.add_argument("--base-collection", default=settings.collection_name)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    support_dir, test_dir = args.support_dir.resolve(), args.test_dir.resolve()
    if not support_dir.is_dir() or not test_dir.is_dir():
        parser.error(f"Support or test directory not found: {support_dir} / {test_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    supports = load_folder(support_dir)
    test_all = load_folder(test_dir)
    test_classes = {row["code"] for row in test_all}
    # Use only labels represented in the held-out folder.
    supports = [row for row in supports if row["code"] in test_classes]
    if not supports or not test_all:
        parser.error("No supported images or test images found after label filtering")

    # Find exact-file copies and original-source-frame overlap, then keep the
    # test split source-disjoint from every gallery example.
    support_hashes = {hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest() for row in supports}
    support_groups = {row["source_group"] for row in supports}
    overlap_exact = 0
    excluded = []
    tests = []
    for row in test_all:
        digest = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        same_source = row["source_group"] in support_groups
        same_content = digest in support_hashes
        overlap_exact += int(same_content)
        if same_source or same_content:
            excluded.append(row)
        else:
            tests.append(row)
    if not tests:
        parser.error("Leakage filtering removed all test images")

    # Deduplicate gallery to at most one crop per original source frame and
    # canonical code. This prevents repeated crops from one frame counting as
    # independent few-shot examples.
    dedup_supports: dict[tuple[str, str], dict[str, Any]] = {}
    for row in supports:
        dedup_supports.setdefault((row["code"], row["source_group"]), row)
    supports = sorted(dedup_supports.values(), key=lambda row: (row["code"], row["source_group"]))
    support_by_code: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(supports):
        support_by_code[row["code"]].append(index)
    test_counts = dict(sorted(Counter(row["code"] for row in tests).items()))
    support_counts = dict(sorted(Counter(row["code"] for row in supports).items()))
    classes_missing_support = sorted(set(test_counts) - set(support_counts))
    print(f"Support: {len(supports)} images across {len(support_counts)} tested labels", flush=True)
    print(f"Held-out queries after source-disjoint filtering: {len(tests)}", flush=True)
    print(f"Excluded test images: {len(excluded)} ({overlap_exact} exact content matches)", flush=True)

    device = choose_device(args.device)
    model_name = settings.embedding_name
    client = QdrantClient(url=settings.qdrant_uri, check_compatibility=False)
    if not client.collection_exists(args.base_collection):
        parser.error(f"LawDB collection not found: {args.base_collection}")
    base_vectors, base_codes = fetch_lawdb_vectors(client, args.base_collection)
    dimension = base_vectors.shape[1]

    # Persist embeddings so an interrupted or temporarily disconnected Qdrant
    # run can resume without repeating the multi-minute image-encoding pass.
    cache_path = args.output_dir / "embedding_cache.npz"
    fingerprint = embedding_fingerprint(model_name, supports, tests)
    metadata_path = args.output_dir / "embedding_cache.json"
    cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    support_embed_seconds = test_embed_seconds = 0.0
    if cache_path.is_file() and cache_metadata.get("fingerprint") == fingerprint:
        with np.load(cache_path) as cached:
            support_vectors = cached["support_vectors"]
            test_vectors = cached["test_vectors"]
        print("Loaded matching cached image embeddings", flush=True)
    else:
        embedder, dimension_from_model = get_embedding_model(model_name, device=device)
        if dimension_from_model != dimension:
            parser.error(f"Embedding dimension mismatch: model={dimension_from_model}, LawDB={dimension}")
        support_vectors, support_embed_seconds = embed_records(embedder, supports, args.batch_size)
        test_vectors, test_embed_seconds = embed_records(embedder, tests, args.batch_size)
        del embedder
        np.savez(cache_path, support_vectors=support_vectors, test_vectors=test_vectors)
        metadata_path.write_text(json.dumps({"fingerprint": fingerprint, "model": model_name}, indent=2), encoding="utf-8")
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    if support_vectors.shape[1] != dimension or test_vectors.shape[1] != dimension:
        parser.error("Cached/model embedding dimensions do not match the LawDB collection")
    baseline_scores = grouped_scores(test_vectors @ base_vectors.T, base_codes)
    baseline_ranks = [rank(scores, row["code"])[0] for scores, row in zip(baseline_scores, tests)]
    baseline_metrics = score_metrics(baseline_ranks, [row["code"] for row in tests])

    cache_similarities = test_vectors @ support_vectors.T
    run_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_rows = []
    available_ks = [k for k in K_VALUES if any(len(indices) >= k for indices in support_by_code.values())]
    for shots in available_ks:
        eligible_codes = {code: indices for code, indices in support_by_code.items() if len(indices) >= shots}
        for seed in SEEDS:
            rng = random.Random(seed)
            selected = {code: sorted(rng.sample(indices, shots)) for code, indices in eligible_codes.items()}
            cache_by_code: dict[str, np.ndarray] = {
                code: np.max(cache_similarities[:, indices], axis=1)
                for code, indices in selected.items()
            }
            shot_ranks, shot_top1 = [], []
            for query_index, (base, row) in enumerate(zip(baseline_scores, tests)):
                combined = dict(base)
                for code, per_query_scores in cache_by_code.items():
                    combined[code] = max(combined.get(code, -1.0), float(per_query_scores[query_index]))
                shot_rank, top1 = rank(combined, row["code"])
                shot_ranks.append(shot_rank)
                shot_top1.append(top1)
                query_rows.append({
                    "query_image": row["image_name"],
                    "source_group": row["source_group"],
                    "gold_code": row["code"],
                    "shots_per_code": shots,
                    "seed": seed,
                    "baseline_rank": baseline_ranks[query_index],
                    "fewshot_rank": shot_rank,
                    "baseline_top1": rank(base, row["code"])[1],
                    "fewshot_top1": top1,
                })
            summary = score_metrics(shot_ranks, [row["code"] for row in tests])
            summary.update({"shots_per_code": shots, "seed": seed, "supported_code_count": len(selected)})
            run_summaries[str(shots)].append(summary)
            print(f"{shots}-shot seed {seed}: R@1 {summary['recall_at_1']:.3f}; R@5 {summary['recall_at_5']:.3f}", flush=True)

    averaged = {}
    for shots, runs in run_summaries.items():
        averaged[shots] = {
            key: mean_sd([float(run[key]) for run in runs])
            for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "macro_recall_at_1")
        }
        averaged[shots]["supported_code_count"] = min(run["supported_code_count"] for run in runs)

    # Per-code means over all seeds at each shot count.
    per_code = []
    for code in sorted(test_counts):
        row = {"sign_code": code, "query_count": test_counts[code], "support_count": support_counts.get(code, 0)}
        selected_base = [r for r, q in zip(baseline_ranks, tests) if q["code"] == code]
        row["baseline_recall_at_1"] = float(np.mean(np.asarray(selected_base) <= 1))
        for shots in available_ks:
            values = []
            for seed in SEEDS:
                selected_rows = [q for q in query_rows if q["gold_code"] == code and q["shots_per_code"] == shots and q["seed"] == seed]
                if selected_rows:
                    values.append(float(np.mean([q["fewshot_rank"] == 1 for q in selected_rows])))
            row[f"{shots}_shot_recall_at_1_mean"] = float(np.mean(values)) if values else None
        per_code.append(row)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "support_dir": str(support_dir),
        "test_dir": str(test_dir),
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "device": device,
        "baseline_collection": args.base_collection,
        "baseline_points": len(base_codes),
        "test_class_count": len(test_counts),
        "support_image_count_after_dedup": len(supports),
        "support_counts_by_code": support_counts,
        "test_queries_after_leakage_filter": len(tests),
        "test_counts_by_code": test_counts,
        "test_images_excluded_for_shared_source_frame_or_exact_duplicate": len(excluded),
        "exact_content_matches_found_before_exclusion": overlap_exact,
        "missing_support_codes": classes_missing_support,
        "fewshot_seeds": list(SEEDS),
        "fewshot_results_mean_sd": averaged,
        "baseline_lawdb_only": baseline_metrics,
        "per_code_recall_at_1": per_code,
        "protocol": "Support is restricted to labels in the tested folder, deduplicated to one image per code/source frame. Test images sharing any original source-frame basename with support are excluded. Each K-shot seed selects K support images per class from distinct source frames. Scores merge by max cosine per sign_code over read-only LawDB and few-shot cache.",
        "runtime": {
            "support_embedding_seconds": round(support_embed_seconds, 3),
            "test_embedding_seconds": round(test_embed_seconds, 3),
            "support_embedding_ms_per_image": round(1000 * support_embed_seconds / max(1, len(supports)), 3),
            "test_embedding_ms_per_image": round(1000 * test_embed_seconds / max(1, len(tests)), 3),
        },
        "limitations": [
            "P.127 and P.127 numeric variants are grouped as the parent identity P.127.",
            "The gallery has only about eight examples per sign; tested signs with fewer than K examples are not few-shot boosted at that K and fall back to LawDB ranking.",
            "This is retrieval-only on pre-cropped signs; it does not evaluate sign detection.",
            "Source-frame grouping uses the filename convention before __line; other near-duplicate scenes without shared names may remain.",
            "No model weights are fine-tuned; this is a nearest-example cache baseline.",
        ],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "query_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(query_rows[0]) if query_rows else []
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(query_rows)
    with (args.output_dir / "per_code.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_code[0]))
        writer.writeheader()
        writer.writerows(per_code)

    lines = [
        "# Few-shot retrieval: all sign crops vs tested sign crops", "",
        f"- Encoder: `{model_name}` on `{device}`; baseline LawDB has {len(base_codes)} points.",
        f"- Few-shot support: {len(supports)} images across {len(support_counts)} labels; held-out queries: {len(tests)} across {len(test_counts)} labels.",
        f"- Excluded {len(excluded)} test crops that shared an original source frame with support ({overlap_exact} exact-image hash matches).",
        f"- Each result averages {len(SEEDS)} random support selections. No model fine-tuning; the Qdrant collection is read-only.", "",
        "| Retrieval | R@1 | R@3 | R@5 | MRR | Macro R@1 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| LawDB only | {baseline_metrics['recall_at_1']:.3f} | {baseline_metrics['recall_at_3']:.3f} | {baseline_metrics['recall_at_5']:.3f} | {baseline_metrics['mrr']:.3f} | {baseline_metrics['macro_recall_at_1']:.3f} |",
    ]
    for shots, values in averaged.items():
        lines.append(
            f"| LawDB + {shots}-shot cache ({values['supported_code_count']} labels; mean ± SD) | "
            + " | ".join(f"{values[k]['mean']:.3f} ± {values[k]['std']:.3f}" for k in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "macro_recall_at_1"))
            + " |"
        )
    lines.extend(["", "## Per-code support and top-1 results", "", "| Sign | Support | Queries | LawDB R@1 | Few-shot R@1 by K |", "|---|---:|---:|---:|---|"])
    for row in per_code:
        ks = ", ".join(f"{k}: {row[f'{k}_shot_recall_at_1_mean']:.3f}" for k in available_ks if row.get(f"{k}_shot_recall_at_1_mean") is not None)
        lines.append(f"| {row['sign_code']} | {row['support_count']} | {row['query_count']} | {row['baseline_recall_at_1']:.3f} | {ks} |")
    lines.extend(["", "## Notes", "", *[f"- {item}" for item in report["limitations"]]])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
