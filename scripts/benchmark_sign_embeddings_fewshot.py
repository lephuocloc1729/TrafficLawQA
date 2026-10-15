"""Few-shot evaluation of frozen traffic-sign image embeddings.

Evaluates nearest-exemplar and class-prototype matching with K examples per
sign code. This deliberately avoids full-backbone fine-tuning on tiny datasets.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_sign_embeddings import (
    DEFAULT_MODELS,
    choose_device,
    load_dataset as load_project_lawdb_dataset,
    load_folder_dataset,
)
from src.deps.embeddings import embedding_model_settings, get_embedding_model


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, 1e-12)


def build_episodes(
    records: list[dict[str, Any]],
    shots: int,
    episode_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_code: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_code[row["code"]].append(index)
    insufficient = {code: len(indices) for code, indices in by_code.items() if len(indices) <= shots}
    if insufficient:
        raise ValueError(f"Each label needs more than {shots} images; insufficient: {insufficient}")

    rng = np.random.default_rng(seed + shots)
    episodes = []
    for episode_id in range(episode_count):
        supports: dict[str, list[int]] = {}
        queries: list[int] = []
        for code in sorted(by_code):
            indices = np.asarray(by_code[code], dtype=np.int64)
            chosen = rng.choice(indices, size=shots, replace=False)
            support_set = set(int(i) for i in chosen)
            supports[code] = sorted(support_set)
            queries.extend(int(i) for i in indices if int(i) not in support_set)
        episodes.append({"episode_id": episode_id, "supports": supports, "queries": queries})
    return episodes


def build_project_lawdb_episodes(
    records: list[dict[str, Any]],
    repeats_per_figure: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    """One-shot episodes with whole LawDB source figures held out.

    A single random gallery image is selected per code. Only codes with at
    least two distinct source figures can be queried. Singleton codes remain
    gallery-only distractors when their source figure is not held out.
    """
    by_code: dict[str, list[int]] = defaultdict(list)
    groups_by_code: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(records):
        by_code[row["code"]].append(index)
        groups_by_code[row["code"]].add(row["source_group"])
    eligible = sorted(code for code, groups in groups_by_code.items() if len(groups) >= 2)
    eligible_counts = {code: len(groups_by_code[code]) for code in eligible}
    heldout_groups = sorted(
        {records[index]["source_group"] for code in eligible for index in by_code[code]}
    )
    rng = np.random.default_rng(seed)
    episodes = []
    for heldout_group in heldout_groups:
        queries = [
            index for code in eligible for index in by_code[code]
            if records[index]["source_group"] == heldout_group
        ]
        if not queries:
            continue
        gallery_by_code: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(records):
            if row["source_group"] != heldout_group:
                gallery_by_code[row["code"]].append(index)
        for repeat in range(repeats_per_figure):
            supports = {
                code: [int(rng.choice(indices))]
                for code, indices in sorted(gallery_by_code.items())
            }
            episodes.append(
                {
                    "episode_id": f"{heldout_group}__{repeat}",
                    "heldout_group": heldout_group,
                    "supports": supports,
                    "queries": queries,
                }
            )
    return episodes, eligible, eligible_counts


def score_episode(
    records: list[dict[str, Any]],
    vectors: np.ndarray,
    episode: dict[str, Any],
    method: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    codes = sorted(episode["supports"])
    support_matrix: dict[str, np.ndarray] = {}
    prototypes: dict[str, np.ndarray] = {}
    for code in codes:
        support_matrix[code] = vectors[episode["supports"][code]]
        prototypes[code] = normalize(support_matrix[code].mean(axis=0, keepdims=True))[0]

    query_rows = []
    correct_by_code: dict[str, list[int]] = defaultdict(list)
    for index in episode["queries"]:
        query_vector = vectors[index]
        scores: dict[str, float] = {}
        for code in codes:
            if method == "prototype":
                scores[code] = float(query_vector @ prototypes[code])
            else:
                scores[code] = float(np.max(support_matrix[code] @ query_vector))
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        gold = records[index]["code"]
        rank = next(position for position, (code, _) in enumerate(ranked, start=1) if code == gold)
        is_correct = int(rank == 1)
        correct_by_code[gold].append(is_correct)
        query_rows.append(
            {
                "episode_id": episode["episode_id"],
                "query_image": records[index]["image_name"],
                "query_path": records[index]["image_path"],
                "query_source_group": records[index].get("source_group"),
                "gold_code": gold,
                "rank": rank,
                "top1_code": ranked[0][0],
                "top1_score": ranked[0][1],
                "top3": [{"code": code, "score": score} for code, score in ranked[:3]],
            }
        )

    class_accuracy = [float(np.mean(values)) for values in correct_by_code.values()]
    metrics = {
        "top1": float(np.mean([row["rank"] == 1 for row in query_rows])),
        "top3": float(np.mean([row["rank"] <= 3 for row in query_rows])),
        "top5": float(np.mean([row["rank"] <= 5 for row in query_rows])),
        "macro_top1": float(np.mean(class_accuracy)),
        "mrr": float(np.mean([1 / row["rank"] for row in query_rows])),
    }
    return metrics, query_rows


def summarize_episode_metrics(
    model: str,
    shots: int,
    method: str,
    metrics_by_episode: list[dict[str, float]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "shots_per_code": shots,
        "matcher": method,
        "episode_count": len(metrics_by_episode),
        "query_count_per_episode": None,
    }
    for metric in ("top1", "top3", "top5", "macro_top1", "mrr"):
        values = np.asarray([entry[metric] for entry in metrics_by_episode], dtype=np.float64)
        row[metric + "_mean"] = float(values.mean())
        row[metric + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--dataset-dir", type=Path)
    dataset_group.add_argument(
        "--project-lawdb",
        action="store_true",
        help="Use the project's parsed LawDB sign images and source-figure groups",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", default="outputs/fewshot_embedding_benchmark_20261008")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()

    if args.dataset_dir is not None and not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_dir}")
    device = choose_device(args.device)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_project_lawdb_dataset() if args.project_lawdb else load_folder_dataset(args.dataset_dir)
    if not records:
        parser.error("No labeled images were found in the selected dataset")
    class_counts = defaultdict(int)
    for record in records:
        class_counts[record["code"]] += 1

    eligible_codes = None
    eligible_source_groups = None
    if args.project_lawdb:
        if args.shots != [1]:
            parser.error("The project LawDB corpus supports only --shots 1 for a comparable source-figure-held-out few-shot test")
        project_episodes, eligible_codes, eligible_source_groups = build_project_lawdb_episodes(
            records, args.episodes, args.seed
        )
        episodes_by_shot = {1: project_episodes}
    else:
        episodes_by_shot = {
            shots: build_episodes(records, shots, args.episodes, args.seed)
            for shots in args.shots
        }
    if not any(episodes_by_shot.values()):
        parser.error("No eligible few-shot episodes could be built from this dataset")
    dataset_path = str(args.dataset_dir.resolve()) if args.dataset_dir is not None else str(
        Path("data/raw/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json").resolve()
    )
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": dataset_path,
        "candidate_image_count": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "device": device,
        "models_requested": args.models,
        "shots": args.shots,
        "episode_count_per_shot": args.episodes,
        "eligible_query_codes": eligible_codes,
        "independent_source_figure_counts_for_eligible_codes": eligible_source_groups,
        "episode_split": (
            "hold out one entire source figure; choose one support image per available code from other figures"
            if args.project_lawdb else
            "random per-image support/query split; physical sign-instance IDs are unavailable"
        ),
        "random_seed": args.seed,
        "matcher_methods": ["prototype", "nearest_exemplar"],
        "backbone_finetuning_performed": False,
        "limitations": (
            [
                "LawDB source figures are illustrations, not independent real-road sign installations.",
                "Only codes with at least two independent source figures are query-evaluated; other codes are gallery-only distractors.",
                "No unknown-sign examples were supplied; open-set rejection is not evaluated.",
            ]
            if args.project_lawdb else
            [
                "Only ten images per code were supplied; results are few-shot pilot estimates, not robust deployment accuracy.",
                "The dataset has no physical sign-instance or capture-sequence IDs, so repeated views may leak across support/query.",
                "No unknown-sign examples were supplied; open-set rejection is not evaluated.",
            ]
        ),
        "model_results": [],
        "model_errors": [],
    }
    results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for model_name in args.models:
        if model_name not in embedding_model_settings:
            manifest["model_errors"].append({"model": model_name, "error": "Unsupported embedding model name"})
            continue
        print(f"[{model_name}] loading on {device}", flush=True)
        embedder = None
        try:
            load_start = time.perf_counter()
            embedder, dim = get_embedding_model(model_name, device=device)
            load_seconds = time.perf_counter() - load_start
            vectors_list = []
            embed_start = time.perf_counter()
            for index, row in enumerate(records, start=1):
                with Image.open(row["image_path"]) as image:
                    vector = embedder.embed_image(image.convert("RGB"))
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if vector.shape[0] != dim:
                    raise ValueError(f"Expected embedding dim {dim}, got {vector.shape[0]}")
                vectors_list.append(vector)
                if index % 50 == 0 or index == len(records):
                    print(f"  embedded {index}/{len(records)}", flush=True)
            vectors = np.stack(vectors_list)
            embed_seconds = time.perf_counter() - embed_start
            model_rows = []
            prediction_rows = []
            for shots, episodes in episodes_by_shot.items():
                for method in ("prototype", "nearest_exemplar"):
                    episode_metrics = []
                    for episode in episodes:
                        metrics, predictions = score_episode(records, vectors, episode, method)
                        episode_metrics.append(metrics)
                        for prediction in predictions:
                            prediction_rows.append(
                                {
                                    "model": model_name,
                                    "shots_per_code": shots,
                                    "matcher": method,
                                    **prediction,
                                }
                            )
                    summary = summarize_episode_metrics(model_name, shots, method, episode_metrics)
                    if args.project_lawdb:
                        # Pool query predictions so a source figure containing
                        # one query does not receive the same weight as one
                        # containing many. Estimate support-set variability by
                        # resampling gallery exemplars for each held-out query.
                        by_query: dict[str, list[int]] = defaultdict(list)
                        by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
                        for prediction in prediction_rows:
                            if prediction["model"] != model_name or prediction["matcher"] != method:
                                continue
                            by_query[prediction["query_path"]].append(int(prediction["rank"] == 1))
                            by_code[prediction["gold_code"]].append(prediction)
                        pooled = [
                            prediction for prediction in prediction_rows
                            if prediction["model"] == model_name and prediction["matcher"] == method
                        ]
                        summary["episode_weighted_top1_mean"] = summary["top1_mean"]
                        summary["episode_weighted_top1_std"] = summary["top1_std"]
                        summary["top1_mean"] = float(np.mean([row["rank"] == 1 for row in pooled]))
                        summary["top3_mean"] = float(np.mean([row["rank"] <= 3 for row in pooled]))
                        summary["top5_mean"] = float(np.mean([row["rank"] <= 5 for row in pooled]))
                        summary["mrr_mean"] = float(np.mean([1 / row["rank"] for row in pooled]))
                        summary["macro_top1_mean"] = float(np.mean([
                            np.mean([row["rank"] == 1 for row in code_rows])
                            for code_rows in by_code.values()
                        ]))
                        summary["top1_std"] = float(np.mean([
                            np.std(correctness, ddof=1) if len(correctness) > 1 else 0.0
                            for correctness in by_query.values()
                        ]))
                        summary["unique_query_count"] = len(by_query)
                        summary["prediction_count"] = len(pooled)
                        summary["per_code_top1"] = {
                            code: {
                                "unique_queries": len({row["query_path"] for row in code_rows}),
                                "top1": float(np.mean([row["rank"] == 1 for row in code_rows])),
                            }
                            for code, code_rows in sorted(by_code.items())
                        }
                    else:
                        summary["query_count_per_episode"] = len(episodes[0]["queries"])
                    summary["embedding_dim"] = int(vectors.shape[1])
                    summary["parameters"] = int(embedder.parameters())
                    summary["model_load_seconds"] = round(load_seconds, 3)
                    summary["embedding_seconds"] = round(embed_seconds, 3)
                    summary["embedding_ms_per_image"] = round(1000 * embed_seconds / len(records), 3)
                    model_rows.append(summary)
            results.extend(model_rows)
            all_predictions.extend(prediction_rows)
            manifest["model_results"].extend(model_rows)
            safe_name = model_name.replace("/", "_").replace(" ", "_")
            np.savez_compressed(
                output_dir / f"embeddings_{safe_name}.npz",
                embeddings=vectors,
                image_paths=np.array([row["image_path"] for row in records]),
                codes=np.array([row["code"] for row in records]),
            )
            del vectors, vectors_list
        except Exception as exc:
            manifest["model_errors"].append({"model": model_name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if embedder is not None:
                del embedder
            gc.collect()
            if device == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    if results:
        fieldnames = list(results[0])
        with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
    (output_dir / "predictions.json").write_text(
        json.dumps(all_predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = [
        "# Few-shot traffic-sign embedding benchmark",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Images: {len(records)} across {len(class_counts)} codes.",
        f"- Device: `{device}`; repeats per held-out source figure: {args.episodes}; seed: {args.seed}.",
        (
            f"- Queryable codes with multiple independent source figures: {len(eligible_codes or [])}; "
            f"counts: {eligible_source_groups}."
            if args.project_lawdb else
            "- Each episode samples K support images per code; remaining images are queries."
        ),
        "- Embedding backbones are frozen; this is few-shot matching, not gradient-based fine-tuning.",
        (
            "- Top-k scores are pooled over held-out query images and repeated one-shot gallery selections; the ± value is mean per-query variation across support resamples."
            if args.project_lawdb else
            "- Episode metrics are averaged across random support/query assignments; standard deviation is across episodes."
        ),
        (
            "- Limitation: LawDB source figures are independent illustrations, not independent real-road sign installations; "
            "single-figure codes are gallery-only and excluded from query evaluation."
            if args.project_lawdb else
            "- Limitation: physical sign-instance IDs and unknown-sign examples were not supplied."
        ),
        "",
        "| Model | Shots | Matcher | Top-1 mean ± SD | Top-3 | Top-5 | Macro top-1 | MRR | embed ms/image |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        report.append(
            f"| {row['model']} | {row['shots_per_code']} | {row['matcher']} | "
            f"{row['top1_mean']:.3f} ± {row['top1_std']:.3f} | {row['top3_mean']:.3f} | "
            f"{row['top5_mean']:.3f} | {row['macro_top1_mean']:.3f} | {row['mrr_mean']:.3f} | "
            f"{row['embedding_ms_per_image']:.1f} |"
        )
    if args.project_lawdb and results:
        prototype_rows = {row["model"]: row for row in results if row["matcher"] == "prototype"}
        report.extend([
            "", "## Per-code top-1", "",
            "| Code | Source figures | Unique queries | CLIP-GmP | DINOv2 | SigLIP2 |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for code in eligible_codes or []:
            accuracies = {
                model: prototype_rows[model]["per_code_top1"].get(code, {})
                for model in prototype_rows
            }
            query_count = next(
                (metrics.get("unique_queries", 0) for metrics in accuracies.values()), 0
            )
            report.append(
                f"| {code} | {eligible_source_groups[code]} | {query_count} | "
                f"{accuracies.get('CLIP-GmP-ViT-L-14', {}).get('top1', 0.0):.3f} | "
                f"{accuracies.get('dinov2-with-registers-large', {}).get('top1', 0.0):.3f} | "
                f"{accuracies.get('siglip2-base-patch16-224', {}).get('top1', 0.0):.3f} |"
            )
    if manifest["model_errors"]:
        report.extend(["", "## Model errors", ""])
        report.extend(f"- `{error['model']}`: {error['error']}" for error in manifest["model_errors"])
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
