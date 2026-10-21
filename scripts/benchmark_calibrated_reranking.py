"""Source-grouped cross-validation for few-shot score-fusion rerankers.

Reuses the frozen embeddings from benchmark_all_sign_crops_fewshot.py. The
tested_sign_crops set has already been inspected, so this is development/CV
evidence rather than a new independent final test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_all_sign_crops_fewshot import (
    SEEDS,
    canonical_code,
    embedding_fingerprint,
    fetch_lawdb_vectors,
    grouped_scores,
    load_folder,
    rank,
    score_metrics,
)
from src.settings import settings

ALPHAS_Z = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)
ALPHAS_RRF = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
ALPHAS_RAW = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
RRF_CONSTANT = 10.0


def load_rows(support_dir: Path, test_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supports = load_folder(support_dir)
    test_all = load_folder(test_dir)
    test_codes = {row["code"] for row in test_all}
    supports = [row for row in supports if row["code"] in test_codes]
    support_hashes = set()
    import hashlib

    for row in supports:
        support_hashes.add(hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest())
    support_groups = {row["source_group"] for row in supports}
    tests = []
    for row in test_all:
        image_hash = hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        if row["source_group"] in support_groups or image_hash in support_hashes:
            continue
        tests.append(row)
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in supports:
        dedup.setdefault((row["code"], row["source_group"]), row)
    supports = sorted(dedup.values(), key=lambda row: (row["code"], row["source_group"]))
    return supports, tests


def group_folds(groups: list[str], fold_count: int, seed: int = 917) -> list[set[str]]:
    values = sorted(set(groups))
    random.Random(seed).shuffle(values)
    folds = [set() for _ in range(fold_count)]
    for i, group in enumerate(values):
        folds[i % fold_count].add(group)
    return folds


def make_score_matrix(score_maps: list[dict[str, float]], codes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mat = np.full((len(score_maps), len(codes)), -1.0, dtype=np.float32)
    present = np.zeros(len(codes), dtype=bool)
    positions = {code: i for i, code in enumerate(codes)}
    for row_i, scores in enumerate(score_maps):
        for code, value in scores.items():
            if code in positions:
                mat[row_i, positions[code]] = value
                present[positions[code]] = True
    return mat, present


def zscore_rows(matrix: np.ndarray, present_columns: np.ndarray, missing_value: float = -4.0) -> np.ndarray:
    selected = matrix[:, present_columns]
    mean = selected.mean(axis=1, keepdims=True)
    std = selected.std(axis=1, keepdims=True)
    z = (matrix - mean) / np.maximum(std, 1e-6)
    z[:, ~present_columns] = missing_value
    return z.astype(np.float32, copy=False)


def ranks_from_scores(scores: np.ndarray, gold_indices: np.ndarray) -> np.ndarray:
    gold_scores = scores[np.arange(len(scores)), gold_indices]
    # Lexicographic tie-breaking by stable column order is effectively exact
    # equality only; cosine similarities rarely tie at float32 precision.
    return (1 + (scores > gold_scores[:, None]).sum(axis=1)).astype(np.int32)


def metrics_for_indices(ranks: np.ndarray, codes: list[str], indices: np.ndarray) -> dict[str, float]:
    subset = ranks[indices]
    subset_codes = [codes[i] for i in indices]
    return score_metrics(subset.tolist(), subset_codes)


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def choose_alpha(
    base_z: np.ndarray,
    cache_z: np.ndarray,
    gold_indices: np.ndarray,
    labels: list[str],
    train_indices: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, float]:
    best_alpha, best_macro = alphas[0], -1.0
    for alpha in alphas:
        ranks = ranks_from_scores(base_z[train_indices] + alpha * cache_z[train_indices], gold_indices[train_indices])
        metric = score_metrics(ranks.tolist(), [labels[i] for i in train_indices])
        objective = metric["macro_recall_at_1"]
        if objective > best_macro:
            best_alpha, best_macro = alpha, objective
    return best_alpha, best_macro


def choose_alpha_rrf(
    base_rank: np.ndarray,
    cache_rank: np.ndarray,
    gold_indices: np.ndarray,
    labels: list[str],
    train_indices: np.ndarray,
) -> tuple[float, float]:
    best_alpha, best_macro = ALPHAS_RRF[0], -1.0
    base_contrib = 1.0 / (RRF_CONSTANT + base_rank.astype(np.float32))
    cache_contrib = np.where(cache_rank > 0, 1.0 / (RRF_CONSTANT + cache_rank), 0.0).astype(np.float32)
    for alpha in ALPHAS_RRF:
        fused = base_contrib[train_indices] + alpha * cache_contrib[train_indices]
        ranks = ranks_from_scores(fused, gold_indices[train_indices])
        metric = score_metrics(ranks.tolist(), [labels[i] for i in train_indices])
        objective = metric["macro_recall_at_1"]
        if objective > best_macro:
            best_alpha, best_macro = alpha, objective
    return best_alpha, best_macro


def choose_alpha_delta(
    baseline: np.ndarray,
    adjustment: np.ndarray,
    gold_indices: np.ndarray,
    labels: list[str],
    train_indices: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[float, float]:
    best_alpha, best_macro = alphas[0], -1.0
    for alpha in alphas:
        ranks = ranks_from_scores(
            baseline[train_indices] + alpha * adjustment[train_indices],
            gold_indices[train_indices],
        )
        metric = score_metrics(ranks.tolist(), [labels[i] for i in train_indices])
        if metric["macro_recall_at_1"] > best_macro:
            best_alpha, best_macro = alpha, metric["macro_recall_at_1"]
    return best_alpha, best_macro


def choose_confidence_gate(
    baseline_ranks: np.ndarray,
    cache_ranks: np.ndarray,
    cache_scores: np.ndarray,
    gold_indices: np.ndarray,
    labels: list[str],
    train_indices: np.ndarray,
    present_columns: np.ndarray,
) -> tuple[float, float, float]:
    """Choose when to trust the few-shot class ranking, using training folds only."""
    present_ids = np.flatnonzero(present_columns)
    top_scores = cache_scores[:, present_ids]
    top_order = np.argsort(-top_scores, axis=1, kind="stable")
    top1 = top_scores[np.arange(len(cache_scores)), top_order[:, 0]]
    top2 = top_scores[np.arange(len(cache_scores)), top_order[:, 1]]
    margin = top1 - top2
    train_labels = np.asarray([labels[i] for i in train_indices])
    _, label_group = np.unique(train_labels, return_inverse=True)
    label_counts = np.bincount(label_group)
    best = (0.0, 0.0, -1.0)
    for min_score in np.arange(0.20, 0.801, 0.02):
        for min_margin in np.arange(0.0, 0.301, 0.01):
            use_cache = (top1[train_indices] >= min_score) & (margin[train_indices] >= min_margin)
            correct = np.where(use_cache, cache_ranks[train_indices], baseline_ranks[train_indices]) == 1
            correct_by_label = np.bincount(label_group, weights=correct, minlength=len(label_counts))
            objective = float(np.mean(correct_by_label / label_counts))
            if objective > best[2]:
                best = (float(min_score), float(min_margin), objective)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-dir", type=Path, default=ROOT / "data/all-traffic-sign-crops")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "data/tested_sign_crops")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "outputs/all_sign_fewshot_20261019")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/calibrated_reranking_20261019")
    parser.add_argument("--base-collection", default=settings.collection_name)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    supports, tests = load_rows(args.support_dir.resolve(), args.test_dir.resolve())
    model_name = settings.embedding_name
    expected_fingerprint = embedding_fingerprint(model_name, supports, tests)
    metadata_path = args.embedding_dir / "embedding_cache.json"
    embeddings_path = args.embedding_dir / "embedding_cache.npz"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        parser.error("Expected cached embeddings from the all-sign-crops benchmark")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != expected_fingerprint:
        parser.error("Cached embeddings do not match the current support/test files")
    with np.load(embeddings_path) as cached:
        support_vectors = cached["support_vectors"]
        test_vectors = cached["test_vectors"]

    client = QdrantClient(url=settings.qdrant_uri, check_compatibility=False)
    if not client.collection_exists(args.base_collection):
        parser.error(f"Baseline collection unavailable: {args.base_collection}")
    base_vectors, base_codes = fetch_lawdb_vectors(client, args.base_collection)
    if base_vectors.shape[1] != test_vectors.shape[1]:
        parser.error("Embedding dimension mismatch")
    base_sim = test_vectors @ base_vectors.T
    base_maps = grouped_scores(base_sim, [canonical_code(code) for code in base_codes])
    codes = sorted({code for scores in base_maps for code in scores} | {row["code"] for row in supports})
    baseline_matrix, base_present = make_score_matrix(base_maps, codes)
    base_z = zscore_rows(baseline_matrix, base_present)
    support_by_code: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(supports):
        support_by_code[row["code"]].append(i)

    test_groups = [row["source_group"] for row in tests]
    folds = group_folds(test_groups, args.folds)
    fold_masks = [np.asarray([group in fold for group in test_groups]) for fold in folds]
    folds_by_query = np.zeros(len(tests), dtype=np.int32)
    for fold_i, mask in enumerate(fold_masks):
        folds_by_query[mask] = fold_i
    labels = [row["code"] for row in tests]
    label_to_i = {code: i for i, code in enumerate(codes)}
    gold_indices = np.asarray([label_to_i[code] for code in labels], dtype=np.int32)
    baseline_rank = ranks_from_scores(baseline_matrix, gold_indices)
    baseline_metric = score_metrics(baseline_rank.tolist(), labels)

    # Join support-to-query cosine similarities once, then evaluate every K and
    # support seed without another encoder or Qdrant call.
    support_sim = test_vectors @ support_vectors.T
    results_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    query_rows: list[dict[str, Any]] = []
    for shots in (4,):
        eligible = {code: ids for code, ids in support_by_code.items() if len(ids) >= shots}
        for seed in SEEDS:
            rng = random.Random(seed)
            selected = {code: rng.sample(ids, shots) for code, ids in eligible.items()}
            cache_matrix = np.full((len(tests), len(codes)), -1.0, dtype=np.float32)
            cache_mean_matrix = np.full((len(tests), len(codes)), -1.0, dtype=np.float32)
            cache_median_matrix = np.full((len(tests), len(codes)), -1.0, dtype=np.float32)
            cache_top2_matrix = np.full((len(tests), len(codes)), -1.0, dtype=np.float32)
            prototype_matrix = np.full((len(tests), len(codes)), -1.0, dtype=np.float32)
            cache_present = np.zeros(len(codes), dtype=bool)
            for code, ids in selected.items():
                class_similarities = support_sim[:, ids]
                cache_matrix[:, label_to_i[code]] = np.max(class_similarities, axis=1)
                cache_mean_matrix[:, label_to_i[code]] = np.mean(class_similarities, axis=1)
                cache_median_matrix[:, label_to_i[code]] = np.median(class_similarities, axis=1)
                cache_top2_matrix[:, label_to_i[code]] = np.sort(class_similarities, axis=1)[:, -2:].mean(axis=1)
                prototype = support_vectors[ids].mean(axis=0)
                prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
                prototype_matrix[:, label_to_i[code]] = test_vectors @ prototype
                cache_present[label_to_i[code]] = True
            cache_z = zscore_rows(cache_matrix, cache_present, missing_value=0.0)

            # The existing raw-max method is retained as a comparison.
            raw_scores = np.maximum(baseline_matrix, cache_matrix)
            raw_ranks = ranks_from_scores(raw_scores, gold_indices)
            results_by_method["raw_max_4shot"].append({"seed": seed, "ranks": raw_ranks})
            top2_scores = np.maximum(baseline_matrix, cache_top2_matrix)
            results_by_method["raw_max_top2mean_4shot"].append({
                "seed": seed,
                "ranks": ranks_from_scores(top2_scores, gold_indices),
            })
            mean_scores = np.maximum(baseline_matrix, cache_mean_matrix)
            results_by_method["raw_max_mean_4shot"].append({
                "seed": seed,
                "ranks": ranks_from_scores(mean_scores, gold_indices),
            })
            median_scores = np.maximum(baseline_matrix, cache_median_matrix)
            results_by_method["raw_max_median_4shot"].append({
                "seed": seed,
                "ranks": ranks_from_scores(median_scores, gold_indices),
            })
            prototype_scores = np.maximum(baseline_matrix, prototype_matrix)
            results_by_method["raw_max_prototype_4shot"].append({
                "seed": seed,
                "ranks": ranks_from_scores(prototype_scores, gold_indices),
            })

            centered_cache = np.zeros_like(cache_matrix)
            centered_cache[:, cache_present] = cache_matrix[:, cache_present] - cache_matrix[:, cache_present].mean(axis=1, keepdims=True)
            delta_cache = np.zeros_like(cache_matrix)
            delta_cache[:, cache_present] = cache_matrix[:, cache_present] - baseline_matrix[:, cache_present]
            centered_prototype = np.zeros_like(prototype_matrix)
            centered_prototype[:, cache_present] = (
                prototype_matrix[:, cache_present]
                - prototype_matrix[:, cache_present].mean(axis=1, keepdims=True)
            )

            # Rank cache candidates among supported classes; zero means no cache result.
            cache_rank = np.zeros_like(cache_matrix, dtype=np.int32)
            for row_i in range(len(tests)):
                ordered = np.argsort(-cache_matrix[row_i, cache_present], kind="stable")
                present_ids = np.flatnonzero(cache_present)
                cache_rank[row_i, present_ids[ordered]] = np.arange(1, len(present_ids) + 1)
            base_rank_matrix = np.empty_like(baseline_matrix, dtype=np.int32)
            for row_i in range(len(tests)):
                order = np.argsort(-baseline_matrix[row_i], kind="stable")
                base_rank_matrix[row_i, order] = np.arange(1, len(codes) + 1)

            # Fit alpha within the other four source-group folds; score only the
            # held-out fold. Macro Recall@1 is the calibration objective.
            z_predictions = np.zeros(len(tests), dtype=np.int32)
            rrf_predictions = np.zeros(len(tests), dtype=np.int32)
            centered_predictions = np.zeros(len(tests), dtype=np.int32)
            delta_predictions = np.zeros(len(tests), dtype=np.int32)
            prototype_predictions = np.zeros(len(tests), dtype=np.int32)
            gate_predictions = np.zeros(len(tests), dtype=np.int32)
            cache_only_rank_matrix = ranks_from_scores(
                np.where(cache_present[None, :], cache_matrix, baseline_matrix), gold_indices,
            )
            fold_choices = []
            for fold_i, held_mask in enumerate(fold_masks):
                held = np.flatnonzero(held_mask)
                train = np.flatnonzero(~held_mask)
                alpha_z, train_macro_z = choose_alpha(base_z, cache_z, gold_indices, labels, train, ALPHAS_Z)
                alpha_rrf, train_macro_rrf = choose_alpha_rrf(base_rank_matrix, cache_rank, gold_indices, labels, train)
                alpha_centered, train_macro_centered = choose_alpha_delta(
                    baseline_matrix, centered_cache, gold_indices, labels, train, ALPHAS_RAW,
                )
                alpha_delta, train_macro_delta = choose_alpha_delta(
                    baseline_matrix, delta_cache, gold_indices, labels, train, ALPHAS_RAW,
                )
                alpha_prototype, train_macro_prototype = choose_alpha_delta(
                    baseline_matrix, centered_prototype, gold_indices, labels, train, ALPHAS_RAW,
                )
                gate_score, gate_margin, train_macro_gate = choose_confidence_gate(
                    baseline_rank, cache_only_rank_matrix, cache_matrix, gold_indices,
                    labels, train, cache_present,
                )
                fused_z = base_z[held] + alpha_z * cache_z[held]
                z_predictions[held] = ranks_from_scores(fused_z, gold_indices[held])
                centered_predictions[held] = ranks_from_scores(
                    baseline_matrix[held] + alpha_centered * centered_cache[held], gold_indices[held],
                )
                delta_predictions[held] = ranks_from_scores(
                    baseline_matrix[held] + alpha_delta * delta_cache[held], gold_indices[held],
                )
                prototype_predictions[held] = ranks_from_scores(
                    baseline_matrix[held] + alpha_prototype * centered_prototype[held], gold_indices[held],
                )
                present_ids = np.flatnonzero(cache_present)
                ordered_scores = np.sort(cache_matrix[held][:, present_ids], axis=1)
                cache_top1 = ordered_scores[:, -1]
                cache_margin = ordered_scores[:, -1] - ordered_scores[:, -2]
                use_cache = (cache_top1 >= gate_score) & (cache_margin >= gate_margin)
                gate_predictions[held] = np.where(
                    use_cache,
                    cache_only_rank_matrix[held],
                    baseline_rank[held],
                )
                base_contrib = 1.0 / (RRF_CONSTANT + base_rank_matrix[held].astype(np.float32))
                cache_contrib = np.where(cache_rank[held] > 0, 1.0 / (RRF_CONSTANT + cache_rank[held]), 0.0).astype(np.float32)
                fused_rrf = base_contrib + alpha_rrf * cache_contrib
                rrf_predictions[held] = ranks_from_scores(fused_rrf, gold_indices[held])
                fold_choices.append({
                    "fold": fold_i,
                    "heldout_queries": len(held),
                    "calibration_queries": len(train),
                    "zscore_alpha": alpha_z,
                    "zscore_calibration_macro_recall_at_1": train_macro_z,
                    "rrf_alpha": alpha_rrf,
                    "rrf_calibration_macro_recall_at_1": train_macro_rrf,
                    "centered_alpha": alpha_centered,
                    "centered_calibration_macro_recall_at_1": train_macro_centered,
                    "delta_alpha": alpha_delta,
                    "delta_calibration_macro_recall_at_1": train_macro_delta,
                    "prototype_alpha": alpha_prototype,
                    "prototype_calibration_macro_recall_at_1": train_macro_prototype,
                    "gate_min_score": gate_score,
                    "gate_min_margin": gate_margin,
                    "gate_calibration_macro_recall_at_1": train_macro_gate,
                })
            results_by_method["zscore_fusion_tuned"].append({"seed": seed, "ranks": z_predictions, "fold_choices": fold_choices})
            results_by_method["rrf_fusion_tuned"].append({"seed": seed, "ranks": rrf_predictions, "fold_choices": fold_choices})
            results_by_method["centered_raw_fusion_tuned"].append({"seed": seed, "ranks": centered_predictions, "fold_choices": fold_choices})
            results_by_method["delta_raw_fusion_tuned"].append({"seed": seed, "ranks": delta_predictions, "fold_choices": fold_choices})
            results_by_method["prototype_fusion_tuned"].append({"seed": seed, "ranks": prototype_predictions, "fold_choices": fold_choices})
            results_by_method["confidence_gate_tuned"].append({"seed": seed, "ranks": gate_predictions, "fold_choices": fold_choices})

    method_summaries = {
        "lawdb_only": {key: baseline_metric[key] for key in ("query_count", "recall_at_1", "recall_at_3", "recall_at_5", "mrr", "macro_recall_at_1")}
    }
    seed_summaries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for method, runs in results_by_method.items():
        for run in runs:
            metrics = score_metrics(run["ranks"].tolist(), labels)
            metrics["seed"] = run["seed"]
            seed_summaries[method].append(metrics)
        keys = ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "macro_recall_at_1")
        method_summaries[method] = {key: mean_sd([float(run[key]) for run in seed_summaries[method]]) for key in keys}

    # Preserve per-query, out-of-fold predictions for inspection.
    for method, runs in results_by_method.items():
        for run in runs:
            seed = run["seed"]
            for i, row in enumerate(tests):
                query_rows.append({
                    "method": method,
                    "seed": seed,
                    "query_image": row["image_name"],
                    "source_group": row["source_group"],
                    "gold_code": row["code"],
                    "fold": int(folds_by_query[i]),
                    "rank": int(run["ranks"][i]),
                })

    selected_params = {
        method: [
            {"seed": run["seed"], "fold_choices": run.get("fold_choices", [])}
            for run in runs if run.get("fold_choices")
        ]
        for method, runs in results_by_method.items()
    }
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "5-fold source_group cross-validation; for each held-out fold, tune alpha on the other folds by macro Recall@1; balanced four-shot supports sampled from all-traffic-sign-crops; cached CLIP embeddings; only held-out folds contribute to the tuned reranker metrics.",
        "support_count": len(supports),
        "query_count": len(tests),
        "source_group_count": len(set(test_groups)),
        "fold_count": args.folds,
        "support_seeds": list(SEEDS),
        "embedding_model": model_name,
        "baseline_collection": args.base_collection,
        "baseline_point_count": len(base_codes),
        "support_coverage_4shot": len([1 for ids in support_by_code.values() if len(ids) >= 4]),
        "methods_mean_sd_over_support_seeds": method_summaries,
        "selected_parameters": selected_params,
        "warnings": [
            "This dataset has already been inspected in earlier benchmark reports; treat this as development evidence, not a pristine final test.",
            "The few-shot cache contains only signs represented in the support folder; unsupported labels use LawDB-only scores.",
            "Source grouping follows the filename prefix before __line; unrelated near-duplicate scenes may remain.",
        ],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir / "oof_query_ranks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)
    lines = [
        "# Calibrated reranking cross-validation", "",
        f"- {len(tests)} source-disjoint queries, {len(supports)} support images, {args.folds} folds; alpha tuned on other folds by macro Recall@1.",
        "- Results are development evidence because this dataset has already been inspected.", "",
        "| Method | R@1 | R@3 | R@5 | MRR | Macro R@1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, label in (
        ("lawdb_only", "LawDB only"),
        ("raw_max_4shot", "Current raw-max 4-shot"),
        ("raw_max_top2mean_4shot", "Raw max + top-2-mean cache"),
        ("raw_max_mean_4shot", "LawDB max + mean support similarity"),
        ("raw_max_median_4shot", "LawDB max + median support similarity"),
        ("raw_max_prototype_4shot", "Raw max + class prototype"),
        ("centered_raw_fusion_tuned", "Tuned centered raw fusion"),
        ("delta_raw_fusion_tuned", "Tuned baseline/cache delta"),
        ("prototype_fusion_tuned", "Tuned class-prototype fusion"),
        ("confidence_gate_tuned", "Tuned few-shot confidence gate"),
        ("zscore_fusion_tuned", "Tuned z-score fusion"),
        ("rrf_fusion_tuned", "Tuned reciprocal-rank fusion"),
    ):
        metrics = method_summaries[method]
        fmt = lambda key: f"{metrics[key]['mean']:.3f} ± {metrics[key]['std']:.3f}" if isinstance(metrics[key], dict) else f"{metrics[key]:.3f}"
        lines.append(f"| {label} | {fmt('recall_at_1')} | {fmt('recall_at_3')} | {fmt('recall_at_5')} | {fmt('mrr')} | {fmt('macro_recall_at_1')} |")
    lines.extend(["", "## Selected alpha by fold and support seed", ""])
    for method in ("centered_raw_fusion_tuned", "delta_raw_fusion_tuned", "prototype_fusion_tuned", "confidence_gate_tuned", "zscore_fusion_tuned", "rrf_fusion_tuned"):
        lines.append(f"### {method}")
        for run in selected_params[method]:
            key = {
                "centered_raw_fusion_tuned": "centered_alpha",
                "delta_raw_fusion_tuned": "delta_alpha",
                "prototype_fusion_tuned": "prototype_alpha",
                "confidence_gate_tuned": "gate_min_score",
                "zscore_fusion_tuned": "zscore_alpha",
                "rrf_fusion_tuned": "rrf_alpha",
            }[method]
            if method == "confidence_gate_tuned":
                chosen = ", ".join(
                    f"F{x['fold']}=score≥{x['gate_min_score']:.2f}/margin≥{x['gate_min_margin']:.2f}"
                    for x in run["fold_choices"]
                )
            else:
                chosen = ", ".join(f"F{x['fold']}={x[key]}" for x in run["fold_choices"])
            lines.append(f"- Seed {run['seed']}: {chosen}")
    lines.extend(["", "## Caveats", "", *[f"- {item}" for item in report["warnings"]]])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written: {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
