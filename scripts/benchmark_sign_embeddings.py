"""Leakage-aware image-to-image benchmark for the LawDB traffic-sign crops.

Runs frozen encoders only. The current LawDB snapshot has too few independent
examples per QCVN code for defensible supervised fine-tuning; the report records
that limitation instead of treating augmented copies as independent data.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.deps.embeddings import embedding_model_settings, get_embedding_model


LAWDB_JSON = ROOT / "data/raw/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json"
SIGN_DIR = ROOT / "data/raw/law_db/signs_imageprocessing"
CODE_RE = re.compile(r"\b((?:DP|IE|P|W|R|I|S)\.?\s*\d{1,3}[a-z]?)\b", re.IGNORECASE)
DEFAULT_MODELS = [
    "CLIP-GmP-ViT-L-14",
    "dinov2-with-registers-large",
    "siglip2-base-patch16-224",
]


def normalize_code(value: str) -> str:
    match = re.fullmatch(r"([A-Za-z]+)\.?\s*(\d+)([A-Za-z]?)", value.strip())
    if not match:
        return value.strip()
    prefix, number, variant = match.groups()
    return f"{prefix.upper()}.{number}{variant.lower()}"


def code_from_record(article: dict[str, Any], sign: dict[str, Any]) -> str | None:
    direct = str(sign.get("sign_code") or sign.get("code") or "").strip()
    if direct:
        return normalize_code(direct)
    name_codes = list(dict.fromkeys(CODE_RE.findall(str(sign.get("name") or ""))))
    if len(name_codes) == 1:
        return normalize_code(name_codes[0])
    if name_codes:
        return None
    title_codes = list(dict.fromkeys(CODE_RE.findall(str(article.get("title") or ""))))
    if len(title_codes) == 1:
        return normalize_code(title_codes[0])
    return None


def load_dataset(max_images: int = 0) -> list[dict[str, Any]]:
    raw = json.loads(LAWDB_JSON.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for law in raw:
        if not isinstance(law, dict):
            continue
        for article in law.get("articles", []):
            for sign in article.get("detailed_signs", []):
                code = code_from_record(article, sign)
                image_name = str(sign.get("image") or "").strip()
                image_path = SIGN_DIR / image_name
                key = (code or "", image_name)
                if not code or not image_name or not image_path.is_file() or key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "image_name": image_name,
                        "image_path": str(image_path),
                        "code": code,
                        "law_id": str(law.get("id") or ""),
                        "article_id": str(article.get("id") or ""),
                        # All crops from one source figure/page stay in one split.
                        "source_group": image_name.split("_crop", 1)[0],
                    }
                )
    records.sort(key=lambda row: (row["code"], row["source_group"], row["image_name"]))
    return records[:max_images] if max_images else records


def load_folder_dataset(dataset_dir: Path, max_images: int = 0) -> list[dict[str, Any]]:
    """Load images from one-label-per-folder datasets.

    Layout: ``dataset_dir/<sign_code>/<image>``. Each file is treated as a
    separate instance because this format has no physical-instance metadata.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    records: list[dict[str, Any]] = []
    for label_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        code = normalize_code(label_dir.name)
        for image_path in sorted(
            path for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ):
            records.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path.resolve()),
                    "code": code,
                    "source_group": image_path.stem,
                    "source_type": "folder_dataset",
                }
            )
    records.sort(key=lambda row: (row["code"], row["image_name"], row["image_path"]))
    return records[:max_images] if max_images else records


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_folds(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups_by_code: dict[str, set[str]] = defaultdict(set)
    for row in records:
        groups_by_code[row["code"]].add(row["source_group"])
    folds = []
    for code, groups in sorted(groups_by_code.items()):
        if len(groups) < 2:
            continue
        for heldout_group in sorted(groups):
            queries = [
                row for row in records
                if row["code"] == code and row["source_group"] == heldout_group
            ]
            gallery = [row for row in records if row["source_group"] != heldout_group]
            if queries and any(row["code"] == code for row in gallery):
                folds.append(
                    {
                        "gold_code": code,
                        "heldout_group": heldout_group,
                        "queries": queries,
                        "gallery": gallery,
                    }
                )
    return folds


def evaluate_model(
    model_name: str,
    records: list[dict[str, Any]],
    embeddings: np.ndarray,
    elapsed_sec: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index_by_path = {row["image_path"]: index for index, row in enumerate(records)}
    folds = make_folds(records)
    per_query: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        gallery = fold["gallery"]
        gallery_indices = np.array([index_by_path[row["image_path"]] for row in gallery])
        gallery_embeddings = embeddings[gallery_indices]
        # Dot product is cosine because every encoder output is L2-normalized.
        for query in fold["queries"]:
            query_embedding = embeddings[index_by_path[query["image_path"]]]
            image_scores = gallery_embeddings @ query_embedding
            best_by_code: dict[str, float] = {}
            for row, score in zip(gallery, image_scores):
                best_by_code[row["code"]] = max(
                    best_by_code.get(row["code"], -1.0), float(score)
                )
            ranked = sorted(best_by_code.items(), key=lambda item: item[1], reverse=True)
            rank = next(
                (position for position, (candidate, _) in enumerate(ranked, start=1)
                 if candidate == fold["gold_code"]),
                len(ranked) + 1,
            )
            per_query.append(
                {
                    "model": model_name,
                    "query_image": query["image_name"],
                    "source_group": query["source_group"],
                    "gold_code": fold["gold_code"],
                    "rank": rank,
                    "top1_code": ranked[0][0] if ranked else None,
                    "top1_score": ranked[0][1] if ranked else None,
                    "top2_code": ranked[1][0] if len(ranked) > 1 else None,
                    "top2_score": ranked[1][1] if len(ranked) > 1 else None,
                    "top5": [
                        {"code": code, "score": round(score, 6)}
                        for code, score in ranked[:5]
                    ],
                    "fold": fold_index,
                    "gallery_code_count": len(best_by_code),
                }
            )

    recalls = {
        f"recall_at_{k}": float(np.mean([row["rank"] <= k for row in per_query]))
        if per_query else None
        for k in (1, 3, 5)
    }
    codes = sorted({row["gold_code"] for row in per_query})
    macro_recall_at_1 = []
    for code in codes:
        code_rows = [row for row in per_query if row["gold_code"] == code]
        macro_recall_at_1.append(float(np.mean([row["rank"] == 1 for row in code_rows])))
    metrics = {
        "model": model_name,
        "candidate_image_count": len(records),
        "eligible_exact_labels": len(codes),
        "query_count": len(per_query),
        **recalls,
        "macro_recall_at_1": float(np.mean(macro_recall_at_1)) if macro_recall_at_1 else None,
        "mrr": float(np.mean([1 / row["rank"] for row in per_query])) if per_query else None,
        "embedding_seconds": round(elapsed_sec, 3),
        "embedding_ms_per_image": round(1000 * elapsed_sec / max(1, len(records)), 3),
        "fine_tuned": False,
    }
    return metrics, per_query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output-dir", default="outputs/embedding_benchmark_20261007")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Optional folder dataset with one sign-code subfolder per label",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Smoke-test limit; 0 uses all images")
    args = parser.parse_args()

    device = choose_device(args.device)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset_dir is not None and not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_dir}")
    records = (
        load_folder_dataset(args.dataset_dir, args.max_images)
        if args.dataset_dir is not None
        else load_dataset(args.max_images)
    )
    folds = make_folds(records)
    code_groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        code_groups[record["code"]].add(record["source_group"])
    trainable_codes = sum(len(groups) >= 3 for groups in code_groups.values())
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_json": str(LAWDB_JSON.relative_to(ROOT)) if args.dataset_dir is None else None,
        "image_dir": str(SIGN_DIR.relative_to(ROOT)) if args.dataset_dir is None else str(args.dataset_dir.resolve()),
        "device": device,
        "split": (
            "leave-one-source-figure-out; all crops from a source figure are held out together"
            if args.dataset_dir is None
            else "leave-one-image-out; one image queried against all other images"
        ),
        "split_limitation": (
            None if args.dataset_dir is None else
            "No physical sign-instance IDs were provided. If multiple images show the same physical sign, "
            "they may occur in both query and gallery; interpret scores as a pilot."
        ),
        "candidate_image_count": len(records),
        "unique_code_count": len(code_groups),
        "codes_with_multiple_independent_figures": sum(len(groups) >= 2 for groups in code_groups.values()),
        "codes_with_three_or_more_independent_figures": trainable_codes,
        "fold_count": len(folds),
        "training_performed": False,
        "training_not_performed_reason": (
            "The available LawDB set has only a few independent illustrations per code, "
            "and only a small number of codes have three or more source figures. "
            "It is not adequate for a leakage-safe train/validation/test split or for "
            "estimating fine-tuned generalization."
        ),
        "models_requested": args.models,
        "model_results": [],
        "model_errors": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_query_rows: list[dict[str, Any]] = []
    for model_name in args.models:
        if model_name not in embedding_model_settings:
            manifest["model_errors"].append({"model": model_name, "error": "Unsupported embedding model name"})
            continue
        print(f"[{model_name}] loading on {device}", flush=True)
        model_load_start = time.perf_counter()
        embedder = None
        try:
            embedder, dim = get_embedding_model(model_name, device=device)
            model_load_seconds = time.perf_counter() - model_load_start
            vectors = []
            embedding_start = time.perf_counter()
            for index, row in enumerate(records, start=1):
                with Image.open(row["image_path"]) as image:
                    vector = embedder.embed_image(image.convert("RGB"))
                vector = np.asarray(vector, dtype=np.float32).reshape(-1)
                if vector.shape[0] != dim:
                    raise ValueError(f"Expected embedding dim {dim}, got {vector.shape[0]}")
                vectors.append(vector)
                if index % 50 == 0 or index == len(records):
                    print(f"  embedded {index}/{len(records)}", flush=True)
            matrix = np.stack(vectors)
            elapsed = time.perf_counter() - embedding_start
            model_metrics, query_rows = evaluate_model(model_name, records, matrix, elapsed)
            model_metrics["model_load_seconds"] = round(model_load_seconds, 3)
            model_metrics["embedding_dim"] = int(matrix.shape[1])
            model_metrics["parameters"] = int(embedder.parameters())
            manifest["model_results"].append(model_metrics)
            all_query_rows.extend(query_rows)
            safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
            np.savez_compressed(
                output_dir / f"embeddings_{safe_name}.npz",
                embeddings=matrix,
                image_paths=np.array([row["image_path"] for row in records]),
                codes=np.array([row["code"] for row in records]),
            )
            del matrix, vectors
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

    if all_query_rows:
        with (output_dir / "query_rows.json").open("w", encoding="utf-8") as stream:
            json.dump(all_query_rows, stream, ensure_ascii=False, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        results = manifest["model_results"]
        if results:
            writer = csv.DictWriter(stream, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    dataset_description = (
        f"- Dataset directory: `{args.dataset_dir.resolve()}`"
        if args.dataset_dir is not None
        else f"- Dataset: `{LAWDB_JSON.relative_to(ROOT)}`"
    )
    split_description = (
        "All crops from the same source figure are excluded from the gallery when that figure is queried."
        if args.dataset_dir is None
        else "Each image is held out once while all remaining images form the gallery. No physical-instance IDs were supplied, "
             "so repeated views of one physical sign could appear on both sides."
    )
    report_lines = [
        "# Sign image-embedding benchmark",
        "",
        f"- Device: `{device}`",
        dataset_description,
        f"- Labeled candidate crops: {len(records)} across {len(code_groups)} code labels.",
        f"- Labels with at least 2 independent groups: {manifest['codes_with_multiple_independent_figures']}.",
        f"- Leave-one-group-out queries: {sum(len(fold['queries']) for fold in folds)}.",
        f"- Labels with at least 3 independent groups: {trainable_codes}.",
        "- Fine-tuning: not run; this comparison evaluates frozen encoders only.",
        "",
        split_description + " Retrieval scores are max-aggregated by sign code, so repeated gallery images do not occupy multiple ranks.",
        "",
        "| Model | Dim | Exact R@1 | Exact R@3 | Exact R@5 | Macro R@1 | MRR | Load (s) | Embed ms/image |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in manifest["model_results"]:
        report_lines.append(
            "| {model} | {embedding_dim} | {recall_at_1:.3f} | {recall_at_3:.3f} | {recall_at_5:.3f} | {macro_recall_at_1:.3f} | {mrr:.3f} | {model_load_seconds:.1f} | {embedding_ms_per_image:.1f} |".format(**result)
        )
    if manifest["model_errors"]:
        report_lines.extend(["", "## Model errors", ""])
        report_lines.extend(f"- `{item['model']}`: {item['error']}" for item in manifest["model_errors"])
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
