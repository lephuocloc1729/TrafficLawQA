"""Capture answers and per-sample latency for the shared 20-image test set.

This runner does not score answers. Keep the gold answers in the test-set
manifest for a later independent review.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI
from tqdm import tqdm

from src.core.baseline1_yolo26s_vlm import answer_image_question as answer_baseline1
from src.core.baseline1_yolo26s_vlm import create_resources as create_baseline1_resources
from src.core.freeform_qa import (
    answer_image_question,
    answer_rag_compatible_image_question,
    create_freeform_qa_resources,
)
from src.eval.freeform_qa_metrics import apply_baseline_config
from src.eval.vlm_only import answer_one as answer_vlm
from src.eval.vlm_only import load_config as load_vlm_config
from src.settings import settings


PIPELINES = (
    ("RAG baseline", "rag"),
    ("Neuro-symbolic (YOLO26s)", "neuro_symbolic"),
    ("Baseline 1: YOLO26s + VLM", "baseline1"),
    ("VLM-only", "vlm"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def configure_pipeline(kind: str) -> dict[str, Any] | None:
    if kind == "rag":
        config = load_json(Path("configs/baseline_eval.json"))
        apply_baseline_config(config)
        return config
    if kind == "neuro_symbolic":
        config = load_json(Path("configs/neuro_symbolic_eval.json"))
        apply_baseline_config(config)
        return config
    return None


def pipeline_resources(kind: str) -> tuple[Any, dict[str, Any] | None]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if kind == "rag":
        configure_pipeline(kind)
        return create_freeform_qa_resources(device=device), None
    if kind == "neuro_symbolic":
        configure_pipeline(kind)
        return create_freeform_qa_resources(device=device), None
    if kind == "baseline1":
        configure_pipeline(kind)
        return create_baseline1_resources(device=device), None
    if kind == "vlm":
        config = load_vlm_config(Path("configs/baseline2_vlm_only.json"))
        model = config["model"]
        base_url = os.environ.get(model.get("base_url_env", "LLM_BASE_URL")) or settings.llm_base_url
        configured_keys = (
            os.environ.get(model.get("api_key_env", "LLM_API_KEYS"))
            or settings.llm_api_keys
            or ""
        )
        api_keys = [key.strip() for key in configured_keys.split(",") if key.strip()]
        if not base_url or not api_keys:
            raise ValueError("Set LLM_BASE_URL and LLM_API_KEYS before running VLM-only.")
        return OpenAI(base_url=base_url, api_key=api_keys[0]), model
    raise ValueError(f"Unknown pipeline kind: {kind}")


def run_one(
    kind: str,
    resources: Any,
    model: dict[str, Any] | None,
    image_path: Path,
    question: str,
) -> tuple[str, float]:
    # Reapply the selected profile for every inference so a previous pipeline
    # cannot leave detector/retrieval/symbolic flags behind in global settings.
    configure_pipeline(kind)
    started = time.perf_counter()
    if kind == "rag":
        result = answer_rag_compatible_image_question(
            image_path=image_path,
            question=question,
            resources=resources,
            use_vlm_filter=True,
        )
        answer = str(result.get("freeform_answer") or "")
    elif kind == "neuro_symbolic":
        result = answer_image_question(
            image_path=image_path,
            question=question,
            resources=resources,
            use_vlm_filter=(kind == "rag"),
        )
        answer = str(result.get("freeform_answer") or "")
    elif kind == "baseline1":
        result = answer_baseline1(
            image_path=image_path,
            question=question,
            resources=resources,
        )
        answer = str(result.get("answer") or "")
    else:
        result = answer_vlm(resources, model or {}, {"question": question}, image_path)
        answer = str(result.get("answer") or "")
    return answer, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture answers and latency for all four pipelines; no answer scoring is performed."
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("examples/traffic_sign_vqa_testcases_20"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/four_pipeline_test_results.json"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse previously captured case/pipeline answers in the output file.",
    )
    args = parser.parse_args()

    manifest_path = args.test_set / "manifest.json"
    manifest = load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError(f"Expected 20 cases in {manifest_path}")

    rows: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        saved = load_json(args.output)
        # Retry failed cases on resume while retaining successful captures.
        rows = [row for row in saved.get("rows", []) if not row.get("error")]
    completed = {
        (str(row.get("case_id")), str(row.get("pipeline"))) for row in rows
    }
    started_at = datetime.now(timezone.utc).isoformat()

    for pipeline_name, kind in PIPELINES:
        missing_cases = [
            case for case in cases
            if (str(case.get("case_id")), pipeline_name) not in completed
        ]
        if not missing_cases:
            continue
        try:
            resources, model = pipeline_resources(kind)
        except Exception as exc:
            for case in missing_cases:
                rows.append(
                    {
                        "case_id": case.get("case_id"),
                        "pipeline": pipeline_name,
                        "question": case.get("question", ""),
                        "answer": "",
                        "latency_seconds": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            save_json(
                args.output,
                {
                    "schema_version": 1,
                    "status": "in_progress",
                    "started_at_utc": started_at,
                    "test_set": str(manifest_path),
                    "latency_scope": "per-case inference; pipeline resource initialization is excluded",
                    "scoring": "not_performed",
                    "rows": rows,
                },
            )
            continue

        for case in tqdm(missing_cases, desc=pipeline_name):
            case_id = str(case.get("case_id"))
            image_path = args.test_set / str(case["image"])
            row: dict[str, Any] = {
                "case_id": case_id,
                "pipeline": pipeline_name,
                "question": case.get("question", ""),
            }
            try:
                answer, latency = run_one(
                    kind=kind,
                    resources=resources,
                    model=model,
                    image_path=image_path,
                    question=str(case.get("question") or ""),
                )
                row.update(
                    {
                        "answer": answer,
                        "latency_seconds": round(latency, 5),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "answer": "",
                        "latency_seconds": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            rows.append(row)
            completed.add((case_id, pipeline_name))
            save_json(
                args.output,
                {
                    "schema_version": 1,
                    "status": "in_progress",
                    "started_at_utc": started_at,
                    "test_set": str(manifest_path),
                    "latency_scope": "per-case inference; pipeline resource initialization is excluded",
                    "scoring": "not_performed",
                    "rows": rows,
                },
            )

    expected = len(cases) * len(PIPELINES)
    complete = len({(str(row.get("case_id")), str(row.get("pipeline"))) for row in rows}) == expected
    save_json(
        args.output,
        {
            "schema_version": 1,
            "status": "complete" if complete else "incomplete",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_set": str(manifest_path),
            "latency_scope": "per-case inference; pipeline resource initialization is excluded",
            "scoring": "not_performed",
            "rows": rows,
        },
    )
    print(f"Saved {len(rows)} answer/latency rows to {args.output}")


if __name__ == "__main__":
    main()
