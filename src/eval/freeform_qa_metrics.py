from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from src.settings import settings
from src.eval.baseline_config import (
    config_fingerprint,
    load_baseline_config,
    select_samples,
)
from src.deps.llm_client import LLMClient
from src.core.freeform_qa import answer_image_question, create_freeform_qa_resources
from src.core.sub_task_2 import extract_json_object
from src.core.utils import load_json


REQUIRED_BASELINE_METRICS = (
    "answer_correctness",
    "citation_precision",
    "citation_recall",
    "retrieval_recall_at_1",
    "retrieval_recall_at_3",
    "retrieval_recall_at_5",
    "json_validity",
    "evidence_faithfulness",
    "latency",
)


def article_uid(article: dict[str, Any]) -> str:
    return f"{article.get('law_id', '')}#{article.get('article_id', '')}"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").casefold()).strip()


def citation_metrics(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    gold = {article_uid(article) for article in sample.get("relevant_articles", [])}
    predicted = {article_uid(article) for article in result.get("citations", [])}
    if not gold:
        return {"citation_precision": 0.0, "citation_recall": 0.0, "citation_accuracy": 0.0}

    overlap = gold & predicted
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold)
    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_accuracy": 1.0 if overlap else 0.0,
    }


def evidence_supported(citation: dict[str, Any], article_contexts: list[dict[str, str]]) -> bool:
    evidence = normalize_text(str(citation.get("evidence") or ""))
    if not evidence:
        return False
    for article in article_contexts:
        if (
            article.get("law_id") == citation.get("law_id")
            and article.get("article_id") == citation.get("article_id")
        ):
            article_text = normalize_text(article.get("text", ""))
            if evidence in article_text:
                return True
            evidence_tokens = set(evidence.split())
            article_tokens = set(article_text.split())
            if evidence_tokens and len(evidence_tokens & article_tokens) / len(evidence_tokens) >= 0.65:
                return True
    return False


def evidence_proxy_metrics(result: dict[str, Any]) -> dict[str, float]:
    citations = result.get("citations", [])
    if not citations:
        return {"evidence_faithfulness_proxy": 0.0}
    supported = [
        evidence_supported(citation, result.get("article_contexts", []))
        for citation in citations
    ]
    return {"evidence_faithfulness_proxy": sum(supported) / len(supported)}


def retrieval_recall_metrics(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    gold = {article_uid(article) for article in sample.get("relevant_articles", [])}
    if not gold:
        return {"retrieval_recall_at_1": 0.0, "retrieval_recall_at_3": 0.0, "retrieval_recall_at_5": 0.0}

    recalls: dict[str, float] = {}
    for k in (1, 3, 5):
        retrieved: set[str] = set()
        for sign in result.get("detected_signs", []):
            if not sign.get("is_chosen"):
                continue
            for payload in sign.get("retrieved_payloads", [])[:k]:
                retrieved.add(article_uid(payload))
        recalls[f"retrieval_recall_at_{k}"] = len(gold & retrieved) / len(gold)
    return recalls


def visual_grounding_proxy(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    gold = {article_uid(article) for article in sample.get("relevant_articles", [])}
    chosen = [sign for sign in result.get("detected_signs", []) if sign.get("is_chosen")]
    if not gold or not chosen:
        return {"visual_grounding_accuracy_proxy": 0.0}

    correct = 0
    for sign in chosen:
        top_payloads = sign.get("retrieved_payloads", [])[:1]
        if any(article_uid(payload) in gold for payload in top_payloads):
            correct += 1
    return {"visual_grounding_accuracy_proxy": correct / len(chosen)}


def json_validity(result: dict[str, Any]) -> dict[str, float]:
    try:
        parsed = extract_json_object(result.get("raw_answer_response", ""))
        required = {
            "answer",
            "citations",
            "visual_evidence",
            "reasoning",
            "confidence",
            "symbolic_decision",
            "rules_fired",
            "traffic_facts",
            "query_intent",
            "query_context_facts",
            "reasoning_trace",
        }
        ok = isinstance(parsed, dict) and required.issubset(parsed)
    except Exception:
        ok = False
    return {"json_validity": 1.0 if ok else 0.0}


def heuristic_answer_correctness(sample: dict[str, Any], result: dict[str, Any]) -> float:
    choices = sample.get("choices") or {}
    answer_key = str(sample.get("answer") or "").strip()
    correct_choice = str(choices.get(answer_key) or "").strip()
    answer = normalize_text(result.get("freeform_answer", ""))
    if not correct_choice:
        return 0.0
    normalized_choice = normalize_text(correct_choice)
    if normalized_choice and normalized_choice in answer:
        return 1.0
    meaningful_tokens = [token for token in normalized_choice.split() if len(token) > 2]
    if meaningful_tokens:
        overlap = sum(token in answer for token in meaningful_tokens) / len(meaningful_tokens)
        return 1.0 if overlap >= 0.55 else 0.0
    return 0.0


def judge_sample(
    llm_client: LLMClient,
    sample: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, float]:
    choices = sample.get("choices") or {}
    correct_key = str(sample.get("answer") or "")
    correct_choice = choices.get(correct_key, "")
    prompt = {
        "question": sample.get("question", ""),
        "choices": choices,
        "correct_answer_key": correct_key,
        "correct_answer_text": correct_choice,
        "model_answer": result.get("freeform_answer", ""),
        "citations": result.get("citations", []),
        "reasoning": result.get("reasoning", ""),
    }
    response = llm_client.generate_content(
        contents=[
            {
                "role": "user",
                "content": (
                    "Bạn là giám khảo. Chấm output QA pháp luật giao thông dưới đây.\n"
                    "Trả strict JSON với schema: "
                    "{\"answer_correctness\": 0|1, \"evidence_faithfulness\": 0|1, "
                    "\"hallucination\": 0|1, \"notes\": \"...\"}.\n"
                    "answer_correctness=1 nếu model_answer đúng ý đáp án đúng, dù không ghi A/B/C/D.\n"
                    "evidence_faithfulness=1 nếu citations/reasoning hỗ trợ trực tiếp câu trả lời.\n"
                    "hallucination=1 nếu câu trả lời bịa thông tin hoặc citation không được hỗ trợ.\n\n"
                    f"{json.dumps(prompt, ensure_ascii=False)}"
                ),
            }
        ],
        temperature=0.0,
        max_tokens=600,
    )
    try:
        parsed = extract_json_object(response)
    except Exception:
        parsed = {}
    return {
        "answer_correctness": float(parsed.get("answer_correctness", 0)),
        "evidence_faithfulness": float(parsed.get("evidence_faithfulness", 0)),
        "hallucination": float(parsed.get("hallucination", 1)),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "answer_correctness",
        "citation_accuracy",
        "citation_precision",
        "citation_recall",
        "evidence_faithfulness",
        "evidence_faithfulness_proxy",
        "visual_grounding_accuracy_proxy",
        "retrieval_recall_at_1",
        "retrieval_recall_at_3",
        "retrieval_recall_at_5",
        "json_validity",
        "hallucination",
        "latency",
    ]
    summary = {"num_samples": len(rows)}
    for key in metric_keys:
        values = [row[key] for row in rows if key in row]
        if values:
            summary[key] = round(mean(values), 4)
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _save_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")
    temporary_path.replace(path)


def apply_baseline_config(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    pipeline = config["pipeline"]
    settings.train_json_path = dataset["json_path"]
    settings.train_image_dir = dataset["image_dir"]
    settings.lawdb_json_path = dataset["lawdb_json_path"]
    settings.llm_model = pipeline["llm_model"]
    settings.limit = int(pipeline["limit"])
    settings.image_query_detection_model = pipeline["query_detector"]
    settings.image_lawdb_detection_model = pipeline["lawdb_detector"]
    settings.embedding_name = pipeline["embedding_model"]
    settings.llm_temperature = float(pipeline["temperature"])
    settings.llm_max_new_tokens = int(pipeline["max_new_tokens"])
    settings.collection_name = pipeline["collection_name"]


def _output_payloads(
    config: dict[str, Any],
    config_path: Path,
    dataset_path: Path,
    rows: list[dict[str, Any]],
    started_at: str,
    complete: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fingerprint = config_fingerprint(config)
    status = "complete" if complete else "in_progress"
    common = {
        "schema_version": 1,
        "baseline_name": config.get("name", config_path.stem),
        "status": status,
        "config_path": str(config_path),
        "config_sha256": fingerprint,
    }
    metrics_payload = {
        **common,
        "run": {
            "started_at_utc": started_at,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_revision": _git_revision(),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "lawdb_path": config["dataset"]["lawdb_json_path"],
            "lawdb_sha256": _sha256_file(Path(config["dataset"]["lawdb_json_path"])),
            "sample_ids": config["dataset"]["sample_ids"],
        },
        "pipeline": config["pipeline"],
        "metrics": summarize(rows),
    }
    rows_payload = {
        **common,
        "detail_level": "full_pipeline_output",
        "rows": rows,
    }
    return metrics_payload, rows_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reproducible free-form cited-QA baseline"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_eval.json"),
        help="Locked sample and pipeline configuration",
    )
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--rows-output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep completed rows from a prior run with the same config",
    )
    args = parser.parse_args()

    config = load_baseline_config(args.config)
    apply_baseline_config(config)
    dataset_path = Path(config["dataset"]["json_path"])
    image_dir = Path(config["dataset"]["image_dir"])
    metrics_path = args.metrics_output or Path(config["outputs"]["metrics"])
    rows_path = args.rows_output or Path(config["outputs"]["rows"])

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Baseline dataset not found: {dataset_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Baseline image directory not found: {image_dir}")
    lawdb_path = Path(config["dataset"]["lawdb_json_path"])
    if not lawdb_path.is_file():
        raise FileNotFoundError(f"Baseline LawDB not found: {lawdb_path}")
    if not settings.llm_api_keys:
        raise RuntimeError(
            "LLM_API_KEYS is required to run the baseline; configure the locked model "
            "through an OpenAI-compatible endpoint first"
        )

    samples = load_json(dataset_path)
    if not isinstance(samples, list):
        raise ValueError("Baseline dataset must be a JSON list")
    selected = select_samples(samples, config["dataset"]["sample_ids"])

    resources = create_freeform_qa_resources()
    judge_client = resources.llm_client if config["pipeline"]["judge"] else None
    rows: list[dict[str, Any]] = []
    fingerprint = config_fingerprint(config)
    if args.resume and rows_path.is_file():
        prior = load_json(rows_path)
        if prior.get("config_sha256") != fingerprint:
            raise ValueError("Cannot resume: baseline config fingerprint changed")
        if prior.get("detail_level") != "full_pipeline_output":
            raise ValueError(
                "Cannot resume from legacy rows without full pipeline output"
            )
        rows = prior.get("rows", [])
    completed_ids = {str(row.get("id")) for row in rows}
    started_at = datetime.now(timezone.utc).isoformat()

    for sample in tqdm(selected, desc="Free-form QA eval"):
        if str(sample.get("id")) in completed_ids:
            continue
        image_path = image_dir / f"{sample['image_id']}.jpg"
        t0 = time.perf_counter()
        result = answer_image_question(
            image_path=image_path,
            question=sample.get("question", ""),
            choices=json.dumps(sample.get("choices", {}), ensure_ascii=False),
            debug=False,
            resources=resources,
            use_vlm_filter=config["pipeline"]["use_vlm_filter"],
        )
        latency = time.perf_counter() - t0

        row = {
            "id": sample.get("id"),
            "image_id": sample.get("image_id"),
            "latency": latency,
            "answer": result.get("freeform_answer", ""),
            "input": sample,
            "output": result,
            **citation_metrics(sample, result),
            **evidence_proxy_metrics(result),
            **retrieval_recall_metrics(sample, result),
            **visual_grounding_proxy(sample, result),
            **json_validity(result),
        }
        row["answer_correctness_heuristic"] = heuristic_answer_correctness(
            sample, result
        )
        if judge_client:
            row.update(judge_sample(judge_client, sample, result))
        else:
            row["answer_correctness"] = row["answer_correctness_heuristic"]
            row["evidence_faithfulness"] = row["evidence_faithfulness_proxy"]
            row["hallucination"] = 1.0 - min(
                row["json_validity"],
                row["citation_precision"],
                row["evidence_faithfulness_proxy"],
            )
        rows.append(row)
        metrics_payload, rows_payload = _output_payloads(
            config, args.config, dataset_path, rows, started_at, complete=False
        )
        _save_json_atomic(metrics_payload, metrics_path)
        _save_json_atomic(rows_payload, rows_path)

    metrics_payload, rows_payload = _output_payloads(
        config,
        args.config,
        dataset_path,
        rows,
        started_at,
        complete=len(rows) == len(selected),
    )
    missing_metrics = [
        key
        for key in REQUIRED_BASELINE_METRICS
        if key not in metrics_payload["metrics"]
    ]
    if missing_metrics:
        raise RuntimeError(f"Baseline output is missing metrics: {missing_metrics}")
    _save_json_atomic(metrics_payload, metrics_path)
    _save_json_atomic(rows_payload, rows_path)
    print(json.dumps(metrics_payload["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved aggregate metrics to {metrics_path}")
    print(f"Saved detailed rows to {rows_path}")


if __name__ == "__main__":
    main()
