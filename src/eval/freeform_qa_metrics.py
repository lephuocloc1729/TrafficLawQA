from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm

from src.settings import settings
from src.deps.llm_client import LLMClient
from src.core.freeform_qa import answer_image_question, create_freeform_qa_resources
from src.core.sub_task_2 import extract_json_object
from src.core.utils import load_json, save_json


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
        required = {"answer", "citations", "visual_evidence", "reasoning", "confidence"}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate free-form cited QA metrics")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--begin-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default="outputs/freeform_qa_metrics.json")
    parser.add_argument("--no-vlm-filter", action="store_true")
    parser.add_argument("--judge", action="store_true")
    args = parser.parse_args()

    samples = load_json(settings.train_json_path)
    selected = samples[args.begin_idx : args.begin_idx + args.num_samples]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resources = create_freeform_qa_resources()
    judge_client = resources.llm_client if args.judge else None
    rows: list[dict[str, Any]] = []

    for sample in tqdm(selected, desc="Free-form QA eval"):
        image_path = Path(settings.train_image_dir) / f"{sample['image_id']}.jpg"
        t0 = time.perf_counter()
        result = answer_image_question(
            image_path=image_path,
            question=sample.get("question", ""),
            choices=json.dumps(sample.get("choices", {}), ensure_ascii=False),
            debug=False,
            resources=resources,
            use_vlm_filter=not args.no_vlm_filter,
        )
        latency = time.perf_counter() - t0

        row = {
            "id": sample.get("id"),
            "image_id": sample.get("image_id"),
            "latency": latency,
            "answer": result.get("freeform_answer", ""),
            **citation_metrics(sample, result),
            **evidence_proxy_metrics(result),
            **retrieval_recall_metrics(sample, result),
            **visual_grounding_proxy(sample, result),
            **json_validity(result),
        }
        row["answer_correctness_heuristic"] = heuristic_answer_correctness(sample, result)
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
        save_json({"summary": summarize(rows), "rows": rows}, output_path)

    payload = {"summary": summarize(rows), "rows": rows}
    save_json(payload, output_path)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
