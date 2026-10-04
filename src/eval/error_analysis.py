from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ERROR_TYPES = (
    "NO_DETECTION",
    "WRONG_SIGN_VARIANT",
    "WRONG_ARTICLE",
    "RIGHT_ARTICLE_LOW_RANK",
    "WRONG_SYMBOLIC_RULE",
    "WRONG_CITATION",
    "WRONG_FINAL_ANSWER",
    "INVALID_JSON",
)

DEFAULT_THRESHOLD = 0.2
SIGN_CODE_RE = re.compile(r"\b([A-Z]{1,3}(?:\.[A-Z])?[,.]?\d+[a-z]?)\b", re.I)


def _uid(article: dict[str, Any]) -> str:
    law_id = str(article.get("law_id") or "").strip()
    article_id = str(article.get("article_id") or "").strip()
    return f"{law_id}#{article_id}" if law_id and article_id else ""


def _article_ids(articles: Any) -> set[str]:
    if not isinstance(articles, list):
        return set()
    ids: set[str] = set()
    for article in articles:
        uid = _uid(article) if isinstance(article, dict) else str(article or "").strip()
        if uid:
            ids.add(uid)
    return ids


def _normalize_sign_code(value: Any) -> str:
    match = SIGN_CODE_RE.search(str(value or ""))
    if not match:
        return ""
    raw = match.group(1).replace(",", ".")
    parsed = re.fullmatch(r"([A-Za-z]{1,3}(?:\.[A-Za-z])?)[.]?(\d+)([A-Za-z]?)", raw)
    if not parsed:
        return ""
    prefix, number, suffix = parsed.groups()
    return f"{prefix.upper()}.{number}{suffix.lower()}"


def _family(sign_code: str) -> str:
    return re.sub(r"[,.]?\d+[a-z]$", "", sign_code, flags=re.I)


def _metric(row: dict[str, Any], output: dict[str, Any], key: str) -> float | None:
    value = row.get(key, output.get(key))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _answer(output: dict[str, Any]) -> str:
    for key in ("freeform_answer", "answer", "final_answer"):
        value = output.get(key)
        if isinstance(value, str):
            return value
    return ""


def _retrieved_article_lists(
    detected_signs: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[str]]:
    by_rank: dict[int, list[str]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for sign in detected_signs:
        if sign.get("is_chosen") is False:
            continue
        payloads = sign.get("retrieved_payloads") or sign.get("payloads") or []
        if not isinstance(payloads, list):
            continue
        for index, payload in enumerate(payloads[:5], start=1):
            if not isinstance(payload, dict):
                continue
            uid = _uid(payload)
            if uid and uid not in by_rank[index]:
                by_rank[index].append(uid)

    top1 = list(by_rank[1])
    top3: list[str] = []
    top5: list[str] = []
    for rank in range(1, 6):
        for uid in by_rank[rank]:
            if uid not in top5:
                top5.append(uid)
            if rank <= 3 and uid not in top3:
                top3.append(uid)
    return top1, top3, top5


def _sign_details(
    detected_signs: list[dict[str, Any]] | None,
    output: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if detected_signs is None:
        return None, None

    details: list[dict[str, Any]] = []
    codes: list[str] = []
    sign_observations = (output.get("traffic_facts") or {}).get("sign_observations", [])
    symbolic_codes = [
        _normalize_sign_code(fact.get("sign_code"))
        for fact in sign_observations
        if isinstance(fact, dict)
    ]
    symbolic_codes = [code for code in symbolic_codes if code]

    for sign in detected_signs:
        if not isinstance(sign, dict):
            continue
        payloads = sign.get("retrieved_payloads") or sign.get("payloads") or []
        names = [
            sign.get("sign_code"),
            sign.get("sign_name"),
            sign.get("name"),
        ]
        if isinstance(payloads, list):
            names.extend(
                payload.get("sign_code") or payload.get("sign_name") or payload.get("name")
                for payload in payloads
                if isinstance(payload, dict)
            )
        code = next((_normalize_sign_code(name) for name in names if _normalize_sign_code(name)), "")
        if code and code not in codes:
            codes.append(code)
        details.append(
            {
                "image_name": sign.get("image_name"),
                "sign_name": sign.get("sign_name") or sign.get("name"),
                "sign_code": code or None,
                "is_chosen": sign.get("is_chosen"),
            }
        )

    selected = symbolic_codes[0] if symbolic_codes else (codes[0] if codes else None)
    return details, selected


def _classify(
    sample: dict[str, Any],
    output: dict[str, Any],
    retrieved: tuple[list[str], list[str], list[str]],
    detected_signs: list[dict[str, Any]] | None,
    sign_code: str | None,
) -> list[str]:
    errors: list[str] = []
    top1, _, top5 = retrieved
    gold = _article_ids(sample.get("relevant_articles", sample.get("gold_articles", [])))

    if detected_signs is not None and (
        not detected_signs
        or (
            any("is_chosen" in sign for sign in detected_signs if isinstance(sign, dict))
            and not any(sign.get("is_chosen") for sign in detected_signs if isinstance(sign, dict))
        )
    ):
        errors.append("NO_DETECTION")

    gold_sign_codes = sample.get("gold_sign_codes") or []
    if isinstance(gold_sign_codes, str):
        gold_sign_codes = [gold_sign_codes]
    if sample.get("gold_sign_code"):
        gold_sign_codes = [*gold_sign_codes, sample["gold_sign_code"]]
    gold_codes = {_normalize_sign_code(code) for code in gold_sign_codes}
    gold_codes.discard("")
    if sign_code and gold_codes and sign_code not in gold_codes:
        if any(_family(sign_code) == _family(code) for code in gold_codes):
            errors.append("WRONG_SIGN_VARIANT")

    recall_at_1 = _metric(sample.get("_row", {}), output, "retrieval_recall_at_1")
    recall_at_5 = _metric(sample.get("_row", {}), output, "retrieval_recall_at_5")
    if gold and detected_signs is not None and detected_signs:
        if gold & set(top5):
            if not (gold & set(top1)):
                errors.append("RIGHT_ARTICLE_LOW_RANK")
        elif recall_at_5 == 0:
            errors.append("WRONG_ARTICLE")
        elif (
            recall_at_5 is not None
            and recall_at_1 is not None
            and recall_at_1 < recall_at_5
        ):
            errors.append("RIGHT_ARTICLE_LOW_RANK")
    elif gold and detected_signs is None and recall_at_5 is not None:
        if recall_at_5 == 0:
            errors.append("WRONG_ARTICLE")
        elif recall_at_1 is not None and recall_at_1 < recall_at_5:
            errors.append("RIGHT_ARTICLE_LOW_RANK")

    symbolic = output.get("symbolic_decision")
    expected_decision = sample.get("gold_symbolic_decision")
    if isinstance(symbolic, dict) and expected_decision:
        actual = str(symbolic.get("decision") or "UNKNOWN").upper()
        if actual != str(expected_decision).upper():
            errors.append("WRONG_SYMBOLIC_RULE")

    citations = output.get("citations")
    if isinstance(citations, list):
        cited = {_uid(item) for item in citations if isinstance(item, dict) and _uid(item)}
        if gold and (not cited or not (gold & cited) or not gold.issubset(cited)):
            errors.append("WRONG_CITATION")
    else:
        precision = _metric(sample.get("_row", {}), output, "citation_precision")
        recall = _metric(sample.get("_row", {}), output, "citation_recall")
        if (precision is not None and precision < 1) or (recall is not None and recall < 1):
            errors.append("WRONG_CITATION")

    correctness = _metric(sample.get("_row", {}), output, "answer_correctness")
    if correctness is None:
        correctness = _metric(sample.get("_row", {}), output, "answer_correctness_heuristic")
    if correctness is not None and correctness < 1:
        errors.append("WRONG_FINAL_ANSWER")

    json_valid = _metric(sample.get("_row", {}), output, "json_validity")
    if json_valid is not None and json_valid < 1:
        errors.append("INVALID_JSON")

    # A low-rank result is a reranking signal only when the observed rate is material.
    # Per-row labels remain independent of the aggregate recommendation threshold.
    return [error for error in ERROR_TYPES if error in errors]


def analyze_rows(
    rows: list[dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    analyzed: list[dict[str, Any]] = []
    retrieval_trace_coverage = 0
    detection_trace_coverage = 0
    for row in rows:
        sample = dict(row.get("input") or row.get("sample") or {})
        sample["_row"] = row
        output_value = row.get("output") or row.get("result") or {}
        output = {
            **row,
            **(dict(output_value) if isinstance(output_value, dict) else {}),
        }
        detected_value = output.get("detected_signs", row.get("detected_signs"))
        detected = detected_value if isinstance(detected_value, list) else None
        if detected is not None:
            detection_trace_coverage += 1
        top1, top3, top5 = _retrieved_article_lists(detected or [])
        if detected is not None and any(
            sign.get("retrieved_payloads") or sign.get("payloads")
            for sign in detected
            if isinstance(sign, dict)
        ):
            retrieval_trace_coverage += 1
        details, sign_code = _sign_details(detected, output)
        gold_articles = sorted(
            _article_ids(sample.get("relevant_articles", sample.get("gold_articles", [])))
        )

        # Expose supplied gold values only; never infer missing retrieval records from recall.
        retrieved_top1: list[str] | None = top1 if detected is not None else None
        retrieved_top3: list[str] | None = top3 if detected is not None else None
        retrieved_top5: list[str] | None = top5 if detected is not None else None
        symbolic_decision = output.get("symbolic_decision")
        errors = _classify(
            sample,
            output,
            (top1, top3, top5),
            detected,
            sign_code,
        )
        analyzed.append(
            {
                "id": row.get("id", sample.get("id")),
                "question": sample.get("question", row.get("question", "")),
                "gold_articles": gold_articles,
                "retrieved_top1": retrieved_top1,
                "retrieved_top3": retrieved_top3,
                "retrieved_top5": retrieved_top5,
                "detected_signs": details,
                "sign_code": sign_code,
                "symbolic_decision": symbolic_decision,
                "final_answer": _answer(output) or row.get("answer", ""),
                "error_type": errors,
                "metrics": {
                    key: row.get(key, output.get(key))
                    for key in (
                        "retrieval_recall_at_1",
                        "retrieval_recall_at_3",
                        "retrieval_recall_at_5",
                        "citation_precision",
                        "citation_recall",
                        "answer_correctness",
                        "json_validity",
                    )
                    if row.get(key, output.get(key)) is not None
                },
            }
        )

    total = len(analyzed)
    counts = {
        error_type: sum(error_type in row["error_type"] for row in analyzed)
        for error_type in ERROR_TYPES
    }
    rates = {
        error_type: round(count / total, 4) if total else 0.0
        for error_type, count in counts.items()
    }
    decisions: list[dict[str, Any]] = []
    recommendations = (
        (
            "RIGHT_ARTICLE_LOW_RANK",
            "Prioritize reranking or hybrid retrieval; gold articles are retrieved but not near the top.",
        ),
        (
            "WRONG_SIGN_VARIANT",
            "Improve sign-code extraction and variant resolution before changing article retrieval.",
        ),
        (
            "WRONG_ARTICLE",
            "Audit detection, embeddings and LawDB coverage; the relevant article is absent from observed top-5 results.",
        ),
    )
    for error_type, recommendation in recommendations:
        rate = rates[error_type]
        decisions.append(
            {
                "signal": error_type,
                "count": counts[error_type],
                "rate": rate,
                "triggered": rate >= threshold and total > 0,
                "recommendation": recommendation,
            }
        )

    retrieval_correct_answer_wrong = 0
    for row in analyzed:
        if "WRONG_FINAL_ANSWER" not in row["error_type"]:
            continue
        retrieved = row["retrieved_top5"]
        gold = set(row["gold_articles"])
        recall_at_5 = row["metrics"].get("retrieval_recall_at_5")
        if (retrieved is not None and gold and gold.issubset(set(retrieved))) or (
            retrieved is None and recall_at_5 is not None and float(recall_at_5) >= 1
        ):
            retrieval_correct_answer_wrong += 1
    if retrieval_correct_answer_wrong:
        decisions.append(
            {
                "signal": "RETRIEVAL_CORRECT_BUT_ANSWER_WRONG",
                "count": retrieval_correct_answer_wrong,
                "rate": round(retrieval_correct_answer_wrong / total, 4) if total else 0.0,
                "triggered": retrieval_correct_answer_wrong / total >= threshold,
                "recommendation": "For these specific cases, focus on symbolic reasoning or the answer prompt rather than retrieval.",
            }
        )

    return {
        "schema_version": 1,
        "threshold": threshold,
        "summary": {
            "num_samples": total,
            "error_counts": counts,
            "error_rates": rates,
            "trace_coverage": {
                "detection_trace_samples": detection_trace_coverage,
                "retrieval_payload_trace_samples": retrieval_trace_coverage,
                "retrieval_trace_rate": round(retrieval_trace_coverage / total, 4) if total else 0.0,
            },
            "recommendations": decisions,
        },
        "provenance": source_metadata or {},
        "limitations": [
            "Unavailable retrieval or detection details are null; they are not treated as empty results.",
            "WRONG_SIGN_VARIANT requires gold_sign_code(s) in the input sample.",
            "WRONG_SYMBOLIC_RULE requires gold_symbolic_decision in the input sample.",
            "When only aggregate per-sample recall is available, low-rank and wrong-article labels are inferred from those metrics; exact article lists remain unavailable.",
        ],
        "rows": analyzed,
    }


def to_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Retrieval Error Analysis",
        "",
        f"Samples: **{summary['num_samples']}** · decision threshold: **{report['threshold']:.0%}**",
        "",
        "## Error frequency",
        "",
        "| Error type | Count | Rate |",
        "|---|---:|---:|",
    ]
    for error_type in ERROR_TYPES:
        count = summary["error_counts"][error_type]
        rate = summary["error_rates"][error_type]
        lines.append(f"| {error_type} | {count} | {rate:.1%} |")
    lines.extend(["", "## Decision guidance", "", "| Signal | Count | Rate | Triggered | Recommendation |", "|---|---:|---:|:---:|---|"])
    for item in summary["recommendations"]:
        lines.append(
            f"| {item['signal']} | {item['count']} | {item['rate']:.1%} | {str(item['triggered']).lower()} | {item['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "## Per-sample analysis",
            "",
            "| ID | Question | Gold articles | Retrieved top-1 | top-3 | top-5 | Sign code | Symbolic decision | Final answer | Error type |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in report["rows"]:
        def cell(value: Any) -> str:
            text = "unavailable" if value is None else json.dumps(value, ensure_ascii=False)
            return text.replace("|", "\\|").replace("\n", " ")

        symbolic = row["symbolic_decision"]
        if isinstance(symbolic, dict):
            symbolic = symbolic.get("decision")
        lines.append(
            "| "
            + " | ".join(
                cell(value)
                for value in (
                    row["id"],
                    row["question"],
                    row["gold_articles"],
                    row["retrieved_top1"],
                    row["retrieved_top3"],
                    row["retrieved_top5"],
                    row["sign_code"],
                    symbolic,
                    row["final_answer"],
                    row["error_type"],
                )
            )
            + " |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-sample retrieval and answer errors")
    parser.add_argument("--input", type=Path, default=Path("outputs/baseline_rows.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/error_analysis.json"))
    parser.add_argument("--markdown", type=Path, default=Path("outputs/error_analysis.md"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Input must be a list of rows or an object containing a rows list")
    source = {
        key: payload[key]
        for key in ("baseline_name", "config_sha256", "detail_level", "provenance")
        if isinstance(payload, dict) and key in payload
    }
    report = analyze_rows(rows, threshold=args.threshold, source_metadata=source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(to_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved analysis to {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
