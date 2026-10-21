from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_DIRECTIONS = {
    "answer_correctness": "higher",
    "citation_precision": "higher",
    "citation_recall": "higher",
    "retrieval_recall_at_1": "higher",
    "retrieval_recall_at_3": "higher",
    "retrieval_recall_at_5": "higher",
    "json_validity": "higher",
    "evidence_faithfulness": "higher",
    "latency": "lower",
}


def load_metrics(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path)
    with path.open(encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Metrics file must contain a JSON object: {path}")
    metrics = payload.get("metrics", payload.get("summary", payload))
    if not isinstance(metrics, dict):
        raise ValueError(f"Metrics object not found in: {path}")
    return payload, metrics


def compare_metrics(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    missing = [key for key in METRIC_DIRECTIONS if key not in candidate]
    if missing:
        raise ValueError(f"Candidate is missing baseline metrics: {missing}")

    for key, direction in METRIC_DIRECTIONS.items():
        if key not in baseline:
            raise ValueError(f"Baseline is missing metric: {key}")
        baseline_value = float(baseline[key])
        candidate_value = float(candidate[key])
        delta = candidate_value - baseline_value
        improved = delta > 0 if direction == "higher" else delta < 0
        comparison[key] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": round(delta, 6),
            "direction": direction,
            "improved": improved,
            "regressed": delta < 0 if direction == "higher" else delta > 0,
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare evaluation metrics to baseline"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/baseline_metrics.json"),
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-config-mismatch",
        action="store_true",
        help="Allow a comparison even when both config fingerprints differ",
    )
    args = parser.parse_args()

    baseline_payload, baseline_metrics = load_metrics(args.baseline)
    candidate_payload, candidate_metrics = load_metrics(args.candidate)
    fingerprints = (
        baseline_payload.get("config_sha256"),
        candidate_payload.get("config_sha256"),
    )
    config_match = (
        fingerprints[0] == fingerprints[1] if all(fingerprints) else None
    )
    if config_match is False and not args.allow_config_mismatch:
        parser.error(
            "baseline and candidate configurations differ; rerun with the locked "
            "config or pass --allow-config-mismatch for an explicitly "
            "non-equivalent comparison"
        )

    comparison = compare_metrics(baseline_metrics, candidate_metrics)
    payload = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "config_match": config_match,
        "metrics": comparison,
    }

    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
