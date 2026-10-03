import json

import pytest

from src.eval.baseline_config import (
    config_fingerprint,
    load_baseline_config,
    select_samples,
)
from src.eval.compare_baseline import METRIC_DIRECTIONS, compare_metrics, load_metrics


def test_baseline_config_locks_20_samples():
    config = load_baseline_config("configs/baseline_eval.json")

    assert len(config["dataset"]["sample_ids"]) == 20
    assert config["pipeline"]["limit"] == 5
    assert config["pipeline"]["llm_model"] == "gemma-3-12b-it"
    assert config["pipeline"]["collection_name"].endswith("clip-gmp-vit-l-14")


def test_select_samples_uses_manifest_order():
    samples = [{"id": "b"}, {"id": "a"}]

    assert select_samples(samples, ["a", "b"]) == [{"id": "a"}, {"id": "b"}]


def test_select_samples_rejects_missing_id():
    with pytest.raises(ValueError, match="missing locked baseline samples"):
        select_samples([{"id": "a"}], ["a", "b"])


def test_compare_metrics_respects_latency_direction():
    baseline = {
        "answer_correctness": 0.5,
        "citation_precision": 0.5,
        "citation_recall": 0.5,
        "retrieval_recall_at_1": 0.5,
        "retrieval_recall_at_3": 0.5,
        "retrieval_recall_at_5": 0.5,
        "json_validity": 0.5,
        "evidence_faithfulness": 0.5,
        "latency": 10.0,
    }
    candidate = {**baseline, "answer_correctness": 0.6, "latency": 8.0}

    comparison = compare_metrics(baseline, candidate)

    assert comparison["answer_correctness"]["improved"] is True
    assert comparison["latency"]["improved"] is True
    assert comparison["latency"]["delta"] == -2.0


def test_load_metrics_supports_current_and_legacy_shapes(tmp_path):
    current = tmp_path / "current.json"
    legacy = tmp_path / "legacy.json"
    current.write_text(json.dumps({"metrics": {"latency": 1}}), encoding="utf-8")
    legacy.write_text(json.dumps({"summary": {"latency": 2}}), encoding="utf-8")

    assert load_metrics(current)[1]["latency"] == 1
    assert load_metrics(legacy)[1]["latency"] == 2


def test_committed_baseline_artifacts_match_locked_config():
    config = load_baseline_config("configs/baseline_eval.json")
    metrics_payload, metrics = load_metrics("outputs/baseline_metrics.json")
    with open("outputs/baseline_rows.json", encoding="utf-8") as file_obj:
        rows_payload = json.load(file_obj)

    fingerprint = config_fingerprint(config)
    assert metrics_payload["config_sha256"] == fingerprint
    assert rows_payload["config_sha256"] == fingerprint
    assert [row["id"] for row in rows_payload["rows"]] == config["dataset"][
        "sample_ids"
    ]
    assert metrics["num_samples"] == len(rows_payload["rows"]) == 20
    assert set(METRIC_DIRECTIONS).issubset(metrics)
