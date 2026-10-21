from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_baseline_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    if not isinstance(config, dict):
        raise ValueError("Baseline config must be a JSON object")

    dataset = config.get("dataset")
    pipeline = config.get("pipeline")
    outputs = config.get("outputs")
    if not all(isinstance(item, dict) for item in (dataset, pipeline, outputs)):
        raise ValueError(
            "Baseline config requires dataset, pipeline, and outputs objects"
        )

    required_dataset_keys = {
        "json_path",
        "image_dir",
        "lawdb_json_path",
        "sample_ids",
    }
    missing_dataset_keys = sorted(required_dataset_keys - set(dataset))
    if missing_dataset_keys:
        raise ValueError(
            "Baseline dataset config is missing: "
            f"{', '.join(missing_dataset_keys)}"
        )

    sample_ids = dataset.get("sample_ids", [])
    if not 20 <= len(sample_ids) <= 50:
        raise ValueError("Baseline config must lock between 20 and 50 sample ids")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Baseline sample ids must be unique")

    required_pipeline_keys = {
        "llm_model",
        "limit",
        "query_detector",
        "lawdb_detector",
        "embedding_model",
        "collection_name",
        "use_vlm_filter",
        "judge",
        "temperature",
        "max_new_tokens",
    }
    missing = sorted(required_pipeline_keys - set(pipeline))
    if missing:
        raise ValueError(f"Baseline pipeline config is missing: {', '.join(missing)}")
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def select_samples(
    samples: list[dict[str, Any]], sample_ids: list[str]
) -> list[dict[str, Any]]:
    samples_by_id = {str(sample.get("id")): sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in samples_by_id]
    if missing:
        raise ValueError(f"Dataset is missing locked baseline samples: {missing}")
    return [samples_by_id[sample_id] for sample_id in sample_ids]
