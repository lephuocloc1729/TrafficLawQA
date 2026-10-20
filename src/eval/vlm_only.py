"""Direct image-question answering baseline with one fixed VLM only.

This runner deliberately has no imports from detection, OCR, LawDB, retrieval,
or symbolic reasoning modules. It sends the original image and question in a
single multimodal request to the configured OpenAI-compatible VLM endpoint.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from src.settings import settings


SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về tình huống giao thông dựa trực tiếp trên ảnh được cung cấp và kiến thức sẵn có của bạn. Trả lời bằng tiếng Việt tự nhiên. Nếu câu hỏi có lựa chọn, hãy chọn một lựa chọn và ghi rõ mã lựa chọn theo mẫu 'Đáp án: A' (thay A bằng mã phù hợp), sau đó giải thích ngắn gọn. Nếu ảnh hoặc thông tin không đủ căn cứ, hãy nói rõ điều đó và không đoán. Không cần viện dẫn điều luật hoặc tạo citation; nếu nhắc đến quy định, hãy trình bày như kiến thức của bạn, không khẳng định đó là trích dẫn đã xác minh."""


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    for key in ("dataset", "model", "outputs"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Config requires a {key} object")
    sample_ids = config["dataset"].get("sample_ids")
    if not isinstance(sample_ids, list) or not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("dataset.sample_ids must be a non-empty list of unique ids")
    return config


def resolve_image(image_dir: Path, image_id: str) -> Path:
    direct = image_dir / image_id
    if direct.is_file():
        return direct
    matches = sorted(path for path in image_dir.glob(f"{image_id}.*") if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one image for {image_id!r} in {image_dir}; found {len(matches)}")
    return matches[0]


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(suffix, "application/octet-stream")


def image_data_url(path: Path) -> str:
    return f"data:{mime_type(path)};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def format_question(sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    choices = sample.get("choices") or {}
    if isinstance(choices, dict) and choices:
        options = "\n".join(f"{key}. {value}" for key, value in choices.items())
        question += f"\n\nCác lựa chọn:\n{options}"
    return question


def answer_one(client: OpenAI, model: dict[str, Any], sample: dict[str, Any], image_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model["model_id"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": format_question(sample)},
                {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
            ]},
        ],
        temperature=float(model.get("temperature", 0.0)),
        top_p=float(model.get("top_p", 0.1)),
        max_tokens=int(model.get("max_tokens", 1000)),
    )
    elapsed = time.perf_counter() - started
    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return {
        "answer": answer,
        "latency_seconds": elapsed,
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def answer_key(answer_text: str, choices: dict[str, Any]) -> str | None:
    keys = [str(key) for key in choices]
    if not keys:
        return None
    match = re.search(r"đáp\s*án\s*:\s*([\w]+)", answer_text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"^\s*([A-Z])\s*[.)]", answer_text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip().upper()
        return next((key for key in keys if key.upper() == candidate), None)
    return None


def summarize(rows: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("answer_correctness") is not None]
    latencies = [row["latency_seconds"] for row in rows]
    input_tokens = [row["input_tokens"] for row in rows if isinstance(row.get("input_tokens"), int)]
    output_tokens = [row["output_tokens"] for row in rows if isinstance(row.get("output_tokens"), int)]
    input_rate = model.get("input_usd_per_million_tokens")
    output_rate = model.get("output_usd_per_million_tokens")
    cost = None
    if input_rate is not None and output_rate is not None and len(input_tokens) == len(rows) and len(output_tokens) == len(rows):
        cost = (sum(input_tokens) * float(input_rate) + sum(output_tokens) * float(output_rate)) / 1_000_000
    return {
        "num_samples": len(rows),
        "answer_correctness": sum(row["answer_correctness"] for row in scored) / len(scored) if scored else None,
        "answer_scored_samples": len(scored),
        "citation": "not_applicable",
        "citation_precision": None,
        "citation_recall": None,
        "latency": mean(latencies) if latencies else None,
        "latency_median_seconds": median(latencies) if latencies else None,
        "input_tokens": sum(input_tokens) if input_tokens else None,
        "output_tokens": sum(output_tokens) if output_tokens else None,
        "estimated_cost_usd": cost,
        "cost_status": "estimated_from_configured_token_rates" if cost is not None else "unavailable_rates_not_configured_or_usage_missing",
    }


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the direct VLM-only baseline")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline2_vlm_only.json"))
    parser.add_argument("--resume", action="store_true", help="Reuse rows already present in the output file")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    model = config["model"]
    outputs = config["outputs"]
    dataset_path = Path(dataset["json_path"])
    image_dir = Path(dataset["image_dir"])
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of samples")
    by_id = {str(sample.get("id")): sample for sample in data}
    missing = [sample_id for sample_id in dataset["sample_ids"] if sample_id not in by_id]
    if missing:
        raise ValueError(f"Dataset is missing configured samples: {missing}")
    selected = [by_id[sample_id] for sample_id in dataset["sample_ids"]]
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    rows_path = Path(outputs["rows"])
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and rows_path.exists():
        saved = json.loads(rows_path.read_text(encoding="utf-8"))
        if saved.get("config_sha256") != config_hash:
            raise ValueError("Cannot resume: output config fingerprint differs")
        existing = {str(row["id"]): row for row in saved.get("rows", [])}

    base_url = os.environ.get(model.get("base_url_env", "LLM_BASE_URL")) or settings.llm_base_url
    configured_keys = os.environ.get(model.get("api_key_env", "LLM_API_KEYS")) or settings.llm_api_keys or ""
    api_keys = [key.strip() for key in configured_keys.split(",") if key.strip()]
    if not base_url or not api_keys:
        raise ValueError(f"Set {model.get('base_url_env', 'LLM_BASE_URL')} and {model.get('api_key_env', 'LLM_API_KEYS')} before running")
    client = OpenAI(base_url=base_url, api_key=api_keys[0])
    rows = dict(existing)
    for sample in tqdm(selected, desc="VLM-only"):
        sample_id = str(sample["id"])
        if sample_id in rows:
            continue
        image_path = resolve_image(image_dir, str(sample["image_id"]))
        prediction = answer_one(client, model, sample, image_path)
        choices = sample.get("choices") or {}
        predicted_key = answer_key(prediction["answer"], choices)
        gold_key = str(sample.get("answer", "")).strip().upper()
        correctness = float(predicted_key == gold_key) if choices and gold_key else None
        rows[sample_id] = {
            "id": sample_id,
            "image_id": sample.get("image_id"),
            "question": sample.get("question"),
            "choices": choices,
            "gold_answer": sample.get("answer"),
            "predicted_choice": predicted_key,
            "answer": prediction["answer"],
            "answer_correctness": correctness,
            "citation_status": "not_applicable",
            "latency_seconds": prediction["latency_seconds"],
            "input_tokens": prediction["input_tokens"],
            "output_tokens": prediction["output_tokens"],
            "total_tokens": prediction["total_tokens"],
        }
        payload = {
            "schema_version": 1,
            "baseline_name": config["name"],
            "status": "in_progress",
            "config_sha256": config_hash,
            "model": model,
            "rows": [rows[sample_id] for sample_id in dataset["sample_ids"] if sample_id in rows],
        }
        rows_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ordered_rows = [rows[sample_id] for sample_id in dataset["sample_ids"]]
    metrics = summarize(ordered_rows, model)
    completed_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    payload = {
        "schema_version": 1,
        "baseline_name": config["name"],
        "status": "complete",
        "completed_at": completed_at,
        "git_revision": git_revision(),
        "config_path": str(args.config),
        "config_sha256": config_hash,
        "model": model,
        "serving": {"base_url": base_url, "api_key_env": model.get("api_key_env", "LLM_API_KEYS")},
        "dataset": {"path": str(dataset_path), "image_dir": str(image_dir), "sample_ids": dataset["sample_ids"]},
        "metrics": metrics,
        "rows": ordered_rows,
    }
    rows_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path = Path(outputs["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({key: value for key, value in payload.items() if key != "rows"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
