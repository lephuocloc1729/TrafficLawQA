from __future__ import annotations

import hashlib
import json
import time
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from PIL import Image

from src.settings import settings
from src.deps.llm_client import LLMClient
from src.deps.qdrant import QdrantVectorStore
from src.deps.embeddings import get_embedding_model
from src.deps.detection import SignExtractor
from src.core.extract_signs import extract_signs
from src.core.filter_signs import filter_relevant_signs
from src.core.sign_parse import build_sign_groups, parse_detected_signs
from src.core.query_signs import query_relevant_signs
from src.core.lawdb.article_lookup import LawDBArticleLookup
from src.core.sub_task_2 import get_freeform_answer, sanitize_prompt_parts


DEMO_WORK_DIR = Path(tempfile.gettempdir()) / "traffic_sign_qa_demo"
DEMO_CACHE_DIR = Path("outputs/demo_cache")
DEMO_PIPELINE_VERSION = "freeform-qa-sign-parse-groups-json-repair-v1"


@dataclass
class FreeformQAResources:
    llm_client: LLMClient
    embedder: Any
    sign_extractor: SignExtractor
    vector_store: QdrantVectorStore
    article_lookup: LawDBArticleLookup
    device: str


def create_freeform_qa_resources(device: str | None = None) -> FreeformQAResources:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        api_keys=[api_key for api_key in settings.llm_api_keys.split(",")],
        model_id=settings.llm_model,
    )
    embedder, _ = get_embedding_model(settings.embedding_name, device=device)
    sign_extractor = SignExtractor(
        detector=settings.image_query_detection_model,
        device=device,
    )
    vector_store = QdrantVectorStore(uri=settings.qdrant_uri)
    article_lookup = LawDBArticleLookup()
    return FreeformQAResources(
        llm_client=llm_client,
        embedder=embedder,
        sign_extractor=sign_extractor,
        vector_store=vector_store,
        article_lookup=article_lookup,
        device=device,
    )


def _prepare_demo_image(image_path: Path, image_dir: Path, image_id: str) -> Path:
    output_path = image_dir / f"{image_id}.jpg"
    try:
        image = Image.open(image_path).convert("RGB")
        image.save(output_path, format="JPEG")
    except Exception:
        shutil.copyfile(image_path, output_path)
    return output_path


def clear_demo_cache(cache_dir: Path | None = None) -> None:
    cache_root = cache_dir or DEMO_CACHE_DIR
    if cache_root.exists():
        shutil.rmtree(cache_root)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _demo_cache_key(
    image_path: Path,
    question: str,
    choices: str | None,
    use_vlm_filter: bool,
) -> str:
    payload = {
        "image_sha256": _hash_file(image_path),
        "pipeline_version": DEMO_PIPELINE_VERSION,
        "question": question.strip(),
        "choices": (choices or "").strip(),
        "use_vlm_filter": use_vlm_filter,
        "llm_model": settings.llm_model,
        "embedding_name": settings.embedding_name,
        "query_detector": settings.image_query_detection_model,
        "law_detector": settings.image_lawdb_detection_model,
        "collection_name": settings.collection_name,
        "retrieval_limit": settings.limit,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_entry_dir(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / cache_key[:2] / cache_key


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _prepare_result_for_cache(result: dict[str, Any], entry_dir: Path) -> dict[str, Any]:
    cached_result = _json_safe(result)
    cached_result["answer_prompt_parts"] = sanitize_prompt_parts(
        cached_result.get("answer_prompt_parts", [])
    )

    entry_dir.mkdir(parents=True, exist_ok=True)
    input_image_path = cached_result.get("input_image_path")
    if input_image_path:
        source = Path(input_image_path)
        if source.exists():
            target = entry_dir / f"input{source.suffix or '.jpg'}"
            shutil.copyfile(source, target)
            cached_result["input_image_path"] = str(target)

    signs_dir = entry_dir / "signs"
    signs_dir.mkdir(parents=True, exist_ok=True)
    for sign in cached_result.get("detected_signs", []):
        image_path = sign.get("image_path")
        if not image_path:
            continue
        source = Path(image_path)
        if not source.exists():
            continue
        target = signs_dir / source.name
        shutil.copyfile(source, target)
        sign["image_path"] = str(target)
    cached_result["sign_dir"] = str(signs_dir)
    return cached_result


def _load_cached_answer(
    cache_dir: Path,
    cache_key: str,
    lookup_start: float,
) -> dict[str, Any] | None:
    result_path = _cache_entry_dir(cache_dir, cache_key) / "result.json"
    if not result_path.exists():
        return None
    with result_path.open("r", encoding="utf-8") as file_obj:
        result = json.load(file_obj)
    original_performance = result.get("performance") or {}
    result["cache_hit"] = True
    result["cache_key"] = cache_key
    result["cached_original_performance"] = original_performance
    lookup_seconds = time.perf_counter() - lookup_start
    result["performance"] = {
        "cache_lookup": round(lookup_seconds, 5),
        "total": round(lookup_seconds, 5),
        "original_total": original_performance.get("total"),
    }
    return result


def _save_cached_answer(
    result: dict[str, Any],
    cache_dir: Path,
    cache_key: str,
) -> None:
    entry_dir = _cache_entry_dir(cache_dir, cache_key)
    cached_result = _prepare_result_for_cache(result, entry_dir)
    cached_result["cache_hit"] = False
    cached_result["cache_key"] = cache_key
    result_path = entry_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as file_obj:
        json.dump(cached_result, file_obj, ensure_ascii=False, indent=2)


def warm_up_demo(resources: FreeformQAResources | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {}
    start = time.perf_counter()
    resources = resources or create_freeform_qa_resources()
    report["load_resources"] = round(time.perf_counter() - start, 5)
    report["device"] = resources.device

    t0 = time.perf_counter()
    try:
        resources.vector_store.client.get_collection(settings.collection_name)
        report["qdrant"] = {
            "ok": True,
            "seconds": round(time.perf_counter() - t0, 5),
            "collection": settings.collection_name,
        }
    except Exception as exc:
        report["qdrant"] = {
            "ok": False,
            "seconds": round(time.perf_counter() - t0, 5),
            "error": f"{type(exc).__name__}: {exc}",
        }

    t0 = time.perf_counter()
    try:
        resources.llm_client.generate_content(
            contents=[
                {
                    "role": "user",
                    "content": "Trả lời đúng một từ: OK",
                }
            ],
            temperature=0.0,
            max_tokens=8,
        )
        report["llm"] = {
            "ok": True,
            "seconds": round(time.perf_counter() - t0, 5),
            "model": settings.llm_model,
        }
    except Exception as exc:
        report["llm"] = {
            "ok": False,
            "seconds": round(time.perf_counter() - t0, 5),
            "error": f"{type(exc).__name__}: {exc}",
        }

    report["total"] = round(time.perf_counter() - start, 5)
    return report


def answer_image_question(
    image_path: Path,
    question: str,
    choices: str | None = None,
    debug: bool = False,
    resources: FreeformQAResources | None = None,
    use_vlm_filter: bool = True,
    use_cache: bool = True,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    cache_root = cache_dir or DEMO_CACHE_DIR
    cache_key: str | None = None
    if use_cache:
        cache_key = _demo_cache_key(
            image_path=Path(image_path),
            question=question,
            choices=choices,
            use_vlm_filter=use_vlm_filter,
        )
        cached_result = _load_cached_answer(cache_root, cache_key, total_start)
        if cached_result is not None:
            return cached_result
    timings["cache_lookup"] = round(time.perf_counter() - total_start, 5)

    run_id = uuid4().hex
    work_dir = DEMO_WORK_DIR / run_id
    image_dir = work_dir / "images"
    sign_dir = work_dir / "signs"
    image_dir.mkdir(parents=True, exist_ok=True)
    sign_dir.mkdir(parents=True, exist_ok=True)

    image_id = "demo"
    prepared_image_path = _prepare_demo_image(Path(image_path), image_dir, image_id)

    sample: dict[str, Any] = {
        "id": run_id,
        "image_id": image_id,
        "question": question,
    }
    if choices:
        sample["choices"] = choices

    t0 = time.perf_counter()
    resources = resources or create_freeform_qa_resources()
    timings["load_resources"] = round(time.perf_counter() - t0, 5)

    t0 = time.perf_counter()
    sample = extract_signs(
        sign_extractor=resources.sign_extractor,
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
    )
    timings["detect_signs"] = round(time.perf_counter() - t0, 5)

    t0 = time.perf_counter()
    sample = parse_detected_signs(
        llm_client=resources.llm_client,
        sample=sample,
        sign_dir=sign_dir,
        temperature=settings.llm_temperature,
        max_tokens=min(settings.llm_max_new_tokens, 700),
    )
    timings["parse_detected_signs"] = round(time.perf_counter() - t0, 5)

    t0 = time.perf_counter()
    if use_vlm_filter:
        sample = filter_relevant_signs(
            llm_client=resources.llm_client,
            sample=sample,
            image_dir=image_dir,
            sign_dir=sign_dir,
            batch_size=10,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_new_tokens,
        )
    else:
        for sign in sample.get("detected_signs", []):
            sign["is_chosen"] = True
            sign["relevance_reason"] = "Selected by fast mode without VLM filtering."
        sample = build_sign_groups(sample)
    timings["filter_relevant_signs"] = round(time.perf_counter() - t0, 5)

    t0 = time.perf_counter()
    sample = query_relevant_signs(
        embedding_model=resources.embedder,
        vector_store=resources.vector_store,
        sample=sample,
        sign_dir=sign_dir,
        collection_name=settings.collection_name,
        limit=settings.limit,
        set_name="public_test",
    )
    timings["retrieve_lawdb"] = round(time.perf_counter() - t0, 5)

    t0 = time.perf_counter()
    result = get_freeform_answer(
        llm_client=resources.llm_client,
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
        embedding_model=resources.embedder,
        vector_store=resources.vector_store,
        collection_name=settings.collection_name,
        article_lookup=resources.article_lookup,
        limit=settings.limit,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_new_tokens,
    )
    timings["answer_generation"] = round(time.perf_counter() - t0, 5)
    timings["total"] = round(time.perf_counter() - total_start, 5)
    result["performance"] = timings
    result["cache_hit"] = False
    result["cache_key"] = cache_key

    result["input_image_path"] = str(prepared_image_path)
    result["sign_dir"] = str(sign_dir)
    result["detected_signs"] = [
        {
            **sign,
            "image_path": str(sign_dir / sign["image_name"]),
        }
        for sign in result.get("detected_signs", [])
    ]

    if not debug:
        result["answer_prompt_parts"] = sanitize_prompt_parts(
            result.get("answer_prompt_parts", [])
        )

    if use_cache and cache_key:
        try:
            _save_cached_answer(result, cache_root, cache_key)
        except Exception as exc:
            result["cache_error"] = f"{type(exc).__name__}: {exc}"

    return result
