from __future__ import annotations

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
from src.core.query_signs import query_relevant_signs
from src.core.lawdb.article_lookup import LawDBArticleLookup
from src.core.sub_task_2 import get_freeform_answer, sanitize_prompt_parts


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


def answer_image_question(
    image_path: Path,
    question: str,
    choices: str | None = None,
    debug: bool = False,
    resources: FreeformQAResources | None = None,
    use_vlm_filter: bool = True,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    run_id = uuid4().hex
    work_dir = Path(tempfile.gettempdir()) / "lexisign_freeform_demo" / run_id
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

    return result
