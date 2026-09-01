import json
import os
import re
import sys
import time
import torch
import logging
import argparse
import unicodedata as ud

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable

from src.settings import settings
from src.deps.llm_client import LLMClient
from src.deps.qdrant import QdrantVectorStore
from src.core.utils import load_json, save_json, pil_to_base64_png
from src.deps.embeddings import get_embedding_model
from src.core.lawdb.article_lookup import LawDBArticleLookup
from src.core.symbolic.reasoner import build_symbolic_reasoning_context
from src.core.symbolic.sign_normalizer import normalize_detected_sign_to_fact
from src.prompts.answer_prompt import (
    SYSTEM_PROMPT_FREEFORM_CITED_QA,
    SYSTEM_PROMPT_SIGN_VARIANT_DISAMBIGUATION,
)


MAX_ARTICLE_CHARS = 6000
MAX_ARTICLES_IN_PROMPT = 8
SIGN_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}(?:\.[A-Z])?[,\.]\d+[a-z]?)\b",
    re.IGNORECASE,
)


def normalize_text(text: str | int) -> str:
    if isinstance(text, int) and text == 40:
        return "A"
    return ud.normalize("NFC", str(text))


def truncate_text(text: str, max_chars: int = MAX_ARTICLE_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def sanitize_prompt_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized_parts: list[dict[str, Any]] = []
    for part in parts:
        copied = dict(part)
        if copied.get("type") == "image_url":
            copied["image_url"] = {"url": "<IMAGE_CONTENT>"}
        sanitized_parts.append(copied)
    return sanitized_parts


def extract_json_object(response_content: str) -> dict[str, Any]:
    cleaned = response_content.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON response must be an object")
    return parsed


def _citation_evidence(article: dict[str, str]) -> str:
    text = truncate_text(article.get("text", ""), max_chars=700)
    if text:
        return text
    return "Không tìm thấy nội dung điều luật trong LawDB context."


def normalize_freeform_response(
    response_content: str,
    article_contexts: list[dict[str, str]],
    visual_evidence: list[dict[str, str]],
    symbolic_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbolic_context = symbolic_context or {}
    allowed_articles = {
        (article["law_id"], article["article_id"]): article
        for article in article_contexts
    }

    try:
        parsed = extract_json_object(response_content)
        parse_error = ""
    except Exception as exc:
        parsed = {}
        parse_error = str(exc)

    answer = str(parsed.get("answer") or "").strip()
    reasoning = str(parsed.get("reasoning") or "").strip()
    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    citations: list[dict[str, str]] = []
    parsed_citations = parsed.get("citations", [])
    if isinstance(parsed_citations, list):
        for citation in parsed_citations:
            if not isinstance(citation, dict):
                continue
            law_id = str(citation.get("law_id") or "").strip()
            article_id = str(citation.get("article_id") or "").strip()
            article = allowed_articles.get((law_id, article_id))
            if not article:
                continue
            citations.append(
                {
                    "law_id": law_id,
                    "article_id": article_id,
                    "title": str(citation.get("title") or article.get("title") or ""),
                    "evidence": str(
                        citation.get("evidence") or _citation_evidence(article)
                    ).strip(),
                }
            )

    normalized_visual_evidence: list[dict[str, str]] = []
    allowed_visual_by_image: dict[str, list[dict[str, str]]] = {}
    for item in visual_evidence:
        image_name = item.get("image_name", "")
        if image_name:
            allowed_visual_by_image.setdefault(image_name, []).append(item)

    parsed_visual_evidence = parsed.get("visual_evidence", [])
    if isinstance(parsed_visual_evidence, list):
        for item in parsed_visual_evidence:
            if not isinstance(item, dict):
                continue
            image_name = str(item.get("image_name") or "").strip()
            candidates = allowed_visual_by_image.get(image_name, [])
            if not candidates:
                continue
            parsed_sign_name = str(item.get("sign_name") or "").strip()
            source = next(
                (
                    candidate
                    for candidate in candidates
                    if parsed_sign_name
                    and parsed_sign_name.lower() == str(candidate.get("sign_name", "")).lower()
                ),
                None,
            )
            if source is None:
                source = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.get("is_primary")
                    ),
                    candidates[0],
                )
            normalized_visual_evidence.append(
                {
                    "sign_name": str(
                        item.get("sign_name") or source.get("sign_name") or ""
                    ),
                    "description": str(
                        item.get("description") or source.get("description") or ""
                    ),
                    "image_name": image_name,
                }
            )

    if not normalized_visual_evidence:
        normalized_visual_evidence = [
            {
                "sign_name": item.get("sign_name", ""),
                "description": item.get("description", ""),
                "image_name": item.get("image_name", ""),
            }
            for item in visual_evidence
        ]

    if not citations and confidence != "low":
        confidence = "low"

    if not citations:
        citations = [
            {
                "law_id": article["law_id"],
                "article_id": article["article_id"],
                "title": article.get("title", ""),
                "evidence": _citation_evidence(article),
            }
            for article in article_contexts[:3]
        ]

    if parse_error:
        answer = (
            "Chưa đủ thông tin để đưa ra câu trả lời có cấu trúc đáng tin cậy vì "
            "model không trả JSON hợp lệ."
        )
        reasoning = f"Fallback parser được dùng do lỗi JSON: {parse_error}"
        confidence = "low"

    if not answer:
        answer = "Chưa đủ thông tin trong context được cung cấp để trả lời chắc chắn."
        confidence = "low"

    return {
        "freeform_answer": answer,
        "citations": citations,
        "visual_evidence": normalized_visual_evidence,
        "reasoning": reasoning,
        "confidence": confidence,
        "raw_answer_response": response_content,
        "symbolic_decision": symbolic_context.get("symbolic_decision", {}),
        "rules_fired": symbolic_context.get("rules_fired", []),
        "symbolic_facts": symbolic_context.get("symbolic_facts", []),
        "question_intent": symbolic_context.get("question_intent", {}),
        "reasoning_trace": symbolic_context.get("reasoning_trace", {}),
    }


def _article_ref(article: dict[str, Any]) -> dict[str, str]:
    return {
        "law_id": str(article.get("law_id") or "").strip(),
        "article_id": str(article.get("article_id") or "").strip(),
    }


def _dedupe_articles(articles: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for article in articles:
        article_ref = _article_ref(article)
        key = (article_ref["law_id"], article_ref["article_id"])
        if not key[0] or not key[1] or key in seen:
            continue
        seen.add(key)
        deduped.append(article_ref)
    return deduped


def _sign_code(payload: dict[str, Any]) -> str:
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("sign_name", "sign_description", "name", "description")
    )
    match = SIGN_CODE_RE.search(text)
    return match.group(1) if match else ""


def _sign_family(sign_code: str) -> str:
    return re.sub(r"([,\.]\d+)[a-z]$", r"\1", sign_code)


def _has_ambiguous_variants(payloads: list[dict[str, Any]]) -> bool:
    families: dict[str, set[str]] = {}
    for payload in payloads:
        code = _sign_code(payload)
        if not code or not re.search(r"[a-z]$", code):
            continue
        families.setdefault(_sign_family(code), set()).add(code)
    return any(len(codes) > 1 for codes in families.values())


def _variant_hint_from_question(
    sample: dict[str, Any],
    payloads: list[dict[str, Any]],
) -> int | None:
    question = str(sample.get("question") or "").lower()
    explicit_code = SIGN_CODE_RE.search(question)
    if explicit_code:
        target_code = explicit_code.group(1).lower()
        for index, payload in enumerate(payloads):
            if _sign_code(payload).lower() == target_code:
                return index

    hint_to_suffix = {
        "ngày lẻ": "b",
        "ngay le": "b",
        "ngày chẵn": "c",
        "ngay chan": "c",
    }
    for hint, suffix in hint_to_suffix.items():
        if hint in question:
            for index, payload in enumerate(payloads):
                code = _sign_code(payload)
                if code.lower().endswith(suffix):
                    return index
    return None


def _parse_selected_index(
    response_content: str,
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        parsed = extract_json_object(response_content)
    except Exception as exc:
        return {
            "selected_index": 0,
            "confidence": "low",
            "reason": f"Fallback to top-1 because variant JSON parse failed: {exc}",
            "raw_response": response_content,
        }

    selected_index = parsed.get("selected_index", 0)
    if (
        not isinstance(selected_index, int)
        or selected_index < 0
        or selected_index >= len(payloads)
    ):
        selected_index = 0

    confidence = str(parsed.get("confidence") or "low").lower().strip()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "selected_index": selected_index,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip(),
        "raw_response": response_content,
    }


def disambiguate_sign_payloads(
    llm_client: LLMClient | None,
    sample: dict[str, Any],
    sign_image: Image.Image,
    sign_payloads: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not sign_payloads:
        return sign_payloads, {}

    hint_index = _variant_hint_from_question(sample, sign_payloads)
    if hint_index is not None:
        reordered = [
            sign_payloads[hint_index],
            *sign_payloads[:hint_index],
            *sign_payloads[hint_index + 1 :],
        ]
        return reordered, {
            "selected_index": hint_index,
            "confidence": "medium",
            "reason": "Selected from explicit question hint.",
            "method": "question_hint",
        }

    if (
        not llm_client
        or len(sign_payloads) < 2
        or not _has_ambiguous_variants(sign_payloads)
    ):
        return sign_payloads, {"selected_index": 0, "method": "top1"}

    candidates = [
        {
            "index": index,
            "sign_name": payload.get("sign_name", ""),
            "description": payload.get("sign_description", ""),
            "law_id": payload.get("law_id", ""),
            "article_id": payload.get("article_id", ""),
        }
        for index, payload in enumerate(sign_payloads)
    ]
    sign_b64, sign_mime = pil_to_base64_png(sign_image)
    try:
        response_content = llm_client.generate_content(
            contents=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT_SIGN_VARIANT_DISAMBIGUATION,
                        },
                        {
                            "type": "text",
                            "text": (
                                "QUESTION\n"
                                f"{sample.get('question', '')}\n\n"
                                "CANDIDATES\n"
                                f"{json.dumps(candidates, ensure_ascii=False)}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{sign_mime};base64,{sign_b64}"},
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        decision = _parse_selected_index(response_content, sign_payloads)
    except Exception as exc:
        decision = {
            "selected_index": 0,
            "confidence": "low",
            "reason": f"Fallback to top-1 because variant disambiguation failed: {exc}",
            "method": "vlm_variant_disambiguation",
        }
    selected_index = decision["selected_index"]
    reordered = [
        sign_payloads[selected_index],
        *sign_payloads[:selected_index],
        *sign_payloads[selected_index + 1 :],
    ]
    decision["method"] = "vlm_variant_disambiguation"
    return reordered, decision


def collect_retrieved_context(
    sample: dict[str, Any],
    sign_dir: Path,
    embedding_model: Callable,
    vector_store: QdrantVectorStore,
    collection_name: str,
    article_lookup: LawDBArticleLookup,
    llm_client: LLMClient | None = None,
    limit: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    article_refs: list[dict[str, Any]] = []
    visual_evidence: list[dict[str, str]] = []
    sign_prompt_parts: list[dict[str, Any]] = []
    symbolic_facts: list[dict[str, Any]] = []

    for sign in sample.get("detected_signs", []):
        if not sign.get("is_chosen"):
            continue

        sign_image_path = sign_dir / sign["image_name"]
        sign_image = Image.open(sign_image_path).convert("RGB")
        sign_b64, sign_mime = pil_to_base64_png(sign_image)

        sign_payloads = [
            dict(payload)
            for payload in sign.get("retrieved_payloads", [])
            if isinstance(payload, dict)
        ]
        if not sign_payloads:
            embedding = embedding_model.embed_image(sign_image)
            retrieved_results = vector_store.query(collection_name, embedding, top_k=limit)
            sign_payloads = [
                dict(retrieved_result.payload or {})
                for retrieved_result in retrieved_results
            ]

        sign_payloads, variant_decision = disambiguate_sign_payloads(
            llm_client=llm_client,
            sample=sample,
            sign_image=sign_image,
            sign_payloads=sign_payloads,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        primary_payload = sign_payloads[0] if sign_payloads else {}
        sign["variant_decision"] = variant_decision
        if primary_payload:
            sign["image_lawdb_path"] = primary_payload.get("image_path", "")
            sign["sign_name"] = primary_payload.get("sign_name", "")

        sign_visual_evidence: list[dict[str, Any]] = []
        for rank, payload in enumerate(sign_payloads, start=1):
            article_ref = {
                "law_id": payload.get("law_id", ""),
                "article_id": payload.get("article_id", ""),
            }
            article_refs.append(article_ref)

            evidence_item = {
                "sign_name": str(payload.get("sign_name") or ""),
                "description": str(payload.get("sign_description") or ""),
                "image_name": str(sign.get("image_name") or ""),
                "retrieval_rank": rank,
                "is_primary": rank == 1,
            }
            sign_visual_evidence.append(evidence_item)
            visual_evidence.append(evidence_item)

        sign["retrieved_payloads"] = sign_payloads
        symbolic_fact = normalize_detected_sign_to_fact(
            sign=sign,
            payloads=sign_payloads[:1],
            visual_evidence=sign_visual_evidence[:1],
        )
        sign["symbolic_fact"] = symbolic_fact.to_dict()
        symbolic_facts.append(sign["symbolic_fact"])
        sign_prompt_parts.extend(
            [
                {
                    "type": "text",
                    "text": (
                        "DETECTED/RETRIEVED SIGN CONTEXT\n"
                        f"image_name: {sign.get('image_name')}\n"
                        f"primary_visual_evidence: {json.dumps(sign_visual_evidence[:1], ensure_ascii=False)}\n"
                        f"variant_decision: {json.dumps(variant_decision, ensure_ascii=False)}\n"
                        f"retrieved_payloads: {json.dumps(sign_payloads, ensure_ascii=False)}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{sign_mime};base64,{sign_b64}"},
                },
            ]
        )

    article_refs.extend(sample.get("relevant_articles", []))
    article_refs.extend(sample.get("predicted_articles", []))
    article_contexts = article_lookup.get_many(_dedupe_articles(article_refs))
    return article_contexts, visual_evidence, sign_prompt_parts, symbolic_facts


def build_freeform_prompt_parts(
    sample: dict[str, Any],
    image_dir: Path,
    article_contexts: list[dict[str, str]],
    visual_evidence: list[dict[str, str]],
    sign_prompt_parts: list[dict[str, Any]],
    symbolic_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    image_path = image_dir / f"{sample['image_id']}.jpg"
    image = Image.open(image_path).convert("RGB")
    org_b64, org_mime = pil_to_base64_png(image)

    symbolic_context = symbolic_context or {}
    choices = sample.get("choices")
    user_context = {
        "question": sample.get("question", ""),
        "choices_optional_reference": choices if choices else None,
        "visual_evidence_available": visual_evidence,
        "symbolic_decision": symbolic_context.get("symbolic_decision", {}),
        "rules_fired": symbolic_context.get("rules_fired", []),
    }

    article_lines: list[str] = []
    for index, article in enumerate(article_contexts[:MAX_ARTICLES_IN_PROMPT], start=1):
        article_lines.append(
            "\n".join(
                [
                    f"ARTICLE {index}",
                    f"law_id: {article['law_id']}",
                    f"article_id: {article['article_id']}",
                    f"title: {article.get('title', '')}",
                    f"text: {truncate_text(article.get('text', ''))}",
                ]
            )
        )
    legal_context = "\n\n".join(article_lines) or "No legal article context available."

    return [
        {"type": "text", "text": SYSTEM_PROMPT_FREEFORM_CITED_QA},
        {
            "type": "text",
            "text": (
                "USER QUESTION CONTEXT\n"
                f"{json.dumps(user_context, ensure_ascii=False)}\n\n"
                "SYMBOLIC REASONING CONTEXT\n"
                f"{json.dumps(symbolic_context, ensure_ascii=False)}\n\n"
                "LEGAL ARTICLE CONTEXT\n"
                f"{legal_context}"
            ),
        },
        {"type": "text", "text": "INPUT IMAGE"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{org_mime};base64,{org_b64}"},
        },
        *sign_prompt_parts,
    ]


def get_freeform_answer(
    llm_client: LLMClient,
    sample: dict[str, Any],
    image_dir: Path,
    sign_dir: Path,
    embedding_model: Callable,
    vector_store: QdrantVectorStore,
    collection_name: str,
    article_lookup: LawDBArticleLookup,
    limit: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 5000,
) -> dict[str, Any]:
    t1 = time.time()
    try:
        (
            article_contexts,
            visual_evidence,
            sign_prompt_parts,
            symbolic_facts,
        ) = collect_retrieved_context(
            sample=sample,
            sign_dir=sign_dir,
            embedding_model=embedding_model,
            vector_store=vector_store,
            collection_name=collection_name,
            article_lookup=article_lookup,
            llm_client=llm_client,
            limit=limit,
            temperature=temperature,
            max_tokens=min(max_tokens, 500),
        )
        symbolic_context = build_symbolic_reasoning_context(
            symbolic_facts=symbolic_facts,
            question=str(sample.get("question") or ""),
            llm_client=llm_client,
            use_llm_fallback=True,
            temperature=temperature,
            max_tokens=min(max_tokens, 300),
        )

        parts = build_freeform_prompt_parts(
            sample=sample,
            image_dir=image_dir,
            article_contexts=article_contexts,
            visual_evidence=visual_evidence,
            sign_prompt_parts=sign_prompt_parts,
            symbolic_context=symbolic_context,
        )

        response_content: str = llm_client.generate_content(
            contents=[{"role": "user", "content": parts}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = normalize_freeform_response(
            response_content=response_content,
            article_contexts=article_contexts,
            visual_evidence=visual_evidence,
            symbolic_context=symbolic_context,
        )
        return {
            **sample,
            **parsed,
            "time_second": round(time.time() - t1, 5),
            "answer_prompt_parts": parts,
            "article_contexts": article_contexts,
        }
    except Exception as exc:
        response_content = f"Error response: {exc}"
        parsed = normalize_freeform_response(
            response_content=response_content,
            article_contexts=[],
            visual_evidence=[],
        )
        return {
            **sample,
            **parsed,
            "time_second": round(time.time() - t1, 5),
            "answer_prompt_parts": [],
            "article_contexts": [],
            "symbolic_decision": {},
            "rules_fired": [],
            "symbolic_facts": [],
            "question_intent": {},
            "reasoning_trace": {},
        }


def get_answer(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return get_freeform_answer(*args, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Get free-form cited answer for each sample")
    parser.add_argument(
        "--set_name",
        "-s",
        type=str,
        default="train",
        choices=["train", "public_test", "private_test"],
        help="Dataset set to process",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        api_keys=[api_key for api_key in settings.llm_api_keys.split(",")],
        model_id=settings.llm_model,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder, _ = get_embedding_model(settings.embedding_name, device=device)
    vector_store = QdrantVectorStore(uri=settings.qdrant_uri)
    article_lookup = LawDBArticleLookup()

    if args.set_name == "train":
        input_json_path = settings.train_extracted_sign_json_path
        original_image_dir = Path(settings.train_image_dir)
        extracted_sign_dir = Path(settings.train_extracted_sign_dir)
        save_json_path = Path(settings.train_extracted_sign_json_path)
    elif args.set_name == "public_test":
        input_json_path = settings.public_test_extracted_sign_json_path
        original_image_dir = Path(settings.public_test_image_dir)
        extracted_sign_dir = Path(settings.public_test_extracted_sign_dir)
        save_json_path = Path(settings.public_test_extracted_sign_json_path)
    elif args.set_name == "private_test":
        input_json_path = settings.private_test_extracted_sign_json_path
        original_image_dir = Path(settings.private_test_image_dir)
        extracted_sign_dir = Path(settings.private_test_extracted_sign_dir)
        save_json_path = Path(settings.private_test_extracted_sign_json_path)
    else:
        print(f"[ERROR] Unknown set name: {args.set_name}")
        sys.exit(1)

    extracted_sign_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[..] Configuration:")
    logging.info("  Set Name: %s", args.set_name)
    logging.info("  Input JSON Path: %s", input_json_path)
    logging.info("  Input Image Dir: %s", original_image_dir)
    logging.info("  Output Image Dir: %s", extracted_sign_dir)
    logging.info("  Output JSON Path: %s", save_json_path)
    logging.info("  LawDB Lookup Path: %s", article_lookup.lawdb_path)

    try:
        samples: list[dict[str, Any]] = load_json(input_json_path)
    except FileNotFoundError:
        print(f"[ERROR] Input JSON file not found: {input_json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load input JSON: {e}")
        sys.exit(1)

    for sample in tqdm(samples, total=len(samples)):
        if args.set_name == "train" and "answer" in sample:
            sample["answer"] = normalize_text(sample["answer"])

        response = get_freeform_answer(
            llm_client=llm_client,
            sample=sample,
            image_dir=original_image_dir,
            sign_dir=extracted_sign_dir,
            embedding_model=embedder,
            vector_store=vector_store,
            collection_name=settings.collection_name,
            article_lookup=article_lookup,
            limit=settings.limit,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_new_tokens,
        )

        for field in (
            "freeform_answer",
            "citations",
            "visual_evidence",
            "reasoning",
            "confidence",
            "raw_answer_response",
            "time_second",
            "article_contexts",
            "symbolic_decision",
            "rules_fired",
            "symbolic_facts",
            "question_intent",
            "reasoning_trace",
        ):
            sample[field] = response[field]

        sample["answer_prompt_parts"] = sanitize_prompt_parts(
            response["answer_prompt_parts"]
        )

        if args.set_name in ["public_test", "private_test"]:
            sample["answer"] = response["freeform_answer"]

        save_json(samples, save_json_path)

    if args.set_name in ["public_test", "private_test"]:
        submission_samples = []
        for sample in samples:
            submission_samples.append(
                {
                    "id": sample.get("id"),
                    "image_id": sample.get("image_id"),
                    "answer": sample.get("freeform_answer"),
                    "citations": sample.get("citations", []),
                    "visual_evidence": sample.get("visual_evidence", []),
                    "reasoning": sample.get("reasoning", ""),
                    "confidence": sample.get("confidence", "low"),
                    "symbolic_decision": sample.get("symbolic_decision", {}),
                    "rules_fired": sample.get("rules_fired", []),
                    "symbolic_facts": sample.get("symbolic_facts", []),
                    "question_intent": sample.get("question_intent", {}),
                    "reasoning_trace": sample.get("reasoning_trace", {}),
                }
            )

        save_json(
            submission_samples,
            os.path.join(os.path.dirname(save_json_path), "submission_task2.json"),
        )


if __name__ == "__main__":
    main()
