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
from src.core.article_context import (
    article_candidate_from_payload,
    build_article_contexts,
    rule_article_candidates,
)
from src.core.symbolic.reasoner import build_symbolic_reasoning_context
from src.core.symbolic.sign_normalizer import normalize_detected_sign_to_observation
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
    article_contexts: list[dict[str, Any]],
    visual_evidence: list[dict[str, Any]],
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

    # Do not let the free-form model turn an uncovered or incomplete sign rule
    # into a permission claim. For sign permission intents, the
    # symbolic outcome is the authority; the LLM may explain it, but it cannot
    # promote UNKNOWN/CONDITIONAL to ALLOWED (or PROHIBITED).
    query_intent = symbolic_context.get("query_intent", {})
    intents = query_intent.get("intents", []) if isinstance(query_intent, dict) else []
    intent_kinds = {
        str(item.get("kind") or "UNKNOWN").upper()
        for item in intents
        if isinstance(item, dict)
    }
    if intent_kinds and intent_kinds.issubset({"PENALTY", "UNKNOWN"}):
        answer = "Hiện hệ thống chưa hỗ trợ câu hỏi về mức phạt hoặc chưa xác định được yêu cầu."
        citations = []
        confidence = "low"
        reasoning = "Intent thuộc nhóm PENALTY/UNKNOWN, hiện chưa được hệ thống hỗ trợ."
    sign_permission_intent = any(
        isinstance(item, dict)
        and str(item.get("kind") or "").upper() == "CHECK_PERMISSION"
        for item in intents
    )
    traffic_facts = symbolic_context.get("facts", {})
    observations = traffic_facts.get("sign_observations", []) if isinstance(traffic_facts, dict) else []
    # An ambiguous/unidentified observation is still a sign-focused request;
    # it must not fall through to a free-form permission inference.
    target_has_sign = isinstance(observations, list) and bool(observations)
    if not target_has_sign:
        for item in intents:
            target = item.get("target") if isinstance(item, dict) else None
            if isinstance(target, str) and target in {"scene", "sign_observation", "sign_code"}:
                target_has_sign = True
                break
            if isinstance(target, dict) and (
                target.get("sign_codes") or target.get("kind") == "sign_observation"
            ):
                target_has_sign = True
                break
    symbolic_decision = symbolic_context.get("symbolic_decision", {})
    outcome = (
        str(symbolic_decision.get("decision") or "UNKNOWN").upper()
        if isinstance(symbolic_decision, dict)
        else "UNKNOWN"
    )
    if sign_permission_intent and target_has_sign and outcome in {"UNKNOWN", "CONDITIONAL"}:
        conclusion = (
            str(symbolic_decision.get("conclusion") or symbolic_decision.get("reason") or "").strip()
            if isinstance(symbolic_decision, dict)
            else ""
        )
        answer = (
            f"Chưa thể kết luận được phép hay bị cấm (kết quả {outcome})."
            + (f" {conclusion}" if conclusion else "")
        )
        reasoning = "Kết quả được giữ ở trạng thái fail-closed từ rule engine; câu trả lời tự do không được nâng thành kết luận pháp lý."
        citations = []
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
        "traffic_facts": symbolic_context.get("facts", {}),
        "query_intent": symbolic_context.get("query_intent", {}),
        "query_context_facts": symbolic_context.get("query_context_facts", []),
        "intent_evaluations": symbolic_context.get("intent_evaluations", []),
        "rule_evaluations": symbolic_context.get("rule_evaluations", []),
        "sign_meaning_evaluations": symbolic_context.get("sign_meaning_evaluations", []),
        "reasoning_trace": symbolic_context.get("reasoning_trace", {}),
    }


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
    embedding_model: Callable | None,
    vector_store: QdrantVectorStore | None,
    collection_name: str,
    article_lookup: LawDBArticleLookup,
    llm_client: LLMClient | None = None,
    limit: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    article_candidates: list[dict[str, Any]] = []
    visual_evidence: list[dict[str, Any]] = []
    sign_prompt_parts: list[dict[str, Any]] = []
    traffic_sign_observations: list[dict[str, Any]] = []

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
        can_query_fallback = (
            not sign_payloads
            and not sign.get("retrieval_attempted")
            and settings.enable_sign_retrieval_fallback
            and embedding_model is not None
            and vector_store is not None
            and not (
                sign.get("identity_status") == "identified" and sign.get("sign_code")
            )
        )
        if can_query_fallback:
            embedding = embedding_model.embed_image(sign_image)
            retrieved_results = vector_store.query(collection_name, embedding, top_k=limit)
            sign_payloads = [
                dict(retrieved_result.payload or {})
                for retrieved_result in retrieved_results
            ]

        if sign_payloads:
            sign_payloads, variant_decision = disambiguate_sign_payloads(
                llm_client=llm_client,
                sample=sample,
                sign_image=sign_image,
                sign_payloads=sign_payloads,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            detector_confidence = float(sign.get("confidence") or 0.0)
            variant_decision = {
                "selected_index": None,
                "confidence": (
                    "high" if detector_confidence >= 0.8 else
                    "medium" if detector_confidence >= 0.6 else "low"
                ),
                "reason": (
                    "Canonical identity came directly from the YOLOv8m QCVN crosswalk."
                    if sign.get("sign_code")
                    else f"YOLOv8m mapped the sign to family {sign.get('sign_family')}; exact variant remains unresolved."
                    if sign.get("sign_family")
                    else "No retrieval fallback was run; sign identity remains unresolved."
                ),
                "method": (
                    "yolov8m_qcvn_crosswalk"
                    if sign.get("sign_code") or sign.get("sign_family")
                    else "retrieval_disabled"
                ),
            }
        primary_payload = sign_payloads[0] if sign_payloads else {}
        sign["variant_decision"] = variant_decision
        if primary_payload:
            sign["image_lawdb_path"] = primary_payload.get("image_path", "")
            sign["sign_name"] = primary_payload.get("sign_name", "")

        sign_visual_evidence: list[dict[str, Any]] = []
        for rank, payload in enumerate(sign_payloads, start=1):
            article_candidate = article_candidate_from_payload(
                payload,
                retrieval_rank=rank,
                context_source="retrieval",
            )
            if article_candidate:
                article_candidates.append(article_candidate)

            evidence_item = {
                "sign_code": _sign_code(payload),
                "sign_family": str(payload.get("sign_family") or ""),
                "sign_name": str(payload.get("sign_name") or ""),
                "description": str(payload.get("sign_description") or ""),
                "legal_conditions": payload.get("legal_conditions", {}),
                "image_name": str(sign.get("image_name") or ""),
                "retrieval_rank": rank,
                "is_primary": rank == 1,
            }
            sign_visual_evidence.append(evidence_item)
            visual_evidence.append(evidence_item)

        sign["retrieved_payloads"] = sign_payloads
        observation_source = "yolov8m" if sign.get("source") == "yolov8m" else "retrieval"
        sign_observation = normalize_detected_sign_to_observation(
            sign=sign,
            payloads=sign_payloads,
            visual_evidence=sign_visual_evidence,
            source=observation_source,
        )
        sign["sign_observation"] = sign_observation.to_dict()
        traffic_sign_observations.append(sign["sign_observation"])
        sign_prompt_parts.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"DETECTED SIGN CROP: {sign.get('image_name')}\n"
                        "VARIANT RESOLUTION: "
                        f"{json.dumps(variant_decision, ensure_ascii=False)}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{sign_mime};base64,{sign_b64}"},
                },
            ]
        )

    predicted_article_refs = sample.get("predicted_articles") or []
    sample_article_refs = predicted_article_refs or sample.get("relevant_articles", [])
    for article_ref in sample_article_refs:
        if not isinstance(article_ref, dict):
            continue
        if (
            article_candidates
            and not article_ref.get("source")
            and not predicted_article_refs
        ):
            continue
        candidate = article_candidate_from_payload(article_ref)
        if candidate:
            article_candidates.append(candidate)

    article_contexts = build_article_contexts(
        article_candidates,
        article_lookup=article_lookup,
        max_articles=MAX_ARTICLES_IN_PROMPT,
    )
    return article_contexts, visual_evidence, sign_prompt_parts, traffic_sign_observations


def build_freeform_prompt_parts(
    sample: dict[str, Any],
    image_dir: Path,
    article_contexts: list[dict[str, Any]],
    visual_evidence: list[dict[str, Any]],
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
        "query_intent": symbolic_context.get("query_intent", {}),
        "symbolic_decision": symbolic_context.get("symbolic_decision", {}),
        "rules_fired": symbolic_context.get("rules_fired", []),
    }

    def format_article_section(
        heading: str,
        articles: list[dict[str, Any]],
    ) -> str:
        article_lines: list[str] = []
        for index, article in enumerate(articles, start=1):
            article_lines.append(
                "\n".join(
                    [
                        f"ARTICLE {index}",
                        f"law_id: {article['law_id']}",
                        f"article_id: {article['article_id']}",
                        f"title: {article.get('title', '')}",
                        f"context_source: {article.get('context_source', '')}",
                        f"retrieval_rank: {article.get('retrieval_rank')}",
                        "related_sign_codes: "
                        f"{json.dumps(article.get('related_sign_codes', []), ensure_ascii=False)}",
                        f"text: {truncate_text(article.get('text', ''))}",
                    ]
                )
            )
        content = "\n\n".join(article_lines) or "None"
        return f"{heading}\n{content}"

    primary_articles = [
        article
        for article in article_contexts
        if article.get("context_source") == "retrieval"
    ]
    related_articles = [
        article
        for article in article_contexts
        if article.get("context_source") != "retrieval"
    ]
    legal_context = "\n\n".join(
        [
            format_article_section("PRIMARY ARTICLE", primary_articles),
            format_article_section("RELATED ARTICLE", related_articles),
        ]
    )
    sign_evidence = [
        {
            "sign_code": item.get("sign_code", ""),
            "sign_family": item.get("sign_family", ""),
            "sign_name": item.get("sign_name", ""),
            "description": truncate_text(
                str(item.get("description") or ""), max_chars=700
            ),
            "legal_conditions": item.get("legal_conditions", {}),
            "image_name": item.get("image_name", ""),
            "retrieval_rank": item.get("retrieval_rank"),
        }
        for item in _primary_evidence_by_sign(visual_evidence)
    ]
    sign_specific_context = (
        json.dumps(sign_evidence, ensure_ascii=False) if sign_evidence else "None"
    )

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
                f"{legal_context}\n\n"
                "SIGN-SPECIFIC EVIDENCE\n"
                f"{sign_specific_context}"
            ),
        },
        {"type": "text", "text": "INPUT IMAGE"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{org_mime};base64,{org_b64}"},
        },
        *sign_prompt_parts,
    ]


def _primary_evidence_by_sign(
    visual_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_by_image: dict[str, dict[str, Any]] = {}
    for item in visual_evidence:
        image_name = str(item.get("image_name") or "")
        if not image_name:
            continue
        if image_name not in primary_by_image or item.get("is_primary"):
            primary_by_image[image_name] = item
    return list(primary_by_image.values())[:MAX_ARTICLES_IN_PROMPT]


def get_freeform_answer(
    llm_client: LLMClient,
    sample: dict[str, Any],
    image_dir: Path,
    sign_dir: Path,
    embedding_model: Callable | None,
    vector_store: QdrantVectorStore | None,
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
            traffic_sign_observations,
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
            traffic_facts=traffic_sign_observations,
            question=str(sample.get("question") or ""),
            llm_client=llm_client,
            use_llm=True,
            temperature=temperature,
            max_tokens=min(max_tokens, 300),
        )
        article_contexts = build_article_contexts(
            [*article_contexts, *rule_article_candidates(symbolic_context)],
            article_lookup=article_lookup,
            max_articles=MAX_ARTICLES_IN_PROMPT,
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
            "traffic_facts": {},
            "query_intent": {},
            "query_context_facts": [],
            "intent_evaluations": [],
            "rule_evaluations": [],
            "sign_meaning_evaluations": [],
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
            "traffic_facts",
            "query_intent",
            "query_context_facts",
            "intent_evaluations",
            "rule_evaluations",
            "sign_meaning_evaluations",
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
                    "traffic_facts": sample.get("traffic_facts", {}),
                    "query_intent": sample.get("query_intent", {}),
                    "query_context_facts": sample.get("query_context_facts", []),
                    "intent_evaluations": sample.get("intent_evaluations", []),
                    "rule_evaluations": sample.get("rule_evaluations", []),
                    "sign_meaning_evaluations": sample.get("sign_meaning_evaluations", []),
                    "reasoning_trace": sample.get("reasoning_trace", {}),
                }
            )

        save_json(
            submission_samples,
            os.path.join(os.path.dirname(save_json_path), "submission_task2.json"),
        )


if __name__ == "__main__":
    main()
