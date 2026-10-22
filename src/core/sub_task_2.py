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
from src.core.symbolic.fact_schema import SignObservation, TrafficFacts
from src.core.symbolic.intent_schema import normalize_intent_kind
from src.core.symbolic.sign_normalizer import normalize_detected_sign_to_observation, normalize_sign_code
from src.core.symbolic.sign_relations import link_supplementary_signs
from src.core.symbolic.traffic_sign_ocr import get_traffic_sign_ocr
from src.core.symbolic.sign_information_facts import extract_nearby_highway_profile_facts
from src.prompts.answer_prompt import (
    SYSTEM_PROMPT_FREEFORM_CITED_QA,
    SYSTEM_PROMPT_SIGN_VARIANT_DISAMBIGUATION,
    SYSTEM_PROMPT_TRAFFICLAWQA_RAG,
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
        normalize_intent_kind(str(item.get("kind") or "UNKNOWN"))
        for item in intents
        if isinstance(item, dict)
    }
    if intent_kinds and intent_kinds.issubset({"UNKNOWN"}):
        answer = "Hiện hệ thống chưa xác định hoặc chưa hỗ trợ yêu cầu này."
        citations = []
        confidence = "low"
        reasoning = "Intent UNKNOWN hiện chưa được hệ thống hỗ trợ."
    sign_permission_intent = any(
        isinstance(item, dict)
        and normalize_intent_kind(str(item.get("kind") or "")) == "CHECK_PERMISSION"
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
    direct = normalize_sign_code(str(payload.get("sign_code") or ""))
    if direct:
        return direct
    text = " ".join(
        str(payload.get(key) or "")
        for key in ("sign_name", "sign_description", "name", "description")
    )
    match = SIGN_CODE_RE.search(text)
    return match.group(1) if match else ""


def _sign_family(sign_code: str) -> str:
    return re.sub(r"([,\.]\d+)[a-z]$", r"\1", sign_code)


def _has_multiple_candidate_identities(payloads: list[dict[str, Any]]) -> bool:
    identities = set()
    for payload in payloads:
        code = _sign_code(payload)
        attributes = payload.get("attributes") or {}
        conditions = payload.get("normalized_conditions") or {}
        if code:
            identities.add((
                code,
                json.dumps(attributes, ensure_ascii=False, sort_keys=True, default=str),
                json.dumps(conditions, ensure_ascii=False, sort_keys=True, default=str),
            ))
        else:
            identities.add((
                str(payload.get("sign_name") or payload.get("name") or "").strip().casefold(),
                str(payload.get("sign_description") or payload.get("description") or "").strip().casefold(),
            ))
    return len(identities) > 1


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
            "selected_index": None,
            "confidence": "low",
            "reason": f"Candidate selection abstained because response JSON was invalid: {exc}",
            "raw_response": response_content,
        }

    selected_index = parsed.get("selected_index")
    if (
        not isinstance(selected_index, int)
        or selected_index < 0
        or selected_index >= len(payloads)
    ):
        selected_index = None

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
    trafficlawqa_rag_compatibility: bool = False,
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
            "identity_confirmed": True,
            "reason": "Selected from explicit question hint.",
            "method": "question_hint",
        }

    if trafficlawqa_rag_compatibility:
        # Match the reference project's policy: only ask the VLM to resolve
        # lettered variants within the same sign family; otherwise keep top-1.
        ambiguous_families: dict[str, set[str]] = {}
        for payload in sign_payloads:
            code = _sign_code(payload)
            if code and re.search(r"[a-z]$", code):
                ambiguous_families.setdefault(_sign_family(code), set()).add(code)
        has_ambiguous_variants = any(len(codes) > 1 for codes in ambiguous_families.values())
        if len(sign_payloads) < 2 or not has_ambiguous_variants or not llm_client:
            return sign_payloads, {"selected_index": 0, "method": "top1"}

    if (
        len(sign_payloads) < 2
    ):
        return sign_payloads, {
            "selected_index": None,
            "confidence": "low",
            "identity_confirmed": False,
            "reason": "No LLM choice was made because fewer than two retrieved candidates were available.",
            "method": "retrieval_candidates_insufficient",
        }

    if not _has_multiple_candidate_identities(sign_payloads):
        return [sign_payloads[0], *sign_payloads[1:]], {
            "selected_index": 0,
            "confidence": "medium",
            "identity_confirmed": True,
            "reason": "Retrieved candidates agree on the same sign identity and attributes.",
            "method": "retrieval_candidate_consensus",
        }

    if not llm_client:
        return sign_payloads, {
            "selected_index": None,
            "confidence": "low",
            "identity_confirmed": False,
            "reason": "Candidate identities differ but no LLM client is available to resolve them.",
            "method": "retrieval_candidates_unresolved",
        }

    if trafficlawqa_rag_compatibility:
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
    else:
        candidates = [
            {
                "index": index,
                "sign_code": _sign_code(payload),
                "retrieval_rank": payload.get("retrieval_rank", index + 1),
                "retrieval_score": payload.get("score"),
                "sign_name": payload.get("sign_name", ""),
                "description": payload.get("sign_description", ""),
                "attributes": payload.get("attributes", {}),
                "normalized_conditions": payload.get("normalized_conditions", {}),
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
                                "RETRIEVED SIGN CANDIDATES (choose only by index)\n"
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
            "selected_index": 0 if trafficlawqa_rag_compatibility else None,
            "confidence": "low",
            "identity_confirmed": trafficlawqa_rag_compatibility,
            "reason": (
                f"Fallback to top-1 because variant disambiguation failed: {exc}"
                if trafficlawqa_rag_compatibility
                else f"Candidate selection abstained because the LLM call failed: {exc}"
            ),
            "method": "vlm_variant_disambiguation",
        }
    selected_index = decision["selected_index"]
    decision["identity_confirmed"] = (
        selected_index is not None and decision.get("confidence") == "high"
    )
    reordered = sign_payloads
    if selected_index is not None:
        reordered = [
            sign_payloads[selected_index],
            *sign_payloads[:selected_index],
            *sign_payloads[selected_index + 1 :],
        ]
    decision["method"] = (
        "vlm_variant_disambiguation"
        if trafficlawqa_rag_compatibility
        else "llm_top10_sign_selection"
    )
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
    allow_sign_retrieval: bool = True,
    trafficlawqa_rag_compatibility: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
    TrafficFacts,
]:
    article_candidates: list[dict[str, Any]] = []
    visual_evidence: list[dict[str, Any]] = []
    sign_prompt_parts: list[dict[str, Any]] = []
    traffic_sign_observations: list[SignObservation] = []
    ocr_engine = None
    ocr_unavailable = False

    for sign in sample.get("detected_signs", []):
        if not sign.get("is_chosen"):
            continue

        sign_image_path = sign_dir / sign["image_name"]
        sign_image = Image.open(sign_image_path).convert("RGB")

        sign_payloads = [
            dict(payload)
            for payload in sign.get("retrieved_payloads", [])
            if isinstance(payload, dict)
        ]
        if not allow_sign_retrieval:
            sign_payloads = []
        can_query_fallback = (
            allow_sign_retrieval
            and
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
            sign_payloads = []
            for rank, retrieved_result in enumerate(retrieved_results, start=1):
                payload = dict(retrieved_result.payload or {})
                payload["retrieval_rank"] = rank
                payload["source"] = "retrieval"
                score = getattr(retrieved_result, "score", None)
                if score is not None:
                    payload["score"] = score
                sign_payloads.append(payload)

        if sign_payloads:
            sign_payloads, variant_decision = disambiguate_sign_payloads(
                llm_client=llm_client,
                sample=sample,
                sign_image=sign_image,
                sign_payloads=sign_payloads,
                temperature=temperature,
                max_tokens=max_tokens,
                trafficlawqa_rag_compatibility=trafficlawqa_rag_compatibility,
            )
        elif isinstance(sign.get("variant_decision"), dict):
            # Direct-class detectors such as YOLO26s already resolve the class;
            # preserve their confidence and mapping status without re-ranking.
            variant_decision = dict(sign["variant_decision"])
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
        selection_confirmed = (
            variant_decision.get(
                "identity_confirmed",
                variant_decision.get("selected_index") is not None
                and variant_decision.get("confidence") in {"high", "medium"},
            ) is True
        )
        primary_payload = sign_payloads[0] if sign_payloads else {}
        sign["variant_decision"] = variant_decision
        if primary_payload:
            sign["image_lawdb_path"] = primary_payload.get("image_path", "")
            sign["sign_name"] = primary_payload.get("sign_name", "")

        sign_visual_evidence: list[dict[str, Any]] = []
        for rank, payload in enumerate(sign_payloads, start=1):
            if trafficlawqa_rag_compatibility or (selection_confirmed and rank == 1):
                article_candidate = article_candidate_from_payload(
                    payload,
                    retrieval_rank=payload.get("retrieval_rank", rank),
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
                "retrieval_rank": payload.get("retrieval_rank", rank),
                "retrieval_score": payload.get("score"),
                "is_primary": (trafficlawqa_rag_compatibility and rank == 1)
                or (selection_confirmed and rank == 1),
                "is_candidate": True,
            }
            sign_visual_evidence.append(evidence_item)
            visual_evidence.append(evidence_item)

        if not sign_payloads and (
            sign.get("sign_code")
            or sign.get("sign_family")
            or sign.get("class_label")
        ):
            direct_evidence = {
                "sign_code": sign.get("sign_code") or "",
                "sign_family": sign.get("sign_family") or "",
                "sign_name": sign.get("sign_name") or sign.get("class_label") or "",
                "description": sign.get("sign_description") or "",
                "legal_conditions": sign.get("normalized_conditions") or {},
                "image_name": str(sign.get("image_name") or ""),
                "retrieval_rank": None,
                "is_primary": bool(
                    sign.get("identity_status") == "identified"
                    and sign.get("sign_code")
                ),
                "is_candidate": False,
            }
            sign_visual_evidence.append(direct_evidence)
            visual_evidence.append(direct_evidence)

            direct_article = article_candidate_from_payload(
                sign,
                context_source="detector",
            )
            if direct_article and selection_confirmed and sign.get("sign_code"):
                article_candidates.append(direct_article)

        sign["retrieved_payloads"] = sign_payloads
        observation_source = str(sign.get("source") or "retrieval")
        sign_observation = normalize_detected_sign_to_observation(
            sign=sign,
            payloads=sign_payloads,
            visual_evidence=sign_visual_evidence,
            source=observation_source,
        )
        if (
            settings.traffic_sign_ocr_enabled
            and not ocr_unavailable
            and not sign_observation.ocr_texts
            and not sign.get("ocr_status")
        ):
            try:
                if ocr_engine is None:
                    ocr_engine = get_traffic_sign_ocr(
                        device=settings.traffic_sign_ocr_device,
                        language=settings.traffic_sign_ocr_language,
                    )
                sign_observation.ocr_texts = ocr_engine.recognize(sign_image_path)
                sign["ocr_status"] = "read" if sign_observation.ocr_texts else "no_text"
            except Exception as exc:
                ocr_unavailable = True
                sign["ocr_status"] = "unavailable"
                logging.getLogger(__name__).warning(
                    "Traffic-sign OCR unavailable for this sample: %s", exc
                )
        if sign_observation.ocr_texts:
            sign["ocr_texts"] = sign_observation.ocr_texts
            sign.setdefault("ocr_status", "read")
        sign["sign_observation"] = sign_observation.to_dict()
        traffic_sign_observations.append(sign_observation)
        sign_prompt_parts.append({
            "type": "text",
            "text": (
                f"DETECTED SIGN CROP: {sign.get('image_name')}\n"
                "VARIANT RESOLUTION: "
                f"{json.dumps(variant_decision, ensure_ascii=False)}"
            ),
        })

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
    traffic_facts = TrafficFacts(
        schema_version="2.0",
        scene_id=str(sample.get("image_id") or "scene-unknown"),
        sign_observations=traffic_sign_observations,
        context_facts=extract_nearby_highway_profile_facts(traffic_sign_observations),
        sign_relations=link_supplementary_signs(traffic_sign_observations),
        sign_inventory_status="partial",
    )
    return article_contexts, visual_evidence, sign_prompt_parts, traffic_facts


def build_freeform_prompt_parts(
    sample: dict[str, Any],
    image_dir: Path,
    article_contexts: list[dict[str, Any]],
    visual_evidence: list[dict[str, Any]],
    sign_prompt_parts: list[dict[str, Any]],
    symbolic_context: dict[str, Any] | None = None,
    trafficlawqa_rag_compatibility: bool = False,
    include_full_image_ocr: bool = False,
) -> list[dict[str, Any]]:
    symbolic_context = symbolic_context or {}
    choices = sample.get("choices")
    user_context = {
        "question": sample.get("question", ""),
        "choices_optional_reference": choices if choices else None,
    }
    if include_full_image_ocr:
        user_context.update({
            "full_image_ocr_status": sample.get("full_image_ocr_status", "unavailable"),
            "full_image_ocr": sample.get("full_image_ocr", []),
        })
    if not trafficlawqa_rag_compatibility:
        user_context.update({
            "query_intent": symbolic_context.get("query_intent", {}),
            "symbolic_decision": symbolic_context.get("symbolic_decision", {}),
            "rules_fired": symbolic_context.get("rules_fired", []),
        })

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
        if article.get("context_source") in {"retrieval", "detector"}
    ]
    related_articles = [
        article
        for article in article_contexts
        if article.get("context_source") not in {"retrieval", "detector"}
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

    question_block = (
        "USER QUESTION CONTEXT\n"
        f"{json.dumps(user_context, ensure_ascii=False)}\n\n"
    )
    if not trafficlawqa_rag_compatibility:
        question_block += (
            "SYMBOLIC REASONING CONTEXT\n"
            f"{json.dumps(symbolic_context, ensure_ascii=False)}\n\n"
        )
    question_block += (
        "LEGAL ARTICLE CONTEXT\n"
        f"{legal_context}\n\n"
        "SIGN-SPECIFIC EVIDENCE\n"
        f"{sign_specific_context}"
    )

    prompt_parts = [
        {
            "type": "text",
            "text": (
                SYSTEM_PROMPT_TRAFFICLAWQA_RAG
                if trafficlawqa_rag_compatibility
                else SYSTEM_PROMPT_FREEFORM_CITED_QA
            ),
        },
        {"type": "text", "text": question_block},
    ]
    if trafficlawqa_rag_compatibility:
        original_image = Image.open(image_dir / f"{sample['image_id']}.jpg").convert("RGB")
        image_b64, image_mime = pil_to_base64_png(original_image)
        prompt_parts.extend([
            {"type": "text", "text": "INPUT IMAGE"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
            },
        ])
    prompt_parts.extend(sign_prompt_parts)
    return prompt_parts


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
    allow_sign_retrieval: bool = True,
    trafficlawqa_rag_compatibility: bool = False,
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
            # The LLM only explains system-produced facts; it does not inspect
            # legal conclusions. It may select a sign identity from retrieved
            # candidates before the symbolic pipeline evaluates any rule.
            llm_client=llm_client,
            limit=(settings.limit if trafficlawqa_rag_compatibility else settings.sign_recognition_candidate_limit),
            temperature=temperature,
            max_tokens=min(max_tokens, 500),
            allow_sign_retrieval=allow_sign_retrieval,
            trafficlawqa_rag_compatibility=trafficlawqa_rag_compatibility,
        )
        symbolic_context = {}
        if settings.enable_symbolic_reasoning and not trafficlawqa_rag_compatibility:
            symbolic_context = build_symbolic_reasoning_context(
                traffic_facts=traffic_sign_observations,
                question=str(sample.get("question") or ""),
                llm_client=llm_client,
                use_llm=True,
                temperature=temperature,
                max_tokens=min(max_tokens, 300),
            )
        if trafficlawqa_rag_compatibility:
            symbolic_context = {}
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
            trafficlawqa_rag_compatibility=trafficlawqa_rag_compatibility,
            include_full_image_ocr=(
                settings.enable_symbolic_reasoning
                and not trafficlawqa_rag_compatibility
            ),
        )

        response_content: str = llm_client.generate_content(
            contents=[{"role": "user", "content": parts}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
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
            limit=settings.sign_recognition_candidate_limit,
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
