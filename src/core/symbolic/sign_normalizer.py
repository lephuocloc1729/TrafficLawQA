from __future__ import annotations

import re
from typing import Any

from src.core.symbolic.sign_schema import Confidence, TrafficSignFact, normalize_confidence


SIGN_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}(?:\.[A-Z])?[,\.]\d+[a-z]?)\b",
    re.IGNORECASE,
)
VARIANT_SUFFIX_RE = re.compile(r"([,.]\d+)[a-z]$", re.IGNORECASE)


SIGN_RULE_SCOPE: dict[str, dict[str, Any]] = {
    "P.130": {
        "restriction_type": "no_stopping_no_parking",
        "target_vehicle": "motor_vehicle",
    },
    "P.131a": {
        "restriction_type": "no_parking",
        "target_vehicle": "motor_vehicle",
    },
    "P.131b": {
        "restriction_type": "no_parking",
        "target_vehicle": "motor_vehicle",
        "time_condition": "odd_days",
    },
    "P.131c": {
        "restriction_type": "no_parking",
        "target_vehicle": "motor_vehicle",
        "time_condition": "even_days",
    },
    "P.127": {
        "restriction_type": "speed_limit",
        "target_vehicle": "motor_vehicle",
    },
    "P.123a": {
        "restriction_type": "no_left_turn",
        "target_vehicle": "motor_vehicle",
    },
    "P.123b": {
        "restriction_type": "no_right_turn",
        "target_vehicle": "motor_vehicle",
    },
    "P.102": {
        "restriction_type": "no_entry",
        "target_vehicle": "vehicle",
    },
}


def normalize_sign_code(sign_code: str | None) -> str:
    raw = str(sign_code or "").strip().replace(",", ".")
    if not raw:
        return ""

    match = re.fullmatch(
        r"([A-Za-z]{1,3}(?:\.[A-Za-z])?)\.(\d+)([A-Za-z]?)",
        raw,
    )
    if not match:
        return raw

    prefix, number, suffix = match.groups()
    return f"{prefix.upper()}.{number}{suffix.lower()}"


def extract_sign_code(*texts: Any) -> str:
    for text in texts:
        if text is None:
            continue
        match = SIGN_CODE_RE.search(str(text))
        if match:
            return normalize_sign_code(match.group(1))
    return ""


def sign_family(sign_code: str | None) -> str:
    normalized = normalize_sign_code(sign_code)
    if not normalized:
        return ""
    return VARIANT_SUFFIX_RE.sub(r"\1", normalized)


def _article_ref(payload: dict[str, Any]) -> dict[str, str] | None:
    law_id = str(payload.get("law_id") or "").strip()
    article_id = str(payload.get("article_id") or "").strip()
    if not law_id or not article_id:
        return None

    article = {"law_id": law_id, "article_id": article_id}
    title = str(payload.get("title") or payload.get("article_title") or "").strip()
    if title:
        article["title"] = title
    return article


def _payload_texts(payload: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(payload.get("sign_code") or ""),
        str(payload.get("sign_name") or ""),
        str(payload.get("sign_description") or ""),
        str(payload.get("name") or ""),
        str(payload.get("description") or ""),
    )


def _metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keep_keys = (
        "image_path",
        "retrieval_rank",
        "score",
        "confidence",
        "sign_description",
    )
    return {
        key: payload[key]
        for key in keep_keys
        if key in payload and payload[key] not in (None, "")
    }


def normalize_payload_to_fact(
    payload: dict[str, Any],
    visual_evidence: list[dict[str, Any]] | None = None,
    confidence: str | None = None,
    source: str = "retrieval",
) -> TrafficSignFact:
    sign_code = extract_sign_code(*_payload_texts(payload))
    family = sign_family(sign_code)
    rule_scope = SIGN_RULE_SCOPE.get(sign_code, {})
    article_ref = _article_ref(payload)
    resolved_confidence: Confidence = normalize_confidence(
        confidence or payload.get("confidence") or ("medium" if rule_scope else "low")
    )

    return TrafficSignFact(
        sign_code=sign_code,
        sign_family=family or sign_code,
        sign_name=str(payload.get("sign_name") or payload.get("name") or "").strip(),
        restriction_type=str(rule_scope.get("restriction_type") or "unknown"),
        target_vehicle=str(rule_scope.get("target_vehicle") or "unknown"),
        time_condition=rule_scope.get("time_condition"),
        direction_scope=rule_scope.get("direction_scope"),
        distance_scope=rule_scope.get("distance_scope"),
        related_articles=[article_ref] if article_ref else [],
        visual_evidence=visual_evidence or [],
        confidence=resolved_confidence,
        source=source,
        metadata=_metadata_from_payload(payload),
    )


def normalize_detected_sign_to_fact(
    sign: dict[str, Any],
    payloads: list[dict[str, Any]] | None = None,
    visual_evidence: list[dict[str, Any]] | None = None,
    source: str = "retrieval",
) -> TrafficSignFact:
    primary_payload = dict(payloads[0]) if payloads else {}
    fallback_payload = {**sign, **primary_payload}
    fact = normalize_payload_to_fact(
        fallback_payload,
        visual_evidence=visual_evidence,
        confidence=(sign.get("confidence") or primary_payload.get("confidence")),
        source=source,
    )
    if not fact.sign_name:
        fact.sign_name = str(sign.get("sign_name") or "").strip()
    return fact


def normalize_payloads_to_facts(
    payloads: list[dict[str, Any]],
    visual_evidence_by_rank: list[dict[str, Any]] | None = None,
    source: str = "retrieval",
) -> list[TrafficSignFact]:
    facts: list[TrafficSignFact] = []
    for index, payload in enumerate(payloads):
        evidence = []
        if visual_evidence_by_rank and index < len(visual_evidence_by_rank):
            evidence = [visual_evidence_by_rank[index]]
        fact = normalize_payload_to_fact(
            payload,
            visual_evidence=evidence,
            source=source,
        )
        facts.append(fact)
    return facts
