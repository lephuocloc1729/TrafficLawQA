from __future__ import annotations

import re
import unicodedata as ud
from typing import Any

from src.core.symbolic.sign_schema import Confidence, TrafficSignFact, normalize_confidence


SIGN_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}(?:\.[A-Z])?[,\.]?\d+[a-z]?)\b",
    re.IGNORECASE,
)
VARIANT_SUFFIX_RE = re.compile(r"([,.]?\d+)[a-z]$", re.IGNORECASE)


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
        r"([A-Za-z]{1,3}(?:\.[A-Za-z])?)[\.]?(\d+)([A-Za-z]?)",
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


def _article_refs(payload: dict[str, Any]) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    raw_refs = payload.get("related_articles")
    if isinstance(raw_refs, list):
        for raw_ref in raw_refs:
            if isinstance(raw_ref, dict):
                article = _article_ref(raw_ref)
                if article and article not in refs:
                    refs.append(article)
    article = _article_ref(payload)
    if article and article not in refs:
        refs.append(article)
    return refs


def _normalize_time_condition(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None
    text = "".join(
        char
        for char in ud.normalize("NFD", normalized.replace("đ", "d"))
        if ud.category(char) != "Mn"
    )
    if "odd" in text or "ngay le" in text:
        return "odd_days"
    if "even" in text or "ngay chan" in text:
        return "even_days"
    return normalized


def _normalize_vehicle_scope(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = raw.lower().replace("đ", "d")
    if not normalized:
        return "unknown"
    aliases = (
        ("motor_vehicle", ("xe co gioi", "phuong tien co gioi", "motor vehicle")),
        ("motorbike", ("xe may", "mo to", "motorbike")),
        ("truck", ("xe tai", "truck", "container")),
        ("bus", ("xe khach", "xe buyt", "bus")),
        ("bicycle", ("xe dap", "bicycle")),
        ("pedestrian", ("nguoi di bo", "pedestrian")),
        ("car", ("o to", "xe con", "xe hoi", "car")),
        ("vehicle", ("tat ca phuong tien", "phuong tien", "vehicle")),
    )
    folded = "".join(
        char
        for char in ud.normalize("NFD", normalized)
        if ud.category(char) != "Mn"
    )
    for canonical, phrases in aliases:
        if folded == canonical or any(phrase in folded for phrase in phrases):
            return canonical
    return normalized


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
    source: str | None = None,
) -> TrafficSignFact:
    sign_code = extract_sign_code(*_payload_texts(payload))
    family = sign_family(sign_code)
    rule_scope = SIGN_RULE_SCOPE.get(sign_code, {})
    legal_conditions = payload.get("legal_conditions")
    if not isinstance(legal_conditions, dict):
        legal_conditions = {}
    resolved_confidence: Confidence = normalize_confidence(
        confidence
        or payload.get("confidence")
        or ("medium" if rule_scope else "low")
    )

    return TrafficSignFact(
        sign_code=sign_code,
        sign_family=family or normalize_sign_code(payload.get("sign_family")),
        sign_name=str(payload.get("sign_name") or payload.get("name") or "").strip(),
        restriction_type=str(
            payload.get("restriction_type") or rule_scope.get("restriction_type") or "unknown"
        ),
        target_vehicle=str(
            _normalize_vehicle_scope(
                payload.get("target_vehicle")
                or legal_conditions.get("vehicle_scope")
                or rule_scope.get("target_vehicle")
            )
        ),
        time_condition=_normalize_time_condition(
            payload.get("time_condition")
            or legal_conditions.get("time_condition")
            or rule_scope.get("time_condition")
        ),
        direction_scope=(
            payload.get("direction_scope")
            or legal_conditions.get("direction_scope")
            or rule_scope.get("direction_scope")
        ),
        distance_scope=payload.get("distance_scope") or rule_scope.get("distance_scope"),
        related_articles=_article_refs(payload),
        visual_evidence=visual_evidence or [],
        confidence=resolved_confidence,
        source=str(source or payload.get("source") or "retrieval"),
        metadata={
            **_metadata_from_payload(payload),
            **(
                {"candidate_count": len(payload.get("candidates", []))}
                if isinstance(payload.get("candidates"), list)
                else {}
            ),
        },
    )


def normalize_detected_sign_to_fact(
    sign: dict[str, Any],
    payloads: list[dict[str, Any]] | None = None,
    visual_evidence: list[dict[str, Any]] | None = None,
    source: str = "retrieval",
) -> TrafficSignFact:
    candidates = [dict(payload) for payload in (payloads or [])]
    primary_payload = next(
        (payload for payload in candidates if payload.get("is_primary")),
        candidates[0] if candidates else {},
    )
    fallback_payload = {**sign, **primary_payload}
    fact = normalize_payload_to_fact(
        fallback_payload,
        visual_evidence=visual_evidence,
        confidence=(sign.get("confidence") or primary_payload.get("confidence")),
        source=source,
    )
    if candidates:
        fact.metadata["retrieval_candidate_count"] = len(candidates)
        fact.metadata["selected_candidate_rank"] = primary_payload.get(
            "retrieval_rank", 1
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
        ranked_payload = dict(payload)
        ranked_payload.setdefault("retrieval_rank", index + 1)
        fact = normalize_payload_to_fact(
            ranked_payload,
            visual_evidence=evidence,
            source=source,
        )
        fact.metadata["retrieval_candidate_count"] = len(payloads)
        facts.append(fact)
    return facts
