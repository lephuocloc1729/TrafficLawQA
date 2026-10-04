from __future__ import annotations

import re
from typing import Any

from src.core.symbolic.fact_schema import SignObservation
from src.core.symbolic.sign_schema import normalize_confidence


SIGN_CODE_RE = re.compile(r"\b([A-Z]{1,3}(?:\.[A-Z])?[,\.]?\d+[a-z]?)\b", re.IGNORECASE)
VARIANT_SUFFIX_RE = re.compile(r"([,.]?\d+)[a-z]$", re.IGNORECASE)


def normalize_sign_code(sign_code: str | None) -> str:
    raw = str(sign_code or "").strip().replace(",", ".")
    if not raw:
        return ""
    match = re.fullmatch(r"([A-Za-z]{1,3}(?:\.[A-Za-z])?)[\.]?(\d+)([A-Za-z]?)", raw)
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
    return VARIANT_SUFFIX_RE.sub(r"\1", normalized) if normalized else ""


def _article_refs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    candidates = payload.get("related_articles", [])
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            law_id = str(candidate.get("law_id") or "").strip()
            article_id = str(candidate.get("article_id") or "").strip()
            if law_id and article_id:
                refs.append({"law_id": law_id, "article_id": article_id,
                             "title": str(candidate.get("title") or candidate.get("article_title") or "")})
    law_id = str(payload.get("law_id") or "").strip()
    article_id = str(payload.get("article_id") or "").strip()
    if law_id and article_id:
        ref = {"law_id": law_id, "article_id": article_id,
               "title": str(payload.get("title") or payload.get("article_title") or "")}
        if ref not in refs:
            refs.append(ref)
    return refs


def _payload_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("candidates", [])
    candidates: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            code = extract_sign_code(item.get("sign_code"), item.get("sign_name"), item.get("name"))
            if code:
                candidates.append({
                    "sign_code": code,
                    "confidence": item.get("confidence", item.get("score")),
                    "source": item.get("source", "retrieval"),
                })
    return candidates


def _observation_id(payload: dict[str, Any], sign_code: str) -> str:
    for key in ("observation_id", "image_name", "image_path", "image_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value.replace("\\", "/")
    return f"sign-observation:{sign_code or 'unidentified'}"


def normalize_payload_to_observation(
    payload: dict[str, Any],
    visual_evidence: list[dict[str, Any]] | None = None,
    confidence: str | None = None,
    source: str | None = None,
) -> SignObservation:
    """Normalize retrieval/detection data into an observation, without inferring legal scope."""
    sign_code = extract_sign_code(
        payload.get("sign_code"), payload.get("sign_name"), payload.get("name"),
        payload.get("sign_description"), payload.get("description"),
    )
    candidates = _payload_candidates(payload)
    if sign_code and not any(item["sign_code"] == sign_code for item in candidates):
        candidates.insert(0, {"sign_code": sign_code, "confidence": confidence, "source": source or "retrieval"})

    identity_status = str(payload.get("identity_status") or "").strip().lower()
    if identity_status not in {"identified", "ambiguous", "unidentified"}:
        identity_status = "identified" if sign_code else "ambiguous" if candidates else "unidentified"

    bbox = payload.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = None
    else:
        try:
            bbox = [float(value) for value in bbox]
        except (TypeError, ValueError):
            bbox = None

    name = str(payload.get("sign_name") or payload.get("name") or "").strip()
    description = str(payload.get("sign_description") or payload.get("description") or "").strip()
    ocr = payload.get("ocr_texts", payload.get("ocr_text"))
    if isinstance(ocr, str):
        ocr_texts = [{"raw_text": ocr, "source": "ocr"}]
    elif isinstance(ocr, list):
        ocr_texts = [item if isinstance(item, dict) else {"raw_text": str(item), "source": "ocr"}
                     for item in ocr]
    else:
        ocr_texts = []

    evidence = [dict(item) for item in visual_evidence or [] if isinstance(item, dict)]
    image_path = str(payload.get("image_path") or payload.get("image_name") or "").strip()
    if image_path and not any(item.get("path") == image_path for item in evidence):
        evidence.append({"type": "crop", "path": image_path, "source": source or "retrieval"})

    return SignObservation(
        observation_id=_observation_id(payload, sign_code),
        role=str(payload.get("role") or payload.get("sign_role") or "unknown"),
        bbox=bbox,
        sign_code=sign_code or None,
        identity_status=identity_status,
        candidates=candidates,
        visual_attributes={"name": name or None, "description": description or None},
        ocr_texts=ocr_texts,
        legal_refs=_article_refs(payload),
        evidence=evidence,
        confidence=normalize_confidence(confidence or payload.get("confidence")),
        source=str(source or payload.get("source") or "retrieval"),
        metadata={
            "image_path": image_path or None,
            "retrieval_rank": payload.get("retrieval_rank"),
            "retrieval_score": payload.get("score"),
            "legal_conditions_unverified": payload.get("legal_conditions"),
            "variant_decision": payload.get("variant_decision"),
            "candidate_count": len(candidates),
        },
    )


def normalize_detected_sign_to_observation(
    sign: dict[str, Any],
    payloads: list[dict[str, Any]] | None = None,
    visual_evidence: list[dict[str, Any]] | None = None,
    source: str = "retrieval",
) -> SignObservation:
    candidates = [dict(payload) for payload in (payloads or []) if isinstance(payload, dict)]
    primary = next((p for p in candidates if p.get("is_primary")), candidates[0] if candidates else {})
    merged = {**sign, **primary}
    if candidates:
        merged["candidates"] = candidates
    observation = normalize_payload_to_observation(
        merged,
        visual_evidence=visual_evidence,
        confidence=(
            sign.get("variant_decision", {}).get("confidence")
            if isinstance(sign.get("variant_decision"), dict)
            else None
        ),
        source=source,
    )
    variant_decision = sign.get("variant_decision")
    if observation.sign_code:
        if isinstance(variant_decision, dict) and variant_decision.get("confidence") in {"high", "medium"}:
            observation.identity_status = "identified"
        else:
            observation.identity_status = "ambiguous"
            observation.metadata["identity_resolution"] = "retrieval_top1_not_visually_confirmed"
    if sign.get("confidence") is not None:
        observation.metadata["detector_confidence"] = sign.get("confidence")
    if not observation.visual_attributes.get("name") and sign.get("sign_name"):
        observation.visual_attributes["name"] = str(sign["sign_name"])
    if not observation.sign_code and sign.get("sign_code"):
        observation.sign_code = normalize_sign_code(str(sign["sign_code"])) or None
        observation.identity_status = "identified" if observation.sign_code else "unidentified"
    return observation


def normalize_payloads_to_observations(
    payloads: list[dict[str, Any]],
    visual_evidence_by_rank: list[dict[str, Any]] | None = None,
    source: str = "retrieval",
) -> list[SignObservation]:
    observations = []
    for index, payload in enumerate(payloads):
        ranked = dict(payload)
        ranked.setdefault("retrieval_rank", index + 1)
        evidence = [visual_evidence_by_rank[index]] if visual_evidence_by_rank and index < len(visual_evidence_by_rank) else []
        observations.append(normalize_payload_to_observation(ranked, evidence, source=source))
    return observations
