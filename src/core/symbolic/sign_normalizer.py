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
                candidate = {
                    "sign_code": code,
                    "confidence": item.get("confidence", item.get("score")),
                    "source": item.get("source", "retrieval"),
                    "retrieval_rank": item.get("retrieval_rank", item.get("rank")),
                    "retrieval_score": item.get("score", item.get("retrieval_score")),
                    "attributes": item.get("attributes", {}),
                    "normalized_conditions": item.get("normalized_conditions", {}),
                }
                candidates.append(candidate)
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
    visual_attributes = payload.get("visual_attributes")
    visual_attributes = dict(visual_attributes) if isinstance(visual_attributes, dict) else {}
    top_level_conditions = payload.get("normalized_conditions")
    if isinstance(top_level_conditions, dict):
        visual_attributes.setdefault("normalized_conditions", dict(top_level_conditions))
    class_attributes = payload.get("attributes")
    if isinstance(class_attributes, dict):
        visual_attributes["class_attributes"] = dict(class_attributes)
        conditions = dict(visual_attributes.get("normalized_conditions") or {})
        attribute_to_condition = {
            "speed_limit_kmh": "speed_limit_value",
            "min_speed_kmh": "minimum_speed_limit_kmh",
            "slope_percent": "slope_percent",
        }
        for attribute, condition in attribute_to_condition.items():
            if attribute in class_attributes:
                conditions.setdefault(condition, class_attributes[attribute])

        # P.131b/P.131c encode the days on which the sign applies. Keep this
        # separate from calendar_parity, which describes the date being asked
        # about and is used by the symbolic rules to evaluate applicability.
        if (
            normalize_sign_code(sign_code) in {"P.131b", "P.131c"}
            and class_attributes.get("time_condition") in {"odd_days", "even_days"}
        ):
            conditions.setdefault(
                "sign_applicability_calendar_parity", class_attributes["time_condition"]
            )
        if conditions:
            visual_attributes["normalized_conditions"] = conditions
    visual_attributes.setdefault("name", name or None)
    visual_attributes.setdefault("description", description or None)
    visual_attributes.setdefault("sign_family", payload.get("sign_family") or None)
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

    catalog_info: dict[str, Any] = {
        "official_name": None,
        "legal_meaning": None,
        "status": "unresolved",
    }
    if sign_code:
        try:
            from src.core.symbolic.sign_catalog import load_default_sign_catalog

            entry = load_default_sign_catalog().lookup(sign_code)
            if entry is None:
                catalog_info["status"] = "not_found"
            else:
                catalog_info = {
                    "official_name": entry.official_name,
                    "legal_meaning": entry.legal_meaning,
                    "status": entry.status,
                }
        except (OSError, ValueError):
            catalog_info["status"] = "unavailable"

    return SignObservation(
        observation_id=_observation_id(payload, sign_code),
        role=str(payload.get("role") or payload.get("sign_role") or "unknown"),
        bbox=bbox,
        sign_code=sign_code or None,
        identity_status=identity_status,
        candidates=candidates,
        visual_attributes=visual_attributes,
        ocr_texts=ocr_texts,
        legal_refs=_article_refs(payload),
        evidence=evidence,
        confidence=normalize_confidence(confidence or payload.get("confidence")),
        source=str(source or payload.get("source") or "retrieval"),
        catalog=catalog_info,
        metadata={
            "image_path": image_path or None,
            "retrieval_rank": payload.get("retrieval_rank"),
            "retrieval_score": payload.get("score"),
            "legal_conditions_unverified": payload.get("legal_conditions"),
            "variant_decision": payload.get("variant_decision"),
            "candidate_count": len(candidates),
            "sign_family": payload.get("sign_family") or sign_family(sign_code),
            "qcvn_mapping_status": payload.get("qcvn_mapping_status"),
            "qcvn_section": payload.get("qcvn_section"),
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
    # Keep geometry and observation identity from the query detector; retrieval
    # payloads describe catalog entries and may carry unrelated source-image data.
    for key in ("bbox", "image_name", "observation_id", "role", "sign_role"):
        if sign.get(key) is not None:
            merged[key] = sign[key]
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
    if observation.role == "unknown" and observation.sign_code:
        normalized_code = normalize_sign_code(observation.sign_code) or ""
        observation.role = (
            "supplementary"
            if normalized_code.startswith("S.") and normalized_code != "S.507"
            else "main"
        )
    variant_decision = sign.get("variant_decision")
    if observation.sign_code:
        llm_candidate_selection = (
            isinstance(variant_decision, dict)
            and variant_decision.get("method") == "llm_top10_sign_selection"
        )
        if (
            isinstance(variant_decision, dict)
            and (
                variant_decision.get("identity_confirmed") is True
                if llm_candidate_selection
                else variant_decision.get("confidence") in {"high", "medium"}
            )
        ):
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
