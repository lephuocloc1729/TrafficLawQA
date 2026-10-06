"""Strict extraction for numeric values printed on main traffic signs."""

from __future__ import annotations

import re
from typing import Any

from src.core.symbolic.fact_schema import ContextFact, SignObservation
from src.core.symbolic.sign_normalizer import normalize_sign_code
from src.core.symbolic.sign_schema import normalize_confidence


_NUMERIC_SIGN_SPECS = {
    "P.117": {
        "predicate": "height_limit_m",
        "unit": "m",
        "pattern": re.compile(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met(?:er|re)s?|mét)?\s*", re.IGNORECASE),
        "numeric_kind": "float",
    },
    "P.118": {
        "predicate": "width_limit_m",
        "unit": "m",
        "pattern": re.compile(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met(?:er|re)s?|mét)?\s*", re.IGNORECASE),
        "numeric_kind": "float",
    },
    "P.119": {
        "predicate": "length_limit_m",
        "unit": "m",
        "pattern": re.compile(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met(?:er|re)s?|mét)?\s*", re.IGNORECASE),
        "numeric_kind": "float",
    },
    "P.120": {
        "predicate": "combination_length_limit_m",
        "unit": "m",
        "pattern": re.compile(r"\s*(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met(?:er|re)s?|mét)?\s*", re.IGNORECASE),
        "numeric_kind": "float",
    },
    "P.121": {
        "predicate": "minimum_following_distance_m",
        "unit": "m",
        "pattern": re.compile(r"\s*(\d{1,4})(?:\s*m)?\s*", re.IGNORECASE),
    },
    "R.306": {
        "predicate": "minimum_speed_limit_kmh",
        "unit": "km/h",
        "pattern": re.compile(r"\s*(\d{1,3})(?:\s*(?:km\s*/\s*h|kmh|kph))?\s*", re.IGNORECASE),
    },
    "R.E.9d": {
        "predicate": "speed_limit_value",
        "unit": "km/h",
        "pattern": re.compile(r"\s*(\d{1,3})(?:\s*(?:km\s*/\s*h|kmh|kph))?\s*", re.IGNORECASE),
    },
    "R.E.9b": {
        "predicate": "time_window",
        "unit": None,
        "value_kind": "time_window",
        "pattern": re.compile(r"\s*(\d{1,2}):([0-5]\d)\s*[-–—]\s*(\d{1,2}):([0-5]\d)\s*"),
    },
}


def extract_main_sign_context_facts(
    observation: SignObservation,
    subject_id: str,
) -> list[ContextFact]:
    """Extract strict numeric values only for sign codes whose units define them."""
    code = normalize_sign_code(observation.sign_code)
    spec = _NUMERIC_SIGN_SPECS.get(code or "")
    if spec is None:
        return []

    readings: list[tuple[Any, str, str, dict[str, Any]]] = []
    for index, item in enumerate(observation.ocr_texts):
        raw = str(item.get("normalized_value") or item.get("raw_text") or "").strip()
        match = spec["pattern"].fullmatch(raw)
        if not match:
            continue
        if spec.get("value_kind") == "time_window":
            start_hour, start_minute, end_hour, end_minute = map(int, match.groups())
            if start_hour > 23 or end_hour > 23:
                continue
            value: Any = {
                "start_minute": start_hour * 60 + start_minute,
                "end_minute": end_hour * 60 + end_minute,
                "raw_text": raw,
            }
        else:
            raw_number = match.group(1).replace(",", ".")
            value = float(raw_number) if spec.get("numeric_kind") == "float" else int(raw_number)
        readings.append((value, raw, normalize_confidence(item.get("confidence")), item))

    if not readings:
        return []
    def fingerprint(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("start_minute"), value.get("end_minute")
        return value

    values = {fingerprint(reading[0]) for reading in readings}
    if len(values) > 1:
        conflicting_values = sorted(values) if all(isinstance(value, (int, float)) for value in values) else sorted(values, key=str)
        return [ContextFact(
            fact_id=f"{observation.observation_id}:ocr:{spec['predicate']}:conflict",
            predicate=spec["predicate"],
            value=None,
            status="ambiguous",
            source="main_sign_ocr",
            raw_text=" | ".join(reading[1] for reading in readings),
            confidence="low",
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "extraction": f"strict_{code.lower().replace('.', '')}_numeric_ocr",
                "conflicting_values": conflicting_values,
                "unit": spec["unit"],
            },
        )]

    value, raw, confidence, source_item = max(
        readings,
        key=lambda reading: {"high": 2, "medium": 1, "low": 0}.get(reading[2], 0),
    )
    evidence = list(observation.evidence)
    if isinstance(source_item.get("evidence"), dict):
        evidence.append(source_item["evidence"])
    status = "observed" if confidence in {"high", "medium"} else "ambiguous"
    return [ContextFact(
        fact_id=f"{observation.observation_id}:ocr:{spec['predicate']}",
        predicate=spec["predicate"],
        value=value,
        status=status,
        source="main_sign_ocr",
        raw_text=raw,
        confidence=confidence,
        evidence=evidence,
        metadata={
            "subject_observation_id": subject_id,
            "source_observation_id": observation.observation_id,
            "extraction": f"strict_{code.lower().replace('.', '')}_numeric_ocr",
            "unit": spec["unit"],
        },
    )]
