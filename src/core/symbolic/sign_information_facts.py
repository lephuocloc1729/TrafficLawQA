"""Structured OCR facts for highway and destination information signs."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.core.symbolic.fact_schema import ContextFact, SignObservation
from src.core.symbolic.sign_normalizer import normalize_sign_code
from src.core.symbolic.sign_schema import normalize_confidence


TIME_RANGE_RE = re.compile(
    r"(?<!\d)(\d{1,2})[:. ]([0-5]\d)\s*[-–—~]\s*"
    r"(\d{1,2})[:. ]([0-5]\d)(?!\d)"
)
DISTANCE_RE = re.compile(r"(?<!\d)(\d{1,5}(?:[.,]\d+)?)\s*(km|m)\b", re.I)
ROUTE_CODE_RE = re.compile(r"\bCT\s*[.\- ]?\s*0*(\d{1,3})\b", re.I)
MAIN_SIGN_NUMERIC_CODES = {"P.117", "P.118", "P.119", "P.120", "P.121", "P.127", "P.127a", "R.306"}
DESTINATION_RE = re.compile(
    r"(?P<destination>[^\d]{2,60}?)\s+(?P<distance>\d{1,4}(?:[.,]\d+)?)\s*km\b",
    re.I,
)


def _plain(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn").replace("đ", "d")


def _ocr_lines(observation: SignObservation) -> list[dict[str, Any]]:
    def position(item: dict[str, Any]) -> tuple[float, float]:
        polygon = item.get("bbox") or []
        points = [point for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not points:
            return 0.0, 0.0
        return min(float(point[1]) for point in points), min(float(point[0]) for point in points)

    return sorted(
        [item for item in observation.ocr_texts if str(item.get("raw_text") or item.get("normalized_value") or "").strip()],
        key=position,
    )


def _fact(
    observation: SignObservation,
    subject_id: str,
    predicate: str,
    value: Any,
    item: dict[str, Any],
    index: int,
    *,
    raw_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextFact:
    confidence = normalize_confidence(item.get("confidence") or observation.confidence)
    return ContextFact(
        fact_id=f"{observation.observation_id}:information:{predicate}:{index}",
        predicate=predicate,
        value=value,
        status="observed" if confidence in {"high", "medium"} else "ambiguous",
        source="traffic_sign_ocr",
        raw_text=raw_text or str(item.get("raw_text") or ""),
        confidence=confidence,
        evidence=[{"type": "ocr_text", "bbox": item.get("bbox")}],
        metadata={
            "subject_observation_id": subject_id,
            "source_observation_id": observation.observation_id,
            "extraction": "structured_traffic_sign_ocr",
            **(metadata or {}),
        },
    )


def extract_sign_information_facts(
    observation: SignObservation,
    subject_id: str | None = None,
) -> list[ContextFact]:
    """Parse explicit time, distance, route and destination data from OCR.

    These are observations for answering INFORMATION intents. They do not by
    themselves decide whether a prohibition is legally active.
    """
    subject_id = subject_id or observation.observation_id
    code = normalize_sign_code(observation.sign_code)
    lines = _ocr_lines(observation)
    result: list[ContextFact] = []
    combined = " ".join(
        str(item.get("normalized_value") or item.get("raw_text") or "").strip()
        for item in lines
    )
    plain_combined = _plain(combined)

    # Keep every displayed time interval, including stacked intervals on one panel.
    time_index = 0
    for line_index, item in enumerate(lines):
        raw = str(item.get("normalized_value") or item.get("raw_text") or "")
        normalized = raw.replace("：", ":").replace("−", "-")
        for match in TIME_RANGE_RE.finditer(normalized):
            start_hour, start_minute, end_hour, end_minute = map(int, match.groups())
            if start_hour > 23 or end_hour > 23:
                continue
            time_index += 1
            result.append(_fact(
                observation,
                subject_id,
                "supplementary_time_window",
                {
                    "start_minute": start_hour * 60 + start_minute,
                    "end_minute": end_hour * 60 + end_minute,
                    "raw_text": match.group(0),
                },
                item,
                time_index,
                raw_text=match.group(0),
                metadata={"ocr_line_index": line_index},
            ))

    is_expressway = (
        code == "I.437"
        or "expressway" in plain_combined
        or "cao toc" in plain_combined
        or "expressway" in _plain(str(observation.visual_attributes.get("name") or ""))
    )
    if is_expressway:
        result.append(_fact(
            observation, subject_id, "road_type", "expressway",
            lines[0] if lines else {}, 0,
            raw_text=combined,
            metadata={"basis": "recognized_expressway_sign_or_text"},
        ))
        route_match = ROUTE_CODE_RE.search(combined)
        if route_match:
            route_code = f"CT.{int(route_match.group(1)):02d}"
            result.append(_fact(
                observation, subject_id, "highway_route_code", route_code,
                next((item for item in lines if ROUTE_CODE_RE.search(str(item.get("raw_text") or ""))), lines[0] if lines else {}),
                1,
                raw_text=route_match.group(0),
            ))

        name_parts: list[str] = []
        for item in lines:
            raw = str(item.get("normalized_value") or item.get("raw_text") or "").strip()
            plain = _plain(raw)
            if not raw or "expressway" in plain or "cao toc" in plain or ROUTE_CODE_RE.search(raw):
                continue
            if DISTANCE_RE.fullmatch(raw.strip()) or re.fullmatch(r"\d{1,3}", raw.strip()):
                continue
            if re.search(r"[A-Za-zÀ-ỹ]{2,}", raw):
                name_parts.append(raw)
        highway_name = " ".join(name_parts).strip(" -–—")
        if highway_name:
            result.append(_fact(
                observation, subject_id, "highway_name", highway_name,
                lines[0] if lines else {}, 2,
                raw_text=highway_name,
            ))

    # A place-name followed by a kilometre value is a strong directional-sign
    # pattern (for example, "Hà Nội 20 km"). This also supports an unresolved
    # detector class when the OCR itself contains the complete pattern.
    match = DESTINATION_RE.search(combined)
    has_destination_pair = match is not None
    if match:
        destination = match.group("destination").strip(" >→-–—:;")
        if re.search(r"[A-Za-zÀ-ỹ]{2,}", destination):
            distance_km = float(match.group("distance").replace(",", "."))
            item = next((line for line in lines if "km" in str(line.get("raw_text") or "").lower()), lines[-1] if lines else {})
            result.extend([
                _fact(observation, subject_id, "destination_name", destination, item, 0, raw_text=match.group(0)),
                _fact(observation, subject_id, "destination_distance_km", distance_km, item, 1, raw_text=match.group(0)),
            ])

    # Generic below-sign distances are retained without guessing whether the
    # number means S.501 effect length or S.502 distance to an object.
    distance_index = 0
    main_bbox = observation.bbox or [0, 0, 0, 0]
    for item in lines:
        raw = str(item.get("normalized_value") or item.get("raw_text") or "")
        bbox = item.get("bbox") or []
        center_y = (
            sum(float(point[1]) for point in bbox if isinstance(point, (list, tuple)) and len(point) > 1) / len(bbox)
            if bbox else None
        )
        is_below = center_y is not None and center_y > main_bbox[3]
        # Some detectors include the attached panel inside the main sign box.
        # Preserve a standalone distance token in that case too.
        is_standalone_distance = (
            DISTANCE_RE.fullmatch(raw.strip()) is not None
            and code not in MAIN_SIGN_NUMERIC_CODES
        )
        if not is_below and not is_standalone_distance:
            continue
        if has_destination_pair and "km" in raw.lower():
            continue
        for match in DISTANCE_RE.finditer(raw):
            amount = float(match.group(1).replace(",", "."))
            unit = match.group(2).lower()
            distance_m = amount * 1000 if unit == "km" else amount
            distance_index += 1
            result.append(_fact(
                observation, subject_id, "supplementary_distance_m", distance_m,
                item, distance_index,
                raw_text=match.group(0),
                metadata={"distance_semantics": "unresolved_s501_or_s502"},
            ))

    return result


def extract_nearby_highway_profile_facts(
    observations: list[SignObservation],
) -> list[ContextFact]:
    """Group nearby YOLO speed signs with a detected expressway information sign."""
    highways = [
        item for item in observations
        if normalize_sign_code(item.sign_code) == "I.437" and item.bbox and len(item.bbox) == 4
    ]
    result: list[ContextFact] = []
    for highway in highways:
        x1, y1, x2, y2 = highway.bbox or [0, 0, 0, 0]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        region = (x1 - width * 0.1, y1 - height * 0.1, x2 + width * 0.1, y2 + height * 2.0)
        maximums: set[int] = set()
        minimums: set[int] = set()
        for candidate in observations:
            if candidate is highway or not candidate.bbox or len(candidate.bbox) != 4:
                continue
            code = normalize_sign_code(candidate.sign_code)
            if code not in {"P.127", "R.306"}:
                continue
            cx1, cy1, cx2, cy2 = candidate.bbox
            center_x, center_y = (cx1 + cx2) / 2, (cy1 + cy2) / 2
            if not (region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3]):
                continue
            conditions = candidate.visual_attributes.get("normalized_conditions", {})
            if not isinstance(conditions, dict):
                continue
            try:
                if code == "P.127" and conditions.get("speed_limit_value") is not None:
                    maximums.add(int(conditions["speed_limit_value"]))
                if code == "R.306" and conditions.get("minimum_speed_limit_kmh") is not None:
                    minimums.add(int(conditions["minimum_speed_limit_kmh"]))
            except (TypeError, ValueError):
                continue

        if not maximums and not minimums:
            continue
        parsed_facts = extract_sign_information_facts(highway, subject_id=highway.observation_id)
        route_code = next((fact.value for fact in parsed_facts if fact.predicate == "highway_route_code"), None)
        highway_name = next((fact.value for fact in parsed_facts if fact.predicate == "highway_name"), None)
        conflicting = len(maximums) > 1 or len(minimums) > 1
        value = {
            "route_code": route_code,
            "highway_name": highway_name,
            "maximum_speed_kmh": next(iter(maximums)) if len(maximums) == 1 else None,
            "minimum_speed_kmh": next(iter(minimums)) if len(minimums) == 1 else None,
        }
        result.append(ContextFact(
            fact_id=f"{highway.observation_id}:highway-profile",
            predicate="highway_profile",
            value=None if conflicting else value,
            status="ambiguous" if conflicting else "derived",
            source="nearby_yolo_sign_association",
            confidence="low" if conflicting else "medium",
            evidence=[{"type": "bbox", "ref": highway.observation_id}],
            metadata={
                "subject_observation_id": highway.observation_id,
                "speed_signs_associated_by_geometry": True,
                "conflicting_maximum_speeds_kmh": sorted(maximums) if len(maximums) > 1 else [],
                "conflicting_minimum_speeds_kmh": sorted(minimums) if len(minimums) > 1 else [],
            },
        ))
    return result
