"""Conservative, evidence-linked extraction of visible supplementary values.

These are observations, not legal rules. A downstream rule still needs reviewed
official authority before it can affect a permission/limit decision.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.core.symbolic.fact_schema import ContextFact, SignObservation
from src.core.symbolic.sign_normalizer import normalize_sign_code, sign_family
from src.core.symbolic.sign_schema import normalize_confidence

TIME_RANGE_RE = re.compile(
    r"(?<!\d)(\d{1,2}):([0-5]\d)\s*[-–—]\s*(\d{1,2}):([0-5]\d)(?!\d)"
)
DISTANCE_RE = re.compile(r"(?<!\d)(\d{1,6})\s*(?:m|met(?:er|re)s?|mét|met)(?!\w)", re.I)
HEIGHT_RE = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:m|met(?:er|re)s?|mét|met)(?!\w)", re.I)
AXLE_LIMIT_RE = re.compile(
    r"\s*(single\s+axle|truc\s+don|tandem\s+axle|truc\s+kep|triple\s+axle|truc\s+ba)"
    r"\s*[:=]?\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:t|tan|tonnes?)\s*",
    re.I,
)
GROSS_WEIGHT_LIMIT_RE = re.compile(
    r"\s*(\d{1,3}(?:[.,]\d{1,2})?)\s*(?:t|tan|tonnes?)\s*",
    re.I,
)
AXLE_PREDICATES = {
    "single axle": "single_axle_load_limit_t",
    "truc don": "single_axle_load_limit_t",
    "tandem axle": "tandem_axle_load_limit_t",
    "truc kep": "tandem_axle_load_limit_t",
    "triple axle": "triple_axle_load_limit_t",
    "truc ba": "triple_axle_load_limit_t",
}

# These are the arrows as depicted on the sign face in QCVN Figures F.3a/F.3b.
# They are deliberately not converted to left/right relative to the road; that
# requires the panel's physical orientation and its link to the main sign.
S503_VISUAL_SEMANTICS = {
    "S.503a": {"depicted_arrow": "right", "effect_axis": "perpendicular_to_road"},
    "S.503b": {"depicted_arrow": "both_horizontal_directions", "effect_axis": "perpendicular_to_road"},
    "S.503c": {"depicted_arrow": "left", "effect_axis": "perpendicular_to_road"},
    "S.503d": {"depicted_arrow": "down", "effect_axis": "parallel_to_road"},
    "S.503e": {"depicted_arrow": "both_vertical_directions", "effect_axis": "parallel_to_road"},
    "S.503f": {"depicted_arrow": "up", "effect_axis": "parallel_to_road"},
}
SCOPE_DIMENSION_BY_CODE = {
    "S.504": "lane",
    "S.505a": "vehicle_class",
    "S.506a": "priority_road_direction_for_I.401",
    "S.506b": "priority_road_direction_for_W.208_or_R.122",
}
S509B_SEMANTICS = {
    "action": "parking",
    "meaning": "no_parking",
    "permitted_primary_sign_codes": ["P.130", "P.131a", "P.131b", "P.131c"],
}
S507_STANDALONE_GUIDANCE = {
    "hazard": "dangerous_turn_ahead",
    "use": "standalone",
    "indicated_turn_direction": None,
    "direction_requires_visual_resolution": True,
    "replaces_W201_or_W202_warning": False,
}
SG_LANE_GUIDANCE = {
    "S.G.11": {
        "kind": "lane_count_and_direction",
        "arrow_count_equals_lanes_per_direction": True,
        "upcoming_lane_direction_mapping": None,
        "requires_visual_mapping": True,
    },
    "S.G.12": {
        "kind": "lane_not_in_operation",
        "affected_lane": None,
        "requires_visual_lane_resolution": True,
    },
}
SH6_EXCEPTION_SEMANTICS = {
    "applies_to": "prohibition_or_restriction_sign",
    "effect": "exception_for_explicitly_displayed_road_user_group",
    "exception_group": None,
    "group_requires_visual_or_text_resolution": True,
    "blanket_permission": False,
}
S510_WARNING_SEMANTICS = {
    "S.510a": {
        "hazard": "slippery_road_with_snow_or_ice",
        "official_context": "days_when_snowing",
        "scene_weather_inferred": False,
    },
    "S.510b": {
        "hazard": "railway_crossing_on_branch_road",
        "official_context": "when_turning_into_branch_road_that_intersects_railway",
        "railway_to_sign_section_threshold_m": 100,
        "threshold_operator": "less_than",
        "scene_route_or_distance_inferred": False,
    },
}


def _plain_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    unmarked = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return unmarked.replace("đ", "d")


def _new_fact(
    observation: SignObservation,
    subject_id: str,
    relation_id: str | None,
    ocr_item: dict[str, Any],
    index: int,
    predicate: str,
    value: Any,
) -> ContextFact:
    confidence = normalize_confidence(ocr_item.get("confidence"))
    raw_text = str(ocr_item.get("raw_text") or ocr_item.get("normalized_value") or "")
    return ContextFact(
        fact_id=f"{observation.observation_id}:supplement:{index}:{predicate}",
        predicate=predicate,
        value=value,
        status="observed" if confidence in {"high", "medium"} else "ambiguous",
        source="supplementary_sign_ocr",
        raw_text=raw_text,
        confidence=confidence,
        evidence=[*observation.evidence, *(
            [ocr_item["evidence"]] if isinstance(ocr_item.get("evidence"), dict) else []
        )],
        metadata={
            "subject_observation_id": subject_id,
            "source_observation_id": observation.observation_id,
            "relation_id": relation_id,
            "supplementary_sign_code": normalize_sign_code(observation.sign_code),
            "extraction": "strict_ocr_pattern",
        },
    )


def extract_supplementary_context_facts(
    observation: SignObservation,
    subject_id: str,
    relation_id: str | None = None,
) -> list[ContextFact]:
    """Extract only explicit OCR values from recognized S-series panels."""
    code = normalize_sign_code(observation.sign_code)
    family = sign_family(code)
    result: list[ContextFact] = []
    if observation.identity_status != "identified":
        return result

    panel_confidence = normalize_confidence(observation.confidence)
    panel_status = "observed" if panel_confidence in {"high", "medium"} else "ambiguous"
    if code in S503_VISUAL_SEMANTICS:
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:panel-visual-semantics",
            predicate="supplementary_panel_visual_semantics",
            value={
                **S503_VISUAL_SEMANTICS[code],
                "orientation_reference": "sign_face",
                "road_relative_direction_resolved": False,
            },
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": "QCVN 41:2024/BGTVT Appendix F.3(a-d), Figures F.3a-F.3b",
                "road_relative_orientation_required": True,
            },
        ))
    elif code in SCOPE_DIMENSION_BY_CODE:
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:scope-dimension",
            predicate="supplementary_scope_dimension",
            value=SCOPE_DIMENSION_BY_CODE[code],
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": (
                    "QCVN 41:2024/BGTVT Appendix F.4" if code == "S.504"
                    else "QCVN 41:2024/BGTVT Appendix F.5" if code == "S.505a"
                    else "QCVN 41:2024/BGTVT Appendix F.8"
                ),
                "specific_lane_vehicle_or_route_value_still_requires_visual_resolution": True,
            },
        ))
    elif code == "S.H.6":
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:exception-semantics",
            predicate="supplementary_exception_semantics",
            value=SH6_EXCEPTION_SEMANTICS,
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": "QCVN 41:2024/BGTVT Appendix F.17, Figure F.17",
                "exception_group_not_inferred_from_sign_code": True,
                "group_match_must_be_evaluated_before_applying_primary_restriction": True,
            },
        ))
        nonempty_ocr: list[tuple[int, dict[str, Any], str]] = []
        for index, item in enumerate(observation.ocr_texts):
            raw = str(item.get("normalized_value") or item.get("raw_text") or "").strip()
            if not raw:
                continue
            nonempty_ocr.append((index, item, raw))
            result.append(_new_fact(
                observation, subject_id, relation_id, item, index,
                "supplementary_exception_group_text", raw,
            ))
        # Reuse the same explicit vehicle vocabulary as query Intent. Resolve
        # only one high/medium-confidence text block and one recognized class;
        # multiple OCR blocks or mixed/unknown labels remain unresolved rather
        # than letting the last OCR item overwrite an earlier one.
        if len(nonempty_ocr) == 1:
            index, item, raw = nonempty_ocr[0]
            if normalize_confidence(item.get("confidence")) in {"high", "medium"}:
                # Local import avoids coupling catalog extraction to Intent at
                # module initialization while sharing its vehicle vocabulary.
                from src.core.symbolic.question_intent import extract_vehicle_mentions

                exception_classes = extract_vehicle_mentions(raw)
                if len(exception_classes) == 1:
                    result.extend([
                        _new_fact(
                            observation, subject_id, relation_id, item, index,
                            "sh6_exception_actor_classes", exception_classes,
                        ),
                        _new_fact(
                            observation, subject_id, relation_id, item, index,
                            "sh6_exception_scope_resolved", True,
                        ),
                    ])
    elif family in SG_LANE_GUIDANCE:
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:lane-guidance-semantics",
            predicate="supplementary_lane_guidance_semantics",
            value=SG_LANE_GUIDANCE[family],
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": (
                    "QCVN 41:2024/BGTVT Appendix F.15, Figure F.15"
                    if family == "S.G.11"
                    else "QCVN 41:2024/BGTVT Appendix F.16, Figure F.16"
                ),
                "numeric_lane_count_or_lane_identity_inferred": False,
            },
        ))
    elif code == "S.509b":
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:parking-scope-semantics",
            predicate="supplementary_panel_action_semantics",
            value=S509B_SEMANTICS,
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": "QCVN 41:2024/BGTVT Appendix F.11(b), Figure F.11",
                "primary_sign_relationship_must_be_confirmed": True,
                "does_not_replace_or_narrow_primary_sign_without_legal_review": True,
            },
        ))
    elif code == "S.507":
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:standalone-guidance-semantics",
            predicate="standalone_sign_guidance_semantics",
            value=S507_STANDALONE_GUIDANCE,
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": "QCVN 41:2024/BGTVT §41.1 and Appendix F.9, Figure F.9",
                "independent_sign_not_a_scope_panel": True,
                "visual_turn_direction_not_inferred_from_code": True,
            },
        ))
    elif code in S510_WARNING_SEMANTICS:
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:supplement:warning-semantics",
            predicate="supplementary_warning_semantics",
            value=S510_WARNING_SEMANTICS[code],
            status=panel_status,
            source="official_supplementary_sign_variant_mapping",
            confidence=panel_confidence,
            evidence=list(observation.evidence),
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
                "supplementary_sign_code": code,
                "legal_basis": "QCVN 41:2024/BGTVT Appendix F.12, Figure F.12",
                "meaning_only_no_permission_or_prohibition_inferred": True,
            },
        ))
    for index, item in enumerate(observation.ocr_texts):
        raw = str(item.get("normalized_value") or item.get("raw_text") or "").strip()
        if not raw:
            continue
        plain = _plain_text(raw)
        if family in {"S.501", "S.502"}:
            match = DISTANCE_RE.search(plain)
            if match:
                predicate = "effect_length_m" if family == "S.501" else "distance_to_object_m"
                result.append(_new_fact(
                    observation, subject_id, relation_id, item, index, predicate, int(match.group(1))
                ))
        elif family == "S.508":
            parity = "odd_days" if "ngay le" in plain else "even_days" if "ngay chan" in plain else None
            if parity:
                result.append(_new_fact(
                    observation, subject_id, relation_id, item, index, "supplementary_calendar_parity", parity
                ))
            time_range = TIME_RANGE_RE.search(plain)
            if time_range:
                start_hour, start_minute, end_hour, end_minute = map(int, time_range.groups())
                if start_hour <= 23 and end_hour <= 23:
                    result.append(_new_fact(
                        observation,
                        subject_id,
                        relation_id,
                        item,
                        index,
                        "supplementary_time_window",
                        {
                            "start_minute": start_hour * 60 + start_minute,
                            "end_minute": end_hour * 60 + end_minute,
                            "raw_text": raw,
                        },
                    ))
        elif family == "S.509" and code == "S.509a":
            match = HEIGHT_RE.search(plain)
            if match:
                result.append(_new_fact(
                    observation,
                    subject_id,
                    relation_id,
                    item,
                    index,
                    "safe_height_limit_m",
                    float(match.group(1).replace(",", ".")),
                ))
        elif code == "S.505c":
            match = AXLE_LIMIT_RE.fullmatch(plain)
            if match:
                axle_kind = re.sub(r"\s+", " ", match.group(1).strip())
                predicate = AXLE_PREDICATES.get(axle_kind)
                if predicate:
                    result.append(_new_fact(
                        observation,
                        subject_id,
                        relation_id,
                        item,
                        index,
                        predicate,
                        float(match.group(2).replace(",", ".")),
                    ))
        elif code == "S.505b":
            match = GROSS_WEIGHT_LIMIT_RE.fullmatch(plain)
            row_index = item.get("table_row_index")
            if match and type(row_index) is int and row_index in {1, 2, 3}:
                fact = _new_fact(
                    observation,
                    subject_id,
                    relation_id,
                    item,
                    index,
                    "s505b_row_gross_weight_limit_t",
                    float(match.group(1).replace(",", ".")),
                )
                fact.metadata["table_row_index"] = row_index
                result.append(fact)
    if code in {"S.505b", "S.505c"}:
        axle_predicates = set(AXLE_PREDICATES.values())
        grouped: dict[tuple[str, int | None], list[ContextFact]] = {}
        unrelated: list[ContextFact] = []
        for fact in result:
            is_axle_limit = code == "S.505c" and fact.predicate in axle_predicates
            is_row_limit = (
                code == "S.505b"
                and fact.predicate == "s505b_row_gross_weight_limit_t"
                and type(fact.metadata.get("table_row_index")) is int
            )
            if is_axle_limit or is_row_limit:
                row_index = fact.metadata.get("table_row_index") if is_row_limit else None
                grouped.setdefault((fact.predicate, row_index), []).append(fact)
            else:
                unrelated.append(fact)
        for (predicate, row_index), readings in grouped.items():
            values = {fact.value for fact in readings}
            if len(values) == 1:
                unrelated.append(max(
                    readings,
                    key=lambda fact: {"high": 2, "medium": 1, "low": 0}.get(fact.confidence, 0),
                ))
                continue
            exemplar = readings[0]
            row_suffix = f":row-{row_index}" if row_index is not None else ""
            unrelated.append(ContextFact(
                fact_id=f"{observation.observation_id}:supplement:conflict:{predicate}{row_suffix}",
                predicate=predicate,
                value=None,
                status="ambiguous",
                source="supplementary_sign_ocr",
                raw_text=" | ".join(fact.raw_text or "" for fact in readings),
                confidence="low",
                evidence=[evidence for fact in readings for evidence in fact.evidence],
                metadata={
                    **exemplar.metadata,
                    "extraction": "strict_ocr_pattern_conflict",
                    "conflicting_values": sorted(values),
                },
            ))
        result = unrelated
    return result
