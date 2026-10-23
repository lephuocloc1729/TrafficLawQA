from __future__ import annotations

from typing import Any

from src.core.symbolic.fact_schema import FACT_PREDICATE_ALIASES, ContextFact, SignObservation
from src.core.symbolic.sign_fact_templates import (
    instantiate_sign_fact_templates,
    instantiate_supplementary_template,
)
from src.core.symbolic.sign_information_facts import extract_sign_information_facts
from src.core.symbolic.sign_normalizer import normalize_sign_code
from src.core.symbolic.sign_numeric_facts import extract_main_sign_context_facts
from src.core.symbolic.sign_schema import normalize_confidence
from src.core.symbolic.supplementary_facts import extract_supplementary_context_facts


def observation_context_facts(
    observation: SignObservation,
    subject_id: str,
    relation_id: str | None = None,
    source: str = "sign_observation_interpretation",
) -> list[ContextFact]:
    normalized_fields = {
        "time_condition": "supplementary_time_condition",
        "time_window": "time_window",
        "vehicle_scope": "vehicle_scope_condition",
        "direction_scope": "direction_scope_condition",
        "distance_scope": "distance_scope_condition",
        "numeric_value": "supplementary_numeric_value",
        "speed_limit_value": "speed_limit_value",
        "minimum_speed_limit_kmh": "minimum_speed_limit_kmh",
        "sign_applicability_calendar_parity": "sign_applicability_calendar_parity",
        "slope_percent": "slope_percent",
        "total_weight_limit_t": "total_weight_limit_t",
        "axle_load_limit_t": "axle_load_limit_t",
        "payload_limit_t": "payload_limit_t",
        "payload_limit_present": "payload_limit_present",
        "payload_limit_relation": "payload_limit_relation",
        "passenger_seat_limit_present": "passenger_seat_limit_present",
        "passenger_seat_limit": "passenger_seat_limit",
        "passenger_seat_limit_relation": "passenger_seat_limit_relation",
        "bus_lane_marking": "bus_lane_marking",
        "combination_length_limit_m": "combination_length_limit_m",
        "height_limit_m": "height_limit_m",
        "width_limit_m": "width_limit_m",
        "length_limit_m": "length_limit_m",
        "minimum_following_distance_m": "minimum_following_distance_m",
        "sg11_lane_direction_mapping_resolved": "sg11_lane_direction_mapping_resolved",
        "sg12_non_traversable_lane_resolved": "sg12_non_traversable_lane_resolved",
        "sh6_exception_scope_resolved": "sh6_exception_scope_resolved",
        "sh6_exception_actor_classes": "sh6_exception_actor_classes",
        "sh6_exception_applies_to_actor_resolved": "sh6_exception_applies_to_actor_resolved",
        # The S.505a vehicle silhouettes are not OCR text. These values must
        # come from an explicit visual classification/review, never from the
        # supplementary sign code or the primary sign's default scope.
        "s505a_vehicle_scope_resolved": "s505a_vehicle_scope_resolved",
        "s505a_vehicle_classes": "s505a_vehicle_classes",
        "exceptions": "supplementary_exceptions",
        "sign_effective_scope_confirmed": "sign_effective_scope_confirmed",
    }
    normalized_units = {
        "speed_limit_value": "km/h",
        "minimum_speed_limit_kmh": "km/h",
        "total_weight_limit_t": "t",
        "axle_load_limit_t": "t",
        "payload_limit_t": "t",
        "height_limit_m": "m",
        "width_limit_m": "m",
        "length_limit_m": "m",
        "combination_length_limit_m": "m",
        "minimum_following_distance_m": "m",
        "passenger_seat_limit": "seats",
    }
    result: list[ContextFact] = []
    main_numeric_ocr_facts = (
        extract_main_sign_context_facts(observation, subject_id=subject_id)
        if observation.sign_code and not observation.sign_code.upper().startswith("S.")
        else []
    )
    numeric_ocr_predicates = {
        FACT_PREDICATE_ALIASES.get(item.predicate, item.predicate)
        for item in main_numeric_ocr_facts
    }
    normalized_field_predicates = {
        "speed_limit_value": "maximum_speed",
        "minimum_speed_limit_kmh": "minimum_speed",
        "total_weight_limit_t": "maximum_vehicle_mass",
        "axle_load_limit_t": "maximum_axle_load",
        "payload_limit_t": "maximum_payload_mass",
        "height_limit_m": "maximum_vehicle_height",
        "width_limit_m": "maximum_vehicle_width",
        "length_limit_m": "maximum_vehicle_length",
        "combination_length_limit_m": "maximum_combination_length",
        "minimum_following_distance_m": "minimum_distance",
    }
    fields = observation.visual_attributes.get("normalized_conditions", {})
    if isinstance(fields, dict):
        for source_key, predicate in normalized_fields.items():
            # P.121's minimum distance must come from OCR of its displayed
            # number. Ignore pre-normalized values so conflicting OCR remains
            # ambiguous and cannot be turned into a firm prohibition.
            if (
                normalize_sign_code(observation.sign_code) == "P.121"
                and source_key == "minimum_following_distance_m"
            ):
                continue
            if source_key in fields:
                if normalized_field_predicates.get(source_key, predicate) in numeric_ocr_predicates:
                    continue
                result.append(ContextFact(
                    fact_id=f"{observation.observation_id}:{predicate}",
                    predicate=predicate,
                    value=fields[source_key],
                    status="derived",
                    source=(
                        "yolo26s_class_attribute"
                        if observation.source == "yolo26s_vtsdb100"
                        else source
                    ),
                    confidence=observation.confidence,
                    evidence=observation.evidence,
                    unit=normalized_units.get(source_key),
                    metadata={
                        "subject_observation_id": subject_id,
                        "source_observation_id": observation.observation_id,
                        "relation_id": relation_id,
                    },
                ))
    for index, ocr_item in enumerate(observation.ocr_texts):
        ocr_confidence = normalize_confidence(ocr_item.get("confidence"))
        raw_text = ocr_item.get("raw_text")
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:ocr:{index}",
            predicate="sign_text" if observation.observation_id == subject_id else "supplementary_text",
            value=ocr_item.get("normalized_value", ocr_item.get("raw_text")),
            status=("observed" if ocr_confidence in {"high", "medium"} else "ambiguous")
            if raw_text else "unknown",
            source="ocr",
            raw_text=raw_text,
            confidence=ocr_confidence,
            evidence=observation.evidence,
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
            },
        ))
    if observation.sign_code and observation.sign_code.upper().startswith("S."):
        result.extend(extract_supplementary_context_facts(
            observation, subject_id=subject_id, relation_id=relation_id
        ))
    elif observation.sign_code:
        result.extend(main_numeric_ocr_facts)
    result.extend(extract_sign_information_facts(
        observation,
        subject_id=subject_id,
        relation_id=relation_id,
    ))
    template_facts = instantiate_sign_fact_templates(observation)
    for template_fact in template_facts:
        if template_fact.metadata.get("fill_from") == "ocr_or_detector_attribute":
            template_predicate = FACT_PREDICATE_ALIASES.get(
                template_fact.predicate, template_fact.predicate
            )
            observed_matches = [
                fact for fact in result
                if FACT_PREDICATE_ALIASES.get(fact.predicate, fact.predicate) == template_predicate
            ]
            if observed_matches:
                # The observed OCR/detector fact supplies the value and status;
                # the sign-code template supplies its unit and legal scope.
                # This preserves OCR ambiguity while avoiding duplicate values.
                for observed in observed_matches:
                    if observed.unit is None:
                        observed.unit = template_fact.unit
                    observed.scope = {**template_fact.scope, **observed.scope}
                    observed.conditions = {**template_fact.conditions, **observed.conditions}
                continue
        result.append(template_fact)
    result.extend(instantiate_supplementary_template(observation))
    return result


