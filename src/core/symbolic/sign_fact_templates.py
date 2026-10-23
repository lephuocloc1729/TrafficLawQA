"""Load and instantiate the checked-in VTSDB100 fact templates."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.symbolic.fact_schema import ContextFact, SignObservation
from src.core.symbolic.condition_facts import expand_condition_facts
from src.core.symbolic.sign_normalizer import normalize_sign_code


TEMPLATE_PATH = Path(__file__).parent / "catalogs" / "vtsdb100_fact_templates.json"


@lru_cache(maxsize=1)
def _payload() -> dict[str, Any]:
    with TEMPLATE_PATH.open(encoding="utf-8") as stream:
        data = json.load(stream)
    return data if isinstance(data, dict) else {}


def _code_template(sign_code: str) -> dict[str, Any]:
    matches = [
        item for item in _payload().get("classes", {}).values()
        if isinstance(item, dict) and normalize_sign_code(item.get("sign_code")) == sign_code
    ]
    if not matches:
        return {"facts": []}

    by_predicate: dict[str, list[dict[str, Any]]] = {}
    for item in matches:
        for fact in item.get("facts", []):
            if isinstance(fact, dict):
                by_predicate.setdefault(str(fact.get("predicate")), []).append(fact)
    facts: list[dict[str, Any]] = []
    for predicate, variants in by_predicate.items():
        if predicate == "conditions":
            by_condition: dict[str, list[dict[str, Any]]] = {}
            for variant in variants:
                for condition in expand_condition_facts([variant]):
                    condition_type = condition.get("predicate")
                    if condition_type:
                        by_condition.setdefault(condition_type, []).append(condition)
            merged_conditions: dict[str, Any] = {}
            for condition_predicate, condition_variants in by_condition.items():
                template_condition = {
                    key: value for key, value in condition_variants[0].items()
                    if key != "predicate"
                }
                invariant = all(
                    item.get("value") == template_condition.get("value")
                    for item in condition_variants
                )
                if template_condition.get("fill_from") == "ocr_or_detector_attribute" or not invariant:
                    template_condition["value"] = None
                    template_condition["status"] = "not_observed"
                    template_condition["confidence"] = "low"
                merged_conditions[condition_predicate] = template_condition
            facts.append({"predicate": "conditions", "value": merged_conditions})
            continue
        # Preserve a value only when it is invariant across all detector classes
        # for the same canonical code. Numeric variants otherwise stay blank.
        first_value = variants[0].get("value")
        invariant = all(item.get("value") == first_value for item in variants)
        template = dict(variants[0])
        if not invariant:
            template["value"] = None
            template["status"] = "not_observed"
            template["confidence"] = "low"
        if template.get("fill_from") == "ocr_or_detector_attribute":
            # A code-only lookup must never inherit a numeric value from one
            # dataset class variant (for example P.106b/2.5 t).
            template["value"] = None
            template["status"] = "not_observed"
            template["confidence"] = "low"
        facts.append(template)
    return {"facts": facts}


def _overlay_class_attributes(fact: dict[str, Any], attributes: dict[str, Any]) -> dict[str, Any]:
    if fact.get("predicate") == "conditions" and isinstance(fact.get("value"), dict):
        by_predicate = {
            "maximum_payload_mass": ("payload_limit_t",),
            "maximum_vehicle_mass": ("total_weight_limit_t",),
            "maximum_axle_load": ("axle_load_limit_t",),
            "maximum_vehicle_height": ("height_limit_m",),
            "maximum_vehicle_width": ("width_limit_m",),
            "maximum_vehicle_length": ("length_limit_m",),
            "maximum_speed": ("speed_limit_value", "speed_limit_kmh"),
            "minimum_speed": ("minimum_speed_limit_kmh", "min_speed_kmh"),
        }
        changed = False
        values = dict(fact["value"])
        for condition_predicate, raw in values.items():
            entries = raw if isinstance(raw, list) else [raw]
            updated = []
            for item in entries:
                if not isinstance(item, dict) or item.get("fill_from") != "ocr_or_detector_attribute":
                    updated.append(item)
                    continue
                value = next((
                    attributes.get(key) for key in by_predicate.get(condition_predicate, ())
                    if attributes.get(key) is not None
                ), None)
                if value is None:
                    updated.append(item)
                else:
                    updated.append({**item, "value": value, "status": "derived", "confidence": "high"})
                    changed = True
            values[condition_predicate] = updated if isinstance(raw, list) else updated[0]
        return {**fact, "value": values} if changed else fact
    if fact.get("fill_from") != "ocr_or_detector_attribute":
        return fact
    attribute_by_predicate = {
        "maximum_payload_mass": ("payload_limit_t",),
        "maximum_vehicle_mass": ("total_weight_limit_t",),
        "maximum_axle_load": ("axle_load_limit_t",),
        "maximum_vehicle_height": ("height_limit_m",),
        "maximum_vehicle_width": ("width_limit_m",),
        "maximum_vehicle_length": ("length_limit_m",),
        "maximum_combination_length": ("combination_length_limit_m",),
        "minimum_distance": ("minimum_following_distance_m",),
        "maximum_speed": ("speed_limit_value", "speed_limit_kmh"),
        "minimum_speed": ("minimum_speed_limit_kmh", "min_speed_kmh"),
    }
    value = next((
        attributes.get(key) for key in attribute_by_predicate.get(fact.get("predicate"), ())
        if attributes.get(key) is not None
    ), None)
    if value is None:
        return fact
    return {**fact, "value": value, "status": "derived", "confidence": "high"}


def instantiate_sign_fact_templates(observation: SignObservation) -> list[ContextFact]:
    """Create default facts for a recognized sign, leaving OCR slots unresolved."""
    if not observation.sign_code or observation.identity_status != "identified":
        return []
    code = normalize_sign_code(observation.sign_code)
    label = str(observation.metadata.get("detector_class_label") or "").strip().lower()
    entry = _payload().get("classes", {}).get(label)
    if not isinstance(entry, dict) or normalize_sign_code(entry.get("sign_code")) != code:
        entry = _code_template(code)

    class_attributes = observation.visual_attributes.get("normalized_conditions", {})
    class_attributes = class_attributes if isinstance(class_attributes, dict) else {}
    templates = [
        _overlay_class_attributes(item, class_attributes)
        for item in entry.get("facts", [])
        if isinstance(item, dict)
    ]

    result: list[ContextFact] = []
    for index, template in enumerate(templates):
        predicate = str(template.get("predicate") or "").strip()
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:template:{predicate}:{index}",
            predicate=predicate,
            value=template.get("value"),
            status=template.get("status", "derived"),
            source="sign_code_template",
            confidence=template.get("confidence", observation.confidence),
            unit=template.get("unit"),
            scope=template.get("scope", {}),
            conditions=template.get("conditions", {}),
            metadata={
                "subject_observation_id": observation.observation_id,
                **({"fill_from": template["fill_from"]} if template.get("fill_from") else {}),
            },
        ))
    return result


def instantiate_supplementary_template(observation: SignObservation) -> list[ContextFact]:
    if observation.source != "paddleocr_text_panel":
        return []
    from src.core.symbolic.sign_information_facts import extract_sign_information_facts

    observed = extract_sign_information_facts(observation, subject_id=observation.observation_id)
    if any(item.predicate in {"supplementary_time_window", "time_window", "active_time"} for item in observed):
        return []
    templates = _payload().get("supplementary_templates", {}).get("ocr_time_panel", [])
    return [
        ContextFact(
            fact_id=f"{observation.observation_id}:template:{item.get('predicate', 'active_time')}",
            predicate=item.get("predicate", "active_time"),
            value=item.get("value"),
            status=item.get("status", "not_observed"),
            source="supplementary_panel_template",
            confidence=item.get("confidence", "low"),
            metadata={"subject_observation_id": observation.observation_id},
        )
        for item in templates
        if isinstance(item, dict)
    ]
