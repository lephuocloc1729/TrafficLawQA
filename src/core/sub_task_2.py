import json
import os
import re
import sys
import time
import torch
import logging
import argparse
import unicodedata as ud
from datetime import datetime, time as datetime_time

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
from src.core.symbolic.condition_facts import CONDITION_PREDICATES, expand_condition_facts
from src.core.symbolic.sign_normalizer import normalize_detected_sign_to_observation, normalize_sign_code
from src.core.symbolic.sign_relations import link_supplementary_signs
from src.core.symbolic.traffic_sign_ocr import get_traffic_sign_ocr
from src.core.symbolic.sign_information_facts import (
    TIME_RANGE_RE,
    extract_nearby_highway_profile_facts,
)
from src.prompts.answer_prompt import (
    SYSTEM_PROMPT_CHECK_PERMISSION,
    SYSTEM_PROMPT_FREEFORM_CITED_QA,
    SYSTEM_PROMPT_INFORMATION,
    SYSTEM_PROMPT_SIGN_VARIANT_DISAMBIGUATION,
    SYSTEM_PROMPT_TRAFFICLAWQA_RAG,
    SYSTEM_PROMPT_UNSUPPORTED_INTENT,
)


MAX_ARTICLE_CHARS = 6000
MAX_ARTICLES_IN_PROMPT = 8
SIGN_CODE_RE = re.compile(
    r"\b([A-Z]{1,3}(?:\.[A-Z])?[,\.]\d+[a-z]?)\b",
    re.IGNORECASE,
)


def _ocr_text_panel_observations(sample: dict[str, Any]) -> tuple[list[SignObservation], set[str]]:
    """Turn OCR time text below a detected sign into a linked panel observation."""
    full_ocr = sample.get("full_image_ocr", [])
    if not isinstance(full_ocr, list):
        return [], set()

    grouped: dict[tuple[float, ...], list[dict[str, Any]]] = {}
    for item in full_ocr:
        if not isinstance(item, dict) or item.get("region_role") != "below_sign_context":
            continue
        associated = item.get("associated_sign_bbox")
        polygon = item.get("bbox")
        points = [
            point for point in polygon or []
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not isinstance(associated, (list, tuple)) or len(associated) != 4 or len(points) < 4:
            continue
        raw_text = str(item.get("normalized_value") or item.get("raw_text") or "")
        if not TIME_RANGE_RE.search(raw_text):
            continue
        try:
            key = tuple(round(float(value), 1) for value in associated)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(key, []).append(item)

    observations: list[SignObservation] = []
    consumed_text_keys: set[str] = set()
    image_id = str(sample.get("image_id") or "scene")
    for index, items in enumerate(grouped.values()):
        all_points = [
            (float(point[0]), float(point[1]))
            for item in items
            for point in item.get("bbox", [])
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(all_points) < 4:
            continue
        panel_bbox = [
            min(point[0] for point in all_points),
            min(point[1] for point in all_points),
            max(point[0] for point in all_points),
            max(point[1] for point in all_points),
        ]
        confidence_labels = [str(item.get("confidence") or "low") for item in items]
        confidence = (
            "low" if "low" in confidence_labels
            else "medium" if "medium" in confidence_labels
            else "high"
        )
        observation_id = f"{image_id}:ocr-time-panel-{index + 1}"
        observations.append(SignObservation(
            observation_id=observation_id,
            role="supplementary",
            bbox=panel_bbox,
            identity_status="unidentified",
            ocr_texts=items,
            confidence=confidence,
            source="paddleocr_text_panel",
            evidence=[{"type": "image", "ref": image_id}],
        ))
        consumed_text_keys.update(
            json.dumps(item.get("bbox"), separators=(",", ":"), sort_keys=True)
            for item in items
        )
    return observations, consumed_text_keys


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


def _without_verification_status(value: Any) -> Any:
    """Hide review-status metadata from the answer model's reasoning context."""
    if isinstance(value, dict):
        return {
            key: _without_verification_status(item)
            for key, item in value.items()
            if key != "verification_status"
        }
    if isinstance(value, list):
        return [_without_verification_status(item) for item in value]
    return value


def _answer_system_prompt(symbolic_context: dict[str, Any]) -> str:
    """Choose a focused answer prompt when the request has one supported intent."""
    query_intent = symbolic_context.get("query_intent", {})
    intents = query_intent.get("intents", []) if isinstance(query_intent, dict) else []
    kinds = {
        normalize_intent_kind(str(item.get("kind") or "UNKNOWN"))
        for item in intents
        if isinstance(item, dict)
    }
    if kinds and kinds == {"CHECK_PERMISSION"}:
        return SYSTEM_PROMPT_CHECK_PERMISSION
    if kinds and kinds == {"INFORMATION"}:
        return SYSTEM_PROMPT_INFORMATION
    if kinds and kinds == {"UNKNOWN"}:
        return SYSTEM_PROMPT_UNSUPPORTED_INTENT
    return SYSTEM_PROMPT_FREEFORM_CITED_QA


_INFO_FACT_PREDICATES = {
    "legal_meaning": {"sign_meaning"},
    "active_time": {"active_time"},
    "calendar_parity": {"calendar_parity"},
    "minimum_speed": {"minimum_speed"},
    "maximum_speed": {"maximum_speed"},
    "speed_limit": {"minimum_speed", "maximum_speed"},
    "minimum_distance": {"minimum_distance", "supplementary_distance_m"},
    "maximum_vehicle_height": {"maximum_vehicle_height"},
    "maximum_vehicle_mass": {"maximum_vehicle_mass"},
    "maximum_payload_mass": {"maximum_payload_mass"},
    "maximum_seat_count": {"maximum_seat_count"},
    "prohibited_actor": {"prohibited_actor", "prohibited_vehicle", "prohibited_subject"},
    "prohibited_vehicle": {"prohibited_actor", "prohibited_vehicle"},
    "prohibited_subject": {"prohibited_actor", "prohibited_subject"},
    "vehicle_scope": {"vehicle_scope"},
    "destination_distance_km": {"destination_distance_km"},
    "destination_name": {"destination_name"},
    "highway_name": {"highway_name"},
    "highway_route_code": {"highway_route_code"},
    "direction_scope": {"direction_scope"},
    "conditions": {"conditions"},
}


def _vehicle_scope_matches(requested: Any, actual: Any) -> bool:
    aliases = {
        "motorcycle_or_moped": {"motorcycle", "moped", "motorcycle_or_moped"},
        "motorcycle": {"motorcycle", "motorcycle_or_moped"},
        "moped": {"moped", "motorcycle_or_moped"},
        "automobile": {"automobile", "passenger_car", "car"},
        "passenger_car": {"automobile", "passenger_car", "car"},
    }
    requested_value = str(requested).strip().lower()
    actual_value = str(actual).strip().lower()
    requested_values = aliases.get(requested_value, {requested_value})
    actual_values = aliases.get(actual_value, {actual_value})
    motor_vehicle_types = {
        "motorcycle", "motorcycle_or_moped", "moped", "automobile", "passenger_car", "car", "bus", "truck",
        "tractor", "tractor_unit", "special_use_vehicle", "special_use_machine", "construction_machine",
        "agricultural_forestry_machine", "other_motor_vehicle", "motorized_tricycle",
    }
    if "vehicle" in actual_values and requested_value != "pedestrian":
        return True
    if "motor_vehicle" in actual_values and requested_value in {
        *motor_vehicle_types,
    }:
        return True
    return bool(requested_values.intersection(actual_values))


def _plain_vietnamese(text: str) -> str:
    value = ud.normalize("NFD", text.lower())
    value = "".join(char for char in value if ud.category(char) != "Mn")
    return value.replace("đ", "d")


def _catalog_meaning_matches_filters(meaning: str, filters: dict[str, Any]) -> bool:
    normalized = _plain_vietnamese(meaning)
    action = str(filters.get("action") or "").lower()
    action_phrases = {
        "turn_left": ("re trai", "queo trai"),
        "turn_right": ("re phai", "queo phai"),
        "u_turn": ("quay dau",),
        "parking": ("do xe", "dau xe"),
        "stopping": ("dung xe", "tam dung"),
        "entering": ("di vao", "cam vao"),
        "exiting": ("di ra", "roi khoi"),
        "going_straight": ("di thang",),
    }
    if action in action_phrases and not any(term in normalized for term in action_phrases[action]):
        return False

    vehicle = str(filters.get("vehicle_type") or "").lower()
    recognized_vehicle_terms = {
        "motorcycle": ("mo to", "xe may", "xe gan may"),
        "motorcycle_or_moped": ("mo to", "xe may", "xe gan may"),
        "moped": ("xe gan may",),
        "automobile": ("o to", "xe con", "xe hoi"),
        "passenger_car": ("o to", "xe con", "xe hoi"),
        "truck": ("xe tai", "o to tai"),
        "bus": ("xe buyt", "xe bus", "xe khach"),
    }
    stated_vehicle_groups = {
        key for key, terms in recognized_vehicle_terms.items()
        if any(term in normalized for term in terms)
    }
    if stated_vehicle_groups:
        if not any(_vehicle_scope_matches(vehicle, stated) for stated in stated_vehicle_groups):
            if "xe co gioi" not in normalized:
                return False
    return True


def _information_condition_state(entry: dict[str, Any], intent_conditions: dict[str, Any]) -> str:
    """Compare supplied query details with one condition attached to an answer fact."""
    predicate = entry.get("predicate")
    vehicle_value_by_condition = {
        "maximum_payload_mass": "vehicle_payload_t",
        "maximum_vehicle_mass": "vehicle_gross_weight_t",
        "maximum_axle_load": "vehicle_axle_load_t",
        "maximum_vehicle_height": "vehicle_height_m",
        "maximum_vehicle_width": "vehicle_width_m",
    }
    actual_value = None
    if predicate in vehicle_value_by_condition:
        actual_value = intent_conditions.get(vehicle_value_by_condition[predicate])
        if isinstance(actual_value, dict):
            actual_value = actual_value.get("value")
    elif predicate == "active_time":
        actual_value = intent_conditions.get("observed_at", intent_conditions.get("observed_time"))
    elif predicate == "calendar_parity":
        actual_value = intent_conditions.get("calendar_parity")
        if actual_value is None:
            observed_date = intent_conditions.get("observed_at", intent_conditions.get("date"))
            if isinstance(observed_date, str):
                try:
                    actual_value = "odd" if datetime.fromisoformat(observed_date).day % 2 else "even"
                except ValueError:
                    pass
    if actual_value is None:
        return "unresolved"

    if predicate in vehicle_value_by_condition:
        try:
            actual = float(actual_value)
            limit = float(entry.get("value"))
        except (TypeError, ValueError):
            return "unresolved"
        operator = entry.get("operator")
        if operator == ">":
            return "satisfied" if actual > limit else "not_satisfied"
        if operator == ">=":
            return "satisfied" if actual >= limit else "not_satisfied"
        if operator == "<":
            return "satisfied" if actual < limit else "not_satisfied"
        if operator == "<=":
            return "satisfied" if actual <= limit else "not_satisfied"
        return "unresolved"

    if predicate == "calendar_parity":
        expected = str(entry.get("value") or "").replace("_days", "").lower()
        actual = str(actual_value).replace("_days", "").lower()
        return "satisfied" if expected == actual else "not_satisfied"

    if predicate == "active_time":
        if isinstance(actual_value, str):
            try:
                observed_time = datetime.fromisoformat(actual_value).time()
            except ValueError:
                try:
                    observed_time = datetime_time.fromisoformat(actual_value)
                except ValueError:
                    return "unresolved"
        else:
            return "unresolved"
        windows = entry.get("value")
        windows = windows if isinstance(windows, list) else [windows]
        for window in windows:
            if not isinstance(window, dict):
                continue
            try:
                start = datetime_time.fromisoformat(str(window.get("from", window.get("start"))))
                end = datetime_time.fromisoformat(str(window.get("to", window.get("end"))))
            except ValueError:
                continue
            if (start <= end and start <= observed_time <= end) or (
                start > end and (observed_time >= start or observed_time <= end)
            ):
                return "satisfied"
        return "not_satisfied"
    return "unresolved"


def _fact_matches_information_filters(
    fact: dict[str, Any],
    filters: dict[str, Any],
    sibling_facts: list[dict[str, Any]],
    requested_fact: str,
) -> bool:
    scope = fact.get("scope") if isinstance(fact.get("scope"), dict) else {}
    vehicle = filters.get("vehicle_type")
    if vehicle:
        included = scope.get("vehicle_types")
        excluded = scope.get("excluded_vehicle_types", [])
        included_values = included if isinstance(included, list) else [included] if included else []
        excluded_values = excluded if isinstance(excluded, list) else [excluded] if excluded else []
        if included_values and not any(_vehicle_scope_matches(vehicle, item) for item in included_values):
            return False
        if any(_vehicle_scope_matches(vehicle, item) for item in excluded_values):
            return False
    for key in ("vehicle_type", "action", "distance_kind"):
        requested = filters.get(key)
        actual = scope.get(key)
        if key == "vehicle_type" and actual is None:
            actual = scope.get("vehicle_types")
        if requested and actual:
            actual_values = actual if isinstance(actual, list) else [actual]
            if key == "vehicle_type":
                if not any(_vehicle_scope_matches(requested, item) for item in actual_values):
                    return False
            elif requested not in actual_values:
                return False

    # A supplementary time fact is attached to its primary sign. If that sign
    # explicitly states a vehicle/action restriction, use it to scope the fact.
    sibling_predicates = {item.get("predicate") for item in sibling_facts if isinstance(item, dict)}
    vehicle = filters.get("vehicle_type")
    actor_scope_predicates = {"prohibited_actor", "prohibited_vehicle", "prohibited_subject", "vehicle_scope_condition", "vehicle_scope"}
    if vehicle and sibling_predicates.intersection(actor_scope_predicates):
        restrictions = [
            item
            for item in sibling_facts
            if isinstance(item, dict) and item.get("predicate") in actor_scope_predicates
        ]
        def flatten(value: Any) -> set[str]:
            if isinstance(value, str):
                return {value.strip().lower()}
            if isinstance(value, dict):
                return {part for item in value.values() for part in flatten(item)}
            if isinstance(value, list):
                return {part for item in value for part in flatten(item)}
            return set()
        matched_restriction = False
        for item in restrictions:
            item_scope = item.get("scope") if isinstance(item.get("scope"), dict) else {}
            values = item_scope.get("vehicle_types")
            if values is None:
                values = item.get("value")
            normalized_values = flatten(values)
            excluded_values = flatten(item_scope.get("excluded_vehicle_types", []))
            conditions = item.get("conditions") if isinstance(item.get("conditions"), dict) else {}
            exceptions = flatten(conditions.get("exceptions", []))
            if normalized_values and any(_vehicle_scope_matches(vehicle, value) for value in normalized_values):
                if any(_vehicle_scope_matches(vehicle, value) for value in excluded_values | exceptions):
                    continue
                matched_restriction = True
                break
        if restrictions and not matched_restriction:
            return False
    action = filters.get("action")
    action_predicates = {"prohibited_action", "restricted_action", "mandatory_action", "recommended_action", "action_scope_condition"}
    if action and sibling_predicates.intersection(action_predicates):
        if not any(
            item.get("predicate") in action_predicates and action in flatten(item.get("value"))
            for item in sibling_facts if isinstance(item, dict)
        ):
            return False
    return True


def _information_answer_facts(
    traffic_facts: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    requested_fact = str(intent.get("requested_fact") or "").strip().lower()
    predicates = _INFO_FACT_PREDICATES.get(requested_fact)
    if predicates is None:
        info_type = str(intent.get("information_type") or "").strip().lower()
        if info_type == "time_condition":
            predicates = {"conditions", "active_time", "calendar_parity"}
        elif info_type == "sign_meaning":
            predicates = {"sign_meaning"}
        elif info_type == "speed":
            predicates = {"conditions", "minimum_speed", "maximum_speed"}
        elif info_type in {"prohibited_subjects", "prohibited_actors"}:
            predicates = {"prohibited_actor", "prohibited_vehicle", "prohibited_subject"}
        elif info_type == "destination_distance":
            predicates = {"destination_distance_km", "destination_name"}
        elif info_type == "road_identity":
            predicates = {"highway_name", "highway_route_code"}
        elif info_type == "direction_scope":
            predicates = {"direction_scope"}
        else:
            predicates = set()

    filters = intent.get("filters") if isinstance(intent.get("filters"), dict) else {}
    target_payload = intent.get("target")
    target = target_payload if isinstance(target_payload, dict) else {}
    target_kind = str(target.get("kind") or target_payload or "").strip().lower()
    target_codes = {
        normalize_sign_code(str(code))
        for code in target.get("sign_codes", [])
        if code
    } if isinstance(target.get("sign_codes"), list) else set()
    target_observation_ids = {
        str(item) for item in target.get("observation_ids", []) if item
    } if isinstance(target.get("observation_ids"), list) else set()
    observations = traffic_facts.get("signs", traffic_facts.get("sign_observations", []))
    observations = observations if isinstance(observations, list) else []
    observation_by_id = {
        str(item.get("id")): item
        for item in observations
        if isinstance(item, dict) and item.get("id")
    }
    relation_payload = traffic_facts.get("relations", traffic_facts.get("sign_relations", []))
    relation_payload = relation_payload if isinstance(relation_payload, list) else []
    panel_targets: dict[str, list[str]] = {}
    for relation in relation_payload:
        if not isinstance(relation, dict):
            continue
        source_id = str(relation.get("from") or relation.get("source_observation_id") or "")
        target_id = str(relation.get("to") or relation.get("target_observation_id") or "")
        if source_id and target_id:
            panel_targets.setdefault(source_id, []).append(target_id)
    selected_observations = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        identity = observation.get("identity") if isinstance(observation.get("identity"), dict) else {}
        sign_code = observation.get("code") or identity.get("sign_code")
        observation_facts = observation.get("facts", [])
        observation_facts = observation_facts if isinstance(observation_facts, list) else []
        meaning = next((
            str(item.get("value") or "") for item in observation_facts
            if isinstance(item, dict) and item.get("predicate") == "sign_meaning"
        ), "")
        condition_type = filters.get("condition_type") or (
            requested_fact if requested_fact in CONDITION_PREDICATES else None
        )
        if condition_type == "active_time" and filters:
            catalog = observation.get("catalog") if isinstance(observation.get("catalog"), dict) else {}
            meaning = str(meaning or catalog.get("legal_meaning") or "").strip()
            if not meaning:
                related_meanings = [
                    str(next((
                        item.get("value") for item in (observation_by_id.get(target_id, {}).get("facts") or [])
                        if isinstance(item, dict) and item.get("predicate") == "sign_meaning"
                    ), None) or (observation_by_id.get(target_id, {}).get("catalog") or {}).get("legal_meaning") or "").strip()
                    for target_id in panel_targets.get(str(observation.get("id") or ""), [])
                ]
                meaning = next((item for item in related_meanings if item), "")
            if meaning and not _catalog_meaning_matches_filters(meaning, filters):
                continue
            if not meaning and (filters.get("vehicle_type") or filters.get("action")):
                continue
        observation_id = str(observation.get("id") or "")
        if target_kind == "sign_code" and target_codes:
            direct_code_match = normalize_sign_code(str(sign_code or "")) in target_codes
            linked_code_match = any(
                normalize_sign_code(str(observation_by_id.get(target_id, {}).get("code") or (observation_by_id.get(target_id, {}).get("identity") or {}).get("sign_code") or "")) in target_codes
                for target_id in panel_targets.get(observation_id, [])
            )
            if not direct_code_match and not linked_code_match:
                continue
        if target_kind == "sign_observation" and target_observation_ids:
            if observation_id not in target_observation_ids and not target_observation_ids.intersection(panel_targets.get(observation_id, [])):
                continue
        facts = [item for item in observation_facts if isinstance(item, dict)]
        condition_type = filters.get("condition_type")
        if (requested_fact == "conditions" or requested_fact in CONDITION_PREDICATES) and any(
            item.get("predicate") == "conditions" for item in facts
        ):
            # Public facts bundle time, calendar and numeric restrictions under
            # one predicate. Select the requested member, then retain any
            # actor-specific numeric threshold needed to qualify the answer.
            condition_fact = next(item for item in facts if item.get("predicate") == "conditions")
            entries = [
                item for item in expand_condition_facts([condition_fact])
                if item.get("predicate") in CONDITION_PREDICATES
            ]
            requested_types = (
                {"minimum_speed", "maximum_speed"}
                if condition_type == "speed_limit"
                else {condition_type or requested_fact}
            )
            requested_entries = [
                item for item in entries
                if isinstance(item, dict)
                and item.get("predicate") in requested_types
                and (
                    not filters.get("vehicle_type")
                    or not isinstance(item.get("scope"), dict)
                    or not item["scope"].get("vehicle_types")
                    or any(
                        _vehicle_scope_matches(filters["vehicle_type"], value)
                        for value in (
                            item["scope"]["vehicle_types"]
                            if isinstance(item["scope"]["vehicle_types"], list)
                            else [item["scope"]["vehicle_types"]]
                        )
                    )
                )
            ]
            related_entries = []
            vehicle = filters.get("vehicle_type")
            query_conditions = intent.get("conditions") if isinstance(intent.get("conditions"), dict) else {}
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or entry in requested_entries
                ):
                    continue
                applies_to = entry.get("applies_to")
                if applies_to and vehicle and not _vehicle_scope_matches(vehicle, applies_to):
                    continue
                entry_scope = entry.get("scope") if isinstance(entry.get("scope"), dict) else {}
                scoped_vehicles = entry_scope.get("vehicle_types")
                scoped_vehicles = scoped_vehicles if isinstance(scoped_vehicles, list) else [scoped_vehicles] if scoped_vehicles else []
                if vehicle and scoped_vehicles and not any(
                    _vehicle_scope_matches(vehicle, scoped_vehicle) for scoped_vehicle in scoped_vehicles
                ):
                    continue
                # Other conditions qualify the requested information. For
                # example, a speed limit may only apply during a time window,
                # and an active-time answer may only concern trucks over a
                # payload threshold.
                qualified = dict(entry)
                qualified["condition_state"] = _information_condition_state(entry, query_conditions)
                related_entries.append(qualified)
            if vehicle:
                actor_predicates = {"prohibited_actor", "prohibited_vehicle", "prohibited_subject"}
                for actor_fact in facts:
                    if actor_fact.get("predicate") not in actor_predicates:
                        continue
                    actor_values = actor_fact.get("value")
                    actor_values = actor_values if isinstance(actor_values, list) else [actor_values]
                    if not any(_vehicle_scope_matches(vehicle, value) for value in actor_values):
                        continue
                    actor_conditions = actor_fact.get("conditions")
                    actor_condition = actor_conditions.get(vehicle) if isinstance(actor_conditions, dict) else None
                    if not isinstance(actor_condition, dict):
                        continue
                    required_predicate = actor_condition.get("predicate")
                    related_entries.extend(
                        {
                            **entry,
                            "operator": actor_condition.get("operator"),
                            "applies_to": vehicle,
                        }
                        for entry in entries
                        if isinstance(entry, dict) and entry.get("predicate") == required_predicate
                    )
            if requested_entries:
                def _condition_map(items):
                    mapped = {}
                    for item in items:
                        predicate = item.get("predicate")
                        if not predicate:
                            continue
                        payload = {key: value for key, value in item.items() if key != "predicate"}
                        if predicate not in mapped:
                            mapped[predicate] = payload
                        elif isinstance(mapped[predicate], list):
                            mapped[predicate].append(payload)
                        else:
                            mapped[predicate] = [mapped[predicate], payload]
                    return mapped

                selected = [{
                    "predicate": "conditions",
                    "value": _condition_map(requested_entries + [
                        item for item in related_entries
                        if item not in requested_entries
                    ]),
                }]
            else:
                selected = []
        else:
            selected = [
                item for item in facts
                if item.get("predicate") in predicates
                and _fact_matches_information_filters(item, filters, facts, requested_fact)
            ]
        if not selected:
            continue
        selected_observations.append({
            "id": observation.get("id"),
            "role": observation.get("role"),
            "code": sign_code,
            "facts": [
                {
                    key: fact[key]
                    for key in ("predicate", "value", "unit", "scope", "conditions", "status", "confidence")
                    if key == "value" and key in fact or fact.get(key) is not None
                }
                for fact in selected
            ],
        })

    scene_facts = traffic_facts.get("scene_facts", [])
    selected_scene_facts = [
        {
            key: fact[key]
            for key in ("predicate", "value", "unit", "scope", "conditions", "status", "confidence")
            if key == "value" and key in fact or fact.get(key) is not None
        }
        for fact in scene_facts
        if isinstance(fact, dict)
        and fact.get("predicate") in predicates
        and _fact_matches_information_filters(fact, filters, scene_facts, requested_fact)
    ] if isinstance(scene_facts, list) else []
    selected_ids = {str(item.get("id")) for item in selected_observations if item.get("id")}
    all_relations = relation_payload
    selected_relations = [
        relation for relation in all_relations
        if isinstance(relation, dict)
        and (
            str(relation.get("from") or relation.get("source_observation_id")) in selected_ids
            or str(relation.get("to") or relation.get("target_observation_id")) in selected_ids
        )
    ] if isinstance(all_relations, list) else []
    related_ids = {
        str(relation.get("to") or relation.get("target_observation_id"))
        for relation in selected_relations
        if str(relation.get("from") or relation.get("source_observation_id")) in selected_ids
        and (relation.get("to") or relation.get("target_observation_id"))
    }
    all_observations = observations
    related_signs = [
        {
            "id": observation.get("id"),
            "code": observation.get("code") or (observation.get("identity") or {}).get("sign_code"),
            "name": observation.get("name") or (observation.get("catalog") or {}).get("official_name"),
            "facts": [
                item for item in (observation.get("facts") or [])
                if isinstance(item, dict) and item.get("predicate") == "sign_meaning"
            ],
        }
        for observation in all_observations
        if isinstance(observation, dict) and str(observation.get("id")) in related_ids
    ] if isinstance(all_observations, list) else []
    return {
        "requested_fact": requested_fact or None,
        "filters": filters,
        "signs": selected_observations,
        "scene_facts": selected_scene_facts,
        "relations": selected_relations,
        "related_signs": related_signs,
    }


def _answer_context_for_vlm(symbolic_context: dict[str, Any]) -> dict[str, Any]:
    """Keep full traces for returned/debug data; send only answer-relevant context to the VLM."""
    query_intent = symbolic_context.get("query_intent", {})
    intent_items = query_intent.get("intents", []) if isinstance(query_intent, dict) else []
    intents = [item for item in intent_items if isinstance(item, dict)]
    kinds = {normalize_intent_kind(str(item.get("kind") or "UNKNOWN")) for item in intents}
    context: dict[str, Any] = {"intents": intents}
    if "INFORMATION" in kinds:
        info_intents = [item for item in intents if normalize_intent_kind(str(item.get("kind") or "")) == "INFORMATION"]
        if len(info_intents) == 1:
            context["answer_facts"] = _information_answer_facts(
                symbolic_context.get("facts", {}) if isinstance(symbolic_context.get("facts"), dict) else {},
                info_intents[0],
            )
        else:
            context["answer_facts"] = [
                _information_answer_facts(
                    symbolic_context.get("facts", {}) if isinstance(symbolic_context.get("facts"), dict) else {},
                    item,
                )
                for item in info_intents
            ]
        if any(item.get("requested_fact") == "legal_meaning" or item.get("information_type") == "sign_meaning" for item in info_intents):
            context["sign_meaning_evaluations"] = [
                {
                    key: item[key]
                    for key in ("observation_id", "status", "issues")
                    if item.get(key) is not None
                }
                for item in symbolic_context.get("sign_meaning_evaluations", [])
                if isinstance(item, dict)
            ]
    if "CHECK_PERMISSION" in kinds:
        decision = symbolic_context.get("symbolic_decision", {})
        if isinstance(decision, dict):
            context["symbolic_decision"] = {
                key: decision[key]
                for key in ("decision", "conclusion", "reason", "confidence", "legal_basis")
                if decision.get(key) is not None
            }
        context["intent_evaluations"] = symbolic_context.get("intent_evaluations", [])
        context["rules_fired"] = symbolic_context.get("rules_fired", [])
    return context


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
    query_intent_for_response = symbolic_context.get("query_intent", {})
    response_intents = query_intent_for_response.get("intents", []) if isinstance(query_intent_for_response, dict) else []
    information_only = bool(response_intents) and all(
        isinstance(item, dict) and normalize_intent_kind(str(item.get("kind") or "UNKNOWN")) == "INFORMATION"
        for item in response_intents
    )
    allowed_articles = {
        (article["law_id"], article["article_id"]): article
        for article in ([] if information_only else article_contexts)
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

    if not citations and confidence != "low" and not information_only:
        confidence = "low"

    if not citations and not information_only:
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

    # The rule engine owns permission outcomes. Use its resolved verdict
    # directly so the free-form model cannot contradict or abstain from it.
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
    observations = (
        traffic_facts.get("signs", traffic_facts.get("sign_observations", []))
        if isinstance(traffic_facts, dict) else []
    )
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
    if sign_permission_intent and target_has_sign and outcome in {"ALLOWED", "PROHIBITED"}:
        conclusion = (
            str(symbolic_decision.get("conclusion") or symbolic_decision.get("reason") or "").strip()
            if isinstance(symbolic_decision, dict)
            else ""
        )
        verdict_text = (
            "Có, theo kết luận của rule engine."
            if outcome == "ALLOWED"
            else "Không, theo kết luận của rule engine."
        )
        answer = f"{verdict_text} {conclusion}".strip()
        reasoning = conclusion or f"Rule engine kết luận {outcome}."
        rule_confidence = str(symbolic_decision.get("confidence") or "").lower()
        if rule_confidence in {"high", "medium", "low"}:
            confidence = rule_confidence
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
    ocr_time_panels, ocr_time_panel_bbox_keys = _ocr_text_panel_observations(sample)
    ocr_engine = None
    ocr_unavailable = False

    for sign in sample.get("detected_signs", []):
        if not sign.get("is_chosen"):
            continue

        # OCR text on a time panel is represented as its own supplementary
        # observation below, then linked geometrically to the nearest main sign.
        if ocr_time_panel_bbox_keys:
            sign["ocr_texts"] = [
                item for item in sign.get("ocr_texts", [])
                if json.dumps(item.get("bbox"), separators=(",", ":"), sort_keys=True)
                not in ocr_time_panel_bbox_keys
            ]

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
                    "medium" if detector_confidence >= 0.5 else "low"
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

    traffic_sign_observations.extend(ocr_time_panels)

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
    answer_context = _answer_context_for_vlm(symbolic_context)
    answer_intents = answer_context.get("intents", [])
    intent_kinds = {
        normalize_intent_kind(str(item.get("kind") or "UNKNOWN"))
        for item in answer_intents if isinstance(item, dict)
    }
    information_only = bool(intent_kinds) and intent_kinds == {"INFORMATION"}
    info_intents = [
        item for item in answer_intents
        if isinstance(item, dict) and normalize_intent_kind(str(item.get("kind") or "")) == "INFORMATION"
    ]
    structured_information = bool(info_intents) and all(
        item.get("requested_fact") in _INFO_FACT_PREDICATES
        or item.get("information_type") == "sign_meaning"
        for item in info_intents
    )
    choices = sample.get("choices")
    user_context = {
        "question": sample.get("question", ""),
        "choices_optional_reference": choices if choices else None,
    }
    if include_full_image_ocr and not structured_information:
        user_context.update({
            "full_image_ocr_status": sample.get("full_image_ocr_status", "unavailable"),
            "full_image_ocr": sample.get("full_image_ocr", []),
        })
    if not trafficlawqa_rag_compatibility:
        user_context.update({
            "query_intent": {"intents": answer_intents},
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
        if not information_only and article.get("context_source") in {"retrieval", "detector"}
    ]
    related_articles = [
        article
        for article in article_contexts
        if not information_only and article.get("context_source") not in {"retrieval", "detector"}
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
        for item in ([] if information_only else _primary_evidence_by_sign(visual_evidence))
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
            f"{json.dumps(_without_verification_status(answer_context), ensure_ascii=False)}\n\n"
        )
    if not information_only:
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
                else _answer_system_prompt(symbolic_context)
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
    if not information_only:
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
