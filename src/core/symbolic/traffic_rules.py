from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.core.symbolic.fact_schema import (
    ContextFact,
    SignObservation,
    TrafficFacts,
)
from src.core.symbolic.intent_schema import Intent, QueryIntent
from src.core.symbolic.rule_schema import RuleCondition, RuleEffect, RuleEvaluation, TrafficRule
from src.core.symbolic.sign_normalizer import normalize_sign_code, sign_family
from src.core.symbolic.sign_fact_resolver import observation_context_facts
from src.core.symbolic.question_intent import extract_vehicle
from src.core.symbolic.vehicle_taxonomy import (
    ACTOR_CLASS_CANDIDATES,
    VEHICLES,
    vehicle_is_covered,
)
PERMISSION_INTENTS = {"CHECK_PERMISSION"}
RULEBOOK_PATH = Path(__file__).parent / "catalogs" / "qcvn_41_2024_rulebook_draft.json"
RULE_DRAFT_REGISTRY_PATH = Path(__file__).parent / "catalogs" / "qcvn_41_2024_rule_drafts.json"
SCOPE_SUPPLEMENTARY_FAMILIES = {
    "S.501", "S.502", "S.503", "S.504", "S.505", "S.506", "S.508", "S.509",
}

# Facts this project can currently extract from the sign, OCR, or explicit
# query wording. Predicates that require locating the vehicle in a lane or
# judging road safety are intentionally excluded. Other scene-assessment
# predicates are evaluated when supplied, but their absence does not block a
# rule.
SUPPORTED_RULE_CONDITION_PREDICATES = frozenset({
    # Actor and vehicle classes supplied by the question or a rule scope.
    "actor_vehicle_category", "actor_is_motor_vehicle", "actor_has_priority_right",
    "actor_eligibility_for_i446_reserved_parking_confirmed",
    "hazardous_cargo_status", "vehicle_trailer_configuration", "vehicle_scope_condition",
    "overtaken_vehicle_class", "overtaken_vehicle_is_motor_vehicle",
    "pedestrian_location", "wheelchair_for_person_with_disability_confirmed",
    "vehicle_class_parking_conditions_resolved",
    # Sign-provided speed, distance, dimension, weight, and passenger limits.
    "speed_limit_value", "minimum_speed_limit_kmh", "slope_percent", "general_max_speed_kmh",
    "applicable_speed_limit_value", "applicable_statutory_max_speed_kmh",
    "total_weight_limit_t", "axle_load_limit_t", "payload_limit_t", "height_limit_m",
    "width_limit_m", "length_limit_m", "combination_length_limit_m",
    "minimum_following_distance_m", "safe_height_limit_m", "passenger_seat_limit",
    "passenger_seat_limit_present", "passenger_seat_limit_relation", "payload_limit_present",
    "payload_limit_relation", "certified_payload_capacity_t", "vehicle_payload_t",
    "vehicle_gross_weight_t", "vehicle_axle_group_loads_t", "vehicle_axle_load_t",
    "vehicle_permitted_gross_weight_t", "vehicle_s505b_row_index",
    "vehicle_s505b_row_match_confirmed", "s505b_row_gross_weight_limits_by_index",
    "vehicle_seat_count", "vehicle_width_m", "vehicle_height_m", "vehicle_length_m",
    "vehicle_combination_length_m", "vehicle_manufacturer_max_speed_kmh",
    # Date/time and sign-depicted direction or explicit route wording.
    "calendar_parity", "sign_applicability_calendar_parity", "time_window", "supplementary_time_condition",
    "supplementary_calendar_parity", "intended_obstacle_passing_side",
    "intended_roundabout_route_follows_arrow", "vehicle_travel_direction_matches_i407_arrow",
    "i416_detour_route_matches_diagram", "vehicle_class_shown_on_i417_resolved",
    "requested_route_matches_i414_direction", "requested_route_matches_i415_arrow",
    "requested_route_matches_i417_diagram_for_vehicle_class", "requested_turn_matches_i413_diagram",
    "requested_path_matches_i418_diagram", "requested_u_turn_matches_i410_diagram",
    "pedestrian_direction_matches_i424_diagram", "i414_road_number_and_destination_text_resolved",
    "direction_scope_condition", "distance_scope_condition",
    "supplementary_direction_scope_resolved", "priority_road_scope_modifier_resolved",
    "i401_priority_road_direction_resolved",
    # Presence and values of identified supplementary or ending signs.
    "linked_s505b_panel_present", "linked_s505c_panel_present", "linked_s506b_panel_present",
    "linked_s508_panel_present", "linked_s508_temporal_scope_active", "active_restriction_sign_codes",
    "s505a_vehicle_scope_resolved", "s505a_vehicle_classes",
    "i434a_bus_service_match_confirmed", "i434b_loading_or_unloading_purpose_confirmed",
    "sh6_exception_scope_resolved", "sh6_exception_actor_classes",
    "sh6_exception_applies_to_actor_resolved",
})
UNOBSERVABLE_DECISION_SCOPE_PREDICATES = frozenset({
    "bus_lane_marking", "current_lane_scope_confirmed", "current_vehicle_lane_scope_resolved",
    "r411_current_lane_resolved", "r411_lane_arrow_mapping_resolved",
    "r411_directional_road_marking_consistent", "r411_requested_route_matches_lane_arrow",
    "r415_lane_count_per_direction_resolved", "r415_vehicle_symbols_mapped_to_each_lane",
    "requested_vehicle_and_lane_resolved", "other_vehicle_class_lane_scope_resolved",
})
def _is_legally_verified(rule: TrafficRule) -> bool:
    return (
        rule.verification_status == "verified"
        and bool(rule.legal_basis)
        and all(ref.get("verification_status") == "verified" for ref in rule.legal_basis)
    )


def _rule_execution_allowed(rule: TrafficRule) -> bool:
    # Draft rules with a recorded legal basis are executable; verification
    # status remains provenance metadata and does not gate matching/decisions.
    return bool(rule.legal_basis)


def _actor_scope_match(actor: str, scopes: list[str]) -> str:
    """Classify a compound actor as all, some, or none of the given rule scopes."""
    candidates = ACTOR_CLASS_CANDIDATES.get(actor)
    if not candidates:
        return "all" if any(vehicle_is_covered(actor, scope) for scope in scopes) else "none"
    matches = [any(vehicle_is_covered(candidate, scope) for scope in scopes) for candidate in candidates]
    if all(matches):
        return "all"
    if any(matches):
        return "some"
    return "none"
SUPPLEMENTARY_RULE_SCOPE_PREDICATES = {
    "S.501": {"effect_length_m", "s501_effect_length_resolved"},
    "S.502": {"distance_to_object_m", "s502_distance_scope_resolved"},
    "S.503": {"s503_direction_scope_resolved", "supplementary_direction_scope_resolved"},
    "S.504": {"s504_affected_lane_resolved", "supplementary_lane_scope_resolved"},
    "S.505a": {"s505a_vehicle_scope_resolved", "vehicle_scope_condition"},
    "S.506a": {"i401_priority_road_direction_resolved", "priority_road_scope_modifier_resolved"},
    "S.509a": {"safe_height_limit_m", "s509a_height_limit_resolved"},
    "S.509b": {"s509b_parking_scope_resolved", "supplementary_parking_scope_resolved"},
    "S.G.11": {"sg11_lane_direction_mapping_resolved"},
    "S.G.12": {"sg12_non_traversable_lane_resolved"},
    "S.H.6": {"sh6_exception_scope_resolved", "sh6_exception_applies_to_actor_resolved"},
}


def _is_scope_supplementary(observation: SignObservation) -> bool:
    family = sign_family(observation.sign_code)
    return (
        family in SCOPE_SUPPLEMENTARY_FAMILIES
        or (observation.role == "supplementary" and family not in {"S.507", "S.510"})
    )


def load_rulebook(path: str | Path = RULEBOOK_PATH) -> tuple[dict[str, Any], tuple[TrafficRule, ...]]:
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    rules = [TrafficRule(**item) for item in payload.get("rules", []) if isinstance(item, dict)]
    draft_metadata: dict[str, Any] = {}
    if Path(path).resolve() == RULEBOOK_PATH.resolve() and RULE_DRAFT_REGISTRY_PATH.exists():
        from src.core.symbolic.sign_catalog import load_default_sign_catalog

        with RULE_DRAFT_REGISTRY_PATH.open(encoding="utf-8") as stream:
            registry = json.load(stream)
        catalog = load_default_sign_catalog()
        registry_rules: list[TrafficRule] = []
        for draft in registry.get("rules", []):
            if not isinstance(draft, dict):
                continue
            code = str(draft.get("code") or "").strip()
            source_entry = catalog.lookup(code)
            if source_entry is None or not source_entry.legal_basis:
                raise ValueError(f"Rule draft {draft.get('rule_id')} lacks official catalog provenance for {code}")
            rule_payload = {
                key: value for key, value in draft.items()
                if key not in {"code", "additional_legal_basis"}
            }
            rule_payload["trigger_sign_codes"] = [code]
            rule_payload["legal_basis"] = [
                *[dict(ref) for ref in source_entry.legal_basis],
                *[dict(ref) for ref in draft.get("additional_legal_basis", []) if isinstance(ref, dict)],
            ]
            rule_payload["verification_status"] = "unverified"
            registry_rules.append(TrafficRule(**rule_payload))
        # The source-first registry is the maintained override layer: matching
        # rule IDs replace their older base-rule definitions rather than firing
        # twice with potentially different scopes or conditions.
        merged_by_id = {rule.rule_id: rule for rule in rules}
        for rule in registry_rules:
            merged_by_id[rule.rule_id] = rule
        rules = list(merged_by_id.values())
        draft_metadata = {
            "rule_draft_registry": registry.get("registry_version"),
            "rule_draft_count": len(registry.get("rules", [])),
            "rule_draft_review_status": registry.get("review_status", "unreviewed_draft"),
        }
    rules = tuple(rules)
    ids = [rule.rule_id for rule in rules]
    if not rules or any(not rule_id for rule_id in ids):
        raise ValueError("Rulebook must contain rules with non-empty rule_id values")
    if len(ids) != len(set(ids)):
        raise ValueError("Rulebook contains duplicate rule_id values")
    metadata = {key: value for key, value in payload.items() if key != "rules"}
    metadata.update(draft_metadata)
    return (metadata, rules)


TRAFFIC_RULEBOOK_METADATA, TRAFFIC_RULES = load_rulebook()


def _supplementary_context(facts: TrafficFacts, main_sign: SignObservation) -> list[ContextFact]:
    observations = {item.observation_id: item for item in facts.sign_observations}
    linked = [
        relation for relation in facts.sign_relations
        if relation.target_observation_id == main_sign.observation_id
        and relation.relation_type in {"supplements", "supplementary_for"}
        and relation.confidence == "high"
    ]
    result: list[ContextFact] = []
    for relation in linked:
        supplementary = observations.get(relation.source_observation_id)
        if (
            supplementary is None
            or (
                supplementary.source != "paddleocr_text_panel"
                and (
                    supplementary.identity_status != "identified"
                    or not _is_scope_supplementary(supplementary)
                )
            )
        ):
            continue
        result.extend(observation_context_facts(
            supplementary,
            subject_id=main_sign.observation_id,
            relation_id=relation.relation_id,
            source="supplementary_sign_interpretation",
        ))
    return result


def _s505c_limit_map_facts(context: list[ContextFact], main_sign: SignObservation) -> list[ContextFact]:
    """Group only OCR-resolved S.505c axle limits; absent groups stay absent."""
    source_facts = [
        fact for fact in context
        if fact.metadata.get("subject_observation_id") == main_sign.observation_id
        and fact.metadata.get("supplementary_sign_code") == "S.505c"
        and fact.predicate in {
            "single_axle_load_limit_t",
            "tandem_axle_load_limit_t",
            "triple_axle_load_limit_t",
        }
    ]
    if not source_facts:
        return []
    predicate_to_group = {
        "single_axle_load_limit_t": "single_axle",
        "tandem_axle_load_limit_t": "tandem_axle",
        "triple_axle_load_limit_t": "triple_axle",
    }
    values_by_group: dict[str, set[float]] = {}
    for fact in source_facts:
        if fact.status not in {"observed", "derived"} or fact.confidence not in {"medium", "high"}:
            continue
        try:
            values_by_group.setdefault(predicate_to_group[fact.predicate], set()).add(float(fact.value))
        except (TypeError, ValueError):
            continue
    conflicting = any(len(values) > 1 for values in values_by_group.values())
    limits = {
        group: next(iter(values))
        for group, values in values_by_group.items()
        if len(values) == 1
    }
    if conflicting:
        limits = {}
    return [ContextFact(
        fact_id=f"{main_sign.observation_id}:s505c-axle-load-limit-map",
        predicate="axle_load_limits_by_group",
        value=limits if limits else None,
        status="ambiguous" if conflicting else "derived" if limits else "unknown",
        source="linked_s505c_strict_ocr",
        confidence=(
            "low" if conflicting else
            "high" if all(fact.confidence == "high" for fact in source_facts) else
            "medium" if limits else "low"
        ),
        raw_text=" | ".join(fact.raw_text or "" for fact in source_facts),
        evidence=[evidence for fact in source_facts for evidence in fact.evidence],
        metadata={
            "subject_observation_id": main_sign.observation_id,
            "source_observation_ids": sorted({
                str(fact.metadata.get("source_observation_id")) for fact in source_facts
            }),
            "observed_axle_groups": sorted(limits),
            "extraction": "grouped_strict_s505c_ocr",
        },
    )]


def _s505b_row_limit_map_facts(context: list[ContextFact], main_sign: SignObservation) -> list[ContextFact]:
    """Group OCR limits only when a layout extractor tied each value to a row."""
    source_facts = [
        fact for fact in context
        if fact.metadata.get("subject_observation_id") == main_sign.observation_id
        and fact.metadata.get("supplementary_sign_code") == "S.505b"
        and fact.predicate == "s505b_row_gross_weight_limit_t"
    ]
    if not source_facts:
        return []
    values_by_row: dict[int, set[float]] = {}
    for fact in source_facts:
        row_index = fact.metadata.get("table_row_index")
        if type(row_index) is not int or row_index not in {1, 2, 3}:
            continue
        if fact.status not in {"observed", "derived"} or fact.confidence not in {"medium", "high"}:
            continue
        try:
            values_by_row.setdefault(row_index, set()).add(float(fact.value))
        except (TypeError, ValueError):
            continue
    limits = {
        row_index: next(iter(values))
        for row_index, values in values_by_row.items()
        if len(values) == 1
    }
    if not limits:
        return []
    return [ContextFact(
        fact_id=f"{main_sign.observation_id}:s505b-row-gross-weight-limit-map",
        predicate="s505b_row_gross_weight_limits_by_index",
        value=limits,
        status="derived",
        source="linked_s505b_strict_ocr_with_row_layout",
        confidence=(
            "high" if all(fact.confidence == "high" for fact in source_facts)
            else "medium"
        ),
        raw_text=" | ".join(fact.raw_text or "" for fact in source_facts),
        evidence=[evidence for fact in source_facts for evidence in fact.evidence],
        metadata={
            "subject_observation_id": main_sign.observation_id,
            "source_observation_ids": sorted({
                str(fact.metadata.get("source_observation_id")) for fact in source_facts
            }),
            "observed_table_row_indices": sorted(limits),
            "extraction": "strict_s505b_ocr_row_map",
        },
    )]


def _supplementary_linkage_unresolved(facts: TrafficFacts, main_sign: SignObservation) -> bool:
    """Fail closed when a detected supplementary sign cannot be confidently attached."""
    observations = {item.observation_id: item for item in facts.sign_observations}
    for supplementary in facts.sign_observations:
        if not _is_scope_supplementary(supplementary):
            continue
        is_ocr_text_panel = supplementary.source == "paddleocr_text_panel"
        if supplementary.identity_status != "identified" and not is_ocr_text_panel:
            return True
        if (
            sign_family(supplementary.sign_code) == "S.505"
            and supplementary.confidence not in {"medium", "high"}
        ):
            return True
        relations = [
            relation for relation in facts.sign_relations
            if relation.relation_type in {"supplements", "supplementary_for"}
            and relation.source_observation_id == supplementary.observation_id
        ]
        if not relations:
            return True
        if is_ocr_text_panel and not any(
            relation.confidence == "high" for relation in relations
        ):
            return True
        attached_here = [
            relation for relation in relations
            if relation.target_observation_id == main_sign.observation_id
        ]
        if attached_here and any(relation.confidence != "high" for relation in attached_here):
            return True
        if not attached_here and not any(
            relation.confidence == "high" and relation.target_observation_id in observations
            for relation in relations
        ):
            return True
    return False


def _unmodeled_supplementary_scope_panels(
    facts: TrafficFacts,
    main_sign: SignObservation,
    rule: TrafficRule,
) -> list[str]:
    """Return linked scope modifiers that the candidate rule does not evaluate."""
    observations = {item.observation_id: item for item in facts.sign_observations}
    rule_predicates = {
        condition.predicate for condition in rule.conditions if condition.required
    }
    unmodeled: set[str] = set()
    for relation in facts.sign_relations:
        if (
            relation.target_observation_id != main_sign.observation_id
            or relation.relation_type not in {"supplements", "supplementary_for"}
            or relation.confidence != "high"
        ):
            continue
        panel = observations.get(relation.source_observation_id)
        if panel is None or panel.identity_status != "identified":
            continue
        code = panel.sign_code or ""
        family = sign_family(code)
        # These have dedicated evaluators/guards already (time and bridge-load
        # panels; S.506b's priority-road branch). Do not duplicate their traces.
        if family == "S.508" or code in {"S.505b", "S.505c", "S.506b"}:
            continue
        required_predicates = SUPPLEMENTARY_RULE_SCOPE_PREDICATES.get(code) or (
            SUPPLEMENTARY_RULE_SCOPE_PREDICATES.get(family)
        )
        if required_predicates and not (rule_predicates & required_predicates):
            unmodeled.add(code)
    return sorted(unmodeled)


def _linked_s508_scope_facts(
    facts: TrafficFacts,
    main_sign: SignObservation,
    context: list[ContextFact],
) -> list[ContextFact]:
    """Summarize linked S.508 presence and whether its observed time/day applies."""
    panels = [
        item for item in facts.sign_observations
        if item.role == "supplementary" and sign_family(item.sign_code) == "S.508"
    ]
    high_links = [
        relation for relation in facts.sign_relations
        if relation.relation_type in {"supplements", "supplementary_for"}
        and relation.confidence == "high"
    ]
    linked_here = [
        relation for relation in high_links
        if relation.target_observation_id == main_sign.observation_id
        and any(panel.observation_id == relation.source_observation_id for panel in panels)
    ]
    unresolved_panels = [
        panel for panel in panels
        if panel.identity_status != "identified"
        or not any(link.source_observation_id == panel.observation_id for link in high_links)
    ]

    presence: bool | None = None
    if linked_here:
        presence = True
    elif not panels:
        presence = False
    elif facts.sign_inventory_status == "complete" and not unresolved_panels:
        presence = False

    result: list[ContextFact] = []
    if presence is None:
        return result
    result.append(ContextFact(
        fact_id=f"{main_sign.observation_id}:linked-s508-present",
        predicate="linked_s508_panel_present",
        value=presence,
        status="derived",
        source="complete_sign_inventory_and_high_confidence_relations",
        confidence="high",
        metadata={"subject_observation_id": main_sign.observation_id},
    ))
    if not presence:
        return result

    observations = {item.observation_id: item for item in facts.sign_observations}
    active_conditions: list[tuple[str, object]] = []
    linked_panel_ids = {relation.source_observation_id for relation in linked_here}
    for item in context:
        panel_id = item.metadata.get("source_observation_id")
        if panel_id not in linked_panel_ids or item.source != "supplementary_sign_ocr":
            continue
        if item.predicate in {"supplementary_time_window", "supplementary_calendar_parity"}:
            if item.status not in {"observed", "derived"} or item.confidence not in {"medium", "high"}:
                return result
            active_conditions.append((item.predicate, item.value))

    # A visible S.508 with illegible/unrecognized text is still a live modifier.
    for panel_id in linked_panel_ids:
        panel = observations.get(panel_id)
        if panel is None or not panel.ocr_texts:
            return result
        extracted = [
            item for item in context
            if item.metadata.get("source_observation_id") == panel_id
            and item.source == "supplementary_sign_ocr"
            and item.predicate in {"supplementary_time_window", "supplementary_calendar_parity"}
        ]
        if not extracted:
            return result
        for index, _ in enumerate(panel.ocr_texts):
            if not any(item.fact_id.startswith(f"{panel_id}:supplement:{index}:") for item in extracted):
                return result

    observed_at, observed_fact = _value_for("observed_at", context)
    if observed_fact is None or not active_conditions:
        return result
    try:
        instant = datetime.fromisoformat(str(observed_at))
    except ValueError:
        return result

    applies = True
    minute = instant.hour * 60 + instant.minute
    parity = "odd_days" if instant.day % 2 else "even_days"
    for predicate, value in active_conditions:
        if predicate == "supplementary_calendar_parity":
            applies = applies and value == parity
        elif predicate == "supplementary_time_window":
            if not isinstance(value, dict):
                return result
            try:
                start, end = int(value["start_minute"]), int(value["end_minute"])
            except (KeyError, TypeError, ValueError):
                return result
            if not (0 <= start < 1440 and 0 <= end < 1440) or minute in {start, end}:
                return result
            in_window = start < minute < end if start < end else minute > start or minute < end
            applies = applies and in_window

    result.append(ContextFact(
        fact_id=f"{main_sign.observation_id}:linked-s508-active",
        predicate="linked_s508_temporal_scope_active",
        value=applies,
        status="derived",
        source="linked_s508_ocr_and_system_time",
        confidence="high",
        metadata={
            "subject_observation_id": main_sign.observation_id,
            "supplementary_observation_ids": sorted(linked_panel_ids),
        },
    ))
    return result


def _linked_s506b_scope_facts(facts: TrafficFacts, main_sign: SignObservation) -> list[ContextFact]:
    """Resolve S.506b only when its identity and attachment to this sign are explicit."""
    observations = {item.observation_id: item for item in facts.sign_observations}
    high_links = [
        relation for relation in facts.sign_relations
        if relation.relation_type in {"supplements", "supplementary_for"}
        and relation.confidence == "high"
    ]
    linked_here = [
        relation for relation in high_links
        if relation.target_observation_id == main_sign.observation_id
        and (panel := observations.get(relation.source_observation_id)) is not None
        and panel.sign_code == "S.506b"
        and panel.identity_status == "identified"
        and panel.confidence in {"medium", "high"}
    ]

    present: bool | None = True if linked_here else None
    if present is None and not any(item.sign_code == "S.506b" for item in facts.sign_observations):
        present = False
    if present is None and facts.sign_inventory_status == "complete":
        unresolved_panel = False
        for panel in facts.sign_observations:
            if not _is_scope_supplementary(panel):
                continue
            panel_links = [
                relation for relation in facts.sign_relations
                if relation.source_observation_id == panel.observation_id
                and relation.relation_type in {"supplements", "supplementary_for"}
            ]
            if (
                panel.identity_status != "identified"
                or panel.confidence not in {"medium", "high"}
                or not any(link.confidence == "high" for link in panel_links)
            ):
                unresolved_panel = True
                break
        if not unresolved_panel:
            present = False

    if present is None:
        return []
    source = (
        "identified_supplementary_sign_and_high_confidence_relation"
        if present else "complete_sign_inventory_and_resolved_supplementary_relations"
    )
    return [ContextFact(
        fact_id=f"{main_sign.observation_id}:linked-s506b-present",
        predicate="linked_s506b_panel_present",
        value=present,
        status="derived",
        source=source,
        confidence=(
            observations[linked_here[0].source_observation_id].confidence if linked_here else "high"
        ),
        evidence=main_sign.evidence,
        metadata={
            "subject_observation_id": main_sign.observation_id,
            "supplementary_observation_ids": [item.source_observation_id for item in linked_here],
        },
    )]


def _linked_s505_bridge_scope_facts(
    facts: TrafficFacts,
    main_sign: SignObservation,
) -> list[ContextFact]:
    """Resolve whether S.505b/c bridge-weight panels modify a P.106a sign."""
    if main_sign.sign_code != "P.106a":
        return []
    observations = {item.observation_id: item for item in facts.sign_observations}
    result: list[ContextFact] = []
    states: dict[str, bool | None] = {}
    for code in ("S.505b", "S.505c"):
        panels = [
            item for item in facts.sign_observations
            if item.role == "supplementary" and item.sign_code == code
        ]
        panel_ids = {item.observation_id for item in panels}
        links = [
            relation for relation in facts.sign_relations
            if relation.relation_type in {"supplements", "supplementary_for"}
            and relation.source_observation_id in panel_ids
        ]
        linked_here = [
            relation for relation in links
            if relation.target_observation_id == main_sign.observation_id
            and relation.confidence == "high"
            and (panel := observations.get(relation.source_observation_id)) is not None
            and panel.identity_status == "identified"
            and panel.confidence in {"medium", "high"}
        ]
        unresolved_panel = any(
            panel.identity_status != "identified"
            or panel.confidence not in {"medium", "high"}
            or not any(
                relation.source_observation_id == panel.observation_id
                and relation.confidence == "high"
                for relation in links
            )
            for panel in panels
        )
        present: bool | None = True if linked_here else (
            False if not panels or (facts.sign_inventory_status == "complete" and not unresolved_panel)
            else None
        )
        states[code] = present
        if present is not None:
            result.append(ContextFact(
                fact_id=f"{main_sign.observation_id}:linked-{code.lower().replace('.', '')}-present",
                predicate=f"linked_{code.lower().replace('.', '')}_panel_present",
                value=present,
                status="derived",
                source=(
                    "identified_panel_and_high_confidence_relation"
                    if present else "complete_inventory_and_resolved_panel_relations"
                ),
                confidence="high",
                evidence=main_sign.evidence,
                metadata={
                    "subject_observation_id": main_sign.observation_id,
                    "supplementary_observation_ids": [
                        relation.source_observation_id for relation in linked_here
                    ],
                },
            ))
    combined: bool | None = (
        True if any(state is True for state in states.values())
        else False if all(state is False for state in states.values())
        else None
    )
    if combined is not None:
        result.append(ContextFact(
            fact_id=f"{main_sign.observation_id}:linked-s505-bridge-panels-present",
            predicate="linked_s505_bridge_panels_present",
            value=combined,
            status="derived",
            source="resolved_s505b_c_panel_presence",
            confidence="high",
            evidence=main_sign.evidence,
            metadata={"subject_observation_id": main_sign.observation_id},
        ))
    return result


def _context_facts(
    facts: TrafficFacts,
    query_facts: Iterable[ContextFact],
    main_sign: SignObservation | None = None,
) -> list[ContextFact]:
    main_id = main_sign.observation_id if main_sign else None
    explicit_query_facts = list(query_facts)
    context = [
        item for item in facts.context_facts
        if not item.metadata.get("subject_observation_id")
        or item.metadata.get("subject_observation_id") == main_id
    ]
    context.extend(explicit_query_facts)
    if main_sign:
        context.extend(observation_context_facts(main_sign, subject_id=main_sign.observation_id))
        context.extend(_supplementary_context(facts, main_sign))
        context.extend(_s505b_row_limit_map_facts(context, main_sign))
        context.extend(_s505c_limit_map_facts(context, main_sign))
    has_parity = any(item.predicate == "calendar_parity" for item in context)
    if not has_parity:
        observed_at = next(
            (item for item in reversed(context)
             if item.predicate == "observed_at" and item.status == "resolved"),
            None,
        )
        if observed_at and isinstance(observed_at.value, str):
            try:
                day = datetime.fromisoformat(observed_at.value).day
                context.append(ContextFact(
                    fact_id=f"{observed_at.fact_id}:calendar-parity",
                    predicate="calendar_parity",
                    value="odd_days" if day % 2 else "even_days",
                    status="derived",
                    source="derived_from_system_clock" if observed_at.source == "system_clock" else "derived",
                    raw_text=observed_at.raw_text,
                    confidence=observed_at.confidence,
                    evidence=observed_at.evidence,
                    metadata={"derived_from": observed_at.fact_id},
                ))
            except ValueError:
                pass
    if main_sign:
        context.extend(_linked_s508_scope_facts(facts, main_sign, context))
        context.extend(_linked_s505_bridge_scope_facts(facts, main_sign))
        s506b_facts = _linked_s506b_scope_facts(facts, main_sign)
        context.extend(s506b_facts)
        if any(item.predicate == "linked_s506b_panel_present" and item.value is False for item in s506b_facts):
            context.append(ContextFact(
                fact_id=f"{main_sign.observation_id}:priority-road-scope-modifier-resolved",
                predicate="priority_road_scope_modifier_resolved",
                value=True,
                status="derived",
                source="complete_inventory_confirms_no_linked_s506b",
                confidence="high",
                metadata={"subject_observation_id": main_sign.observation_id},
            ))
    # Explicit values in the question take precedence over values inferred
    # from the sign image (for example, asking about an odd date on a sign
    # that restricts parking on odd days).
    context.extend(explicit_query_facts)
    return context


def _derive_sh6_actor_match(context: list[ContextFact], actor: str, intent_id: str) -> ContextFact | None:
    """Compare a resolved S.H.6 group only against an explicit actor class."""
    if not actor or actor == "unknown":
        return None
    resolved, resolved_fact = _value_for("sh6_exception_scope_resolved", context)
    groups, groups_fact = _value_for("sh6_exception_actor_classes", context)
    if resolved is not True or resolved_fact is None or groups_fact is None:
        return None
    if resolved_fact.status not in {"observed", "provided", "resolved"}:
        return None
    if groups_fact.status not in {"observed", "provided", "resolved"}:
        return None
    if groups_fact.confidence not in {"high", "medium"} or not isinstance(groups, list) or not groups:
        return None
    if any(not isinstance(group, str) or group == "unknown" for group in groups):
        return None
    match_state = _actor_scope_match(actor, groups)
    if match_state == "some":
        return None
    matches = match_state == "all"
    return ContextFact(
        fact_id=f"{intent_id}:sh6-exception-actor-match",
        predicate="sh6_exception_applies_to_actor",
        value=matches,
        status="resolved",
        source="explicit_actor_and_resolved_sh6_group_comparison",
        confidence="high",
        evidence=[*resolved_fact.evidence, *groups_fact.evidence],
        metadata={
            "exception_group_fact_id": groups_fact.fact_id,
            "actor_class": actor,
            "matched_exception_classes": list(groups),
        },
    )


def _derive_s505a_actor_match(context: list[ContextFact], actor: str, intent_id: str) -> ContextFact | None:
    """Resolve whether an explicit actor is in the vehicle class shown on S.505a."""
    if not actor or actor == "unknown":
        return None
    resolved, resolved_fact = _value_for("s505a_vehicle_scope_resolved", context)
    classes, classes_fact = _value_for("s505a_vehicle_classes", context)
    if resolved is not True or resolved_fact is None or classes_fact is None:
        return None
    if classes_fact.status not in {"observed", "provided", "resolved", "derived"}:
        return None
    if classes_fact.confidence not in {"high", "medium"} or not isinstance(classes, list) or not classes:
        return None
    if any(not isinstance(item, str) or item not in VEHICLES for item in classes):
        return None
    match_state = _actor_scope_match(actor, classes)
    if match_state == "some":
        return None
    matches = match_state == "all"
    return ContextFact(
        fact_id=f"{intent_id}:s505a-actor-match",
        predicate="s505a_vehicle_scope_applies_to_actor",
        value=matches,
        status="resolved",
        source="explicit_actor_and_visually_resolved_s505a_vehicle_classes",
        confidence="high",
        evidence=[*resolved_fact.evidence, *classes_fact.evidence],
        metadata={
            "vehicle_scope_fact_id": classes_fact.fact_id,
            "actor_class": actor,
            "depicted_vehicle_classes": list(classes),
            "legal_basis": "QCVN 41:2024/BGTVT Appendix F.5, Figure F.5",
        },
    )


def _value_for(predicate: str, context: list[ContextFact]) -> tuple[Any, ContextFact | None]:
    for fact in reversed(context):
        if fact.predicate != predicate:
            continue
        if fact.status in {"unknown", "needs_resolution", "ambiguous", "not_observed"}:
            # A later unresolved observation must shadow any earlier value for
            # the same predicate; otherwise a conflicting OCR result could be
            # silently ignored in favor of stale/model-derived data.
            return None, None
        if fact.confidence not in {"medium", "high"}:
            return None, None
        return fact.value, fact
    return None, None


def _condition_result(condition: RuleCondition, context: list[ContextFact]) -> tuple[str, Any, str | None]:
    value, fact = _value_for(condition.predicate, context)
    if fact is None:
        # Optional conditions may be irrelevant for fixed-outcome rules, but
        # still need to report absence so a conditional threshold rule cannot
        # mistake an unknown comparison for permission.
        return "missing", None, None
    if condition.operator == "exists":
        return ("satisfied" if value not in (None, "", []) else "missing", value, fact.fact_id)
    if condition.operator == "equals":
        if condition.predicate in {"calendar_parity", "sign_applicability_calendar_parity"}:
            aliases = {"odd": "odd_days", "even": "even_days"}
            value = aliases.get(value, value)
            expected = aliases.get(condition.expected_value, condition.expected_value)
            return ("satisfied" if value == expected else "mismatch", value, fact.fact_id)
        return ("satisfied" if value == condition.expected_value else "mismatch", value, fact.fact_id)
    if condition.operator == "equals_fact":
        other_value, other_fact = _value_for(str(condition.expected_value or ""), context)
        if other_fact is None:
            return "missing", value, fact.fact_id
        return ("satisfied" if value == other_value else "mismatch", value, fact.fact_id)
    if condition.operator in {"lte_fact", "gte_fact", "gt_fact", "lt_fact"}:
        other_value, other_fact = _value_for(str(condition.expected_value or ""), context)
        if other_fact is None:
            return "missing", value, fact.fact_id
        try:
            left, right = float(value), float(other_value)
        except (TypeError, ValueError):
            return "missing", value, fact.fact_id
        matched = {
            "lte_fact": left <= right,
            "gte_fact": left >= right,
            "gt_fact": left > right,
            "lt_fact": left < right,
        }[condition.operator]
        return ("satisfied" if matched else "mismatch", value, fact.fact_id)
    if condition.operator == "axle_group_loads_within_limits":
        limit_map, limit_fact = _value_for(str(condition.expected_value or ""), context)
        if limit_fact is None or not isinstance(value, dict) or not isinstance(limit_map, dict):
            return "missing", value, fact.fact_id
        if not value:
            return "missing", value, fact.fact_id
        comparisons: list[bool] = []
        for axle_group, loads in value.items():
            if axle_group not in {"single_axle", "tandem_axle", "triple_axle"}:
                return "missing", value, fact.fact_id
            if axle_group not in limit_map:
                return "missing", value, fact.fact_id
            load_values = loads if isinstance(loads, (list, tuple)) else [loads]
            if not load_values:
                return "missing", value, fact.fact_id
            if isinstance(limit_map[axle_group], bool) or any(isinstance(load, bool) for load in load_values):
                return "missing", value, fact.fact_id
            try:
                limit = float(limit_map[axle_group])
                numeric_loads = [float(load) for load in load_values]
            except (TypeError, ValueError):
                return "missing", value, fact.fact_id
            if limit < 0 or not math.isfinite(limit) or any(
                load < 0 or not math.isfinite(load) for load in numeric_loads
            ):
                return "missing", value, fact.fact_id
            comparisons.extend(load <= limit for load in numeric_loads)
        if not comparisons:
            return "missing", value, fact.fact_id
        return ("satisfied" if all(comparisons) else "mismatch", value, fact.fact_id)
    if condition.operator == "s505b_gross_weight_within_selected_row":
        spec = condition.expected_value if isinstance(condition.expected_value, dict) else {}
        limits, limits_fact = _value_for(
            str(spec.get("limit_map_predicate") or "s505b_row_gross_weight_limits_by_index"), context
        )
        row_index, row_fact = _value_for(
            str(spec.get("row_index_predicate") or "vehicle_s505b_row_index"), context
        )
        if limits_fact is None or row_fact is None or not isinstance(limits, dict):
            return "missing", value, fact.fact_id
        if isinstance(row_index, bool) or type(row_index) is not int or row_index not in {1, 2, 3}:
            return "missing", value, fact.fact_id
        limit_value = limits.get(row_index, limits.get(str(row_index)))
        if limit_value is None or isinstance(value, bool) or isinstance(limit_value, bool):
            return "missing", value, fact.fact_id
        try:
            permitted_gross_weight = float(value)
            row_limit = float(limit_value)
        except (TypeError, ValueError):
            return "missing", value, fact.fact_id
        if (
            permitted_gross_weight <= 0 or row_limit <= 0
            or not math.isfinite(permitted_gross_weight)
            or not math.isfinite(row_limit)
        ):
            return "missing", value, fact.fact_id
        return (
            "satisfied" if permitted_gross_weight <= row_limit else "mismatch",
            value,
            fact.fact_id,
        )
    if condition.operator == "time_window_contains_now":
        window = value if isinstance(value, dict) else {}
        observed_at, observed_fact = _value_for("observed_at", context)
        if observed_fact is None:
            return "missing", value, fact.fact_id
        try:
            instant = datetime.fromisoformat(str(observed_at))
            current_minute = instant.hour * 60 + instant.minute
            start = int(window["start_minute"])
            end = int(window["end_minute"])
            if not (0 <= start < 1440 and 0 <= end < 1440):
                raise ValueError("time window minute outside day")
        except (KeyError, TypeError, ValueError):
            return "missing", value, fact.fact_id
        if current_minute in {start, end}:
            return "missing", value, fact.fact_id
        if start < end:
            matched = start < current_minute < end
        else:
            matched = current_minute > start or current_minute < end
        return ("satisfied" if matched else "mismatch", value, fact.fact_id)
    if condition.operator == "not_equals":
        return ("satisfied" if value != condition.expected_value else "mismatch", value, fact.fact_id)
    if condition.operator == "contains":
        try:
            return ("satisfied" if condition.expected_value in value else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    if condition.operator == "in":
        candidates = ACTOR_CLASS_CANDIDATES.get(str(value)) if isinstance(value, str) else None
        if candidates:
            try:
                matches = [candidate in condition.expected_value for candidate in candidates]
            except TypeError:
                return "mismatch", value, fact.fact_id
            if all(matches):
                return "satisfied", value, fact.fact_id
            if any(matches):
                return "missing", value, fact.fact_id
            return "mismatch", value, fact.fact_id
        try:
            return ("satisfied" if value in condition.expected_value else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    if condition.operator == "not_in":
        candidates = ACTOR_CLASS_CANDIDATES.get(str(value)) if isinstance(value, str) else None
        if candidates:
            try:
                matches = [candidate not in condition.expected_value for candidate in candidates]
            except TypeError:
                return "mismatch", value, fact.fact_id
            if all(matches):
                return "satisfied", value, fact.fact_id
            if any(matches):
                return "missing", value, fact.fact_id
            return "mismatch", value, fact.fact_id
        try:
            return ("satisfied" if value not in condition.expected_value else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    if condition.operator in {"gte", "lte", "gt", "lt"}:
        try:
            comparisons = {
                "gte": value >= condition.expected_value,
                "lte": value <= condition.expected_value,
                "gt": value > condition.expected_value,
                "lt": value < condition.expected_value,
            }
            matched = comparisons[condition.operator]
            return ("satisfied" if matched else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    if condition.operator == "between":
        try:
            lower, upper = condition.expected_value
            return ("satisfied" if lower <= value <= upper else "mismatch", value, fact.fact_id)
        except (TypeError, ValueError):
            return "mismatch", value, fact.fact_id
    return "missing", value, fact.fact_id


def _resolve_effects(rule: TrafficRule, context: list[ContextFact]) -> list[RuleEffect]:
    resolved: list[RuleEffect] = []
    for effect in rule.effects:
        if not effect.value_predicate:
            resolved.append(effect)
            continue
        value, _ = _value_for(effect.value_predicate, context)
        if value is None:
            continue
        payload = effect.to_dict()
        payload["value"] = value
        payload["value_predicate"] = None
        resolved.append(RuleEffect(**payload))
    return resolved


def _permission_effect_scope_state(
    rule: TrafficRule,
    actor: str,
    context: list[ContextFact],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Check that any declared permission effect is limited to a known scope.

    Rule effects remain declarative: a location list reads the actor's
    ``<actor>_location`` fact, while ``*_fact`` scope entries name the fact
    that must establish that part of the permission.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    trace: list[dict[str, Any]] = []
    for effect in rule.effects:
        if effect.effect_type not in {"permit_action", "allow_direction"}:
            continue
        scope = effect.scope if isinstance(effect.scope, dict) else {}
        for key, requirement in scope.items():
            if key == "location" and isinstance(requirement, (list, tuple, set)):
                predicate = f"{actor}_location"
                value, fact = _value_for(predicate, context)
                if fact is None:
                    missing.append(predicate)
                    trace.append({"predicate": predicate, "state": "missing", "scope": list(requirement)})
                elif value not in requirement:
                    mismatched.append(predicate)
                    trace.append({"predicate": predicate, "state": "mismatch", "value": value,
                                  "scope": list(requirement), "fact_id": fact.fact_id})
                else:
                    trace.append({"predicate": predicate, "state": "satisfied", "value": value,
                                  "fact_id": fact.fact_id})
            elif key.endswith("_fact") and isinstance(requirement, str):
                value, fact = _value_for(requirement, context)
                if fact is None:
                    missing.append(requirement)
                    trace.append({"predicate": requirement, "state": "missing"})
                elif value is False:
                    mismatched.append(requirement)
                    trace.append({"predicate": requirement, "state": "mismatch", "value": value,
                                  "fact_id": fact.fact_id})
                else:
                    trace.append({"predicate": requirement, "state": "satisfied", "value": value,
                                  "fact_id": fact.fact_id})
    return list(dict.fromkeys(missing)), list(dict.fromkeys(mismatched)), trace


def _evaluate_candidate(
    rule: TrafficRule,
    intent: Intent,
    observation: SignObservation,
    context: list[ContextFact],
    *,
    identity_ambiguous: bool = False,
    actor_unresolved: bool = False,
    actor_missing_facts: list[str] | None = None,
    sign_inventory_status: str = "unknown",
    supplementary_linkage_unresolved: bool = False,
    unmodeled_supplementary_scope_panels: list[str] | None = None,
) -> RuleEvaluation:
    if identity_ambiguous:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            intent_id=intent.intent_id,
            observation_ids=[observation.observation_id],
            decision="UNKNOWN",
            status="unresolved",
            missing_facts=["canonical_sign_identity"],
            legal_basis=rule.legal_basis,
            reason="Mã biển còn mơ hồ; chưa áp dụng rule dứt khoát.",
            trace=[{"step": "sign_candidate_match", "candidate_code": rule.trigger_sign_codes[0]}],
            verification_status=rule.verification_status,
        )

    if actor_unresolved:
        missing = list(actor_missing_facts or ["actor_class"])
        return RuleEvaluation(
            rule_id=rule.rule_id,
            intent_id=intent.intent_id,
            observation_ids=[observation.observation_id],
            decision="UNKNOWN",
            status="unresolved",
            missing_facts=missing,
            legal_basis=rule.legal_basis,
            reason=(
                "Chưa xác định đủ thuộc tính pháp lý của phương tiện để kiểm tra phạm vi áp dụng của biển: "
                + ", ".join(missing)
                + "."
            ),
            confidence="low",
            trace=[{"step": "actor_scope_match", "state": "unknown"}],
            verification_status=rule.verification_status,
        )

    missing: list[str] = []
    matched: list[str] = []
    trace: list[dict[str, Any]] = []
    skipped_unobservable: list[str] = []
    condition_states: list[tuple[RuleCondition, str]] = []
    decision = rule.outcome
    for condition in rule.conditions:
        _, condition_fact = _value_for(condition.predicate, context)
        if (
            condition_fact is None
            and condition.predicate not in SUPPORTED_RULE_CONDITION_PREDICATES
        ):
            skipped_unobservable.append(condition.predicate)
            trace.append({
                "predicate": condition.predicate,
                "state": "skipped_unobservable_scene_fact",
                "value": None,
            })
            continue
        state, value, fact_id = _condition_result(condition, context)
        condition_states.append((condition, state))
        trace.append({"predicate": condition.predicate, "state": state, "value": value})
        if state == "satisfied":
            if fact_id:
                matched.append(fact_id)
        elif state == "missing":
            if condition.required or rule.outcome == "CONDITIONAL":
                missing.append(
                    "motorcycle_or_moped_subtype"
                    if condition.predicate == "actor_vehicle_category"
                    and isinstance(value, str)
                    and value in ACTOR_CLASS_CANDIDATES
                    else condition.predicate
                )
                decision = condition.on_missing  # type: ignore[assignment]
        elif state == "mismatch":
            if condition.on_mismatch == "NOT_APPLICABLE":
                return RuleEvaluation(
                    rule_id=rule.rule_id, intent_id=intent.intent_id,
                    observation_ids=[observation.observation_id], decision="UNKNOWN",
                    status="not_applicable", matched_facts=matched,
                    legal_basis=rule.legal_basis,
                    reason="Điều kiện của rule không khớp tình huống đã biết.", trace=trace,
                    verification_status=rule.verification_status,
                )
            decision = condition.on_mismatch  # type: ignore[assignment]
            if fact_id:
                matched.append(fact_id)

    linked_s508_present, _ = _value_for("linked_s508_panel_present", context)
    if linked_s508_present is True and not any(
        condition.predicate == "linked_s508_temporal_scope_active"
        for condition in rule.conditions
    ):
        missing.append("supplementary_time_scope_not_evaluated")
        decision = "CONDITIONAL"
        trace.append({
            "step": "supplementary_time_scope",
            "state": "panel_present_but_not_modeled_by_rule",
        })

    linked_s505b, _ = _value_for("linked_s505b_panel_present", context)
    linked_s505c, _ = _value_for("linked_s505c_panel_present", context)
    has_s505b_evaluation = {
        "linked_s505b_panel_present", "vehicle_s505b_row_match_confirmed",
        "vehicle_s505b_row_index", "vehicle_permitted_gross_weight_t",
    }.issubset({condition.predicate for condition in rule.conditions}) and any(
        condition.operator == "s505b_gross_weight_within_selected_row"
        for condition in rule.conditions
    )
    has_s505c_evaluation = (
        "linked_s505c_panel_present" in {condition.predicate for condition in rule.conditions}
        and any(condition.operator == "axle_group_loads_within_limits" for condition in rule.conditions)
    )
    unmodeled_bridge_panels = []
    if linked_s505b is True and not has_s505b_evaluation:
        unmodeled_bridge_panels.append("S.505b")
    if linked_s505c is True and not has_s505c_evaluation:
        unmodeled_bridge_panels.append("S.505c")
    if observation.sign_code == "P.106a" and unmodeled_bridge_panels:
        missing.append("supplementary_bridge_weight_scope_not_evaluated")
        if not ((has_s505b_evaluation or has_s505c_evaluation) and decision == "PROHIBITED"):
            decision = "CONDITIONAL"
        trace.append({
            "step": "supplementary_bridge_weight_scope",
            "state": "linked_panel_present_but_not_modeled_by_rule",
            "unmodeled_panels": unmodeled_bridge_panels,
        })

    unmodeled_scope_panels = list(unmodeled_supplementary_scope_panels or [])
    sh6_applies, sh6_match_fact = _value_for("sh6_exception_applies_to_actor", context)
    if sh6_match_fact is not None and sh6_applies is False:
        # The panel was resolved and does not exempt this explicit actor, so it
        # need not block the primary restriction candidate.
        unmodeled_scope_panels = [
            code for code in unmodeled_scope_panels
            if normalize_sign_code(code) != "S.H.6"
        ]
        trace.append({
            "step": "supplementary_exception_scope",
            "state": "resolved_not_applicable_to_actor",
            "fact_id": sh6_match_fact.fact_id,
        })
    s505a_applies, s505a_match_fact = _value_for("s505a_vehicle_scope_applies_to_actor", context)
    if s505a_match_fact is not None:
        # A resolved nonmatching class is outside this primary sign's S.505a
        # scope; a matching class is subject to the primary sign. Neither
        # branch establishes blanket permission.
        unmodeled_scope_panels = [
            code for code in unmodeled_scope_panels
            if normalize_sign_code(code) != "S.505a"
        ]
        trace.append({
            "step": "supplementary_vehicle_scope",
            "state": "resolved_applies_to_actor" if s505a_applies else "resolved_not_applicable_to_actor",
            "fact_id": s505a_match_fact.fact_id,
        })
    if unmodeled_scope_panels:
        missing.extend(
            f"supplementary_{code.replace('.', '').lower()}_scope_not_evaluated"
            for code in unmodeled_scope_panels
        )
        decision = "CONDITIONAL"
        trace.append({
            "step": "supplementary_scope_evaluation",
            "state": "linked_panel_present_but_not_modeled_by_rule",
            "unmodeled_panels": unmodeled_scope_panels,
        })

    rule_execution_allowed = _rule_execution_allowed(rule)
    if not rule_execution_allowed:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            intent_id=intent.intent_id,
            observation_ids=[observation.observation_id],
            decision="UNKNOWN",
            status="unresolved",
            matched_facts=matched,
            missing_facts=list(dict.fromkeys([*missing, "legal_basis"])),
            legal_basis=rule.legal_basis,
            reason="Rule chưa có căn cứ pháp lý được ghi nhận.",
            confidence="low",
            trace=[*trace, {"step": "rule_basis_check", "state": "missing"}],
            verification_status=rule.verification_status,
        )

    if sh6_match_fact is not None and sh6_applies is True and any(
        effect.effect_type in {"prohibit_action", "maximum", "minimum", "exact_value"}
        for effect in rule.effects
    ):
        return RuleEvaluation(
            rule_id=rule.rule_id,
            intent_id=intent.intent_id,
            observation_ids=[observation.observation_id],
            decision="UNKNOWN",
            status="not_applicable",
            matched_facts=[sh6_match_fact.fact_id],
            missing_facts=[],
            legal_basis=rule.legal_basis,
            reason=(
                "S.H.6 xác định ngoại lệ cho nhóm phương tiện đang hỏi đối với hạn chế của biển này; "
                "điều đó chỉ loại trừ hạn chế hiện tại, không tự xác lập quyền lưu thông hoặc miễn các quy tắc khác."
            ),
            confidence="high",
            trace=[*trace, {
                "step": "supplementary_exception_scope",
                "state": "exception_applies_to_actor_restriction_not_applied",
                "fact_id": sh6_match_fact.fact_id,
            }],
            verification_status=rule.verification_status,
        )

    if s505a_match_fact is not None and s505a_applies is False:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            intent_id=intent.intent_id,
            observation_ids=[observation.observation_id],
            decision="UNKNOWN",
            status="not_applicable",
            matched_facts=[s505a_match_fact.fact_id],
            missing_facts=[],
            legal_basis=rule.legal_basis,
            reason=(
                "Loại xe được hỏi không thuộc nhóm hình xe đã xác nhận trên S.505a, nên biển chính này "
                "không áp dụng cho xe đó; điều này không tự xác lập quyền lưu thông hoặc loại trừ biển/quy tắc khác."
            ),
            confidence="high",
            trace=[*trace, {
                "step": "supplementary_vehicle_scope",
                "state": "primary_sign_not_applied_to_actor_other_constraints_remain",
                "fact_id": s505a_match_fact.fact_id,
            }],
            verification_status=rule.verification_status,
        )

    if supplementary_linkage_unresolved:
        missing.append("supplementary_sign_association")
        decision = "CONDITIONAL"
        trace.append({
            "step": "supplementary_sign_association",
            "state": "unresolved",
            "required_for_definitive_permission_assessment": True,
        })

    scope_missing, scope_mismatches, scope_trace = _permission_effect_scope_state(
        rule, intent.actor.value, context,
    )
    trace.extend({"step": "permission_effect_scope", **item} for item in scope_trace)
    if scope_mismatches:
        return RuleEvaluation(
            rule_id=rule.rule_id,
            intent_id=intent.intent_id,
            observation_ids=[observation.observation_id],
            decision="UNKNOWN",
            status="not_applicable",
            matched_facts=matched,
            legal_basis=rule.legal_basis,
            reason="Dữ kiện đã biết nằm ngoài phạm vi cho phép của rule; không thể suy ra quyền từ hiệu ứng này.",
            confidence="high",
            trace=trace,
            verification_status=rule.verification_status,
        )
    if scope_missing:
        missing.extend(scope_missing)
        decision = "CONDITIONAL"

    # A conditional threshold rule becomes ALLOWED only after its comparison
    # facts are present and the comparison has passed. A missing measurement
    # must stay conditional; it must never be treated as proof of permission.
    passed_prohibitive_threshold = any(
        condition.operator in {
            "lte_fact", "gte_fact", "gt_fact", "lt_fact",
            "lte", "gte", "gt", "lt",
        }
        and condition.on_mismatch == "PROHIBITED"
        and state == "satisfied"
        for condition, state in condition_states
    )
    if (
        rule.outcome == "CONDITIONAL"
        and decision == "CONDITIONAL"
        and passed_prohibitive_threshold
        and not missing
        and not skipped_unobservable
    ):
        decision = "ALLOWED"
        trace.append({
            "step": "generic_threshold_evaluation",
            "state": "within_limit",
        })

    conditions_incomplete = bool(missing)
    decision_scope_unresolved = bool(
        set(skipped_unobservable).intersection(UNOBSERVABLE_DECISION_SCOPE_PREDICATES)
    ) or (decision == "ALLOWED" and bool(skipped_unobservable))
    if decision_scope_unresolved and decision in {"ALLOWED", "PROHIBITED"}:
        decision = "CONDITIONAL"
        trace.append({
            "step": "permission_scope_guard",
            "state": "conditional_due_to_unobservable_scope",
            "skipped_predicates": skipped_unobservable,
        })
    effects = (
        _resolve_effects(rule, context)
        if not conditions_incomplete and not decision_scope_unresolved else []
    )
    reason = rule.reason if not missing else f"{rule.reason} Thiếu dữ kiện: {', '.join(missing)}."
    if decision_scope_unresolved:
        reason += " Chưa thể xác nhận phạm vi áp dụng từ dữ liệu hiện có."
    if supplementary_linkage_unresolved:
        reason += " Chưa xác định được biển phụ nào gắn với biển chính."
    if not missing and intent.kind == "CHECK_PERMISSION":
        requested_speed = intent.requested_answer.get("requested_speed_kmh")
        maximum_effect = next((
            effect for effect in effects
            if effect.effect_type == "maximum" and effect.action == "speed" and effect.value is not None
        ), None)
        minimum_effect = next((
            effect for effect in effects
            if effect.effect_type == "minimum" and effect.action == "speed" and effect.value is not None
        ), None)
        if requested_speed is not None and maximum_effect is not None:
            try:
                requested = float(str(requested_speed).replace(",", "."))
                maximum = float(str(maximum_effect.value).replace(",", "."))
            except (TypeError, ValueError):
                missing.append("numeric_speed_comparison")
                decision = "CONDITIONAL"
                reason = "Không thể so sánh tốc độ người dùng nêu với trị số đã đọc trên biển."
                trace.append({"step": "numeric_limit_comparison", "state": "unresolved"})
            else:
                trace.append({
                    "step": "numeric_limit_comparison",
                    "requested_speed_kmh": requested,
                    "maximum_speed_kmh": maximum,
                    "state": "exceeds_limit" if requested > maximum else "within_or_at_limit",
                })
                if requested > maximum:
                    decision = "PROHIBITED"
                    reason = f"Tốc độ được hỏi ({requested:g} km/h) vượt quá giới hạn trên biển ({maximum:g} km/h)."
                else:
                    decision = "ALLOWED"
                    reason = (
                        f"Tốc độ được hỏi ({requested:g} km/h) không vượt giới hạn trên biển "
                        f"({maximum:g} km/h); rule này không kết luận cấm."
                    )
        elif requested_speed is not None and minimum_effect is not None:
            try:
                requested = float(str(requested_speed).replace(",", "."))
                minimum = float(str(minimum_effect.value).replace(",", "."))
            except (TypeError, ValueError):
                missing.append("numeric_speed_comparison")
                decision = "CONDITIONAL"
                reason = "Không thể so sánh tốc độ người dùng nêu với tốc độ tối thiểu trên biển."
                trace.append({"step": "numeric_limit_comparison", "state": "unresolved"})
            else:
                trace.append({
                    "step": "numeric_limit_comparison",
                    "requested_speed_kmh": requested,
                    "minimum_speed_kmh": minimum,
                    "state": "below_limit" if requested < minimum else "at_or_above_limit",
                })
                if requested < minimum:
                    decision = "PROHIBITED"
                    reason = (
                        f"Tốc độ được hỏi ({requested:g} km/h) thấp hơn tốc độ tối thiểu "
                        f"trên biển ({minimum:g} km/h) trong điều kiện đã xác nhận thuận lợi và an toàn."
                    )
                else:
                    decision = "ALLOWED"
                    reason = (
                        f"Tốc độ được hỏi ({requested:g} km/h) không thấp hơn mức tối thiểu "
                        f"trên biển ({minimum:g} km/h); rule này không kết luận cấm."
                    )

    incomplete = bool(missing)
    return RuleEvaluation(
        rule_id=rule.rule_id,
        intent_id=intent.intent_id,
        observation_ids=[observation.observation_id],
        decision=decision,
        status="incomplete" if incomplete else "applied",
        matched_facts=matched,
        missing_facts=missing,
        legal_basis=rule.legal_basis,
        reason=reason,
        confidence=(
            "low" if incomplete else "medium"
            if rule.verification_status != "verified"
            or any(ref.get("verification_status") != "verified" for ref in rule.legal_basis)
            else "high"
        ),
        trace=trace,
        verification_status=rule.verification_status,
        effects=effects if rule_execution_allowed and not conditions_incomplete else [],
    )


def find_matching_rules(
    facts: TrafficFacts,
    intent: QueryIntent,
    context_facts: Iterable[ContextFact] = (),
    rules: Iterable[TrafficRule] = TRAFFIC_RULES,
) -> list[RuleEvaluation]:
    matches: list[RuleEvaluation] = []
    scenario_facts = list(context_facts)
    for query_intent in intent.intents:
        if query_intent.kind != "CHECK_PERMISSION":
            continue
        action = query_intent.action.value
        actor = query_intent.actor.value
        actor = {
            "motorcycle": "motorbike",
            "automobile": "car",
            "passenger_car": "four_wheel_passenger_vehicle",
            "other_motor_vehicle": "motor_vehicle",
        }.get(actor, actor)
        # The intent schema uses a compact actor group. Refine from the original
        # wording when a legally specific class is needed for rule matching.
        if query_intent.actor.raw_text:
            detailed_actor = extract_vehicle(query_intent.actor.raw_text)
            if detailed_actor != "unknown":
                actor = detailed_actor
        intent_context = list(scenario_facts)
        if (
            query_intent.actor.status in {"provided", "defaulted"}
            and actor not in {"unknown", "vehicle", "motor_vehicle", "non_motor_vehicle", "priority_vehicle", "emergency_vehicle"}
            and vehicle_is_covered(actor, "vehicle")
        ):
            intent_context.append(ContextFact(
                fact_id=f"{query_intent.intent_id}:actor-vehicle-category",
                predicate="actor_vehicle_category",
                value=actor,
                status="provided",
                source="intent_actor_slot",
                raw_text=query_intent.actor.raw_text,
                confidence="high",
                metadata={"intent_id": query_intent.intent_id},
            ))
            intent_context.append(ContextFact(
                fact_id=f"{query_intent.intent_id}:actor-is-motor-vehicle",
                predicate="actor_is_motor_vehicle",
                value=vehicle_is_covered(actor, "motor_vehicle"),
                status="provided",
                source="intent_actor_slot",
                raw_text=query_intent.actor.raw_text,
                confidence="high",
                metadata={"intent_id": query_intent.intent_id},
            ))
        if (
            query_intent.actor.status == "provided"
            and actor == "priority_vehicle"
        ):
            intent_context.append(ContextFact(
                fact_id=f"{query_intent.intent_id}:actor-priority-right",
                predicate="actor_has_priority_right",
                value=True,
                status="provided",
                source="intent_actor_slot",
                raw_text=query_intent.actor.raw_text,
                confidence="high",
                metadata={"intent_id": query_intent.intent_id},
            ))
        for observation in facts.sign_observations:
            context = _context_facts(facts, intent_context, observation)
            sh6_match = _derive_sh6_actor_match(context, actor, query_intent.intent_id)
            if sh6_match is not None:
                context.append(sh6_match)
            s505a_match = _derive_s505a_actor_match(context, actor, query_intent.intent_id)
            if s505a_match is not None:
                context.append(s505a_match)
            if query_intent.target.kind == "sign_observation" and query_intent.target.observation_ids:
                if observation.observation_id not in query_intent.target.observation_ids:
                    continue
            if query_intent.target.kind == "sign_code" and query_intent.target.sign_codes:
                candidate_codes = {str(item.get("sign_code") or "") for item in observation.candidates}
                if observation.sign_code not in query_intent.target.sign_codes and not candidate_codes.intersection(query_intent.target.sign_codes):
                    continue
            for rule in rules:
                if action not in rule.actions:
                    continue
                actor_unresolved = (
                    query_intent.kind in PERMISSION_INTENTS
                    and (actor == "unknown" or query_intent.actor.status in {"not_stated", "ambiguous", "needs_resolution"})
                )
                actor_missing_facts = ["actor_class"]
                if actor_unresolved and actor in {"priority_vehicle", "emergency_vehicle"}:
                    # Some sign rules name a priority/emergency vehicle explicitly,
                    # while also depending on its underlying vehicle class (for
                    # example, a priority truck with a certified payload limit).
                    # A resolved actor_vehicle_category may settle that subtype,
                    # but only when this rule actually checks that fact. The
                    # condition evaluator still validates the category and any
                    # additional legal thresholds before returning an outcome.
                    actor_category, actor_category_fact = _value_for("actor_vehicle_category", context)
                    category_is_checked = any(
                        condition.predicate == "actor_vehicle_category" and condition.required
                        for condition in rule.conditions
                    )
                    if (
                        actor_category_fact is not None
                        and category_is_checked
                        and any(vehicle_is_covered(str(actor_category), scope) for scope in rule.actor_scope)
                    ):
                        actor_unresolved = False
                if (
                    query_intent.kind in PERMISSION_INTENTS
                    and actor == "emergency_vehicle"
                    and "priority_vehicle" in rule.excluded_actor_scope
                ):
                    # Emergency vehicles are only within the legal priority class
                    # in qualifying circumstances; vehicle type alone is not proof.
                    actor_unresolved = True
                    actor_missing_facts = ["priority_vehicle_status"]
                if actor in ACTOR_CLASS_CANDIDATES and not actor_unresolved:
                    scope_match = _actor_scope_match(actor, rule.actor_scope)
                    excluded_match = _actor_scope_match(actor, rule.excluded_actor_scope)
                    if scope_match == "none" or excluded_match == "all":
                        continue
                    if scope_match == "some" or excluded_match == "some":
                        actor_unresolved = True
                        actor_missing_facts = ["motorcycle_or_moped_subtype"]
                else:
                    if not actor_unresolved and any(
                        actor == excluded or vehicle_is_covered(actor, excluded)
                        for excluded in rule.excluded_actor_scope
                    ):
                        continue
                    if not actor_unresolved and actor != "unknown" and not any(
                        vehicle_is_covered(actor, scope) for scope in rule.actor_scope
                    ):
                        continue
                code = observation.sign_code
                if code in rule.trigger_sign_codes and observation.identity_status == "identified":
                    matches.append(_evaluate_candidate(
                        rule, query_intent, observation, context,
                        actor_unresolved=actor_unresolved,
                        actor_missing_facts=actor_missing_facts,
                        sign_inventory_status=facts.sign_inventory_status,
                        supplementary_linkage_unresolved=_supplementary_linkage_unresolved(facts, observation),
                        unmodeled_supplementary_scope_panels=_unmodeled_supplementary_scope_panels(
                            facts, observation, rule
                        ),
                    ))
                    continue
                candidate_codes = {str(item.get("sign_code") or "") for item in observation.candidates}
                if observation.identity_status == "ambiguous" and candidate_codes.intersection(rule.trigger_sign_codes):
                    matches.append(_evaluate_candidate(
                        rule, query_intent, observation, context,
                        identity_ambiguous=True,
                        actor_unresolved=actor_unresolved,
                        actor_missing_facts=actor_missing_facts,
                        sign_inventory_status=facts.sign_inventory_status,
                        supplementary_linkage_unresolved=_supplementary_linkage_unresolved(facts, observation),
                        unmodeled_supplementary_scope_panels=_unmodeled_supplementary_scope_panels(
                            facts, observation, rule
                        ),
                    ))
    return sorted(matches, key=lambda item: (item.intent_id, item.rule_id, item.observation_ids))
