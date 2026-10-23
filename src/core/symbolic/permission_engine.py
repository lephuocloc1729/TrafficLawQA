"""Permission decisions from facts and intent, without per-sign rule lookup.

The engine interprets the shared fact vocabulary: actor/action prohibitions,
explicit permissions, scopes, temporal conditions, and numeric limits. Sign
meaning is supplied by the fact templates/OCR pipeline; this module does not
load or evaluate the traffic rulebook.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Any

from src.core.symbolic.fact_schema import FACT_PREDICATE_ALIASES, TrafficFacts
from src.core.symbolic.condition_facts import expand_condition_facts
from src.core.symbolic.intent_schema import Intent, IntentParseResult
from src.core.symbolic.rule_schema import IntentEvaluation, RuleEvaluation
from src.core.symbolic.sign_catalog import load_default_sign_catalog
from src.core.symbolic.sign_normalizer import normalize_sign_code
from src.core.symbolic.vehicle_taxonomy import ACTOR_CLASS_CANDIDATES, vehicle_is_covered


ACTION_ALIASES = {
    "parking": "park", "stopping": "stop", "going_straight": "go_straight",
    "roundabout_circulation": "circulation", "crossing": "crossing",
    "turn_left": "turn_left", "turn_right": "turn_right", "u_turn": "u_turn",
    "entering": "entering", "exiting": "exiting", "circulation": "circulation",
    "speed": "speed", "reduce_speed": "reduce_speed", "yielding": "yield",
    "using_horn": "honk", "honk": "honk", "overtaking": "overtake",
    "overtake": "overtake", "reversing": "reverse", "reverse": "reverse",
    "lane_change": "change_lane", "following_distance": "following_distance",
}
ACTOR_ALIASES = {
    "automobile": "car", "passenger_car": "four_wheel_passenger_vehicle",
    "motorcycle": "motorcycle_or_moped", "other_motor_vehicle": "motor_vehicle",
}

LIMITS: dict[str, tuple[str, str, set[str]]] = {
    "maximum_speed": ("requested_speed_kmh", "maximum", {"speed", "circulation"}),
    "minimum_speed": ("requested_speed_kmh", "minimum", {"speed", "circulation"}),
    "maximum_vehicle_mass": ("vehicle_gross_weight_t", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_payload_mass": ("vehicle_payload_t", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_axle_load": ("vehicle_axle_load_t", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_vehicle_height": ("vehicle_height_m", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_vehicle_width": ("vehicle_width_m", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_vehicle_length": ("vehicle_length_m", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_combination_length": ("vehicle_combination_length_m", "maximum", {"entering", "circulation", "crossing"}),
    "maximum_seat_count": ("vehicle_seat_count", "maximum", {"entering", "circulation", "crossing"}),
    "minimum_distance": ("requested_distance_m", "minimum", {"following_distance", "circulation"}),
}


def _canonical_predicate(value: Any) -> str:
    predicate = str(value or "unknown").strip().lower()
    return FACT_PREDICATE_ALIASES.get(predicate, predicate)


def _actor_matches(actor: str, subject: Any) -> bool:
    if not isinstance(subject, str):
        return False
    subject = subject.strip().lower()
    if actor in ACTOR_CLASS_CANDIDATES:
        return all(vehicle_is_covered(candidate, subject) for candidate in ACTOR_CLASS_CANDIDATES[actor])
    if actor in {"unknown", "vehicle", "motor_vehicle", "non_motor_vehicle"}:
        return actor == subject
    return actor == subject or vehicle_is_covered(actor, subject)


def _actor_matches_any(actor: str, subjects: list[Any]) -> bool:
    if actor in ACTOR_CLASS_CANDIDATES:
        return all(
            any(vehicle_is_covered(candidate, str(subject)) for subject in subjects)
            for candidate in ACTOR_CLASS_CANDIDATES[actor]
        )
    return any(_actor_matches(actor, subject) for subject in subjects)


def _fact_is_usable(fact: dict[str, Any]) -> bool:
    return str(fact.get("status", "observed")).lower() not in {
        "unknown", "ambiguous", "needs_resolution", "not_observed",
    }


def _number(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _query_values(query: IntentParseResult, intent: Intent) -> dict[str, Any]:
    candidates: dict[str, list[Any]] = {}
    for item in query.context_facts:
        candidates.setdefault(_canonical_predicate(item.predicate), []).append(item.value)
    values = {
        key: (items[0] if all(item == items[0] for item in items) else None)
        for key, items in candidates.items()
    }
    for key, item in intent.conditions.items():
        values[key] = item.get("value") if isinstance(item, dict) else item
    values.update({key: value for key, value in intent.requested_answer.items() if value is not None})
    return values


def _targeted(intent: Intent, sign: dict[str, Any]) -> bool:
    target = intent.target
    if target.kind == "sign_observation" and target.observation_ids:
        return sign.get("id") in target.observation_ids
    if target.kind == "sign_code" and target.sign_codes:
        code = normalize_sign_code(sign.get("code"))
        return code in {normalize_sign_code(item) for item in target.sign_codes}
    return True


def _time_match(value: Any, query_values: dict[str, Any]) -> tuple[str, str | None]:
    observed = query_values.get("observed_at")
    if not isinstance(observed, str):
        return "missing", "observed_at"
    try:
        current = datetime.fromisoformat(observed)
    except ValueError:
        return "missing", "observed_at"
    windows = value if isinstance(value, list) else [value]
    for window in windows:
        if not isinstance(window, dict):
            continue
        start = window.get("from", window.get("start"))
        end = window.get("to", window.get("end"))
        try:
            begin, finish = time.fromisoformat(str(start)), time.fromisoformat(str(end))
        except ValueError:
            continue
        now = current.timetz().replace(tzinfo=None)
        if begin <= finish and begin <= now <= finish:
            return "satisfied", None
        if begin > finish and (now >= begin or now <= finish):
            return "satisfied", None
    return "mismatch", None


def _temporal_state(sign_facts: list[dict[str, Any]], values: dict[str, Any]) -> tuple[str, str | None]:
    for fact in sign_facts:
        predicate = _canonical_predicate(fact.get("predicate"))
        if predicate == "calendar_parity" and _fact_is_usable(fact):
            parity = values.get("calendar_parity")
            if parity is None:
                observed = values.get("observed_at")
                if isinstance(observed, str):
                    try:
                        parity = "odd" if datetime.fromisoformat(observed).day % 2 else "even"
                    except ValueError:
                        pass
            if parity is None:
                return "missing", "calendar_parity"
            expected = str(fact.get("value", "")).replace("_days", "").lower()
            actual = str(parity).replace("_days", "").lower()
            if expected != actual:
                return "mismatch", None
        if predicate == "active_time" and _fact_is_usable(fact):
            state, missing = _time_match(fact.get("value"), values)
            if state != "satisfied":
                return state, missing
    return "satisfied", None


def _scope_state(scope: Any, actor: str, values: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(scope, dict):
        return "satisfied", []
    missing: list[str] = []
    vehicle_types = scope.get("vehicle_types")
    if vehicle_types:
        types = vehicle_types if isinstance(vehicle_types, list) else [vehicle_types]
        if not _actor_matches_any(actor, types):
            return "mismatch", []
    excluded = scope.get("excluded_vehicle_types", [])
    if isinstance(excluded, str):
        excluded = [excluded]
    if any(_actor_matches(actor, subject) for subject in excluded):
        return "mismatch", []
    location = scope.get("location")
    if location:
        location_fact = values.get("pedestrian_location")
        if location_fact is None:
            missing.append("pedestrian_location")
        elif str(location_fact).lower() not in {
            str(item).lower() for item in (location if isinstance(location, list) else [location])
        }:
            return "mismatch", []
    supported = {"vehicle_types", "excluded_vehicle_types", "location", "exclusive", "exclusive_actions", "actions"}
    missing.extend(f"scope:{key}" for key in scope if key not in supported)
    return ("missing", missing) if missing else ("satisfied", [])


def _threshold_state(
    fact: dict[str, Any], actor: str, action: str, values: dict[str, Any],
) -> tuple[str, list[str], str]:
    predicate = _canonical_predicate(fact.get("predicate"))
    spec = LIMITS.get(predicate)
    if spec is None:
        return "irrelevant", [], ""
    input_predicate, direction, actions = spec
    scope_state, scope_missing = _scope_state(fact.get("scope"), actor, values)
    if scope_state == "mismatch":
        return "irrelevant", [], ""
    if action not in actions:
        return "irrelevant", [], ""
    if scope_state == "missing":
        return "conditional", scope_missing, "Chưa đủ dữ kiện xác nhận phạm vi áp dụng của giới hạn."
    if not _fact_is_usable(fact) or fact.get("value") is None:
        return "conditional", [predicate], "Chưa đọc được giá trị giới hạn trên biển."
    actual = _number(values.get(input_predicate))
    limit = _number(fact.get("value"))
    if actual is None:
        return "conditional", [input_predicate], "Chưa có thông số của phương tiện để so sánh với giới hạn."
    if limit is None:
        return "conditional", [predicate], "Giá trị giới hạn không hợp lệ hoặc chưa được xác nhận."
    violation = actual > limit if direction == "maximum" else actual < limit
    if violation:
        return "prohibited", [], f"Giá trị phương tiện ({actual:g}) vượt giới hạn áp dụng ({limit:g})."
    return "allowed", [], f"Giá trị phương tiện ({actual:g}) nằm trong giới hạn ({limit:g})."


def _evaluate_sign(
    intent: Intent, sign: dict[str, Any], values: dict[str, Any],
) -> RuleEvaluation | None:
    code = str(sign.get("code") or "")
    observation_id = str(sign.get("id") or "unknown")
    if sign.get("role") == "supplementary":
        return None
    if not _targeted(intent, sign):
        return None
    if sign.get("identity_status", "identified" if code else "unidentified") != "identified" or not code:
        return RuleEvaluation(
            "FACT_COMPARISON", intent.intent_id, [observation_id], "UNKNOWN", "unresolved",
            missing_facts=["canonical_sign_identity"],
            reason="Chưa nhận diện được biển để đối chiếu facts với hành vi được hỏi.",
            confidence="low", trace=[{"step": "fact_comparison", "state": "identity_unresolved"}],
        )

    sign_facts = expand_condition_facts(
        [item for item in sign.get("facts", []) if isinstance(item, dict)]
    )
    actor = ACTOR_ALIASES.get(intent.actor.value, intent.actor.value)
    action = ACTION_ALIASES.get(intent.action.value, intent.action.value)
    matched: list[str] = []
    trace: list[dict[str, Any]] = []
    relevant = False
    known_allowed = False
    conditional_missing: list[str] = []
    prohibition_facts = [
        item for item in sign_facts
        if _canonical_predicate(item.get("predicate")) == "prohibited_actor" and _fact_is_usable(item)
    ]
    prohibited_actions = [
        item for item in sign_facts
        if _canonical_predicate(item.get("predicate")) == "prohibited_action" and _fact_is_usable(item)
    ]
    prohibits_action = any(
        action == ACTION_ALIASES.get(str(value), str(value))
        for item in prohibited_actions
        for value in (item.get("value") if isinstance(item.get("value"), list) else [item.get("value")])
    )
    actor_unresolved = actor == "unknown" or intent.actor.status in {
        "ambiguous", "needs_resolution", "not_stated",
    }
    matching_prohibition = any(
        action in [
            ACTION_ALIASES.get(str(value), str(value))
            for effect in prohibited_actions
            for value in (effect.get("value") if isinstance(effect.get("value"), list) else [effect.get("value")])
        ]
        for fact in prohibition_facts
    )
    actor_dependent = matching_prohibition or any(
        _canonical_predicate(item.get("predicate")) in {"permitted_actor", *LIMITS}
        and isinstance(item.get("scope"), dict)
        and bool(item["scope"].get("vehicle_types"))
        for item in sign_facts
    )
    if actor_unresolved and actor_dependent:
        return RuleEvaluation(
            "FACT_COMPARISON", intent.intent_id, [observation_id], "UNKNOWN", "unresolved",
            missing_facts=["actor_class"], reason="Chưa xác định được loại phương tiện để đối chiếu.",
            confidence="low", trace=[{"step": "fact_comparison", "state": "actor_unresolved"}],
        )

    for actor_fact in prohibition_facts:
        if not prohibits_action:
            continue
        subjects = actor_fact.get("value")
        subjects = subjects if isinstance(subjects, list) else [subjects]
        if not _actor_matches_any(actor, subjects):
            continue
        actor_scope_state, missing_scope = _scope_state(actor_fact.get("scope"), actor, values)
        if actor_scope_state == "mismatch":
            continue
        if actor_scope_state == "missing":
            relevant = True
            conditional_missing.extend(missing_scope)
            continue
        conditions = actor_fact.get("conditions", {})
        exceptions = conditions.get("exceptions", []) if isinstance(conditions, dict) else []
        if isinstance(conditions, dict):
            actor_condition = conditions.get(actor)
            linked_condition_facts = [
                item for item in sign_facts
                if _canonical_predicate(item.get("predicate")) in LIMITS
                and item.get("applies_to") == actor
                and item.get("operator")
            ]
            legacy_limit_fact = None
            if isinstance(actor_condition, dict):
                condition_predicate = _canonical_predicate(actor_condition.get("predicate"))
                legacy_limit_fact = next((
                    item for item in sign_facts
                    if _canonical_predicate(item.get("predicate")) == condition_predicate
                    and (item.get("applies_to") in {None, actor})
                ), None)
                if legacy_limit_fact is None:
                    relevant = True
                    conditional_missing.append(condition_predicate)
                    continue
                linked_condition_facts.append(legacy_limit_fact)
            if linked_condition_facts:
                # A restriction with several applicable condition entries is
                # active only when every required threshold is exceeded.
                condition_states = []
                for limit_fact in linked_condition_facts:
                    condition_predicate = _canonical_predicate(limit_fact.get("predicate"))
                    threshold_state, missing, _reason = _threshold_state(limit_fact, actor, action, values)
                    if threshold_state == "irrelevant":
                        continue
                    relevant = True
                    matched.append(f"{observation_id}:{condition_predicate}")
                    trace.append({"step": "threshold_comparison", "predicate": condition_predicate, "state": threshold_state})
                    condition_states.append(threshold_state)
                    if threshold_state == "conditional":
                        conditional_missing.extend(missing)
                if condition_states:
                    if "conditional" in condition_states:
                        continue
                    if "allowed" in condition_states:
                        known_allowed = True
                        continue
                    temporal_state, temporal_missing = _temporal_state(sign_facts, values)
                    if temporal_state == "mismatch":
                        trace.append({"step": "temporal_scope", "state": "inactive"})
                        continue
                    if temporal_state == "missing":
                        conditional_missing.append(temporal_missing or "temporal_scope")
                        trace.append({"step": "temporal_scope", "state": "unresolved"})
                        continue
                    return RuleEvaluation(
                        "FACT_COMPARISON", intent.intent_id, [observation_id], "PROHIBITED", "applied",
                        matched, [], reason="Đã thỏa các ngưỡng cấm áp dụng cho phương tiện.",
                        confidence=actor_fact.get("confidence", "medium"), trace=trace,
                    )
            unsupported_conditions = [
                key for key in conditions
                if key not in {"exceptions", actor}
            ]
            if unsupported_conditions:
                relevant = True
                conditional_missing.extend(f"condition:{key}" for key in unsupported_conditions)
                continue
            exceptions = conditions.get("exceptions", []) if isinstance(conditions, dict) else []
            if "priority_vehicle" in exceptions and actor in {"priority_vehicle", "emergency_vehicle"}:
                relevant = True
                conditional_missing.append("priority_vehicle_status")
                continue
        if any(action in [ACTION_ALIASES.get(str(value), str(value)) for value in (item.get("value") if isinstance(item.get("value"), list) else [item.get("value")])] for item in prohibited_actions):
            matched.extend([f"{observation_id}:prohibited_actor", f"{observation_id}:prohibited_action"])
            temporal_state, temporal_missing = _temporal_state(sign_facts, values)
            if temporal_state == "mismatch":
                trace.append({"step": "temporal_scope", "state": "inactive"})
                continue
            relevant = True
            if temporal_state == "missing":
                conditional_missing.extend([temporal_missing] if temporal_missing else [])
                trace.append({"step": "temporal_scope", "state": "unresolved"})
                continue
            if actor in {"priority_vehicle", "emergency_vehicle"} and "priority_vehicle" in exceptions:
                continue
            return RuleEvaluation(
                "FACT_COMPARISON", intent.intent_id, [observation_id], "PROHIBITED", "applied",
                list(dict.fromkeys(matched)), reason=f"Facts của biển {code} cấm {actor} thực hiện hành vi {intent.action.value}.",
                confidence=min(actor_fact.get("confidence", "medium"), prohibited_actions[0].get("confidence", "medium")),
                trace=[*trace, {"step": "actor_action_comparison", "state": "prohibited", "sign_code": code}],
            )

    # Numeric facts are generic limits. Their values and vehicle measurements
    # are compared here instead of requiring one rule per sign class.
    for fact in sign_facts:
        state, missing, reason = _threshold_state(fact, actor, action, values)
        if state == "irrelevant":
            continue
        temporal_state, temporal_missing = _temporal_state(sign_facts, values)
        if temporal_state == "mismatch":
            trace.append({"step": "temporal_scope", "state": "inactive"})
            continue
        relevant = True
        matched.append(f"{observation_id}:{_canonical_predicate(fact.get('predicate'))}")
        trace.append({"step": "threshold_comparison", "predicate": _canonical_predicate(fact.get("predicate")), "state": state})
        if state == "prohibited":
            if temporal_state == "missing":
                conditional_missing.append(temporal_missing or "temporal_scope")
                trace.append({"step": "temporal_scope", "state": "unresolved"})
                continue
            return RuleEvaluation("FACT_COMPARISON", intent.intent_id, [observation_id], "PROHIBITED", "applied", matched, [], reason=reason, confidence=fact.get("confidence", "medium"), trace=trace)
        if state == "conditional":
            conditional_missing.extend(missing)
        elif state == "allowed":
            known_allowed = True
        if temporal_state == "missing":
            conditional_missing.append(temporal_missing or "temporal_scope")

    mandatory = next((item for item in sign_facts if _canonical_predicate(item.get("predicate")) == "mandatory_action" and _fact_is_usable(item)), None)
    if mandatory:
        mandatory_scope, missing_scope = _scope_state(mandatory.get("scope"), actor, values)
        mandatory_actions = mandatory.get("value") if isinstance(mandatory.get("value"), list) else [mandatory.get("value")]
        mandatory_actions = [ACTION_ALIASES.get(str(value), str(value)) for value in mandatory_actions]
        scoped_actions = mandatory.get("scope", {}).get("actions", []) if isinstance(mandatory.get("scope"), dict) else []
        scoped_actions = [ACTION_ALIASES.get(str(value), str(value)) for value in (scoped_actions if isinstance(scoped_actions, list) else [scoped_actions])]
        if mandatory_scope == "missing":
            relevant = True
            conditional_missing.extend(missing_scope)
        elif mandatory_scope == "satisfied" and (not scoped_actions or action in scoped_actions):
            if action in mandatory_actions:
                relevant = True
                return RuleEvaluation("FACT_COMPARISON", intent.intent_id, [observation_id], "ALLOWED", "applied", [f"{observation_id}:mandatory_action"], reason=f"Biển {code} yêu cầu thực hiện hành vi {intent.action.value}; hành vi này được phép và là bắt buộc trong phạm vi biển.", confidence=mandatory.get("confidence", "medium"), trace=[{"step": "mandatory_action_comparison", "state": "required", "action": action}])
            if isinstance(mandatory.get("scope"), dict) and mandatory["scope"].get("exclusive"):
                relevant = True
                return RuleEvaluation("FACT_COMPARISON", intent.intent_id, [observation_id], "PROHIBITED", "applied", [f"{observation_id}:mandatory_action"], reason=f"Biển {code} chỉ cho phép hướng/hành vi được chỉ định; {intent.action.value} không nằm trong hướng bắt buộc.", confidence=mandatory.get("confidence", "medium"), trace=[{"step": "mandatory_action_comparison", "state": "exclusive_mismatch", "action": action, "required_actions": mandatory_actions}])

    vehicle_scope = next((item for item in sign_facts if _canonical_predicate(item.get("predicate")) == "vehicle_scope" and _fact_is_usable(item)), None)
    if vehicle_scope and isinstance(vehicle_scope.get("scope"), dict):
        scoped_actions = vehicle_scope["scope"].get("actions", [])
        scoped_actions = [ACTION_ALIASES.get(str(value), str(value)) for value in (scoped_actions if isinstance(scoped_actions, list) else [scoped_actions])]
        if (not scoped_actions or action in scoped_actions) and vehicle_scope["scope"].get("exclusive"):
            subjects = vehicle_scope.get("value") if isinstance(vehicle_scope.get("value"), list) else [vehicle_scope.get("value")]
            relevant = True
            if _actor_matches_any(actor, subjects):
                known_allowed = True
            else:
                return RuleEvaluation("FACT_COMPARISON", intent.intent_id, [observation_id], "PROHIBITED", "applied", [f"{observation_id}:vehicle_scope"], reason=f"Biển {code} giới hạn đoạn đường này cho {', '.join(str(item) for item in subjects)}.", confidence=vehicle_scope.get("confidence", "medium"), trace=[{"step": "vehicle_scope_comparison", "state": "actor_outside_scope", "actor": actor}])

    # Explicit positive facts are the only facts that can establish allowance
    # on their own. A sign that simply does not mention the queried actor does
    # not prove that the whole segment is safe to traverse.
    allowed_actor = next((item for item in sign_facts if _canonical_predicate(item.get("predicate")) == "permitted_actor" and _fact_is_usable(item)), None)
    allowed_action = next((item for item in sign_facts if _canonical_predicate(item.get("predicate")) == "permitted_action" and _fact_is_usable(item)), None)
    if allowed_actor and allowed_action:
        actors = allowed_actor.get("value") if isinstance(allowed_actor.get("value"), list) else [allowed_actor.get("value")]
        actions = allowed_action.get("value") if isinstance(allowed_action.get("value"), list) else [allowed_action.get("value")]
        normalized_actions = [ACTION_ALIASES.get(str(value), str(value)) for value in actions]
        if _actor_matches_any(actor, actors) and action in normalized_actions:
            allowed_scope, allowed_scope_missing = _scope_state(allowed_actor.get("scope"), actor, values)
            if allowed_scope == "mismatch":
                allowed_scope_missing = []
            elif allowed_scope == "missing":
                conditional_missing.extend(allowed_scope_missing)
                relevant = True
            else:
                relevant = True
                return RuleEvaluation("FACT_COMPARISON", intent.intent_id, [observation_id], "ALLOWED", "applied", [f"{observation_id}:permitted_actor", f"{observation_id}:permitted_action"], reason=f"Facts của biển {code} cho phép {actor} thực hiện hành vi {intent.action.value} trong phạm vi đã nêu.", confidence=min(allowed_actor.get("confidence", "medium"), allowed_action.get("confidence", "medium")), trace=[{"step": "actor_action_comparison", "state": "allowed", "sign_code": code}])

    if not relevant:
        return RuleEvaluation(
            "FACT_COMPARISON", intent.intent_id, [observation_id], "UNKNOWN", "not_applicable",
            reason=f"Facts của biển {code} chưa nêu giới hạn áp dụng cho actor/action này.",
            confidence="low", trace=[{"step": "fact_comparison", "state": "no_matching_permission_fact", "sign_code": code}],
        )
    if conditional_missing:
        missing = list(dict.fromkeys(conditional_missing))
        return RuleEvaluation(
            "FACT_COMPARISON", intent.intent_id, [observation_id], "CONDITIONAL", "incomplete",
            matched_facts=list(dict.fromkeys(matched)), missing_facts=missing,
            reason="Còn thiếu dữ kiện để đối chiếu đầy đủ facts của biển: " + ", ".join(missing) + ".",
            confidence="low", trace=trace,
        )
    if known_allowed:
        return RuleEvaluation(
            "FACT_COMPARISON", intent.intent_id, [observation_id], "ALLOWED", "applied",
            matched_facts=list(dict.fromkeys(matched)),
            reason="Thông số đã biết của phương tiện nằm trong giới hạn thể hiện trên biển.",
            confidence="medium", trace=trace,
        )
    return RuleEvaluation(
        "FACT_COMPARISON", intent.intent_id, [observation_id], "UNKNOWN", "not_applicable",
        reason=f"Facts của biển {code} không tạo kết luận cho hành vi được hỏi.",
        confidence="low", trace=trace,
    )


def evaluate_check_permission(
    facts: TrafficFacts,
    query: IntentParseResult,
    intent: Intent,
) -> tuple[list[RuleEvaluation], IntentEvaluation]:
    """Evaluate one permission intent solely from the resolved fact inventory."""
    payload = facts.to_dict()
    signs = [item for item in payload.get("signs", []) if isinstance(item, dict) and _targeted(intent, item)]
    values = _query_values(query, intent)
    evaluations: list[RuleEvaluation] = []
    catalog = load_default_sign_catalog()
    for sign in signs:
        # A panel's own facts are modifiers. The compact facts serializer moves
        # linked active_time rows to the primary sign before this evaluation.
        result = _evaluate_sign(intent, sign, values)
        if result is None:
            continue
        entry = catalog.lookup(sign.get("code")) if sign.get("code") else None
        if entry is not None:
            result.legal_basis = list(entry.legal_basis)
        evaluations.append(result)

    relevant = [item for item in evaluations if item.status != "not_applicable"]
    if any(item.decision == "PROHIBITED" and item.status == "applied" for item in relevant):
        result = next(item for item in relevant if item.decision == "PROHIBITED" and item.status == "applied")
        decision, reason = "PROHIBITED", result.reason
    elif any(item.status == "unresolved" or item.decision == "UNKNOWN" for item in relevant):
        decision, reason = "UNKNOWN", "Chưa đủ facts để xác định biển áp dụng và kết luận quyền thực hiện hành vi."
    elif any(item.status == "incomplete" or item.decision == "CONDITIONAL" for item in relevant):
        result = next(item for item in relevant if item.status == "incomplete" or item.decision == "CONDITIONAL")
        decision, reason = "CONDITIONAL", result.reason
    elif relevant and all(item.status == "applied" and item.decision == "ALLOWED" for item in relevant):
        if facts.sign_inventory_status == "complete":
            decision, reason = "ALLOWED", "Các facts áp dụng đã được đối chiếu và không phát hiện giới hạn bị vi phạm."
        else:
            decision, reason = "CONDITIONAL", "Chưa xác nhận đã kiểm kê đầy đủ biển trong cảnh."
            evaluations.append(RuleEvaluation(
                "FACT_COMPARISON", intent.intent_id, decision="CONDITIONAL", status="incomplete",
                missing_facts=["complete_sign_inventory"], reason=reason,
                trace=[{"step": "scene_coverage", "state": "inventory_incomplete"}],
            ))
    else:
        decision, reason = "UNKNOWN", "Facts hiện có không mô tả đủ một giới hạn hoặc điều kiện áp dụng để kết luận."

    missing = list(dict.fromkeys(field for item in relevant for field in item.missing_facts))
    return evaluations, IntentEvaluation(intent.intent_id, decision, evaluations, missing, reason)
