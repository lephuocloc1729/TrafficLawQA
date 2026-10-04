from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from src.core.symbolic.fact_schema import ContextFact, SignObservation, TrafficFacts
from src.core.symbolic.intent_schema import Intent, QueryIntent
from src.core.symbolic.rule_schema import RuleCondition, RuleEvaluation, TrafficRule


MOTOR_VEHICLES = {"car", "motorbike", "truck", "bus", "motor_vehicle"}
VEHICLES = {*MOTOR_VEHICLES, "bicycle", "vehicle"}
PERMISSION_INTENTS = {"CHECK_PERMISSION"}


def vehicle_is_covered(vehicle: str, target_vehicle: str) -> bool:
    vehicle = (vehicle or "unknown").strip().lower()
    target_vehicle = (target_vehicle or "unknown").strip().lower()
    if target_vehicle == "unknown" or vehicle == "unknown":
        return True
    if target_vehicle == "motor_vehicle":
        return vehicle in MOTOR_VEHICLES
    if target_vehicle == "vehicle":
        return vehicle in VEHICLES
    return vehicle == target_vehicle


def _basis(article: str) -> list[dict[str, Any]]:
    # These legacy article references have not been independently validated here.
    return [{"law_id": "QCVN 41:2024/BGTVT", "article_id": article, "verification_status": "unverified"}]


TRAFFIC_RULES: tuple[TrafficRule, ...] = (
    TrafficRule("P130_NO_STOPPING_NO_PARKING", _basis("30"), ["P.130"], ["CHECK_PERMISSION"],
                ["parking", "stopping"], ["motor_vehicle"], "PROHIBITED",
                reason="Biển P.130 cấm dừng xe và đỗ xe trong phạm vi hiệu lực của biển."),
    TrafficRule("P131A_NO_PARKING", _basis("31"), ["P.131a"], ["CHECK_PERMISSION"],
                ["parking"], ["motor_vehicle"], "PROHIBITED",
                reason="Biển P.131a cấm đỗ xe trong phạm vi hiệu lực của biển."),
    TrafficRule("P131B_ODD_DAY_NO_PARKING", _basis("31"), ["P.131b"], ["CHECK_PERMISSION"],
                ["parking"], ["motor_vehicle"], "PROHIBITED",
                [RuleCondition("calendar_parity", "equals", "odd_days", on_mismatch="ALLOWED")],
                "Biển P.131b cấm đỗ xe vào ngày lẻ."),
    TrafficRule("P131C_EVEN_DAY_NO_PARKING", _basis("31"), ["P.131c"], ["CHECK_PERMISSION"],
                ["parking"], ["motor_vehicle"], "PROHIBITED",
                [RuleCondition("calendar_parity", "equals", "even_days", on_mismatch="ALLOWED")],
                "Biển P.131c cấm đỗ xe vào ngày chẵn."),
    TrafficRule("P123A_NO_LEFT_TURN", _basis("30"), ["P.123a"], ["CHECK_PERMISSION"],
                ["turn_left"], ["motor_vehicle"], "PROHIBITED",
                reason="Biển P.123a cấm rẽ trái trong phạm vi hiệu lực của biển."),
    TrafficRule("P123B_NO_RIGHT_TURN", _basis("30"), ["P.123b"], ["CHECK_PERMISSION"],
                ["turn_right"], ["motor_vehicle"], "PROHIBITED",
                reason="Biển P.123b cấm rẽ phải trong phạm vi hiệu lực của biển."),
    TrafficRule("P102_NO_ENTRY", _basis("30"), ["P.102"], ["CHECK_PERMISSION"],
                ["entering", "circulation"], ["vehicle"], "PROHIBITED",
                reason="Biển P.102 cấm phương tiện đi vào theo chiều đặt biển."),
    TrafficRule("P127_SPEED_LIMIT", _basis("30"), ["P.127"], ["ASK_CONDITION", "CHECK_PERMISSION"],
                ["speed"], ["motor_vehicle"], "CONDITIONAL",
                [RuleCondition("speed_limit_value", "exists", None, on_missing="CONDITIONAL")],
                "Biển P.127 quy định tốc độ tối đa; cần đọc trị số trên biển hoặc biển phụ."),
)


def _observation_context_facts(
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
        "exceptions": "supplementary_exceptions",
    }
    result: list[ContextFact] = []
    fields = observation.visual_attributes.get("normalized_conditions", {})
    if isinstance(fields, dict):
        for source_key, predicate in normalized_fields.items():
            if source_key in fields:
                result.append(ContextFact(
                    fact_id=f"{observation.observation_id}:{predicate}",
                    predicate=predicate,
                    value=fields[source_key],
                    status="derived",
                    source=source,
                    confidence=observation.confidence,
                    evidence=observation.evidence,
                    metadata={
                        "subject_observation_id": subject_id,
                        "source_observation_id": observation.observation_id,
                        "relation_id": relation_id,
                    },
                ))
    for index, ocr_item in enumerate(observation.ocr_texts):
        result.append(ContextFact(
            fact_id=f"{observation.observation_id}:ocr:{index}",
            predicate="sign_text" if observation.observation_id == subject_id else "supplementary_text",
            value=ocr_item.get("normalized_value", ocr_item.get("raw_text")),
            status="observed" if ocr_item.get("raw_text") else "unknown",
            source="ocr",
            raw_text=ocr_item.get("raw_text"),
            confidence=observation.confidence,
            evidence=observation.evidence,
            metadata={
                "subject_observation_id": subject_id,
                "source_observation_id": observation.observation_id,
                "relation_id": relation_id,
            },
        ))
    return result


def _supplementary_context(facts: TrafficFacts, main_sign: SignObservation) -> list[ContextFact]:
    observations = {item.observation_id: item for item in facts.sign_observations}
    linked = [
        relation for relation in facts.sign_relations
        if relation.target_observation_id == main_sign.observation_id
        and relation.relation_type in {"supplements", "supplementary_for"}
    ]
    result: list[ContextFact] = []
    for relation in linked:
        supplementary = observations.get(relation.source_observation_id)
        if supplementary is None:
            continue
        result.extend(_observation_context_facts(
            supplementary,
            subject_id=main_sign.observation_id,
            relation_id=relation.relation_id,
            source="supplementary_sign_interpretation",
        ))
    return result


def _context_facts(
    facts: TrafficFacts,
    query_facts: Iterable[ContextFact],
    main_sign: SignObservation | None = None,
) -> list[ContextFact]:
    main_id = main_sign.observation_id if main_sign else None
    context = [
        item for item in facts.context_facts
        if not item.metadata.get("subject_observation_id")
        or item.metadata.get("subject_observation_id") == main_id
    ]
    context.extend(query_facts)
    if main_sign:
        context.extend(_observation_context_facts(main_sign, subject_id=main_sign.observation_id))
        context.extend(_supplementary_context(facts, main_sign))
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
    return context


def _value_for(predicate: str, context: list[ContextFact]) -> tuple[Any, ContextFact | None]:
    for fact in reversed(context):
        if fact.predicate == predicate and fact.status not in {"unknown", "needs_resolution", "ambiguous"}:
            return fact.value, fact
    return None, None


def _condition_result(condition: RuleCondition, context: list[ContextFact]) -> tuple[str, Any, str | None]:
    value, fact = _value_for(condition.predicate, context)
    if fact is None:
        return ("missing" if condition.required else "satisfied", None, None)
    if condition.operator == "exists":
        return ("satisfied" if value not in (None, "", []) else "missing", value, fact.fact_id)
    if condition.operator == "equals":
        return ("satisfied" if value == condition.expected_value else "mismatch", value, fact.fact_id)
    if condition.operator == "not_equals":
        return ("satisfied" if value != condition.expected_value else "mismatch", value, fact.fact_id)
    if condition.operator == "contains":
        try:
            return ("satisfied" if condition.expected_value in value else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    if condition.operator == "in":
        try:
            return ("satisfied" if value in condition.expected_value else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    if condition.operator in {"gte", "lte"}:
        try:
            matched = value >= condition.expected_value if condition.operator == "gte" else value <= condition.expected_value
            return ("satisfied" if matched else "mismatch", value, fact.fact_id)
        except TypeError:
            return "mismatch", value, fact.fact_id
    return "missing", value, fact.fact_id


def _evaluate_candidate(
    rule: TrafficRule,
    intent: Intent,
    observation: SignObservation,
    context: list[ContextFact],
    *,
    identity_ambiguous: bool = False,
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
        )

    missing: list[str] = []
    matched: list[str] = []
    trace: list[dict[str, Any]] = []
    decision = rule.outcome
    for condition in rule.conditions:
        state, value, fact_id = _condition_result(condition, context)
        trace.append({"predicate": condition.predicate, "state": state, "value": value})
        if state == "satisfied":
            if fact_id:
                matched.append(fact_id)
        elif state == "missing":
            if condition.required:
                missing.append(condition.predicate)
                decision = condition.on_missing  # type: ignore[assignment]
        elif state == "mismatch":
            if condition.on_mismatch == "NOT_APPLICABLE":
                return RuleEvaluation(
                    rule_id=rule.rule_id, intent_id=intent.intent_id,
                    observation_ids=[observation.observation_id], decision="UNKNOWN",
                    status="not_applicable", matched_facts=matched,
                    legal_basis=rule.legal_basis,
                    reason="Điều kiện của rule không khớp tình huống đã biết.", trace=trace,
                )
            decision = condition.on_mismatch  # type: ignore[assignment]
            if fact_id:
                matched.append(fact_id)

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
        reason=rule.reason if not missing else f"{rule.reason} Thiếu dữ kiện: {', '.join(missing)}.",
        confidence=(
            "low" if incomplete else "medium"
            if rule.verification_status != "verified"
            or any(ref.get("verification_status") != "verified" for ref in rule.legal_basis)
            else "high"
        ),
        trace=trace,
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
        action = query_intent.action.value
        actor = query_intent.actor.value
        for observation in facts.sign_observations:
            context = _context_facts(facts, scenario_facts, observation)
            if query_intent.target.kind == "sign_observation" and query_intent.target.observation_ids:
                if observation.observation_id not in query_intent.target.observation_ids:
                    continue
            if query_intent.target.kind == "sign_code" and query_intent.target.sign_codes:
                candidate_codes = {str(item.get("sign_code") or "") for item in observation.candidates}
                if observation.sign_code not in query_intent.target.sign_codes and not candidate_codes.intersection(query_intent.target.sign_codes):
                    continue
            for rule in rules:
                if query_intent.kind not in rule.intent_kinds or action not in rule.actions:
                    continue
                if actor != "unknown" and not any(vehicle_is_covered(actor, scope) for scope in rule.actor_scope):
                    continue
                code = observation.sign_code
                if code in rule.trigger_sign_codes and observation.identity_status == "identified":
                    matches.append(_evaluate_candidate(rule, query_intent, observation, context))
                    continue
                candidate_codes = {str(item.get("sign_code") or "") for item in observation.candidates}
                if observation.identity_status == "ambiguous" and candidate_codes.intersection(rule.trigger_sign_codes):
                    matches.append(_evaluate_candidate(rule, query_intent, observation, context, identity_ambiguous=True))
    return sorted(matches, key=lambda item: (item.intent_id, item.rule_id, item.observation_ids))
