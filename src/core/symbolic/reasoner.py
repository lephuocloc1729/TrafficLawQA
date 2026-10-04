from __future__ import annotations

from typing import Any, Iterable

from src.core.symbolic.fact_schema import SignObservation, TrafficFacts
from src.core.symbolic.intent_schema import IntentParseResult, QueryIntent
from src.core.symbolic.question_intent import parse_query_intent
from src.core.symbolic.rule_schema import IntentEvaluation, RuleEvaluation, TrafficRule
from src.core.symbolic.sign_schema import ReasoningTrace, SymbolicDecision
from src.core.symbolic.traffic_rules import TRAFFIC_RULES, find_matching_rules


def _coerce_facts(facts: TrafficFacts | Iterable[SignObservation | dict[str, Any]]) -> TrafficFacts:
    if isinstance(facts, TrafficFacts):
        return facts
    if isinstance(facts, dict):
        return TrafficFacts(**facts)
    observations = [
        item if isinstance(item, SignObservation) else SignObservation(**item)
        for item in facts
        if isinstance(item, (SignObservation, dict))
    ]
    return TrafficFacts(schema_version="2.0", scene_id="scene-1", sign_observations=observations)


def _coerce_query(query: IntentParseResult | QueryIntent | dict[str, Any] | str) -> IntentParseResult:
    if isinstance(query, IntentParseResult):
        return query
    if isinstance(query, QueryIntent):
        return IntentParseResult("2.0", query)
    if isinstance(query, dict):
        return IntentParseResult(**query)
    return parse_query_intent(str(query))


def _evaluate_intent(intent_id: str, evaluations: list[RuleEvaluation]) -> IntentEvaluation:
    relevant = [item for item in evaluations if item.intent_id == intent_id and item.status != "not_applicable"]
    if not relevant:
        return IntentEvaluation(intent_id, "UNKNOWN", [], ["no_matching_rule"], "Không tìm thấy rule phù hợp với dữ kiện đã xác minh.")
    decisions = {item.decision for item in relevant}
    missing = sorted({field for item in relevant for field in item.missing_facts})
    if "PROHIBITED" in decisions and "ALLOWED" in decisions:
        decision = "CONDITIONAL"
        reason = "Các rule phù hợp đưa ra kết quả đối nghịch; cần kiểm tra phạm vi hoặc biển phụ."
        missing.append("conflicting_rule_outcomes")
    elif "PROHIBITED" in decisions:
        decision = "PROHIBITED"
        reason = next(item.reason for item in relevant if item.decision == decision)
    elif "CONDITIONAL" in decisions:
        decision = "CONDITIONAL"
        reason = next(item.reason for item in relevant if item.decision == decision)
    elif "ALLOWED" in decisions:
        decision = "ALLOWED"
        reason = next(item.reason for item in relevant if item.decision == decision)
    else:
        decision = "UNKNOWN"
        reason = next((item.reason for item in relevant if item.reason), "Chưa đủ dữ kiện.")
    return IntentEvaluation(intent_id, decision, relevant, missing, reason)


def build_symbolic_decision(
    intent_evaluations: list[IntentEvaluation],
    rule_evaluations: list[RuleEvaluation],
) -> SymbolicDecision:
    if not intent_evaluations:
        return SymbolicDecision(decision="UNKNOWN", conclusion="Không có intent để đánh giá.")
    decisions = {item.decision for item in intent_evaluations}
    if len(intent_evaluations) > 1:
        # Keep separate answers per requested task; only collapse when they agree.
        decision = next(iter(decisions)) if len(decisions) == 1 else "UNKNOWN"
        conclusion = "Một hoặc nhiều yêu cầu cần được trả lời riêng; xem intent_evaluations."
    else:
        decision = intent_evaluations[0].decision
        conclusion = intent_evaluations[0].reason
    fired = [
        item.rule_id for item in rule_evaluations
        if item.status in {"applied", "incomplete"}
    ]
    unique_fired = list(dict.fromkeys(fired))
    basis, seen = [], set()
    for evaluation in rule_evaluations:
        if evaluation.status not in {"applied", "incomplete", "unresolved"}:
            continue
        for ref in evaluation.legal_basis:
            key = (ref.get("law_id"), ref.get("article_id"), ref.get("document"), ref.get("section"))
            if key not in seen:
                seen.add(key)
                basis.append(ref)
    fact_ids = list(dict.fromkeys(
        fact_id for evaluation in rule_evaluations for fact_id in evaluation.matched_facts
    ))
    unresolved = [note for result in intent_evaluations for note in result.unresolved_items]
    applied_confidences = [
        item.confidence for item in rule_evaluations
        if item.status in {"applied", "incomplete"}
    ]
    confidence = "low"
    if (
        decision in {"ALLOWED", "PROHIBITED"}
        and applied_confidences
        and all(value == "high" for value in applied_confidences)
        and not unresolved
    ):
        confidence = "high"
    elif applied_confidences and all(value != "low" for value in applied_confidences):
        confidence = "medium"
    return SymbolicDecision(
        decision=decision,
        conclusion=conclusion,
        rules_fired=unique_fired,
        intent_evaluations=[item.to_dict() for item in intent_evaluations],
        rule_evaluations=[item.to_dict() for item in rule_evaluations],
        matched_fact_ids=fact_ids,
        legal_basis=basis,
        confidence=confidence,
        reason="; ".join(dict.fromkeys(result.reason for result in intent_evaluations if result.reason)),
        metadata={"unresolved_items": unresolved},
    )


def reason_over_facts(
    facts: TrafficFacts | Iterable[SignObservation | dict[str, Any]],
    query_intent: IntentParseResult | QueryIntent | dict[str, Any] | str,
    rules: Iterable[TrafficRule] = TRAFFIC_RULES,
) -> ReasoningTrace:
    fact_set = _coerce_facts(facts)
    query = _coerce_query(query_intent)
    rule_list = tuple(rules)
    evaluations = find_matching_rules(
        fact_set,
        query.query_intent,
        query.context_facts,
        rules=rule_list,
    )
    intent_results = [
        _evaluate_intent(intent.intent_id, evaluations)
        for intent in query.query_intent.intents
    ]
    decision = build_symbolic_decision(intent_results, evaluations)
    trace = ReasoningTrace(
        facts=fact_set.to_dict(),
        query_intent=query.query_intent.to_dict(),
        query_context_facts=[item.to_dict() for item in query.context_facts],
        rule_evaluations=[item.to_dict() for item in evaluations],
        intent_evaluations=[item.to_dict() for item in intent_results],
        decision=decision,
    )
    trace.add_step(
        "normalize_inputs",
        inputs={"observation_count": len(fact_set.sign_observations),
                "intent_count": len(query.query_intent.intents),
                "context_fact_count": len([*fact_set.context_facts, *query.context_facts])},
        output={"sign_codes": [item.sign_code for item in fact_set.sign_observations]},
    )
    trace.add_step(
        "evaluate_rules",
        inputs={"rule_count": len(rule_list)},
        output={"evaluations": [item.to_dict() for item in evaluations]},
    )
    trace.add_step(
        "aggregate_intents",
        inputs={"intent_results": [item.to_dict() for item in intent_results]},
        output={"decision": decision.decision, "reason": decision.reason},
    )
    return trace


def decide_symbolically(
    facts: TrafficFacts | Iterable[SignObservation | dict[str, Any]],
    query_intent: IntentParseResult | QueryIntent | dict[str, Any] | str,
    rules: Iterable[TrafficRule] = TRAFFIC_RULES,
) -> SymbolicDecision:
    return reason_over_facts(facts, query_intent, rules).decision or SymbolicDecision()


def build_symbolic_reasoning_context(
    traffic_facts: TrafficFacts | Iterable[SignObservation | dict[str, Any]],
    question: str,
    llm_client: Any | None = None,
    use_llm_fallback: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict[str, Any]:
    query = parse_query_intent(
        question=question,
        llm_client=llm_client,
        use_llm_fallback=use_llm_fallback,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    fact_set = _coerce_facts(traffic_facts)
    trace = reason_over_facts(fact_set, query)
    decision = trace.decision or SymbolicDecision()
    return {
        "facts": fact_set.to_dict(),
        "query_intent": query.query_intent.to_dict(),
        "query_context_facts": [item.to_dict() for item in query.context_facts],
        "intent_evaluations": decision.intent_evaluations,
        "rule_evaluations": decision.rule_evaluations,
        "rules_fired": decision.rules_fired,
        "symbolic_decision": decision.to_dict(),
        "reasoning_trace": trace.to_dict(),
    }
