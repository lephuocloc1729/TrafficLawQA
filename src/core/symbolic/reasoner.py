from __future__ import annotations

from typing import Any, Iterable

from src.core.symbolic.question_intent import parse_question_intent
from src.core.symbolic.sign_schema import (
    QuestionIntent,
    ReasoningTrace,
    RuleMatch,
    SymbolicDecision,
    TrafficSignFact,
)
from src.core.symbolic.traffic_rules import TRAFFIC_RULES, TrafficRule, find_matching_rules


DECISION_PRIORITY = {
    "PROHIBITED": 0,
    "CONDITIONAL": 1,
    "ALLOWED": 2,
    "UNKNOWN": 3,
}


def _coerce_fact(fact: TrafficSignFact | dict) -> TrafficSignFact:
    if isinstance(fact, TrafficSignFact):
        return fact
    return TrafficSignFact(**fact)


def _coerce_intent(intent: QuestionIntent | dict | str) -> QuestionIntent:
    if isinstance(intent, QuestionIntent):
        return intent
    if isinstance(intent, dict):
        return QuestionIntent(**intent)
    return parse_question_intent(str(intent))


def _dedupe_articles(rule_matches: Iterable[RuleMatch]) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    articles: list[dict[str, str]] = []
    for rule_match in rule_matches:
        for article in rule_match.legal_basis:
            law_id = str(article.get("law_id") or "").strip()
            article_id = str(article.get("article_id") or "").strip()
            key = (law_id, article_id)
            if not law_id or not article_id or key in seen:
                continue
            seen.add(key)
            articles.append(article)
    return articles


def build_symbolic_decision(rule_matches: list[RuleMatch]) -> SymbolicDecision:
    if not rule_matches:
        return SymbolicDecision(
            decision="UNKNOWN",
            conclusion="Không có rule thủ công nào khớp với câu hỏi và biển báo đã chuẩn hóa.",
            confidence="low",
            reason="No manual traffic-law rule fired.",
        )

    selected = sorted(
        rule_matches,
        key=lambda match: DECISION_PRIORITY.get(match.decision, 99),
    )[0]
    matched_fact_ids = sorted(
        {
            fact_id
            for rule_match in rule_matches
            for fact_id in rule_match.matched_fact_ids
        }
    )

    return SymbolicDecision(
        decision=selected.decision,
        conclusion=selected.reason,
        rules_fired=[rule_match.rule_id for rule_match in rule_matches],
        rule_matches=rule_matches,
        matched_fact_ids=matched_fact_ids,
        legal_basis=_dedupe_articles(rule_matches),
        confidence=selected.confidence,
        reason=selected.reason,
    )


def reason_over_facts(
    facts: Iterable[TrafficSignFact | dict],
    question_intent: QuestionIntent | dict | str,
    rules: Iterable[TrafficRule] = TRAFFIC_RULES,
) -> ReasoningTrace:
    coerced_facts = [_coerce_fact(fact) for fact in facts]
    coerced_intent = _coerce_intent(question_intent)
    rule_list = tuple(rules)
    rule_matches = find_matching_rules(
        facts=coerced_facts,
        intent=coerced_intent,
        rules=rule_list,
    )
    decision = build_symbolic_decision(rule_matches)
    trace = ReasoningTrace(
        facts=coerced_facts,
        question_intent=coerced_intent,
        rule_matches=rule_matches,
        decision=decision,
    )
    trace.add_step(
        "normalize_inputs",
        inputs={
            "num_facts": len(coerced_facts),
            "intent": coerced_intent.intent,
            "action": coerced_intent.action,
            "vehicle": coerced_intent.vehicle,
        },
        output={"candidate_sign_codes": [fact.sign_code for fact in coerced_facts]},
    )
    trace.add_step(
        "apply_manual_rules",
        inputs={"rule_count": len(rule_list)},
        output={
            "rules_fired": decision.rules_fired,
            "decision": decision.decision,
            "confidence": decision.confidence,
        },
    )
    return trace


def decide_symbolically(
    facts: Iterable[TrafficSignFact | dict],
    question_intent: QuestionIntent | dict | str,
    rules: Iterable[TrafficRule] = TRAFFIC_RULES,
) -> SymbolicDecision:
    return reason_over_facts(
        facts=facts,
        question_intent=question_intent,
        rules=rules,
    ).decision or SymbolicDecision()


def build_symbolic_reasoning_context(
    symbolic_facts: Iterable[TrafficSignFact | dict],
    question: str,
    llm_client: Any | None = None,
    use_llm_fallback: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 300,
) -> dict[str, Any]:
    question_intent = parse_question_intent(
        question=question,
        llm_client=llm_client,
        use_llm_fallback=use_llm_fallback,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    reasoning_trace = reason_over_facts(
        facts=symbolic_facts,
        question_intent=question_intent,
    )
    symbolic_decision = reasoning_trace.decision or SymbolicDecision()
    return {
        "symbolic_facts": [fact.to_dict() for fact in reasoning_trace.facts],
        "question_intent": question_intent.to_dict(),
        "rules_fired": symbolic_decision.rules_fired,
        "symbolic_decision": symbolic_decision.to_dict(),
        "reasoning_trace": reasoning_trace.to_dict(),
    }
