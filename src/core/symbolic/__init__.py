"""Typed intent, fact and rule-engine primitives for traffic-law reasoning."""

from src.core.symbolic.fact_schema import ContextFact, SignObservation, SignRelation, TrafficFacts
from src.core.symbolic.intent_schema import (
    Intent,
    IntentParseResult,
    IntentSlot,
    IntentTarget,
    QueryIntent,
)
from src.core.symbolic.question_intent import (
    VALID_ACTIONS,
    VALID_INTENTS,
    VALID_VEHICLES,
    extract_action,
    extract_time_context,
    extract_vehicle,
    normalize_question_text,
    parse_query_intent,
    parse_query_intent_deterministic,
    parse_query_intent_with_llm,
)
from src.core.symbolic.reasoner import (
    build_symbolic_decision,
    build_symbolic_reasoning_context,
    decide_symbolically,
    reason_over_facts,
)
from src.core.symbolic.rule_schema import IntentEvaluation, RuleCondition, RuleEvaluation, TrafficRule
from src.core.symbolic.sign_normalizer import (
    extract_sign_code,
    normalize_detected_sign_to_observation,
    normalize_payload_to_observation,
    normalize_payloads_to_observations,
    normalize_sign_code,
    sign_family,
)
from src.core.symbolic.sign_schema import ReasoningTrace, SymbolicDecision
from src.core.symbolic.traffic_rules import TRAFFIC_RULES, find_matching_rules, vehicle_is_covered

__all__ = [
    "ContextFact", "Intent", "IntentEvaluation", "IntentParseResult", "IntentSlot", "IntentTarget",
    "QueryIntent", "ReasoningTrace", "RuleCondition", "RuleEvaluation",
    "SignObservation", "SignRelation", "SymbolicDecision", "TRAFFIC_RULES",
    "TrafficFacts", "TrafficRule", "VALID_ACTIONS", "VALID_INTENTS", "VALID_VEHICLES",
    "build_symbolic_decision", "build_symbolic_reasoning_context", "decide_symbolically",
    "extract_action", "extract_sign_code", "extract_time_context", "extract_vehicle",
    "find_matching_rules", "normalize_detected_sign_to_observation", "normalize_payload_to_observation",
    "normalize_payloads_to_observations", "normalize_question_text", "normalize_sign_code",
    "parse_query_intent", "parse_query_intent_deterministic",
    "parse_query_intent_with_llm", "reason_over_facts", "sign_family",
    "vehicle_is_covered",
]
