"""Symbolic reasoning models and utilities for TrafficLawVQA."""

from src.core.symbolic.sign_schema import (
    QuestionIntent,
    ReasoningTrace,
    RuleMatch,
    SymbolicDecision,
    TrafficSignFact,
)
from src.core.symbolic.sign_normalizer import (
    SIGN_RULE_SCOPE,
    extract_sign_code,
    normalize_detected_sign_to_fact,
    normalize_payload_to_fact,
    normalize_payloads_to_facts,
    normalize_sign_code,
    sign_family,
)
from src.core.symbolic.question_intent import (
    VALID_ACTIONS,
    VALID_INTENTS,
    VALID_VEHICLES,
    detect_intent,
    extract_action,
    extract_vehicle,
    normalize_question_text,
    parse_question_intent,
    parse_question_intent_deterministic,
    parse_question_intent_with_llm,
)

__all__ = [
    "QuestionIntent",
    "ReasoningTrace",
    "RuleMatch",
    "SIGN_RULE_SCOPE",
    "SymbolicDecision",
    "TrafficSignFact",
    "VALID_ACTIONS",
    "VALID_INTENTS",
    "VALID_VEHICLES",
    "detect_intent",
    "extract_action",
    "extract_sign_code",
    "extract_vehicle",
    "normalize_question_text",
    "normalize_detected_sign_to_fact",
    "normalize_payload_to_fact",
    "normalize_payloads_to_facts",
    "normalize_sign_code",
    "parse_question_intent",
    "parse_question_intent_deterministic",
    "parse_question_intent_with_llm",
    "sign_family",
]
