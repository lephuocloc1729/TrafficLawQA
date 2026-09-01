from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Literal


Confidence = Literal["high", "medium", "low"]
Decision = Literal["ALLOWED", "PROHIBITED", "CONDITIONAL", "UNKNOWN"]

VALID_CONFIDENCE: set[str] = {"high", "medium", "low"}
VALID_DECISIONS: set[str] = {"ALLOWED", "PROHIBITED", "CONDITIONAL", "UNKNOWN"}


def normalize_confidence(value: str | None) -> Confidence:
    normalized = str(value or "low").strip().lower()
    if normalized not in VALID_CONFIDENCE:
        return "low"
    return normalized  # type: ignore[return-value]


def normalize_decision(value: str | None) -> Decision:
    normalized = str(value or "UNKNOWN").strip().upper()
    if normalized not in VALID_DECISIONS:
        return "UNKNOWN"
    return normalized  # type: ignore[return-value]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, SerializableMixin):
        return value.to_dict()
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


class SerializableMixin:
    """Small helper for stable JSON output from symbolic dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class TrafficSignFact(SerializableMixin):
    sign_code: str
    sign_family: str
    sign_name: str = ""
    restriction_type: str = "unknown"
    target_vehicle: str = "unknown"
    time_condition: str | None = None
    direction_scope: str | None = None
    distance_scope: str | None = None
    related_articles: list[dict[str, str]] = field(default_factory=list)
    visual_evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: Confidence = "low"
    source: str = "retrieval"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sign_code = str(self.sign_code or "").strip()
        self.sign_family = str(self.sign_family or "").strip()
        self.sign_name = str(self.sign_name or "").strip()
        self.restriction_type = str(self.restriction_type or "unknown").strip()
        self.target_vehicle = str(self.target_vehicle or "unknown").strip()
        self.confidence = normalize_confidence(self.confidence)
        self.source = str(self.source or "retrieval").strip()


@dataclass
class QuestionIntent(SerializableMixin):
    intent: str = "UNKNOWN"
    action: str = "unknown"
    vehicle: str = "unknown"
    time_context: str | None = None
    location_context: str | None = None
    raw_question: str = ""
    confidence: Confidence = "low"
    source: str = "parser"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intent = str(self.intent or "UNKNOWN").strip().upper()
        self.action = str(self.action or "unknown").strip()
        self.vehicle = str(self.vehicle or "unknown").strip()
        self.raw_question = str(self.raw_question or "").strip()
        self.confidence = normalize_confidence(self.confidence)
        self.source = str(self.source or "parser").strip()


@dataclass
class RuleMatch(SerializableMixin):
    rule_id: str
    decision: Decision
    matched_fact_ids: list[str] = field(default_factory=list)
    legal_basis: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""
    confidence: Confidence = "low"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rule_id = str(self.rule_id or "").strip()
        self.decision = normalize_decision(self.decision)
        self.reason = str(self.reason or "").strip()
        self.confidence = normalize_confidence(self.confidence)


@dataclass
class SymbolicDecision(SerializableMixin):
    decision: Decision = "UNKNOWN"
    conclusion: str = ""
    rules_fired: list[str] = field(default_factory=list)
    rule_matches: list[RuleMatch] = field(default_factory=list)
    matched_fact_ids: list[str] = field(default_factory=list)
    legal_basis: list[dict[str, str]] = field(default_factory=list)
    confidence: Confidence = "low"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.decision = normalize_decision(self.decision)
        self.conclusion = str(self.conclusion or "").strip()
        self.reason = str(self.reason or "").strip()
        self.confidence = normalize_confidence(self.confidence)
        if not self.rules_fired and self.rule_matches:
            self.rules_fired = [match.rule_id for match in self.rule_matches]


@dataclass
class ReasoningTrace(SerializableMixin):
    facts: list[TrafficSignFact] = field(default_factory=list)
    question_intent: QuestionIntent | None = None
    rule_matches: list[RuleMatch] = field(default_factory=list)
    decision: SymbolicDecision | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        name: str,
        inputs: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        self.steps.append(
            {
                "name": name,
                "inputs": inputs or {},
                "output": output or {},
            }
        )
