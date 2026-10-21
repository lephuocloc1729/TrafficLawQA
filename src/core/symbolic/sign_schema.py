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
class SymbolicDecision(SerializableMixin):
    decision: Decision = "UNKNOWN"
    conclusion: str = ""
    rules_fired: list[str] = field(default_factory=list)
    intent_evaluations: list[dict[str, Any]] = field(default_factory=list)
    rule_evaluations: list[dict[str, Any]] = field(default_factory=list)
    matched_fact_ids: list[str] = field(default_factory=list)
    legal_basis: list[dict[str, str]] = field(default_factory=list)
    confidence: Confidence = "low"
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sign_meaning_evaluations: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.decision = normalize_decision(self.decision)
        self.conclusion = str(self.conclusion or "").strip()
        self.reason = str(self.reason or "").strip()
        self.confidence = normalize_confidence(self.confidence)
        if not self.rules_fired and self.rule_evaluations:
            self.rules_fired = [
                str(item.get("rule_id"))
                for item in self.rule_evaluations
                if item.get("rule_id") and item.get("status") == "applied"
            ]


@dataclass
class ReasoningTrace(SerializableMixin):
    facts: dict[str, Any] = field(default_factory=dict)
    query_intent: dict[str, Any] = field(default_factory=dict)
    query_context_facts: list[dict[str, Any]] = field(default_factory=list)
    rule_evaluations: list[dict[str, Any]] = field(default_factory=list)
    intent_evaluations: list[dict[str, Any]] = field(default_factory=list)
    decision: SymbolicDecision | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    sign_meaning_evaluations: list[dict[str, Any]] = field(default_factory=list)

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
