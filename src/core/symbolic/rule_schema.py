from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.symbolic.sign_schema import Decision, SerializableMixin, normalize_confidence, normalize_decision


@dataclass
class RuleCondition(SerializableMixin):
    predicate: str
    operator: str
    expected_value: Any = None
    required: bool = True
    on_missing: str = "CONDITIONAL"
    on_mismatch: str = "NOT_APPLICABLE"

    def __post_init__(self) -> None:
        self.predicate = str(self.predicate or "unknown").strip().lower()
        self.operator = str(self.operator or "equals").strip().lower()
        if self.operator not in {
            "equals", "not_equals", "exists", "contains", "gte", "lte", "gt", "lt",
            "in", "not_in", "between", "equals_fact", "lte_fact", "gte_fact", "gt_fact", "lt_fact",
            "time_window_contains_now", "axle_group_loads_within_limits",
            "s505b_gross_weight_within_selected_row",
        }:
            self.operator = "equals"
        self.on_missing = normalize_decision(self.on_missing)
        if self.on_missing == "ALLOWED":
            self.on_missing = "CONDITIONAL"
        self.on_mismatch = str(self.on_mismatch or "NOT_APPLICABLE").strip().upper()
        if self.on_mismatch not in {"PROHIBITED", "CONDITIONAL", "UNKNOWN", "NOT_APPLICABLE"}:
            self.on_mismatch = "NOT_APPLICABLE"


@dataclass
class RuleEffect(SerializableMixin):
    """A reusable legal consequence, separate from conditions and query wording."""

    effect_type: str
    action: str | None = None
    value: Any = None
    unit: str | None = None
    value_predicate: str | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        self.effect_type = str(self.effect_type or "unknown").strip().lower()
        if self.effect_type not in {
            "prohibit_action", "permit_action", "require_action", "maximum", "minimum",
            "exact_value", "allow_direction", "sign_meaning", "unknown",
        }:
            self.effect_type = "unknown"
        self.action = str(self.action).strip().lower() if self.action else None
        self.unit = str(self.unit).strip().lower() if self.unit else None
        self.value_predicate = str(self.value_predicate).strip().lower() if self.value_predicate else None
        self.scope = dict(self.scope) if isinstance(self.scope, dict) else {}
        self.description = str(self.description or "").strip()


@dataclass
class TrafficRule(SerializableMixin):
    rule_id: str
    legal_basis: list[dict[str, Any]]
    trigger_sign_codes: list[str]
    actions: list[str]
    actor_scope: list[str]
    outcome: Decision
    conditions: list[RuleCondition] = field(default_factory=list)
    reason: str = ""
    verification_status: str = "unverified"
    effects: list[RuleEffect] = field(default_factory=list)
    excluded_actor_scope: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rule_id = str(self.rule_id or "").strip()
        self.trigger_sign_codes = [str(x).strip() for x in self.trigger_sign_codes]
        self.actions = [str(x).strip().lower() for x in self.actions]
        self.actor_scope = [str(x).strip().lower() for x in self.actor_scope]
        self.excluded_actor_scope = [str(x).strip().lower() for x in self.excluded_actor_scope]
        self.outcome = normalize_decision(self.outcome)
        self.conditions = [
            condition if isinstance(condition, RuleCondition) else RuleCondition(**condition)
            for condition in self.conditions
            if isinstance(condition, (RuleCondition, dict))
        ]
        self.effects = [
            effect if isinstance(effect, RuleEffect) else RuleEffect(**effect)
            for effect in self.effects
            if isinstance(effect, (RuleEffect, dict))
        ]
        if self.outcome == "ALLOWED" and not any(
            effect.effect_type in {"permit_action", "allow_direction"}
            for effect in self.effects
        ):
            raise ValueError(
                f"Rule {self.rule_id!r} cannot return ALLOWED without an explicit "
                "permit_action or allow_direction effect"
            )
        required_effect_facts = {
            effect.value_predicate for effect in self.effects if effect.value_predicate
        }
        declared_condition_facts = {
            condition.predicate for condition in self.conditions if condition.required
        }
        undeclared = required_effect_facts - declared_condition_facts
        if undeclared:
            raise ValueError(
                f"Rule {self.rule_id!r} has value-dependent effects without required conditions: "
                f"{', '.join(sorted(undeclared))}"
            )
        self.reason = str(self.reason or "").strip()
        self.legal_basis = [item for item in self.legal_basis if isinstance(item, dict)]
        self.verification_status = str(self.verification_status or "unverified").strip().lower()
        if self.verification_status not in {"verified", "unverified", "draft", "superseded"}:
            self.verification_status = "unverified"


@dataclass
class RuleEvaluation(SerializableMixin):
    rule_id: str
    intent_id: str
    observation_ids: list[str] = field(default_factory=list)
    decision: Decision = "UNKNOWN"
    status: Literal["applied", "incomplete", "unresolved", "not_applicable"] = "unresolved"
    matched_facts: list[str] = field(default_factory=list)
    missing_facts: list[str] = field(default_factory=list)
    legal_basis: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""
    confidence: str = "low"
    trace: list[dict[str, Any]] = field(default_factory=list)
    verification_status: str = "unverified"
    effects: list[RuleEffect] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.decision = normalize_decision(self.decision)
        self.confidence = normalize_confidence(self.confidence)
        self.verification_status = str(self.verification_status or "unverified").strip().lower()
        self.effects = [
            effect if isinstance(effect, RuleEffect) else RuleEffect(**effect)
            for effect in self.effects
            if isinstance(effect, (RuleEffect, dict))
        ]


@dataclass
class IntentEvaluation(SerializableMixin):
    intent_id: str
    decision: Decision = "UNKNOWN"
    rule_evaluations: list[RuleEvaluation] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    reason: str = ""

    def __post_init__(self) -> None:
        self.decision = normalize_decision(self.decision)
