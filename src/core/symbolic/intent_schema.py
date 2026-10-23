from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.symbolic.fact_schema import ContextFact
from src.core.symbolic.sign_schema import SerializableMixin


IntentKind = Literal["CHECK_PERMISSION", "INFORMATION", "UNKNOWN"]
VALID_INTENT_KINDS = frozenset({
    "CHECK_PERMISSION", "INFORMATION", "UNKNOWN",
})
LEGACY_INTENT_KIND_ALIASES = {
    "QUERY_SIGN": "INFORMATION",
    "QUERY_SIGN_MEANING": "INFORMATION",
    "IDENTIFY_SIGN": "INFORMATION",
    "ASK_CONDITION": "CHECK_PERMISSION",
    "PENALTY": "UNKNOWN",
    "ASK_PENALTY": "UNKNOWN",
    "QUERY_PROHIBITED_SUBJECTS": "INFORMATION",
    "QUERY_TIME_CONDITION": "INFORMATION",
    "QUERY_DIRECTION_SCOPE": "INFORMATION",
    "QUERY_DISTANCE_SCOPE": "INFORMATION",
}


def normalize_intent_kind(value: str | None) -> IntentKind:
    kind = str(value or "UNKNOWN").strip().upper()
    kind = LEGACY_INTENT_KIND_ALIASES.get(kind, kind)
    if kind not in VALID_INTENT_KINDS:
        kind = "UNKNOWN"
    return kind  # type: ignore[return-value]


@dataclass
class IntentSlot(SerializableMixin):
    value: str = "unknown"
    raw_text: str | None = None
    status: str = "not_stated"

    def __post_init__(self) -> None:
        self.value = str(self.value or "unknown").strip().lower()
        self.status = str(self.status or "not_stated").strip().lower()
        if self.raw_text is not None:
            self.raw_text = str(self.raw_text)


@dataclass
class IntentTarget(SerializableMixin):
    kind: str = "unresolved"
    observation_ids: list[str] = field(default_factory=list)
    sign_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.kind = str(self.kind or "unresolved").strip().lower()
        if self.kind not in {"scene", "sign_observation", "sign_code", "unresolved"}:
            self.kind = "unresolved"
        self.observation_ids = [str(item) for item in self.observation_ids if item]
        self.sign_codes = [str(item) for item in self.sign_codes if item]


@dataclass
class Intent(SerializableMixin):
    intent_id: str
    kind: IntentKind
    target: IntentTarget = field(default_factory=IntentTarget)
    action: IntentSlot = field(default_factory=IntentSlot)
    actor: IntentSlot = field(default_factory=IntentSlot)
    requested_answer: dict[str, Any] = field(default_factory=dict)
    information_type: str | None = None
    requested_fact: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.intent_id = str(self.intent_id or "intent-1").strip()
        legacy_kind = str(self.kind or "UNKNOWN").strip().upper()
        self.kind = normalize_intent_kind(self.kind)
        if legacy_kind in {"QUERY_SIGN", "QUERY_SIGN_MEANING", "IDENTIFY_SIGN"} and not self.information_type:
            self.information_type = "sign_meaning"
        if isinstance(self.target, str):
            self.target = IntentTarget(kind=self.target)
        if isinstance(self.target, dict):
            self.target = IntentTarget(**self.target)
        if isinstance(self.action, str):
            self.action = IntentSlot(self.action, status="provided" if self.action != "unknown" else "not_stated")
        if isinstance(self.action, dict):
            self.action = IntentSlot(**{
                key: value for key, value in self.action.items()
                if key in {"value", "raw_text", "status"}
            })
        if isinstance(self.actor, str):
            self.actor = IntentSlot(self.actor, status="provided" if self.actor != "unknown" else "not_stated")
        if isinstance(self.actor, dict):
            self.actor = IntentSlot(**{
                key: value for key, value in self.actor.items()
                if key in {"value", "raw_text", "status"}
            })
        self.requested_answer = dict(self.requested_answer or {})
        self.information_type = str(self.information_type).strip().lower() if self.information_type else None
        self.requested_fact = str(self.requested_fact).strip().lower() if self.requested_fact else None
        self.filters = dict(self.filters or {}) if isinstance(self.filters, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        """Expose the compact intent contract while retaining richer runtime slots."""
        payload: dict[str, Any] = {"kind": self.kind}
        if self.kind == "CHECK_PERMISSION":
            if self.action.value != "unknown":
                payload["action"] = self.action.value
            if self.actor.value != "unknown":
                payload["actor"] = self.actor.value
            target = self._compact_target()
            if target is not None:
                payload["target"] = target
            requested_values = {
                key: value for key, value in self.requested_answer.items()
                if key.startswith("requested_") and value is not None
            }
            if requested_values:
                payload["requested_answer"] = requested_values
        elif self.kind == "INFORMATION":
            if self.information_type:
                payload["information_type"] = self.information_type
            if self.requested_fact:
                payload["requested_fact"] = self.requested_fact
            if self.filters:
                payload["filters"] = self.filters
            target = self._compact_target()
            if target is not None:
                payload["target"] = target
        return payload

    def _compact_target(self) -> str | dict[str, Any] | None:
        if self.target.kind == "sign_code" and self.target.sign_codes:
            return {"kind": "sign_code", "sign_codes": self.target.sign_codes}
        if self.target.kind == "sign_observation" and self.target.observation_ids:
            return {"kind": "sign_observation", "observation_ids": self.target.observation_ids}
        if self.target.kind == "unresolved":
            return None
        return "scene" if self.target.kind in {"scene", "sign_observation", "sign_code"} else self.target.kind


@dataclass
class QueryIntent(SerializableMixin):
    """What the user asks, independent from facts about the traffic scene."""

    schema_version: str
    query_id: str
    raw_query: str
    intents: list[Intent] = field(default_factory=list)
    parser: dict[str, Any] = field(default_factory=dict)
    unresolved_items: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or "3.0").strip()
        self.query_id = str(self.query_id or "query-unknown").strip()
        self.raw_query = str(self.raw_query or "").strip()
        self.intents = [
            item if isinstance(item, Intent) else Intent(**{
                "intent_id": item.get("intent_id") or f"intent-{index + 1}",
                **{
                    key: value for key, value in item.items()
                    if key in {
                        "kind", "target", "action", "actor", "requested_answer",
                        "information_type", "requested_fact", "filters",
                    }
                },
            })
            for index, item in enumerate(self.intents)
            if isinstance(item, (Intent, dict))
        ]
        self.parser = dict(self.parser or {}) if isinstance(self.parser, dict) else {}
        self.unresolved_items = [str(item) for item in self.unresolved_items]
        intent_ids = [item.intent_id for item in self.intents]
        if len(intent_ids) != len(set(intent_ids)):
            self.unresolved_items.append("duplicate_intent_id")
        self.unresolved_items = list(dict.fromkeys(self.unresolved_items))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question": self.raw_query,
            "intents": [item.to_dict() for item in self.intents],
        }


@dataclass
class IntentParseResult(SerializableMixin):
    """Parser output keeps user intent and query-derived facts as separate collections."""

    schema_version: str
    query_intent: QueryIntent
    context_facts: list[ContextFact] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or "3.0").strip()
        if isinstance(self.query_intent, dict):
            self.query_intent = QueryIntent(**self.query_intent)
        self.context_facts = [
            item if isinstance(item, ContextFact) else ContextFact(**item)
            for item in self.context_facts
            if isinstance(item, (ContextFact, dict))
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.query_intent.to_dict(),
            "context_facts": [item.to_dict() for item in self.context_facts],
        }
