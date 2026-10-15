from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from src.core.symbolic.fact_schema import ContextFact
from src.core.symbolic.sign_schema import Confidence, SerializableMixin, normalize_confidence


IntentKind = Literal["CHECK_PERMISSION", "QUERY_SIGN", "INFORMATION", "PENALTY", "UNKNOWN"]
VALID_INTENT_KINDS = frozenset({
    "CHECK_PERMISSION", "QUERY_SIGN", "INFORMATION", "PENALTY", "UNKNOWN",
})
LEGACY_INTENT_KIND_ALIASES = {
    "QUERY_SIGN_MEANING": "QUERY_SIGN",
    "IDENTIFY_SIGN": "QUERY_SIGN",
    "ASK_CONDITION": "CHECK_PERMISSION",
    "ASK_PENALTY": "PENALTY",
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
    source: str = "user_query"
    confidence: float | None = None

    def __post_init__(self) -> None:
        self.value = str(self.value or "unknown").strip().lower()
        self.status = str(self.status or "not_stated").strip().lower()
        self.source = str(self.source or "user_query").strip().lower()
        if self.raw_text is not None:
            self.raw_text = str(self.raw_text)
        if self.confidence is not None:
            try:
                self.confidence = max(0.0, min(1.0, float(self.confidence)))
            except (TypeError, ValueError):
                self.confidence = None


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
    confidence: Confidence = "low"
    unresolved_items: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.intent_id = str(self.intent_id or "intent-1").strip()
        self.kind = normalize_intent_kind(self.kind)
        if isinstance(self.target, dict):
            self.target = IntentTarget(**self.target)
        if isinstance(self.action, dict):
            self.action = IntentSlot(**self.action)
        if isinstance(self.actor, dict):
            self.actor = IntentSlot(**self.actor)
        self.confidence = normalize_confidence(self.confidence)
        self.unresolved_items = [str(item) for item in self.unresolved_items]


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
        self.schema_version = str(self.schema_version or "1.0").strip()
        self.query_id = str(self.query_id or "query-unknown").strip()
        self.raw_query = str(self.raw_query or "").strip()
        self.intents = [
            item if isinstance(item, Intent) else Intent(**item)
            for item in self.intents
            if isinstance(item, (Intent, dict))
        ]
        self.parser = dict(self.parser or {}) if isinstance(self.parser, dict) else {}
        self.unresolved_items = [str(item) for item in self.unresolved_items]
        intent_ids = [item.intent_id for item in self.intents]
        if len(intent_ids) != len(set(intent_ids)):
            self.unresolved_items.append("duplicate_intent_id")
        self.unresolved_items = list(dict.fromkeys(self.unresolved_items))


@dataclass
class IntentParseResult(SerializableMixin):
    """Parser output keeps user intent and query-derived facts as separate collections."""

    schema_version: str
    query_intent: QueryIntent
    context_facts: list[ContextFact] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or "2.0").strip()
        if isinstance(self.query_intent, dict):
            self.query_intent = QueryIntent(**self.query_intent)
        self.context_facts = [
            item if isinstance(item, ContextFact) else ContextFact(**item)
            for item in self.context_facts
            if isinstance(item, (ContextFact, dict))
        ]
