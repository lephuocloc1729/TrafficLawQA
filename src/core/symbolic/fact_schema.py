from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.symbolic.sign_schema import Confidence, SerializableMixin, normalize_confidence


@dataclass
class SignObservation(SerializableMixin):
    observation_id: str
    role: str = "unknown"
    bbox: list[float] | None = None
    sign_code: str | None = None
    identity_status: str = "unidentified"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    visual_attributes: dict[str, Any] = field(default_factory=dict)
    ocr_texts: list[dict[str, Any]] = field(default_factory=list)
    legal_refs: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: Confidence = "low"
    source: str = "image"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observation_id = str(self.observation_id or "observation-unknown").strip()
        self.role = str(self.role or "unknown").strip().lower()
        self.sign_code = str(self.sign_code or "").strip() or None
        self.identity_status = str(self.identity_status or "unidentified").strip().lower()
        if self.role not in {"main", "supplementary", "temporary", "unknown"}:
            self.role = "unknown"
        if self.identity_status not in {"identified", "ambiguous", "unidentified"}:
            self.identity_status = "unidentified"
        if self.identity_status == "identified" and not self.sign_code:
            self.identity_status = "unidentified"
        self.confidence = normalize_confidence(self.confidence)
        self.source = str(self.source or "image").strip()
        self.candidates = [item for item in self.candidates if isinstance(item, dict)]
        self.visual_attributes = dict(self.visual_attributes) if isinstance(self.visual_attributes, dict) else {}
        self.ocr_texts = [
            item if isinstance(item, dict) else {"raw_text": str(item), "source": "ocr"}
            for item in self.ocr_texts
        ]
        self.legal_refs = [item for item in self.legal_refs if isinstance(item, dict)]
        self.evidence = [item for item in self.evidence if isinstance(item, dict)]
        self.metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}


@dataclass
class ContextFact(SerializableMixin):
    fact_id: str
    predicate: str
    value: Any = None
    status: str = "unknown"
    source: str = "unknown"
    raw_text: str | None = None
    confidence: Confidence = "low"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fact_id = str(self.fact_id or "fact-unknown").strip()
        self.predicate = str(self.predicate or "unknown").strip().lower()
        self.status = str(self.status or "unknown").strip().lower()
        self.source = str(self.source or "unknown").strip().lower()
        if self.status not in {
            "provided", "resolved", "observed", "derived", "ambiguous",
            "needs_resolution", "not_observed", "unknown",
        }:
            self.status = "unknown"
        self.confidence = normalize_confidence(self.confidence)


@dataclass
class SignRelation(SerializableMixin):
    relation_id: str
    relation_type: str
    source_observation_id: str
    target_observation_id: str
    confidence: Confidence = "low"
    source: str = "image"

    def __post_init__(self) -> None:
        self.relation_id = str(self.relation_id or "relation-unknown").strip()
        self.relation_type = str(self.relation_type or "unknown").strip().lower()
        self.source_observation_id = str(self.source_observation_id or "").strip()
        self.target_observation_id = str(self.target_observation_id or "").strip()
        self.confidence = normalize_confidence(self.confidence)


@dataclass
class TrafficFacts(SerializableMixin):
    schema_version: str
    scene_id: str
    sign_observations: list[SignObservation] = field(default_factory=list)
    context_facts: list[ContextFact] = field(default_factory=list)
    sign_relations: list[SignRelation] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or "1.0").strip()
        self.scene_id = str(self.scene_id or "scene-unknown").strip()
        self.sign_observations = [
            item if isinstance(item, SignObservation) else SignObservation(**item)
            for item in self.sign_observations
            if isinstance(item, (SignObservation, dict))
        ]
        self.context_facts = [
            item if isinstance(item, ContextFact) else ContextFact(**item)
            for item in self.context_facts
            if isinstance(item, (ContextFact, dict))
        ]
        self.sign_relations = [
            item if isinstance(item, SignRelation) else SignRelation(**item)
            for item in self.sign_relations
            if isinstance(item, (SignRelation, dict))
        ]
        observation_ids = [item.observation_id for item in self.sign_observations]
        if len(observation_ids) != len(set(observation_ids)):
            self.unresolved_items.append("duplicate_observation_id")
        known_ids = set(observation_ids)
        for relation in self.sign_relations:
            if relation.source_observation_id not in known_ids or relation.target_observation_id not in known_ids:
                self.unresolved_items.append(f"unresolved_relation:{relation.relation_id}")
        self.unresolved_items = list(dict.fromkeys(str(item) for item in self.unresolved_items))
