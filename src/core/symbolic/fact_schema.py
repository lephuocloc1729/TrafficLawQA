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
    catalog: dict[str, Any] = field(default_factory=dict)

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
        self.catalog = dict(self.catalog) if isinstance(self.catalog, dict) else {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize an observation using the compact public traffic-facts schema."""
        identity = {
            "sign_code": self.sign_code,
            "status": self.identity_status,
            "confidence": self.confidence,
            "candidates": [
                {
                    "sign_code": item.get("sign_code"),
                    "confidence": item.get("confidence", item.get("score")),
                }
                for item in self.candidates
            ],
        }
        return {
            "id": self.observation_id,
            "role": self.role,
            "bbox": self.bbox,
            "identity": identity,
            "catalog": {
                "official_name": self.catalog.get("official_name"),
                "legal_meaning": self.catalog.get("legal_meaning"),
                "status": self.catalog.get("status", "unresolved"),
            },
            "ocr_texts": [
                {
                    "raw_text": item.get("raw_text"),
                    "normalized_value": item.get("normalized_value"),
                    "confidence": normalize_confidence(item.get("confidence")),
                    "score": item.get("score"),
                    "bbox": item.get("bbox"),
                    "source": item.get("source", "ocr"),
                }
                for item in self.ocr_texts
            ],
            "facts": [],
            "evidence": [
                {
                    "type": item.get("type", "image"),
                    "ref": item.get("ref", item.get("path")),
                }
                for item in self.evidence
            ],
        }


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
        # Accept caller spellings such as "Speed_limit_value", then keep a
        # lowercase canonical name for rule matching.
        self.predicate = str(self.predicate or "unknown").strip().lower()
        self.status = str(self.status or "unknown").strip().lower()
        self.source = str(self.source or "unknown").strip().lower()
        if self.status not in {
            "provided", "resolved", "observed", "derived", "ambiguous",
            "needs_resolution", "not_observed", "unknown",
        }:
            self.status = "unknown"
        self.confidence = normalize_confidence(self.confidence)

    def to_compact_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "value": self.value,
            "status": self.status,
            "source": self.source,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.relation_type,
            "from": self.source_observation_id,
            "to": self.target_observation_id,
            "confidence": self.confidence,
        }


@dataclass
class TrafficFacts(SerializableMixin):
    schema_version: str
    scene_id: str
    sign_observations: list[SignObservation] = field(default_factory=list)
    context_facts: list[ContextFact] = field(default_factory=list)
    sign_relations: list[SignRelation] = field(default_factory=list)
    unresolved_items: list[str] = field(default_factory=list)
    sign_inventory_status: str = "unknown"
    sign_inventory_evidence: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.schema_version = str(self.schema_version or "1.0").strip()
        self.scene_id = str(self.scene_id or "scene-unknown").strip()
        self.sign_inventory_status = str(self.sign_inventory_status or "unknown").strip().lower()
        if self.sign_inventory_status not in {"complete", "partial", "unknown"}:
            self.sign_inventory_status = "unknown"
        self.sign_inventory_evidence = [
            item for item in self.sign_inventory_evidence if isinstance(item, dict)
        ]
        has_inventory_provenance = any(
            str(item.get("source") or "").strip() and item.get("evidence")
            for item in self.sign_inventory_evidence
        )
        if self.sign_inventory_status == "complete" and not has_inventory_provenance:
            self.sign_inventory_status = "unknown"
            self.unresolved_items.append("sign_inventory_complete_without_evidence")
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compact schema, grouping sign-specific facts by observation."""
        observations = [item.to_dict() for item in self.sign_observations]
        observation_by_id = {item.observation_id: index for index, item in enumerate(self.sign_observations)}
        scene_facts: list[dict[str, Any]] = []

        # Preserve explicitly supplied facts, grouping subject-specific facts under
        # their observation so callers do not need a duplicated subject ID field.
        for fact in self.context_facts:
            subject_id = fact.metadata.get("subject_observation_id")
            if subject_id in observation_by_id:
                observations[observation_by_id[subject_id]]["facts"].append(fact.to_compact_dict())
            else:
                scene_facts.append(fact.to_compact_dict())

        # Facts extracted from OCR and normalized sign attributes are computed by
        # the reasoning layer. Include them in the public fact payload as well.
        try:
            from src.core.symbolic.traffic_rules import _observation_context_facts

            for index, observation in enumerate(self.sign_observations):
                existing = {
                    (fact.get("predicate"), str(fact.get("value")))
                    for fact in observations[index]["facts"]
                }
                for fact in _observation_context_facts(
                    observation, subject_id=observation.observation_id
                ):
                    compact = fact.to_compact_dict()
                    fingerprint = (compact["predicate"], str(compact["value"]))
                    if fingerprint not in existing:
                        observations[index]["facts"].append(compact)
                        existing.add(fingerprint)
        except ImportError:
            # Schema serialization remains available if the optional rule layer
            # cannot be imported (for example, while generating documentation).
            pass

        return {
            "sign_inventory": {
                "status": self.sign_inventory_status,
                "evidence": [
                    {
                        "source": item.get("source"),
                        "ref": item.get("ref", item.get("evidence")),
                    }
                    for item in self.sign_inventory_evidence
                ],
            },
            "sign_observations": observations,
            "sign_relations": [item.to_dict() for item in self.sign_relations],
            "scene_facts": scene_facts,
        }
