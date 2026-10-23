from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.symbolic.sign_schema import Confidence, SerializableMixin, normalize_confidence


# Query-facing names are stable and descriptive; internal aliases remain
# accepted so existing extraction and rule definitions keep working.
FACT_PREDICATE_ALIASES = {
    "speed_limit_value": "maximum_speed",
    "minimum_speed_limit_kmh": "minimum_speed",
    "height_limit_m": "maximum_vehicle_height",
    "safe_height_limit_m": "maximum_vehicle_height",
    "width_limit_m": "maximum_vehicle_width",
    "length_limit_m": "maximum_vehicle_length",
    "combination_length_limit_m": "maximum_combination_length",
    "minimum_following_distance_m": "minimum_distance",
    "total_weight_limit_t": "maximum_vehicle_mass",
    "axle_load_limit_t": "maximum_axle_load",
    "payload_limit_t": "maximum_payload_mass",
    "passenger_seat_limit": "maximum_seat_count",
    "vehicle_scope_condition": "prohibited_actor",
    "prohibited_vehicle": "prohibited_actor",
    "prohibited_subject": "prohibited_actor",
    "time_window": "active_time",
    "supplementary_time_window": "active_time",
    "supplementary_calendar_parity": "calendar_parity",
    "sign_applicability_calendar_parity": "calendar_parity",
}


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
        ocr_texts = [
            {
                "raw_text": item.get("raw_text"),
                "normalized_value": item.get("normalized_value"),
                "confidence": normalize_confidence(item.get("confidence")),
                "score": item.get("score"),
                "bbox": item.get("bbox"),
                "source": item.get("source", "ocr"),
            }
            for item in self.ocr_texts
        ]
        evidence = [
            {
                "type": item.get("type", "image"),
                "ref": item.get("ref", item.get("path")),
            }
            for item in self.evidence
        ]
        if self.source == "paddleocr_text_panel":
            return {
                "id": self.observation_id,
                "role": "supplementary",
                "type": "ocr_text_panel",
                "bbox": self.bbox,
                "ocr_texts": ocr_texts,
                "facts": [],
                "evidence": evidence,
            }

        identity = {
            "sign_code": self.sign_code,
            "status": self.identity_status,
            "confidence": self.confidence,
            "candidates": [
                {
                    "sign_code": item.get("sign_code"),
                    "confidence": item.get("confidence", item.get("score")),
                    **({"retrieval_rank": item.get("retrieval_rank", item.get("rank"))}
                       if item.get("retrieval_rank", item.get("rank")) is not None else {}),
                    **({"retrieval_score": item.get("retrieval_score", item.get("score"))}
                       if item.get("retrieval_score", item.get("score")) is not None else {}),
                    **({"attributes": item["attributes"]}
                       if isinstance(item.get("attributes"), dict) and item["attributes"] else {}),
                    **({"normalized_conditions": item["normalized_conditions"]}
                       if isinstance(item.get("normalized_conditions"), dict) and item["normalized_conditions"] else {}),
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
            "ocr_texts": ocr_texts,
            "facts": [],
            "evidence": evidence,
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
    unit: str | None = None
    scope: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)

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
        self.metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}
        # Keep compatibility with existing producers while promoting these
        # commonly used fact attributes into the public schema.
        if self.unit is None and self.metadata.get("unit") is not None:
            self.unit = str(self.metadata["unit"])
        else:
            self.unit = str(self.unit).strip() if self.unit is not None else None
        if not self.scope and isinstance(self.metadata.get("scope"), dict):
            self.scope = dict(self.metadata["scope"])
        else:
            self.scope = dict(self.scope) if isinstance(self.scope, dict) else {}
        if not self.conditions and isinstance(self.metadata.get("conditions"), dict):
            self.conditions = dict(self.metadata["conditions"])
        else:
            self.conditions = dict(self.conditions) if isinstance(self.conditions, dict) else {}

    def to_compact_dict(self) -> dict[str, Any]:
        predicate = FACT_PREDICATE_ALIASES.get(self.predicate, self.predicate)
        scope = dict(self.scope)
        value = self.value
        if self.predicate == "minimum_following_distance_m":
            scope.setdefault("distance_kind", "following_distance")
        if self.predicate in {"calendar_parity", "supplementary_calendar_parity", "sign_applicability_calendar_parity"}:
            value = {"odd_days": "odd", "even_days": "even"}.get(value, value)
        if self.predicate in {"time_window", "supplementary_time_window"} and isinstance(value, dict):
            start = value.get("start_minute")
            end = value.get("end_minute")
            if isinstance(start, int) and isinstance(end, int):
                value = {
                    "from": f"{start // 60:02d}:{start % 60:02d}",
                    "to": f"{end // 60:02d}:{end % 60:02d}",
                }
        payload = {
            "predicate": predicate,
            "value": value,
            "status": self.status,
            "source": self.source,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
        }
        if self.unit:
            payload["unit"] = self.unit
        if scope:
            payload["scope"] = scope
        if self.conditions:
            payload["conditions"] = self.conditions
        return payload


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
        """Serialize answer-facing sign facts without internal extraction metadata."""
        signs: list[dict[str, Any]] = []
        observation_by_id: dict[str, int] = {}
        scene_facts: list[dict[str, Any]] = []

        for observation in self.sign_observations:
            is_text_panel = observation.source == "paddleocr_text_panel"
            sign: dict[str, Any] = {
                "id": observation.observation_id,
                "role": "supplementary" if is_text_panel else observation.role,
                "facts": [],
            }
            if is_text_panel:
                sign["type"] = "ocr_text_panel"
                sign["ocr_text"] = [
                    str(item.get("normalized_value") or item.get("raw_text") or "")
                    for item in observation.ocr_texts
                    if item.get("normalized_value") or item.get("raw_text")
                ]
            else:
                if observation.sign_code:
                    sign["code"] = observation.sign_code
                if observation.catalog.get("official_name"):
                    sign["name"] = observation.catalog["official_name"]
                if observation.identity_status != "identified":
                    sign["identity_status"] = observation.identity_status
                if observation.identity_status != "identified" or observation.confidence != "high":
                    sign["confidence"] = observation.confidence
                if observation.ocr_texts:
                    sign["ocr_text"] = [
                        str(item.get("normalized_value") or item.get("raw_text") or "")
                        for item in observation.ocr_texts
                        if item.get("normalized_value") or item.get("raw_text")
                    ]
            observation_by_id[observation.observation_id] = len(signs)
            signs.append(sign)

        # Preserve explicitly supplied facts, grouping subject-specific facts under
        # their observation so callers do not need a duplicated subject ID field.
        for fact in self.context_facts:
            if fact.predicate in {"sign_text", "supplementary_text"}:
                continue
            subject_id = fact.metadata.get("subject_observation_id")
            if subject_id in observation_by_id:
                signs[observation_by_id[subject_id]]["facts"].append(_answer_fact_dict(fact))
            else:
                scene_facts.append(_answer_fact_dict(fact))

        # Facts extracted from OCR and normalized sign attributes are computed by
        # the reasoning layer. Include them in the public fact payload as well.
        try:
            from src.core.symbolic.sign_fact_resolver import observation_context_facts

            for observation_index, observation in enumerate(self.sign_observations):
                existing = {
                    (fact.get("predicate"), str(fact.get("value")))
                    for fact in signs[observation_index]["facts"]
                }
                for fact in observation_context_facts(
                    observation, subject_id=observation.observation_id
                ):
                    # The OCR transcription is already preserved under
                    # `ocr_texts`; keep only its normalized semantic facts here.
                    if fact.predicate in {"sign_text", "supplementary_text"}:
                        continue
                    compact = _answer_fact_dict(fact)
                    fingerprint = (compact["predicate"], str(compact["value"]))
                    if fingerprint not in existing:
                        signs[observation_index]["facts"].append(compact)
                        existing.add(fingerprint)
        except ImportError:
            # Schema serialization remains available if the optional fact resolver
            # cannot be imported (for example, while generating documentation).
            pass

        # Materialize a high-confidence OCR time panel onto its parent sign in
        # the compact answer-facing view. Keep the panel observation and
        # relation for traceability, while avoiding duplicate active_time rows.
        for relation in self.sign_relations:
            if relation.relation_type not in {"supplements", "supplementary_for"}:
                continue
            if relation.confidence != "high":
                continue
            source_index = observation_by_id.get(relation.source_observation_id)
            target_index = observation_by_id.get(relation.target_observation_id)
            if source_index is None or target_index is None:
                continue
            source_observation = self.sign_observations[source_index]
            target_observation = self.sign_observations[target_index]
            if (
                source_observation.source != "paddleocr_text_panel"
                or target_observation.role != "main"
            ):
                continue

            panel_facts = signs[source_index]["facts"]
            active_times = [
                fact for fact in panel_facts
                if fact.get("predicate") == "active_time"
            ]
            if not active_times:
                continue
            main_facts = signs[target_index]["facts"]
            existing = {
                (fact.get("predicate"), str(fact.get("value")))
                for fact in main_facts
            }
            for fact in active_times:
                fingerprint = (fact.get("predicate"), str(fact.get("value")))
                if fingerprint not in existing:
                    main_facts.append(dict(fact))
                    existing.add(fingerprint)
            signs[source_index]["facts"] = [
                fact for fact in panel_facts
                if fact.get("predicate") != "active_time"
            ]

        from src.core.symbolic.condition_facts import bundle_condition_facts

        for sign in signs:
            sign["facts"] = bundle_condition_facts(sign.get("facts", []))
        scene_facts = bundle_condition_facts(scene_facts)

        return {
            "signs": signs,
            "relations": [
                {
                    "predicate": item.relation_type,
                    "from": item.source_observation_id,
                    "to": item.target_observation_id,
                    "confidence": item.confidence,
                }
                for item in self.sign_relations
            ],
            "scene_facts": scene_facts,
        }


def _answer_fact_dict(fact: ContextFact) -> dict[str, Any]:
    """Keep the predicate/value contract and only answer/debug-relevant fields."""
    compact = fact.to_compact_dict()
    return {
        key: compact[key]
        for key in ("predicate", "value", "unit", "scope", "conditions", "status", "confidence")
        if key == "value" or (compact.get(key) is not None and compact.get(key) != {})
    }
