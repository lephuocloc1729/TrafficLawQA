from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.symbolic.fact_schema import TrafficFacts
from src.core.symbolic.intent_schema import Intent, QueryIntent

DEFAULT_CATALOG_PATH = Path(__file__).parent / "catalogs" / "qcvn_41_2024_sign_catalog_draft.json"
SUPPLEMENTARY_ADDENDUM_PATH = Path(__file__).parent / "catalogs" / "qcvn_41_2024_supplementary_sign_addendum.json"


def normalize_official_sign_code(value: str | None) -> str:
    """Normalize typography only; do not infer or expand unspecified variants."""
    text = str(value or "").strip().replace(",", ".")
    text = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"([A-Za-z.]+)\.?([0-9]+)([A-Za-z]?)", text)
    if not match:
        return text
    prefix = match.group(1).rstrip(".").upper()
    return f"{prefix}.{match.group(2)}{match.group(3).lower()}"


@dataclass(frozen=True)
class SignMeaningLookup:
    sign_code: str
    official_name: str | None
    legal_meaning: str | None
    legal_basis: tuple[dict[str, Any], ...]
    status: str
    visual_description: str | None = None
    issues: tuple[str, ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()

    @property
    def is_verified(self) -> bool:
        return self.status == "verified"


@dataclass(frozen=True)
class SignMeaningEvaluation:
    intent_id: str
    observation_id: str | None
    sign_code: str | None
    official_name: str | None
    legal_meaning: str | None
    legal_basis: tuple[dict[str, Any], ...] = ()
    status: str = "unresolved"
    issues: tuple[str, ...] = ()
    candidates: tuple[dict[str, Any], ...] = ()


class OfficialSignCatalog:
    """Source-first sign catalog; LawDB fields are deliberately not consumed."""

    def __init__(self, payload: dict[str, Any]):
        self.metadata = {key: value for key, value in payload.items() if key != "items"}
        self._by_code: dict[str, dict[str, Any]] = {}
        seen_canonical_ids: set[str] = set()
        self._by_alias: dict[str, str] = {}
        self._alias_conflicts: set[str] = set()
        for item in payload.get("items", []):
            code = normalize_official_sign_code(item.get("official_code"))
            if not code:
                continue
            if code in self._by_code:
                raise ValueError(f"Duplicate official sign code in catalog: {code}")
            canonical_id = str(item.get("canonical_sign_id") or "").strip()
            if canonical_id and canonical_id in seen_canonical_ids:
                raise ValueError(f"Duplicate canonical sign id in catalog: {canonical_id}")
            if canonical_id:
                seen_canonical_ids.add(canonical_id)
            self._by_code[code] = item
            for alias in item.get("aliases", []):
                normalized = normalize_official_sign_code(alias)
                if normalized and normalized != code:
                    previous = self._by_alias.get(normalized)
                    if previous and previous != code:
                        self._alias_conflicts.add(normalized)
                    else:
                        self._by_alias[normalized] = code

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CATALOG_PATH) -> "OfficialSignCatalog":
        path = Path(path)
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if path.resolve() == DEFAULT_CATALOG_PATH.resolve() and SUPPLEMENTARY_ADDENDUM_PATH.exists():
            with SUPPLEMENTARY_ADDENDUM_PATH.open(encoding="utf-8") as stream:
                addendum = json.load(stream)
            payload["source_addenda"] = [
                {key: value for key, value in addendum.items() if key != "items"}
            ]
            for family in addendum.get("items", []):
                for code in family.get("codes", []):
                    variant = code.removeprefix(family["family"])
                    basis_overrides = family.get("legal_basis_by_code", {}).get(code, [])
                    legal_basis = [
                        {
                            "document": addendum["source_document"],
                            "edition": "2024",
                            "section_or_figure": basis.get("section_or_figure", family["section_or_figure"]),
                            "pdf_pages": family["pdf_pages"],
                            "source_url": addendum["source_url"],
                            "accessed_at": addendum["accessed_at"],
                            "verification_status": basis.get("verification_status", "unverified"),
                        }
                        for basis in (basis_overrides or [{"section_or_figure": family["section_or_figure"]}])
                    ]
                    item = {
                        "canonical_sign_id": f"qcvn41_2024_{code.lower().replace('.', '_')}",
                        "official_code": code,
                        "aliases": [],
                        "family": family["family"],
                        "variant": variant or None,
                        "official_name": family.get("official_names_by_code", {}).get(
                            code, family.get("official_name")
                        ),
                        "sign_category": family["sign_category"],
                        "visual_description": family.get("visual_descriptions_by_code", {}).get(code),
                        "legal_meaning": family.get("legal_meaning_by_code", {}).get(
                            code, family.get("legal_meaning")
                        ),
                        "scope_attributes": {
                            "vehicle_scope": None,
                            "action_scope": None,
                            "time_condition": None,
                            "direction_scope": None,
                            "distance_scope": None,
                            "numeric_value": None,
                            "exceptions": [],
                        },
                        "supplementary_sign_relationships": family.get(
                            "supplementary_sign_relationships", []
                        ),
                        "legal_basis": legal_basis,
                        "review_status": addendum["review_status"],
                        "eligible_for_automatic_decision": addendum["eligible_for_automatic_decision"],
                        "source_evidence": {
                            "official_pdf_sha256": addendum["source_file_sha256"],
                            "source_text_excerpts": [family["source_excerpt"]],
                        },
                        "issues": ["not_in_initial_automated_extraction", "team_legal_review_required"],
                        "reviewed_by": None,
                        "reviewed_at": None,
                        "review_notes": family.get("review_notes_by_code", {}).get(
                            code, family.get("review_notes", "")
                        ),
                    }
                    payload.setdefault("items", []).append(item)
            payload["items_count"] = len(payload.get("items", []))
        return cls(payload)

    def lookup(self, sign_code: str | None) -> SignMeaningLookup | None:
        normalized = normalize_official_sign_code(sign_code)
        family_variants = sorted(
            code for code, item in self._by_code.items()
            if normalize_official_sign_code(item.get("family")) == normalized
            and item.get("variant")
        )
        family_heading_codes = {
            normalize_official_sign_code(code)
            for code in self.metadata.get("family_heading_codes", [])
        }
        if normalized in family_heading_codes and family_variants:
            item = self._by_code.get(normalized, {})
            return SignMeaningLookup(
                sign_code=normalized,
                official_name=item.get("official_name"),
                legal_meaning=item.get("legal_meaning"),
                legal_basis=tuple(item.get("legal_basis", [])),
                status="variant_required",
                issues=("official_code_identifies_a_family_not_a_single_variant",),
                candidates=tuple(self._candidate_summary(code) for code in family_variants),
            )
        if normalized not in self._by_code and family_variants:
            return SignMeaningLookup(
                sign_code=normalized,
                official_name=None,
                legal_meaning=None,
                legal_basis=(),
                status="variant_required",
                issues=("official_code_identifies_a_family_not_a_single_variant",),
                candidates=tuple(self._candidate_summary(code) for code in family_variants),
            )
        # An explicit official code always takes precedence over aliases that
        # happen to use the same spelling. Alias ambiguity only applies when
        # the query itself is not a canonical official code.
        if normalized in self._alias_conflicts and normalized not in self._by_code:
            codes = [
                code for code, item in self._by_code.items()
                if normalized in {normalize_official_sign_code(alias) for alias in item.get("aliases", [])}
            ]
            return SignMeaningLookup(
                sign_code=normalized,
                official_name=None,
                legal_meaning=None,
                legal_basis=(),
                status="ambiguous_alias",
                issues=("catalog_alias_maps_to_multiple_official_codes",),
                candidates=tuple(self._candidate_summary(code) for code in sorted(codes)),
            )
        canonical = normalized if normalized in self._by_code else self._by_alias.get(normalized)
        if canonical is None:
            return None
        item = self._by_code[canonical]
        status = "verified" if (
            item.get("review_status") == "verified"
            and item.get("eligible_for_automatic_decision") is True
            and bool(item.get("legal_basis"))
            and all(ref.get("verification_status") == "verified" for ref in item.get("legal_basis", []))
        ) else "unverified"
        return SignMeaningLookup(
            sign_code=canonical,
            official_name=item.get("official_name"),
            visual_description=item.get("visual_description"),
            legal_meaning=item.get("legal_meaning"),
            legal_basis=tuple(item.get("legal_basis", [])),
            status=status,
            issues=tuple(item.get("issues", [])),
        )

    def _candidate_summary(self, code: str) -> dict[str, Any]:
        item = self._by_code[code]
        return {
            "sign_code": code,
            "official_name": item.get("official_name"),
            "legal_meaning": item.get("legal_meaning"),
            "catalog_status": "verified" if item.get("review_status") == "verified" else "unverified",
        }

    @property
    def items(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._by_code.values())

    def evaluate_meaning(self, facts: TrafficFacts, intent: Intent) -> list[SignMeaningEvaluation]:
        """Resolve sign-meaning intents only from this official-source catalog."""
        if intent.kind != "QUERY_SIGN":
            return []
        if intent.target.kind == "sign_code" and intent.target.sign_codes:
            results = []
            for requested_code in intent.target.sign_codes:
                entry = self.lookup(requested_code)
                observation_id = next((
                    observation.observation_id for observation in facts.sign_observations
                    if normalize_official_sign_code(observation.sign_code) == normalize_official_sign_code(requested_code)
                ), None)
                if entry is None:
                    results.append(SignMeaningEvaluation(
                        intent_id=intent.intent_id,
                        observation_id=observation_id,
                        sign_code=normalize_official_sign_code(requested_code),
                        official_name=None,
                        legal_meaning=None,
                        status="not_found_in_official_catalog",
                        issues=("official_source_catalog_gap",),
                    ))
                else:
                    results.append(SignMeaningEvaluation(
                        intent_id=intent.intent_id,
                        observation_id=observation_id,
                        sign_code=entry.sign_code,
                        official_name=entry.official_name,
                        legal_meaning=entry.legal_meaning,
                        legal_basis=entry.legal_basis,
                        status=entry.status,
                        issues=entry.issues,
                        candidates=entry.candidates,
                    ))
            return results

        observations = facts.sign_observations
        if intent.target.kind == "sign_observation" and intent.target.observation_ids:
            wanted = set(intent.target.observation_ids)
            observations = [item for item in observations if item.observation_id in wanted]
        elif intent.target.kind == "sign_code" and intent.target.sign_codes:
            wanted = {normalize_official_sign_code(code) for code in intent.target.sign_codes}
            observations = [
                item for item in observations
                if normalize_official_sign_code(item.sign_code) in wanted
                or any(normalize_official_sign_code(c.get("sign_code")) in wanted for c in item.candidates)
            ]

        results: list[SignMeaningEvaluation] = []
        for observation in observations:
            if observation.identity_status == "identified" and observation.sign_code:
                entry = self.lookup(observation.sign_code)
                if entry is None:
                    results.append(SignMeaningEvaluation(
                        intent_id=intent.intent_id,
                        observation_id=observation.observation_id,
                        sign_code=normalize_official_sign_code(observation.sign_code),
                        official_name=None,
                        legal_meaning=None,
                        status="not_found_in_official_catalog",
                        issues=("official_source_catalog_gap",),
                    ))
                else:
                    results.append(SignMeaningEvaluation(
                        intent_id=intent.intent_id,
                        observation_id=observation.observation_id,
                        sign_code=entry.sign_code,
                        official_name=entry.official_name,
                        legal_meaning=entry.legal_meaning,
                        legal_basis=entry.legal_basis,
                        status=entry.status,
                        issues=entry.issues,
                        candidates=entry.candidates,
                    ))
                continue

            candidate_rows = []
            for candidate in observation.candidates:
                entry = self.lookup(candidate.get("sign_code"))
                if entry is not None:
                    candidate_rows.append({
                        "sign_code": entry.sign_code,
                        "official_name": entry.official_name,
                        "legal_meaning": entry.legal_meaning,
                        "legal_basis": list(entry.legal_basis),
                        "catalog_status": entry.status,
                        "candidate_confidence": candidate.get("confidence"),
                    })
            status = "ambiguous" if candidate_rows or observation.identity_status == "ambiguous" else "unresolved"
            results.append(SignMeaningEvaluation(
                intent_id=intent.intent_id,
                observation_id=observation.observation_id,
                sign_code=None,
                official_name=None,
                legal_meaning=None,
                status=status,
                issues=("sign_identity_not_resolved",),
                candidates=tuple(candidate_rows),
            ))
        if not results:
            results.append(SignMeaningEvaluation(
                intent_id=intent.intent_id,
                observation_id=None,
                sign_code=None,
                official_name=None,
                legal_meaning=None,
                status="unresolved",
                issues=("no_target_sign_observation",),
            ))
        return results


@lru_cache(maxsize=1)
def load_default_sign_catalog() -> OfficialSignCatalog:
    return OfficialSignCatalog.load()
