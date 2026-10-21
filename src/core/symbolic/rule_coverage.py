"""Coverage diagnostics for the source catalog and declarative sign rulebook.

This module reports missing review work; it does not infer that a sign has no
legal effect merely because no rule currently references it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.core.symbolic.rule_schema import TrafficRule
from src.core.symbolic.sign_catalog import (
    DEFAULT_CATALOG_PATH,
    OfficialSignCatalog,
    normalize_official_sign_code,
)
from src.core.symbolic.traffic_rules import RULEBOOK_PATH, load_rulebook

UNCODED_FIGURES_PATH = Path(__file__).parent / "catalogs" / "qcvn_41_2024_uncoded_figure_review.json"


def build_rule_coverage_report(
    catalog_payload: dict[str, Any],
    rules: Iterable[TrafficRule],
    uncoded_figures_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return code-level coverage with separate draft/verified rule counts."""
    coded_items = [
        item for item in catalog_payload.get("items", [])
        if normalize_official_sign_code(item.get("official_code"))
        and item.get("code_namespace") != "project_defined_not_qcvn"
    ]
    all_catalog_codes = {
        normalize_official_sign_code(item.get("official_code"))
        for item in catalog_payload.get("items", [])
        if normalize_official_sign_code(item.get("official_code"))
    }
    family_heading_codes = {
        normalize_official_sign_code(code)
        for code in catalog_payload.get("family_heading_codes", [])
    }
    rules_by_code: dict[str, list[TrafficRule]] = defaultdict(list)
    for rule in rules:
        for raw_code in rule.trigger_sign_codes:
            code = normalize_official_sign_code(raw_code)
            if code:
                rules_by_code[code].append(rule)

    rows: list[dict[str, Any]] = []
    catalog_codes = set()
    for item in coded_items:
        code = normalize_official_sign_code(item.get("official_code"))
        catalog_codes.add(code)
        matching_rules = rules_by_code.get(code, [])
        is_family_heading = code in family_heading_codes
        verified_count = sum(
            rule.verification_status == "verified"
            and bool(rule.legal_basis)
            and all(ref.get("verification_status") == "verified" for ref in rule.legal_basis)
            for rule in matching_rules
        )
        draft_count = sum(rule.verification_status != "verified" for rule in matching_rules)
        if is_family_heading:
            rule_status = "variant_required"
        elif verified_count:
            rule_status = "verified_rule_present"
        elif draft_count:
            rule_status = "draft_rule_review_required"
        else:
            rule_status = "no_rule_record_review_required"
        rows.append({
            "official_code": code,
            "canonical_sign_id": item.get("canonical_sign_id"),
            "official_name": item.get("official_name"),
            "catalog_review_status": item.get("review_status", "unverified"),
            "catalog_entry_kind": "family_heading" if is_family_heading else "sign_or_variant",
            "meaning_available": bool(item.get("legal_meaning")),
            "rule_status": rule_status,
            "rule_ids": [rule.rule_id for rule in matching_rules],
            "verified_rule_count": verified_count,
            "draft_rule_count": draft_count,
        })

    unknown_rule_codes = sorted(set(rules_by_code) - all_catalog_codes)
    canonical_ids = [str(item.get("canonical_sign_id") or "") for item in coded_items]
    leaf_rows = [row for row in rows if row["catalog_entry_kind"] == "sign_or_variant"]
    family_rows = [row for row in rows if row["catalog_entry_kind"] == "family_heading"]
    uncoded_items = (uncoded_figures_payload or {}).get("items", [])
    return {
        "report_version": "1.0",
        "catalog_source": catalog_payload.get("official_document"),
        "catalog_item_count": len(coded_items),
        "coded_sign_count": len(leaf_rows),
        "family_heading_count": len(family_rows),
        "duplicate_catalog_code_count": len(coded_items) - len(catalog_codes),
        "duplicate_canonical_sign_id_count": len(canonical_ids) - len(set(canonical_ids)),
        "meaning_available_count": sum(row["meaning_available"] for row in leaf_rows),
        "meaning_missing_count": sum(not row["meaning_available"] for row in leaf_rows),
        "verified_rule_sign_count": sum(row["rule_status"] == "verified_rule_present" for row in leaf_rows),
        "draft_rule_sign_count": sum(row["rule_status"] == "draft_rule_review_required" for row in leaf_rows),
        "no_rule_record_sign_count": sum(row["rule_status"] == "no_rule_record_review_required" for row in leaf_rows),
        "family_heading_rule_ids": {
            row["official_code"]: row["rule_ids"] for row in family_rows if row["rule_ids"]
        },
        "rule_ids_without_catalog_code": unknown_rule_codes,
        "uncoded_figure_review_count": len(uncoded_items),
        "unresolved_figure_review_count": sum(
            item.get("review_status", "unclassified_illustration") == "unclassified_illustration"
            for item in uncoded_items
        ),
        "figure_linked_to_catalog_sign_count": sum(
            item.get("review_status") == "linked_to_catalog_signs" for item in uncoded_items
        ),
        "out_of_scope_figure_count": sum(
            item.get("review_status") == "out_of_scope" for item in uncoded_items
        ),
        "technical_diagram_count": sum(
            item.get("review_status") == "technical_diagram" for item in uncoded_items
        ),
        "items": rows,
        "review_note": (
            "No rule record is a coverage gap requiring review, not evidence that the sign has no legal effect. "
            "Family-heading rows identify code groups requiring a concrete variant; they are not counted as "
            "individual signs. A verified rule must have a reviewed status and verified legal-basis references."
        ),
    }


def load_default_rule_coverage_report() -> dict[str, Any]:
    with DEFAULT_CATALOG_PATH.open(encoding="utf-8") as stream:
        catalog = json.load(stream)
    merged_catalog = OfficialSignCatalog.load()
    catalog["items"] = list(merged_catalog.items)
    catalog["items_count"] = len(catalog["items"])
    with UNCODED_FIGURES_PATH.open(encoding="utf-8") as stream:
        uncoded = json.load(stream)
    _, rules = load_rulebook(RULEBOOK_PATH)
    return build_rule_coverage_report(catalog, rules, uncoded)


if __name__ == "__main__":
    print(json.dumps(load_default_rule_coverage_report(), ensure_ascii=False, indent=2))
