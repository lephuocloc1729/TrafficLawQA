"""Add conservative temporary/work-zone sign rules from QCVN Appendix E.43-E.44."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(predicate: str) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": True,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "CONDITIONAL",
    }


def make_rule(code: str, rule_id: str, section: str, predicates: list[str], reason: str) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["circulation", "entering", "speed"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"), *map(condition, predicates)],
        "reason": reason,
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {"workzone_facts": predicates, "blanket_permission": False},
            "description": reason,
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": section,
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }


def make_rules() -> list[dict]:
    rules = [make_rule(
        "I.440", "I440_ACTIVE_WORKZONE_SEGMENT_GUIDANCE", "Appendix E.43(a-b), Figure E.43",
        ["i440_active_workzone_segment_confirmed",
         "i440_start_end_and_chainage_resolved",
         "i441_warning_sign_and_other_workzone_panels_resolved",
         "workzone_vehicle_restrictions_and_speed_signs_resolved",
         "other_applicable_route_constraints_resolved"],
        "QCVN 41:2024/BGTVT Appendix E.43(a-b): I.440 identifies a road section undergoing construction, repair, upgrading, or improvement; it is placed at both ends of that section, after I.441, and any displayed chainage is rounded to meters. The sign itself does not impose an unstated closure or speed limit; resolve the active segment and all accompanying restriction signs.",
    )]
    predicates = [
        "i441_worksite_ahead_and_approach_confirmed",
        "i441_distance_stage_and_workzone_extent_resolved",
        "i441_linked_w227_warning_resolved",
        "i441_p127_p134_and_other_panels_resolved_when_present",
        "workzone_vehicle_restrictions_and_speed_signs_resolved",
        "other_applicable_route_constraints_resolved",
    ]
    reason = (
        "QCVN 41:2024/BGTVT Appendix E.44(a-c): I.441a/b/c warns that a construction, repair, or "
        "upgrading worksite is ahead. The signs are placed at both ends of the work section at 500 m, "
        "100 m, and 50 m before I.440, with number and distances adjustable to the traffic plan; W.227 "
        "is placed with them, and P.127/P.134 may be added when needed. Do not infer a specific speed cap, "
        "vehicle prohibition, or route closure from I.441 or its suffix alone; resolve the actual signs and scope."
    )
    for variant in "abc":
        code = f"I.441{variant}"
        rules.append(make_rule(
            code, f"I441{variant.upper()}_WORKZONE_AHEAD_GUIDANCE",
            "Appendix E.44(a-c), Figure E.44", predicates, reason,
        ))
    return rules


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-47"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
