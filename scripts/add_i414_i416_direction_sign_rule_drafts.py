"""Add route/destination sign drafts from official QCVN Appendix E.14-E.16."""

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


def make_rule(code: str, rule_id: str, section: str, conditions: list[str], reason: str) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["circulation", "entering"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"), *map(condition, conditions)],
        "reason": reason,
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {"route_context_facts": conditions, "blanket_permission": False},
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
    result = []
    for variant in "abcd":
        code = f"I.414{variant}"
        result.append(make_rule(
            code, f"I414{variant.upper()}_DESTINATION_ROUTE_GUIDANCE",
            "Appendix E.14, Figure E.14",
            ["i414_road_number_and_destination_text_resolved",
             "i414_arrow_geometry_matches_intersection",
             "requested_route_matches_i414_direction",
             "other_applicable_route_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.14: I.414a/b/c/d directs road users at an intersection toward named destinations or populated places and displays road identification and distance; the sketched direction must correspond to the real intersection. It identifies a route, not permission to use every road along it.",
        ))
    result.append(make_rule(
        "I.415", "I415_DESTINATION_DIRECTION_ARROW_GUIDANCE",
        "Appendix E.15, Figure E.15",
        ["i415_destination_and_arrow_resolved",
         "requested_route_matches_i415_arrow",
         "other_applicable_route_constraints_resolved"],
        "QCVN 41:2024/BGTVT Appendix E.15: I.415 indicates the direction to the next nearby destination and its rounded-kilometer distance; for a nearby destination the distance need not be shown. The sign gives direction/distance information, not permission to travel every road on that route.",
    ))
    result.append(make_rule(
        "I.416", "I416_DETOUR_ROUTE_GUIDANCE_WITH_RESTRICTION_CONTEXT",
        "Appendix E.16(a-b), Figure E.16",
        ["i416_detour_route_matches_diagram",
         "i416_embedded_prohibition_sign_resolved_when_present",
         "affected_vehicle_class_and_underlying_restriction_resolved",
         "alternative_route_applicability_confirmed",
         "other_applicable_route_constraints_resolved"],
        "QCVN 41:2024/BGTVT Appendix E.16(a-b): I.416 guides users onto a detour when a road is closed due to road/bridge obstruction or works and some vehicle types are prohibited; a prohibition symbol from P.101-P.120 may be shown. The actual restriction pictogram, affected class, and detour diagram must be resolved; I.416 alone does not grant route permission.",
    ))
    return result


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-44"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
