"""Add fail-closed, variant-level route-guidance rules from QCVN Appendix E."""

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


def make_rule(code: str, rule_id: str, section: str, conditions: list[str], reason: str,
              action: str = "circulation") -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION"],
        "actions": list(dict.fromkeys([action, "circulation"])),
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"), *map(condition, conditions)],
        "reason": reason,
        "effects": [{
            "effect_type": "sign_meaning",
            "action": action,
            "scope": {"route_and_actor_class_facts": conditions, "blanket_permission": False},
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
        "I.413a", "I413A_BUS_LANE_OPPOSITE_DIRECTION_GUIDANCE", "Appendix E.13(a), Figure E.13",
        ["i413a_diagram_and_opposite_direction_resolved",
         "bus_lane_effective_scope_resolved",
         "actor_bus_lane_eligibility_resolved",
         "other_applicable_route_constraints_resolved"],
        "QCVN 41:2024/BGTVT Appendix E.13(a): I.413a informs road users that a bus-only lane exists ahead in the opposite direction and is placed at the start of a one-way road where buses may travel in the opposing direction. The sign alone does not establish that the querying vehicle may use that lane.",
    )]
    for code in ("I.413b", "I.413c"):
        rules.append(make_rule(
            code, f"{code.replace('.', '').upper()}_BUS_LANE_ROUTE_GUIDANCE",
            "Appendix E.13(b), Figure E.13",
            ["requested_turn_matches_i413_diagram",
             "destination_bus_lane_scope_resolved",
             "actor_bus_lane_eligibility_resolved",
             "other_applicable_route_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.13(b): I.413b/c indicates that turning at the intersection leads to a road with a lane reserved for buses. The diagram and requested turn must match; this sign does not determine whether the querying vehicle is eligible to use the bus lane.",
        ))

    for variant in "abc":
        code = f"I.417{variant}"
        rules.append(make_rule(
            code, f"I417{variant.upper()}_VEHICLE_SPECIFIC_ROUTE_GUIDANCE",
            "Appendix E.17(a-b), Figure E.17",
            ["vehicle_class_shown_on_i417_resolved",
             "requested_route_matches_i417_diagram_for_vehicle_class",
             "underlying_vehicle_restriction_and_route_scope_resolved",
             "other_applicable_route_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.17(a-b): I.417a/b/c may direct particular vehicle types to a route toward the next populated area where some vehicle types are prohibited or restricted. The shown vehicle class and route must be read from the actual sign; it does not independently establish blanket permission or prohibition.",
        ))
    return rules


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-43"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
