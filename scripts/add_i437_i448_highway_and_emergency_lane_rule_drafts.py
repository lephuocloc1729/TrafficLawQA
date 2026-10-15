"""Add fail-closed highway-start and emergency-lane sign drafts from QCVN."""

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


def make_rules() -> list[dict]:
    definitions = [
        {
            "code": "I.437",
            "rule_id": "I437_HIGHWAY_START_NOT_VEHICLE_ELIGIBILITY",
            "section": "Appendix E.40, Figure E.40",
            "conditions": [
                "i437_highway_start_location_confirmed",
                "actor_vehicle_class_and_highway_eligibility_resolved",
                "other_applicable_highway_sign_and_legal_constraints_resolved",
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix E.40: I.437 indicates the start of an expressway. "
                "It does not by itself establish that a particular vehicle class is eligible to enter "
                "or that entry is permitted under all other applicable restrictions."
            ),
            "scope": {
                "meaning_fact": "i437_highway_start_location_confirmed",
                "eligibility_fact": "actor_vehicle_class_and_highway_eligibility_resolved",
                "blanket_permission": False,
            },
        },
        {
            "code": "I.448",
            "rule_id": "I448_EMERGENCY_LANE_LOCATION_AND_DISTANCE_GUIDANCE",
            "section": "Appendix E.51",
            "conditions": [
                "i448_emergency_lane_location_confirmed",
                "i448_distance_and_turn_point_resolved",
                "actor_vehicle_class_and_reason_for_use_resolved",
                "other_applicable_emergency_lane_constraints_resolved",
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix E.51: I.448 signs indicate the location and distance "
                "to an emergency escape lane, using four pairs at 2 km, 1 km, 300 m, and the turn point; "
                "the distance may be adjusted. This sign alone does not authorize general driving in or "
                "use of the emergency lane."
            ),
            "scope": {
                "location_fact": "i448_emergency_lane_location_confirmed",
                "distance_fact": "i448_distance_and_turn_point_resolved",
                "blanket_permission": False,
            },
        },
    ]
    return [{
        "rule_id": item["rule_id"],
        "code": item["code"],
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["entering", "circulation"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"),
                       *map(condition, item["conditions"])],
        "reason": item["reason"],
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "entering" if item["code"] == "I.437" else "circulation",
            "scope": item["scope"],
            "description": item["reason"],
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": item["section"],
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    } for item in definitions]


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-45"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
