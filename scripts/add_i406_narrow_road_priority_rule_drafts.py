"""Add source-first, fail-closed I.406 narrow-road priority rule drafts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(predicate: str, expected: object, *, mismatch: str = "NOT_APPLICABLE") -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": mismatch,
    }


def basis(section: str) -> dict:
    return {
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": section,
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }


def make_priority_rule() -> dict:
    return {
        "rule_id": "I406_MOTOR_VEHICLE_PRIORITY_NARROW_ROAD",
        "code": "I.406",
        "actions": ["entering", "circulation", "going_straight", "crossing"],
        "actor_scope": ["motor_vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_on_i406_priority_approach", True),
            condition("opposing_vehicle_already_entered_narrow_section", False),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.6(a-b), Figure E.6: I.406 indicates that a motor vehicle "
            "on the signed approach has priority to go first through a narrow section, but it must "
            "still yield if an oncoming motorized or non-motorized vehicle has already entered that "
            "section. This draft reports only the sign's priority relation; it does not establish "
            "general permission to proceed. Missing occupancy or approach facts remain conditional."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {
                "priority_relation": "motor_vehicle_on_signed_approach_over_oncoming_traffic",
                "priority_is_not_blanket_permission": True,
                "narrow_section_occupancy_fact": "opposing_vehicle_already_entered_narrow_section",
            },
            "description": (
                "I.406 gives the signed motor-vehicle approach priority at the narrow section only "
                "subject to the duty to yield to an oncoming vehicle already inside it."
            ),
        }],
        "additional_legal_basis": [
            basis("Appendix E.6(a-b), Figure E.6: I.406 priority through a narrow road and the oncoming-vehicle exception"),
            basis("Appendix E.6(c), Figure E.6: P.132 is required on the opposite approach"),
        ],
    }


def make_yield_rule() -> dict:
    return {
        "rule_id": "I406_YIELD_WHEN_ONCOMING_VEHICLE_ALREADY_IN_NARROW_SECTION",
        "code": "I.406",
        "actions": ["yielding", "entering", "circulation", "going_straight", "crossing"],
        "actor_scope": ["motor_vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_on_i406_priority_approach", True),
            condition("opposing_vehicle_already_entered_narrow_section", True),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.6(b): a vehicle on the I.406 priority approach must yield "
            "if an oncoming motorized or non-motorized vehicle has already entered the narrow section. "
            "The engine requires explicit approach and occupancy evidence; it does not infer absence "
            "of a vehicle or grant permission after yielding."
        ),
        "effects": [{
            "effect_type": "require_action",
            "action": "yielding",
            "description": "Yield to an oncoming motorized or non-motorized vehicle already inside the narrow section.",
        }],
        "additional_legal_basis": [
            basis("Appendix E.6(b), Figure E.6: duty to yield when an oncoming vehicle has entered the narrow section"),
        ],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    additions = [make_priority_rule(), make_yield_rule()]
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-36"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
