"""Add I.436 duty drafts for QCVN Appendix E.39."""

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
        "on_mismatch": "NOT_APPLICABLE",
    }


def basis(section: str) -> list[dict]:
    return [{
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": f"Appendix {section}",
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }]


def duty_rule(rule_id: str, action: str, effect_type: str, description: str) -> dict:
    return {
        "rule_id": rule_id,
        "code": "I.436",
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": [action],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed"),
            condition("i436_vehicle_passing_through_police_station_area_confirmed"),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.39 states that vehicles must reduce speed to a safe level "
            "and must not overtake while passing through the indicated area. " + description
        ),
        "effects": [{
            "effect_type": effect_type,
            "action": action,
            "scope": {"area_fact": "i436_vehicle_passing_through_police_station_area_confirmed"},
            "description": description,
        }],
        "additional_legal_basis": basis("E.39, Figure E.39"),
    }


def make_rules() -> list[dict]:
    return [
        duty_rule(
            "I436_REQUIRE_SAFE_SPEED_WHILE_PASSING_POLICE_STATION_AREA",
            "speed", "require_action", "Reduce speed to a safe level while passing through the area.",
        ),
        duty_rule(
            "I436_PROHIBIT_OVERTAKING_WHILE_PASSING_POLICE_STATION_AREA",
            "overtaking", "prohibit_action", "Do not overtake while passing through the area.",
        ),
    ]


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-55"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
