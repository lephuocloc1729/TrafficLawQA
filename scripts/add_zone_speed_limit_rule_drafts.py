"""Add source-first zone-speed-start/end rule drafts for R.E.9d and R.E.10d."""

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


def start_zone_rule() -> dict:
    return {
        "rule_id": "RE9D_MAXIMUM_SPEED_WITHIN_CONFIRMED_ZONE",
        "code": "R.E.9d",
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": ["speed"],
        "actor_scope": ["motor_vehicle"],
        "excluded_actor_scope": ["priority_vehicle"],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_within_re9d_zone_confirmed", True),
            {
                "predicate": "speed_limit_value",
                "operator": "exists",
                "expected_value": None,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "NOT_APPLICABLE",
            },
        ],
        "reason": (
            "QCVN 41:2024/BGTVT D.19: R.E.9d starts an area-wide maximum-speed restriction; "
            "the rule applies across all roads in that zone, not just the sign location. The posted "
            "number must be read from this sign, and the vehicle must be confirmed inside the zone. "
            "The evaluator can prohibit a queried speed above that number; a speed at or below it "
            "remains conditional and does not establish overall permission. Actor-scope mapping and "
            "interaction with the zone exit require team review."
        ),
        "effects": [{
            "effect_type": "maximum",
            "action": "speed",
            "unit": "km/h",
            "value_predicate": "speed_limit_value",
            "scope": {"area_fact": "vehicle_within_re9d_zone_confirmed", "all_roads_in_zone": True},
        }],
        "additional_legal_basis": [
            basis("Chapter 6, D.19: R.E.9 signs apply throughout the zone; R.E.9d sets the maximum-speed restriction"),
            basis("Appendix M, group R.E.9d: zone maximum-speed sign face and displayed numeric value"),
            basis("Appendix B.27(b-c): posted maximum-speed value is in km/h and is a ceiling, not a safe target speed"),
        ],
    }


def end_zone_rule() -> dict:
    return {
        "rule_id": "RE10D_END_OF_ZONE_MAXIMUM_SPEED_RESTRICTION",
        "code": "R.E.10d",
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": ["speed", "circulation", "entering"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_beyond_re10d_zone_exit_confirmed", True),
            condition("other_applicable_speed_limit_resolved", True),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT D.20: R.E.10d indicates exit from the area where the corresponding "
            "zone restriction applied. It ends that area restriction only; it does not create a new "
            "speed limit or imply that the road is unrestricted. The engine requires explicit evidence "
            "that the vehicle has passed the zone-exit sign and returns CONDITIONAL pending other limits."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "speed",
            "scope": {"ends_zone_restriction": "R.E.9d", "blanket_permission": False},
            "description": "R.E.10d marks the end of the corresponding area-wide restriction; other speed limits remain unresolved.",
        }],
        "additional_legal_basis": [
            basis("Chapter 6, D.20: R.E.10 signs indicate exit from a zone where a restriction or direction had area-wide effect"),
            basis("Appendix M, group R.E.10d: end of area-wide maximum-speed restriction sign face"),
        ],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    additions = [start_zone_rule(), end_zone_rule()]
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-38"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
