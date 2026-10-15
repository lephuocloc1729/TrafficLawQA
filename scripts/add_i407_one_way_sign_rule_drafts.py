"""Add conservative I.407(a/b/c) one-way direction rule drafts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"
VARIANTS = ("I.407a", "I.407b", "I.407c")


def make_rule(code: str) -> dict:
    return {
        "rule_id": f"{code.replace('.', '').upper()}_ONE_WAY_DIRECTION_ONLY",
        "code": code,
        "actions": [
            "entering", "circulation", "going_straight", "turn_left", "turn_right", "u_turn", "crossing",
        ],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": ["priority_vehicle"],
        "outcome": "CONDITIONAL",
        "conditions": [{
            "predicate": "sign_effective_scope_confirmed",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "NOT_APPLICABLE",
        }, {
            "predicate": "vehicle_travel_direction_matches_i407_arrow",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "PROHIBITED",
        }],
        "reason": (
            f"QCVN 41:2024/BGTVT Appendix E.7(a-b), Figure E.7: {code} identifies a one-way road; "
            "traffic may travel only in the direction indicated by the arrow, and turning back in the "
            "opposite direction is prohibited except for vehicles with priority rights under the law. "
            "The engine requires the sign scope and vehicle-to-arrow direction to be explicitly resolved. "
            "Matching the arrow is only this sign's direction condition, not blanket permission."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {
                "one_way_direction_fact": "vehicle_travel_direction_matches_i407_arrow",
                "priority_vehicle_exception": True,
                "variant": code,
            },
            "description": (
                f"{code} indicates a one-way direction. Movement matching the arrow satisfies only this "
                "sign's direction condition; it does not establish overall permission."
            ),
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix E.7(a-b), Figure E.7: I.407(a,b,c) one-way road and travel direction",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    additions = [make_rule(code) for code in VARIANTS]
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-37"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
