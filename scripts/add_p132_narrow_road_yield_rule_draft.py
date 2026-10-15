"""Add a review-gated P.132 stop/yield rule from official QCVN Appendix B.32."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not any(item.get("official_code") == "P.132" for item in catalog.get("items", [])):
        raise ValueError("Official-source catalog is missing P.132")

    rule = {
        "rule_id": "P132_STOP_AND_YIELD_TO_ONCOMING_MOTOR_VEHICLES",
        "code": "P.132",
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": [
            "stopping", "yielding", "entering", "circulation", "going_straight", "turn_left",
            "turn_right", "u_turn", "crossing",
        ],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            {
                "predicate": "sign_effective_scope_confirmed",
                "operator": "equals",
                "expected_value": True,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "NOT_APPLICABLE",
            },
            {
                "predicate": "vehicle_at_or_approaching_p132_narrow_section",
                "operator": "equals",
                "expected_value": True,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "NOT_APPLICABLE",
            },
            {
                "predicate": "oncoming_motorized_traffic_present",
                "operator": "equals",
                "expected_value": True,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "NOT_APPLICABLE",
            },
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix B.32(a-b), Figure B.32: all motorized and non-motorized "
            "vehicles, including priority vehicles, must yield to oncoming motorized vehicles at the "
            "signed narrow road/bridge; on the signed narrow section, they must stop at a suitable "
            "position to yield. Missing scene evidence does not establish that oncoming traffic is absent."
        ),
        "effects": [
            {
                "effect_type": "require_action",
                "action": "stopping",
                "description": "Stop at a suitable place when oncoming motorized traffic is present on the signed narrow section.",
            },
            {
                "effect_type": "require_action",
                "action": "yielding",
                "description": "Yield to motorized vehicles approaching from the opposite direction.",
            },
        ],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix B, B.32(a-b), Figure B.32 (Gazette page 12)",
            "source_url": SOURCE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
            "gazette_source_url": GAZETTE_URL,
        }],
    }

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id[rule["rule_id"]] = rule
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-26"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
