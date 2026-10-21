"""Add a review-gated W.208 yield rule tied to official QCVN C.8 and S.506b scope."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
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


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not any(item.get("official_code") == "W.208" for item in catalog.get("items", [])):
        raise ValueError("Official-source catalog is missing W.208")

    rule = {
        "rule_id": "W208_YIELD_TO_PRIORITY_ROAD_AT_UNCONTROLLED_INTERSECTION",
        "code": "W.208",
        "actions": [
            "yielding", "entering", "circulation", "going_straight", "turn_left",
            "turn_right", "u_turn", "crossing",
        ],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_approaching_w208_intersection", True),
            condition("intersection_traffic_control_active", False),
            condition("priority_road_scope_modifier_resolved", True),
            condition("priority_road_traffic_requires_yield", True),
            condition("actor_has_priority_right", False),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix C.8(a-b), Figure C.10: W.208 is placed on a non-priority "
            "road before its intersection with a priority road; vehicles on that approach must yield "
            "to vehicles on the priority road, except vehicles with priority rights. The rule requires "
            "the approach, priority-road direction (including any linked S.506b), traffic conflict, and "
            "absence of active intersection control to be explicit. Missing facts remain conditional; "
            "this does not establish blanket permission after yielding."
        ),
        "effects": [{
            "effect_type": "require_action",
            "action": "yielding",
            "description": "Yield to traffic on the priority road at the intersection, subject to the confirmed approach and priority-direction scope.",
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix C, C.8(a-d), Figure C.10 (Gazette page 19); supplementary sign S.506b relationship in C.8(c)",
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
