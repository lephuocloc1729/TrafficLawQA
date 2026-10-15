"""Add a review-gated R.411 lane-direction rule from official QCVN text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(predicate: str, expected: object, *, on_mismatch: str = "NOT_APPLICABLE") -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": on_mismatch,
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if "R.411" not in {item.get("official_code") for item in catalog.get("items", [])}:
        raise ValueError("Official-source sign catalog is missing R.411")

    rule = {
        "rule_id": "R411_FOLLOW_DIRECTION_FOR_CURRENT_LANE",
        "code": "R.411",
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": ["going_straight", "turn_left", "turn_right", "u_turn", "circulation", "entering"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("r411_current_lane_resolved", True),
            condition("r411_lane_arrow_mapping_resolved", True),
            condition("r411_directional_road_marking_consistent", True),
            condition("r411_requested_route_matches_lane_arrow", True, on_mismatch="PROHIBITED"),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix D.13, Figure D.15: R.411 indicates the number of lanes and "
            "direction for each lane in coordination with road marking type 9.3; road users must use the "
            "lane whose indicated direction is appropriate for their route. The engine requires an explicit "
            "mapping of the observed sign/marking to the actor's current lane and requested route. If the "
            "route is confirmed not to match the lane arrow, it is prohibited under this sign rule. If it "
            "matches, the result remains conditional and is not blanket permission."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {
                "lane_arrow_and_marking_required": True,
                "current_lane_fact": "r411_current_lane_resolved",
                "route_match_fact": "r411_requested_route_matches_lane_arrow",
            },
            "description": "R.411 requires the vehicle to use the lane indicated for its intended route, read together with the corresponding road arrows.",
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix D.13, Figure D.15; Gazette page 42",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id[rule["rule_id"]] = rule
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-31"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
