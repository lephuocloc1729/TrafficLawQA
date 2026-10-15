"""Add a conservative, diagram-dependent I.418 route-guidance draft."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(predicate: str, expected: object) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "CONDITIONAL",
    }


def make_rule() -> dict:
    return {
        "rule_id": "I418_DIAGRAM_DEPENDENT_ROUTE_GUIDANCE",
        "code": "I.418",
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["turn_left", "turn_right", "going_straight", "crossing", "entering", "circulation"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("intersection_where_turn_is_prohibited_confirmed", True),
            condition("requested_path_matches_i418_diagram", True),
            condition("other_applicable_sign_constraints_resolved", True),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.18(a-b), Figure E.18: I.418 indicates the route through "
            "an intersection where a turn is prohibited; the sign is placed at the intersection before "
            "the prohibited turn, and its diagram varies with the actual traffic arrangement. The rule "
            "requires an explicit match between the requested path and the observed diagram. It does not "
            "infer permission from an unreadable/mismatched diagram or override another applicable sign."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {
                "route_fact": "requested_path_matches_i418_diagram",
                "diagram_varies_with_actual_traffic_arrangement": True,
                "blanket_permission": False,
            },
            "description": (
                "I.418 guides the road user along the path drawn through an intersection where a turn is prohibited; "
                "the sign alone does not establish overall permission."
            ),
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix E.18(a-b), Figure E.18: I.418 route through a prohibited-turn intersection; diagram depends on actual traffic arrangement",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    addition = make_rule()
    by_id[addition["rule_id"]] = addition
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-41"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
