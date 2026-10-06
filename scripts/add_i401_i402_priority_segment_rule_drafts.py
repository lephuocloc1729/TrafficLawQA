"""Add review-gated I.401/I.402 drafts from official QCVN 41:2024 text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
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
        "on_mismatch": "NOT_APPLICABLE",
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


def make_i401_rule() -> dict:
    return {
        "rule_id": "I401_PRIORITY_ROAD_UNCONTROLLED_INTERSECTION_MEANING",
        "code": "I.401",
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION"],
        "actions": ["yielding", "entering", "circulation", "going_straight", "turn_left", "turn_right", "crossing"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_on_i401_priority_road_segment", True),
            condition("i401_priority_road_direction_resolved", True),
            condition("intersection_traffic_control_active", False),
            condition("priority_vehicle_conflict_present", False),
            condition("actor_has_priority_right", False),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.1(a-b): I.401 marks the start of a priority-road segment. "
            "At an uncontrolled intersection on that segment, traffic on the priority road has priority; "
            "traffic entering/crossing from another road yields, except priority vehicles. If traffic "
            "control is active, this priority-road principle ceases to apply (except for priority vehicles). "
            "Direction/scope, control state, conflict, and actor priority status are not inferred. This is "
            "not blanket permission. S.506a below I.401 indicates the priority-road direction where it changes."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {
                "priority_road_right_of_way": True,
                "exception": "vehicles_with_priority_rights",
                "segment_start_sign": "I.401",
                "supplementary_direction_sign": "S.506a",
            },
            "description": "I.401 marks the beginning of a priority-road segment; priority at an intersection is limited to the conditions in Appendix E.1(b).",
        }],
        "additional_legal_basis": [
            basis("Appendix E.1(a-b), Figure E.1: I.401 priority-road start, placement, and uncontrolled-intersection effect"),
            basis("Appendix E.1(c), Figure E.1; Appendix F.8: S.506a is placed below I.401 where the priority road changes direction and indicates its direction"),
        ],
    }


def make_i402_rule() -> dict:
    return {
        "rule_id": "I402_END_OF_PRIORITY_ROAD_SEGMENT_MEANING",
        "code": "I.402",
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION"],
        "actions": ["entering", "circulation", "going_straight", "turn_left", "turn_right", "crossing"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_beyond_i402_end_point", True),
            condition("other_applicable_signs_and_controls_resolved", True),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.2: I.402 is placed at the end of the designated priority-road segment. "
            "This sign meaning alone does not establish which road user has priority at a later intersection or "
            "grant permission; other signs, controls, and applicable rules must be resolved."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {"priority_road_segment_ends": True},
            "description": "The designated priority-road segment ends at the applicable I.402 sign location; no broader permission or priority result is inferred.",
        }],
        "additional_legal_basis": [basis("Appendix E.2, Figure E.2: I.402, end of priority-road segment")],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_codes = {item.get("official_code") for item in catalog.get("items", [])}
    if not {"I.401", "I.402"}.issubset(catalog_codes):
        raise ValueError("Official-source sign catalog is missing I.401 or I.402")

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    additions = [make_i401_rule(), make_i402_rule()]
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-28"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
