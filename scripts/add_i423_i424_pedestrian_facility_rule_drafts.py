"""Add pedestrian-crossing/facility drafts from official QCVN Appendix E.23a-E.25."""

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


def make_rule(code: str, rule_id: str, actors: list[str], actions: list[str], section: str,
              predicates: list[str], reason: str, effects: list[dict]) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "actions": actions,
        "actor_scope": actors,
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"), *map(condition, predicates)],
        "reason": reason,
        "effects": effects,
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": section,
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }


def make_rules() -> list[dict]:
    pedestrian_crossing_facts = [
        "i423_crossing_location_and_sign_direction_confirmed",
        "pedestrian_presence_at_or_approaching_crossing_resolved",
        "crosswalk_marking_and_traffic_control_resolved",
        "other_applicable_pedestrian_and_vehicle_controls_resolved",
    ]
    crossing_reason = (
        "QCVN 41:2024/BGTVT Appendix E.23a(a-b): I.423a/b identifies a pedestrian crossing place. "
        "It may be used alone at an uncontrolled crossing or together with road markings. A road user "
        "encountering it must slow down, observe, and give priority to pedestrians crossing; the pictured "
        "pedestrian direction must face toward the road. This draft requires actual approach and pedestrian "
        "facts and does not infer unrelated right-of-way or permission conclusions."
    )
    crossing_effects = [
        {"effect_type": "require_action", "action": "reduce_speed", "scope": {"location_fact": "i423_crossing_location_and_sign_direction_confirmed"},
         "description": "Slow down on approach to the indicated pedestrian crossing."},
        {"effect_type": "require_action", "action": "observe", "scope": {"location_fact": "i423_crossing_location_and_sign_direction_confirmed"},
         "description": "Pay attention and observe for pedestrians at the indicated crossing."},
        {"effect_type": "require_action", "action": "yielding", "scope": {"condition": "pedestrian_presence_at_or_approaching_crossing"},
         "description": "Give priority to pedestrians crossing when the crossing fact is confirmed."},
    ]
    rules = []
    for variant in "ab":
        code = f"I.423{variant}"
        rules.append(make_rule(
            code, f"I423{variant.upper()}_PEDESTRIAN_CROSSING_DUTY",
            ["vehicle"], ["circulation", "crossing", "yielding"],
            "Appendix E.23a(a-b), Figure E.23a",
            pedestrian_crossing_facts, crossing_reason, crossing_effects,
        ))

    rules.append(make_rule(
        "I.423c", "I423C_START_OF_PEDESTRIAN_PATH_GUIDANCE", ["vehicle", "pedestrian"],
        ["circulation", "crossing", "entering"], "Appendix E.23b, Figure E.23b",
        ["i423c_start_of_pedestrian_path_confirmed",
         "requested_location_is_not_a_road_crossing_confirmed",
         "vehicle_access_or_pedestrian_route_rules_resolved"],
        "QCVN 41:2024/BGTVT Appendix E.23b: I.423c identifies the beginning of a section reserved for pedestrians and is not used for locations where pedestrians cross a road. The sign alone does not establish vehicle permission on the pedestrian path or any crossing priority.",
        [{"effect_type": "sign_meaning", "action": "circulation",
          "scope": {"pedestrian_path_start_fact": "i423c_start_of_pedestrian_path_confirmed",
                    "not_a_crossing_fact": "requested_location_is_not_a_road_crossing_confirmed",
                    "blanket_permission": False}}],
    ))

    for variant in "ab":
        code = f"I.424{variant}"
        rules.append(make_rule(
            code, f"I424{variant.upper()}_PEDESTRIAN_OVERPASS_GUIDANCE", ["pedestrian"],
            ["crossing", "walking"], "Appendix E.24, Figure E.24",
            ["i424_overpass_location_confirmed", "pedestrian_direction_matches_i424_diagram"],
            "QCVN 41:2024/BGTVT Appendix E.24: I.424a/b directs pedestrians to use the pedestrian overpass; select the variant matching the actual direction of pedestrian travel. This sign does not grant vehicle permission or independently establish vehicle right-of-way.",
            [{"effect_type": "sign_meaning", "action": "crossing",
              "scope": {"facility": "pedestrian_overpass", "direction_fact": "pedestrian_direction_matches_i424_diagram",
                        "blanket_permission": False}}],
        ))
    for variant in "cd":
        code = f"I.424{variant}"
        rules.append(make_rule(
            code, f"I424{variant.upper()}_PEDESTRIAN_UNDERPASS_GUIDANCE", ["pedestrian"],
            ["crossing", "walking"], "Appendix E.25, Figure E.25",
            ["i424_underpass_location_confirmed", "pedestrian_direction_matches_i424_diagram"],
            "QCVN 41:2024/BGTVT Appendix E.25: I.424c/d directs pedestrians to use the pedestrian underpass; select the variant matching the actual direction of pedestrian travel. This sign does not grant vehicle permission or independently establish vehicle right-of-way.",
            [{"effect_type": "sign_meaning", "action": "crossing",
              "scope": {"facility": "pedestrian_underpass", "direction_fact": "pedestrian_direction_matches_i424_diagram",
                        "blanket_permission": False}}],
        ))
    return rules


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-48"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
