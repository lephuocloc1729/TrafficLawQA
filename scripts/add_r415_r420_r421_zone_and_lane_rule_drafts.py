"""Add cautious QCVN drafts for R.415 lane boards and R.420/R.421 area limits."""

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


def legal_basis(section: str) -> list[dict]:
    return [{
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": section,
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }]


def make_rule(
    rule_id: str,
    code: str,
    intent_kinds: list[str],
    actions: list[str],
    predicates: list[str],
    description: str,
    section: str,
    *,
    scope: dict | None = None,
) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": intent_kinds,
        "actions": actions,
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition(name) for name in predicates],
        "reason": description,
        "effects": [{
            "effect_type": "sign_meaning",
            "action": actions[0],
            "scope": {"blanket_permission": False, **(scope or {})},
            "description": description,
        }],
        "additional_legal_basis": legal_basis(section),
    }


def make_rules() -> list[dict]:
    return [
        make_rule(
            "R415A_LANE_AND_VEHICLE_DIAGRAM_MEANING",
            "R.415a",
            ["QUERY_SIGN_MEANING", "CHECK_PERMISSION", "ASK_CONDITION"],
            ["unknown", "circulation", "entering", "lane_change"],
            [
                "sign_effective_scope_confirmed",
                "r415_lane_count_per_direction_resolved",
                "r415_vehicle_symbols_mapped_to_each_lane",
                "requested_vehicle_and_lane_resolved",
                "road_markings_for_lane_change_resolved",
                "other_applicable_lane_rules_resolved",
            ],
            (
                "QCVN 41:2024/BGTVT Appendix D.16(a,c-d), Figure D.18a: R.415a shows the number "
                "of lanes and vehicle types permitted on each lane; the displayed layout is only an example. "
                "It applies to roads with two to four motor-vehicle lanes per direction. Lane changes must "
                "follow actual road markings and applicable rules; the sign does not itself grant overall "
                "permission to travel or change lanes. It does not apply to vehicles entering/exiting or stopping/parking."
            ),
            "Appendix D.16(a,c-d), Figure D.18a",
            scope={
                "interpretation": "positive_vehicle_to_lane_mapping_only",
                "excluded_use_cases": ["entering_or_exiting_road", "stopping", "parking"],
            },
        ),
        make_rule(
            "R415B_END_OF_VEHICLE_LANE_SCOPE_MEANING",
            "R.415b",
            ["QUERY_SIGN_MEANING", "CHECK_PERMISSION", "ASK_CONDITION"],
            ["unknown", "circulation", "entering", "lane_change"],
            [
                "sign_effective_scope_confirmed",
                "r415b_ending_lane_segment_resolved",
                "subsequent_lane_rule_for_vehicle_and_lane_resolved",
            ],
            (
                "QCVN 41:2024/BGTVT Appendix D.16(a-b), Figure D.18b: R.415b indicates the end "
                "of the regulated lane-by-lane vehicle-use segment. It does not by itself establish which "
                "lane a particular vehicle may use after the sign or grant blanket permission."
            ),
            "Appendix D.16(a-b), Figure D.18b",
            scope={"blanket_permission": False},
        ),
        make_rule(
            "R420_START_POPULATED_AREA_SCOPE_NOT_SPEED_VALUE",
            "R.420",
            ["QUERY_SIGN_MEANING", "CHECK_PERMISSION", "ASK_CONDITION"],
            ["unknown", "speed", "circulation"],
            [
                "sign_effective_scope_confirmed",
                "r420_area_scope_until_r421_resolved",
                "applicable_populated_area_rule_for_vehicle_and_action_resolved",
            ],
            (
                "QCVN 41:2024/BGTVT Appendix D.17(a-d), Figure D.19: R.420 marks the start of the "
                "populated-area segment; its scope ends at R.421. It signals that populated-area road rules "
                "apply and specifically directs drivers to obey a posted P.127 maximum-speed sign. R.420 "
                "does not itself state a numeric speed limit; the applicable vehicle/action-specific rule and "
                "any posted signs must be resolved separately."
            ),
            "Appendix D.17(a-d), Figure D.19",
            scope={"zone_start": True, "zone_end_sign": "R.421", "numeric_speed_limit": None},
        ),
        make_rule(
            "R421_END_POPULATED_AREA_SCOPE_NO_PERMISSION_INFERENCE",
            "R.421",
            ["QUERY_SIGN_MEANING", "CHECK_PERMISSION", "ASK_CONDITION"],
            ["unknown", "speed", "circulation"],
            [
                "sign_effective_scope_confirmed",
                "vehicle_past_r421_confirmed",
                "applicable_rule_after_populated_area_for_vehicle_and_action_resolved",
            ],
            (
                "QCVN 41:2024/BGTVT Appendix D.18, Figure D.20: R.421 marks the end of the "
                "populated-area segment and of the road rules that apply by virtue of that area scope. "
                "It does not establish a new speed limit or blanket permission; the rules and signs applying "
                "after the boundary must be resolved separately."
            ),
            "Appendix D.18, Figure D.20",
            scope={"zone_end": True, "numeric_speed_limit": None, "blanket_permission": False},
        ),
    ]


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-51"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
