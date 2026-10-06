"""Add review-gated parking and U-turn facility rules from QCVN Appendix E."""

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


def basis(section: str, figure: str, role: str) -> dict:
    return {
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": f"Appendix E.{section}, Figure E.{figure}; {role}",
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }


def rule(code: str, rule_id: str, action: str, section: str, extra_conditions: list[str],
         reason: str, scope: dict, figure: str | None = None) -> dict:
    section_role = {
        "8": "I.408 parking-place meaning, defined extent, and termination 10 m before the next intersection",
        "8a": "I.408a partial-sidewalk parking; passenger-side wheels must be on the sidewalk",
        "9": "I.409 indicates an authorized U-turn location; S.502 may state distance",
        "10": "I.410 indicates an authorized U-turn area and depicts the maneuver",
    }[section]
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION"],
        "actions": [action],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "ALLOWED",
        "conditions": [condition("sign_effective_scope_confirmed"), *map(condition, extra_conditions)],
        "reason": reason,
        "effects": [{
            "effect_type": "permit_action",
            "action": action,
            "scope": scope,
            "description": reason,
        }],
        "additional_legal_basis": [basis(section, figure or section, section_role)],
    }


def make_rules() -> list[dict]:
    return [
        rule(
            "I.408", "I408_PERMIT_PARKING_WITHIN_DESIGNATED_PLACE", "parking", "8",
            ["parking_location_within_i408_designated_place",
             "i408_scope_and_supplementary_panels_resolved",
             "other_applicable_parking_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.8(a-b): I.408 indicates a place where parking is permitted. Its scope ends 10 m before the next intersection unless a supplementary sign defines the permitted zone. This permission applies only at the confirmed designated place, after supplementary scope and other parking constraints are resolved.",
            {"location_fact": "parking_location_within_i408_designated_place",
             "supplementary_scope_fact": "i408_scope_and_supplementary_panels_resolved",
             "blanket_permission": False},
        ),
        rule(
            "I.408a", "I408A_PERMIT_PARTIAL_SIDEWALK_PARKING", "parking", "8a",
            ["parking_location_within_i408a_designated_place",
             "i408a_scope_and_supplementary_panels_resolved",
             "passenger_side_wheels_on_sidewalk_confirmed",
             "other_applicable_parking_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.8a(a-b): I.408a indicates a place where partial-sidewalk parking is permitted; the passenger-side wheels must be on the sidewalk. The rule applies only at the confirmed place and with the required wheel placement.",
            {"location_fact": "parking_location_within_i408a_designated_place",
             "required_wheel_placement": "passenger_side_wheels_on_sidewalk_confirmed",
             "blanket_permission": False},
            figure="8a",
        ),
        rule(
            "I.409", "I409_PERMIT_U_TURN_AT_DESIGNATED_PLACE", "u_turn", "9",
            ["u_turn_location_matches_i409_designated_place",
             "i409_position_and_s502_distance_resolved",
             "other_applicable_turn_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.9(a-b): I.409 indicates a position where U-turning is permitted; S.502 may be placed below it to state distance. Permission is limited to the resolved indicated position and any supplementary distance.",
            {"location_fact": "u_turn_location_matches_i409_designated_place",
             "distance_scope_fact": "i409_position_and_s502_distance_resolved",
             "blanket_permission": False},
        ),
        rule(
            "I.410", "I410_PERMIT_U_TURN_WITHIN_DESIGNATED_AREA", "u_turn", "10",
            ["u_turn_location_within_i410_designated_area",
             "requested_u_turn_matches_i410_diagram",
             "i410_position_and_s502_distance_resolved",
             "other_applicable_turn_constraints_resolved"],
            "QCVN 41:2024/BGTVT Appendix E.10(a-b): I.410 indicates an area where U-turning is permitted and depicts how to perform the maneuver; S.502 may state distance. Permission is limited to the confirmed area and depicted maneuver.",
            {"area_fact": "u_turn_location_within_i410_designated_area",
             "maneuver_fact": "requested_u_turn_matches_i410_diagram",
             "distance_scope_fact": "i410_position_and_s502_distance_resolved",
             "blanket_permission": False},
        ),
    ]


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-42"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
