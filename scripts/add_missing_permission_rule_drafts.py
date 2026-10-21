"""Add conservative permission rules for designated stops and priority entry."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rulebook_draft.json"
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-16"


def condition(predicate: str, value: object = True) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": value,
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


def stop_rule(
    *, rule_id: str, code: str, actor: str, action_description: str,
    section: str, extra_condition: str,
) -> dict:
    actions = ["stopping", "parking"]
    return {
        "rule_id": rule_id,
        "code": code,
        "actions": actions,
        "actor_scope": [actor],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed"),
            condition("i434_designated_stop_location_resolved"),
            condition(extra_condition),
            condition("other_applicable_stopping_constraints_resolved"),
        ],
        "reason": (
            f"QCVN 41:2024/BGTVT {section} identifies a designated location for {action_description}. "
            "The sign alone does not establish that the queried stop/parking maneuver is lawful in the "
            "current scene; exact sign scope, relevant service/use, and other stopping constraints must be "
            "resolved. This rule never grants blanket permission."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "stopping",
            "scope": {"designated_location_fact": "i434_designated_stop_location_resolved"},
            "description": action_description,
        }],
        "additional_legal_basis": [basis(section)],
    }


def make_rules() -> list[dict]:
    return [
        {
            "rule_id": "P102_PRIORITY_VEHICLE_EXCEPTION_REQUIRES_ROUTE_REVIEW",
            "code": "P.102",
            "actions": ["entering", "circulation"],
            "actor_scope": ["priority_vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed"),
                condition("priority_vehicle_status_confirmed"),
                condition("opposite_direction_route_passable"),
                condition("expressway_opposite_direction_limit_resolved"),
                condition("other_applicable_signs_and_controls_resolved"),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix B.2 excepts vehicles with statutory priority rights from P.102. "
                "The exception does not itself establish blanket permission: qualifying priority status, a "
                "passable route, expressway-specific limits, and other applicable controls must be reviewed."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "entering",
                "scope": {"exception": "vehicles_with_confirmed_priority_rights"},
                "description": "P.102's prohibition may not apply to a qualifying priority vehicle; other route rules remain unresolved.",
            }],
            "additional_legal_basis": [
                basis("Appendix B.2: P.102 exception for priority vehicles"),
                {
                    "document": "Luật Trật tự, an toàn giao thông đường bộ số 36/2024/QH15",
                    "edition": "2024; effective 2025-01-01",
                    "section_or_figure": "Điều 27 khoản 3: quyền và giới hạn của xe ưu tiên khi làm nhiệm vụ",
                    "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15_tiep.pdf",
                    "accessed_at": ACCESS_DATE,
                    "verification_status": "verified",
                },
            ],
        },
        stop_rule(
            rule_id="I434A_BUS_STOP_PERMISSION_REQUIRES_SERVICE_AND_SCOPE",
            code="I.434a",
            actor="bus",
            action_description="a bus stopping place for passenger boarding and alighting",
            section="Appendix E.36, Figure E.36",
            extra_condition="i434a_bus_service_match_confirmed",
        ),
        stop_rule(
            rule_id="I434B_TRUCK_STOP_PERMISSION_REQUIRES_LOADING_SCOPE",
            code="I.434b",
            actor="truck",
            action_description="a truck stopping/parking place for receiving or delivering goods",
            section="Appendix E.37, Figure E.37",
            extra_condition="i434b_loading_or_unloading_purpose_confirmed",
        ),
    ]


def main() -> None:
    rulebook = json.loads(RULEBOOK_PATH.read_text(encoding="utf-8"))
    p102 = next((rule for rule in rulebook.get("rules", []) if rule.get("rule_id") == "P102_NO_ENTRY"), None)
    if p102 is None or "priority_vehicle" not in p102.get("excluded_actor_scope", []):
        raise ValueError("P102_NO_ENTRY must retain its priority-vehicle exclusion")

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {rule.get("rule_id"): rule for rule in registry.get("rules", [])}

    # Keep R.412a aligned with D.14: passenger coaches/buses belong here;
    # passenger cars belong to R.412b.
    for rule_id in (
        "R412A_ALLOW_DESIGNATED_VEHICLE",
        "R412A_ALLOW_BELOW_DISPLAYED_SEAT_LIMIT",
        "R412A_PROHIBIT_AT_OR_ABOVE_DISPLAYED_SEAT_LIMIT",
    ):
        if rule_id not in by_id:
            raise ValueError(f"Expected existing rule {rule_id}")
        by_id[rule_id]["actor_scope"] = ["bus"]
    other_vehicle_rule = by_id.get("R412A_PROHIBIT_OTHER_VEHICLES")
    if other_vehicle_rule is None:
        raise ValueError("Expected existing rule R412A_PROHIBIT_OTHER_VEHICLES")
    other_vehicle_rule["excluded_actor_scope"] = ["bus", "priority_vehicle"]

    for rule in make_rules():
        by_id[rule["rule_id"]] = rule

    registry["rules"] = list(by_id.values())
    registry["registry_version"] = "qcvn41-rule-drafts-56"
    registry["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
