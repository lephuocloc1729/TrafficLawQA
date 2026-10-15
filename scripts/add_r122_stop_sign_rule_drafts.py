"""Add review-gated R.122 rules transcribed from official QCVN 41:2024 sources."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/van-ban/thong-tu-so-51-2024-tt-bgtvt-43387.htm"


def condition(predicate: str, expected: object, *, mismatch: str = "NOT_APPLICABLE") -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": mismatch,
    }


def basis(section: str) -> dict:
    return {
        "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": section,
        "source_url": SOURCE_URL,
        "accessed_at": "2026-10-05",
        "verification_status": "verified",
        "gazette_source_url": GAZETTE_URL,
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not any(item.get("official_code") == "R.122" for item in catalog.get("items", [])):
        raise ValueError("Official-source catalog is missing R.122")

    drafts = [
        {
            "rule_id": "R122_REQUIRE_STOP_BEFORE_SIGN_OR_LINE",
            "code": "R.122",
            "intent_kinds": ["CHECK_PERMISSION"],
            "actions": ["stopping"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed", True),
                condition("vehicle_approaching_r122_stop_position", True),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix D.1(a-b), Figure D.1: R.122 requires motorized and "
                "non-motorized vehicles, including priority vehicles, to stop before the sign or stop line. "
                "The engine requires the sign's applicability and the relevant approach/stop position to be confirmed."
            ),
            "effects": [{
                "effect_type": "require_action",
                "action": "stopping",
                "description": "Stop before R.122 or the stop line when its scope and approach are confirmed.",
            }],
            "additional_legal_basis": [basis("Appendix D, D.1(a-b), Figure D.1")],
        },
        {
            "rule_id": "R122_PROCEED_ONLY_AFTER_CONTROL_PERMISSION",
            "code": "R.122",
            "intent_kinds": ["CHECK_PERMISSION"],
            "actions": ["yielding", "circulation", "entering", "going_straight", "turn_left", "turn_right", "u_turn", "crossing"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed", True),
                condition("vehicle_stopped_before_r122", True),
                condition("r122_control_signal_or_officer_permits_proceeding", True),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix D.1(b), Figure D.1: after stopping before R.122 or the stop line, "
                "proceed only when the traffic controller or signal/flag permits it. A matching permission "
                "condition is not blanket permission; other applicable controls and signs remain to be checked."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "proceeding",
                "description": "Proceeding is conditional on an explicit permitting traffic signal/controller after stopping.",
            }],
            "additional_legal_basis": [basis("Appendix D, D.1(b), Figure D.1")],
        },
        {
            "rule_id": "R122_PROCEED_WITHOUT_ACTIVE_CONTROL_ONLY_IF_SAFE",
            "code": "R.122",
            "intent_kinds": ["CHECK_PERMISSION"],
            "actions": ["yielding", "circulation", "entering", "going_straight", "turn_left", "turn_right", "u_turn", "crossing"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed", True),
                condition("vehicle_stopped_before_r122", True),
                condition("r122_active_traffic_control_present", False),
                condition("intersection_safe_to_proceed", True),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix D.1(b), Figure D.1: if no traffic controller is present and "
                "signals are absent or not operating, proceeding is conditional on there being no traffic-safety "
                "hazard. The engine does not infer safety from missing observations."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "proceeding",
                "description": "Proceeding without active control requires a confirmed safe intersection after stopping.",
            }],
            "additional_legal_basis": [basis("Appendix D, D.1(b), Figure D.1")],
        },
        {
            "rule_id": "R122_S506B_YIELD_TO_PRIORITY_ROAD_TRAFFIC",
            "code": "R.122",
            "intent_kinds": ["CHECK_PERMISSION"],
            "actions": ["circulation", "entering", "going_straight", "turn_left", "turn_right", "u_turn", "crossing"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed", True),
                condition("vehicle_stopped_before_r122", True),
                condition("linked_s506b_panel_present", True),
                condition("actor_has_priority_right", False),
                condition("priority_road_traffic_requires_yield", True),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT Appendix D.1(c) and Appendix F.8: when S.506b is confidently identified "
                "and linked below R.122, traffic on the non-priority road must yield to traffic on the priority "
                "road, except traffic with priority rights. The engine requires explicit actor and scene facts "
                "and does not treat this result as blanket permission after yielding."
            ),
            "effects": [{
                "effect_type": "require_action",
                "action": "yielding",
                "description": "Yield to priority-road traffic when the linked S.506b relationship and scene conditions are confirmed.",
            }],
            "additional_legal_basis": [
                basis("Appendix D, D.1(c), Figure D.1"),
                basis("Appendix F.8, Figure F.8"),
            ],
        },
    ]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id.update({rule["rule_id"]: rule for rule in drafts})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-26"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
