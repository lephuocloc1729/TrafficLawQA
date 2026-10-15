"""Add W.207 variant rules for priority-road intersections from official QCVN C.7."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"
VARIANTS = ("W.207a", "W.207b", "W.207c", "W.207d", "W.207e", "W.207f",
            "W.207g", "W.207h", "W.207i", "W.207k", "W.207l")


def condition(predicate: str, expected: object) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def make_rule(code: str) -> dict:
    scope_conditions = [
        condition("sign_effective_scope_confirmed", True),
        condition("vehicle_on_w207_priority_road_approach", True),
        condition("w207_intersection_geometry_matches_sign", True),
        condition("intersection_traffic_control_active", False),
        condition("actor_has_priority_right", False),
        condition("priority_vehicle_conflict_present", False),
    ]
    if code in {"W.207d", "W.207e"}:
        # C.7(d) requires an auxiliary scope sign for the multi-intersection
        # form; confirming only this one diagram is not enough to infer extent.
        scope_conditions.append(condition("w207_multiple_intersection_scope_resolved", True))
    figure = "Figure C.7" if code in {"W.207a", "W.207b", "W.207c", "W.207d", "W.207e"} else (
        "Figure C.8" if code in {"W.207f", "W.207g", "W.207h"} else "Figure C.9"
    )
    return {
        "rule_id": f"{code.replace('.', '').upper()}_PRIORITY_THROUGH_UNCONTROLLED_INTERSECTION",
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": [
            "yielding", "entering", "circulation", "going_straight", "turn_left",
            "turn_right", "u_turn", "crossing",
        ],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": scope_conditions,
        "reason": (
            f"QCVN 41:2024/BGTVT Appendix C.7(a-d), {figure}: {code} warns that the vehicle is on a "
            "priority road approaching an intersection with a non-priority road. Vehicles on the signed "
            "road have priority through the intersection, except they must yield to vehicles with priority "
            "rights. The rule requires the actual approach and depicted junction geometry to match; it is "
            "limited to intersections without active traffic control under the priority-road principle in "
            "Appendix E.1(b). It does not establish blanket permission."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {
                "priority_road_right_of_way": True,
                "exception": "vehicles_with_priority_rights",
                "variant": code,
            },
            "description": "Vehicles on this confirmed priority-road approach have priority through the depicted uncontrolled intersection, except for priority vehicles.",
        }],
        "additional_legal_basis": [
            {
                "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
                "edition": "2024; effective 2025-01-01",
                "section_or_figure": f"Appendix C, C.7(a-d), {figure}, Figure C.9 variants; Gazette page 18",
                "source_url": SOURCE_URL,
                "accessed_at": ACCESS_DATE,
                "verification_status": "verified",
                "gazette_source_url": GAZETTE_URL,
            },
            {
                "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
                "edition": "2024; effective 2025-01-01",
                "section_or_figure": "Appendix E, E.1(b): priority-road principle at uncontrolled intersections and its effect when traffic control is present",
                "source_url": SOURCE_URL,
                "accessed_at": ACCESS_DATE,
                "verification_status": "verified",
                "gazette_source_url": GAZETTE_URL,
            },
        ],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_codes = {item.get("official_code") for item in catalog.get("items", [])}
    if not set(VARIANTS).issubset(catalog_codes):
        raise ValueError("Official-source sign catalog is missing one or more W.207 variants")

    additions = [make_rule(code) for code in VARIANTS]
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-27"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
