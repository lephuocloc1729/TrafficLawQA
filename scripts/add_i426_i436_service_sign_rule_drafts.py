"""Add official-source meaning and I.436 duty drafts for QCVN Appendix E.27-E.39."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"

# Each description is a close English summary of the cited official QCVN text;
# none of these location signs is treated as permission to perform an action.
SIGN_MEANINGS = {
    "I.426": ("E.27, Figure E.27", "Indicates a nearby emergency medical station."),
    "I.427a": ("E.28, Figure E.28", "Indicates a workshop or station that repairs cars or motorcycles that have broken down on the road."),
    "I.427b": ("E.29, Figure E.29", "Indicates the location of a vehicle weight-control facility."),
    "I.428a": ("E.30, Figure E.30", "Indicates a fuel station serving road users."),
    "I.428b": ("E.30, Figure E.30", "Indicates an electric-vehicle charging station serving road users."),
    "I.428c": ("E.30, Figure E.30", "Indicates a place providing both a fuel station and an electric-vehicle charging station."),
    "I.429": ("E.31, Figure E.31", "Indicates a place where vehicles can be washed."),
    "I.430": ("E.32, Figure E.32", "Indicates a public telephone station provided for road users."),
    "I.431": ("E.33, Figure E.33", "Indicates a roadside rest stop with services for travelers, such as food, rest, or fuel; the depicted symbols are selected to match the services available."),
    "I.432": ("E.34, Figure E.34", "Indicates a hotel serving road travelers."),
    "I.433a": ("E.35a, Figure E.35a", "Indicates a holiday/recreation destination."),
    "I.433b": ("E.35b, Figure E.35b(a)", "On foreign-affairs routes and routes with substantial foreign traffic, indicates an upcoming camping place."),
    "I.433c": ("E.35b, Figure E.35b(b)", "On foreign-affairs routes and routes with substantial foreign traffic, indicates an upcoming place for mobile homes/caravans."),
    "I.433d": ("E.35b, Figure E.35b(c)", "On foreign-affairs routes and routes with substantial foreign traffic, indicates an upcoming place for camping and mobile homes/caravans."),
    "I.433e": ("E.35c, Figure E.35c", "On foreign-affairs routes and routes with substantial foreign traffic, indicates an upcoming guesthouse/youth hostel."),
    "I.434a": ("E.36, Figure E.36", "Indicates a bus boarding/alighting stop. Supplementary panels may show route number, route name, origin, destination, or route; those panels may be beside the sign or in the passenger area. The official text also describes alternate wording for non-terminal bus stops and fixed-route passenger stops."),
    "I.434b": ("E.37, Figure E.37", "Indicates a truck parking area for loading and unloading goods; where it is not legally a bus-terminal-type facility, the sign text is changed to 'Truck Stop'."),
    "I.435": ("E.38, Figure E.38", "Indicates a tram stop for passenger boarding and alighting."),
    "I.436": ("E.39, Figure E.39", "Indicates the location of a traffic-police station."),
}


def condition(predicate: str) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": True,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def basis(section: str) -> list[dict]:
    return [{
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": f"Appendix {section}",
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }]


def meaning_rule(code: str, section: str, description: str) -> dict:
    return {
        "rule_id": f"{code.replace('.', '').upper().replace('-', '_')}_OFFICIAL_SIGN_MEANING",
        "code": code,
        "intent_kinds": ["QUERY_SIGN"],
        "actions": ["unknown"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed")],
        "reason": f"QCVN 41:2024/BGTVT Appendix {section}: {description}",
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "unknown",
            "scope": {"blanket_permission": False},
            "description": description,
        }],
        "additional_legal_basis": basis(section),
    }


def duty_rule(rule_id: str, action: str, effect_type: str, description: str) -> dict:
    return {
        "rule_id": rule_id,
        "code": "I.436",
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": [action],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed"),
            condition("i436_vehicle_passing_through_police_station_area_confirmed"),
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix E.39 states that vehicles must reduce speed to a safe level "
            "and must not overtake while passing through the indicated area. " + description
        ),
        "effects": [{
            "effect_type": effect_type,
            "action": action,
            "scope": {"area_fact": "i436_vehicle_passing_through_police_station_area_confirmed"},
            "description": description,
        }],
        "additional_legal_basis": basis("E.39, Figure E.39"),
    }


def make_rules() -> list[dict]:
    rules = [meaning_rule(code, section, description)
             for code, (section, description) in SIGN_MEANINGS.items()]
    rules.extend([
        duty_rule(
            "I436_REQUIRE_SAFE_SPEED_WHILE_PASSING_POLICE_STATION_AREA",
            "speed", "require_action", "Reduce speed to a safe level while passing through the area.",
        ),
        duty_rule(
            "I436_PROHIBIT_OVERTAKING_WHILE_PASSING_POLICE_STATION_AREA",
            "overtaking", "prohibit_action", "Do not overtake while passing through the area.",
        ),
    ])
    return rules


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-50"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
