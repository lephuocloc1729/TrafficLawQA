"""Add fail-closed drafts for DP.133–DP.135 using QCVN 41:2024/BGTVT."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"


def condition(predicate: str, operator: str = "exists", expected: object = None) -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def make_rule(rule_id: str, code: str, actions: list[str], actor_scope: list[str],
              conditions: list[dict], reason: str, *, effect: dict | None = None) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "actions": actions,
        "actor_scope": actor_scope,
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": conditions,
        "reason": reason,
        "effects": [effect] if effect else [],
    }


SCOPE = condition("sign_effective_scope_confirmed", "equals", True)


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {item.get("official_code"): item for item in catalog.get("items", [])}
    basis_by_code = {
        "DP.133": "Phụ lục B.33 (PDF page 95)",
        "DP.134": "Phụ lục B.34 (PDF page 95)",
        "DP.135": "Phụ lục B.35 (PDF pages 95–96)",
    }
    for code in basis_by_code:
        if code not in entries or not entries[code].get("legal_basis"):
            raise ValueError(f"Missing official catalog provenance for {code}")

    motorized_actors = ["motor_vehicle", "priority_vehicle", "emergency_vehicle"]
    rules = [
        make_rule(
            "DP133_ENDS_OVERTAKING_PROHIBITIONS", "DP.133", ["overtaking"],
            motorized_actors, [SCOPE],
            "B.33: DP.133 ends the effects of P.125 and P.126; motor vehicles may overtake subject to the legal conditions for overtaking. This sign alone does not establish that a particular maneuver satisfies those conditions.",
            effect={"effect_type": "sign_meaning", "action": "overtaking",
                    "description": "The P.125/P.126 overtaking prohibition ends here; other overtaking conditions remain applicable."},
        ),
        make_rule(
            "DP134_ENDS_POSTED_MAXIMUM_SPEED", "DP.134", ["speed"],
            motorized_actors,
            [SCOPE, condition("general_max_speed_kmh")],
            "B.34: DP.134 ends the P.127 maximum-speed restriction; the applicable statutory maximum remains controlling. The value must be resolved for this road and vehicle class.",
            effect={"effect_type": "maximum", "action": "speed", "unit": "km/h",
                    "value_predicate": "general_max_speed_kmh"},
        ),
    ]

    dp135_cases = [
        ("OVERTAKING_P125", ["overtaking"], "P.125"),
        ("OVERTAKING_P126", ["overtaking"], "P.126"),
        ("SPEED_P127", ["speed"], "P.127"),
        ("HORN_P128", ["using_horn"], "P.128"),
        ("STOPPING_P130", ["stopping", "parking"], "P.130"),
        ("PARKING_P131A", ["parking"], "P.131a"),
        ("PARKING_P131B", ["parking"], "P.131b"),
        ("PARKING_P131C", ["parking"], "P.131c"),
        ("FOLLOWING_DISTANCE_P121", ["following_distance"], "P.121"),
    ]
    for suffix, actions, ended_code in dp135_cases:
        value_predicate = "general_max_speed_kmh" if "SPEED" in suffix else None
        conditions = [SCOPE, condition("active_restriction_sign_codes", "contains", ended_code)]
        if value_predicate:
            conditions.append(condition(value_predicate))
        effect = {
            "effect_type": "sign_meaning",
            "action": actions[0],
            "description": f"DP.135 ends the known active {ended_code} prohibition, but does not establish overall permission.",
        }
        if value_predicate:
            effect = {"effect_type": "maximum", "action": "speed", "unit": "km/h",
                      "value_predicate": value_predicate}
        rules.append(make_rule(
            f"DP135_ENDS_{suffix}", "DP.135", actions,
            motorized_actors, conditions,
            f"B.35: DP.135 ends the applicable previously placed {ended_code} restriction. Only that known sign effect is removed; other legal restrictions and conditions are not inferred away.",
            effect=effect,
        ))

    for rule in rules:
        code = rule["code"]
        rule["additional_legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": basis_by_code[code],
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
        }]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update({item["rule_id"]: item for item in rules})
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
