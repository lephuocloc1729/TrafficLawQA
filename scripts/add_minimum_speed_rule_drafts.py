"""Add conservative R.306/R.307 minimum-speed rules from official QCVN text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
ACTORS = ["motor_vehicle", "priority_vehicle", "emergency_vehicle"]


def condition(predicate: str, operator: str = "exists", expected: object = None,
              *, on_mismatch: str = "NOT_APPLICABLE") -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": on_mismatch,
    }


def rule(rule_id: str, code: str, actions: list[str], outcome: str,
         conditions: list[dict], reason: str, effects: list[dict]) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": actions,
        "actor_scope": ACTORS,
        "excluded_actor_scope": [],
        "outcome": outcome,
        "conditions": conditions,
        "reason": reason,
        "effects": effects,
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {item.get("official_code"): item for item in catalog.get("items", [])}
    for code in ("R.306", "R.307"):
        if code not in entries or not entries[code].get("legal_basis"):
            raise ValueError(f"Missing official catalog provenance for {code}")

    r306_scope = condition("sign_effective_scope_confirmed", "equals", True)
    r306_ref = {
        "document": "QCVN 41:2024/BGTVT", "edition": "2024",
        "section_or_figure": "Phụ lục D.6 (PDF page 118)",
        "source_url": SOURCE_URL, "accessed_at": "2026-10-05",
        "verification_status": "verified",
    }
    r307_ref = {
        "document": "QCVN 41:2024/BGTVT", "edition": "2024",
        "section_or_figure": "Phụ lục D.7 (PDF pages 118–119)",
        "source_url": SOURCE_URL, "accessed_at": "2026-10-05",
        "verification_status": "verified",
    }
    rules = [
        rule(
            "R306_MINIMUM_SPEED_IN_SAFE_CONDITIONS", "R.306", ["speed"], "CONDITIONAL",
            [r306_scope, condition("minimum_speed_limit_kmh"),
             condition("traffic_conditions_favorable_and_safe", "equals", True)],
            "D.6(a–c): motor vehicles must not operate below the displayed minimum in favorable and safe traffic conditions. The displayed km/h value and those conditions must be known; this rule does not treat the minimum as a recommended speed above which every speed is safe.",
            [{"effect_type": "minimum", "action": "speed", "unit": "km/h",
              "value_predicate": "minimum_speed_limit_kmh"}],
        ),
        rule(
            "R306_NO_ENTRY_BELOW_MANUFACTURER_CAPABILITY", "R.306", ["entering"], "PROHIBITED",
            [r306_scope, condition("minimum_speed_limit_kmh"),
             condition("vehicle_manufacturer_max_speed_kmh", "lt_fact", "minimum_speed_limit_kmh")],
            "D.6(b): a vehicle whose manufacturer-specified maximum speed is below the sign's minimum speed is not permitted to enter this road. Both values and the sign scope must be resolved.",
            [{"effect_type": "prohibit_action", "action": "entering"}],
        ),
        rule(
            "R307_MINIMUM_SPEED_ENDS_WHEN_NOT_IMPEDING", "R.307", ["speed"], "CONDITIONAL",
            [condition("sign_effective_scope_confirmed", "equals", True),
             condition("would_impede_other_vehicles", "equals", False)],
            "D.7(b): R.307 ends the R.306 minimum-speed requirement; vehicles may travel below that former value, provided they do not impede other vehicles. Other speed limits and road conditions remain unchecked.",
            [{"effect_type": "sign_meaning", "action": "speed",
              "description": "The R.306 minimum-speed requirement has ended; slower travel is not a blanket permission and must not impede other vehicles."}],
        ),
        rule(
            "R307_PROHIBIT_SLOW_SPEED_THAT_IMPEDES", "R.307", ["speed"], "PROHIBITED",
            [condition("sign_effective_scope_confirmed", "equals", True),
             condition("would_impede_other_vehicles", "equals", True)],
            "D.7(b): although R.307 ends the R.306 minimum-speed requirement, slower driving must not impede other vehicles. This outcome requires an explicit resolved fact that the proposed speed would impede traffic.",
            [{"effect_type": "prohibit_action", "action": "speed"}],
        ),
    ]
    for draft in rules:
        draft["additional_legal_basis"] = [dict(r306_ref if draft["code"] == "R.306" else r307_ref)]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update({item["rule_id"]: item for item in rules})
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
