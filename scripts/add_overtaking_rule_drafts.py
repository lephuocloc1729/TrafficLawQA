"""Add conservative P.125/P.126 overtaking rules from QCVN 41:2024/BGTVT."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
SPECIAL_OVERTAKEN_CLASSES = ["two_wheeled_motorcycle", "motorbike", "motorcycle", "moped"]
SCOPE = {
    "predicate": "sign_effective_scope_confirmed",
    "operator": "equals",
    "expected_value": True,
    "required": True,
    "on_missing": "CONDITIONAL",
    "on_mismatch": "NOT_APPLICABLE",
}


def condition(predicate: str, operator: str = "exists", expected: object = None,
              *, mismatch: str = "NOT_APPLICABLE") -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": mismatch,
    }


def rule(rule_id: str, code: str, outcome: str, conditions: list[dict], reason: str,
         *, actors: list[str], effect: dict | None = None) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": ["overtaking"],
        "actor_scope": actors,
        "excluded_actor_scope": [],
        "outcome": outcome,
        "conditions": conditions,
        "reason": reason,
        "effects": [effect] if effect else [],
    }


def meaning_effect(text: str) -> dict:
    return {"effect_type": "sign_meaning", "action": "overtaking", "description": text}


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {item.get("official_code"): item for item in catalog.get("items", [])}
    scope_and_motor_actor = [SCOPE, condition("actor_is_motor_vehicle", "equals", True)]
    reasons = {
        "P.125": (
            "B.25(a,b): biển cấm các loại xe cơ giới vượt nhau, kể cả xe được ưu tiên; "
            "ngoại lệ nêu rõ là được vượt xe máy hai bánh và xe gắn máy. Thiếu phân loại phương tiện "
            "đang bị vượt thì không kết luận."
        ),
        "P.126": (
            "B.26(a,b): cấm ô tô tải có khối lượng chuyên chở ghi trong giấy chứng nhận "
            "lớn hơn 3.500 kg vượt xe cơ giới khác, kể cả xe ưu tiên; được vượt xe máy hai bánh "
            "và xe gắn máy. Chỉ dùng sức chở theo giấy chứng nhận, không dùng khối lượng hàng đang chở."
        ),
    }
    rules = [
        rule(
            "P125_NO_OVERTAKING_OTHER_MOTOR_VEHICLES", "P.125", "PROHIBITED",
            [*scope_and_motor_actor,
             condition("overtaken_vehicle_is_motor_vehicle", "equals", True),
             condition("overtaken_vehicle_class", "not_in", SPECIAL_OVERTAKEN_CLASSES)],
            reasons["P.125"], actors=["motor_vehicle", "emergency_vehicle", "priority_vehicle"],
            effect={"effect_type": "prohibit_action", "action": "overtaking"},
        ),
        rule(
            "P125_TWO_WHEEL_MOTORCYCLE_MOPED_EXCEPTION", "P.125", "CONDITIONAL",
            [*scope_and_motor_actor,
             condition("overtaken_vehicle_class", "in", SPECIAL_OVERTAKEN_CLASSES)],
            "B.25(b): P.125 expressly permits overtaking a two-wheeled motorcycle or moped in the scope of this sign. This is only an exception to P.125, not confirmation that all other traffic rules allow the maneuver.",
            actors=["motor_vehicle", "emergency_vehicle", "priority_vehicle"],
            effect=meaning_effect("P.125 does not prohibit overtaking this stated target class; other applicable rules remain unchecked."),
        ),
        rule(
            "P126_HEAVY_TRUCK_NO_OVERTAKING_OTHER_MOTOR_VEHICLES", "P.126", "PROHIBITED",
            [SCOPE, condition("actor_vehicle_category", "equals", "truck"),
             condition("certified_payload_capacity_t", "gt", 3.5),
             condition("overtaken_vehicle_is_motor_vehicle", "equals", True),
             condition("overtaken_vehicle_class", "not_in", SPECIAL_OVERTAKEN_CLASSES)],
            reasons["P.126"],
            actors=["truck", "emergency_vehicle", "priority_vehicle"],
            effect={"effect_type": "prohibit_action", "action": "overtaking"},
        ),
        rule(
            "P126_TWO_WHEEL_MOTORCYCLE_MOPED_EXCEPTION", "P.126", "CONDITIONAL",
            [SCOPE, condition("actor_vehicle_category", "equals", "truck"),
             condition("certified_payload_capacity_t", "gt", 3.5),
             condition("overtaken_vehicle_class", "in", SPECIAL_OVERTAKEN_CLASSES)],
            "B.26(b): this stated exception applies to vehicles that this sign otherwise restricts; the sign itself does not prohibit overtaking a two-wheeled motorcycle or moped. This is not overall permission to overtake.",
            actors=["truck", "emergency_vehicle", "priority_vehicle"],
            effect=meaning_effect("P.126 does not prohibit overtaking this stated target class; other applicable rules remain unchecked."),
        ),
        rule(
            "P126_CAPACITY_THRESHOLD_NOT_MET", "P.126", "CONDITIONAL",
            [SCOPE, condition("actor_vehicle_category", "equals", "truck"),
             condition("certified_payload_capacity_t", "lte", 3.5)],
            "B.26(b) sets the covered truck threshold at a certified carrying capacity greater than 3.500 t. This sign alone does not establish a prohibition for a truck at or below that threshold; other rules remain unchecked.",
            actors=["truck", "emergency_vehicle", "priority_vehicle"],
            effect=meaning_effect("P.126's stated >3.500 t threshold is not met; this is not overall permission to overtake."),
        ),
    ]

    for item in rules:
        code = item["code"]
        if code not in entries or not entries[code].get("legal_basis"):
            raise ValueError(f"Missing official catalog provenance for {code}")
        item["additional_legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": "Phụ lục B.25 (PDF page 88)" if code == "P.125" else "Phụ lục B.26 (PDF page 88)",
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
