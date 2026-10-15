"""Add conservative, official-source-backed drafts for QCVN speed-limit signs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
SCOPE = {
    "predicate": "sign_effective_scope_confirmed",
    "operator": "equals",
    "expected_value": True,
    "required": True,
    "on_missing": "CONDITIONAL",
    "on_mismatch": "NOT_APPLICABLE",
}


def condition(predicate: str, operator: str = "exists", expected: object = None) -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def make_rule(rule_id: str, code: str, reason: str, conditions: list[dict],
              value_predicate: str, actors: list[str] | None = None) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["speed"],
        "actor_scope": actors or ["motor_vehicle"],
        "excluded_actor_scope": ["priority_vehicle"],
        "outcome": "CONDITIONAL",
        "conditions": conditions,
        "reason": reason,
        "effects": [{
            "effect_type": "maximum",
            "action": "speed",
            "unit": "km/h",
            "value_predicate": value_predicate,
        }],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_code = {item.get("official_code"): item for item in catalog.get("items", [])}
    references = {
        code: {
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": section,
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
        }
        for code, section in {
            "P.127": "Phụ lục B.27 (PDF pages 89–90)",
            "P.127a": "Phụ lục B.27a (PDF page 91)",
            "P.127b": "Phụ lục B.27b (PDF page 91)",
            "P.127c": "Phụ lục B.27c (PDF page 91)",
        }.items()
    }

    rules = [
        make_rule(
            "P127_SPEED_LIMIT", "P.127",
            "B.27(b,c): phương tiện cơ giới không được vượt trị số tốc độ ghi trên biển; trị số trên biển tính bằng km/h. Điều kiện mặt đường, thời tiết và giao thông vẫn cần người lái tự điều chỉnh an toàn; rule này chỉ trả giới hạn tối đa ghi trên biển.",
            [SCOPE, condition("speed_limit_value")],
            "speed_limit_value",
        ),
        make_rule(
            "P127A_NIGHT_SPEED_LIMIT", "P.127a",
            "B.27a: giới hạn tốc độ ban đêm chỉ có hiệu lực trong thời gian ghi trên biển và từ vị trí đặt biển đến R.421; P.127a được đặt sau R.420. Cần xác nhận liên kết khu vực, đọc trị số và thời gian; thiếu bất kỳ dữ kiện nào thì chưa kết luận.",
            [SCOPE, condition("night_speed_zone_scope_confirmed", "equals", True),
             condition("speed_limit_value"),
             condition("time_window", "time_window_contains_now")],
            "speed_limit_value",
        ),
        make_rule(
            "P127B_CURRENT_LANE_SPEED_LIMIT", "P.127b",
            "B.27b: xe phải tuân thủ tốc độ tối đa ghi cho làn đang chạy. Cần xác nhận làn hiện tại và trị số tương ứng trên biển ghép.",
            [SCOPE, condition("current_lane_scope_confirmed", "equals", True),
             condition("current_lane_speed_limit_value")],
            "current_lane_speed_limit_value",
        ),
        make_rule(
            "P127C_CURRENT_VEHICLE_LANE_SPEED_LIMIT", "P.127c",
            "B.27c: phương tiện phải đi đúng làn theo biểu tượng phương tiện và tuân thủ tốc độ tối đa trên làn đó. Cần xác nhận đồng thời loại phương tiện, làn hiện tại, biểu tượng tương ứng và trị số đọc được.",
            [SCOPE, condition("current_vehicle_lane_scope_confirmed", "equals", True),
             condition("applicable_speed_limit_value")],
            "applicable_speed_limit_value",
        ),
    ]

    for rule in rules:
        code = rule["code"]
        if code not in by_code or not by_code[code].get("legal_basis"):
            raise ValueError(f"Missing official catalog provenance for {code}")
        rule["additional_legal_basis"] = [references[code]]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update({rule["rule_id"]: rule for rule in rules})
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
