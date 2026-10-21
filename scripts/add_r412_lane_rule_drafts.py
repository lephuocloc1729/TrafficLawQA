"""Add source-specific meanings and review-gated lane-use rules for R.412a-h."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
LAW_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15_tiep.pdf"
ACCESS_DATE = "2026-10-05"
LAW_BASIS = {
    "document": "Luật Trật tự, an toàn giao thông đường bộ số 36/2024/QH15",
    "edition": "2024; effective 2025-01-01",
    "section_or_figure": "Điều 34 khoản 1, khoản 2 (phân loại xe cơ giới và xe thô sơ)",
    "source_url": LAW_URL,
    "accessed_at": ACCESS_DATE,
    "verification_status": "verified",
}
SCOPE = {
    "predicate": "sign_effective_scope_confirmed", "operator": "equals",
    "expected_value": True, "required": True,
    "on_missing": "CONDITIONAL", "on_mismatch": "NOT_APPLICABLE",
}

ALLOWED_ACTORS = {
    # QCVN D.14 distinguishes passenger coaches (including buses) on R.412a
    # from passenger cars on R.412b.
    "R.412a": ["bus"],
    "R.412b": ["car"],
    "R.412c": ["truck"],
    "R.412d": ["motorbike", "motorcycle", "moped"],
    "R.412e": ["bus"],
    "R.412f": ["car", "truck", "bus", "taxi"],
    "R.412g": ["motorbike", "motorcycle", "moped", "non_motor_vehicle"],
    "R.412h": ["non_motor_vehicle"],
}
VARIANT_MEANINGS = {
    "R.412a": "Báo hiệu làn đường dành riêng cho ô tô khách, kể cả ô tô buýt. Khi cần phân làn theo số chỗ, số chỗ ngồi cho phép có thể được ghi trên hình xe; biển R.412a có thể được bổ sung chữ BRT cho làn xe buýt nhanh.",
    "R.412b": "Báo hiệu làn đường dành cho xe ô tô con.",
    "R.412c": "Báo hiệu làn đường dành cho xe ô tô tải. Khi cần phân loại theo khối lượng chuyên chở cho phép, trị số này có thể được ghi trên hình xe tải.",
    "R.412d": "Báo hiệu làn đường dành cho xe máy và xe gắn máy.",
    "R.412e": "Báo hiệu làn đường dành cho xe buýt. Xe khác không được đi vào; riêng khi vạch phân làn là nét đứt, xe khác có thể đi vào nhưng phải nhường đường và chuyển làn để không ảnh hưởng hoạt động của xe buýt. Vạch nét liền thì xe khác không được đi vào.",
    "R.412f": "Báo hiệu làn đường dành cho các loại xe ô tô.",
    "R.412g": "Báo hiệu làn đường dành riêng cho xe máy, kể cả xe gắn máy, và xe đạp, kể cả các loại xe thô sơ khác.",
    "R.412h": "Báo hiệu làn đường dành riêng cho xe đạp, kể cả các loại xe thô sơ khác.",
}
SCOPE_BY_CODE = {
    "R.412a": "passenger_vehicle",
    "R.412b": "passenger_car",
    "R.412c": "truck",
    "R.412d": "motorcycle_or_moped",
    "R.412e": "bus",
    "R.412f": "automobile",
    "R.412g": "motorcycle_moped_or_non_motor_vehicle",
    "R.412h": "non_motor_vehicle",
}


def condition(predicate: str, operator: str, value=None, *, mismatch="NOT_APPLICABLE") -> dict:
    result = {
        "predicate": predicate, "operator": operator, "required": True,
        "on_missing": "CONDITIONAL", "on_mismatch": mismatch,
    }
    if value is not None:
        result["expected_value"] = value
    return result


def effect(effect_type: str, action: str, scope: dict | None = None) -> dict:
    result = {"effect_type": effect_type, "action": action}
    if scope:
        result["scope"] = scope
    return result


def make_rule(rule_id: str, code: str, actors: list[str], outcome: str, reason: str,
              conditions: list[dict], effects: list[dict], *, excluded: list[str] | None = None,
              actions: list[str] | None = None) -> dict:
    basis_section = "Phụ lục D.14 (PDF pages 123,124,125)"
    return {
        "rule_id": rule_id,
        "code": code,
        "actions": actions or ["entering", "circulation"],
        "actor_scope": actors,
        "excluded_actor_scope": excluded or [],
        "outcome": outcome,
        "conditions": conditions,
        "reason": reason,
        "effects": effects,
        "additional_legal_basis": [
            {
                "document": "QCVN 41:2024/BGTVT",
                "edition": "2024",
                "section_or_figure": basis_section,
                "source_url": SOURCE_URL,
                "accessed_at": ACCESS_DATE,
                "verification_status": "verified",
            },
            LAW_BASIS,
        ],
    }


def lane_effects(code: str, kind: str = "permit_action") -> list[dict]:
    return [effect(kind, action, {"application_scope_fact": "sign_effective_scope_confirmed"})
            for action in ("entering", "circulation")]


def make_rules() -> list[dict]:
    rules: list[dict] = []
    general = (
        "QCVN 41:2024/BGTVT Phụ lục D.14 quy định làn dành riêng cho nhóm phương tiện "
        "được biểu thị trên biển; xe khác không được đi vào, trừ xe ưu tiên."
    )
    for code, actors in ALLOWED_ACTORS.items():
        rules.append(make_rule(
            f"{code.replace('.', '').upper()}_ALLOW_DESIGNATED_VEHICLE",
            code, actors, "ALLOWED",
            f"{general} Phạm vi biển/làn phải được xác nhận; riêng các giới hạn ghi trên biển phải được đọc đủ.",
            [SCOPE], lane_effects(code),
        ))
        # The QCVN's optional printed restrictions must not be silently ignored.
        if code == "R.412a":
            rules[-1]["conditions"].append(condition(
                "passenger_seat_limit_present", "equals", False,
                mismatch="NOT_APPLICABLE",
            ))
            rules.extend([
                make_rule(
                    "R412A_ALLOW_BELOW_DISPLAYED_SEAT_LIMIT", code, actors, "ALLOWED",
                    "D.14(a) cho phép ghi số chỗ ngồi cho phép trên biển. Cần xác nhận giới hạn và số chỗ theo hồ sơ xe; chỉ kết luận trong phạm vi giới hạn đã đọc.",
                    [SCOPE, condition("passenger_seat_limit_present", "equals", True),
                     condition("passenger_seat_limit_relation", "equals", "lt"),
                     condition("passenger_seat_limit", "exists"),
                     condition("vehicle_seat_count", "lt_fact", "passenger_seat_limit")],
                    lane_effects(code),
                ),
                make_rule(
                    "R412A_PROHIBIT_AT_OR_ABOVE_DISPLAYED_SEAT_LIMIT", code, actors, "PROHIBITED",
                    "D.14(a) cho phép ghi số chỗ ngồi cho phép trên biển. Số chỗ xe đạt/vượt giới hạn đã đọc thì không thuộc phạm vi làn này.",
                    [SCOPE, condition("passenger_seat_limit_present", "equals", True),
                     condition("passenger_seat_limit_relation", "equals", "lt"),
                     condition("passenger_seat_limit", "exists"),
                     condition("vehicle_seat_count", "gte_fact", "passenger_seat_limit")],
                    [effect("prohibit_action", action) for action in ("entering", "circulation")],
                ),
            ])
        if code == "R.412c":
            rules[-1]["conditions"].append(condition(
                "payload_limit_present", "equals", False, mismatch="NOT_APPLICABLE"
            ))
            rules.extend([
                make_rule(
                    "R412C_ALLOW_BELOW_DISPLAYED_PAYLOAD_LIMIT", code, actors, "ALLOWED",
                    "D.14(c) cho phép ghi trị số khối lượng chuyên chở cho phép trên biển; cần trị số xe theo giấy chứng nhận và trị số giới hạn đọc được.",
                    [SCOPE, condition("payload_limit_present", "equals", True),
                     condition("payload_limit_relation", "equals", "lt"),
                     condition("payload_limit_t", "exists"),
                     condition("vehicle_payload_t", "lt_fact", "payload_limit_t")],
                    lane_effects(code),
                ),
                make_rule(
                    "R412C_PROHIBIT_AT_OR_ABOVE_DISPLAYED_PAYLOAD_LIMIT", code, actors, "PROHIBITED",
                    "D.14(c) cho phép ghi trị số khối lượng chuyên chở cho phép trên biển; trị số xe đạt/vượt giới hạn đã đọc thì không thuộc phạm vi làn này.",
                    [SCOPE, condition("payload_limit_present", "equals", True),
                     condition("payload_limit_relation", "equals", "lt"),
                     condition("payload_limit_t", "exists"),
                     condition("vehicle_payload_t", "gte_fact", "payload_limit_t")],
                    [effect("prohibit_action", action) for action in ("entering", "circulation")],
                ),
            ])

        if code == "R.412e":
            rules.extend([
                make_rule(
                    "R412E_PROHIBIT_OTHER_VEHICLE_ON_SOLID_BUS_LANE", code,
                    ["vehicle"], "PROHIBITED",
                    "D.14(a) quy định xe khác không được đi vào làn xe buýt khi vạch phân làn là nét liền.",
                    [SCOPE, condition("bus_lane_marking", "equals", "solid")],
                    [effect("prohibit_action", action) for action in ("entering", "circulation")],
                    excluded=["bus", "priority_vehicle"],
                ),
                make_rule(
                    "R412E_ALLOW_OTHER_VEHICLE_ON_BROKEN_BUS_LANE_WITH_YIELD", code,
                    ["vehicle"], "ALLOWED",
                    "D.14(a) cho phép xe khác đi vào làn xe buýt khi vạch nét đứt nhưng bắt buộc nhường đường và chuyển làn để không ảnh hưởng hoạt động của xe buýt.",
                    [SCOPE, condition("bus_lane_marking", "equals", "broken")],
                    [effect("permit_action", "entering"), effect("permit_action", "circulation"),
                     effect("require_action", "yield_to_bus")],
                    excluded=["bus", "priority_vehicle"],
                ),
            ])

        # For all other dedicated-lane variants, QCVN expressly bars vehicles
        # outside the pictured class, except priority vehicles. Priority status
        # is not itself an allow result; it only removes this specific ban.
        if code != "R.412e":
            rules.append(make_rule(
                f"{code.replace('.', '').upper()}_PROHIBIT_OTHER_VEHICLES",
                code, ["vehicle"], "PROHIBITED", general,
                [SCOPE],
                [effect("prohibit_action", action) for action in ("entering", "circulation")],
                excluded=[*actors, "priority_vehicle"],
            ))
    return rules


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    items = {item.get("official_code"): item for item in catalog.get("items", [])}
    for code, meaning in VARIANT_MEANINGS.items():
        item = items.get(code)
        if item is None:
            raise ValueError(f"Official catalog entry missing for {code}")
        item["legal_meaning"] = meaning
        item.setdefault("scope_attributes", {})["vehicle_scope"] = SCOPE_BY_CODE[code]
        item["scope_attributes"]["action_scope"] = ["entering", "circulation"]
        item["scope_attributes"]["exceptions"] = ["priority vehicles"]
    CATALOG_PATH.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    additions = make_rules()
    replacements = {item["rule_id"]: item for item in additions}
    registry["rules"] = [
        replacements.pop(item.get("rule_id"), item)
        for item in registry.get("rules", [])
    ]
    registry["rules"].extend(replacements.values())
    registry["registry_version"] = "qcvn41-rule-drafts-19"
    registry["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
