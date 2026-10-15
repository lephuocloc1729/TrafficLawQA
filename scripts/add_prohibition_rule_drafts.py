"""Add conservative, source-backed drafts for common prohibition signs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
LAW_BASIS = {
    "document": "Luật Trật tự, an toàn giao thông đường bộ số 36/2024/QH15",
    "edition": "2024; effective 2025-01-01",
    "section_or_figure": "Điều 34 khoản 1, khoản 2 (phân loại xe cơ giới và xe thô sơ)",
    "source_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15_tiep.pdf",
    "accessed_at": "2026-10-05",
    "verification_status": "verified",
}
SCOPE_CONDITION = {
    "predicate": "sign_effective_scope_confirmed",
    "operator": "equals",
    "expected_value": True,
    "required": True,
    "on_missing": "CONDITIONAL",
    "on_mismatch": "NOT_APPLICABLE",
}
PRIORITY_EXCEPTIONS = ["priority_vehicle"]
MOTORCYCLE_AND_MOPED_EXCEPTIONS = ["priority_vehicle", "motorbike", "motorcycle", "moped"]


def _rule(
    rule_id: str,
    code: str,
    actors: list[str],
    actions: list[str],
    reason: str,
    *,
    excluded: list[str] | None = None,
    conditions: list[dict] | None = None,
) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": actions,
        "actor_scope": actors,
        "excluded_actor_scope": excluded or [],
        "outcome": "PROHIBITED",
        "conditions": conditions if conditions is not None else [SCOPE_CONDITION],
        "reason": reason,
        "effects": [{"effect_type": "prohibit_action", "action": action,
                     "scope": {"application_scope_fact": "sign_effective_scope_confirmed"}}
                    for action in actions],
        "additional_legal_basis": [LAW_BASIS],
    }


def make_drafts() -> list[dict]:
    return [
        _rule("P103A_NO_ENTRY_MOTOR_VEHICLES", "P.103a", ["motor_vehicle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.3(a): cấm xe cơ giới đi qua, trừ xe máy hai bánh, xe gắn máy và xe ưu tiên.",
              excluded=MOTORCYCLE_AND_MOPED_EXCEPTIONS),
        _rule("P103B_NO_MOTOR_VEHICLE_RIGHT_TURN", "P.103b", ["motor_vehicle"],
              ["turn_right"],
              "QCVN 41:2024/BGTVT Phụ lục B.3(b): cấm xe cơ giới rẽ phải, trừ xe máy hai bánh, xe gắn máy và xe ưu tiên.",
              excluded=MOTORCYCLE_AND_MOPED_EXCEPTIONS),
        _rule("P103C_NO_MOTOR_VEHICLE_LEFT_TURN", "P.103c", ["motor_vehicle"],
              ["turn_left"],
              "QCVN 41:2024/BGTVT Phụ lục B.3(b): cấm xe cơ giới rẽ trái, trừ xe máy hai bánh, xe gắn máy và xe ưu tiên.",
              excluded=MOTORCYCLE_AND_MOPED_EXCEPTIONS),

        _rule("P106A_NO_TRUCK_TRACTOR_SPECIAL_VEHICLE", "P.106a",
              ["truck", "tractor", "tractor_unit", "special_use_vehicle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.6(a): cấm xe ô tô tải; hiệu lực cấm cũng áp dụng cho máy kéo và xe máy chuyên dùng, trừ xe ưu tiên theo đoạn nguồn.",
              excluded=PRIORITY_EXCEPTIONS),
        {
            "rule_id": "P106B_PAYLOAD_LIMIT", "code": "P.106b",
            "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
            "actions": ["entering", "circulation"], "actor_scope": ["truck"],
            "outcome": "CONDITIONAL",
            "conditions": [
                SCOPE_CONDITION,
                {"predicate": "vehicle_payload_t", "operator": "lte_fact",
                 "expected_value": "payload_limit_t", "required": True,
                 "on_missing": "CONDITIONAL", "on_mismatch": "PROHIBITED"},
                {"predicate": "payload_limit_t", "operator": "exists", "required": True,
                 "on_missing": "CONDITIONAL", "on_mismatch": "CONDITIONAL"},
            ],
            "reason": "QCVN 41:2024/BGTVT Phụ lục B.6(b): cấm xe ô tô tải có khối lượng chuyên chở theo giấy chứng nhận lớn hơn trị số tấn ghi trên biển; xe vượt ngưỡng bị cấm, thiếu một trong hai trị số thì CONDITIONAL.",
            "effects": [{"effect_type": "maximum", "action": "payload_capacity", "unit": "t",
                         "value_predicate": "payload_limit_t"}],
            "additional_legal_basis": [LAW_BASIS],
        },
        _rule("P106B_PROHIBIT_TRACTOR_SPECIAL_USE", "P.106b",
              ["tractor", "tractor_unit", "special_use_vehicle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.6(b): nêu riêng rằng biển cũng cấm máy kéo và xe máy chuyên dùng; trị số tải trọng của xe tải không được áp sang nhóm này.",
              excluded=PRIORITY_EXCEPTIONS),
        _rule("P106C_NO_HAZARDOUS_GOODS_VEHICLES", "P.106c", ["vehicle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.6(c): cấm xe chở hàng nguy hiểm; cần fact xác nhận xe đang chở hàng nguy hiểm.",
              conditions=[SCOPE_CONDITION,
                          {"predicate": "hazardous_cargo_status", "operator": "equals", "expected_value": True,
                           "required": True, "on_missing": "CONDITIONAL", "on_mismatch": "NOT_APPLICABLE"}]),

        _rule("P107_PROHIBIT_PASSENGER_AND_TRUCK_VEHICLES", "P.107",
              ["bus", "truck", "tractor", "tractor_unit", "special_use_vehicle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.7: cấm ô tô chở khách và xe ô tô tải, kể cả máy kéo và xe máy chuyên dùng, trừ xe ưu tiên; actor taxonomy cần nhóm review xác nhận.",
              excluded=PRIORITY_EXCEPTIONS),

        _rule("P110A_NO_BICYCLE", "P.110a", ["bicycle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.10(a): cấm xe đạp đi qua, nhưng không cấm người dắt xe đạp."),
        _rule("P110B_NO_CARGO_BICYCLE", "P.110b", ["cargo_bicycle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.10(b): cấm xe đạp thồ đi qua, nhưng không cấm người dắt loại xe này."),
        _rule("P111A_NO_MOPED", "P.111a", ["moped"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.11(a): cấm xe gắn máy; văn bản nêu biển không có giá trị đối với xe đạp."),
        _rule("P111B_NO_MOTORIZED_TRICYCLE", "P.111b", ["motorized_tricycle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.11(b): cấm xe ba bánh loại có động cơ."),
        _rule("P111C_NO_MOTORIZED_TRICYCLE", "P.111c", ["motorized_tricycle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.11(b): cấm xe ba bánh loại có động cơ."),
        _rule("P111D_NO_NON_MOTORIZED_TRICYCLE", "P.111d", ["non_motorized_tricycle"],
              ["entering", "circulation"],
              "QCVN 41:2024/BGTVT Phụ lục B.11(c): cấm xe ba bánh loại không có động cơ."),

        _rule("P123A_NO_LEFT_TURN", "P.123a", ["vehicle"],
              ["turn_left"],
              "QCVN 41:2024/BGTVT Phụ lục B.23(a,b): P.123a cấm các loại xe cơ giới và thô sơ rẽ trái, trừ xe ưu tiên; không cấm quay đầu.",
              excluded=PRIORITY_EXCEPTIONS),
        _rule("P123B_NO_RIGHT_TURN", "P.123b", ["vehicle"],
              ["turn_right"],
              "QCVN 41:2024/BGTVT Phụ lục B.23(a,b): P.123b cấm các loại xe cơ giới và thô sơ rẽ phải, trừ xe ưu tiên; không cấm quay đầu.",
              excluded=PRIORITY_EXCEPTIONS),

        _rule("P124A_NO_U_TURN", "P.124a", ["vehicle"],
              ["u_turn"],
              "QCVN 41:2024/BGTVT Phụ lục B.24(a,g): cấm các loại xe cơ giới và thô sơ quay đầu; xe ưu tiên được loại trừ theo đoạn nguồn.",
              excluded=PRIORITY_EXCEPTIONS),
        _rule("P124B_NO_U_TURN_VARIANT_B", "P.124b",
              ["car", "motorized_tricycle"], ["u_turn"],
              "QCVN 41:2024/BGTVT Phụ lục B.24(b,g): cấm ô tô và xe mô tô ba bánh có thùng quay đầu; xe ưu tiên được loại trừ theo đoạn nguồn.",
              excluded=PRIORITY_EXCEPTIONS),
        _rule("P124C_NO_LEFT_OR_U_TURN", "P.124c", ["vehicle"],
              ["turn_left", "u_turn"],
              "QCVN 41:2024/BGTVT Phụ lục B.24(c): cấm các loại xe rẽ trái và quay đầu."),
        _rule("P124D_NO_RIGHT_OR_U_TURN", "P.124d", ["vehicle"],
              ["turn_right", "u_turn"],
              "QCVN 41:2024/BGTVT Phụ lục B.24(d): cấm các loại xe rẽ phải và quay đầu."),
        _rule("P124E_NO_CAR_LEFT_OR_U_TURN", "P.124e", ["car"],
              ["turn_left", "u_turn"],
              "QCVN 41:2024/BGTVT Phụ lục B.24(e): cấm ô tô rẽ trái và quay đầu."),
        _rule("P124F_NO_CAR_RIGHT_OR_U_TURN", "P.124f", ["car"],
              ["turn_right", "u_turn"],
              "QCVN 41:2024/BGTVT Phụ lục B.24(f): cấm ô tô rẽ phải và quay đầu."),
    ]


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    additions = make_drafts()
    replacements = {item["rule_id"]: item for item in additions}
    stale_rule_ids = {
        "P103A_PROHIBIT_MOTOR_VEHICLE_ENTRY", "P103B_PROHIBIT_MOTOR_VEHICLE_RIGHT_TURN",
        "P103C_PROHIBIT_MOTOR_VEHICLE_LEFT_TURN", "P106A_PROHIBIT_TRUCK_TRACTOR_SPECIAL_USE",
        "P106B_PROHIBIT_TRUCK_OVER_SIGNED_CAPACITY", "P106C_PROHIBIT_HAZARDOUS_CARGO_VEHICLE",
        "P110A_PROHIBIT_BICYCLE_RIDING", "P123A_PROHIBIT_LEFT_TURN_ALL_VEHICLES",
        "P123B_PROHIBIT_RIGHT_TURN_ALL_VEHICLES", "P124A_PROHIBIT_U_TURN_ALL_VEHICLES",
        "P124B_PROHIBIT_U_TURN_CAR_AND_SIDECAR", "P124C_PROHIBIT_LEFT_AND_U_TURN",
        "P124D_PROHIBIT_RIGHT_AND_U_TURN", "P124E_PROHIBIT_CAR_LEFT_AND_U_TURN",
        "P124F_PROHIBIT_CAR_RIGHT_AND_U_TURN", "P110B_PROHIBIT_CARGO_BICYCLE_RIDING",
        "P111A_PROHIBIT_MOPED", "P111B_PROHIBIT_MOTORIZED_TRICYCLE",
        "P111C_PROHIBIT_MOTORIZED_TRICYCLE", "P111D_PROHIBIT_NON_MOTORIZED_TRICYCLE",
    }
    payload["rules"] = [
        replacements.pop(item.get("rule_id"), item)
        for item in payload.get("rules", [])
        if item.get("rule_id") not in stale_rule_ids
    ]
    payload["rules"].extend(replacements.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
