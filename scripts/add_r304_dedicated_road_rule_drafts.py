"""Add official-source-backed, review-gated R.304 rule drafts."""

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


def draft(
    rule_id: str,
    actor_scope: list[str],
    outcome: str,
    reason: str,
    *,
    conditions: list[dict] | None = None,
    effects: list[dict] | None = None,
    actions: list[str] | None = None,
) -> dict:
    return {
        "rule_id": rule_id,
        "code": "R.304",
        "actions": actions or ["entering", "circulation"],
        "actor_scope": actor_scope,
        "outcome": outcome,
        "conditions": conditions if conditions is not None else [SCOPE_CONDITION],
        "reason": reason,
        "effects": effects or [],
        "additional_legal_basis": [LAW_BASIS],
    }


def make_drafts() -> list[dict]:
    scope = "sign_effective_scope_confirmed"
    qcvn = "QCVN 41:2024/BGTVT, Phụ lục D.4(b)"
    return [
        draft(
            "R304_PERMIT_PEDESTRIAN_USE",
            ["pedestrian"], "ALLOWED",
            f"{qcvn} quy định người đi bộ phải dùng đường dành riêng có biển R.304; "
            "chỉ kết luận trong phạm vi biển đã được xác nhận.",
            effects=[
                {"effect_type": "permit_action", "action": action,
                 "scope": {"application_scope_fact": scope}}
                for action in ("entering", "circulation")
            ],
        ),
        draft(
            "R304_PERMIT_NON_MOTOR_VEHICLE_USE",
            ["non_motor_vehicle"], "ALLOWED",
            f"{qcvn} quy định xe thô sơ phải dùng đường dành riêng có biển R.304. "
            "Nhóm xe thô sơ được phân loại theo Điều 34(2) Luật 36/2024/QH15; "
            "phương tiện không phân loại được vẫn phải để UNKNOWN.",
            effects=[
                {"effect_type": "permit_action", "action": action,
                 "scope": {"application_scope_fact": scope}}
                for action in ("entering", "circulation")
            ],
        ),
        draft(
            "R304_PROHIBIT_MOTORIZED_AND_PRIORITY_VEHICLE_ENTRY",
            ["motor_vehicle", "priority_vehicle"], "PROHIBITED",
            f"{qcvn} cấm xe cơ giới (kể cả xe gắn máy) và xe ưu tiên đi vào đường "
            "đặt biển R.304. Ngoại lệ đi cắt ngang được xử lý riêng, không mở rộng "
            "thành quyền lưu thông.",
            effects=[{"effect_type": "prohibit_action", "action": "entering"},
                     {"effect_type": "prohibit_action", "action": "circulation"}],
        ),
        draft(
            "R304_CROSSING_REQUIRES_ABSOLUTE_SAFETY_REVIEW",
            ["motor_vehicle", "priority_vehicle"], "CONDITIONAL",
            f"{qcvn} nêu ngoại lệ đi cắt ngang nhưng yêu cầu đảm bảo tuyệt đối an toàn "
            "cho xe thô sơ và người đi bộ. Engine không tự kết luận điều kiện an toàn "
            "này từ ảnh; cần đánh giá/tái thẩm định riêng.",
            conditions=[
                SCOPE_CONDITION,
                {"predicate": "crossing_safety_confirmed", "operator": "equals",
                 "expected_value": True, "required": True, "on_missing": "CONDITIONAL",
                 "on_mismatch": "CONDITIONAL"},
            ],
            effects=[{"effect_type": "sign_meaning", "action": "crossing",
                      "description": "Safety assessment is required; this is not permission."}],
            actions=["crossing"],
        ),
    ]


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    additions = make_drafts()
    replacements = {item["rule_id"]: item for item in additions}
    payload["rules"] = [
        replacements.pop(item.get("rule_id"), item)
        for item in payload.get("rules", [])
    ]
    payload["rules"].extend(replacements.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
