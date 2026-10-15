"""Add source-grounded P.130/P.131 rules with explicit S.508 fail-closed scope."""

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


def condition(predicate: str, expected: object, operator: str = "equals") -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def prohibition(rule_id: str, code: str, actions: list[str], reason: str,
                conditions: list[dict], *, actor_scope: list[str] | None = None,
                excluded: list[str] | None = None) -> dict:
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": actions,
        "actor_scope": actor_scope or ["motor_vehicle"],
        "excluded_actor_scope": excluded or ["priority_vehicle"],
        "outcome": "PROHIBITED",
        "conditions": conditions,
        "reason": reason,
        "effects": [{
            "effect_type": "prohibit_action",
            "action": action,
            "scope": {"application_scope_fact": "sign_effective_scope_confirmed"},
        } for action in actions],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    by_code = {item.get("official_code"): item for item in catalog.get("items", [])}

    shared_scope = [SCOPE, condition("linked_s508_panel_present", False)]
    timed_scope = [
        SCOPE,
        condition("linked_s508_panel_present", True),
        condition("linked_s508_temporal_scope_active", True),
    ]
    p130_base = (
        "B.30(a,b): cấm xe cơ giới dừng và đỗ phía đường đặt biển, trừ xe ưu tiên; "
        "phạm vi bắt đầu tại biển và kết thúc tại nút giao/vị trí dừng đỗ quy định hoặc DP.135. "
        "Nếu có S.508 thì chỉ áp dụng trong điều kiện thời gian/ngày đã đọc và đối chiếu."
    )
    p131_base = (
        "B.31(a,b): cấm xe cơ giới đỗ phía đường đặt biển, trừ xe ưu tiên; "
        "P.131b áp dụng ngày lẻ, P.131c ngày chẵn. Quy định về phạm vi và thời gian tại B.30 "
        "cũng áp dụng; S.508 chưa đọc/đối chiếu thì không kết luận dứt khoát."
    )
    rules = [
        prohibition(
            "P130_NO_STOPPING_NO_PARKING", "P.130", ["parking", "stopping"], p130_base,
            shared_scope, excluded=["priority_vehicle", "bus"],
        ),
        prohibition(
            "P130_NO_STOPPING_NO_PARKING_DURING_S508", "P.130",
            ["parking", "stopping"], p130_base, timed_scope,
            excluded=["priority_vehicle", "bus"],
        ),
        prohibition(
            "P131A_NO_PARKING", "P.131a", ["parking"], p131_base,
            shared_scope,
        ),
        prohibition(
            "P131A_NO_PARKING_DURING_S508", "P.131a", ["parking"], p131_base,
            timed_scope,
        ),
        prohibition(
            "P131B_ODD_DAY_NO_PARKING", "P.131b", ["parking"], p131_base,
            [*shared_scope, condition("calendar_parity", "odd_days")],
        ),
        prohibition(
            "P131B_ODD_DAY_NO_PARKING_DURING_S508", "P.131b", ["parking"], p131_base,
            [*timed_scope, condition("calendar_parity", "odd_days")],
        ),
        prohibition(
            "P131C_EVEN_DAY_NO_PARKING", "P.131c", ["parking"], p131_base,
            [*shared_scope, condition("calendar_parity", "even_days")],
        ),
        prohibition(
            "P131C_EVEN_DAY_NO_PARKING_DURING_S508", "P.131c", ["parking"], p131_base,
            [*timed_scope, condition("calendar_parity", "even_days")],
        ),
        {
            "rule_id": "P130_BUS_STOPPING_OR_PARKING_REQUIRES_CONTEXT",
            "code": "P.130",
            "intent_kinds": ["CHECK_PERMISSION"],
            "actions": ["parking", "stopping"],
            "actor_scope": ["bus"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                SCOPE,
                condition("scheduled_bus_route_status", None, "exists"),
                condition("designated_bus_stopping_position_confirmed", True),
            ],
            "reason": (
                "B.30(a) nói xe buýt chạy theo hành trình quy định được hướng dẫn vị trí dừng thích hợp; "
                "biển này không tự xác nhận xe buýt đang chạy đúng hành trình hay đang ở vị trí dừng được chỉ định. "
                "Cần xác minh các dữ kiện đó; rule không kết luận ALLOWED."
            ),
            "effects": [],
        },
    ]

    sections = {
        "P.130": "Phụ lục B.30 (PDF pages 93–94)",
        "P.131a": "Phụ lục B.31 (PDF page 94)",
        "P.131b": "Phụ lục B.31 (PDF page 94)",
        "P.131c": "Phụ lục B.31 (PDF page 94)",
    }
    for rule in rules:
        code = rule["code"]
        entry = by_code.get(code)
        if not entry or not entry.get("legal_basis"):
            raise ValueError(f"Missing official catalog provenance for {code}")
        rule["additional_legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": sections[code],
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
        }]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update({rule["rule_id"]: rule for rule in rules})
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
