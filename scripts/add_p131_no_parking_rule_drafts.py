"""Add source-first P.131a/b/c parking-prohibition drafts from official QCVN text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"
VARIANTS = {
    "P.131a": (None, "cấm xe cơ giới đỗ phía đường có đặt biển"),
    "P.131b": ("odd_days", "cấm xe cơ giới đỗ phía đường có đặt biển vào ngày lẻ"),
    "P.131c": ("even_days", "cấm xe cơ giới đỗ phía đường có đặt biển vào ngày chẵn"),
}


def condition(predicate: str, expected: object) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def basis(section: str) -> dict:
    return {
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": section,
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }


def make_rule(code: str, with_s508: bool) -> dict:
    parity, meaning = VARIANTS[code]
    conditions = [condition("sign_effective_scope_confirmed", True)]
    if parity:
        conditions.append(condition("calendar_parity", parity))
    conditions.append(condition("linked_s508_panel_present", with_s508))
    if with_s508:
        conditions.append(condition("linked_s508_temporal_scope_active", True))
    rule_id_base = {
        "P.131a": "P131A_NO_PARKING",
        "P.131b": "P131B_ODD_DAY_NO_PARKING",
        "P.131c": "P131C_EVEN_DAY_NO_PARKING",
    }[code]
    rule_id = rule_id_base + ("_DURING_S508" if with_s508 else "")
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["parking"],
        "actor_scope": ["motor_vehicle"],
        "excluded_actor_scope": ["priority_vehicle"],
        "outcome": "PROHIBITED",
        "conditions": conditions,
        "reason": (
            f"QCVN 41:2024/BGTVT Appendix B.31(a-c), {code}: {meaning}. "
            "The prohibition applies on the sign's side of the road, except to priority vehicles. "
            "Its extent and any S.508 time condition must be resolved; this draft does not infer "
            "permission from an inactive/unreadable panel or from absence of another prohibition."
        ),
        "effects": [{
            "effect_type": "prohibit_action",
            "action": "parking",
            "scope": {
                "application_scope_fact": "sign_effective_scope_confirmed",
                "sign_variant": code,
                "supplementary_time_scope_fact": "linked_s508_temporal_scope_active" if with_s508 else None,
            },
            "description": f"{code} prohibits parking within its confirmed effect and time scope, subject to the stated priority-vehicle exception.",
        }],
        "additional_legal_basis": [
            basis(f"Appendix B.31(a-c), Figure B.31: {code} no-parking meaning and odd/even-day variants"),
            basis("Appendix B.30(b-c), incorporated by B.31(b): sign extent and S.508 time condition"),
            basis("Chapter 4, Article 26.8: P.131(a,b,c) effect to next intersection/end sign, also subject to supplementary signs"),
        ],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if "P.131" not in {item.get("official_code") for item in catalog.get("items", [])}:
        raise ValueError("Official-source sign catalog is missing P.131")

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    additions = [make_rule(code, with_s508) for code in VARIANTS for with_s508 in (False, True)]
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for duplicate_id in (
        "P131A_NO_PARKING_WITHOUT_S508",
        "P131B_NO_PARKING_WITHOUT_S508",
        "P131B_NO_PARKING_DURING_S508",
        "P131C_NO_PARKING_WITHOUT_S508",
        "P131C_NO_PARKING_DURING_S508",
    ):
        by_id.pop(duplicate_id, None)
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-29"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
