"""Add conservative, review-gated R.305 drafts from official QCVN text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"


def condition(predicate: str, expected: object = True) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "CONDITIONAL",
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not any(item.get("official_code") == "R.305" for item in catalog.get("items", [])):
        raise ValueError("Missing R.305 in the official source catalog")

    basis = {
        "document": "QCVN 41:2024/BGTVT",
        "edition": "2024",
        "section_or_figure": "Phụ lục D, D.5(a–b), Hình D.7 (PDF page 117)",
        "source_url": SOURCE_URL,
        "accessed_at": "2026-10-05",
        "verification_status": "verified",
    }
    common = {
        "code": "R.305",
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION"],
        "excluded_actor_scope": [],
        "additional_legal_basis": [dict(basis)],
    }
    rules = [
        {
            **common,
            "rule_id": "R305_PROHIBIT_NONEXEMPT_VEHICLE_ENTRY",
            "actions": ["entering", "circulation"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": ["bicycle", "mobility_aid"],
            "outcome": "PROHIBITED",
            "conditions": [condition("sign_effective_scope_confirmed")],
            "reason": (
                "QCVN 41:2024/BGTVT D.5(b): motor and non-motor vehicles, including priority vehicles, "
                "except bicycles and wheelchairs for persons with disabilities, may not enter a road "
                "marked R.305. Crossing through is a separate exception and is not treated as ordinary "
                "entry or circulation. Confirm sign scope and the actor's legal class."
            ),
            "effects": [{"effect_type": "prohibit_action", "action": "entering"}],
        },
        {
            **common,
            "rule_id": "R305_CROSSING_REQUIRES_ABSOLUTE_PEDESTRIAN_SAFETY",
            "actions": ["crossing"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": ["bicycle", "mobility_aid"],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed"),
                condition("absolute_safety_for_pedestrians_confirmed"),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT D.5(b) allows crossing through the pedestrian road only when "
                "absolute safety for pedestrians is ensured. The engine requires an explicit safety "
                "assessment; even when supplied, this sign alone does not establish overall permission."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "crossing",
                "description": "Crossing is an exception only when absolute safety for pedestrians is ensured.",
            }],
        },
        {
            **common,
            "rule_id": "R305_BICYCLE_EXCEPTION_NOT_BLANKET_PERMISSION",
            "actions": ["entering", "circulation", "crossing"],
            "actor_scope": ["bicycle"],
            "outcome": "CONDITIONAL",
            "conditions": [condition("sign_effective_scope_confirmed")],
            "reason": (
                "QCVN 41:2024/BGTVT D.5(b) excludes bicycles from this sign's vehicle prohibition. "
                "This exception is not a blanket permission; other signs, controls, and bicycle rules "
                "remain unchecked."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "circulation",
                "description": "Bicycles are excluded from the R.305 vehicle prohibition; overall permission is unresolved.",
            }],
        },
        {
            **common,
            "rule_id": "R305_WHEELCHAIR_EXCEPTION_REQUIRES_CLASSIFICATION",
            "actions": ["entering", "circulation", "crossing"],
            "actor_scope": ["mobility_aid"],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed"),
                condition("wheelchair_for_person_with_disability_confirmed"),
            ],
            "reason": (
                "QCVN 41:2024/BGTVT D.5(b) excludes wheelchairs for persons with disabilities from "
                "the vehicle prohibition. The observed mobility aid must be classified before applying "
                "that exception; the exception is not blanket permission under other controls."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "circulation",
                "description": "A confirmed wheelchair for a person with a disability is excluded from the R.305 vehicle prohibition.",
            }],
        },
    ]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update({item["rule_id"]: item for item in rules})
    for item in rules:
        item["additional_legal_basis"] = [{**item["additional_legal_basis"][0], "gazette_source_url": GAZETTE_URL}]
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
