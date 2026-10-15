"""Add Article 27 warning-duty drafts for every W-coded catalog entry."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"
ACTIONS = [
    "parking", "stopping", "yielding", "turn_left", "turn_right", "u_turn",
    "going_straight", "entering", "exiting", "speed", "circulation", "crossing",
    "overtaking", "reversing", "lane_change", "using_horn", "following_distance",
    "passing_obstacle", "roundabout_circulation", "unknown",
]


def make_rule(code: str) -> dict:
    return {
        "rule_id": f"{code.replace('.', '').upper()}_COMMON_WARNING_SIGN_DUTY",
        "code": code,
        "intent_kinds": ["QUERY_SIGN", "CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": ACTIONS,
        "actor_scope": ["vehicle", "pedestrian"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [{
            "predicate": "sign_effective_scope_confirmed",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "NOT_APPLICABLE",
        }],
        "reason": (
            "QCVN 41:2024/BGTVT Chapter 5, Article 27 applies to danger and warning signs: they warn "
            "road users in advance of hazards or precautions on the route; when encountering one, road "
            "users must reduce speed to the necessary level, pay attention and observe, and be ready to "
            "handle possible situations to prevent crashes. This general duty does not establish a numeric "
            "speed limit or permission for a particular maneuver. The specific hazard meaning remains the "
            "identified sign's own catalog meaning."
        ),
        "effects": [
            {
                "effect_type": "require_action",
                "action": "reduce_speed_to_necessary_level",
                "scope": {"trigger_sign_code": code},
                "description": "Reduce speed to the level necessary when encountering this danger/warning sign; no numeric speed is inferred.",
            },
            {
                "effect_type": "require_action",
                "action": "pay_attention_and_observe",
                "scope": {"trigger_sign_code": code},
                "description": "Pay attention and observe for the warned hazard or precaution.",
            },
            {
                "effect_type": "require_action",
                "action": "prepare_to_handle_possible_situations",
                "scope": {"trigger_sign_code": code},
                "description": "Be ready to handle possible situations to prevent a crash.",
            },
            {
                "effect_type": "sign_meaning",
                "action": "circulation",
                "scope": {"general_warning_duty": True, "specific_hazard_from_sign_catalog": code},
                "description": "This W-coded sign warns in advance; its specific hazard is supplied by the identified sign meaning.",
            },
        ],
        "additional_legal_basis": [
            {
                "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
                "edition": "2024; effective 2025-01-01",
                "section_or_figure": "Chapter 5, Article 27: common effect of danger and warning signs",
                "source_url": SOURCE_URL,
                "gazette_source_url": GAZETTE_URL,
                "accessed_at": ACCESS_DATE,
                "verification_status": "verified",
            },
            {
                "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
                "edition": "2024; effective 2025-01-01",
                "section_or_figure": "Chapter 5, Article 28.1: W-coded danger and warning sign code inventory",
                "source_url": SOURCE_URL,
                "gazette_source_url": GAZETTE_URL,
                "accessed_at": ACCESS_DATE,
                "verification_status": "verified",
            },
        ],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    warning_codes = sorted({
        str(item.get("official_code"))
        for item in catalog.get("items", [])
        if str(item.get("official_code") or "").startswith("W.")
    })
    if not warning_codes:
        raise ValueError("The official-source catalog contains no W-coded warning signs")
    for code in warning_codes:
        item = next((x for x in catalog["items"] if x.get("official_code") == code), None)
        if not item or not item.get("legal_basis") or not str(item["legal_basis"][0].get("source_url", "")).startswith("https://datafiles.chinhphu.vn/"):
            raise ValueError(f"W-coded sign lacks official-source provenance: {code}")

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    additions = [make_rule(code) for code in warning_codes]
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-35"
    payload["review_status"] = "unreviewed_draft"
    payload["warning_code_inventory"] = {
        "source": "QCVN 41:2024/BGTVT Chapter 5, Article 28.1, reconciled against the official-source catalog",
        "count": len(warning_codes),
        "codes": warning_codes,
        "review_status": "unreviewed_extraction",
    }
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
