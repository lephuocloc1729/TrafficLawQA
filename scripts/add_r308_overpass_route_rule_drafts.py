"""Add variant-specific, conservative R.308 route rules from official Figure D.10."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
QCVN_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"

VARIANTS = {
    # The official Figure D.10 was visually checked in the official QCVN PDF,
    # page 119: R.308a illustrates a left branch; R.308b a right branch.
    "R.308a": {"shown_turn": "turn_left", "other_turn": "turn_right", "word": "left"},
    "R.308b": {"shown_turn": "turn_right", "other_turn": "turn_left", "word": "right"},
}


def condition() -> dict:
    return {
        "predicate": "sign_effective_scope_confirmed",
        "operator": "equals",
        "expected_value": True,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def make_rule(code: str, *, other_turn: bool) -> dict:
    variant = VARIANTS[code]
    shown_turn = variant["shown_turn"]
    actions = [variant["other_turn"]] if other_turn else ["going_straight", shown_turn]
    if other_turn:
        rule_id = f"{code.replace('.', '').upper()}_OPPOSITE_TURN_NOT_INDICATED"
        reason = (
            f"QCVN 41:2024/BGTVT Appendix D.8 and Figure D.10: {code} shows a straight route and a "
            f"{variant['word']}-turn branch. The requested opposite turn is not depicted. D.8 does not "
            "expressly classify that unpictured movement as prohibited here, so this rule returns "
            "CONDITIONAL and does not infer permission or prohibition."
        )
        description = (
            f"The requested turn is not shown on {code}; this sign alone does not establish whether it is permitted."
        )
    else:
        rule_id = f"{code.replace('.', '').upper()}_STRAIGHT_OR_SHOWN_TURN"
        reason = (
            f"QCVN 41:2024/BGTVT Appendix D.8 and Figure D.10: {code} indicates the route can continue "
            f"straight or follow the illustrated branch to turn {variant['word']}. This is sign meaning, "
            "not blanket permission; the route, other signs, and traffic controls still require assessment."
        )
        description = f"{code} illustrates straight travel or a {variant['word']}-turn branch at the overpass."
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": actions,
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition()],
        "reason": reason,
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "circulation",
            "scope": {"illustrated_turn": shown_turn, "straight_route_indicated": True},
            "description": description,
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix D, D.8, Figure D.10 (official QCVN PDF page 119; Gazette page 38)",
            "source_url": QCVN_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
            "gazette_source_url": GAZETTE_URL,
        }],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    codes = {item.get("official_code") for item in catalog.get("items", [])}
    if not set(VARIANTS).issubset(codes):
        raise ValueError("Official-source catalog is missing an R.308 variant")
    additions = [make_rule(code, other_turn=other_turn)
                 for code in VARIANTS for other_turn in (False, True)]
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-24"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
