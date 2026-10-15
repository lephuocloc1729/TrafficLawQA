"""Add a conservative, review-gated R.309 horn-requirement draft."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not any(item.get("official_code") == "R.309" for item in catalog.get("items", [])):
        raise ValueError("Missing R.309 in the official source catalog")

    rule = {
        "rule_id": "R309_HORN_REQUIRED_AT_SIGN_POSITION",
        "code": "R.309",
        "actions": ["using_horn"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            {
                "predicate": "sign_effective_scope_confirmed",
                "operator": "equals",
                "expected_value": True,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "CONDITIONAL",
            },
            {
                "predicate": "vehicle_at_r309_sign_position",
                "operator": "equals",
                "expected_value": True,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "CONDITIONAL",
            },
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix D, D.9, Figure D.11: vehicles arriving at the position "
            "where R.309 is placed must sound the horn. The QCVN says the sign is placed before a "
            "sharp bend or before a mountain-pass road with a longitudinal grade greater than 10% "
            "where visibility is limited. Seeing the sign alone does not establish that the vehicle "
            "is at that position or that its scope is effective; this draft never returns ALLOWED."
        ),
        "effects": [{
            "effect_type": "require_action",
            "action": "using_horn",
            "scope": {"application_scope_fact": "vehicle_at_r309_sign_position"},
            "description": "Sound the horn at the R.309 sign position.",
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": "Phụ lục D, D.9, Hình D.11 (PDF page 119)",
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
            "gazette_source_url": GAZETTE_URL,
        }],
    }
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rules = {item.get("rule_id"): item for item in payload.get("rules", [])}
    rules[rule["rule_id"]] = rule
    payload["rules"] = list(rules.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
