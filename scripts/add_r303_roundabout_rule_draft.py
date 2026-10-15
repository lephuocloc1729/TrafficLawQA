"""Add a review-gated R.303 roundabout-direction draft from the official QCVN."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if not any(item.get("official_code") == "R.303" for item in catalog.get("items", [])):
        raise ValueError("Missing R.303 in the official source catalog")

    rule = {
        "rule_id": "R303_MANDATORY_ROUNDABOUT_DIRECTION",
        "code": "R.303",
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": ["roundabout_circulation", "circulation", "turn_left", "turn_right", "u_turn"],
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
                "predicate": "intended_roundabout_route_follows_arrow",
                "operator": "equals",
                "expected_value": True,
                "required": True,
                "on_missing": "CONDITIONAL",
                "on_mismatch": "CONDITIONAL",
            },
        ],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix D, D.3(a–b), Figure D.5: R.303 marks an "
            "intersection where motorized and non-motorized vehicles must travel around "
            "the safety island; vehicles intending to change direction must circulate around "
            "that island in the direction of the arrow. This sign does not by itself establish "
            "that any particular destination movement is available or permitted. Confirm the "
            "sign's effective scope, the proposed route relative to the arrow, and all other "
            "applicable signs and controls. This draft never returns ALLOWED."
        ),
        "effects": [{
            "effect_type": "require_action",
            "action": "roundabout_circulation",
            "scope": {"route_direction_fact": "intended_roundabout_route_follows_arrow"},
            "description": "If changing direction at the marked intersection, circulate around the safety island in the arrow direction.",
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": "Phụ lục D, D.3(a–b), Hình D.5 (PDF page 117)",
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
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
