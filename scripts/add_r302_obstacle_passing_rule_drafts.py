"""Add review-gated R.302(a/b/c) drafts from the official QCVN text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
VARIANTS = {
    "R.302a": ("right", "chỉ được vòng sang phải"),
    "R.302b": ("left", "chỉ được vòng sang trái"),
    "R.302c": (["left", "right"], "được vòng sang trái hoặc sang phải"),
}


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {item.get("official_code") for item in catalog.get("items", [])}
    missing = set(VARIANTS) - entries
    if missing:
        raise ValueError(f"Missing official catalog entries: {', '.join(sorted(missing))}")

    basis = {
        "document": "QCVN 41:2024/BGTVT",
        "edition": "2024",
        "section_or_figure": "Phụ lục D, D.2(a–c), Hình D.4 (PDF pages 116–117)",
        "source_url": SOURCE_URL,
        "accessed_at": "2026-10-05",
        "verification_status": "verified",
    }
    rules = []
    for code, (allowed_sides, meaning) in VARIANTS.items():
        rules.append({
            "rule_id": f"{code.replace('.', '').upper()}_OBSTACLE_PASSING_DIRECTION",
            "code": code,
            "intent_kinds": ["CHECK_PERMISSION"],
            "actions": ["passing_obstacle"],
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
                    "predicate": "intended_obstacle_passing_side",
                    "operator": "in" if isinstance(allowed_sides, list) else "equals",
                    "expected_value": allowed_sides,
                    "required": True,
                    "on_missing": "CONDITIONAL",
                    "on_mismatch": "CONDITIONAL",
                },
            ],
            "reason": (
                f"QCVN 41:2024/BGTVT Phụ lục D.2(a–c), Hình D.4: {code} directs "
                f"motorized and non-motorized vehicles around an obstacle; this variant means {meaning}. "
                "R.302(a/b) may also be used with P.102 at a median to indicate one-way travel direction. "
                "A route-specific conclusion remains conditional until the sign's effective scope, the "
                "intended side, full sign inventory, and other applicable controls are verified. This rule "
                "never returns ALLOWED."
            ),
            "effects": [{
                "effect_type": "require_action",
                "action": "passing_obstacle",
                "scope": {"required_side": allowed_sides},
                "description": f"Follow the R.302 arrow direction: {meaning}.",
            }],
            "additional_legal_basis": [dict(basis)],
        })

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update({item["rule_id"]: item for item in rules})
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
