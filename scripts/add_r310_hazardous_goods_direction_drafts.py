"""Draft variant-specific R.310 direction rules from official QCVN Figure D.12."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
VARIANTS = {
    "R.310a": ("turn_left", "rẽ trái"),
    "R.310b": ("going_straight", "đi thẳng"),
    "R.310c": ("turn_right", "rẽ phải"),
}
ACTION_LABELS = {
    "turn_left": "LEFT_TURN",
    "going_straight": "STRAIGHT",
    "turn_right": "RIGHT_TURN",
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


def make_rule(code: str, action: str, required_action: str, label: str,
              *, follows_arrow: bool) -> dict:
    if follows_arrow:
        rule_id = f"{code.replace('.', '').upper()}_REQUIRE_{label.upper()}_HAZMAT"
        outcome = "CONDITIONAL"
        effects = [{
            "effect_type": "require_action",
            "action": action,
            "scope": {"hazardous_cargo_fact": "hazardous_cargo_status"},
            "description": f"For a vehicle carrying dangerous goods, follow {required_action} as shown by {code}.",
        }]
        reason = (
            f"QCVN 41:2024/BGTVT D.10 and Figure D.12: {code} displays {required_action}; "
            "vehicles carrying dangerous goods must follow the direction shown. A matching route "
            "is not blanket permission because other signs and controls are not checked."
        )
    else:
        rule_id = f"{code.replace('.', '').upper()}_PROHIBIT_{label.upper()}_HAZMAT"
        outcome = "PROHIBITED"
        effects = [{"effect_type": "prohibit_action", "action": action}]
        reason = (
            f"QCVN 41:2024/BGTVT D.10 and Figure D.12: {code} displays {required_action}; "
            f"a vehicle confirmed to be carrying dangerous goods must follow that indicated direction, "
            f"not {action}. The sign's effective scope and hazardous-cargo status must both be confirmed."
        )
    return {
        "rule_id": rule_id,
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION", "CHECK_PERMISSION"],
        "actions": [action],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": outcome,
        "conditions": [
            condition("hazardous_cargo_status", True),
            condition("sign_effective_scope_confirmed", True),
        ],
        "reason": reason,
        "effects": effects,
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": "Phụ lục D, D.10, Hình D.12 (PDF page 120; Công báo page 38)",
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
            "gazette_source_url": GAZETTE_URL,
        }],
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_codes = {item.get("official_code") for item in catalog.get("items", [])}
    if not set(VARIANTS).issubset(catalog_codes):
        raise ValueError("Official source catalog is missing one or more R.310 variants")

    rules = []
    for code, (required_action, direction_text) in VARIANTS.items():
        rules.append(make_rule(
            code, required_action, direction_text, ACTION_LABELS[required_action], follows_arrow=True
        ))
        for other_action, _ in VARIANTS.values():
            if other_action != required_action:
                rules.append(make_rule(
                    code, other_action, direction_text, ACTION_LABELS[other_action], follows_arrow=False
                ))

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    r310_ids = {f"R{code[2:].replace('.', '').upper()}" for code in VARIANTS}
    existing = {
        item.get("rule_id"): item for item in payload.get("rules", [])
        if not (
            str(item.get("rule_id", "")).startswith(tuple(r310_ids))
            and (not str(item.get("rule_id", "")).isascii()
                 or any(char.isspace() for char in str(item.get("rule_id", ""))))
        )
    }
    existing.update({rule["rule_id"]: rule for rule in rules})
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
