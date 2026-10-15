"""Add source-backed, review-gated R.301 direction rule drafts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
ALLOWED_DIRECTIONS = {
    "a": ["going_straight"],
    "b": ["turn_right"],
    "c": ["turn_left"],
    "d": ["turn_right"],
    "e": ["turn_left"],
    "f": ["going_straight", "turn_right"],
    "g": ["going_straight", "turn_left", "u_turn"],
    "h": ["turn_left", "turn_right", "u_turn"],
}
ALL_DIRECTIONS = ["going_straight", "turn_left", "turn_right", "u_turn"]


def draft(code: str, directions: list[str], allowed: bool) -> dict:
    label = "ALLOW" if allowed else "PROHIBIT"
    effect = "allow_direction" if allowed else "prohibit_action"
    effect_label = "allowed" if allowed else "prohibited"
    return {
        "rule_id": f"R301_{code[-1].upper()}_{label}_{'_'.join(directions).upper()}",
        "code": code,
        "actions": directions,
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": ["priority_vehicle"],
        "outcome": "ALLOWED" if allowed else "PROHIBITED",
        "conditions": [{
            "predicate": "sign_effective_scope_confirmed",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "NOT_APPLICABLE",
        }],
        "reason": (
            f"QCVN 41:2024/BGTVT Appendix D, D.1a(b) lists {', '.join(directions)} as "
            f"{effect_label} by {code}; the sign applies to motorized and non-motorized vehicles, "
            "except priority vehicles as prescribed. Spatial applicability must be confirmed."
        ),
        "effects": [{"effect_type": effect, "action": action,
                     "scope": {"application_scope_fact": "sign_effective_scope_confirmed"}}
                    for action in directions],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    additions: list[dict] = []
    for suffix, allowed in ALLOWED_DIRECTIONS.items():
        code = f"R.301{suffix}"
        prohibited = [action for action in ALL_DIRECTIONS if action not in allowed]
        additions.extend([draft(code, allowed, True), draft(code, prohibited, False)])
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
