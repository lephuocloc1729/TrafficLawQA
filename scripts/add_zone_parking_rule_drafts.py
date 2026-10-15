"""Add conservative area-wide parking rule drafts for R.E.9a/b and R.E.10a/b."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(
    predicate: str,
    expected: object,
    *,
    operator: str = "equals",
    mismatch: str = "NOT_APPLICABLE",
) -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": mismatch,
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


def start_rule(code: str, timed: bool) -> dict:
    time_conditions = [condition(
        "time_window", None, operator="time_window_contains_now", mismatch="CONDITIONAL",
    )] if timed else []
    label = "timed no-parking" if timed else "no-parking"
    return {
        "rule_id": "RE9B_NO_PARKING_DURING_DISPLAYED_TIME_IN_ZONE" if timed else "RE9A_NO_PARKING_IN_ZONE",
        "code": code,
        "actions": ["parking"],
        "actor_scope": ["motor_vehicle"],
        "excluded_actor_scope": ["priority_vehicle"],
        "outcome": "PROHIBITED",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_within_zone_confirmed", True),
            *time_conditions,
        ],
        "reason": (
            f"QCVN 41:2024/BGTVT D.19: {code} begins an area-wide {label} restriction covering all roads "
            "in the confirmed zone. The rule requires explicit zone membership; for R.E.9b, it also "
            "requires a strictly read time window and the system's current local time. The actor scope "
            "and priority-vehicle treatment are review-gated mappings; no permission is inferred when "
            "the time window is inactive or any fact is missing."
        ),
        "effects": [{
            "effect_type": "prohibit_action",
            "action": "parking",
            "scope": {
                "zone_fact": "vehicle_within_zone_confirmed",
                "all_roads_in_zone": True,
                "time_window_fact": "time_window" if timed else None,
            },
            "description": f"No parking under {code} within the confirmed zone{ ' during the displayed time window' if timed else '' }.",
        }],
        "additional_legal_basis": [
            basis("Chapter 6, D.19: R.E.9 starts the restriction on every road in the area; R.E.9a/b parking effects"),
            *([basis("Appendix M, group R.E.9b, Figure D.21: displayed time-range panel example")] if timed else []),
            basis("Appendix B.31(a), parking-sign actor/priority scope reference; application to zone signs requires team review"),
        ],
    }


def end_rule(code: str, timed: bool) -> dict:
    restriction = "R.E.9b time-based no-parking" if timed else "R.E.9a no-parking"
    return {
        "rule_id": "RE10B_END_OF_TIMED_NO_PARKING_ZONE" if timed else "RE10A_END_OF_NO_PARKING_ZONE",
        "code": code,
        "actions": ["parking"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [
            condition("sign_effective_scope_confirmed", True),
            condition("vehicle_beyond_zone_exit_confirmed", True),
            condition("other_applicable_parking_restrictions_resolved", True),
        ],
        "reason": (
            f"QCVN 41:2024/BGTVT D.20: {code} indicates exit from the area where {restriction} applied. "
            "It ends only that area-wide restriction. The vehicle must be confirmed past the exit sign, "
            "and other parking restrictions must be resolved; the sign does not itself establish permission."
        ),
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "parking",
            "scope": {"ends_zone_restriction": restriction, "blanket_permission": False},
            "description": f"{code} marks the end of the corresponding zone parking restriction; other restrictions remain unresolved.",
        }],
        "additional_legal_basis": [
            basis("Chapter 6, D.20: corresponding R.E.10 sign indicates exit from an area-wide restriction"),
        ],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    additions = [
        start_rule("R.E.9a", timed=False),
        start_rule("R.E.9b", timed=True),
        end_rule("R.E.10a", timed=False),
        end_rule("R.E.10b", timed=True),
    ]
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-39"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
