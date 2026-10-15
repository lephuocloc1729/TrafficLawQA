"""Add conservative DP.127a/b/c drafts for ending combined-sign speed limits."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"
VARIANTS = ("DP.127a", "DP.127b", "DP.127c")


def make_rule(code: str) -> dict:
    return {
        "rule_id": f"{code.replace('.', '').upper()}_END_COMBINED_SPEED_LIMIT",
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["speed"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [{
            "predicate": "sign_effective_scope_confirmed",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "NOT_APPLICABLE",
        }, {
            "predicate": "vehicle_beyond_dp127_sign_confirmed",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "NOT_APPLICABLE",
        }, {
            "predicate": "applicable_statutory_max_speed_kmh",
            "operator": "exists",
            "expected_value": None,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "CONDITIONAL",
        }, {
            "predicate": "other_applicable_speed_limits_resolved",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "CONDITIONAL",
        }],
        "reason": (
            f"QCVN 41:2024/BGTVT B.27d: {code} is a variant of DP.127, which ends the P.127b/P.127c "
            "combined maximum-speed restriction. From this sign the applicable maximum speed is the "
            "one prescribed by the Road Law and Road Traffic Safety Law. This sign-only rule does not "
            "invent or retrieve that statutory number: it requires an explicitly supplied, applicable "
            "legal maximum and resolution of any other active speed restrictions. Missing facts remain "
            "CONDITIONAL; at-or-below-limit is not blanket permission."
        ),
        "effects": [{
            "effect_type": "maximum",
            "action": "speed",
            "unit": "km/h",
            "value_predicate": "applicable_statutory_max_speed_kmh",
            "scope": {
                "ends_sign_codes": ["P.127b", "P.127c"],
                "from_sign": code,
                "statutory_limit_fact_source_required": True,
            },
            "description": (
                "The P.127b/P.127c combined-sign ceiling ends here; use only the separately resolved "
                "statutory maximum for this road and vehicle. This does not establish overall permission."
            ),
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix B.27d, Figure B.27d: DP.127(a,b,c) ends P.127(b,c); maximum thereafter is set by applicable road-traffic laws",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }, {
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": f"Appendix M, group DP.127, Figure/design for {code}; variant is selected according to traffic organization",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    additions = [make_rule(code) for code in VARIANTS]
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-40"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
