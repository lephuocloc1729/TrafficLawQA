"""Add a conservative I.446 accessible-parking interpretation draft."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(predicate: str) -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": True,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "CONDITIONAL",
    }


def make_rule() -> dict:
    predicates = [
        "i446_reserved_parking_place_confirmed",
        "actor_eligibility_for_i446_reserved_parking_confirmed",
        "i446_related_p131a_i408_signs_and_markings_resolved",
        "other_applicable_parking_constraints_resolved",
    ]
    reason = (
        "QCVN 41:2024/BGTVT Appendix E.49, Figure E.55: I.446 identifies a parking place for "
        "persons with disabilities and may be combined with P.131a and I.408. QCVN E.49 does not "
        "specify the individual eligibility test or how those co-displayed signs modify one another; "
        "the engine requires explicit eligibility and full sign/marking resolution and does not treat "
        "I.446 alone as an unrestricted parking authorization."
    )
    return {
        "rule_id": "I446_ACCESSIBLE_PARKING_SCOPE_REQUIRES_ELIGIBILITY_AND_OVERLAY_REVIEW",
        "code": "I.446",
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION"],
        "actions": ["parking"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"), *map(condition, predicates)],
        "reason": reason,
        "effects": [{
            "effect_type": "sign_meaning",
            "action": "parking",
            "scope": {"required_facts": predicates, "blanket_permission": False},
            "description": reason,
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix E.49, Figure E.55: I.446 disabled-person parking-place sign; may be combined with P.131a and I.408",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    rule = make_rule()
    by_id[rule["rule_id"]] = rule
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-46"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
