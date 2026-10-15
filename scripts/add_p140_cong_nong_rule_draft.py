"""Add a review-gated P.140 draft from official QCVN 41:2024/BGTVT text."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    if "P.140" not in {item.get("official_code") for item in catalog.get("items", [])}:
        raise ValueError("Official-source sign catalog is missing P.140")

    rule = {
        "rule_id": "P140_PROHIBIT_CONG_NONG_VEHICLES",
        "code": "P.140",
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["entering", "circulation"],
        "actor_scope": ["cong_nong_vehicle"],
        "excluded_actor_scope": [],
        "outcome": "PROHIBITED",
        "conditions": [{
            "predicate": "sign_effective_scope_confirmed",
            "operator": "equals",
            "expected_value": True,
            "required": True,
            "on_missing": "CONDITIONAL",
            "on_mismatch": "NOT_APPLICABLE",
        }],
        "reason": (
            "QCVN 41:2024/BGTVT Appendix B.40, Figure B.40: P.140 is used to prohibit agricultural "
            "tractors (xe công nông) and similar vehicles on the road. The QCVN text does not enumerate "
            "the vehicles considered similar; the engine maps only an explicitly identified xe công nông. "
            "Other vehicle types remain unresolved unless their classification is supplied and reviewed. "
            "The rule also requires the sign's effective scope to be confirmed."
        ),
        "effects": [{
            "effect_type": "prohibit_action",
            "action": "circulation",
            "scope": {
                "application_scope_fact": "sign_effective_scope_confirmed",
                "vehicle_class": "cong_nong_vehicle",
            },
            "description": "P.140 prohibits circulation of an explicitly identified xe công nông within the confirmed sign scope.",
        }],
        "additional_legal_basis": [{
            "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
            "edition": "2024; effective 2025-01-01",
            "section_or_figure": "Appendix B.40, Figure B.40",
            "source_url": SOURCE_URL,
            "gazette_source_url": GAZETTE_URL,
            "accessed_at": ACCESS_DATE,
            "verification_status": "verified",
        }],
    }

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id[rule["rule_id"]] = rule
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-30"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
