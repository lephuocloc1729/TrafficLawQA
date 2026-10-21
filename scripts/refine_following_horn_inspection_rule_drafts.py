"""Refine P.121/P.128/P.129 drafts against QCVN 41:2024/BGTVT wording."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"


def condition(predicate: str, operator: str = "exists", expected: object = None,
              *, on_mismatch: str = "NOT_APPLICABLE") -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": on_mismatch,
    }


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = {item.get("official_code"): item for item in catalog.get("items", [])}
    rule_specs = {
        "P121_MINIMUM_FOLLOWING_DISTANCE": {
            "code": "P.121",
            "actions": ["following_distance"],
            "actor_scope": ["car", "taxi", "bus", "truck", "priority_vehicle", "emergency_vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed", "equals", True),
                condition("minimum_following_distance_m"),
                condition("vehicle_following_distance_m", "gte_fact", "minimum_following_distance_m",
                          on_mismatch="PROHIBITED"),
            ],
            "reason": (
                "B.21(a–c): P.121 applies to automobiles, including priority vehicles, and prohibits "
                "a following distance shorter than the value shown in meters. The displayed value, "
                "vehicle class, sign scope, and actual following distance must be established. "
                "Meeting this minimum alone does not establish overall permission."
            ),
            "effects": [{
                "effect_type": "minimum", "action": "following_distance", "unit": "m",
                "value_predicate": "minimum_following_distance_m",
            }],
            "section": "Phụ lục B.21 (PDF page 87)",
        },
        "P128_NO_HORN": {
            "code": "P.128",
            "actions": ["using_horn"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": [],
            "outcome": "PROHIBITED",
            "conditions": [condition("sign_effective_scope_confirmed", "equals", True)],
            "reason": (
                "B.28(a,b): P.128 prohibits vehicles from using the horn within the confirmed sign scope. "
                "S.501 may specify the effect length; an unresolved supplementary-sign relationship "
                "keeps the answer conditional."
            ),
            "effects": [{"effect_type": "prohibit_action", "action": "using_horn"}],
            "section": "Phụ lục B.28 (PDF page 93)",
        },
        "P129_INSPECTION_STOP_REQUIRED": {
            "code": "P.129",
            "actions": ["stopping", "circulation"],
            "actor_scope": ["vehicle"],
            "excluded_actor_scope": [],
            "outcome": "CONDITIONAL",
            "conditions": [
                condition("sign_effective_scope_confirmed", "equals", True),
                condition("vehicle_is_transport_vehicle", "equals", True),
                condition("passing_inspection_station", "equals", True),
            ],
            "reason": (
                "B.29: P.129 identifies an inspection station; transport vehicles passing it must stop "
                "for inspection/control as prescribed. Do not extend this requirement to every road user "
                "without resolving the vehicle and checkpoint facts; the sign does not itself specify "
                "the inspection procedure."
            ),
            "effects": [{"effect_type": "require_action", "action": "stopping"}],
            "section": "Phụ lục B.29 (PDF page 93)",
        },
    }

    for rule_id, spec in rule_specs.items():
        code = spec["code"]
        if code not in entries or not entries[code].get("legal_basis"):
            raise ValueError(f"Missing official catalog provenance for {code}")
        spec["rule_id"] = rule_id
        spec["additional_legal_basis"] = [{
            "document": "QCVN 41:2024/BGTVT",
            "edition": "2024",
            "section_or_figure": spec.pop("section"),
            "source_url": SOURCE_URL,
            "accessed_at": "2026-10-05",
            "verification_status": "verified",
        }]

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    existing = {item.get("rule_id"): item for item in payload.get("rules", [])}
    existing.update(rule_specs)
    payload["rules"] = list(existing.values())
    payload["registry_version"] = "qcvn41-rule-drafts-19"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
