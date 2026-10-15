"""Draft R.403 variant rules from QCVN 41:2024 and vehicle classes in Law 36/2024."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
QCVN_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_QCVN_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
GAZETTE_LAW_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-36-2024-qh15-42555-51422?format=pdf"
ACCESS_DATE = "2026-10-05"

# Exact vehicle categories are deliberately conservative. Broad or ambiguous
# actor labels (for example priority_vehicle without an underlying class) stay
# unresolved rather than being coerced into one of these groups.
DESIGNATED_CLASSES = {
    "R.403a": ["car", "truck", "bus", "taxi"],
    "R.403b": ["car", "truck", "bus", "taxi", "motorbike", "motorcycle"],
    "R.403c": ["bus"],
    "R.403d": ["car", "taxi"],
    "R.403e": ["motorbike", "motorcycle"],
    "R.403f": [
        "motorbike", "motorcycle", "moped", "bicycle", "electric_bicycle",
        "cargo_bicycle", "non_motor_vehicle", "non_motorized_tricycle", "hand_cart",
        "human_pushed_vehicle", "animal_drawn_vehicle", "mobility_aid",
    ],
    "R.403g": ["four_wheel_goods_vehicle"],
    "R.403h": ["four_wheel_passenger_vehicle"],
    "R.403k": ["four_wheel_goods_vehicle", "four_wheel_passenger_vehicle"],
}


def condition(predicate: str, operator: str, expected: object) -> dict:
    return {
        "predicate": predicate,
        "operator": operator,
        "expected_value": expected,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": "NOT_APPLICABLE",
    }


def basis(document: str, section: str, source_url: str) -> dict:
    return {
        "document": document,
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": section,
        "source_url": source_url,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }


def make_rules(code: str, designated: list[str]) -> list[dict]:
    stem = code.replace(".", "").upper()
    shared_basis = [
        basis("QCVN 41:2024/BGTVT (Circular 51/2024/TT-BGTVT)",
              "Appendix D, D.11(a-m), Figure D.13", QCVN_URL),
        basis("Law 36/2024/QH15 on Road Traffic Order and Safety",
              "Article 34(1)(a), (d), (đ), (e), (g), and (2)(a-b): vehicle classes",
              "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/9/36-2024-qh15_tiep.pdf"),
    ]
    common = {
        "code": code,
        "intent_kinds": ["CHECK_PERMISSION"],
        "actions": ["entering", "circulation"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "conditions": [condition("sign_effective_scope_confirmed", "equals", True)],
        "additional_legal_basis": shared_basis,
    }
    sign_wording = {
        "R.403a": "all types of automobiles",
        "R.403b": "automobiles and motorcycles",
        "R.403c": "buses",
        "R.403d": "passenger cars",
        "R.403e": "motorcycles",
        "R.403f": "motorcycles including mopeds, and bicycles including non-motor vehicles",
        "R.403g": "motorized four-wheel goods vehicles",
        "R.403h": "motorized four-wheel passenger vehicles",
        "R.403k": "motorized four-wheel goods and passenger vehicles",
    }[code]
    return [
        {
            **common,
            "rule_id": f"{stem}_DESIGNATED_VEHICLE_SCOPE",
            "conditions": [
                *common["conditions"],
                condition("actor_vehicle_category", "in", designated),
            ],
            "outcome": "CONDITIONAL",
            "reason": (
                f"QCVN 41:2024/BGTVT Appendix D.11 and Figure D.13: {code} marks a road for "
                f"{sign_wording}. A matching class means only that the sign's designated-class "
                "restriction does not exclude this vehicle; it is not blanket permission."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "entering",
                "scope": {"designated_actor_classes": designated},
                "description": f"This vehicle class is represented within the designated scope of {code}.",
            }],
        },
        {
            **common,
            "rule_id": f"{stem}_PROHIBIT_OTHER_VEHICLE_CLASSES",
            "conditions": [
                *common["conditions"],
                condition("actor_vehicle_category", "not_in", designated),
            ],
            "outcome": "PROHIBITED",
            "reason": (
                f"QCVN 41:2024/BGTVT Appendix D.11(l): vehicle types other than those shown on "
                f"{code} may not enter the road section where the sign applies. Vehicle-class "
                "identity and effective sign scope must both be explicit."
            ),
            "effects": [{"effect_type": "prohibit_action", "action": "entering"}],
        },
    ]


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    catalog_codes = {item.get("official_code") for item in catalog.get("items", [])}
    if not set(DESIGNATED_CLASSES).issubset(catalog_codes):
        raise ValueError("The official-source sign catalog is missing one or more R.403 variants")

    additions = [
        rule for code, actors in DESIGNATED_CLASSES.items()
        for rule in make_rules(code, actors)
    ]
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rules_by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    rules_by_id.update({rule["rule_id"]: rule for rule in additions})
    payload["rules"] = list(rules_by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-22"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
