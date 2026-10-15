"""Add review-gated R.404 rules without interpreting an ended scope as blanket permission."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
CATALOG_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"

ENDED_CLASSES = {
    "R.404a": ["car", "truck", "bus", "taxi"],
    "R.404b": ["car", "truck", "bus", "taxi", "motorbike", "motorcycle"],
    "R.404c": ["bus"],
    "R.404d": ["car", "taxi"],
    "R.404e": ["motorbike", "motorcycle"],
    "R.404f": [
        "motorbike", "motorcycle", "moped", "bicycle", "electric_bicycle",
        "cargo_bicycle", "non_motor_vehicle", "non_motorized_tricycle", "hand_cart",
        "human_pushed_vehicle", "animal_drawn_vehicle", "mobility_aid",
    ],
    "R.404g": ["four_wheel_goods_vehicle"],
    "R.404h": ["four_wheel_passenger_vehicle"],
    "R.404k": ["four_wheel_goods_vehicle", "four_wheel_passenger_vehicle"],
}


def main() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    codes = {item.get("official_code") for item in catalog.get("items", [])}
    if not set(ENDED_CLASSES).issubset(codes):
        raise ValueError("Official-source catalog is missing one or more R.404 variants")

    additions = []
    for code, vehicle_classes in ENDED_CLASSES.items():
        additions.append({
            "rule_id": f"{code.replace('.', '').upper()}_END_DESIGNATED_VEHICLE_ROAD_SCOPE",
            "code": code,
            "actions": ["entering", "circulation"],
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
                    "on_mismatch": "NOT_APPLICABLE",
                },
                {
                    "predicate": "vehicle_beyond_r404_end_point",
                    "operator": "equals",
                    "expected_value": True,
                    "required": True,
                    "on_missing": "CONDITIONAL",
                    "on_mismatch": "NOT_APPLICABLE",
                },
                {
                    "predicate": "actor_vehicle_category",
                    "operator": "in",
                    "expected_value": vehicle_classes,
                    "required": True,
                    "on_missing": "CONDITIONAL",
                    "on_mismatch": "NOT_APPLICABLE",
                },
            ],
            "reason": (
                f"QCVN 41:2024/BGTVT Appendix D.12, Figure D.14: {code} marks the end of the "
                "road section dedicated to the vehicle class(es) shown. This ends that sign-defined "
                "road scope at the sign; it does not by itself establish that the vehicle may enter "
                "or continue under all other applicable controls and restrictions."
            ),
            "effects": [{
                "effect_type": "sign_meaning",
                "action": "circulation",
                "scope": {"ended_designated_actor_classes": vehicle_classes},
                "description": "The dedicated-road segment for these vehicle classes ends at this sign.",
            }],
            "additional_legal_basis": [{
                "document": "QCVN 41:2024/BGTVT (issued by Circular 51/2024/TT-BGTVT)",
                "edition": "2024; effective 2025-01-01",
                "section_or_figure": "Appendix D, D.12(a-k), Figure D.14",
                "source_url": SOURCE_URL,
                "accessed_at": ACCESS_DATE,
                "verification_status": "verified",
                "gazette_source_url": GAZETTE_URL,
            }],
        })

    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    by_id.update({item["rule_id"]: item for item in additions})
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-23"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
