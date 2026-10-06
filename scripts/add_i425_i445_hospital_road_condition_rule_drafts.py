"""Add hospital and road-condition sign drafts from QCVN Appendix E.26/E.48."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src/core/symbolic/catalogs/qcvn_41_2024_rule_drafts.json"
SOURCE_URL = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
GAZETTE_URL = "https://congbao.chinhphu.vn/tai-ve-van-ban-so-51-2024-tt-bgtvt-43387-53152?format=pdf"
ACCESS_DATE = "2026-10-05"


def condition(predicate: str, mismatch: str = "CONDITIONAL") -> dict:
    return {
        "predicate": predicate,
        "operator": "equals",
        "expected_value": True,
        "required": True,
        "on_missing": "CONDITIONAL",
        "on_mismatch": mismatch,
    }


def legal_basis(section: str) -> list[dict]:
    return [{
        "document": "QCVN 41:2024/BGTVT, issued with Circular 51/2024/TT-BGTVT",
        "edition": "2024; effective 2025-01-01",
        "section_or_figure": section,
        "source_url": SOURCE_URL,
        "gazette_source_url": GAZETTE_URL,
        "accessed_at": ACCESS_DATE,
        "verification_status": "verified",
    }]


def make_rule(rule_id: str, codes: list[str], actions: list[str], predicates: list[dict],
              reason: str, section: str, effects: list[dict]) -> dict:
    return {
        "rule_id": rule_id,
        "code": codes[0],
        "trigger_sign_codes": codes,
        "intent_kinds": ["CHECK_PERMISSION", "ASK_CONDITION", "QUERY_SIGN_MEANING"],
        # `unknown` lets a sign-meaning query reach the rule; it does not
        # declare any driving action permitted.
        "actions": [*actions, "unknown"],
        "actor_scope": ["vehicle"],
        "excluded_actor_scope": [],
        "outcome": "CONDITIONAL",
        "conditions": [condition("sign_effective_scope_confirmed"), *predicates],
        "reason": reason,
        "effects": effects,
        "additional_legal_basis": legal_basis(section),
    }


def make_rules() -> list[dict]:
    hospital_reason = (
        "QCVN 41:2024/BGTVT Appendix E.26(a-b): I.425 indicates an upcoming nearby treatment "
        "facility such as a hospital, infirmary, or clinic. A road user encountering it must slow down, "
        "observe carefully, and not use the horn. The duty applies at the confirmed indicated approach; "
        "the sign does not establish a new speed number."
    )
    hospital_predicates = [
        condition("i425_medical_facility_and_approach_confirmed"),
        condition("i425_effective_road_segment_resolved"),
        condition("other_applicable_speed_and_traffic_controls_resolved"),
    ]
    rules = [make_rule(
        "I425_HOSPITAL_APPROACH_SLOW_OBSERVE_NO_HORN", ["I.425"],
        ["circulation", "speed", "using_horn"], hospital_predicates, hospital_reason,
        "Appendix E.26(a-b), Figure E.26",
        [
            {"effect_type": "require_action", "action": "reduce_speed",
             "scope": {"approach_fact": "i425_medical_facility_and_approach_confirmed"},
             "description": "Slow down when approaching the indicated nearby medical facility."},
            {"effect_type": "require_action", "action": "observe",
             "scope": {"approach_fact": "i425_medical_facility_and_approach_confirmed"},
             "description": "Pay attention and observe carefully."},
            {"effect_type": "prohibit_action", "action": "using_horn",
             "scope": {"approach_fact": "i425_medical_facility_and_approach_confirmed"},
             "description": "Do not use the horn in the indicated approach area."},
        ],
    )]

    variants = list("abcdefgh")
    codes = [f"I.445{variant}" for variant in variants]
    general_reason = (
        "QCVN 41:2024/BGTVT Appendix E.48: I.445a-h describe unsafe or adverse road conditions "
        "ahead. The family-level text says these signs are placed before the affected section to remind "
        "drivers to pay attention and reduce speed. The exact hazard/driver-specific instruction must be "
        "resolved from the identified variant; do not invent a numeric speed limit."
    )
    # The registry loader binds one rule record to its `code` field. Keep one
    # rule per official variant instead of relying on a multi-code field that
    # the loader intentionally does not consume.
    for variant, code in zip(variants, codes):
        rules.append(make_rule(
            f"I445{variant.upper()}_COMMON_CAUTION_AND_SPEED_REDUCTION", [code],
            ["circulation", "speed"],
            [condition("i445_variant_identity_and_hazard_confirmed"),
             condition("i445_unsafe_road_segment_scope_resolved"),
             condition("other_applicable_speed_and_traffic_controls_resolved")],
            general_reason, "Appendix E.48, Figures E.52-E.54",
            [
                {"effect_type": "require_action", "action": "observe",
                 "scope": {"hazard_fact": "i445_variant_identity_and_hazard_confirmed"},
                 "description": "Focus attention on the adverse or unsafe road condition ahead."},
                {"effect_type": "require_action", "action": "reduce_speed",
                 "scope": {"hazard_fact": "i445_variant_identity_and_hazard_confirmed"},
                 "description": "Reduce speed before entering the affected road section."},
            ],
        ))

    # Variant-specific meanings/instructions from E.48; code suffix is not used to guess pictogram meaning.
    variant_rules = {
        "a": ("slippery_road_confirmed", "I.445a marks a slippery road, for example when wet or oil-covered; QCVN instructs drivers to travel slowly."),
        "b": ("steep_grade_and_restricted_sight_distance_confirmed", "I.445b marks a steep road with restricted sight distance; QCVN instructs drivers to travel slowly and carefully."),
        "c": ("fog_and_restricted_visibility_confirmed", "I.445c marks a foggy section with restricted visibility; QCVN instructs drivers to travel slowly and concentrate on observation."),
        "d": ("weak_or_uneven_roadbed_confirmed", "I.445d marks a weak, subsiding, or uneven roadbed; QCVN reminds drivers to travel slowly and carefully."),
        "e": ("large_or_oversize_vehicle_confirmed", "I.445e guides large or oversize vehicles on a road with at least two lanes to travel at low speed and not occupy lanes for other vehicle types."),
        "f": ("strong_crosswind_area_confirmed", "I.445f warns of a strong crosswind area before a large bridge, bay crossing, or mountain pass where such winds commonly occur."),
        "g": ("accident_prone_road_segment_confirmed", "I.445g marks a road segment where accidents frequently occur."),
        "h": ("continuous_steep_descent_confirmed", "I.445h marks a continuous steep descent; QCVN reminds drivers to travel slowly and keep to the right."),
    }
    for variant, (hazard_predicate, meaning) in variant_rules.items():
        if variant == "e":
            rules.append(make_rule(
                "I445E_LARGE_OVERSIZE_VEHICLE_SPECIFIC_LOW_SPEED_AND_LANE_SCOPE", ["I.445e"],
                ["circulation", "speed", "lane_change"],
                [condition(hazard_predicate), condition("multi_lane_road_confirmed"),
                 condition("other_vehicle_class_lane_scope_resolved")],
                "QCVN 41:2024/BGTVT Appendix E.48: " + meaning,
                "Appendix E.48, Figure E.53 (I.445e)",
                [
                    {"effect_type": "require_action", "action": "reduce_speed",
                     "scope": {"actor_class_fact": hazard_predicate},
                     "description": "Large or oversize vehicles must travel at low speed."},
                    {"effect_type": "prohibit_action", "action": "occupy_lane_of_other_vehicle_classes",
                     "scope": {"actor_class_fact": hazard_predicate},
                     "description": "Do not occupy lanes for other vehicle types."},
                ],
            ))
        else:
            rules.append(make_rule(
                f"I445{variant.upper()}_HAZARD_SPECIFIC_CAUTION", [f"I.445{variant}"],
                ["circulation", "speed"],
                [condition(hazard_predicate), condition("i445_unsafe_road_segment_scope_resolved")],
                "QCVN 41:2024/BGTVT Appendix E.48: " + meaning,
                "Appendix E.48, " + ("Figure E.52" if variant in "abc" else "Figure E.53" if variant in "def" else "Figure E.54"),
                [{"effect_type": "sign_meaning", "action": "circulation",
                  "scope": {"hazard_fact": hazard_predicate, "blanket_permission": False},
                  "description": meaning}],
            ))
    return rules


def main() -> None:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    by_id = {item.get("rule_id"): item for item in payload.get("rules", [])}
    for item in make_rules():
        by_id[item["rule_id"]] = item
    payload["rules"] = list(by_id.values())
    payload["registry_version"] = "qcvn41-rule-drafts-49"
    payload["review_status"] = "unreviewed_draft"
    REGISTRY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
