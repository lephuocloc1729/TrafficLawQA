import unittest
from dataclasses import replace
from datetime import datetime
import json
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from src.core.symbolic.fact_schema import ContextFact, SignObservation, SignRelation, TrafficFacts
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation
from src.core.symbolic.rule_schema import RuleCondition, RuleEffect, TrafficRule
from src.core.symbolic.traffic_rules import (
    TRAFFIC_RULEBOOK_METADATA,
    TRAFFIC_RULES,
    find_matching_rules,
    load_rulebook,
    vehicle_is_covered,
)


def facts_for(payload: dict) -> TrafficFacts:
    observation = normalize_payload_to_observation(payload)
    return TrafficFacts(
        "2.0", "scene-1", [observation], sign_inventory_status="complete",
        sign_inventory_evidence=[{"source": "test_fixture", "evidence": "fixture contains all scene signs"}],
    )


def _scoped_sign_facts(code: str) -> TrafficFacts:
    facts = facts_for({"sign_code": code})
    facts.sign_observations[0].visual_attributes["normalized_conditions"] = {
        "sign_effective_scope_confirmed": True
    }
    facts.sign_observations[0].confidence = "high"
    return facts


def query(text: str, now: datetime | None = None):
    stable_even_day = now or datetime(2026, 10, 4, 12, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    return parse_query_intent_deterministic(text, now=stable_even_day)


def evaluate(facts: TrafficFacts, parsed, rules=TRAFFIC_RULES):
    if rules is TRAFFIC_RULES:
        rules = reviewed_rules(rules)
    return find_matching_rules(
        facts,
        parsed.query_intent,
        parsed.context_facts,
        rules=rules,
    )


def reviewed_rules(rules):
    return tuple(replace(
        rule,
        verification_status="verified",
        legal_basis=[{**basis, "verification_status": "verified"} for basis in rule.legal_basis],
    ) for rule in rules)


def has_unresolved_condition(evaluations, predicates):
    predicates = set(predicates)
    return any(
        predicates.intersection(item.missing_facts)
        or any(
            trace.get("predicate") in predicates
            and trace.get("state") == "skipped_unobservable_scene_fact"
            for trace in item.trace
        )
        for item in evaluations
    )


class TrafficRulesTest(unittest.TestCase):
    def test_rules_are_declarative_and_registered(self) -> None:
        self.assertEqual(len(TRAFFIC_RULES), 373)
        self.assertEqual(TRAFFIC_RULEBOOK_METADATA["status"], "draft_unreviewed")
        self.assertTrue(all(rule.trigger_sign_codes and rule.actions for rule in TRAFFIC_RULES))
        self.assertTrue(all(rule.verification_status == "unverified" for rule in TRAFFIC_RULES))
        self.assertTrue(all(rule.legal_basis[0].get("section_or_figure") for rule in TRAFFIC_RULES))
        self.assertTrue(all(rule.legal_basis[0].get("source_url", "").startswith("https://datafiles.chinhphu.vn/") for rule in TRAFFIC_RULES))
        self.assertTrue(all(rule.effects or rule.outcome == "CONDITIONAL" for rule in TRAFFIC_RULES))
        self.assertEqual(TRAFFIC_RULEBOOK_METADATA["rule_draft_count"], 372)
        self.assertTrue(all("intent_kinds" not in rule.to_dict() for rule in TRAFFIC_RULES))

    def test_r122_stop_and_proceed_drafts_fail_closed(self) -> None:
        facts = _scoped_sign_facts("R.122")
        r122_rules = [rule for rule in TRAFFIC_RULES if "R.122" in rule.trigger_sign_codes]
        proceed_query = query("Xe ô tô có được đi tiếp không?")

        unreviewed = evaluate(facts, proceed_query, r122_rules)
        self.assertTrue(unreviewed)
        self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
        self.assertTrue(all("legal_review_required" not in item.missing_facts for item in unreviewed))

        reviewed = reviewed_rules(r122_rules)
        no_context = evaluate(facts, proceed_query, reviewed)
        control_rule = next(
            item for item in no_context
            if item.rule_id == "R122_PROCEED_ONLY_AFTER_CONTROL_PERMISSION"
        )
        self.assertEqual(control_rule.decision, "CONDITIONAL")
        self.assertNotIn("vehicle_stopped_before_r122", control_rule.missing_facts)
        self.assertTrue(any(
            item.get("predicate") == "vehicle_stopped_before_r122"
            and item.get("state") == "skipped_unobservable_scene_fact"
            for item in control_rule.trace
        ))
        self.assertNotIn("r122_control_signal_or_officer_permits_proceeding", control_rule.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in control_rule.effects))

        facts.context_facts.extend([
            ContextFact(
                "r122-stop-confirmed", "vehicle_stopped_before_r122", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "r122-control-permits", "r122_control_signal_or_officer_permits_proceeding", True,
                status="provided", source="observed_traffic_control", confidence="high",
            ),
        ])
        with_permission = evaluate(facts, proceed_query, reviewed)
        control_rule = next(
            item for item in with_permission
            if item.rule_id == "R122_PROCEED_ONLY_AFTER_CONTROL_PERMISSION"
        )
        self.assertEqual(control_rule.decision, "CONDITIONAL")
        self.assertNotEqual(control_rule.decision, "ALLOWED")
        self.assertNotIn("other_applicable_sign_constraints", control_rule.missing_facts)

        stop_query = query("Xe ô tô có phải dừng tại biển này không?")
        facts.context_facts.append(ContextFact(
            "r122-approach-confirmed", "vehicle_approaching_r122_stop_position", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        stop_rules = [rule for rule in reviewed if rule.rule_id == "R122_REQUIRE_STOP_BEFORE_SIGN_OR_LINE"]
        stop_result = evaluate(facts, stop_query, stop_rules)
        self.assertTrue(stop_result)
        self.assertEqual(stop_result[0].decision, "CONDITIONAL")
        self.assertTrue(any(effect.effect_type == "require_action" for effect in stop_result[0].effects))

    def test_i425_and_i445_rules_cover_each_official_variant_and_fail_closed(self) -> None:
        expected_codes = ["I.425", *(f"I.445{variant}" for variant in "abcdefgh")]
        for code in expected_codes:
            with self.subTest(code=code):
                matched = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                self.assertTrue(matched)
                self.assertTrue(all(rule.verification_status == "unverified" for rule in matched))
                parsed = query(f"Xe ô tô có được lưu thông qua khu vực đặt biển {code} không?")
                result = evaluate(_scoped_sign_facts(code), parsed, matched)
                self.assertTrue(result)
                self.assertTrue(all(item.verification_status == "unverified" for item in result))
                self.assertTrue(all("legal_review_required" not in item.missing_facts for item in result))

        hospital = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "I425_HOSPITAL_APPROACH_SLOW_OBSERVE_NO_HORN")
        self.assertEqual(
            [(effect.effect_type, effect.action) for effect in hospital.effects],
            [("require_action", "reduce_speed"), ("require_action", "observe"), ("prohibit_action", "using_horn")],
        )
        i445e = [rule for rule in TRAFFIC_RULES if "I.445e" in rule.trigger_sign_codes]
        self.assertEqual(len(i445e), 2)
        self.assertTrue(any(any(effect.action == "occupy_lane_of_other_vehicle_classes" for effect in rule.effects) for rule in i445e))

    def test_i426_i435_meanings_are_not_permission_rules(self) -> None:
        codes = [
            "I.426", "I.427a", "I.427b", "I.428a", "I.428b", "I.428c", "I.429",
            "I.430", "I.431", "I.432", "I.433a", "I.433b", "I.433c", "I.433d",
            "I.433e", "I.435",
        ]
        for code in codes:
            with self.subTest(code=code):
                self.assertFalse(any(code in rule.trigger_sign_codes for rule in TRAFFIC_RULES))

        i436_rules = [rule for rule in TRAFFIC_RULES if "I.436" in rule.trigger_sign_codes]
        self.assertEqual(len(i436_rules), 2)
        self.assertTrue(all(rule.verification_status == "unverified" for rule in i436_rules))
        self.assertTrue(any(
            effect.effect_type == "require_action" and effect.action == "speed"
            for rule in i436_rules for effect in rule.effects
        ))
        self.assertTrue(any(
            effect.effect_type == "prohibit_action" and effect.action == "overtaking"
            for rule in i436_rules for effect in rule.effects
        ))

        stop_sign_rules = [rule for rule in TRAFFIC_RULES
                           if set(rule.trigger_sign_codes).intersection({"I.434a", "I.434b"})]
        self.assertEqual(len(stop_sign_rules), 2)
        self.assertTrue(all(rule.outcome == "CONDITIONAL" for rule in stop_sign_rules))
        self.assertTrue(all(
            all(effect.effect_type == "sign_meaning" for effect in rule.effects)
            for rule in stop_sign_rules
        ))

    def test_r415_and_populated_area_sign_rules_stay_conditional_and_do_not_invent_limits(self) -> None:
        cases = [
            ("R.415a", "Biển R.415a có ý nghĩa gì?", "INFORMATION"),
            ("R.415b", "Biển R.415b có ý nghĩa gì?", "INFORMATION"),
            ("R.420", "Tốc độ tối đa trong khu đông dân cư là bao nhiêu?", "INFORMATION"),
            ("R.421", "Tốc độ tối đa sau biển R.421 là bao nhiêu?", "INFORMATION"),
        ]
        for code, question, missing_predicate in cases:
            with self.subTest(code=code):
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                self.assertEqual(len(rules), 1)
                self.assertTrue(all(rule.verification_status == "unverified" for rule in rules))
                facts = _scoped_sign_facts(code)
                parsed = query(question)

                self.assertEqual(parsed.query_intent.intents[0].kind, missing_predicate)
                self.assertEqual(evaluate(facts, parsed, rules), [])

        r415a = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "R415A_LANE_AND_VEHICLE_DIAGRAM_MEANING")
        self.assertIn("two to four motor-vehicle lanes per direction", r415a.reason)
        self.assertIn("stopping/parking", r415a.reason)
        r420 = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "R420_START_POPULATED_AREA_SCOPE_NOT_SPEED_VALUE")
        self.assertIn("posted P.127", r420.reason)

    def test_r412_r413_ending_lane_codes_remain_unruled_pending_official_code_conflict_review(self) -> None:
        conflict_codes = [*(f"R.412{variant}" for variant in "ijklmnop"), *(f"R.413{variant}" for variant in "ijklmnop")]
        for code in conflict_codes:
            with self.subTest(code=code):
                self.assertFalse(any(code in rule.trigger_sign_codes for rule in TRAFFIC_RULES))

    def test_i418_route_diagram_never_implies_blanket_permission(self) -> None:
        facts = _scoped_sign_facts("I.418")
        rules = [rule for rule in TRAFFIC_RULES if "I.418" in rule.trigger_sign_codes]
        parsed = query("Xe ô tô có được rẽ trái theo biển chỉ dẫn không?")

        unreviewed = find_matching_rules(
            facts, parsed.query_intent, parsed.context_facts, rules=rules
        )
        self.assertTrue(unreviewed)
        self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
        self.assertTrue(all("legal_review_required" not in item.missing_facts for item in unreviewed))

        reviewed = reviewed_rules(rules)
        incomplete = find_matching_rules(
            facts, parsed.query_intent, parsed.context_facts, rules=reviewed
        )
        route_rule = next(item for item in incomplete if item.rule_id == "I418_DIAGRAM_DEPENDENT_ROUTE_GUIDANCE")
        self.assertEqual(route_rule.decision, "CONDITIONAL")
        self.assertIn("requested_path_matches_i418_diagram", route_rule.missing_facts)
        self.assertFalse(any(effect.effect_type in {"permit_action", "allow_direction"} for effect in route_rule.effects))

        facts.context_facts.extend([
            ContextFact("i418-intersection", "intersection_where_turn_is_prohibited_confirmed", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("i418-path", "requested_path_matches_i418_diagram", True,
                        status="provided", source="reviewed_diagram_match", confidence="high"),
            ContextFact("i418-other-signs", "other_applicable_sign_constraints_resolved", True,
                        status="provided", source="complete_scene_review", confidence="high"),
        ])
        resolved = find_matching_rules(
            facts, parsed.query_intent, parsed.context_facts, rules=reviewed
        )
        route_rule = next(item for item in resolved if item.rule_id == "I418_DIAGRAM_DEPENDENT_ROUTE_GUIDANCE")
        self.assertEqual(route_rule.decision, "CONDITIONAL")
        self.assertTrue(all(effect.effect_type == "sign_meaning" for effect in route_rule.effects))

    def test_i408_i410_positive_facility_signs_require_complete_local_facts(self) -> None:
        cases = [
            ("I.408", "Xe ô tô có được đỗ ở đây không?", "I408_PERMIT_PARKING_WITHIN_DESIGNATED_PLACE",
             ["parking_location_within_i408_designated_place", "i408_scope_and_supplementary_panels_resolved",
              "other_applicable_parking_constraints_resolved"]),
            ("I.408a", "Xe ô tô có được đỗ ở đây không?", "I408A_PERMIT_PARTIAL_SIDEWALK_PARKING",
             ["parking_location_within_i408a_designated_place", "i408a_scope_and_supplementary_panels_resolved",
              "passenger_side_wheels_on_sidewalk_confirmed", "other_applicable_parking_constraints_resolved"]),
            ("I.409", "Xe ô tô có được quay đầu ở đây không?", "I409_PERMIT_U_TURN_AT_DESIGNATED_PLACE",
             ["u_turn_location_matches_i409_designated_place", "i409_position_and_s502_distance_resolved",
              "other_applicable_turn_constraints_resolved"]),
            ("I.410", "Xe ô tô có được quay đầu ở đây không?", "I410_PERMIT_U_TURN_WITHIN_DESIGNATED_AREA",
             ["u_turn_location_within_i410_designated_area", "requested_u_turn_matches_i410_diagram",
              "i410_position_and_s502_distance_resolved", "other_applicable_turn_constraints_resolved"]),
        ]
        for code, question, rule_id, required_facts in cases:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                parsed = query(question)

                unreviewed = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=rules
                )
                self.assertTrue(unreviewed)
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))

                reviewed = reviewed_rules(rules)
                incomplete = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                result = next(item for item in incomplete if item.rule_id == rule_id)
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(has_unresolved_condition([result], required_facts))

                facts.context_facts.extend(ContextFact(
                    f"{code}-{index}", predicate, True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ) for index, predicate in enumerate(required_facts))
                resolved = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                result = next(item for item in resolved if item.rule_id == rule_id)
                self.assertEqual(result.decision, "ALLOWED")
                self.assertTrue(any(effect.effect_type == "permit_action" for effect in result.effects))
                self.assertIs(result.effects[0].scope.get("blanket_permission"), False)

    def test_i413_i417_route_guidance_never_infers_lane_or_route_permission(self) -> None:
        cases = [
            ("I.413a", ["i413a_diagram_and_opposite_direction_resolved", "bus_lane_effective_scope_resolved",
                        "actor_bus_lane_eligibility_resolved", "other_applicable_route_constraints_resolved"]),
            ("I.413b", ["requested_turn_matches_i413_diagram", "destination_bus_lane_scope_resolved",
                        "actor_bus_lane_eligibility_resolved", "other_applicable_route_constraints_resolved"]),
            ("I.413c", ["requested_turn_matches_i413_diagram", "destination_bus_lane_scope_resolved",
                        "actor_bus_lane_eligibility_resolved", "other_applicable_route_constraints_resolved"]),
            *[(f"I.417{variant}", ["vehicle_class_shown_on_i417_resolved",
                                    "requested_route_matches_i417_diagram_for_vehicle_class",
                                    "underlying_vehicle_restriction_and_route_scope_resolved",
                                    "other_applicable_route_constraints_resolved"])
              for variant in "abc"],
        ]
        parsed = query("Xe ô tô có được lưu thông theo hướng này không?")
        for code, required_facts in cases:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                unreviewed = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=rules
                )
                self.assertTrue(unreviewed)
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))

                reviewed = reviewed_rules(rules)
                incomplete = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                self.assertTrue(incomplete)
                self.assertTrue(all(item.decision == "CONDITIONAL" for item in incomplete))
                self.assertTrue(any(
                    has_unresolved_condition([item], required_facts) for item in incomplete
                ))
                facts.context_facts.extend(ContextFact(
                    f"{code}-{index}", predicate, True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ) for index, predicate in enumerate(required_facts))
                resolved = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                self.assertTrue(all(item.decision == "CONDITIONAL" for item in resolved))
                self.assertTrue(all(
                    all(effect.effect_type == "sign_meaning" for effect in item.effects)
                    for item in resolved if item.status != "not_applicable"
                ))

    def test_i414_i416_direction_signs_remain_conditional_without_route_permission(self) -> None:
        cases = [
            (f"I.414{variant}", ["i414_road_number_and_destination_text_resolved",
                                  "i414_arrow_geometry_matches_intersection",
                                  "requested_route_matches_i414_direction",
                                  "other_applicable_route_constraints_resolved"])
            for variant in "abcd"
        ] + [
            ("I.415", ["i415_destination_and_arrow_resolved", "requested_route_matches_i415_arrow",
                        "other_applicable_route_constraints_resolved"]),
            ("I.416", ["i416_detour_route_matches_diagram",
                        "i416_embedded_prohibition_sign_resolved_when_present",
                        "affected_vehicle_class_and_underlying_restriction_resolved",
                        "alternative_route_applicability_confirmed",
                        "other_applicable_route_constraints_resolved"]),
        ]
        parsed = query("Xe ô tô có được lưu thông theo hướng này không?")
        for code, required_facts in cases:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                self.assertTrue(rules)
                unreviewed = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=rules
                )
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
                reviewed = reviewed_rules(rules)
                incomplete = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                self.assertTrue(all(item.decision == "CONDITIONAL" for item in incomplete))
                self.assertTrue(any(
                    has_unresolved_condition([item], required_facts) for item in incomplete
                ))
                facts.context_facts.extend(ContextFact(
                    f"{code}-{index}", predicate, True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ) for index, predicate in enumerate(required_facts))
                resolved = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                self.assertTrue(all(item.decision == "CONDITIONAL" for item in resolved))
                self.assertTrue(all(
                    all(effect.effect_type == "sign_meaning" for effect in item.effects)
                    for item in resolved if item.status != "not_applicable"
                ))

    def test_i437_i448_never_turn_guidance_into_general_vehicle_permission(self) -> None:
        cases = [
            ("I.437", ["i437_highway_start_location_confirmed",
                       "actor_vehicle_class_and_highway_eligibility_resolved",
                       "other_applicable_highway_sign_and_legal_constraints_resolved"]),
            ("I.448", ["i448_emergency_lane_location_confirmed",
                       "i448_distance_and_turn_point_resolved",
                       "actor_vehicle_class_and_reason_for_use_resolved",
                       "other_applicable_emergency_lane_constraints_resolved"]),
        ]
        parsed = query("Xe ô tô có được lưu thông trên làn này không?")
        for code, required_facts in cases:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                unreviewed = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=rules
                )
                self.assertTrue(unreviewed)
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
                reviewed = reviewed_rules(rules)
                incomplete = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                result = next(item for item in incomplete if item.rule_id == rules[0].rule_id)
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(has_unresolved_condition([result], required_facts))
                facts.context_facts.extend(ContextFact(
                    f"{code}-{index}", predicate, True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ) for index, predicate in enumerate(required_facts))
                resolved = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                result = next(item for item in resolved if item.rule_id == rules[0].rule_id)
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(all(effect.effect_type == "sign_meaning" for effect in result.effects))

    def test_i446_accessible_parking_requires_explicit_eligibility_and_sign_overlay(self) -> None:
        facts = _scoped_sign_facts("I.446")
        rules = [rule for rule in TRAFFIC_RULES if "I.446" in rule.trigger_sign_codes]
        parsed = query("Xe ô tô có được đỗ ở đây không?")
        unreviewed = find_matching_rules(
            facts, parsed.query_intent, parsed.context_facts, rules=rules
        )
        self.assertTrue(unreviewed)
        self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))

        reviewed = reviewed_rules(rules)
        incomplete = find_matching_rules(
            facts, parsed.query_intent, parsed.context_facts, rules=reviewed
        )
        result = next(item for item in incomplete if item.rule_id.startswith("I446_"))
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertIn("actor_eligibility_for_i446_reserved_parking_confirmed", result.missing_facts)

        predicates = [
            "i446_reserved_parking_place_confirmed",
            "actor_eligibility_for_i446_reserved_parking_confirmed",
            "i446_related_p131a_i408_signs_and_markings_resolved",
            "other_applicable_parking_constraints_resolved",
        ]
        facts.context_facts.extend(ContextFact(
            f"i446-{index}", predicate, True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ) for index, predicate in enumerate(predicates))
        resolved = find_matching_rules(
            facts, parsed.query_intent, parsed.context_facts, rules=reviewed
        )
        result = next(item for item in resolved if item.rule_id.startswith("I446_"))
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertTrue(all(effect.effect_type == "sign_meaning" for effect in result.effects))
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in result.effects))

    def test_i440_i441_workzone_signs_do_not_invent_closures_or_speed_limits(self) -> None:
        shared = ["workzone_vehicle_restrictions_and_speed_signs_resolved",
                  "other_applicable_route_constraints_resolved"]
        cases = [
            ("I.440", ["i440_active_workzone_segment_confirmed", "i440_start_end_and_chainage_resolved",
                       "i441_warning_sign_and_other_workzone_panels_resolved", *shared]),
            *[(f"I.441{variant}", ["i441_worksite_ahead_and_approach_confirmed",
                                   "i441_distance_stage_and_workzone_extent_resolved",
                                   "i441_linked_w227_warning_resolved",
                                   "i441_p127_p134_and_other_panels_resolved_when_present", *shared])
              for variant in "abc"],
        ]
        parsed = query("Xe ô tô có được đi qua đoạn đường này không?")
        for code, required_facts in cases:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                unreviewed = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=rules
                )
                self.assertTrue(unreviewed)
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
                reviewed = reviewed_rules(rules)
                incomplete = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                result = next(item for item in incomplete if item.rule_id == rules[0].rule_id)
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(has_unresolved_condition([result], required_facts))
                facts.context_facts.extend(ContextFact(
                    f"{code}-{index}", predicate, True,
                    status="provided", source="reviewed_workzone_assessment", confidence="high",
                ) for index, predicate in enumerate(required_facts))
                resolved = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                result = next(item for item in resolved if item.rule_id == rules[0].rule_id)
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(all(effect.effect_type == "sign_meaning" for effect in result.effects))

    def test_i423_i424_pedestrian_signs_preserve_crossing_duties_and_facility_scope(self) -> None:
        cases = [
            ("I.423a", "Xe ô tô có phải nhường người đi bộ tại đây không?",
             ["i423_crossing_location_and_sign_direction_confirmed",
              "pedestrian_presence_at_or_approaching_crossing_resolved",
              "crosswalk_marking_and_traffic_control_resolved",
              "other_applicable_pedestrian_and_vehicle_controls_resolved"]),
            ("I.423b", "Xe ô tô có phải nhường người đi bộ tại đây không?",
             ["i423_crossing_location_and_sign_direction_confirmed",
              "pedestrian_presence_at_or_approaching_crossing_resolved",
              "crosswalk_marking_and_traffic_control_resolved",
              "other_applicable_pedestrian_and_vehicle_controls_resolved"]),
            ("I.423c", "Xe ô tô có được đi vào đoạn đường này không?",
             ["i423c_start_of_pedestrian_path_confirmed",
              "requested_location_is_not_a_road_crossing_confirmed",
              "vehicle_access_or_pedestrian_route_rules_resolved"]),
            ("I.424a", "Người đi bộ có được sang đường bằng cầu vượt không?",
             ["i424_overpass_location_confirmed", "pedestrian_direction_matches_i424_diagram"]),
            ("I.424b", "Người đi bộ có được sang đường bằng cầu vượt không?",
             ["i424_overpass_location_confirmed", "pedestrian_direction_matches_i424_diagram"]),
            ("I.424c", "Người đi bộ có được sang đường bằng hầm chui không?",
             ["i424_underpass_location_confirmed", "pedestrian_direction_matches_i424_diagram"]),
            ("I.424d", "Người đi bộ có được sang đường bằng hầm chui không?",
             ["i424_underpass_location_confirmed", "pedestrian_direction_matches_i424_diagram"]),
        ]
        for code, question, required_facts in cases:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                parsed = query(question)
                unreviewed = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=rules
                )
                self.assertTrue(unreviewed)
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
                reviewed = reviewed_rules(rules)
                incomplete = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                relevant = [item for item in incomplete if item.status != "not_applicable"]
                self.assertTrue(relevant)
                self.assertTrue(all(item.decision == "CONDITIONAL" for item in relevant))
                self.assertTrue(has_unresolved_condition(relevant, required_facts))
                facts.context_facts.extend(ContextFact(
                    f"{code}-{index}", predicate, True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ) for index, predicate in enumerate(required_facts))
                resolved = find_matching_rules(
                    facts, parsed.query_intent, parsed.context_facts, rules=reviewed
                )
                relevant = [item for item in resolved if item.status != "not_applicable"]
                self.assertTrue(all(item.decision == "CONDITIONAL" for item in relevant))
                if code in {"I.423a", "I.423b"}:
                    required_actions = {effect.action for item in relevant for effect in item.effects
                                        if effect.effect_type == "require_action"}
                    self.assertTrue({"reduce_speed", "observe", "yielding"}.issubset(required_actions))

    def test_i406_priority_is_conditional_and_never_becomes_blanket_permission(self) -> None:
        facts = _scoped_sign_facts("I.406")
        rules = [rule for rule in TRAFFIC_RULES if "I.406" in rule.trigger_sign_codes]
        self.assertEqual(len(rules), 2)
        parsed = query("Xe ô tô có được đi trước qua đường hẹp không?")

        unreviewed = evaluate(facts, parsed, rules)
        self.assertTrue(unreviewed)
        self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))
        self.assertTrue(all("legal_review_required" not in item.missing_facts for item in unreviewed))

        reviewed = reviewed_rules(rules)
        missing_occupancy = evaluate(facts, parsed, reviewed)
        priority_result = next(item for item in missing_occupancy if item.rule_id == "I406_MOTOR_VEHICLE_PRIORITY_NARROW_ROAD")
        self.assertEqual(priority_result.decision, "CONDITIONAL")
        self.assertNotIn("opposing_vehicle_already_entered_narrow_section", priority_result.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in priority_result.effects))

        facts.context_facts.extend([
            ContextFact(
                "i406-approach", "vehicle_on_i406_priority_approach", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "i406-occupancy", "opposing_vehicle_already_entered_narrow_section", True,
                status="provided", source="observed_road_scene", confidence="high",
            ),
        ])
        yield_query = query("Xe ô tô phải nhường đường không?")
        yield_result = evaluate(facts, yield_query, reviewed)
        required_yield = next(
            item for item in yield_result
            if item.rule_id == "I406_YIELD_WHEN_ONCOMING_VEHICLE_ALREADY_IN_NARROW_SECTION"
        )
        self.assertEqual(required_yield.decision, "CONDITIONAL")
        self.assertTrue(any(
            effect.effect_type == "require_action" and effect.action == "yielding"
            for effect in required_yield.effects
        ))
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in required_yield.effects))

    def test_i407_one_way_variants_require_explicit_arrow_direction_and_fail_closed(self) -> None:
        for code in ("I.407a", "I.407b", "I.407c"):
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                parsed = query("Xe tải có được đi ngược chiều đường một chiều không?")

                unreviewed = evaluate(facts, parsed, rules)
                self.assertTrue(unreviewed)
                self.assertTrue(all(item.verification_status == "unverified" for item in unreviewed))

                reviewed = reviewed_rules(rules)
                missing_direction = evaluate(facts, parsed, reviewed)
                unresolved = next(item for item in missing_direction if item.status != "not_applicable")
                self.assertEqual(unresolved.decision, "CONDITIONAL")
                self.assertIn("vehicle_travel_direction_matches_i407_arrow", unresolved.missing_facts)

                facts.context_facts.append(ContextFact(
                    f"{code}-opposite", "vehicle_travel_direction_matches_i407_arrow", False,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ))
                opposite_direction = evaluate(facts, parsed, reviewed)
                prohibited = next(item for item in opposite_direction if item.status != "not_applicable")
                self.assertEqual(prohibited.decision, "PROHIBITED")

                facts.context_facts[-1] = ContextFact(
                    f"{code}-matching", "vehicle_travel_direction_matches_i407_arrow", True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                )
                matching_direction = evaluate(facts, parsed, reviewed)
                conditional = next(item for item in matching_direction if item.status != "not_applicable")
                self.assertEqual(conditional.decision, "CONDITIONAL")
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in conditional.effects))

    def test_re9d_zone_speed_rule_needs_zone_membership_and_never_infers_allowed(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "zone-speed", role="main", sign_code="R.E.9d", identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True,
                "speed_limit_value": 50,
            }}, confidence="high",
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "complete scene includes the zone sign"
        }])
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE9D_MAXIMUM_SPEED_WITHIN_CONFIRMED_ZONE")
        reviewed = reviewed_rules([rule])
        parsed = query("Xe ô tô có được chạy ở tốc độ 60 km/h không?")

        missing_zone = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(missing_zone.decision, "PROHIBITED")
        self.assertNotIn("vehicle_within_re9d_zone_confirmed", missing_zone.missing_facts)

        facts.context_facts.append(ContextFact(
            "re9d-zone", "vehicle_within_re9d_zone_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        above_limit = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(above_limit.decision, "PROHIBITED")
        below_limit = evaluate(
            facts, query("Xe ô tô có được chạy ở tốc độ 40 km/h không?"), reviewed,
        )[0]
        self.assertEqual(below_limit.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in below_limit.effects))

    def test_re10d_only_ends_zone_limit_and_requires_other_limit_resolution(self) -> None:
        facts = _scoped_sign_facts("R.E.10d")
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE10D_END_OF_ZONE_MAXIMUM_SPEED_RESTRICTION")
        reviewed = reviewed_rules([rule])
        facts.context_facts.append(ContextFact(
            "re10d-exit", "vehicle_beyond_re10d_zone_exit_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        result = evaluate(facts, query("Xe ô tô có được chạy ở tốc độ 70 km/h không?"), reviewed)[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_speed_limit_resolved", result.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in result.effects))

    def test_re9a_area_wide_parking_ban_requires_zone_membership(self) -> None:
        facts = _scoped_sign_facts("R.E.9a")
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE9A_NO_PARKING_IN_ZONE")
        reviewed = reviewed_rules([rule])
        parsed = query("Xe ô tô có được đỗ xe ở đây không?")

        missing_zone = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(missing_zone.decision, "PROHIBITED")
        self.assertNotIn("vehicle_within_zone_confirmed", missing_zone.missing_facts)

        facts.context_facts.append(ContextFact(
            "re9a-zone", "vehicle_within_zone_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        in_zone = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(in_zone.decision, "PROHIBITED")

    def test_re9c_designates_parking_zone_but_never_establishes_permission(self) -> None:
        facts = _scoped_sign_facts("R.E.9c")
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE9C_PARKING_AREA_ZONE_DESIGNATION")
        reviewed = reviewed_rules([rule])
        parsed = query("Xe tải có được đỗ xe ở đây không?")

        no_zone_scope = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(no_zone_scope.decision, "CONDITIONAL")
        self.assertNotIn("vehicle_within_re9c_zone_confirmed", no_zone_scope.missing_facts)

        facts.context_facts.append(ContextFact(
            "re9c-zone", "vehicle_within_re9c_zone_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        unresolved_vehicle_conditions = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(unresolved_vehicle_conditions.decision, "CONDITIONAL")
        self.assertIn("vehicle_class_parking_conditions_resolved", unresolved_vehicle_conditions.missing_facts)

        facts.context_facts.extend([
            ContextFact(
                "re9c-class", "vehicle_class_parking_conditions_resolved", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "re9c-other", "other_local_parking_restrictions_resolved", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
        ])
        designation_only = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(designation_only.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in designation_only.effects))

    def test_re10c_ends_parking_zone_without_permitting_or_prohibiting_after_it(self) -> None:
        facts = _scoped_sign_facts("R.E.10c")
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE10C_END_OF_PARKING_AREA_ZONE")
        reviewed = reviewed_rules([rule])
        parsed = query("Xe máy có được đỗ xe sau biển này không?")

        before_exit_confirmed = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(before_exit_confirmed.decision, "CONDITIONAL")
        self.assertNotIn("vehicle_beyond_re10c_zone_exit_confirmed", before_exit_confirmed.missing_facts)

        facts.context_facts.extend([
            ContextFact(
                "re10c-exit", "vehicle_beyond_re10c_zone_exit_confirmed", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "re10c-other", "other_local_parking_restrictions_resolved", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
        ])
        result = evaluate(facts, parsed, reviewed)[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in result.effects))

    def test_re11_tunnel_signs_require_tunnel_scope_and_never_grant_blanket_permission(self) -> None:
        entry_facts = _scoped_sign_facts("R.E.11a")
        entry_rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE11A_TUNNEL_SPECIAL_REGULATIONS_APPLY")
        entry_reviewed = reviewed_rules([entry_rule])
        parsed = query("Xe buýt có được đi vào đường hầm này không?")

        missing_tunnel_scope = evaluate(entry_facts, parsed, entry_reviewed)[0]
        self.assertEqual(missing_tunnel_scope.decision, "CONDITIONAL")
        self.assertNotIn("vehicle_on_tunnel_segment_confirmed", missing_tunnel_scope.missing_facts)

        entry_facts.context_facts.extend([
            ContextFact(
                "tunnel-presence", "vehicle_on_tunnel_segment_confirmed", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "tunnel-controls", "tunnel_specific_rules_resolved", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
        ])
        entry = evaluate(entry_facts, parsed, entry_reviewed)[0]
        self.assertEqual(entry.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in entry.effects))

        exit_facts = _scoped_sign_facts("R.E.11b")
        exit_rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE11B_END_TUNNEL_SPECIAL_REGULATIONS")
        exit_reviewed = reviewed_rules([exit_rule])
        exit_parsed = query("Xe buýt có được đi sau biển này không?")
        missing_exit = evaluate(exit_facts, exit_parsed, exit_reviewed)[0]
        self.assertEqual(missing_exit.decision, "CONDITIONAL")
        self.assertNotIn("vehicle_beyond_tunnel_exit_confirmed", missing_exit.missing_facts)

        exit_facts.context_facts.extend([
            ContextFact(
                "tunnel-exit", "vehicle_beyond_tunnel_exit_confirmed", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "road-controls", "other_applicable_road_rules_resolved", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
        ])
        after_exit = evaluate(exit_facts, exit_parsed, exit_reviewed)[0]
        self.assertEqual(after_exit.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in after_exit.effects))

    def test_w239a_compares_explicit_vehicle_height_only_with_linked_s509a_safe_height(self) -> None:
        main = SignObservation(
            "powerline-warning", role="main", sign_code="W.239a", identity_status="identified",
            confidence="high", visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True,
            }},
        )
        panel = SignObservation(
            "safe-height-panel", role="supplementary", sign_code="S.509a",
            identity_status="identified", confidence="high",
            ocr_texts=[{"raw_text": "4,2 m", "confidence": "high"}],
        )
        facts = TrafficFacts(
            "2.0", "scene-safe-height", [main, panel],
            sign_relations=[SignRelation(
                "safe-height-for-warning", "supplements", "safe-height-panel", "powerline-warning", "high",
            )],
            context_facts=[ContextFact(
                "route-crosses-powerline", "vehicle_route_crosses_powerline_location_confirmed", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            )],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack and route"}],
        )
        comparison_rule = next(
            rule for rule in TRAFFIC_RULES
            if rule.rule_id == "W239A_COMPARE_VEHICLE_HEIGHT_WITH_SAFE_HEIGHT_PANEL"
        )
        warning_rule = next(
            rule for rule in TRAFFIC_RULES if rule.rule_id == "W239A_COMMON_WARNING_SIGN_DUTY"
        )
        rules = reviewed_rules([comparison_rule, warning_rule])

        taller_query = query("Xe tải cao 4,35 m có được lưu thông qua biển này không?")
        taller = evaluate(facts, taller_query, rules)
        height_result = next(item for item in taller if item.rule_id == comparison_rule.rule_id)
        self.assertEqual(height_result.decision, "CONDITIONAL")
        self.assertEqual(height_result.verification_status, "verified")
        self.assertTrue(any(
            item.get("predicate") == "vehicle_height_m"
            and item.get("state") == "satisfied"
            and item.get("value") == 4.35
            for item in height_result.trace
        ))
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in height_result.effects))

        shorter_query = query("Xe tải cao 4,1 m có được lưu thông qua biển này không?")
        shorter = evaluate(facts, shorter_query, rules)
        height_result = next(item for item in shorter if item.rule_id == comparison_rule.rule_id)
        self.assertEqual(height_result.status, "not_applicable")
        self.assertTrue(all(item.decision not in {"ALLOWED", "PROHIBITED"} for item in shorter))

        missing_height_query = query("Xe tải có được lưu thông qua biển này không?")
        missing_height = evaluate(facts, missing_height_query, rules)
        height_result = next(item for item in missing_height if item.rule_id == comparison_rule.rule_id)
        self.assertEqual(height_result.decision, "CONDITIONAL")
        self.assertIn("vehicle_height_m", height_result.missing_facts)

        # A sign that reports actual physical clearance (W.239b) must not consume
        # the electrical safe-height value as though it were the same quantity.
        physical_clearance_facts = TrafficFacts(
            "2.0", "scene-physical-clearance", [
                SignObservation("physical-warning", role="main", sign_code="W.239b", identity_status="identified"),
                panel,
            ],
            sign_relations=[SignRelation(
                "wrong-height-concept", "supplements", "safe-height-panel", "physical-warning", "high",
            )],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "separate W.239b sign"}],
        )
        physical = evaluate(physical_clearance_facts, taller_query, rules)
        self.assertFalse(any(item.rule_id == comparison_rule.rule_id for item in physical))

    def test_re9b_time_restriction_uses_current_system_time_and_never_allows(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "zone-time", role="main", sign_code="R.E.9b", identity_status="identified",
            ocr_texts=[{"raw_text": "6:00 - 18:00", "confidence": "high"}],
            visual_attributes={"normalized_conditions": {"sign_effective_scope_confirmed": True}},
            confidence="high",
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "complete scene includes zone time sign"
        }])
        facts.context_facts.append(ContextFact(
            "re9b-zone", "vehicle_within_zone_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE9B_NO_PARKING_DURING_DISPLAYED_TIME_IN_ZONE")
        reviewed = reviewed_rules([rule])
        active = evaluate(facts, query("Xe ô tô có được đỗ xe lúc này không?"), reviewed)[0]
        self.assertEqual(active.decision, "PROHIBITED")

        outside = evaluate(
            facts,
            query("Xe ô tô có được đỗ xe lúc này không?", datetime(2026, 10, 4, 20, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))),
            reviewed,
        )[0]
        self.assertEqual(outside.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in outside.effects))

    def test_re10a_ends_only_zone_ban_and_does_not_permit_parking(self) -> None:
        facts = _scoped_sign_facts("R.E.10a")
        facts.context_facts.append(ContextFact(
            "re10a-exit", "vehicle_beyond_zone_exit_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "RE10A_END_OF_NO_PARKING_ZONE")
        result = evaluate(facts, query("Xe ô tô có được đỗ xe không?"), reviewed_rules([rule]))[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_parking_restrictions_resolved", result.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in result.effects))

    def test_dp127_variants_end_only_combined_limits_and_require_statutory_speed_fact(self) -> None:
        for code, rule_id in (
            ("DP.127a", "DP127A_END_COMBINED_SPEED_LIMIT"),
            ("DP.127b", "DP127B_END_COMBINED_SPEED_LIMIT"),
            ("DP.127c", "DP127C_END_COMBINED_SPEED_LIMIT"),
        ):
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                facts.context_facts.append(ContextFact(
                    f"{code}-past", "vehicle_beyond_dp127_sign_confirmed", True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ))
                rules = [rule for rule in TRAFFIC_RULES if rule.rule_id == rule_id]
                parsed = query("Xe ô tô có được chạy ở tốc độ 70 km/h không?")
                unreviewed = evaluate(facts, parsed, rules)[0]
                self.assertEqual(unreviewed.verification_status, "unverified")

                reviewed = reviewed_rules(rules)
                no_statutory_limit = evaluate(facts, parsed, reviewed)[0]
                self.assertEqual(no_statutory_limit.decision, "CONDITIONAL")
                self.assertIn("applicable_statutory_max_speed_kmh", no_statutory_limit.missing_facts)

                facts.context_facts.extend([
                    ContextFact(
                        f"{code}-statutory-limit", "applicable_statutory_max_speed_kmh", 60,
                        status="provided", source="authoritative_road_law_pipeline", confidence="high",
                    ),
                    ContextFact(
                        f"{code}-other-limits", "other_applicable_speed_limits_resolved", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                ])
                too_fast = evaluate(facts, parsed, reviewed)[0]
                self.assertEqual(too_fast.decision, "PROHIBITED")
                at_limit = evaluate(
                    facts, query("Xe ô tô có được chạy ở tốc độ 50 km/h không?"), reviewed,
                )[0]
                self.assertEqual(at_limit.decision, "CONDITIONAL")
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in at_limit.effects))

    def test_r122_s506b_yield_rule_requires_identified_and_linked_supplementary_sign(self) -> None:
        facts = _scoped_sign_facts("R.122")
        panel = SignObservation(
            observation_id="panel-s506b", role="supplementary", sign_code="S.506b",
            identity_status="identified", confidence="high",
        )
        facts.sign_observations.append(panel)
        facts.sign_relations.append(SignRelation(
            relation_id="s506b-under-r122", relation_type="supplements",
            source_observation_id=panel.observation_id,
            target_observation_id=facts.sign_observations[0].observation_id,
            confidence="high",
        ))
        facts.context_facts.extend([
            ContextFact(
                "r122-stop-confirmed", "vehicle_stopped_before_r122", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "actor-not-priority", "actor_has_priority_right", False,
                status="provided", source="user_or_scene_facts", confidence="high",
            ),
            ContextFact(
                "priority-traffic-present", "priority_road_traffic_requires_yield", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
        ])
        rule = next(
            item for item in TRAFFIC_RULES
            if item.rule_id == "R122_S506B_YIELD_TO_PRIORITY_ROAD_TRAFFIC"
        )
        reviewed = reviewed_rules([rule])
        result = evaluate(facts, query("Xe ô tô có được đi tiếp không?"), reviewed)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].decision, "CONDITIONAL")
        linked_trace = next(item for item in result[0].trace if item.get("predicate") == "linked_s506b_panel_present")
        self.assertEqual(linked_trace["value"], True)
        self.assertTrue(any(
            effect.effect_type == "require_action" and effect.action == "yielding"
            for effect in result[0].effects
        ))

        facts.sign_relations.clear()
        unresolved = evaluate(facts, query("Xe ô tô có được đi tiếp không?"), reviewed)
        self.assertEqual(unresolved[0].decision, "CONDITIONAL")
        self.assertIn("linked_s506b_panel_present", unresolved[0].missing_facts)

    def test_r403_vehicle_classes_follow_official_variant_scope_and_fail_closed(self) -> None:
        cases = [
            ("R.403c", "Xe buýt có được đi vào đoạn đường này không?", "bus", "CONDITIONAL"),
            ("R.403c", "Xe tải có được đi vào đoạn đường này không?", "truck", "PROHIBITED"),
            ("R.403f", "Xe đạp điện có được đi vào đoạn đường này không?", "non_motor_vehicle", "CONDITIONAL"),
            ("R.403g", "Xe chở hàng bốn bánh gắn động cơ có được đi vào không?", "truck", "CONDITIONAL"),
            ("R.403h", "Xe chở hàng bốn bánh gắn động cơ có được đi vào không?", "truck", "PROHIBITED"),
        ]
        for code, text, expected_actor, expected_decision in cases:
            with self.subTest(code=code, text=text):
                parsed = query(text)
                self.assertEqual(parsed.query_intent.intents[0].actor.value, expected_actor)
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]

                unreviewed = evaluate(facts, parsed, rules)
                matching_unreviewed = next(
                    item for item in unreviewed
                    if item.status != "not_applicable"
                )
                self.assertEqual(matching_unreviewed.verification_status, "unverified")
                self.assertNotIn("legal_review_required", matching_unreviewed.missing_facts)

                reviewed = evaluate(facts, parsed, reviewed_rules(rules))
                applicable = next(item for item in reviewed if item.status != "not_applicable")
                self.assertEqual(applicable.decision, expected_decision)
                self.assertNotEqual(applicable.decision, "ALLOWED")

        # A generic actor is not treated as an unlisted vehicle class.
        generic = query("Phương tiện có được đi vào đoạn đường này không?")
        generic_result = evaluate(
            _scoped_sign_facts("R.403c"), generic,
            reviewed_rules([rule for rule in TRAFFIC_RULES if "R.403c" in rule.trigger_sign_codes]),
        )
        self.assertTrue(any(
            item.decision == "CONDITIONAL" and "actor_vehicle_category" in item.missing_facts
            for item in generic_result
        ))

    def test_r404_variants_end_only_their_designated_road_scope(self) -> None:
        cases = [
            ("R.404a", "Xe tải có được tiếp tục đi qua không?", "truck"),
            ("R.404b", "Xe máy có được tiếp tục đi qua không?", "motorcycle_or_moped"),
            ("R.404c", "Xe buýt có được tiếp tục đi qua không?", "bus"),
            ("R.404d", "Ô tô con có được tiếp tục đi qua không?", "passenger_car"),
            ("R.404e", "Xe máy có được tiếp tục đi qua không?", "motorcycle_or_moped"),
            ("R.404f", "Xe đạp có được tiếp tục đi qua không?", "non_motor_vehicle"),
            ("R.404g", "Xe chở hàng bốn bánh gắn động cơ có được tiếp tục đi qua không?", "truck"),
            ("R.404h", "Xe chở người bốn bánh gắn động cơ có được tiếp tục đi qua không?", "passenger_car"),
            ("R.404k", "Xe chở hàng bốn bánh gắn động cơ có được tiếp tục đi qua không?", "truck"),
        ]
        for code, text, expected_actor in cases:
            with self.subTest(code=code):
                parsed = query(text)
                self.assertEqual(parsed.query_intent.intents[0].actor.value, expected_actor)
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                unreviewed = evaluate(facts, parsed, rules)
                self.assertTrue(unreviewed)
                self.assertEqual(unreviewed[0].verification_status, "unverified")
                self.assertNotIn("legal_review_required", unreviewed[0].missing_facts)

                reviewed = reviewed_rules(rules)
                unresolved_position = evaluate(facts, parsed, reviewed)
                self.assertEqual(unresolved_position[0].decision, "CONDITIONAL")
                self.assertNotIn("vehicle_beyond_r404_end_point", unresolved_position[0].missing_facts)

                facts.context_facts.append(ContextFact(
                    f"{code}-after-point", "vehicle_beyond_r404_end_point", True,
                    status="provided", source="reviewed_scene_assessment", confidence="high",
                ))
                located = evaluate(facts, parsed, reviewed)
                self.assertEqual(located[0].decision, "CONDITIONAL")
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in located[0].effects))
                if code in {"R.404b", "R.404e"}:
                    self.assertIn("motorcycle_or_moped_subtype", located[0].missing_facts)
                else:
                    self.assertNotIn("other_applicable_sign_constraints", located[0].missing_facts)

    def test_r308_variants_use_official_figure_turn_mapping_without_inventing_prohibition(self) -> None:
        cases = [
            ("R.308a", "Xe ô tô có được đi thẳng không?", "going_straight", "turn_left"),
            ("R.308a", "Xe ô tô có được rẽ trái không?", "turn_left", "turn_left"),
            ("R.308a", "Xe ô tô có được rẽ phải không?", "turn_right", "turn_left"),
            ("R.308b", "Xe ô tô có được đi thẳng không?", "going_straight", "turn_right"),
            ("R.308b", "Xe ô tô có được rẽ phải không?", "turn_right", "turn_right"),
            ("R.308b", "Xe ô tô có được rẽ trái không?", "turn_left", "turn_right"),
        ]
        for code, text, expected_action, illustrated_turn in cases:
            with self.subTest(code=code, text=text):
                parsed = query(text)
                self.assertEqual(parsed.query_intent.intents[0].action.value, expected_action)
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                result = evaluate(facts, parsed, reviewed_rules(rules))
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].decision, "CONDITIONAL")
                self.assertNotEqual(result[0].decision, "ALLOWED")
                self.assertFalse(any(effect.effect_type == "prohibit_action" for effect in result[0].effects))
                self.assertTrue(any(
                    effect.effect_type == "sign_meaning"
                    and effect.scope.get("illustrated_turn") == illustrated_turn
                    for effect in result[0].effects
                ))

                unreviewed = evaluate(facts, parsed, rules)
                self.assertEqual(unreviewed[0].verification_status, "unverified")
                self.assertNotIn("legal_review_required", unreviewed[0].missing_facts)

    def test_p132_applies_stop_and_yield_to_motorized_and_non_motorized_vehicles(self) -> None:
        cases = [
            ("Xe ô tô có được đi qua đoạn đường hẹp này không khi có xe ngược chiều?", "automobile"),
            ("Xe đạp có phải dừng khi có xe ngược chiều không?", "non_motor_vehicle"),
            ("Xe ưu tiên có phải dừng khi có xe ngược chiều không?", "other_motor_vehicle"),
        ]
        for text, expected_actor in cases:
            with self.subTest(actor=expected_actor):
                parsed = query(text)
                self.assertEqual(parsed.query_intent.intents[0].actor.value, expected_actor)
                facts = _scoped_sign_facts("P.132")
                rules = [rule for rule in TRAFFIC_RULES if "P.132" in rule.trigger_sign_codes]

                unreviewed = evaluate(facts, parsed, rules)
                self.assertEqual(unreviewed[0].verification_status, "unverified")
                self.assertNotIn("legal_review_required", unreviewed[0].missing_facts)

                reviewed = reviewed_rules(rules)
                missing_traffic = evaluate(facts, parsed, reviewed)
                self.assertEqual(missing_traffic[0].decision, "CONDITIONAL")
                self.assertNotIn("oncoming_motorized_traffic_present", missing_traffic[0].missing_facts)
                self.assertTrue(missing_traffic[0].effects)
                self.assertFalse(any(
                    effect.effect_type == "permit_action" for effect in missing_traffic[0].effects
                ))

                facts.context_facts.extend([
                    ContextFact(
                        "p132-approach", "vehicle_at_or_approaching_p132_narrow_section", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                    ContextFact(
                        "p132-oncoming-traffic", "oncoming_motorized_traffic_present", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                ])
                active = evaluate(facts, parsed, reviewed)
                self.assertEqual(active[0].decision, "CONDITIONAL")
                self.assertTrue(any(
                    effect.effect_type == "require_action" and effect.action == "stopping"
                    for effect in active[0].effects
                ))
                self.assertTrue(any(
                    effect.effect_type == "require_action" and effect.action == "yielding"
                    for effect in active[0].effects
                ))
                self.assertNotEqual(active[0].decision, "ALLOWED")

    def test_article_27_warning_duty_covers_every_w_code_without_granting_permission(self) -> None:
        warning_rules = [rule for rule in TRAFFIC_RULES if "COMMON_WARNING_SIGN_DUTY" in rule.rule_id]
        self.assertEqual(len(warning_rules), 96)
        warning_codes = [code for item in warning_rules for code in item.trigger_sign_codes]
        self.assertEqual(len(warning_codes), len(set(warning_codes)))
        rule = next(item for item in TRAFFIC_RULES if item.rule_id == "W212_COMMON_WARNING_SIGN_DUTY")
        self.assertTrue(all(code.startswith("W.") for code in rule.trigger_sign_codes))
        self.assertTrue(all(
            any(basis.get("section_or_figure", "").startswith("Chapter 5, Article 27") for basis in item.legal_basis)
            for item in [rule]
        ))

        facts = _scoped_sign_facts("W.212")
        meaning_query = query("Biển W.212 có ý nghĩa gì?")
        self.assertEqual(meaning_query.query_intent.intents[0].kind, "INFORMATION")
        self.assertEqual(meaning_query.query_intent.intents[0].information_type, "sign_meaning")
        self.assertEqual(evaluate(facts, meaning_query, [rule]), [])

        reviewed = reviewed_rules([rule])
        self.assertEqual(evaluate(facts, meaning_query, reviewed), [])

        permission = evaluate(
            facts, query("Xe ô tô có được vượt xe khác không?"), reviewed
        )
        self.assertEqual(permission[0].decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in permission[0].effects))

    def test_w208_yield_rule_resolves_intersection_and_optional_s506b_scope(self) -> None:
        text = "Xe ô tô có phải nhường đường cho xe trên đường ưu tiên không?"
        parsed = query(text)
        self.assertEqual(parsed.query_intent.intents[0].action.value, "yielding")
        rules = [
            rule for rule in TRAFFIC_RULES
            if "W.208" in rule.trigger_sign_codes and "COMMON_WARNING_SIGN_DUTY" not in rule.rule_id
        ]
        facts = _scoped_sign_facts("W.208")

        unreviewed = evaluate(facts, parsed, rules)
        self.assertEqual(unreviewed[0].verification_status, "unverified")
        self.assertNotIn("legal_review_required", unreviewed[0].missing_facts)

        reviewed = reviewed_rules(rules)
        unresolved = evaluate(facts, parsed, reviewed)
        self.assertEqual(unresolved[0].decision, "CONDITIONAL")
        self.assertNotIn("vehicle_approaching_w208_intersection", unresolved[0].missing_facts)
        self.assertNotIn("priority_road_traffic_requires_yield", unresolved[0].missing_facts)
        self.assertTrue(any(
            item.predicate == "priority_road_scope_modifier_resolved" and item.value is True
            for item in facts.context_facts
        ) or any(
            item.get("predicate") == "priority_road_scope_modifier_resolved"
            for item in unresolved[0].trace
        ))

        facts.context_facts.extend([
            ContextFact(
                "w208-approach", "vehicle_approaching_w208_intersection", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "w208-no-control", "intersection_traffic_control_active", False,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "w208-conflict", "priority_road_traffic_requires_yield", True,
                status="provided", source="reviewed_scene_assessment", confidence="high",
            ),
            ContextFact(
                "w208-not-priority", "actor_has_priority_right", False,
                status="provided", source="user_or_scene_facts", confidence="high",
            ),
        ])
        applicable = evaluate(facts, parsed, reviewed)
        self.assertEqual(applicable[0].decision, "CONDITIONAL")
        self.assertTrue(any(
            effect.effect_type == "require_action" and effect.action == "yielding"
            for effect in applicable[0].effects
        ))
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in applicable[0].effects))

        priority_query = query("Xe ưu tiên có phải nhường đường cho xe trên đường ưu tiên không?")
        priority_result = evaluate(facts, priority_query, reviewed)
        self.assertEqual(priority_result[0].status, "not_applicable")
        self.assertFalse(any(effect.action == "yielding" for effect in priority_result[0].effects))

        # If S.506b is linked below W.208, an explicit assessment must resolve
        # whether the priority-road traffic follows the indicated bend.
        linked_facts = _scoped_sign_facts("W.208")
        panel = SignObservation(
            observation_id="w208-s506b", role="supplementary", sign_code="S.506b",
            identity_status="identified", confidence="high",
        )
        linked_facts.sign_observations.append(panel)
        linked_facts.sign_relations.append(SignRelation(
            relation_id="s506b-under-w208", relation_type="supplements",
            source_observation_id=panel.observation_id,
            target_observation_id=linked_facts.sign_observations[0].observation_id,
            confidence="high",
        ))
        linked_facts.context_facts.extend(facts.context_facts)
        scoped = evaluate(linked_facts, parsed, reviewed)
        self.assertEqual(scoped[0].decision, "CONDITIONAL")
        self.assertIn("priority_road_scope_modifier_resolved", scoped[0].missing_facts)

        linked_facts.context_facts.append(ContextFact(
            "w208-s506b-scope-reviewed", "priority_road_scope_modifier_resolved", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        scoped = evaluate(linked_facts, parsed, reviewed)
        self.assertEqual(scoped[0].decision, "CONDITIONAL")
        self.assertTrue(any(effect.action == "yielding" for effect in scoped[0].effects))

    def test_w207_variants_require_matching_priority_road_approach_and_scope(self) -> None:
        variants = ("W.207a", "W.207b", "W.207c", "W.207d", "W.207e", "W.207f",
                    "W.207g", "W.207h", "W.207i", "W.207k", "W.207l")
        text = "Xe ô tô có được đi qua giao lộ này trước xe từ đường nhánh không?"
        parsed = query(text)
        self.assertEqual(parsed.query_intent.intents[0].action.value, "circulation")
        for code in variants:
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [
                    rule for rule in TRAFFIC_RULES
                    if code in rule.trigger_sign_codes and "COMMON_WARNING_SIGN_DUTY" not in rule.rule_id
                ]
                self.assertEqual(len(rules), 1)
                unreviewed = evaluate(facts, parsed, rules)
                self.assertEqual(unreviewed[0].verification_status, "unverified")
                self.assertNotIn("legal_review_required", unreviewed[0].missing_facts)

                facts.context_facts.extend([
                    ContextFact(
                        f"{code}-approach", "vehicle_on_w207_priority_road_approach", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                    ContextFact(
                        f"{code}-geometry", "w207_intersection_geometry_matches_sign", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                    ContextFact(
                        f"{code}-uncontrolled", "intersection_traffic_control_active", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                    ContextFact(
                        f"{code}-actor-not-priority", "actor_has_priority_right", False,
                        status="provided", source="user_or_scene_facts", confidence="high",
                    ),
                    ContextFact(
                        f"{code}-no-priority-conflict", "priority_vehicle_conflict_present", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    ),
                ])
                if code in {"W.207d", "W.207e"}:
                    facts.context_facts.append(ContextFact(
                        f"{code}-range", "w207_multiple_intersection_scope_resolved", True,
                        status="provided", source="reviewed_supplementary_scope", confidence="high",
                    ))
                reviewed = reviewed_rules(rules)
                result = evaluate(facts, parsed, reviewed)
                self.assertEqual(result[0].decision, "CONDITIONAL")
                self.assertTrue(any(
                    effect.effect_type == "sign_meaning"
                    and effect.scope.get("priority_road_right_of_way") is True
                    for effect in result[0].effects
                ))
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in result[0].effects))

        # The multi-intersection variants d/e are not applied without their
        # explicitly resolved supplementary range.
        facts = _scoped_sign_facts("W.207d")
        facts.context_facts.extend([
            ContextFact("approach", "vehicle_on_w207_priority_road_approach", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("geometry", "w207_intersection_geometry_matches_sign", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("uncontrolled", "intersection_traffic_control_active", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("actor-not-priority", "actor_has_priority_right", False,
                        status="provided", source="user_or_scene_facts", confidence="high"),
            ContextFact("no-priority-conflict", "priority_vehicle_conflict_present", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
        ])
        scoped = evaluate(
            facts, parsed,
            reviewed_rules([
                rule for rule in TRAFFIC_RULES
                if "W.207d" in rule.trigger_sign_codes and "COMMON_WARNING_SIGN_DUTY" not in rule.rule_id
            ]),
        )
        self.assertEqual(scoped[0].decision, "CONDITIONAL")
        self.assertNotIn("w207_multiple_intersection_scope_resolved", scoped[0].missing_facts)

        rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES
            if "W.207a" in rule.trigger_sign_codes and "COMMON_WARNING_SIGN_DUTY" not in rule.rule_id
        ])
        priority_actor = query("Xe ưu tiên có được đi qua giao lộ này trước xe từ đường nhánh không?")
        priority_facts = _scoped_sign_facts("W.207a")
        priority_facts.context_facts.extend([
            ContextFact("approach", "vehicle_on_w207_priority_road_approach", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("geometry", "w207_intersection_geometry_matches_sign", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("uncontrolled", "intersection_traffic_control_active", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("no-priority-conflict", "priority_vehicle_conflict_present", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
        ])
        excepted = evaluate(priority_facts, priority_actor, rules)
        self.assertEqual(excepted[0].status, "not_applicable")
        self.assertFalse(any(effect.effect_type == "sign_meaning" for effect in excepted[0].effects))

        regular_actor = query("Xe ô tô có được đi qua giao lộ này trước xe từ đường nhánh không?")
        conflict_facts = _scoped_sign_facts("W.207a")
        conflict_facts.context_facts.extend([
            ContextFact("approach", "vehicle_on_w207_priority_road_approach", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("geometry", "w207_intersection_geometry_matches_sign", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("uncontrolled", "intersection_traffic_control_active", False,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
            ContextFact("actor-not-priority", "actor_has_priority_right", False,
                        status="provided", source="user_or_scene_facts", confidence="high"),
            ContextFact("priority-conflict", "priority_vehicle_conflict_present", True,
                        status="provided", source="reviewed_scene_assessment", confidence="high"),
        ])
        excepted = evaluate(conflict_facts, regular_actor, rules)
        self.assertEqual(excepted[0].status, "not_applicable")

    def test_i401_i402_priority_segment_rules_remain_conditional(self) -> None:
        for code, rule_id, question_text, required_facts in (
            (
                "I.401", "I401_PRIORITY_ROAD_UNCONTROLLED_INTERSECTION_MEANING",
                "Xe ô tô có được đi thẳng qua giao lộ này không?",
                {
                    "vehicle_on_i401_priority_road_segment": True,
                    "i401_priority_road_direction_resolved": True,
                    "intersection_traffic_control_active": False,
                    "priority_vehicle_conflict_present": False,
                    "actor_has_priority_right": False,
                },
            ),
            (
                "I.402", "I402_END_OF_PRIORITY_ROAD_SEGMENT_MEANING",
                "Xe ô tô có được đi qua giao lộ này không?",
                {
                    "vehicle_beyond_i402_end_point": True,
                    "other_applicable_signs_and_controls_resolved": True,
                },
            ),
        ):
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                self.assertEqual(len(rules), 1)
                self.assertEqual(rules[0].rule_id, rule_id)
                unreviewed = evaluate(facts, query(question_text), rules)
                self.assertEqual(unreviewed[0].verification_status, "unverified")
                self.assertNotIn("legal_review_required", unreviewed[0].missing_facts)

                reviewed = reviewed_rules(rules)
                partial = evaluate(facts, query(question_text), reviewed)
                self.assertEqual(partial[0].decision, "CONDITIONAL")
                self.assertTrue(has_unresolved_condition(partial, required_facts))

                facts.context_facts.extend(
                    ContextFact(
                        f"{code}-{predicate}", predicate, value,
                        status="provided", source="reviewed_scene_assessment", confidence="high",
                    )
                    for predicate, value in required_facts.items()
                )
                complete = evaluate(facts, query(question_text), reviewed)
                self.assertEqual(complete[0].decision, "CONDITIONAL")
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in complete[0].effects))

    def test_p131_no_parking_variants_use_scope_time_and_priority_exception(self) -> None:
        odd_day = datetime(2026, 10, 5, 12, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        even_day = datetime(2026, 10, 4, 12, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        for code in ("P.131a", "P.131b", "P.131c"):
            with self.subTest(code=code):
                facts = _scoped_sign_facts(code)
                active_day_query = query(
                    "Xe ô tô có được đỗ xe ở đây không?",
                    now=even_day if code == "P.131c" else odd_day,
                )
                rules = [rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes]
                self.assertEqual(len(rules), 2)
                unreviewed = evaluate(facts, active_day_query, rules)
                unreviewed_no_panel = next(item for item in unreviewed if not item.rule_id.endswith("_DURING_S508"))
                self.assertEqual(unreviewed_no_panel.decision, "PROHIBITED")
                self.assertNotIn("legal_review_required", unreviewed_no_panel.missing_facts)

                reviewed = reviewed_rules(rules)
                result = evaluate(facts, active_day_query, reviewed)
                applicable = next(item for item in result if not item.rule_id.endswith("_DURING_S508"))
                self.assertEqual(applicable.decision, "PROHIBITED")
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in applicable.effects))

                priority_query = query(
                    "Xe ưu tiên có được đỗ xe ở đây không?",
                    now=even_day if code == "P.131c" else odd_day,
                )
                priority_result = evaluate(facts, priority_query, reviewed)
                self.assertFalse(any(item.decision == "PROHIBITED" for item in priority_result))

                # A linked but unreadable S.508 panel is not treated as absent or inactive.
                scoped_panel_facts = _scoped_sign_facts(code)
                panel = SignObservation(
                    "p131-s508", role="supplementary", sign_code="S.508a",
                    identity_status="identified", confidence="high", ocr_texts=[],
                )
                scoped_panel_facts.sign_observations.append(panel)
                scoped_panel_facts.sign_relations.append(SignRelation(
                    relation_id="s508-under-p131", relation_type="supplements",
                    source_observation_id=panel.observation_id,
                    target_observation_id=scoped_panel_facts.sign_observations[0].observation_id,
                    confidence="high",
                ))
                panel_result = evaluate(scoped_panel_facts, active_day_query, reviewed)
                with_panel = next(item for item in panel_result if item.rule_id.endswith("_DURING_S508"))
                self.assertEqual(with_panel.decision, "CONDITIONAL")
                self.assertIn("linked_s508_temporal_scope_active", with_panel.missing_facts)

        even_facts = _scoped_sign_facts("P.131b")
        even_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if "P.131b" in rule.trigger_sign_codes
        ])
        even_result = evaluate(even_facts, query("Xe ô tô có được đỗ xe ở đây không?", now=even_day), even_rules)
        self.assertTrue(all(item.status == "not_applicable" for item in even_result))

    def test_p140_requires_explicit_cong_nong_identity_and_official_scope(self) -> None:
        exact_query = query("Xe công nông có được đi vào đoạn đường này không?")
        self.assertEqual(exact_query.query_intent.intents[0].actor.value, "other_motor_vehicle")
        rule = next(item for item in TRAFFIC_RULES if item.rule_id == "P140_PROHIBIT_CONG_NONG_VEHICLES")
        facts = _scoped_sign_facts("P.140")

        draft_result = evaluate(facts, exact_query, [rule])
        self.assertEqual(draft_result[0].verification_status, "unverified")
        self.assertNotIn("legal_review_required", draft_result[0].missing_facts)

        reviewed = evaluate(facts, exact_query, reviewed_rules([rule]))
        self.assertEqual(reviewed[0].decision, "PROHIBITED")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in reviewed[0].effects))

        missing_scope = evaluate(
            facts_for({"sign_code": "P.140"}), exact_query, reviewed_rules([rule])
        )
        self.assertEqual(missing_scope[0].decision, "PROHIBITED")
        self.assertNotIn("sign_effective_scope_confirmed", missing_scope[0].missing_facts)

        similar_query = query("Xe tương tự xe công nông có được đi vào không?")
        self.assertEqual(similar_query.query_intent.intents[0].actor.value, "unknown")
        unresolved = evaluate(facts, similar_query, reviewed_rules([rule]))
        self.assertTrue(unresolved)
        self.assertEqual(unresolved[0].decision, "UNKNOWN")
        self.assertIn("actor_class", unresolved[0].missing_facts)
        listed_similar_query = query("Xe công nông hoặc xe tương tự có được đi vào không?")
        self.assertEqual(listed_similar_query.query_intent.intents[0].actor.value, "unknown")

        tractor_query = query("Máy kéo có được đi vào đoạn đường này không?")
        self.assertEqual(tractor_query.query_intent.intents[0].actor.value, "other_motor_vehicle")
        tractor_result = evaluate(facts, tractor_query, reviewed_rules([rule]))
        self.assertFalse(any(item.decision == "PROHIBITED" for item in tractor_result))

    def test_r411_requires_lane_mapping_and_never_treats_a_matching_arrow_as_permission(self) -> None:
        parsed = query("Xe ô tô có được rẽ trái từ làn này không?")
        self.assertEqual(parsed.query_intent.intents[0].action.value, "turn_left")
        facts = _scoped_sign_facts("R.411")
        rule = next(item for item in TRAFFIC_RULES if item.rule_id == "R411_FOLLOW_DIRECTION_FOR_CURRENT_LANE")

        draft = evaluate(facts, parsed, [rule])
        self.assertEqual(draft[0].verification_status, "unverified")
        self.assertNotIn("legal_review_required", draft[0].missing_facts)

        reviewed_rule = reviewed_rules([rule])
        unresolved = evaluate(facts, parsed, reviewed_rule)
        self.assertEqual(unresolved[0].decision, "CONDITIONAL")
        self.assertNotIn("r411_current_lane_resolved", unresolved[0].missing_facts)

        for route_matches, expected in ((True, "CONDITIONAL"), (False, "PROHIBITED")):
            with self.subTest(route_matches=route_matches):
                mapped_facts = _scoped_sign_facts("R.411")
                mapped_facts.context_facts.extend([
                    ContextFact("r411-current-lane", "r411_current_lane_resolved", True,
                                status="provided", source="reviewed_scene_assessment", confidence="high"),
                    ContextFact("r411-arrows", "r411_lane_arrow_mapping_resolved", True,
                                status="provided", source="reviewed_sign_and_marking_read", confidence="high"),
                    ContextFact("r411-road-marking", "r411_directional_road_marking_consistent", True,
                                status="provided", source="reviewed_scene_assessment", confidence="high"),
                    ContextFact("r411-route-match", "r411_requested_route_matches_lane_arrow", route_matches,
                                status="provided", source="reviewed_intent_lane_assessment", confidence="high"),
                ])
                result = evaluate(mapped_facts, parsed, reviewed_rule)
                self.assertEqual(result[0].decision, expected)
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in result[0].effects))
                if route_matches:
                    self.assertTrue(any(effect.effect_type == "sign_meaning" for effect in result[0].effects))

    def test_r302_requires_an_explicit_route_side_and_never_grants_permission(self) -> None:
        def run(code: str, question_text: str):
            facts = _scoped_sign_facts(code)
            parsed = query(question_text)
            selected = reviewed_rules([rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes])
            return evaluate(facts, parsed, selected)

        right_only = run("R.302a", "Xe máy có được đi vòng bên trái chướng ngại vật không?")
        self.assertEqual(right_only[0].decision, "CONDITIONAL")
        self.assertIn("intended_obstacle_passing_side", right_only[0].trace[1]["predicate"])
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in right_only[0].effects))

        missing_side = run("R.302a", "Xe máy có được đi vòng qua chướng ngại vật không?")
        self.assertEqual(missing_side[0].decision, "CONDITIONAL")
        self.assertIn("intended_obstacle_passing_side", missing_side[0].missing_facts)

        either_side = run("R.302c", "Xe đạp có được đi vòng bên trái chướng ngại vật không?")
        self.assertEqual(either_side[0].decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in either_side[0].effects))

    def test_ie468_variants_preserve_official_obstacle_route_guidance_conservatively(self) -> None:
        cases = (
            ("IE.468a", "Xe tải có được đi qua bên trái chướng ngại vật không?"),
            ("IE.468b", "Xe đạp có được đi qua bên phải chướng ngại vật không?"),
            ("IE.468c", "Xe máy có được đi qua bên phải chướng ngại vật không?"),
        )
        for code, question_text in cases:
            with self.subTest(code=code):
                rules = reviewed_rules([rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes])
                result = evaluate(_scoped_sign_facts(code), query(question_text), rules)[0]
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(any(effect.effect_type == "require_action" for effect in result.effects))
                self.assertFalse(any(effect.effect_type == "permit_action" for effect in result.effects))

        mismatch = evaluate(
            _scoped_sign_facts("IE.468a"),
            query("Xe tải có được đi qua bên phải chướng ngại vật không?"),
            reviewed_rules([rule for rule in TRAFFIC_RULES if "IE.468a" in rule.trigger_sign_codes]),
        )[0]
        self.assertEqual(mismatch.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "prohibit_action" for effect in mismatch.effects))

        unspecified_side = evaluate(
            _scoped_sign_facts("IE.468b"),
            query("Xe ô tô có được đi qua chướng ngại vật không?"),
            reviewed_rules([rule for rule in TRAFFIC_RULES if "IE.468b" in rule.trigger_sign_codes]),
        )[0]
        self.assertEqual(unspecified_side.decision, "CONDITIONAL")
        self.assertIn("intended_obstacle_passing_side", unspecified_side.missing_facts)

    def test_ie471_and_ie473_do_not_invent_numeric_distance_or_speed(self) -> None:
        ie471 = evaluate(
            _scoped_sign_facts("IE.471"),
            query("Xe ô tô có được giữ khoảng cách an toàn 5 mét không?"),
            reviewed_rules([rule for rule in TRAFFIC_RULES if "IE.471" in rule.trigger_sign_codes]),
        )[0]
        self.assertEqual(ie471.decision, "CONDITIONAL")
        self.assertNotIn("safe_following_distance_resolved", ie471.missing_facts)
        self.assertIn("does not itself state a numeric distance", ie471.reason)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in ie471.effects))

        ie473 = evaluate(
            _scoped_sign_facts("IE.473"),
            query("Xe ô tô có phải giảm tốc độ khi gặp biển không?"),
            reviewed_rules([rule for rule in TRAFFIC_RULES if "IE.473" in rule.trigger_sign_codes]),
        )[0]
        self.assertEqual(ie473.decision, "CONDITIONAL")
        self.assertNotIn("safe_speed_for_area_resolved", ie473.missing_facts)
        self.assertIn("no numeric speed", ie473.reason)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in ie473.effects))

    def test_r303_requires_arrow_relation_and_never_grants_destination_permission(self) -> None:
        facts = _scoped_sign_facts("R.303")
        r303_rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "R.303" in rule.trigger_sign_codes])
        unknown_route = evaluate(
            facts, query("Xe ô tô có được chạy vòng xuyến không?"), r303_rules
        )
        self.assertEqual(unknown_route[0].decision, "CONDITIONAL")
        self.assertIn("intended_roundabout_route_follows_arrow", unknown_route[0].missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in unknown_route[0].effects))

        stated_route = evaluate(
            facts,
            query("Xe ô tô có được chạy vòng xuyến theo hướng mũi tên không?"),
            r303_rules,
        )
        self.assertEqual(stated_route[0].decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_sign_constraints", stated_route[0].missing_facts)

    def test_r305_preserves_vehicle_exceptions_and_crossing_safety_condition(self) -> None:
        facts = _scoped_sign_facts("R.305")
        r305_rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "R.305" in rule.trigger_sign_codes])

        prohibited_vehicle = evaluate(
            facts, query("Xe máy có được đi vào đường này không?"), r305_rules
        )
        prohibition = next(item for item in prohibited_vehicle
                           if item.rule_id == "R305_PROHIBIT_NONEXEMPT_VEHICLE_ENTRY")
        self.assertEqual(prohibition.decision, "PROHIBITED")

        bicycle = evaluate(
            facts, query("Xe đạp có được đi vào đường này không?"), r305_rules
        )
        exception = next(item for item in bicycle
                         if item.rule_id == "R305_BICYCLE_EXCEPTION_NOT_BLANKET_PERMISSION")
        self.assertEqual(exception.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in exception.effects))

        crossing = evaluate(
            facts, query("Xe ô tô có được đi cắt ngang qua đường này không?"), r305_rules
        )
        crossing_rule = next(item for item in crossing
                             if item.rule_id == "R305_CROSSING_REQUIRES_ABSOLUTE_PEDESTRIAN_SAFETY")
        self.assertEqual(crossing_rule.decision, "CONDITIONAL")
        self.assertNotIn("absolute_safety_for_pedestrians_confirmed", crossing_rule.missing_facts)

        facts.context_facts.append(ContextFact(
            "pedestrian-safety-assessment", "absolute_safety_for_pedestrians_confirmed", True,
            status="provided", source="reviewed_scene_assessment", confidence="high",
        ))
        assessed_crossing = evaluate(
            facts, query("Xe ô tô có được đi cắt ngang qua đường này không?"), r305_rules
        )
        assessed_crossing_rule = next(item for item in assessed_crossing
                                      if item.rule_id == "R305_CROSSING_REQUIRES_ABSOLUTE_PEDESTRIAN_SAFETY")
        self.assertEqual(assessed_crossing_rule.decision, "CONDITIONAL")
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in assessed_crossing_rule.effects))

        wheelchair = evaluate(
            facts, query("Xe lăn có được đi vào đường này không?"), r305_rules
        )
        wheelchair_exception = next(item for item in wheelchair
                                    if item.rule_id == "R305_WHEELCHAIR_EXCEPTION_REQUIRES_CLASSIFICATION")
        self.assertEqual(wheelchair_exception.decision, "CONDITIONAL")
        self.assertIn("wheelchair_for_person_with_disability_confirmed", wheelchair_exception.missing_facts)

    def test_r309_horn_requirement_needs_sign_scope_and_position(self) -> None:
        facts = _scoped_sign_facts("R.309")
        r309_rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "R.309" in rule.trigger_sign_codes])

        missing_position = evaluate(
            facts, query("Xe ô tô có phải ấn còi không?"), r309_rules
        )
        self.assertEqual(missing_position[0].decision, "CONDITIONAL")
        self.assertNotIn("vehicle_at_r309_sign_position", missing_position[0].missing_facts)

        stated_position = evaluate(
            facts,
            query("Xe ô tô có phải ấn còi tại vị trí đặt biển không?"),
            r309_rules,
        )
        self.assertEqual(stated_position[0].decision, "CONDITIONAL")
        self.assertTrue(any(effect.effect_type == "require_action" for effect in stated_position[0].effects))
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in stated_position[0].effects))

    def test_r310_variant_arrows_are_source_mapped_and_hazard_cargo_is_required(self) -> None:
        for code, direction in (("R.310a", "turn_left"), ("R.310b", "going_straight"),
                                ("R.310c", "turn_right")):
            facts = _scoped_sign_facts(code)
            sign_rules = reviewed_rules([rule for rule in TRAFFIC_RULES if code in rule.trigger_sign_codes])
            question_text = {
                "turn_left": "Xe tải chở hàng nguy hiểm có được rẽ trái không?",
                "going_straight": "Xe tải chở hàng nguy hiểm có được đi thẳng không?",
                "turn_right": "Xe tải chở hàng nguy hiểm có được rẽ phải không?",
            }[direction]
            matched_direction = evaluate(facts, query(question_text), sign_rules)
            direction_label = {"turn_left": "LEFT_TURN", "going_straight": "STRAIGHT",
                               "turn_right": "RIGHT_TURN"}[direction]
            direction_rule = next(item for item in matched_direction
                                  if item.rule_id.endswith(f"REQUIRE_{direction_label}_HAZMAT"))
            self.assertEqual(direction_rule.decision, "CONDITIONAL")
            self.assertTrue(any(effect.effect_type == "require_action" for effect in direction_rule.effects))
            self.assertFalse(any(effect.effect_type == "permit_action" for effect in direction_rule.effects))

            wrong_direction = {
                "turn_left": "Xe tải chở hàng nguy hiểm có được rẽ phải không?",
                "going_straight": "Xe tải chở hàng nguy hiểm có được rẽ trái không?",
                "turn_right": "Xe tải chở hàng nguy hiểm có được đi thẳng không?",
            }[direction]
            prohibited = evaluate(facts, query(wrong_direction), sign_rules)
            self.assertTrue(any(item.decision == "PROHIBITED" for item in prohibited))

        unknown_cargo = evaluate(
            _scoped_sign_facts("R.310a"),
            query("Xe tải có được rẽ phải không?"),
            reviewed_rules([rule for rule in TRAFFIC_RULES if "R.310a" in rule.trigger_sign_codes]),
        )
        self.assertTrue(unknown_cargo)
        self.assertTrue(all(item.decision in {"UNKNOWN", "CONDITIONAL"} for item in unknown_cargo))
        self.assertTrue(any("hazardous_cargo_status" in item.missing_facts for item in unknown_cargo))

        non_hazardous = evaluate(
            _scoped_sign_facts("R.310a"),
            query("Xe tải không chở hàng nguy hiểm có được rẽ phải không?"),
            reviewed_rules([rule for rule in TRAFFIC_RULES if "R.310a" in rule.trigger_sign_codes]),
        )
        self.assertTrue(non_hazardous)
        self.assertTrue(all(item.decision == "UNKNOWN" for item in non_hazardous))

    def test_rulebook_loader_rejects_duplicate_rule_ids(self) -> None:
        payload = {
            "rules": [TRAFFIC_RULES[0].to_dict(), TRAFFIC_RULES[0].to_dict()]
        }
        with TemporaryDirectory() as temporary_directory:
            path = f"{temporary_directory}/duplicate-rules.json"
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream)
            with self.assertRaisesRegex(ValueError, "duplicate rule_id"):
                load_rulebook(path)

    def test_vehicle_scope(self) -> None:
        self.assertTrue(vehicle_is_covered("car", "motor_vehicle"))
        self.assertTrue(vehicle_is_covered("motorbike", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("bicycle", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("pedestrian", "vehicle"))
        self.assertTrue(vehicle_is_covered("semi_trailer", "vehicle"))
        self.assertFalse(vehicle_is_covered("unknown", "vehicle"))
        self.assertTrue(vehicle_is_covered("electric_bicycle", "non_motor_vehicle"))
        self.assertTrue(vehicle_is_covered("wheelchair", "non_motor_vehicle"))
        self.assertTrue(vehicle_is_covered("wheelchair", "vehicle"))
        self.assertTrue(vehicle_is_covered("pedicab", "non_motor_vehicle"))
        self.assertTrue(vehicle_is_covered("semi_trailer", "motor_vehicle"))
        self.assertTrue(vehicle_is_covered("construction_machine", "special_use_machine"))
        self.assertTrue(vehicle_is_covered("agricultural_forestry_machine", "special_use_machine"))
        self.assertTrue(vehicle_is_covered("tractor", "special_use_machine"))
        self.assertFalse(vehicle_is_covered("tractor", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("mobility_aid", "non_motor_vehicle"))
        self.assertTrue(vehicle_is_covered("non_motorized_tricycle", "non_motor_vehicle"))
        self.assertTrue(vehicle_is_covered("cargo_bicycle", "non_motor_vehicle"))
        self.assertFalse(vehicle_is_covered("electric_scooter", "non_motor_vehicle"))
        self.assertTrue(vehicle_is_covered("motorcycle_or_moped", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("motorcycle_or_moped", "motorbike"))
        self.assertFalse(vehicle_is_covered("motorcycle_or_moped", "moped"))
        self.assertTrue(vehicle_is_covered("semi_trailer", "vehicle"))
        self.assertFalse(vehicle_is_covered("unknown", "vehicle"))

    def test_ambiguous_everyday_motorcycle_term_only_matches_rules_covering_both_legal_classes(self) -> None:
        parsed = query("Xe máy có được đi vào không?")
        self.assertEqual(parsed.query_intent.intents[0].actor.value, "motorcycle_or_moped")
        p104_facts = _scoped_sign_facts("P.104")
        p104_rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "P.104" in rule.trigger_sign_codes])
        result = evaluate(p104_facts, parsed, p104_rules)
        self.assertTrue(result)
        self.assertTrue(all(item.decision == "UNKNOWN" for item in result))
        self.assertTrue(any("motorcycle_or_moped_subtype" in item.missing_facts for item in result))

        broad_facts = _scoped_sign_facts("R.305")
        broad_rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "R.305" in rule.trigger_sign_codes])
        broad_result = evaluate(broad_facts, parsed, broad_rules)
        self.assertTrue(broad_result)
        self.assertTrue(any(item.decision == "PROHIBITED" for item in broad_result))
        self.assertFalse(any(
            "motorcycle_or_moped_subtype" in item.missing_facts
            for item in broad_result
        ))

    def test_p130_matches_parking_and_stopping(self) -> None:
        facts = _scoped_sign_facts("P.130")
        parking = evaluate(facts, query("Xe ô tô có được phép đỗ không?"))
        stopping = evaluate(facts, query("Xe ô tô có được dừng xe không?"))
        self.assertEqual(parking[0].rule_id, "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(stopping[0].rule_id, "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(parking[0].decision, "PROHIBITED")

    def test_p130_bus_context_never_becomes_an_implicit_permission(self) -> None:
        result = evaluate(
            _scoped_sign_facts("P.130"),
            query("Xe buýt có được dừng xe ở đây không?"),
            reviewed_rules(TRAFFIC_RULES),
        )
        bus_rule = next(
            item for item in result
            if item.rule_id == "P130_BUS_STOPPING_OR_PARKING_REQUIRES_CONTEXT"
        )
        self.assertEqual(bus_rule.decision, "CONDITIONAL")
        self.assertNotIn("scheduled_bus_route_status", bus_rule.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in bus_rule.effects))

    def test_p125_overtaking_distinguishes_motor_vehicle_target_from_exceptions(self) -> None:
        sign = SignObservation(
            "p125", role="main", sign_code="P.125", identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True,
            }}, confidence="high",
        )
        facts = TrafficFacts(
            "2.0", "p125-truck-target", [sign], context_facts=[
                ContextFact("actor-motor", "actor_is_motor_vehicle", True,
                            status="provided", source="test", confidence="high"),
                ContextFact("target-motor", "overtaken_vehicle_is_motor_vehicle", True,
                            status="provided", source="test", confidence="high"),
                ContextFact("target-class", "overtaken_vehicle_class", "truck",
                            status="provided", source="test", confidence="high"),
            ], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        p125_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if "P.125" in rule.trigger_sign_codes
        ])
        truck_target = evaluate(
            facts, query("Ô tô có được vượt xe tải không?"), p125_rules
        )
        prohibition = next(
            item for item in truck_target
            if item.rule_id == "P125_NO_OVERTAKING_OTHER_MOTOR_VEHICLES"
        )
        self.assertEqual(prohibition.decision, "PROHIBITED")

        facts.context_facts[-1].value = "moped"
        moped_target = evaluate(
            facts, query("Ô tô có được vượt xe gắn máy không?"), p125_rules
        )
        exception = next(
            item for item in moped_target
            if item.rule_id == "P125_TWO_WHEEL_MOTORCYCLE_MOPED_EXCEPTION"
        )
        self.assertEqual(exception.decision, "CONDITIONAL")
        self.assertTrue(any(effect.effect_type == "sign_meaning" for effect in exception.effects))
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in exception.effects))

    def test_p126_uses_certified_payload_capacity_not_cargo_mass_and_covers_priority_trucks(self) -> None:
        def p126_facts(actor_category: str, payload_fact: ContextFact | None) -> TrafficFacts:
            context = [
                ContextFact("target-motor", "overtaken_vehicle_is_motor_vehicle", True,
                            status="provided", source="test", confidence="high"),
                ContextFact("target-class", "overtaken_vehicle_class", "car",
                            status="provided", source="test", confidence="high"),
                ContextFact("actor-category", "actor_vehicle_category", actor_category,
                            status="provided", source="test", confidence="high"),
            ]
            if payload_fact:
                context.append(payload_fact)
            return TrafficFacts(
                "2.0", "p126-scene", [SignObservation(
                    "p126", role="main", sign_code="P.126", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                )], context_facts=context, sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
            )

        p126_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if "P.126" in rule.trigger_sign_codes
        ])
        certified = ContextFact(
            "certificate-payload", "certified_payload_capacity_t", 4.0,
            status="provided", source="vehicle_certificate", confidence="high",
        )
        truck = evaluate(
            p126_facts("truck", certified),
            query("Xe tải có được vượt xe ô tô không?"), p126_rules,
        )
        heavy_truck_prohibition = next(
            item for item in truck
            if item.rule_id == "P126_HEAVY_TRUCK_NO_OVERTAKING_OTHER_MOTOR_VEHICLES"
        )
        self.assertEqual(heavy_truck_prohibition.decision, "PROHIBITED")

        priority_truck = evaluate(
            p126_facts("truck", certified),
            query("Xe ưu tiên có được vượt xe ô tô không?"), p126_rules,
        )
        priority_heavy_truck_prohibition = next(
            item for item in priority_truck
            if item.rule_id == "P126_HEAVY_TRUCK_NO_OVERTAKING_OTHER_MOTOR_VEHICLES"
        )
        self.assertEqual(priority_heavy_truck_prohibition.decision, "PROHIBITED")

        unresolved_priority_actor = query("Xe ưu tiên có được vượt xe ô tô không?")
        unresolved_priority_actor.query_intent.intents[0].actor.status = "needs_resolution"
        resolved_by_vehicle_facts = evaluate(
            p126_facts("truck", certified), unresolved_priority_actor, p126_rules,
        )
        resolved_heavy_truck_prohibition = next(
            item for item in resolved_by_vehicle_facts
            if item.rule_id == "P126_HEAVY_TRUCK_NO_OVERTAKING_OTHER_MOTOR_VEHICLES"
        )
        self.assertEqual(resolved_heavy_truck_prohibition.decision, "PROHIBITED")

        wrong_fact = ContextFact(
            "cargo-mass", "vehicle_payload_t", 4.0,
            status="provided", source="user", confidence="high",
        )
        unresolved = evaluate(
            p126_facts("truck", wrong_fact),
            query("Xe tải có được vượt xe ô tô không?"), p126_rules,
        )
        self.assertEqual(unresolved[0].decision, "CONDITIONAL")
        self.assertIn("certified_payload_capacity_t", unresolved[0].missing_facts)

        threshold = ContextFact(
            "certificate-payload", "certified_payload_capacity_t", 3.5,
            status="provided", source="vehicle_certificate", confidence="high",
        )
        at_threshold = evaluate(
            p126_facts("truck", threshold),
            query("Xe tải có được vượt xe ô tô không?"), p126_rules,
        )
        below_prohibition = next(
            item for item in at_threshold
            if item.rule_id == "P126_HEAVY_TRUCK_NO_OVERTAKING_OTHER_MOTOR_VEHICLES"
        )
        self.assertEqual(below_prohibition.status, "not_applicable")
        self.assertFalse(any(item.decision == "ALLOWED" for item in at_threshold))

    def test_end_restriction_signs_never_infer_overall_permission(self) -> None:
        def end_sign_facts(code: str, context_facts: list[ContextFact] | None = None) -> TrafficFacts:
            return TrafficFacts(
                "2.0", f"scene-{code}", [SignObservation(
                    f"obs-{code}", role="main", sign_code=code, identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                )], context_facts=context_facts or [], sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
            )

        end_sign_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if rule.trigger_sign_codes[0] in {"DP.133", "DP.134", "DP.135"}
        ])
        dp133 = evaluate(
            end_sign_facts("DP.133"),
            query("Xe ô tô có được vượt xe khác không?"), end_sign_rules,
        )
        dp133_result = next(item for item in dp133 if item.rule_id == "DP133_ENDS_OVERTAKING_PROHIBITIONS")
        self.assertEqual(dp133_result.decision, "CONDITIONAL")
        self.assertFalse(any(item.decision == "ALLOWED" for item in dp133))
        self.assertIn("conditions for overtaking", dp133_result.reason)

        general_speed = ContextFact(
            "general-speed", "general_max_speed_kmh", 50,
            status="provided", source="road_and_vehicle_context", confidence="high",
        )
        dp134 = evaluate(
            end_sign_facts("DP.134", [general_speed]),
            query("Xe ô tô được chạy tốc độ tối đa bao nhiêu?"), end_sign_rules,
        )
        self.assertEqual(dp134, [])

        active_p125 = ContextFact(
            "active-prohibitions", "active_restriction_sign_codes", ["P.125"],
            status="provided", source="scene_history", confidence="high",
        )
        dp135 = evaluate(
            end_sign_facts("DP.135", [active_p125]),
            query("Xe ô tô có được vượt xe khác không?"), end_sign_rules,
        )
        dp135_result = next(item for item in dp135 if item.rule_id == "DP135_ENDS_OVERTAKING_P125")
        self.assertEqual(dp135_result.decision, "CONDITIONAL")
        self.assertFalse(any(item.decision == "ALLOWED" for item in dp135))

        missing_prior_sign = evaluate(
            end_sign_facts("DP.135"),
            query("Xe ô tô có được vượt xe khác không?"), end_sign_rules,
        )
        missing_result = next(item for item in missing_prior_sign if item.rule_id == "DP135_ENDS_OVERTAKING_P125")
        self.assertEqual(missing_result.decision, "CONDITIONAL")
        self.assertIn("active_restriction_sign_codes", missing_result.missing_facts)

    def test_p121_uses_official_scope_priority_classes_and_main_sign_ocr(self) -> None:
        p121_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if "P.121" in rule.trigger_sign_codes
        ])

        def p121_facts(actor_distance: float) -> TrafficFacts:
            return TrafficFacts(
                "2.0", "p121-scene", [SignObservation(
                    "p121", role="main", sign_code="P.121", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }},
                    ocr_texts=[{"raw_text": "8 m", "confidence": "high"}],
                    confidence="high",
                )], context_facts=[ContextFact(
                    "actual-distance", "vehicle_following_distance_m", actor_distance,
                    status="provided", source="range_sensor", confidence="high",
                )], sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
            )

        too_close = evaluate(
            p121_facts(5), query("Xe ưu tiên có được giữ khoảng cách 5 m không?"), p121_rules,
        )
        p121_result = next(item for item in too_close if item.rule_id == "P121_MINIMUM_FOLLOWING_DISTANCE")
        self.assertEqual(p121_result.decision, "PROHIBITED")
        self.assertTrue(any(
            item.get("predicate") == "minimum_following_distance_m" and item.get("state") == "satisfied"
            for item in p121_result.trace
        ))

        at_minimum = evaluate(
            p121_facts(8), query("Xe ô tô có được giữ khoảng cách 8 m không?"), p121_rules,
        )
        at_minimum_result = next(item for item in at_minimum if item.rule_id == "P121_MINIMUM_FOLLOWING_DISTANCE")
        self.assertEqual(at_minimum_result.decision, "CONDITIONAL")
        self.assertEqual(at_minimum_result.effects[0].value, 8)
        self.assertFalse(any(item.decision == "ALLOWED" for item in at_minimum))

        conflicting_ocr = TrafficFacts(
            "2.0", "p121-conflicting-ocr", [SignObservation(
                "p121-conflict", role="main", sign_code="P.121", identity_status="identified",
                visual_attributes={"normalized_conditions": {
                    "sign_effective_scope_confirmed": True,
                    "minimum_following_distance_m": 6,
                }},
                ocr_texts=[
                    {"raw_text": "8 m", "confidence": "high"},
                    {"raw_text": "10 m", "confidence": "medium"},
                ], confidence="high",
            )], context_facts=[ContextFact(
                "actual-distance", "vehicle_following_distance_m", 5,
                status="provided", source="range_sensor", confidence="high",
            )], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        conflict_result = evaluate(
            conflicting_ocr, query("Xe ô tô có được giữ khoảng cách 5 m không?"), p121_rules,
        )
        unresolved_limit = next(
            item for item in conflict_result if item.rule_id == "P121_MINIMUM_FOLLOWING_DISTANCE"
        )
        self.assertEqual(unresolved_limit.decision, "CONDITIONAL")
        self.assertIn("minimum_following_distance_m", unresolved_limit.missing_facts)

    def test_p128_and_p129_require_confirmed_scope_and_vehicle_facts(self) -> None:
        reviewed_p128_p129 = reviewed_rules([
            rule for rule in TRAFFIC_RULES if set(rule.trigger_sign_codes) & {"P.128", "P.129"}
        ])

        p128_facts = TrafficFacts(
            "2.0", "p128-scene", [SignObservation(
                "p128", role="main", sign_code="P.128", identity_status="identified",
                visual_attributes={"normalized_conditions": {
                    "sign_effective_scope_confirmed": True,
                }}, confidence="high",
            )], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        p128 = evaluate(p128_facts, query("Xe đạp có được bấm còi ở đây không?"), reviewed_p128_p129)
        horn = next(item for item in p128 if item.rule_id == "P128_NO_HORN")
        self.assertEqual(horn.decision, "PROHIBITED")

        p129_facts = TrafficFacts(
            "2.0", "p129-scene", [SignObservation(
                "p129", role="main", sign_code="P.129", identity_status="identified",
                visual_attributes={"normalized_conditions": {
                    "sign_effective_scope_confirmed": True,
                }}, confidence="high",
            )], context_facts=[
                ContextFact("transport", "vehicle_is_transport_vehicle", True,
                            status="provided", source="vehicle_profile", confidence="high"),
                ContextFact("passing", "passing_inspection_station", True,
                            status="provided", source="scene_context", confidence="high"),
            ], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        p129 = evaluate(p129_facts, query("Xe tải có phải dừng ở trạm kiểm tra không?"), reviewed_p128_p129)
        inspection = next(item for item in p129 if item.rule_id == "P129_INSPECTION_STOP_REQUIRED")
        self.assertEqual(inspection.decision, "CONDITIONAL")
        self.assertTrue(any(effect.effect_type == "require_action" for effect in inspection.effects))

        unresolved_p129_facts = TrafficFacts(
            "2.0", "p129-unknown-vehicle", [SignObservation(
                "p129-unknown", role="main", sign_code="P.129", identity_status="identified",
                visual_attributes={"normalized_conditions": {
                    "sign_effective_scope_confirmed": True,
                }}, confidence="high",
            )], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        unresolved = evaluate(
            unresolved_p129_facts, query("Xe có phải dừng ở trạm kiểm tra không?"), reviewed_p128_p129,
        )
        unresolved_inspection = next(item for item in unresolved if item.rule_id == "P129_INSPECTION_STOP_REQUIRED")
        self.assertEqual(unresolved_inspection.decision, "UNKNOWN")
        self.assertIn("actor_class", unresolved_inspection.missing_facts)

    def test_r306_r307_minimum_speed_scope_and_end_conditions(self) -> None:
        speed_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if set(rule.trigger_sign_codes) & {"R.306", "R.307"}
        ])

        def sign_facts(code: str, facts: list[ContextFact]) -> TrafficFacts:
            return TrafficFacts(
                "2.0", f"scene-{code}", [SignObservation(
                    f"obs-{code}", role="main", sign_code=code, identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }},
                    ocr_texts=[{"raw_text": "60", "confidence": "high"}] if code == "R.306" else [],
                    confidence="high",
                )], context_facts=facts, sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
            )

        favorable = ContextFact(
            "safe-traffic", "traffic_conditions_favorable_and_safe", True,
            status="provided", source="road_context", confidence="high",
        )
        under_minimum = evaluate(
            sign_facts("R.306", [favorable]),
            query("Xe ô tô có được chạy chậm hơn 50 km/h ở đây không?"), speed_rules,
        )
        below = next(item for item in under_minimum if item.rule_id == "R306_MINIMUM_SPEED_IN_SAFE_CONDITIONS")
        self.assertEqual(below.decision, "PROHIBITED")

        max_capability = ContextFact(
            "manufacturer-max", "vehicle_manufacturer_max_speed_kmh", 50,
            status="provided", source="vehicle_specification", confidence="high",
        )
        cannot_enter = evaluate(
            sign_facts("R.306", [max_capability]),
            query("Xe ô tô có được đi vào đường này không?"), speed_rules,
        )
        entry_rule = next(
            item for item in cannot_enter
            if item.rule_id == "R306_NO_ENTRY_BELOW_MANUFACTURER_CAPABILITY"
        )
        self.assertEqual(entry_rule.decision, "PROHIBITED")

        would_not_impede = ContextFact(
            "not-impede", "would_impede_other_vehicles", False,
            status="provided", source="traffic_context", confidence="high",
        )
        r307_slow = evaluate(
            sign_facts("R.307", [would_not_impede]),
            query("Xe ô tô có được chạy chậm hơn 50 km/h ở đây không?"), speed_rules,
        )
        ended_requirement = next(
            item for item in r307_slow
            if item.rule_id == "R307_MINIMUM_SPEED_ENDS_WHEN_NOT_IMPEDING"
        )
        self.assertEqual(ended_requirement.decision, "CONDITIONAL")
        self.assertFalse(any(item.decision == "ALLOWED" for item in r307_slow))

        would_impede = ContextFact(
            "impede", "would_impede_other_vehicles", True,
            status="provided", source="traffic_context", confidence="high",
        )
        r307_impeding = evaluate(
            sign_facts("R.307", [would_impede]),
            query("Xe ô tô có được chạy chậm hơn 50 km/h ở đây không?"), speed_rules,
        )
        prohibited = next(
            item for item in r307_impeding
            if item.rule_id == "R307_PROHIBIT_SLOW_SPEED_THAT_IMPEDES"
        )
        self.assertEqual(prohibited.decision, "PROHIBITED")

    def test_odd_even_conditions_are_evaluated_from_context_facts(self) -> None:
        odd_results = evaluate(
            _scoped_sign_facts("P.131b"),
            query("Ngày lẻ, xe ô tô có được đỗ xe không?"),
        )
        even_results = evaluate(
            _scoped_sign_facts("P.131b"),
            query("Ngày chẵn, xe ô tô có được đỗ xe không?"),
        )
        odd = next(item for item in odd_results if item.status != "not_applicable")
        self.assertEqual((odd.decision, odd.status), ("PROHIBITED", "applied"))
        self.assertTrue(even_results)
        self.assertTrue(all((item.decision, item.status) == ("UNKNOWN", "not_applicable") for item in even_results))

    def test_unreviewed_rule_with_basis_can_issue_a_legal_decision(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.130"}),
            query("Xe ô tô có được phép đỗ không?"),
            rules=list(TRAFFIC_RULES),
        )[0]
        self.assertEqual((result.decision, result.status), ("PROHIBITED", "applied"))
        self.assertNotIn("legal_review_required", result.missing_facts)
        self.assertEqual(result.verification_status, "unverified")

    def test_unverified_rule_executes_without_eval_override(self) -> None:
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P110A_NO_BICYCLE")
        facts = _scoped_sign_facts("P.110a")
        parsed = query("Xe đạp có được lưu thông trên đoạn đường này không?")

        result = find_matching_rules(
            facts,
            parsed.query_intent,
            parsed.context_facts,
            rules=[rule],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].decision, result[0].status), ("PROHIBITED", "applied"))
        self.assertEqual(result[0].verification_status, "unverified")
        self.assertNotIn("legal_review_gate", [item.get("step") for item in result[0].trace])

    def test_permission_query_requires_vehicle_class_when_sign_scope_depends_on_it(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.130"}),
            query("Có được đỗ xe ở đây không?"),
            rules=reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual((result.decision, result.status), ("UNKNOWN", "unresolved"))
        self.assertIn("actor_class", result.missing_facts)

    def test_permission_query_requires_vehicle_class_when_sign_scope_depends_on_it(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.130"}),
            query("Có được đỗ xe ở đây không?"),
            rules=reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual((result.decision, result.status), ("UNKNOWN", "unresolved"))
        self.assertIn("actor_class", result.missing_facts)

    def test_explicit_query_time_resolves_calendar_parity(self) -> None:
        result = evaluate(
            _scoped_sign_facts("P.131c"),
            query("Xe ô tô có được đỗ lúc 7 giờ hôm nay không?"),
        )[0]
        self.assertEqual(result.decision, "PROHIBITED")
        self.assertEqual(result.status, "applied")
        self.assertEqual(result.missing_facts, [])
        self.assertTrue(any(
            item.get("predicate") == "calendar_parity" and item.get("value") == "even_days"
            for item in result.trace
        ))

    def test_system_clock_fact_supplies_calendar_parity(self) -> None:
        result = evaluate(
            _scoped_sign_facts("P.131c"),
            query("Xe ô tô có được đỗ không?"),
        )[0]
        self.assertEqual(result.decision, "PROHIBITED")
        self.assertTrue(any(item.get("predicate") == "calendar_parity" for item in result.trace))

    def test_p127_missing_numeric_value_is_incomplete(self) -> None:
        facts = facts_for({"sign_code": "P.127"})
        facts.sign_observations[0].visual_attributes["normalized_conditions"] = {
            "sign_effective_scope_confirmed": True,
        }
        facts.sign_observations[0].confidence = "high"
        result = evaluate(
            facts, query("Xe ô tô có được chạy ở tốc độ 60 km/h không?")
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertEqual(result.missing_facts, ["speed_limit_value"])

    def test_low_confidence_numeric_reading_is_not_used_as_a_rule_fact(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "speed-sign", role="main", sign_code="P.127", identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "speed_limit_value": 60, "sign_effective_scope_confirmed": True,
            }},
            confidence="low",
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "complete scene, low-confidence value read"
        }])
        result = evaluate(facts, query("Xe ô tô có được chạy ở tốc độ 60 km/h không?"))[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertIn("speed_limit_value", result.missing_facts)
        self.assertEqual(result.effects, [])

    def test_speed_rule_resolves_an_effect_from_supported_sign_fact(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "speed-sign",
            role="main",
            sign_code="P.127",
            identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "speed_limit_value": 60, "sign_effective_scope_confirmed": True,
            }},
            confidence="high",
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "fixture includes speed sign"
        }])
        rule = reviewed_rules([next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P127_SPEED_LIMIT")])
        result = evaluate(facts, query("Xe ô tô có được chạy ở tốc độ tối đa của biển không?"), rules=rule)[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertEqual(result.effects[0].effect_type, "maximum")
        self.assertEqual(result.effects[0].value, 60)

    def test_speed_permission_compares_requested_value_without_inferring_allowed(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "speed-sign",
            role="main",
            sign_code="P.127",
            identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "speed_limit_value": 60, "sign_effective_scope_confirmed": True,
            }},
            confidence="high",
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "fixture includes speed sign"
        }])
        speed_rule = reviewed_rules([next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P127_SPEED_LIMIT")])
        above = evaluate(facts, query("Xe ô tô có được chạy ở tốc độ 70 km/h không?"), rules=speed_rule)[0]
        below = evaluate(facts, query("Xe ô tô có được chạy ở tốc độ 50 km/h không?"), rules=speed_rule)[0]
        self.assertEqual(above.decision, "PROHIBITED")
        self.assertEqual(above.effects[0].value, 60)
        self.assertEqual(below.decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_sign_constraints", below.missing_facts)

    def test_p127_variants_require_their_distinct_time_lane_and_vehicle_facts(self) -> None:
        variants = {
            "P.127a": ("P127A_NIGHT_SPEED_LIMIT", [
                ContextFact("night-zone", "night_speed_zone_scope_confirmed", True,
                            status="provided", source="test", confidence="high"),
            ]),
            "P.127b": ("P127B_CURRENT_LANE_SPEED_LIMIT", [
                ContextFact("lane-scope", "current_lane_scope_confirmed", True,
                            status="provided", source="test", confidence="high"),
                ContextFact("lane-limit", "current_lane_speed_limit_value", 60,
                            status="provided", source="test", confidence="high"),
            ]),
            "P.127c": ("P127C_CURRENT_VEHICLE_LANE_SPEED_LIMIT", [
                ContextFact("vehicle-lane-scope", "current_vehicle_lane_scope_confirmed", True,
                            status="provided", source="test", confidence="high"),
                ContextFact("vehicle-lane-limit", "applicable_speed_limit_value", 50,
                            status="provided", source="test", confidence="high"),
            ]),
        }
        for code, (rule_id, extra_facts) in variants.items():
            with self.subTest(code=code):
                sign = SignObservation(
                    f"sign-{code}", role="main", sign_code=code, identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True, "speed_limit_value": 70,
                        **({"time_window": {"start_minute": 1260, "end_minute": 300}}
                           if code == "P.127a" else {}),
                    }}, confidence="high",
                )
                facts = TrafficFacts(
                    "2.0", f"scene-{code}", [sign], context_facts=extra_facts,
                    sign_inventory_status="complete",
                    sign_inventory_evidence=[{"source": "test", "evidence": "complete scene"}],
                )
                rules = reviewed_rules([rule for rule in TRAFFIC_RULES if rule.rule_id == rule_id])
                instant = datetime(2026, 10, 4, 23, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")) \
                    if code == "P.127a" else None
                question = "Xe ô tô có được chạy ở tốc độ 60 km/h không?" \
                    if code == "P.127a" else "Xe ô tô có được chạy ở tốc độ tối đa của biển không?"
                result = evaluate(facts, query(question, instant), rules)[0]
                self.assertEqual(result.decision, "CONDITIONAL")
                self.assertTrue(result.effects, result.to_dict())
                self.assertEqual(result.effects[0].effect_type, "maximum")
                expected = {"P.127a": 70, "P.127b": 60, "P.127c": 50}[code]
                self.assertEqual(result.effects[0].value, expected)

        # A P.127a time window must be active; without the S.508 reading,
        # the engine must report the missing fact instead of assuming a time.
        night_sign = SignObservation(
            "p127a-night", role="main", sign_code="P.127a", identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True, "speed_limit_value": 70,
            }}, confidence="high",
        )
        facts = TrafficFacts(
            "2.0", "night-window", [night_sign], context_facts=[
                ContextFact("night-zone", "night_speed_zone_scope_confirmed", True,
                            status="provided", source="test", confidence="high"),
            ], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test", "evidence": "complete scene"}],
        )
        night_rule = reviewed_rules([
            rule for rule in TRAFFIC_RULES if rule.rule_id == "P127A_NIGHT_SPEED_LIMIT"
        ])
        active = evaluate(
            facts, query("Xe ô tô có được chạy ở tốc độ 60 km/h không?", datetime(
                2026, 10, 4, 23, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")
            )), night_rule,
        )[0]
        self.assertEqual(active.decision, "CONDITIONAL")
        self.assertIn("time_window", active.missing_facts)

    def test_missing_condition_cannot_be_configured_to_grant_permission(self) -> None:
        self.assertEqual(RuleCondition("some_fact", "exists", on_missing="ALLOWED").on_missing, "CONDITIONAL")
        self.assertEqual(RuleCondition("some_fact", "equals", True, on_mismatch="ALLOWED").on_mismatch, "NOT_APPLICABLE")

    def test_allowed_rule_requires_an_explicit_affirmative_effect(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit permit_action or allow_direction"):
            TrafficRule(
                "unsafe-implicit-allow", [], ["R.301"], ["straight"],
                ["vehicle"], "ALLOWED",
            )

    def test_priority_exception_and_bus_exception_are_not_applied_to_excluded_actors(self) -> None:
        rules = reviewed_rules(TRAFFIC_RULES)
        p101 = [rule for rule in rules if rule.rule_id == "P101_NO_ENTRY_ALL_DIRECTIONS"]
        p107a = [rule for rule in rules if rule.rule_id == "P107A_NO_PASSENGER_CAR"]
        priority = evaluate(facts_for({"sign_code": "P.101"}), query("Xe ưu tiên có được đi vào không?"), p101)
        regular_car = evaluate(facts_for({"sign_code": "P.101"}), query("Ô tô có được đi vào không?"), p101)
        bus = evaluate(facts_for({"sign_code": "P.107a"}), query("Xe buýt có được đi qua không?"), p107a)
        passenger_car = evaluate(facts_for({"sign_code": "P.107a"}), query("Ô tô có được đi qua không?"), p107a)
        self.assertEqual(priority, [])
        self.assertEqual(regular_car[0].decision, "PROHIBITED")
        self.assertEqual(bus, [])
        self.assertEqual(passenger_car[0].decision, "PROHIBITED")

    def test_p103_variants_keep_turn_direction_specific(self) -> None:
        right_facts = facts_for({"sign_code": "P.103b"})
        left_facts = facts_for({"sign_code": "P.103c"})
        right_facts.sign_observations[0].visual_attributes["normalized_conditions"] = {
            "sign_effective_scope_confirmed": True
        }
        left_facts.sign_observations[0].visual_attributes["normalized_conditions"] = {
            "sign_effective_scope_confirmed": True
        }
        right_facts.sign_observations[0].confidence = "high"
        left_facts.sign_observations[0].confidence = "high"
        right = evaluate(
            right_facts,
            query("Xe ô tô có được rẽ phải không?"),
            reviewed_rules(TRAFFIC_RULES),
        )[0]
        left = evaluate(
            left_facts,
            query("Xe ô tô có được rẽ trái không?"),
            reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual((right.rule_id, right.decision), ("P103B_NO_MOTOR_VEHICLE_RIGHT_TURN", "PROHIBITED"))
        self.assertEqual((left.rule_id, left.decision), ("P103C_NO_MOTOR_VEHICLE_LEFT_TURN", "PROHIBITED"))

    def test_p106b_uses_registered_payload_capacity_and_displayed_limit(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "weight-sign", role="main", sign_code="P.106b", identity_status="identified",
            confidence="high", visual_attributes={"normalized_conditions": {
                "payload_limit_t": 3.5, "sign_effective_scope_confirmed": True,
            }},
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "complete scene and confident sign reading"
        }])
        intent = query("Xe tải có được đi vào không?")
        intent.context_facts.append(ContextFact(
            "vehicle-payload", "vehicle_payload_t", 4.0,
            status="provided", source="user_query", confidence="high",
        ))
        result = evaluate(facts, intent, reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual((result.decision, result.status), ("PROHIBITED", "applied"))
        self.assertEqual(result.effects[0].value, 3.5)

    def test_p106c_needs_explicit_hazardous_cargo_fact(self) -> None:
        facts = facts_for({"sign_code": "P.106c"})
        facts.sign_observations[0].visual_attributes["normalized_conditions"] = {
            "sign_effective_scope_confirmed": True
        }
        facts.sign_observations[0].confidence = "high"
        query_result = query("Xe tải có được đi vào không?")
        missing = evaluate(facts, query_result, reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual(missing.decision, "CONDITIONAL")
        self.assertIn("hazardous_cargo_status", missing.missing_facts)

        query_result.context_facts.append(ContextFact(
            "cargo-status", "hazardous_cargo_status", True,
            status="provided", source="user_query", confidence="high",
        ))
        hazardous = evaluate(facts, query_result, reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual(hazardous.decision, "PROHIBITED")

    def test_p110b_rule_is_scoped_to_cargo_bicycles(self) -> None:
        facts = facts_for({"sign_code": "P.110b"})
        facts.sign_observations[0].visual_attributes["normalized_conditions"] = {
            "sign_effective_scope_confirmed": True
        }
        facts.sign_observations[0].confidence = "high"
        result = evaluate(
            facts,
            query("Xe đạp thồ có được đi vào không?"),
            reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual(result.rule_id, "P110B_NO_CARGO_BICYCLE")
        self.assertEqual(result.decision, "PROHIBITED")

    def test_p108_requires_known_trailer_configuration(self) -> None:
        facts = facts_for({"sign_code": "P.108"})
        parsed = query("Xe ô tô có được đi vào không?")
        missing = evaluate(facts, parsed, reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual(missing.decision, "CONDITIONAL")
        self.assertIn("vehicle_trailer_configuration", missing.missing_facts)

        parsed.context_facts.append(ContextFact(
            "trailer-config", "vehicle_trailer_configuration", "towing_trailer",
            status="provided", source="user_query", confidence="high",
        ))
        towing = evaluate(facts, parsed, reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual(towing.decision, "PROHIBITED")

    def test_vehicle_trailer_configuration_is_not_inferred_from_sign_visual_fields(self) -> None:
        facts = facts_for({
            "sign_code": "P.108",
            "visual_attributes": {"normalized_conditions": {
                "vehicle_trailer_configuration": "towing_trailer",
            }},
        })
        result = evaluate(
            facts,
            query("Xe ô tô có được đi vào không?"),
            reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertIn("vehicle_trailer_configuration", result.missing_facts)

    def test_p108_does_not_apply_to_semi_trailer_configuration(self) -> None:
        parsed = query("Xe đầu kéo có được đi vào không?")
        parsed.context_facts.append(ContextFact(
            "trailer-config", "vehicle_trailer_configuration", "semi_trailer",
            status="provided", source="user_query", confidence="high",
        ))
        result = evaluate(
            facts_for({"sign_code": "P.108"}), parsed, reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual(result.status, "not_applicable")

    def test_p111_codes_use_the_distinct_vehicle_classes(self) -> None:
        cases = (
            ("P.111a", "Xe gắn máy có được đi vào không?", "P111A_NO_MOPED"),
            ("P.111b", "Xe ba gác máy có được đi vào không?", "P111B_NO_MOTORIZED_TRICYCLE"),
            ("P.111d", "Xích lô có được đi vào không?", "P111D_NO_NON_MOTORIZED_TRICYCLE"),
        )
        for code, question_text, expected_rule_id in cases:
            with self.subTest(code=code):
                result = evaluate(
                    _scoped_sign_facts(code), query(question_text), reviewed_rules(TRAFFIC_RULES),
                )[0]
                self.assertEqual(result.rule_id, expected_rule_id)
                self.assertEqual(result.decision, "PROHIBITED")

    def test_p120_compares_full_vehicle_trailer_and_cargo_length(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "length-sign", role="main", sign_code="P.120", identity_status="identified",
            confidence="high", visual_attributes={"normalized_conditions": {"combination_length_limit_m": 12}},
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "complete scene and confident limit reading"
        }])
        parsed = query("Xe đầu kéo có được đi vào không?")
        parsed.context_facts.extend([
            ContextFact("trailer-config", "vehicle_trailer_configuration", "semi_trailer",
                        status="provided", source="user_query", confidence="high"),
            ContextFact("combination-length", "vehicle_combination_length_m", 14,
                        status="provided", source="user_query", confidence="high"),
        ])
        result = evaluate(facts, parsed, reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual(result.decision, "PROHIBITED")
        self.assertEqual(result.effects[0].value, 12)

    def test_p117_uses_sign_ocr_and_explicit_vehicle_height_without_inferring_permission(self) -> None:
        observation = SignObservation(
            "height-sign", role="main", sign_code="P.117", identity_status="identified",
            ocr_texts=[{"raw_text": "4,2 m", "confidence": "high"}],
            confidence="high", visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True,
            }},
        )
        facts = TrafficFacts(
            "2.0", "scene-1", [observation], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete fixture"}],
        )

        exceeds_limit = evaluate(
            facts, query("Xe tải cao 4,35 m có được đi qua không?"), reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual(exceeds_limit.rule_id, "P117_HEIGHT_LIMIT")
        self.assertEqual(exceeds_limit.decision, "PROHIBITED")
        self.assertEqual(exceeds_limit.effects[0].value, 4.2)

        within_limit = evaluate(
            facts, query("Xe tải cao 4,0 m có được đi qua không?"), reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual(within_limit.decision, "CONDITIONAL")

        missing_vehicle_measurement = evaluate(
            facts, query("Xe tải có được đi qua không?"), reviewed_rules(TRAFFIC_RULES),
        )[0]
        self.assertEqual(missing_vehicle_measurement.decision, "CONDITIONAL")
        self.assertIn("vehicle_height_m", missing_vehicle_measurement.missing_facts)

    def test_emergency_vehicle_does_not_automatically_get_priority_exception(self) -> None:
        p101 = [rule for rule in reviewed_rules(TRAFFIC_RULES)
                if rule.rule_id == "P101_NO_ENTRY_ALL_DIRECTIONS"]
        result = evaluate(
            facts_for({"sign_code": "P.101"}),
            query("Xe cứu hỏa có được đi vào không?"),
            p101,
        )[0]
        self.assertEqual(result.decision, "UNKNOWN")
        self.assertEqual(result.status, "unresolved")
        self.assertIn("priority_vehicle_status", result.missing_facts)

    def test_weight_limit_rule_compares_vehicle_fact_to_observed_sign_value(self) -> None:
        sign_facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            "weight-sign", role="main", sign_code="P.115", identity_status="identified",
            confidence="high", visual_attributes={"normalized_conditions": {"total_weight_limit_t": 10}},
        )], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "complete scene and confident sign reading"
        }])
        rule = reviewed_rules([next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P115_TOTAL_WEIGHT_LIMIT")])
        over_limit_query = query("Xe tải 12 tấn có được đi qua không?")
        over_limit_query.context_facts.append(ContextFact(
            "vehicle-weight", "vehicle_gross_weight_t", 12,
            status="provided", source="user_query", confidence="high",
        ))
        below_limit_query = query("Xe tải 8 tấn có được đi qua không?")
        below_limit_query.context_facts.append(ContextFact(
            "vehicle-weight", "vehicle_gross_weight_t", 8,
            status="provided", source="user_query", confidence="high",
        ))
        over_limit = evaluate(sign_facts, over_limit_query, rule)[0]
        below_limit = evaluate(sign_facts, below_limit_query, rule)[0]
        self.assertEqual(over_limit.decision, "PROHIBITED")
        self.assertEqual(below_limit.decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_sign_constraints", below_limit.missing_facts)

    def test_value_dependent_effect_requires_a_required_condition(self) -> None:
        with self.assertRaisesRegex(ValueError, "without required conditions"):
            TrafficRule(
                "unsafe-missing-value", [], ["P.127"], ["speed"],
                ["motor_vehicle"], "CONDITIONAL",
                effects=[RuleEffect("maximum", action="speed", value_predicate="speed_limit_value")],
            )

    def test_explicit_verified_permission_can_allow_only_when_inventory_is_complete(self) -> None:
        rule = TrafficRule(
            "explicit-route-permission",
            [{"document": "reviewed source", "verification_status": "verified"}],
            ["R.301"], ["going_straight"], ["vehicle"], "ALLOWED",
            verification_status="verified",
            effects=[RuleEffect("permit_action", action="going_straight", scope={"actor_class": "vehicle"})],
        )
        complete = TrafficFacts(
            "2.0", "scene-1", [normalize_payload_to_observation({"sign_code": "R.301"})],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "full controlled scene fixture"}],
        )
        incomplete = TrafficFacts(
            "2.0", "scene-1", [normalize_payload_to_observation({"sign_code": "R.301"})],
            sign_inventory_status="unknown",
        )
        full_result = evaluate(complete, query("Xe ô tô có được đi thẳng không?"), [rule])[0]
        partial_result = evaluate(incomplete, query("Xe ô tô có được đi thẳng không?"), [rule])[0]
        self.assertEqual(full_result.decision, "ALLOWED")
        self.assertEqual(partial_result.decision, "ALLOWED")
        self.assertNotIn("complete_sign_inventory", partial_result.missing_facts)

    def test_r301_directions_are_explicit_and_require_confirmed_spatial_scope(self) -> None:
        r301_rules = [rule for rule in TRAFFIC_RULES if rule.rule_id.startswith("R301_")]
        self.assertEqual(len(r301_rules), 16)
        self.assertTrue(all(rule.verification_status == "unverified" for rule in r301_rules))
        self.assertTrue(all(
            any(condition.predicate == "sign_effective_scope_confirmed" for condition in rule.conditions)
            for rule in r301_rules
        ))

        expected_allowed = {
            "R.301a": ["going_straight"],
            "R.301b": ["turn_right"],
            "R.301c": ["turn_left"],
            "R.301d": ["turn_right"],
            "R.301e": ["turn_left"],
            "R.301f": ["going_straight", "turn_right"],
            "R.301g": ["going_straight", "turn_left", "u_turn"],
            "R.301h": ["turn_left", "turn_right", "u_turn"],
        }
        for code, allowed in expected_allowed.items():
            pair = [rule for rule in r301_rules if code in rule.trigger_sign_codes]
            allow_rule = next(rule for rule in pair if rule.outcome == "ALLOWED")
            prohibit_rule = next(rule for rule in pair if rule.outcome == "PROHIBITED")
            self.assertEqual(allow_rule.actions, allowed)
            self.assertEqual(
                set(prohibit_rule.actions),
                {"going_straight", "turn_left", "turn_right", "u_turn"} - set(allowed),
            )

        def scene(code: str, scope: bool | None) -> TrafficFacts:
            normalized = {"sign_effective_scope_confirmed": scope} if scope is not None else {}
            return TrafficFacts(
                "2.0", "scene-r301", [SignObservation(
                    observation_id="sign-r301",
                    sign_code=code,
                    identity_status="identified",
                    visual_attributes={"normalized_conditions": normalized},
                    confidence="high",
                )],
                sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "all signs in the described junction"}],
            )

        pair = [rule for rule in r301_rules if "R.301a" in rule.trigger_sign_codes]
        left_query = query("Ô tô có được rẽ trái tại nút giao này không?")
        straight_query = query("Ô tô có được đi thẳng tại nút giao này không?")
        unresolved_scope = evaluate(scene("R.301a", None), left_query, reviewed_rules(pair))[0]
        self.assertEqual(unresolved_scope.decision, "PROHIBITED")
        self.assertNotIn("sign_effective_scope_confirmed", unresolved_scope.missing_facts)

        prohibited = evaluate(scene("R.301a", True), left_query, reviewed_rules(pair))[0]
        self.assertEqual(prohibited.decision, "PROHIBITED")
        permitted = evaluate(scene("R.301a", True), straight_query, pair)[0]
        self.assertEqual(permitted.decision, "ALLOWED")
        self.assertNotIn("legal_review_required", permitted.missing_facts)
        approved_permitted = evaluate(scene("R.301a", True), straight_query, reviewed_rules(pair))[0]
        self.assertEqual(approved_permitted.decision, "ALLOWED")

    def test_r304_covers_explicit_actor_classes_and_keeps_crossing_conditional(self) -> None:
        r304_rules = [rule for rule in TRAFFIC_RULES if rule.rule_id.startswith("R304_")]
        self.assertEqual(len(r304_rules), 4)
        self.assertTrue(all(rule.verification_status == "unverified" for rule in r304_rules))
        self.assertTrue(all(
            any(ref.get("document", "").startswith("Luật Trật tự")
                and "Điều 34" in ref.get("section_or_figure", "")
                for ref in rule.legal_basis)
            for rule in r304_rules
        ))

        def scene(scope: bool | None = True) -> TrafficFacts:
            normalized = {"sign_effective_scope_confirmed": scope} if scope is not None else {}
            return TrafficFacts(
                "2.0", "scene-r304", [SignObservation(
                    observation_id="sign-r304", role="main", sign_code="R.304",
                    identity_status="identified",
                    visual_attributes={"normalized_conditions": normalized}, confidence="high",
                )],
                sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "all signs included"}],
            )

        reviewed = reviewed_rules(r304_rules)
        facts = scene()
        pedestrian = evaluate(facts, query("Người đi bộ có được đi vào đường này không?"), reviewed)
        bicycle = evaluate(facts, query("Xe đạp có được đi vào đường này không?"), reviewed)
        motorbike = evaluate(facts, query("Xe máy có được đi vào đường này không?"), reviewed)
        self.assertEqual(pedestrian[0].decision, "ALLOWED")
        self.assertEqual(bicycle[0].decision, "ALLOWED")
        self.assertEqual(motorbike[0].decision, "PROHIBITED")

        crossing = evaluate(
            facts, query("Ô tô có được cắt ngang đường này không?"),
            [rule for rule in reviewed if rule.rule_id.endswith("ABSOLUTE_SAFETY_REVIEW")],
        )
        self.assertEqual(crossing[0].decision, "CONDITIONAL")

        unreviewed = evaluate(
            facts, query("Xe đạp có được đi vào đường này không?"), r304_rules,
        )
        self.assertEqual(unreviewed[0].decision, "ALLOWED")
        missing_scope = evaluate(
            scene(None), query("Xe đạp có được đi vào đường này không?"), reviewed,
        )
        self.assertEqual(missing_scope[0].decision, "CONDITIONAL")

    def test_common_prohibition_variants_are_source_backed_and_review_gated(self) -> None:
        selected_codes = {
            "P.103a", "P.103b", "P.103c", "P.106a", "P.106b", "P.106c", "P.107",
            "P.110a", "P.110b", "P.111a", "P.111b", "P.111c", "P.111d",
            "P.123a", "P.123b", "P.124a", "P.124b", "P.124c", "P.124d", "P.124e", "P.124f",
        }
        rules = [rule for rule in TRAFFIC_RULES
                 if set(rule.trigger_sign_codes).intersection(selected_codes)]
        registered = {code for rule in rules for code in rule.trigger_sign_codes}
        self.assertEqual(registered, selected_codes)
        self.assertTrue(all(rule.verification_status == "unverified" for rule in rules))
        self.assertTrue(all(
            any("Phụ lục" in ref.get("section_or_figure", "")
                and ref.get("source_url", "").startswith("https://datafiles.chinhphu.vn/")
                for ref in rule.legal_basis)
            for rule in rules
        ))
        self.assertTrue(all(
            any("Điều 34" in ref.get("section_or_figure", "") for ref in rule.legal_basis)
            for rule in rules
        ))

        def scene(code: str, **normalized) -> TrafficFacts:
            return TrafficFacts(
                "2.0", "prohibition-scene", [SignObservation(
                    observation_id=f"sign-{code}", role="main", sign_code=code,
                    identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True, **normalized,
                    }}, confidence="high",
                )], sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "all signs included"}],
            )

        approved = reviewed_rules(rules)
        p103_car = evaluate(scene("P.103a"), query("Ô tô có được đi vào không?"),
                            [rule for rule in approved if "P.103a" in rule.trigger_sign_codes])
        p103_motorbike = evaluate(scene("P.103a"), query("Xe máy có được đi vào không?"),
                                  [rule for rule in approved if "P.103a" in rule.trigger_sign_codes])
        self.assertEqual(p103_car[0].decision, "PROHIBITED")
        self.assertEqual(p103_motorbike, [])

        left_turn = evaluate(scene("P.123a"), query("Xe đạp có được rẽ trái không?"),
                             [rule for rule in approved if "P.123a" in rule.trigger_sign_codes])
        u_turn = evaluate(scene("P.123a"), query("Xe đạp có được quay đầu không?"),
                          [rule for rule in approved if "P.123a" in rule.trigger_sign_codes])
        self.assertEqual(left_turn[0].decision, "PROHIBITED")
        self.assertEqual(u_turn, [])

        pushed_bicycle = query("Người dắt xe đạp có được đi qua không?").query_intent.intents[0]
        self.assertEqual(pushed_bicycle.actor.value, "pedestrian")
        p110_push = evaluate(scene("P.110a"), query("Người dắt xe đạp có được đi qua không?"),
                             [rule for rule in approved if "P.110a" in rule.trigger_sign_codes])
        self.assertEqual(p110_push, [])

    def test_r412_lane_rules_cover_variants_exceptions_and_unknown_markings(self) -> None:
        codes = {f"R.412{suffix}" for suffix in "abcdefgh"}
        lane_rules = [rule for rule in TRAFFIC_RULES
                      if set(rule.trigger_sign_codes).intersection(codes)]
        self.assertEqual(len(lane_rules), 21)
        self.assertEqual({code for rule in lane_rules for code in rule.trigger_sign_codes}, codes)
        self.assertTrue(all(rule.verification_status == "unverified" for rule in lane_rules))
        self.assertTrue(all(
            any(ref.get("source_url") == "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/51-bgtvt-kem.pdf"
                and "D.14" in ref.get("section_or_figure", "") for ref in rule.legal_basis)
            for rule in lane_rules
        ))

        def scene(code: str, normalized: dict | None = None,
                  extra_observations: list[SignObservation] | None = None) -> TrafficFacts:
            signs = [SignObservation(
                f"sign-{code}", role="main", sign_code=code, identity_status="identified",
                visual_attributes={"normalized_conditions": {
                    "sign_effective_scope_confirmed": True, **(normalized or {}),
                }}, confidence="high",
            )]
            signs.extend(extra_observations or [])
            return TrafficFacts(
                "2.0", "r412-scene", signs, sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "all signs in scene included"}],
            )

        reviewed = reviewed_rules(lane_rules)
        r412d_rules = [r for r in reviewed if "R.412d" in r.trigger_sign_codes]
        motorbike = evaluate(scene("R.412d"), query("Xe máy có được đi vào làn này không?"), r412d_rules)
        car = evaluate(scene("R.412d"), query("Ô tô có được đi vào làn này không?"), r412d_rules)
        self.assertEqual(motorbike[0].decision, "ALLOWED")
        self.assertEqual(car[0].decision, "PROHIBITED")

        r412e_rules = [r for r in reviewed if "R.412e" in r.trigger_sign_codes]
        dashed = evaluate(
            scene("R.412e", {"bus_lane_marking": "broken"}),
            query("Ô tô có được đi vào làn này không?"), r412e_rules,
        )
        solid = evaluate(
            scene("R.412e", {"bus_lane_marking": "solid"}),
            query("Ô tô có được đi vào làn này không?"), r412e_rules,
        )
        unknown_marking = evaluate(
            scene("R.412e"), query("Ô tô có được đi vào làn này không?"), r412e_rules,
        )
        self.assertEqual(dashed[0].decision, "ALLOWED")
        self.assertTrue(any(effect.effect_type == "require_action" and effect.action == "yield_to_bus"
                            for effect in dashed[0].effects))
        solid_prohibition_rule = next(
            rule.rule_id for rule in r412e_rules
            if "PROHIBIT_OTHER_VEHICLE_ON_SOLID_BUS_LANE" in rule.rule_id
        )
        solid_prohibition = next(item for item in solid if item.rule_id == solid_prohibition_rule)
        self.assertEqual(solid_prohibition.decision, "PROHIBITED")
        self.assertTrue(unknown_marking)
        self.assertTrue(all(item.decision == "CONDITIONAL" for item in unknown_marking))

        r412c_rules = [r for r in reviewed if "R.412c" in r.trigger_sign_codes]
        limited_scene = scene("R.412c", {
            "payload_limit_present": True, "payload_limit_relation": "lt", "payload_limit_t": 3.5,
        })
        under = TrafficFacts(
            "2.0", "r412-under", limited_scene.sign_observations,
            context_facts=[ContextFact("payload", "vehicle_payload_t", 3.0,
                                      status="provided", source="user", confidence="high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        over = TrafficFacts(
            "2.0", "r412-over", limited_scene.sign_observations,
            context_facts=[ContextFact("payload", "vehicle_payload_t", 4.0,
                                      status="provided", source="user", confidence="high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        self.assertEqual(evaluate(under, query("Xe tải có được vào làn này không?"), r412c_rules)[0].decision, "ALLOWED")
        over_evaluations = evaluate(over, query("Xe tải có được vào làn này không?"), r412c_rules)
        over_prohibition = next(
            item for item in over_evaluations
            if "PROHIBIT_AT_OR_ABOVE_DISPLAYED_PAYLOAD_LIMIT" in item.rule_id
        )
        self.assertEqual(over_prohibition.decision, "PROHIBITED")

        unrelated_supplement = SignObservation(
            "s504", role="supplementary", sign_code="S.504", identity_status="identified", confidence="high",
        )
        linked_scene = scene("R.412b", extra_observations=[unrelated_supplement])
        r412b_rules = [r for r in reviewed if "R.412b" in r.trigger_sign_codes]
        unresolved_link = evaluate(
            linked_scene, query("Ô tô có được đi vào làn này không?"), r412b_rules,
        )[0]
        self.assertEqual(unresolved_link.decision, "CONDITIONAL")
        self.assertIn("supplementary_sign_association", unresolved_link.missing_facts)

    def test_p106b_compares_vehicle_certificate_capacity_and_never_defaults_to_allowed(self) -> None:
        code_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if "P.106b" in rule.trigger_sign_codes
        ])
        fact = ContextFact(
            "vehicle-capacity", "vehicle_payload_t", 7,
            status="provided", source="user", confidence="high",
        )
        sign = SignObservation(
            "p106b-sign", role="main", sign_code="P.106b", identity_status="identified",
            visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True, "payload_limit_t": 5,
            }}, confidence="high",
        )
        facts = TrafficFacts(
            "2.0", "payload-scene", [sign], context_facts=[fact],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "all signs included"}],
        )
        above = evaluate(facts, query("Xe tải có được đi vào không?"), code_rules)
        self.assertEqual(above[0].decision, "PROHIBITED")

        missing_capacity = TrafficFacts(
            "2.0", "payload-scene", [sign], sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "all signs included"}],
        )
        unresolved = evaluate(missing_capacity, query("Xe tải có được đi vào không?"), code_rules)
        self.assertEqual(unresolved[0].decision, "CONDITIONAL")
        self.assertIn("vehicle_payload_t", unresolved[0].missing_facts)

    def test_complete_inventory_claim_without_evidence_is_downgraded(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1", sign_inventory_status="complete",
        )
        self.assertEqual(facts.sign_inventory_status, "unknown")
        self.assertIn("sign_inventory_complete_without_evidence", facts.unresolved_items)

    def test_penalty_intent_does_not_fire_permission_rule(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.131a"}), query("Mức phạt khi đỗ xe là bao nhiêu?")
        )
        self.assertEqual(result, [])

    def test_ambiguous_identity_does_not_produce_confident_match(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [SignObservation(
            observation_id="crop-1", identity_status="ambiguous",
            candidates=[{"sign_code": "P.131b", "confidence": 0.48}], confidence="low",
        )])
        result = evaluate(facts, query("Xe ô tô có được đỗ không?"))[0]
        self.assertEqual(result.status, "unresolved")
        self.assertEqual(result.decision, "UNKNOWN")

    def test_explicit_target_limits_rule_matching_to_referenced_observation(self) -> None:
        facts = TrafficFacts("2.0", "scene-1", [
            normalize_payload_to_observation({"sign_code": "P.131a", "image_name": "crop-a.jpg"}),
            normalize_payload_to_observation({"sign_code": "P.123a", "image_name": "crop-b.jpg"}),
        ], sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "fixture includes both signs"
        }])
        parsed = query("Xe ô tô có được đỗ không?")
        parsed.query_intent.intents[0].target.kind = "sign_observation"
        parsed.query_intent.intents[0].target.observation_ids = ["crop-b.jpg"]
        self.assertEqual(evaluate(facts, parsed), [])

    def test_supplementary_fact_is_scoped_to_its_main_sign(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.131a", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation(
                    "panel", role="supplementary", sign_code="S.508a", identity_status="identified",
                    confidence="high",
                    ocr_texts=[{"raw_text": "6:00-9:00", "confidence": "high"}],
                    visual_attributes={"normalized_conditions": {"time_window": {"start": "06:00", "end": "09:00"}}},
                ),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "fixture includes main and supplementary sign"}],
        )
        rule = next(
            item for item in TRAFFIC_RULES if item.rule_id == "P131A_NO_PARKING_DURING_S508"
        )
        evaluation = evaluate(
            facts,
            query("Xe ô tô có được đỗ không?", datetime(
                2026, 10, 4, 8, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")
            )),
            reviewed_rules([rule]),
        )[0]
        self.assertEqual(evaluation.decision, "PROHIBITED")
        self.assertTrue(any(step["predicate"] == "linked_s508_temporal_scope_active" for step in evaluation.trace))

    def test_low_confidence_supplementary_link_blocks_definitive_result(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.130", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation("panel", role="supplementary", ocr_texts=[{"raw_text": "6:00-9:00"}]),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "low")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "both signs are present"}],
        )
        rule = reviewed_rules(TRAFFIC_RULES)[0]
        evaluation = evaluate(facts, query("Xe ô tô có được đỗ không?"), [rule])[0]
        self.assertEqual(evaluation.decision, "CONDITIONAL")
        self.assertIn("supplementary_sign_association", evaluation.missing_facts)

    def test_s508_time_window_can_be_evaluated_against_system_time(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.130", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation(
                    "panel", role="supplementary", sign_code="S.508a", identity_status="identified",
                    ocr_texts=[{"raw_text": "06:00-09:00", "confidence": "high"}],
                ),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
        )
        timed_rule = next(
            rule for rule in TRAFFIC_RULES
            if rule.rule_id == "P130_NO_STOPPING_NO_PARKING_DURING_S508"
        )
        rules = reviewed_rules([timed_rule])
        inside = evaluate(
            facts,
            query("Xe ô tô có được đỗ không?", datetime(2026, 10, 4, 8, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))),
            rules,
        )[0]
        outside = evaluate(facts, query("Xe ô tô có được đỗ không?"), rules)[0]
        boundary = evaluate(
            facts,
            query("Xe ô tô có được đỗ không?", datetime(2026, 10, 4, 9, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))),
            rules,
        )[0]
        self.assertEqual(inside.decision, "PROHIBITED")
        self.assertEqual(outside.status, "not_applicable")
        self.assertEqual(boundary.decision, "CONDITIONAL")
        self.assertIn("linked_s508_temporal_scope_active", boundary.missing_facts)

    def test_s508_day_parity_compares_panel_condition_to_current_day_fact(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.130", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation(
                    "panel", role="supplementary", sign_code="S.508b", identity_status="identified",
                    ocr_texts=[{"raw_text": "ngày chẵn", "confidence": "high"}],
                ),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
        )
        parity_rule = next(
            rule for rule in TRAFFIC_RULES
            if rule.rule_id == "P130_NO_STOPPING_NO_PARKING_DURING_S508"
        )
        result = evaluate(
            facts, query("Xe ô tô có được đỗ không?"), reviewed_rules([parity_rule])
        )[0]
        self.assertEqual(result.decision, "PROHIBITED")

    def test_s508_low_confidence_ocr_keeps_p130_conditional(self) -> None:
        facts = TrafficFacts(
            "2.0", "low-s508-ocr",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.130", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation(
                    "panel", role="supplementary", sign_code="S.508a", identity_status="identified",
                    ocr_texts=[{"raw_text": "06:00-09:00", "confidence": "low"}],
                ),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete stack"}],
        )
        timed_rule = next(
            rule for rule in TRAFFIC_RULES
            if rule.rule_id == "P130_NO_STOPPING_NO_PARKING_DURING_S508"
        )
        result = evaluate(
            facts,
            query("Xe ô tô có được đỗ không?", datetime(
                2026, 10, 4, 8, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")
            )),
            reviewed_rules([timed_rule]),
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertIn("linked_s508_temporal_scope_active", result.missing_facts)

    def test_linked_s508_blocks_a_prohibition_rule_that_does_not_model_its_time_scope(self) -> None:
        facts = TrafficFacts(
            "2.0", "s508-unmodeled-sign",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.103a", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation(
                    "panel", role="supplementary", sign_code="S.508a", identity_status="identified",
                    ocr_texts=[{"raw_text": "06:00-09:00", "confidence": "high"}],
                ),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
        )
        p103_rules = reviewed_rules([
            rule for rule in TRAFFIC_RULES if "P.103a" in rule.trigger_sign_codes
        ])
        result = evaluate(
            facts,
            query("Xe ô tô có được đi vào không?", datetime(
                2026, 10, 4, 8, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")
            )),
            p103_rules,
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertIn("supplementary_time_scope_not_evaluated", result.missing_facts)

    def test_linked_unmodeled_scope_panels_force_conditional(self) -> None:
        cases = [
            ("P.130", "S.501", "Xe ô tô có được đỗ ở đây không?", "30 m",
             "supplementary_s501_scope_not_evaluated"),
            ("P.130", "S.502", "Xe ô tô có được đỗ ở đây không?", "30 m",
             "supplementary_s502_scope_not_evaluated"),
            ("P.130", "S.503a", "Xe ô tô có được đỗ ở đây không?", "",
             "supplementary_s503a_scope_not_evaluated"),
            ("P.130", "S.504", "Xe ô tô có được đỗ ở đây không?", "",
             "supplementary_s504_scope_not_evaluated"),
            ("P.103a", "S.505a", "Xe tải có được đi vào không?", "",
             "supplementary_s505a_scope_not_evaluated"),
            ("W.239", "S.509a", "Xe tải có được đi qua không?", "4,2 m",
             "supplementary_s509a_scope_not_evaluated"),
            ("P.130", "S.509b", "Xe ô tô có được đỗ ở đây không?", "CẤM ĐỖ XE",
             "supplementary_s509b_scope_not_evaluated"),
            ("P.130", "S.G.11a", "Xe ô tô có được đỗ ở đây không?", "",
             "supplementary_sg11a_scope_not_evaluated"),
            ("P.130", "S.G.12b", "Xe ô tô có được đỗ ở đây không?", "",
             "supplementary_sg12b_scope_not_evaluated"),
            ("P.103a", "S.H.6", "Xe ô tô có được đi vào không?", "NGOẠI LỆ / EXCEPT",
             "supplementary_sh6_scope_not_evaluated"),
        ]
        for main_code, panel_code, question_text, panel_text, missing_fact in cases:
            with self.subTest(panel_code=panel_code):
                main = SignObservation(
                    "main", role="main", sign_code=main_code, identity_status="identified",
                    confidence="high", visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }},
                )
                panel = SignObservation(
                    "panel", role="supplementary", sign_code=panel_code, identity_status="identified",
                    confidence="high", ocr_texts=([{"raw_text": panel_text, "confidence": "high"}]
                                                   if panel_text else []),
                )
                facts = TrafficFacts(
                    "2.0", f"scope-{panel_code}", [main, panel],
                    sign_relations=[SignRelation("panel-main", "supplements", "panel", "main", "high")],
                    sign_inventory_status="complete",
                    sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
                )
                rules = reviewed_rules([rule for rule in TRAFFIC_RULES if main_code in rule.trigger_sign_codes])
                result = evaluate(facts, query(question_text), rules)
                self.assertTrue(result)
                self.assertTrue(any(missing_fact in item.missing_facts for item in result))
                self.assertTrue(all(item.decision in {"CONDITIONAL", "UNKNOWN"} for item in result))
                self.assertFalse(any(item.decision in {"ALLOWED", "PROHIBITED"} for item in result))
                self.assertTrue(any(
                    step.get("step") == "supplementary_scope_evaluation"
                    for item in result for step in item.trace
                ))

    def test_sh6_exception_scope_exempts_only_resolved_actor_without_granting_permission(self) -> None:
        def run(actor_question: str, panel_text: str | None):
            main = SignObservation(
                "main", role="main", sign_code="P.101", identity_status="identified", confidence="high",
            )
            panel = SignObservation(
                "sh6", role="supplementary", sign_code="S.H.6", identity_status="identified",
                confidence="high",
                ocr_texts=([{"raw_text": panel_text, "confidence": "high"}] if panel_text else []),
            )
            facts = TrafficFacts(
                "2.0", "p101-sh6", [main, panel],
                sign_relations=[SignRelation("sh6-main", "supplements", "sh6", "main", "high")],
                sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
            )
            rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "P.101" in rule.trigger_sign_codes])
            return evaluate(facts, query(actor_question), rules)[0]

        exempt_actor = run("Xe buýt có được đi vào không?", "XE BUÝT - NGOẠI LỆ / EXCEPT")
        self.assertEqual(exempt_actor.decision, "UNKNOWN")
        self.assertEqual(exempt_actor.status, "not_applicable")
        self.assertNotIn("other_applicable_sign_constraints", exempt_actor.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in exempt_actor.effects))

        non_exempt_actor = run("Xe tải có được đi vào không?", "XE BUÝT - NGOẠI LỆ / EXCEPT")
        self.assertEqual(non_exempt_actor.decision, "PROHIBITED")
        self.assertTrue(any(
            step.get("state") == "resolved_not_applicable_to_actor"
            for step in non_exempt_actor.trace
        ))

        unresolved_group = run("Xe buýt có được đi vào không?", None)
        self.assertEqual(unresolved_group.decision, "CONDITIONAL")
        self.assertIn("supplementary_sh6_scope_not_evaluated", unresolved_group.missing_facts)

    def test_s505a_applies_primary_sign_only_to_visually_resolved_vehicle_class(self) -> None:
        def run(actor_question: str, resolved_classes: list[str] | None):
            main = SignObservation(
                "main", role="main", sign_code="P.101", identity_status="identified", confidence="high",
            )
            panel = SignObservation(
                "s505a", role="supplementary", sign_code="S.505a", identity_status="identified",
                confidence="high",
            )
            if resolved_classes is not None:
                panel.visual_attributes["normalized_conditions"] = {
                    "s505a_vehicle_scope_resolved": True,
                    "s505a_vehicle_classes": resolved_classes,
                }
            facts = TrafficFacts(
                "2.0", "p101-s505a", [main, panel],
                sign_relations=[SignRelation("s505a-main", "supplements", "s505a", "main", "high")],
                sign_inventory_status="complete",
                sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
            )
            rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "P.101" in rule.trigger_sign_codes])
            return evaluate(facts, query(actor_question), rules)[0]

        matching = run("Xe buýt có được đi vào không?", ["bus"])
        self.assertEqual(matching.decision, "PROHIBITED")
        self.assertTrue(any(step.get("state") == "resolved_applies_to_actor" for step in matching.trace))

        nonmatching = run("Xe tải có được đi vào không?", ["bus"])
        self.assertEqual(nonmatching.decision, "UNKNOWN")
        self.assertEqual(nonmatching.status, "not_applicable")
        self.assertNotIn("other_applicable_sign_constraints", nonmatching.missing_facts)
        self.assertFalse(any(effect.effect_type == "permit_action" for effect in nonmatching.effects))

        unresolved = run("Xe buýt có được đi vào không?", None)
        self.assertEqual(unresolved.decision, "CONDITIONAL")
        self.assertIn("supplementary_s505a_scope_not_evaluated", unresolved.missing_facts)

    def test_s506a_direction_scope_is_explicitly_part_of_i401_rule(self) -> None:
        main = SignObservation(
            "main", role="main", sign_code="I.401", identity_status="identified", confidence="high",
            visual_attributes={"normalized_conditions": {"sign_effective_scope_confirmed": True}},
        )
        panel = SignObservation(
            "panel", role="supplementary", sign_code="S.506a", identity_status="identified", confidence="high",
        )
        facts = TrafficFacts(
            "2.0", "s506a-linked", [main, panel],
            sign_relations=[SignRelation("panel-main", "supplements", "panel", "main", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete sign stack"}],
        )
        rules = reviewed_rules([rule for rule in TRAFFIC_RULES if "I.401" in rule.trigger_sign_codes])
        result = evaluate(facts, query("Xe ô tô có được rẽ trái ở giao lộ này không?"), rules)
        self.assertTrue(result)
        self.assertTrue(all("i401_priority_road_direction_resolved" in item.missing_facts for item in result))
        self.assertFalse(any("supplementary_s506a_scope_not_evaluated" in item.missing_facts for item in result))
        self.assertTrue(all(item.decision == "CONDITIONAL" for item in result))

    def test_linked_s505_bridge_panels_block_unqualified_p106a_prohibition(self) -> None:
        facts = TrafficFacts(
            "2.0", "s505-bridge-scope",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.106a", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation(
                    "s505b", role="supplementary", sign_code="S.505b",
                    identity_status="identified", ocr_texts=[{
                        "raw_text": "00 t", "confidence": "high",
                    }], confidence="high",
                ),
                SignObservation(
                    "s505c", role="supplementary", sign_code="S.505c",
                    identity_status="identified", ocr_texts=[{
                        "raw_text": "Trục đơn 10 t", "confidence": "high",
                    }], confidence="high",
                ),
            ],
            sign_relations=[
                SignRelation("rel-b", "supplements", "s505b", "main", "high"),
                SignRelation("rel-c", "supplements", "s505c", "main", "high"),
            ],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{
                "source": "test_fixture", "evidence": "complete bridge sign stack",
            }],
        )
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P106A_NO_TRUCK_TRACTOR_SPECIAL_VEHICLE")
        result = evaluate(
            facts,
            query("Xe tải có được đi vào không?"),
            reviewed_rules([rule]),
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertIn("supplementary_bridge_weight_scope_not_evaluated", result.missing_facts)
        self.assertTrue(any(
            step.get("step") == "supplementary_bridge_weight_scope"
            for step in result.trace
        ))
        self.assertTrue(any("F.6" in basis.get("section_or_figure", "") for basis in result.legal_basis))
        self.assertTrue(any("F.7" in basis.get("section_or_figure", "") for basis in result.legal_basis))

    def test_s505c_prohibits_over_limit_axle_group_but_never_grants_bridge_permission(self) -> None:
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P106A_S505C_AXLE_GROUP_LOAD_LIMIT")
        panel = SignObservation(
            "s505c", role="supplementary", sign_code="S.505c", identity_status="identified",
            confidence="high", ocr_texts=[{
                "raw_text": "Trục đơn 10 t", "confidence": "high",
            }],
        )
        main = SignObservation(
            "main", role="main", sign_code="P.106a", identity_status="identified",
            confidence="high", visual_attributes={"normalized_conditions": {
                "sign_effective_scope_confirmed": True,
            }},
        )
        complete_inventory = {
            "sign_inventory_status": "complete",
            "sign_inventory_evidence": [{"source": "test_fixture", "evidence": "complete bridge sign stack"}],
        }

        def evaluate_with(loads: dict | None):
            context = []
            if loads is not None:
                context.extend([
                    ContextFact(
                        "axle-inventory", "vehicle_axle_inventory_complete", True,
                        status="provided", source="vehicle_record", confidence="high",
                    ),
                    ContextFact(
                        "axle-loads", "vehicle_axle_group_loads_t", loads,
                        status="provided", source="vehicle_record", confidence="high",
                    ),
                ])
            facts = TrafficFacts(
                "2.0", "s505c-rule", [main, panel], context_facts=context,
                sign_relations=[SignRelation("rel-c", "supplements", "s505c", "main", "high")],
                **complete_inventory,
            )
            return evaluate(facts, query("Xe tải có được đi vào không?"), reviewed_rules([rule]))[0]

        over = evaluate_with({"single_axle": [10.1]})
        self.assertEqual(over.decision, "PROHIBITED")
        self.assertTrue(any("F.7" in basis.get("section_or_figure", "") for basis in over.legal_basis))

        within = evaluate_with({"single_axle": [9.9]})
        self.assertEqual(within.decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_sign_constraints", within.missing_facts)

        missing = evaluate_with(None)
        self.assertEqual(missing.decision, "CONDITIONAL")
        self.assertNotIn("vehicle_axle_inventory_complete", missing.missing_facts)

        unlisted_group = evaluate_with({"tandem_axle": [18.0]})
        self.assertEqual(unlisted_group.decision, "CONDITIONAL")
        self.assertIn("vehicle_axle_group_loads_t", unlisted_group.missing_facts)

    def test_s505b_uses_row_matched_vehicle_record_weight_without_granting_permission(self) -> None:
        rule = next(rule for rule in TRAFFIC_RULES if rule.rule_id == "P106A_S505B_ROW_GROSS_WEIGHT_LIMIT")

        def evaluate_case(limit_ocr: list[dict], permitted_weight: float):
            main = SignObservation(
                "main", role="main", sign_code="P.106a", identity_status="identified",
                confidence="high", visual_attributes={"normalized_conditions": {
                    "sign_effective_scope_confirmed": True,
                }},
            )
            panel = SignObservation(
                "s505b", role="supplementary", sign_code="S.505b", identity_status="identified",
                confidence="high", ocr_texts=limit_ocr,
            )
            facts = TrafficFacts(
                "2.0", "s505b-row", [main, panel],
                context_facts=[
                    ContextFact(
                        "row-match", "vehicle_s505b_row_match_confirmed", True,
                        status="provided", source="vehicle_to_sign_symbol_match", confidence="high",
                    ),
                    ContextFact(
                        "row-index", "vehicle_s505b_row_index", 2,
                        status="provided", source="vehicle_to_sign_symbol_match", confidence="high",
                    ),
                    ContextFact(
                        "permitted-gross-weight", "vehicle_permitted_gross_weight_t", permitted_weight,
                        status="provided", source="vehicle_registration", confidence="high",
                    ),
                ],
                sign_relations=[SignRelation("rel-b", "supplements", "s505b", "main", "high")],
                sign_inventory_status="complete",
                sign_inventory_evidence=[{
                    "source": "test_fixture", "evidence": "complete bridge sign stack",
                }],
            )
            return evaluate(facts, query("Xe tải có được đi vào không?"), reviewed_rules([rule]))[0]

        over = evaluate_case([
            {"raw_text": "18 t", "table_row_index": 1, "confidence": "high"},
            {"raw_text": "20 t", "table_row_index": 2, "confidence": "high"},
        ], 20.1)
        self.assertEqual(over.decision, "PROHIBITED")
        self.assertTrue(any("F.6" in basis.get("section_or_figure", "") for basis in over.legal_basis))

        within = evaluate_case([
            {"raw_text": "18 t", "table_row_index": 1, "confidence": "high"},
            {"raw_text": "20 t", "table_row_index": 2, "confidence": "high"},
        ], 20.0)
        self.assertEqual(within.decision, "CONDITIONAL")
        self.assertNotIn("other_applicable_sign_constraints", within.missing_facts)

        rowless_ocr = evaluate_case([{"raw_text": "20 t", "confidence": "high"}], 20.1)
        self.assertEqual(rowless_ocr.decision, "CONDITIONAL")
        self.assertIn("vehicle_permitted_gross_weight_t", rowless_ocr.missing_facts)

    def test_standalone_s507_is_not_mistaken_for_an_unattached_scope_panel(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1",
            sign_observations=[
                SignObservation(
                    "main", role="main", sign_code="P.130", identity_status="identified",
                    visual_attributes={"normalized_conditions": {
                        "sign_effective_scope_confirmed": True,
                    }}, confidence="high",
                ),
                SignObservation("turn", role="supplementary", sign_code="S.507", identity_status="identified"),
            ],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete scene"}],
        )
        result = evaluate(facts, query("Xe ô tô có được đỗ không?"), reviewed_rules(TRAFFIC_RULES))[0]
        self.assertEqual(result.decision, "PROHIBITED")

    def test_non_matching_action_and_unknown_sign_do_not_fire(self) -> None:
        facts = facts_for({"sign_code": "P.131a"})
        self.assertEqual(evaluate(facts, query("Xe ô tô có được rẽ trái không?")), [])
        self.assertEqual(evaluate(facts_for({"sign_code": "P.999"}), query("Xe ô tô có được đỗ không?")), [])


if __name__ == "__main__":
    unittest.main()
