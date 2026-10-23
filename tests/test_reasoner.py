import unittest
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.symbolic.fact_schema import ContextFact, SignRelation, TrafficFacts
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.reasoner import (
    build_symbolic_reasoning_context,
    decide_symbolically,
    reason_over_facts,
)
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation
from src.core.symbolic.traffic_rules import TRAFFIC_RULES


def reviewed_rules():
    return tuple(replace(
        rule,
        verification_status="verified",
        legal_basis=[{**basis, "verification_status": "verified"} for basis in rule.legal_basis],
    ) for rule in TRAFFIC_RULES)


def facts(code: str) -> TrafficFacts:
    observation = normalize_payload_to_observation({
        "sign_code": code,
        "image_name": "crop-1.jpg",
        "law_id": "QCVN 41:2024/BGTVT",
        "article_id": "31",
    })
    observation.visual_attributes["normalized_conditions"] = {
        "sign_effective_scope_confirmed": True,
    }
    observation.confidence = "high"
    return TrafficFacts("2.0", "scene-1", [observation], sign_inventory_status="complete", sign_inventory_evidence=[{
        "source": "test_fixture", "evidence": "fixture explicitly includes all scene signs"
    }])


def query(text: str):
    fixed_now = datetime(2026, 10, 5, 8, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    return parse_query_intent_deterministic(text, now=fixed_now)


class ReasonerTest(unittest.TestCase):
    def test_p102_permission_requires_location_fact_before_allowing(self) -> None:
        decision = decide_symbolically(
            facts("P.102"),
            "Người đi bộ có được đi ở đây không?",
            rules=reviewed_rules(),
        )
        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertIn("pedestrian_location", decision.rule_evaluations[0]["missing_facts"])

    def test_p102_pedestrian_permission_requires_sidewalk_or_road_shoulder_fact(self) -> None:
        parsed = query("Người đi bộ có được đi trên vỉa hè ở đây không?")
        self.assertTrue(any(
            item.predicate == "pedestrian_location" and item.value == "sidewalk"
            for item in parsed.context_facts
        ))
        decision = decide_symbolically(facts("P.102"), parsed, rules=reviewed_rules())
        self.assertEqual(decision.decision, "ALLOWED")
        self.assertTrue(any("permit_action" in str(item) for item in decision.rule_evaluations))

    def test_p102_carriageway_is_outside_the_rule_permission_scope(self) -> None:
        parsed = query("Người đi bộ có được đi ở đây không?")
        parsed.context_facts.append(ContextFact(
            fact_id="location-2", predicate="pedestrian_location", value="carriageway",
            status="provided", source="test_fixture", confidence="high",
        ))
        decision = decide_symbolically(facts("P.102"), parsed, rules=reviewed_rules())
        self.assertEqual(decision.decision, "UNKNOWN")
        self.assertEqual(decision.rule_evaluations[0]["status"], "not_applicable")

    def test_reasoner_returns_evaluation_trace(self) -> None:
        trace = reason_over_facts(facts("P.130"), "Xe ô tô có được phép đỗ tại đây không?", rules=reviewed_rules())
        payload = trace.to_dict()
        self.assertEqual(payload["decision"]["decision"], "PROHIBITED")
        self.assertIn("P130_NO_STOPPING_NO_PARKING", payload["decision"]["rules_fired"])
        self.assertTrue(set(payload["decision"]["rules_fired"]))
        self.assertTrue(all(
            item["decision"] in {"PROHIBITED", "ALLOWED"}
            for item in payload["rule_evaluations"]
            if item["status"] in {"applied", "incomplete"}
        ))
        self.assertEqual(payload["rule_evaluations"][0]["rule_id"], "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(payload["steps"][1]["name"], "evaluate_rules")

    def test_permission_uses_date_of_explicit_query_time_for_calendar_parity(self) -> None:
        decision = decide_symbolically(
            facts("P.131b"),
            "Xe ô tô có được đỗ lúc 7 giờ ngày 17/10/2026 không?",
            rules=reviewed_rules(),
        )
        self.assertEqual(decision.decision, "PROHIBITED")
        self.assertEqual(decision.rule_evaluations[0]["missing_facts"], [])
        self.assertTrue(any(
            item.get("predicate") == "calendar_parity" and item.get("value") == "odd_days"
            for item in decision.rule_evaluations[0]["trace"]
        ))

    def test_in_scope_permission_rule_preserves_missing_context_and_trace(self) -> None:
        decision = decide_symbolically(
            facts("I.401"),
            "Xe ô tô có được đi thẳng qua giao lộ này không?",
            rules=reviewed_rules(),
        )
        evaluation = next(
            item for item in decision.rule_evaluations
            if item["rule_id"] == "I401_PRIORITY_ROAD_UNCONTROLLED_INTERSECTION_MEANING"
        )
        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertEqual(evaluation["decision"], "CONDITIONAL")

    def test_permission_uses_default_system_clock_if_user_omits_time(self) -> None:
        decision = decide_symbolically(facts("P.131b"), query("Xe ô tô có được đỗ không?"), rules=reviewed_rules())
        self.assertEqual(decision.decision, "PROHIBITED")  # injected date is an odd day
        self.assertTrue(decision.rules_fired)

    def test_opposite_parity_sign_is_not_a_scene_wide_permission_proof(self) -> None:
        decision = decide_symbolically(facts("P.131b"), query("Ngày chẵn, xe ô tô có được đỗ không?"), rules=reviewed_rules())
        self.assertEqual(decision.decision, "UNKNOWN")
        self.assertTrue(all(item["status"] == "not_applicable" for item in decision.rule_evaluations))

    def test_unreviewed_rules_with_basis_can_execute(self) -> None:
        decision = decide_symbolically(facts("P.130"), "Xe ô tô có được đỗ tại đây không?")
        self.assertEqual(decision.decision, "PROHIBITED")
        self.assertEqual(decision.rule_evaluations[0]["verification_status"], "unverified")

    def test_conflicting_signs_are_reported_per_intent(self) -> None:
        observations = [
            normalize_payload_to_observation({"sign_code": "P.131b", "image_name": "a.jpg"}),
            normalize_payload_to_observation({"sign_code": "P.131c", "image_name": "b.jpg"}),
        ]
        for observation in observations:
            observation.visual_attributes["normalized_conditions"] = {
                "sign_effective_scope_confirmed": True,
            }
            observation.confidence = "high"
        both = TrafficFacts("2.0", "scene-1", observations,
            sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "fixture explicitly includes both signs"
        }])
        decision = decide_symbolically(both, query("Ngày lẻ, xe ô tô có được đỗ không?"), rules=reviewed_rules())
        self.assertIn("PROHIBITED", {item["decision"] for item in decision.intent_evaluations})

    def test_unrelated_sign_does_not_block_matching_turn_left_rule(self) -> None:
        relevant = normalize_payload_to_observation({"sign_code": "R.301e", "image_name": "turn-left.jpg"})
        relevant.visual_attributes["normalized_conditions"] = {"sign_effective_scope_confirmed": True}
        relevant.confidence = "high"
        unrelated = normalize_payload_to_observation({"sign_code": "P.130", "image_name": "no-parking.jpg"})
        unrelated.confidence = "high"
        scene = TrafficFacts(
            "2.0", "two-signs", [relevant, unrelated],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "both signs listed"}],
        )

        decision = decide_symbolically(
            scene, "Xe ô tô có được rẽ trái không?", rules=reviewed_rules(),
        )

        self.assertEqual(decision.decision, "ALLOWED")
        self.assertIn("R301_E_ALLOW_TURN_LEFT", decision.rules_fired)
        self.assertFalse(any(
            item["rule_id"] == "COVERAGE_GAP_no-parking.jpg"
            for item in decision.rule_evaluations
        ))

    def test_multi_intent_query_keeps_results_separate(self) -> None:
        query_intent = query("Biển này có ý nghĩa gì và ô tô có được đỗ ở đây không?")
        trace = reason_over_facts(facts("P.131a"), query_intent, rules=reviewed_rules())
        payload = trace.to_dict()
        self.assertEqual(len(payload["intent_evaluations"]), 2)
        self.assertEqual(payload["intent_evaluations"][0]["decision"], "UNKNOWN")
        self.assertEqual(payload["intent_evaluations"][1]["decision"], "PROHIBITED")

    def test_sign_meaning_intent_uses_official_catalog_and_preserves_review_status(self) -> None:
        trace = reason_over_facts(facts("P.123a"), "Biển này có ý nghĩa gì?")
        payload = trace.to_dict()
        result = payload["sign_meaning_evaluations"][0]
        self.assertEqual(result["sign_code"], "P.123a")
        self.assertEqual(result["status"], "unverified")
        self.assertTrue(result["legal_basis"][0]["source_url"].startswith("https://datafiles.chinhphu.vn/"))

    def test_unmatched_sign_stays_unknown_and_keeps_coverage_gap(self) -> None:
        decision = decide_symbolically(facts("P.999"), query("Xe ô tô có được đỗ không?"))
        self.assertEqual(decision.decision, "UNKNOWN")
        self.assertEqual(decision.rules_fired, [])
        gap = next(item for item in decision.rule_evaluations if item["rule_id"].startswith("COVERAGE_GAP_"))
        self.assertIn("sign_rule_coverage:P.999", gap["missing_facts"])

    def test_unrelated_sign_does_not_block_rule_for_another_sign(self) -> None:
        observations = [
            normalize_payload_to_observation({"sign_code": "P.130", "image_name": "a.jpg"}),
            normalize_payload_to_observation({"sign_code": "R.301", "image_name": "b.jpg"}),
        ]
        for observation in observations:
            observation.visual_attributes["normalized_conditions"] = {
                "sign_effective_scope_confirmed": True,
            }
            observation.confidence = "high"
        scene = TrafficFacts("2.0", "scene-1", observations,
            sign_inventory_status="complete", sign_inventory_evidence=[{
            "source": "test_fixture", "evidence": "fixture explicitly includes both signs"
        }])
        decision = decide_symbolically(scene, query("Xe ô tô có được đỗ không?"), rules=reviewed_rules())
        self.assertEqual(decision.decision, "PROHIBITED")
        self.assertFalse(any(
            item["rule_id"] == "COVERAGE_GAP_b.jpg"
            for item in decision.rule_evaluations
        ))

    def test_linked_supplementary_panel_is_evaluated_by_primary_sign_not_double_counted_as_unruled(self) -> None:
        main = normalize_payload_to_observation({"sign_code": "P.130", "image_name": "main.jpg"})
        main.visual_attributes["normalized_conditions"] = {"sign_effective_scope_confirmed": True}
        main.confidence = "high"
        panel = normalize_payload_to_observation({
            "sign_code": "S.508a", "image_name": "panel.jpg", "role": "supplementary",
        })
        panel.ocr_texts = [{"raw_text": "8:00-10:00", "confidence": "high"}]
        panel.confidence = "high"
        scene = TrafficFacts(
            "2.0", "p130-s508", [main, panel],
            sign_relations=[SignRelation("panel-main", "supplements", "panel.jpg", "main.jpg", "high")],
            sign_inventory_status="complete",
            sign_inventory_evidence=[{"source": "test_fixture", "evidence": "complete linked sign stack"}],
        )
        decision = decide_symbolically(
            scene, "Lúc 08:30 xe ô tô có được đỗ ở đây không?", rules=reviewed_rules(),
        )
        self.assertTrue(any(
            item["rule_id"] == "P130_NO_STOPPING_NO_PARKING_DURING_S508"
            for item in decision.rule_evaluations
        ))
        self.assertFalse(any(
            item["rule_id"] == "COVERAGE_GAP_panel.jpg"
            for item in decision.rule_evaluations
        ))

    def test_known_sign_rule_does_not_require_complete_inventory_or_reached_sign_fact(self) -> None:
        partial_facts = TrafficFacts(
            "2.0", "scene-1", [normalize_payload_to_observation({"sign_code": "P.130"})],
            sign_inventory_status="partial",
        )
        decision = decide_symbolically(partial_facts, "Xe ô tô có được đỗ không?", rules=reviewed_rules())
        self.assertEqual(decision.decision, "PROHIBITED")
        self.assertNotIn("complete_sign_inventory", decision.rule_evaluations[0]["missing_facts"])
        self.assertNotIn("sign_effective_scope_confirmed", decision.rule_evaluations[0]["missing_facts"])

    def test_context_output_exposes_new_intent_and_fact_schemas(self) -> None:
        context = build_symbolic_reasoning_context(facts("P.131a"), "Xe ô tô có được đỗ không?")
        self.assertEqual(context["query_intent"]["schema_version"], "3.0")
        self.assertEqual(context["facts"]["signs"][0]["code"], "P.131a")
        self.assertEqual(context["symbolic_decision"]["decision"], "PROHIBITED")
        self.assertEqual(context["sign_meaning_evaluations"], [])
        self.assertNotIn("legal_review_required", context["rule_evaluations"][0]["missing_facts"])
        self.assertIn("intent_evaluations", context["symbolic_decision"])

    def test_one_generic_threshold_logic_checks_p115_gross_weight(self) -> None:
        scene = facts("P.115")
        scene.context_facts.append(ContextFact(
            fact_id="p115-limit", predicate="total_weight_limit_t", value=10,
            status="observed", source="test_fixture", confidence="high", unit="t",
        ))
        over_limit = decide_symbolically(
            scene,
            query("Xe tải có tổng khối lượng 12 tấn có được đi qua đường này không?"),
            rules=reviewed_rules(),
        )
        within_limit = decide_symbolically(
            scene,
            query("Xe tải có tổng khối lượng 8 tấn có được qua cầu không?"),
            rules=reviewed_rules(),
        )
        self.assertEqual(over_limit.decision, "PROHIBITED")
        self.assertEqual(within_limit.decision, "ALLOWED")
        self.assertIn("P115_TOTAL_WEIGHT_LIMIT", over_limit.rules_fired)
        self.assertIn("P115_TOTAL_WEIGHT_LIMIT", within_limit.rules_fired)

    def test_missing_or_ambiguous_vehicle_mass_never_means_allowed(self) -> None:
        scene = facts("P.115")
        scene.context_facts.append(ContextFact(
            fact_id="p115-limit", predicate="total_weight_limit_t", value=10,
            status="observed", source="test_fixture", confidence="high", unit="t",
        ))
        decision = decide_symbolically(
            scene, query("Xe tải 2 tấn có được lên cầu không?"), rules=reviewed_rules(),
        )
        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertIn("vehicle_gross_weight_t", decision.rule_evaluations[0]["missing_facts"])


if __name__ == "__main__":
    unittest.main()
