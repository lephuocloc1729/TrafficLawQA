import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.symbolic.fact_schema import SignObservation, SignRelation, TrafficFacts
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation
from src.core.symbolic.rule_schema import RuleCondition, TrafficRule
from src.core.symbolic.traffic_rules import TRAFFIC_RULES, find_matching_rules, vehicle_is_covered


def facts_for(payload: dict) -> TrafficFacts:
    observation = normalize_payload_to_observation(payload)
    return TrafficFacts("2.0", "scene-1", [observation])


def query(text: str):
    stable_even_day = datetime(2026, 10, 4, 12, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    return parse_query_intent_deterministic(text, now=stable_even_day)


def evaluate(facts: TrafficFacts, parsed, rules=TRAFFIC_RULES):
    return find_matching_rules(
        facts,
        parsed.query_intent,
        parsed.context_facts,
        rules=rules,
    )


class TrafficRulesTest(unittest.TestCase):
    def test_rules_are_declarative_and_registered(self) -> None:
        self.assertEqual(len(TRAFFIC_RULES), 8)
        self.assertTrue(all(rule.trigger_sign_codes and rule.intent_kinds for rule in TRAFFIC_RULES))

    def test_vehicle_scope(self) -> None:
        self.assertTrue(vehicle_is_covered("car", "motor_vehicle"))
        self.assertTrue(vehicle_is_covered("motorbike", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("bicycle", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("pedestrian", "vehicle"))

    def test_p130_matches_parking_and_stopping(self) -> None:
        facts = facts_for({"sign_code": "P.130", "image_name": "crop-1.jpg"})
        parking = evaluate(facts, query("Xe ô tô có được phép đỗ không?"))
        stopping = evaluate(facts, query("Xe ô tô có được dừng xe không?"))
        self.assertEqual(parking[0].rule_id, "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(stopping[0].rule_id, "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(parking[0].decision, "PROHIBITED")

    def test_odd_even_conditions_are_evaluated_from_context_facts(self) -> None:
        odd = evaluate(
            facts_for({"sign_code": "P.131b"}),
            query("Ngày lẻ, xe ô tô có được đỗ xe không?"),
        )[0]
        even = evaluate(
            facts_for({"sign_code": "P.131b"}),
            query("Ngày chẵn, xe ô tô có được đỗ xe không?"),
        )[0]
        self.assertEqual((odd.decision, odd.status), ("PROHIBITED", "applied"))
        self.assertEqual((even.decision, even.status), ("ALLOWED", "applied"))

    def test_time_rule_is_incomplete_when_temporal_reference_unresolved(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.131c"}),
            query("Xe ô tô có được đỗ lúc 7 giờ hôm nay không?"),
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.missing_facts, ["calendar_parity"])

    def test_system_clock_fact_supplies_calendar_parity(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.131c"}),
            query("Xe ô tô có được đỗ không?"),
        )[0]
        self.assertEqual(result.decision, "PROHIBITED")
        self.assertTrue(any(item.get("predicate") == "calendar_parity" for item in result.trace))

    def test_p127_missing_numeric_value_is_incomplete(self) -> None:
        result = evaluate(
            facts_for({"sign_code": "P.127"}), query("Tốc độ tối đa là bao nhiêu?")
        )[0]
        self.assertEqual(result.decision, "CONDITIONAL")
        self.assertEqual(result.missing_facts, ["speed_limit_value"])

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
        ])
        parsed = query("Xe ô tô có được đỗ không?")
        parsed.query_intent.intents[0].target.kind = "sign_observation"
        parsed.query_intent.intents[0].target.observation_ids = ["crop-b.jpg"]
        self.assertEqual(evaluate(facts, parsed), [])

    def test_supplementary_fact_is_scoped_to_its_main_sign(self) -> None:
        facts = TrafficFacts(
            "2.0", "scene-1",
            sign_observations=[
                SignObservation("main", role="main", sign_code="P.131a", identity_status="identified"),
                SignObservation(
                    "panel", role="supplementary", ocr_texts=[{"raw_text": "6:00-9:00"}],
                    visual_attributes={"normalized_conditions": {"time_window": {"start": "06:00", "end": "09:00"}}},
                ),
            ],
            sign_relations=[SignRelation("r1", "supplements", "panel", "main", "high")],
        )
        rule = TrafficRule(
            "test-supplementary", [], ["P.131a"], ["CHECK_PERMISSION"], ["parking"],
            ["motor_vehicle"], "PROHIBITED",
            [RuleCondition("time_window", "exists")],
        )
        evaluation = evaluate(facts, query("Xe ô tô có được đỗ không?"), [rule])[0]
        self.assertEqual(evaluation.decision, "PROHIBITED")
        self.assertTrue(any(step["predicate"] == "time_window" for step in evaluation.trace))

    def test_non_matching_action_and_unknown_sign_do_not_fire(self) -> None:
        facts = facts_for({"sign_code": "P.131a"})
        self.assertEqual(evaluate(facts, query("Xe ô tô có được rẽ trái không?")), [])
        self.assertEqual(evaluate(facts_for({"sign_code": "P.999"}), query("Xe ô tô có được đỗ không?")), [])


if __name__ == "__main__":
    unittest.main()
