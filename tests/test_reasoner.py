import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.symbolic.fact_schema import TrafficFacts
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.reasoner import (
    build_symbolic_reasoning_context,
    decide_symbolically,
    reason_over_facts,
)
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation


def facts(code: str) -> TrafficFacts:
    return TrafficFacts("2.0", "scene-1", [normalize_payload_to_observation({
        "sign_code": code,
        "image_name": "crop-1.jpg",
        "law_id": "QCVN 41:2024/BGTVT",
        "article_id": "31",
    })])


def query(text: str):
    fixed_now = datetime(2026, 10, 5, 8, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    return parse_query_intent_deterministic(text, now=fixed_now)


class ReasonerTest(unittest.TestCase):
    def test_reasoner_returns_evaluation_trace(self) -> None:
        trace = reason_over_facts(facts("P.130"), "Xe ô tô có được phép đỗ tại đây không?")
        payload = trace.to_dict()
        self.assertEqual(payload["decision"]["decision"], "PROHIBITED")
        self.assertEqual(payload["decision"]["rules_fired"], ["P130_NO_STOPPING_NO_PARKING"])
        self.assertEqual(payload["rule_evaluations"][0]["rule_id"], "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(payload["steps"][1]["name"], "evaluate_rules")

    def test_permission_decision_waits_for_time_fact_when_explicit_time_unresolved(self) -> None:
        decision = decide_symbolically(
            facts("P.131b"),
            "Xe ô tô có được đỗ lúc 7 giờ hôm nay không?",
        )
        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertEqual(decision.rule_evaluations[0]["missing_facts"], ["calendar_parity"])

    def test_permission_uses_default_system_clock_if_user_omits_time(self) -> None:
        decision = decide_symbolically(facts("P.131b"), query("Xe ô tô có được đỗ không?"))
        self.assertEqual(decision.decision, "PROHIBITED")  # injected date is an odd day
        self.assertTrue(decision.rules_fired)

    def test_opposite_parity_sign_is_allowed_for_that_calendar_day(self) -> None:
        decision = decide_symbolically(facts("P.131b"), query("Ngày chẵn, xe ô tô có được đỗ không?"))
        self.assertEqual(decision.decision, "ALLOWED")

    def test_conflicting_signs_are_reported_per_intent(self) -> None:
        both = TrafficFacts("2.0", "scene-1", [
            normalize_payload_to_observation({"sign_code": "P.131b", "image_name": "a.jpg"}),
            normalize_payload_to_observation({"sign_code": "P.131c", "image_name": "b.jpg"}),
        ])
        decision = decide_symbolically(both, query("Ngày lẻ, xe ô tô có được đỗ không?"))
        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertIn("conflicting_rule_outcomes", decision.metadata["unresolved_items"])

    def test_multi_intent_query_keeps_results_separate(self) -> None:
        query_intent = query("Biển này có ý nghĩa gì và ô tô có được đỗ ở đây không?")
        trace = reason_over_facts(facts("P.131a"), query_intent)
        payload = trace.to_dict()
        self.assertEqual(len(payload["intent_evaluations"]), 2)
        self.assertEqual(payload["intent_evaluations"][0]["decision"], "UNKNOWN")
        self.assertEqual(payload["intent_evaluations"][1]["decision"], "PROHIBITED")

    def test_unrecognized_sign_stays_unknown(self) -> None:
        decision = decide_symbolically(facts("P.999"), query("Xe ô tô có được đỗ không?"))
        self.assertEqual(decision.decision, "UNKNOWN")
        self.assertEqual(decision.rules_fired, [])

    def test_context_output_exposes_new_intent_and_fact_schemas(self) -> None:
        context = build_symbolic_reasoning_context(facts("P.131a"), "Xe ô tô có được đỗ không?")
        self.assertEqual(context["query_intent"]["schema_version"], "2.0")
        self.assertEqual(context["facts"]["sign_observations"][0]["sign_code"], "P.131a")
        self.assertEqual(context["symbolic_decision"]["decision"], "PROHIBITED")
        self.assertIn("intent_evaluations", context["symbolic_decision"])


if __name__ == "__main__":
    unittest.main()
