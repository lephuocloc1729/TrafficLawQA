import json
import unittest

from src.core.symbolic.sign_schema import (
    QuestionIntent,
    ReasoningTrace,
    RuleMatch,
    SymbolicDecision,
    TrafficSignFact,
)


class SignSchemaTest(unittest.TestCase):
    def test_traffic_sign_fact_serializes_required_fields(self) -> None:
        fact = TrafficSignFact(
            sign_code=" P.131b ",
            sign_family=" P.131 ",
            sign_name="Cấm đỗ xe ngày lẻ",
            restriction_type="no_parking",
            target_vehicle="motor_vehicle",
            time_condition="odd_days",
            direction_scope="side_of_road_with_sign",
            distance_scope=None,
            related_articles=[
                {"law_id": "QCVN 41:2024/BGTVT", "article_id": "31"}
            ],
            visual_evidence=[{"image_name": "demo_crop0.jpg"}],
            confidence="high",
            source="retrieval",
        )

        payload = fact.to_dict()

        self.assertEqual(payload["sign_code"], "P.131b")
        self.assertEqual(payload["sign_family"], "P.131")
        self.assertEqual(payload["restriction_type"], "no_parking")
        self.assertEqual(payload["target_vehicle"], "motor_vehicle")
        self.assertEqual(payload["time_condition"], "odd_days")
        self.assertEqual(payload["confidence"], "high")
        self.assertEqual(payload["related_articles"][0]["article_id"], "31")
        self.assertEqual(payload["visual_evidence"][0]["image_name"], "demo_crop0.jpg")

        decoded = json.loads(fact.to_json())
        self.assertEqual(decoded, payload)

    def test_invalid_confidence_and_decision_fall_back_safely(self) -> None:
        fact = TrafficSignFact(
            sign_code="P.102",
            sign_family="P.102",
            confidence="certain",  # type: ignore[arg-type]
        )
        decision = SymbolicDecision(
            decision="MAYBE",  # type: ignore[arg-type]
            confidence="sure",  # type: ignore[arg-type]
        )

        self.assertEqual(fact.confidence, "low")
        self.assertEqual(decision.decision, "UNKNOWN")
        self.assertEqual(decision.confidence, "low")

    def test_nested_reasoning_trace_serializes_cleanly(self) -> None:
        fact = TrafficSignFact(
            sign_code="P.130",
            sign_family="P.130",
            restriction_type="no_stopping_no_parking",
            target_vehicle="motor_vehicle",
            confidence="medium",
        )
        intent = QuestionIntent(
            intent="check_permission",
            action="parking",
            vehicle="car",
            raw_question="Xe ô tô có được đỗ ở đây không?",
            confidence="medium",
        )
        rule_match = RuleMatch(
            rule_id="P130_NO_STOPPING_NO_PARKING",
            decision="PROHIBITED",
            matched_fact_ids=["P.130"],
            legal_basis=[{"law_id": "QCVN 41:2024/BGTVT", "article_id": "30"}],
            reason="P.130 prohibits stopping and parking.",
            confidence="high",
        )
        decision = SymbolicDecision(
            decision="PROHIBITED",
            conclusion="Xe ô tô không được đỗ tại vị trí có biển P.130.",
            rule_matches=[rule_match],
            matched_fact_ids=["P.130"],
            confidence="high",
        )
        trace = ReasoningTrace(
            facts=[fact],
            question_intent=intent,
            rule_matches=[rule_match],
            decision=decision,
        )
        trace.add_step(
            "apply_rule",
            inputs={"sign_code": "P.130", "action": "parking"},
            output={"decision": "PROHIBITED"},
        )

        payload = trace.to_dict()

        self.assertEqual(payload["facts"][0]["sign_code"], "P.130")
        self.assertEqual(payload["question_intent"]["intent"], "CHECK_PERMISSION")
        self.assertEqual(payload["rule_matches"][0]["rule_id"], "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(payload["decision"]["rules_fired"], ["P130_NO_STOPPING_NO_PARKING"])
        self.assertEqual(payload["steps"][0]["name"], "apply_rule")


if __name__ == "__main__":
    unittest.main()
