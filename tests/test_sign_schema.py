import json
import unittest

from src.core.symbolic.fact_schema import ContextFact, SignObservation, TrafficFacts
from src.core.symbolic.intent_schema import Intent, IntentParseResult, IntentSlot, QueryIntent
from src.core.symbolic.rule_schema import RuleCondition, RuleEvaluation, TrafficRule
from src.core.symbolic.sign_schema import ReasoningTrace, SymbolicDecision


class SymbolicSchemaTest(unittest.TestCase):
    def test_intent_and_query_facts_serialize_as_separate_contracts(self) -> None:
        query = QueryIntent(
            schema_version="2.0",
            query_id="q1",
            raw_query="Biển này là gì, xe ô tô có được đỗ không?",
            intents=[
                Intent("i1", "QUERY_SIGN_MEANING"),
                Intent("i2", "CHECK_PERMISSION", action=IntentSlot("parking"), actor=IntentSlot("car")),
            ],
        )
        parsed = IntentParseResult(
            schema_version="2.0",
            query_intent=query,
            context_facts=[{
                "fact_id": "time-1", "predicate": "observed_at",
                "value": "2026-10-04T12:00:00+07:00", "source": "system_clock",
            }],
        )
        decoded = json.loads(parsed.to_json())
        self.assertEqual([item["kind"] for item in decoded["query_intent"]["intents"]], ["QUERY_SIGN_MEANING", "CHECK_PERMISSION"])
        self.assertEqual(decoded["context_facts"][0]["source"], "system_clock")

    def test_observation_and_fact_set_keep_identity_separate_from_legal_scope(self) -> None:
        observation = SignObservation(
            observation_id="crop-1", role="main", sign_code="P.131b",
            identity_status="identified", candidates=[{"sign_code": "P.131b", "score": 0.9}],
            evidence=[{"type": "crop", "path": "crop.jpg"}], confidence="high",
        )
        facts = TrafficFacts(
            schema_version="2.0", scene_id="scene-1", sign_observations=[observation],
            context_facts=[ContextFact("t1", "calendar_parity", "odd_days", "provided", "user_query")],
        )
        payload = facts.to_dict()
        self.assertEqual(payload["sign_observations"][0]["sign_code"], "P.131b")
        self.assertNotIn("restriction_type", payload["sign_observations"][0])
        self.assertEqual(payload["context_facts"][0]["predicate"], "calendar_parity")

    def test_rule_schema_is_declarative_and_serializable(self) -> None:
        rule = TrafficRule(
            rule_id="test-rule", legal_basis=[], trigger_sign_codes=["P.131b"],
            intent_kinds=["CHECK_PERMISSION"], actions=["parking"],
            actor_scope=["motor_vehicle"], outcome="PROHIBITED",
            conditions=[RuleCondition("calendar_parity", "equals", "odd_days")],
        )
        self.assertEqual(rule.to_dict()["conditions"][0]["expected_value"], "odd_days")
        evaluation = RuleEvaluation("test-rule", "i1", decision="CONDITIONAL", status="incomplete")
        self.assertEqual(evaluation.to_dict()["status"], "incomplete")

    def test_reasoning_trace_serializes_new_schema(self) -> None:
        decision = SymbolicDecision(decision="UNKNOWN", conclusion="Need more facts")
        trace = ReasoningTrace(facts={}, query_intent={}, decision=decision)
        trace.add_step("evaluate_rules", inputs={"count": 1}, output={"decision": "UNKNOWN"})
        payload = trace.to_dict()
        self.assertEqual(payload["steps"][0]["name"], "evaluate_rules")
        self.assertEqual(payload["decision"]["decision"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
