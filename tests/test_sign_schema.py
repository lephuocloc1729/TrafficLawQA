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
                Intent("i1", "QUERY_SIGN"),
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
        self.assertEqual([item["kind"] for item in decoded["intents"]], ["QUERY_SIGN", "CHECK_PERMISSION"])
        self.assertEqual(decoded["context_facts"][0]["source"], "system_clock")
        permission = decoded["intents"][1]
        self.assertEqual(set(permission), {"kind", "action", "actor"})
        self.assertEqual(permission["action"], "parking")
        self.assertEqual(permission["actor"], "car")

    def test_legacy_intent_metadata_is_accepted_but_not_repeated(self) -> None:
        query = QueryIntent(
            schema_version="2.0", query_id="q1", raw_query="Xe tải có được đi vào không?",
            intents=[{
                "intent_id": "i1", "kind": "CHECK_PERMISSION", "confidence": "high",
                "unresolved_items": [], "action": {"value": "entering", "source": "user_query"},
                "actor": {"value": "truck", "status": "provided", "source": "user_query", "confidence": 0.9},
            }],
        )
        payload = query.to_dict()
        self.assertEqual(payload["intents"][0]["actor"], "truck")
        self.assertNotIn("confidence", payload["intents"][0])
        self.assertNotIn("unresolved_items", payload["intents"][0])

    def test_compact_intent_values_can_be_loaded_again(self) -> None:
        query = QueryIntent(
            schema_version="3.0", query_id="q1", raw_query="Biển này cấm những xe nào?",
            intents=[{
                "kind": "INFORMATION", "information_type": "prohibited_subjects", "target": "scene",
            }],
        )
        self.assertEqual(query.intents[0].information_type, "prohibited_subjects")
        self.assertEqual(query.intents[0].target.kind, "scene")
        self.assertEqual(query.to_dict()["intents"], [{
            "kind": "INFORMATION", "information_type": "prohibited_subjects", "target": "scene",
        }])

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
        self.assertEqual(payload["sign_observations"][0]["identity"]["sign_code"], "P.131b")
        self.assertNotIn("restriction_type", payload["sign_observations"][0])
        self.assertEqual(payload["scene_facts"][0]["predicate"], "calendar_parity")
        self.assertNotIn("schema_version", payload)
        self.assertNotIn("scene_id", payload)

    def test_rule_schema_is_declarative_and_serializable(self) -> None:
        rule = TrafficRule(
            rule_id="test-rule", legal_basis=[], trigger_sign_codes=["P.131b"],
            actions=["parking"],
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
