import json
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from src.core.symbolic.question_intent import (
    extract_action,
    extract_time_context,
    extract_vehicle,
    normalize_question_text,
    parse_query_intent,
    parse_query_intent_deterministic,
)


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return self.response


class QueryIntentTest(unittest.TestCase):
    def test_accent_normalization_and_basic_slots(self) -> None:
        query = parse_query_intent_deterministic("Xe ô tô có được phép đỗ tại đây không?")
        self.assertEqual(normalize_question_text("Xe ô tô"), "xe o to")
        self.assertEqual(query.query_intent.intents[0].kind, "CHECK_PERMISSION")
        self.assertEqual(query.query_intent.intents[0].action.value, "parking")
        self.assertEqual(query.query_intent.intents[0].actor.value, "car")
        self.assertEqual(query.query_intent.intents[0].target.kind, "scene")
        self.assertEqual(query.query_intent.intents[0].requested_answer["kind"], "decision_with_explanation")

    def test_parser_supports_multiple_intents(self) -> None:
        query = parse_query_intent_deterministic(
            "Biển này có ý nghĩa gì và ô tô có được đỗ ở đây không?"
        )
        self.assertEqual(
            [item.kind for item in query.query_intent.intents],
            ["QUERY_SIGN_MEANING", "CHECK_PERMISSION"],
        )

    def test_permission_defaults_to_a_system_clock_fact(self) -> None:
        now = datetime(2026, 10, 5, 8, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        query = parse_query_intent_deterministic("Xe ô tô có được đỗ ở đây không?", now=now)
        temporal = query.context_facts[0]
        self.assertEqual(temporal.predicate, "observed_at")
        self.assertEqual(temporal.value, "2026-10-05T08:30:00+07:00")
        self.assertEqual(temporal.source, "system_clock")
        self.assertEqual(temporal.metadata["resolution"], "default_current")

    def test_explicit_time_is_kept_as_query_fact_not_replaced_by_now(self) -> None:
        query = parse_query_intent_deterministic("Xe ô tô có được đỗ lúc 7 giờ hôm nay không?")
        temporal = query.context_facts[0]
        self.assertEqual(temporal.predicate, "temporal_reference")
        self.assertEqual(temporal.status, "needs_resolution")
        self.assertEqual(temporal.source, "user_query")

    def test_calendar_parity_is_an_explicit_context_fact(self) -> None:
        query = parse_query_intent_deterministic(
            "Hôm nay là ngày lẻ, xe ô tô có được đỗ không?"
        )
        self.assertEqual(query.context_facts[0].predicate, "calendar_parity")
        self.assertEqual(query.context_facts[0].value, "odd_days")
        self.assertEqual(query.context_facts[0].source, "user_query")

    def test_action_and_vehicle_extractors(self) -> None:
        self.assertEqual(extract_action("Xe máy có được rẽ trái không?"), "turn_left")
        self.assertEqual(extract_vehicle("Xe tải có được đi vào không?"), "truck")
        self.assertEqual(extract_time_context("Ngày chẵn có được đỗ không?"), "even_days")

    def test_llm_fallback_parses_multiple_intents_and_validates(self) -> None:
        client = FakeLLMClient(json.dumps({
            "schema_version": "2.0",
            "intents": [
                {"intent_id": "i1", "kind": "QUERY_SIGN_MEANING", "confidence": "high"},
                {"intent_id": "i2", "kind": "CHECK_PERMISSION",
                 "action": {"value": "entering", "status": "provided"},
                 "actor": {"value": "truck", "status": "provided"}, "confidence": "medium"},
            ],
            "confidence": "high",
        }))
        query = parse_query_intent("Trường hợp xe chở hàng như hình thì sao?", llm_client=client)
        self.assertEqual(client.calls, 1)
        self.assertEqual([item.kind for item in query.query_intent.intents], ["QUERY_SIGN_MEANING", "CHECK_PERMISSION"])
        self.assertEqual(query.query_intent.intents[1].action.value, "entering")
        self.assertEqual(query.query_intent.intents[1].actor.value, "truck")
        self.assertEqual(query.context_facts[0].metadata["resolution"], "default_current")

    def test_clear_deterministic_parse_does_not_call_llm(self) -> None:
        client = FakeLLMClient("{}")
        query = parse_query_intent("Xe ô tô có được đỗ không?", llm_client=client)
        self.assertEqual(client.calls, 0)
        self.assertEqual(query.query_intent.intents[0].kind, "CHECK_PERMISSION")

    def test_invalid_llm_labels_are_reduced_to_unknown(self) -> None:
        client = FakeLLMClient(json.dumps({
            "intents": [{"kind": "FREE_TEXT", "action": {"value": "fly"}, "confidence": "certain"}]
        }))
        query = parse_query_intent("Một câu hỏi mơ hồ", llm_client=client)
        self.assertEqual(query.query_intent.intents[0].kind, "UNKNOWN")
        self.assertEqual(query.query_intent.intents[0].action.value, "unknown")
        self.assertEqual(query.query_intent.intents[0].confidence, "low")


if __name__ == "__main__":
    unittest.main()
