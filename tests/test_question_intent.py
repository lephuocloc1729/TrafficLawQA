import json
import unittest

from src.core.symbolic.question_intent import (
    extract_action,
    extract_time_context,
    extract_vehicle,
    normalize_question_text,
    parse_question_intent,
    parse_question_intent_deterministic,
)


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return self.response


class QuestionIntentTest(unittest.TestCase):
    def test_normalize_question_text_removes_vietnamese_accents(self) -> None:
        self.assertEqual(
            normalize_question_text("Xe ô tô có được phép đỗ không?"),
            "xe o to co duoc phep do khong",
        )

    def test_permission_question_extracts_intent_action_and_vehicle(self) -> None:
        intent = parse_question_intent_deterministic(
            "Xe ô tô có được phép đỗ tại vị trí này không?"
        )

        self.assertEqual(intent.intent, "CHECK_PERMISSION")
        self.assertEqual(intent.action, "parking")
        self.assertEqual(intent.vehicle, "car")
        self.assertEqual(intent.confidence, "high")
        self.assertEqual(intent.source, "keyword")

    def test_all_supported_query_intents_are_detected(self) -> None:
        cases = {
            "Biển báo này có ý nghĩa gì?": "QUERY_SIGN_MEANING",
            "Biển này cấm những phương tiện nào?": "QUERY_PROHIBITED_SUBJECTS",
            "Biển này có hiệu lực vào thời gian nào?": "QUERY_TIME_CONDITION",
            "Biển cấm theo hướng nào?": "QUERY_DIRECTION_SCOPE",
            "Biển phụ này áp dụng trong phạm vi bao nhiêu mét?": "QUERY_DISTANCE_SCOPE",
        }

        for question, expected_intent in cases.items():
            with self.subTest(question=question):
                parsed = parse_question_intent_deterministic(question)
                self.assertEqual(parsed.intent, expected_intent)
                self.assertEqual(parsed.confidence, "high")

    def test_extracts_supported_actions(self) -> None:
        cases = {
            "Có được đỗ xe không?": "parking",
            "Có được dừng xe không?": "stopping",
            "Có được rẽ trái không?": "turn_left",
            "Có được rẽ phải không?": "turn_right",
            "Xe có được đi vào đường này không?": "entering",
            "Tốc độ tối đa là bao nhiêu?": "speed",
            "Xe máy có được lưu thông không?": "circulation",
        }

        for question, expected_action in cases.items():
            with self.subTest(question=question):
                self.assertEqual(extract_action(question), expected_action)

    def test_week8_required_question_intent_parsing_cases(self) -> None:
        cases = [
            ("Xe ô tô có được phép đỗ không?", "CHECK_PERMISSION", "parking", "car"),
            ("Xe máy có được dừng xe tại đây không?", "CHECK_PERMISSION", "stopping", "motorbike"),
            ("Xe ô tô có được rẽ trái không?", "CHECK_PERMISSION", "turn_left", "car"),
            ("Xe tải có được đi vào đường này không?", "CHECK_PERMISSION", "entering", "truck"),
            ("Tốc độ tối đa là bao nhiêu?", "UNKNOWN", "speed", "unknown"),
        ]

        for question, expected_intent, expected_action, expected_vehicle in cases:
            with self.subTest(question=question):
                parsed = parse_question_intent_deterministic(question)
                self.assertEqual(parsed.intent, expected_intent)
                self.assertEqual(parsed.action, expected_action)
                self.assertEqual(parsed.vehicle, expected_vehicle)

    def test_extracts_odd_even_day_time_context(self) -> None:
        self.assertEqual(extract_time_context("Hôm nay là ngày lẻ, xe có được đỗ không?"), "odd_days")
        self.assertEqual(extract_time_context("Vào ngày chẵn có được đỗ xe không?"), "even_days")
        self.assertIsNone(extract_time_context("Xe có được đỗ không?"))

    def test_extracts_supported_vehicle_subjects(self) -> None:
        cases = {
            "Xe ô tô có được đi không?": "car",
            "Xe máy có bị cấm không?": "motorbike",
            "Xe tải có được vào không?": "truck",
            "Xe buýt có được lưu thông không?": "bus",
            "Xe đạp có được đi qua không?": "bicycle",
            "Người đi bộ có được qua đường không?": "pedestrian",
            "Xe cơ giới có được đỗ không?": "motor_vehicle",
        }

        for question, expected_vehicle in cases.items():
            with self.subTest(question=question):
                self.assertEqual(extract_vehicle(question), expected_vehicle)

    def test_unknown_question_can_use_llm_fallback(self) -> None:
        client = FakeLLMClient(
            json.dumps(
                {
                    "intent": "CHECK_PERMISSION",
                    "action": "entering",
                    "vehicle": "truck",
                    "time_context": None,
                    "location_context": "current_location",
                    "confidence": "medium",
                }
            )
        )

        parsed = parse_question_intent(
            "Trường hợp xe chở hàng như hình thì sao?",
            llm_client=client,
        )

        self.assertEqual(client.calls, 1)
        self.assertEqual(parsed.intent, "CHECK_PERMISSION")
        self.assertEqual(parsed.action, "entering")
        self.assertEqual(parsed.vehicle, "truck")
        self.assertEqual(parsed.location_context, "current_location")
        self.assertEqual(parsed.confidence, "medium")
        self.assertEqual(parsed.source, "llm")

    def test_llm_fallback_is_not_called_for_clear_keyword_parse(self) -> None:
        client = FakeLLMClient('{"intent":"UNKNOWN"}')

        parsed = parse_question_intent(
            "Xe ô tô có được phép đỗ không?",
            llm_client=client,
        )

        self.assertEqual(client.calls, 0)
        self.assertEqual(parsed.intent, "CHECK_PERMISSION")
        self.assertEqual(parsed.action, "parking")

    def test_llm_fallback_payload_is_validated(self) -> None:
        client = FakeLLMClient(
            json.dumps(
                {
                    "intent": "FREE_TEXT",
                    "action": "fly",
                    "vehicle": "spaceship",
                    "confidence": "certain",
                }
            )
        )

        parsed = parse_question_intent("Một câu hỏi mơ hồ", llm_client=client)

        self.assertEqual(parsed.intent, "UNKNOWN")
        self.assertEqual(parsed.action, "unknown")
        self.assertEqual(parsed.vehicle, "unknown")
        self.assertEqual(parsed.confidence, "low")

    def test_llm_fallback_error_is_preserved(self) -> None:
        client = FakeLLMClient("not json")

        parsed = parse_question_intent("Một câu hỏi mơ hồ", llm_client=client)

        self.assertEqual(parsed.intent, "UNKNOWN")
        self.assertIn("llm_fallback_error", parsed.metadata)


if __name__ == "__main__":
    unittest.main()
