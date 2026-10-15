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
            ["QUERY_SIGN", "CHECK_PERMISSION"],
        )

    def test_permission_defaults_to_a_system_clock_fact(self) -> None:
        now = datetime(2026, 10, 5, 8, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        query = parse_query_intent_deterministic("Xe ô tô có được đỗ ở đây không?", now=now)
        temporal = query.context_facts[0]
        self.assertEqual(temporal.predicate, "observed_at")
        self.assertEqual(temporal.value, "2026-10-05T08:30:00+07:00")
        self.assertEqual(temporal.source, "system_clock")
        self.assertEqual(temporal.metadata["resolution"], "default_current")

    def test_pedestrian_location_is_extracted_only_when_explicit_and_unambiguous(self) -> None:
        sidewalk = parse_query_intent_deterministic(
            "Người đi bộ có được đi trên vỉa hè ở đây không?"
        )
        fact = next(item for item in sidewalk.context_facts if item.predicate == "pedestrian_location")
        self.assertEqual((fact.value, fact.source, fact.status), ("sidewalk", "user_query", "provided"))

        ambiguous = parse_query_intent_deterministic(
            "Người đi bộ có được đi trên vỉa hè hay lề đường ở đây không?"
        )
        self.assertFalse(any(item.predicate == "pedestrian_location" for item in ambiguous.context_facts))

        denied_location = parse_query_intent_deterministic(
            "Không có vỉa hè, người đi bộ có được đi ở đây không?"
        )
        self.assertFalse(any(item.predicate == "pedestrian_location" for item in denied_location.context_facts))

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
        speed_intent = parse_query_intent_deterministic("Xe ô tô có được chạy ở tốc độ 70 km/h không?").query_intent.intents[0]
        self.assertEqual(speed_intent.action.value, "speed")
        self.assertEqual(speed_intent.requested_answer["requested_speed_kmh"], 70)
        self.assertEqual(extract_time_context("Ngày chẵn có được đỗ không?"), "even_days")

    def test_explicit_vehicle_height_is_a_query_fact_not_inferred_from_vehicle_class(self) -> None:
        parsed = parse_query_intent_deterministic(
            "Xe tải cao 4,35 m có được lưu thông qua biển này không?"
        )
        intent = parsed.query_intent.intents[0]
        height = next(fact for fact in parsed.context_facts if fact.predicate == "vehicle_height_m")
        self.assertEqual(intent.kind, "CHECK_PERMISSION")
        self.assertEqual(intent.actor.value, "truck")
        self.assertEqual(height.value, 4.35)
        self.assertEqual(height.source, "user_query")
        self.assertEqual(height.status, "provided")
        self.assertEqual(height.metadata["unit"], "m")
        self.assertEqual(intent.requested_answer["requested_vehicle_height_m"], 4.35)

        ambiguous = parse_query_intent_deterministic(
            "Xe tải cao 4,2 m, nhưng chiều cao xe là 4,4 m; có được đi qua không?"
        )
        conflicted = next(fact for fact in ambiguous.context_facts if fact.predicate == "vehicle_height_m")
        self.assertIsNone(conflicted.value)
        self.assertEqual(conflicted.status, "ambiguous")
        self.assertEqual(conflicted.metadata["conflicting_values"], [4.2, 4.4])
        self.assertNotIn("requested_vehicle_height_m", ambiguous.query_intent.intents[0].requested_answer)

    def test_explicit_width_length_and_trailer_combination_measurements_become_query_facts(self) -> None:
        width_query = parse_query_intent_deterministic(
            "Xe tải rộng 2,6 m có được lưu thông qua biển này không?"
        )
        width = next(fact for fact in width_query.context_facts if fact.predicate == "vehicle_width_m")
        self.assertEqual((width.value, width.status, width.metadata["unit"]), (2.6, "provided", "m"))
        self.assertEqual(width_query.query_intent.intents[0].requested_answer["requested_vehicle_width_m"], 2.6)

        length_query = parse_query_intent_deterministic(
            "Xe tải dài 8,5 m có được lưu thông qua biển này không?"
        )
        length = next(fact for fact in length_query.context_facts if fact.predicate == "vehicle_length_m")
        self.assertEqual((length.value, length.status), (8.5, "provided"))

        combination_query = parse_query_intent_deterministic(
            "Xe đầu kéo kéo sơ mi rơ moóc có tổng chiều dài đoàn xe là 18 m có được đi qua không?"
        )
        combination_length = next(
            fact for fact in combination_query.context_facts
            if fact.predicate == "vehicle_combination_length_m"
        )
        configuration = next(
            fact for fact in combination_query.context_facts
            if fact.predicate == "vehicle_trailer_configuration"
        )
        self.assertEqual((combination_length.value, combination_length.status), (18.0, "provided"))
        self.assertEqual((configuration.value, configuration.status), ("semi_trailer", "provided"))

        ambiguous = parse_query_intent_deterministic(
            "Xe tải rộng 2,6 m nhưng chiều ngang xe là 2,8 m có được đi qua không?"
        )
        width_fact = next(fact for fact in ambiguous.context_facts if fact.predicate == "vehicle_width_m")
        self.assertIsNone(width_fact.value)
        self.assertEqual(width_fact.status, "ambiguous")

    def test_action_and_identifier_support_include_supplementary_signs_and_vehicle_classes(self) -> None:
        self.assertEqual(extract_action("Xe đầu kéo có được đi thẳng không?"), "going_straight")
        self.assertEqual(extract_vehicle("Xe đầu kéo có được đi thẳng không?"), "tractor_unit")
        for sign_code in ("S.501", "S.503e", "S.G.11a", "S.H.3b"):
            parsed = parse_query_intent_deterministic(f"{sign_code} có ý nghĩa gì?")
            self.assertEqual(parsed.query_intent.intents[0].target.sign_codes, [sign_code])
        self.assertEqual(extract_vehicle("Xe ô tô taxi có được đi vào không?"), "taxi")
        self.assertEqual(extract_vehicle("Xe máy có được đi vào không?"), "motorcycle_or_moped")
        self.assertEqual(extract_vehicle("Xe mô tô có được đi vào không?"), "motorbike")
        self.assertEqual(extract_vehicle("Xe gắn máy có được đi vào không?"), "moped")
        self.assertEqual(extract_vehicle("Xe đạp thồ có được đi vào không?"), "cargo_bicycle")
        self.assertEqual(extract_vehicle("Xe ba gác máy có được đi vào không?"), "motorized_tricycle")
        self.assertEqual(extract_vehicle("Ô tô khách có được đi vào không?"), "bus")
        self.assertEqual(extract_vehicle("Xe máy thi công có được đi vào không?"), "construction_machine")
        self.assertEqual(extract_vehicle("Xe máy nông nghiệp có được đi vào không?"), "agricultural_forestry_machine")
        self.assertEqual(extract_vehicle("Xe lăn có được đi vào không?"), "wheelchair")
        self.assertEqual(extract_vehicle("Người đi xích lô có được đi vào không?"), "non_motorized_tricycle")
        self.assertEqual(extract_vehicle("Xe ưu tiên có được đi thẳng không?"), "priority_vehicle")
        self.assertEqual(extract_action("Khoảng cách giữa hai xe có đủ chưa?"), "following_distance")

    def test_obstacle_passing_action_and_explicit_side_become_query_facts(self) -> None:
        left = parse_query_intent_deterministic("Xe máy có được đi vòng bên trái chướng ngại vật không?")
        intent = left.query_intent.intents[0]
        self.assertEqual(intent.kind, "CHECK_PERMISSION")
        self.assertEqual(intent.action.value, "passing_obstacle")
        side = next(item for item in left.context_facts if item.predicate == "intended_obstacle_passing_side")
        self.assertEqual((side.value, side.source, side.status), ("left", "user_query", "provided"))

        ambiguous = parse_query_intent_deterministic(
            "Xe máy có được vòng bên trái hay bên phải chướng ngại vật không?"
        )
        self.assertFalse(any(item.predicate == "intended_obstacle_passing_side" for item in ambiguous.context_facts))

    def test_roundabout_action_and_arrow_relation_are_explicit_only(self) -> None:
        parsed = parse_query_intent_deterministic(
            "Xe máy có được chạy vòng xuyến theo hướng mũi tên không?"
        )
        self.assertEqual(parsed.query_intent.intents[0].action.value, "roundabout_circulation")
        fact = next(item for item in parsed.context_facts
                    if item.predicate == "intended_roundabout_route_follows_arrow")
        self.assertTrue(fact.value)

        ambiguous = parse_query_intent_deterministic(
            "Xe máy có được chạy vòng xuyến không?"
        )
        self.assertFalse(any(item.predicate == "intended_roundabout_route_follows_arrow"
                             for item in ambiguous.context_facts))

    def test_r309_horn_obligation_and_sign_position_are_parsed_explicitly(self) -> None:
        parsed = parse_query_intent_deterministic(
            "Xe ô tô có phải ấn còi tại vị trí đặt biển không?"
        )
        intent = parsed.query_intent.intents[0]
        self.assertEqual(intent.kind, "CHECK_PERMISSION")
        self.assertEqual(intent.action.value, "using_horn")
        position = next(item for item in parsed.context_facts
                        if item.predicate == "vehicle_at_r309_sign_position")
        self.assertTrue(position.value)

    def test_hazardous_cargo_is_never_inferred_from_vehicle_class(self) -> None:
        hazardous = parse_query_intent_deterministic(
            "Xe tải chở hàng nguy hiểm có được rẽ trái không?"
        )
        intent = hazardous.query_intent.intents[0]
        self.assertEqual((intent.actor.value, intent.action.value), ("truck", "turn_left"))
        status = next(item for item in hazardous.context_facts
                      if item.predicate == "hazardous_cargo_status")
        self.assertTrue(status.value)

        non_hazardous = parse_query_intent_deterministic(
            "Xe tải không chở hàng nguy hiểm có được rẽ trái không?"
        )
        status = next(item for item in non_hazardous.context_facts
                      if item.predicate == "hazardous_cargo_status")
        self.assertFalse(status.value)

        unspecified = parse_query_intent_deterministic("Xe tải có được rẽ trái không?")
        self.assertFalse(any(item.predicate == "hazardous_cargo_status"
                             for item in unspecified.context_facts))

    def test_inspection_obligation_question_maps_to_stopping_condition(self) -> None:
        parsed = parse_query_intent_deterministic("Xe tải có phải dừng ở trạm kiểm tra không?")
        intent = parsed.query_intent.intents[0]
        self.assertEqual(intent.kind, "CHECK_PERMISSION")
        self.assertEqual(intent.action.value, "stopping")
        self.assertEqual(intent.actor.value, "truck")

    def test_slower_speed_question_maps_to_speed_action(self) -> None:
        intent = parse_query_intent_deterministic(
            "Xe ô tô có được chạy chậm hơn 50 km/h ở đây không?"
        ).query_intent.intents[0]
        self.assertEqual(intent.kind, "CHECK_PERMISSION")
        self.assertEqual(intent.action.value, "speed")
        self.assertEqual(intent.requested_answer["requested_speed_kmh"], 50)

    def test_speed_reduction_and_safe_distance_questions_keep_distinct_actions(self) -> None:
        slow_down = parse_query_intent_deterministic(
            "Xe ô tô có phải giảm tốc độ khi gặp biển không?"
        ).query_intent.intents[0]
        self.assertEqual((slow_down.kind, slow_down.action.value), ("CHECK_PERMISSION", "reduce_speed"))

        safe_distance = parse_query_intent_deterministic(
            "Xe ô tô phải giữ khoảng cách an toàn bao nhiêu mét?"
        ).query_intent.intents[0]
        self.assertEqual((safe_distance.kind, safe_distance.action.value), ("INFORMATION", "following_distance"))

    def test_llm_fallback_parses_multiple_intents_and_validates(self) -> None:
        client = FakeLLMClient(json.dumps({
            "schema_version": "2.0",
            "intents": [
                {"intent_id": "i1", "kind": "QUERY_SIGN", "confidence": "high"},
                {"intent_id": "i2", "kind": "CHECK_PERMISSION",
                 "action": {"value": "entering", "status": "provided"},
                 "actor": {"value": "truck", "status": "provided"}, "confidence": "medium"},
            ],
            "confidence": "high",
        }))
        query = parse_query_intent("Trường hợp xe chở hàng như hình thì sao?", llm_client=client)
        self.assertEqual(client.calls, 1)
        self.assertEqual([item.kind for item in query.query_intent.intents], ["QUERY_SIGN", "CHECK_PERMISSION"])
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
