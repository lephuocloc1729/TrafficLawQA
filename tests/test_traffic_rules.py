import unittest

from src.core.symbolic.question_intent import parse_question_intent_deterministic
from src.core.symbolic.sign_normalizer import normalize_payload_to_fact
from src.core.symbolic.traffic_rules import (
    TRAFFIC_RULES,
    find_matching_rules,
    vehicle_is_covered,
)


def fact_for(sign_name: str):
    return normalize_payload_to_fact(
        {
            "law_id": "QCVN 41:2024/BGTVT",
            "article_id": "31",
            "sign_name": sign_name,
        }
    )


class TrafficRulesTest(unittest.TestCase):
    def test_required_rules_are_registered(self) -> None:
        rule_ids = {rule.rule_id for rule in TRAFFIC_RULES}

        self.assertEqual(
            {
                "P130_NO_STOPPING_NO_PARKING",
                "P131A_NO_PARKING",
                "P131B_ODD_DAY_NO_PARKING",
                "P131C_EVEN_DAY_NO_PARKING",
                "P123A_NO_LEFT_TURN",
                "P123B_NO_RIGHT_TURN",
                "P102_NO_ENTRY",
                "P127_SPEED_LIMIT",
            },
            rule_ids,
        )

    def test_vehicle_coverage_for_motor_vehicle_rules(self) -> None:
        self.assertTrue(vehicle_is_covered("car", "motor_vehicle"))
        self.assertTrue(vehicle_is_covered("motorbike", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("bicycle", "motor_vehicle"))
        self.assertFalse(vehicle_is_covered("pedestrian", "vehicle"))

    def test_p130_matches_parking_and_stopping(self) -> None:
        fact = fact_for("P.130 Cấm dừng xe và đỗ xe")

        parking_matches = find_matching_rules(
            [fact],
            parse_question_intent_deterministic("Xe ô tô có được phép đỗ không?"),
        )
        stopping_matches = find_matching_rules(
            [fact],
            parse_question_intent_deterministic("Xe ô tô có được dừng xe không?"),
        )

        self.assertEqual(parking_matches[0].rule_id, "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(stopping_matches[0].rule_id, "P130_NO_STOPPING_NO_PARKING")
        self.assertEqual(parking_matches[0].decision, "PROHIBITED")

    def test_variant_parking_rules_match_expected_signs(self) -> None:
        cases = {
            "P.131a Cấm đỗ xe": ("P131A_NO_PARKING", "PROHIBITED"),
            "P.131b Cấm đỗ xe ngày lẻ": ("P131B_ODD_DAY_NO_PARKING", "CONDITIONAL"),
            "P.131c Cấm đỗ xe ngày chẵn": ("P131C_EVEN_DAY_NO_PARKING", "CONDITIONAL"),
        }
        intent = parse_question_intent_deterministic("Xe ô tô có được đỗ xe không?")

        for sign_name, expected in cases.items():
            with self.subTest(sign_name=sign_name):
                matches = find_matching_rules([fact_for(sign_name)], intent)
                self.assertEqual((matches[0].rule_id, matches[0].decision), expected)

    def test_turn_entry_and_speed_rules_match(self) -> None:
        cases = [
            ("P.123a Cấm rẽ trái", "Xe ô tô có được rẽ trái không?", "P123A_NO_LEFT_TURN"),
            ("P.123b Cấm rẽ phải", "Xe ô tô có được rẽ phải không?", "P123B_NO_RIGHT_TURN"),
            ("P.102 Cấm đi ngược chiều", "Xe ô tô có được đi vào không?", "P102_NO_ENTRY"),
            ("P.127 Tốc độ tối đa cho phép", "Tốc độ tối đa là bao nhiêu?", "P127_SPEED_LIMIT"),
        ]

        for sign_name, question, expected_rule in cases:
            with self.subTest(sign_name=sign_name):
                matches = find_matching_rules(
                    [fact_for(sign_name)],
                    parse_question_intent_deterministic(question),
                )
                self.assertEqual(matches[0].rule_id, expected_rule)

    def test_no_match_for_unrelated_action(self) -> None:
        matches = find_matching_rules(
            [fact_for("P.131a Cấm đỗ xe")],
            parse_question_intent_deterministic("Xe ô tô có được rẽ trái không?"),
        )

        self.assertEqual(matches, [])


if __name__ == "__main__":
    unittest.main()
