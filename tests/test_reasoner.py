import unittest

from src.core.symbolic.reasoner import (
    build_symbolic_reasoning_context,
    decide_symbolically,
    reason_over_facts,
)
from src.core.symbolic.sign_normalizer import normalize_payload_to_fact


class ReasonerTest(unittest.TestCase):
    def test_reasoner_returns_trace_and_prohibited_decision(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "QCVN 41:2024/BGTVT",
                "article_id": "30",
                "sign_name": "P.130 Cấm dừng xe và đỗ xe",
            }
        )

        trace = reason_over_facts(
            [fact],
            "Xe ô tô có được phép đỗ tại đây không?",
        )
        payload = trace.to_dict()

        self.assertEqual(payload["decision"]["decision"], "PROHIBITED")
        self.assertEqual(payload["decision"]["rules_fired"], ["P130_NO_STOPPING_NO_PARKING"])
        self.assertEqual(payload["rule_matches"][0]["matched_facts"][0]["sign_code"], "P.130")
        self.assertEqual(payload["steps"][0]["name"], "normalize_inputs")
        self.assertEqual(payload["steps"][1]["name"], "apply_manual_rules")

    def test_reasoner_returns_conditional_for_odd_day_parking_sign(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "QCVN 41:2024/BGTVT",
                "article_id": "31",
                "sign_name": "P.131b Cấm đỗ xe ngày lẻ",
            }
        )

        decision = decide_symbolically(
            [fact.to_dict()],
            {"intent": "CHECK_PERMISSION", "action": "parking", "vehicle": "car"},
        )

        self.assertEqual(decision.decision, "CONDITIONAL")
        self.assertEqual(decision.rules_fired, ["P131B_ODD_DAY_NO_PARKING"])
        self.assertEqual(decision.legal_basis[0]["article_id"], "31")

    def test_reasoner_returns_prohibited_for_p131b_on_odd_day(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "QCVN 41:2024/BGTVT",
                "article_id": "31",
                "sign_name": "P.131b Cấm đỗ xe ngày lẻ",
            }
        )

        decision = decide_symbolically(
            [fact],
            "Hôm nay là ngày lẻ, xe ô tô có được đỗ ở đây không?",
        )

        self.assertEqual(decision.decision, "PROHIBITED")
        self.assertEqual(decision.rules_fired, ["P131B_ODD_DAY_NO_PARKING"])
        self.assertEqual(decision.confidence, "high")

    def test_reasoner_no_rule_fallback(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "QCVN 41:2024/BGTVT",
                "article_id": "99",
                "sign_name": "P.999 Biển chưa hỗ trợ",
            }
        )

        decision = decide_symbolically(
            [fact],
            {"intent": "CHECK_PERMISSION", "action": "parking", "vehicle": "car"},
        )

        self.assertEqual(decision.decision, "UNKNOWN")
        self.assertEqual(decision.rules_fired, [])
        self.assertEqual(decision.confidence, "low")

    def test_symbolic_reasoning_context_contains_final_answer_fields(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "QCVN 41:2024/BGTVT",
                "article_id": "31",
                "sign_name": "P.131a Cấm đỗ xe",
            }
        )

        context = build_symbolic_reasoning_context(
            [fact],
            "Xe ô tô có được phép đỗ ở đây không?",
        )

        self.assertEqual(
            set(context),
            {
                "symbolic_facts",
                "question_intent",
                "rules_fired",
                "symbolic_decision",
                "reasoning_trace",
            },
        )
        self.assertEqual(context["question_intent"]["intent"], "CHECK_PERMISSION")
        self.assertEqual(context["rules_fired"], ["P131A_NO_PARKING"])
        self.assertEqual(context["symbolic_decision"]["decision"], "PROHIBITED")
        self.assertEqual(context["symbolic_facts"][0]["sign_code"], "P.131a")
        self.assertEqual(
            context["reasoning_trace"]["decision"]["rules_fired"],
            ["P131A_NO_PARKING"],
        )

    def test_ten_manual_symbolic_examples_have_expected_decisions(self) -> None:
        cases = [
            ("P.130 Cấm dừng xe và đỗ xe", "Xe ô tô có được đỗ không?", "P130_NO_STOPPING_NO_PARKING", "PROHIBITED"),
            ("P.130 Cấm dừng xe và đỗ xe", "Xe máy có được dừng xe không?", "P130_NO_STOPPING_NO_PARKING", "PROHIBITED"),
            ("P.131a Cấm đỗ xe", "Xe ô tô có được đỗ ở đây không?", "P131A_NO_PARKING", "PROHIBITED"),
            ("P.131b Cấm đỗ xe ngày lẻ", "Xe tải có được đỗ ở đây không?", "P131B_ODD_DAY_NO_PARKING", "CONDITIONAL"),
            ("P.131c Cấm đỗ xe ngày chẵn", "Xe buýt có được đỗ ở đây không?", "P131C_EVEN_DAY_NO_PARKING", "CONDITIONAL"),
            ("P.123a Cấm rẽ trái", "Xe ô tô có được rẽ trái không?", "P123A_NO_LEFT_TURN", "PROHIBITED"),
            ("P.123b Cấm rẽ phải", "Xe ô tô có được rẽ phải không?", "P123B_NO_RIGHT_TURN", "PROHIBITED"),
            ("P.102 Cấm đi ngược chiều", "Xe ô tô có được đi vào đường này không?", "P102_NO_ENTRY", "PROHIBITED"),
            ("P.127 Tốc độ tối đa cho phép", "Tốc độ tối đa là bao nhiêu?", "P127_SPEED_LIMIT", "CONDITIONAL"),
            ("P.999 Biển chưa hỗ trợ", "Xe ô tô có được đỗ không?", None, "UNKNOWN"),
        ]

        for sign_name, question, expected_rule, expected_decision in cases:
            with self.subTest(sign_name=sign_name, question=question):
                fact = normalize_payload_to_fact(
                    {
                        "law_id": "QCVN 41:2024/BGTVT",
                        "article_id": "31",
                        "sign_name": sign_name,
                    }
                )
                context = build_symbolic_reasoning_context([fact], question)
                self.assertEqual(
                    context["symbolic_decision"]["decision"],
                    expected_decision,
                )
                if expected_rule is None:
                    self.assertEqual(context["rules_fired"], [])
                else:
                    self.assertEqual(context["rules_fired"], [expected_rule])


if __name__ == "__main__":
    unittest.main()
