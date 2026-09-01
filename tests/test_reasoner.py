import unittest

from src.core.symbolic.reasoner import decide_symbolically, reason_over_facts
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


if __name__ == "__main__":
    unittest.main()
