import unittest

from src.core.symbolic.sign_normalizer import (
    extract_sign_code,
    normalize_detected_sign_to_fact,
    normalize_payload_to_fact,
    normalize_payloads_to_facts,
    normalize_sign_code,
    sign_family,
)


class SignNormalizerTest(unittest.TestCase):
    def test_extract_sign_code_from_vietnamese_sign_text(self) -> None:
        self.assertEqual(
            extract_sign_code("Biển số P.131a Cấm đỗ xe"),
            "P.131a",
        )
        self.assertEqual(extract_sign_code("bien P,131B cam do xe"), "P.131b")
        self.assertEqual(extract_sign_code("Không có mã biển"), "")

    def test_sign_family_removes_variant_suffix(self) -> None:
        self.assertEqual(sign_family("P.131b"), "P.131")
        self.assertEqual(sign_family("P.123A"), "P.123")
        self.assertEqual(sign_family("P.127"), "P.127")
        self.assertEqual(normalize_sign_code("p,131C"), "P.131c")

    def test_known_sign_payload_maps_to_symbolic_meaning(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "QCVN 41:2024/BGTVT",
                "article_id": "31",
                "sign_name": "Biển số P.131b Cấm đỗ xe ngày lẻ",
                "sign_description": "Cấm đỗ xe vào ngày lẻ.",
                "image_path": "data/law_db/signs/p131b.jpg",
            },
            visual_evidence=[{"image_name": "demo_crop0.jpg", "is_primary": True}],
        )

        payload = fact.to_dict()

        self.assertEqual(payload["sign_code"], "P.131b")
        self.assertEqual(payload["sign_family"], "P.131")
        self.assertEqual(payload["restriction_type"], "no_parking")
        self.assertEqual(payload["target_vehicle"], "motor_vehicle")
        self.assertEqual(payload["time_condition"], "odd_days")
        self.assertEqual(payload["confidence"], "medium")
        self.assertEqual(payload["related_articles"][0]["article_id"], "31")
        self.assertEqual(payload["visual_evidence"][0]["image_name"], "demo_crop0.jpg")

    def test_unknown_sign_fallback_preserves_article_metadata(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "law_id": "36/2024/QH15",
                "article_id": "10",
                "sign_name": "Biển báo chưa có trong rule scope",
                "sign_description": "Không có mã biển rõ ràng.",
            }
        )

        payload = fact.to_dict()

        self.assertEqual(payload["sign_code"], "")
        self.assertEqual(payload["restriction_type"], "unknown")
        self.assertEqual(payload["target_vehicle"], "unknown")
        self.assertEqual(payload["confidence"], "low")
        self.assertEqual(
            payload["related_articles"],
            [{"law_id": "36/2024/QH15", "article_id": "10"}],
        )

    def test_detected_sign_uses_primary_payload_only(self) -> None:
        fact = normalize_detected_sign_to_fact(
            sign={"image_name": "demo_crop0.jpg", "is_chosen": True},
            payloads=[
                {
                    "law_id": "QCVN 41:2024/BGTVT",
                    "article_id": "30",
                    "sign_name": "P.130 Cấm dừng xe và đỗ xe",
                },
                {
                    "law_id": "QCVN 41:2024/BGTVT",
                    "article_id": "31",
                    "sign_name": "P.131a Cấm đỗ xe",
                },
            ],
            visual_evidence=[{"image_name": "demo_crop0.jpg", "retrieval_rank": 1}],
        )

        self.assertEqual(fact.sign_code, "P.130")
        self.assertEqual(fact.restriction_type, "no_stopping_no_parking")
        self.assertEqual(fact.related_articles[0]["article_id"], "30")

    def test_payloads_to_facts_preserves_rank_specific_evidence(self) -> None:
        facts = normalize_payloads_to_facts(
            [
                {"sign_name": "P.123a Cấm rẽ trái"},
                {"sign_name": "P.123b Cấm rẽ phải"},
            ],
            visual_evidence_by_rank=[
                {"image_name": "crop0.jpg", "retrieval_rank": 1},
                {"image_name": "crop0.jpg", "retrieval_rank": 2},
            ],
        )

        self.assertEqual([fact.sign_code for fact in facts], ["P.123a", "P.123b"])
        self.assertEqual(facts[0].restriction_type, "no_left_turn")
        self.assertEqual(facts[1].restriction_type, "no_right_turn")
        self.assertEqual(facts[1].visual_evidence[0]["retrieval_rank"], 2)


if __name__ == "__main__":
    unittest.main()
