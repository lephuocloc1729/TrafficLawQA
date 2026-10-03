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

    def test_week8_required_sign_code_extraction_cases(self) -> None:
        cases = {
            "Biển số P.131a Cấm đỗ xe": "P.131a",
            "Biển P.131b cấm đỗ xe ngày lẻ": "P.131b",
            "P.131c - cấm đỗ xe ngày chẵn": "P.131c",
            "P.130 Cấm dừng xe và đỗ xe": "P.130",
            "Biển báo P.127 tốc độ tối đa": "P.127",
            "Biển báo không có mã hợp lệ": "",
            "Mã P.ABC không đúng định dạng": "",
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_sign_code(text), expected)

    def test_sign_family_removes_variant_suffix(self) -> None:
        self.assertEqual(sign_family("P.131b"), "P.131")
        self.assertEqual(sign_family("P.123A"), "P.123")
        self.assertEqual(sign_family("P.127"), "P.127")
        self.assertEqual(normalize_sign_code("p,131C"), "P.131c")

    def test_normalizes_code_spacing_prefix_case_and_suffix_case(self) -> None:
        cases = {
            "P.131b": "P.131b",
            "P131b": "P.131b",
            "Biển số P.131b": "P.131b",
            "p.131B": "P.131b",
            "P,131B": "P.131b",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(extract_sign_code(raw), expected)

    def test_payload_normalizes_legal_condition_fields_and_all_article_refs(self) -> None:
        fact = normalize_payload_to_fact(
            {
                "sign_code": "P131b",
                "sign_family": "P131",
                "restriction_type": "parking_restriction",
                "legal_conditions": {
                    "time_condition": "cấm vào ngày lẻ",
                    "vehicle_scope": "xe cơ giới",
                    "direction_scope": "theo chiều đặt biển",
                },
                "related_articles": [
                    {"law_id": "QCVN", "article_id": "31"},
                    {"law_id": "QCVN", "article_id": "31"},
                    {"law_id": "QCVN", "article_id": "32"},
                ],
                "source": "lawdb",
            }
        )
        self.assertEqual(fact.sign_code, "P.131b")
        self.assertEqual(fact.sign_family, "P.131")
        self.assertEqual(fact.restriction_type, "parking_restriction")
        self.assertEqual(fact.target_vehicle, "motor_vehicle")
        self.assertEqual(fact.time_condition, "odd_days")
        self.assertEqual(fact.direction_scope, "theo chiều đặt biển")
        self.assertEqual(len(fact.related_articles), 2)
        self.assertEqual(fact.source, "lawdb")

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
                    "is_primary": True,
                },
            ],
            visual_evidence=[{"image_name": "demo_crop0.jpg", "retrieval_rank": 1}],
        )

        self.assertEqual(fact.sign_code, "P.131a")
        self.assertEqual(fact.restriction_type, "no_parking")
        self.assertEqual(fact.related_articles[0]["article_id"], "31")
        self.assertEqual(fact.metadata["retrieval_candidate_count"], 2)

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
