import unittest

from src.core.symbolic.sign_normalizer import (
    extract_sign_code,
    normalize_detected_sign_to_observation,
    normalize_payload_to_observation,
    normalize_payloads_to_observations,
    normalize_sign_code,
    sign_family,
)


class SignObservationNormalizerTest(unittest.TestCase):
    def test_extracts_and_normalizes_official_style_codes(self) -> None:
        self.assertEqual(extract_sign_code("Biển số P.131a Cấm đỗ xe"), "P.131a")
        self.assertEqual(extract_sign_code("bien P,131B cam do xe"), "P.131b")
        self.assertEqual(normalize_sign_code("p131C"), "P.131c")
        self.assertEqual(sign_family("P.131b"), "P.131")

    def test_payload_becomes_observation_without_inferred_legal_scope(self) -> None:
        observation = normalize_payload_to_observation({
            "sign_code": "P131b",
            "sign_name": "Biển số P.131b Cấm đỗ xe ngày lẻ",
            "legal_conditions": {"time_condition": "ngày lẻ", "vehicle_scope": "xe cơ giới"},
            "related_articles": [{"law_id": "QCVN", "article_id": "31"}],
            "image_path": "data/signs/p131b.jpg",
        }, visual_evidence=[{"image_name": "crop.jpg"}])
        payload = observation.to_dict()
        self.assertEqual(payload["sign_code"], "P.131b")
        self.assertEqual(payload["identity_status"], "identified")
        self.assertEqual(payload["legal_refs"][0]["article_id"], "31")
        self.assertEqual(payload["evidence"][0]["image_name"], "crop.jpg")
        self.assertNotIn("time_condition", payload)
        self.assertNotIn("target_vehicle", payload)
        self.assertEqual(payload["metadata"]["legal_conditions_unverified"]["time_condition"], "ngày lẻ")

    def test_unidentified_observation_preserves_candidates_and_ocr(self) -> None:
        observation = normalize_payload_to_observation({
            "image_name": "crop-1.jpg",
            "identity_status": "ambiguous",
            "candidates": [
                {"sign_code": "P.131b", "score": 0.55},
                {"sign_name": "P.131c Cấm đỗ ngày chẵn", "score": 0.42},
            ],
            "ocr_text": "6:00-9:00",
        })
        self.assertIsNone(observation.sign_code)
        self.assertEqual(observation.identity_status, "ambiguous")
        self.assertEqual([item["sign_code"] for item in observation.candidates], ["P.131b", "P.131c"])
        self.assertEqual(observation.ocr_texts[0]["raw_text"], "6:00-9:00")

    def test_detected_sign_uses_primary_retrieval_payload(self) -> None:
        observation = normalize_detected_sign_to_observation(
            sign={"image_name": "crop-0.jpg", "role": "main"},
            payloads=[
                {"sign_name": "P.130 Cấm dừng xe và đỗ xe"},
                {"sign_name": "P.131a Cấm đỗ xe", "is_primary": True,
                 "law_id": "QCVN", "article_id": "31"},
            ],
        )
        self.assertEqual(observation.sign_code, "P.131a")
        self.assertEqual(observation.role, "main")
        self.assertEqual(observation.legal_refs[0]["article_id"], "31")

    def test_payload_list_creates_separate_observations(self) -> None:
        observations = normalize_payloads_to_observations([
            {"sign_name": "P.123a Cấm rẽ trái", "image_name": "crop-a.jpg"},
            {"sign_name": "P.123b Cấm rẽ phải", "image_name": "crop-b.jpg"},
        ])
        self.assertEqual([item.sign_code for item in observations], ["P.123a", "P.123b"])
        self.assertEqual([item.observation_id for item in observations], ["crop-a.jpg", "crop-b.jpg"])


if __name__ == "__main__":
    unittest.main()
