import unittest

from src.core.symbolic.fact_schema import SignObservation, TrafficFacts
from src.core.symbolic.sign_relations import link_supplementary_signs
from src.core.symbolic.traffic_rules import _supplementary_context


class TrafficSignOCRPipelineTest(unittest.TestCase):
    def test_p127_ocr_value_is_a_fact_of_its_own_sign(self) -> None:
        speed_sign = SignObservation(
            observation_id="speed-sign",
            role="main",
            bbox=[10, 10, 90, 90],
            sign_code="P.127",
            identity_status="identified",
            confidence="high",
            ocr_texts=[{
                "raw_text": "60",
                "normalized_value": "60",
                "confidence": "high",
                "source": "paddleocr_ppocrv6",
            }],
        )
        facts = TrafficFacts("2.0", "scene", [speed_sign]).to_dict()

        speed_facts = facts["sign_observations"][0]["facts"]
        self.assertIn({
            "predicate": "speed_limit_value",
            "value": 60,
            "status": "observed",
            "source": "main_sign_ocr",
            "raw_text": "60",
            "confidence": "high",
        }, speed_facts)

    def test_supplement_is_linked_to_nearest_main_sign_above(self) -> None:
        farther_sign = SignObservation(
            observation_id="farther-main", role="main", bbox=[0, 0, 100, 60],
            sign_code="P.127", identity_status="identified", confidence="high",
        )
        nearest_sign = SignObservation(
            observation_id="nearest-main", role="main", bbox=[10, 100, 90, 160],
            sign_code="P.130", identity_status="identified", confidence="high",
        )
        panel = SignObservation(
            observation_id="panel", role="supplementary", bbox=[15, 170, 85, 200],
            sign_code="S.508", identity_status="identified", confidence="high",
            ocr_texts=[{
                "raw_text": "Ngày lẻ",
                "normalized_value": "Ngày lẻ",
                "confidence": "high",
                "source": "paddleocr_ppocrv6",
            }],
        )
        observations = [farther_sign, nearest_sign, panel]
        relations = link_supplementary_signs(observations)
        facts = TrafficFacts(
            "2.0", "scene", observations, sign_relations=relations,
        )

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations[0].source_observation_id, "panel")
        self.assertEqual(relations[0].target_observation_id, "nearest-main")
        self.assertEqual(relations[0].confidence, "high")
        inherited = _supplementary_context(facts, nearest_sign)
        parity_fact = next(
            fact for fact in inherited
            if fact.predicate == "supplementary_calendar_parity"
        )
        self.assertEqual(parity_fact.value, "odd_days")
        self.assertEqual(parity_fact.metadata["subject_observation_id"], "nearest-main")


if __name__ == "__main__":
    unittest.main()
