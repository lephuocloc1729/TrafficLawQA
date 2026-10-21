import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.core.symbolic.fact_schema import SignObservation, TrafficFacts
from src.core.symbolic.reasoner import build_symbolic_reasoning_context
from src.core.symbolic.sign_relations import link_supplementary_signs
from src.core.symbolic.traffic_rules import _supplementary_context
from src.core.sub_task_2 import build_freeform_prompt_parts
from src.core.extract_signs import _ocr_yolo_detections


class TrafficSignOCRPipelineTest(unittest.TestCase):
    def test_full_image_ocr_runs_once_and_associates_text_by_position(self) -> None:
        engine = Mock()
        engine.recognize.return_value = [
            {
                "raw_text": "CAM XE",
                "normalized_value": "CAM XE",
                "confidence": "high",
                "bbox": [[20, 20], [80, 20], [80, 35], [20, 35]],
            },
            {
                "raw_text": "7:30-10:30",
                "normalized_value": "7:30-10:30",
                "confidence": "high",
                "bbox": [[25, 110], [85, 110], [85, 125], [25, 125]],
            },
            {
                "raw_text": "HA NOI",
                "normalized_value": "HA NOI",
                "confidence": "high",
                "bbox": [[400, 400], [470, 400], [470, 415], [400, 415]],
            },
        ]
        detections = [{
            "source": "yolo26m_vtsdb100",
            "bbox": [10, 10, 100, 100],
        }]

        with patch("src.core.extract_signs.settings.traffic_sign_ocr_enabled", True), \
             patch("src.core.extract_signs.get_traffic_sign_ocr", return_value=engine):
            _ocr_yolo_detections(Path("scene.jpg"), detections)

        engine.recognize.assert_called_once_with(Path("scene.jpg"))
        self.assertEqual(
            [item["raw_text"] for item in detections[0]["ocr_texts"]],
            ["CAM XE", "7:30-10:30"],
        )
        self.assertEqual(detections[0]["ocr_texts"][1]["region_role"], "below_sign_context")

    def test_information_intent_receives_structured_ocr_facts_in_llm_prompt(self) -> None:
        observation = SignObservation(
            observation_id="main-sign",
            role="main",
            bbox=[100, 100, 300, 300],
            sign_code="P.130",
            identity_status="identified",
            confidence="high",
            source="yolo26m_vtsdb100",
            ocr_texts=[
                {
                    "raw_text": "7:30-10:30",
                    "normalized_value": "7:30-10:30",
                    "confidence": "high",
                    "bbox": [[120, 320], [210, 320], [210, 340], [120, 340]],
                    "source": "paddleocr_ppocrv6",
                    "region_role": "below_sign_context",
                },
                {
                    "raw_text": "16:00-18:30",
                    "normalized_value": "16:00-18:30",
                    "confidence": "high",
                    "bbox": [[120, 345], [220, 345], [220, 365], [120, 365]],
                    "source": "paddleocr_ppocrv6",
                    "region_role": "below_sign_context",
                },
                {
                    "raw_text": "500 m",
                    "normalized_value": "500 m",
                    "confidence": "high",
                    "bbox": [[120, 370], [180, 370], [180, 390], [120, 390]],
                    "source": "paddleocr_ppocrv6",
                    "region_role": "below_sign_context",
                },
            ],
        )
        facts = TrafficFacts("2.0", "scene-1", [observation])
        questions = [
            ("Biển hạn chế này có hiệu lực vào thời gian nào?", "time_condition"),
            ("Biển báo có hiệu lực trong phạm vi bao xa?", "distance_scope"),
        ]

        for question, information_type in questions:
            with self.subTest(question=question):
                symbolic = build_symbolic_reasoning_context(
                    traffic_facts=facts,
                    question=question,
                    use_llm=False,
                )
                intent = symbolic["query_intent"]["intents"][0]
                self.assertEqual(intent["kind"], "INFORMATION")
                self.assertEqual(intent["information_type"], information_type)

                observed_facts = symbolic["facts"]["sign_observations"][0]["facts"]
                time_facts = [
                    item for item in observed_facts
                    if item["predicate"] == "supplementary_time_window"
                ]
                distance_facts = [
                    item for item in observed_facts
                    if item["predicate"] == "supplementary_distance_m"
                ]
                self.assertEqual(len(time_facts), 2)
                self.assertEqual(time_facts[0]["value"]["start_minute"], 450)
                self.assertEqual(time_facts[0]["value"]["end_minute"], 630)
                self.assertEqual(time_facts[1]["value"]["start_minute"], 960)
                self.assertEqual(time_facts[1]["value"]["end_minute"], 1110)
                self.assertEqual(len(distance_facts), 1)
                self.assertEqual(distance_facts[0]["value"], 500.0)
                self.assertTrue(all(item["status"] == "observed" for item in time_facts + distance_facts))

                prompt_parts = build_freeform_prompt_parts(
                    sample={"question": question},
                    image_dir=Path("."),
                    article_contexts=[],
                    visual_evidence=[],
                    sign_prompt_parts=[],
                    symbolic_context=symbolic,
                )
                llm_context = prompt_parts[1]["text"]
                self.assertIn('"predicate": "supplementary_time_window"', llm_context)
                self.assertIn('"predicate": "supplementary_distance_m"', llm_context)
                self.assertIn("7:30-10:30", llm_context)
                self.assertIn("16:00-18:30", llm_context)
                self.assertIn("500.0", llm_context)

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
