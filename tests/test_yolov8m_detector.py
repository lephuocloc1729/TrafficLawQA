from types import SimpleNamespace
import json
from pathlib import Path

from PIL import Image

from src.deps.detection.yolov8m import YoloV8mSignDetector, class_label_to_qcvn
from src.core.symbolic.sign_normalizer import normalize_payload_to_observation


class Value:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class FakeModel:
    names = {0: "131b", 1: "127.4"}

    def predict(self, **kwargs):
        self.kwargs = kwargs
        boxes = [
            SimpleNamespace(cls=Value(0), conf=Value(0.91), xyxy=[Value([1, 2, 8, 9])]),
            SimpleNamespace(cls=Value(1), conf=Value(0.83), xyxy=[Value([9, 2, 16, 9])]),
        ]
        return [SimpleNamespace(boxes=boxes, names=self.names)]


def test_detector_retains_classes_confidence_and_boxes():
    model = FakeModel()
    detector = YoloV8mSignDetector(model=model, confidence_threshold=0.4)
    detections = detector.detect(Image.new("RGB", (20, 12)))

    assert detections[0] == {
        "bbox": [1.0, 2.0, 8.0, 9.0],
        "confidence": 0.91,
        "class_id": 0,
        "class_label": "131b",
        "sign_code": "P.131b",
        "sign_family": "P.131",
        "identity_status": "identified",
        "candidates": [],
        "attributes": {},
        "qcvn_mapping_status": "exact_code",
        "qcvn_section": "Appendix B.31",
        "source": "yolov8m",
    }
    assert detections[1]["sign_code"] == "P.127"
    assert detections[1]["attributes"] == {"speed_limit_kmh": 40}
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["conf"] == 0.4


def test_detector_crops_are_clamped_to_image_bounds():
    image = Image.new("RGB", (10, 10), "white")
    crops = YoloV8mSignDetector.crop_boxes(
        [{"bbox": [-4, 2, 14, 11]}], image
    )

    assert len(crops) == 1
    assert crops[0].size == (10, 8)


def test_family_labels_remain_ambiguous_with_variant_candidates():
    mapping = class_label_to_qcvn("207", confidence=0.94)

    assert mapping["sign_code"] is None
    assert mapping["sign_family"] == "W.207"
    assert mapping["identity_status"] == "ambiguous"
    assert "W.207a" in {candidate["sign_code"] for candidate in mapping["candidates"]}


def test_other_labels_do_not_get_guessed():
    mapping = class_label_to_qcvn("other_1", confidence=0.99)

    assert mapping["sign_code"] is None
    assert mapping["identity_status"] == "unidentified"
    assert mapping["qcvn_mapping_status"] == "unmapped"


def test_exact_label_below_identity_threshold_is_ambiguous():
    mapping = class_label_to_qcvn("131b", confidence=0.49)

    assert mapping["sign_code"] == "P.131b"
    assert mapping["identity_status"] == "ambiguous"
    assert mapping["candidates"][0]["sign_code"] == "P.131b"


def test_exact_label_at_identity_threshold_is_identified():
    mapping = class_label_to_qcvn("131b", confidence=0.5)

    assert mapping["sign_code"] == "P.131b"
    assert mapping["identity_status"] == "identified"


def test_crosswalk_covers_checkpoint_class_ids_without_duplicates():
    path = (
        Path(__file__).parents[1]
        / "src/core/symbolic/catalogs/yolov8m_qcvn_crosswalk.json"
    )
    crosswalk = json.loads(path.read_text(encoding="utf-8"))
    items = crosswalk["items"]

    assert len(items) == 49
    assert [item["class_id"] for item in items] == list(range(49))
    assert len({item["class_label"] for item in items}) == 49


def test_speed_class_attribute_survives_sign_fact_normalization():
    detection = class_label_to_qcvn("127.6", confidence=0.93)
    observation = normalize_payload_to_observation(
        {"image_name": "crop.jpg", "class_label": "127.6", **detection},
        source="yolov8m",
    )

    assert observation.sign_code == "P.127"
    assert observation.visual_attributes["normalized_conditions"]["speed_limit_value"] == 60
