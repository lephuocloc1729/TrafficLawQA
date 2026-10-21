from PIL import Image
from .grounding_dino import GroundingDINO
from .image_processing import LawDBSignExtractor
from .yoloE import YoloE
from .yolov8m import YoloV8mSignDetector
from .yolo26s import Yolo26sSignDetector
from src.constants import DetectionName


class SignExtractor:
    def __init__(
        self, detector: str = "YoloE", device: str = "cpu", **detector_options
    ):
        if detector == DetectionName.GroundingDINO.value:
            self.sign_extractor = GroundingDINO(device=device)
        elif detector == DetectionName.ImageProcessing.value:
            self.sign_extractor = LawDBSignExtractor()
        elif detector == DetectionName.YoloE.value:
            self.sign_extractor = YoloE(device=device)
        elif detector == DetectionName.YoloV8m.value:
            self.sign_extractor = YoloV8mSignDetector(device=device, **detector_options)
        elif detector == DetectionName.Yolo26s.value:
            self.sign_extractor = Yolo26sSignDetector(device=device, **detector_options)
        else:
            raise ValueError(f"Detector {detector} not supported")

    def crop_signs(self, image_path: str) -> list[Image.Image]:
        image = Image.open(image_path).convert("RGB")
        if isinstance(self.sign_extractor, (YoloV8mSignDetector, Yolo26sSignDetector)):
            detections = self.sign_extractor.detect(image)
            return self.sign_extractor.crop_boxes(detections, image)
        boxes, confidences, class_names = self.sign_extractor(image_path)
        cropped_signs = self.sign_extractor.crop_boxes(boxes, image)
        return cropped_signs

    def detect_and_crop(self, image_path: str) -> list[tuple[Image.Image, dict]]:
        """Return crops with detector metadata where the backend supports it."""
        image = Image.open(image_path).convert("RGB")
        if isinstance(self.sign_extractor, (YoloV8mSignDetector, Yolo26sSignDetector)):
            detections = self.sign_extractor.detect(image)
            return list(zip(self.sign_extractor.crop_boxes(detections, image), detections))
        boxes, confidences, class_names = self.sign_extractor(image_path)
        crops = self.sign_extractor.crop_boxes(boxes, image)
        records = []
        for index, crop in enumerate(crops):
            box = boxes[index] if index < len(boxes) else None
            if hasattr(box, "tolist"):
                box = box.tolist()
            if box is not None:
                box = [float(value) for value in box]
            confidence = confidences[index] if index < len(confidences) else None
            if hasattr(confidence, "item"):
                confidence = float(confidence.item())
            label = class_names[index] if index < len(class_names) else None
            records.append(
                (
                    crop,
                    {
                        "bbox": box,
                        "confidence": confidence,
                        "class_label": str(label) if label is not None else None,
                        "source": type(self.sign_extractor).__name__,
                    },
                )
            )
        return records
