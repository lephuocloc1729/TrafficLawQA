from PIL import Image
from .grounding_dino import GroundingDINO
from .image_processing import LawDBSignExtractor
from .yoloE import YoloE
from src.constants import DetectionName


class SignExtractor:
    def __init__(self, detector: str = "YoloE", device: str = "cpu"):
        if detector == DetectionName.GroundingDINO.value:
            self.sign_extractor = GroundingDINO(device=device)
        elif detector == DetectionName.ImageProcessing.value:
            self.sign_extractor = LawDBSignExtractor()
        elif detector == DetectionName.YoloE.value:
            self.sign_extractor = YoloE(device=device)
        else:
            raise ValueError(f"Detector {detector} not supported")

    def crop_signs(self, image_path: str) -> list[Image.Image]:
        image = Image.open(image_path).convert("RGB")
        boxes, confidences, class_names = self.sign_extractor(image_path)
        cropped_signs = self.sign_extractor.crop_boxes(boxes, image)
        return cropped_signs
