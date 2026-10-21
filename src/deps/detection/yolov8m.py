"""Adapter for the Traffic-sign-VietNam-recognition YOLOv8m checkpoint."""

from __future__ import annotations

import os
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_MODEL_PATH = Path("models/yolov8m_final.pt")
CROSSWALK_PATH = (
    Path(__file__).resolve().parents[2]
    / "core"
    / "symbolic"
    / "catalogs"
    / "yolov8m_qcvn_crosswalk.json"
)


@lru_cache(maxsize=1)
def _crosswalk_by_label() -> dict[str, dict[str, Any]]:
    with CROSSWALK_PATH.open(encoding="utf-8") as stream:
        crosswalk = json.load(stream)
    entries: dict[str, dict[str, Any]] = {}
    for item in crosswalk["items"]:
        entries[_normalize_label(item["class_label"])] = item
        for alias in item.get("accepted_aliases", []):
            entries[_normalize_label(alias)] = item
    return entries


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", "", str(label or "")).lower()


def class_label_to_qcvn(label: str, confidence: float = 1.0) -> dict[str, Any]:
    """Resolve a checkpoint label using the explicit, reviewed crosswalk data."""
    mapping = _crosswalk_by_label().get(_normalize_label(label))
    if not mapping:
        return {
            "sign_code": _canonical_code(label),
            "sign_family": None,
            "identity_status": "ambiguous" if _canonical_code(label) else "unidentified",
            "candidates": [],
            "qcvn_mapping_status": "unmapped",
        }

    mapping_status = mapping["mapping_status"]
    exact_code = mapping.get("sign_code")
    candidates = list(mapping.get("candidate_codes", []))
    if exact_code and confidence < 0.6:
        candidates = list(dict.fromkeys([exact_code, *candidates]))
    identity_status = (
        "identified"
        if exact_code and confidence >= 0.6
        else "ambiguous"
        if exact_code or candidates
        else "unidentified"
    )
    return {
        "sign_code": exact_code,
        "sign_family": mapping.get("sign_family"),
        "identity_status": identity_status,
        "candidates": [
            {"sign_code": code, "confidence": confidence, "source": "yolov8m_qcvn_crosswalk"}
            for code in candidates
        ],
        "attributes": mapping.get("attributes", {}),
        "qcvn_mapping_status": mapping_status,
        "qcvn_section": mapping.get("qcvn_section"),
    }


class YoloV8mSignDetector:
    """Run YOLOv8m and retain the detector's label, score, and box per crop."""

    def __init__(
        self,
        device: str = "cpu",
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.model_path = Path(
            model_path or os.getenv("SIGN_MODEL_PATH") or DEFAULT_MODEL_PATH
        )
        self.confidence_threshold = float(confidence_threshold)
        # Injection keeps unit tests independent of loading third-party checkpoints.
        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"YOLOv8m checkpoint not found at {self.model_path}. "
                    "Set SIGN_MODEL_PATH to its location."
                )
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
        self.model = model

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        result_list = self.model.predict(
            source=image,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not result_list:
            return []
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coords = box.xyxy[0].tolist()
            if isinstance(names, dict):
                label = names.get(class_id, names.get(str(class_id), str(class_id)))
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                label = names[class_id]
            else:
                label = str(class_id)
            label = str(label)
            mapping = class_label_to_qcvn(label, confidence)
            detections.append(
                {
                    "bbox": [float(value) for value in coords],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_label": label,
                    **mapping,
                    "source": "yolov8m",
                }
            )
        return detections

    @staticmethod
    def crop_boxes(
        detections: list[dict[str, Any]], image: Image.Image
    ) -> list[Image.Image]:
        width, height = image.size
        crops: list[Image.Image] = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            box = (
                max(0, int(x1)),
                max(0, int(y1)),
                min(width, int(x2)),
                min(height, int(y2)),
            )
            crops.append(image.crop(box))
        return crops


def _canonical_code(label: str) -> str | None:
    """Accept only fully qualified labels; numeric dataset labels need a reviewed map."""
    match = re.fullmatch(r"([A-Za-z])\.?\s*(\d{1,3})([A-Za-z]?)", label.strip())
    if not match:
        return None
    prefix, number, variant = match.groups()
    return f"{prefix.upper()}.{number}{variant.lower()}"
