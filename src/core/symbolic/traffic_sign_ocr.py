"""Vietnamese full-image OCR using Paddle text detection and VietOCR recognition."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from PIL import Image


PADDLEOCR_VERSION = "3.7.0"
VIETOCR_VERSION = "0.3.13"
TEXT_DETECTION_MODEL = "PP-OCRv6_medium_det"
VIETOCR_MODEL = "vgg_transformer"


def resolve_ocr_device(requested: str = "gpu") -> str:
    """Resolve an OCR device, requiring CUDA when GPU is explicitly requested."""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("OCR_DEVICE must be one of: auto, cpu, gpu")
    if normalized == "cpu":
        return "cpu"

    try:
        import torch
        import paddle

        gpu_available = (
            torch.cuda.is_available()
            and paddle.is_compiled_with_cuda()
            and paddle.device.cuda.device_count() > 0
        )
    except (ImportError, RuntimeError, AttributeError) as exc:
        if normalized == "gpu":
            raise RuntimeError(
                "GPU OCR is the default, but PyTorch or PaddlePaddle could not "
                "initialize CUDA. Install CUDA-enabled builds or set "
                "TRAFFIC_SIGN_OCR_DEVICE=cpu."
            ) from exc
        return "cpu"

    if gpu_available:
        return "gpu"
    if normalized == "gpu":
        raise RuntimeError(
            "GPU OCR is the default, but CUDA is unavailable in PyTorch or "
            "PaddlePaddle. Install CUDA-enabled builds or set "
            "TRAFFIC_SIGN_OCR_DEVICE=cpu."
        )
    return "cpu"


def _result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "json", None)
        if callable(payload):
            payload = payload()
        if not isinstance(payload, dict):
            payload = getattr(result, "res", None)
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]
    return payload if isinstance(payload, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else []


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def _score_value(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        scores = [_score_value(item) for item in value]
        return sum(scores) / len(scores) if scores else 0.0
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "mean"):
        try:
            value = value.mean()
        except (TypeError, ValueError, RuntimeError):
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, RuntimeError):
            pass
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class TrafficSignOCR:
    """Full-image OCR: PaddleOCR detects text regions; VietOCR reads each crop."""

    def __init__(self, device: str = "gpu", language: str = "vi") -> None:
        self.device = resolve_ocr_device(device)
        self.language = language
        if language.lower() not in {"vi", "vietnamese"}:
            raise ValueError("VietOCR is configured for Vietnamese text (vi).")

        self._require_version("paddleocr", PADDLEOCR_VERSION)
        self._require_version("vietocr", VIETOCR_VERSION)

        from paddleocr import TextDetection
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        # VietOCR is a recognizer, so retain a text-region detector and replace
        # PP-OCR's recognition stage with VietOCR for each detected line.
        self.detector = TextDetection(
            model_name=TEXT_DETECTION_MODEL,
            device=self.device,
        )
        config = Cfg.load_config_from_name(VIETOCR_MODEL)
        config["device"] = "cuda:0" if self.device == "gpu" else "cpu"
        self.recognizer = Predictor(config)
        self.ocr_version = f"{TEXT_DETECTION_MODEL}+VietOCR-{VIETOCR_MODEL}"

    @staticmethod
    def _require_version(package: str, required: str) -> None:
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Install the OCR extra with {package}=={required}."
            ) from exc
        if installed != required:
            raise RuntimeError(
                f"This pipeline requires {package}=={required}; found {installed}."
            )

    def recognize(self, image_path: str | Path) -> list[dict[str, Any]]:
        image_path = Path(image_path)
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        results = self.detector.predict(str(image_path))
        extracted: list[dict[str, Any]] = []
        for result in results or []:
            payload = _result_dict(result)
            polygons = _as_list(payload.get("dt_polys"))

            for polygon in polygons:
                if hasattr(polygon, "tolist"):
                    polygon = polygon.tolist()
                if not isinstance(polygon, list) or not polygon:
                    continue
                try:
                    points = [
                        [float(point[0]), float(point[1])]
                        for point in polygon
                        if isinstance(point, (list, tuple)) and len(point) >= 2
                    ]
                except (TypeError, ValueError, IndexError):
                    continue
                if len(points) < 4:
                    continue

                left = max(0, int(min(point[0] for point in points)))
                top = max(0, int(min(point[1] for point in points)))
                right = min(image.width, int(max(point[0] for point in points)) + 1)
                bottom = min(image.height, int(max(point[1] for point in points)) + 1)
                if right <= left or bottom <= top:
                    continue

                crop = image.crop((left, top, right, bottom))
                try:
                    text, probability = self.recognizer.predict(crop, return_prob=True)
                except TypeError:
                    # Support VietOCR releases whose predictor returns text only.
                    text, probability = self.recognizer.predict(crop), 0.0
                text = str(text or "").strip()
                if not text:
                    continue
                score = _score_value(probability)
                extracted.append({
                    "raw_text": text,
                    "normalized_value": text,
                    "confidence": _confidence_label(score),
                    "score": score,
                    "bbox": points,
                    "source": "vietocr_vgg_transformer",
                })
        return extracted


@lru_cache(maxsize=4)
def get_traffic_sign_ocr(device: str = "gpu", language: str = "vi") -> TrafficSignOCR:
    """Load one detector/recognizer pair per device and language."""
    return TrafficSignOCR(device=device, language=language)
