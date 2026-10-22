"""GPU-capable OCR adapter for full-image text and traffic-sign crops."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


def resolve_ocr_device(requested: str = "gpu") -> str:
    """Resolve OCR device, requiring a usable CUDA backend when GPU is requested."""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("OCR_DEVICE must be one of: auto, cpu, gpu")
    if normalized == "cpu":
        return normalized

    try:
        import paddle

        gpu_available = (
            paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        )
    except (ImportError, RuntimeError, AttributeError) as exc:
        if normalized == "gpu":
            raise RuntimeError(
                "GPU OCR is the default, but PaddlePaddle could not initialize CUDA. "
                "Install a CUDA-enabled PaddlePaddle build or set "
                "TRAFFIC_SIGN_OCR_DEVICE=cpu."
            ) from exc
        return "cpu"

    if gpu_available:
        return "gpu"
    if normalized == "gpu":
        raise RuntimeError(
            "GPU OCR is the default, but no CUDA-enabled Paddle GPU is available. "
            "Install a CUDA-enabled PaddlePaddle build or set "
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


class TrafficSignOCR:
    """Pinned PaddleOCR 3.7.0 PP-OCRv6 medium wrapper."""

    PADDLEOCR_VERSION = "3.7.0"
    DETECTION_MODEL = "PP-OCRv6_medium_det"
    RECOGNITION_MODEL = "PP-OCRv6_medium_rec"

    def __init__(self, device: str = "gpu", language: str = "vi") -> None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed_version = version("paddleocr")
        except PackageNotFoundError as exc:
            raise RuntimeError(
                f"Install the OCR extra with paddleocr=={self.PADDLEOCR_VERSION}."
            ) from exc
        if installed_version != self.PADDLEOCR_VERSION:
            raise RuntimeError(
                f"This pipeline requires paddleocr=={self.PADDLEOCR_VERSION}; "
                f"found {installed_version}."
            )

        from paddleocr import PaddleOCR

        self.device = resolve_ocr_device(device)
        self.language = language
        options = {
            "lang": language,
            "device": self.device,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        self.pipeline = PaddleOCR(
            ocr_version="PP-OCRv6",
            text_detection_model_name=self.DETECTION_MODEL,
            text_recognition_model_name=self.RECOGNITION_MODEL,
            **options,
        )
        self.ocr_version = "PP-OCRv6"

    def recognize(self, image_path: str | Path) -> list[dict[str, Any]]:
        results = self.pipeline.predict(str(image_path))
        extracted: list[dict[str, Any]] = []
        for result in results or []:
            payload = _result_dict(result)
            texts = _as_list(payload.get("rec_texts"))
            scores = _as_list(payload.get("rec_scores"))
            polygons_value = payload.get("rec_polys")
            if polygons_value is None:
                polygons_value = payload.get("dt_polys")
            polygons = _as_list(polygons_value)

            for index, text_value in enumerate(texts):
                text = str(text_value or "").strip()
                if not text:
                    continue
                try:
                    score = float(scores[index]) if index < len(scores) else 0.0
                except (TypeError, ValueError):
                    score = 0.0
                polygon = polygons[index] if index < len(polygons) else None
                if hasattr(polygon, "tolist"):
                    polygon = polygon.tolist()
                if isinstance(polygon, list):
                    polygon = [
                        [float(value) for value in point]
                        for point in polygon
                        if isinstance(point, (list, tuple))
                    ]
                else:
                    polygon = None
                extracted.append({
                    "raw_text": text,
                    "normalized_value": text,
                    "confidence": _confidence_label(score),
                    "score": score,
                    "bbox": polygon,
                    "source": "paddleocr_ppocrv6_medium",
                })
        return extracted


@lru_cache(maxsize=4)
def get_traffic_sign_ocr(device: str = "gpu", language: str = "vi") -> TrafficSignOCR:
    """Load one OCR model per device/language and reuse it across sign crops."""
    return TrafficSignOCR(device=device, language=language)
