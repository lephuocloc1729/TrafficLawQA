"""GPU-capable OCR adapter for text printed on traffic-sign crops."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image


def resolve_ocr_device(requested: str = "auto") -> str:
    """Resolve auto to Paddle's GPU only when this Paddle build can use CUDA."""
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "cpu", "gpu"}:
        raise ValueError("OCR_DEVICE must be one of: auto, cpu, gpu")
    if normalized != "auto":
        return normalized

    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except (ImportError, RuntimeError, AttributeError):
        pass
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
    """PaddleOCR PP-OCRv6 wrapper returning the project's OCR fact schema."""

    def __init__(self, device: str = "auto", language: str = "vi") -> None:
        from paddleocr import PaddleOCR

        self.device = resolve_ocr_device(device)
        self.language = language
        self.pipeline = PaddleOCR(
            lang=language,
            ocr_version="PP-OCRv6",
            device=self.device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

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
                    "source": "paddleocr_ppocrv6",
                })
        return extracted


@lru_cache(maxsize=4)
def get_traffic_sign_ocr(device: str = "auto", language: str = "vi") -> TrafficSignOCR:
    """Load one OCR model per device/language and reuse it across sign crops."""
    return TrafficSignOCR(device=device, language=language)
