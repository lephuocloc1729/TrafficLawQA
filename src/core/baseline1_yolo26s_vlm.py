"""Baseline 1: fixed YOLO26s detection followed by multimodal VLM answering.

The VLM receives the original image, question/choices, and YOLO26s detections.
This baseline does not use OCR, LawDB, BM25, vector retrieval, or symbolic
reasoning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from src.core.utils import pil_to_base64_png
from src.deps.llm_client import LLMClient


@dataclass
class BaselineResources:
    detector: Any
    llm: LLMClient
    max_tokens: int = 1000


def create_resources(
    device: str | None = None,
    model_path: str | Path | None = None,
) -> BaselineResources:
    """Load the configured YOLO26s detector and OpenAI-compatible VLM client."""
    from src.settings import settings
    from src.deps.detection.yolo26s import Yolo26sSignDetector

    resolved_device = device or (_cuda_available() and "cuda") or "cpu"
    api_keys = [
        key.strip()
        for key in (settings.llm_api_keys or "").split(",")
        if key.strip()
    ]
    if not api_keys:
        raise ValueError("Set LLM_API_KEYS in .env before running this baseline")

    detector = Yolo26sSignDetector(
        device=resolved_device,
        model_path=model_path or settings.sign_model_path,
        confidence_threshold=settings.sign_detection_confidence_threshold,
    )
    llm = LLMClient(settings.llm_base_url, api_keys, settings.llm_model)
    return BaselineResources(
        detector=detector,
        llm=llm,
        max_tokens=min(settings.llm_max_new_tokens, 1000),
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _format_detection(detection: dict[str, Any], index: int) -> str:
    label = detection.get("class_label") or "không rõ lớp"
    code = detection.get("sign_code") or "chưa ánh xạ mã biển"
    name = detection.get("sign_name") or detection.get("sign_description") or ""
    confidence = detection.get("confidence")
    confidence_text = (
        f", độ tin cậy {float(confidence):.2f}"
        if confidence is not None else ""
    )
    identity_status = detection.get("identity_status") or detection.get("qcvn_mapping_status")
    uncertainty = (
        f", trạng thái nhận diện: {identity_status}"
        if identity_status and identity_status not in {"identified", "exact", "mapped"}
        else ""
    )
    detail = f" ({name})" if name and name != label else ""
    return (
        f"- Biển #{index}: lớp YOLO `{label}`{detail}; mã: {code}"
        f"{confidence_text}{uncertainty}."
    )


def answer_image_question(
    image_path: str | Path,
    question: str,
    choices: str | list[str] | None = None,
    resources: BaselineResources | None = None,
) -> dict[str, Any]:
    """Detect signs, then send the image, detections, and question to the VLM."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not question.strip():
        raise ValueError("question must not be empty")
    resources = resources or create_resources()

    image = Image.open(image_path).convert("RGB")
    detections = resources.detector.detect(image)
    sign_context = "\n".join(
        _format_detection(detection, index)
        for index, detection in enumerate(detections, start=1)
    ) or "- YOLO26s không phát hiện được biển báo."
    choice_text = "\n".join(choices) if isinstance(choices, list) else (choices or "")
    prompt = (
        "Bạn là trợ lý hỏi đáp về tình huống giao thông Việt Nam. Trả lời ngắn gọn, "
        "tự nhiên bằng tiếng Việt. Hãy xem ảnh gốc cùng kết quả phát hiện biển YOLO26s "
        "dưới đây để trả lời câu hỏi. Xem nhãn YOLO là gợi ý thị giác, không phải căn cứ "
        "pháp lý đã xác minh; nếu ảnh, nhận diện hoặc kiến thức không đủ chắc chắn, hãy "
        "nêu rõ giới hạn và không đoán. Không bịa điều khoản hay citation. Nếu có lựa "
        "chọn A/B/C/D, hãy nêu chữ cái lựa chọn và giải thích ngắn.\n\n"
        f"CÂU HỎI:\n{question.strip()}\n\n"
        f"LỰA CHỌN:\n{choice_text or '(không có)'}\n\n"
        f"BIỂN YOLO26s NHẬN DIỆN:\n{sign_context}"
    )
    image_b64, image_mime = pil_to_base64_png(image)
    answer = resources.llm.generate_content(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image_mime};base64,{image_b64}"
                    }},
                ],
            }
        ],
        temperature=0,
        max_tokens=resources.max_tokens,
    ) or "Chưa đủ thông tin từ ảnh và kết quả nhận diện để trả lời chắc chắn."

    return {
        "answer": answer.strip(),
        "detected_signs": detections,
        "image_sent_to_vlm": True,
        "citations": [],
        "retrieval": "disabled",
        "reasoning": "disabled",
        "pipeline": "yolo26s + image + VLM",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline 1: YOLO26s + original image + VLM (no OCR/retrieval)"
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--choices", help="Optional choices, one per line")
    parser.add_argument("--model", type=Path, help="YOLO26s checkpoint path")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Inference device")
    args = parser.parse_args()
    resources = create_resources(device=args.device, model_path=args.model)
    result = answer_image_question(
        args.image,
        args.question,
        choices=args.choices.splitlines() if args.choices else None,
        resources=resources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
