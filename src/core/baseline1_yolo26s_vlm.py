"""Baseline 1: fixed YOLO26s detection followed by multimodal VLM answering.

The VLM receives the original image, question/choices, and YOLO26s detections.
This baseline does not use OCR, LawDB, BM25, vector retrieval, or symbolic
reasoning.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from openai import OpenAI

from src.settings import settings


@dataclass
class BaselineResources:
    detector: Any
    llm: OpenAI
    model: dict[str, Any]


def create_resources(
    device: str | None = None,
    model_path: str | Path | None = None,
    vlm_config_path: str | Path = "configs/baseline2_vlm_only.json",
) -> BaselineResources:
    """Load YOLO26s and the same VLM configuration used by Baseline 2."""
    from src.deps.detection.yolo26s import Yolo26sSignDetector

    vlm_config = json.loads(Path(vlm_config_path).read_text(encoding="utf-8"))
    model = vlm_config["model"]
    base_url = os.environ.get(model.get("base_url_env", "LLM_BASE_URL")) or settings.llm_base_url
    configured_keys = (
        os.environ.get(model.get("api_key_env", "LLM_API_KEYS"))
        or settings.llm_api_keys
        or ""
    )
    api_keys = [key.strip() for key in configured_keys.split(",") if key.strip()]
    if not base_url or not api_keys:
        raise ValueError(
            f"Set {model.get('base_url_env', 'LLM_BASE_URL')} and "
            f"{model.get('api_key_env', 'LLM_API_KEYS')} before running this baseline"
        )

    resolved_device = device or (_cuda_available() and "cuda") or "cpu"
    detector = Yolo26sSignDetector(
        device=resolved_device,
        model_path=model_path or settings.sign_model_path,
        confidence_threshold=settings.sign_detection_confidence_threshold,
    )
    llm = OpenAI(base_url=base_url, api_key=api_keys[0])
    return BaselineResources(
        detector=detector,
        llm=llm,
        model=model,
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


def _mime_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "application/octet-stream")


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
        f"CÂU HỎI:\n{question.strip()}\n\n"
        f"LỰA CHỌN:\n{choice_text or '(không có)'}\n\n"
        f"BIỂN YOLO26s NHẬN DIỆN:\n{sign_context}"
    )
    image_url = (
        f"data:{_mime_type(image_path)};base64,"
        f"{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    )
    model = resources.model
    response = resources.llm.chat.completions.create(
        model=model["model_id"],
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý trả lời câu hỏi về tình huống giao thông dựa trực tiếp "
                    "trên ảnh được cung cấp và kiến thức sẵn có của bạn. Chỉ trả lời câu "
                    "hỏi bằng tiếng Việt tự nhiên, trực tiếp và ngắn gọn. Xem danh sách "
                    "YOLO26s là gợi ý nhận diện thị giác, không phải căn cứ pháp lý đã "
                    "xác minh. Nếu ảnh hoặc thông tin không đủ căn cứ, hãy nói rõ điều "
                    "đó và không đoán. Không cần viện dẫn điều luật hoặc tạo citation; "
                    "nếu nhắc đến quy định, không khẳng định đó là trích dẫn đã xác minh."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=float(model.get("temperature", 0.0)),
        top_p=float(model.get("top_p", 0.1)),
        max_tokens=int(model.get("max_tokens", 1000)),
    )
    answer = response.choices[0].message.content or ""
    answer = answer or "Chưa đủ thông tin từ ảnh và kết quả nhận diện để trả lời chắc chắn."

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
    parser.add_argument(
        "--vlm-config",
        type=Path,
        default=Path("configs/baseline2_vlm_only.json"),
        help="VLM configuration; defaults to Baseline 2's configuration",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Inference device")
    args = parser.parse_args()
    resources = create_resources(
        device=args.device,
        model_path=args.model,
        vlm_config_path=args.vlm_config,
    )
    result = answer_image_question(
        args.image,
        args.question,
        choices=args.choices.splitlines() if args.choices else None,
        resources=resources,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
