from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from PIL import Image

from src.core.utils import pil_to_base64_png
from src.deps.llm_client import LLMClient
from src.prompts.answer_prompt import SYSTEM_PROMPT_SIGN_CROP_PARSE


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def infer_question_intent(question: str) -> dict[str, Any]:
    normalized = _strip_accents(question)
    intent = {
        "is_parking_question": any(
            token in normalized for token in ("do xe", "dung xe", "cam do", "cam dung")
        ),
        "is_speed_question": any(
            token in normalized for token in ("toc do", "km/h", "kmh", "gioi han")
        ),
        "is_lane_question": any(token in normalized for token in ("lan", "lane")),
        "is_direction_question": any(
            token in normalized
            for token in ("re trai", "re phai", "di thang", "quay dau", "huong")
        ),
        "is_vehicle_scope_question": any(
            token in normalized
            for token in (
                "xe may",
                "xe mo to",
                "oto",
                "o to",
                "xe tai",
                "xe khach",
                "xe con",
                "nguoi di bo",
                "phuong tien",
            )
        ),
        "is_time_or_date_question": any(
            token in normalized
            for token in ("ngay le", "ngay chan", "gio", "thoi gian", "h", "phut")
        ),
        "is_prohibition_question": any(
            token in normalized for token in ("cam", "khong duoc", "bi cam")
        ),
        "is_permission_question": any(
            token in normalized for token in ("duoc", "co duoc", "cho phep")
        ),
        "numbers_in_question": re.findall(r"\d+(?:[.,]\d+)?", question),
        "times_in_question": re.findall(
            r"\b\d{1,2}\s*(?:h|giờ|:)\s*\d{0,2}\b", question, flags=re.IGNORECASE
        ),
        "date_parity": (
            "odd"
            if "ngay le" in normalized
            else "even"
            if "ngay chan" in normalized
            else ""
        ),
    }
    return intent


def _extract_json_object(response_content: str) -> dict[str, Any]:
    cleaned = response_content.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Crop parse response must be a JSON object")
    return parsed


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def normalize_crop_parse(
    parsed: dict[str, Any] | None,
    image_name: str,
    raw_response: str = "",
    error: str = "",
) -> dict[str, Any]:
    parsed = parsed or {}
    kind = str(parsed.get("kind") or "unknown").strip().lower()
    if kind not in {"main", "supplementary", "unknown"}:
        kind = "unknown"

    confidence = str(parsed.get("confidence") or "low").strip().lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"

    return {
        "image_name": image_name,
        "kind": kind,
        "text": str(parsed.get("text") or "").strip(),
        "number": _as_list(parsed.get("number")),
        "time": _as_list(parsed.get("time")),
        "vehicle": _as_list(parsed.get("vehicle")),
        "arrow": _as_list(parsed.get("arrow")),
        "colors": _as_list(parsed.get("colors")),
        "shape": str(parsed.get("shape") or "").strip(),
        "symbols": _as_list(parsed.get("symbols")),
        "description": str(parsed.get("description") or "").strip(),
        "confidence": confidence,
        "raw_parse_response": raw_response,
        "parse_error": error,
    }


def compact_crop_parse(crop_parse: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(crop_parse, dict):
        return {}
    return {
        key: value
        for key, value in crop_parse.items()
        if key not in {"raw_parse_response", "parse_error"}
    }


def compact_sign_groups(groups: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for group in groups or []:
        copied = dict(group)
        copied["primary_crop_parse"] = compact_crop_parse(
            copied.get("primary_crop_parse")
        )
        copied["supplementary_signs"] = [
            {
                **item,
                "crop_parse": compact_crop_parse(item.get("crop_parse")),
            }
            for item in copied.get("supplementary_signs", [])
            if isinstance(item, dict)
        ]
        compacted.append(copied)
    return compacted


def parse_sign_crop(
    llm_client: LLMClient,
    sign_image: Image.Image,
    image_name: str,
    question: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> dict[str, Any]:
    sign_b64, sign_mime = pil_to_base64_png(sign_image)
    try:
        response_content = llm_client.generate_content(
            contents=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT_SIGN_CROP_PARSE},
                        {
                            "type": "text",
                            "text": (
                                "QUESTION FOR CONTEXT ONLY\n"
                                f"{question}\n\n"
                                "Do not answer the question. Only parse this crop."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{sign_mime};base64,{sign_b64}"},
                        },
                    ],
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parsed = _extract_json_object(response_content)
        return normalize_crop_parse(parsed, image_name, raw_response=response_content)
    except Exception as exc:
        return normalize_crop_parse(
            None,
            image_name,
            raw_response=locals().get("response_content", ""),
            error=f"{type(exc).__name__}: {exc}",
        )


def parse_detected_signs(
    llm_client: LLMClient,
    sample: dict[str, Any],
    sign_dir: Path,
    temperature: float = 0.0,
    max_tokens: int = 700,
) -> dict[str, Any]:
    question = str(sample.get("question") or "")
    sample["question_intent"] = infer_question_intent(question)
    parse_responses: list[dict[str, Any]] = []

    for sign in sample.get("detected_signs", []):
        if sign.get("crop_parse"):
            parse_responses.append(sign["crop_parse"])
            continue
        image_name = str(sign.get("image_name") or "")
        sign_path = sign_dir / image_name
        if not image_name or not sign_path.exists():
            sign["crop_parse"] = normalize_crop_parse(
                None,
                image_name,
                error="Missing sign crop image.",
            )
            parse_responses.append(sign["crop_parse"])
            continue
        sign_image = Image.open(sign_path).convert("RGB")
        sign["crop_parse"] = parse_sign_crop(
            llm_client=llm_client,
            sign_image=sign_image,
            image_name=image_name,
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        parse_responses.append(sign["crop_parse"])

    sample["sign_parse_responses"] = parse_responses
    return build_sign_groups(sample)


def build_sign_groups(sample: dict[str, Any]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    current_group: dict[str, Any] | None = None

    for index, sign in enumerate(sample.get("detected_signs", []), start=1):
        crop_parse = sign.get("crop_parse") or {}
        kind = crop_parse.get("kind", "unknown")
        is_supplementary = kind == "supplementary"

        if current_group is None or not is_supplementary:
            current_group = {
                "group_id": f"group_{len(groups) + 1}",
                "primary_image_name": sign.get("image_name", ""),
                "primary_crop_parse": crop_parse,
                "supplementary_signs": [],
                "all_image_names": [sign.get("image_name", "")],
                "detected_indices": [index],
                "is_chosen": bool(sign.get("is_chosen")),
            }
            groups.append(current_group)
        else:
            current_group["supplementary_signs"].append(
                {
                    "image_name": sign.get("image_name", ""),
                    "crop_parse": crop_parse,
                    "detected_index": index,
                }
            )
            current_group["all_image_names"].append(sign.get("image_name", ""))
            current_group["detected_indices"].append(index)
            current_group["is_chosen"] = current_group["is_chosen"] or bool(
                sign.get("is_chosen")
            )

        sign["sign_group_id"] = current_group["group_id"]

    chosen_groups = [group for group in groups if group.get("is_chosen")]
    primary_group = chosen_groups[0] if chosen_groups else (groups[0] if groups else None)
    if primary_group:
        primary_group["is_primary"] = True
    sample["sign_groups"] = groups
    sample["primary_sign_group"] = primary_group
    if "question_intent" not in sample:
        sample["question_intent"] = infer_question_intent(str(sample.get("question") or ""))
    return sample
