import re
import ast
import json
from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

from src.deps.llm_client import LLMClient
from src.prompts.sign_filter_prompt import SIGN_FILTER_PROMPT
from src.core.sign_parse import (
    build_sign_groups,
    compact_crop_parse,
    compact_sign_groups,
    infer_question_intent,
)
from src.core.utils import (
    pil_to_base64_png,
    chunked,
    normalize_bool_tokens,
)


def filter_batch_signs(
    llm_client: LLMClient,
    image: Image.Image,
    question: str,
    signs: Sequence[Image.Image],
    sign_metadata: Sequence[dict[str, Any]] | None = None,
    question_intent: dict[str, Any] | None = None,
    sign_groups: Sequence[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 5000,
) -> str:
    try:
        org_b64, org_mime = pil_to_base64_png(image)
        compact_groups_json = json.dumps(
            compact_sign_groups(list(sign_groups or [])),
            ensure_ascii=False,
        )

        parts: List[dict] = [
            {
                "type": "text",
                "text": SIGN_FILTER_PROMPT.replace("<<NUM_SIGNS>>", str(len(signs)))
                + f"\nQuestion: {question}\n"
                + "QUESTION INTENT:\n"
                + f"{json.dumps(question_intent or {}, ensure_ascii=False)}\n"
                + "SIGN GROUPS:\n"
                + f"{compact_groups_json}\n",
            },
            {"type": "text", "text": "Original Image:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{org_mime};base64,{org_b64}"},
            },
            {"type": "text", "text": "Detected Signs (keep this order):"},
        ]

        # Precompute base64 for detected signs (avoid duplicate conversions)
        for i, sign in enumerate(signs, start=1):
            b64, mt = pil_to_base64_png(sign)
            metadata = (
                sign_metadata[i - 1]
                if sign_metadata is not None and i - 1 < len(sign_metadata)
                else {}
            )
            parts.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"<<<SIGN {i}>>>\n"
                            "SIGN METADATA:\n"
                            f"{json.dumps(metadata, ensure_ascii=False)}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mt};base64,{b64}"},
                    },
                    {"type": "text", "text": f"<<<END SIGN {i}>>>"},
                ]
            )

        return llm_client.generate_content(
            contents=[{"role": "user", "content": parts}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}") from e


def extract_final_answer(response_text: str) -> Union[List[bool], str]:
    """
    Extract a list of booleans from a block marked with:
        <<ANSWER>> ... <</ANSWER>>
    Accepts:
        [True, False]
        True, False
    Returns the boolean list, or the raw answer text if unparseable,
    or "No answer found" if the tags are missing.
    """
    ANSWER_RE = re.compile(
        r"<<ANSWER>>(.*?)<</ANSWER>>", flags=re.DOTALL | re.IGNORECASE
    )

    m = ANSWER_RE.search(response_text)
    if not m:
        return "No answer found"

    FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE)
    answer_text = FENCE_RE.sub("", m.group(1).strip())
    normalized = normalize_bool_tokens(answer_text)

    # Try structured list first
    try:
        parsed = ast.literal_eval(
            normalized if normalized.startswith("[") else f"[{normalized}]"
        )
        if isinstance(parsed, list) and all(isinstance(x, bool) for x in parsed):
            return parsed
    except Exception:
        pass

    # Fallback: CSV like "True, False"
    parts = [p.strip() for p in normalized.strip("[]").split(",") if p.strip()]
    if parts and all(p in ("True", "False") for p in parts):
        return [p == "True" for p in parts]

    # If not parseable, return raw text for visibility
    return answer_text


def filter_relevant_signs(
    llm_client: LLMClient,
    sample: Dict[str, Any],
    image_dir: Path,
    sign_dir: Path,
    batch_size: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 5000,
) -> Tuple[Image.Image, Union[str, None], List[Image.Image], List[bool]]:
    image_id = sample["image_id"]
    question = sample["question"]
    detected_signs = sample["detected_signs"]
    sample["question_intent"] = sample.get("question_intent") or infer_question_intent(
        question
    )
    sample = build_sign_groups(sample)

    image_path = image_dir / f"{image_id}.jpg"
    image = Image.open(image_path).convert("RGB")

    if not detected_signs:
        return sample

    if len(detected_signs) == 1:
        detected_signs[0]["is_chosen"] = True
        return build_sign_groups(sample)

    llm_responses: List[str] = []
    chosen_signs: List[bool] = []

    for batch_items in chunked(detected_signs, batch_size):
        batch_signs = [
            Image.open(sign_dir / sign["image_name"]).convert("RGB")
            for sign in batch_items
        ]
        sign_metadata = [
            {
                "image_name": sign.get("image_name", ""),
                "sign_group_id": sign.get("sign_group_id", ""),
                "crop_parse": compact_crop_parse(sign.get("crop_parse")),
            }
            for sign in batch_items
        ]
        llm_response = filter_batch_signs(
            llm_client,
            image,
            question,
            batch_signs,
            sign_metadata=sign_metadata,
            question_intent=sample.get("question_intent", {}),
            sign_groups=sample.get("sign_groups", []),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        parsed = extract_final_answer(llm_response)

        if isinstance(parsed, list):
            # Pad/truncate to match batch length
            needed = len(batch_signs)
            if len(parsed) < needed:
                parsed = parsed + [False] * (needed - len(parsed))
            elif len(parsed) > needed:
                parsed = parsed[:needed]
            chosen_signs.extend(parsed)
        else:
            chosen_signs.extend([False] * len(batch_signs))

        llm_responses.append(llm_response)

    sample["filter_relevant_signs_explanation"] = "\n============\n".join(llm_responses)

    for i, detected_sign in enumerate(detected_signs):
        detected_sign["is_chosen"] = chosen_signs[i] if i < len(chosen_signs) else False

    return build_sign_groups(sample)
