import argparse
import re
import json
import time
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Sequence

from src.deps.llm_client import LLMClient
from src.core.lawdb.sign_metadata import (
    normalize_legal_conditions,
    normalize_parsed_sign_info,
)
from src.prompts.parse_signs_prompt import PARSE_SIGNS_PROMPT
from src.core.utils import pil_to_base64_png, load_json, save_json
from src.settings import settings


def build_prompt_parts(
    article_title: str,
    article_content: str,
    extracted_signs: Sequence[Image.Image],
    from_index: int,
    to_index: int,
) -> list[dict[str, any]]:

    image_parts: list[dict[str, any]] = []
    for img in extracted_signs:
        b64, mime = pil_to_base64_png(img)
        image_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )

    num_signs = len(extracted_signs)
    text = (
        PARSE_SIGNS_PROMPT.replace("<<TITLE>>", article_title)
        .replace("<<CONTENT>>", article_content)
        .replace("<<NUM_SIGNS>>", str(num_signs))
        .replace("<<FROM_INDEX>>", str(from_index))
        .replace("<<TO_INDEX>>", str(to_index))
    )

    return [{"type": "text", "text": text}, *image_parts]


def parse_traffic_signs(
    llm_client: LLMClient,
    article_content: str,
    extracted_signs: Sequence[Image.Image],
    article_title: str,
    from_index: int,
    to_index: int,
    temperature: float = 0.0,
    max_tokens: int = 5000,
) -> tuple[str, list[dict[str, any]]]:
    """
    Call the VLM to parse traffic signs in an article section.

    Returns:
        (response_text, parts) where parts is the message payload sent to the LLM.
    """
    try:
        parts = build_prompt_parts(
            article_title=article_title,
            article_content=article_content,
            extracted_signs=extracted_signs,
            from_index=from_index,
            to_index=to_index,
        )

        response_text: str = llm_client.generate_content(
            contents=[{"role": "user", "content": parts}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response_text, parts
    except Exception as e:
        raise RuntimeError(f"Error generating response: {e}") from e


def safe_json_from_llm(text: str) -> any:
    """
    Strip common code fences and parse JSON.
    Accepts ```json ... ``` or ``` ... ``` fences.
    """
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text.strip(),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start < 0 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def fallback_sign_info(
    article: dict[str, any],
    sign_name: str,
    absolute_index: int,
) -> dict[str, str]:
    title = article.get("title") or f"Điều {article.get('id', '')}"
    text = re.sub(r"\s+", " ", article.get("text") or "").strip()
    return {
        "image_tag": f"<<IMAGE_{absolute_index}>>",
        "sign_code": "",
        "sign_family": "",
        "name": title,
        "description": (
            f"Fallback metadata: ảnh crop {sign_name} thuộc điều {article.get('id', '')} "
            f"({title}). {text[:700]}"
        ),
        "legal_conditions": normalize_legal_conditions({}),
        "__fallback": True,
    }


def group_signs_by_image(article: dict[str, any]) -> tuple[list[str], list[list[str]]]:
    """
    Given an article with 'images' and 'signs', return:
    - images: base image ids (without .jpg)
    - detailed_images: lists of cropped sign filenames grouped by base image
    """
    base_image_ids = [p.split(".jpg")[0] for p in article["images"]]
    detailed_images: list[list[str]] = [[] for _ in base_image_ids]
    for sign in article["signs"]:
        image_id = sign.split("_crop")[0]
        try:
            idx = base_image_ids.index(image_id)
        except ValueError:
            # In case a cropped sign refers to a base that isn't in images
            print("Base image id %s not found in article images", image_id)
            continue
        detailed_images[idx].append(sign)
    return base_image_ids, detailed_images


def absolute_to_local_indices(
    detailed_images: Sequence[Sequence[str]],
) -> list[list[int]]:
    """
    For a nested list of signs grouped per image, return the absolute indices list per image.
    Example: [[a,b], [c]] -> [[0,1],[2]]
    """
    output_indices: list[list[int]] = []
    offset = 0
    for group in detailed_images:
        group_len = len(group)
        output_indices.append(list(range(offset, offset + group_len)))
        offset += group_len
    return output_indices


def reindex_placeholders(content: str, local_indices: Sequence[Sequence[int]]) -> str:
    """
    Replace placeholders <<IMAGE_i>> with the sequence of local indices of corresponding cropped signs extracted from that image in article content,
    e.g., <<IMAGE_0>> -> <<IMAGE_0>><<IMAGE_1>>
    """
    updated = content
    # Replace from the end to keep earlier indices stable (as in your code)
    for i in reversed(range(len(local_indices))):
        replacement = "".join(f"<<IMAGE_{j}>>" for j in local_indices[i])
        updated = updated.replace(f"<<IMAGE_{i}>>", replacement)
    return updated


def open_cropped_images(
    cropped_dir: Path,
    filenames: Sequence[str],
) -> list[Image.Image]:
    """Open cropped images as RGB and return them. Ensures files are closed."""
    out: list[Image.Image] = []
    for name in filenames:
        path = cropped_dir / name
        with Image.open(str(path)) as im:
            out.append(im.convert("RGB"))
    return out


def process_article(
    article: dict[str, any],
    llm_client: LLMClient,
    cropped_images_dir: Path,
    chunk_size: int = 10,
    sleep_sec: float = 1.0,
    max_tokens: int = 5000,
) -> None:
    """Mutates `article` with detailed parsing results and bookkeeping fields."""
    article.setdefault("signs", [])

    # Nothing to parse
    if not article["signs"]:
        article["__is_sucessfully_parsing_sign"] = True
        article["detailed_signs"] = []
        return

    # Already parsed
    if article.get("__is_sucessfully_parsing_sign", False):
        return

    _, detailed_images = group_signs_by_image(article)
    absolute_indices_by_image = absolute_to_local_indices(detailed_images)
    num_cropped = len(article["signs"])

    parsed_response_all: list[dict[str, any]] = []

    for start in tqdm(range(0, num_cropped, chunk_size), leave=False):
        stop = min(start + chunk_size, num_cropped)
        allowed = list(range(start, stop))

        # Filter absolute indices per image to just this chunk
        local_indices = [
            [j for j in idxs if j in allowed] for idxs in absolute_indices_by_image
        ]

        content = reindex_placeholders(article["text"], local_indices)

        # Prepare image batch in the chunk order
        cropped_batch_names = [article["signs"][idx] for idx in allowed]
        try:
            cropped_images = open_cropped_images(
                cropped_images_dir, cropped_batch_names
            )
        except FileNotFoundError as e:
            article["__is_sucessfully_parsing_sign"] = False
            article["detailed_signs"] = []
            article["__error"] = f"Missing image file: {e}"
            article["__detailed_signs_for_debug_only"] = []
            print("Missing image file: %s", e)
            return

        # VLM call
        llm_text, _parts = parse_traffic_signs(
            llm_client=llm_client,
            extracted_signs=cropped_images,
            article_content=content,
            article_title=article["title"],
            from_index=allowed[0],
            to_index=allowed[-1],
            max_tokens=max_tokens,
        )

        # Parse LLM JSON
        try:
            parsed = safe_json_from_llm(llm_text)
            if not isinstance(parsed, list):
                raise ValueError("LLM JSON is not a list")
        except Exception as e:
            print("Failed to parse response chunk [%d:%d]: %s", start, stop, e)
            parsed = []

        if len(parsed) != len(allowed) and len(allowed) > 1:
            parsed = []
            for absolute_index in allowed:
                one_name = article["signs"][absolute_index]
                try:
                    one_image = open_cropped_images(cropped_images_dir, [one_name])
                    one_content = reindex_placeholders(
                        article["text"], [[absolute_index]]
                    )
                    one_text, _ = parse_traffic_signs(
                        llm_client=llm_client,
                        extracted_signs=one_image,
                        article_content=one_content,
                        article_title=article["title"],
                        from_index=absolute_index,
                        to_index=absolute_index,
                        max_tokens=max_tokens,
                    )
                    one_parsed = safe_json_from_llm(one_text)
                    if isinstance(one_parsed, list) and one_parsed:
                        parsed.append(one_parsed[0])
                    else:
                        parsed.append(
                            fallback_sign_info(article, one_name, absolute_index)
                        )
                except Exception as retry_error:
                    print(
                        "Failed single-sign retry article %s sign %s: %s",
                        article.get("id"),
                        one_name,
                        retry_error,
                    )
                    parsed.append(fallback_sign_info(article, one_name, absolute_index))
                time.sleep(sleep_sec)

        # Truncate to #items in this chunk
        if len(parsed) < len(allowed):
            parsed.extend(
                fallback_sign_info(article, article["signs"][idx], idx)
                for idx in allowed[len(parsed) :]
            )
        parsed_response_all.extend(parsed[: len(allowed)])

        time.sleep(sleep_sec)

    # Validate & shape final output
    error_msg = ""
    detailed_signs: list[dict[str, any]] = []
    if len(parsed_response_all) != len(article["signs"]):
        is_ok = False
        error_msg = f"Failed to parse response: {len(parsed_response_all)} != {len(article['signs'])}"
    else:
        try:
            for sign_i, sign_parsed in enumerate(parsed_response_all):
                if not isinstance(sign_parsed, dict):
                    raise ValueError(f"Parsed sign {sign_i} is not an object")
                sp = normalize_parsed_sign_info(sign_parsed, article)
                sp["image"] = article["signs"][sign_i]
                detailed_signs.append(sp)
                # Remove volatile tag if present
                if "image_tag" in sp:
                    del sp["image_tag"]
            is_ok = True
        except Exception as e:
            detailed_signs = []
            is_ok = False
            error_msg = f"Failed to extract response: {e}"

    article["__is_sucessfully_parsing_sign"] = is_ok
    article["detailed_signs"] = detailed_signs
    article["__error"] = error_msg
    article["__detailed_signs_for_debug_only"] = parsed_response_all

    if not is_ok:
        print("Parse error: %s (article id: %s)", error_msg, article.get("id"))
        print("Detailed signs (partial): %s", detailed_signs)


def main():
    parser = argparse.ArgumentParser(description="Parse LawDB traffic signs with a VLM")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2500)
    args = parser.parse_args()

    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        api_keys=[api_key for api_key in settings.llm_api_keys.split(",")],
        model_id=settings.llm_model,
    )

    lawdb = load_json(settings.lawdb_json_sign_parsing_path)

    for law in lawdb:
        for article in tqdm(law.get("articles", []), desc="Articles"):
            process_article(
                article=article,
                llm_client=llm_client,
                cropped_images_dir=Path(settings.lawdb_extracted_sign_dir),
                chunk_size=args.chunk_size,
                sleep_sec=args.sleep_sec,
                max_tokens=args.max_tokens,
            )
            # Persist incrementally to avoid large-loss on crash
            save_json(lawdb, settings.lawdb_json_sign_parsing_path)


if __name__ == "__main__":
    main()
