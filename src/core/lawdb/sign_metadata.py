from __future__ import annotations

import copy
from typing import Any

from src.core.symbolic.sign_normalizer import (
    SIGN_CODE_RE,
    extract_sign_code,
    normalize_sign_code,
    sign_family,
)


LEGAL_CONDITION_KEYS = ("time_condition", "vehicle_scope", "direction_scope")


def normalize_legal_conditions(value: Any) -> dict[str, str]:
    source = value if isinstance(value, dict) else {}
    return {
        key: str(source.get(key) or "").strip()
        for key in LEGAL_CONDITION_KEYS
    }


def build_sign_payload(
    law_id: str,
    article_id: str,
    image_path: str,
    sign_info: dict[str, Any],
) -> dict[str, Any]:
    normalized_law_id = str(law_id or "").strip()
    normalized_article_id = str(article_id or "").strip()
    sign_name = str(sign_info.get("name") or sign_info.get("sign_name") or "").strip()
    sign_description = str(
        sign_info.get("description") or sign_info.get("sign_description") or ""
    ).strip()
    sign_code = extract_sign_code(
        sign_info.get("sign_code"),
        sign_name,
        sign_description,
    )

    return {
        "law_id": normalized_law_id,
        "article_id": normalized_article_id,
        "article_uid": f"{normalized_law_id}#{normalized_article_id}",
        "parent_article_id": normalized_article_id,
        "sign_code": sign_code,
        "sign_family": sign_family(sign_code),
        "sign_name": sign_name,
        "sign_description": sign_description,
        "legal_conditions": normalize_legal_conditions(
            sign_info.get("legal_conditions")
        ),
        "image_path": str(image_path),
    }


def normalize_parsed_sign_info(
    sign_parsed: dict[str, Any],
    article: dict[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(sign_parsed)
    article_text = " ".join(
        str(article.get(key) or "") for key in ("title", "text")
    )
    article_codes = {
        normalize_sign_code(match)
        for match in SIGN_CODE_RE.findall(article_text)
    }
    proposed_code = extract_sign_code(
        normalized.get("sign_code"),
        normalized.get("name"),
        normalized.get("description"),
    )
    accepted_code = proposed_code if proposed_code in article_codes else ""

    normalized["sign_code"] = accepted_code
    normalized["sign_family"] = sign_family(accepted_code)
    normalized["name"] = str(
        normalized.get("name") or "Biển báo chưa xác định từ context"
    ).strip()
    normalized["description"] = str(normalized.get("description") or "").strip()
    normalized["legal_conditions"] = normalize_legal_conditions(
        normalized.get("legal_conditions")
    )
    return normalized
