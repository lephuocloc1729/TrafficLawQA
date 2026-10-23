from __future__ import annotations

from typing import Any, Iterable, Protocol

from src.core.symbolic.sign_normalizer import extract_sign_code


class ArticleLookup(Protocol):
    def get_article(self, law_id: str, article_id: str) -> dict[str, str] | None: ...


CONTEXT_SOURCE_PRIORITY = {
    "retrieval": 0,
    "detector": 0,
    "rule": 1,
    "fallback_default_article": 2,
}


def article_key(article: dict[str, Any]) -> tuple[str, str]:
    return (
        str(article.get("law_id") or "").strip(),
        str(article.get("article_id") or "").strip(),
    )


def article_candidate_from_payload(
    payload: dict[str, Any],
    retrieval_rank: int | None = None,
    context_source: str | None = None,
) -> dict[str, Any] | None:
    law_id, article_id = article_key(payload)
    if not law_id or not article_id:
        return None

    source = str(
        context_source
        or payload.get("context_source")
        or payload.get("source")
        or "retrieval"
    ).strip()
    if source not in CONTEXT_SOURCE_PRIORITY:
        source = "retrieval"

    rank_value = payload.get("retrieval_rank", retrieval_rank)
    rank = rank_value if isinstance(rank_value, int) and rank_value > 0 else None
    sign_code = extract_sign_code(
        payload.get("sign_code"),
        payload.get("sign_name"),
        payload.get("sign_description"),
        payload.get("name"),
        payload.get("description"),
    )
    raw_related_codes = payload.get("related_sign_codes", [])
    if not isinstance(raw_related_codes, list):
        raw_related_codes = []
    related_codes = [
        str(code).strip()
        for code in raw_related_codes
        if str(code).strip()
    ]
    if sign_code and sign_code not in related_codes:
        related_codes.append(sign_code)

    return {
        "law_id": law_id,
        "article_id": article_id,
        "context_source": source,
        "retrieval_rank": rank,
        "related_sign_codes": related_codes,
        "reason": str(payload.get("reason") or "").strip(),
    }


def merge_article_candidates(
    candidates: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    insertion_order: list[tuple[str, str]] = []

    for raw_candidate in candidates:
        candidate = article_candidate_from_payload(raw_candidate)
        if candidate is None:
            continue
        key = article_key(candidate)
        if key not in merged:
            merged[key] = candidate
            insertion_order.append(key)
            continue

        current = merged[key]
        current_priority = CONTEXT_SOURCE_PRIORITY[current["context_source"]]
        candidate_priority = CONTEXT_SOURCE_PRIORITY[candidate["context_source"]]
        if candidate_priority < current_priority:
            current["context_source"] = candidate["context_source"]

        ranks = [
            rank
            for rank in (current.get("retrieval_rank"), candidate.get("retrieval_rank"))
            if isinstance(rank, int)
        ]
        current["retrieval_rank"] = min(ranks) if ranks else None
        for sign_code in candidate.get("related_sign_codes", []):
            if sign_code not in current["related_sign_codes"]:
                current["related_sign_codes"].append(sign_code)
        if candidate.get("reason") and not current.get("reason"):
            current["reason"] = candidate["reason"]

    order_index = {key: index for index, key in enumerate(insertion_order)}
    return sorted(
        merged.values(),
        key=lambda item: (
            CONTEXT_SOURCE_PRIORITY[item["context_source"]],
            item["retrieval_rank"] if item["retrieval_rank"] is not None else 10**9,
            order_index[article_key(item)],
        ),
    )


def build_article_contexts(
    candidates: Iterable[dict[str, Any]],
    article_lookup: ArticleLookup,
    max_articles: int,
) -> list[dict[str, Any]]:
    if max_articles <= 0:
        return []
    contexts: list[dict[str, Any]] = []
    for candidate in merge_article_candidates(candidates):
        law_id, article_id = article_key(candidate)
        article = article_lookup.get_article(law_id, article_id)
        if not article:
            continue
        contexts.append(
            {
                "law_id": law_id,
                "article_id": article_id,
                "title": str(article.get("title") or ""),
                "text": str(article.get("text") or ""),
                "context_source": candidate["context_source"],
                "retrieval_rank": candidate["retrieval_rank"],
                "related_sign_codes": candidate["related_sign_codes"],
            }
        )
        if len(contexts) >= max_articles:
            break
    return contexts


def rule_article_candidates(symbolic_context: dict[str, Any]) -> list[dict[str, Any]]:
    sign_codes = [
        str(observation.get("code") or observation.get("sign_code") or "").strip()
        for observation in (
            symbolic_context.get("facts", {}).get("signs")
            or symbolic_context.get("facts", {}).get("sign_observations", [])
        )
        if str(observation.get("code") or observation.get("sign_code") or "").strip()
    ]
    candidates: list[dict[str, Any]] = []
    legal_basis = symbolic_context.get("symbolic_decision", {}).get(
        "legal_basis", []
    )
    for article in legal_basis:
        if not isinstance(article, dict):
            continue
        candidate = article_candidate_from_payload(
            {
                **article,
                "related_sign_codes": sign_codes,
            },
            context_source="rule",
        )
        if candidate:
            candidates.append(candidate)
    return candidates
