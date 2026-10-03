from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from PIL import Image

from src.core.article_context import (
    article_candidate_from_payload,
    merge_article_candidates,
)
from src.constants import DEFAULT_ARTICLES

if TYPE_CHECKING:
    from src.deps.qdrant import QdrantVectorStore


DEFAULT_ARTICLE = {"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}


def retrieved_article_references(
    payload_groups: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for payloads in payload_groups:
        for rank, payload in enumerate(payloads, start=1):
            candidate = article_candidate_from_payload(
                payload,
                retrieval_rank=rank,
                context_source="retrieval",
            )
            if candidate:
                candidates.append(candidate)

    primary_candidates = merge_article_candidates(candidates)
    references: list[dict[str, Any]] = []
    for candidate in primary_candidates:
        references.append(
            {
                "law_id": candidate["law_id"],
                "article_id": candidate["article_id"],
                "source": "retrieval",
                "reason": "Retrieved from a selected traffic-sign embedding.",
                "retrieval_rank": candidate["retrieval_rank"],
                "related_sign_codes": candidate["related_sign_codes"],
            }
        )

    primary_keys = {
        (candidate["law_id"], candidate["article_id"])
        for candidate in primary_candidates
    }
    related_candidates: list[dict[str, Any]] = []
    for candidate in primary_candidates:
        article_id = candidate["article_id"]
        mapping = next(
            (
                (prefix, related_article)
                for prefix, related_article in DEFAULT_ARTICLES.items()
                if article_id.startswith(prefix)
            ),
            None,
        )
        if mapping is None:
            continue
        prefix, related_article = mapping
        key = (
            str(related_article.get("law_id") or "").strip(),
            str(related_article.get("article_id") or "").strip(),
        )
        if not key[0] or not key[1] or key in primary_keys:
            continue
        related_candidates.append(
            {
                **related_article,
                "context_source": "rule",
                "source": "rule",
                "reason": (
                    f"DEFAULT_ARTICLES maps retrieved {prefix} article "
                    "to this related article."
                ),
                "retrieval_rank": None,
                "related_sign_codes": candidate["related_sign_codes"],
            }
        )
    for candidate in merge_article_candidates(related_candidates):
        references.append(
            {
                "law_id": candidate["law_id"],
                "article_id": candidate["article_id"],
                "source": "rule",
                "reason": candidate["reason"],
                "retrieval_rank": None,
                "related_sign_codes": candidate["related_sign_codes"],
            }
        )
    return references


def fallback_article_reference() -> dict[str, Any]:
    return {
        **DEFAULT_ARTICLE,
        "source": "fallback_default_article",
        "reason": "No valid article was returned by sign retrieval.",
        "retrieval_rank": None,
        "related_sign_codes": [],
    }


def query_relevant_signs(
    sample: dict[str, Any],
    sign_dir: Path,
    embedding_model: Callable,
    vector_store: QdrantVectorStore,
    collection_name: str,
    limit: int = 1,
    set_name: str = "train",
) -> dict[str, Any]:
    payload_groups: list[list[dict[str, Any]]] = []
    for sign_info in sample.get("detected_signs", []):
        if not sign_info.get("is_chosen"):
            continue

        image_path = sign_dir / sign_info["image_name"]
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
        embedding = embedding_model.embed_image(image)
        retrieved_results = vector_store.query(
            collection_name,
            embedding,
            top_k=limit,
        )

        payloads: list[dict[str, Any]] = []
        for rank, retrieved_result in enumerate(retrieved_results, start=1):
            payload = dict(retrieved_result.payload or {})
            payload["retrieval_rank"] = rank
            payload["source"] = "retrieval"
            score = getattr(retrieved_result, "score", None)
            if score is not None:
                payload["score"] = score
            payloads.append(payload)

        sign_info["retrieved_payloads"] = payloads
        payload_groups.append(payloads)
        if payloads:
            primary_payload = payloads[0]
            sign_info["image_lawdb_path"] = primary_payload.get("image_path", "")
            sign_info["sign_name"] = primary_payload.get("sign_name", "")

    articles = retrieved_article_references(payload_groups)
    if not articles:
        articles = [fallback_article_reference()]

    if set_name == "train":
        sample["predicted_articles"] = articles
    elif set_name in {"public_test", "private_test"}:
        sample["relevant_articles"] = articles
    else:
        raise ValueError(f"Unsupported set_name: {set_name}")

    return sample
