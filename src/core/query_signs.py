from PIL import Image
from pathlib import Path
from typing import Callable

from src.deps.qdrant import QdrantVectorStore
from src.constants import DEFAULT_ARTICLES


default_article = {"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}


def query_relevant_signs(
    sample: dict[str, any],
    sign_dir: Path,
    embedding_model: Callable,
    vector_store: QdrantVectorStore,
    collection_name: str,
    limit: int = 1,
    set_name: str = "train",
):
    predicted_articles = []
    for sign_info in sample["detected_signs"]:
        if sign_info["is_chosen"]:
            image_path = sign_dir / sign_info["image_name"]
            image = Image.open(image_path)
            embedding = embedding_model.embed_image(image)
            retrieved_results = vector_store.query(
                collection_name,
                embedding,
                top_k=limit,
            )

            sign_info["image_lawdb_path"] = retrieved_results[0].payload["image_path"]
            sign_info["sign_name"] = retrieved_results[0].payload["sign_name"]
            sign_info["retrieved_payloads"] = [
                dict(retrieved_result.payload or {})
                for retrieved_result in retrieved_results
            ]

            for retrieved_result in retrieved_results:
                payload = retrieved_result.payload
                article = {
                    "law_id": payload["law_id"],
                    "article_id": payload["article_id"],
                }

                if article not in predicted_articles:
                    predicted_articles.append(article)

    final_relevant_articles = []
    has_specical_signs = False
    for relevant_article in predicted_articles:
        article_id = relevant_article["article_id"]

        final_relevant_articles.append(relevant_article)

        # Rule 2
        for first_char, article in DEFAULT_ARTICLES.items():
            if article_id.startswith(first_char):
                if article not in final_relevant_articles:
                    final_relevant_articles.append(article)
                    has_specical_signs = True
                break

    # Rule 3
    if not has_specical_signs:
        final_relevant_articles.append(default_article)

    # Rule 1
    if not len(final_relevant_articles):
        final_relevant_articles = [default_article]

    if set_name == "train":
        sample["predicted_articles"] = final_relevant_articles
    elif set_name in ["public_test", "private_test"]:
        sample["relevant_articles"] = final_relevant_articles

    return sample
