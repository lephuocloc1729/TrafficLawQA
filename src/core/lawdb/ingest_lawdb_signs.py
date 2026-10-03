import sys
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from src.core.utils import load_json
from src.core.lawdb.sign_metadata import build_sign_payload
from src.deps.embeddings import get_embedding_model
from src.deps.qdrant import QdrantVectorStore
from src.settings import settings


def main():
    embedder, embedding_dim = get_embedding_model(
        settings.embedding_name, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    vector_store = QdrantVectorStore(uri=settings.qdrant_uri)
    vector_store.create_collection(
        collection_name=settings.collection_name,
        embedding_size=embedding_dim,
        distance="cosine",
    )

    lawdb_image_dir = Path(settings.lawdb_image_dir)
    lawdb_extracted_sign_dir = Path(settings.lawdb_extracted_sign_dir)
    lawdb_json_sign_parsing_path = settings.lawdb_json_sign_parsing_path

    print("[..] Configuration:")
    print(f"\tCollection Name: {settings.collection_name}")
    print(f"\tLawDB Image Dir: {lawdb_image_dir}")
    print(f"\tSign Image Dir: {lawdb_extracted_sign_dir}")
    print(f"\tLawDB Save Path: {lawdb_json_sign_parsing_path}")

    # Load LawDB data
    try:
        lawdb = load_json(lawdb_json_sign_parsing_path)
    except FileNotFoundError:
        print(f"[ERROR] LawDB JSON file not found: {lawdb_json_sign_parsing_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load LawDB JSON: {e}")
        sys.exit(1)

    sign_embeddings = []
    sign_payloads = []

    for law in lawdb:
        law_id = law["id"]
        num_articles = len(law["articles"])
        print(f"[..] Processing LAW {law_id}: {num_articles} articles")

        for article in tqdm(law["articles"], total=num_articles):
            images = article.get("images", [])
            if not images:
                continue

            signs = article.get("detailed_signs", [])
            if not signs:
                continue

            article_id = article.get("id")

            for _, detail_sign_info in enumerate(article["detailed_signs"]):
                image_path = str(lawdb_extracted_sign_dir / detail_sign_info["image"])
                image = Image.open(image_path)
                embedding = embedder.embed_image(image)
                payload = build_sign_payload(
                    law_id=law_id,
                    article_id=article_id,
                    image_path=image_path,
                    sign_info=detail_sign_info,
                )
                sign_embeddings.append(embedding)
                sign_payloads.append(payload)

    vector_store.add_embeddings(
        collection_name=settings.collection_name,
        embeddings=sign_embeddings,
        payloads=sign_payloads,
    )

    print(
        f"[..] Successfully ingested {len(sign_embeddings)} embeddings into collection: {settings.collection_name}"
    )


if __name__ == "__main__":
    main()
