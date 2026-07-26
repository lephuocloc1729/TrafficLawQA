import sys
import torch
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Callable

from src.deps.llm_client import LLMClient
from src.deps.detection import SignExtractor
from src.deps.embeddings import get_embedding_model
from src.deps.qdrant import QdrantVectorStore
from src.core.utils import load_json, save_json
from src.core.extract_signs import extract_signs
from src.core.filter_signs import filter_relevant_signs
from src.core.query_signs import query_relevant_signs
from src.settings import settings


def get_relevant_articles(
    llm_client: LLMClient,
    embedding_model: Callable,
    vector_store: QdrantVectorStore,
    sign_extractor: SignExtractor,
    sample: dict[str, any],
    image_dir: Path,
    sign_dir: Path,
    collection_name: str,
    limit: int = 1,
    batch_size: int = 10,
    temperature: float = 0.0,
    max_tokens: int = 5000,
    set_name: str = "train",
) -> dict[str, any]:
    # Step 1. Extract all signs from the given image
    sample = extract_signs(
        sign_extractor=sign_extractor,
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
    )
    # Step 2. Filter relevant signs
    sample = filter_relevant_signs(
        llm_client=llm_client,
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
        batch_size=batch_size,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # Step 3. Query relevant articles from the given signs
    sample = query_relevant_signs(
        embedding_model=embedding_model,
        vector_store=vector_store,
        sample=sample,
        sign_dir=sign_dir,
        collection_name=collection_name,
        limit=limit,
        set_name=set_name,
    )

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter signs from query images using LLM"
    )
    parser.add_argument(
        "--set_name",
        "-s",
        type=str,
        default="train",
        choices=["train", "public_test", "private_test"],
        help="Dataset set to process",
    )
    parser.add_argument(
        "--begin_idx",
        "-b",
        type=int,
        default=0,
        help="Starting index for processing samples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="LLM batching size for detected signs",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    # Setup LLM client
    llm_client = LLMClient(
        base_url=settings.llm_base_url,
        api_keys=[api_key for api_key in settings.llm_api_keys.split(",")],
        model_id=settings.llm_model,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder, _ = get_embedding_model(
        settings.embedding_name,
        device=device,
    )
    sign_extractor = SignExtractor(
        detector=settings.image_query_detection_model,
        device=device,
    )
    vector_store = QdrantVectorStore(uri=settings.qdrant_uri)

    # Set up paths based on configuration and set name
    if args.set_name == "train":
        input_json_path = settings.train_json_path
        original_image_dir = Path(settings.train_image_dir)
        extracted_sign_dir = Path(settings.train_extracted_sign_dir)
        save_json_path = Path(settings.train_extracted_sign_json_path)
    elif args.set_name == "public_test":
        input_json_path = settings.public_test_json_path
        original_image_dir = Path(settings.public_test_image_dir)
        extracted_sign_dir = Path(settings.public_test_extracted_sign_dir)
        save_json_path = Path(settings.public_test_extracted_sign_json_path)
    elif args.set_name == "private_test":
        input_json_path = settings.private_test_json_path
        original_image_dir = Path(settings.private_test_image_dir)
        extracted_sign_dir = Path(settings.private_test_extracted_sign_dir)
        save_json_path = Path(settings.private_test_extracted_sign_json_path)
    else:
        print(f"[ERROR] Unknown set name: {args.set_name}")
        sys.exit(1)

    extracted_sign_dir.mkdir(parents=True, exist_ok=True)

    logging.info("[..] Configuration:")
    logging.info("  Set Name: %s", args.set_name)
    logging.info("  Input JSON Path: %s", input_json_path)
    logging.info("  Input Image Dir: %s", original_image_dir)
    logging.info("  Output Image Dir: %s", extracted_sign_dir)
    logging.info("  Output JSON Path: %s", save_json_path)
    logging.info("  Begin Index: %d", args.begin_idx)
    logging.info("  Batch Size: %d", args.batch_size)

    # Load samples
    try:
        samples: list[dict[str, any]] = load_json(input_json_path)
    except FileNotFoundError:
        print(f"[ERROR] Input JSON file not found: {input_json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load input JSON: {e}")
        sys.exit(1)

    # Process samples
    for sample in tqdm(samples[args.begin_idx :], total=len(samples[args.begin_idx :])):
        sample = get_relevant_articles(
            llm_client=llm_client,
            embedding_model=embedder,
            vector_store=vector_store,
            sign_extractor=sign_extractor,
            sample=sample,
            image_dir=original_image_dir,
            sign_dir=extracted_sign_dir,
            collection_name=settings.collection_name,
            limit=settings.limit,
            batch_size=args.batch_size,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_new_tokens,
            set_name=args.set_name,
        )

        # Persist incrementally
        save_json(samples, save_json_path)

    print(f"[..] Successfully processed {len(samples[args.begin_idx:])} samples")
    print(f"[..] Results saved to: {save_json_path}")


if __name__ == "__main__":
    main()
