import argparse
from loguru import logger
from pathlib import Path
from src.deps.detection import SignExtractor


def extract_signs(
    sign_extractor: SignExtractor,
    sample: dict[str, any],
    image_dir: Path,
    sign_dir: str | None = None,
) -> dict[str, any]:
    sample["detected_signs"] = []

    try:
        image_id = sample["image_id"]
        image_path = image_dir / f"{image_id}.jpg"
        cropped_signs = sign_extractor.crop_signs(image_path=image_path)
        for i, sign in enumerate(cropped_signs):
            sign = sign.convert("RGB")
            sign_name = f"{sample['image_id']}_crop{i}.jpg"
            sign.save(sign_dir / sign_name, format="JPEG")
            sample["detected_signs"].append(
                {"image_name": sign_name, "is_chosen": False}
            )
    except Exception as e:
        logger.error(f"Error extracting signs: {e}")
        return sample

    return sample


if __name__ == "__main__":
    import sys
    import torch
    from tqdm import tqdm
    from src.settings import settings
    from src.core.utils import load_json, save_json

    parser = argparse.ArgumentParser(description="Extract signs from images")
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
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sign_extractor = SignExtractor(
        detector=settings.image_query_detection_model,
        device=device,
    )

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

    logger.info("[..] Configuration:")
    logger.info("  Set Name: %s", args.set_name)
    logger.info("  Input JSON Path: %s", input_json_path)
    logger.info("  Input Image Dir: %s", original_image_dir)
    logger.info("  Output Image Dir: %s", extracted_sign_dir)
    logger.info("  Output JSON Path: %s", save_json_path)

    try:
        samples: list[dict[str, any]] = load_json(input_json_path)
    except FileNotFoundError:
        print(f"[ERROR] Input JSON file not found: {input_json_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load input JSON: {e}")
        sys.exit(1)

    for sample in tqdm(samples[args.begin_idx :], total=len(samples[args.begin_idx :])):
        sample = extract_signs(
            sign_extractor=sign_extractor,
            sample=sample,
            image_dir=original_image_dir,
            sign_dir=extracted_sign_dir,
        )

        # Persist incrementally
        save_json(samples, save_json_path)

    print(f"[..] Successfully processed {len(samples[args.begin_idx:])} samples")
    print(f"[..] Results saved to: {save_json_path}")
