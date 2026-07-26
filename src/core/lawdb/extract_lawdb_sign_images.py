import sys
import torch
from PIL import Image
from pathlib import Path

from src.deps.detection import SignExtractor
from src.core.utils import load_json, save_json
from src.constants import IGNORED_ARTICLE_ID
from src.settings import settings


def main():
    lawdb_image_dir = Path(settings.lawdb_image_dir)
    lawdb_json_preprocessing_path = Path(settings.lawdb_json_preprocessing_path)
    lawdb_json_sign_parsing_path = Path(settings.lawdb_json_sign_parsing_path)
    lawdb_extracted_sign_dir = Path(settings.lawdb_extracted_sign_dir)
    lawdb_extracted_sign_dir.mkdir(parents=True, exist_ok=True)

    sign_extractor = SignExtractor(
        detector=settings.image_lawdb_detection_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("[..] Configuration:")
    print(f"\tDetection Name: {settings.image_lawdb_detection_model}")
    print(f"\tLawDB Image Dir: {lawdb_image_dir}")
    print(f"\tLawDB JSON Path: {lawdb_json_preprocessing_path}")
    print(f"\tSign Image Dir: {lawdb_extracted_sign_dir}")
    print(f"\tLawDB Save Path: {lawdb_json_sign_parsing_path}")

    try:
        lawdb = load_json(lawdb_json_preprocessing_path)
    except FileNotFoundError:
        print(f"[ERROR] LawDB JSON file not found: {lawdb_json_preprocessing_path}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Failed to load LawDB JSON: {e}")
        sys.exit(1)

    counts = {}

    for law in lawdb:
        law_id = law["id"]
        print(f"[..] Processing LAW {law_id}\n")
        counts[law_id] = 0

        for i, article in enumerate(law["articles"], 1):
            print(f"\t{i}/{len(law['articles'])} - article id {article['id']}")

            if article["id"].startswith(IGNORED_ARTICLE_ID):
                continue

            signs = []
            for image_name in article["images"]:
                image_path = lawdb_image_dir / image_name

                if not image_path.exists():
                    print(f"\t\t[WARNING] Image not found: {image_path}")
                    continue

                try:
                    # signs detection
                    cropped_images = sign_extractor.crop_signs(image_path)

                    for j, cropped_image in enumerate(cropped_images):
                        cropped_name = f"{image_path.stem}_crop{j}.jpg"
                        cropped_image.save(lawdb_extracted_sign_dir / cropped_name)
                        signs.append(cropped_name)

                except Exception as e:
                    print(f"\t\t[ERROR] Failed to process image {image_name}: {e}")
                    continue

            # Add new field
            article["signs"] = signs
            article["num_signs"] = len(signs)
            counts[law_id] += len(signs)

    # Save new lawdb
    try:
        save_json(lawdb, lawdb_json_sign_parsing_path)
        print(f"[..] Processed LawDB saved to: {lawdb_json_sign_parsing_path}")
        print(f"[..] Total signs: {counts}")
    except Exception as e:
        print(f"[ERROR] Failed to save LawDB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
