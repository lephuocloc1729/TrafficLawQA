import argparse
from loguru import logger
from pathlib import Path
from src.deps.detection import SignExtractor
from src.settings import settings
from src.core.symbolic.traffic_sign_ocr import get_traffic_sign_ocr


def _ocr_yolo_detections(
    image_path: Path,
    detections: list[dict],
) -> None:
    """OCR the source image once, then associate text boxes with YOLO signs."""
    if not settings.traffic_sign_ocr_enabled or not detections:
        return
    if not any("yolo26" in str(item.get("source") or "").lower() for item in detections):
        return

    try:
        engine = get_traffic_sign_ocr(
            device=settings.traffic_sign_ocr_device,
            language=settings.traffic_sign_ocr_language,
        )
        # PaddleOCR returns text polygons in coordinates of the original image.
        image_texts = engine.recognize(image_path)
    except Exception as exc:
        logger.warning("Could not initialize traffic-sign OCR: {}", exc)
        for detection in detections:
            detection["ocr_texts"] = []
            detection["ocr_status"] = "unavailable"
        return

    associations: list[list[dict]] = [[] for _ in detections]
    for text_item in image_texts:
        polygon = text_item.get("bbox")
        if not isinstance(polygon, list) or not polygon:
            continue
        points = [
            point for point in polygon
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if not points:
            continue
        tx1 = min(float(point[0]) for point in points)
        ty1 = min(float(point[1]) for point in points)
        tx2 = max(float(point[0]) for point in points)
        ty2 = max(float(point[1]) for point in points)
        text_width = max(1.0, tx2 - tx1)
        text_height = max(1.0, ty2 - ty1)
        text_area = text_width * text_height
        text_center_x = (tx1 + tx2) / 2
        text_center_y = (ty1 + ty2) / 2

        candidates: list[tuple[tuple[float, ...], int, str]] = []
        for index, detection in enumerate(detections):
            bbox = detection.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in bbox)
            box_width = max(1.0, x2 - x1)
            box_height = max(1.0, y2 - y1)
            overlap_x = max(0.0, min(x2, tx2) - max(x1, tx1))
            overlap_y = max(0.0, min(y2, ty2) - max(y1, ty1))
            overlap_ratio = (overlap_x * overlap_y) / text_area

            # Text inside a detected sign belongs to that sign. Prefer the box
            # with greatest text overlap, then the tighter box if detections overlap.
            if overlap_ratio >= 0.5 or (
                x1 <= text_center_x <= x2 and y1 <= text_center_y <= y2
            ):
                box_area = box_width * box_height
                candidates.append(((0.0, -overlap_ratio, box_area), index, "detected_sign"))
                continue

            # A panel directly below a sign can be attached when its text is
            # horizontally aligned and no farther than 1.5 sign heights away.
            vertical_gap = ty1 - y2
            horizontal_overlap_ratio = overlap_x / min(box_width, text_width)
            center_offset = abs(text_center_x - (x1 + x2) / 2) / box_width
            if (
                vertical_gap >= -text_height * 0.25
                and vertical_gap <= max(32.0, box_height * 1.5)
                and horizontal_overlap_ratio >= 0.3
            ):
                candidates.append((
                    (1.0, max(0.0, vertical_gap) / box_height, center_offset),
                    index,
                    "below_sign_context",
                ))

        if candidates:
            _, selected_index, role = min(candidates, key=lambda item: item[0])
            associated = dict(text_item)
            associated["region_role"] = role
            associated["associated_sign_bbox"] = detections[selected_index].get("bbox")
            associations[selected_index].append(associated)

    for detection, texts in zip(detections, associations):
        detection["ocr_texts"] = texts
        detection["ocr_status"] = "read" if texts else "no_text"


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
        detections = sign_extractor.detect_and_crop(image_path=str(image_path))
        detection_metadata = [metadata for _, metadata in detections]
        detection_dir = Path(sign_dir) if sign_dir is not None else image_dir
        _ocr_yolo_detections(image_path, detection_metadata)
        for i, (sign, metadata) in enumerate(detections):
            sign = sign.convert("RGB")
            sign_name = f"{sample['image_id']}_crop{i}.jpg"
            sign.save(detection_dir / sign_name, format="JPEG")
            detected = {"image_name": sign_name, "is_chosen": False, **metadata}
            if detected.get("source") == "yolo26s_vtsdb100":
                detected.setdefault("role", "main")
            sign_code = detected.get("sign_code")
            detected["identity_status"] = detected.get("identity_status") or (
                "identified" if sign_code else "ambiguous"
            )
            detected["candidates"] = detected.get("candidates") or (
                [
                    {
                        "sign_code": sign_code,
                        "confidence": detected.get("confidence"),
                        "source": "yolov8m",
                    }
                ]
                if sign_code
                else []
            )
            sample["detected_signs"].append(detected)
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
