import cv2
import numpy as np
import supervision as sv

from PIL import Image
from ultralytics import YOLOE

from .utils_yoloE import (
    draw_bboxes,
    remove_tiny_boxes,
    remove_duplicated_boxes,
)


CLASS_NAMES = [
    "blue rectangle traffic sign",
    "red circular traffic sign",
    "blue circular traffic sign",
    "red triangle traffic sign",
    "green rectangle traffic sign",
    "white rectangle text traffic sign",
]


class YoloE:
    def __init__(
        self,
        model_name_or_path: str = "models/yoloe-v8l-seg.pt",
        device: str = "cpu",
        score_threshold: float = 0.2,
    ):
        self.score_threshold = score_threshold
        self.model = YOLOE(model=model_name_or_path).to(device)
        self.model.set_classes(CLASS_NAMES, self.model.get_text_pe(CLASS_NAMES))
        self.model.to(device).eval()

    def __call__(self, image_path: str):
        image = Image.open(str(image_path)).convert("RGB")
        results = self.model(image_path, conf=self.score_threshold, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        boxes = detections.xyxy
        confidences = detections.confidence
        class_names = detections.data["class_name"]
        boxes, confidences, class_names = remove_tiny_boxes(
            boxes, confidences, class_names, image.size, thres=0.05
        )
        boxes, confidences, class_names = remove_duplicated_boxes(
            boxes, confidences, class_names, iou_thres=0.8
        )

        xyxy_boxes = boxes.tolist()

        return xyxy_boxes, confidences, class_names

    def crop_boxes(
        self,
        bboxes: list,
        image: Image.Image,
    ) -> list[Image.Image]:
        cropped_images = [image.crop(box) for box in bboxes]
        return cropped_images

    def draw_boxes(
        self,
        image: Image.Image,
        xyxy_boxes: list,
        confidences: list[float],
        class_names: list[str],
    ) -> Image.Image:
        image = draw_bboxes(image, xyxy_boxes, confidences, class_names)
        return image


class YoloESegment:
    def __init__(
        self,
        model_name_or_path: str = "models/yoloe-v8l-seg.pt",
        device: str = "cpu",
        score_threshold: float = 0.2,
    ):
        self.score_threshold = score_threshold
        self.model = YOLOE(model=model_name_or_path).to(device)
        self.model.set_classes(CLASS_NAMES, self.model.get_text_pe(CLASS_NAMES))
        self.model.to(device).eval()

        # --- Count parameters ---
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"🔢 Total parameters: {num_params:,} | Trainable: {trainable_params:,}")

    def __call__(self, image_path: str):
        image = Image.open(str(image_path)).convert("RGB")
        results = self.model(image_path, conf=self.score_threshold, verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])

        # Get segmentation masks
        masks = detections.mask
        boxes = detections.xyxy
        confidences = detections.confidence
        class_names = detections.data["class_name"]

        # Get bounding boxes from masks for filtering
        if masks is not None:
            boxes, confidences, class_names, masks = remove_tiny_boxes(
                boxes, confidences, class_names, image.size, thres=0.05, masks=masks
            )
            boxes, confidences, class_names, masks = remove_duplicated_boxes(
                boxes, confidences, class_names, iou_thres=0.8, masks=masks
            )
        else:
            masks = []
            confidences = []
            class_names = []
            boxes = []

        return masks, boxes, confidences, class_names

    def crop_segments(
        self,
        masks: np.ndarray,
        bboxes: np.ndarray,
        image: Image.Image,
    ) -> list[Image.Image]:
        """
        Crop images based on segmentation masks and bounding boxes.
        Pixels outside the mask are filled with 0.
        """
        if masks is None or len(masks) == 0:
            return []

        cropped_images = []
        image_array = np.array(image)

        for mask, bbox in zip(masks, bboxes):
            # Convert boolean mask to uint8 for cv2.findContours
            mask_uint8 = mask.astype(np.uint8) * 255

            x1, y1, x2, y2 = bbox.astype(int).tolist()

            # Crop the image to bbox
            cropped = image_array[y1:y2, x1:x2]

            # Crop the mask to bbox
            mask_cropped = mask_uint8[y1:y2, x1:x2]

            # padding more 10 pixels for mask_cropped -> prepare to operate smooth mask
            mask_cropped = np.pad(
                mask_cropped, ((10, 10), (10, 10)), mode="constant", constant_values=0
            )

            # make the mask smooth
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_cropped = cv2.morphologyEx(
                mask_cropped,
                cv2.MORPH_OPEN,
                kernel,
                iterations=min(mask_cropped.shape[0], mask_cropped.shape[1]) // 15 + 1,
            )

            # de-padding mask_cropped
            mask_cropped = mask_cropped[10:-10, 10:-10]

            # Create a 3-channel mask
            mask_3ch = np.stack([mask_cropped] * 3, axis=-1).astype(bool)

            # Apply mask: set pixels outside mask to 255
            cropped[~mask_3ch] = 255

            # Convert back to PIL Image
            cropped_pil = Image.fromarray(cropped)
            cropped_images.append(cropped_pil)

        return cropped_images

    def draw_segments(
        self,
        image: Image.Image,
        masks: np.ndarray,
        confidences: list[float],
        class_names: list[str],
    ) -> Image.Image:
        """
        Draw segmentation masks on the image with labels.
        """
        if masks is None or len(masks) == 0:
            return image

        image_array = np.array(image)

        for mask, conf, cls_name in zip(masks, confidences, class_names):
            # Create colored mask overlay
            color = np.random.randint(0, 255, 3)
            mask_colored = np.zeros_like(image_array)
            mask_colored[mask] = color

            # Blend mask with image
            alpha = 0.5
            image_array = cv2.addWeighted(
                image_array, 1 - alpha, mask_colored, alpha, 0
            )

        return Image.fromarray(image_array)

    def draw_boxes(
        self,
        image: Image.Image,
        xyxy_boxes: list,
        confidences: list[float],
        class_names: list[str],
    ) -> Image.Image:
        image = draw_bboxes(image, xyxy_boxes, confidences, class_names)
        return image


if __name__ == "__main__":
    from pathlib import Path

    image_path = Path("data/public_test/public_test_images/public_test_2_1.jpg")
    image = Image.open(image_path).convert("RGB")

    # Demo for YoloE (original bbox detection)
    detector = YoloE()

    bboxes, confidences, class_names = detector(str(image_path))
    cropped_images = detector.crop_boxes(bboxes, image)
    draw_image = detector.draw_boxes(image.copy(), bboxes, confidences, class_names)
    output_dir = Path("test_yoloE/yoloE")
    output_dir.mkdir(exist_ok=True, parents=True)
    draw_image.save(output_dir / image_path.name)

    # Demo for YoloESegment (segmentation-based detection)
    print("\n" + "=" * 50)
    print("Testing YoloESegment class...")
    print("=" * 50)

    segment_detector = YoloESegment()

    masks, bboxes, confidences, class_names = segment_detector(str(image_path))

    if masks is not None and len(masks) > 0:
        print(f"Found {len(masks)} segments")

        # save all masks in to folder masks
        output_dir = Path("test_yoloE/masks")
        output_dir.mkdir(exist_ok=True, parents=True)
        for i, mask in enumerate(masks):
            mask_image = Image.fromarray(mask)
            mask_image.save(output_dir / f"{image_path.stem}_mask_{i}.jpg")

        # Crop segments
        cropped_segments = segment_detector.crop_segments(masks, bboxes, image)
        print(f"Cropped {len(cropped_segments)} segments")

        # Save cropped segments
        output_dir = Path("test_yoloE/yoloE_segments")
        output_dir.mkdir(exist_ok=True, parents=True)

        for i, (cropped_img, conf, cls_name) in enumerate(
            zip(cropped_segments, confidences, class_names)
        ):
            output_path = (
                output_dir
                / f"{image_path.stem}_segment_{i}_{cls_name.replace(' ', '_')}.jpg"
            )
            cropped_img.save(output_path)
            print(
                f"Saved segment {i}: {cls_name} (confidence: {conf:.3f}) -> {output_path}"
            )

        # Draw segments on original image
        draw_boxes_image = segment_detector.draw_boxes(
            image, bboxes.copy(), confidences, class_names
        )
        draw_boxes_image.save(output_dir / f"{image_path.stem}_with_boxes.jpg")

        draw_segments_image = segment_detector.draw_segments(
            image, masks, confidences, class_names
        )
        draw_segments_image.save(output_dir / f"{image_path.stem}_with_segments.jpg")
        print(
            f"Saved image with segments: {output_dir / f'{image_path.stem}_with_segments.jpg'}"
        )
    else:
        print("No segments found")
