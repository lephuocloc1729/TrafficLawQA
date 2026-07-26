import torch

from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


SIGN_EXTRACTION_PROMPT = "rectangle blue and white sign. circular green and white sign. triangle red and yellow sign. circular red and white traffic sign."


class GroundingDINO:
    def __init__(
        self,
        model_name_or_path: str = "IDEA-Research/grounding-dino-base",
        device: str = "cpu",
    ):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name_or_path
        )
        self.model.to(device).eval()

        # --- Count parameters ---
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"🔢 Total parameters: {num_params:,} | Trainable: {trainable_params:,}")

    def __call__(self, image_path: str):
        image = Image.open(image_path)
        image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            text=SIGN_EXTRACTION_PROMPT,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            # box_threshold=0.15,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )

        result = self.remove_boxes(results[0], image.size)
        result = self.apply_simple_nms(result, iou_threshold=0.3)

        boxes = result["boxes"].detach().cpu().numpy().tolist()
        scores = result["scores"].detach().cpu().numpy().tolist()
        labels = result["labels"]

        return boxes, scores, labels

    def remove_boxes(
        self,
        result: dict[str, any],
        image_size: tuple[int, int],
        aspect_ratio_threshold: float = 3,
        area_ratio_threshold: float = 0.005,
    ):
        w, h = image_size
        image_area = h * w

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        text_labels = result["text_labels"]

        keep_indices = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            box_area = width * height
            if box_area / image_area < area_ratio_threshold:
                continue

            if height / width > aspect_ratio_threshold:
                continue

            if width / height > aspect_ratio_threshold:
                continue

            keep_indices.append(i)

        filtered_result = {
            "boxes": boxes[keep_indices],
            "scores": scores[keep_indices],
            "labels": [labels[i] for i in keep_indices],
            "text_labels": [text_labels[i] for i in keep_indices],
        }
        return filtered_result

    def apply_simple_nms(
        self,
        result: dict[str, any],
        iou_threshold: float = 0.5,
    ) -> dict[str, any]:
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        text_labels = result["text_labels"]

        if boxes.size(0) == 0:
            return result

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ious = box_iou(boxes, boxes)

        keep = []
        suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

        for i in range(len(boxes)):
            if suppressed[i]:
                continue

            overlap_inds = (ious[i] > iou_threshold).nonzero(as_tuple=False).flatten()

            labeled_inds = [
                j.item() for j in overlap_inds if text_labels[j] not in ["", None]
            ]

            if labeled_inds:
                selected_idx = max(labeled_inds, key=lambda j: areas[j])
            else:
                selected_idx = overlap_inds[torch.argmax(areas[overlap_inds])].item()

            keep.append(selected_idx)
            suppressed[overlap_inds] = True

        filtered_result = {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": [labels[i] for i in keep],
            "text_labels": [text_labels[i] for i in keep],
        }
        return filtered_result

    def crop_boxes(
        self, boxes: list[tuple[float, float, float, float]], image: Image.Image
    ) -> list[Image.Image]:
        cropped_images = [image.crop(box) for box in boxes]
        return cropped_images

    def draw_boxes(
        self,
        image: Image.Image,
        result: dict[str, any],
        chosen_bboxes: list[bool] = None,
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["text_labels"]

        if not chosen_bboxes:
            chosen_bboxes = [True] * len(boxes)

        for i in range(min(len(boxes), len(chosen_bboxes))):
            score = scores[i].item()
            color = "green" if chosen_bboxes[i] else "red"

            box = boxes[i].tolist()
            label = labels[i]

            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 10), f"{label} ({score:.2f})", fill="red")

        return image


if __name__ == "__main__":
    from pathlib import Path

    detector = GroundingDINO()

    image_path = Path("data/public_test/public_test_images/public_test_2_1.jpg")
    image = Image.open(image_path)

    bboxes = detector(str(image_path))
    cropped_images = detector.crop_boxes(bboxes, image)
    draw_image = detector.draw_boxes(image, bboxes)
    draw_image.save(f"src/deps/detectors/outputs/{image_path.name}")
