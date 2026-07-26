import cv2
from PIL import Image
from typing import List, Tuple


class LawDBSignExtractor:
    def __init__(
        self,
        min_area_ratio: float = 0.02,  # keep contours >= 5% of the image area
        border_size: int = 20,  # add border to the image to avoid edge effects
    ):
        self.min_area_ratio = min_area_ratio
        self.border_size = border_size

    def __call__(self, image_path: str) -> List[Image.Image]:
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        min_contour_area = self.min_area_ratio * image_bgr.shape[0] * image_bgr.shape[1]
        image_bgr = cv2.copyMakeBorder(
            image_bgr,
            self.border_size,
            self.border_size,
            self.border_size,
            self.border_size,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at: {image_path}")

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Invert by default so dark shapes on light background become white (foreground)
        _, thresh = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY_INV)

        # Close small gaps then find only external contours (discard inner/sub-contours) -> No need to do this
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        coutour_bboxes = [cv2.boundingRect(c) for c in contours]
        contours = [
            c
            for i, c in enumerate(contours)
            if coutour_bboxes[i][2] / coutour_bboxes[i][3] <= 4
            and coutour_bboxes[i][2] > 10
            and coutour_bboxes[i][2] > 10
        ]  # remove contours that width/height ratio is too large, width or height <= 10
        contours = [
            c for c in contours if cv2.contourArea(c) >= min_contour_area
        ]  # remove contours that area is too small

        # draw contours on the thresh image
        # thresh = cv2.drawContours(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), contours, -1, (0, 0, 255), 1)
        # os.makedirs("tmp", exist_ok=True)
        # cv2.imwrite(f"tmp/{os.path.basename(image_path)}", thresh)

        bboxes: List[Tuple[int, int, int, int]] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x -= self.border_size
            y -= self.border_size
            bboxes.append((x, y, x + w, y + h))

        # split image into 10 x 10 grids, calculate the center belong to which grid, sort top bottom left right
        bboxes.sort(
            key=lambda bbox: ((bbox[1] + bbox[3]) // 20, (bbox[0] + bbox[2]) // 20)
        )

        return bboxes, [], []

    def crop_boxes(
        self, bboxes: List[Tuple[int, int, int, int]], image: Image.Image
    ) -> List[Image.Image]:
        return [image.crop(bbox) for bbox in bboxes]
