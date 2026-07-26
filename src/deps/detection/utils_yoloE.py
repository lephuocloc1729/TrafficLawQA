import numpy as np
from PIL import ImageDraw, ImageFont


def draw_bboxes(image, xyxy, confidence, class_name, color="red"):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    for box, conf, cname in zip(xyxy, confidence, class_name):
        x1, y1, x2, y2 = box
        # label = f"{cname} {conf:.2f}"
        label = f"{conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    for box, conf, cname in zip(xyxy, confidence, class_name):
        x1, y1, x2, y2 = box
        # Get text size
        try:
            # Pillow >= 10
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            text_w, text_h = right - left, bottom - top
        except AttributeError:
            # Pillow cũ
            text_w, text_h = draw.textsize(label, font=font)

        draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)

        draw.text((x1, y1 - text_h), label, fill=(255, 255, 255), font=font)

    return image


def remove_tiny_boxes(xyxy, confidence, class_name, image_size, thres=0.005, masks=None):
    """
    Remove bounding boxes whose area < thres
    """
    # Opt 1: Compare with image size
    # w_img, h_img = image_size
    # img_area = w_img * h_img
    # box_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    # keep_mask = (box_area / img_area) >= thres

    # Opt 2: Compare with the largest sign
    box_area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    if len(box_area) == 0:
        return xyxy, confidence, class_name  # Không có box nào thì trả nguyên
    max_area = box_area.max()
    keep_mask = (box_area / max_area) >= thres

    xyxy_new = xyxy[keep_mask]
    confidence_new = confidence[keep_mask]
    class_name_new = class_name[keep_mask]
    
    try:
        masks_new = masks[keep_mask]
    except Exception:
        masks_new = None

    if masks_new is None:
        return xyxy_new, confidence_new, class_name_new
    else:
        return xyxy_new, confidence_new, class_name_new, masks_new


def remove_duplicated_boxes(xyxy, confidence, class_name, iou_thres=0.8, masks=None):
    num_boxes = len(xyxy)
    keep_mask = np.ones(num_boxes, dtype=bool)

    for i in range(num_boxes):
        if not keep_mask[i]:
            continue
        x1_i, y1_i, x2_i, y2_i = xyxy[i]
        area_i = (x2_i - x1_i) * (y2_i - y1_i)

        for j in range(i + 1, num_boxes):
            if not keep_mask[j]:
                continue
            x1_j, y1_j, x2_j, y2_j = xyxy[j]
            area_j = (x2_j - x1_j) * (y2_j - y1_j)

            # Intersection
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            # Union
            union_area = area_i + area_j - inter_area
            iou = inter_area / union_area if union_area > 0 else 0

            # If IoU > thres, remove the smaller
            if iou > iou_thres:
                if area_i >= area_j:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break

    xyxy_new = xyxy[keep_mask]
    confidence_new = confidence[keep_mask]
    class_name_new = class_name[keep_mask]

    try:
        masks_new = masks[keep_mask]
    except Exception:
        masks_new = None

    if masks_new is None:
        return xyxy_new, confidence_new, class_name_new
    else:
        return xyxy_new, confidence_new, class_name_new, masks_new
