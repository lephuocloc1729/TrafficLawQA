"""Adapter for the fine-tuned VTSDB100 YOLO26s sign detector."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from PIL import Image

from src.core.symbolic.sign_normalizer import normalize_sign_code, sign_family
from src.core.symbolic.sign_catalog import load_default_sign_catalog


DEFAULT_MODEL_PATH = Path("models/yolo26s_vtsdb100_best.pt")
QCVN_LAW_ID = "QCVN 41:2024/BGTVT"
PROJECT_DEFINED_SIGN_CODES = frozenset({"LOCAL.001", "LOCAL.002"})


# This mapping follows the user-reviewed VTSDB100 workbook. Dataset-specific
# spellings are preserved as dataset_sign_code; speed classes retain the
# canonical QCVN code plus a numeric attribute for symbolic rule evaluation.
_EXACT_CLASSES: dict[str, tuple[str, dict[str, Any], str]] = {
    "no_entry": ("P.102", {}, "B.2"),
    "no_cars": ("P.103a", {}, "B.3"),
    "no_cars_turn_right": ("P.103b", {}, "B.3"),
    "no_cars_turn_left": ("P.103c", {}, "B.3"),
    "no_motorcycles": ("P.104", {}, "B.4"),
    "no_trucks": ("P.106a", {}, "B.6"),
    # This VTSDB100 class is specifically the 2.5 t variant. QCVN's rule
    # depends on the payload limit printed on the sign, so retain that value.
    "no_trucks_over_2.5t": ("P.106b", {"payload_limit_t": 2.5}, "B.6"),
    "no_pedestrians": ("P.112", {}, "B.12"),
    "height_limit": ("P.117", {}, "B.17"),
    "no_turn_left": ("P.123a", {}, "B.23"),
    "no_turn_right": ("P.123b", {}, "B.23"),
    "no_u_turn": ("P.124a", {}, "B.24"),
    "no_turn_left_and_u_turn": ("P.124c", {}, "B.24"),
    "no_honking": ("P.128", {}, "B.28"),
    "no_stopping_and_parking": ("P.130", {}, "B.30"),
    "no_parking": ("P.131a", {}, "B.31"),
    "no_parking_odd_days": ("P.131b", {"time_condition": "odd_days"}, "B.31"),
    "no_parking_even_days": ("P.131c", {"time_condition": "even_days"}, "B.31"),
    "no_straight_ahead": ("P.136", {}, "B.36"),
    "no_turn_left_and_right": ("P.137", {}, "B.37"),
    "no_cars_u_turn": ("P.124b", {}, "B.24"),
    "no_buses_and_trucks": ("P.107", {}, "B.7"),
    "no_trailers": ("P.108", {}, "B.8"),
    "no_vehicles_carrying_dangerous_goods": ("P.106c", {}, "B.6"),
    "no_bicycles": ("P.110a", {}, "B.10"),
    "no_slow_moving_vehicles": ("LOCAL.001", {}, ""),
    "no_motorcycles_and_slow_vehicles": ("LOCAL.002", {}, ""),
    "give_way": ("W.208", {}, "C.8"),
    "width_limit": ("P.118", {}, "B.18"),
    "weight_limit": ("P.115", {}, "B.15"),
    "axle_weight_limit": ("P.116", {}, "B.16"),
    "no_overtaking": ("P.125", {}, "B.25"),
    "min_distance_between_vehicles": ("P.121", {}, "B.21"),
    "end_no_overtaking": ("DP.133", {}, "B.33"),
    "end_all_prohibitions": ("DP.135", {}, "B.35"),
    "road_closed": ("P.101", {}, "B.1"),
    "pedestrian_crossing": ("W.224", {}, "C.24"),
    "children_crossing": ("W.225", {}, "C.25"),
    "road_works": ("W.227", {}, "C.27"),
    "two_way_traffic": ("W.204", {}, "C.4"),
    "railway_crossing_with_gates": ("W.211", {}, "C.11"),
    "go_straight": ("R.301a", {}, "D.1a"),
    "turn_left_only": ("R.301e", {}, "D.1a"),
    "turn_right_only": ("R.301d", {}, "D.1a"),
    "roundabout": ("R.303", {}, "D.3"),
    "main_road": ("I.401", {}, "E.1"),
    "end_main_road": ("I.402", {}, "E.2"),
    "hospital": ("I.425", {}, "E.26"),
    "bus_stop": ("I.434a", {}, "E.36"),
    "market": ("I.442", {}, "E.45"),
    "checkpoint": ("P.129", {}, "B.29"),
    "one_way_road": ("I.407a", {}, "E.7"),
    "u_turn_area": ("I.409", {}, "E.12"),
    "clearance_height": ("W.239b", {}, "C.39"),
    "slow_down": ("W.245a", {}, "C.45"),
    "accident_area": ("W.244", {}, "C.44"),
    "end_dual_carriageway": ("W.236", {}, "C.36"),
    "tunnel": ("R.E.11a", {}, "D.21"),
    "end_tunnel": ("R.E.11b", {}, "D.21"),
    "road_reserved_for_cars": ("R.403a", {}, "D.11"),
    "expressway": ("I.437", {}, "E.40"),
    "reserved_for_disabled": ("I.446", {}, "E.49"),
    "residential_area": ("R.420", {}, "D.20"),
    "end_residential_area": ("R.421", {}, "D.21"),
    "parking_area": ("I.408", {}, "E.8"),
    "speed_limit_40": ("P.127", {"speed_limit_kmh": 40}, "B.27"),
    "speed_limit_50": ("P.127", {"speed_limit_kmh": 50}, "B.27"),
    "speed_limit_60": ("P.127", {"speed_limit_kmh": 60}, "B.27"),
    "speed_limit_70": ("P.127", {"speed_limit_kmh": 70}, "B.27"),
    "speed_limit_80": ("P.127", {"speed_limit_kmh": 80}, "B.27"),
    "speed_limit_90": ("P.127", {"speed_limit_kmh": 90}, "B.27"),
    "speed_limit_100": ("P.127", {"speed_limit_kmh": 100}, "B.27"),
    "speed_limit_120": ("P.127", {"speed_limit_kmh": 120}, "B.27"),
    "min_speed_limit_30": ("R.306", {"min_speed_kmh": 30}, "D.6"),
    "min_speed_limit_60": ("R.306", {"min_speed_kmh": 60}, "D.6"),
    "end_min_speed_limit_30": ("R.307", {}, "D.7"),
    "steep_descent_5_percent": ("W.219", {"slope_percent": 5}, "C.19"),
    "steep_ascent_5_percent": ("W.220", {"slope_percent": 5}, "C.20"),
    "dangerous_curve_left": ("W.201a", {}, "C.1"),
    "dangerous_curve_right": ("W.201b", {}, "C.1"),
    "cliff_ahead": ("W.215a", {}, "C.15"),
    "road_narrows": ("W.203a", {}, "C.3"),
    "crossroad": ("W.205a", {}, "C.5"),
    "intersection_with_non_priority_road": ("W.207a", {}, "C.7"),
    "road_bump": ("W.221b", {}, "C.21"),
    "pedestrian_crossing_zone": ("I.423b", {}, "E.23a"),
    "pedestrian_overpass": ("I.424d", {}, "E.25"),
    "multiple_curves": ("W.202a", {}, "C.2"),
    "intersection_with_non_priority_road_right": ("W.207b", {}, "C.7"),
    "t_junction": ("W.205d", {}, "C.5"),
    "t_junction_right": ("W.205b", {}, "C.5"),
    "t_junction_left": ("W.205c", {}, "C.5"),
    "intersection_with_non_priority_road_left": ("W.207c", {}, "C.7"),
    "dead_end": ("I.405c", {}, "E.5"),
    "underpass": ("W.218", {}, "C.18"),
    "road_narrows_left": ("W.203b", {}, "C.3"),
    "road_narrows_right": ("W.203c", {}, "C.3"),
    "keep_right": ("R.302a", {}, "D.2"),
}

# These dataset labels do not identify a single reviewed QCVN sign code. In
# particular, a broad "other prohibition" class must not be treated as P.109
# (tractor prohibition), and the A.01/A.02 labels are legacy dataset codes.
# Keep them detectable while preventing downstream rules from receiving a
# fabricated legal sign identity.
_AMBIGUOUS_CLASSES: dict[str, str] = {
    "other_prohibition_signs": "The class groups multiple prohibition signs; the exact sign is not identified.",
}

_FAMILY_CLASSES: dict[str, tuple[str, list[str], str]] = {
    "overhead_power_lines": ("W.239", ["W.239a", "W.239b"], "C.39"),
}

# Keep dataset-specific spellings that encode a numeric class or use a comma
# separator, while using canonical QCVN codes and attributes for reasoning.
_DATASET_CODE_OVERRIDES = {
    "speed_limit_40": "P.127-40",
    "speed_limit_50": "P.127-50",
    "speed_limit_60": "P.127-60",
    "speed_limit_70": "P.127-70",
    "speed_limit_80": "P.127-80",
    "speed_limit_90": "P.127-90",
    "speed_limit_100": "P.127-100",
    "speed_limit_120": "P.127-120",
    "min_speed_limit_30": "R.306-30",
    "min_speed_limit_60": "R.306-60",
    "tunnel": "R.E,11a",
    "end_tunnel": "R.E,11b",
}


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "_", str(label or "").strip().lower()).strip("_")


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def _article_id_from_catalog(entry: Any, fallback: str) -> str:
    for reference in getattr(entry, "legal_basis", ()) or ():
        section = str(reference.get("section_or_figure") or "")
        match = re.search(r"\b([A-Z]\.\d+[a-z]?)\b", section, re.IGNORECASE)
        if match:
            return match.group(1)
    return fallback


def class_label_to_qcvn(label: str, confidence: float) -> dict[str, Any]:
    """Map a VTSDB100 class to the application's traffic-sign schema."""
    key = _normalize_label(label)
    if key in _AMBIGUOUS_CLASSES:
        is_other_prohibition = key == "other_prohibition_signs"
        return {
            "sign_code": None,
            "sign_family": None,
            "dataset_sign_code": _DATASET_CODE_OVERRIDES.get(key, label),
            "identity_status": "unidentified" if is_other_prohibition else "ambiguous",
            "candidates": [],
            "attributes": {},
            "qcvn_mapping_status": "unmapped" if is_other_prohibition else "unverified_dataset_label",
            "qcvn_section": None,
            "law_id": None,
            "article_id": None,
            "related_articles": [],
            "sign_name": "Unknown" if is_other_prohibition else label.replace("_", " "),
            "sign_description": "",
            "visual_attributes": {"class_attributes": {}},
            "variant_decision": {
                "confidence": _confidence_label(confidence),
                "identity_confirmed": False,
                "method": "yolo26s_unmapped_class" if is_other_prohibition else "yolo26s_unverified_dataset_class",
                "reason": _AMBIGUOUS_CLASSES[key],
            },
        }

    if key in _EXACT_CLASSES:
        code, attributes, article_id = _EXACT_CLASSES[key]
        catalog_entry = load_default_sign_catalog().lookup(code)
        article_id = _article_id_from_catalog(catalog_entry, article_id)
        dataset_code = _DATASET_CODE_OVERRIDES.get(key, code)
        confidence_label = _confidence_label(confidence)
        has_catalog_meaning = catalog_entry is not None and catalog_entry.status != "variant_required"
        identity_status = "identified" if confidence >= 0.5 and has_catalog_meaning else "ambiguous"
        candidates = [] if identity_status == "identified" or not has_catalog_meaning else [
            {
                "sign_code": code,
                "confidence": confidence,
                "source": "yolo26s_vtsdb100",
                "attributes": attributes,
            }
        ]
        related_articles = (
            [{"law_id": QCVN_LAW_ID, "article_id": article_id}]
            if has_catalog_meaning and article_id
            else []
        )
        is_project_defined = code in PROJECT_DEFINED_SIGN_CODES
        return {
            "sign_code": code if has_catalog_meaning else None,
            "sign_family": sign_family(code) if has_catalog_meaning else None,
            "dataset_sign_code": dataset_code,
            "identity_status": identity_status,
            "candidates": candidates,
            "attributes": attributes,
            "qcvn_mapping_status": (
                "dataset_code_unverified"
                if not has_catalog_meaning
                else "project_defined"
                if is_project_defined and identity_status == "identified"
                else "exact_code"
                if identity_status == "identified"
                else "low_confidence"
            ),
            "qcvn_section": article_id if related_articles else None,
            "law_id": QCVN_LAW_ID if related_articles else None,
            "article_id": article_id if related_articles else None,
            "related_articles": related_articles,
            "sign_name": catalog_entry.official_name if has_catalog_meaning else label.replace("_", " "),
            "sign_description": (
                (catalog_entry.visual_description or catalog_entry.legal_meaning)
                if has_catalog_meaning else ""
            ),
            "visual_attributes": {"class_attributes": attributes},
            "variant_decision": {
                "confidence": confidence_label,
                "identity_confirmed": identity_status == "identified",
                "method": "yolo26s_direct_classification",
                "reason": (
                    "Project-defined code based on the supplied dataset label and sign interpretation; not an official QCVN code."
                    if is_project_defined
                    else "Canonical sign class predicted directly by the fine-tuned detector."
                ),
            },
        }

    family = _FAMILY_CLASSES.get(key)
    if family:
        family_code, codes, article_id = family
        return {
            "sign_code": None,
            "sign_family": family_code,
            "dataset_sign_code": _DATASET_CODE_OVERRIDES.get(key, family_code),
            "identity_status": "ambiguous",
            "candidates": [
                {"sign_code": code, "confidence": confidence, "source": "yolo26s_vtsdb100"}
                for code in codes
            ],
            "qcvn_mapping_status": "family_ambiguous",
            "qcvn_section": article_id,
            "law_id": QCVN_LAW_ID,
            "article_id": article_id,
            "related_articles": [{"law_id": QCVN_LAW_ID, "article_id": article_id}],
            "sign_name": label.replace("_", " "),
            "sign_description": label.replace("_", " "),
            "variant_decision": {
                "confidence": _confidence_label(confidence),
                "identity_confirmed": False,
                "method": "yolo26s_family_classification",
                "reason": "The detector class identifies a sign family but not its exact variant.",
            },
        }

    # Allow a future checkpoint whose class names are already official codes.
    code = normalize_sign_code(label.replace("_", "."))
    if re.fullmatch(r"[A-Z]{1,3}(?:\.[A-Z])?\.\d+[a-z]?", code):
        catalog_entry = load_default_sign_catalog().lookup(code)
        if catalog_entry:
            return {
                "sign_code": code,
                "sign_family": sign_family(code),
                "identity_status": "identified" if confidence >= 0.5 else "ambiguous",
                "candidates": [],
                "attributes": {},
                "qcvn_mapping_status": "canonical_label",
                "sign_name": catalog_entry.official_name,
                "sign_description": catalog_entry.legal_meaning or "",
                "variant_decision": {
                    "confidence": _confidence_label(confidence),
                    "identity_confirmed": confidence >= 0.5,
                    "method": "yolo26s_direct_classification",
                    "reason": "The model class label is an official sign code.",
                },
            }

    return {
        "sign_code": None,
        "sign_family": None,
        "identity_status": "unidentified",
        "candidates": [],
        "qcvn_mapping_status": "unmapped",
        "sign_name": label.replace("_", " "),
        "sign_description": "",
        "variant_decision": {
            "confidence": _confidence_label(confidence),
            "identity_confirmed": False,
            "method": "yolo26s_unmapped_class",
            "reason": "No reviewed QCVN class mapping is available for this detector label.",
        },
    }


class Yolo26sSignDetector:
    """Run YOLO26s and return detections compatible with the QA pipeline."""

    def __init__(
        self,
        device: str = "cpu",
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.25,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.model_path = Path(
            model_path or os.getenv("SIGN_MODEL_PATH") or DEFAULT_MODEL_PATH
        )
        self.confidence_threshold = float(confidence_threshold)
        if model is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"YOLO26s checkpoint not found at {self.model_path}. "
                    "Set SIGN_MODEL_PATH to the fine-tuned yolo26s_vtsdb100_best.pt file."
                )
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
        self.model = model

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        result_list = self.model.predict(
            source=image,
            device=self.device,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not result_list:
            return []
        result = result_list[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", None) or getattr(self.model, "names", {})
        detections: list[dict[str, Any]] = []
        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            coords = box.xyxy[0].tolist()
            if isinstance(names, dict):
                label = names.get(class_id, names.get(str(class_id), str(class_id)))
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                label = names[class_id]
            else:
                label = str(class_id)
            label = str(label)
            detections.append({
                "bbox": [float(value) for value in coords],
                "confidence": confidence,
                "class_id": class_id,
                "class_label": label,
                **class_label_to_qcvn(label, confidence),
                "source": "yolo26s_vtsdb100",
            })
        return detections

    @staticmethod
    def crop_boxes(
        detections: list[dict[str, Any]], image: Image.Image
    ) -> list[Image.Image]:
        width, height = image.size
        crops: list[Image.Image] = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            box = (
                max(0, int(x1)),
                max(0, int(y1)),
                min(width, int(x2)),
                min(height, int(y2)),
            )
            crops.append(image.crop(box))
        return crops
