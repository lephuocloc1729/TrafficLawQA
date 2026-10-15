from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont

from src.deps.detection import SignExtractor
from src.deps.detection.yolov8m import YoloV8mSignDetector
from src.deps.embeddings import get_embedding_model
from src.deps.qdrant import QdrantVectorStore
from src.settings import settings


st.set_page_config(page_title="Sign Model Comparison", page_icon="🚦", layout="wide")


def available_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner="Loading YOLOv8m, image embeddings, and Qdrant…")
def load_comparison_resources(device: str):
    yolov8m_detector = YoloV8mSignDetector(
        device=device,
        model_path=settings.sign_model_path,
        confidence_threshold=settings.sign_detection_confidence_threshold,
    )
    # YoloE's MobileCLIP TorchScript text encoder cannot be loaded on MPS on
    # this machine, so keep the legacy detector on CPU while YOLOv8m and CLIP
    # use the selected device.
    legacy_sign_extractor = SignExtractor(detector="YoloE", device="cpu")
    embedder, _ = get_embedding_model(settings.embedding_name, device=device)
    vector_store = QdrantVectorStore(uri=settings.qdrant_uri)
    return yolov8m_detector, legacy_sign_extractor, embedder, vector_store


def annotate_detections(
    image: Image.Image, detections: list[dict[str, Any]]
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["bbox"])
        color = "#32d583"
        label = f"{detection['class_label']}  {detection['confidence']:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), label, fill="#10261b", font=font)
    return annotated


def payload_image_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path if path.is_file() else None


def render_retrieval_results(
    crop: Image.Image,
    detection: dict[str, Any],
    embedder: Any,
    vector_store: QdrantVectorStore,
    top_k: int,
) -> None:
    st.markdown(
        f"**YOLO region {detection['class_label']}** · "
        f"YOLO confidence `{detection['confidence']:.3f}`"
    )
    st.image(crop, caption="Region sent to image retrieval", width=190)
    embedding = embedder.embed_image(crop)
    matches = vector_store.query(
        settings.collection_name,
        embedding,
        top_k=top_k,
    )
    if not matches:
        st.info("No LawDB image matches were returned.")
        return

    for rank, match in enumerate(matches, start=1):
        payload = dict(match.payload or {})
        score = getattr(match, "score", None)
        image_path = payload_image_path(payload.get("image_path"))
        match_col, detail_col = st.columns([1, 2], vertical_alignment="center")
        with match_col:
            if image_path:
                st.image(str(image_path), use_container_width=True)
            else:
                st.caption("Indexed image file is unavailable locally")
        with detail_col:
            st.markdown(f"**#{rank} · {payload.get('sign_name') or 'LawDB sign'}**")
            if score is not None:
                st.caption(f"Similarity: `{float(score):.4f}`")
            st.caption(
                f"{payload.get('law_id') or 'LawDB'} · "
                f"Article {payload.get('article_id') or 'unknown'}"
            )
            if payload.get("sign_description"):
                st.caption(payload["sign_description"])
        st.divider()


def main() -> None:
    st.title("YOLOv8m vs. LawDB Image Retrieval")
    st.caption(
        "The two branches run independently on the same image: the original "
        "LexiSign YoloE → CLIP → Qdrant retrieval pipeline, and the separately "
        "trained YOLOv8m detector. Retrieval does not use YOLOv8m detections."
    )

    with st.sidebar:
        st.header("Comparison settings")
        confidence = st.slider("Detection confidence threshold", 0.05, 0.95, 0.25, 0.05)
        top_k = st.slider("LawDB matches per detected sign", 1, 10, 5)
        st.caption(f"Device: {available_device()}")
        st.caption(f"Embedding model: {settings.embedding_name}")

    uploaded = st.file_uploader(
        "Choose a traffic image", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )
    if uploaded is None:
        st.info("Upload an image to compare the model and retrieval results.")
        return

    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open this image: {exc}")
        return

    st.image(image, caption="Uploaded image", use_container_width=True)
    device = available_device()
    try:
        yolov8m_detector, legacy_sign_extractor, embedder, vector_store = (
            load_comparison_resources(device)
        )
        yolov8m_detector.confidence_threshold = confidence
        legacy_sign_extractor.sign_extractor.score_threshold = confidence
        with st.spinner("Running both sign detectors and searching the LawDB index…"):
            yolov8m_detections = yolov8m_detector.detect(image)
            with tempfile.TemporaryDirectory(prefix="lexisign_comparison_") as temp_dir:
                query_image_path = Path(temp_dir) / "query.png"
                image.save(query_image_path)
                legacy_regions = legacy_sign_extractor.detect_and_crop(
                    str(query_image_path)
                )
            collection_exists = vector_store.client.collection_exists(
                collection_name=settings.collection_name
            )
    except Exception as exc:
        st.error(f"Could not initialize the comparison: {type(exc).__name__}: {exc}")
        st.caption(
            "Check that the YOLO checkpoint, embedding model, and Qdrant service "
            "are available. Start Qdrant with `make docker-up` if needed."
        )
        return

    if not collection_exists:
        st.error(
            f"Qdrant collection `{settings.collection_name}` does not exist. "
            "Ingest the LawDB sign images before running retrieval."
        )
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("YOLOv8m detections")
        st.metric("Signs detected", len(yolov8m_detections))
        st.image(
            annotate_detections(image, yolov8m_detections),
            caption="Bounding boxes and YOLO class labels",
            use_container_width=True,
        )
        if yolov8m_detections:
            st.dataframe(
                [
                    {
                        "YOLO class": item["class_label"],
                        "Confidence": round(item["confidence"], 4),
                    }
                    for item in yolov8m_detections
                ],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("YOLOv8m did not detect any signs at this confidence threshold.")

    with right:
        st.subheader("Original YoloE + image retrieval")
        st.caption("YoloE regions are embedded with CLIP and searched in Qdrant.")
        if not legacy_regions:
            st.info("The original YoloE detector did not find any sign regions.")
            return
        for index, (crop, metadata) in enumerate(legacy_regions, start=1):
            detection = {
                "class_label": metadata.get("class_label") or "traffic sign",
                "confidence": float(metadata.get("confidence") or 0.0),
            }
            if metadata.get("bbox"):
                detection["bbox"] = metadata["bbox"]
            with st.container(border=True):
                st.markdown(f"#### Sign region {index}")
                try:
                    render_retrieval_results(
                        crop,
                        detection,
                        embedder,
                        vector_store,
                        top_k,
                    )
                except Exception as exc:
                    st.warning(
                        f"Retrieval failed for this region: {type(exc).__name__}: {exc}"
                    )


if __name__ == "__main__":
    main()
