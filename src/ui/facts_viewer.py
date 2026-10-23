from __future__ import annotations

import json
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

# Allow `streamlit run src/ui/facts_viewer.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.core.sub_task_2 import collect_retrieved_context
from src.core.extract_signs import extract_signs
from src.core.lawdb.article_lookup import LawDBArticleLookup
from src.deps.detection import SignExtractor
from src.settings import settings
from src.core.symbolic.sign_fact_templates import TEMPLATE_PATH


def template_digest() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()


def pipeline_digest() -> str:
    """Invalidate displayed results when either templates or fact assembly changes."""
    digest = hashlib.sha256()
    source_paths = [
        TEMPLATE_PATH,
        PROJECT_ROOT / "src/core/sub_task_2.py",
        PROJECT_ROOT / "src/core/symbolic/fact_schema.py",
        PROJECT_ROOT / "src/core/symbolic/sign_fact_templates.py",
        PROJECT_ROOT / "src/core/symbolic/sign_normalizer.py",
        PROJECT_ROOT / "src/core/symbolic/traffic_rules.py",
    ]
    for path in source_paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


@st.cache_resource(show_spinner="Đang tải mô hình nhận diện biển báo...")
def load_resources() -> tuple[SignExtractor, LawDBArticleLookup]:
    extractor = SignExtractor(
        detector=settings.image_query_detection_model,
        device="cpu",
        model_path=settings.sign_model_path,
        confidence_threshold=settings.sign_detection_confidence_threshold,
    )
    return extractor, LawDBArticleLookup()


def analyze_image(upload: Any) -> dict[str, Any]:
    run_dir = Path(tempfile.gettempdir()) / "lexisign_facts_viewer" / uuid4().hex
    image_dir = run_dir / "images"
    sign_dir = run_dir / "signs"
    image_dir.mkdir(parents=True, exist_ok=True)
    sign_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "uploaded.jpg"
    with Image.open(upload) as image:
        image.convert("RGB").save(image_path, format="JPEG")

    # The app intentionally uses no LLM; OCR and sign detection run locally.
    settings.traffic_sign_ocr_device = "cpu"
    sign_extractor, article_lookup = load_resources()
    sample: dict[str, Any] = {
        "id": run_dir.name,
        "image_id": "uploaded",
        "question": "",
    }
    sample = extract_signs(
        sign_extractor=sign_extractor,
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
        include_full_image_ocr=bool(settings.enable_symbolic_reasoning),
    )
    for sign in sample.get("detected_signs", []):
        sign["is_chosen"] = True

    _, _, _, facts = collect_retrieved_context(
        sample=sample,
        sign_dir=sign_dir,
        embedding_model=None,
        vector_store=None,
        collection_name=settings.collection_name,
        article_lookup=article_lookup,
        llm_client=None,
        allow_sign_retrieval=False,
    )
    return {
        "facts": facts.to_dict(),
        "sample": sample,
        "run_dir": str(run_dir),
        "template_digest": template_digest(),
        "pipeline_digest": pipeline_digest(),
    }


def format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_fact_table(facts: list[dict[str, Any]]) -> None:
    if not facts:
        st.caption("Chưa trích xuất được fact cho biển này.")
        return
    rows = []
    for fact in facts:
        value = "Chưa nhận diện" if fact.get("value") is None else format_value(fact.get("value"))
        if fact.get("unit"):
            value = f"{value} {fact['unit']}"
        row = {
            "Predicate": fact.get("predicate", ""),
            "Giá trị": value,
            "Phạm vi": format_value(fact.get("scope", {})) if fact.get("scope") else "—",
            "Điều kiện": format_value(fact.get("conditions", {})) if fact.get("conditions") else "—",
            "Trạng thái": fact.get("status", ""),
        }
        rows.append(row)
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_results(result: dict[str, Any]) -> None:
    payload = result["facts"]
    observations = payload.get("signs", [])
    sample = result.get("sample", {})
    detections = sample.get("detected_signs", [])

    st.subheader("Facts trích xuất")
    st.caption(f"Phát hiện {len(observations)} biển báo. Kết quả bên dưới là facts đã chuẩn hóa từ nhận diện biển và OCR.")

    if not observations:
        st.info("Không phát hiện được biển báo trong ảnh.")
    main_sign_index = 0
    for index, observation in enumerate(observations, start=1):
        code = observation.get("code") or "Chưa xác định mã biển"
        is_text_panel = observation.get("type") == "ocr_text_panel"
        title = (
            f"Bảng chữ phụ {index} · OCR"
            if is_text_panel
            else f"Biển {index}: {code} · {observation.get('role', 'unknown')}"
        )
        with st.expander(title, expanded=True):
            columns = st.columns([1, 2])
            if not is_text_panel:
                detection = detections[main_sign_index] if main_sign_index < len(detections) else {}
                main_sign_index += 1
            else:
                detection = {}
            crop_path = Path(result["run_dir"]) / "signs" / str(detection.get("image_name", ""))
            if detection and crop_path.is_file():
                columns[0].image(str(crop_path), caption="Vùng biển được nhận diện", use_container_width=True)
            if is_text_panel:
                columns[1].markdown("**Loại:** Bảng chữ thời gian được nhận diện bằng OCR")
            else:
                columns[1].markdown(f"**Tên biển:** {observation.get('name') or detection.get('class_label') or 'Chưa có'}")
                if detection.get("class_label"):
                    columns[1].caption(
                        f"Class YOLO: {detection['class_label']} · Mã dùng tra template: {observation.get('code') or 'chưa xác định'}"
                    )
                meaning = next((
                    fact.get("value") for fact in observation.get("facts", [])
                    if fact.get("predicate") == "sign_meaning"
                ), None)
                if meaning:
                    columns[1].markdown(f"**Ý nghĩa:** {meaning}")
                if observation.get("identity_status") or observation.get("confidence"):
                    columns[1].markdown(
                        f"**Định danh:** {observation.get('identity_status', 'identified')} · "
                        f"**Độ tin cậy:** {observation.get('confidence', 'high')}"
                    )
            display_facts = [
                fact for fact in observation.get("facts", [])
                if fact.get("predicate") != "sign_meaning"
            ]
            render_fact_table(display_facts)
            if not display_facts and not is_text_panel:
                st.caption("Không có fact: identity chưa được xác nhận hoặc chưa tìm thấy template cho mã biển này.")
            ocr_items = observation.get("ocr_text", [])
            if ocr_items:
                st.caption("OCR: " + " · ".join(str(text) for text in ocr_items))

    scene_facts = payload.get("scene_facts", [])
    if scene_facts:
        st.subheader("Facts áp dụng cho toàn cảnh")
        render_fact_table(scene_facts)

    relations = payload.get("relations", [])
    if relations:
        st.subheader("Quan hệ giữa các biển")
        st.dataframe(
            [
                {"Quan hệ": relation.get("predicate"), "Từ biển": relation.get("from"), "Đến biển": relation.get("to")}
                for relation in relations
            ],
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("JSON facts gọn", expanded=False):
        st.json(payload, expanded=True)


def main() -> None:
    st.set_page_config(page_title="Traffic Sign Facts", page_icon="🚦", layout="wide")
    st.title("Traffic Sign Facts")
    st.caption("Tải ảnh biển báo lên để xem facts đã nhận diện và chuẩn hóa. Ứng dụng không gửi ảnh đến VLM/LLM.")

    upload = st.file_uploader("Chọn ảnh giao thông", type=["jpg", "jpeg", "png"])
    if upload is not None:
        st.image(upload, caption=upload.name, width=600)

    if st.button("Nhận diện và xem facts", type="primary", disabled=upload is None):
        try:
            with st.spinner("Đang nhận diện biển báo và đọc chữ trong ảnh..."):
                st.session_state["facts_viewer_result"] = analyze_image(upload)
            st.success("Đã trích xuất facts.")
        except Exception as exc:
            st.error(f"Không thể phân tích ảnh: {exc}")
            st.exception(exc)

    result = st.session_state.get("facts_viewer_result")
    if result and (
        result.get("template_digest") != template_digest()
        or result.get("pipeline_digest") != pipeline_digest()
    ):
        st.session_state.pop("facts_viewer_result", None)
        st.info("Pipeline hoặc fact template đã được cập nhật. Hãy chạy lại nhận diện để xem kết quả mới.")
        result = None
    if result:
        render_results(result)


if __name__ == "__main__":
    main()
