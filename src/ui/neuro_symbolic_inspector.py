from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

# Streamlit may put src/ui rather than the repository root on sys.path when
# this file is launched directly with `streamlit run src/ui/...`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.core.article_context import build_article_contexts, rule_article_candidates
from src.core.extract_signs import extract_signs
from src.core.freeform_qa import FreeformQAResources
from src.core.lawdb.article_lookup import LawDBArticleLookup
from src.deps.detection import SignExtractor
from src.core.sub_task_2 import (
    MAX_ARTICLES_IN_PROMPT,
    _answer_context_for_vlm,
    build_freeform_prompt_parts,
    collect_retrieved_context,
)
from src.core.symbolic.fact_schema import TrafficFacts
from src.core.symbolic.intent_schema import (
    Intent,
    IntentParseResult,
    IntentSlot,
    IntentTarget,
    QueryIntent,
)
from src.core.symbolic.reasoner import reason_over_facts
from src.core.symbolic.sign_fact_templates import TEMPLATE_PATH
from src.settings import settings


st.set_page_config(page_title="Neuro-symbolic Pipeline Inspector", layout="wide")

INTENT_LABELS = {
    "CHECK_PERMISSION": "Kiểm tra hành vi có được phép không",
    "INFORMATION": "Tra cứu thông tin/ý nghĩa biển báo",
    "UNKNOWN": "Intent khác / chưa hỗ trợ",
}
ACTION_OPTIONS = [
    "parking", "stopping", "yielding", "turn_left", "turn_right", "u_turn",
    "going_straight", "entering", "exiting", "speed", "circulation", "crossing",
    "overtaking", "reversing", "lane_change", "using_horn", "following_distance",
    "passing_obstacle", "roundabout_circulation", "reduce_speed", "unknown",
]
ACTOR_OPTIONS = [
    "motorcycle", "moped", "motorcycle_or_moped", "automobile", "passenger_car",
    "bus", "truck", "tractor_unit", "other_motor_vehicle", "non_motor_vehicle",
    "pedestrian", "unknown",
]
INFO_FACT_OPTIONS = {
    "Ý nghĩa biển báo": ("legal_meaning", "sign_meaning"),
    "Đối tượng bị cấm/bắt buộc": ("prohibited_actor", "prohibited_subjects"),
    "Khung thời gian có hiệu lực": ("active_time", "time_condition"),
    "Tốc độ tối thiểu": ("minimum_speed", "speed"),
    "Tốc độ tối đa": ("maximum_speed", "speed"),
    "Giới hạn tốc độ (tối thiểu và tối đa)": ("speed_limit", "speed"),
    "Ngày chẵn/lẻ": ("calendar_parity", "time_condition"),
    "Khoảng cách tối thiểu": ("minimum_distance", "distance_scope"),
    "Chiều cao xe tối đa": ("maximum_vehicle_height", "conditions"),
    "Khối lượng xe tối đa": ("maximum_vehicle_mass", "conditions"),
    "Khối lượng chuyên chở tối đa": ("maximum_payload_mass", "conditions"),
    "Số chỗ ngồi tối đa": ("maximum_seat_count", "conditions"),
    "Thông tin khác": ("", "other"),
}
ANSWER_JSON_CONTRACT = {
    "answer": "Câu trả lời tiếng Việt",
    "citations": [{"law_id": "", "article_id": "", "title": "", "evidence": ""}],
    "visual_evidence": [{"sign_name": "", "description": "", "image_name": ""}],
    "reasoning": "Giải thích ngắn dựa trên facts/rules",
    "confidence": "high | medium | low",
}


def template_digest() -> str:
    return hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest()


@st.cache_resource(show_spinner="Đang tải detector và tài nguyên pipeline trên CPU...")
def load_resources():
    # Keep LLM construction out of this app entirely; only local resources are used.
    return FreeformQAResources(
        llm_client=None,  # type: ignore[arg-type]
        embedder=None,
        sign_extractor=SignExtractor(
            detector=settings.image_query_detection_model,
            device="cpu",
            model_path=settings.sign_model_path,
            confidence_threshold=settings.sign_detection_confidence_threshold,
        ),
        vector_store=None,
        article_lookup=LawDBArticleLookup(),
        device="cpu",
    )


def serialize_final_parts(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep prompt text intact while replacing base64 payloads with readable markers."""
    safe = []
    for part in parts:
        copied = dict(part)
        if copied.get("type") == "image_url":
            copied["image_url"] = {"url": "<ẢNH CROP ĐÍNH KÈM; BASE64 ĐÃ ẨN>"}
        safe.append(copied)
    return safe


def run_fact_extraction(upload: Any) -> dict[str, Any]:
    """Run only image detection, OCR, template hydration, and fact normalization."""
    settings.traffic_sign_ocr_device = "cpu"
    run_id = uuid4().hex
    root = Path(tempfile.gettempdir()) / "lexisign_neuro_symbolic_inspector" / run_id
    image_dir, sign_dir = root / "images", root / "signs"
    image_dir.mkdir(parents=True, exist_ok=True)
    sign_dir.mkdir(parents=True, exist_ok=True)
    source_path = image_dir / "uploaded.jpg"
    with Image.open(upload) as image:
        image.convert("RGB").save(source_path, format="JPEG")

    resources = load_resources()
    sample: dict[str, Any] = {
        "id": run_id,
        "image_id": "uploaded",
        "question": "",
    }
    stages: list[tuple[str, Any]] = [
        ("Ảnh đầu vào", {"filename": upload.name, "size_bytes": upload.size})
    ]
    start = time.perf_counter()
    sample = extract_signs(
        sign_extractor=resources.sign_extractor,
        sample=sample,
        image_dir=image_dir,
        sign_dir=sign_dir,
        include_full_image_ocr=bool(settings.enable_symbolic_reasoning),
    )
    for sign in sample.get("detected_signs", []):
        sign["is_chosen"] = True
    stages.append(("Phát hiện biển báo", {
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "detected_signs": sample.get("detected_signs", []),
    }))

    start = time.perf_counter()
    article_candidates, visual_evidence, sign_prompt_parts, traffic_facts = collect_retrieved_context(
        sample=sample,
        sign_dir=sign_dir,
        embedding_model=resources.embedder,
        vector_store=resources.vector_store,
        collection_name=settings.collection_name,
        article_lookup=resources.article_lookup,
        llm_client=None,
        limit=settings.sign_recognition_candidate_limit,
        temperature=0.0,
        max_tokens=0,
        allow_sign_retrieval=False,
    )
    stages.append(("OCR toàn ảnh và liên kết chữ với biển gần đó", {
        "status": sample.get("full_image_ocr_status", "unavailable"),
        "all_text_regions": sample.get("full_image_ocr", []),
        "associated_text_by_sign": [
            {
                "image_name": sign.get("image_name"),
                "sign_code": sign.get("sign_code"),
                "ocr_status": sign.get("ocr_status", "not_run"),
                "ocr_texts": sign.get("ocr_texts", []),
            }
            for sign in sample.get("detected_signs", [])
        ],
    }))
    stages.append(("Chuẩn hóa biển thành quan sát/facts", {
        "elapsed_seconds": round(time.perf_counter() - start, 3),
        "traffic_facts": traffic_facts.to_dict(),
        "visual_evidence": visual_evidence,
        "article_candidates_from_signs": article_candidates,
    }))

    return {
        "root": str(root),
        "image_dir": str(image_dir),
        "sign_dir": str(sign_dir),
        "source_path": str(source_path),
        "sample": sample,
        "traffic_facts": traffic_facts,
        "traffic_facts_json": traffic_facts.to_dict(),
        "article_candidates": article_candidates,
        "visual_evidence": visual_evidence,
        "sign_prompt_parts": sign_prompt_parts,
        "stages": stages,
        "template_digest": template_digest(),
    }


def make_intent(
    *,
    kind: str,
    target_kind: str,
    target_id: str | None,
    question: str,
    action: str = "unknown",
    actor: str = "unknown",
    information_type: str | None = None,
    requested_fact: str | None = None,
    vehicle_filter: str = "unknown",
) -> Intent:
    target = IntentTarget(
        kind=target_kind,
        observation_ids=[target_id] if target_kind == "sign_observation" and target_id else [],
    )
    filters = {"vehicle_type": vehicle_filter} if vehicle_filter != "unknown" else {}
    return Intent(
        intent_id="intent-1",
        kind=kind,
        target=target,
        action=IntentSlot(value=action, status="provided" if action != "unknown" else "not_stated"),
        actor=IntentSlot(value=actor, status="provided" if actor != "unknown" else "not_stated"),
        information_type=information_type,
        requested_fact=requested_fact or None,
        filters=filters,
    )


def manual_symbolic_context(facts: TrafficFacts, intent: Intent, question: str) -> dict[str, Any]:
    query = QueryIntent(
        schema_version="3.0",
        query_id=f"manual-{uuid4().hex[:10]}",
        raw_query=question,
        intents=[intent],
        parser={"method": "manual_ui_selection", "llm_used": False},
    )
    trace = reason_over_facts(facts, IntentParseResult("3.0", query))
    decision = trace.decision
    return {
        "facts": facts.to_dict(),
        "query_intent": query.to_dict(),
        "query_context_facts": [],
        "intent_evaluations": decision.intent_evaluations if decision else [],
        "rule_evaluations": decision.rule_evaluations if decision else [],
        "sign_meaning_evaluations": decision.sign_meaning_evaluations if decision else [],
        "rules_fired": decision.rules_fired if decision else [],
        "symbolic_decision": decision.to_dict() if decision else {},
        "reasoning_trace": trace.to_dict(),
    }


def build_intent_preview(extraction: dict[str, Any], intent: Intent, question: str) -> dict[str, Any]:
    symbolic_context = manual_symbolic_context(extraction["traffic_facts"], intent, question)
    answer_context = _answer_context_for_vlm(symbolic_context)
    resources = load_resources()
    article_contexts = build_article_contexts(
        [*extraction["article_candidates"], *rule_article_candidates(symbolic_context)],
        article_lookup=resources.article_lookup,
        max_articles=MAX_ARTICLES_IN_PROMPT,
    )
    sample = dict(extraction["sample"])
    sample["question"] = question
    prompt_parts = build_freeform_prompt_parts(
        sample=sample,
        image_dir=Path(extraction["image_dir"]),
        article_contexts=article_contexts,
        visual_evidence=extraction["visual_evidence"],
        sign_prompt_parts=extraction["sign_prompt_parts"],
        symbolic_context=symbolic_context,
        include_full_image_ocr=bool(settings.enable_symbolic_reasoning),
    )
    return {
        "intent_json": symbolic_context["query_intent"],
        "answer_context_json": answer_context,
        "symbolic_context": symbolic_context,
        "article_contexts": article_contexts,
        "prompt_parts": serialize_final_parts(prompt_parts),
    }


def show_facts(extraction: dict[str, Any]) -> None:
    normalized_signs = extraction["traffic_facts_json"].get("signs", [])
    st.subheader("Facts đã trích xuất")
    if not normalized_signs:
        st.info("Pipeline chưa tạo được quan sát biển báo nào.")
    for sign in normalized_signs:
        code = sign.get("code") or sign.get("sign_code") or "Chưa xác định mã"
        with st.expander(f"{code} · {sign.get('role', 'main')}", expanded=True):
            st.caption(sign.get("name") or sign.get("meaning") or "")
            rows = []
            for fact in sign.get("facts", []):
                value = fact.get("value")
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, ensure_ascii=False)
                if fact.get("unit"):
                    value = f"{value} {fact['unit']}"
                rows.append({
                    "Predicate": fact.get("predicate", ""),
                    "Giá trị": value,
                    "Scope": json.dumps(fact.get("scope", {}), ensure_ascii=False) if fact.get("scope") else "—",
                    "Điều kiện": json.dumps(fact.get("conditions", {}), ensure_ascii=False) if fact.get("conditions") else "—",
                    "Trạng thái": fact.get("status", ""),
                })
            if rows:
                st.dataframe(rows, hide_index=True, use_container_width=True)
            else:
                st.warning("Không có fact cho biển này; kiểm tra mã nhận diện và template tương ứng.")


st.title("Neuro-symbolic Pipeline Inspector")
st.caption("Bước 1: trích xuất facts từ ảnh. Bước 2: chọn intent và xem JSON trước khi gửi đến VLM.")

upload = st.file_uploader("Ảnh giao thông", type=["jpg", "jpeg", "png"])
upload_digest = hashlib.sha256(upload.getvalue()).hexdigest() if upload else None
if upload_digest != st.session_state.get("neuro_symbolic_inspector_upload_digest"):
    st.session_state.pop("neuro_symbolic_inspector_extraction", None)
    st.session_state.pop("neuro_symbolic_inspector_preview", None)
    st.session_state["neuro_symbolic_inspector_upload_digest"] = upload_digest

if upload:
    st.image(upload, caption=upload.name, width=500)

if st.button("1 · Trích xuất facts", type="primary", disabled=upload is None):
    try:
        with st.spinner("Đang nhận diện biển, chạy OCR và điền fact template trên CPU..."):
            extraction = run_fact_extraction(upload)
        st.session_state["neuro_symbolic_inspector_extraction"] = extraction
        st.session_state.pop("neuro_symbolic_inspector_preview", None)
    except Exception as exc:
        st.exception(exc)

extraction = st.session_state.get("neuro_symbolic_inspector_extraction")
if extraction and extraction.get("template_digest") != template_digest():
    st.session_state.pop("neuro_symbolic_inspector_extraction", None)
    st.session_state.pop("neuro_symbolic_inspector_preview", None)
    extraction = None

if extraction:
    st.success("Đã trích xuất facts. Chọn intent bên dưới để dựng JSON xem trước.")
    show_facts(extraction)
    with st.expander("Các bước nhận diện và OCR", expanded=False):
        for index, (name, payload) in enumerate(extraction["stages"], start=1):
            with st.expander(f"Bước {index}: {name}", expanded=False):
                st.json(payload, expanded=False)
                if name == "Phát hiện biển báo":
                    detections = payload.get("detected_signs", [])
                    if detections:
                        cols = st.columns(min(len(detections), 4))
                        for idx, sign in enumerate(detections):
                            crop_path = Path(extraction["root"]) / "signs" / str(sign.get("image_name", ""))
                            if crop_path.is_file():
                                cols[idx % len(cols)].image(str(crop_path), caption=f"Crop {idx + 1}")

    st.divider()
    st.subheader("2 · Chọn intent và tạo JSON xem trước")
    intent_kind = st.selectbox(
        "Intent",
        list(INTENT_LABELS),
        format_func=lambda item: INTENT_LABELS[item],
        key="inspector_intent_kind",
    )
    sign_options = [("Toàn cảnh", "scene", None)]
    for sign in extraction["traffic_facts_json"].get("signs", []):
        obs_id = sign.get("id")
        if obs_id:
            label = f"{sign.get('code') or 'Chưa rõ mã'} · {sign.get('name') or obs_id} · {obs_id}"
            sign_options.append((label, "sign_observation", str(obs_id)))
    with st.form("intent_preview_form"):
        question = st.text_input("Câu hỏi mẫu", placeholder="Ví dụ: Cấm quẹo phải vào lúc nào?")
        target_label = st.selectbox("Đối tượng câu hỏi", [item[0] for item in sign_options])
        action = "unknown"
        actor = "unknown"
        information_type = None
        requested_fact = None
        vehicle_filter = "unknown"
        if intent_kind == "CHECK_PERMISSION":
            action = st.selectbox("Hành vi", ACTION_OPTIONS, format_func=lambda value: value.replace("_", " "))
            actor = st.selectbox("Loại phương tiện/người", ACTOR_OPTIONS, format_func=lambda value: value.replace("_", " "))
        elif intent_kind == "INFORMATION":
            info_label = st.selectbox(
                "Chọn thông tin cần hỏi",
                list(INFO_FACT_OPTIONS),
                help="Chọn loại fact cần lấy từ biển, ví dụ khung giờ, tốc độ hoặc giới hạn khối lượng.",
            )
            requested_fact, information_type = INFO_FACT_OPTIONS[info_label]
            vehicle_filter = st.selectbox(
                "Lọc theo loại xe (nếu cần)",
                ["unknown", *ACTOR_OPTIONS[:-1]],
                format_func=lambda value: "Không lọc" if value == "unknown" else value.replace("_", " "),
            )
        submitted = st.form_submit_button("Tạo JSON intent và answer/context", type="primary")

    if submitted:
        target = next(item for item in sign_options if item[0] == target_label)
        selected_intent = make_intent(
            kind=intent_kind,
            target_kind=target[1],
            target_id=target[2],
            question=question,
            action=action,
            actor=actor,
            information_type=information_type,
            requested_fact=requested_fact,
            vehicle_filter=vehicle_filter,
        )
        try:
            with st.spinner("Đang chạy symbolic reasoning và dựng payload xem trước..."):
                preview = build_intent_preview(extraction, selected_intent, question)
            st.session_state["neuro_symbolic_inspector_preview"] = preview
        except Exception as exc:
            st.exception(exc)

preview = st.session_state.get("neuro_symbolic_inspector_preview")
if extraction and preview:
    st.success("Đã tạo xong JSON. Chưa có dữ liệu nào được gửi đến VLM.")
    json_col, answer_col = st.columns(2)
    with json_col:
        st.markdown("#### Intent JSON")
        st.json(preview["intent_json"], expanded=True)
    with answer_col:
        st.markdown("#### Answer context JSON cho VLM")
        st.caption("Đây là facts/rule liên quan đã lọc theo intent, không phải câu trả lời ngôn ngữ tự nhiên đã sinh.")
        st.json(preview["answer_context_json"], expanded=True)
    with st.expander("Schema JSON câu trả lời VLM cần trả về", expanded=True):
        st.json(ANSWER_JSON_CONTRACT, expanded=True)
    with st.expander("Symbolic reasoning đầy đủ để debug", expanded=False):
        st.json(preview["symbolic_context"], expanded=False)
    with st.expander("Payload prompt cuối cùng (đã ẩn dữ liệu ảnh)", expanded=False):
        st.info("Bản xem trước để kiểm tra; app chưa gửi payload này đến bất kỳ VLM/LLM nào.")
        st.json(preview["prompt_parts"], expanded=True)
