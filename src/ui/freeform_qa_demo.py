from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.core.freeform_qa import answer_image_question, create_freeform_qa_resources


ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
DEMO_CONTENT_MAX_WIDTH = 920
DEMO_IMAGE_WIDTH = 520


st.set_page_config(page_title="Traffic Sign QA", layout="wide")


@st.cache_resource(show_spinner="Loading detector and embedding model...")
def load_resources():
    return create_freeform_qa_resources()


def inject_demo_styles() -> None:
    st.markdown(
        f"""
        <style>
        [data-testid="stMainBlockContainer"],
        section[data-testid="stMain"] .block-container,
        [data-testid="stAppViewContainer"] .block-container,
        .main .block-container {{
            width: min(100%, {DEMO_CONTENT_MAX_WIDTH}px) !important;
            max-width: {DEMO_CONTENT_MAX_WIDTH}px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }}

        [data-testid="stMarkdownContainer"] p {{
            line-height: 1.6;
        }}

        div[data-testid="stExpander"] {{
            border-radius: 0.75rem;
        }}

        div[data-testid="stForm"] {{
            border: 1px solid rgba(250, 250, 250, 0.14);
            border-radius: 1.25rem;
            padding: 0.45rem 0.6rem 0.5rem 0.75rem;
            background: rgba(255, 255, 255, 0.035);
            box-shadow: none !important;
        }}

        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
            align-items: center;
            gap: 0.5rem;
        }}

        div[data-testid="stForm"] [data-testid="stTextInput"] {{
            margin-bottom: 0;
        }}

        div[data-testid="stForm"] [data-testid="stTextInput"] > div,
        div[data-testid="stForm"] [data-testid="stTextInput"] > div:focus-within {{
            border: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
        }}

        div[data-testid="stForm"] [data-testid="stTextInput"] input {{
            border: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            padding-left: 0.25rem !important;
            outline: none !important;
        }}

        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {{
            border-radius: 999px;
            min-height: 2.4rem;
        }}

        div[data-testid="stFileUploader"] section {{
            border-radius: 1rem;
        }}

        @media (max-width: 720px) {{
            [data-testid="stMainBlockContainer"],
            section[data-testid="stMain"] .block-container,
            [data-testid="stAppViewContainer"] .block-container,
            .main .block-container {{
                width: 100% !important;
                max-width: 100% !important;
                padding-left: 1rem;
                padding-right: 1rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def persist_uploaded_image(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    if suffix.lower().lstrip(".") not in ALLOWED_IMAGE_TYPES:
        suffix = ".jpg"
    temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_image.write(uploaded_file.getbuffer())
    temp_image.close()
    return Path(temp_image.name)


def render_selected_image(uploaded_file: Any | None) -> Path | None:
    if uploaded_file is None:
        return None
    image_path = persist_uploaded_image(uploaded_file)
    st.image(
        uploaded_file,
        caption=getattr(uploaded_file, "name", "Uploaded traffic image"),
        width=DEMO_IMAGE_WIDTH,
    )
    return image_path


def render_answer(result: dict[str, Any]) -> None:
    st.subheader("Answer")
    answer = result.get("freeform_answer")
    if answer:
        st.success(answer)
    else:
        st.warning("No valid free-form answer was produced.")

    confidence = result.get("confidence", "low")
    performance = result.get("performance") or {}
    caption_parts = [f"Confidence: `{confidence}`"]
    if performance.get("total") is not None:
        caption_parts.append(f"Processing time: `{performance.get('total')}s`")
    st.caption(" | ".join(caption_parts))

    reasoning = result.get("reasoning")
    if reasoning:
        st.markdown("**Reasoning**")
        st.write(reasoning)


def render_detected_signs(result: dict[str, Any]) -> None:
    detected_signs = result.get("detected_signs", [])
    st.subheader("Detected Signs")
    if not detected_signs:
        st.info("No signs were detected.")
        return

    columns = st.columns(min(3, len(detected_signs)))
    for index, sign in enumerate(detected_signs):
        with columns[index % len(columns)]:
            st.image(sign.get("image_path"), use_container_width=True)
            status = "Relevant" if sign.get("is_chosen") else "Not selected"
            st.caption(status)
            sign_name = sign.get("sign_name")
            if sign_name:
                st.write(sign_name)
            variant_decision = sign.get("variant_decision") or {}
            if variant_decision and variant_decision.get("method") != "top1":
                st.caption(
                    f"Variant: {variant_decision.get('method')} | "
                    f"{variant_decision.get('confidence', 'low')}"
                )


def render_citations(result: dict[str, Any]) -> None:
    citations = result.get("citations", [])
    st.subheader("Citations")
    if not citations:
        st.info("No citations were returned.")
        return

    for citation in citations:
        label = (
            f"{citation.get('law_id', '')} #"
            f"{citation.get('article_id', '')}"
        )
        title = citation.get("title") or "(untitled article)"
        with st.expander(f"{label} - {title}", expanded=True):
            evidence = citation.get("evidence")
            if evidence:
                st.write(evidence)


def render_visual_evidence(result: dict[str, Any]) -> None:
    visual_evidence = result.get("visual_evidence", [])
    st.subheader("Visual Evidence")
    if not visual_evidence:
        st.info("No visual evidence was returned.")
        return

    for item in visual_evidence:
        sign_name = item.get("sign_name") or "Detected sign"
        image_name = item.get("image_name") or ""
        with st.expander(f"{image_name} - {sign_name}", expanded=False):
            st.write(item.get("description", ""))


def render_debug(result: dict[str, Any]) -> None:
    st.subheader("Raw Response")
    st.code(result.get("raw_answer_response", ""), language="json")
    st.subheader("Debug Info")
    debug_payload = {
        "performance": result.get("performance"),
        "relevant_articles": result.get("relevant_articles"),
        "article_contexts": result.get("article_contexts"),
        "query_intent": result.get("query_intent"),
        "query_context_facts": result.get("query_context_facts"),
        "symbolic_decision": result.get("symbolic_decision"),
        "rules_fired": result.get("rules_fired"),
        "traffic_facts": result.get("traffic_facts"),
        "intent_evaluations": result.get("intent_evaluations"),
        "rule_evaluations": result.get("rule_evaluations"),
        "reasoning_trace": result.get("reasoning_trace"),
        "detected_signs": result.get("detected_signs"),
        "answer_prompt_parts": result.get("answer_prompt_parts"),
    }
    st.json(json.loads(json.dumps(debug_payload, ensure_ascii=False)))


def main() -> None:
    inject_demo_styles()
    st.title("Traffic Sign QA")

    with st.sidebar:
        st.header("Settings")
        st.caption(
            "The demo detects signs, retrieves LawDB evidence from Qdrant, "
            "and generates a cited free-form answer."
        )
        debug = st.checkbox("Debug", value=False)
        use_vlm_filter = st.checkbox("Use VLM sign filter", value=True)
        with st.expander("Optional reference choices", expanded=False):
            choices = st.text_area(
                "Choices",
                height=140,
                placeholder="Paste A/B/C/D choices if the sample has them.",
                label_visibility="collapsed",
            )

    uploaded_file = st.file_uploader(
        "Upload a traffic image",
        type=ALLOWED_IMAGE_TYPES,
        accept_multiple_files=False,
    )
    image_path = render_selected_image(uploaded_file)

    with st.form("freeform_question_form", clear_on_submit=False):
        question_col, send_col = st.columns([8, 1], gap="small")
        with question_col:
            question = st.text_input(
                "Ask a free-form legal question",
                placeholder="Ví dụ: Biển báo trong ảnh có ý nghĩa gì?",
                label_visibility="collapsed",
            )
        with send_col:
            run_clicked = st.form_submit_button(
                "Send",
                type="primary",
                use_container_width=True,
            )

    if not run_clicked:
        return
    if image_path is None:
        st.warning("Please upload a traffic image.")
        return
    if not question.strip():
        st.warning("Please enter a free-form question.")
        return

    try:
        with st.spinner(
            "Detecting signs, retrieving legal evidence, and preparing answer..."
        ):
            resources = load_resources()
            result = answer_image_question(
                image_path=image_path,
                question=question.strip(),
                choices=choices.strip() or None,
                debug=debug,
                resources=resources,
                use_vlm_filter=use_vlm_filter,
            )
    except Exception as exc:
        st.error("Free-form demo run failed.")
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    st.divider()
    render_answer(result)
    render_detected_signs(result)
    render_citations(result)
    render_visual_evidence(result)
    if debug:
        render_debug(result)


if __name__ == "__main__":
    main()
