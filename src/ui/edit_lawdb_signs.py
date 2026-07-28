from __future__ import annotations

import json
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import streamlit as st

from src.settings import settings


st.set_page_config(page_title="Edit LawDB Signs", layout="wide")


def candidate_json_paths() -> list[Path]:
    paths = [
        Path(settings.lawdb_json_sign_parsing_path),
        Path("outputs/vlsp2025_law_new_preprocessed_with_vlm_parse_quality_run.json"),
        Path("data/raw/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json"),
        Path("data/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json"),
    ]
    seen: set[Path] = set()
    existing: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            existing.append(path)
    return existing


def default_sign_dir() -> Path:
    sign_dir = Path(settings.lawdb_extracted_sign_dir)
    if sign_dir.exists():
        return sign_dir
    fallback = Path("data/law_db/signs_imageprocessing")
    if fallback.exists():
        return fallback
    return sign_dir


@st.cache_data(show_spinner="Loading LawDB sign parse JSON...")
def load_lawdb(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text)
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_lawdb(data: list[dict[str, Any]], path: Path) -> None:
    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copyfile(path, backup_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        suffix=".tmp",
    ) as temp_file:
        json.dump(data, temp_file, ensure_ascii=False, indent=2)
        temp_path = Path(temp_file.name)
    temp_path.replace(path)
    load_lawdb.clear()


def flatten_signs(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for law_index, law in enumerate(data):
        law_id = law.get("id", "")
        for article_index, article in enumerate(law.get("articles", [])):
            detailed_signs = article.get("detailed_signs", [])
            for sign_index, sign in enumerate(detailed_signs):
                rows.append(
                    {
                        "law_index": law_index,
                        "article_index": article_index,
                        "sign_index": sign_index,
                        "law_id": law_id,
                        "article_id": article.get("id", ""),
                        "article_title": article.get("title", ""),
                        "article_text": article.get("text", ""),
                        "image": sign.get("image", ""),
                        "name": sign.get("name", ""),
                        "description": sign.get("description", ""),
                        "sign": sign,
                    }
                )
    return rows


def find_row_index(rows: list[dict[str, Any]], key: tuple[int, int, int]) -> int:
    for index, row in enumerate(rows):
        if (
            row["law_index"],
            row["article_index"],
            row["sign_index"],
        ) == key:
            return index
    return 0


def get_sign(data: list[dict[str, Any]], row: dict[str, Any]) -> dict[str, Any]:
    return data[row["law_index"]]["articles"][row["article_index"]]["detailed_signs"][
        row["sign_index"]
    ]


def sign_image_path(sign_dir: Path, image_name: str) -> Path:
    path = sign_dir / image_name
    if path.exists():
        return path
    raw_path = Path(settings.lawdb_image_dir) / image_name
    if raw_path.exists():
        return raw_path
    return path


def sidebar_controls(paths: list[Path]) -> tuple[Path, Path, str]:
    with st.sidebar:
        st.header("Data")
        selected_path = st.selectbox(
            "Parse JSON",
            options=[str(path) for path in paths] or [""],
            index=0,
        )
        custom_path = st.text_input(
            "Custom JSON path",
            value=selected_path,
            help="File will be updated directly when you save.",
        )
        sign_dir = Path(
            st.text_input(
                "Sign crop directory",
                value=str(default_sign_dir()),
            )
        )
        query = st.text_input(
            "Search",
            placeholder="P.131b, cấm đỗ, B.31, image002_crop0...",
        )
    return Path(custom_path), sign_dir, query.strip()


def filter_rows(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    if not query:
        return rows
    normalized = query.casefold()
    return [
        row
        for row in rows
        if normalized
        in " ".join(
            [
                str(row.get("law_id", "")),
                str(row.get("article_id", "")),
                str(row.get("article_title", "")),
                str(row.get("image", "")),
                str(row.get("name", "")),
                str(row.get("description", "")),
            ]
        ).casefold()
    ]


def render_editor(data: list[dict[str, Any]], rows: list[dict[str, Any]], path: Path, sign_dir: Path) -> None:
    if not rows:
        st.warning("No parsed signs found.")
        return

    selected_key = st.session_state.get("selected_sign_key")
    if selected_key is None:
        selected_key = (
            rows[0]["law_index"],
            rows[0]["article_index"],
            rows[0]["sign_index"],
        )
        st.session_state["selected_sign_key"] = selected_key

    current_index = min(find_row_index(rows, selected_key), len(rows) - 1)
    selected_index = st.sidebar.number_input(
        "Sign index",
        min_value=0,
        max_value=max(len(rows) - 1, 0),
        value=current_index,
        step=1,
    )
    row = rows[int(selected_index)]
    st.session_state["selected_sign_key"] = (
        row["law_index"],
        row["article_index"],
        row["sign_index"],
    )
    sign = get_sign(data, row)

    st.title("LawDB Sign Description Editor")
    st.caption(f"{len(rows)} parsed signs in current view")

    info_col, edit_col = st.columns([1, 1], gap="large")

    with info_col:
        st.subheader("Sign Crop")
        image_path = sign_image_path(sign_dir, sign.get("image", ""))
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        else:
            st.error(f"Image not found: {image_path}")

        st.markdown("**Source**")
        st.write(f"Law: `{row['law_id']}`")
        st.write(f"Article: `{row['article_id']}`")
        st.write(row["article_title"] or "(untitled)")
        with st.expander("Article text", expanded=False):
            st.write(row["article_text"])
        with st.expander("Raw sign JSON", expanded=False):
            st.json(sign)

    with edit_col:
        st.subheader("Parsed Description")
        with st.form("edit_sign_form"):
            new_name = st.text_input("Name", value=str(sign.get("name", "")))
            new_description = st.text_area(
                "Description",
                value=str(sign.get("description", "")),
                height=260,
            )
            status = st.selectbox(
                "Review status",
                ["reviewed", "pending", "needs_fix"],
                index=0 if sign.get("__review_status") == "reviewed" else 1,
            )
            note = st.text_area(
                "Review note",
                value=str(sign.get("__review_note", "")),
                height=90,
            )

            save_col, prev_col, next_col = st.columns([2, 1, 1])
            with save_col:
                save_clicked = st.form_submit_button("Save changes", type="primary")
            with prev_col:
                prev_clicked = st.form_submit_button("Prev")
            with next_col:
                next_clicked = st.form_submit_button("Next")

        if save_clicked:
            sign["name"] = new_name.strip()
            sign["description"] = new_description.strip()
            sign["__review_status"] = status
            sign["__review_note"] = note.strip()
            save_lawdb(data, path)
            st.success(f"Saved to {path}")

        if prev_clicked:
            st.session_state["selected_sign_key"] = (
                rows[max(0, int(selected_index) - 1)]["law_index"],
                rows[max(0, int(selected_index) - 1)]["article_index"],
                rows[max(0, int(selected_index) - 1)]["sign_index"],
            )
            st.rerun()
        if next_clicked:
            st.session_state["selected_sign_key"] = (
                rows[min(len(rows) - 1, int(selected_index) + 1)]["law_index"],
                rows[min(len(rows) - 1, int(selected_index) + 1)]["article_index"],
                rows[min(len(rows) - 1, int(selected_index) + 1)]["sign_index"],
            )
            st.rerun()


def main() -> None:
    paths = candidate_json_paths()
    path, sign_dir, query = sidebar_controls(paths)
    if not path.exists():
        st.error(f"JSON file not found: {path}")
        return

    data = load_lawdb(str(path))
    rows = filter_rows(flatten_signs(data), query)
    render_editor(data, rows, path, sign_dir)


if __name__ == "__main__":
    main()
