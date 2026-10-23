from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow `streamlit run src/ui/sign_template_review.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.core.symbolic.sign_catalog import load_default_sign_catalog


TEMPLATE_PATH = PROJECT_ROOT / "src/core/symbolic/catalogs/vtsdb100_fact_templates.json"
FACT_CORE_FIELDS = {
    "predicate", "value", "unit", "status", "confidence", "fill_from", "scope", "conditions",
}
FACT_STATUSES = ["derived", "observed", "not_observed", "ambiguous", "unknown"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]


def read_template_file() -> dict[str, Any]:
    with TEMPLATE_PATH.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("classes"), dict):
        raise ValueError("JSON template phải có trường `classes` dạng object.")
    return payload


@st.cache_data(show_spinner=False)
def catalog_entries() -> dict[str, dict[str, Any]]:
    catalog = load_default_sign_catalog()
    entries: dict[str, dict[str, Any]] = {}
    for item in catalog.items:
        entries[item["official_code"]] = {
            "official_name": item.get("official_name"),
            "legal_meaning": item.get("legal_meaning"),
            "status": item.get("review_status", "unverified"),
            "legal_basis": item.get("legal_basis", []),
        }
    return entries


def json_text(value: Any, default: Any) -> str:
    return json.dumps(default if value is None else value, ensure_ascii=False)


def fact_rows(facts: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for fact in facts:
        extra = {key: value for key, value in fact.items() if key not in FACT_CORE_FIELDS}
        rows.append({
            "predicate": str(fact.get("predicate") or ""),
            "value_json": json_text(fact.get("value"), None),
            "unit": str(fact.get("unit") or ""),
            "status": str(fact.get("status") or "derived"),
            "confidence": str(fact.get("confidence") or "low"),
            "fill_from": str(fact.get("fill_from") or ""),
            "scope_json": json_text(fact.get("scope"), {}),
            "conditions_json": json_text(fact.get("conditions"), {}),
            "other_fields_json": json.dumps(extra, ensure_ascii=False),
        })
    return rows


def parse_json_cell(value: Any, default: Any, field_name: str) -> Any:
    if value is None or str(value).strip() == "":
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name}: JSON không hợp lệ — {exc.msg} (vị trí {exc.pos}).") from exc


def rows_to_facts(rows: Any) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    records = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
    for row_number, row in enumerate(records, start=1):
        if not any(str(value or "").strip() for value in row.values()):
            continue
        predicate = str(row.get("predicate") or "").strip()
        if not predicate:
            raise ValueError(f"Fact dòng {row_number}: cần có predicate.")
        value = parse_json_cell(row.get("value_json"), None, f"Fact dòng {row_number}, value")
        scope = parse_json_cell(row.get("scope_json"), {}, f"Fact dòng {row_number}, scope")
        conditions = parse_json_cell(row.get("conditions_json"), {}, f"Fact dòng {row_number}, conditions")
        extra = parse_json_cell(row.get("other_fields_json"), {}, f"Fact dòng {row_number}, other fields")
        if not isinstance(scope, dict):
            raise ValueError(f"Fact dòng {row_number}: scope phải là JSON object.")
        if not isinstance(conditions, dict):
            raise ValueError(f"Fact dòng {row_number}: conditions phải là JSON object.")
        if not isinstance(extra, dict):
            raise ValueError(f"Fact dòng {row_number}: other fields phải là JSON object.")
        fact = {
            "predicate": predicate,
            "value": value,
            "status": str(row.get("status") or "derived"),
            "confidence": str(row.get("confidence") or "low"),
        }
        unit = str(row.get("unit") or "").strip()
        fill_from = str(row.get("fill_from") or "").strip()
        if unit:
            fact["unit"] = unit
        if fill_from:
            fact["fill_from"] = fill_from
        if scope:
            fact["scope"] = scope
        if conditions:
            fact["conditions"] = conditions
        fact.update(extra)
        facts.append(fact)
    return facts


def atomic_save(payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=TEMPLATE_PATH.parent,
            prefix=f".{TEMPLATE_PATH.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, TEMPLATE_PATH)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()


def move_selected_class(options: list[str], step: int) -> None:
    current = st.session_state.get("sign_template_selected_class")
    if current not in options:
        current = options[0]
    current_index = options.index(current)
    next_index = current_index + step
    if 0 <= next_index < len(options):
        st.session_state.sign_template_selected_class = options[next_index]


st.set_page_config(page_title="Rà soát fact template biển báo", page_icon="🚦", layout="wide")
st.title("Rà soát 100 class biển báo")
st.caption("Đối chiếu mã biển và ý nghĩa trong catalog, chỉnh facts template, rồi lưu trực tiếp vào file JSON.")

if "sign_template_review_data" not in st.session_state:
    try:
        st.session_state.sign_template_review_data = read_template_file()
    except Exception as exc:
        st.error(f"Không đọc được file template: {exc}")
        st.stop()

payload = st.session_state.sign_template_review_data
classes: dict[str, dict[str, Any]] = payload["classes"]
catalog = catalog_entries()

identified_count = sum(item.get("identity_status") == "identified" for item in classes.values())
unresolved_count = len(classes) - identified_count
left, middle, right = st.columns(3)
left.metric("Tổng class", len(classes))
middle.metric("Có mã biển cụ thể", identified_count)
right.metric("Chưa phân giải", unresolved_count)

search = st.text_input("Tìm class hoặc mã biển", placeholder="Ví dụ: speed_limit_40, P.106b")
labels = sorted(classes)
filtered_labels = [
    label for label in labels
    if not search
    or search.casefold() in label.casefold()
    or search.casefold() in str(classes[label].get("sign_code") or "").casefold()
]

overview_rows = []
for label in filtered_labels:
    item = classes[label]
    catalog_item = catalog.get(str(item.get("sign_code") or ""), {})
    facts = item.get("facts", [])
    overview_rows.append({
        "Class": label,
        "Mã biển": item.get("sign_code") or "Chưa xác định",
        "Tên catalog": catalog_item.get("official_name") or "—",
        "Số facts": len(facts),
        "Predicates": ", ".join(str(fact.get("predicate", "")) for fact in facts),
        "Trạng thái ID": item.get("identity_status", "—"),
    })

with st.expander(f"Danh sách class ({len(overview_rows)})", expanded=True):
    st.dataframe(overview_rows, width="stretch", hide_index=True)

if not filtered_labels:
    st.info("Không tìm thấy class phù hợp.")
    st.stop()

selection_key = "sign_template_selected_class"
if st.session_state.get(selection_key) not in filtered_labels:
    st.session_state[selection_key] = filtered_labels[0]
selected_index = filtered_labels.index(st.session_state[selection_key])
previous_col, selection_col, next_col = st.columns([1, 6, 1])
with previous_col:
    st.button(
        "← Trước",
        disabled=selected_index == 0,
        width="stretch",
        shortcut="Left",
        help="Phím ←",
        on_click=move_selected_class,
        args=(filtered_labels, -1),
    )
with selection_col:
    selected = st.selectbox(
        f"Class cần xem hoặc sửa · {selected_index + 1}/{len(filtered_labels)}",
        filtered_labels,
        format_func=lambda label: f"{label}  ·  {classes[label].get('sign_code') or 'chưa có mã'}",
        key=selection_key,
    )
with next_col:
    st.button(
        "Tiếp →",
        disabled=selected_index == len(filtered_labels) - 1,
        width="stretch",
        shortcut="Right",
        help="Phím →",
        on_click=move_selected_class,
        args=(filtered_labels, 1),
    )
entry = classes[selected]
catalog_item = catalog.get(str(entry.get("sign_code") or ""), {})

st.subheader(f"{selected}  ·  {entry.get('sign_code') or 'Chưa xác định mã biển'}")
if entry.get("identity_status") != "identified":
    st.warning(
        "Class này chưa ánh xạ tới một mã biển duy nhất. Bạn có thể rà soát facts, "
        "nhưng hãy kiểm tra identity trước khi thêm ý nghĩa pháp lý cụ thể."
    )

catalog_col, template_col = st.columns(2)
with catalog_col:
    st.markdown("#### Ý nghĩa trong catalog")
    if catalog_item:
        st.markdown(f"**{catalog_item.get('official_name') or 'Chưa có tên chính thức'}**")
        st.write(catalog_item.get("legal_meaning") or "Catalog chưa có phần diễn giải.")
        st.caption(f"Trạng thái catalog: {catalog_item.get('status', 'unverified')}")
        basis = catalog_item.get("legal_basis") or []
        if basis:
            with st.expander("Căn cứ catalog"):
                for source in basis:
                    st.write(
                        f"{source.get('document', 'Nguồn')} · "
                        f"{source.get('section_or_figure', '')} · "
                        f"{source.get('verification_status', 'unverified')}"
                    )
    else:
        st.info("Không tìm thấy mã này trong catalog chính thức.")

with template_col:
    st.markdown("#### Thông tin nhận diện template")
    st.write(f"**Trạng thái identity:** {entry.get('identity_status', '—')}")
    if entry.get("sign_family"):
        st.write(f"**Nhóm biển:** {entry['sign_family']}")
    if entry.get("candidate_codes"):
        st.write("**Mã ứng viên:** " + ", ".join(entry["candidate_codes"]))
    st.caption("Phần catalog chỉ đọc. Muốn sửa ý nghĩa dùng trong suy luận, chỉnh predicate `sign_meaning` ở bảng facts bên dưới.")

st.markdown("#### Fact template")
st.caption(
    "Các cột value, scope, conditions và other fields dùng JSON. Scope xe dùng "
    "`vehicle_types`; có thể thêm `excluded_vehicle_types` cho ngoại lệ. Ví dụ: "
    "`{\"vehicle_types\":[\"motor_vehicle\"],\"excluded_vehicle_types\":[\"motorcycle\",\"moped\"]}`. "
    "`vehicle` chỉ mọi loại xe; `motor_vehicle` chỉ xe cơ giới. "
    "Ví dụ value: số `40`, chuỗi `\"xe cơ giới\"`, "
    "mảng `[\"truck\"]`, object `{}` hoặc giá trị chưa quan sát `null`. Có thể thêm/xóa dòng trong bảng."
)

with st.form(f"edit_facts_{selected}"):
    edited_rows = st.data_editor(
        fact_rows(entry.get("facts", [])),
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        key=f"facts_editor_{selected}",
        column_config={
            "predicate": st.column_config.TextColumn("predicate", required=True),
            "value_json": st.column_config.TextColumn("value (JSON)", width="large"),
            "unit": st.column_config.TextColumn("unit", width="small"),
            "status": st.column_config.SelectboxColumn("status", options=FACT_STATUSES),
            "confidence": st.column_config.SelectboxColumn("confidence", options=CONFIDENCE_LEVELS),
            "fill_from": st.column_config.TextColumn("fill_from"),
            "scope_json": st.column_config.TextColumn("scope (JSON)", width="large"),
            "conditions_json": st.column_config.TextColumn("conditions (JSON)", width="large"),
            "other_fields_json": st.column_config.TextColumn("other fields (JSON)", width="large"),
        },
    )
    save_clicked = st.form_submit_button("Lưu facts của class này", type="primary")

if save_clicked:
    try:
        new_facts = rows_to_facts(edited_rows)
        updated_payload = json.loads(json.dumps(payload, ensure_ascii=False))
        updated_payload["classes"][selected]["facts"] = new_facts
        atomic_save(updated_payload)
        st.session_state.sign_template_review_data = updated_payload
        st.success(f"Đã lưu {len(new_facts)} facts cho {selected} vào {TEMPLATE_PATH.name}.")
        st.rerun()
    except (ValueError, OSError) as exc:
        st.error(f"Chưa lưu được thay đổi: {exc}")

st.download_button(
    "Tải xuống toàn bộ template JSON",
    data=json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    file_name="vtsdb100_fact_templates.json",
    mime="application/json",
)

with st.sidebar:
    st.header("Quản lý template")
    st.caption(f"File đang xem: `{TEMPLATE_PATH}`")
    if st.button("Nạp lại từ file", width="stretch"):
        st.session_state.sign_template_review_data = read_template_file()
        st.rerun()
