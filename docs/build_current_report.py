from __future__ import annotations

import json
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"
ASSET_DIR = OUT_DIR / "report_assets"
OUT_DOCX = OUT_DIR / "DSP391_Final_Report_Traffic_Sign_Legal_QA_Current.docx"
TEMPLATE_LOGO = Path("/tmp/traffic_legal_report_docx/template_media/image1.png")
METRICS_CURRENT = ROOT / "outputs/freeform_qa_metrics_reverted_previous_begin39_20_judge_20260728.json"
LAWDB = ROOT / "data/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json"


RED = RGBColor(192, 0, 0)
BLUE = RGBColor(31, 78, 121)
GRAY = RGBColor(51, 51, 51)
LIGHT_BLUE = "D9EAF7"
PALE_BLUE = "F3F8FD"
BORDER = "7F7F7F"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, color: str = BORDER, size: str = "8") -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:{}".format(edge)
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=120, bottom=90, end=120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_width(table, width_dxa: int = 9360) -> None:
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(width_dxa))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_pr.append(OxmlElement("w:tblLayout"))
    tbl_pr[-1].set(qn("w:type"), "fixed")


def format_table(table, header=True, widths=None) -> None:
    set_table_width(table)
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if widths and c_idx < len(widths):
                cell.width = Inches(widths[c_idx])
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)
                paragraph.paragraph_format.line_spacing = 1.05
                for run in paragraph.runs:
                    run.font.size = Pt(9.5)
            if header and r_idx == 0:
                set_cell_shading(cell, LIGHT_BLUE)
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.bold = True


def add_paragraph(doc, text: str, style: str = "Normal", bold: bool = False):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.bold = bold
    return p


def add_bullets(doc, items: list[str]) -> None:
    for item in items:
        p = doc.add_paragraph(style="List Paragraph")
        p.paragraph_format.left_indent = Inches(0.18)
        p.paragraph_format.first_line_indent = Inches(-0.12)
        p.add_run("• ")
        p.add_run(item)
        p.paragraph_format.space_after = Pt(3)


def add_caption(doc, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(9.5)


def add_heading(doc, text: str, level: int) -> None:
    doc.add_heading(text, level=level)


def add_named_table(doc, caption: str, headers: list[str], rows: list[list[str]], widths=None):
    add_caption(doc, caption)
    table = doc.add_table(rows=1, cols=len(headers))
    for i, header in enumerate(headers):
        table.rows[0].cells[i].text = header
    for row_data in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row_data):
            cells[i].text = value
    format_table(table, header=True, widths=widths)
    return table


def add_figure_placeholder(
    doc,
    caption: str,
    description: str,
    suggested_content: list[str],
    height_lines: int = 5,
) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_width(table)
    cell = table.cell(0, 0)
    set_cell_border(cell, color="5B9BD5", size="10")
    set_cell_shading(cell, PALE_BLUE)
    set_cell_margins(cell, top=180, bottom=180, start=220, end=220)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("FIGURE PLACEHOLDER")
    r.bold = True
    r.font.color.rgb = BLUE
    r.font.size = Pt(13)

    p = cell.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(description)
    r.italic = True
    r.font.size = Pt(10)

    for item in suggested_content:
        p = cell.add_paragraph(style=None)
        p.paragraph_format.left_indent = Inches(0.2)
        p.add_run("- ").bold = True
        p.add_run(item)

    for _ in range(height_lines):
        cell.add_paragraph("")

    add_caption(doc, caption)


def load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("summary", {})


def make_pipeline_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1500, 850
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    except Exception:
        font_title = font = font_small = ImageFont.load_default()

    boxes = [
        ("1. Upload", "Road image +\nVietnamese question"),
        ("2. Detect", "YOLOE crops\ntraffic signs"),
        ("3. Filter", "VLM selects\nrelevant signs"),
        ("4. Retrieve", "CLIP embedding\nqueries Qdrant"),
        ("5. Ground", "LawDB lookup\nadds article text"),
        ("6. Answer", "Strict JSON:\nanswer + citations"),
    ]
    x0, y0, bw, bh = 90, 145, 380, 210
    gap_x, gap_y = 70, 95
    draw.text((60, 45), "Traffic Sign Legal QA Runtime Pipeline", fill=(31, 78, 121), font=font_title)
    draw.line((55, 95, w - 55, 95), fill=(84, 141, 212), width=3)
    for idx, (title, body) in enumerate(boxes):
        col = idx % 3
        row = idx // 3
        x = x0 + col * (bw + gap_x)
        y = y0 + row * (bh + gap_y)
        draw.rounded_rectangle((x, y, x + bw, y + bh), radius=14, outline=(84, 141, 212), width=3, fill=(243, 248, 253))
        draw.ellipse((x + 20, y + 22, x + 64, y + 66), fill=(31, 78, 121))
        draw.text((x + 42, y + 44), str(idx + 1), anchor="mm", fill="white", font=font_small)
        draw.text((x + 82, y + 24), title, fill=(0, 0, 0), font=font_title)
        draw.multiline_text((x + 28, y + 96), body, fill=(40, 40, 40), font=font, spacing=8)
        if idx < len(boxes) - 1:
            if col < 2:
                ax = x + bw + 8
                ay = y + bh // 2
                draw.line((ax, ay, ax + gap_x - 18, ay), fill=(84, 141, 212), width=4)
                draw.polygon([(ax + gap_x - 18, ay), (ax + gap_x - 34, ay - 11), (ax + gap_x - 34, ay + 11)], fill=(84, 141, 212))
            else:
                ax = x + bw // 2
                ay = y + bh + 10
                draw.line((ax, ay, ax, ay + gap_y - 20), fill=(84, 141, 212), width=4)
                draw.polygon([(ax, ay + gap_y - 20), (ax - 11, ay + gap_y - 36), (ax + 11, ay + gap_y - 36)], fill=(84, 141, 212))
    draw.rounded_rectangle((140, 735, w - 140, 810), radius=12, outline=(127, 127, 127), width=2, fill=(255, 255, 255))
    draw.text((170, 755), "Stable version:", fill=(192, 0, 0), font=font_title)
    draw.text((450, 760), "free-form JSON, LawDB citations, visual evidence, local GPU LLM endpoint.", fill=(0, 0, 0), font=font)
    img.save(path)


def make_evaluation_figure(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 1500, 650
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
        bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except Exception:
        title_font = font = bold = ImageFont.load_default()
    draw.text((50, 35), "Current Stable Version Metrics", fill=(31, 78, 121), font=title_font)
    draw.line((50, 88, w - 50, 88), fill=(84, 141, 212), width=3)
    values = [
        ("Answer correctness", metrics.get("answer_correctness", 0)),
        ("Citation accuracy", metrics.get("citation_accuracy", 0)),
        ("Evidence faithfulness", metrics.get("evidence_faithfulness", 0)),
        ("Visual grounding proxy", metrics.get("visual_grounding_accuracy_proxy", 0)),
        ("JSON validity", metrics.get("json_validity", 0)),
        ("1 - hallucination", 1 - metrics.get("hallucination", 0)),
    ]
    left, top, bar_w, row_h = 420, 140, 760, 64
    for i, (label, value) in enumerate(values):
        y = top + i * row_h
        draw.text((70, y + 8), label, fill=(0, 0, 0), font=bold)
        draw.rectangle((left, y + 10, left + bar_w, y + 42), outline=(127, 127, 127), width=2)
        draw.rectangle((left, y + 10, left + int(bar_w * value), y + 42), fill=(91, 155, 213))
        draw.text((left + bar_w + 25, y + 8), f"{value:.2f}", fill=(0, 0, 0), font=font)
    latency = metrics.get("latency", 0)
    draw.rounded_rectangle((90, 555, 1410, 620), radius=12, outline=(127, 127, 127), fill=(243, 248, 253), width=2)
    draw.text((120, 573), f"Average latency: {latency:.2f}s per sample on the evaluated GPU run. Metrics are measured on 20 sign-centric training samples with LLM judge.", fill=(0, 0, 0), font=font)
    img.save(path)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Inches(1)
    section.right_margin = Inches(0.98)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08

    for name, size, color in [
        ("Heading 1", 16, RED),
        ("Heading 2", 13, BLUE),
        ("Heading 3", 11.5, GRAY),
    ]:
        st = styles[name]
        st.font.name = "Arial"
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.paragraph_format.space_before = Pt(10)
        st.paragraph_format.space_after = Pt(6)


def add_cover(doc: Document) -> None:
    table = doc.add_table(rows=1, cols=2)
    table.autofit = False
    table.cell(0, 0).width = Inches(2.45)
    table.cell(0, 1).width = Inches(5.1)
    left = table.cell(0, 0)
    right = table.cell(0, 1)
    for cell in (left, right):
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if TEMPLATE_LOGO.exists():
        left.paragraphs[0].add_run().add_picture(str(TEMPLATE_LOGO), width=Inches(1.75))
    p = right.paragraphs[0]
    r = p.add_run("MINISTRY OF EDUCATION AND TRAINING")
    r.bold = True
    r.font.size = Pt(14)
    for row in table.rows:
        for cell in row.cells:
            cell._tc.get_or_add_tcPr().append(OxmlElement("w:tcBorders"))

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("FPT UNIVERSITY")
    r.bold = True
    r.font.size = Pt(28)

    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Data Science Project Document")
    r.font.size = Pt(26)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24)
    p_pr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "5B9BD5")
    pbdr.append(bottom)
    p_pr.append(pbdr)
    r = p.add_run("Multimodal Legal Question Answering for Traffic Sign Regulations")
    r.bold = True
    r.font.size = Pt(15)

    doc.add_paragraph()
    info = doc.add_table(rows=5, cols=2)
    rows = [
        ("Traffic Legal VLM Team", ""),
        ("Group Members", "Le Phuoc Loc - SE190518\nLe Nhut Tien - SE192973\nNguyen Xuan An - SE192634\nNguyen Thien Ha - SE192649"),
        ("Supervisor", "Nguyen Trong Tai"),
        ("Ext Supervisor", "FPT University Academic Board"),
        ("Capstone Project code", "DSP391m"),
    ]
    for i, (a, b) in enumerate(rows):
        info.rows[i].cells[0].text = a
        info.rows[i].cells[1].text = b
    info.cell(0, 0).merge(info.cell(0, 1))
    info.rows[0].cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in info.rows:
        for cell in row.cells:
            set_cell_border(cell, color="000000")
            set_cell_margins(cell, top=80, bottom=80)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(11.5)
            row.cells[0].paragraphs[0].runs[0].bold = True
    set_table_width(info)

    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run("- HCMC, July/2026 -").font.size = Pt(13)
    doc.add_page_break()


def add_front_matter(doc: Document) -> None:
    add_heading(doc, "Table of Contents", 1)
    toc = [
        "Acknowledgement ........................................................................ 2",
        "Definition and Acronyms ............................................................ 3",
        "List of Tables ........................................................................... 4",
        "List of Figures .......................................................................... 4",
        "I. Project Introduction .............................................................. 5",
        "II. Literature Review and Reference Approach .................................. 8",
        "III. System Analysis and Design ................................................... 11",
        "IV. Implementation ................................................................. 15",
        "V. Evaluation and Results ........................................................ 20",
        "VI. Discussion, Limitations, and Future Work .................................. 24",
        "VII. Conclusion ....................................................................... 27",
        "References ............................................................................ 28",
    ]
    for line in toc:
        add_paragraph(doc, line)
    doc.add_page_break()

    add_heading(doc, "Acknowledgement", 1)
    add_paragraph(doc, "The team would like to thank the DSP391 lecturers, supervisor, and reviewers for guidance during the design, implementation, and evaluation of this project. The current report also acknowledges the VLSP 2025 MLQA-TSR shared task and Vietnamese traffic-sign regulations as important context for building a training-free multimodal legal QA pipeline.")

    add_heading(doc, "Definition and Acronyms", 1)
    add_named_table(
        doc,
        "Table 1. Definitions and acronyms used in this report",
        ["Acronym", "Definition"],
        [
            ["AI", "Artificial Intelligence"],
            ["VQA", "Visual Question Answering"],
            ["VLM", "Vision-Language Model"],
            ["LLM", "Large Language Model"],
            ["MLQA-TSR", "Multimodal Legal Question Answering on Traffic Sign Rules"],
            ["LawDB", "Legal database containing Vietnamese traffic-law articles and traffic-sign regulation content"],
            ["Qdrant", "Vector database used to store traffic-sign embeddings and article metadata"],
            ["Citation", "Structured reference to law_id, article_id, article title, and evidence text"],
            ["Evidence faithfulness", "Whether the answer is supported by provided legal and visual evidence"],
        ],
        widths=[1.6, 5.8],
    )
    doc.add_page_break()

    add_heading(doc, "List of Tables", 1)
    for line in [
        "Table 1. Definitions and acronyms used in this report",
        "Table 2. Final project scope and non-scope",
        "Table 3. Reference task requirements and current project design",
        "Table 4. Runtime components and configuration",
        "Table 5. Free-form answer JSON schema",
        "Table 6. Evaluation metrics",
        "Table 7. Version comparison on the sign-centric 20-sample slice",
        "Table 8. Current stable version metric summary",
        "Table 9. Risk and limitation register",
    ]:
        add_paragraph(doc, line)
    add_heading(doc, "List of Figures", 1)
    for line in [
        "Figure 1. Runtime pipeline for Traffic Sign Legal QA",
        "Figure 2. Current stable version metric profile",
    ]:
        add_paragraph(doc, line)
    doc.add_page_break()


def add_project_intro(doc: Document) -> None:
    add_heading(doc, "I. Project Introduction", 1)
    add_heading(doc, "1. Overview", 2)
    add_heading(doc, "1.1 Project Information", 3)
    add_paragraph(doc, "Project name in English: Traffic Sign Legal QA with Free-form Cited Answers.")
    add_paragraph(doc, "Project name in Vietnamese: Hỏi đáp pháp luật giao thông từ ảnh biển báo với câu trả lời tự do và trích dẫn căn cứ.")
    add_paragraph(doc, "Project type: Data Science / Multimodal AI capstone prototype. The system is designed as a research and educational assistant, not as official legal advice.")
    add_paragraph(doc, "Primary product: an end-to-end Streamlit application that accepts a road-scene image and a natural-language question, detects relevant traffic signs, retrieves LawDB evidence, and returns a Vietnamese free-form answer with legal citations and visual evidence.")

    add_heading(doc, "1.2 Project Background", 3)
    add_paragraph(doc, "Vietnamese traffic-sign interpretation is naturally multimodal. The legal answer depends not only on a written regulation, but also on the visual sign variant, any supplementary plate, and the user's question intent. A text-only legal search engine may retrieve an article, but it cannot reliably connect the visual sign in a real road scene to the correct legal provision.")
    add_paragraph(doc, "The VLSP 2025 MLQA-TSR task formalizes this challenge through two linked subtasks: retrieving relevant legal articles and answering legal questions from road images. The original benchmark requires answers in multiple-choice or Yes/No form. This project extends the practical user experience by changing Subtask 2 into a free-form Vietnamese QA mode with structured citations and evidence.")

    add_heading(doc, "2. Project Objectives", 2)
    add_bullets(doc, [
        "Build a LawDB processing pipeline that aligns traffic-sign crops with law_id and article_id metadata.",
        "Detect traffic signs in uploaded road-scene images and filter signs according to the question.",
        "Retrieve top-k legal sign candidates from Qdrant using image embeddings.",
        "Load full article title and text context for citations.",
        "Generate strict JSON answers containing answer, citations, visual_evidence, reasoning, confidence, and raw debug traces.",
        "Provide a Streamlit demo that can run locally or on a rented GPU through an OpenAI-compatible LLM endpoint.",
        "Evaluate the final system using correctness, citation quality, retrieval recall, grounding, JSON validity, hallucination, and latency metrics.",
    ])

    add_named_table(
        doc,
        "Table 2. Final project scope and non-scope",
        ["Scope item", "Included?", "Notes"],
        [
            ["LawDB preprocessing and sign embedding", "Yes", "The current LawDB contains 489 detailed sign entries after manual quality editing."],
            ["Traffic sign detection in uploaded images", "Yes", "YOLOE is used for query-image sign detection."],
            ["Question-conditioned sign filtering", "Yes", "A VLM/LLM call selects which detected signs are relevant to the question."],
            ["Free-form cited Vietnamese answer", "Yes", "The final answer is strict JSON and no longer forced to A/B/C/D or Yes/No."],
            ["Official legal decision making", "No", "The project is a research prototype and should not be treated as legal authority."],
            ["Fine-tuned model training", "No", "The implementation is training-free and relies on retrieval, prompting, and local GPU inference."],
        ],
        widths=[2.3, 1.1, 4.3],
    )
    doc.add_page_break()


def add_related_work(doc: Document) -> None:
    add_heading(doc, "II. Literature Review and Reference Approach", 1)
    add_heading(doc, "1. Reference Task Direction", 2)
    add_paragraph(doc, "The VLSP 2025 MLQA-TSR setting motivates a training-free pipeline that connects traffic-sign images with Vietnamese legal articles. In this direction, the system preprocesses LawDB, detects signs from road-scene images, embeds sign crops, retrieves related legal evidence, and uses LLM/VLM components for final question answering.")
    add_paragraph(doc, "MLQA-TSR contains two useful engineering targets: retrieving relevant legal articles from LawDB and answering legal questions from road images. LawDB includes the Law on Road Traffic Order and Safety 36/2024/QH15 and QCVN 41:2024/BGTVT. The original benchmark answer format is multiple-choice or Yes/No, while this project presents a user-facing free-form cited answer.")
    add_paragraph(doc, "A key design insight is that clean LawDB sign images and real-world road images should be handled differently. Clean LawDB images can be segmented during preprocessing, while road-scene images benefit from learning-based detectors such as YOLOE or GroundingDINO. The resulting sign crops can be embedded and indexed with legal metadata.")

    add_named_table(
        doc,
        "Table 3. Reference task requirements and current project design",
        ["Aspect", "Reference task direction", "Current project"],
        [
            ["Overall strategy", "Training-free multi-stage retrieval and QA pipeline.", "End-to-end demo pipeline with a product-oriented free-form cited QA layer."],
            ["LawDB processing", "Convert legal text, crop clean sign images, parse sign information, embed and store metadata.", "Manually reviewed LawDB sign descriptions and 489 detailed sign entries prepared for Qdrant ingest."],
            ["Question-image processing", "Detect signs, filter relevant signs, retrieve legal articles, answer benchmark format.", "Detect signs, filter relevant signs, retrieve top-k sign candidates, load full article context, answer strict JSON."],
            ["Subtask 2 output", "Multiple-choice or Yes/No answer according to benchmark rules.", "Vietnamese free-form answer with law_id, article_id, evidence, visual evidence, reasoning, confidence."],
            ["Serving", "Local LLM/VLM serving is suitable for reproducible experiments.", "Latest GPU demo uses Ollama gemma3:12b through an OpenAI-compatible endpoint at port 18989."],
        ],
        widths=[1.6, 3.0, 3.0],
    )

    add_heading(doc, "2. Design Implications", 2)
    add_bullets(doc, [
        "Article retrieval and answer generation should be separated so citations can be validated independently.",
        "Traffic-sign crops should carry payload metadata, especially law_id, article_id, sign_name, and sign_description.",
        "The answer prompt should prevent unsupported legal claims by requiring citations only from provided article context.",
        "A training-free system is practical for a capstone environment because it avoids collecting labels for model fine-tuning and focuses effort on retrieval quality and evidence traceability.",
    ])


def add_design(doc: Document) -> None:
    metrics = load_metrics(METRICS_CURRENT)
    pipeline = ASSET_DIR / "runtime_pipeline.png"
    eval_fig = ASSET_DIR / "metric_profile.png"
    make_pipeline_figure(pipeline)
    make_evaluation_figure(eval_fig, metrics)

    add_heading(doc, "III. System Analysis and Design", 1)
    add_heading(doc, "1. Functional Requirements", 2)
    add_bullets(doc, [
        "The user can upload a .jpg, .jpeg, or .png traffic image.",
        "The user can ask a natural-language question in Vietnamese and optionally provide reference choices.",
        "The system detects sign crops and shows which crops are selected as relevant.",
        "The system retrieves legal evidence from Qdrant and resolves article content through LawDBArticleLookup.",
        "The system returns a free-form answer, citations, visual evidence, reasoning, confidence, and raw JSON/debug information.",
    ])

    add_heading(doc, "2. Runtime Architecture", 2)
    add_paragraph(doc, "The current stable architecture is intentionally simple. The application layer is Streamlit. The core pipeline is Python. Qdrant stores sign embeddings and metadata. A local GPU LLM is accessed through an OpenAI-compatible API, so the code can switch between Ollama, llama.cpp, vLLM, or hosted endpoints by changing environment variables.")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(pipeline), width=Inches(6.8))
    add_caption(doc, "Figure 1. Runtime pipeline for Traffic Sign Legal QA")

    add_named_table(
        doc,
        "Table 4. Runtime components and configuration",
        ["Component", "Latest GPU demo value", "Role"],
        [
            ["Streamlit", "src/ui/freeform_qa_demo.py", "Interactive demo for upload, analysis, citations, visual evidence, and debug output."],
            ["Detector", "YoloE", "Detects candidate traffic signs in user-uploaded road images."],
            ["Embedding", "CLIP-GmP-ViT-L-14", "Embeds sign crops and LawDB sign images for vector retrieval."],
            ["Vector store", "Qdrant on port 6333", "Stores LawDB sign embeddings and article payload metadata."],
            ["LLM endpoint", "Ollama OpenAI-compatible API on port 18989", "Runs gemma3:12b in the latest GPU demo for sign filtering and answer generation."],
            ["Retrieval limit", "LIMIT=5", "Retrieves top-k LawDB sign candidates per selected crop."],
            ["Answer temperature", "0.0", "Reduces randomness and improves JSON consistency."],
        ],
        widths=[1.8, 2.4, 3.4],
    )

    add_heading(doc, "3. Output Contract", 2)
    add_paragraph(doc, "The most important design change is that Subtask 2 no longer returns only A/B/C/D or Yes/No. The model is required to return strict JSON. The parser then normalizes citations and visual evidence against the provided context. If JSON parsing fails, the system keeps the raw response and returns a low-confidence fallback.")
    add_named_table(
        doc,
        "Table 5. Free-form answer JSON schema",
        ["Field", "Description"],
        [
            ["answer", "Short Vietnamese free-form answer grounded in the provided legal and visual context."],
            ["citations", "List of law_id, article_id, title, and evidence from LawDB article context."],
            ["visual_evidence", "Detected or retrieved signs used by the answer, including sign_name, description, and image_name."],
            ["reasoning", "Short explanation connecting the sign evidence to the legal article."],
            ["confidence", "One of high, medium, or low."],
            ["raw_answer_response", "Stored raw model response for debugging and audit."],
        ],
        widths=[1.8, 5.8],
    )
    doc.add_page_break()


def add_implementation(doc: Document) -> None:
    add_heading(doc, "IV. Implementation", 1)
    add_heading(doc, "1. LawDB Processing", 2)
    law_hash = hashlib.sha256(LAWDB.read_bytes()).hexdigest() if LAWDB.exists() else "N/A"
    add_paragraph(doc, "The LawDB pipeline preprocesses legal articles, extracts clean sign images, parses sign names and descriptions, embeds sign crops, and prepares one retrieval entry per detailed sign. The latest manually reviewed LawDB file is used by both ingest and article lookup.")
    add_bullets(doc, [
        f"Current LawDB file: data/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json.",
        f"Traceability hash: {law_hash}.",
        "Number of detailed sign entries prepared for Qdrant ingest: 489.",
        "Each Qdrant payload stores law_id, article_id, image_path, sign_name, and sign_description.",
    ])

    add_heading(doc, "2. Image QA Pipeline", 2)
    add_paragraph(doc, "At runtime, src/core/freeform_qa.py orchestrates the single-image pipeline used by Streamlit. The uploaded image is copied to a temporary working directory, signs are detected, selected signs are passed to Qdrant retrieval, article contexts are loaded, and src/core/sub_task_2.py builds the final cited QA prompt.")
    add_paragraph(doc, "Question-conditioned filtering is handled by src/core/filter_signs.py. If only one sign is detected, it is selected automatically. If multiple signs are present, the model receives the original image, crop sequence, and question, then returns a boolean relevance decision per crop.")

    add_heading(doc, "3. Prompting and Parsing", 2)
    add_paragraph(doc, "The prompt in src/prompts/answer_prompt.py enforces four constraints: use only supplied context, do not invent law articles, cite only articles that appear in the legal context, and provide visual evidence only from detected/retrieved signs. The parser in src/core/sub_task_2.py extracts JSON, validates allowed citations, normalizes visual evidence, and falls back to low confidence if the raw model response is malformed.")

    add_heading(doc, "4. Deployment on GPU", 2)
    add_paragraph(doc, "The current tested deployment used a rented RTX 4090 GPU server. Qdrant ran on internal port 6333, Ollama served gemma3:12b through an OpenAI-compatible endpoint on internal port 18989, and Streamlit served the demo on internal port 8501. The provider mapped Streamlit to the public URL used for manual testing.")
    add_bullets(doc, [
        "LLM_BASE_URL=http://localhost:18989/v1",
        "LLM_MODEL=gemma3:12b",
        "QDRANT_URI=http://localhost:6333",
        "IMAGE_QUERY_DETECTION_MODEL=YoloE",
        "EMBEDDING_NAME=CLIP-GmP-ViT-L-14",
        "LIMIT=5",
    ])
    doc.add_page_break()


def add_evaluation(doc: Document) -> None:
    current = load_metrics(METRICS_CURRENT)
    baseline = load_metrics(METRICS_BASELINE)
    optimized = load_metrics(METRICS_OPTIMIZED)
    eval_fig = ASSET_DIR / "metric_profile.png"

    add_heading(doc, "V. Evaluation and Results", 1)
    add_heading(doc, "1. Evaluation Setup", 2)
    add_paragraph(doc, "The final reported metrics are measured on a 20-sample sign-centric training slice using --begin-idx 39 --num-samples 20 --judge. This slice is more representative for traffic-sign retrieval and visual grounding than the first 20 samples, which contain many general legal questions.")
    add_paragraph(doc, "The judge mode uses the configured LLM to evaluate answer correctness, evidence faithfulness, and hallucination. Retrieval and citation metrics are computed programmatically from sample relevant_articles, returned citations, and retrieved payloads.")

    add_named_table(
        doc,
        "Table 6. Evaluation metrics",
        ["Metric", "Purpose"],
        [
            ["Answer correctness", "Whether the free-form answer matches the benchmark ground-truth answer meaning."],
            ["Citation accuracy / precision / recall", "Whether returned law_id and article_id match gold relevant articles."],
            ["Evidence faithfulness", "Whether citations and reasoning support the answer."],
            ["Visual grounding accuracy proxy", "Whether selected signs retrieve gold-related payloads at top-1."],
            ["Retrieval Recall@1/3/5", "Whether gold articles appear in top-k retrieved payloads for selected signs."],
            ["JSON validity", "Whether raw_answer_response is parseable strict JSON with required fields."],
            ["Hallucination rate", "Whether the answer invents unsupported information."],
            ["Average latency", "Mean end-to-end sample processing time."],
        ],
        widths=[2.4, 5.2],
    )

    add_heading(doc, "2. Version Comparison", 2)
    rows = []
    for name, data in [
        ("Baseline sign-centric run", baseline),
        ("Optimized query-time parse version", optimized),
        ("Current stable reverted version", current),
    ]:
        rows.append([
            name,
            f"{data.get('answer_correctness', 0):.2f}",
            f"{data.get('citation_accuracy', 0):.2f}",
            f"{data.get('visual_grounding_accuracy_proxy', 0):.3f}",
            f"{data.get('retrieval_recall_at_5', 0):.3f}",
            f"{data.get('hallucination', 0):.2f}",
            f"{data.get('latency', 0):.2f}s",
        ])
    add_named_table(
        doc,
        "Table 7. Version comparison on the sign-centric 20-sample slice",
        ["Version", "Answer", "Citation", "Visual", "R@5", "Halluc.", "Latency"],
        rows,
        widths=[2.4, 0.75, 0.85, 0.85, 0.75, 0.75, 0.9],
    )

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(eval_fig), width=Inches(6.7))
    add_caption(doc, "Figure 2. Current stable version metric profile")

    add_named_table(
        doc,
        "Table 8. Current stable version metric summary",
        ["Metric", "Value"],
        [
            ["Answer correctness", f"{current.get('answer_correctness', 0):.2f}"],
            ["Citation accuracy", f"{current.get('citation_accuracy', 0):.2f}"],
            ["Citation precision", f"{current.get('citation_precision', 0):.2f}"],
            ["Citation recall", f"{current.get('citation_recall', 0):.2f}"],
            ["Evidence faithfulness", f"{current.get('evidence_faithfulness', 0):.2f}"],
            ["Visual grounding accuracy proxy", f"{current.get('visual_grounding_accuracy_proxy', 0):.3f}"],
            ["Retrieval Recall@1 / @3 / @5", f"{current.get('retrieval_recall_at_1', 0):.3f} / {current.get('retrieval_recall_at_3', 0):.3f} / {current.get('retrieval_recall_at_5', 0):.3f}"],
            ["JSON validity", f"{current.get('json_validity', 0):.2f}"],
            ["Hallucination rate", f"{current.get('hallucination', 0):.2f}"],
            ["Average latency", f"{current.get('latency', 0):.2f}s"],
        ],
        widths=[3.3, 2.1],
    )
    add_paragraph(doc, "The current stable version is preferred for presentation because it preserves 0.80 answer correctness, improves citation accuracy over the earlier baseline, eliminates hallucination in the judged slice, and reduces latency to approximately 17.20 seconds per sample.")
    doc.add_page_break()


def add_discussion_conclusion(doc: Document) -> None:
    add_heading(doc, "VI. Discussion, Limitations, and Future Work", 1)
    add_heading(doc, "1. Main Findings", 2)
    add_bullets(doc, [
        "Manual quality review of LawDB sign descriptions is valuable because retrieval quality depends heavily on sign_name and sign_description metadata.",
        "Increasing query-time reasoning steps can improve interpretability in some cases, but the measured optimized version did not improve answer correctness and increased latency.",
        "The stable reverted version is a better demo candidate because it is faster, simpler, and produced stronger citation and visual grounding metrics in the latest comparable run.",
        "Strict JSON output and parser fallback are necessary for Streamlit rendering and metric computation.",
    ])

    add_heading(doc, "2. Limitations", 2)
    add_named_table(
        doc,
        "Table 9. Risk and limitation register",
        ["Limitation", "Impact", "Mitigation"],
        [
            ["Small judged sample size", "Metrics may not represent the full dataset.", "Report the exact slice and keep JSON artifacts for traceability."],
            ["LLM judge variability", "Answer correctness and hallucination may shift with model behavior.", "Use temperature 0.0 and combine judge metrics with programmatic metrics."],
            ["Similar sign variants", "P.131a/P.131b/P.131c style variants can be confused.", "Continue manual LawDB review and add targeted retrieval tests for common variant families."],
            ["Legal reliability", "Generated answers may still be incomplete or overconfident.", "Show citations, confidence, raw JSON, and state that the demo is not official legal advice."],
            ["Network and GPU dependency", "Live demo can slow down if GPU rental or school network is unstable.", "Prepare the GPU in advance and keep Streamlit/Qdrant/Ollama warmed before presentation."],
        ],
        widths=[2.0, 2.4, 3.1],
    )

    add_heading(doc, "3. Future Work", 2)
    add_bullets(doc, [
        "Evaluate on a larger sign-centric and full training set with fixed seeds and archived raw outputs.",
        "Add targeted regression tests for confusing sign families and supplementary plates.",
        "Improve retrieval metadata without adding expensive query-time calls unless metrics justify the latency.",
        "Add optional demo cache for presentation images when network quality is poor.",
        "Experiment with other OpenAI-compatible local serving backends only after the stable version is frozen.",
    ])

    add_heading(doc, "VII. Conclusion", 1)
    add_paragraph(doc, "This project delivers a working multimodal legal QA prototype for Vietnamese traffic signs. The system combines a training-free retrieval pipeline with a practical free-form cited QA demo. The implementation detects relevant signs, retrieves LawDB evidence, loads article context, and generates Vietnamese answers with citations and visual evidence.")
    add_paragraph(doc, "The latest stable version is the recommended presentation version. It maintains 0.80 answer correctness on the sign-centric slice, reaches 0.65 citation accuracy, keeps JSON validity at 1.00, reports zero hallucination under the judge metric, and reduces average latency to 17.20 seconds. These results make it a balanced choice between quality, explainability, and live-demo reliability.")

    add_heading(doc, "References", 1)
    refs = [
        "The Association for Vietnamese Language and Speech Processing. 2025. VLSP 2025 Multimodal Legal Question Answering on Traffic Sign Rules shared task.",
        "QCVN 41:2024/BGTVT. National Technical Regulation on Road Signs and Signals.",
        "Law 36/2024/QH15. Law on Road Traffic Order and Safety.",
        "Project repository artifacts: README.md, src/core/freeform_qa.py, src/core/sub_task_2.py, src/core/lawdb/article_lookup.py, outputs/freeform_qa_metrics_reverted_previous_begin39_20_judge_20260728.json.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Paragraph")
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.add_run("• ")
        p.add_run(ref)


def add_front_matter_extended(doc: Document) -> None:
    add_heading(doc, "Table of Contents", 1)
    toc = [
        "Acknowledgement ........................................................................ 2",
        "Definition and Acronyms ............................................................ 3",
        "List of Tables ........................................................................... 4",
        "List of Figures .......................................................................... 4",
        "I. Project Introduction .............................................................. 5",
        "II. Project Management Plan ....................................................... 9",
        "III. Existing Systems/State of the Art ........................................... 12",
        "IV. Methodology .................................................................... 16",
        "V. System Design and Implementation ............................................ 21",
        "VI. Results and Discussion ....................................................... 25",
        "VII. Conclusion .................................................................... 31",
        "VIII. References ................................................................... 33",
        "IX. Appendices .................................................................... 34",
    ]
    for line in toc:
        add_paragraph(doc, line)
    doc.add_page_break()

    add_heading(doc, "Acknowledgement", 1)
    add_paragraph(doc, "The team would like to thank the DSP391 lecturers, supervisor, and reviewers for guidance during the design, implementation, and evaluation of this project. The current report also acknowledges the VLSP 2025 MLQA-TSR shared task and Vietnamese traffic-sign regulations as important context for building a training-free multimodal legal QA pipeline for Vietnamese traffic-sign rules.")
    add_paragraph(doc, "This report is written to support final presentation and project defense. Therefore, it explains the system architecture, implementation choices, evaluation evidence, and limitations clearly, while avoiding unnecessary low-level details that would make the defense depend on fragile implementation minutiae.")

    add_heading(doc, "Definition and Acronyms", 1)
    add_named_table(
        doc,
        "Table 1. Definitions and acronyms used in this report",
        ["Acronym", "Definition"],
        [
            ["AI", "Artificial Intelligence"],
            ["VQA", "Visual Question Answering"],
            ["VLM", "Vision-Language Model"],
            ["LLM", "Large Language Model"],
            ["MLQA-TSR", "Multimodal Legal Question Answering on Traffic Sign Rules"],
            ["LawDB", "Legal database containing Vietnamese traffic-law articles and traffic-sign regulation content"],
            ["Qdrant", "Vector database used to store traffic-sign embeddings and article metadata"],
            ["OpenAI-compatible endpoint", "A local or hosted API that follows the /v1/chat/completions interface"],
            ["Citation", "Structured reference to law_id, article_id, article title, and evidence text"],
            ["Visual evidence", "Detected or retrieved traffic-sign crop used as visual support for the answer"],
            ["Evidence faithfulness", "Whether the answer is supported by provided legal and visual evidence"],
            ["JSON validity", "Whether the answer response follows the required structured JSON schema"],
        ],
        widths=[1.8, 5.8],
    )
    doc.add_page_break()

    add_heading(doc, "List of Tables", 1)
    for line in [
        "Table 1. Definitions and acronyms used in this report",
        "Table 2. Final project scope and non-scope",
        "Table 3. Project work package overview",
        "Table 4. Quality control strategy",
        "Table 5. Reference task requirements and current project design",
        "Table 6. Runtime components and configuration",
        "Table 7. Free-form answer JSON schema",
        "Table 8. Main modules in the implementation",
        "Table 9. Evaluation metrics",
        "Table 10. Final evaluated system metric summary",
        "Table 11. Demonstration readiness checklist",
        "Table 12. Risk and limitation register",
    ]:
        add_paragraph(doc, line)

    add_heading(doc, "List of Figures", 1)
    for line in [
        "Figure 1. User scenario for traffic-sign legal QA",
        "Figure 2. Training-free multimodal legal QA pipeline",
        "Figure 3. Current system architecture",
        "Figure 4. LawDB processing workflow",
        "Figure 5. Streamlit demo screen layout",
        "Figure 6. Metric profile for the final evaluated system",
        "Figure 7. GPU demo deployment topology",
    ]:
        add_paragraph(doc, line)
    doc.add_page_break()


def add_project_intro_extended(doc: Document) -> None:
    add_heading(doc, "I. Project Introduction", 1)
    add_heading(doc, "1. Overview", 2)
    add_heading(doc, "1.1 Project Information", 3)
    add_paragraph(doc, "Project name in English: Multimodal Legal Question Answering for Traffic Sign Regulations.")
    add_paragraph(doc, "Project name in Vietnamese: Hoi dap phap luat da phuong thuc cho quy dinh bien bao giao thong.")
    add_paragraph(doc, "Current prototype focus: Traffic Sign Legal QA with free-form cited Vietnamese answers.")
    add_paragraph(doc, "Project type: Data Science / Multimodal AI capstone prototype. The system is designed as a research and educational assistant, not as official legal advice.")
    add_paragraph(doc, "Primary product: an end-to-end Streamlit application that accepts a road-scene image and a natural-language question, detects relevant traffic signs, retrieves LawDB evidence, and returns a Vietnamese free-form answer with legal citations and visual evidence.")

    add_heading(doc, "1.2 Motivation", 3)
    add_paragraph(doc, "Vietnamese traffic-sign interpretation is naturally multimodal. The legal answer depends not only on a written regulation, but also on the visual sign variant, any supplementary plate, and the user's question intent. A text-only legal search engine may retrieve an article, but it cannot reliably connect the visual sign in a real road scene to the correct legal provision.")
    add_paragraph(doc, "In everyday use, a driver or learner does not ask for an article ID first. They usually ask a practical question such as whether a vehicle may enter, whether stopping is allowed, or what time range a prohibition applies to. The project therefore prioritizes a user-facing answer with citations and evidence rather than a benchmark-only label.")
    add_figure_placeholder(
        doc,
        "Figure 1. User scenario for traffic-sign legal QA",
        "Insert a simple 4-step visual scenario: user sees sign, uploads image, asks question, receives answer with citations.",
        [
            "Left side: a road-scene image containing a traffic sign.",
            "Middle: user asks a Vietnamese legal question.",
            "Right side: answer card with law_id, article_id, and visual evidence.",
            "Use the same blue accent and white background as the report template.",
        ],
        height_lines=4,
    )

    add_heading(doc, "2. Problem Statement", 2)
    add_paragraph(doc, "The core problem is to answer legal questions from traffic-sign images while making the evidence auditable. The system must combine object-level visual detection, sign-level retrieval, article lookup, and controlled generation. A useful answer must not only be plausible; it must show which legal article and which detected sign support the conclusion.")
    add_paragraph(doc, "The original benchmark format for Subtask 2 is multiple-choice or Yes/No. That format is useful for scoring but not ideal for a demo because it hides the legal reasoning process. The current project changes the output into a structured free-form JSON answer with citations, visual evidence, reasoning, confidence, and raw response diagnostics.")

    add_heading(doc, "3. Project Objectives", 2)
    add_bullets(doc, [
        "Build a LawDB processing pipeline that aligns traffic-sign crops with law_id and article_id metadata.",
        "Detect traffic signs in uploaded road-scene images and filter signs according to the question.",
        "Retrieve top-k legal sign candidates from Qdrant using image embeddings.",
        "Load full article title and text context for citations.",
        "Generate strict JSON answers containing answer, citations, visual_evidence, reasoning, confidence, and raw debug traces.",
        "Provide a Streamlit demo that can run locally or on a rented GPU through an OpenAI-compatible LLM endpoint.",
        "Evaluate the final system using correctness, citation quality, retrieval recall, grounding, JSON validity, hallucination, and latency metrics.",
    ])

    add_named_table(
        doc,
        "Table 2. Final project scope and non-scope",
        ["Scope item", "Included?", "Notes"],
        [
            ["LawDB preprocessing and sign embedding", "Yes", "The current LawDB contains 489 detailed sign entries after manual quality editing."],
            ["Traffic sign detection in uploaded images", "Yes", "YOLOE is used for query-image sign detection."],
            ["Question-conditioned sign filtering", "Yes", "A VLM/LLM call selects which detected signs are relevant to the question."],
            ["Free-form cited Vietnamese answer", "Yes", "The final answer is strict JSON and no longer forced to A/B/C/D or Yes/No."],
            ["Official legal decision making", "No", "The project is a research prototype and should not be treated as legal authority."],
            ["Fine-tuned model training", "No", "The implementation is training-free and relies on retrieval, prompting, and local GPU inference."],
        ],
        widths=[2.3, 1.1, 4.3],
    )
    doc.add_page_break()


def add_project_management_extended(doc: Document) -> None:
    add_heading(doc, "II. Project Management Plan", 1)
    add_heading(doc, "1. Work Package Overview", 2)
    add_paragraph(doc, "The project was managed as a sequence of small work packages rather than as a single large implementation. This was important because multimodal legal QA combines several moving parts: data preparation, vector retrieval, prompt design, local GPU serving, UI design, and evaluation. Separating the work into packages made it easier to test each layer independently.")
    add_named_table(
        doc,
        "Table 3. Project work package overview",
        ["Work package", "Main output", "Acceptance evidence"],
        [
            ["LawDB preparation", "Preprocessed LawDB, extracted sign crops, manually reviewed sign metadata.", "489 sign entries ingested into Qdrant; LawDB hash recorded."],
            ["Retrieval pipeline", "Sign crop embedding and Qdrant top-k retrieval.", "Recall@1/3/5 metrics and retrieved payload inspection."],
            ["Free-form QA", "Strict JSON answer with citations and visual evidence.", "JSON validity, citation metrics, raw response logs."],
            ["Streamlit demo", "Upload UI, answer panel, citations, visual evidence, debug section.", "Demo launched on local and GPU environments."],
            ["GPU deployment", "Ollama/Qdrant/Streamlit services on rented GPU.", "Public Streamlit URL and benchmark run."],
            ["Reporting", "Final DOCX report, metrics summary, defense-friendly explanation.", "Rendered document and archived JSON metrics."],
        ],
        widths=[2.0, 3.0, 2.7],
    )

    add_heading(doc, "2. Quality Management", 2)
    add_paragraph(doc, "The main quality principle is traceability. Reported numbers should come from JSON artifacts, cited articles should exist in the LawDB context, and demo behavior should be reproducible from environment variables and Makefile commands. This avoids relying only on manual impressions of the demo.")
    add_named_table(
        doc,
        "Table 4. Quality control strategy",
        ["Quality concern", "Control used in the project"],
        [
            ["Wrong or unsupported legal claim", "Prompt restricts citations to provided article context; parser filters citations against allowed articles."],
            ["Invalid model output", "Strict JSON schema plus fallback handling with low confidence and raw response retention."],
            ["Wrong LawDB file", "Manual LawDB file hash was compared between local and GPU before metric runs."],
            ["Unstable demo setup", "Qdrant, Ollama, and Streamlit are warmed and checked by HTTP endpoints before manual testing."],
            ["Metric traceability", "The final reported metrics are linked to archived JSON artifacts and the exact evaluated sample slice."],
        ],
        widths=[2.5, 5.0],
    )

    add_heading(doc, "3. Change Management", 2)
    add_paragraph(doc, "Change management focused on preserving a clear and reproducible demonstration path. Each major change was checked against compile results, manual demo behavior, LawDB consistency, and archived metric outputs.")
    add_paragraph(doc, "For presentation, the report describes the final evaluated system as a single coherent pipeline. Internal development attempts are intentionally omitted so reviewers can focus on the delivered behavior, evidence flow, and measured results.")
    doc.add_page_break()


def add_related_work_extended(doc: Document) -> None:
    add_heading(doc, "III. Existing Systems/State of the Art", 1)
    add_heading(doc, "1. Overview of the Field", 2)
    add_paragraph(doc, "The VLSP 2025 Multimodal Legal Question Answering on Traffic Sign Rules task is built around the need to connect legal text with road-scene images. Subtask 1 focuses on retrieving relevant legal articles. Subtask 2 extends retrieval into question answering, where the original benchmark expects either a multiple-choice answer or a Yes/No answer.")
    add_paragraph(doc, "This project keeps the retrieval-grounded structure and changes the answer format for a practical demo. Instead of returning only a label, the system returns a short Vietnamese answer and exposes citations and visual evidence. This makes the output easier to inspect during a presentation and closer to what a real user would expect.")

    add_heading(doc, "2. Historical Context", 2)
    add_paragraph(doc, "Early legal information systems were mostly text-search tools. They could return legal documents, but they did not interpret road-scene images. Modern multimodal QA systems add vision models and retrieval, which makes it possible to connect an uploaded traffic image with legal text and structured answer generation.")

    add_heading(doc, "3. Key Studies and Theories", 2)
    add_paragraph(doc, "A suitable approach for this project is a training-free multimodal pipeline. The system can combine traffic sign detection, image embeddings, question-conditioned VLM filtering, LawDB preprocessing, and vector retrieval without requiring expensive fine-tuning. This makes the project easier to reproduce in a capstone environment while still producing auditable answers.")
    add_paragraph(doc, "The key insight is that clean LawDB images and real-world road images should be handled differently. Clean LawDB images can be segmented during preprocessing, while road-scene images benefit from learning-based detectors such as YOLOE or GroundingDINO. The resulting sign crops can be embedded and indexed with legal metadata.")
    add_figure_placeholder(
        doc,
        "Figure 2. Training-free multimodal legal QA pipeline",
        "Insert a simplified version of the project's multi-stage pipeline.",
        [
            "Show LawDB preprocessing on the left and road-scene query processing on the right.",
            "Connect sign detection, VLM filtering, embedding, Qdrant retrieval, and QA.",
            "Show that the final answer is free-form JSON with legal citations and visual evidence.",
        ],
        height_lines=5,
    )

    add_heading(doc, "4. Technological Advancements", 2)
    add_paragraph(doc, "Recent progress in open-source VLMs, image embedding models, vector databases, and local GPU serving makes this project feasible without training a new foundation model. The current system uses detector-based visual localization, CLIP-style embeddings for sign retrieval, Qdrant for vector search, and an OpenAI-compatible LLM endpoint for final reasoning.")

    add_named_table(
        doc,
        "Table 5. Reference task requirements and current project design",
        ["Aspect", "Reference task direction", "Current project"],
        [
            ["Overall strategy", "Training-free multi-stage retrieval and QA pipeline.", "End-to-end demo pipeline with a product-oriented free-form cited QA layer."],
            ["LawDB processing", "Convert legal text, crop clean sign images, parse sign information, embed and store metadata.", "Manually reviewed LawDB sign descriptions and 489 detailed sign entries prepared for Qdrant ingest."],
            ["Question-image processing", "Detect signs, filter relevant signs, retrieve legal articles, answer benchmark format.", "Detect signs, filter relevant signs, retrieve top-k sign candidates, load full article context, answer strict JSON."],
            ["Subtask 2 output", "Multiple-choice or Yes/No answer according to benchmark rules.", "Vietnamese free-form answer with law_id, article_id, evidence, visual evidence, reasoning, confidence."],
            ["Serving", "Local LLM/VLM serving is suitable for reproducible experiments.", "Latest GPU demo uses Ollama gemma3:12b through an OpenAI-compatible endpoint at port 18989."],
        ],
        widths=[1.6, 3.0, 3.0],
    )

    add_heading(doc, "5. Comparison of Existing Systems", 2)
    add_paragraph(doc, "Compared with a text-only legal lookup tool, the current system uses visual evidence from the uploaded image. Compared with a direct VLM answer, it retrieves LawDB context first so the generated answer can expose law_id, article_id, title, and evidence text. Compared with a pure benchmark classifier, it returns a defense-friendly free-form answer that can be inspected by users.")

    add_heading(doc, "6. Gaps in the Literature/Technology", 2)
    add_bullets(doc, [
        "Similar sign variants can still be confused when their visual differences are small.",
        "Generated legal answers need strict grounding because an LLM can otherwise produce plausible but unsupported statements.",
        "A demo system must balance answer quality with latency, especially when running on rented GPU infrastructure.",
    ])

    add_heading(doc, "7. Justification for the Project", 2)
    add_bullets(doc, [
        "Article retrieval and answer generation should be separated so citations can be validated independently.",
        "Traffic-sign crops should carry payload metadata, especially law_id, article_id, sign_name, and sign_description.",
        "The answer prompt should prevent unsupported legal claims by requiring citations only from provided article context.",
        "A training-free system is practical for a capstone environment because it avoids collecting labels for model fine-tuning and focuses effort on retrieval quality and evidence traceability.",
    ])
    doc.add_page_break()


def add_design_extended(doc: Document) -> None:
    add_heading(doc, "IV. Methodology", 1)
    add_heading(doc, "1. Research Questions and Objectives", 2)
    add_paragraph(doc, "The project investigates whether a training-free multimodal pipeline can answer Vietnamese traffic-sign legal questions with traceable citations. The practical objective is to build a demo that accepts an image and a question, detects relevant signs, retrieves LawDB evidence, and generates a structured answer that can be checked by reviewers.")
    add_bullets(doc, [
        "The user can upload a .jpg, .jpeg, or .png traffic image.",
        "The user can ask a natural-language question in Vietnamese and optionally provide reference choices.",
        "The system detects sign crops and shows which crops are selected as relevant.",
        "The system retrieves legal evidence from Qdrant and resolves article content through LawDBArticleLookup.",
        "The system returns a free-form answer, citations, visual evidence, reasoning, confidence, and raw JSON/debug information.",
    ])
    add_paragraph(doc, "The non-functional requirements are stability, explainability, reasonable latency, and easy deployment on a rented GPU. The system is allowed to be slower than a pure classifier because it returns richer evidence, but it should not be so slow that a live presentation becomes uncomfortable.")

    add_heading(doc, "2. Data Collection and Preprocessing", 2)
    add_paragraph(doc, "The data layer starts from LawDB legal articles and associated traffic-sign images. The preprocessing step extracts sign entries, stores sign_name and sign_description metadata, and prepares sign crops for embedding. During development, the sign descriptions were manually reviewed because retrieval performance depends heavily on this metadata.")
    add_figure_placeholder(
        doc,
        "Figure 4. LawDB processing workflow",
        "Insert a workflow figure showing how raw LawDB becomes a Qdrant collection.",
        [
            "Raw LawDB JSON and clean legal-document sign images.",
            "Preprocessing, sign crop extraction, VLM/manual sign description review.",
            "Embedding generation and Qdrant ingest.",
            "Payload fields: law_id, article_id, image_path, sign_name, sign_description.",
        ],
        height_lines=5,
    )

    add_heading(doc, "3. Feature Selection and Engineering", 2)
    add_paragraph(doc, "The main engineered features are visual sign crops, CLIP-style sign embeddings, LawDB article identifiers, sign names, sign descriptions, and article text snippets. These features are chosen because they preserve the link between the visual sign and the legal context needed for citation-based answering.")

    add_heading(doc, "4. Model Training and Validation", 2)
    add_paragraph(doc, "The current system is training-free. Instead of fine-tuning a new model, it integrates a traffic-sign detector, an embedding model, vector retrieval, and a local LLM/VLM endpoint. Validation is performed through compile checks, manual demo tests, Qdrant retrieval inspection, and a judged 20-sample metric run.")

    add_heading(doc, "5. Evaluation Metrics", 2)
    add_paragraph(doc, "The evaluation focuses on answer correctness, citation accuracy, evidence faithfulness, visual grounding, retrieval recall, JSON validity, hallucination rate, and latency. These metrics are selected because the application is useful only when it answers correctly, cites valid legal evidence, and remains usable in a live demo.")

    add_heading(doc, "6. Implementation Plan", 2)
    add_paragraph(doc, "The most important design change is that Subtask 2 no longer returns only A/B/C/D or Yes/No. The model is required to return strict JSON. The parser then normalizes citations and visual evidence against the provided context. If JSON parsing fails, the system keeps the raw response and returns a low-confidence fallback.")
    add_named_table(
        doc,
        "Table 7. Free-form answer JSON schema",
        ["Field", "Description"],
        [
            ["answer", "Short Vietnamese free-form answer grounded in the provided legal and visual context."],
            ["citations", "List of law_id, article_id, title, and evidence from LawDB article context."],
            ["visual_evidence", "Detected or retrieved signs used by the answer, including sign_name, description, and image_name."],
            ["reasoning", "Short explanation connecting the sign evidence to the legal article."],
            ["confidence", "One of high, medium, or low."],
            ["raw_answer_response", "Stored raw model response for debugging and audit."],
        ],
        widths=[1.8, 5.8],
    )

    add_heading(doc, "7. Ethical Considerations", 2)
    add_paragraph(doc, "The system is designed as an educational and research prototype. It does not replace official legal interpretation or professional advice. The UI exposes citations, confidence, visual evidence, and raw debug information so users can inspect the basis of an answer rather than treating model output as unquestionable authority.")
    doc.add_page_break()


def add_implementation_extended(doc: Document) -> None:
    add_heading(doc, "V. System Design and Implementation", 1)
    add_heading(doc, "1. AI Model Integration", 2)
    add_paragraph(doc, "The architecture contains four main layers: user interface, vision processing, retrieval, and answer generation. Streamlit provides the user interface. YOLOE detects traffic sign crops. CLIP-GmP-ViT-L-14 embeds sign crops. Qdrant retrieves similar LawDB signs. The LLM then generates a JSON answer from the question, visual evidence, retrieved sign metadata, and full article context.")
    add_figure_placeholder(
        doc,
        "Figure 3. Current system architecture",
        "Insert a block diagram of the final system architecture.",
        [
            "UI layer: Streamlit upload, question input, debug panel.",
            "Vision layer: YOLOE detector and sign crop folder.",
            "Retrieval layer: CLIP embedding model and Qdrant collection.",
            "Reasoning layer: LawDBArticleLookup and OpenAI-compatible LLM endpoint.",
        ],
        height_lines=6,
    )
    doc.add_page_break()

    add_named_table(
        doc,
        "Table 6. Runtime components and configuration",
        ["Component", "Latest GPU demo value", "Role"],
        [
            ["Streamlit", "src/ui/freeform_qa_demo.py", "Interactive demo for upload, analysis, citations, visual evidence, and debug output."],
            ["Detector", "YoloE", "Detects candidate traffic signs in user-uploaded road images."],
            ["Embedding", "CLIP-GmP-ViT-L-14", "Embeds sign crops and LawDB sign images for vector retrieval."],
            ["Vector store", "Qdrant on port 6333", "Stores LawDB sign embeddings and article payload metadata."],
            ["LLM endpoint", "Ollama OpenAI-compatible API on port 18989", "Runs gemma3:12b in the latest GPU demo for sign filtering and answer generation."],
            ["Retrieval limit", "LIMIT=5", "Retrieves top-k LawDB sign candidates per selected crop."],
            ["Answer temperature", "0.0", "Reduces randomness and improves JSON consistency."],
        ],
        widths=[1.8, 2.4, 3.4],
    )
    doc.add_page_break()

    add_heading(doc, "2. Data Flow and Processing", 2)
    add_paragraph(doc, "The runtime data flow begins with an uploaded image and a question. The image is normalized into a temporary working directory. Detected sign crops are saved so they can be displayed in the UI and reused by the retrieval step. For each selected sign, an image embedding is sent to Qdrant, and retrieved payloads are used to build the legal article context.")

    add_heading(doc, "3. LawDB Processing", 2)
    law_hash = hashlib.sha256(LAWDB.read_bytes()).hexdigest() if LAWDB.exists() else "N/A"
    add_paragraph(doc, "The LawDB pipeline preprocesses legal articles, extracts clean sign images, parses sign names and descriptions, embeds sign crops, and prepares one retrieval entry per detailed sign. The latest manually reviewed LawDB file is used by both ingest and article lookup.")
    add_bullets(doc, [
        "Current LawDB file: data/law_db/vlsp2025_law_new_preprocessed_with_imageprocessing.json.",
        f"Traceability hash: {law_hash}.",
        "Number of detailed sign entries prepared for Qdrant ingest: 489.",
        "Each Qdrant payload stores law_id, article_id, image_path, sign_name, and sign_description.",
    ])

    add_heading(doc, "4. Deployment Strategy", 2)
    add_paragraph(doc, "The project uses an OpenAI-compatible API client. This avoids hard-coding one serving backend. In the latest GPU run, Ollama served gemma3:12b on port 18989. Qdrant used port 6333 and Streamlit used port 8501. The same code can be pointed to another compatible backend by changing LLM_BASE_URL and LLM_MODEL.")

    add_heading(doc, "5. Scalability and Maintenance", 2)
    add_paragraph(doc, "The design keeps detection, retrieval, article lookup, prompting, and UI rendering in separate modules. This makes it possible to improve one layer at a time, such as replacing the detector, updating LawDB metadata, changing the LLM backend, or increasing the evaluated sample set.")

    add_heading(doc, "6. Streamlit Demo Design", 2)
    add_paragraph(doc, "The Streamlit UI is designed for a presentation setting. It should show the image, detected sign crops, the final answer, citations, visual evidence, and optional debug JSON. The interface should not hide the evidence because the main value of the project is not only answering, but also explaining where the answer came from.")
    add_figure_placeholder(
        doc,
        "Figure 5. Streamlit demo screen layout",
        "Insert a screenshot or mockup of the current Streamlit demo.",
        [
            "Left or top: uploaded traffic image.",
            "Middle: question input and Analyze button.",
            "Below: detected sign crops with relevant/not selected marker.",
            "Right or lower section: answer, citations expanders, visual evidence, debug panel.",
        ],
        height_lines=4,
    )

    add_heading(doc, "7. Main Modules", 2)
    add_named_table(
        doc,
        "Table 8. Main modules in the implementation",
        ["Module", "Responsibility"],
        [
            ["src/core/freeform_qa.py", "Single-image pipeline used by Streamlit: prepare image, detect signs, filter, retrieve, answer."],
            ["src/core/sub_task_2.py", "Free-form cited QA prompt assembly, JSON parsing, citation normalization, fallback handling."],
            ["src/core/filter_signs.py", "Question-conditioned relevant sign selection using original image and detected crops."],
            ["src/core/query_signs.py", "Embed selected sign crops and query Qdrant top-k LawDB sign payloads."],
            ["src/core/lawdb/article_lookup.py", "Resolve law_id and article_id into article title and text/content context."],
            ["src/deps/llm_client.py", "OpenAI-compatible client wrapper for local or hosted LLM endpoints."],
            ["src/ui/freeform_qa_demo.py", "Streamlit demo rendering upload, answer, citations, visual evidence, and debug output."],
        ],
        widths=[2.5, 5.0],
    )

    add_heading(doc, "8. Image QA Pipeline", 2)
    add_paragraph(doc, "At runtime, src/core/freeform_qa.py orchestrates the single-image pipeline used by Streamlit. The uploaded image is copied to a temporary working directory, signs are detected, selected signs are passed to Qdrant retrieval, article contexts are loaded, and src/core/sub_task_2.py builds the final cited QA prompt.")
    add_paragraph(doc, "Question-conditioned filtering is handled by src/core/filter_signs.py. If only one sign is detected, it is selected automatically. If multiple signs are present, the model receives the original image, crop sequence, and question, then returns a boolean relevance decision per crop.")
    add_paragraph(doc, "The final system intentionally avoids full query-time OCR and sign-group parsing. It still keeps lightweight variant disambiguation for ambiguous retrieved candidates, such as visually similar P.131a/P.131b/P.131c signs. This keeps the demo easier to explain while preserving the sign-variant handling needed for common traffic-sign families.")
    doc.add_page_break()

    add_heading(doc, "9. Prompting and Parsing", 2)
    add_paragraph(doc, "The prompt in src/prompts/answer_prompt.py enforces four constraints: use only supplied context, do not invent law articles, cite only articles that appear in the legal context, and provide visual evidence only from detected/retrieved signs. The parser in src/core/sub_task_2.py extracts JSON, validates allowed citations, normalizes visual evidence, and falls back to low confidence if the raw model response is malformed.")
    add_paragraph(doc, "This parser behavior is important for the user interface. A demo should not crash because a model returns a markdown fence, an incomplete JSON object, or extra text around JSON. The project therefore keeps raw_answer_response for diagnosis and still returns a safe low-confidence answer when the schema cannot be fully trusted.")

    add_heading(doc, "10. Local GPU LLM Serving", 2)
    add_paragraph(doc, "The project uses an OpenAI-compatible API client. This avoids hard-coding one serving backend. In the latest GPU run, Ollama served gemma3:12b on port 18989. Qdrant used port 6333 and Streamlit used port 8501. The same code can be pointed to another compatible backend by changing LLM_BASE_URL and LLM_MODEL.")
    add_bullets(doc, [
        "LLM_BASE_URL=http://localhost:18989/v1",
        "LLM_MODEL=gemma3:12b",
        "QDRANT_URI=http://localhost:6333",
        "IMAGE_QUERY_DETECTION_MODEL=YoloE",
        "EMBEDDING_NAME=CLIP-GmP-ViT-L-14",
        "LIMIT=5",
        "LLM_TEMPERATURE=0.0",
    ])
    add_paragraph(doc, "This setup is enough for a defense demo: it shows that the project can run on a rented GPU without changing the codebase, and it also keeps deployment understandable for reviewers.")
    doc.add_page_break()


def add_evaluation_extended(doc: Document) -> None:
    current = load_metrics(METRICS_CURRENT)

    add_heading(doc, "VI. Results and Discussion", 1)
    add_heading(doc, "1. Results and Analysis", 2)
    add_heading(doc, "1.1 Evaluation Setup", 3)
    add_paragraph(doc, "The final reported metrics are measured on a 20-sample sign-centric training slice using --begin-idx 39 --num-samples 20 --judge. This slice is more representative for traffic-sign retrieval and visual grounding than the first 20 samples, which contain many general legal questions.")
    add_paragraph(doc, "The judge mode uses the configured LLM to evaluate answer correctness, evidence faithfulness, and hallucination. Retrieval and citation metrics are computed programmatically from sample relevant_articles, returned citations, and retrieved payloads. This mixed approach is appropriate because some metrics require semantic judgment while others can be checked from IDs and JSON fields.")

    add_named_table(
        doc,
        "Table 9. Evaluation metrics",
        ["Metric", "Purpose"],
        [
            ["Answer correctness", "Whether the free-form answer matches the benchmark ground-truth answer meaning."],
            ["Citation accuracy / precision / recall", "Whether returned law_id and article_id match gold relevant articles."],
            ["Evidence faithfulness", "Whether citations and reasoning support the answer."],
            ["Visual grounding accuracy proxy", "Whether selected signs retrieve gold-related payloads at top-1."],
            ["Retrieval Recall@1/3/5", "Whether gold articles appear in top-k retrieved payloads for selected signs."],
            ["JSON validity", "Whether raw_answer_response is parseable strict JSON with required fields."],
            ["Hallucination rate", "Whether the answer invents unsupported information."],
            ["Average latency", "Mean end-to-end sample processing time."],
        ],
        widths=[2.4, 5.2],
    )
    doc.add_page_break()

    add_heading(doc, "1.2 Final Evaluated System Results", 3)
    add_figure_placeholder(
        doc,
        "Figure 6. Metric profile for the final evaluated system",
        "Insert a compact bar chart or radar chart for the final evaluated system metrics.",
        [
            "Show answer correctness 0.80, citation accuracy 0.65, evidence faithfulness 1.00.",
            "Show visual grounding proxy 0.575, JSON validity 1.00, hallucination 0.00.",
            "Add a small note: evaluated on 20 sign-centric samples with LLM judge.",
        ],
        height_lines=5,
    )
    add_named_table(
        doc,
        "Table 10. Final evaluated system metric summary",
        ["Metric", "Value"],
        [
            ["Answer correctness", f"{current.get('answer_correctness', 0):.2f}"],
            ["Citation accuracy", f"{current.get('citation_accuracy', 0):.2f}"],
            ["Citation precision", f"{current.get('citation_precision', 0):.2f}"],
            ["Citation recall", f"{current.get('citation_recall', 0):.2f}"],
            ["Evidence faithfulness", f"{current.get('evidence_faithfulness', 0):.2f}"],
            ["Visual grounding accuracy proxy", f"{current.get('visual_grounding_accuracy_proxy', 0):.3f}"],
            ["Retrieval Recall@1 / @3 / @5", f"{current.get('retrieval_recall_at_1', 0):.3f} / {current.get('retrieval_recall_at_3', 0):.3f} / {current.get('retrieval_recall_at_5', 0):.3f}"],
            ["JSON validity", f"{current.get('json_validity', 0):.2f}"],
            ["Hallucination rate", f"{current.get('hallucination', 0):.2f}"],
            ["Average latency", f"{current.get('latency', 0):.2f}s"],
        ],
        widths=[3.3, 2.1],
    )
    add_paragraph(doc, "The final evaluated system is appropriate for presentation because it reaches 0.80 answer correctness, 0.65 citation accuracy, 1.00 JSON validity, zero hallucination under the judge metric, and approximately 17.20 seconds average latency per sample.")

    add_heading(doc, "2. Discussion", 2)
    add_paragraph(doc, "The result should be interpreted as a practical prototype metric rather than a final benchmark claim. The evaluated slice is small but useful because it contains sign-centric questions where visual grounding matters. The final system is balanced: it answers with traceable citations, avoids unsupported legal statements in the judged slice, and remains usable for a live demo.")
    add_paragraph(doc, "For presentation, the most important metrics are answer correctness, citation quality, hallucination rate, JSON validity, and latency. A demo system should answer correctly, cite traceable evidence, avoid unsupported legal statements, and finish within a reasonable time.")
    doc.add_page_break()


def add_deployment_extended(doc: Document) -> None:
    add_heading(doc, "3. Recommendations", 2)
    add_heading(doc, "3.1 GPU Deployment Topology", 3)
    add_paragraph(doc, "The latest tested deployment used a rented RTX 4090 GPU server. The project folder was synchronized to the GPU machine, dependencies were installed in a virtual environment, Qdrant was started, Ollama served gemma3:12b through an OpenAI-compatible endpoint, and Streamlit was exposed through the provider's mapped public port.")
    add_figure_placeholder(
        doc,
        "Figure 7. GPU demo deployment topology",
        "Insert a deployment diagram showing local browser, public GPU port mapping, Streamlit, Qdrant, Ollama, and GPU.",
        [
            "Browser accesses public Streamlit URL.",
            "Streamlit calls local Python pipeline on GPU host.",
            "Pipeline calls Qdrant at 6333 and Ollama at 18989.",
            "Ollama uses RTX GPU for gemma3:12b inference.",
        ],
        height_lines=5,
    )

    add_heading(doc, "3.2 Demonstration Checklist", 3)
    add_named_table(
        doc,
        "Table 12. Demonstration readiness checklist",
        ["Check", "Expected result"],
        [
            ["Qdrant collection", "Collection lawdb_processed_by_imageprocessing_and_clip-gmp-vit-l-14 exists."],
            ["LawDB file", "Manual LawDB hash matches local verified file."],
            ["Ollama model", "OpenAI-compatible /v1/models returns gemma3:12b."],
            ["Streamlit", "Public URL loads the demo page."],
            ["Warm-up", "Detector, embedding model, and LLM are loaded before presentation."],
            ["Demo sample", "At least two tested images are prepared in case school network is unstable."],
            ["Debug mode", "Can be enabled to show retrieved payloads, citations, and raw JSON if reviewers ask."],
        ],
        widths=[2.3, 5.2],
    )

    add_heading(doc, "4. Personal Reflection", 2)
    add_paragraph(doc, "The recommended presentation flow is simple: first explain the user problem, then show one image-question example, then open citations and visual evidence. Avoid starting with implementation details. The implementation can be discussed after reviewers understand the product behavior.")
    add_paragraph(doc, "If a question asks why the system uses retrieval rather than direct model answering, the answer is that legal QA needs traceability. Retrieval gives the model bounded evidence and gives the user visible citations, which is more suitable for an educational legal QA demo.")
    doc.add_page_break()


def add_discussion_conclusion_extended(doc: Document) -> None:
    add_heading(doc, "VII. Conclusion", 1)
    add_heading(doc, "1. Summary of Findings", 2)
    add_bullets(doc, [
        "Manual quality review of LawDB sign descriptions is valuable because retrieval quality depends heavily on sign_name and sign_description metadata.",
        "The final pipeline keeps the answer process traceable by exposing detected signs, retrieved evidence, citations, reasoning, confidence, and raw JSON when needed.",
        "The measured result is strong enough for a capstone demonstration while still leaving clear future work for broader evaluation and sign-family regression tests.",
        "Strict JSON output and parser fallback are necessary for Streamlit rendering and metric computation.",
    ])

    add_heading(doc, "2. Contributions and Reflections on the Project", 2)
    add_paragraph(doc, "This project delivers a working multimodal legal QA prototype for Vietnamese traffic signs. The system combines a training-free retrieval pipeline with a practical free-form cited QA demo. The implementation detects relevant signs, retrieves LawDB evidence, loads article context, and generates Vietnamese answers with citations and visual evidence.")
    add_paragraph(doc, "The final evaluated system maintains 0.80 answer correctness on the sign-centric slice, reaches 0.65 citation accuracy, keeps JSON validity at 1.00, reports zero hallucination under the judge metric, and averages 17.20 seconds per sample. These results make it a balanced choice between quality, explainability, and live-demo reliability.")
    add_paragraph(doc, "The project is not presented as a complete legal-advice system. Its contribution is a clear and testable prototype showing how traffic-sign detection, LawDB retrieval, citation grounding, and local GPU LLM serving can be combined into a usable educational QA experience.")

    add_heading(doc, "3. Limitations and Future Work", 2)
    add_named_table(
        doc,
        "Table 13. Risk and limitation register",
        ["Limitation", "Impact", "Mitigation"],
        [
            ["Small judged sample size", "Metrics may not represent the full dataset.", "Report the exact slice and keep JSON artifacts for traceability."],
            ["LLM judge variability", "Answer correctness and hallucination may shift with model behavior.", "Use temperature 0.0 and combine judge metrics with programmatic metrics."],
            ["Similar sign variants", "P.131a/P.131b/P.131c style variants can be confused.", "Continue manual LawDB review and add targeted retrieval tests for common variant families."],
            ["Legal reliability", "Generated answers may still be incomplete or overconfident.", "Show citations, confidence, raw JSON, and state that the demo is not official legal advice."],
            ["Network and GPU dependency", "Live demo can slow down if GPU rental or school network is unstable.", "Prepare the GPU in advance and keep Streamlit/Qdrant/Ollama warmed before presentation."],
        ],
        widths=[2.0, 2.4, 3.1],
    )

    add_heading(doc, "3.1 Future Work", 3)
    add_bullets(doc, [
        "Evaluate on a larger sign-centric and full training set with fixed seeds and archived raw outputs.",
        "Add targeted regression tests for confusing sign families and supplementary plates.",
        "Improve retrieval metadata without adding expensive query-time calls unless metrics justify the latency.",
        "Add optional demo cache for presentation images when network quality is poor.",
        "Experiment with other OpenAI-compatible local serving backends after the final demo setup is documented.",
    ])
    doc.add_page_break()

    add_heading(doc, "VIII. References", 1)
    refs = [
        "The Association for Vietnamese Language and Speech Processing. 2025. VLSP 2025 Multimodal Legal Question Answering on Traffic Sign Rules shared task.",
        "QCVN 41:2024/BGTVT. National Technical Regulation on Road Signs and Signals.",
        "Law 36/2024/QH15. Law on Road Traffic Order and Safety.",
        "Project repository artifacts: README.md, src/core/freeform_qa.py, src/core/sub_task_2.py, src/core/lawdb/article_lookup.py, and the archived final evaluation metrics JSON artifact.",
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Paragraph")
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.18)
        p.add_run("• ")
        p.add_run(ref)
    doc.add_page_break()


def add_appendix_extended(doc: Document) -> None:
    add_heading(doc, "IX. Appendices", 1)
    add_heading(doc, "A. Suggested Figures to Add Later", 2)
    add_paragraph(doc, "The report intentionally uses figure placeholders so the final team can add polished screenshots or diagrams after agreeing on the visual style. The following list describes what each placeholder should contain.")
    add_bullets(doc, [
        "Figure 1: user scenario from traffic sign observation to cited answer.",
        "Figure 2: simplified training-free multimodal legal QA pipeline.",
        "Figure 3: current architecture with Streamlit, detector, embedding, Qdrant, LawDB lookup, and LLM endpoint.",
        "Figure 4: LawDB processing workflow from raw legal data to Qdrant points.",
        "Figure 5: Streamlit demo screenshot or mockup.",
        "Figure 6: bar chart or radar chart of final evaluated system metrics.",
        "Figure 7: GPU deployment topology with public port mapping.",
    ])

    add_heading(doc, "B. Defense Notes", 2)
    add_paragraph(doc, "The safest technical story for defense is that the project is retrieval-grounded, evidence-aware, and deliberately training-free. The system does not claim to replace official legal interpretation. It demonstrates how a multimodal assistant can connect detected signs to legal articles and expose the evidence used to answer.")
    add_paragraph(doc, "When asked about performance, use the final evaluated system numbers and state the measured subset clearly. When asked about limitations, mention small judged sample size, similar sign variants, and reliance on local GPU inference.")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_document(doc)
    add_cover(doc)
    add_front_matter_extended(doc)
    add_project_intro_extended(doc)
    add_project_management_extended(doc)
    add_related_work_extended(doc)
    add_design_extended(doc)
    add_implementation_extended(doc)
    add_evaluation_extended(doc)
    add_deployment_extended(doc)
    add_discussion_conclusion_extended(doc)
    add_appendix_extended(doc)
    doc.save(OUT_DOCX)
    print(OUT_DOCX)


if __name__ == "__main__":
    main()
