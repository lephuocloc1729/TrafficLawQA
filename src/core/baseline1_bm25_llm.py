"""Baseline 1: fixed sign detection/OCR, BM25 over LawDB, then LLM answer."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.utils import load_json
from src.deps.llm_client import LLMClient


TOKEN_RE = re.compile(r"[\w]+(?:[./-][\w]+)*", re.UNICODE)
CLAUSE_RE = re.compile(r"(?m)^\s*(\d{1,2})\s*[.)]\s+")
POINT_RE = re.compile(r"(?m)^\s*([a-zđ])\s*[.)]\s+")
MAX_UNIT_CHARS = 5000


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", str(text or "")).lower()


def tokenize(text: str) -> list[str]:
    """Small dependency-free tokenizer; preserve Vietnamese accents and sign codes."""
    return TOKEN_RE.findall(_normalize(text))


@dataclass(frozen=True)
class LegalUnit:
    law_id: str
    article_id: str
    title: str
    text: str
    clause_id: str = ""
    point_id: str = ""

    def citation(self) -> dict[str, str]:
        return {
            "law_id": self.law_id,
            "article_id": self.article_id,
            "clause_id": self.clause_id,
            "point_id": self.point_id,
            "title": self.title,
            "evidence": self.text,
        }


def _split_structured_text(text: str) -> list[tuple[str, str, str]]:
    """Split visibly numbered clauses/points, falling back to the whole article."""
    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    clauses = list(CLAUSE_RE.finditer(normalized))
    if not clauses:
        return [("", "", normalized)] if normalized else []

    units: list[tuple[str, str, str]] = []
    for index, match in enumerate(clauses):
        end = clauses[index + 1].start() if index + 1 < len(clauses) else len(normalized)
        clause_text = normalized[match.start():end].strip()
        clause_id = match.group(1)
        points = list(POINT_RE.finditer(clause_text))
        if points:
            for point_index, point in enumerate(points):
                point_end = points[point_index + 1].start() if point_index + 1 < len(points) else len(clause_text)
                units.append((clause_id, point.group(1), clause_text[point.start():point_end].strip()))
        else:
            units.append((clause_id, "", clause_text))
    return units


def load_legal_units(path: str | Path) -> list[LegalUnit]:
    """Load LawDB article records and retain IDs/titles through clause splitting."""
    payload = load_json(str(path))
    if isinstance(payload, dict):
        payload = [
            {"id": law_id, "articles": articles}
            for law_id, articles in payload.items()
        ]
    if not isinstance(payload, list):
        raise ValueError("LawDB must be a list or a mapping of laws to articles")
    units: list[LegalUnit] = []
    for law in payload:
        if not isinstance(law, dict):
            continue
        law_id = str(law.get("id") or law.get("law_id") or "").strip()
        for article in law.get("articles", []):
            if not isinstance(article, dict):
                continue
            article_id = str(article.get("id") or article.get("article_id") or "").strip()
            title = str(article.get("title") or "").strip()
            body = str(article.get("text") or article.get("content") or article.get("body") or "").strip()
            if not law_id or not article_id or not body:
                continue
            for clause_id, point_id, text in _split_structured_text(body):
                if len(text) > MAX_UNIT_CHARS:
                    text = text[:MAX_UNIT_CHARS]
                units.append(LegalUnit(law_id, article_id, title, text, clause_id, point_id))
    return units


class BM25Index:
    """A compact BM25Okapi-style index over legal units."""

    def __init__(self, units: list[LegalUnit], k1: float = 1.5, b: float = 0.75):
        if not units:
            raise ValueError("No legal units found in LawDB")
        self.units = units
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(f"{u.title} {u.text}") for u in units]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokens]
        self.avg_doc_len = sum(map(len, self.tokens)) / len(self.tokens) or 1.0
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log(1 + (len(units) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms or limit < 1:
            return []
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            doc_len = len(self.tokens[index])
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {**self.units[index].citation(), "score": round(score, 6)}
            for score, index in scored[:limit]
        ]


class OptionalOCR:
    """Tesseract OCR adapter. Install pytesseract and the Vietnamese language data."""

    def __init__(self, language: str = "vie+eng"):
        self.language = language
        try:
            import pytesseract
            self._reader = pytesseract
        except ImportError:
            self._reader = None

    def read(self, image: Any) -> tuple[list[str], str]:
        if self._reader is None:
            return [], "unavailable: install the optional pytesseract dependency and Tesseract OCR"
        try:
            text = self._reader.image_to_string(image, lang=self.language)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return lines, "available"
        except Exception as exc:
            return [], f"error: {exc}"


@dataclass
class BaselineResources:
    detector: Any
    llm: LLMClient
    index: BM25Index
    ocr: OptionalOCR
    max_tokens: int = 1000


def create_resources(lawdb_path: str | Path | None = None, device: str = "cpu") -> BaselineResources:
    from src.settings import settings

    # Load vision dependencies only when the image pipeline is actually run.
    from src.deps.detection import SignExtractor

    api_keys = [key.strip() for key in (settings.llm_api_keys or "").split(",") if key.strip()]
    if not api_keys:
        raise ValueError("Set LLM_API_KEYS in .env before running Baseline 1")
    return BaselineResources(
        detector=SignExtractor(detector=settings.image_query_detection_model, device=device),
        llm=LLMClient(settings.llm_base_url, api_keys, settings.llm_model),
        index=BM25Index(load_legal_units(lawdb_path or settings.lawdb_json_path)),
        ocr=OptionalOCR(),
        max_tokens=min(settings.llm_max_new_tokens, 1000),
    )


def _read_image(path: Path) -> Any:
    from PIL import Image
    return Image.open(path).convert("RGB")


def answer_image_question(
    image_path: str | Path,
    question: str,
    choices: str | list[str] | None = None,
    resources: BaselineResources | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Answer using fixed detector crops, OCR, BM25, and a text-only LLM prompt."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    resources = resources or create_resources()
    image = _read_image(image_path)
    crops = resources.detector.crop_signs(str(image_path))

    # This branch's detector localizes broad sign shapes only; do not claim it
    # recognized a sign code. OCR text and the question remain usable evidence.
    ocr_texts: list[str] = []
    ocr_statuses: list[str] = []
    for crop in crops:
        lines, status = resources.ocr.read(crop)
        ocr_texts.extend(lines)
        ocr_statuses.append(status)
    if not crops:
        lines, status = resources.ocr.read(image)
        ocr_texts.extend(lines)
        ocr_statuses.append(status)

    detected_signs = [
        {"index": i, "identity_status": "unidentified", "description": "Biển được phát hiện; detector hiện tại không trả mã hoặc tên biển."}
        for i, _ in enumerate(crops)
    ]
    query_parts = [question.strip()]
    if ocr_texts:
        query_parts.append("OCR: " + " ".join(dict.fromkeys(ocr_texts)))
    query = "\n".join(part for part in query_parts if part)
    retrieved = resources.index.search(query, limit=top_k)

    if not retrieved:
        return {
            "answer": "Chưa đủ thông tin để trả lời: BM25 không tìm được điều khoản có từ khóa phù hợp.",
            "citations": [], "retrieved_units": [], "query": query,
            "detected_signs": detected_signs, "ocr_texts": ocr_texts,
            "ocr_status": ocr_statuses, "recognition_limit": "Detector chỉ định vị vùng biển, chưa nhận dạng mã/ý nghĩa biển.",
        }

    choice_text = "\n".join(choices) if isinstance(choices, list) else (choices or "")
    evidence = "\n\n".join(
        f"[{i}] {unit['law_id']} | Điều {unit['article_id']}"
        f"{(' | Khoản ' + unit['clause_id']) if unit['clause_id'] else ''}"
        f"{(' | Điểm ' + unit['point_id']) if unit['point_id'] else ''}"
        f" | {unit['title']}\n{unit['evidence']}"
        for i, unit in enumerate(retrieved, start=1)
    )
    signs_text = "\n".join(
        f"- Biển #{sign['index']}: chưa nhận dạng được mã hoặc ý nghĩa bằng detector cố định."
        for sign in detected_signs
    ) or "- Không phát hiện được vùng biển."
    prompt = (
        "Bạn là trợ lý hỏi đáp pháp luật giao thông Việt Nam. Trả lời ngắn gọn, tự nhiên bằng tiếng Việt. "
        "Chỉ dựa trên câu hỏi, lựa chọn, thông tin nhận diện/OCR và các đơn vị pháp lý bên dưới. "
        "Không suy diễn vượt quá bằng chứng. Chỉ được dẫn nguồn bằng số [n] có trong danh sách điều khoản; "
        "không bịa luật hoặc điều khoản. Nếu không đủ căn cứ, nói rõ chưa đủ thông tin. "
        "Nếu có lựa chọn, nêu chữ cái lựa chọn và giải thích ngắn.\n\n"
        f"CÂU HỎI:\n{question}\n\nLỰA CHỌN:\n{choice_text or '(không có)'}\n\n"
        f"BIỂN ĐÃ PHÁT HIỆN:\n{signs_text}\n\nOCR:\n{'; '.join(ocr_texts) or '(không đọc được chữ)'}\n\n"
        f"ĐIỀU KHOẢN BM25 TRUY XUẤT (chỉ các mục này được phép trích dẫn):\n{evidence}"
    )
    answer = resources.llm.generate_content(
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=resources.max_tokens,
    ) or "Chưa đủ thông tin trong điều khoản truy xuất để trả lời chắc chắn."

    # Citations come from retrieval itself, never from unvalidated model output.
    cited_indices = {int(match) for match in re.findall(r"\[(\d+)\]", answer)}
    citation_fields = (
        "law_id", "article_id", "clause_id", "point_id", "title", "evidence"
    )
    citations = [
        {key: unit[key] for key in citation_fields}
        for i, unit in enumerate(retrieved, start=1)
        if i in cited_indices
    ]
    return {
        "answer": answer.strip(), "citations": citations,
        "retrieved_units": retrieved, "query": query,
        "detected_signs": detected_signs, "ocr_texts": ocr_texts,
        "ocr_status": ocr_statuses,
        "recognition_limit": "Detector chỉ định vị vùng biển, chưa nhận dạng mã/ý nghĩa biển.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline 1: fixed detector/OCR + BM25 + LLM")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--choices", help="Optional choices, one per line")
    parser.add_argument("--lawdb", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    resources = create_resources(args.lawdb, args.device)
    result = answer_image_question(
        args.image, args.question,
        choices=args.choices.splitlines() if args.choices else None,
        resources=resources, top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
