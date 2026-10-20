import json

from PIL import Image

from src.core.baseline1_bm25_llm import (
    BM25Index,
    LegalUnit,
    answer_image_question,
    load_legal_units,
)


def test_load_lawdb_keeps_article_clause_and_point_ids(tmp_path):
    lawdb_path = tmp_path / "lawdb.json"
    lawdb_path.write_text(
        json.dumps(
            [
                {
                    "id": "QCVN 41:2024/BGTVT",
                    "articles": [
                        {
                            "id": "B.31",
                            "title": "Biển cấm dừng xe",
                            "text": "1. Cấm dừng xe.\n   a) Áp dụng với xe cơ giới.\n   b) Ngoại lệ xe ưu tiên.",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    units = load_legal_units(lawdb_path)

    assert [(unit.clause_id, unit.point_id) for unit in units] == [
        ("1", "a"),
        ("1", "b"),
    ]
    assert units[0].law_id == "QCVN 41:2024/BGTVT"
    assert units[0].article_id == "B.31"


def test_bm25_ranks_matching_vietnamese_unit_and_preserves_citation():
    index = BM25Index(
        [
            LegalUnit("L1", "1", "Cấm dừng xe", "Cấm dừng và đỗ xe."),
            LegalUnit("L2", "2", "Tốc độ", "Giới hạn tốc độ tối đa."),
        ]
    )

    results = index.search("Có được dừng xe không?", limit=2)

    assert results[0]["law_id"] == "L1"
    assert results[0]["article_id"] == "1"
    assert results[0]["score"] > 0
    assert len(results) == 1  # Zero-score passages are excluded from context.


def test_answer_only_returns_retrieved_citations_and_uses_text_only(tmp_path):
    class Detector:
        def crop_signs(self, image_path):
            return [Image.new("RGB", (10, 10))]

    class OCR:
        def read(self, image):
            return ["P.131a"], "available"

    class LLM:
        def generate_content(self, contents, **kwargs):
            prompt = contents[0]["content"]
            assert "P.131a" in prompt
            assert "L1 | Điều B.31" in prompt
            assert "image_url" not in prompt
            return "Chọn B. Điều khoản này cấm dừng xe [1]."

    image_path = tmp_path / "scene.png"
    Image.new("RGB", (20, 20)).save(image_path)
    index = BM25Index(
        [
            LegalUnit("L1", "B.31", "Cấm dừng xe", "Cấm dừng xe tại vị trí có biển P.131a."),
            LegalUnit("L2", "B.32", "Tốc độ", "Giới hạn tốc độ tối đa."),
        ]
    )
    resources = type(
        "Resources",
        (),
        {"detector": Detector(), "ocr": OCR(), "index": index, "llm": LLM(), "max_tokens": 1000},
    )()

    result = answer_image_question(
        image_path,
        "Xe máy có được dừng tại đây không?",
        choices=["A. Có", "B. Không"],
        resources=resources,
        top_k=1,
    )

    assert result["answer"].startswith("Chọn B")
    assert [citation["article_id"] for citation in result["citations"]] == ["B.31"]
    assert result["detected_signs"][0]["identity_status"] == "unidentified"
