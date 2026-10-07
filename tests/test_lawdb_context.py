from types import SimpleNamespace

from PIL import Image

from src.core.article_context import build_article_contexts
from src.core.lawdb.sign_metadata import build_sign_payload
from src.core.lawdb.sign_metadata import normalize_parsed_sign_info
from src.core.sub_task_2 import normalize_freeform_response
from src.core.query_signs import (
    DEFAULT_ARTICLE,
    query_relevant_signs,
    retrieved_article_references,
)
from src.prompts.parse_signs_prompt import PARSE_SIGNS_PROMPT


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeArticleLookup:
    def __init__(self, article_ids):
        self.articles = {
            (LAW_ID, article_id): {
                "law_id": LAW_ID,
                "article_id": article_id,
                "title": f"Điều {article_id}",
                "text": f"Nội dung điều {article_id}",
            }
            for article_id in article_ids
        }

    def get_article(self, law_id, article_id):
        return self.articles.get((law_id, article_id))


def payload(article_id, sign_code):
    return {
        "law_id": LAW_ID,
        "article_id": article_id,
        "sign_code": sign_code,
        "sign_name": f"Biển {sign_code}",
    }


def test_one_sign_points_to_correct_article():
    references = retrieved_article_references([[payload("B.10", "P.110a")]])
    contexts = build_article_contexts(
        references,
        FakeArticleLookup(["B.10"]),
        max_articles=8,
    )

    assert len(contexts) == 1
    assert contexts[0]["article_id"] == "B.10"
    assert contexts[0]["context_source"] == "retrieval"
    assert contexts[0]["retrieval_rank"] == 1
    assert contexts[0]["related_sign_codes"] == ["P.110a"]


def test_multiple_signs_same_article_are_merged():
    references = retrieved_article_references(
        [
            [payload("B.10", "P.110a")],
            [payload("B.10", "P.110b")],
        ]
    )

    assert len(references) == 2
    assert references[0]["related_sign_codes"] == ["P.110a", "P.110b"]
    assert references[1]["source"] == "rule"


def test_default_article_mapping_is_related_and_records_why():
    references = retrieved_article_references([[payload("B.11", "P.131b")]])

    assert [(item["article_id"], item["source"]) for item in references] == [
        ("B.11", "retrieval"),
        ("22", "rule"),
    ]
    assert "DEFAULT_ARTICLES" in references[1]["reason"]


def test_shared_default_related_article_merges_all_sign_codes():
    references = retrieved_article_references(
        [
            [payload("B.10", "P.110a")],
            [payload("B.11", "P.110b")],
        ]
    )
    related = [item for item in references if item["article_id"] == "22"]

    assert len(related) == 1
    assert related[0]["related_sign_codes"] == ["P.110a", "P.110b"]


def test_unmapped_valid_retrieval_does_not_add_unrelated_default():
    references = retrieved_article_references([[payload("35", "P.100")]])

    assert [item["article_id"] for item in references] == ["35"]


def test_one_sign_with_multiple_candidates_preserves_ranks():
    references = retrieved_article_references(
        [[payload("B.10", "P.110a"), payload("B.11", "P.111a")]]
    )
    contexts = build_article_contexts(
        references,
        FakeArticleLookup(["B.10", "B.11"]),
        max_articles=8,
    )

    assert [item["article_id"] for item in contexts] == ["B.10", "B.11"]
    assert [item["retrieval_rank"] for item in contexts] == [1, 2]


def test_no_retrieval_result_uses_only_documented_fallback(tmp_path):
    image_path = tmp_path / "crop.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_path)

    class Embedder:
        def embed_image(self, image):
            return [0.0]

    class VectorStore:
        def query(self, collection_name, embedding, top_k):
            return []

    sample = {
        "detected_signs": [{"image_name": image_path.name, "is_chosen": True}]
    }
    result = query_relevant_signs(
        sample=sample,
        sign_dir=tmp_path,
        embedding_model=Embedder(),
        vector_store=VectorStore(),
        collection_name="test",
        set_name="train",
    )

    assert result["detected_signs"][0]["retrieved_payloads"] == []
    assert result["predicted_articles"] == [
        {
            **DEFAULT_ARTICLE,
            "source": "fallback_default_article",
            "reason": "No valid article was returned by sign retrieval.",
            "retrieval_rank": None,
            "related_sign_codes": [],
        }
    ]


def test_unknown_only_retrieval_skips_canonical_yolo_detection(tmp_path):
    image_path = tmp_path / "crop.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_path)

    class Embedder:
        called = False

        def embed_image(self, image):
            self.called = True
            return [0.0]

    class VectorStore:
        def query(self, *args, **kwargs):
            raise AssertionError("known YOLO signs must bypass retrieval")

    embedder = Embedder()
    sample = {
        "detected_signs": [
            {
                "image_name": image_path.name,
                "is_chosen": True,
                "identity_status": "identified",
                "sign_code": "P.131b",
            }
        ]
    }
    result = query_relevant_signs(
        sample=sample,
        sign_dir=tmp_path,
        embedding_model=embedder,
        vector_store=VectorStore(),
        collection_name="test",
        set_name="train",
        unknown_only=True,
    )

    assert embedder.called is False
    assert "predicted_articles" not in result


def test_missing_lawdb_article_is_excluded_from_context():
    contexts = build_article_contexts(
        [payload("missing", "P.999")],
        FakeArticleLookup([]),
        max_articles=8,
    )

    assert contexts == []


def test_duplicate_context_prefers_retrieval_and_merges_sign_codes():
    contexts = build_article_contexts(
        [
            {
                **payload("31", "P.131a"),
                "context_source": "rule",
            },
            {
                **payload("31", "P.131b"),
                "context_source": "retrieval",
                "retrieval_rank": 2,
            },
        ],
        FakeArticleLookup(["31"]),
        max_articles=8,
    )

    assert len(contexts) == 1
    assert contexts[0]["context_source"] == "retrieval"
    assert contexts[0]["retrieval_rank"] == 2
    assert contexts[0]["related_sign_codes"] == ["P.131a", "P.131b"]


def test_context_over_limit_is_capped_in_retrieval_order():
    article_ids = [str(index) for index in range(12)]
    candidates = [
        {
            **payload(article_id, f"P.{100 + index}"),
            "retrieval_rank": index + 1,
        }
        for index, article_id in enumerate(article_ids)
    ]
    contexts = build_article_contexts(
        candidates,
        FakeArticleLookup(article_ids),
        max_articles=8,
    )

    assert len(contexts) == 8
    assert [item["article_id"] for item in contexts] == article_ids[:8]


def test_ingest_payload_normalizes_sign_metadata():
    result = build_sign_payload(
        law_id=LAW_ID,
        article_id="B.11",
        image_path="signs/p131b.jpg",
        sign_info={
            "name": "Biển số P,131B - Cấm đỗ xe ngày lẻ",
            "description": "Cấm đỗ xe vào ngày lẻ.",
            "legal_conditions": {"time_condition": "ngày lẻ"},
        },
    )

    assert result["article_uid"] == f"{LAW_ID}#B.11"
    assert result["parent_article_id"] == "B.11"
    assert result["sign_code"] == "P.131b"
    assert result["sign_family"] == "P.131"
    assert result["legal_conditions"]["time_condition"] == "ngày lẻ"


def test_parse_prompt_requires_empty_code_when_context_is_uncertain():
    assert '"sign_code": "P.131b"' in PARSE_SIGNS_PROMPT
    assert 'sign_code phải là chuỗi rỗng ""' in PARSE_SIGNS_PROMPT
    assert '"legal_conditions"' in PARSE_SIGNS_PROMPT


def test_parser_rejects_sign_code_not_present_in_article_text():
    parsed = normalize_parsed_sign_info(
        {
            "sign_code": "P.131b",
            "sign_family": "P.131",
            "name": "Biển P.131b",
            "description": "Cấm đỗ ngày lẻ.",
        },
        {"title": "Biển cấm xe", "text": "Quy định về biển P.131a."},
    )

    assert parsed["sign_code"] == ""
    assert parsed["sign_family"] == ""
    assert parsed["legal_conditions"] == {
        "time_condition": "",
        "vehicle_scope": "",
        "direction_scope": "",
    }


def test_citations_are_whitelisted_to_context_articles():
    contexts = [
        {
            "law_id": LAW_ID,
            "article_id": "B.10",
            "title": "Điều B.10",
            "text": "Căn cứ pháp lý.",
            "context_source": "retrieval",
            "retrieval_rank": 1,
            "related_sign_codes": ["P.110a"],
        }
    ]
    response = (
        '{"answer":"Câu trả lời", "confidence":"high", "citations": ['
        '{"law_id":"' + LAW_ID + '","article_id":"B.10","evidence":"Căn cứ"},'
        '{"law_id":"' + LAW_ID + '","article_id":"99","evidence":"Bịa"}]}'
    )

    result = normalize_freeform_response(response, contexts, [])

    assert [(citation["law_id"], citation["article_id"])
            for citation in result["citations"]] == [(LAW_ID, "B.10")]


def test_llm_cannot_promote_unknown_sign_permission_to_allowed():
    symbolic_context = {
        "query_intent": {"intents": [{"kind": "CHECK_PERMISSION", "target": {"kind": "scene"}}]},
        "facts": {"sign_observations": [{"sign_code": None, "identity_status": "ambiguous"}]},
        "symbolic_decision": {
            "decision": "UNKNOWN",
            "conclusion": "Chưa có rule đã được rà soát cho phạm vi này.",
        },
    }
    result = normalize_freeform_response(
        '{"answer":"Được phép.","confidence":"high"}',
        [{"law_id": LAW_ID, "article_id": "B.10", "title": "Điều B.10", "text": "text"}],
        [],
        symbolic_context,
    )

    assert "UNKNOWN" in result["freeform_answer"]
    assert "Được phép" not in result["freeform_answer"]
    assert result["confidence"] == "low"
    assert result["citations"] == []


def test_old_pipeline_is_not_overridden_for_non_sign_intents():
    symbolic_context = {
        "query_intent": {"intents": [{"kind": "GENERAL_LEGAL_QA"}]},
        "facts": {"sign_observations": []},
        "symbolic_decision": {"decision": "UNKNOWN"},
    }
    result = normalize_freeform_response(
        '{"answer":"Câu trả lời cũ.","confidence":"medium"}', [], [], symbolic_context,
    )

    assert result["freeform_answer"] == "Câu trả lời cũ."
