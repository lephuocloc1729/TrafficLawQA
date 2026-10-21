import json
from types import SimpleNamespace

from PIL import Image

from src.core.sub_task_2 import collect_retrieved_context, disambiguate_sign_payloads


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeArticleLookup:
    def get_article(self, law_id, article_id):
        return {
            "law_id": law_id,
            "article_id": article_id,
            "title": f"Article {article_id}",
            "text": f"Legal text {article_id}",
        }


def ten_candidates():
    return [
        {
            "law_id": "QCVN",
            "article_id": f"B.{index}",
            "sign_code": f"P.{100 + index}",
            "sign_name": f"Sign P.{100 + index}",
            "sign_description": f"Visual description {index}",
            "retrieval_rank": index,
            "score": 0.9 - index / 100,
        }
        for index in range(1, 11)
    ]


def test_llm_can_choose_any_of_ten_candidates_and_choice_is_first():
    llm = FakeLLM(
        '{"selected_index": 7, "confidence": "high", "reason": "Matches the arrow shape."}'
    )
    payloads = ten_candidates()

    selected, decision = disambiguate_sign_payloads(
        llm,
        {"question": "What does this sign mean?"},
        Image.new("RGB", (16, 16), "white"),
        payloads,
    )

    assert len(llm.calls) == 1
    candidate_prompt = llm.calls[0]["contents"][0]["content"][1]["text"]
    candidate_json = candidate_prompt.split(
        "RETRIEVED SIGN CANDIDATES (choose only by index)\n", 1,
    )[1]
    candidates = json.loads(candidate_json)
    assert len(candidates) == 10
    assert candidates[7]["sign_code"] == "P.108"
    assert decision["selected_index"] == 7
    assert decision["method"] == "llm_top10_sign_selection"
    assert decision["confidence"] == "high"
    assert selected[0]["sign_code"] == "P.108"


def test_llm_abstention_does_not_silently_choose_retrieval_top1():
    llm = FakeLLM('{"selected_index": null, "confidence": "low", "reason": "Blurry crop."}')
    payloads = ten_candidates()

    selected, decision = disambiguate_sign_payloads(
        llm,
        {"question": "What does this sign mean?"},
        Image.new("RGB", (16, 16), "white"),
        payloads,
    )

    assert selected == payloads
    assert decision["selected_index"] is None
    assert decision["confidence"] == "low"


def test_selected_candidate_alone_supplies_legal_context_and_top10_remain_auditable(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.sub_task_2.settings.traffic_sign_ocr_enabled", False)
    image_path = tmp_path / "sign.jpg"
    Image.new("RGB", (16, 16), "white").save(image_path)
    llm = FakeLLM(
        '{"selected_index": 6, "confidence": "high", "reason": "The symbol matches."}'
    )
    payloads = ten_candidates()
    sample = {
        "question": "What does this sign mean?",
        "detected_signs": [{
            "image_name": image_path.name,
            "is_chosen": True,
            "retrieved_payloads": payloads,
        }],
    }

    contexts, _, _, facts = collect_retrieved_context(
        sample=sample,
        sign_dir=tmp_path,
        embedding_model=None,
        vector_store=None,
        collection_name="test",
        article_lookup=FakeArticleLookup(),
        llm_client=llm,
        limit=10,
    )

    assert [item["article_id"] for item in contexts] == ["B.7"]
    observation = facts.sign_observations[0]
    assert observation.sign_code == "P.107"
    assert observation.identity_status == "identified"
    assert len(observation.candidates) == 10


def test_low_confidence_choice_does_not_supply_legal_article_context(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.sub_task_2.settings.traffic_sign_ocr_enabled", False)
    image_path = tmp_path / "sign.jpg"
    Image.new("RGB", (16, 16), "white").save(image_path)
    llm = FakeLLM(
        '{"selected_index": 6, "confidence": "low", "reason": "Ambiguous crop."}'
    )
    sample = {
        "question": "What does this sign mean?",
        "detected_signs": [{
            "image_name": image_path.name,
            "is_chosen": True,
            "retrieved_payloads": ten_candidates(),
        }],
    }

    contexts, _, _, facts = collect_retrieved_context(
        sample=sample,
        sign_dir=tmp_path,
        embedding_model=None,
        vector_store=None,
        collection_name="test",
        article_lookup=FakeArticleLookup(),
        llm_client=llm,
        limit=10,
    )

    assert contexts == []
    assert facts.sign_observations[0].identity_status == "ambiguous"


def test_unresolved_detection_retrieves_ten_candidates_before_llm_selection(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.sub_task_2.settings.traffic_sign_ocr_enabled", False)
    monkeypatch.setattr("src.core.sub_task_2.settings.enable_sign_retrieval_fallback", True)
    image_path = tmp_path / "sign.jpg"
    Image.new("RGB", (16, 16), "white").save(image_path)
    llm = FakeLLM(
        '{"selected_index": 2, "confidence": "high", "reason": "Visual match."}'
    )

    class Embedder:
        def embed_image(self, image):
            return [0.1, 0.2]

    class VectorStore:
        requested_top_k = None

        def query(self, collection_name, embedding, top_k):
            self.requested_top_k = top_k
            return [
                SimpleNamespace(payload=payload, score=payload["score"])
                for payload in ten_candidates()
            ]

    vector_store = VectorStore()
    sample = {
        "question": "What does this sign mean?",
        "detected_signs": [{
            "image_name": image_path.name,
            "is_chosen": True,
            "identity_status": "ambiguous",
        }],
    }

    contexts, _, _, facts = collect_retrieved_context(
        sample=sample,
        sign_dir=tmp_path,
        embedding_model=Embedder(),
        vector_store=vector_store,
        collection_name="test",
        article_lookup=FakeArticleLookup(),
        llm_client=llm,
        limit=10,
    )

    assert vector_store.requested_top_k == 10
    assert len(llm.calls) == 1
    assert [item["article_id"] for item in contexts] == ["B.3"]
    assert facts.sign_observations[0].sign_code == "P.103"
