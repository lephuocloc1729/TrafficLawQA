from src.eval.error_analysis import analyze_rows, to_markdown


def article(article_id: str) -> dict[str, str]:
    return {"law_id": "QCVN", "article_id": article_id}


def test_full_trace_classifies_rank_variant_symbolic_citation_answer_and_json():
    row = {
        "id": "case-1",
        "input": {
            "question": "Xe có được đỗ ngày lẻ không?",
            "relevant_articles": [article("31")],
            "gold_sign_code": "P.131b",
            "gold_symbolic_decision": "PROHIBITED",
        },
        "output": {
            "detected_signs": [
                {
                    "image_name": "crop.png",
                    "sign_name": "Biển P.131c",
                    "is_chosen": True,
                    "retrieved_payloads": [
                        article("30"),
                        article("31"),
                    ],
                }
            ],
            "symbolic_facts": [{"sign_code": "P.131c"}],
            "symbolic_decision": {"decision": "CONDITIONAL"},
            "citations": [article("30")],
            "freeform_answer": "Được phép.",
        },
        "retrieval_recall_at_1": 0.0,
        "retrieval_recall_at_5": 1.0,
        "answer_correctness": 0.0,
        "json_validity": 0.0,
    }

    report = analyze_rows([row])
    analyzed = report["rows"][0]

    assert analyzed["gold_articles"] == ["QCVN#31"]
    assert analyzed["retrieved_top1"] == ["QCVN#30"]
    assert analyzed["retrieved_top3"] == ["QCVN#30", "QCVN#31"]
    assert analyzed["retrieved_top5"] == ["QCVN#30", "QCVN#31"]
    assert analyzed["sign_code"] == "P.131c"
    assert set(analyzed["error_type"]) == {
        "WRONG_SIGN_VARIANT",
        "RIGHT_ARTICLE_LOW_RANK",
        "WRONG_SYMBOLIC_RULE",
        "WRONG_CITATION",
        "WRONG_FINAL_ANSWER",
        "INVALID_JSON",
    }


def test_no_detection_is_not_mislabelled_as_wrong_article():
    report = analyze_rows(
        [
            {
                "input": {"relevant_articles": [article("31")]},
                "output": {"detected_signs": []},
                "retrieval_recall_at_5": 0,
            }
        ]
    )

    assert report["rows"][0]["error_type"] == ["NO_DETECTION"]


def test_legacy_metrics_allow_error_classification_but_not_fabricated_article_lists():
    report = analyze_rows(
        [
            {
                "id": "low-rank",
                "input": {"relevant_articles": [article("31")]},
                "output": {"freeform_answer": "answer"},
                "retrieval_recall_at_1": 0,
                "retrieval_recall_at_5": 1,
            },
            {
                "id": "missing",
                "input": {"relevant_articles": [article("32")]},
                "output": {"freeform_answer": "answer"},
                "retrieval_recall_at_1": 0,
                "retrieval_recall_at_5": 0,
            },
        ]
    )

    low_rank, missing = report["rows"]
    assert low_rank["retrieved_top1"] is None
    assert low_rank["retrieved_top5"] is None
    assert low_rank["error_type"] == ["RIGHT_ARTICLE_LOW_RANK"]
    assert missing["error_type"] == ["WRONG_ARTICLE"]
    assert report["summary"]["recommendations"][0]["triggered"] is True
    assert report["summary"]["trace_coverage"]["retrieval_payload_trace_samples"] == 0


def test_unknown_trace_is_not_treated_as_zero_recall_and_markdown_lists_errors():
    report = analyze_rows(
        [{"input": {"question": "Q?", "relevant_articles": [article("31")]}, "output": {}}]
    )
    row = report["rows"][0]

    assert row["retrieved_top5"] is None
    assert row["error_type"] == []
    assert "RIGHT_ARTICLE_LOW_RANK" in to_markdown(report)


def test_wrong_answer_with_complete_retrieval_recommends_downstream_work():
    report = analyze_rows(
        [
            {
                "input": {"relevant_articles": [article("31")]},
                "output": {"freeform_answer": "wrong"},
                "retrieval_recall_at_5": 1,
                "answer_correctness": 0,
            }
        ]
    )

    signal = report["summary"]["recommendations"][-1]
    assert signal["signal"] == "RETRIEVAL_CORRECT_BUT_ANSWER_WRONG"
    assert signal["triggered"] is True
    assert report["rows"][0]["error_type"] == ["WRONG_FINAL_ANSWER"]
