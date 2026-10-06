from src.core.symbolic.fact_schema import SignObservation
from src.core.symbolic.sign_numeric_facts import extract_main_sign_context_facts


def observation(code: str, ocr: list[dict]) -> SignObservation:
    return SignObservation(
        observation_id="main-sign",
        role="main",
        sign_code=code,
        identity_status="identified",
        ocr_texts=ocr,
        confidence="high",
        evidence=[{"source": "test_crop", "bbox": [0, 0, 1, 1]}],
    )


def test_p121_ocr_extracts_minimum_distance_with_its_code_defined_unit():
    facts = extract_main_sign_context_facts(
        observation("P.121", [{"raw_text": "8 m", "confidence": "high"}]), "main-sign",
    )

    assert len(facts) == 1
    assert (facts[0].predicate, facts[0].value, facts[0].status) == (
        "minimum_following_distance_m", 8, "observed",
    )
    assert facts[0].metadata["unit"] == "m"
    assert facts[0].evidence[0]["source"] == "test_crop"


def test_p117_to_p120_ocr_extracts_code_specific_metric_limits():
    for code, predicate, text, expected in (
        ("P.117", "height_limit_m", "4,2 m", 4.2),
        ("P.118", "width_limit_m", "3,2m", 3.2),
        ("P.119", "length_limit_m", "9 m", 9.0),
        ("P.120", "combination_length_limit_m", "14m", 14.0),
    ):
        facts = extract_main_sign_context_facts(
            observation(code, [{"raw_text": text, "confidence": "high"}]), "main-sign",
        )
        assert len(facts) == 1, code
        assert (facts[0].predicate, facts[0].value, facts[0].metadata["unit"]) == (
            predicate, expected, "m",
        )


def test_p117_to_p120_conflicting_numeric_values_shadow_limit():
    for code, predicate in (
        ("P.117", "height_limit_m"),
        ("P.118", "width_limit_m"),
        ("P.119", "length_limit_m"),
        ("P.120", "combination_length_limit_m"),
    ):
        facts = extract_main_sign_context_facts(observation(code, [
            {"raw_text": "3,2 m", "confidence": "high"},
            {"raw_text": "3,5 m", "confidence": "high"},
        ]), "main-sign")
        assert len(facts) == 1, code
        assert facts[0].predicate == predicate
        assert facts[0].value is None
        assert facts[0].status == "ambiguous"


def test_p121_conflicting_ocr_values_remain_ambiguous():
    facts = extract_main_sign_context_facts(observation("P.121", [
        {"raw_text": "8 m", "confidence": "high"},
        {"raw_text": "10 m", "confidence": "medium"},
    ]), "main-sign")

    assert len(facts) == 1
    assert facts[0].predicate == "minimum_following_distance_m"
    assert facts[0].status == "ambiguous"
    assert facts[0].metadata["conflicting_values"] == [8, 10]


def test_main_sign_numeric_extraction_does_not_guess_other_sign_values():
    assert extract_main_sign_context_facts(
        observation("P.130", [{"raw_text": "8 m", "confidence": "high"}]), "main-sign",
    ) == []
    assert extract_main_sign_context_facts(
        observation("P.121", [{"raw_text": "about eight meters", "confidence": "high"}]), "main-sign",
    ) == []


def test_r306_ocr_extracts_minimum_speed_with_code_defined_kmh_unit():
    facts = extract_main_sign_context_facts(
        observation("R.306", [{"raw_text": "60", "confidence": "high"}]), "main-sign",
    )

    assert len(facts) == 1
    assert (facts[0].predicate, facts[0].value, facts[0].metadata["unit"]) == (
        "minimum_speed_limit_kmh", 60, "km/h",
    )


def test_re9d_ocr_extracts_zone_speed_limit_without_assuming_its_scope():
    facts = extract_main_sign_context_facts(
        observation("R.E.9d", [{"raw_text": "50", "confidence": "high"}]), "main-sign",
    )

    assert len(facts) == 1
    assert (facts[0].predicate, facts[0].value, facts[0].metadata["unit"]) == (
        "speed_limit_value", 50, "km/h",
    )


def test_re9d_conflicting_speed_ocr_readings_remain_ambiguous():
    facts = extract_main_sign_context_facts(observation("R.E.9d", [
        {"raw_text": "50", "confidence": "high"},
        {"raw_text": "60", "confidence": "medium"},
    ]), "main-sign")

    assert len(facts) == 1
    assert facts[0].predicate == "speed_limit_value"
    assert facts[0].status == "ambiguous"
    assert facts[0].metadata["conflicting_values"] == [50, 60]


def test_re9b_time_range_is_extracted_only_from_strict_hhmm_range_text():
    facts = extract_main_sign_context_facts(
        observation("R.E.9b", [{"raw_text": "6:00 – 18:00", "confidence": "high"}]), "main-sign",
    )

    assert len(facts) == 1
    assert facts[0].predicate == "time_window"
    assert facts[0].value["start_minute"] == 360
    assert facts[0].value["end_minute"] == 1080

    assert extract_main_sign_context_facts(
        observation("R.E.9b", [{"raw_text": "morning until evening", "confidence": "high"}]), "main-sign",
    ) == []


def test_re9b_conflicting_time_ranges_remain_ambiguous():
    facts = extract_main_sign_context_facts(observation("R.E.9b", [
        {"raw_text": "6:00 - 18:00", "confidence": "high"},
        {"raw_text": "7:00 - 19:00", "confidence": "medium"},
    ]), "main-sign")

    assert len(facts) == 1
    assert facts[0].predicate == "time_window"
    assert facts[0].status == "ambiguous"
