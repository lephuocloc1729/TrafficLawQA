from __future__ import annotations

from src.core.symbolic.fact_schema import SignObservation
from src.core.symbolic.supplementary_facts import extract_supplementary_context_facts


def _observation(code: str, text: str, confidence: str | None = "high") -> SignObservation:
    item = {"raw_text": text}
    if confidence is not None:
        item["confidence"] = confidence
    return SignObservation(
        "panel-1",
        role="supplementary",
        sign_code=code,
        identity_status="identified",
        ocr_texts=[item],
    )


def test_s501_and_s502_numbers_keep_distinct_distance_semantics() -> None:
    effect_length = extract_supplementary_context_facts(_observation("S.501", "300 m"), "main")
    distance = extract_supplementary_context_facts(_observation("S.502", "300 m"), "main")
    assert [(fact.predicate, fact.value) for fact in effect_length] == [("effect_length_m", 300)]
    assert [(fact.predicate, fact.value) for fact in distance] == [("distance_to_object_m", 300)]


def test_s508_extracts_time_and_day_scope_without_conflating_current_date() -> None:
    facts = extract_supplementary_context_facts(
        _observation("S.508a", "Ngày lẻ, 06:00-09:30"), "main", "relation-1"
    )
    assert [(fact.predicate, fact.value) for fact in facts] == [
        ("supplementary_calendar_parity", "odd_days"),
        ("supplementary_time_window", {
            "start_minute": 360, "end_minute": 570, "raw_text": "Ngày lẻ, 06:00-09:30"
        }),
    ]
    assert all(fact.status == "observed" and fact.metadata["relation_id"] == "relation-1" for fact in facts)


def test_low_or_unspecified_ocr_confidence_never_becomes_a_resolved_condition() -> None:
    low = extract_supplementary_context_facts(_observation("S.508b", "06:00-09:30", "low"), "main")
    unspecified = extract_supplementary_context_facts(_observation("S.508b", "06:00-09:30", None), "main")
    assert low[0].status == "ambiguous"
    assert unspecified[0].status == "ambiguous"


def test_s509a_height_value_is_extracted_as_a_measurement() -> None:
    facts = extract_supplementary_context_facts(_observation("S.509a", "4,2 m"), "main")
    assert [(fact.predicate, fact.value) for fact in facts] == [("safe_height_limit_m", 4.2)]


def test_s509b_is_typed_as_a_parking_modifier_without_becoming_a_standalone_rule() -> None:
    observation = _observation("S.509b", "CẤM ĐỖ XE")
    observation.confidence = "high"
    facts = extract_supplementary_context_facts(observation, "main", "relation-1")
    assert len(facts) == 1
    fact = facts[0]
    assert fact.predicate == "supplementary_panel_action_semantics"
    assert fact.value == {
        "action": "parking",
        "meaning": "no_parking",
        "permitted_primary_sign_codes": ["P.130", "P.131a", "P.131b", "P.131c"],
    }
    assert fact.status == "observed"
    assert fact.metadata["relation_id"] == "relation-1"
    assert fact.metadata["legal_basis"] == "QCVN 41:2024/BGTVT Appendix F.11(b), Figure F.11"
    assert fact.metadata["does_not_replace_or_narrow_primary_sign_without_legal_review"] is True


def test_s509b_uncertain_panel_identity_does_not_emit_definitive_semantics() -> None:
    observation = _observation("S.509b", "CẤM ĐỖ XE", "low")
    observation.identity_status = "ambiguous"
    assert extract_supplementary_context_facts(observation, "main") == []


def test_s507_is_recorded_as_standalone_dangerous_turn_guidance_not_a_scope_modifier() -> None:
    observation = _observation("S.507", "")
    observation.confidence = "high"
    facts = extract_supplementary_context_facts(observation, "scene-1")
    assert len(facts) == 1
    fact = facts[0]
    assert fact.predicate == "standalone_sign_guidance_semantics"
    assert fact.value == {
        "hazard": "dangerous_turn_ahead",
        "use": "standalone",
        "indicated_turn_direction": None,
        "direction_requires_visual_resolution": True,
        "replaces_W201_or_W202_warning": False,
    }
    assert fact.metadata["independent_sign_not_a_scope_panel"] is True
    assert fact.metadata["visual_turn_direction_not_inferred_from_code"] is True
    assert "Appendix F.9" in fact.metadata["legal_basis"]


def test_sg11_and_sg12_record_lane_guidance_without_guessing_the_lane_mapping() -> None:
    lane_direction = extract_supplementary_context_facts(
        _observation("S.G.11a", ""), "main-sign", "relation-1",
    )
    assert len(lane_direction) == 1
    assert lane_direction[0].predicate == "supplementary_lane_guidance_semantics"
    assert lane_direction[0].value == {
        "kind": "lane_count_and_direction",
        "arrow_count_equals_lanes_per_direction": True,
        "upcoming_lane_direction_mapping": None,
        "requires_visual_mapping": True,
    }
    assert lane_direction[0].metadata["numeric_lane_count_or_lane_identity_inferred"] is False
    assert "F.15" in lane_direction[0].metadata["legal_basis"]

    closed_lane = extract_supplementary_context_facts(
        _observation("S.G.12b", ""), "main-sign", "relation-1",
    )
    assert len(closed_lane) == 1
    assert closed_lane[0].predicate == "supplementary_lane_guidance_semantics"
    assert closed_lane[0].value == {
        "kind": "lane_not_in_operation",
        "affected_lane": None,
        "requires_visual_lane_resolution": True,
    }
    assert "F.16" in closed_lane[0].metadata["legal_basis"]


def test_sh6_preserves_exception_group_as_unresolved_ocr_instead_of_inferring_actor_class() -> None:
    observation = _observation("S.H.6", "XE BUÝT - NGOẠI LỆ / EXCEPT")
    observation.confidence = "high"
    facts = extract_supplementary_context_facts(observation, "main-sign", "relation-1")
    semantics = next(fact for fact in facts if fact.predicate == "supplementary_exception_semantics")
    group_text = next(fact for fact in facts if fact.predicate == "supplementary_exception_group_text")

    assert semantics.value == {
        "applies_to": "prohibition_or_restriction_sign",
        "effect": "exception_for_explicitly_displayed_road_user_group",
        "exception_group": None,
        "group_requires_visual_or_text_resolution": True,
        "blanket_permission": False,
    }
    assert semantics.metadata["group_match_must_be_evaluated_before_applying_primary_restriction"] is True
    assert group_text.value == "XE BUÝT - NGOẠI LỆ / EXCEPT"
    assert group_text.metadata["relation_id"] == "relation-1"
    exception_class = next(fact for fact in facts if fact.predicate == "sh6_exception_actor_classes")
    scope_resolved = next(fact for fact in facts if fact.predicate == "sh6_exception_scope_resolved")
    assert exception_class.value == ["bus"]
    assert scope_resolved.value is True

    ambiguous = _observation("S.H.6", "XE BUÝT VÀ XE TAXI - NGOẠI LỆ")
    ambiguous.confidence = "high"
    ambiguous_facts = extract_supplementary_context_facts(ambiguous, "main-sign")
    assert any(fact.predicate == "supplementary_exception_group_text" for fact in ambiguous_facts)
    assert not any(fact.predicate == "sh6_exception_scope_resolved" for fact in ambiguous_facts)

    conflicting = _observation("S.H.6", "")
    conflicting.confidence = "high"
    conflicting.ocr_texts = [
        {"raw_text": "XE BUÝT - NGOẠI LỆ", "confidence": "high"},
        {"raw_text": "XE TẢI - NGOẠI LỆ", "confidence": "high"},
    ]
    conflicting_facts = extract_supplementary_context_facts(conflicting, "main-sign")
    assert not any(fact.predicate == "sh6_exception_scope_resolved" for fact in conflicting_facts)


def test_s510_warning_semantics_preserve_qcvn_context_without_asserting_scene_conditions() -> None:
    snow = _observation("S.510a", "CHÚ Ý ĐƯỜNG TRƠN CÓ BĂNG TUYẾT")
    snow.confidence = "high"
    snow_fact = extract_supplementary_context_facts(snow, "main")[0]
    assert snow_fact.predicate == "supplementary_warning_semantics"
    assert snow_fact.value == {
        "hazard": "slippery_road_with_snow_or_ice",
        "official_context": "days_when_snowing",
        "scene_weather_inferred": False,
    }
    assert snow_fact.metadata["meaning_only_no_permission_or_prohibition_inferred"] is True

    railway = _observation("S.510b", "CHÚ Ý ĐƯỜNG SẮT 30 m")
    railway.confidence = "high"
    railway_fact = extract_supplementary_context_facts(railway, "main")[0]
    assert railway_fact.value == {
        "hazard": "railway_crossing_on_branch_road",
        "official_context": "when_turning_into_branch_road_that_intersects_railway",
        "railway_to_sign_section_threshold_m": 100,
        "threshold_operator": "less_than",
        "scene_route_or_distance_inferred": False,
    }
    assert railway_fact.metadata["legal_basis"] == "QCVN 41:2024/BGTVT Appendix F.12, Figure F.12"


def test_s505c_axle_limits_require_explicit_axle_class_and_tonne_unit() -> None:
    observation = SignObservation(
        "panel-1",
        role="supplementary",
        sign_code="S.505c",
        identity_status="identified",
        ocr_texts=[
            {"raw_text": "TRỤC ĐƠN 10 t", "confidence": "high"},
            {"raw_text": "TRỤC KÉP 18,5 t", "confidence": "medium"},
            {"raw_text": "TRỤC BA 24 t", "confidence": "high"},
            {"raw_text": "10 t", "confidence": "high"},
        ],
    )
    facts = extract_supplementary_context_facts(observation, "main", "relation-1")
    assert [(fact.predicate, fact.value) for fact in facts] == [
        ("single_axle_load_limit_t", 10.0),
        ("tandem_axle_load_limit_t", 18.5),
        ("triple_axle_load_limit_t", 24.0),
    ]
    assert all(fact.status == "observed" and fact.metadata["relation_id"] == "relation-1" for fact in facts)


def test_conflicting_s505c_ocr_readings_shadow_numeric_limit() -> None:
    observation = _observation("S.505c", "Trục đơn 10 t")
    observation.ocr_texts.append({"raw_text": "Trục đơn 12 t", "confidence": "high"})
    facts = extract_supplementary_context_facts(observation, "main")
    assert len(facts) == 1
    assert facts[0].predicate == "single_axle_load_limit_t"
    assert facts[0].value is None
    assert facts[0].status == "ambiguous"
    assert facts[0].metadata["conflicting_values"] == [10.0, 12.0]


def test_s505b_gross_weight_values_require_explicit_table_row_mapping() -> None:
    observation = SignObservation(
        "panel-1",
        role="supplementary",
        sign_code="S.505b",
        identity_status="identified",
        ocr_texts=[
            {"raw_text": "18 t", "table_row_index": 1, "confidence": "high"},
            {"raw_text": "24,5 t", "table_row_index": 3, "confidence": "medium"},
            {"raw_text": "99 t", "confidence": "high"},
        ],
    )
    facts = extract_supplementary_context_facts(observation, "main", "relation-1")
    assert [(fact.value, fact.metadata["table_row_index"]) for fact in facts] == [
        (18.0, 1), (24.5, 3),
    ]


def test_conflicting_s505b_values_shadow_only_the_affected_row() -> None:
    observation = SignObservation(
        "panel-1",
        role="supplementary",
        sign_code="S.505b",
        identity_status="identified",
        ocr_texts=[
            {"raw_text": "18 t", "table_row_index": 1, "confidence": "high"},
            {"raw_text": "20 t", "table_row_index": 1, "confidence": "high"},
            {"raw_text": "24 t", "table_row_index": 2, "confidence": "high"},
        ],
    )
    facts = extract_supplementary_context_facts(observation, "main")
    assert len(facts) == 2
    row_one = next(fact for fact in facts if fact.metadata.get("table_row_index") == 1)
    row_two = next(fact for fact in facts if fact.metadata.get("table_row_index") == 2)
    assert row_one.value is None and row_one.status == "ambiguous"
    assert row_two.value == 24.0 and row_two.status == "observed"


def test_unknown_sign_codes_do_not_trigger_supplementary_value_parsing() -> None:
    assert extract_supplementary_context_facts(_observation("P.130", "300 m"), "main") == []


def test_s503_variant_mapping_records_sign_face_arrow_not_road_relative_direction() -> None:
    expected = {
        "S.503a": ("right", "perpendicular_to_road"),
        "S.503b": ("both_horizontal_directions", "perpendicular_to_road"),
        "S.503c": ("left", "perpendicular_to_road"),
        "S.503d": ("down", "parallel_to_road"),
        "S.503e": ("both_vertical_directions", "parallel_to_road"),
        "S.503f": ("up", "parallel_to_road"),
    }
    for code, (arrow, axis) in expected.items():
        observation = _observation(code, "", confidence="high")
        observation.confidence = "high"
        facts = extract_supplementary_context_facts(observation, "main", "linked-panel")
        visual = next(fact for fact in facts if fact.predicate == "supplementary_panel_visual_semantics")
        assert visual.value["depicted_arrow"] == arrow
        assert visual.value["effect_axis"] == axis
        assert visual.value["orientation_reference"] == "sign_face"
        assert visual.value["road_relative_direction_resolved"] is False
        assert visual.status == "observed"
        assert visual.metadata["relation_id"] == "linked-panel"
        assert visual.metadata["legal_basis"].endswith("Figures F.3a-F.3b")


def test_scope_panel_facts_do_not_invent_specific_lane_vehicle_or_route_value() -> None:
    expected = {
        "S.504": "lane",
        "S.505a": "vehicle_class",
        "S.506a": "priority_road_direction_for_I.401",
        "S.506b": "priority_road_direction_for_W.208_or_R.122",
    }
    for code, dimension in expected.items():
        observation = _observation(code, "", confidence="high")
        observation.confidence = "high"
        facts = extract_supplementary_context_facts(observation, "main", "linked-panel")
        scope = next(fact for fact in facts if fact.predicate == "supplementary_scope_dimension")
        assert scope.value == dimension
        assert scope.status == "observed"
        assert scope.metadata["specific_lane_vehicle_or_route_value_still_requires_visual_resolution"] is True


def test_ambiguous_supplementary_identity_does_not_emit_code_derived_semantics() -> None:
    observation = _observation("S.503a", "", confidence="high")
    observation.identity_status = "ambiguous"
    assert extract_supplementary_context_facts(observation, "main") == []
