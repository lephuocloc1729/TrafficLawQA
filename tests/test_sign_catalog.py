from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.core.symbolic.fact_schema import SignObservation, TrafficFacts
from src.core.symbolic.intent_schema import Intent, IntentTarget
from src.core.symbolic.question_intent import parse_query_intent_deterministic
from src.core.symbolic.sign_catalog import (
    OfficialSignCatalog,
    load_default_sign_catalog,
    normalize_official_sign_code,
)


def test_official_catalog_is_source_first_and_independent_of_lawdb() -> None:
    catalog = load_default_sign_catalog()
    assert catalog.metadata["official_document"] == "QCVN 41:2024/BGTVT"
    assert catalog.metadata["items_count"] == 442
    assert catalog.lookup("P.123a") is not None
    assert catalog.lookup("P.123a").status == "unverified"
    assert catalog.lookup("P.123a").legal_basis[0]["source_url"].startswith("https://datafiles.chinhphu.vn/")
    payload = json.loads((Path(__file__).parents[1] / "src/core/symbolic/catalogs/qcvn_41_2024_sign_catalog_draft.json").read_text())
    ids = [item["canonical_sign_id"] for item in payload["items"]]
    assert len(ids) == len(set(ids))
    assert all(item["legal_basis"] for item in payload["items"])
    assert all(item["eligible_for_automatic_decision"] is False for item in payload["items"])
    assert not {"P.133", "P.134", "P.135"}.intersection(
        item["official_code"] for item in payload["items"]
    )
    for code in ("DP.133", "DP.134", "DP.135"):
        item = catalog.lookup(code)
        assert item is not None
        assert item.legal_basis[0]["section_or_figure"].startswith("Phụ lục B.")


def test_catalog_json_sources_reject_duplicate_object_keys() -> None:
    root = Path(__file__).parents[1] / "src/core/symbolic/catalogs"

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    for path in root.glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def test_family_heading_codes_with_lettered_variants_do_not_resolve_as_single_signs() -> None:
    catalog = load_default_sign_catalog()
    for family_code, expected in {
        "DP.127": {"DP.127a", "DP.127b", "DP.127c"},
        "I.407": {"I.407a", "I.407b", "I.407c"},
        "P.131": {"P.131a", "P.131b", "P.131c"},
    }.items():
        result = catalog.lookup(family_code)
        assert result is not None
        assert result.status == "variant_required"
        assert {candidate["sign_code"] for candidate in result.candidates} == expected
        assert result.legal_basis

    # These are real unlettered signs even though their families have variants.
    assert catalog.lookup("P.127").status == "unverified"
    assert catalog.lookup("I.408").status == "unverified"


def test_suspected_p133_to_p135_extraction_conflicts_are_auditable_not_canonical() -> None:
    root = Path(__file__).parents[1] / "src/core/symbolic/catalogs"
    anomalies = json.loads((root / "qcvn_41_2024_catalog_extraction_anomalies.json").read_text())
    catalog = load_default_sign_catalog()

    assert {item["candidate_code"] for item in anomalies["items"]} == {
        "P.133", "P.134", "P.135", "I.453c", "R.413i–R.413o",
    }
    assert all(item["status"] == "confirmed_official_text_code_conflict" for item in anomalies["items"])
    assert all("pending_team_review" in item["disposition"] for item in anomalies["items"])
    for code in ("P.133", "P.134", "P.135"):
        assert catalog.lookup(code) is None
    for code in ("DP.133", "DP.134", "DP.135"):
        assert catalog.lookup(code) is not None
    assert catalog.lookup("I.453c") is None
    ie453c = catalog.lookup("IE.453c")
    assert ie453c is not None
    assert ie453c.legal_meaning == "Biển chỉ dẫn hết đường cao tốc."
    assert "47.4.2" in ie453c.legal_basis[0]["section_or_figure"]
    assert "appendix_figure_subcaption_mislabels_IE_453c_as_I_453c" in ie453c.issues
    family_lookup = catalog.lookup("IE.450")
    assert family_lookup is not None
    assert family_lookup.status == "variant_required"
    assert {row["sign_code"] for row in family_lookup.candidates} == {"IE.450a", "IE.450b"}
    for family_code in ("IE.457", "IE.460", "IE.465", "R.302", "R.413", "R.415", "W.201"):
        family_result = catalog.lookup(family_code)
        assert family_result is not None
        assert family_result.status == "variant_required"
        assert family_result.candidates
    for code, expected in {
        "IE.451a": "khoảng cách đến lối vào",
        "IE.451b": "lối vào đường cao tốc",
        "IE.456a": "khoảng cách đến trạm dừng nghỉ",
        "IE.456b": "hướng rẽ vào trạm dừng nghỉ",
        "IE.456c": "lối vào trạm dừng nghỉ",
        "IE.472c": "thu phí không dừng",
        "IE.472d": "rút thẻ",
        "IE.472e": "thu phí",
    }.items():
        entry = catalog.lookup(code)
        assert entry is not None
        assert expected in entry.legal_meaning.lower()
    for code in ("R.413i", "R.413j", "R.413k", "R.413l", "R.413m", "R.413n", "R.413o", "R.413p"):
        item = catalog.lookup(code)
        assert item is not None
        assert item.legal_meaning
        assert item.status == "unverified"
        assert "official_prose_and_figure_subcaptions_conflict_on_R412_vs_R413_code" in item.issues
    bridge = catalog.lookup("I.439")
    assert bridge is not None
    assert bridge.legal_meaning == "Biển cung cấp thông tin về cầu; trên biển ghi tên cầu, lý trình và tên đường."
    assert "E.42" in bridge.legal_basis[0]["section_or_figure"]
    for code in ("S.H.3a", "S.H.3b", "S.H.3c"):
        item = catalog.lookup(code)
        assert item is not None
        assert item.official_name == "Hướng tác dụng của biển"
        assert "chỉ hướng tác dụng của biển" in item.legal_meaning.lower()
        assert item.status == "unverified"
        assert "variant_specific_direction_not_textually_defined_in_QCVN" in item.issues
        assert "Điều 41.2.1" in item.legal_basis[0]["section_or_figure"]
        assert item.legal_basis[0]["verification_status"] == "unverified"
    assert catalog.lookup("S.H,3a").sign_code == "S.H.3a"


def test_uncoded_figures_are_preserved_without_fake_official_codes() -> None:
    path = Path(__file__).parents[1] / "src/core/symbolic/catalogs/qcvn_41_2024_uncoded_figure_review.json"
    payload = json.loads(path.read_text())
    assert payload["items_count"] == 118
    assert all(item["official_code"] is None for item in payload["items"])
    assert payload["classified_existing_sign_figure_count"] == 13
    assert payload["unresolved_figure_count"] == 6
    assert all(item.get("related_official_codes") for item in payload["items"] if item["review_status"] == "linked_to_catalog_signs")
    assert any(
        item["official_figure_caption"].startswith("Hình I.3")
        and item["review_status"] == "unclassified_illustration"
        for item in payload["items"]
    )


def test_sign_code_normalization_handles_alias_typography() -> None:
    assert normalize_official_sign_code(" p, 123A ") == "P.123a"


def test_catalog_does_not_assert_unreviewed_extraction_as_verified() -> None:
    catalog = load_default_sign_catalog()
    result = catalog.lookup("P.123a")
    assert result is not None
    assert result.is_verified is False
    assert "team_legal_review_required" in result.issues


def test_sign_meaning_evaluation_returns_unknown_when_identity_is_ambiguous() -> None:
    catalog = load_default_sign_catalog()
    facts = TrafficFacts("2.0", "scene-1", [SignObservation(
        "sign-1",
        identity_status="ambiguous",
        candidates=[{"sign_code": "P.123a", "confidence": 0.52}],
    )])
    intent = Intent(
        "intent-1", "QUERY_SIGN_MEANING",
        target=IntentTarget(kind="scene"),
    )
    result = catalog.evaluate_meaning(facts, intent)[0]
    assert result.status == "ambiguous"
    assert result.sign_code is None
    assert result.candidates[0]["sign_code"] == "P.123a"


def test_catalog_constructor_indexes_only_explicit_official_code() -> None:
    catalog = OfficialSignCatalog({"items": [{
        "official_code": "P.123a",
        "aliases": ["P,123a"],
        "official_name": "test",
        "legal_meaning": "test",
        "legal_basis": [{"verification_status": "verified"}],
        "review_status": "verified",
        "eligible_for_automatic_decision": True,
        "issues": [],
    }]})
    assert catalog.lookup("P,123a").status == "verified"


def test_catalog_rejects_duplicate_canonical_sign_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate canonical sign id"):
        OfficialSignCatalog({"items": [
            {"canonical_sign_id": "duplicate", "official_code": "P.123a"},
            {"canonical_sign_id": "duplicate", "official_code": "P.123b"},
        ]})


def test_sign_meaning_query_can_lookup_an_explicit_code_without_lawdb_or_image() -> None:
    catalog = load_default_sign_catalog()
    parsed = parse_query_intent_deterministic("P.123a có ý nghĩa gì?")
    result = catalog.evaluate_meaning(TrafficFacts("2.0", "empty"), parsed.query_intent.intents[0])[0]
    assert result.sign_code == "P.123a"
    assert result.status == "unverified"
    assert result.legal_basis[0]["source_url"].startswith("https://datafiles.chinhphu.vn/")


def test_ambiguous_alias_is_not_resolved_to_the_first_match() -> None:
    catalog = OfficialSignCatalog({"items": [
        {"official_code": "P.123a", "aliases": ["P-123"]},
        {"official_code": "P.123b", "aliases": ["P-123"]},
    ]})
    result = catalog.lookup("P-123")
    assert result is not None
    assert result.status == "ambiguous_alias"


def test_canonical_code_takes_precedence_over_conflicting_alias() -> None:
    catalog = load_default_sign_catalog()
    entry = catalog.lookup("P.127")
    assert entry is not None
    assert entry.sign_code == "P.127"
    assert entry.status == "unverified"


def test_default_catalog_includes_numbered_supplementary_signs_from_appendix_f() -> None:
    catalog = load_default_sign_catalog()
    expected_codes = {
        "S.501", "S.502", "S.503a", "S.503b", "S.503c", "S.503d", "S.503e", "S.503f",
        "S.504", "S.505a", "S.505b", "S.505c", "S.506a", "S.506b", "S.507",
        "S.508a", "S.508b", "S.509a", "S.509b", "S.510a", "S.510b",
    }
    for code in expected_codes:
        entry = catalog.lookup(code)
        assert entry is not None, code
        assert entry.status == "unverified"
        assert entry.legal_basis[0]["section_or_figure"].startswith("Phụ lục F.")
    assert len(catalog.items) == 442
    family_result = catalog.lookup("S.503")
    assert family_result is not None
    assert family_result.status == "variant_required"
    assert {candidate["sign_code"] for candidate in family_result.candidates} == {
        "S.503a", "S.503b", "S.503c", "S.503d", "S.503e", "S.503f",
    }
    expected_arrows = {
        "S.503a": "sang phải",
        "S.503b": "trái và phải",
        "S.503c": "sang trái",
        "S.503d": "hướng xuống",
        "S.503e": "lên và xuống",
        "S.503f": "hướng lên",
    }
    for code, arrow in expected_arrows.items():
        entry = catalog.lookup(code)
        assert entry is not None
        assert entry.status == "unverified"
        assert arrow in entry.visual_description
        assert entry.legal_basis[0]["verification_status"] == "verified"
        assert "Hình F.3" in entry.legal_basis[0]["section_or_figure"]


def test_official_sign_variants_keep_distinct_code_specific_meanings() -> None:
    catalog = load_default_sign_catalog()
    right = catalog.lookup("P.103b")
    left = catalog.lookup("P.103c")
    cargo_bicycle = catalog.lookup("P.110b")
    assert right and left and cargo_bicycle
    assert right.official_name == "Cấm xe ô tô rẽ phải"
    assert left.official_name == "Cấm xe ô tô rẽ trái"
    assert "rẽ phải" in right.legal_meaning
    assert "rẽ trái" in left.legal_meaning
    assert right.legal_meaning != left.legal_meaning
    assert cargo_bicycle.legal_meaning == "Cấm xe đạp thồ đi qua; không cấm người dắt loại xe này."
    assert all(entry.legal_basis[0]["section_or_figure"] for entry in (right, left, cargo_bicycle))

    s505b = catalog.lookup("S.505b")
    assert s505b is not None
    assert s505b.status == "unverified"
    assert "ba dòng" in s505b.visual_description
    assert "không đặt tên riêng" in s505b.visual_description
    assert s505b.legal_basis[0]["section_or_figure"] == "Phụ lục F.6, Hình F.6"
    assert s505b.legal_basis[0]["verification_status"] == "verified"

    p111a = catalog.lookup("P.111a")
    p111b = catalog.lookup("P.111b")
    p111d = catalog.lookup("P.111d")
    assert p111a and p111b and p111d
    assert "xe gắn máy" in p111a.legal_meaning
    assert "có động cơ" in p111b.legal_meaning
    assert "không có động cơ" in p111d.legal_meaning
