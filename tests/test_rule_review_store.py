from __future__ import annotations

import pytest

from src.core.symbolic.rule_review_store import latest_reviews, record_review, review_history


CHECKS = {
    "source": True,
    "scope": True,
    "effect": True,
    "tests": True,
}


def test_review_store_keeps_history_and_returns_latest_decision(tmp_path):
    path = tmp_path / "reviews.sqlite3"
    record_review(
        rule_id="P123A_NO_LEFT_TURN",
        verdict="NEEDS_REVISION",
        reviewer="reviewer-a",
        notes="Confirm the exact variant scope.",
        checks={"source": False},
        path=path,
    )
    record_review(
        rule_id="P123A_NO_LEFT_TURN",
        verdict="APPROVED",
        reviewer="reviewer-b",
        notes="Source and scope checked against the cited figure.",
        checks=CHECKS,
        rule_fingerprint="abc123",
        path=path,
    )

    latest = latest_reviews(path)["P123A_NO_LEFT_TURN"]
    assert latest["verdict"] == "APPROVED"
    assert latest["reviewer"] == "reviewer-b"
    assert latest["checks"] == CHECKS
    assert len(review_history("P123A_NO_LEFT_TURN", path)) == 2


def test_approval_requires_all_review_checklist_items(tmp_path):
    with pytest.raises(ValueError, match="every review checklist"):
        record_review(
            rule_id="P123A_NO_LEFT_TURN",
            verdict="APPROVED",
            reviewer="reviewer",
            notes="Approved",
            checks={"source": True, "scope": False},
            path=tmp_path / "reviews.sqlite3",
        )

def test_review_requires_named_reviewer_and_notes(tmp_path):
    with pytest.raises(ValueError, match="review notes are required"):
        record_review(
            rule_id="P123A_NO_LEFT_TURN",
            verdict="REJECTED",
            reviewer="",
            notes="",
            checks={},
            path=tmp_path / "reviews.sqlite3",
        )
