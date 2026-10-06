"""Local, append-only storage for human review of draft traffic rules."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_DB = Path(__file__).parents[3] / "data" / "reviews" / "qcvn_41_2024_rule_reviews.sqlite3"


def _connect(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS rule_review_events (
            review_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id TEXT NOT NULL,
            verdict TEXT NOT NULL CHECK (verdict IN ('APPROVED', 'NEEDS_REVISION', 'REJECTED')),
            reviewer TEXT NOT NULL,
            notes TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            checks_json TEXT NOT NULL,
            proposed_changes TEXT NOT NULL,
            rule_fingerprint TEXT NOT NULL
        )"""
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_review_events_rule_id ON rule_review_events(rule_id, review_id)"
    )
    return connection


def latest_reviews(path: str | Path = DEFAULT_REVIEW_DB) -> dict[str, dict[str, Any]]:
    """Return the latest decision for each rule without discarding review history."""
    with _connect(path) as connection:
        rows = connection.execute(
            """SELECT event.* FROM rule_review_events AS event
               JOIN (SELECT rule_id, MAX(review_id) AS latest_id
                     FROM rule_review_events GROUP BY rule_id) AS latest
                 ON event.review_id = latest.latest_id"""
        ).fetchall()
    result = {}
    for row in rows:
        item = dict(row)
        item["checks"] = json.loads(item.pop("checks_json"))
        result[item["rule_id"]] = item
    return result


def review_history(rule_id: str, path: str | Path = DEFAULT_REVIEW_DB) -> list[dict[str, Any]]:
    with _connect(path) as connection:
        rows = connection.execute(
            "SELECT * FROM rule_review_events WHERE rule_id = ? ORDER BY review_id DESC",
            (rule_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["checks"] = json.loads(item.pop("checks_json"))
        result.append(item)
    return result


def record_review(
    *,
    rule_id: str,
    verdict: str,
    reviewer: str,
    notes: str,
    checks: dict[str, bool],
    proposed_changes: str = "",
    rule_fingerprint: str = "",
    path: str | Path = DEFAULT_REVIEW_DB,
) -> int:
    verdict = verdict.strip().upper()
    if verdict not in {"APPROVED", "NEEDS_REVISION", "REJECTED"}:
        raise ValueError(f"Unsupported review verdict: {verdict}")
    reviewer = reviewer.strip()
    notes = notes.strip()
    if not rule_id.strip() or not reviewer or not notes:
        raise ValueError("rule_id, reviewer, and review notes are required")
    if verdict == "APPROVED" and not all(checks.values()):
        raise ValueError("Approval requires every review checklist item to be confirmed")

    with _connect(path) as connection:
        cursor = connection.execute(
            """INSERT INTO rule_review_events
               (rule_id, verdict, reviewer, notes, reviewed_at, checks_json,
                proposed_changes, rule_fingerprint)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rule_id.strip(), verdict, reviewer, notes,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(checks, ensure_ascii=False, sort_keys=True),
                proposed_changes.strip(), rule_fingerprint,
            ),
        )
        return int(cursor.lastrowid)
