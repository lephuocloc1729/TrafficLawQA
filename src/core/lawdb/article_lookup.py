from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.utils import load_json
from src.settings import settings


ArticleKey = tuple[str, str]


def _candidate_lawdb_paths() -> list[Path]:
    configured_paths = [
        settings.lawdb_json_sign_parsing_path,
        settings.lawdb_json_preprocessing_path,
        settings.lawdb_json_path,
    ]
    candidates = [Path(path) for path in configured_paths if path]
    candidates.extend(
        [
            Path("data/raw/law_db/vlsp2025_law_new.json"),
            Path("data/law_db/vlsp2025_law_new.json"),
        ]
    )

    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)
    return unique_candidates


def _article_text(article: dict[str, Any]) -> str:
    for key in ("text", "content", "markdown", "body"):
        value = article.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class LawDBArticleLookup:
    def __init__(self, lawdb_path: str | Path | None = None):
        self.lawdb_path = self._resolve_path(lawdb_path)
        self.index = self._build_index(self.lawdb_path)

    @staticmethod
    def _resolve_path(lawdb_path: str | Path | None) -> Path:
        if lawdb_path:
            path = Path(lawdb_path)
            if not path.exists():
                raise FileNotFoundError(f"LawDB file not found: {path}")
            return path

        for path in _candidate_lawdb_paths():
            if path.exists():
                return path

        searched = ", ".join(str(path) for path in _candidate_lawdb_paths())
        raise FileNotFoundError(f"No LawDB JSON file found. Searched: {searched}")

    @staticmethod
    def _build_index(lawdb_path: Path) -> dict[ArticleKey, dict[str, str]]:
        lawdb = load_json(str(lawdb_path))
        index: dict[ArticleKey, dict[str, str]] = {}

        for law in lawdb:
            law_id = str(law.get("id", "")).strip()
            for article in law.get("articles", []):
                article_id = str(article.get("id", "")).strip()
                if not law_id or not article_id:
                    continue

                title = str(article.get("title") or "").strip()
                text = _article_text(article)
                index[(law_id, article_id)] = {
                    "law_id": law_id,
                    "article_id": article_id,
                    "title": title,
                    "text": text,
                }

        return index

    def get_article(
        self,
        law_id: str,
        article_id: str,
    ) -> dict[str, str] | None:
        return self.index.get((str(law_id).strip(), str(article_id).strip()))

    def get_many(self, articles: list[dict[str, Any]]) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        seen: set[ArticleKey] = set()
        for article in articles:
            law_id = str(article.get("law_id", "")).strip()
            article_id = str(article.get("article_id", "")).strip()
            key = (law_id, article_id)
            if key in seen:
                continue
            seen.add(key)

            found_article = self.get_article(law_id, article_id)
            if found_article:
                found.append(found_article)
            elif law_id and article_id:
                found.append(
                    {
                        "law_id": law_id,
                        "article_id": article_id,
                        "title": "",
                        "text": "",
                    }
                )
        return found
