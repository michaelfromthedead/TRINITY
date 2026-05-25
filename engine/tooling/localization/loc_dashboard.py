"""
Localization dashboard for progress tracking.

Provides overview of localization status, missing strings,
and translation progress.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
from datetime import datetime


@dataclass(slots=True)
class LanguageProgress:
    """
    Progress information for a single language.
    """
    language_code: str
    language_name: str
    total_strings: int
    translated_strings: int
    approved_strings: int
    needs_review: int
    word_count: int
    translated_word_count: int
    last_updated: Optional[float] = None

    @property
    def completion_percent(self) -> float:
        """Get completion percentage."""
        if self.total_strings == 0:
            return 0.0
        return (self.translated_strings / self.total_strings) * 100

    @property
    def approval_percent(self) -> float:
        """Get approval percentage."""
        if self.translated_strings == 0:
            return 0.0
        return (self.approved_strings / self.translated_strings) * 100

    @property
    def word_completion_percent(self) -> float:
        """Get word count completion percentage."""
        if self.word_count == 0:
            return 0.0
        return (self.translated_word_count / self.word_count) * 100


@dataclass(slots=True)
class TranslationStats:
    """
    Overall translation statistics.
    """
    total_strings: int = 0
    total_words: int = 0
    total_characters: int = 0
    languages: int = 0
    fully_translated: int = 0
    partially_translated: int = 0
    not_started: int = 0
    average_completion: float = 0.0
    strings_needing_review: int = 0
    last_export: Optional[float] = None
    last_import: Optional[float] = None


@dataclass(slots=True)
class MissingString:
    """
    Information about a missing translation.
    """
    key: str
    source_text: str
    language: str
    category: str = ""
    priority: int = 0
    context: str = ""
    source_file: str = ""
    word_count: int = 0

    def __post_init__(self):
        """Calculate word count."""
        if self.word_count == 0:
            self.word_count = len(self.source_text.split())


class LocalizationDashboard:
    """
    Dashboard for monitoring localization progress.

    Provides comprehensive view of translation status,
    missing strings, and progress tracking.
    """
    __slots__ = (
        "_string_table",
        "_language_progress",
        "_stats",
        "_missing_strings",
        "_filters",
        "_sort_key",
        "_sort_reverse",
    )

    def __init__(self):
        """Initialize dashboard."""
        self._string_table: Optional[Any] = None  # StringTableManager reference
        self._language_progress: dict[str, LanguageProgress] = {}
        self._stats = TranslationStats()
        self._missing_strings: list[MissingString] = []
        self._filters: dict[str, Any] = {}
        self._sort_key = "key"
        self._sort_reverse = False

    def set_string_table(self, table: Any) -> None:
        """Set the string table to monitor."""
        self._string_table = table
        self.refresh()

    def refresh(self) -> None:
        """Refresh all dashboard data."""
        if self._string_table is None:
            return

        self._calculate_progress()
        self._calculate_stats()
        self._find_missing_strings()

    def _calculate_progress(self) -> None:
        """Calculate progress for each language."""
        self._language_progress.clear()

        if self._string_table is None:
            return

        all_languages = self._string_table.get_all_languages()
        source_lang = getattr(self._string_table, '_fallback_language', 'en')

        # Get all entries
        all_entries = self._string_table.get_all_entries()
        total_strings = len(all_entries)

        # Calculate source word count
        total_words = sum(
            len(entry[1].source_text.split())
            for entry in all_entries
        )

        for lang in all_languages:
            if lang == source_lang:
                continue

            translated = 0
            approved = 0
            needs_review = 0
            translated_words = 0

            for table_name, entry in all_entries:
                if entry.has_translation(lang):
                    translated += 1
                    translated_words += len(entry.source_text.split())

                    if hasattr(entry, 'is_approved') and entry.is_approved:
                        approved += 1

                    if hasattr(entry, 'needs_review') and entry.needs_review:
                        needs_review += 1

            progress = LanguageProgress(
                language_code=lang,
                language_name=self._get_language_name(lang),
                total_strings=total_strings,
                translated_strings=translated,
                approved_strings=approved,
                needs_review=needs_review,
                word_count=total_words,
                translated_word_count=translated_words,
            )

            self._language_progress[lang] = progress

    def _calculate_stats(self) -> None:
        """Calculate overall statistics."""
        if self._string_table is None:
            return

        all_entries = self._string_table.get_all_entries()

        self._stats.total_strings = len(all_entries)
        self._stats.total_words = sum(
            len(entry[1].source_text.split())
            for entry in all_entries
        )
        self._stats.total_characters = sum(
            len(entry[1].source_text)
            for entry in all_entries
        )
        self._stats.languages = len(self._language_progress)

        # Count completion status
        fully = 0
        partial = 0
        not_started = 0

        for progress in self._language_progress.values():
            if progress.completion_percent >= 100:
                fully += 1
            elif progress.completion_percent > 0:
                partial += 1
            else:
                not_started += 1

        self._stats.fully_translated = fully
        self._stats.partially_translated = partial
        self._stats.not_started = not_started

        # Calculate average completion
        if self._language_progress:
            total_completion = sum(
                p.completion_percent for p in self._language_progress.values()
            )
            self._stats.average_completion = total_completion / len(self._language_progress)

        # Count strings needing review
        self._stats.strings_needing_review = sum(
            p.needs_review for p in self._language_progress.values()
        )

    def _find_missing_strings(self) -> None:
        """Find all missing translations."""
        self._missing_strings.clear()

        if self._string_table is None:
            return

        all_entries = self._string_table.get_all_entries()

        for lang in self._language_progress.keys():
            for table_name, entry in all_entries:
                if not entry.has_translation(lang):
                    missing = MissingString(
                        key=entry.key,
                        source_text=entry.source_text,
                        language=lang,
                        category=entry.context.category if hasattr(entry.context, 'category') else "",
                        context=entry.context.description if hasattr(entry.context, 'description') else "",
                        source_file=entry.source_file if hasattr(entry, 'source_file') else "",
                    )
                    self._missing_strings.append(missing)

    def _get_language_name(self, code: str) -> str:
        """Get display name for language code."""
        names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "pl": "Polish",
            "nl": "Dutch",
            "tr": "Turkish",
            "th": "Thai",
            "vi": "Vietnamese",
        }
        return names.get(code, code.upper())

    # Progress queries
    def get_language_progress(self, language: str) -> Optional[LanguageProgress]:
        """Get progress for a specific language."""
        return self._language_progress.get(language)

    def get_all_progress(self) -> list[LanguageProgress]:
        """Get progress for all languages."""
        return list(self._language_progress.values())

    def get_progress_sorted(
        self,
        by: str = "completion",
        descending: bool = True
    ) -> list[LanguageProgress]:
        """
        Get progress sorted by criteria.

        Args:
            by: Sort key ("completion", "name", "translated", "words")
            descending: Sort in descending order

        Returns:
            Sorted progress list
        """
        progress_list = list(self._language_progress.values())

        key_map = {
            "completion": lambda p: p.completion_percent,
            "name": lambda p: p.language_name,
            "translated": lambda p: p.translated_strings,
            "words": lambda p: p.translated_word_count,
        }

        key_func = key_map.get(by, key_map["completion"])
        return sorted(progress_list, key=key_func, reverse=descending)

    def get_stats(self) -> TranslationStats:
        """Get overall statistics."""
        return self._stats

    # Missing strings queries
    def get_missing_strings(
        self,
        language: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 0
    ) -> list[MissingString]:
        """
        Get missing strings with optional filtering.

        Args:
            language: Filter by language
            category: Filter by category
            limit: Maximum results (0 = no limit)

        Returns:
            Filtered missing strings
        """
        result = self._missing_strings

        if language:
            result = [m for m in result if m.language == language]

        if category:
            result = [m for m in result if m.category == category]

        # Sort
        result = sorted(
            result,
            key=lambda m: getattr(m, self._sort_key, m.key),
            reverse=self._sort_reverse
        )

        if limit > 0:
            result = result[:limit]

        return result

    def get_missing_count(self, language: Optional[str] = None) -> int:
        """Get count of missing strings."""
        if language:
            return sum(1 for m in self._missing_strings if m.language == language)
        return len(self._missing_strings)

    def get_missing_by_category(self, language: str) -> dict[str, int]:
        """Get missing string counts by category for a language."""
        counts: dict[str, int] = {}

        for missing in self._missing_strings:
            if missing.language == language:
                cat = missing.category or "Uncategorized"
                counts[cat] = counts.get(cat, 0) + 1

        return counts

    # Filtering and sorting
    def set_filter(self, key: str, value: Any) -> None:
        """Set a filter."""
        self._filters[key] = value

    def clear_filters(self) -> None:
        """Clear all filters."""
        self._filters.clear()

    def set_sort(self, key: str, reverse: bool = False) -> None:
        """Set sort parameters."""
        self._sort_key = key
        self._sort_reverse = reverse

    # Export
    def export_report(self, format: str = "text") -> str:
        """
        Export dashboard report.

        Args:
            format: Output format ("text", "json", "csv")

        Returns:
            Formatted report
        """
        if format == "json":
            return self._export_json()
        elif format == "csv":
            return self._export_csv()
        else:
            return self._export_text()

    def _export_text(self) -> str:
        """Export as text report."""
        lines = [
            "=" * 60,
            "LOCALIZATION DASHBOARD REPORT",
            "=" * 60,
            "",
            "OVERALL STATISTICS",
            "-" * 40,
            f"Total Strings: {self._stats.total_strings}",
            f"Total Words: {self._stats.total_words}",
            f"Languages: {self._stats.languages}",
            f"Fully Translated: {self._stats.fully_translated}",
            f"Partially Translated: {self._stats.partially_translated}",
            f"Not Started: {self._stats.not_started}",
            f"Average Completion: {self._stats.average_completion:.1f}%",
            "",
            "LANGUAGE PROGRESS",
            "-" * 40,
        ]

        for progress in self.get_progress_sorted():
            lines.append(
                f"{progress.language_name} ({progress.language_code}): "
                f"{progress.completion_percent:.1f}% "
                f"({progress.translated_strings}/{progress.total_strings})"
            )

        lines.extend([
            "",
            "MISSING TRANSLATIONS",
            "-" * 40,
            f"Total Missing: {len(self._missing_strings)}",
        ])

        for lang, progress in self._language_progress.items():
            missing = self.get_missing_count(lang)
            if missing > 0:
                lines.append(f"  {progress.language_name}: {missing} missing")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _export_json(self) -> str:
        """Export as JSON."""
        import json

        data = {
            "generated_at": datetime.now().isoformat(),
            "stats": {
                "total_strings": self._stats.total_strings,
                "total_words": self._stats.total_words,
                "languages": self._stats.languages,
                "average_completion": self._stats.average_completion,
            },
            "languages": [
                {
                    "code": p.language_code,
                    "name": p.language_name,
                    "completion": p.completion_percent,
                    "translated": p.translated_strings,
                    "total": p.total_strings,
                }
                for p in self.get_progress_sorted()
            ],
            "missing_count": len(self._missing_strings),
        }

        return json.dumps(data, indent=2)

    def _export_csv(self) -> str:
        """Export as CSV."""
        lines = ["Language,Code,Completion,Translated,Total"]

        for progress in self.get_progress_sorted():
            lines.append(
                f"{progress.language_name},{progress.language_code},"
                f"{progress.completion_percent:.1f},{progress.translated_strings},"
                f"{progress.total_strings}"
            )

        return "\n".join(lines)

    # Quick status methods
    def get_status_summary(self) -> dict[str, Any]:
        """Get quick status summary."""
        return {
            "total_strings": self._stats.total_strings,
            "languages": self._stats.languages,
            "average_completion": round(self._stats.average_completion, 1),
            "fully_complete": self._stats.fully_translated,
            "missing_count": len(self._missing_strings),
            "needs_review": self._stats.strings_needing_review,
        }

    def is_complete(self, language: str) -> bool:
        """Check if a language is fully translated."""
        progress = self._language_progress.get(language)
        return progress is not None and progress.completion_percent >= 100

    def get_priority_missing(self, language: str, count: int = 10) -> list[MissingString]:
        """Get highest priority missing strings for a language."""
        missing = [m for m in self._missing_strings if m.language == language]
        missing.sort(key=lambda m: m.priority, reverse=True)
        return missing[:count]
