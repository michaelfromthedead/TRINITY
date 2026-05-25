"""Tests for localization dashboard."""

import pytest
from engine.tooling.localization.loc_dashboard import (
    LanguageProgress,
    TranslationStats,
    MissingString,
    LocalizationDashboard,
)


class TestLanguageProgress:
    """Tests for language progress."""

    def test_creation(self):
        """Test progress creation."""
        progress = LanguageProgress(
            language_code="fr",
            language_name="French",
            total_strings=100,
            translated_strings=50,
            approved_strings=40,
            needs_review=5,
            word_count=1000,
            translated_word_count=500,
        )
        assert progress.language_code == "fr"
        assert progress.total_strings == 100

    def test_completion_percent(self):
        """Test completion percentage calculation."""
        progress = LanguageProgress(
            language_code="fr",
            language_name="French",
            total_strings=100,
            translated_strings=50,
            approved_strings=0,
            needs_review=0,
            word_count=0,
            translated_word_count=0,
        )
        assert progress.completion_percent == 50.0

    def test_completion_percent_zero_total(self):
        """Test completion with zero total strings."""
        progress = LanguageProgress(
            language_code="fr",
            language_name="French",
            total_strings=0,
            translated_strings=0,
            approved_strings=0,
            needs_review=0,
            word_count=0,
            translated_word_count=0,
        )
        assert progress.completion_percent == 0.0

    def test_approval_percent(self):
        """Test approval percentage calculation."""
        progress = LanguageProgress(
            language_code="fr",
            language_name="French",
            total_strings=100,
            translated_strings=50,
            approved_strings=25,
            needs_review=0,
            word_count=0,
            translated_word_count=0,
        )
        assert progress.approval_percent == 50.0

    def test_word_completion_percent(self):
        """Test word count completion."""
        progress = LanguageProgress(
            language_code="fr",
            language_name="French",
            total_strings=100,
            translated_strings=50,
            approved_strings=0,
            needs_review=0,
            word_count=1000,
            translated_word_count=400,
        )
        assert progress.word_completion_percent == 40.0


class TestTranslationStats:
    """Tests for translation stats."""

    def test_default_values(self):
        """Test default stats values."""
        stats = TranslationStats()
        assert stats.total_strings == 0
        assert stats.total_words == 0
        assert stats.languages == 0

    def test_custom_values(self):
        """Test custom stats values."""
        stats = TranslationStats(
            total_strings=500,
            total_words=5000,
            languages=5,
            fully_translated=3,
            average_completion=80.0,
        )
        assert stats.total_strings == 500
        assert stats.fully_translated == 3


class TestMissingString:
    """Tests for missing string."""

    def test_creation(self):
        """Test missing string creation."""
        missing = MissingString(
            key="hello",
            source_text="Hello World",
            language="fr",
        )
        assert missing.key == "hello"
        assert missing.language == "fr"

    def test_word_count_auto(self):
        """Test automatic word count."""
        missing = MissingString(
            key="test",
            source_text="Hello World Test",
            language="fr",
        )
        assert missing.word_count == 3

    def test_word_count_provided(self):
        """Test provided word count."""
        missing = MissingString(
            key="test",
            source_text="Hello World",
            language="fr",
            word_count=10,
        )
        assert missing.word_count == 10


class TestLocalizationDashboard:
    """Tests for localization dashboard."""

    def setup_method(self):
        """Set up test dashboard."""
        self.dashboard = LocalizationDashboard()

    def test_creation(self):
        """Test dashboard creation."""
        assert self.dashboard is not None

    def test_get_stats_empty(self):
        """Test getting stats without data."""
        stats = self.dashboard.get_stats()
        assert stats.total_strings == 0

    def test_get_language_progress_none(self):
        """Test getting progress for nonexistent language."""
        progress = self.dashboard.get_language_progress("fr")
        assert progress is None

    def test_get_all_progress_empty(self):
        """Test getting all progress when empty."""
        progress = self.dashboard.get_all_progress()
        assert len(progress) == 0

    def test_get_missing_strings_empty(self):
        """Test getting missing strings when empty."""
        missing = self.dashboard.get_missing_strings()
        assert len(missing) == 0

    def test_get_missing_count(self):
        """Test getting missing count."""
        count = self.dashboard.get_missing_count()
        assert count == 0

    def test_get_status_summary(self):
        """Test getting status summary."""
        summary = self.dashboard.get_status_summary()
        assert "total_strings" in summary
        assert "languages" in summary
        assert "missing_count" in summary

    def test_is_complete(self):
        """Test completion check."""
        assert not self.dashboard.is_complete("fr")

    def test_get_priority_missing(self):
        """Test getting priority missing strings."""
        missing = self.dashboard.get_priority_missing("fr", 10)
        assert len(missing) == 0

    def test_export_text_report(self):
        """Test exporting text report."""
        report = self.dashboard.export_report("text")
        assert "LOCALIZATION DASHBOARD" in report
        assert "OVERALL STATISTICS" in report

    def test_export_json_report(self):
        """Test exporting JSON report."""
        report = self.dashboard.export_report("json")
        import json
        data = json.loads(report)
        assert "stats" in data
        assert "languages" in data

    def test_export_csv_report(self):
        """Test exporting CSV report."""
        report = self.dashboard.export_report("csv")
        assert "Language,Code,Completion" in report

    def test_set_filter(self):
        """Test setting filter."""
        self.dashboard.set_filter("language", "fr")
        # Filter is set (tested via get_missing_strings)

    def test_clear_filters(self):
        """Test clearing filters."""
        self.dashboard.set_filter("language", "fr")
        self.dashboard.clear_filters()

    def test_set_sort(self):
        """Test setting sort."""
        self.dashboard.set_sort("key", reverse=True)

    def test_get_progress_sorted(self):
        """Test getting sorted progress."""
        progress = self.dashboard.get_progress_sorted(by="completion")
        assert isinstance(progress, list)

    def test_get_missing_by_category(self):
        """Test getting missing by category."""
        counts = self.dashboard.get_missing_by_category("fr")
        assert isinstance(counts, dict)

    def test_refresh(self):
        """Test refresh method."""
        # Should not raise with no string table
        self.dashboard.refresh()


class TestDashboardWithMockData:
    """Tests with mock string table data."""

    def setup_method(self):
        """Set up dashboard with mock data."""
        self.dashboard = LocalizationDashboard()

        # Create mock string table manager
        class MockEntry:
            def __init__(self, key, source_text, translations=None):
                self.key = key
                self.source_text = source_text
                self.translations = translations or {}
                self.context = MockContext()
                self.source_file = ""
                self.needs_review = False
                self.is_approved = False

            def has_translation(self, lang):
                return lang in self.translations

        class MockContext:
            category = "UI"
            description = ""

        class MockTable:
            def __init__(self):
                self.entries = [
                    MockEntry("hello", "Hello", {"fr": "Bonjour"}),
                    MockEntry("bye", "Goodbye", {}),
                ]
                self._fallback_language = "en"

            def get_all_languages(self):
                return {"en", "fr"}

            def get_all_entries(self):
                return [("main", e) for e in self.entries]

        self.mock_table = MockTable()
        self.dashboard.set_string_table(self.mock_table)

    def test_refresh_with_data(self):
        """Test refresh with mock data."""
        self.dashboard.refresh()
        stats = self.dashboard.get_stats()
        assert stats.total_strings == 2

    def test_get_language_progress_with_data(self):
        """Test getting progress with data."""
        self.dashboard.refresh()
        progress = self.dashboard.get_language_progress("fr")
        assert progress is not None
        assert progress.total_strings == 2
        assert progress.translated_strings == 1

    def test_get_missing_with_data(self):
        """Test getting missing strings with data."""
        self.dashboard.refresh()
        missing = self.dashboard.get_missing_strings("fr")
        assert len(missing) == 1
        assert missing[0].key == "bye"

    def test_is_complete_with_data(self):
        """Test completion check with data."""
        self.dashboard.refresh()
        assert not self.dashboard.is_complete("fr")
