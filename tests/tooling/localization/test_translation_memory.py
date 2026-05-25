"""Tests for translation memory."""

import pytest
from engine.tooling.localization.translation_memory import (
    TMEntry,
    TMMatch,
    TMMatchType,
    TranslationMemory,
    TranslationMemoryManager,
)


class TestTMEntry:
    """Tests for TM entry."""

    def test_creation(self):
        """Test entry creation."""
        entry = TMEntry(
            id=1,
            source_text="Hello",
            target_text="Bonjour",
            source_language="en",
            target_language="fr",
        )
        assert entry.id == 1
        assert entry.source_text == "Hello"
        assert entry.target_text == "Bonjour"

    def test_normalized_source(self):
        """Test source text normalization."""
        entry = TMEntry(
            id=1,
            source_text="  HELLO  World  ",
            target_text="Test",
            source_language="en",
            target_language="fr",
        )
        normalized = entry.get_source_normalized()
        assert normalized == "hello world"


class TestTMMatch:
    """Tests for TM match."""

    def test_exact_match(self):
        """Test exact match detection."""
        entry = TMEntry(1, "Hello", "Bonjour", "en", "fr")
        match = TMMatch(
            entry=entry,
            match_type=TMMatchType.EXACT,
            similarity=1.0,
            source_text="Hello",
        )
        assert match.is_exact_match()

    def test_fuzzy_match(self):
        """Test fuzzy match detection."""
        entry = TMEntry(1, "Hello", "Bonjour", "en", "fr")
        match = TMMatch(
            entry=entry,
            match_type=TMMatchType.FUZZY,
            similarity=0.8,
            source_text="Hello World",
        )
        assert not match.is_exact_match()


class TestTranslationMemory:
    """Tests for translation memory."""

    def setup_method(self):
        """Set up test memory."""
        self.tm = TranslationMemory("en", "fr")

    def test_creation(self):
        """Test memory creation."""
        assert self.tm.source_language == "en"
        assert self.tm.target_language == "fr"
        assert self.tm.entry_count == 0

    def test_add_entry(self):
        """Test adding entry."""
        entry = self.tm.add_entry("Hello", "Bonjour")
        assert entry.id == 1
        assert self.tm.entry_count == 1

    def test_add_entry_with_context(self):
        """Test adding entry with context."""
        entry = self.tm.add_entry(
            "Hello",
            "Bonjour",
            context="greeting",
            domain="UI",
        )
        assert entry.context == "greeting"
        assert entry.domain == "UI"

    def test_update_entry(self):
        """Test updating entry."""
        entry = self.tm.add_entry("Hello", "Bonjour")
        assert self.tm.update_entry(entry.id, "Salut")
        assert self.tm.get_entry(entry.id).target_text == "Salut"

    def test_remove_entry(self):
        """Test removing entry."""
        entry = self.tm.add_entry("Hello", "Bonjour")
        assert self.tm.remove_entry(entry.id)
        assert self.tm.get_entry(entry.id) is None
        assert self.tm.entry_count == 0

    def test_find_exact(self):
        """Test exact match lookup."""
        self.tm.add_entry("Hello", "Bonjour")
        entry = self.tm.find_exact("Hello")
        assert entry is not None
        assert entry.target_text == "Bonjour"

    def test_find_exact_normalized(self):
        """Test exact match with different casing."""
        self.tm.add_entry("Hello World", "Bonjour le monde")
        entry = self.tm.find_exact("  hello  world  ")
        assert entry is not None

    def test_find_exact_not_found(self):
        """Test exact match not found."""
        self.tm.add_entry("Hello", "Bonjour")
        entry = self.tm.find_exact("Goodbye")
        assert entry is None

    def test_find_fuzzy(self):
        """Test fuzzy match lookup."""
        self.tm.add_entry("Hello World", "Bonjour le monde")
        matches = self.tm.find_fuzzy("Hello there", min_similarity=0.3)
        assert len(matches) > 0

    def test_find_fuzzy_exact_first(self):
        """Test fuzzy returns exact match first."""
        self.tm.add_entry("Hello", "Bonjour")
        self.tm.add_entry("Hello World", "Bonjour le monde")

        matches = self.tm.find_fuzzy("Hello")
        assert matches[0].match_type == TMMatchType.EXACT

    def test_find_fuzzy_with_context(self):
        """Test fuzzy match with context boost."""
        self.tm.add_entry("Click", "Cliquer", context="button")
        self.tm.add_entry("Click", "Clique", context="action")

        matches = self.tm.find_fuzzy("Click", context="button")
        assert len(matches) > 0

    def test_find_fuzzy_max_results(self):
        """Test fuzzy match max results."""
        for i in range(10):
            self.tm.add_entry(f"Test {i}", f"Teste {i}")

        matches = self.tm.find_fuzzy("Test 0", max_results=3)
        assert len(matches) <= 3

    def test_find_by_context(self):
        """Test finding by context."""
        self.tm.add_entry("Hello", "Bonjour", context="greeting")
        self.tm.add_entry("Goodbye", "Au revoir", context="farewell")

        entries = self.tm.find_by_context("greeting")
        assert len(entries) == 1
        assert entries[0].source_text == "Hello"

    def test_export_import(self):
        """Test export and import."""
        self.tm.add_entry("Hello", "Bonjour")
        self.tm.add_entry("Goodbye", "Au revoir")

        exported = self.tm.export_to_dict()

        new_tm = TranslationMemory("en", "fr")
        count = new_tm.import_from_dict(exported)
        assert count == 2
        assert new_tm.entry_count == 2


class TestTranslationMemoryManager:
    """Tests for TM manager."""

    def setup_method(self):
        """Set up test manager."""
        self.manager = TranslationMemoryManager("en")

    def test_creation(self):
        """Test manager creation."""
        assert self.manager.source_language == "en"

    def test_get_or_create_memory(self):
        """Test getting or creating memory."""
        tm = self.manager.get_or_create_memory("fr")
        assert tm is not None
        assert tm.source_language == "en"
        assert tm.target_language == "fr"

        # Same memory returned
        tm2 = self.manager.get_or_create_memory("fr")
        assert tm2 is tm

    def test_get_memory_nonexistent(self):
        """Test getting nonexistent memory."""
        tm = self.manager.get_memory("fr")
        assert tm is None

    def test_find_translation(self):
        """Test finding translation."""
        tm = self.manager.get_or_create_memory("fr")
        tm.add_entry("Hello", "Bonjour")

        match = self.manager.find_translation("Hello", "fr")
        assert match is not None
        assert match.entry.target_text == "Bonjour"

    def test_add_translation(self):
        """Test adding translation."""
        entry = self.manager.add_translation("Hello", "Bonjour", "fr")
        assert entry is not None
        assert entry.target_text == "Bonjour"

    def test_get_statistics(self):
        """Test getting statistics."""
        self.manager.add_translation("Hello", "Bonjour", "fr")
        self.manager.add_translation("Goodbye", "Auf Wiedersehen", "de")

        stats = self.manager.get_statistics()
        assert stats["total_entries"] == 2
        assert len(stats["language_pairs"]) == 2

    def test_suggest_translations(self):
        """Test translation suggestions."""
        tm = self.manager.get_or_create_memory("fr")
        tm.add_entry("Hello World", "Bonjour le monde")
        tm.add_entry("Hello Friend", "Bonjour ami")

        suggestions = self.manager.suggest_translations("Hello there", "fr")
        assert len(suggestions) > 0

    def test_export_import_all(self):
        """Test exporting and importing all memories."""
        self.manager.add_translation("Hello", "Bonjour", "fr")
        self.manager.add_translation("Hello", "Hallo", "de")

        exported = self.manager.export_all()

        new_manager = TranslationMemoryManager()
        count = new_manager.import_all(exported)
        assert count == 2
