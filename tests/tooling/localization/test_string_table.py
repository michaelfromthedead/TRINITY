"""Tests for string table management."""

import pytest
from engine.tooling.localization.string_table import (
    PluralForm,
    PluralRule,
    StringContext,
    StringEntry,
    StringTable,
    StringTableManager,
)


class TestPluralForm:
    """Tests for plural form enum."""

    def test_all_forms_exist(self):
        """Test all plural forms are defined."""
        assert PluralForm.ZERO.value == "zero"
        assert PluralForm.ONE.value == "one"
        assert PluralForm.TWO.value == "two"
        assert PluralForm.FEW.value == "few"
        assert PluralForm.MANY.value == "many"
        assert PluralForm.OTHER.value == "other"


class TestPluralRule:
    """Tests for plural rules."""

    def test_english_rule(self):
        """Test English plural rule."""
        rule = PluralRule("en", "english")
        assert rule.get_form(1) == PluralForm.ONE
        assert rule.get_form(0) == PluralForm.OTHER
        assert rule.get_form(2) == PluralForm.OTHER
        assert rule.get_form(100) == PluralForm.OTHER

    def test_slavic_rule(self):
        """Test Slavic plural rule."""
        rule = PluralRule("ru", "slavic")
        assert rule.get_form(1) == PluralForm.ONE
        assert rule.get_form(2) == PluralForm.FEW
        assert rule.get_form(5) == PluralForm.MANY
        assert rule.get_form(21) == PluralForm.ONE
        assert rule.get_form(22) == PluralForm.FEW
        assert rule.get_form(25) == PluralForm.MANY

    def test_arabic_rule(self):
        """Test Arabic plural rule."""
        rule = PluralRule("ar", "arabic")
        assert rule.get_form(0) == PluralForm.ZERO
        assert rule.get_form(1) == PluralForm.ONE
        assert rule.get_form(2) == PluralForm.TWO
        assert rule.get_form(5) == PluralForm.FEW
        assert rule.get_form(15) == PluralForm.MANY

    def test_default_rule(self):
        """Test default rule fallback."""
        rule = PluralRule("xx", "unknown")
        assert rule.get_form(1) == PluralForm.ONE
        assert rule.get_form(2) == PluralForm.OTHER


class TestStringContext:
    """Tests for string context."""

    def test_default_values(self):
        """Test default context values."""
        ctx = StringContext()
        assert ctx.description == ""
        assert ctx.max_length == 0
        assert ctx.category == ""

    def test_custom_values(self):
        """Test custom context values."""
        ctx = StringContext(
            description="Button label",
            max_length=20,
            category="UI",
            tags=["button", "action"],
        )
        assert ctx.description == "Button label"
        assert ctx.max_length == 20
        assert "button" in ctx.tags


class TestStringEntry:
    """Tests for string entry."""

    def test_creation(self):
        """Test entry creation."""
        entry = StringEntry(key="test.key", source_text="Hello")
        assert entry.key == "test.key"
        assert entry.source_text == "Hello"

    def test_get_translation_existing(self):
        """Test getting existing translation."""
        entry = StringEntry(key="test", source_text="Hello")
        entry.translations["fr"] = "Bonjour"
        assert entry.get_translation("fr") == "Bonjour"

    def test_get_translation_missing(self):
        """Test getting missing translation falls back to source."""
        entry = StringEntry(key="test", source_text="Hello")
        assert entry.get_translation("fr") == "Hello"

    def test_set_translation(self):
        """Test setting translation."""
        entry = StringEntry(key="test", source_text="Hello")
        entry.set_translation("de", "Hallo")
        assert entry.translations["de"] == "Hallo"

    def test_get_plural(self):
        """Test getting plural form."""
        entry = StringEntry(key="test", source_text="item")
        entry.plurals["en"] = {"one": "item", "other": "items"}
        assert entry.get_plural("en", PluralForm.ONE) == "item"
        assert entry.get_plural("en", PluralForm.OTHER) == "items"

    def test_set_plural(self):
        """Test setting plural form."""
        entry = StringEntry(key="test", source_text="item")
        entry.set_plural("en", PluralForm.ONE, "item")
        entry.set_plural("en", PluralForm.OTHER, "items")
        assert entry.plurals["en"]["one"] == "item"

    def test_has_translation(self):
        """Test translation existence check."""
        entry = StringEntry(key="test", source_text="Hello")
        assert not entry.has_translation("fr")
        entry.set_translation("fr", "Bonjour")
        assert entry.has_translation("fr")

    def test_get_all_languages(self):
        """Test getting all languages."""
        entry = StringEntry(key="test", source_text="Hello")
        entry.set_translation("fr", "Bonjour")
        entry.set_translation("de", "Hallo")
        langs = entry.get_all_languages()
        assert "fr" in langs
        assert "de" in langs


class TestStringTable:
    """Tests for string table."""

    def setup_method(self):
        """Set up test table."""
        self.table = StringTable("main", "en")

    def test_creation(self):
        """Test table creation."""
        assert self.table.name == "main"
        assert self.table.source_language == "en"

    def test_add_entry(self):
        """Test adding entry."""
        entry = StringEntry(key="test", source_text="Hello")
        assert self.table.add_entry(entry)
        assert self.table.get_entry("test") == entry

    def test_add_duplicate_entry(self):
        """Test adding duplicate entry fails."""
        entry = StringEntry(key="test", source_text="Hello")
        self.table.add_entry(entry)
        assert not self.table.add_entry(entry)

    def test_update_entry(self):
        """Test updating entry."""
        entry = StringEntry(key="test", source_text="Hello")
        self.table.add_entry(entry)

        updated = StringEntry(key="test", source_text="Hi")
        assert self.table.update_entry(updated)
        assert self.table.get_entry("test").source_text == "Hi"

    def test_update_locked_entry(self):
        """Test updating locked entry fails."""
        entry = StringEntry(key="test", source_text="Hello", is_locked=True)
        self.table.add_entry(entry)

        updated = StringEntry(key="test", source_text="Hi")
        assert not self.table.update_entry(updated)

    def test_remove_entry(self):
        """Test removing entry."""
        entry = StringEntry(key="test", source_text="Hello")
        self.table.add_entry(entry)
        assert self.table.remove_entry("test")
        assert self.table.get_entry("test") is None

    def test_get_all_entries(self):
        """Test getting all entries."""
        self.table.add_entry(StringEntry(key="a", source_text="A"))
        self.table.add_entry(StringEntry(key="b", source_text="B"))
        entries = self.table.get_all_entries()
        assert len(entries) == 2

    def test_add_target_language(self):
        """Test adding target language."""
        self.table.add_target_language("fr")
        self.table.add_target_language("de")
        assert "fr" in self.table.target_languages
        assert "de" in self.table.target_languages

    def test_get_text_basic(self):
        """Test getting localized text."""
        entry = StringEntry(key="hello", source_text="Hello")
        entry.set_translation("fr", "Bonjour")
        self.table.add_entry(entry)

        assert self.table.get_text("hello", "en") == "Hello"
        assert self.table.get_text("hello", "fr") == "Bonjour"

    def test_get_text_with_format(self):
        """Test getting text with format arguments."""
        entry = StringEntry(key="greet", source_text="Hello, {name}!")
        entry.set_translation("fr", "Bonjour, {name}!")
        self.table.add_entry(entry)

        assert self.table.get_text("greet", "en", name="World") == "Hello, World!"
        assert self.table.get_text("greet", "fr", name="Monde") == "Bonjour, Monde!"

    def test_get_text_with_plural(self):
        """Test getting text with plural."""
        entry = StringEntry(key="items", source_text="item")
        entry.plurals["en"] = {"one": "{count} item", "other": "{count} items"}
        self.table.add_entry(entry)

        assert self.table.get_text("items", "en", count=1) == "{count} item"
        assert self.table.get_text("items", "en", count=5) == "{count} items"

    def test_find_entries(self):
        """Test searching entries."""
        self.table.add_entry(StringEntry(key="hello", source_text="Hello World"))
        self.table.add_entry(StringEntry(key="bye", source_text="Goodbye"))

        results = self.table.find_entries("hello")
        assert len(results) == 1

    def test_get_missing_translations(self):
        """Test getting missing translations."""
        entry1 = StringEntry(key="a", source_text="A")
        entry1.set_translation("fr", "A")
        entry2 = StringEntry(key="b", source_text="B")

        self.table.add_entry(entry1)
        self.table.add_entry(entry2)

        missing = self.table.get_missing_translations("fr")
        assert len(missing) == 1
        assert missing[0].key == "b"

    def test_export_import(self):
        """Test export and import."""
        self.table.add_entry(StringEntry(key="test", source_text="Test"))
        self.table.add_target_language("fr")

        exported = self.table.export_to_dict()

        new_table = StringTable("imported", "en")
        count = new_table.import_from_dict(exported)
        assert count == 1


class TestStringTableManager:
    """Tests for string table manager."""

    def setup_method(self):
        """Set up test manager."""
        self.manager = StringTableManager()

    def test_creation(self):
        """Test manager creation."""
        assert self.manager.active_language == "en"
        assert self.manager.fallback_language == "en"

    def test_create_table(self):
        """Test creating table."""
        table = self.manager.create_table("ui")
        assert table.name == "ui"
        assert self.manager.get_table("ui") == table

    def test_remove_table(self):
        """Test removing table."""
        self.manager.create_table("ui")
        assert self.manager.remove_table("ui")
        assert self.manager.get_table("ui") is None

    def test_get_text_from_table(self):
        """Test getting text from specific table."""
        table = self.manager.create_table("ui")
        table.add_entry(StringEntry(key="hello", source_text="Hello"))

        text = self.manager.get_text("hello", table_name="ui")
        assert text == "Hello"

    def test_get_text_search_all(self):
        """Test getting text searches all tables."""
        table = self.manager.create_table("ui")
        table.add_entry(StringEntry(key="hello", source_text="Hello"))

        text = self.manager.get_text("hello")
        assert text == "Hello"

    def test_active_language(self):
        """Test active language setting."""
        self.manager.active_language = "fr"
        assert self.manager.active_language == "fr"

    def test_get_all_languages(self):
        """Test getting all languages."""
        table = self.manager.create_table("ui")
        table.add_target_language("fr")
        table.add_target_language("de")

        langs = self.manager.get_all_languages()
        assert "en" in langs
        assert "fr" in langs
        assert "de" in langs

    def test_export_import_all(self):
        """Test exporting and importing all tables."""
        table = self.manager.create_table("ui")
        table.add_entry(StringEntry(key="test", source_text="Test"))

        exported = self.manager.export_all()

        new_manager = StringTableManager()
        count = new_manager.import_all(exported)
        assert count == 1
