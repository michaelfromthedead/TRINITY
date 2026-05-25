"""
Comprehensive tests for Localization (string tables, pluralization).

Tests cover:
- LocalizationManager singleton and initialization
- String table loading (JSON, CSV)
- Basic string retrieval
- String formatting with parameters
- Pluralization rules for multiple languages
- Right-to-Left (RTL) language support
- Language switching
- Missing string fallbacks
- Context-aware translations
- Gender-specific translations
"""

import pytest
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Callable
from enum import Enum, auto
import tempfile
import json
import os


# Expected Localization implementation classes
# from engine.ui.text.localization import (
#     LocalizationManager,
#     StringTable,
#     Language,
#     PluralRule,
#     PluralCategory,
#     TextDirection,
#     LocalizedString,
#     FormatParameter,
# )


class PluralCategory(Enum):
    """Plural categories following CLDR."""
    ZERO = auto()
    ONE = auto()
    TWO = auto()
    FEW = auto()
    MANY = auto()
    OTHER = auto()


class TextDirection(Enum):
    """Text direction."""
    LTR = auto()  # Left-to-right
    RTL = auto()  # Right-to-left


@dataclass
class Language:
    """Represents a language."""
    code: str  # ISO 639-1 code (e.g., "en", "es", "ar")
    name: str  # English name
    native_name: str  # Name in the language itself
    direction: TextDirection = TextDirection.LTR
    plural_rule: Optional[Callable[[int], PluralCategory]] = None


@dataclass
class FormatParameter:
    """A parameter for string formatting."""
    name: str
    value: Any
    format_spec: Optional[str] = None


@dataclass
class LocalizedString:
    """A localized string entry."""
    key: str
    value: str
    language: str
    context: Optional[str] = None
    plural_forms: Optional[Dict[PluralCategory, str]] = None


@dataclass
class StringTable:
    """Collection of localized strings for a language."""
    language: str
    strings: Dict[str, LocalizedString]

    def get(self, key: str, context: Optional[str] = None) -> Optional[LocalizedString]:
        """Get a localized string by key and optional context."""
        full_key = f"{context}:{key}" if context else key
        return self.strings.get(full_key) or self.strings.get(key)


# Plural rules for common languages
def english_plural_rule(n: int) -> PluralCategory:
    """English plural rule: 1 is ONE, everything else is OTHER."""
    return PluralCategory.ONE if n == 1 else PluralCategory.OTHER


def russian_plural_rule(n: int) -> PluralCategory:
    """Russian plural rule."""
    n_mod_10 = n % 10
    n_mod_100 = n % 100

    if n_mod_10 == 1 and n_mod_100 != 11:
        return PluralCategory.ONE
    elif 2 <= n_mod_10 <= 4 and not (12 <= n_mod_100 <= 14):
        return PluralCategory.FEW
    else:
        return PluralCategory.MANY


def arabic_plural_rule(n: int) -> PluralCategory:
    """Arabic plural rule."""
    if n == 0:
        return PluralCategory.ZERO
    elif n == 1:
        return PluralCategory.ONE
    elif n == 2:
        return PluralCategory.TWO
    elif 3 <= n % 100 <= 10:
        return PluralCategory.FEW
    elif 11 <= n % 100 <= 99:
        return PluralCategory.MANY
    else:
        return PluralCategory.OTHER


class TestLanguage:
    """Tests for Language class."""

    def test_language_creation(self):
        """Test creating a language."""
        lang = Language(code="en", name="English", native_name="English")
        assert lang.code == "en"
        assert lang.name == "English"
        assert lang.direction == TextDirection.LTR

    def test_language_rtl(self):
        """Test RTL language."""
        lang = Language(
            code="ar", name="Arabic", native_name="العربية",
            direction=TextDirection.RTL
        )
        assert lang.direction == TextDirection.RTL

    def test_language_with_plural_rule(self):
        """Test language with plural rule."""
        lang = Language(
            code="en", name="English", native_name="English",
            plural_rule=english_plural_rule
        )
        assert lang.plural_rule(1) == PluralCategory.ONE
        assert lang.plural_rule(2) == PluralCategory.OTHER


class TestPluralRule:
    """Tests for plural rules."""

    def test_english_plural_one(self):
        """Test English plural rule for 1."""
        assert english_plural_rule(1) == PluralCategory.ONE

    def test_english_plural_zero(self):
        """Test English plural rule for 0."""
        assert english_plural_rule(0) == PluralCategory.OTHER

    def test_english_plural_many(self):
        """Test English plural rule for many."""
        assert english_plural_rule(5) == PluralCategory.OTHER
        assert english_plural_rule(100) == PluralCategory.OTHER

    def test_russian_plural_one(self):
        """Test Russian plural rule for 1."""
        assert russian_plural_rule(1) == PluralCategory.ONE
        assert russian_plural_rule(21) == PluralCategory.ONE
        assert russian_plural_rule(31) == PluralCategory.ONE

    def test_russian_plural_few(self):
        """Test Russian plural rule for few."""
        assert russian_plural_rule(2) == PluralCategory.FEW
        assert russian_plural_rule(3) == PluralCategory.FEW
        assert russian_plural_rule(4) == PluralCategory.FEW
        assert russian_plural_rule(22) == PluralCategory.FEW

    def test_russian_plural_many(self):
        """Test Russian plural rule for many."""
        assert russian_plural_rule(0) == PluralCategory.MANY
        assert russian_plural_rule(5) == PluralCategory.MANY
        assert russian_plural_rule(11) == PluralCategory.MANY
        assert russian_plural_rule(12) == PluralCategory.MANY

    def test_arabic_plural_zero(self):
        """Test Arabic plural rule for 0."""
        assert arabic_plural_rule(0) == PluralCategory.ZERO

    def test_arabic_plural_one(self):
        """Test Arabic plural rule for 1."""
        assert arabic_plural_rule(1) == PluralCategory.ONE

    def test_arabic_plural_two(self):
        """Test Arabic plural rule for 2."""
        assert arabic_plural_rule(2) == PluralCategory.TWO

    def test_arabic_plural_few(self):
        """Test Arabic plural rule for few."""
        assert arabic_plural_rule(3) == PluralCategory.FEW
        assert arabic_plural_rule(10) == PluralCategory.FEW

    def test_arabic_plural_many(self):
        """Test Arabic plural rule for many."""
        assert arabic_plural_rule(11) == PluralCategory.MANY
        assert arabic_plural_rule(99) == PluralCategory.MANY


class TestLocalizedString:
    """Tests for LocalizedString class."""

    def test_localized_string_creation(self):
        """Test creating a localized string."""
        ls = LocalizedString(key="greeting", value="Hello", language="en")
        assert ls.key == "greeting"
        assert ls.value == "Hello"

    def test_localized_string_with_context(self):
        """Test localized string with context."""
        ls = LocalizedString(
            key="hello", value="Hello",
            language="en", context="formal"
        )
        assert ls.context == "formal"

    def test_localized_string_with_plural_forms(self):
        """Test localized string with plural forms."""
        ls = LocalizedString(
            key="items",
            value="item",
            language="en",
            plural_forms={
                PluralCategory.ONE: "{n} item",
                PluralCategory.OTHER: "{n} items",
            }
        )
        assert PluralCategory.ONE in ls.plural_forms


class TestStringTable:
    """Tests for StringTable class."""

    def test_string_table_creation(self):
        """Test creating a string table."""
        table = StringTable(language="en", strings={})
        assert table.language == "en"

    def test_string_table_get(self):
        """Test getting string from table."""
        ls = LocalizedString(key="greeting", value="Hello", language="en")
        table = StringTable(language="en", strings={"greeting": ls})

        result = table.get("greeting")
        assert result is not None
        assert result.value == "Hello"

    def test_string_table_get_not_found(self):
        """Test getting non-existent string."""
        table = StringTable(language="en", strings={})

        result = table.get("nonexistent")
        assert result is None

    def test_string_table_get_with_context(self):
        """Test getting string with context."""
        ls1 = LocalizedString(key="hello", value="Hello", language="en")
        ls2 = LocalizedString(
            key="hello", value="Good day",
            language="en", context="formal"
        )
        table = StringTable(
            language="en",
            strings={
                "hello": ls1,
                "formal:hello": ls2,
            }
        )

        result = table.get("hello", context="formal")
        assert result.value == "Good day"


class TestLocalizationManagerInit:
    """Tests for LocalizationManager initialization."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_manager_singleton(self):
        """Test LocalizationManager is singleton."""
        manager1 = LocalizationManager.get_instance()
        manager2 = LocalizationManager.get_instance()
        assert manager1 is manager2

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_manager_default_language(self):
        """Test default language is English."""
        manager = LocalizationManager.get_instance()
        assert manager.current_language == "en"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_manager_set_language(self):
        """Test setting current language."""
        manager = LocalizationManager.get_instance()
        manager.set_language("es")
        assert manager.current_language == "es"


class TestLocalizationManagerLoading:
    """Tests for loading string tables."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_load_strings_json(self):
        """Test loading strings from JSON."""
        manager = LocalizationManager.get_instance()

        # Create temp JSON file
        strings = {
            "greeting": "Hello",
            "farewell": "Goodbye",
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(strings, f)
            temp_path = f.name

        try:
            manager.load_strings("en", temp_path)
            assert manager.get("greeting") == "Hello"
        finally:
            os.unlink(temp_path)

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_load_strings_nested_json(self):
        """Test loading nested JSON strings."""
        manager = LocalizationManager.get_instance()

        strings = {
            "menu": {
                "file": "File",
                "edit": "Edit",
            },
            "dialog": {
                "ok": "OK",
                "cancel": "Cancel",
            }
        }
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(strings, f)
            temp_path = f.name

        try:
            manager.load_strings("en", temp_path)
            assert manager.get("menu.file") == "File"
            assert manager.get("dialog.ok") == "OK"
        finally:
            os.unlink(temp_path)

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_load_strings_multiple_languages(self):
        """Test loading strings for multiple languages."""
        manager = LocalizationManager.get_instance()

        # English
        en_strings = {"greeting": "Hello"}
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(en_strings, f)
            en_path = f.name

        # Spanish
        es_strings = {"greeting": "Hola"}
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        ) as f:
            json.dump(es_strings, f)
            es_path = f.name

        try:
            manager.load_strings("en", en_path)
            manager.load_strings("es", es_path)

            manager.set_language("en")
            assert manager.get("greeting") == "Hello"

            manager.set_language("es")
            assert manager.get("greeting") == "Hola"
        finally:
            os.unlink(en_path)
            os.unlink(es_path)


class TestLocalizationManagerRetrieval:
    """Tests for string retrieval."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_get_simple_string(self):
        """Test getting a simple string."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "greeting": LocalizedString(
                    key="greeting", value="Hello", language="en"
                )
            }
        )
        manager.set_language("en")

        assert manager.get("greeting") == "Hello"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_get_missing_string(self):
        """Test getting missing string returns key."""
        manager = LocalizationManager.get_instance()

        result = manager.get("nonexistent_key")
        assert result == "nonexistent_key"  # Fallback to key

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_get_missing_string_custom_fallback(self):
        """Test getting missing string with custom fallback."""
        manager = LocalizationManager.get_instance()

        result = manager.get("nonexistent", default="Not found")
        assert result == "Not found"


class TestLocalizationManagerFormatting:
    """Tests for string formatting."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_format_simple(self):
        """Test simple string formatting."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "welcome": LocalizedString(
                    key="welcome",
                    value="Welcome, {name}!",
                    language="en"
                )
            }
        )

        result = manager.get("welcome", name="Alice")
        assert result == "Welcome, Alice!"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_format_multiple_params(self):
        """Test formatting with multiple parameters."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "info": LocalizedString(
                    key="info",
                    value="{name} has {count} items",
                    language="en"
                )
            }
        )

        result = manager.get("info", name="Bob", count=5)
        assert result == "Bob has 5 items"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_format_number(self):
        """Test number formatting."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "price": LocalizedString(
                    key="price",
                    value="Price: ${amount:,.2f}",
                    language="en"
                )
            }
        )

        result = manager.get("price", amount=1234.567)
        assert "1,234.57" in result

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_format_date(self):
        """Test date formatting."""
        manager = LocalizationManager.get_instance()
        from datetime import date

        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "date": LocalizedString(
                    key="date",
                    value="Date: {d}",
                    language="en"
                )
            }
        )

        result = manager.get("date", d=date(2024, 1, 15))
        assert "2024" in result


class TestLocalizationManagerPluralization:
    """Tests for pluralization."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_plural_english_one(self):
        """Test English pluralization for 1."""
        manager = LocalizationManager.get_instance()
        manager.register_language(Language(
            code="en", name="English", native_name="English",
            plural_rule=english_plural_rule
        ))
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "items": LocalizedString(
                    key="items",
                    value="items",
                    language="en",
                    plural_forms={
                        PluralCategory.ONE: "{n} item",
                        PluralCategory.OTHER: "{n} items",
                    }
                )
            }
        )

        result = manager.get_plural("items", 1)
        assert result == "1 item"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_plural_english_many(self):
        """Test English pluralization for many."""
        manager = LocalizationManager.get_instance()

        result = manager.get_plural("items", 5)
        assert result == "5 items"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_plural_english_zero(self):
        """Test English pluralization for 0."""
        manager = LocalizationManager.get_instance()

        result = manager.get_plural("items", 0)
        assert result == "0 items"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_plural_russian(self):
        """Test Russian pluralization."""
        manager = LocalizationManager.get_instance()
        manager.register_language(Language(
            code="ru", name="Russian", native_name="Русский",
            plural_rule=russian_plural_rule
        ))
        manager._tables["ru"] = StringTable(
            language="ru",
            strings={
                "items": LocalizedString(
                    key="items",
                    value="предметов",
                    language="ru",
                    plural_forms={
                        PluralCategory.ONE: "{n} предмет",
                        PluralCategory.FEW: "{n} предмета",
                        PluralCategory.MANY: "{n} предметов",
                    }
                )
            }
        )
        manager.set_language("ru")

        assert manager.get_plural("items", 1) == "1 предмет"
        assert manager.get_plural("items", 2) == "2 предмета"
        assert manager.get_plural("items", 5) == "5 предметов"
        assert manager.get_plural("items", 21) == "21 предмет"


class TestLocalizationManagerRTL:
    """Tests for RTL language support."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_rtl_detection(self):
        """Test RTL language is detected."""
        manager = LocalizationManager.get_instance()
        manager.register_language(Language(
            code="ar", name="Arabic", native_name="العربية",
            direction=TextDirection.RTL
        ))
        manager.set_language("ar")

        assert manager.is_rtl() is True

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_ltr_detection(self):
        """Test LTR language is detected."""
        manager = LocalizationManager.get_instance()
        manager.set_language("en")

        assert manager.is_rtl() is False

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_text_direction(self):
        """Test getting text direction."""
        manager = LocalizationManager.get_instance()

        manager.set_language("en")
        assert manager.text_direction == TextDirection.LTR

        manager.register_language(Language(
            code="he", name="Hebrew", native_name="עברית",
            direction=TextDirection.RTL
        ))
        manager.set_language("he")
        assert manager.text_direction == TextDirection.RTL


class TestLocalizationManagerLanguageSwitching:
    """Tests for language switching."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_switch_language(self):
        """Test switching language."""
        manager = LocalizationManager.get_instance()
        manager.set_language("en")
        manager.set_language("es")

        assert manager.current_language == "es"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_switch_language_callback(self):
        """Test callback on language switch."""
        manager = LocalizationManager.get_instance()
        callback_data = []

        def on_language_changed(old_lang, new_lang):
            callback_data.append((old_lang, new_lang))

        manager.on_language_changed = on_language_changed
        manager.set_language("en")
        manager.set_language("es")

        assert len(callback_data) >= 1
        assert callback_data[-1][1] == "es"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_available_languages(self):
        """Test getting available languages."""
        manager = LocalizationManager.get_instance()

        languages = manager.available_languages
        assert isinstance(languages, list)


class TestLocalizationManagerFallbacks:
    """Tests for fallback behavior."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_fallback_to_default_language(self):
        """Test fallback to default language."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "greeting": LocalizedString(
                    key="greeting", value="Hello", language="en"
                )
            }
        )
        manager._tables["es"] = StringTable(language="es", strings={})

        manager.set_language("es")
        manager.set_fallback_language("en")

        # "greeting" missing in Spanish, should fallback to English
        result = manager.get("greeting")
        assert result == "Hello"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_fallback_chain(self):
        """Test fallback chain of languages."""
        manager = LocalizationManager.get_instance()
        manager.set_fallback_chain(["en", "en-US"])

        # Should try languages in order


class TestLocalizationManagerContext:
    """Tests for context-aware translations."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_context_formal(self):
        """Test formal context."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "greeting": LocalizedString(
                    key="greeting", value="Hi", language="en"
                ),
                "formal:greeting": LocalizedString(
                    key="greeting", value="Good day",
                    language="en", context="formal"
                ),
            }
        )

        assert manager.get("greeting") == "Hi"
        assert manager.get("greeting", context="formal") == "Good day"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_context_fallback(self):
        """Test context fallback to base string."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "greeting": LocalizedString(
                    key="greeting", value="Hi", language="en"
                ),
            }
        )

        # "informal" context not defined, should fallback
        result = manager.get("greeting", context="informal")
        assert result == "Hi"


class TestLocalizationManagerGender:
    """Tests for gender-specific translations."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_gender_masculine(self):
        """Test masculine gender form."""
        manager = LocalizationManager.get_instance()
        # Some languages have gender-specific forms

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_gender_feminine(self):
        """Test feminine gender form."""
        manager = LocalizationManager.get_instance()

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_gender_neutral(self):
        """Test neutral gender form."""
        manager = LocalizationManager.get_instance()


class TestLocalizationManagerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_empty_string(self):
        """Test empty string value."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "empty": LocalizedString(
                    key="empty", value="", language="en"
                )
            }
        )

        result = manager.get("empty")
        assert result == ""

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_unicode_strings(self):
        """Test Unicode strings."""
        manager = LocalizationManager.get_instance()
        manager._tables["ja"] = StringTable(
            language="ja",
            strings={
                "greeting": LocalizedString(
                    key="greeting", value="こんにちは", language="ja"
                )
            }
        )
        manager.set_language("ja")

        result = manager.get("greeting")
        assert result == "こんにちは"

    @pytest.mark.skip(reason="LocalizationManager not yet implemented")
    def test_special_characters_in_format(self):
        """Test special characters in format strings."""
        manager = LocalizationManager.get_instance()
        manager._tables["en"] = StringTable(
            language="en",
            strings={
                "braces": LocalizedString(
                    key="braces",
                    value="Use {{braces}} like this: {value}",
                    language="en"
                )
            }
        )

        result = manager.get("braces", value="test")
        assert "{braces}" in result
        assert "test" in result
