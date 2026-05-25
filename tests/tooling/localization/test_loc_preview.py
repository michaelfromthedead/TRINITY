"""Tests for localization preview."""

import pytest
from engine.tooling.localization.loc_preview import (
    PreviewMode,
    PseudoLocSettings,
    PseudoLocalizer,
    LocalizationPreview,
    LanguageSwitcher,
)


class TestPseudoLocSettings:
    """Tests for pseudo-localization settings."""

    def test_default_values(self):
        """Test default settings."""
        settings = PseudoLocSettings()
        assert settings.enable_accents
        assert settings.expansion_factor == 1.3
        assert settings.add_brackets

    def test_custom_values(self):
        """Test custom settings."""
        settings = PseudoLocSettings(
            enable_accents=False,
            expansion_factor=1.5,
            add_brackets=False,
        )
        assert not settings.enable_accents
        assert settings.expansion_factor == 1.5


class TestPseudoLocalizer:
    """Tests for pseudo localizer."""

    def setup_method(self):
        """Set up test localizer."""
        self.localizer = PseudoLocalizer()

    def test_creation(self):
        """Test localizer creation."""
        assert self.localizer.settings is not None

    def test_transform_basic(self):
        """Test basic transformation."""
        result = self.localizer.transform("Hello")
        assert result != "Hello"
        assert len(result) > len("Hello")

    def test_transform_with_brackets(self):
        """Test brackets are added."""
        localizer = PseudoLocalizer(PseudoLocSettings(
            add_brackets=True,
            bracket_style="[! !]",
        ))
        result = localizer.transform("Hello")
        assert "[!" in result
        assert "!]" in result

    def test_transform_without_brackets(self):
        """Test without brackets."""
        localizer = PseudoLocalizer(PseudoLocSettings(add_brackets=False))
        result = localizer.transform("Hello")
        assert "[!" not in result

    def test_transform_expansion(self):
        """Test text expansion."""
        localizer = PseudoLocalizer(PseudoLocSettings(
            expansion_factor=2.0,
            add_brackets=False,
            enable_accents=False,
        ))
        result = localizer.transform("Hello World")
        assert len(result) > len("Hello World")

    def test_transform_no_expansion(self):
        """Test without expansion."""
        localizer = PseudoLocalizer(PseudoLocSettings(
            expansion_factor=1.0,
            add_brackets=False,
            enable_accents=False,
        ))
        result = localizer.transform("Hello")
        assert result == "Hello"

    def test_preserve_placeholders(self):
        """Test placeholders are preserved."""
        result = self.localizer.transform("Hello {name}!")
        assert "{name}" in result

    def test_preserve_multiple_placeholders(self):
        """Test multiple placeholders preserved."""
        result = self.localizer.transform("{greeting} {name}, {message}")
        assert "{greeting}" in result
        assert "{name}" in result
        assert "{message}" in result

    def test_rtl_simulation(self):
        """Test RTL simulation."""
        localizer = PseudoLocalizer(PseudoLocSettings(
            simulate_rtl=True,
            add_brackets=False,
        ))
        result = localizer.transform("Hello")
        # Should have RTL markers
        assert "\u200F" in result

    def test_set_settings(self):
        """Test updating settings."""
        new_settings = PseudoLocSettings(add_brackets=False)
        self.localizer.set_settings(new_settings)
        assert not self.localizer.settings.add_brackets


class TestLocalizationPreview:
    """Tests for localization preview."""

    def setup_method(self):
        """Set up test preview."""
        self.strings = {
            ("hello", "en"): "Hello",
            ("hello", "fr"): "Bonjour",
            ("bye", "en"): "Goodbye",
        }

        def string_source(key, lang):
            return self.strings.get((key, lang))

        self.preview = LocalizationPreview(string_source)

    def test_creation(self):
        """Test preview creation."""
        assert self.preview.mode == PreviewMode.NORMAL
        assert self.preview.current_language == "en"

    def test_normal_mode(self):
        """Test normal preview mode."""
        self.preview.mode = PreviewMode.NORMAL
        text = self.preview.get_text("hello")
        assert text == "Hello"

    def test_normal_mode_translated(self):
        """Test normal mode with translation."""
        self.preview.mode = PreviewMode.NORMAL
        self.preview.current_language = "fr"
        text = self.preview.get_text("hello")
        assert text == "Bonjour"

    def test_keys_only_mode(self):
        """Test keys only mode."""
        self.preview.mode = PreviewMode.KEYS_ONLY
        text = self.preview.get_text("hello")
        assert text == "[hello]"

    def test_pseudo_loc_mode(self):
        """Test pseudo-localization mode."""
        self.preview.mode = PreviewMode.PSEUDO_LOC
        text = self.preview.get_text("hello")
        assert text != "Hello"

    def test_missing_string(self):
        """Test missing string handling."""
        self.preview.mode = PreviewMode.NORMAL
        text = self.preview.get_text("nonexistent")
        assert "[MISSING:" in text

    def test_custom_missing_marker(self):
        """Test custom missing marker."""
        self.preview.set_missing_marker("???{key}???")
        text = self.preview.get_text("nonexistent")
        assert "???nonexistent???" in text

    def test_fallback_to_source(self):
        """Test fallback to source language."""
        self.preview.current_language = "de"  # Not available
        text = self.preview.get_text("hello")
        assert text == "Hello"  # Falls back to en

    def test_format_arguments(self):
        """Test format argument substitution."""
        self.strings[("greet", "en")] = "Hello, {name}!"
        text = self.preview.get_text("greet", name="World")
        assert text == "Hello, World!"

    def test_long_text_mode(self):
        """Test long text mode."""
        self.preview.mode = PreviewMode.LONG_TEXT
        text = self.preview.get_text("hello")
        assert "Hello" in text
        # Text should be doubled
        assert text.count("Hello") == 2

    def test_get_all_missing(self):
        """Test getting all missing strings."""
        missing = self.preview.get_all_missing(["hello", "bye", "nonexistent"])
        assert "nonexistent" in missing

    def test_highlight_callback(self):
        """Test highlight callback."""
        highlighted = []

        def callback(key, is_missing):
            highlighted.append((key, is_missing))

        self.preview.on_highlight(callback)
        self.preview.get_text("hello")
        self.preview.get_text("nonexistent")

        assert ("hello", False) in highlighted
        assert ("nonexistent", True) in highlighted


class TestLanguageSwitcher:
    """Tests for language switcher."""

    def setup_method(self):
        """Set up test switcher."""
        self.switcher = LanguageSwitcher()

    def test_creation(self):
        """Test switcher creation."""
        assert self.switcher.current_language == "en"

    def test_set_available_languages(self):
        """Test setting available languages."""
        self.switcher.set_available_languages(["en", "fr", "de"])
        assert "fr" in self.switcher.available_languages

    def test_add_language(self):
        """Test adding language."""
        self.switcher.add_language("ja", "Japanese")
        assert "ja" in self.switcher.available_languages
        assert self.switcher.get_language_name("ja") == "Japanese"

    def test_remove_language(self):
        """Test removing language."""
        self.switcher.set_available_languages(["en", "fr", "de"])
        self.switcher.remove_language("de")
        assert "de" not in self.switcher.available_languages

    def test_get_language_name(self):
        """Test getting language name."""
        assert self.switcher.get_language_name("en") == "English"
        assert self.switcher.get_language_name("fr") == "Francais"

    def test_get_unknown_language_name(self):
        """Test getting unknown language name."""
        name = self.switcher.get_language_name("xx")
        assert name == "XX"

    def test_switch_to(self):
        """Test switching language."""
        self.switcher.set_available_languages(["en", "fr"])
        assert self.switcher.switch_to("fr")
        assert self.switcher.current_language == "fr"

    def test_switch_to_unavailable(self):
        """Test switching to unavailable language."""
        self.switcher.set_available_languages(["en", "fr"])
        assert not self.switcher.switch_to("de")
        assert self.switcher.current_language == "en"

    def test_switch_next(self):
        """Test switching to next language."""
        self.switcher.set_available_languages(["en", "fr", "de"])
        next_lang = self.switcher.switch_next()
        assert next_lang == "fr"
        assert self.switcher.current_language == "fr"

    def test_switch_next_wraps(self):
        """Test next wraps to first."""
        self.switcher.set_available_languages(["en", "fr"])
        self.switcher.switch_to("fr")
        next_lang = self.switcher.switch_next()
        assert next_lang == "en"

    def test_switch_previous(self):
        """Test switching to previous language."""
        self.switcher.set_available_languages(["en", "fr", "de"])
        self.switcher.switch_to("de")
        prev_lang = self.switcher.switch_previous()
        assert prev_lang == "fr"

    def test_switch_previous_wraps(self):
        """Test previous wraps to last."""
        self.switcher.set_available_languages(["en", "fr", "de"])
        prev_lang = self.switcher.switch_previous()
        assert prev_lang == "de"

    def test_callback(self):
        """Test language change callback."""
        changes = []

        def on_change(lang):
            changes.append(lang)

        self.switcher.set_available_languages(["en", "fr"])
        self.switcher.on_language_change(on_change)
        self.switcher.switch_to("fr")

        assert "fr" in changes

    def test_link_preview(self):
        """Test linking to preview."""
        def string_source(key, lang):
            return "test"

        preview = LocalizationPreview(string_source)
        self.switcher.set_available_languages(["en", "fr"])
        self.switcher.link_preview(preview)
        self.switcher.switch_to("fr")

        assert preview.current_language == "fr"

    def test_get_selection_items(self):
        """Test getting selection items."""
        self.switcher.set_available_languages(["en", "fr"])
        items = self.switcher.get_selection_items()
        assert len(items) == 2
        assert ("en", "English") in items
        assert ("fr", "Francais") in items
