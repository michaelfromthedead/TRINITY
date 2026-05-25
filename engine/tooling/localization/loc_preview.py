"""
Localization preview and testing tools.

Provides in-game preview with language switching and pseudo-localization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import random


class PreviewMode(Enum):
    """Preview display modes."""
    NORMAL = auto()  # Show actual translations
    PSEUDO_LOC = auto()  # Show pseudo-localized text
    KEYS_ONLY = auto()  # Show string keys
    MISSING_ONLY = auto()  # Highlight missing translations
    LONG_TEXT = auto()  # Expand text for UI testing


@dataclass(slots=True)
class PseudoLocSettings:
    """
    Settings for pseudo-localization.

    Pseudo-localization helps test UI with different text lengths
    and character sets without actual translations.
    """
    # Character replacement (e.g., 'a' -> 'a')
    enable_accents: bool = True
    # Expand text length by this factor
    expansion_factor: float = 1.3
    # Add brackets around text
    add_brackets: bool = True
    bracket_style: str = "[! !]"  # e.g., "[! text !]"
    # Add prefix/suffix
    prefix: str = ""
    suffix: str = ""
    # Simulate right-to-left
    simulate_rtl: bool = False
    # Random seed for consistent output
    seed: int = 42


class PseudoLocalizer:
    """
    Generates pseudo-localized text for testing.

    Transforms English text to look like a foreign language while
    remaining readable, exposing UI issues.
    """
    __slots__ = ("_settings", "_char_map", "_random")

    def __init__(self, settings: Optional[PseudoLocSettings] = None):
        """
        Initialize pseudo-localizer.

        Args:
            settings: Pseudo-localization settings
        """
        self._settings = settings or PseudoLocSettings()
        self._random = random.Random(self._settings.seed)

        # Character map for accent replacement
        self._char_map = {
            'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e',
            'f': 'f', 'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j',
            'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o',
            'p': 'p', 'q': 'q', 'r': 'r', 's': 's', 't': 't',
            'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x', 'y': 'y', 'z': 'z',
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
            'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
            'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
            'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
            'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
        }

    @property
    def settings(self) -> PseudoLocSettings:
        """Get settings."""
        return self._settings

    def set_settings(self, settings: PseudoLocSettings) -> None:
        """Update settings."""
        self._settings = settings
        self._random = random.Random(settings.seed)

    def _replace_chars(self, text: str) -> str:
        """Replace characters with accented versions."""
        if not self._settings.enable_accents:
            return text

        result = []
        for char in text:
            if char in self._char_map:
                result.append(self._char_map[char])
            else:
                result.append(char)
        return ''.join(result)

    def _expand_text(self, text: str) -> str:
        """Expand text length."""
        factor = self._settings.expansion_factor
        if factor <= 1.0:
            return text

        # Add extra characters based on word length
        words = text.split()
        expanded_words = []

        for word in words:
            if len(word) > 3:
                # Add filler after longer words
                extra_len = int(len(word) * (factor - 1))
                extra = '~' * extra_len
                expanded_words.append(word + extra)
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words)

    def _add_brackets(self, text: str) -> str:
        """Add brackets around text."""
        if not self._settings.add_brackets:
            return text

        style = self._settings.bracket_style
        parts = style.split(' ')

        if len(parts) >= 2:
            return f"{parts[0]} {text} {parts[-1]}"
        return f"[{text}]"

    def _simulate_rtl(self, text: str) -> str:
        """Simulate right-to-left text."""
        if not self._settings.simulate_rtl:
            return text

        # Add RTL markers
        return f"\u200F{text}\u200F"

    def transform(self, text: str) -> str:
        """
        Apply pseudo-localization to text.

        Args:
            text: Original text

        Returns:
            Pseudo-localized text
        """
        # Skip placeholders
        import re
        placeholder_pattern = re.compile(r'\{[^}]+\}')
        placeholders = placeholder_pattern.findall(text)

        # Replace placeholders with markers
        for i, ph in enumerate(placeholders):
            text = text.replace(ph, f"__PH{i}__", 1)

        # Apply transformations
        result = text
        result = self._replace_chars(result)
        result = self._expand_text(result)
        result = self._add_brackets(result)
        result = self._simulate_rtl(result)

        # Add prefix/suffix
        if self._settings.prefix:
            result = self._settings.prefix + result
        if self._settings.suffix:
            result = result + self._settings.suffix

        # Restore placeholders
        for i, ph in enumerate(placeholders):
            result = result.replace(f"__PH{i}__", ph, 1)

        return result


class LocalizationPreview:
    """
    Preview localized text in-game.

    Provides real-time preview of localizations with various
    testing modes.
    """
    __slots__ = (
        "_mode",
        "_current_language",
        "_fallback_language",
        "_string_source",
        "_pseudo_localizer",
        "_missing_marker",
        "_highlight_callback",
    )

    def __init__(
        self,
        string_source: Callable[[str, str], Optional[str]],
        fallback_language: str = "en"
    ):
        """
        Initialize preview.

        Args:
            string_source: Function to get string (key, lang) -> text
            fallback_language: Fallback language code
        """
        self._mode = PreviewMode.NORMAL
        self._current_language = fallback_language
        self._fallback_language = fallback_language
        self._string_source = string_source
        self._pseudo_localizer = PseudoLocalizer()
        self._missing_marker = "[MISSING: {key}]"
        self._highlight_callback: Optional[Callable[[str, bool], None]] = None

    @property
    def mode(self) -> PreviewMode:
        """Get current preview mode."""
        return self._mode

    @mode.setter
    def mode(self, value: PreviewMode) -> None:
        """Set preview mode."""
        self._mode = value

    @property
    def current_language(self) -> str:
        """Get current language."""
        return self._current_language

    @current_language.setter
    def current_language(self, value: str) -> None:
        """Set current language."""
        self._current_language = value

    def set_pseudo_loc_settings(self, settings: PseudoLocSettings) -> None:
        """Set pseudo-localization settings."""
        self._pseudo_localizer.set_settings(settings)

    def set_missing_marker(self, marker: str) -> None:
        """Set marker for missing translations."""
        self._missing_marker = marker

    def on_highlight(self, callback: Callable[[str, bool], None]) -> None:
        """Set callback for highlighting strings."""
        self._highlight_callback = callback

    def get_text(self, key: str, **format_args: Any) -> str:
        """
        Get preview text for a key.

        Args:
            key: String key
            **format_args: Format arguments

        Returns:
            Previewed text
        """
        # Keys only mode
        if self._mode == PreviewMode.KEYS_ONLY:
            return f"[{key}]"

        # Get the string
        text = self._string_source(key, self._current_language)
        is_missing = text is None

        # Fallback to source language
        if is_missing and self._current_language != self._fallback_language:
            text = self._string_source(key, self._fallback_language)

        # Handle missing
        if text is None:
            if self._highlight_callback:
                self._highlight_callback(key, True)

            if self._mode == PreviewMode.MISSING_ONLY:
                return self._missing_marker.format(key=key)
            else:
                return self._missing_marker.format(key=key)

        # Notify highlight callback
        if self._highlight_callback:
            self._highlight_callback(key, is_missing)

        # Apply mode transformations
        if self._mode == PreviewMode.PSEUDO_LOC:
            text = self._pseudo_localizer.transform(text)

        elif self._mode == PreviewMode.LONG_TEXT:
            # Double the text for UI testing
            text = text + " " + text

        elif self._mode == PreviewMode.MISSING_ONLY and not is_missing:
            # In missing only mode, show normal text for non-missing
            pass

        # Apply format arguments
        if format_args:
            try:
                text = text.format(**format_args)
            except (KeyError, ValueError):
                pass

        return text

    def get_all_missing(self, keys: list[str]) -> list[str]:
        """
        Get all keys with missing translations.

        Args:
            keys: List of keys to check

        Returns:
            List of missing keys
        """
        missing = []

        for key in keys:
            text = self._string_source(key, self._current_language)
            if text is None:
                missing.append(key)

        return missing


class LanguageSwitcher:
    """
    UI component for switching languages.

    Provides language selection with preview support.
    """
    __slots__ = (
        "_available_languages",
        "_current_language",
        "_language_names",
        "_on_change",
        "_preview",
    )

    def __init__(self):
        """Initialize language switcher."""
        self._available_languages: list[str] = []
        self._current_language = "en"
        self._language_names: dict[str, str] = {
            "en": "English",
            "es": "Espanol",
            "fr": "Francais",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Portugues",
            "ru": "Russkiy",
            "ja": "Nihongo",
            "ko": "Hangugeo",
            "zh": "Zhongwen",
            "ar": "Arabiy",
        }
        self._on_change: Optional[Callable[[str], None]] = None
        self._preview: Optional[LocalizationPreview] = None

    @property
    def current_language(self) -> str:
        """Get current language."""
        return self._current_language

    @property
    def available_languages(self) -> list[str]:
        """Get available languages."""
        return self._available_languages.copy()

    def set_available_languages(self, languages: list[str]) -> None:
        """Set available languages."""
        self._available_languages = languages.copy()

    def add_language(self, code: str, name: Optional[str] = None) -> None:
        """Add an available language."""
        if code not in self._available_languages:
            self._available_languages.append(code)
        if name:
            self._language_names[code] = name

    def remove_language(self, code: str) -> None:
        """Remove an available language."""
        if code in self._available_languages:
            self._available_languages.remove(code)

    def get_language_name(self, code: str) -> str:
        """Get display name for a language code."""
        return self._language_names.get(code, code.upper())

    def set_language_name(self, code: str, name: str) -> None:
        """Set display name for a language."""
        self._language_names[code] = name

    def on_language_change(self, callback: Callable[[str], None]) -> None:
        """Set callback for language changes."""
        self._on_change = callback

    def link_preview(self, preview: LocalizationPreview) -> None:
        """Link to a localization preview."""
        self._preview = preview

    def switch_to(self, language: str) -> bool:
        """
        Switch to a language.

        Args:
            language: Language code

        Returns:
            True if switched successfully
        """
        if language not in self._available_languages:
            return False

        old_language = self._current_language
        self._current_language = language

        # Update linked preview
        if self._preview:
            self._preview.current_language = language

        # Notify callback
        if self._on_change and old_language != language:
            self._on_change(language)

        return True

    def switch_next(self) -> str:
        """
        Switch to next language.

        Returns:
            New language code
        """
        if not self._available_languages:
            return self._current_language

        try:
            idx = self._available_languages.index(self._current_language)
            next_idx = (idx + 1) % len(self._available_languages)
        except ValueError:
            next_idx = 0

        next_lang = self._available_languages[next_idx]
        self.switch_to(next_lang)
        return next_lang

    def switch_previous(self) -> str:
        """
        Switch to previous language.

        Returns:
            New language code
        """
        if not self._available_languages:
            return self._current_language

        try:
            idx = self._available_languages.index(self._current_language)
            prev_idx = (idx - 1) % len(self._available_languages)
        except ValueError:
            prev_idx = 0

        prev_lang = self._available_languages[prev_idx]
        self.switch_to(prev_lang)
        return prev_lang

    def get_selection_items(self) -> list[tuple[str, str]]:
        """
        Get items for a selection UI.

        Returns:
            List of (code, display_name) tuples
        """
        return [
            (code, self.get_language_name(code))
            for code in self._available_languages
        ]
