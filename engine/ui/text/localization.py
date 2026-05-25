"""
Localization system for UI text.

Provides:
- LocalizationManager singleton
- String tables (JSON format)
- Language switching
- Pluralization rules (one, few, many, other)
- Parameter substitution: {name}, {count}
- RTL language support detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator
import json
import re
import threading


class TextDirection(Enum):
    """Text direction for layout purposes."""
    LTR = auto()  # Left-to-right
    RTL = auto()  # Right-to-left
    AUTO = auto()  # Detect from content


class PluralCategory(Enum):
    """Plural categories following CLDR (Unicode Common Locale Data Repository)."""
    ZERO = "zero"
    ONE = "one"
    TWO = "two"
    FEW = "few"
    MANY = "many"
    OTHER = "other"


@dataclass
class PluralRule:
    """Defines pluralization rules for a language.

    Different languages have different pluralization rules.
    For example:
    - English: 1 = one, everything else = other
    - Russian: 1,21,31... = one, 2-4,22-24... = few, 5-20,25-30... = many
    - Arabic: 0 = zero, 1 = one, 2 = two, 3-10 = few, 11-99 = many, 100+ = other
    """

    # Function to determine plural category from a count
    get_category: Callable[[int | float], PluralCategory]

    @staticmethod
    def english() -> PluralRule:
        """Create English pluralization rule."""
        def _get_category(n: int | float) -> PluralCategory:
            if n == 1:
                return PluralCategory.ONE
            return PluralCategory.OTHER
        return PluralRule(get_category=_get_category)

    @staticmethod
    def russian() -> PluralRule:
        """Create Russian pluralization rule."""
        def _get_category(n: int | float) -> PluralCategory:
            n = abs(int(n))
            n10 = n % 10
            n100 = n % 100

            if n10 == 1 and n100 != 11:
                return PluralCategory.ONE
            if 2 <= n10 <= 4 and not (12 <= n100 <= 14):
                return PluralCategory.FEW
            if n10 == 0 or 5 <= n10 <= 9 or 11 <= n100 <= 14:
                return PluralCategory.MANY
            return PluralCategory.OTHER
        return PluralRule(get_category=_get_category)

    @staticmethod
    def arabic() -> PluralRule:
        """Create Arabic pluralization rule."""
        def _get_category(n: int | float) -> PluralCategory:
            n = abs(int(n))
            n100 = n % 100

            if n == 0:
                return PluralCategory.ZERO
            if n == 1:
                return PluralCategory.ONE
            if n == 2:
                return PluralCategory.TWO
            if 3 <= n100 <= 10:
                return PluralCategory.FEW
            if 11 <= n100 <= 99:
                return PluralCategory.MANY
            return PluralCategory.OTHER
        return PluralRule(get_category=_get_category)

    @staticmethod
    def french() -> PluralRule:
        """Create French pluralization rule (0 and 1 = singular)."""
        def _get_category(n: int | float) -> PluralCategory:
            if n == 0 or n == 1:
                return PluralCategory.ONE
            return PluralCategory.OTHER
        return PluralRule(get_category=_get_category)

    @staticmethod
    def japanese() -> PluralRule:
        """Create Japanese pluralization rule (no pluralization)."""
        def _get_category(n: int | float) -> PluralCategory:
            return PluralCategory.OTHER
        return PluralRule(get_category=_get_category)

    @staticmethod
    def polish() -> PluralRule:
        """Create Polish pluralization rule."""
        def _get_category(n: int | float) -> PluralCategory:
            n = abs(int(n))
            n10 = n % 10
            n100 = n % 100

            if n == 1:
                return PluralCategory.ONE
            if 2 <= n10 <= 4 and not (12 <= n100 <= 14):
                return PluralCategory.FEW
            if n10 == 0 or n10 == 1 or 5 <= n10 <= 9 or 12 <= n100 <= 14:
                return PluralCategory.MANY
            return PluralCategory.OTHER
        return PluralRule(get_category=_get_category)


@dataclass(frozen=True)
class FormatParameter:
    """A parameter in a formatted string.

    Attributes:
        name: Parameter name
        format_spec: Optional format specification (e.g., ".2f" for floats)
        default: Default value if parameter not provided
    """
    name: str
    format_spec: str | None = None
    default: Any = None

    def format(self, value: Any) -> str:
        """Format a value using this parameter's format spec.

        Args:
            value: Value to format

        Returns:
            Formatted string
        """
        if value is None:
            value = self.default if self.default is not None else ""

        if self.format_spec:
            return format(value, self.format_spec)
        return str(value)


@dataclass
class LocalizedString:
    """A localized string with parameter substitution support.

    Attributes:
        key: String key for lookup
        template: Template string with {param} placeholders
        parameters: List of extracted parameters
        plural_forms: Dict of plural category -> template for pluralized strings
    """
    key: str
    template: str
    parameters: list[FormatParameter] = field(default_factory=list)
    plural_forms: dict[PluralCategory, str] = field(default_factory=dict)

    # Regex for parameter extraction
    _PARAM_PATTERN = re.compile(r"\{(\w+)(?::([^}]+))?\}")

    def __post_init__(self) -> None:
        """Extract parameters from template."""
        if not self.parameters:
            self.parameters = self._extract_parameters(self.template)

    def _extract_parameters(self, template: str) -> list[FormatParameter]:
        """Extract parameter definitions from template."""
        params = []
        seen = set()

        for match in self._PARAM_PATTERN.finditer(template):
            name = match.group(1)
            format_spec = match.group(2)

            if name not in seen:
                params.append(FormatParameter(name=name, format_spec=format_spec))
                seen.add(name)

        return params

    def format(
        self,
        plural_rule: PluralRule | None = None,
        **kwargs: Any,
    ) -> str:
        """Format the string with provided parameters.

        Args:
            plural_rule: Optional plural rule for count-based strings
            **kwargs: Parameter values

        Returns:
            Formatted string
        """
        # Select template based on plural form if applicable
        template = self.template

        if self.plural_forms and plural_rule and "count" in kwargs:
            count = kwargs["count"]
            category = plural_rule.get_category(count)

            if category in self.plural_forms:
                template = self.plural_forms[category]
            elif PluralCategory.OTHER in self.plural_forms:
                template = self.plural_forms[PluralCategory.OTHER]

        # Perform parameter substitution
        result = template
        for param in self.parameters:
            placeholder = f"{{{param.name}}}"
            if param.format_spec:
                placeholder = f"{{{param.name}:{param.format_spec}}}"

            value = kwargs.get(param.name, param.default)
            formatted = param.format(value)
            result = result.replace(placeholder, formatted)

        return result

    def has_parameter(self, name: str) -> bool:
        """Check if string has a specific parameter."""
        return any(p.name == name for p in self.parameters)


@dataclass
class Language:
    """Represents a language configuration.

    Attributes:
        code: Language code (e.g., "en", "fr", "ja")
        name: Native name of the language
        english_name: English name of the language
        direction: Text direction (LTR or RTL)
        plural_rule: Pluralization rule for this language
        fallback: Fallback language code (e.g., "en")
    """
    code: str
    name: str
    english_name: str
    direction: TextDirection = TextDirection.LTR
    plural_rule: PluralRule = field(default_factory=PluralRule.english)
    fallback: str | None = None

    # RTL languages
    RTL_LANGUAGES = frozenset([
        "ar", "arc", "arz", "az", "dv", "fa", "ha", "he", "khw", "ks",
        "ku", "nqo", "pa", "ps", "sd", "syr", "ug", "ur", "uz", "yi",
    ])

    def __post_init__(self) -> None:
        """Set default direction based on language code."""
        if self.direction == TextDirection.AUTO:
            if self.code.split("-")[0] in self.RTL_LANGUAGES:
                self.direction = TextDirection.RTL
            else:
                self.direction = TextDirection.LTR

    @property
    def is_rtl(self) -> bool:
        """Check if this language is right-to-left."""
        return self.direction == TextDirection.RTL


@dataclass
class StringTable:
    """A table of localized strings for one language.

    Attributes:
        language: Language this table is for
        strings: Dict of key -> LocalizedString
    """
    language: Language
    strings: dict[str, LocalizedString] = field(default_factory=dict)

    def get(self, key: str) -> LocalizedString | None:
        """Get a localized string by key."""
        return self.strings.get(key)

    def set(self, key: str, template: str) -> LocalizedString:
        """Set a localized string.

        Args:
            key: String key
            template: Template string

        Returns:
            Created LocalizedString
        """
        localized = LocalizedString(key=key, template=template)
        self.strings[key] = localized
        return localized

    def set_plural(
        self,
        key: str,
        forms: dict[str, str],
    ) -> LocalizedString:
        """Set a pluralized string.

        Args:
            key: String key
            forms: Dict of category name -> template

        Returns:
            Created LocalizedString
        """
        # Convert string keys to PluralCategory
        plural_forms = {}
        default_template = ""

        for cat_name, template in forms.items():
            try:
                category = PluralCategory(cat_name.lower())
                plural_forms[category] = template
                if category == PluralCategory.OTHER:
                    default_template = template
            except ValueError:
                if cat_name.lower() == "default":
                    default_template = template

        if not default_template and plural_forms:
            default_template = next(iter(plural_forms.values()))

        localized = LocalizedString(
            key=key,
            template=default_template,
            plural_forms=plural_forms,
        )
        self.strings[key] = localized
        return localized

    def load_json(self, data: dict[str, Any]) -> int:
        """Load strings from JSON data.

        Args:
            data: JSON data with string definitions

        Returns:
            Number of strings loaded

        Expected format:
        {
            "key1": "Simple string",
            "key2": "Hello, {name}!",
            "items": {
                "one": "{count} item",
                "other": "{count} items"
            }
        }
        """
        count = 0

        for key, value in data.items():
            if isinstance(value, str):
                self.set(key, value)
                count += 1
            elif isinstance(value, dict):
                # Check if it's a plural form
                if any(k.lower() in ("zero", "one", "two", "few", "many", "other")
                       for k in value.keys()):
                    self.set_plural(key, value)
                    count += 1

        return count

    def load_file(self, path: str | Path) -> int:
        """Load strings from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Number of strings loaded
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.load_json(data)

    def export_json(self) -> dict[str, Any]:
        """Export strings to JSON format.

        Returns:
            Dict suitable for JSON serialization
        """
        result: dict[str, Any] = {}

        for key, localized in self.strings.items():
            if localized.plural_forms:
                result[key] = {
                    cat.value: template
                    for cat, template in localized.plural_forms.items()
                }
            else:
                result[key] = localized.template

        return result

    def __len__(self) -> int:
        return len(self.strings)

    def __contains__(self, key: str) -> bool:
        return key in self.strings

    def __iter__(self) -> Iterator[str]:
        return iter(self.strings)


class LocalizationManager:
    """Singleton manager for localization.

    Handles loading and switching between languages, and provides
    string lookup with parameter substitution and pluralization.
    """

    _instance: LocalizationManager | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        """Initialize the localization manager."""
        self._languages: dict[str, Language] = {}
        self._string_tables: dict[str, StringTable] = {}
        self._current_language: str | None = None
        self._fallback_language: str = "en"
        self._missing_key_callback: Callable[[str], str] | None = None

        # Pre-register common languages
        self._register_default_languages()

    @classmethod
    def get_instance(cls) -> LocalizationManager:
        """Get the singleton instance.

        Returns:
            LocalizationManager singleton
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    def _register_default_languages(self) -> None:
        """Register common languages with their configurations."""
        self.register_language(Language(
            code="en",
            name="English",
            english_name="English",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.english(),
        ))

        self.register_language(Language(
            code="fr",
            name="Francais",
            english_name="French",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.french(),
            fallback="en",
        ))

        self.register_language(Language(
            code="de",
            name="Deutsch",
            english_name="German",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.english(),
            fallback="en",
        ))

        self.register_language(Language(
            code="es",
            name="Espanol",
            english_name="Spanish",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.english(),
            fallback="en",
        ))

        self.register_language(Language(
            code="ru",
            name="Russkiy",
            english_name="Russian",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.russian(),
            fallback="en",
        ))

        self.register_language(Language(
            code="ja",
            name="Nihongo",
            english_name="Japanese",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.japanese(),
            fallback="en",
        ))

        self.register_language(Language(
            code="zh",
            name="Zhongwen",
            english_name="Chinese",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.japanese(),
            fallback="en",
        ))

        self.register_language(Language(
            code="ko",
            name="Hangugeo",
            english_name="Korean",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.japanese(),
            fallback="en",
        ))

        self.register_language(Language(
            code="ar",
            name="al-Arabiyyah",
            english_name="Arabic",
            direction=TextDirection.RTL,
            plural_rule=PluralRule.arabic(),
            fallback="en",
        ))

        self.register_language(Language(
            code="he",
            name="Ivrit",
            english_name="Hebrew",
            direction=TextDirection.RTL,
            plural_rule=PluralRule.english(),
            fallback="en",
        ))

        self.register_language(Language(
            code="pl",
            name="Polski",
            english_name="Polish",
            direction=TextDirection.LTR,
            plural_rule=PluralRule.polish(),
            fallback="en",
        ))

    def register_language(self, language: Language) -> None:
        """Register a language configuration.

        Args:
            language: Language to register
        """
        self._languages[language.code] = language

    def get_language(self, code: str) -> Language | None:
        """Get a language by code.

        Args:
            code: Language code

        Returns:
            Language if found, None otherwise
        """
        return self._languages.get(code)

    @property
    def available_languages(self) -> list[str]:
        """Get list of available language codes."""
        return list(self._languages.keys())

    @property
    def current_language(self) -> str | None:
        """Get the current language code."""
        return self._current_language

    @property
    def current_language_info(self) -> Language | None:
        """Get the current language configuration."""
        if self._current_language:
            return self._languages.get(self._current_language)
        return None

    def set_language(self, code: str) -> bool:
        """Set the current language.

        Args:
            code: Language code

        Returns:
            True if language was set, False if not found
        """
        if code in self._languages:
            self._current_language = code
            return True
        return False

    def set_fallback_language(self, code: str) -> None:
        """Set the fallback language for missing strings.

        Args:
            code: Language code
        """
        self._fallback_language = code

    def load_strings(
        self,
        language_code: str,
        path: str | Path,
    ) -> int:
        """Load strings for a language from a JSON file.

        Args:
            language_code: Language code
            path: Path to JSON file

        Returns:
            Number of strings loaded
        """
        if language_code not in self._languages:
            # Create a basic language config
            self.register_language(Language(
                code=language_code,
                name=language_code,
                english_name=language_code,
            ))

        language = self._languages[language_code]

        if language_code not in self._string_tables:
            self._string_tables[language_code] = StringTable(language=language)

        return self._string_tables[language_code].load_file(path)

    def load_strings_from_dict(
        self,
        language_code: str,
        data: dict[str, Any],
    ) -> int:
        """Load strings for a language from a dictionary.

        Args:
            language_code: Language code
            data: String data dictionary

        Returns:
            Number of strings loaded
        """
        if language_code not in self._languages:
            self.register_language(Language(
                code=language_code,
                name=language_code,
                english_name=language_code,
            ))

        language = self._languages[language_code]

        if language_code not in self._string_tables:
            self._string_tables[language_code] = StringTable(language=language)

        return self._string_tables[language_code].load_json(data)

    def get(
        self,
        key: str,
        default: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Get a localized string.

        Args:
            key: String key
            default: Default value if key not found
            **kwargs: Parameter values for substitution

        Returns:
            Localized string with parameters substituted
        """
        if not self._current_language:
            return self._handle_missing(key, default)

        # Try current language
        result = self._lookup(self._current_language, key, **kwargs)
        if result is not None:
            return result

        # Try language fallback chain
        language = self._languages.get(self._current_language)
        while language and language.fallback:
            result = self._lookup(language.fallback, key, **kwargs)
            if result is not None:
                return result
            language = self._languages.get(language.fallback)

        # Try global fallback
        if self._fallback_language != self._current_language:
            result = self._lookup(self._fallback_language, key, **kwargs)
            if result is not None:
                return result

        return self._handle_missing(key, default)

    def _lookup(
        self,
        language_code: str,
        key: str,
        **kwargs: Any,
    ) -> str | None:
        """Look up a string in a specific language.

        Args:
            language_code: Language to look in
            key: String key
            **kwargs: Parameter values

        Returns:
            Formatted string or None if not found
        """
        table = self._string_tables.get(language_code)
        if not table:
            return None

        localized = table.get(key)
        if not localized:
            return None

        language = self._languages.get(language_code)
        plural_rule = language.plural_rule if language else None

        return localized.format(plural_rule=plural_rule, **kwargs)

    def _handle_missing(self, key: str, default: str | None) -> str:
        """Handle a missing string key.

        Args:
            key: Missing key
            default: Default value

        Returns:
            Default value, callback result, or key itself
        """
        if default is not None:
            return default

        if self._missing_key_callback:
            return self._missing_key_callback(key)

        # Return the key itself as a fallback
        return f"[{key}]"

    def set_missing_key_callback(
        self,
        callback: Callable[[str], str] | None,
    ) -> None:
        """Set callback for missing keys.

        Args:
            callback: Function(key) -> fallback string
        """
        self._missing_key_callback = callback

    def has_key(self, key: str, language_code: str | None = None) -> bool:
        """Check if a key exists.

        Args:
            key: String key
            language_code: Specific language (or current if None)

        Returns:
            True if key exists
        """
        code = language_code or self._current_language
        if not code:
            return False

        table = self._string_tables.get(code)
        return table is not None and key in table

    def get_text_direction(self, language_code: str | None = None) -> TextDirection:
        """Get the text direction for a language.

        Args:
            language_code: Language code (or current if None)

        Returns:
            TextDirection for the language
        """
        code = language_code or self._current_language
        if not code:
            return TextDirection.LTR

        language = self._languages.get(code)
        return language.direction if language else TextDirection.LTR

    def is_rtl(self, language_code: str | None = None) -> bool:
        """Check if a language is right-to-left.

        Args:
            language_code: Language code (or current if None)

        Returns:
            True if RTL
        """
        return self.get_text_direction(language_code) == TextDirection.RTL

    def get_string_table(self, language_code: str) -> StringTable | None:
        """Get the string table for a language.

        Args:
            language_code: Language code

        Returns:
            StringTable if found
        """
        return self._string_tables.get(language_code)

    def clear(self) -> None:
        """Clear all loaded strings."""
        self._string_tables.clear()
        self._current_language = None


def detect_text_direction(text: str) -> TextDirection:
    """Detect text direction from content.

    Uses Unicode bidirectional algorithm heuristics.

    Args:
        text: Text to analyze

    Returns:
        TextDirection.LTR or TextDirection.RTL
    """
    if not text:
        return TextDirection.LTR

    # Check first strong directional character
    for char in text:
        # RTL characters (Arabic, Hebrew, etc.)
        if "\u0590" <= char <= "\u08FF":
            return TextDirection.RTL
        if "\uFB00" <= char <= "\uFDFF":
            return TextDirection.RTL
        if "\uFE70" <= char <= "\uFEFF":
            return TextDirection.RTL

        # LTR characters (Latin, etc.)
        if "\u0041" <= char <= "\u005A":  # A-Z
            return TextDirection.LTR
        if "\u0061" <= char <= "\u007A":  # a-z
            return TextDirection.LTR
        if "\u00C0" <= char <= "\u024F":  # Latin Extended
            return TextDirection.LTR

    return TextDirection.LTR
