"""
String table management for localization.

Provides structured storage of localizable strings with support
for keys, contexts, and plural forms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import json
import re


class PluralForm(Enum):
    """Standard plural forms for localization."""
    ZERO = "zero"
    ONE = "one"
    TWO = "two"
    FEW = "few"
    MANY = "many"
    OTHER = "other"


@dataclass(slots=True)
class PluralRule:
    """
    Plural rule for a language.

    Defines how to select plural form based on count.
    """
    language: str
    # Function that takes count and returns PluralForm
    # For serialization, we store the rule name
    rule_name: str = "default"

    def get_form(self, count: int) -> PluralForm:
        """
        Get plural form for a count.

        Args:
            count: The number to get plural form for

        Returns:
            Appropriate plural form
        """
        if self.rule_name == "english":
            return PluralForm.ONE if count == 1 else PluralForm.OTHER
        elif self.rule_name == "slavic":
            # Russian-style plural rules
            if count % 10 == 1 and count % 100 != 11:
                return PluralForm.ONE
            elif count % 10 >= 2 and count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
                return PluralForm.FEW
            else:
                return PluralForm.MANY
        elif self.rule_name == "arabic":
            if count == 0:
                return PluralForm.ZERO
            elif count == 1:
                return PluralForm.ONE
            elif count == 2:
                return PluralForm.TWO
            elif count % 100 >= 3 and count % 100 <= 10:
                return PluralForm.FEW
            elif count % 100 >= 11:
                return PluralForm.MANY
            else:
                return PluralForm.OTHER
        else:
            # Default to English-style
            return PluralForm.ONE if count == 1 else PluralForm.OTHER


@dataclass(slots=True)
class StringContext:
    """
    Context information for a localizable string.

    Provides translators with information about where and how
    the string is used.
    """
    description: str = ""
    screenshot_path: str = ""
    max_length: int = 0  # 0 = no limit
    category: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class StringEntry:
    """
    A single localizable string entry.

    Contains the key, source text, translations, and metadata.
    """
    key: str
    source_text: str  # Usually English
    context: StringContext = field(default_factory=StringContext)
    # Translations by language code
    translations: dict[str, str] = field(default_factory=dict)
    # Plural forms by language code
    plurals: dict[str, dict[str, str]] = field(default_factory=dict)
    # Metadata
    created_at: float = 0.0
    modified_at: float = 0.0
    source_file: str = ""
    source_line: int = 0
    is_locked: bool = False
    needs_review: bool = False

    def get_translation(self, language: str) -> str:
        """
        Get translation for a language.

        Args:
            language: Language code (e.g., "en", "fr", "ja")

        Returns:
            Translated text or source text if not found
        """
        return self.translations.get(language, self.source_text)

    def set_translation(self, language: str, text: str) -> None:
        """
        Set translation for a language.

        Args:
            language: Language code
            text: Translated text
        """
        self.translations[language] = text

    def get_plural(self, language: str, form: PluralForm) -> Optional[str]:
        """
        Get plural form translation.

        Args:
            language: Language code
            form: Plural form

        Returns:
            Plural translation or None
        """
        if language in self.plurals:
            return self.plurals[language].get(form.value)
        return None

    def set_plural(self, language: str, form: PluralForm, text: str) -> None:
        """
        Set plural form translation.

        Args:
            language: Language code
            form: Plural form
            text: Translated text
        """
        if language not in self.plurals:
            self.plurals[language] = {}
        self.plurals[language][form.value] = text

    def has_translation(self, language: str) -> bool:
        """Check if translation exists for language."""
        return language in self.translations

    def get_all_languages(self) -> set[str]:
        """Get all languages with translations."""
        return set(self.translations.keys())


class StringTable:
    """
    Table of localizable strings.

    Manages a collection of string entries organized by key.
    """
    __slots__ = (
        "_name",
        "_entries",
        "_source_language",
        "_target_languages",
        "_plural_rules",
        "_categories",
    )

    def __init__(
        self,
        name: str,
        source_language: str = "en"
    ):
        """
        Initialize string table.

        Args:
            name: Table name
            source_language: Source language code
        """
        self._name = name
        self._entries: dict[str, StringEntry] = {}
        self._source_language = source_language
        self._target_languages: set[str] = set()
        self._plural_rules: dict[str, PluralRule] = {}
        self._categories: set[str] = set()

        # Set up default plural rules
        self._plural_rules["en"] = PluralRule("en", "english")
        self._plural_rules["ru"] = PluralRule("ru", "slavic")
        self._plural_rules["ar"] = PluralRule("ar", "arabic")

    @property
    def name(self) -> str:
        """Get table name."""
        return self._name

    @property
    def source_language(self) -> str:
        """Get source language."""
        return self._source_language

    @property
    def target_languages(self) -> set[str]:
        """Get target languages."""
        return self._target_languages.copy()

    def add_target_language(self, language: str) -> None:
        """Add a target language."""
        self._target_languages.add(language)

    def remove_target_language(self, language: str) -> None:
        """Remove a target language."""
        self._target_languages.discard(language)

    def set_plural_rule(self, language: str, rule: PluralRule) -> None:
        """Set plural rule for a language."""
        self._plural_rules[language] = rule

    def get_plural_rule(self, language: str) -> Optional[PluralRule]:
        """Get plural rule for a language."""
        return self._plural_rules.get(language)

    def add_entry(self, entry: StringEntry) -> bool:
        """
        Add a string entry.

        Args:
            entry: Entry to add

        Returns:
            True if added, False if key already exists
        """
        if entry.key in self._entries:
            return False

        self._entries[entry.key] = entry

        if entry.context.category:
            self._categories.add(entry.context.category)

        return True

    def update_entry(self, entry: StringEntry) -> bool:
        """
        Update an existing entry.

        Args:
            entry: Entry with updated data

        Returns:
            True if updated, False if not found
        """
        if entry.key not in self._entries:
            return False

        existing = self._entries[entry.key]
        if existing.is_locked:
            return False

        self._entries[entry.key] = entry
        return True

    def remove_entry(self, key: str) -> bool:
        """
        Remove an entry by key.

        Args:
            key: Entry key

        Returns:
            True if removed
        """
        if key in self._entries:
            if self._entries[key].is_locked:
                return False
            del self._entries[key]
            return True
        return False

    def get_entry(self, key: str) -> Optional[StringEntry]:
        """Get an entry by key."""
        return self._entries.get(key)

    def get_all_entries(self) -> list[StringEntry]:
        """Get all entries."""
        return list(self._entries.values())

    def get_entries_by_category(self, category: str) -> list[StringEntry]:
        """Get entries in a category."""
        return [e for e in self._entries.values() if e.context.category == category]

    def get_categories(self) -> set[str]:
        """Get all categories."""
        return self._categories.copy()

    def get_text(
        self,
        key: str,
        language: str,
        count: Optional[int] = None,
        **format_args: Any
    ) -> str:
        """
        Get localized text for a key.

        Args:
            key: String key
            language: Target language
            count: Count for plural forms
            **format_args: Format arguments

        Returns:
            Localized and formatted text
        """
        entry = self._entries.get(key)
        if entry is None:
            return f"[{key}]"

        if count is not None and language in self._plural_rules:
            rule = self._plural_rules[language]
            form = rule.get_form(count)
            text = entry.get_plural(language, form)
            if text is None:
                text = entry.get_translation(language)
        else:
            text = entry.get_translation(language)

        # Apply format arguments
        if format_args:
            try:
                text = text.format(**format_args)
            except (KeyError, ValueError):
                pass  # Return unformatted if formatting fails

        return text

    def find_entries(self, query: str, search_translations: bool = True) -> list[StringEntry]:
        """
        Search for entries containing query text.

        Args:
            query: Search query
            search_translations: Also search in translations

        Returns:
            Matching entries
        """
        query_lower = query.lower()
        results = []

        for entry in self._entries.values():
            if query_lower in entry.key.lower():
                results.append(entry)
            elif query_lower in entry.source_text.lower():
                results.append(entry)
            elif search_translations:
                for translation in entry.translations.values():
                    if query_lower in translation.lower():
                        results.append(entry)
                        break

        return results

    def get_missing_translations(self, language: str) -> list[StringEntry]:
        """Get entries missing translation for a language."""
        return [e for e in self._entries.values() if not e.has_translation(language)]

    def export_to_dict(self) -> dict[str, Any]:
        """Export table to dictionary format."""
        return {
            "name": self._name,
            "source_language": self._source_language,
            "target_languages": list(self._target_languages),
            "entries": [
                {
                    "key": e.key,
                    "source_text": e.source_text,
                    "context": {
                        "description": e.context.description,
                        "category": e.context.category,
                        "max_length": e.context.max_length,
                        "tags": e.context.tags,
                    },
                    "translations": e.translations,
                    "plurals": e.plurals,
                    "needs_review": e.needs_review,
                }
                for e in self._entries.values()
            ],
        }

    def import_from_dict(self, data: dict[str, Any]) -> int:
        """
        Import entries from dictionary.

        Args:
            data: Dictionary data

        Returns:
            Number of entries imported
        """
        count = 0

        if "target_languages" in data:
            for lang in data["target_languages"]:
                self.add_target_language(lang)

        for entry_data in data.get("entries", []):
            context = StringContext(
                description=entry_data.get("context", {}).get("description", ""),
                category=entry_data.get("context", {}).get("category", ""),
                max_length=entry_data.get("context", {}).get("max_length", 0),
                tags=entry_data.get("context", {}).get("tags", []),
            )

            entry = StringEntry(
                key=entry_data["key"],
                source_text=entry_data.get("source_text", ""),
                context=context,
                translations=entry_data.get("translations", {}),
                plurals=entry_data.get("plurals", {}),
                needs_review=entry_data.get("needs_review", False),
            )

            if self.add_entry(entry):
                count += 1
            else:
                self.update_entry(entry)
                count += 1

        return count

    @property
    def entry_count(self) -> int:
        """Get number of entries."""
        return len(self._entries)


class StringTableManager:
    """
    Manages multiple string tables.

    Provides a unified interface for working with multiple
    localization tables.
    """
    __slots__ = (
        "_tables",
        "_active_language",
        "_fallback_language",
    )

    def __init__(self, fallback_language: str = "en"):
        """
        Initialize manager.

        Args:
            fallback_language: Fallback language when translation missing
        """
        self._tables: dict[str, StringTable] = {}
        self._active_language = fallback_language
        self._fallback_language = fallback_language

    @property
    def active_language(self) -> str:
        """Get active language."""
        return self._active_language

    @active_language.setter
    def active_language(self, language: str) -> None:
        """Set active language."""
        self._active_language = language

    @property
    def fallback_language(self) -> str:
        """Get fallback language."""
        return self._fallback_language

    def create_table(self, name: str, source_language: str = "en") -> StringTable:
        """
        Create a new string table.

        Args:
            name: Table name
            source_language: Source language

        Returns:
            Created table
        """
        table = StringTable(name, source_language)
        self._tables[name] = table
        return table

    def get_table(self, name: str) -> Optional[StringTable]:
        """Get a table by name."""
        return self._tables.get(name)

    def remove_table(self, name: str) -> bool:
        """Remove a table."""
        if name in self._tables:
            del self._tables[name]
            return True
        return False

    def get_all_tables(self) -> list[StringTable]:
        """Get all tables."""
        return list(self._tables.values())

    def get_text(
        self,
        key: str,
        table_name: Optional[str] = None,
        language: Optional[str] = None,
        count: Optional[int] = None,
        **format_args: Any
    ) -> str:
        """
        Get localized text.

        Args:
            key: String key
            table_name: Specific table (or search all)
            language: Language (or use active)
            count: Count for plurals
            **format_args: Format arguments

        Returns:
            Localized text
        """
        language = language or self._active_language

        if table_name:
            table = self._tables.get(table_name)
            if table:
                return table.get_text(key, language, count, **format_args)
        else:
            # Search all tables
            for table in self._tables.values():
                entry = table.get_entry(key)
                if entry:
                    return table.get_text(key, language, count, **format_args)

        return f"[{key}]"

    def get_all_languages(self) -> set[str]:
        """Get all languages across all tables."""
        languages = set()
        for table in self._tables.values():
            languages.add(table.source_language)
            languages.update(table.target_languages)
        return languages

    def get_all_entries(self) -> list[tuple[str, StringEntry]]:
        """Get all entries with their table names."""
        results = []
        for table_name, table in self._tables.items():
            for entry in table.get_all_entries():
                results.append((table_name, entry))
        return results

    def export_all(self) -> dict[str, Any]:
        """Export all tables to dictionary."""
        return {
            "active_language": self._active_language,
            "fallback_language": self._fallback_language,
            "tables": {
                name: table.export_to_dict()
                for name, table in self._tables.items()
            },
        }

    def import_all(self, data: dict[str, Any]) -> int:
        """
        Import tables from dictionary.

        Returns:
            Total entries imported
        """
        total = 0

        self._active_language = data.get("active_language", self._active_language)
        self._fallback_language = data.get("fallback_language", self._fallback_language)

        for name, table_data in data.get("tables", {}).items():
            table = self.create_table(
                name,
                table_data.get("source_language", "en")
            )
            total += table.import_from_dict(table_data)

        return total
