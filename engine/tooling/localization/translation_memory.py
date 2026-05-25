"""
Translation memory for localization.

Provides storage and lookup of previous translations for
consistency across the project.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Any, Callable
import re
import math
from collections import defaultdict


class TMMatchType(Enum):
    """Types of translation memory matches."""
    EXACT = auto()  # 100% match
    FUZZY = auto()  # Partial match
    CONTEXT = auto()  # Match based on context
    MACHINE = auto()  # Machine-suggested translation


@dataclass(slots=True)
class TMEntry:
    """
    A translation memory entry.

    Stores a source text and its translation with metadata.
    """
    id: int
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    context: str = ""
    domain: str = ""  # e.g., "UI", "Dialogue", "Items"
    quality_score: float = 1.0  # 0.0 - 1.0
    usage_count: int = 0
    created_at: float = 0.0
    modified_at: float = 0.0
    created_by: str = ""
    is_approved: bool = False

    def get_source_normalized(self) -> str:
        """Get normalized source text for matching."""
        return self._normalize(self.source_text)

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text


@dataclass(slots=True)
class TMMatch:
    """
    A translation memory match result.
    """
    entry: TMEntry
    match_type: TMMatchType
    similarity: float  # 0.0 - 1.0
    source_text: str  # The query text
    differences: list[tuple[str, str]] = field(default_factory=list)

    def is_exact_match(self) -> bool:
        """Check if this is an exact match."""
        return self.match_type == TMMatchType.EXACT and self.similarity >= 1.0


class TranslationMemory:
    """
    Translation memory for a language pair.

    Stores and retrieves translations with fuzzy matching support.
    """
    __slots__ = (
        "_source_language",
        "_target_language",
        "_entries",
        "_next_id",
        "_source_index",
        "_context_index",
    )

    def __init__(self, source_language: str, target_language: str):
        """
        Initialize translation memory.

        Args:
            source_language: Source language code
            target_language: Target language code
        """
        self._source_language = source_language
        self._target_language = target_language
        self._entries: dict[int, TMEntry] = {}
        self._next_id = 1

        # Indices for faster lookup
        self._source_index: dict[str, list[int]] = defaultdict(list)
        self._context_index: dict[str, list[int]] = defaultdict(list)

    @property
    def source_language(self) -> str:
        """Get source language."""
        return self._source_language

    @property
    def target_language(self) -> str:
        """Get target language."""
        return self._target_language

    @property
    def entry_count(self) -> int:
        """Get number of entries."""
        return len(self._entries)

    def add_entry(
        self,
        source_text: str,
        target_text: str,
        context: str = "",
        domain: str = "",
        quality_score: float = 1.0,
        created_by: str = ""
    ) -> TMEntry:
        """
        Add a new translation memory entry.

        Args:
            source_text: Source text
            target_text: Target translation
            context: Context information
            domain: Domain/category
            quality_score: Quality rating
            created_by: Creator identifier

        Returns:
            Created entry
        """
        entry = TMEntry(
            id=self._next_id,
            source_text=source_text,
            target_text=target_text,
            source_language=self._source_language,
            target_language=self._target_language,
            context=context,
            domain=domain,
            quality_score=quality_score,
            created_by=created_by,
        )
        self._next_id += 1

        self._entries[entry.id] = entry

        # Update indices
        normalized = entry.get_source_normalized()
        self._source_index[normalized].append(entry.id)

        if context:
            self._context_index[context.lower()].append(entry.id)

        return entry

    def update_entry(self, entry_id: int, target_text: str) -> bool:
        """
        Update an entry's translation.

        Args:
            entry_id: Entry ID
            target_text: New translation

        Returns:
            True if updated
        """
        if entry_id in self._entries:
            self._entries[entry_id].target_text = target_text
            self._entries[entry_id].usage_count += 1
            return True
        return False

    def remove_entry(self, entry_id: int) -> bool:
        """Remove an entry."""
        if entry_id in self._entries:
            entry = self._entries[entry_id]

            # Update indices
            normalized = entry.get_source_normalized()
            if normalized in self._source_index:
                self._source_index[normalized].remove(entry_id)

            if entry.context:
                context_key = entry.context.lower()
                if context_key in self._context_index:
                    self._context_index[context_key].remove(entry_id)

            del self._entries[entry_id]
            return True
        return False

    def get_entry(self, entry_id: int) -> Optional[TMEntry]:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def find_exact(self, source_text: str) -> Optional[TMEntry]:
        """
        Find exact match for source text.

        Args:
            source_text: Text to match

        Returns:
            Matching entry or None
        """
        normalized = source_text.lower().strip()
        normalized = re.sub(r'\s+', ' ', normalized)

        if normalized in self._source_index:
            entry_ids = self._source_index[normalized]
            if entry_ids:
                # Return highest quality match
                entries = [self._entries[eid] for eid in entry_ids]
                entries.sort(key=lambda e: (e.quality_score, e.usage_count), reverse=True)
                return entries[0]

        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Levenshtein-based approach.

        Returns:
            Similarity score (0.0 - 1.0)
        """
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        if t1 == t2:
            return 1.0

        if not t1 or not t2:
            return 0.0

        # Word-based similarity (faster for longer texts)
        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        jaccard = len(intersection) / len(union)

        # Length similarity
        len_ratio = min(len(t1), len(t2)) / max(len(t1), len(t2))

        # Combined score
        return (jaccard * 0.7 + len_ratio * 0.3)

    def find_fuzzy(
        self,
        source_text: str,
        min_similarity: float = 0.7,
        max_results: int = 5,
        context: Optional[str] = None
    ) -> list[TMMatch]:
        """
        Find fuzzy matches for source text.

        Args:
            source_text: Text to match
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results
            context: Optional context for filtering

        Returns:
            List of matches sorted by similarity
        """
        matches: list[TMMatch] = []

        # Check for exact match first
        exact = self.find_exact(source_text)
        if exact:
            matches.append(TMMatch(
                entry=exact,
                match_type=TMMatchType.EXACT,
                similarity=1.0,
                source_text=source_text,
            ))
            exact.usage_count += 1

        # Find fuzzy matches
        for entry in self._entries.values():
            if exact and entry.id == exact.id:
                continue

            similarity = self._calculate_similarity(source_text, entry.source_text)

            # Boost similarity if context matches
            if context and entry.context:
                if context.lower() in entry.context.lower():
                    similarity = min(1.0, similarity + 0.1)

            if similarity >= min_similarity:
                matches.append(TMMatch(
                    entry=entry,
                    match_type=TMMatchType.FUZZY,
                    similarity=similarity,
                    source_text=source_text,
                ))

        # Sort by similarity
        matches.sort(key=lambda m: (m.similarity, m.entry.quality_score), reverse=True)

        return matches[:max_results]

    def find_by_context(self, context: str) -> list[TMEntry]:
        """Find entries by context."""
        context_key = context.lower()
        if context_key in self._context_index:
            return [self._entries[eid] for eid in self._context_index[context_key]]
        return []

    def get_all_entries(self) -> list[TMEntry]:
        """Get all entries."""
        return list(self._entries.values())

    def export_to_dict(self) -> dict[str, Any]:
        """Export memory to dictionary."""
        return {
            "source_language": self._source_language,
            "target_language": self._target_language,
            "entries": [
                {
                    "id": e.id,
                    "source_text": e.source_text,
                    "target_text": e.target_text,
                    "context": e.context,
                    "domain": e.domain,
                    "quality_score": e.quality_score,
                    "usage_count": e.usage_count,
                    "is_approved": e.is_approved,
                }
                for e in self._entries.values()
            ],
        }

    def import_from_dict(self, data: dict[str, Any]) -> int:
        """
        Import entries from dictionary.

        Returns:
            Number of entries imported
        """
        count = 0

        for entry_data in data.get("entries", []):
            self.add_entry(
                source_text=entry_data["source_text"],
                target_text=entry_data["target_text"],
                context=entry_data.get("context", ""),
                domain=entry_data.get("domain", ""),
                quality_score=entry_data.get("quality_score", 1.0),
            )
            count += 1

        return count


class TranslationMemoryManager:
    """
    Manages translation memories for multiple language pairs.
    """
    __slots__ = ("_memories", "_source_language")

    def __init__(self, source_language: str = "en"):
        """
        Initialize manager.

        Args:
            source_language: Default source language
        """
        self._source_language = source_language
        self._memories: dict[tuple[str, str], TranslationMemory] = {}

    @property
    def source_language(self) -> str:
        """Get source language."""
        return self._source_language

    def get_or_create_memory(
        self,
        target_language: str,
        source_language: Optional[str] = None
    ) -> TranslationMemory:
        """
        Get or create a translation memory for a language pair.

        Args:
            target_language: Target language
            source_language: Source language (defaults to manager's source)

        Returns:
            Translation memory
        """
        source = source_language or self._source_language
        key = (source, target_language)

        if key not in self._memories:
            self._memories[key] = TranslationMemory(source, target_language)

        return self._memories[key]

    def get_memory(
        self,
        target_language: str,
        source_language: Optional[str] = None
    ) -> Optional[TranslationMemory]:
        """Get a translation memory if it exists."""
        source = source_language or self._source_language
        key = (source, target_language)
        return self._memories.get(key)

    def get_all_memories(self) -> list[TranslationMemory]:
        """Get all translation memories."""
        return list(self._memories.values())

    def find_translation(
        self,
        source_text: str,
        target_language: str,
        context: Optional[str] = None
    ) -> Optional[TMMatch]:
        """
        Find best translation for a text.

        Args:
            source_text: Text to translate
            target_language: Target language
            context: Optional context

        Returns:
            Best match or None
        """
        memory = self.get_memory(target_language)
        if memory is None:
            return None

        matches = memory.find_fuzzy(source_text, context=context, max_results=1)
        return matches[0] if matches else None

    def add_translation(
        self,
        source_text: str,
        target_text: str,
        target_language: str,
        context: str = "",
        domain: str = ""
    ) -> TMEntry:
        """
        Add a translation to memory.

        Args:
            source_text: Source text
            target_text: Translation
            target_language: Target language
            context: Context
            domain: Domain

        Returns:
            Created entry
        """
        memory = self.get_or_create_memory(target_language)
        return memory.add_entry(
            source_text=source_text,
            target_text=target_text,
            context=context,
            domain=domain,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for all memories."""
        stats = {
            "total_entries": 0,
            "language_pairs": [],
        }

        for (source, target), memory in self._memories.items():
            count = memory.entry_count
            stats["total_entries"] += count
            stats["language_pairs"].append({
                "source": source,
                "target": target,
                "entries": count,
            })

        return stats

    def export_all(self) -> dict[str, Any]:
        """Export all memories."""
        return {
            "source_language": self._source_language,
            "memories": {
                f"{source}-{target}": memory.export_to_dict()
                for (source, target), memory in self._memories.items()
            },
        }

    def import_all(self, data: dict[str, Any]) -> int:
        """
        Import all memories from dictionary.

        Returns:
            Total entries imported
        """
        self._source_language = data.get("source_language", self._source_language)
        total = 0

        for key, memory_data in data.get("memories", {}).items():
            parts = key.split("-")
            if len(parts) == 2:
                source, target = parts
                memory = self.get_or_create_memory(target, source)
                total += memory.import_from_dict(memory_data)

        return total

    def suggest_translations(
        self,
        source_text: str,
        target_language: str,
        max_suggestions: int = 3
    ) -> list[TMMatch]:
        """
        Get translation suggestions from memory.

        Args:
            source_text: Text to translate
            target_language: Target language
            max_suggestions: Maximum suggestions

        Returns:
            List of suggested translations
        """
        memory = self.get_memory(target_language)
        if memory is None:
            return []

        return memory.find_fuzzy(
            source_text,
            min_similarity=0.5,
            max_results=max_suggestions
        )
