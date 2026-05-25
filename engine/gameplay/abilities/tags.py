"""
Gameplay Tag System.

Provides hierarchical gameplay tags for categorizing abilities, effects, and
game objects. Supports exact, parent, child, and wildcard matching.

Example tags:
    ability.offensive.fire
    status.buff.strength
    item.consumable.potion.health
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    Iterator,
    Optional,
    Set,
    TypeVar,
)

from engine.gameplay.abilities.constants import (
    MAX_TAG_DEPTH,
    TAG_SEPARATOR,
    TAG_WILDCARD,
    TAG_REGISTRY_CACHE_SIZE,
)

T = TypeVar("T")


# =============================================================================
# GAMEPLAY TAG
# =============================================================================


@dataclass(frozen=True, slots=True)
class GameplayTag:
    """
    A hierarchical gameplay tag.

    Tags form a hierarchy separated by dots (e.g., "ability.offensive.fire").
    This enables flexible matching:
    - Exact: tag == "ability.offensive.fire"
    - Parent: tag.is_child_of("ability.offensive")
    - Child: tag.is_parent_of("ability.offensive.fire.fireball")
    - Wildcard: tag.matches("ability.*.fire")
    """

    hierarchy: str

    def __post_init__(self) -> None:
        """Validate tag hierarchy."""
        if not self.hierarchy:
            raise ValueError("Tag hierarchy cannot be empty")

        parts = self.hierarchy.split(TAG_SEPARATOR)

        if len(parts) > MAX_TAG_DEPTH:
            raise ValueError(
                f"Tag depth {len(parts)} exceeds maximum {MAX_TAG_DEPTH}"
            )

        for part in parts:
            if not part:
                raise ValueError("Tag parts cannot be empty")
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$|^\*$", part):
                raise ValueError(
                    f"Invalid tag part '{part}': must be alphanumeric with "
                    "underscores, starting with letter/underscore, or '*'"
                )

    @property
    def parts(self) -> tuple[str, ...]:
        """Get tag hierarchy as tuple of parts."""
        return tuple(self.hierarchy.split(TAG_SEPARATOR))

    @property
    def depth(self) -> int:
        """Get the depth of this tag (number of parts)."""
        return len(self.parts)

    @property
    def parent(self) -> Optional[GameplayTag]:
        """Get parent tag, or None if this is a root tag."""
        parts = self.parts
        if len(parts) <= 1:
            return None
        return GameplayTag(TAG_SEPARATOR.join(parts[:-1]))

    @property
    def root(self) -> GameplayTag:
        """Get the root (first part) of this tag."""
        return GameplayTag(self.parts[0])

    @property
    def leaf(self) -> str:
        """Get the leaf (last part) of this tag."""
        return self.parts[-1]

    def is_child_of(self, other: GameplayTag | str) -> bool:
        """Check if this tag is a child of (or equal to) another tag."""
        if isinstance(other, str):
            other = GameplayTag(other)

        other_parts = other.parts
        self_parts = self.parts

        if len(self_parts) < len(other_parts):
            return False

        return self_parts[: len(other_parts)] == other_parts

    def is_parent_of(self, other: GameplayTag | str) -> bool:
        """Check if this tag is a parent of (or equal to) another tag."""
        if isinstance(other, str):
            other = GameplayTag(other)
        return other.is_child_of(self)

    def is_sibling_of(self, other: GameplayTag | str) -> bool:
        """Check if this tag shares the same parent as another tag."""
        if isinstance(other, str):
            other = GameplayTag(other)

        self_parent = self.parent
        other_parent = other.parent

        if self_parent is None and other_parent is None:
            return True
        if self_parent is None or other_parent is None:
            return False
        return self_parent == other_parent

    def matches(self, pattern: str) -> bool:
        """
        Check if this tag matches a pattern with optional wildcards.

        Patterns support:
        - Exact match: "ability.offensive.fire"
        - Single wildcard: "ability.*.fire" (any single part)
        - Trailing wildcard: "ability.*" (any descendants)
        """
        pattern_parts = pattern.split(TAG_SEPARATOR)
        self_parts = self.parts

        # Handle trailing wildcard
        if pattern_parts and pattern_parts[-1] == TAG_WILDCARD:
            prefix_parts = pattern_parts[:-1]
            if len(self_parts) < len(prefix_parts):
                return False
            for i, part in enumerate(prefix_parts):
                if part != TAG_WILDCARD and part != self_parts[i]:
                    return False
            return True

        # Exact length match with wildcards
        if len(pattern_parts) != len(self_parts):
            return False

        for pattern_part, self_part in zip(pattern_parts, self_parts):
            if pattern_part != TAG_WILDCARD and pattern_part != self_part:
                return False

        return True

    def child(self, name: str) -> GameplayTag:
        """Create a child tag with the given name."""
        return GameplayTag(f"{self.hierarchy}{TAG_SEPARATOR}{name}")

    def ancestors(self) -> Iterator[GameplayTag]:
        """Iterate through all ancestor tags (parent to root)."""
        current = self.parent
        while current is not None:
            yield current
            current = current.parent

    def __str__(self) -> str:
        return self.hierarchy

    def __repr__(self) -> str:
        return f"GameplayTag({self.hierarchy!r})"


# =============================================================================
# GAMEPLAY TAG CONTAINER
# =============================================================================


@dataclass
class GameplayTagContainer:
    """
    Container for managing a set of gameplay tags.

    Provides efficient querying for tag presence, matching, and filtering.
    """

    _tags: Set[GameplayTag] = field(default_factory=set)
    _on_change: Optional[Callable[[GameplayTagContainer], None]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def tags(self) -> FrozenSet[GameplayTag]:
        """Get immutable view of contained tags."""
        return frozenset(self._tags)

    def add(self, tag: GameplayTag | str) -> bool:
        """Add a tag. Returns True if tag was added (not already present)."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)

        if tag in self._tags:
            return False

        self._tags.add(tag)
        self._notify_change()
        return True

    def add_many(self, tags: Iterable[GameplayTag | str]) -> int:
        """Add multiple tags. Returns count of tags added."""
        count = 0
        for tag in tags:
            if isinstance(tag, str):
                tag = GameplayTag(tag)
            if tag not in self._tags:
                self._tags.add(tag)
                count += 1

        if count > 0:
            self._notify_change()
        return count

    def remove(self, tag: GameplayTag | str) -> bool:
        """Remove a tag. Returns True if tag was removed."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)

        if tag not in self._tags:
            return False

        self._tags.remove(tag)
        self._notify_change()
        return True

    def remove_many(self, tags: Iterable[GameplayTag | str]) -> int:
        """Remove multiple tags. Returns count of tags removed."""
        count = 0
        for tag in tags:
            if isinstance(tag, str):
                tag = GameplayTag(tag)
            if tag in self._tags:
                self._tags.remove(tag)
                count += 1

        if count > 0:
            self._notify_change()
        return count

    def clear(self) -> int:
        """Remove all tags. Returns count of tags removed."""
        count = len(self._tags)
        if count > 0:
            self._tags.clear()
            self._notify_change()
        return count

    def has(self, tag: GameplayTag | str) -> bool:
        """Check if container has exactly this tag."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)
        return tag in self._tags

    def has_any(self, tags: Iterable[GameplayTag | str]) -> bool:
        """Check if container has any of the given tags."""
        for tag in tags:
            if self.has(tag):
                return True
        return False

    def has_all(self, tags: Iterable[GameplayTag | str]) -> bool:
        """Check if container has all of the given tags."""
        for tag in tags:
            if not self.has(tag):
                return False
        return True

    def has_any_matching(self, pattern: str) -> bool:
        """Check if any tag matches the pattern."""
        for tag in self._tags:
            if tag.matches(pattern):
                return True
        return False

    def has_parent_of(self, tag: GameplayTag | str) -> bool:
        """Check if container has a parent tag of the given tag."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)
        for container_tag in self._tags:
            if container_tag.is_parent_of(tag):
                return True
        return False

    def has_child_of(self, tag: GameplayTag | str) -> bool:
        """Check if container has a child tag of the given tag."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)
        for container_tag in self._tags:
            if container_tag.is_child_of(tag):
                return True
        return False

    def filter_matching(self, pattern: str) -> FrozenSet[GameplayTag]:
        """Get all tags matching the pattern."""
        return frozenset(tag for tag in self._tags if tag.matches(pattern))

    def filter_children_of(self, tag: GameplayTag | str) -> FrozenSet[GameplayTag]:
        """Get all tags that are children of the given tag."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)
        return frozenset(t for t in self._tags if t.is_child_of(tag))

    def filter_parents_of(self, tag: GameplayTag | str) -> FrozenSet[GameplayTag]:
        """Get all tags that are parents of the given tag."""
        if isinstance(tag, str):
            tag = GameplayTag(tag)
        return frozenset(t for t in self._tags if t.is_parent_of(tag))

    def intersection(self, other: GameplayTagContainer) -> FrozenSet[GameplayTag]:
        """Get tags present in both containers."""
        return frozenset(self._tags & other._tags)

    def union(self, other: GameplayTagContainer) -> FrozenSet[GameplayTag]:
        """Get tags present in either container."""
        return frozenset(self._tags | other._tags)

    def difference(self, other: GameplayTagContainer) -> FrozenSet[GameplayTag]:
        """Get tags in this container but not in other."""
        return frozenset(self._tags - other._tags)

    def _notify_change(self) -> None:
        """Notify callback of tag changes."""
        if self._on_change is not None:
            self._on_change(self)

    def __len__(self) -> int:
        return len(self._tags)

    def __iter__(self) -> Iterator[GameplayTag]:
        return iter(self._tags)

    def __contains__(self, tag: GameplayTag | str) -> bool:
        return self.has(tag)

    def __bool__(self) -> bool:
        return len(self._tags) > 0


# =============================================================================
# GAMEPLAY TAG QUERY
# =============================================================================


@dataclass(frozen=True, slots=True)
class GameplayTagQuery:
    """
    Query for filtering entities by their gameplay tags.

    Supports complex queries with required, optional, and excluded tags.
    """

    require_all: FrozenSet[GameplayTag] = field(default_factory=frozenset)
    require_any: FrozenSet[GameplayTag] = field(default_factory=frozenset)
    exclude: FrozenSet[GameplayTag] = field(default_factory=frozenset)

    @classmethod
    def all_of(cls, *tags: GameplayTag | str) -> GameplayTagQuery:
        """Create query requiring all given tags."""
        parsed = frozenset(
            t if isinstance(t, GameplayTag) else GameplayTag(t) for t in tags
        )
        return cls(require_all=parsed)

    @classmethod
    def any_of(cls, *tags: GameplayTag | str) -> GameplayTagQuery:
        """Create query requiring any of the given tags."""
        parsed = frozenset(
            t if isinstance(t, GameplayTag) else GameplayTag(t) for t in tags
        )
        return cls(require_any=parsed)

    @classmethod
    def none_of(cls, *tags: GameplayTag | str) -> GameplayTagQuery:
        """Create query excluding all given tags."""
        parsed = frozenset(
            t if isinstance(t, GameplayTag) else GameplayTag(t) for t in tags
        )
        return cls(exclude=parsed)

    def and_all(self, *tags: GameplayTag | str) -> GameplayTagQuery:
        """Add required tags (all must be present)."""
        parsed = frozenset(
            t if isinstance(t, GameplayTag) else GameplayTag(t) for t in tags
        )
        return GameplayTagQuery(
            require_all=self.require_all | parsed,
            require_any=self.require_any,
            exclude=self.exclude,
        )

    def and_any(self, *tags: GameplayTag | str) -> GameplayTagQuery:
        """Add optional tags (any can be present)."""
        parsed = frozenset(
            t if isinstance(t, GameplayTag) else GameplayTag(t) for t in tags
        )
        return GameplayTagQuery(
            require_all=self.require_all,
            require_any=self.require_any | parsed,
            exclude=self.exclude,
        )

    def and_none(self, *tags: GameplayTag | str) -> GameplayTagQuery:
        """Add excluded tags (none can be present)."""
        parsed = frozenset(
            t if isinstance(t, GameplayTag) else GameplayTag(t) for t in tags
        )
        return GameplayTagQuery(
            require_all=self.require_all,
            require_any=self.require_any,
            exclude=self.exclude | parsed,
        )

    def matches(self, container: GameplayTagContainer) -> bool:
        """Check if a tag container matches this query."""
        # Check excluded tags first (fastest rejection)
        for tag in self.exclude:
            if container.has(tag):
                return False

        # Check required tags (all must be present)
        for tag in self.require_all:
            if not container.has(tag):
                return False

        # Check optional tags (any must be present, if specified)
        if self.require_any:
            if not container.has_any(self.require_any):
                return False

        return True

    def matches_with_parents(self, container: GameplayTagContainer) -> bool:
        """
        Check if container matches, considering parent tag relationships.

        A tag "ability.offensive.fire" also matches requirement "ability.offensive".
        """
        # Check excluded tags first
        for tag in self.exclude:
            if container.has_child_of(tag):
                return False

        # Check required tags
        for tag in self.require_all:
            if not container.has_child_of(tag):
                return False

        # Check optional tags
        if self.require_any:
            for tag in self.require_any:
                if container.has_child_of(tag):
                    return True
            return False

        return True


# =============================================================================
# DECORATOR
# =============================================================================


def gameplay_tag(hierarchy: str) -> Callable[[type[T]], type[T]]:
    """
    Decorator to attach a gameplay tag to a class.

    Usage:
        @gameplay_tag("ability.offensive.fire")
        class Fireball:
            pass
    """

    def decorator(cls: type[T]) -> type[T]:
        tag = GameplayTag(hierarchy)
        cls._gameplay_tag = True  # type: ignore
        cls._tag = tag  # type: ignore
        cls._tag_hierarchy = hierarchy  # type: ignore

        # Ensure tag container exists
        if not hasattr(cls, "_tag_container"):
            cls._tag_container = GameplayTagContainer()  # type: ignore
        cls._tag_container.add(tag)  # type: ignore

        return cls

    return decorator


# =============================================================================
# TAG REGISTRY
# =============================================================================


class GameplayTagRegistry:
    """
    Global registry for gameplay tags.

    Provides caching and efficient lookup of tags.
    """

    _instance: Optional[GameplayTagRegistry] = None
    _tags: dict[str, GameplayTag]

    def __new__(cls) -> GameplayTagRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tags = {}
        return cls._instance

    @classmethod
    def get(cls, hierarchy: str) -> GameplayTag:
        """Get or create a tag from the registry."""
        instance = cls()
        if hierarchy not in instance._tags:
            instance._tags[hierarchy] = GameplayTag(hierarchy)
        return instance._tags[hierarchy]

    @classmethod
    @lru_cache(maxsize=TAG_REGISTRY_CACHE_SIZE)
    def get_cached(cls, hierarchy: str) -> GameplayTag:
        """Get a cached tag (faster for repeated lookups)."""
        return cls.get(hierarchy)

    @classmethod
    def all_tags(cls) -> FrozenSet[GameplayTag]:
        """Get all registered tags."""
        instance = cls()
        return frozenset(instance._tags.values())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        instance = cls()
        instance._tags.clear()
        cls.get_cached.cache_clear()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "GameplayTag",
    "GameplayTagContainer",
    "GameplayTagQuery",
    "GameplayTagRegistry",
    "gameplay_tag",
]
