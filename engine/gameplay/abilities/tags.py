"""
Gameplay Tag System.

Provides hierarchical gameplay tags for categorizing abilities, effects, and
game objects. Supports exact, parent, child, and wildcard matching.

Wired to Foundation Registry for runtime discovery:
    Registry.query(tag="gameplay_tag") -> all tags
    Registry.query(tag="gameplay_tag", parent="Combat") -> combat subtags

Example tags:
    ability.offensive.fire
    status.buff.strength
    item.consumable.potion.health
    Combat.Damage.Fire
"""

from __future__ import annotations

import re
import warnings
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
# FOUNDATION REGISTRY INTEGRATION
# =============================================================================


def _get_foundation_registry():
    """Get Foundation Registry, returning None if unavailable."""
    try:
        from foundation import registry
        return registry
    except ImportError:
        return None


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

    def to_dict(self) -> dict[str, Any]:
        """Serialize tag to dictionary."""
        return {
            "hierarchy": self.hierarchy,
            "parts": list(self.parts),
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameplayTag:
        """Deserialize tag from dictionary."""
        return cls(hierarchy=data["hierarchy"])

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
    Supports hierarchical matching: "Combat" matches "Combat.Damage.Fire".
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
        return self.add_tag(tag)

    def add_tag(self, tag: GameplayTag | str) -> bool:
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
        return self.remove_tag(tag)

    def remove_tag(self, tag: GameplayTag | str) -> bool:
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

    def has_tag(self, tag: GameplayTag | str, hierarchical: bool = False) -> bool:
        """
        Check if container has this tag.

        Args:
            tag: The tag to check for.
            hierarchical: If True, parent tags match children.
                         E.g., "Combat" matches "Combat.Damage.Fire"
        """
        if isinstance(tag, str):
            tag = GameplayTag(tag)

        # Exact match first
        if tag in self._tags:
            return True

        # Hierarchical match: check if any contained tag is a child of the query
        if hierarchical:
            return self.has_child_of(tag)

        return False

    def has_any(self, tags: Iterable[GameplayTag | str], hierarchical: bool = False) -> bool:
        """Check if container has any of the given tags."""
        for tag in tags:
            if self.has_tag(tag, hierarchical=hierarchical):
                return True
        return False

    def has_all(self, tags: Iterable[GameplayTag | str], hierarchical: bool = False) -> bool:
        """Check if container has all of the given tags."""
        for tag in tags:
            if not self.has_tag(tag, hierarchical=hierarchical):
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize container to dictionary."""
        return {
            "tags": [tag.hierarchy for tag in self._tags],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameplayTagContainer:
        """Deserialize container from dictionary."""
        container = cls()
        for hierarchy in data.get("tags", []):
            container.add_tag(hierarchy)
        return container

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
# DECORATOR - WIRED TO FOUNDATION REGISTRY
# =============================================================================


def gameplay_tag(
    name: str,
    parent: Optional[str] = None,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator to attach a gameplay tag to a class and register with Foundation.

    Registers the class with Foundation Registry:
    - Tagged as "gameplay_tag" for discovery
    - Metadata includes name, parent, and hierarchy

    Args:
        name: The tag hierarchy (e.g., "Combat.Damage.Fire")
        parent: Optional parent tag for hierarchical structure

    Usage:
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        @gameplay_tag("Buff.Speed", parent="Buff")
        class SpeedBuff:
            pass

    Discovery:
        Registry.query(tag="gameplay_tag") -> all tags
        Registry.query(tag="gameplay_tag", parent="Combat") -> combat subtags
    """

    def decorator(cls: type[T]) -> type[T]:
        # Validate tag hierarchy
        hierarchy = name
        if parent:
            # Verify name starts with parent
            if not name.startswith(parent + TAG_SEPARATOR) and name != parent:
                # Auto-prefix with parent if not already present
                hierarchy = f"{parent}{TAG_SEPARATOR}{name}"

        # Create and attach the tag
        tag = GameplayTag(hierarchy)
        cls._gameplay_tag = True  # type: ignore
        cls._tag = tag  # type: ignore
        cls._tag_hierarchy = hierarchy  # type: ignore
        cls._tag_parent = parent  # type: ignore
        cls._tag_name = name  # type: ignore

        # Ensure tag container exists
        if not hasattr(cls, "_tag_container"):
            cls._tag_container = GameplayTagContainer()  # type: ignore
        cls._tag_container.add(tag)  # type: ignore

        # Register with Foundation Registry
        foundation_registry = _get_foundation_registry()
        if foundation_registry is not None:
            try:
                # Register class if not already registered
                if not foundation_registry.is_registered(cls):
                    foundation_registry.register(cls, track_instances=False)

                # Add gameplay_tag discovery tag
                foundation_registry.add_tag(cls, "gameplay_tag")

                # Set metadata for queries
                foundation_registry.set_metadata(cls, "tag_hierarchy", hierarchy)
                foundation_registry.set_metadata(cls, "tag_name", name)
                if parent:
                    foundation_registry.set_metadata(cls, "parent", parent)

                # Set root tag for hierarchical queries
                root = tag.root.hierarchy
                foundation_registry.set_metadata(cls, "root", root)

                # Track all parent tags for matching
                parent_hierarchies = [p.hierarchy for p in tag.ancestors()]
                foundation_registry.set_metadata(cls, "ancestors", parent_hierarchies)

            except Exception as exc:
                warnings.warn(
                    f"gameplay_tag: Foundation registration failed for {cls.__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return cls

    return decorator


# =============================================================================
# TAG REGISTRY
# =============================================================================


class GameplayTagRegistry:
    """
    Global registry for gameplay tags.

    Provides caching and efficient lookup of tags.
    Also integrates with Foundation Registry for cross-system discovery.
    """

    _instance: Optional[GameplayTagRegistry] = None
    _tags: dict[str, GameplayTag]
    _classes: dict[str, type]  # hierarchy -> decorated class

    def __new__(cls) -> GameplayTagRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tags = {}
            cls._instance._classes = {}
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
    def register_class(cls, hierarchy: str, target_cls: type) -> None:
        """Register a decorated class by its tag hierarchy."""
        instance = cls()
        instance._classes[hierarchy] = target_cls

    @classmethod
    def get_class(cls, hierarchy: str) -> Optional[type]:
        """Get the class decorated with a specific tag hierarchy."""
        instance = cls()
        return instance._classes.get(hierarchy)

    @classmethod
    def all_tags(cls) -> FrozenSet[GameplayTag]:
        """Get all registered tags."""
        instance = cls()
        return frozenset(instance._tags.values())

    @classmethod
    def all_classes(cls) -> dict[str, type]:
        """Get all registered classes by hierarchy."""
        instance = cls()
        return dict(instance._classes)

    @classmethod
    def query_by_parent(cls, parent: str) -> list[type]:
        """Query all classes whose tag is a child of the given parent."""
        instance = cls()
        parent_tag = GameplayTag(parent)
        result = []
        for hierarchy, target_cls in instance._classes.items():
            tag = instance._tags.get(hierarchy)
            if tag and tag.is_child_of(parent_tag):
                result.append(target_cls)
        return result

    @classmethod
    def query_foundation(
        cls,
        tag: str = "gameplay_tag",
        **metadata_filters: Any,
    ) -> list[type]:
        """
        Query Foundation Registry for gameplay tags.

        Args:
            tag: The registry tag to filter by (default: "gameplay_tag")
            **metadata_filters: Additional metadata filters (parent, root, etc.)

        Returns:
            List of classes matching the criteria.

        Examples:
            >>> GameplayTagRegistry.query_foundation()  # All gameplay tags
            >>> GameplayTagRegistry.query_foundation(parent="Combat")  # Combat subtags
            >>> GameplayTagRegistry.query_foundation(root="Buff")  # All buff tags
        """
        foundation_registry = _get_foundation_registry()
        if foundation_registry is None:
            return []
        return foundation_registry.query(tag=tag, **metadata_filters)

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (mainly for testing)."""
        instance = cls()
        instance._tags.clear()
        instance._classes.clear()
        cls.get_cached.cache_clear()


# =============================================================================
# ABILITY INTEGRATION
# =============================================================================


def ability_with_tags(
    required_tags: Optional[list[str]] = None,
    granted_tags: Optional[list[str]] = None,
    blocked_by_tags: Optional[list[str]] = None,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator that adds tag requirements to an ability class.

    Args:
        required_tags: Tags that must be present to activate the ability.
        granted_tags: Tags applied when the ability is active.
        blocked_by_tags: Tags that prevent ability activation.

    Usage:
        @ability_with_tags(
            required_tags=["Combat"],
            granted_tags=["Buff.Speed"],
            blocked_by_tags=["Status.Stunned"]
        )
        class SprintAbility:
            pass
    """

    def decorator(cls: type[T]) -> type[T]:
        # Parse tag strings to GameplayTag objects
        cls._required_tags = GameplayTagContainer()  # type: ignore
        cls._granted_tags = GameplayTagContainer()  # type: ignore
        cls._blocked_by_tags = GameplayTagContainer()  # type: ignore

        if required_tags:
            for tag_str in required_tags:
                cls._required_tags.add_tag(tag_str)

        if granted_tags:
            for tag_str in granted_tags:
                cls._granted_tags.add_tag(tag_str)

        if blocked_by_tags:
            for tag_str in blocked_by_tags:
                cls._blocked_by_tags.add_tag(tag_str)

        # Add check methods
        def can_activate(self, owner_tags: GameplayTagContainer) -> bool:
            """Check if ability can be activated based on owner's tags."""
            # Check blocked tags
            if self._blocked_by_tags:
                for blocked_tag in self._blocked_by_tags:
                    if owner_tags.has_tag(blocked_tag, hierarchical=True):
                        return False

            # Check required tags
            if self._required_tags:
                for req_tag in self._required_tags:
                    if not owner_tags.has_tag(req_tag, hierarchical=True):
                        return False

            return True

        def apply_granted_tags(self, target_tags: GameplayTagContainer) -> None:
            """Apply granted tags to target."""
            for tag in self._granted_tags:
                target_tags.add_tag(tag)

        def remove_granted_tags(self, target_tags: GameplayTagContainer) -> None:
            """Remove granted tags from target."""
            for tag in self._granted_tags:
                target_tags.remove_tag(tag)

        cls.can_activate = can_activate  # type: ignore
        cls.apply_granted_tags = apply_granted_tags  # type: ignore
        cls.remove_granted_tags = remove_granted_tags  # type: ignore

        return cls

    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "GameplayTag",
    "GameplayTagContainer",
    "GameplayTagQuery",
    "GameplayTagRegistry",
    # Decorators
    "gameplay_tag",
    "ability_with_tags",
]
