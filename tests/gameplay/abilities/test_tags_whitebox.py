"""
WHITEBOX Tests for the Gameplay Tag System.

Comprehensive internal testing of gameplay tags with full source access.

Tests cover:
- GameplayTag hierarchy parsing and validation
- GameplayTag matching (exact, parent, child, sibling, wildcard)
- GameplayTagContainer internal operations
- GameplayTagQuery complex queries
- GameplayTagRegistry caching and lookup
- @gameplay_tag decorator mechanics
- @ability_with_tags decorator mechanics
- Foundation Registry integration
- Edge cases: max depth, invalid characters, wildcards

Total: 50+ tests for tag system internals
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import pytest

from engine.gameplay.abilities.constants import (
    MAX_TAG_DEPTH,
    TAG_SEPARATOR,
    TAG_WILDCARD,
    TAG_REGISTRY_CACHE_SIZE,
)
from engine.gameplay.abilities.tags import (
    GameplayTag,
    GameplayTagContainer,
    GameplayTagQuery,
    GameplayTagRegistry,
    ability_with_tags,
    gameplay_tag,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear tag registry before and after each test."""
    GameplayTagRegistry.clear()
    yield
    GameplayTagRegistry.clear()


# =============================================================================
# GAMEPLAY TAG PARSING TESTS
# =============================================================================


class TestGameplayTagParsing:
    """Whitebox tests for GameplayTag hierarchy parsing."""

    def test_simple_tag_creation(self):
        """Test creating a simple single-part tag."""
        tag = GameplayTag("Combat")
        assert tag.hierarchy == "Combat"
        assert tag.parts == ("Combat",)
        assert tag.depth == 1

    def test_hierarchical_tag_creation(self):
        """Test creating a multi-part hierarchical tag."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.hierarchy == "ability.offensive.fire"
        assert tag.parts == ("ability", "offensive", "fire")
        assert tag.depth == 3

    def test_tag_parent_property(self):
        """Test parent property returns parent tag."""
        tag = GameplayTag("ability.offensive.fire")
        parent = tag.parent

        assert parent is not None
        assert parent.hierarchy == "ability.offensive"

    def test_tag_parent_of_root(self):
        """Test parent of root tag is None."""
        tag = GameplayTag("Combat")
        assert tag.parent is None

    def test_tag_root_property(self):
        """Test root property returns first part."""
        tag = GameplayTag("ability.offensive.fire.fireball")
        root = tag.root

        assert root.hierarchy == "ability"

    def test_tag_leaf_property(self):
        """Test leaf property returns last part."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.leaf == "fire"

    def test_tag_empty_raises_error(self):
        """Test empty tag hierarchy raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GameplayTag("")

    def test_tag_empty_part_raises_error(self):
        """Test tag with empty part raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GameplayTag("ability..fire")

    def test_tag_max_depth_exceeded(self):
        """Test exceeding max depth raises error."""
        parts = ["part"] * (MAX_TAG_DEPTH + 1)
        hierarchy = TAG_SEPARATOR.join(parts)

        with pytest.raises(ValueError, match="exceeds maximum"):
            GameplayTag(hierarchy)

    def test_tag_invalid_characters(self):
        """Test invalid characters raise error."""
        with pytest.raises(ValueError, match="Invalid tag part"):
            GameplayTag("ability.fire-ball")  # Hyphen not allowed

    def test_tag_numeric_start_raises_error(self):
        """Test part starting with number raises error."""
        with pytest.raises(ValueError, match="Invalid tag part"):
            GameplayTag("ability.1stLevel")

    def test_tag_wildcard_part_allowed(self):
        """Test wildcard character is allowed in parts."""
        tag = GameplayTag("ability.*.fire")
        assert "*" in tag.parts

    def test_tag_underscore_allowed(self):
        """Test underscore is allowed in tag parts."""
        tag = GameplayTag("ability_system.buff_effect")
        assert tag.parts == ("ability_system", "buff_effect")


# =============================================================================
# GAMEPLAY TAG MATCHING TESTS
# =============================================================================


class TestGameplayTagMatching:
    """Whitebox tests for GameplayTag matching operations."""

    def test_is_child_of_exact_match(self):
        """Test is_child_of matches exact tag."""
        tag = GameplayTag("ability.offensive")
        assert tag.is_child_of("ability.offensive")

    def test_is_child_of_parent(self):
        """Test is_child_of matches parent."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.is_child_of("ability.offensive")
        assert tag.is_child_of("ability")

    def test_is_child_of_not_parent(self):
        """Test is_child_of doesn't match non-parent."""
        tag = GameplayTag("ability.offensive.fire")
        assert not tag.is_child_of("ability.defensive")
        assert not tag.is_child_of("status")

    def test_is_child_of_with_string(self):
        """Test is_child_of accepts string."""
        tag = GameplayTag("ability.offensive")
        assert tag.is_child_of("ability")

    def test_is_parent_of_child(self):
        """Test is_parent_of matches child."""
        tag = GameplayTag("ability")
        child = GameplayTag("ability.offensive.fire")
        assert tag.is_parent_of(child)

    def test_is_parent_of_not_child(self):
        """Test is_parent_of doesn't match non-child."""
        tag = GameplayTag("ability")
        other = GameplayTag("status.buff")
        assert not tag.is_parent_of(other)

    def test_is_sibling_of_same_parent(self):
        """Test is_sibling_of with same parent."""
        a = GameplayTag("ability.offensive")
        b = GameplayTag("ability.defensive")
        assert a.is_sibling_of(b)

    def test_is_sibling_of_different_parent(self):
        """Test is_sibling_of with different parents."""
        a = GameplayTag("ability.offensive")
        b = GameplayTag("status.buff")
        assert not a.is_sibling_of(b)

    def test_is_sibling_of_root_tags(self):
        """Test root tags are siblings of each other."""
        a = GameplayTag("ability")
        b = GameplayTag("status")
        assert a.is_sibling_of(b)

    def test_matches_exact(self):
        """Test matches with exact pattern."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.matches("ability.offensive.fire")
        assert not tag.matches("ability.offensive.ice")

    def test_matches_single_wildcard(self):
        """Test matches with single wildcard."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.matches("ability.*.fire")
        assert tag.matches("*.offensive.fire")
        assert not tag.matches("ability.*.ice")

    def test_matches_trailing_wildcard(self):
        """Test matches with trailing wildcard."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.matches("ability.*")
        assert tag.matches("ability.offensive.*")
        assert not tag.matches("status.*")

    def test_matches_multiple_wildcards(self):
        """Test matches with multiple wildcards."""
        tag = GameplayTag("ability.offensive.fire")
        assert tag.matches("*.*.*")
        assert tag.matches("ability.*.*")

    def test_matches_wrong_length(self):
        """Test matches fails with wrong depth."""
        tag = GameplayTag("ability.offensive.fire")
        assert not tag.matches("ability.offensive")
        assert not tag.matches("ability.offensive.fire.extra")


# =============================================================================
# GAMEPLAY TAG NAVIGATION TESTS
# =============================================================================


class TestGameplayTagNavigation:
    """Tests for tag hierarchy navigation."""

    def test_child_method(self):
        """Test creating child tag."""
        parent = GameplayTag("ability")
        child = parent.child("offensive")

        assert child.hierarchy == "ability.offensive"

    def test_ancestors_iterator(self):
        """Test ancestors iterator."""
        tag = GameplayTag("ability.offensive.fire.fireball")

        ancestors = list(tag.ancestors())

        assert len(ancestors) == 3
        assert ancestors[0].hierarchy == "ability.offensive.fire"
        assert ancestors[1].hierarchy == "ability.offensive"
        assert ancestors[2].hierarchy == "ability"

    def test_ancestors_of_root(self):
        """Test ancestors of root tag is empty."""
        tag = GameplayTag("Combat")
        ancestors = list(tag.ancestors())
        assert ancestors == []


# =============================================================================
# GAMEPLAY TAG CONTAINER TESTS
# =============================================================================


class TestGameplayTagContainerInternals:
    """Whitebox tests for GameplayTagContainer operations."""

    def test_container_initialization(self):
        """Test container initializes empty."""
        container = GameplayTagContainer()
        assert len(container) == 0
        assert container._tags == set()

    def test_add_tag_returns_true_on_new(self):
        """Test add returns True when tag is new."""
        container = GameplayTagContainer()
        result = container.add("ability.fire")
        assert result is True

    def test_add_tag_returns_false_on_duplicate(self):
        """Test add returns False for duplicate."""
        container = GameplayTagContainer()
        container.add("ability.fire")
        result = container.add("ability.fire")
        assert result is False

    def test_add_many_returns_count(self):
        """Test add_many returns count of added tags."""
        container = GameplayTagContainer()
        count = container.add_many(["a", "b", "c"])
        assert count == 3

    def test_add_many_skips_duplicates(self):
        """Test add_many skips duplicates."""
        container = GameplayTagContainer()
        container.add("a")
        count = container.add_many(["a", "b", "c"])
        assert count == 2

    def test_remove_returns_true_on_existing(self):
        """Test remove returns True for existing tag."""
        container = GameplayTagContainer()
        container.add("ability.fire")
        result = container.remove("ability.fire")
        assert result is True

    def test_remove_returns_false_on_missing(self):
        """Test remove returns False for missing tag."""
        container = GameplayTagContainer()
        result = container.remove("nonexistent")
        assert result is False

    def test_clear_returns_count(self):
        """Test clear returns count of removed tags."""
        container = GameplayTagContainer()
        container.add_many(["a", "b", "c"])
        count = container.clear()
        assert count == 3

    def test_has_exact_match(self):
        """Test has checks exact match."""
        container = GameplayTagContainer()
        container.add("ability.fire")

        assert container.has("ability.fire")
        assert not container.has("ability")

    def test_has_tag_hierarchical(self):
        """Test has_tag with hierarchical matching."""
        container = GameplayTagContainer()
        container.add("ability.offensive.fire")

        assert container.has_tag("ability.offensive.fire", hierarchical=False)
        assert container.has_tag("ability.offensive", hierarchical=True)
        assert container.has_tag("ability", hierarchical=True)
        assert not container.has_tag("ability", hierarchical=False)

    def test_has_any_tags(self):
        """Test has_any returns True if any tag present."""
        container = GameplayTagContainer()
        container.add("ability.fire")

        assert container.has_any(["ability.ice", "ability.fire"])
        assert not container.has_any(["ability.ice", "ability.lightning"])

    def test_has_all_tags(self):
        """Test has_all requires all tags."""
        container = GameplayTagContainer()
        container.add_many(["ability.fire", "ability.ice"])

        assert container.has_all(["ability.fire", "ability.ice"])
        assert not container.has_all(["ability.fire", "ability.lightning"])

    def test_has_any_matching_pattern(self):
        """Test has_any_matching with pattern."""
        container = GameplayTagContainer()
        container.add("ability.offensive.fire")

        assert container.has_any_matching("ability.*.fire")
        assert container.has_any_matching("ability.*")
        assert not container.has_any_matching("status.*")

    def test_has_parent_of(self):
        """Test has_parent_of checks for parent tags."""
        container = GameplayTagContainer()
        container.add("ability")
        container.add("status.buff")

        assert container.has_parent_of("ability.offensive.fire")
        assert not container.has_parent_of("effect.damage")

    def test_has_child_of(self):
        """Test has_child_of checks for child tags."""
        container = GameplayTagContainer()
        container.add("ability.offensive.fire")
        container.add("status.debuff.slow")

        assert container.has_child_of("ability")
        assert container.has_child_of("ability.offensive")
        assert not container.has_child_of("effect")

    def test_filter_matching(self):
        """Test filter_matching returns matching tags."""
        container = GameplayTagContainer()
        container.add_many([
            "ability.offensive.fire",
            "ability.offensive.ice",
            "ability.defensive.shield",
            "status.buff.speed"
        ])

        matches = container.filter_matching("ability.offensive.*")
        assert len(matches) == 2

    def test_filter_children_of(self):
        """Test filter_children_of returns child tags."""
        container = GameplayTagContainer()
        container.add_many([
            "ability.offensive.fire",
            "ability.offensive.ice",
            "status.buff"
        ])

        children = container.filter_children_of("ability.offensive")
        assert len(children) == 2

    def test_container_set_operations(self):
        """Test container set operations."""
        a = GameplayTagContainer()
        a.add_many(["tag1", "tag2", "tag3"])

        b = GameplayTagContainer()
        b.add_many(["tag2", "tag3", "tag4"])

        intersection = a.intersection(b)
        assert len(intersection) == 2

        union = a.union(b)
        assert len(union) == 4

        difference = a.difference(b)
        assert len(difference) == 1

    def test_container_protocols(self):
        """Test container supports standard protocols."""
        container = GameplayTagContainer()
        container.add_many(["a", "b", "c"])

        # __len__
        assert len(container) == 3

        # __contains__
        assert "a" in container

        # __iter__
        tags = list(container)
        assert len(tags) == 3

        # __bool__
        assert bool(container) is True
        assert bool(GameplayTagContainer()) is False

    def test_change_callback(self):
        """Test change callback is called."""
        notifications = []

        def on_change(container):
            notifications.append(len(container))

        container = GameplayTagContainer()
        container._on_change = on_change

        container.add("tag1")
        container.add("tag2")
        container.remove("tag1")

        assert len(notifications) == 3


# =============================================================================
# GAMEPLAY TAG QUERY TESTS
# =============================================================================


class TestGameplayTagQueryInternals:
    """Whitebox tests for GameplayTagQuery complex queries."""

    def test_query_all_of(self):
        """Test Query.all_of factory."""
        query = GameplayTagQuery.all_of("ability", "status")

        assert len(query.require_all) == 2

    def test_query_any_of(self):
        """Test Query.any_of factory."""
        query = GameplayTagQuery.any_of("fire", "ice", "lightning")

        assert len(query.require_any) == 3

    def test_query_none_of(self):
        """Test Query.none_of factory."""
        query = GameplayTagQuery.none_of("immune", "invulnerable")

        assert len(query.exclude) == 2

    def test_query_chained_builder(self):
        """Test query builder chaining."""
        query = (
            GameplayTagQuery
            .all_of("ability.offensive")
            .and_any("fire", "ice")
            .and_none("blocked")
        )

        assert len(query.require_all) == 1
        assert len(query.require_any) == 2
        assert len(query.exclude) == 1

    def test_query_matches_require_all(self):
        """Test query matches require_all."""
        query = GameplayTagQuery.all_of("a", "b")

        match = GameplayTagContainer()
        match.add_many(["a", "b", "c"])

        no_match = GameplayTagContainer()
        no_match.add_many(["a", "c"])

        assert query.matches(match)
        assert not query.matches(no_match)

    def test_query_matches_require_any(self):
        """Test query matches require_any."""
        query = GameplayTagQuery.any_of("fire", "ice")

        match = GameplayTagContainer()
        match.add("fire")

        no_match = GameplayTagContainer()
        no_match.add("lightning")

        assert query.matches(match)
        assert not query.matches(no_match)

    def test_query_matches_exclude(self):
        """Test query matches excludes."""
        query = GameplayTagQuery.none_of("immune")

        match = GameplayTagContainer()
        match.add("normal")

        no_match = GameplayTagContainer()
        no_match.add("immune")

        assert query.matches(match)
        assert not query.matches(no_match)

    def test_query_matches_combined(self):
        """Test query with all conditions."""
        query = (
            GameplayTagQuery
            .all_of("combat")
            .and_any("attack", "skill")
            .and_none("blocked")
        )

        # All conditions met
        good = GameplayTagContainer()
        good.add_many(["combat", "attack"])
        assert query.matches(good)

        # Missing required
        missing_required = GameplayTagContainer()
        missing_required.add("attack")
        assert not query.matches(missing_required)

        # Missing any
        missing_any = GameplayTagContainer()
        missing_any.add("combat")
        assert not query.matches(missing_any)

        # Has excluded
        has_blocked = GameplayTagContainer()
        has_blocked.add_many(["combat", "attack", "blocked"])
        assert not query.matches(has_blocked)

    def test_query_matches_with_parents(self):
        """Test query matches considering parent relationships."""
        query = GameplayTagQuery.all_of("ability.offensive")

        # Direct match
        direct = GameplayTagContainer()
        direct.add("ability.offensive")
        assert query.matches_with_parents(direct)

        # Child matches parent requirement
        child = GameplayTagContainer()
        child.add("ability.offensive.fire")
        assert query.matches_with_parents(child)

        # No match
        no_match = GameplayTagContainer()
        no_match.add("ability.defensive")
        assert not query.matches_with_parents(no_match)


# =============================================================================
# GAMEPLAY TAG REGISTRY TESTS
# =============================================================================


class TestGameplayTagRegistryInternals:
    """Whitebox tests for GameplayTagRegistry caching."""

    def test_registry_singleton(self):
        """Test registry is singleton."""
        a = GameplayTagRegistry()
        b = GameplayTagRegistry()
        assert a is b

    def test_registry_get_creates_tag(self):
        """Test get creates and caches tag."""
        tag = GameplayTagRegistry.get("new.tag")

        assert tag.hierarchy == "new.tag"
        assert "new.tag" in GameplayTagRegistry()._tags

    def test_registry_get_cached_same_instance(self):
        """Test get returns same instance for same hierarchy."""
        a = GameplayTagRegistry.get("test.tag")
        b = GameplayTagRegistry.get("test.tag")

        assert a is b

    def test_registry_get_cached_uses_lru(self):
        """Test get_cached uses LRU cache."""
        # First call
        a = GameplayTagRegistry.get_cached("cached.tag")

        # Second call should hit cache
        b = GameplayTagRegistry.get_cached("cached.tag")

        assert a == b

    def test_registry_all_tags(self):
        """Test all_tags returns all registered tags."""
        GameplayTagRegistry.get("tag1")
        GameplayTagRegistry.get("tag2")
        GameplayTagRegistry.get("tag3")

        all_tags = GameplayTagRegistry.all_tags()
        assert len(all_tags) >= 3

    def test_registry_clear(self):
        """Test clear removes all tags."""
        GameplayTagRegistry.get("tag1")
        GameplayTagRegistry.get("tag2")

        GameplayTagRegistry.clear()

        assert len(GameplayTagRegistry()._tags) == 0


# =============================================================================
# GAMEPLAY TAG DECORATOR TESTS
# =============================================================================


class TestGameplayTagDecorator:
    """Tests for @gameplay_tag decorator."""

    def test_decorator_adds_tag(self):
        """Test decorator adds _gameplay_tag attribute."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert hasattr(FireDamage, "_gameplay_tag")
        assert FireDamage._gameplay_tag is True

    def test_decorator_sets_hierarchy(self):
        """Test decorator sets _tag_hierarchy."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert FireDamage._tag_hierarchy == "Combat.Damage.Fire"

    def test_decorator_creates_tag_instance(self):
        """Test decorator creates GameplayTag instance."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert isinstance(FireDamage._tag, GameplayTag)
        assert FireDamage._tag.hierarchy == "Combat.Damage.Fire"

    def test_decorator_with_parent(self):
        """Test decorator with parent parameter."""
        @gameplay_tag("Speed", parent="Buff")
        class SpeedBuff:
            pass

        assert "Buff" in SpeedBuff._tag_hierarchy

    def test_decorator_creates_container(self):
        """Test decorator creates tag container."""
        @gameplay_tag("Ability.Offensive")
        class OffensiveAbility:
            pass

        assert hasattr(OffensiveAbility, "_tag_container")
        assert OffensiveAbility._tag_container.has("Ability.Offensive")


# =============================================================================
# ABILITY WITH TAGS DECORATOR TESTS
# =============================================================================


class TestAbilityWithTagsDecorator:
    """Tests for @ability_with_tags decorator."""

    def test_decorator_sets_required_tags(self):
        """Test decorator sets required tags container."""
        @ability_with_tags(required_tags=["Combat", "Weapon"])
        class WeaponAbility:
            pass

        assert hasattr(WeaponAbility, "_required_tags")
        assert WeaponAbility._required_tags.has("Combat")
        assert WeaponAbility._required_tags.has("Weapon")

    def test_decorator_sets_granted_tags(self):
        """Test decorator sets granted tags container."""
        @ability_with_tags(granted_tags=["Buff.Speed", "State.Running"])
        class SprintAbility:
            pass

        assert hasattr(SprintAbility, "_granted_tags")
        assert SprintAbility._granted_tags.has("Buff.Speed")

    def test_decorator_sets_blocked_tags(self):
        """Test decorator sets blocked_by_tags container."""
        @ability_with_tags(blocked_by_tags=["Status.Stunned", "Status.Silenced"])
        class BlockableAbility:
            pass

        assert hasattr(BlockableAbility, "_blocked_by_tags")
        assert BlockableAbility._blocked_by_tags.has("Status.Stunned")

    def test_can_activate_method(self):
        """Test decorator adds can_activate method."""
        @ability_with_tags(
            required_tags=["Combat"],
            blocked_by_tags=["Status.Stunned"]
        )
        class TestAbility:
            pass

        ability = TestAbility()

        # Can activate with required tag
        valid_tags = GameplayTagContainer()
        valid_tags.add("Combat")
        assert ability.can_activate(valid_tags)

        # Cannot activate without required tag
        no_combat = GameplayTagContainer()
        assert not ability.can_activate(no_combat)

        # Cannot activate with blocked tag
        stunned = GameplayTagContainer()
        stunned.add_many(["Combat", "Status.Stunned"])
        assert not ability.can_activate(stunned)

    def test_apply_granted_tags_method(self):
        """Test decorator adds apply_granted_tags method."""
        @ability_with_tags(granted_tags=["State.Casting"])
        class CastAbility:
            pass

        ability = CastAbility()
        tags = GameplayTagContainer()

        ability.apply_granted_tags(tags)
        assert tags.has("State.Casting")

    def test_remove_granted_tags_method(self):
        """Test decorator adds remove_granted_tags method."""
        @ability_with_tags(granted_tags=["State.Casting"])
        class CastAbility:
            pass

        ability = CastAbility()
        tags = GameplayTagContainer()
        tags.add("State.Casting")

        ability.remove_granted_tags(tags)
        assert not tags.has("State.Casting")


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestTagSerialization:
    """Tests for tag serialization."""

    def test_tag_to_dict(self):
        """Test GameplayTag to_dict serialization."""
        tag = GameplayTag("ability.offensive.fire")
        data = tag.to_dict()

        assert data["hierarchy"] == "ability.offensive.fire"
        assert data["parts"] == ["ability", "offensive", "fire"]
        assert data["depth"] == 3

    def test_tag_from_dict(self):
        """Test GameplayTag from_dict deserialization."""
        data = {
            "hierarchy": "ability.offensive.fire",
            "parts": ["ability", "offensive", "fire"],
            "depth": 3
        }

        tag = GameplayTag.from_dict(data)
        assert tag.hierarchy == "ability.offensive.fire"

    def test_container_to_dict(self):
        """Test container to_dict serialization."""
        container = GameplayTagContainer()
        container.add_many(["tag1", "tag2"])

        data = container.to_dict()
        assert "tags" in data
        assert len(data["tags"]) == 2

    def test_container_from_dict(self):
        """Test container from_dict deserialization."""
        data = {"tags": ["tag1", "tag2", "tag3"]}

        container = GameplayTagContainer.from_dict(data)
        assert len(container) == 3
        assert container.has("tag1")


# =============================================================================
# EDGE CASES
# =============================================================================


class TestTagEdgeCases:
    """Edge case tests for tag system."""

    def test_single_character_tag(self):
        """Test single character tag part."""
        tag = GameplayTag("a.b.c")
        assert tag.depth == 3

    def test_very_long_tag_part(self):
        """Test very long tag part name."""
        long_name = "a" * 100
        tag = GameplayTag(f"prefix.{long_name}.suffix")
        assert long_name in tag.parts

    def test_max_depth_tag(self):
        """Test tag at exactly max depth."""
        parts = [f"level{i}" for i in range(MAX_TAG_DEPTH)]
        hierarchy = TAG_SEPARATOR.join(parts)
        tag = GameplayTag(hierarchy)
        assert tag.depth == MAX_TAG_DEPTH

    def test_unicode_not_allowed(self):
        """Test unicode characters not allowed."""
        with pytest.raises(ValueError):
            GameplayTag("ability.fireé")  # e with accent

    def test_spaces_not_allowed(self):
        """Test spaces not allowed."""
        with pytest.raises(ValueError):
            GameplayTag("ability.fire ball")

    def test_wildcard_only_tag(self):
        """Test wildcard-only tag is valid."""
        tag = GameplayTag("*")
        assert tag.hierarchy == "*"

    def test_tag_str_representation(self):
        """Test tag string representation."""
        tag = GameplayTag("ability.fire")
        assert str(tag) == "ability.fire"
        assert repr(tag) == "GameplayTag('ability.fire')"

    def test_tag_equality(self):
        """Test tag equality."""
        a = GameplayTag("ability.fire")
        b = GameplayTag("ability.fire")
        c = GameplayTag("ability.ice")

        assert a == b
        assert a != c

    def test_tag_hash(self):
        """Test tag is hashable."""
        tag = GameplayTag("ability.fire")
        tag_set = {tag}
        assert tag in tag_set

    def test_container_with_same_tag_twice(self):
        """Test adding same tag twice doesn't duplicate."""
        container = GameplayTagContainer()
        container.add("tag")
        container.add("tag")
        assert len(container) == 1

    def test_query_empty_matches_all(self):
        """Test empty query matches any container."""
        query = GameplayTagQuery()

        empty = GameplayTagContainer()
        assert query.matches(empty)

        full = GameplayTagContainer()
        full.add_many(["a", "b", "c"])
        assert query.matches(full)
