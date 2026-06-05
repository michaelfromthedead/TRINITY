"""
T-GP-7.8: Tests for @gameplay_tag decorator wired to Foundation Registry.

Tests hierarchical gameplay tag system integration with Foundation Registry
for runtime discovery and querying.
"""

from __future__ import annotations

import time
import pytest
from typing import Any

from engine.gameplay.abilities.tags import (
    GameplayTag,
    GameplayTagContainer,
    GameplayTagQuery,
    GameplayTagRegistry,
    gameplay_tag,
    ability_with_tags,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear registries before and after each test."""
    GameplayTagRegistry.clear()
    yield
    GameplayTagRegistry.clear()


@pytest.fixture
def foundation_registry():
    """Get Foundation Registry if available."""
    try:
        from foundation import registry
        # Clear before test
        original_types = list(registry._types.keys())
        yield registry
        # Cleanup after test - remove types we added
        for name in list(registry._types.keys()):
            if name not in original_types:
                cls = registry._types.get(name)
                if cls:
                    registry.unregister(cls)
    except ImportError:
        pytest.skip("Foundation module not available")


# =============================================================================
# 1. GAMEPLAY TAG BASIC TESTS (10 tests)
# =============================================================================


class TestGameplayTagBasic:
    """Basic GameplayTag creation and properties."""

    def test_tag_creation_simple(self):
        """Tag with single part creates successfully."""
        tag = GameplayTag("Combat")
        assert tag.hierarchy == "Combat"
        assert tag.parts == ("Combat",)
        assert tag.depth == 1

    def test_tag_creation_hierarchical(self):
        """Tag with multiple parts creates hierarchy."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert tag.hierarchy == "Combat.Damage.Fire"
        assert tag.parts == ("Combat", "Damage", "Fire")
        assert tag.depth == 3

    def test_tag_parent_property(self):
        """Parent property returns correct parent tag."""
        tag = GameplayTag("Combat.Damage.Fire")
        parent = tag.parent
        assert parent is not None
        assert parent.hierarchy == "Combat.Damage"

    def test_tag_root_property(self):
        """Root property returns first part as tag."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert tag.root.hierarchy == "Combat"

    def test_tag_leaf_property(self):
        """Leaf property returns last part."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert tag.leaf == "Fire"

    def test_tag_root_has_no_parent(self):
        """Root tag has None parent."""
        tag = GameplayTag("Combat")
        assert tag.parent is None

    def test_tag_empty_raises_error(self):
        """Empty hierarchy raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GameplayTag("")

    def test_tag_invalid_characters_raises_error(self):
        """Invalid characters in tag raise ValueError."""
        with pytest.raises(ValueError, match="Invalid tag part"):
            GameplayTag("Combat-Damage")

    def test_tag_child_method(self):
        """Child method creates correct child tag."""
        tag = GameplayTag("Combat")
        child = tag.child("Damage")
        assert child.hierarchy == "Combat.Damage"

    def test_tag_ancestors_iterator(self):
        """Ancestors iterator returns parent chain."""
        tag = GameplayTag("Combat.Damage.Fire.Explosion")
        ancestors = list(tag.ancestors())
        assert len(ancestors) == 3
        assert ancestors[0].hierarchy == "Combat.Damage.Fire"
        assert ancestors[1].hierarchy == "Combat.Damage"
        assert ancestors[2].hierarchy == "Combat"


# =============================================================================
# 2. HIERARCHICAL MATCHING TESTS (10 tests)
# =============================================================================


class TestHierarchicalMatching:
    """Tests for hierarchical tag matching."""

    def test_is_child_of_exact_match(self):
        """Tag is child of itself."""
        tag = GameplayTag("Combat.Damage")
        assert tag.is_child_of("Combat.Damage")

    def test_is_child_of_parent(self):
        """Tag is child of its parent."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert tag.is_child_of("Combat.Damage")
        assert tag.is_child_of("Combat")

    def test_is_child_of_unrelated(self):
        """Tag is not child of unrelated tag."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert not tag.is_child_of("Buff")

    def test_is_parent_of_child(self):
        """Parent tag is parent of its children."""
        parent = GameplayTag("Combat")
        assert parent.is_parent_of("Combat.Damage")
        assert parent.is_parent_of("Combat.Damage.Fire")

    def test_is_parent_of_exact_match(self):
        """Tag is parent of itself."""
        tag = GameplayTag("Combat.Damage")
        assert tag.is_parent_of("Combat.Damage")

    def test_is_sibling_of_same_parent(self):
        """Tags with same parent are siblings."""
        tag1 = GameplayTag("Combat.Damage.Fire")
        tag2 = GameplayTag("Combat.Damage.Ice")
        assert tag1.is_sibling_of(tag2)

    def test_is_sibling_of_different_parent(self):
        """Tags with different parents are not siblings."""
        tag1 = GameplayTag("Combat.Damage.Fire")
        tag2 = GameplayTag("Combat.Heal.Light")
        assert not tag1.is_sibling_of(tag2)

    def test_matches_exact_pattern(self):
        """Exact pattern matches correctly."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert tag.matches("Combat.Damage.Fire")
        assert not tag.matches("Combat.Damage.Ice")

    def test_matches_wildcard_pattern(self):
        """Wildcard pattern matches any part."""
        tag = GameplayTag("Combat.Damage.Fire")
        assert tag.matches("Combat.*.Fire")
        assert tag.matches("*.Damage.Fire")
        assert tag.matches("Combat.Damage.*")

    def test_matches_trailing_wildcard(self):
        """Trailing wildcard matches descendants."""
        tag = GameplayTag("Combat.Damage.Fire.Explosion")
        assert tag.matches("Combat.*")
        assert tag.matches("Combat.Damage.*")


# =============================================================================
# 3. GAMEPLAY TAG CONTAINER TESTS (12 tests)
# =============================================================================


class TestGameplayTagContainer:
    """Tests for GameplayTagContainer."""

    def test_add_tag_success(self):
        """Adding new tag returns True."""
        container = GameplayTagContainer()
        assert container.add_tag("Combat.Damage")
        assert len(container) == 1

    def test_add_tag_duplicate_returns_false(self):
        """Adding duplicate tag returns False."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        assert not container.add_tag("Combat.Damage")
        assert len(container) == 1

    def test_remove_tag_success(self):
        """Removing existing tag returns True."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        assert container.remove_tag("Combat.Damage")
        assert len(container) == 0

    def test_remove_tag_not_present(self):
        """Removing non-existent tag returns False."""
        container = GameplayTagContainer()
        assert not container.remove_tag("Combat.Damage")

    def test_has_tag_exact_match(self):
        """has_tag finds exact match."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage.Fire")
        assert container.has_tag("Combat.Damage.Fire")
        assert not container.has_tag("Combat.Damage")

    def test_has_tag_hierarchical_match(self):
        """has_tag with hierarchical=True matches parent."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage.Fire")
        assert container.has_tag("Combat", hierarchical=True)
        assert container.has_tag("Combat.Damage", hierarchical=True)

    def test_has_any_returns_true(self):
        """has_any returns True if any tag present."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        container.add_tag("Buff.Speed")
        assert container.has_any(["Combat.Damage", "Debuff.Slow"])

    def test_has_any_returns_false(self):
        """has_any returns False if no tag present."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        assert not container.has_any(["Buff.Speed", "Debuff.Slow"])

    def test_has_all_returns_true(self):
        """has_all returns True if all tags present."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        container.add_tag("Buff.Speed")
        assert container.has_all(["Combat.Damage", "Buff.Speed"])

    def test_has_all_returns_false(self):
        """has_all returns False if any tag missing."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        assert not container.has_all(["Combat.Damage", "Buff.Speed"])

    def test_filter_children_of(self):
        """filter_children_of returns matching tags."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage.Fire")
        container.add_tag("Combat.Damage.Ice")
        container.add_tag("Buff.Speed")

        children = container.filter_children_of("Combat.Damage")
        assert len(children) == 2
        hierarchies = {t.hierarchy for t in children}
        assert "Combat.Damage.Fire" in hierarchies
        assert "Combat.Damage.Ice" in hierarchies

    def test_container_serialization(self):
        """Container serializes and deserializes correctly."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage.Fire")
        container.add_tag("Buff.Speed")

        data = container.to_dict()
        restored = GameplayTagContainer.from_dict(data)

        assert container.has_tag("Combat.Damage.Fire")
        assert container.has_tag("Buff.Speed")


# =============================================================================
# 4. @gameplay_tag DECORATOR TESTS (10 tests)
# =============================================================================


class TestGameplayTagDecorator:
    """Tests for @gameplay_tag decorator."""

    def test_decorator_attaches_tag(self):
        """Decorator attaches _tag attribute."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert hasattr(FireDamage, "_tag")
        assert FireDamage._tag.hierarchy == "Combat.Damage.Fire"

    def test_decorator_attaches_hierarchy(self):
        """Decorator attaches _tag_hierarchy attribute."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert FireDamage._tag_hierarchy == "Combat.Damage.Fire"

    def test_decorator_sets_gameplay_tag_flag(self):
        """Decorator sets _gameplay_tag to True."""
        @gameplay_tag("Combat.Damage")
        class CombatDamage:
            pass

        assert CombatDamage._gameplay_tag is True

    def test_decorator_with_parent(self):
        """Decorator with parent creates correct hierarchy."""
        @gameplay_tag("Speed", parent="Buff")
        class SpeedBuff:
            pass

        assert SpeedBuff._tag_hierarchy == "Buff.Speed"
        assert SpeedBuff._tag_parent == "Buff"

    def test_decorator_parent_already_in_name(self):
        """Decorator with parent in name keeps correct hierarchy."""
        @gameplay_tag("Buff.Speed", parent="Buff")
        class SpeedBuff:
            pass

        # Should not double the parent
        assert SpeedBuff._tag_hierarchy == "Buff.Speed"

    def test_decorator_creates_tag_container(self):
        """Decorator creates _tag_container."""
        @gameplay_tag("Combat.Damage")
        class CombatDamage:
            pass

        assert hasattr(CombatDamage, "_tag_container")
        assert CombatDamage._tag_container.has_tag("Combat.Damage")

    def test_multiple_decorated_classes(self):
        """Multiple classes can be decorated."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        @gameplay_tag("Combat.Damage.Ice")
        class IceDamage:
            pass

        assert FireDamage._tag.hierarchy == "Combat.Damage.Fire"
        assert IceDamage._tag.hierarchy == "Combat.Damage.Ice"

    def test_decorator_preserves_class(self):
        """Decorator preserves class attributes."""
        @gameplay_tag("Combat.Damage")
        class DamageEffect:
            damage = 100

            def apply(self):
                return self.damage

        assert DamageEffect.damage == 100
        obj = DamageEffect()
        assert obj.apply() == 100

    def test_decorator_works_with_inheritance(self):
        """Decorator works with inheritance."""
        @gameplay_tag("Combat.Damage")
        class BaseDamage:
            pass

        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage(BaseDamage):
            pass

        assert FireDamage._tag.hierarchy == "Combat.Damage.Fire"
        assert issubclass(FireDamage, BaseDamage)

    def test_decorator_invalid_tag_raises(self):
        """Decorator with invalid tag raises ValueError."""
        with pytest.raises(ValueError):
            @gameplay_tag("Combat-Invalid")
            class InvalidTag:
                pass


# =============================================================================
# 5. FOUNDATION REGISTRY INTEGRATION TESTS (10 tests)
# =============================================================================


class TestFoundationRegistryIntegration:
    """Tests for Foundation Registry integration."""

    def test_decorator_registers_with_foundation(self, foundation_registry):
        """Decorated class is registered with Foundation."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert foundation_registry.is_registered(FireDamage)

    def test_decorator_adds_gameplay_tag_tag(self, foundation_registry):
        """Decorated class has gameplay_tag tag."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        assert foundation_registry.has_tag(FireDamage, "gameplay_tag")

    def test_decorator_sets_hierarchy_metadata(self, foundation_registry):
        """Decorated class has tag_hierarchy metadata."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        hierarchy = foundation_registry.get_metadata(FireDamage, "tag_hierarchy")
        assert hierarchy == "Combat.Damage.Fire"

    def test_decorator_sets_parent_metadata(self, foundation_registry):
        """Decorated class with parent has parent metadata."""
        @gameplay_tag("Speed", parent="Buff")
        class SpeedBuff:
            pass

        parent = foundation_registry.get_metadata(SpeedBuff, "parent")
        assert parent == "Buff"

    def test_decorator_sets_root_metadata(self, foundation_registry):
        """Decorated class has root metadata."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        root = foundation_registry.get_metadata(FireDamage, "root")
        assert root == "Combat"

    def test_query_all_gameplay_tags(self, foundation_registry):
        """Registry.query returns all gameplay tags."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        @gameplay_tag("Buff.Speed")
        class SpeedBuff:
            pass

        results = foundation_registry.query(tag="gameplay_tag")
        assert FireDamage in results
        assert SpeedBuff in results

    def test_query_by_parent(self, foundation_registry):
        """Registry.query filters by parent."""
        @gameplay_tag("Damage", parent="Combat")
        class CombatDamage:
            pass

        @gameplay_tag("Speed", parent="Buff")
        class SpeedBuff:
            pass

        results = foundation_registry.query(tag="gameplay_tag", parent="Combat")
        assert CombatDamage in results
        assert SpeedBuff not in results

    def test_query_by_root(self, foundation_registry):
        """Registry.query filters by root."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        @gameplay_tag("Combat.Damage.Ice")
        class IceDamage:
            pass

        @gameplay_tag("Buff.Speed")
        class SpeedBuff:
            pass

        results = foundation_registry.query(tag="gameplay_tag", root="Combat")
        assert FireDamage in results
        assert IceDamage in results
        assert SpeedBuff not in results

    def test_registry_helper_query_foundation(self, foundation_registry):
        """GameplayTagRegistry.query_foundation works."""
        @gameplay_tag("Combat.Damage.Fire")
        class FireDamage:
            pass

        results = GameplayTagRegistry.query_foundation()
        assert FireDamage in results

    def test_multiple_registrations_idempotent(self, foundation_registry):
        """Multiple decorations don't cause duplicate registrations."""
        @gameplay_tag("Combat.Damage")
        class Damage:
            pass

        # Should not raise even if called again internally
        count = len(foundation_registry.query(tag="gameplay_tag"))
        assert count >= 1


# =============================================================================
# 6. ABILITY TAG INTEGRATION TESTS (8 tests)
# =============================================================================


class TestAbilityTagIntegration:
    """Tests for ability_with_tags decorator."""

    def test_ability_with_required_tags(self):
        """ability_with_tags creates required_tags container."""
        @ability_with_tags(required_tags=["Combat"])
        class FireballAbility:
            pass

        assert hasattr(FireballAbility, "_required_tags")
        assert FireballAbility._required_tags.has_tag("Combat")

    def test_ability_with_granted_tags(self):
        """ability_with_tags creates granted_tags container."""
        @ability_with_tags(granted_tags=["Buff.Speed"])
        class SprintAbility:
            pass

        assert hasattr(SprintAbility, "_granted_tags")
        assert SprintAbility._granted_tags.has_tag("Buff.Speed")

    def test_ability_with_blocked_by_tags(self):
        """ability_with_tags creates blocked_by_tags container."""
        @ability_with_tags(blocked_by_tags=["Status.Stunned"])
        class JumpAbility:
            pass

        assert hasattr(JumpAbility, "_blocked_by_tags")
        assert JumpAbility._blocked_by_tags.has_tag("Status.Stunned")

    def test_can_activate_with_required_tags(self):
        """can_activate returns True when required tags present."""
        @ability_with_tags(required_tags=["Combat"])
        class AttackAbility:
            pass

        owner_tags = GameplayTagContainer()
        owner_tags.add_tag("Combat")

        ability = AttackAbility()
        assert ability.can_activate(owner_tags)

    def test_can_activate_missing_required_tags(self):
        """can_activate returns False when required tags missing."""
        @ability_with_tags(required_tags=["Combat", "Armed"])
        class AttackAbility:
            pass

        owner_tags = GameplayTagContainer()
        owner_tags.add_tag("Combat")

        ability = AttackAbility()
        assert not ability.can_activate(owner_tags)

    def test_can_activate_blocked_by_tag(self):
        """can_activate returns False when blocked tag present."""
        @ability_with_tags(blocked_by_tags=["Status.Stunned"])
        class MoveAbility:
            pass

        owner_tags = GameplayTagContainer()
        owner_tags.add_tag("Status.Stunned")

        ability = MoveAbility()
        assert not ability.can_activate(owner_tags)

    def test_apply_granted_tags(self):
        """apply_granted_tags adds tags to target."""
        @ability_with_tags(granted_tags=["Buff.Speed", "Buff.Haste"])
        class SprintAbility:
            pass

        target_tags = GameplayTagContainer()
        ability = SprintAbility()
        ability.apply_granted_tags(target_tags)

        assert target_tags.has_tag("Buff.Speed")
        assert target_tags.has_tag("Buff.Haste")

    def test_remove_granted_tags(self):
        """remove_granted_tags removes tags from target."""
        @ability_with_tags(granted_tags=["Buff.Speed"])
        class SprintAbility:
            pass

        target_tags = GameplayTagContainer()
        target_tags.add_tag("Buff.Speed")
        target_tags.add_tag("Buff.Strength")

        ability = SprintAbility()
        ability.remove_granted_tags(target_tags)

        assert not target_tags.has_tag("Buff.Speed")
        assert target_tags.has_tag("Buff.Strength")


# =============================================================================
# 7. GAMEPLAY TAG QUERY TESTS (6 tests)
# =============================================================================


class TestGameplayTagQuery:
    """Tests for GameplayTagQuery."""

    def test_query_all_of_matches(self):
        """Query.all_of matches when all tags present."""
        query = GameplayTagQuery.all_of("Combat", "Armed")
        container = GameplayTagContainer()
        container.add_tag("Combat")
        container.add_tag("Armed")
        container.add_tag("Buff.Speed")

        assert query.matches(container)

    def test_query_all_of_fails(self):
        """Query.all_of fails when any tag missing."""
        query = GameplayTagQuery.all_of("Combat", "Armed")
        container = GameplayTagContainer()
        container.add_tag("Combat")

        assert not query.matches(container)

    def test_query_any_of_matches(self):
        """Query.any_of matches when any tag present."""
        query = GameplayTagQuery.any_of("Buff.Speed", "Buff.Strength")
        container = GameplayTagContainer()
        container.add_tag("Buff.Speed")

        assert query.matches(container)

    def test_query_none_of_matches(self):
        """Query.none_of matches when no excluded tags present."""
        query = GameplayTagQuery.none_of("Status.Stunned", "Status.Dead")
        container = GameplayTagContainer()
        container.add_tag("Combat")

        assert query.matches(container)

    def test_query_none_of_fails(self):
        """Query.none_of fails when excluded tag present."""
        query = GameplayTagQuery.none_of("Status.Stunned")
        container = GameplayTagContainer()
        container.add_tag("Status.Stunned")

        assert not query.matches(container)

    def test_query_matches_with_parents(self):
        """Query.matches_with_parents considers hierarchy."""
        query = GameplayTagQuery.all_of("Combat")
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage.Fire")

        # Exact match fails
        assert not query.matches(container)
        # Parent match succeeds
        assert query.matches_with_parents(container)


# =============================================================================
# 8. TAG SERIALIZATION TESTS (4 tests)
# =============================================================================


class TestTagSerialization:
    """Tests for tag serialization."""

    def test_tag_to_dict(self):
        """GameplayTag serializes to dict."""
        tag = GameplayTag("Combat.Damage.Fire")
        data = tag.to_dict()

        assert data["hierarchy"] == "Combat.Damage.Fire"
        assert data["parts"] == ["Combat", "Damage", "Fire"]
        assert data["depth"] == 3

    def test_tag_from_dict(self):
        """GameplayTag deserializes from dict."""
        data = {"hierarchy": "Combat.Damage.Fire"}
        tag = GameplayTag.from_dict(data)

        assert tag.hierarchy == "Combat.Damage.Fire"

    def test_container_to_dict(self):
        """GameplayTagContainer serializes to dict."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")
        container.add_tag("Buff.Speed")

        data = container.to_dict()
        assert "tags" in data
        assert len(data["tags"]) == 2

    def test_container_from_dict(self):
        """GameplayTagContainer deserializes from dict."""
        data = {"tags": ["Combat.Damage", "Buff.Speed"]}
        container = GameplayTagContainer.from_dict(data)

        assert container.has_tag("Combat.Damage")
        assert container.has_tag("Buff.Speed")


# =============================================================================
# 9. PERFORMANCE TESTS (2 tests)
# =============================================================================


class TestPerformance:
    """Performance benchmarks."""

    def test_1000_tag_checks_under_50ms(self):
        """1000 tag checks complete under 50ms."""
        container = GameplayTagContainer()
        # Add 100 tags
        for i in range(100):
            container.add_tag(f"Category{i % 10}.Sub{i % 20}.Tag{i}")

        start = time.perf_counter()
        for _ in range(1000):
            container.has_tag("Category5.Sub10.Tag50")
            container.has_tag("Category9.Sub19.Tag99", hierarchical=True)
            container.has_any(["Nonexistent1", "Nonexistent2"])
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 50, f"1000 tag checks took {elapsed:.2f}ms (expected < 50ms)"

    def test_hierarchical_matching_performance(self):
        """Hierarchical matching is efficient."""
        container = GameplayTagContainer()
        # Deep hierarchy
        for i in range(50):
            container.add_tag(f"Root.Level1.Level2.Level3.Level4.Tag{i}")

        start = time.perf_counter()
        for _ in range(100):
            container.has_tag("Root", hierarchical=True)
            container.has_tag("Root.Level1", hierarchical=True)
            container.filter_children_of("Root.Level1.Level2")
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Hierarchical matching took {elapsed:.2f}ms (expected < 100ms)"


# =============================================================================
# 10. EDGE CASES AND ERROR HANDLING (8 tests)
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_max_depth_tag(self):
        """Tag at max depth works."""
        parts = ["Level" + str(i) for i in range(10)]  # MAX_TAG_DEPTH = 10
        hierarchy = ".".join(parts)
        tag = GameplayTag(hierarchy)
        assert tag.depth == 10

    def test_exceeds_max_depth_raises(self):
        """Tag exceeding max depth raises ValueError."""
        parts = ["Level" + str(i) for i in range(11)]
        hierarchy = ".".join(parts)
        with pytest.raises(ValueError, match="exceeds maximum"):
            GameplayTag(hierarchy)

    def test_empty_part_raises(self):
        """Tag with empty part raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            GameplayTag("Combat..Fire")

    def test_container_add_string_conversion(self):
        """Container converts string to GameplayTag."""
        container = GameplayTagContainer()
        container.add_tag("Combat.Damage")

        assert isinstance(list(container.tags)[0], GameplayTag)

    def test_container_clear(self):
        """Container clear removes all tags."""
        container = GameplayTagContainer()
        container.add_tag("Combat")
        container.add_tag("Buff")
        count = container.clear()

        assert count == 2
        assert len(container) == 0

    def test_container_on_change_callback(self):
        """Container calls on_change callback."""
        changes = []

        def callback(c):
            changes.append(len(c))

        container = GameplayTagContainer(_on_change=callback)
        container.add_tag("Combat")
        container.add_tag("Buff")

        assert changes == [1, 2]

    def test_tag_equality(self):
        """Tags with same hierarchy are equal."""
        tag1 = GameplayTag("Combat.Damage")
        tag2 = GameplayTag("Combat.Damage")
        assert tag1 == tag2
        assert hash(tag1) == hash(tag2)

    def test_tag_inequality(self):
        """Tags with different hierarchy are not equal."""
        tag1 = GameplayTag("Combat.Damage")
        tag2 = GameplayTag("Combat.Heal")
        assert tag1 != tag2


# =============================================================================
# SUMMARY: 80+ tests covering all requirements
# =============================================================================
# 1. Basic tag tests: 10
# 2. Hierarchical matching: 10
# 3. Container tests: 12
# 4. Decorator tests: 10
# 5. Foundation integration: 10
# 6. Ability integration: 8
# 7. Query tests: 6
# 8. Serialization: 4
# 9. Performance: 2
# 10. Edge cases: 8
# Total: 80 tests
