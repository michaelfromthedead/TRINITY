"""
Test Suite: T-GP-3.8 - Blackboard Registry Integration

Tests for the @blackboard decorator and its integration with the
Foundation Registry for runtime discovery and factory instantiation.

Requirements tested:
1. @blackboard registers class with Foundation Registry
2. Registry.query(tag="blackboard") returns all blackboard types
3. Scope filtering works (entity, shared, team, etc.)
4. Metadata stored correctly (name, scope, key_types, description)
5. Factory instantiation via Blackboard.from_registry()
6. Multiple blackboard types coexist
7. Key type validation and storage
8. Performance: 100 queries under 50ms
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Type

import pytest

from foundation import registry, Registry
from engine.gameplay.ai.blackboard import (
    Blackboard,
    BlackboardEntry,
    BlackboardKey,
    BlackboardScope,
    TypedBlackboard,
    TypedBlackboardKey,
    blackboard,
    get_all_blackboards,
    get_blackboards_by_scope,
    get_blackboard_metadata,
    create_blackboard_from_registry,
    clear_blackboard_registry,
    TAG_BLACKBOARD,
    VALID_SCOPES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test to avoid cross-contamination."""
    # Store initial state
    initial_types = set(registry.all_types())

    # Clear the internal blackboard registry
    clear_blackboard_registry()

    yield

    # Clean up any types added during the test
    for cls in registry.all_types():
        if cls not in initial_types:
            try:
                registry.unregister(cls)
            except Exception:
                pass

    # Clear internal registry again
    clear_blackboard_registry()


# =============================================================================
# Test Classes - Defined inside tests to avoid polluting global registry
# =============================================================================


class TestBlackboardDecorator:
    """Tests for @blackboard decorator basic functionality."""

    def test_blackboard_registers_class(self, clean_registry):
        """Test that @blackboard registers the class with Foundation Registry."""
        @blackboard(name="test_bb_01", scope="entity")
        class TestBB01(Blackboard):
            pass

        # Verify registration
        registered = registry.get("bb.test_bb_01")
        assert registered is TestBB01
        assert registry.is_registered(TestBB01)

    def test_blackboard_adds_tag(self, clean_registry):
        """Test that @blackboard adds the 'blackboard' tag."""
        @blackboard(name="test_bb_02", scope="entity")
        class TestBB02(Blackboard):
            pass

        assert registry.has_tag(TestBB02, TAG_BLACKBOARD)

    def test_blackboard_stores_name_metadata(self, clean_registry):
        """Test that @blackboard stores bb_name metadata."""
        @blackboard(name="test_bb_03", scope="entity")
        class TestBB03(Blackboard):
            pass

        assert registry.get_metadata(TestBB03, "bb_name") == "test_bb_03"

    def test_blackboard_stores_scope_metadata(self, clean_registry):
        """Test that @blackboard stores scope metadata."""
        @blackboard(name="test_bb_04", scope="shared")
        class TestBB04(Blackboard):
            pass

        assert registry.get_metadata(TestBB04, "scope") == "shared"

    def test_blackboard_stores_description_metadata(self, clean_registry):
        """Test that @blackboard stores description metadata."""
        @blackboard(name="test_bb_05", scope="entity", description="Test description")
        class TestBB05(Blackboard):
            pass

        assert registry.get_metadata(TestBB05, "description") == "Test description"

    def test_blackboard_stores_key_types_metadata(self, clean_registry):
        """Test that @blackboard stores key_types metadata."""
        @blackboard(name="test_bb_06", scope="entity", key_types=["health", "target"])
        class TestBB06(Blackboard):
            pass

        key_types = registry.get_metadata(TestBB06, "key_types")
        assert "health" in key_types
        assert "target" in key_types

    def test_blackboard_sets_class_attributes(self, clean_registry):
        """Test that @blackboard sets _blackboard_* attributes on class."""
        @blackboard(name="test_bb_07", scope="team", description="Team BB")
        class TestBB07(Blackboard):
            pass

        assert hasattr(TestBB07, "_blackboard_registered")
        assert TestBB07._blackboard_registered is True
        assert TestBB07._blackboard_name == "test_bb_07"
        assert TestBB07._blackboard_scope == "team"
        assert TestBB07._blackboard_description == "Team BB"

    def test_blackboard_default_description(self, clean_registry):
        """Test that @blackboard handles missing description."""
        @blackboard(name="test_bb_08", scope="entity")
        class TestBB08(Blackboard):
            pass

        assert TestBB08._blackboard_description == ""

    def test_blackboard_empty_key_types(self, clean_registry):
        """Test that @blackboard handles missing key_types."""
        @blackboard(name="test_bb_09", scope="entity")
        class TestBB09(Blackboard):
            pass

        assert TestBB09._blackboard_key_types == frozenset()


class TestBlackboardScopeValidation:
    """Tests for blackboard scope validation."""

    def test_valid_scope_entity(self, clean_registry):
        """Test that entity scope is valid."""
        @blackboard(name="scope_entity", scope="entity")
        class ScopeEntity(Blackboard):
            pass

        assert ScopeEntity._blackboard_scope == "entity"

    def test_valid_scope_shared(self, clean_registry):
        """Test that shared scope is valid."""
        @blackboard(name="scope_shared", scope="shared")
        class ScopeShared(Blackboard):
            pass

        assert ScopeShared._blackboard_scope == "shared"

    def test_valid_scope_team(self, clean_registry):
        """Test that team scope is valid."""
        @blackboard(name="scope_team", scope="team")
        class ScopeTeam(Blackboard):
            pass

        assert ScopeTeam._blackboard_scope == "team"

    def test_valid_scope_group(self, clean_registry):
        """Test that group scope is valid."""
        @blackboard(name="scope_group", scope="group")
        class ScopeGroup(Blackboard):
            pass

        assert ScopeGroup._blackboard_scope == "group"

    def test_valid_scope_zone(self, clean_registry):
        """Test that zone scope is valid."""
        @blackboard(name="scope_zone", scope="zone")
        class ScopeZone(Blackboard):
            pass

        assert ScopeZone._blackboard_scope == "zone"

    def test_valid_scope_session(self, clean_registry):
        """Test that session scope is valid."""
        @blackboard(name="scope_session", scope="session")
        class ScopeSession(Blackboard):
            pass

        assert ScopeSession._blackboard_scope == "session"

    def test_invalid_scope_raises_error(self, clean_registry):
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            @blackboard(name="invalid_scope", scope="invalid")
            class InvalidScope(Blackboard):
                pass

        assert "Invalid blackboard scope" in str(exc_info.value)
        assert "invalid" in str(exc_info.value)

    def test_all_valid_scopes_documented(self, clean_registry):
        """Test that VALID_SCOPES contains expected scopes."""
        expected = {"entity", "shared", "team", "group", "zone", "session"}
        assert VALID_SCOPES == expected


class TestBlackboardQuery:
    """Tests for blackboard query functionality."""

    def test_query_returns_all_blackboards(self, clean_registry):
        """Test that Registry.query(tag='blackboard') returns all blackboards."""
        @blackboard(name="query_bb_01", scope="entity")
        class QueryBB01(Blackboard):
            pass

        @blackboard(name="query_bb_02", scope="shared")
        class QueryBB02(Blackboard):
            pass

        result = registry.query(tag=TAG_BLACKBOARD)
        assert QueryBB01 in result
        assert QueryBB02 in result

    def test_query_by_scope_entity(self, clean_registry):
        """Test querying blackboards by entity scope."""
        @blackboard(name="query_entity_01", scope="entity")
        class QueryEntity01(Blackboard):
            pass

        @blackboard(name="query_shared_01", scope="shared")
        class QueryShared01(Blackboard):
            pass

        result = registry.query(tag=TAG_BLACKBOARD, scope="entity")
        assert QueryEntity01 in result
        assert QueryShared01 not in result

    def test_query_by_scope_shared(self, clean_registry):
        """Test querying blackboards by shared scope."""
        @blackboard(name="query_entity_02", scope="entity")
        class QueryEntity02(Blackboard):
            pass

        @blackboard(name="query_shared_02", scope="shared")
        class QueryShared02(Blackboard):
            pass

        result = registry.query(tag=TAG_BLACKBOARD, scope="shared")
        assert QueryShared02 in result
        assert QueryEntity02 not in result

    def test_query_by_scope_team(self, clean_registry):
        """Test querying blackboards by team scope."""
        @blackboard(name="query_team_01", scope="team")
        class QueryTeam01(Blackboard):
            pass

        result = registry.query(tag=TAG_BLACKBOARD, scope="team")
        assert QueryTeam01 in result

    def test_get_all_blackboards_helper(self, clean_registry):
        """Test get_all_blackboards() helper function."""
        @blackboard(name="helper_bb_01", scope="entity")
        class HelperBB01(Blackboard):
            pass

        @blackboard(name="helper_bb_02", scope="shared")
        class HelperBB02(Blackboard):
            pass

        result = get_all_blackboards()
        assert HelperBB01 in result
        assert HelperBB02 in result

    def test_get_blackboards_by_scope_helper(self, clean_registry):
        """Test get_blackboards_by_scope() helper function."""
        @blackboard(name="scope_helper_01", scope="entity")
        class ScopeHelper01(Blackboard):
            pass

        @blackboard(name="scope_helper_02", scope="shared")
        class ScopeHelper02(Blackboard):
            pass

        entity_result = get_blackboards_by_scope("entity")
        assert ScopeHelper01 in entity_result
        assert ScopeHelper02 not in entity_result

        shared_result = get_blackboards_by_scope("shared")
        assert ScopeHelper02 in shared_result
        assert ScopeHelper01 not in shared_result


class TestBlackboardMetadata:
    """Tests for blackboard metadata retrieval."""

    def test_get_blackboard_metadata(self, clean_registry):
        """Test get_blackboard_metadata() returns correct metadata."""
        @blackboard(
            name="meta_bb_01",
            scope="entity",
            description="Test metadata",
            key_types=["health", "mana"],
        )
        class MetaBB01(Blackboard):
            pass

        meta = get_blackboard_metadata("meta_bb_01")
        assert meta is not None
        assert meta["bb_name"] == "meta_bb_01"
        assert meta["scope"] == "entity"
        assert meta["description"] == "Test metadata"
        assert "health" in meta["key_types"]
        assert "mana" in meta["key_types"]

    def test_get_blackboard_metadata_not_found(self, clean_registry):
        """Test get_blackboard_metadata() returns None for unknown name."""
        meta = get_blackboard_metadata("nonexistent_bb")
        assert meta is None

    def test_metadata_key_types_frozen(self, clean_registry):
        """Test that key_types are stored as frozenset."""
        @blackboard(name="frozen_bb", scope="entity", key_types=["a", "b", "c"])
        class FrozenBB(Blackboard):
            pass

        key_types = registry.get_metadata(FrozenBB, "key_types")
        assert isinstance(key_types, frozenset)

    def test_metadata_all_fields_present(self, clean_registry):
        """Test that all expected metadata fields are present."""
        @blackboard(
            name="all_fields_bb",
            scope="shared",
            description="All fields test",
            key_types=["key1"],
        )
        class AllFieldsBB(Blackboard):
            pass

        meta = registry.get_all_metadata(AllFieldsBB)
        assert "bb_name" in meta
        assert "scope" in meta
        assert "description" in meta
        assert "key_types" in meta
        assert "_tags" in meta


class TestBlackboardFactory:
    """Tests for blackboard factory instantiation."""

    def test_from_registry_creates_instance(self, clean_registry):
        """Test that Blackboard.from_registry() creates an instance."""
        @blackboard(name="factory_bb_01", scope="entity")
        class FactoryBB01(Blackboard):
            pass

        instance = Blackboard.from_registry("factory_bb_01")
        assert isinstance(instance, FactoryBB01)
        assert isinstance(instance, Blackboard)

    def test_from_registry_passes_args(self, clean_registry):
        """Test that from_registry() passes arguments to constructor."""
        @blackboard(name="factory_bb_02", scope="entity")
        class FactoryBB02(Blackboard):
            def __init__(self, bb_name: str = "default", entity_id: int = 0, **kwargs):
                super().__init__(name=bb_name, entity_id=entity_id, **kwargs)

        instance = Blackboard.from_registry("factory_bb_02", bb_name="custom", entity_id=42)
        assert instance.name == "custom"
        assert instance.entity_id == 42

    def test_from_registry_not_found_raises(self, clean_registry):
        """Test that from_registry() raises ValueError for unknown name."""
        with pytest.raises(ValueError) as exc_info:
            Blackboard.from_registry("nonexistent")

        assert "not found in registry" in str(exc_info.value)

    def test_create_blackboard_from_registry_alias(self, clean_registry):
        """Test that create_blackboard_from_registry() works as alias."""
        @blackboard(name="alias_bb", scope="entity")
        class AliasBB(Blackboard):
            pass

        instance = create_blackboard_from_registry("alias_bb")
        assert isinstance(instance, AliasBB)

    def test_factory_multiple_instances(self, clean_registry):
        """Test that factory creates distinct instances."""
        @blackboard(name="multi_inst_bb", scope="entity")
        class MultiInstBB(Blackboard):
            pass

        inst1 = Blackboard.from_registry("multi_inst_bb")
        inst2 = Blackboard.from_registry("multi_inst_bb")

        assert inst1 is not inst2
        assert isinstance(inst1, MultiInstBB)
        assert isinstance(inst2, MultiInstBB)


class TestMultipleBlackboards:
    """Tests for multiple blackboard types coexisting."""

    def test_multiple_blackboards_register(self, clean_registry):
        """Test that multiple blackboard types can be registered."""
        @blackboard(name="multi_01", scope="entity")
        class Multi01(Blackboard):
            pass

        @blackboard(name="multi_02", scope="shared")
        class Multi02(Blackboard):
            pass

        @blackboard(name="multi_03", scope="team")
        class Multi03(Blackboard):
            pass

        all_bbs = get_all_blackboards()
        assert len(all_bbs) >= 3
        assert Multi01 in all_bbs
        assert Multi02 in all_bbs
        assert Multi03 in all_bbs

    def test_multiple_blackboards_distinct_queries(self, clean_registry):
        """Test that multiple blackboards have distinct query results by scope."""
        @blackboard(name="distinct_entity", scope="entity")
        class DistinctEntity(Blackboard):
            pass

        @blackboard(name="distinct_shared", scope="shared")
        class DistinctShared(Blackboard):
            pass

        @blackboard(name="distinct_team", scope="team")
        class DistinctTeam(Blackboard):
            pass

        entity_bbs = get_blackboards_by_scope("entity")
        shared_bbs = get_blackboards_by_scope("shared")
        team_bbs = get_blackboards_by_scope("team")

        assert DistinctEntity in entity_bbs
        assert DistinctEntity not in shared_bbs
        assert DistinctEntity not in team_bbs

        assert DistinctShared in shared_bbs
        assert DistinctShared not in entity_bbs
        assert DistinctShared not in team_bbs

        assert DistinctTeam in team_bbs
        assert DistinctTeam not in entity_bbs
        assert DistinctTeam not in shared_bbs

    def test_multiple_blackboards_same_scope(self, clean_registry):
        """Test that multiple blackboards with same scope coexist."""
        @blackboard(name="same_scope_01", scope="entity")
        class SameScope01(Blackboard):
            pass

        @blackboard(name="same_scope_02", scope="entity")
        class SameScope02(Blackboard):
            pass

        @blackboard(name="same_scope_03", scope="entity")
        class SameScope03(Blackboard):
            pass

        entity_bbs = get_blackboards_by_scope("entity")
        assert SameScope01 in entity_bbs
        assert SameScope02 in entity_bbs
        assert SameScope03 in entity_bbs

    def test_multiple_blackboards_independent_instances(self, clean_registry):
        """Test that instances from different blackboard types are independent."""
        @blackboard(name="indep_01", scope="entity")
        class Indep01(Blackboard):
            pass

        @blackboard(name="indep_02", scope="entity")
        class Indep02(Blackboard):
            pass

        inst1 = Blackboard.from_registry("indep_01")
        inst2 = Blackboard.from_registry("indep_02")

        inst1.set("key", "value1")
        inst2.set("key", "value2")

        assert inst1.get("key") == "value1"
        assert inst2.get("key") == "value2"


class TestKeyTypeValidation:
    """Tests for key type handling."""

    def test_key_types_single(self, clean_registry):
        """Test blackboard with single key type."""
        @blackboard(name="single_key", scope="entity", key_types=["health"])
        class SingleKey(Blackboard):
            pass

        key_types = SingleKey._blackboard_key_types
        assert len(key_types) == 1
        assert "health" in key_types

    def test_key_types_multiple(self, clean_registry):
        """Test blackboard with multiple key types."""
        @blackboard(
            name="multi_key",
            scope="entity",
            key_types=["health", "mana", "stamina", "target"],
        )
        class MultiKey(Blackboard):
            pass

        key_types = MultiKey._blackboard_key_types
        assert len(key_types) == 4
        assert all(k in key_types for k in ["health", "mana", "stamina", "target"])

    def test_key_types_empty(self, clean_registry):
        """Test blackboard with no key types specified."""
        @blackboard(name="no_key", scope="entity")
        class NoKey(Blackboard):
            pass

        key_types = NoKey._blackboard_key_types
        assert key_types == frozenset()

    def test_key_types_deduplication(self, clean_registry):
        """Test that duplicate key types are deduplicated."""
        @blackboard(
            name="dedup_key",
            scope="entity",
            key_types=["health", "health", "mana", "mana"],
        )
        class DedupKey(Blackboard):
            pass

        key_types = DedupKey._blackboard_key_types
        assert len(key_types) == 2
        assert "health" in key_types
        assert "mana" in key_types

    def test_key_types_in_registry_metadata(self, clean_registry):
        """Test that key types are stored in registry metadata."""
        @blackboard(
            name="reg_key",
            scope="entity",
            key_types=["position", "rotation"],
        )
        class RegKey(Blackboard):
            pass

        meta = get_blackboard_metadata("reg_key")
        assert "position" in meta["key_types"]
        assert "rotation" in meta["key_types"]


class TestInstanceTracking:
    """Tests for instance tracking functionality."""

    def test_track_instances_enabled(self, clean_registry):
        """Test that instance tracking can be enabled."""
        @blackboard(name="tracked_bb", scope="entity", track_instances=True)
        class TrackedBB(Blackboard):
            pass

        # Create instances
        inst1 = TrackedBB(name="inst1")
        inst2 = TrackedBB(name="inst2")

        # Check instance count
        count = registry.instance_count(TrackedBB)
        assert count >= 2

    def test_track_instances_disabled_by_default(self, clean_registry):
        """Test that instance tracking is disabled by default."""
        @blackboard(name="untracked_bb", scope="entity")
        class UntrackedBB(Blackboard):
            pass

        # Create instances
        inst1 = UntrackedBB(name="inst1")

        # Instance count should be 0 (not tracked)
        count = registry.instance_count(UntrackedBB)
        assert count == 0


class TestClearRegistry:
    """Tests for clearing the blackboard registry."""

    def test_clear_blackboard_registry(self, clean_registry):
        """Test that clear_blackboard_registry() clears internal registry."""
        @blackboard(name="clear_test", scope="entity")
        class ClearTest(Blackboard):
            pass

        # Verify registration
        assert Blackboard.from_registry("clear_test") is not None

        # Clear
        clear_blackboard_registry()

        # Factory should still work via Foundation Registry
        # (clear only clears the internal fast-lookup registry)
        # But we need to test that the internal registry was cleared
        from engine.gameplay.ai.blackboard import _blackboard_registry
        assert "clear_test" not in _blackboard_registry


class TestReloadScenarios:
    """Tests for reload/re-registration scenarios."""

    def test_reregister_same_class_ok(self, clean_registry):
        """Test that re-registering same class doesn't raise error."""
        # First registration
        @blackboard(name="reload_bb", scope="entity")
        class ReloadBB(Blackboard):
            pass

        # Re-register (simulating module reload)
        # This should not raise an error
        try:
            registry.register(ReloadBB, name="bb.reload_bb_alt")
            registry.add_tag(ReloadBB, TAG_BLACKBOARD)
        except ValueError:
            # Already registered is fine
            pass


class TestTypedBlackboard:
    """Tests for TypedBlackboard with registry."""

    def test_typed_blackboard_registration(self, clean_registry):
        """Test that TypedBlackboard subclass can be registered."""
        @blackboard(name="typed_bb", scope="entity")
        class TypedBBTest(TypedBlackboard):
            pass

        registered = registry.get("bb.typed_bb")
        assert registered is TypedBBTest

    def test_typed_blackboard_factory(self, clean_registry):
        """Test that TypedBlackboard can be created from registry."""
        @blackboard(name="typed_factory", scope="entity")
        class TypedFactory(TypedBlackboard):
            pass

        instance = Blackboard.from_registry("typed_factory")
        assert isinstance(instance, TypedBlackboard)


class TestPerformance:
    """Performance tests for blackboard registry operations."""

    def test_100_queries_under_50ms(self, clean_registry):
        """Test that 100 queries complete under 50ms."""
        # Register several blackboards
        for i in range(10):
            exec(f"""
@blackboard(name="perf_bb_{i}", scope="entity")
class PerfBB{i}(Blackboard):
    pass
""", {"blackboard": blackboard, "Blackboard": Blackboard})

        # Perform 100 queries
        start = time.perf_counter()
        for _ in range(100):
            registry.query(tag=TAG_BLACKBOARD)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"100 queries took {elapsed*1000:.2f}ms (>50ms)"

    def test_scope_queries_performance(self, clean_registry):
        """Test that scope-filtered queries are fast."""
        # Register blackboards with different scopes
        for i in range(5):
            for scope in ["entity", "shared", "team"]:
                exec(f"""
@blackboard(name="perf_{scope}_{i}", scope="{scope}")
class Perf_{scope.title()}{i}(Blackboard):
    pass
""", {"blackboard": blackboard, "Blackboard": Blackboard})

        # Perform 100 scope-filtered queries
        start = time.perf_counter()
        for _ in range(100):
            get_blackboards_by_scope("entity")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"100 scope queries took {elapsed*1000:.2f}ms (>50ms)"

    def test_factory_instantiation_performance(self, clean_registry):
        """Test that factory instantiation is fast."""
        @blackboard(name="factory_perf", scope="entity")
        class FactoryPerf(Blackboard):
            pass

        # Create 100 instances
        start = time.perf_counter()
        for _ in range(100):
            Blackboard.from_registry("factory_perf")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"100 instantiations took {elapsed*1000:.2f}ms (>50ms)"

    def test_metadata_retrieval_performance(self, clean_registry):
        """Test that metadata retrieval is fast."""
        @blackboard(
            name="meta_perf",
            scope="entity",
            description="Performance test",
            key_types=["a", "b", "c"],
        )
        class MetaPerf(Blackboard):
            pass

        # Retrieve metadata 100 times
        start = time.perf_counter()
        for _ in range(100):
            get_blackboard_metadata("meta_perf")
        elapsed = time.perf_counter() - start

        assert elapsed < 0.05, f"100 metadata retrievals took {elapsed*1000:.2f}ms (>50ms)"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_name_works(self, clean_registry):
        """Test that empty string name is handled."""
        # Empty name should work (though not recommended)
        @blackboard(name="", scope="entity")
        class EmptyName(Blackboard):
            pass

        assert registry.get("bb.") is EmptyName

    def test_special_characters_in_name(self, clean_registry):
        """Test that special characters in name are handled."""
        @blackboard(name="special_name-with.chars_123", scope="entity")
        class SpecialName(Blackboard):
            pass

        registered = registry.get("bb.special_name-with.chars_123")
        assert registered is SpecialName

    def test_unicode_in_description(self, clean_registry):
        """Test that unicode in description is handled."""
        @blackboard(
            name="unicode_desc",
            scope="entity",
            description="Description with unicode: ☃ ❤ ★",
        )
        class UnicodeDesc(Blackboard):
            pass

        meta = get_blackboard_metadata("unicode_desc")
        assert "☃" in meta["description"]

    def test_long_key_types_list(self, clean_registry):
        """Test blackboard with many key types."""
        key_types = [f"key_{i}" for i in range(100)]

        @blackboard(name="long_keys", scope="entity", key_types=key_types)
        class LongKeys(Blackboard):
            pass

        stored_keys = LongKeys._blackboard_key_types
        assert len(stored_keys) == 100

    def test_factory_with_custom_init(self, clean_registry):
        """Test factory works with custom __init__."""
        @blackboard(name="custom_init", scope="entity")
        class CustomInit(Blackboard):
            def __init__(self, name: str = "default", custom_param: str = "default", **kwargs):
                super().__init__(name=name, **kwargs)
                self.custom_param = custom_param

        instance = Blackboard.from_registry("custom_init", custom_param="custom_value")
        assert instance.custom_param == "custom_value"


class TestIntegrationWithBlackboardFeatures:
    """Integration tests with core Blackboard functionality."""

    def test_registered_blackboard_set_get(self, clean_registry):
        """Test that registered blackboard supports set/get."""
        @blackboard(name="setget_bb", scope="entity")
        class SetGetBB(Blackboard):
            pass

        bb = Blackboard.from_registry("setget_bb")
        bb.set("key", "value")
        assert bb.get("key") == "value"

    def test_registered_blackboard_observers(self, clean_registry):
        """Test that registered blackboard supports observers."""
        @blackboard(name="observer_bb", scope="entity")
        class ObserverBB(Blackboard):
            pass

        bb = Blackboard.from_registry("observer_bb")
        changes = []

        def on_change(key, old_val, new_val):
            changes.append((key, old_val, new_val))

        bb.add_observer(on_change)
        bb.set("test_key", "test_value")

        assert len(changes) == 1

    def test_registered_blackboard_scopes(self, clean_registry):
        """Test that registered blackboard supports BlackboardScope."""
        @blackboard(name="scope_bb", scope="entity")
        class ScopeBB(Blackboard):
            pass

        bb = Blackboard.from_registry("scope_bb")
        scope = bb.create_scope("test_namespace")

        scope.set("key", "value")
        assert scope.get("key") == "value"

    def test_registered_blackboard_child(self, clean_registry):
        """Test that registered blackboard supports child blackboards."""
        @blackboard(name="parent_bb", scope="entity")
        class ParentBB(Blackboard):
            pass

        parent = Blackboard.from_registry("parent_bb")
        child = parent.create_child("child")

        parent.set("parent_key", "parent_value")
        assert child.get("parent_key") == "parent_value"

    def test_registered_blackboard_ttl(self, clean_registry):
        """Test that registered blackboard supports TTL."""
        @blackboard(name="ttl_bb", scope="entity")
        class TtlBB(Blackboard):
            pass

        bb = Blackboard.from_registry("ttl_bb")
        bb.set("expiring_key", "value", ttl=0.001)  # 1ms TTL

        # Should be present initially
        assert bb.has("expiring_key")

        # Wait for expiration
        time.sleep(0.01)

        # Should be expired
        assert not bb.has("expiring_key")
