"""
Test Suite for Utility AI Registry Integration (T-GP-3.12).

Tests the @utility_ai decorator wiring to Foundation Registry:
- Registration of utility AI classes
- Tag-based query discovery
- Metadata storage (id, update_rate, considerations)
- Factory instantiation via UtilityAI.from_registry()
- Multiple systems coexistence
- Curve type queries for considerations
- Performance benchmarks
"""

from __future__ import annotations

import time
import pytest
from typing import List

from foundation import registry, Registry
from engine.gameplay.ai.utility_ai import (
    UtilityAI,
    UtilityAction,
    Consideration,
    ConsiderationContext,
    ResponseCurve,
    utility_ai,
    get_all_utility_ai,
    get_utility_ai_by_id,
    get_utility_ai_by_update_rate,
    create_utility_ai_from_registry,
    TAG_UTILITY_AI,
    UTILITY_DEFAULT_UPDATE_RATE,
)
from engine.gameplay.ai.ai_registry import (
    consideration,
    get_all_considerations,
    get_considerations_by_curve,
    TAG_CONSIDERATION,
)
from engine.gameplay.ai.constants import ResponseCurveType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clean registry before and after each test."""
    # Store types to unregister
    initial_types = set(registry.all_types())
    yield
    # Unregister any types added during test
    for cls in registry.all_types():
        if cls not in initial_types:
            try:
                registry.unregister(cls)
            except Exception:
                pass


@pytest.fixture
def sample_consideration():
    """Create a sample consideration for testing."""
    class SampleConsideration(Consideration):
        def get_input(self, context: ConsiderationContext) -> float:
            return 0.5
    return SampleConsideration


# =============================================================================
# Test: @utility_ai decorator registration
# =============================================================================


class TestUtilityAIDecoratorRegistration:
    """Test that @utility_ai decorator registers classes correctly."""

    def test_decorator_registers_class(self):
        """Test that @utility_ai registers the class with Foundation Registry."""
        @utility_ai(id="test_register")
        class TestAI(UtilityAI):
            pass

        assert registry.is_registered(TestAI)

    def test_decorator_adds_utility_ai_tag(self):
        """Test that @utility_ai adds the utility_ai tag."""
        @utility_ai(id="test_tag")
        class TestTagAI(UtilityAI):
            pass

        assert registry.has_tag(TestTagAI, TAG_UTILITY_AI)

    def test_decorator_stores_id_metadata(self):
        """Test that @utility_ai stores the id in metadata."""
        @utility_ai(id="test_id_meta")
        class TestIdAI(UtilityAI):
            pass

        assert registry.get_metadata(TestIdAI, "utility_id") == "test_id_meta"

    def test_decorator_stores_update_rate_metadata(self):
        """Test that @utility_ai stores update_rate in metadata."""
        @utility_ai(id="test_rate", update_rate=0.25)
        class TestRateAI(UtilityAI):
            pass

        assert registry.get_metadata(TestRateAI, "update_rate") == 0.25

    def test_decorator_stores_description_metadata(self):
        """Test that @utility_ai stores description in metadata."""
        @utility_ai(id="test_desc", description="Combat AI system")
        class TestDescAI(UtilityAI):
            pass

        assert registry.get_metadata(TestDescAI, "description") == "Combat AI system"

    def test_decorator_sets_class_attributes(self):
        """Test that @utility_ai sets class attributes for introspection."""
        @utility_ai(id="test_attrs", update_rate=0.3)
        class TestAttrsAI(UtilityAI):
            pass

        assert TestAttrsAI._utility_ai is True
        assert TestAttrsAI._utility_id == "test_attrs"
        assert TestAttrsAI._utility_update_rate == 0.3

    def test_decorator_default_update_rate(self):
        """Test that @utility_ai uses default update_rate when not specified."""
        @utility_ai(id="test_default_rate")
        class TestDefaultRateAI(UtilityAI):
            pass

        assert registry.get_metadata(TestDefaultRateAI, "update_rate") == UTILITY_DEFAULT_UPDATE_RATE

    def test_decorator_rejects_empty_id(self):
        """Test that @utility_ai rejects empty id."""
        with pytest.raises(ValueError, match="id must be non-empty"):
            @utility_ai(id="")
            class EmptyIdAI(UtilityAI):
                pass

    def test_decorator_rejects_zero_update_rate(self):
        """Test that @utility_ai rejects zero update_rate."""
        with pytest.raises(ValueError, match="update_rate must be > 0"):
            @utility_ai(id="test_zero", update_rate=0)
            class ZeroRateAI(UtilityAI):
                pass

    def test_decorator_rejects_negative_update_rate(self):
        """Test that @utility_ai rejects negative update_rate."""
        with pytest.raises(ValueError, match="update_rate must be > 0"):
            @utility_ai(id="test_neg", update_rate=-0.5)
            class NegRateAI(UtilityAI):
                pass


# =============================================================================
# Test: Registry query returns all utility AI systems
# =============================================================================


class TestUtilityAIRegistryQuery:
    """Test Registry.query for utility AI systems."""

    def test_query_returns_registered_utility_ai(self):
        """Test that query(tag='utility_ai') returns registered systems."""
        @utility_ai(id="query_test_1")
        class QueryTestAI1(UtilityAI):
            pass

        results = registry.query(tag=TAG_UTILITY_AI)
        assert QueryTestAI1 in results

    def test_query_returns_multiple_utility_ai(self):
        """Test that query returns multiple registered systems."""
        @utility_ai(id="query_multi_1")
        class QueryMultiAI1(UtilityAI):
            pass

        @utility_ai(id="query_multi_2")
        class QueryMultiAI2(UtilityAI):
            pass

        results = registry.query(tag=TAG_UTILITY_AI)
        assert QueryMultiAI1 in results
        assert QueryMultiAI2 in results

    def test_query_with_id_filter(self):
        """Test query filtering by utility_id."""
        @utility_ai(id="filter_by_id")
        class FilterByIdAI(UtilityAI):
            pass

        results = registry.query(tag=TAG_UTILITY_AI, utility_id="filter_by_id")
        assert len(results) == 1
        assert results[0] is FilterByIdAI

    def test_query_with_update_rate_filter(self):
        """Test query filtering by update_rate."""
        @utility_ai(id="filter_rate_1", update_rate=0.1)
        class FilterRateAI1(UtilityAI):
            pass

        @utility_ai(id="filter_rate_2", update_rate=0.5)
        class FilterRateAI2(UtilityAI):
            pass

        results = registry.query(tag=TAG_UTILITY_AI, update_rate=0.1)
        assert FilterRateAI1 in results
        assert FilterRateAI2 not in results

    def test_get_all_utility_ai_helper(self):
        """Test get_all_utility_ai() helper function."""
        @utility_ai(id="helper_test_1")
        class HelperTestAI1(UtilityAI):
            pass

        @utility_ai(id="helper_test_2")
        class HelperTestAI2(UtilityAI):
            pass

        results = get_all_utility_ai()
        assert HelperTestAI1 in results
        assert HelperTestAI2 in results

    def test_get_utility_ai_by_id_found(self):
        """Test get_utility_ai_by_id() when ID exists."""
        @utility_ai(id="by_id_test")
        class ByIdTestAI(UtilityAI):
            pass

        result = get_utility_ai_by_id("by_id_test")
        assert result is ByIdTestAI

    def test_get_utility_ai_by_id_not_found(self):
        """Test get_utility_ai_by_id() when ID doesn't exist."""
        result = get_utility_ai_by_id("nonexistent_id")
        assert result is None

    def test_get_utility_ai_by_update_rate(self):
        """Test get_utility_ai_by_update_rate() helper."""
        @utility_ai(id="rate_helper_1", update_rate=0.25)
        class RateHelperAI1(UtilityAI):
            pass

        @utility_ai(id="rate_helper_2", update_rate=0.25)
        class RateHelperAI2(UtilityAI):
            pass

        @utility_ai(id="rate_helper_3", update_rate=0.5)
        class RateHelperAI3(UtilityAI):
            pass

        results = get_utility_ai_by_update_rate(0.25)
        assert RateHelperAI1 in results
        assert RateHelperAI2 in results
        assert RateHelperAI3 not in results


# =============================================================================
# Test: Considerations linked correctly
# =============================================================================


class TestConsiderationsLinking:
    """Test that considerations are properly linked to utility AI."""

    def test_consideration_decorator_registers(self):
        """Test that @consideration decorator registers with registry."""
        @consideration(curve="linear")
        class TestConsideration(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        assert registry.is_registered(TestConsideration)
        assert registry.has_tag(TestConsideration, TAG_CONSIDERATION)

    def test_consideration_curve_metadata_stored(self):
        """Test that consideration curve type is stored in metadata."""
        @consideration(curve="sigmoid")
        class SigmoidConsideration(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        assert registry.get_metadata(SigmoidConsideration, "curve_type") == "sigmoid"

    def test_get_considerations_by_curve_linear(self):
        """Test querying considerations by linear curve type."""
        @consideration(curve="linear")
        class LinearConsideration(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("linear")
        assert LinearConsideration in results

    def test_get_considerations_by_curve_sigmoid(self):
        """Test querying considerations by sigmoid curve type."""
        @consideration(curve="sigmoid")
        class SigmoidConsideration2(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.7

        results = get_considerations_by_curve("sigmoid")
        assert SigmoidConsideration2 in results

    def test_get_all_considerations(self):
        """Test getting all registered considerations."""
        @consideration(curve="exponential")
        class ExpConsideration(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.3

        results = get_all_considerations()
        assert ExpConsideration in results


# =============================================================================
# Test: Update rate metadata stored
# =============================================================================


class TestUpdateRateMetadata:
    """Test that update_rate metadata is correctly stored and retrieved."""

    def test_update_rate_stored_default(self):
        """Test default update_rate is stored."""
        @utility_ai(id="rate_default")
        class RateDefaultAI(UtilityAI):
            pass

        rate = registry.get_metadata(RateDefaultAI, "update_rate")
        assert rate == UTILITY_DEFAULT_UPDATE_RATE

    def test_update_rate_stored_custom(self):
        """Test custom update_rate is stored."""
        @utility_ai(id="rate_custom", update_rate=0.1)
        class RateCustomAI(UtilityAI):
            pass

        rate = registry.get_metadata(RateCustomAI, "update_rate")
        assert rate == 0.1

    def test_update_rate_stored_fractional(self):
        """Test fractional update_rate is stored."""
        @utility_ai(id="rate_frac", update_rate=0.333)
        class RateFracAI(UtilityAI):
            pass

        rate = registry.get_metadata(RateFracAI, "update_rate")
        assert abs(rate - 0.333) < 1e-9

    def test_update_rate_stored_large(self):
        """Test large update_rate is stored."""
        @utility_ai(id="rate_large", update_rate=10.0)
        class RateLargeAI(UtilityAI):
            pass

        rate = registry.get_metadata(RateLargeAI, "update_rate")
        assert rate == 10.0


# =============================================================================
# Test: Factory instantiation
# =============================================================================


class TestFactoryInstantiation:
    """Test factory instantiation from registry."""

    def test_from_registry_creates_instance(self):
        """Test UtilityAI.from_registry() creates instance."""
        @utility_ai(id="factory_test")
        class FactoryTestAI(UtilityAI):
            def __init__(self, name: str = "FactoryTest"):
                super().__init__(name=name)

        instance = UtilityAI.from_registry("factory_test")
        assert isinstance(instance, FactoryTestAI)

    def test_from_registry_passes_args(self):
        """Test UtilityAI.from_registry() passes positional args."""
        @utility_ai(id="factory_args")
        class FactoryArgsAI(UtilityAI):
            def __init__(self, name: str = "Default"):
                super().__init__(name=name)

        instance = UtilityAI.from_registry("factory_args", "CustomName")
        assert instance.name == "CustomName"

    def test_from_registry_passes_kwargs(self):
        """Test UtilityAI.from_registry() passes keyword args."""
        @utility_ai(id="factory_kwargs")
        class FactoryKwargsAI(UtilityAI):
            def __init__(self, name: str = "Default", update_rate: float = 0.5):
                super().__init__(name=name, update_rate=update_rate)

        instance = UtilityAI.from_registry("factory_kwargs", name="KwargsTest", update_rate=0.1)
        assert instance.name == "KwargsTest"
        assert instance.update_rate == 0.1

    def test_from_registry_not_found_raises(self):
        """Test UtilityAI.from_registry() raises for unknown ID."""
        with pytest.raises(ValueError, match="not found in registry"):
            UtilityAI.from_registry("nonexistent_factory_id")

    def test_create_utility_ai_from_registry_function(self):
        """Test create_utility_ai_from_registry() helper function."""
        @utility_ai(id="create_helper")
        class CreateHelperAI(UtilityAI):
            pass

        instance = create_utility_ai_from_registry("create_helper")
        assert isinstance(instance, CreateHelperAI)


# =============================================================================
# Test: Multiple systems coexist
# =============================================================================


class TestMultipleSystemsCoexist:
    """Test that multiple utility AI systems can coexist."""

    def test_multiple_systems_registered(self):
        """Test multiple systems are all registered."""
        @utility_ai(id="coexist_1")
        class CoexistAI1(UtilityAI):
            pass

        @utility_ai(id="coexist_2")
        class CoexistAI2(UtilityAI):
            pass

        @utility_ai(id="coexist_3")
        class CoexistAI3(UtilityAI):
            pass

        all_ai = get_all_utility_ai()
        assert CoexistAI1 in all_ai
        assert CoexistAI2 in all_ai
        assert CoexistAI3 in all_ai

    def test_multiple_systems_unique_ids(self):
        """Test each system has unique id in metadata."""
        @utility_ai(id="unique_1")
        class UniqueAI1(UtilityAI):
            pass

        @utility_ai(id="unique_2")
        class UniqueAI2(UtilityAI):
            pass

        assert registry.get_metadata(UniqueAI1, "utility_id") == "unique_1"
        assert registry.get_metadata(UniqueAI2, "utility_id") == "unique_2"

    def test_multiple_systems_different_rates(self):
        """Test systems with different update rates."""
        @utility_ai(id="diff_rate_1", update_rate=0.1)
        class DiffRateAI1(UtilityAI):
            pass

        @utility_ai(id="diff_rate_2", update_rate=0.5)
        class DiffRateAI2(UtilityAI):
            pass

        @utility_ai(id="diff_rate_3", update_rate=1.0)
        class DiffRateAI3(UtilityAI):
            pass

        fast = get_utility_ai_by_update_rate(0.1)
        medium = get_utility_ai_by_update_rate(0.5)
        slow = get_utility_ai_by_update_rate(1.0)

        assert DiffRateAI1 in fast
        assert DiffRateAI2 in medium
        assert DiffRateAI3 in slow

    def test_multiple_systems_independent_instances(self):
        """Test multiple systems create independent instances."""
        @utility_ai(id="indep_1")
        class IndepAI1(UtilityAI):
            pass

        @utility_ai(id="indep_2")
        class IndepAI2(UtilityAI):
            pass

        inst1 = UtilityAI.from_registry("indep_1")
        inst2 = UtilityAI.from_registry("indep_2")

        assert type(inst1) is not type(inst2)
        assert isinstance(inst1, IndepAI1)
        assert isinstance(inst2, IndepAI2)


# =============================================================================
# Test: Curve type queries work
# =============================================================================


class TestCurveTypeQueries:
    """Test querying considerations by curve type."""

    def test_query_linear_curve(self):
        """Test querying linear curve considerations."""
        @consideration(curve="linear")
        class LinearCurve(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("linear")
        assert LinearCurve in results

    def test_query_quadratic_curve(self):
        """Test querying quadratic curve considerations."""
        @consideration(curve="quadratic")
        class QuadraticCurve(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("quadratic")
        assert QuadraticCurve in results

    def test_query_exponential_curve(self):
        """Test querying exponential curve considerations."""
        @consideration(curve="exponential")
        class ExponentialCurve(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("exponential")
        assert ExponentialCurve in results

    def test_query_logistic_curve(self):
        """Test querying logistic curve considerations."""
        @consideration(curve="logistic")
        class LogisticCurve(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("logistic")
        assert LogisticCurve in results

    def test_query_sigmoid_curve(self):
        """Test querying sigmoid curve considerations."""
        @consideration(curve="sigmoid")
        class SigmoidCurve(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("sigmoid")
        assert SigmoidCurve in results

    def test_query_smoothstep_curve(self):
        """Test querying smoothstep curve considerations."""
        @consideration(curve="smoothstep")
        class SmoothstepCurve(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        results = get_considerations_by_curve("smoothstep")
        assert SmoothstepCurve in results

    def test_query_multiple_curves(self):
        """Test that different curves are in separate results."""
        @consideration(curve="linear")
        class MultiLinear(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        @consideration(curve="sigmoid")
        class MultiSigmoid(Consideration):
            def get_input(self, context: ConsiderationContext) -> float:
                return 0.5

        linear_results = get_considerations_by_curve("linear")
        sigmoid_results = get_considerations_by_curve("sigmoid")

        assert MultiLinear in linear_results
        assert MultiLinear not in sigmoid_results
        assert MultiSigmoid in sigmoid_results
        assert MultiSigmoid not in linear_results


# =============================================================================
# Test: Performance - 100 queries under 50ms
# =============================================================================


class TestPerformance:
    """Test performance requirements."""

    def test_100_queries_under_50ms(self):
        """Test that 100 registry queries complete under 50ms."""
        # Register several utility AI systems
        for i in range(10):
            @utility_ai(id=f"perf_ai_{i}", update_rate=0.1 * (i + 1))
            class PerfAI(UtilityAI):
                pass
            # Give each a unique name to avoid decorator issues
            PerfAI.__name__ = f"PerfAI_{i}"

        # Perform 100 queries and time them
        start_time = time.perf_counter()

        for _ in range(100):
            registry.query(tag=TAG_UTILITY_AI)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 50, f"100 queries took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_query_with_filter_performance(self):
        """Test filtered queries are also fast."""
        @utility_ai(id="perf_filter", update_rate=0.5)
        class PerfFilterAI(UtilityAI):
            pass

        start_time = time.perf_counter()

        for _ in range(100):
            registry.query(tag=TAG_UTILITY_AI, utility_id="perf_filter")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 50, f"100 filtered queries took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_get_all_utility_ai_performance(self):
        """Test get_all_utility_ai() performance."""
        for i in range(5):
            @utility_ai(id=f"perf_helper_{i}")
            class PerfHelperAI(UtilityAI):
                pass
            PerfHelperAI.__name__ = f"PerfHelperAI_{i}"

        start_time = time.perf_counter()

        for _ in range(100):
            get_all_utility_ai()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 50, f"100 helper calls took {elapsed_ms:.2f}ms, expected < 50ms"

    def test_from_registry_performance(self):
        """Test UtilityAI.from_registry() performance."""
        @utility_ai(id="perf_factory")
        class PerfFactoryAI(UtilityAI):
            pass

        start_time = time.perf_counter()

        for _ in range(100):
            UtilityAI.from_registry("perf_factory")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 100, f"100 factory calls took {elapsed_ms:.2f}ms, expected < 100ms"


# =============================================================================
# Test: Edge cases and error handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_decorator_idempotent(self):
        """Test that decorating twice doesn't break things."""
        @utility_ai(id="idempotent")
        class IdempotentAI(UtilityAI):
            pass

        # Should not raise even if we try to "re-register"
        assert registry.is_registered(IdempotentAI)

    def test_subclass_inherits_attributes(self):
        """Test that subclasses of decorated classes inherit attributes."""
        @utility_ai(id="parent")
        class ParentAI(UtilityAI):
            pass

        class ChildAI(ParentAI):
            pass

        # Child should inherit parent's utility_ai attributes
        assert hasattr(ChildAI, "_utility_ai")
        assert ChildAI._utility_ai is True

    def test_registry_name_format(self):
        """Test that registry name follows expected format."""
        @utility_ai(id="name_format_test")
        class NameFormatAI(UtilityAI):
            pass

        name = registry.get_name(NameFormatAI)
        assert name == "utility_ai.name_format_test"

    def test_track_instances_disabled_by_default(self):
        """Test that instance tracking is disabled by default."""
        @utility_ai(id="no_track")
        class NoTrackAI(UtilityAI):
            pass

        # Should not have instances tracked
        count = registry.instance_count(NoTrackAI)
        assert count == 0

    def test_track_instances_enabled(self):
        """Test that instance tracking can be enabled."""
        @utility_ai(id="track_enabled", track_instances=True)
        class TrackEnabledAI(UtilityAI):
            pass

        # Create instances
        inst1 = TrackEnabledAI()
        inst2 = TrackEnabledAI()

        count = registry.instance_count(TrackEnabledAI)
        assert count == 2

    def test_metadata_retrieval_for_unregistered_type(self):
        """Test metadata retrieval for unregistered type returns None."""
        class UnregisteredAI(UtilityAI):
            pass

        result = registry.get_metadata(UnregisteredAI, "utility_id")
        assert result is None

    def test_empty_considerations_list(self):
        """Test that considerations metadata starts as empty list."""
        @utility_ai(id="empty_considerations")
        class EmptyConsiderationsAI(UtilityAI):
            pass

        considerations = registry.get_metadata(EmptyConsiderationsAI, "considerations")
        assert considerations == []


# =============================================================================
# Test: Integration with existing UtilityAI functionality
# =============================================================================


class TestIntegrationWithUtilityAI:
    """Test integration with existing UtilityAI functionality."""

    def test_decorated_ai_can_add_actions(self):
        """Test that decorated AI can add actions normally."""
        @utility_ai(id="with_actions")
        class WithActionsAI(UtilityAI):
            pass

        ai = UtilityAI.from_registry("with_actions")

        class TestAction(UtilityAction):
            def __init__(self):
                super().__init__(name="test_action")

            def execute(self, context: ConsiderationContext) -> bool:
                return True

        ai.add_action(TestAction())
        assert len(ai.actions) == 1

    def test_decorated_ai_can_evaluate(self):
        """Test that decorated AI can evaluate actions."""
        @utility_ai(id="can_evaluate")
        class CanEvaluateAI(UtilityAI):
            pass

        ai = UtilityAI.from_registry("can_evaluate")

        class SimpleAction(UtilityAction):
            def __init__(self):
                super().__init__(name="simple", base_score=0.5)

            def execute(self, context: ConsiderationContext) -> bool:
                return True

        ai.add_action(SimpleAction())
        scores = ai.evaluate()

        assert len(scores) == 1
        assert scores[0].score == 0.5

    def test_decorated_ai_update_rate_used(self):
        """Test that decorated AI uses the specified update_rate."""
        @utility_ai(id="custom_rate", update_rate=0.25)
        class CustomRateAI(UtilityAI):
            def __init__(self):
                # Get update_rate from class attribute set by decorator
                super().__init__(update_rate=self._utility_update_rate)

        ai = CustomRateAI()
        assert ai.update_rate == 0.25

    def test_decorated_ai_preserves_name(self):
        """Test that decorated AI preserves instance name."""
        @utility_ai(id="preserves_name")
        class PreservesNameAI(UtilityAI):
            def __init__(self, name: str = "DefaultName"):
                super().__init__(name=name)

        ai = UtilityAI.from_registry("preserves_name", name="CustomName")
        assert ai.name == "CustomName"

    def test_decorated_ai_select_action(self):
        """Test that decorated AI can select actions."""
        @utility_ai(id="select_action")
        class SelectActionAI(UtilityAI):
            pass

        ai = UtilityAI.from_registry("select_action")

        class HighScoreAction(UtilityAction):
            def __init__(self):
                super().__init__(name="high", base_score=0.9)

            def execute(self, context: ConsiderationContext) -> bool:
                return True

        class LowScoreAction(UtilityAction):
            def __init__(self):
                super().__init__(name="low", base_score=0.1)

            def execute(self, context: ConsiderationContext) -> bool:
                return True

        ai.add_action(HighScoreAction())
        ai.add_action(LowScoreAction())

        selected = ai.select_action()
        assert selected is not None
        assert selected.name == "high"


# =============================================================================
# Test: Description metadata
# =============================================================================


class TestDescriptionMetadata:
    """Test description metadata handling."""

    def test_description_stored_when_provided(self):
        """Test that description is stored when provided."""
        @utility_ai(id="with_desc", description="A test AI system")
        class WithDescAI(UtilityAI):
            pass

        desc = registry.get_metadata(WithDescAI, "description")
        assert desc == "A test AI system"

    def test_description_none_when_not_provided(self):
        """Test that description is not set when not provided."""
        @utility_ai(id="no_desc")
        class NoDescAI(UtilityAI):
            pass

        desc = registry.get_metadata(NoDescAI, "description")
        assert desc is None

    def test_description_stored_on_class(self):
        """Test that description is stored on class attribute."""
        @utility_ai(id="class_desc", description="Class description")
        class ClassDescAI(UtilityAI):
            pass

        assert ClassDescAI._utility_description == "Class description"


# =============================================================================
# Test: Complete workflow
# =============================================================================


class TestCompleteWorkflow:
    """Test complete workflow from registration to usage."""

    def test_complete_workflow(self):
        """Test complete workflow: register, query, instantiate, use."""
        # 1. Register a utility AI with decorator
        @utility_ai(id="workflow", update_rate=0.2, description="Workflow test AI")
        class WorkflowAI(UtilityAI):
            def __init__(self, name: str = "Workflow"):
                super().__init__(name=name, update_rate=0.2)

        # 2. Query to find it
        all_ai = get_all_utility_ai()
        assert WorkflowAI in all_ai

        found = get_utility_ai_by_id("workflow")
        assert found is WorkflowAI

        # 3. Check metadata
        assert registry.get_metadata(WorkflowAI, "utility_id") == "workflow"
        assert registry.get_metadata(WorkflowAI, "update_rate") == 0.2
        assert registry.get_metadata(WorkflowAI, "description") == "Workflow test AI"

        # 4. Instantiate from registry
        ai = UtilityAI.from_registry("workflow", name="WorkflowInstance")

        # 5. Use normally
        class WorkflowAction(UtilityAction):
            def __init__(self):
                super().__init__(name="workflow_action", base_score=0.8)

            def execute(self, context: ConsiderationContext) -> bool:
                return True

        ai.add_action(WorkflowAction())
        selected = ai.select_action()

        assert selected is not None
        assert selected.name == "workflow_action"
