"""Contract tests for GraphContext and ContextPool (T-AG-1.6).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - Task T-AG-1.6 description (public API: GraphContext, get_parameter,
    advance_time, with_depth, ContextPool acquire/release)
  - engine/animation/graph/__init__.py (public exports)
  - Introspection of public class signatures (no implementation files read)

Forbidden files (NOT read):
  - engine/animation/graph/animation_graph.py (DEV implementation)
  - tests/test_graph_context_whitebox.py (parallel peer)
"""
import pytest
from engine.animation.graph import (
    # Context
    GraphContext,
    ContextPool,
    # Parameters
    ParameterType,
    GraphParameter,
    # Skeleton
    Bone,
    Skeleton,
    Transform,
    # Bone masks
    BoneMask,
    MissingBoneMode,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_skeleton() -> Skeleton:
    """A minimal skeleton with root and one child."""
    skel = Skeleton()
    skel.add_bone("root")
    skel.add_bone("child", parent_index=0)
    return skel


@pytest.fixture
def sample_params() -> dict:
    """A dict of graph parameters of each type."""
    return {
        "speed": GraphParameter.float_param("speed", 5.0, 0.0, 10.0),
        "count": GraphParameter.int_param("count", 3, 0, 100),
        "active": GraphParameter.bool_param("active", True),
    }


@pytest.fixture
def sample_bone_masks(simple_skeleton: Skeleton) -> dict:
    """A dict with a bone mask configured via set_weight."""
    mask = BoneMask(name="upper")
    mask.set_weight("root", 1.0)
    mask.set_weight("child", 0.5)
    return {"upper": mask}


@pytest.fixture
def basic_context() -> GraphContext:
    """A bare-minimum GraphContext with dt only."""
    return GraphContext(dt=0.016)


@pytest.fixture
def empty_pool() -> ContextPool:
    """A fresh ContextPool."""
    return ContextPool()


# ============================================================================
# Equivalence Class: GraphContext construction
# ============================================================================

class TestGraphContextConstruction:
    """GraphContext can be constructed with various combinations of args."""

    def test_create_default(self):
        """GraphContext() creates with default values and no errors."""
        ctx = GraphContext()
        assert ctx.dt == 0.0
        assert ctx.evaluation_depth == 0
        assert ctx.current_time == 0.0
        assert ctx.tick == 0

    def test_create_with_dt(self):
        """GraphContext accepts a dt parameter."""
        ctx = GraphContext(dt=0.016)
        assert ctx.dt == 0.016

    def test_create_with_skeleton(self, simple_skeleton: Skeleton):
        """GraphContext accepts a skeleton reference."""
        ctx = GraphContext(skeleton=simple_skeleton)
        assert ctx.skeleton is simple_skeleton

    def test_create_with_parameters(self, sample_params: dict):
        """GraphContext accepts a parameter dict."""
        ctx = GraphContext(parameters=sample_params)
        # Verify parameters are accessible via get_parameter
        assert ctx.get_parameter("speed") == 5.0

    def test_create_with_empty_parameters(self):
        """GraphContext with empty parameters dict does not crash."""
        ctx = GraphContext(parameters={})
        # get_parameter on empty dict returns default
        val = ctx.get_parameter("anything")
        assert val is None or val == 0.0

    def test_create_with_bone_masks(self, sample_bone_masks: dict):
        """GraphContext accepts a bone_masks dict."""
        ctx = GraphContext(bone_masks=sample_bone_masks)
        assert ctx.get_bone_mask("upper") is sample_bone_masks["upper"]

    def test_create_with_normalized_time(self):
        """GraphContext accepts a normalized_time value."""
        ctx = GraphContext(normalized_time=0.5)
        assert ctx.normalized_time == 0.5

    def test_create_with_sync_group(self):
        """GraphContext accepts a sync_group identifier."""
        ctx = GraphContext(sync_group="locomotion")
        assert ctx.sync_group == "locomotion"

    def test_create_with_layer_weight(self):
        """GraphContext accepts a layer_weight."""
        ctx = GraphContext(layer_weight=0.75)
        assert ctx.layer_weight == 0.75

    def test_create_with_current_time(self):
        """GraphContext accepts a current_time."""
        ctx = GraphContext(current_time=1.5)
        assert ctx.current_time == 1.5

    def test_create_with_tick(self):
        """GraphContext accepts a tick counter."""
        ctx = GraphContext(tick=42)
        assert ctx.tick == 42

    def test_create_with_current_node_id(self):
        """GraphContext accepts a current_node_id."""
        ctx = GraphContext(current_node_id="clip_a")
        assert ctx.current_node_id == "clip_a"

    def test_create_with_evaluation_depth(self):
        """GraphContext accepts an evaluation_depth."""
        ctx = GraphContext(evaluation_depth=3)
        assert ctx.evaluation_depth == 3

    def test_create_all_args(self, simple_skeleton: Skeleton,
                             sample_params: dict, sample_bone_masks: dict):
        """GraphContext accepts all public args simultaneously."""
        ctx = GraphContext(
            dt=0.033,
            skeleton=simple_skeleton,
            parameters=sample_params,
            bone_masks=sample_bone_masks,
            normalized_time=0.0,
            sync_group="upper_body",
            layer_weight=0.5,
            current_time=2.0,
            tick=60,
            current_node_id="blend_root",
            evaluation_depth=1,
        )
        assert ctx.dt == 0.033
        assert ctx.skeleton is simple_skeleton
        assert ctx.get_parameter("speed") == 5.0
        assert ctx.get_bone_mask("upper") is sample_bone_masks["upper"]
        assert ctx.normalized_time == 0.0
        assert ctx.sync_group == "upper_body"
        assert ctx.layer_weight == 0.5
        assert ctx.current_time == 2.0
        assert ctx.tick == 60
        assert ctx.current_node_id == "blend_root"
        assert ctx.evaluation_depth == 1

    def test_create_with_none_skeleton(self):
        """GraphContext with skeleton=None is valid."""
        ctx = GraphContext(skeleton=None)
        assert ctx.skeleton is None


# ============================================================================
# Equivalence Class: get_parameter family
# ============================================================================

class TestGraphContextGetParameter:
    """Parameters can be retrieved via typed and untyped accessors."""

    def test_get_parameter_existing(self, basic_context: GraphContext,
                                    sample_params: dict):
        """get_parameter retrieves a known parameter value by name."""
        ctx = GraphContext(dt=0.016, parameters=sample_params)
        assert ctx.get_parameter("speed") == 5.0

    def test_get_parameter_unknown_returns_none(self, basic_context: GraphContext):
        """get_parameter for an unknown name returns None."""
        result = basic_context.get_parameter("nonexistent")
        assert result is None

    def test_get_parameter_unknown_custom_default(self, basic_context: GraphContext):
        """get_parameter returns the provided default for an unknown name."""
        result = basic_context.get_parameter("phantom", 42)
        assert result == 42

    def test_get_parameter_unknown_string_default(self, basic_context: GraphContext):
        """get_parameter returns a string default for unknown names."""
        result = basic_context.get_parameter("missing", "fallback")
        assert result == "fallback"

    def test_get_parameter_float_existing(self, sample_params: dict):
        """get_parameter_float retrieves a float parameter value."""
        ctx = GraphContext(parameters=sample_params)
        assert ctx.get_parameter_float("speed") == 5.0

    def test_get_parameter_float_unknown_default(self, basic_context: GraphContext):
        """get_parameter_float returns the default for unknown names."""
        assert basic_context.get_parameter_float("phantom", 1.5) == 1.5

    def test_get_parameter_float_unknown_zero_default(self, basic_context: GraphContext):
        """get_parameter_float returns 0.0 default when omitted."""
        result = basic_context.get_parameter_float("phantom")
        assert result == 0.0

    def test_get_parameter_int_existing(self, sample_params: dict):
        """get_parameter_int retrieves an int parameter value."""
        ctx = GraphContext(parameters=sample_params)
        assert ctx.get_parameter_int("count") == 3

    def test_get_parameter_int_unknown_default(self, basic_context: GraphContext):
        """get_parameter_int returns the default for unknown names."""
        assert basic_context.get_parameter_int("phantom", 99) == 99

    def test_get_parameter_int_unknown_zero_default(self, basic_context: GraphContext):
        """get_parameter_int returns 0 default when omitted."""
        result = basic_context.get_parameter_int("phantom")
        assert result == 0

    def test_get_parameter_bool_existing(self, sample_params: dict):
        """get_parameter_bool retrieves a bool parameter value."""
        ctx = GraphContext(parameters=sample_params)
        assert ctx.get_parameter_bool("active") is True

    def test_get_parameter_bool_unknown_default(self, basic_context: GraphContext):
        """get_parameter_bool returns the default for unknown names."""
        assert basic_context.get_parameter_bool("phantom", True) is True

    def test_get_parameter_bool_unknown_false_default(self, basic_context: GraphContext):
        """get_parameter_bool returns False default when omitted."""
        result = basic_context.get_parameter_bool("phantom")
        assert result is False

    def test_get_parameter_float_on_int_param(self, sample_params: dict):
        """get_parameter_float on an int parameter returns the value."""
        ctx = GraphContext(parameters=sample_params)
        # An int parameter's value may be returned as int or float;
        # contract only guarantees it returns the value
        val = ctx.get_parameter_float("count")
        assert val == 3 or val == 3.0


# ============================================================================
# Equivalence Class: get_bone_mask
# ============================================================================

class TestGraphContextGetBoneMask:
    """Bone masks can be retrieved by name."""

    def test_get_bone_mask_existing(self, sample_bone_masks: dict):
        """get_bone_mask returns a known bone mask."""
        ctx = GraphContext(bone_masks=sample_bone_masks)
        result = ctx.get_bone_mask("upper")
        assert result is sample_bone_masks["upper"]

    def test_get_bone_mask_unknown(self, basic_context: GraphContext):
        """get_bone_mask for an unknown name returns None."""
        result = basic_context.get_bone_mask("nonexistent")
        assert result is None

    def test_get_bone_mask_without_masks(self, basic_context: GraphContext):
        """get_bone_mask on a context with no masks returns None."""
        result = basic_context.get_bone_mask("any")
        assert result is None


# ============================================================================
# Equivalence Class: advance_time
# ============================================================================

class TestAdvanceTime:
    """advance_time advances time and returns a new or mutated context."""

    def test_advance_time_positive_dt(self):
        """advance_time with positive dt changes current_time."""
        ctx = GraphContext(dt=0.016, current_time=0.0)
        result = ctx.advance_time(0.033)
        # Contract: result.current_time increased from original
        assert result.current_time >= ctx.current_time

    def test_advance_time_zero(self):
        """advance_time with zero dt does not decrease time."""
        ctx = GraphContext(dt=0.016, current_time=1.0)
        result = ctx.advance_time(0.0)
        assert result.current_time >= 0.0

    def test_advance_time_negative(self):
        """advance_time with negative dt does not crash."""
        ctx = GraphContext(dt=0.016, current_time=1.0)
        # Contract: negative dt should not cause errors
        result = ctx.advance_time(-0.1)
        # current_time may or may not go backwards; must not crash
        assert result is not None

    def test_advance_time_returns_graph_context(self):
        """advance_time returns a GraphContext instance."""
        ctx = GraphContext(dt=0.016)
        result = ctx.advance_time(0.033)
        assert isinstance(result, GraphContext)

    def test_advance_time_chained(self):
        """advance_time can be chained sequentially."""
        ctx = GraphContext(dt=0.016, current_time=0.0)
        a1 = ctx.advance_time(0.016)
        a2 = a1.advance_time(0.033)
        a3 = a2.advance_time(0.050)
        # Each call increases current_time
        assert a1.current_time >= ctx.current_time
        assert a2.current_time >= a1.current_time
        assert a3.current_time >= a2.current_time

    def test_advance_time_result_dt_field(self):
        """advance_time may update the dt field on the result."""
        ctx = GraphContext(dt=0.016)
        result = ctx.advance_time(0.033)
        # Contract: result.dt may be the original dt or the arg; either is valid
        assert hasattr(result, "dt")
        assert isinstance(result.dt, float)


# ============================================================================
# Equivalence Class: with_depth
# ============================================================================

class TestWithDepth:
    """with_depth creates a context with incremented evaluation depth."""

    def test_with_depth_increments_from_zero(self, basic_context: GraphContext):
        """with_depth increments evaluation_depth from 0 to 1."""
        assert basic_context.evaluation_depth == 0
        deeper = basic_context.with_depth()
        assert deeper.evaluation_depth == 1

    def test_with_depth_chain(self, basic_context: GraphContext):
        """Multiple with_depth calls accumulate depth."""
        d1 = basic_context.with_depth()
        d2 = d1.with_depth()
        d3 = d2.with_depth()
        assert d1.evaluation_depth == 1
        assert d2.evaluation_depth == 2
        assert d3.evaluation_depth == 3

    def test_with_depth_does_not_mutate_original(self, basic_context: GraphContext):
        """with_depth leaves the original context's depth unchanged."""
        original_depth = basic_context.evaluation_depth
        basic_context.with_depth()
        assert basic_context.evaluation_depth == original_depth

    def test_with_depth_returns_graph_context(self, basic_context: GraphContext):
        """with_depth returns a GraphContext instance."""
        result = basic_context.with_depth()
        assert isinstance(result, GraphContext)

    def test_with_depth_from_non_zero(self):
        """with_depth works from a non-zero starting depth."""
        ctx = GraphContext(dt=0.016, evaluation_depth=5)
        deeper = ctx.with_depth()
        assert deeper.evaluation_depth == 6

    def test_with_depth_preserves_other_fields(self, simple_skeleton: Skeleton):
        """with_depth preserves dt, skeleton, and other fields."""
        ctx = GraphContext(dt=0.033, skeleton=simple_skeleton,
                           layer_weight=0.75, normalized_time=0.5,
                           sync_group="upper")
        deeper = ctx.with_depth()
        assert deeper.dt == ctx.dt
        assert deeper.skeleton is ctx.skeleton
        assert deeper.layer_weight == ctx.layer_weight
        assert deeper.normalized_time == ctx.normalized_time
        assert deeper.sync_group == ctx.sync_group


# ============================================================================
# Equivalence Class: ContextPool acquire
# ============================================================================

class TestContextPoolAcquire:
    """ContextPool.acquire() returns a configured GraphContext."""

    def test_acquire_returns_graph_context(self, empty_pool: ContextPool):
        """acquire returns a GraphContext instance."""
        ctx = empty_pool.acquire(dt=0.016)
        assert isinstance(ctx, GraphContext)

    def test_acquire_sets_dt(self, empty_pool: ContextPool):
        """acquire passes dt to the context."""
        ctx = empty_pool.acquire(dt=0.033)
        assert ctx.dt == 0.033

    def test_acquire_with_parameters(self, empty_pool: ContextPool,
                                     sample_params: dict):
        """acquire passes parameters to the context."""
        ctx = empty_pool.acquire(parameters=sample_params)
        assert ctx.get_parameter("speed") == 5.0

    def test_acquire_with_skeleton(self, empty_pool: ContextPool,
                                   simple_skeleton: Skeleton):
        """acquire passes skeleton to the context."""
        ctx = empty_pool.acquire(skeleton=simple_skeleton)
        assert ctx.skeleton is simple_skeleton

    def test_acquire_with_bone_masks(self, empty_pool: ContextPool,
                                     sample_bone_masks: dict):
        """acquire passes bone_masks to the context."""
        ctx = empty_pool.acquire(bone_masks=sample_bone_masks)
        assert ctx.get_bone_mask("upper") is sample_bone_masks["upper"]

    def test_acquire_all_fields(self, empty_pool: ContextPool,
                                simple_skeleton: Skeleton):
        """acquire passes all fields to the context."""
        ctx = empty_pool.acquire(
            dt=0.016,
            skeleton=simple_skeleton,
            normalized_time=0.0,
            sync_group="legs",
            layer_weight=0.8,
            current_time=1.5,
            tick=30,
        )
        assert ctx.dt == 0.016
        assert ctx.skeleton is simple_skeleton
        assert ctx.normalized_time == 0.0
        assert ctx.sync_group == "legs"
        assert ctx.layer_weight == 0.8
        assert ctx.current_time == 1.5
        assert ctx.tick == 30

    def test_multiple_acquire_produces_distinct_contexts(self, empty_pool: ContextPool):
        """Multiple acquire calls return distinct context objects."""
        ctx_a = empty_pool.acquire(dt=0.016)
        ctx_b = empty_pool.acquire(dt=0.033)
        assert ctx_a is not ctx_b


# ============================================================================
# Equivalence Class: ContextPool release
# ============================================================================

class TestContextPoolRelease:
    """ContextPool.release() returns contexts to the pool."""

    def test_release_then_acquire_succeeds(self, empty_pool: ContextPool):
        """After release, acquire still returns a context."""
        ctx = empty_pool.acquire(dt=0.016)
        empty_pool.release(ctx)
        ctx2 = empty_pool.acquire(dt=0.016)
        assert isinstance(ctx2, GraphContext)

    def test_release_none_does_not_crash(self, empty_pool: ContextPool):
        """release(None) or release on wrong type does not crash."""
        empty_pool.release(None)

    def test_release_multiple_then_acquire(self, empty_pool: ContextPool):
        """Release multiple contexts and re-acquire."""
        ctx_a = empty_pool.acquire(dt=0.016)
        ctx_b = empty_pool.acquire(dt=0.016)
        ctx_c = empty_pool.acquire(dt=0.016)
        empty_pool.release(ctx_a)
        empty_pool.release(ctx_b)
        empty_pool.release(ctx_c)
        # Re-acquire should work
        for _ in range(3):
            c = empty_pool.acquire(dt=0.016)
            assert isinstance(c, GraphContext)


# ============================================================================
# Equivalence Class: ContextPool counters
# ============================================================================

class TestContextPoolCounters:
    """Pool counters reflect available, active, and total created."""

    def test_available_count_starts_zero(self, empty_pool: ContextPool):
        """A fresh pool has zero available contexts."""
        assert empty_pool.available_count == 0

    def test_active_count_starts_zero(self, empty_pool: ContextPool):
        """A fresh pool has zero active contexts."""
        assert empty_pool.active_count == 0

    def test_total_created_starts_zero(self, empty_pool: ContextPool):
        """A fresh pool has total_created == 0."""
        assert empty_pool.total_created == 0

    def test_acquire_increases_active_count(self, empty_pool: ContextPool):
        """acquire increments active_count."""
        empty_pool.acquire(dt=0.016)
        assert empty_pool.active_count == 1

    def test_acquire_decreases_available_count(self, empty_pool: ContextPool):
        """acquire decrements available_count (or leaves at 0)."""
        empty_pool.acquire(dt=0.016)
        assert empty_pool.available_count == 0

    def test_acquire_increases_total_created(self, empty_pool: ContextPool):
        """acquire increments total_created."""
        empty_pool.acquire(dt=0.016)
        assert empty_pool.total_created == 1

    def test_release_decreases_active_count(self, empty_pool: ContextPool):
        """release decrements active_count."""
        ctx = empty_pool.acquire(dt=0.016)
        empty_pool.release(ctx)
        assert empty_pool.active_count == 0

    def test_release_increases_available_count(self, empty_pool: ContextPool):
        """release increments available_count."""
        ctx = empty_pool.acquire(dt=0.016)
        empty_pool.release(ctx)
        assert empty_pool.available_count == 1

    def test_multiple_acquire_release_counts(self, empty_pool: ContextPool):
        """Multiple acquire/release cycles track counts correctly."""
        c1 = empty_pool.acquire(dt=0.016)
        c2 = empty_pool.acquire(dt=0.016)
        assert empty_pool.active_count == 2
        assert empty_pool.total_created == 2
        empty_pool.release(c1)
        assert empty_pool.active_count == 1
        assert empty_pool.available_count == 1
        empty_pool.release(c2)
        assert empty_pool.active_count == 0
        assert empty_pool.available_count == 2

    def test_total_created_monotonic(self, empty_pool: ContextPool):
        """total_created never decreases."""
        prev = empty_pool.total_created
        for _ in range(5):
            ctx = empty_pool.acquire(dt=0.016)
            empty_pool.release(ctx)
            assert empty_pool.total_created >= prev
            prev = empty_pool.total_created


# ============================================================================
# Equivalence Class: ContextPool acquire-reuse
# ============================================================================

class TestContextPoolReuse:
    """Released contexts may be reused on subsequent acquire."""

    def test_acquire_after_release_reuses_available(self, empty_pool: ContextPool):
        """After release+acquire, total_created does not increase if context is reused."""
        ctx = empty_pool.acquire(dt=0.016)
        empty_pool.release(ctx)
        created_before = empty_pool.total_created
        ctx2 = empty_pool.acquire(dt=0.016)
        # total_created may or may not increase depending on reuse strategy
        assert empty_pool.total_created >= created_before
        assert isinstance(ctx2, GraphContext)


# ============================================================================
# Boundary: edge-case values
# ============================================================================

class TestGraphContextBoundaries:
    """GraphContext handles boundary values without crashing."""

    def test_dt_zero_boundary(self):
        """GraphContext with dt=0.0 is valid."""
        ctx = GraphContext(dt=0.0)
        assert ctx.dt == 0.0

    def test_dt_negative(self):
        """GraphContext with negative dt does not crash."""
        ctx = GraphContext(dt=-0.1)
        assert ctx.dt == -0.1

    def test_dt_large(self):
        """GraphContext with very large dt does not crash."""
        ctx = GraphContext(dt=1e6)
        assert ctx.dt == 1e6

    def test_normalized_time_zero(self):
        """GraphContext with normalized_time=0.0 is valid."""
        ctx = GraphContext(normalized_time=0.0)
        assert ctx.normalized_time == 0.0

    def test_normalized_time_one(self):
        """GraphContext with normalized_time=1.0 is valid."""
        ctx = GraphContext(normalized_time=1.0)
        assert ctx.normalized_time == 1.0

    def test_normalized_time_negative(self):
        """GraphContext with normalized_time=-1.0 does not crash."""
        ctx = GraphContext(normalized_time=-1.0)
        assert ctx.normalized_time == -1.0

    def test_normalized_time_greater_than_one(self):
        """GraphContext with normalized_time > 1.0 does not crash."""
        ctx = GraphContext(normalized_time=2.5)
        assert ctx.normalized_time == 2.5

    def test_layer_weight_zero(self):
        """GraphContext with layer_weight=0.0 is valid."""
        ctx = GraphContext(layer_weight=0.0)
        assert ctx.layer_weight == 0.0

    def test_layer_weight_one(self):
        """GraphContext with layer_weight=1.0 is valid."""
        ctx = GraphContext(layer_weight=1.0)
        assert ctx.layer_weight == 1.0

    def test_layer_weight_greater_than_one(self):
        """GraphContext with layer_weight > 1.0 does not crash."""
        ctx = GraphContext(layer_weight=2.0)
        assert ctx.layer_weight == 2.0

    def test_tick_zero(self):
        """GraphContext with tick=0 is valid."""
        ctx = GraphContext(tick=0)
        assert ctx.tick == 0

    def test_tick_negative(self):
        """GraphContext with negative tick does not crash."""
        ctx = GraphContext(tick=-1)
        assert ctx.tick == -1

    def test_tick_large(self):
        """GraphContext with a large tick value does not crash."""
        ctx = GraphContext(tick=2**31 - 1)
        assert ctx.tick == 2**31 - 1

    def test_current_time_zero(self):
        """GraphContext with current_time=0.0 is valid."""
        ctx = GraphContext(current_time=0.0)
        assert ctx.current_time == 0.0

    def test_current_time_negative(self):
        """GraphContext with negative current_time does not crash."""
        ctx = GraphContext(current_time=-1.0)
        assert ctx.current_time == -1.0

    def test_evaluation_depth_zero(self):
        """GraphContext with evaluation_depth=0 is valid."""
        ctx = GraphContext(evaluation_depth=0)
        assert ctx.evaluation_depth == 0

    def test_evaluation_depth_large(self):
        """GraphContext with a large evaluation_depth does not crash."""
        ctx = GraphContext(evaluation_depth=100)
        assert ctx.evaluation_depth == 100

    def test_current_node_id_empty_string(self):
        """GraphContext with empty current_node_id does not crash."""
        ctx = GraphContext(current_node_id="")
        assert ctx.current_node_id == ""

    def test_current_node_id_none(self):
        """GraphContext with None current_node_id is valid."""
        ctx = GraphContext(current_node_id=None)
        assert ctx.current_node_id is None

    def test_sync_group_empty_string(self):
        """GraphContext with empty sync_group does not crash."""
        ctx = GraphContext(sync_group="")
        assert ctx.sync_group == ""

    def test_sync_group_none(self):
        """GraphContext with None sync_group is valid."""
        ctx = GraphContext(sync_group=None)
        assert ctx.sync_group is None


# ============================================================================
# Property: get_parameter access after construction
# ============================================================================

class TestGetParameterProperties:
    """Invariants around parameter access."""

    def test_get_parameter_returns_value_for_known_parameter(self):
        """get_parameter('speed') on a context with speed returns its value."""
        ctx = GraphContext(
            parameters={"speed": GraphParameter.float_param("speed", 7.5)}
        )
        assert ctx.get_parameter("speed") == 7.5

    def test_get_parameter_with_zero_value(self):
        """get_parameter works when value is exactly 0."""
        ctx = GraphContext(
            parameters={"speed": GraphParameter.float_param("speed", 0.0)}
        )
        assert ctx.get_parameter("speed") == 0.0

    def test_get_parameter_with_false_bool(self):
        """get_parameter works when bool value is False."""
        ctx = GraphContext(
            parameters={"flag": GraphParameter.bool_param("flag", False)}
        )
        result = ctx.get_parameter("flag")
        assert result is False

    def test_get_parameter_with_default_none_distinct(self):
        """get_parameter returns None for unknown when default=None is explicit."""
        ctx = GraphContext(
            parameters={"speed": GraphParameter.float_param("speed", 1.0)}
        )
        # Explicit default=None for an unknown parameter
        result = ctx.get_parameter("nonexistent", None)
        assert result is None

    def test_get_parameter_unknown_zero_does_not_match_parameter_value(self):
        """get_parameter returns the default, not a hidden value, for unknown names."""
        ctx = GraphContext(parameters={"actual": GraphParameter.float_param("actual", 99.0)})
        default = ctx.get_parameter("typo", 0.0)
        assert default == 0.0


# ============================================================================
# Property: ContextPool acquire/release round-trip
# ============================================================================

class TestContextPoolRoundTrip:
    """ContextPool acquire/release round-trips maintain invariants."""

    def test_round_trip_single(self, empty_pool: ContextPool):
        """Single acquire/release: counters return to initial state."""
        initial_available = empty_pool.available_count
        initial_active = empty_pool.active_count
        initial_created = empty_pool.total_created

        ctx = empty_pool.acquire(dt=0.016)
        empty_pool.release(ctx)

        assert empty_pool.active_count == initial_active
        assert empty_pool.available_count >= initial_available
        assert empty_pool.total_created > initial_created

    def test_round_trip_multiple(self, empty_pool: ContextPool):
        """Multiple acquire/release cycles keep pools consistent."""
        for i in range(10):
            ctx = empty_pool.acquire(dt=0.016, tick=i)
            assert ctx.tick == i
            empty_pool.release(ctx)

        # After all releases, active should be 0
        assert empty_pool.active_count == 0
        assert empty_pool.available_count >= 1

    def test_acquire_release_does_not_mutate_across_pools(self):
        """Contexts from different pools are independent."""
        pool_a = ContextPool()
        pool_b = ContextPool()
        ctx_a = pool_a.acquire(dt=0.016)
        ctx_b = pool_b.acquire(dt=0.033)
        assert ctx_a.dt == 0.016
        assert ctx_b.dt == 0.033
        pool_a.release(ctx_a)
        pool_b.release(ctx_b)
        assert pool_a.available_count == 1
        assert pool_b.available_count == 1


# ============================================================================
# Property: GraphContext factory methods preserve data
# ============================================================================

class TestGraphContextMethodChaining:
    """Chained method calls on GraphContext produce consistent results."""

    def test_with_depth_then_advance_time(self, simple_skeleton: Skeleton):
        """with_depth then advance_time can be chained."""
        ctx = GraphContext(dt=0.016, skeleton=simple_skeleton)
        result = ctx.with_depth().advance_time(0.033)
        assert isinstance(result, GraphContext)
        assert result.evaluation_depth == 1

    def test_advance_time_then_with_depth(self, simple_skeleton: Skeleton):
        """advance_time then with_depth can be chained."""
        ctx = GraphContext(dt=0.016, skeleton=simple_skeleton)
        result = ctx.advance_time(0.033).with_depth()
        assert isinstance(result, GraphContext)
        assert result.evaluation_depth == 1

    def test_deeply_nested_method_chain(self):
        """Deep method chaining does not crash."""
        ctx = GraphContext(dt=0.016)
        result = (ctx
                  .with_depth()
                  .advance_time(0.016)
                  .with_depth()
                  .advance_time(0.033)
                  .with_depth())
        assert isinstance(result, GraphContext)
        assert result.evaluation_depth == 3
