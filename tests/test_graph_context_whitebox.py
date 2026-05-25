"""
Whitebox tests for GraphContext and ContextPool (T-AG-1.6).

Covers every field, method, and lifecycle path of:

  GraphContext
    - Default field values
    - get_parameter / get_parameter_float / get_parameter_int / get_parameter_bool
    - get_bone_mask
    - advance_time (all fields, reference sharing)
    - with_depth (all fields, cache preservation)

  ContextPool
    - acquire (empty pool path)
    - acquire (reuse path, full field reset)
    - release
    - active_count / available_count / total_created invariants
    - multiple acquire/release cycles
"""

from __future__ import annotations

import pytest

from engine.animation.graph.animation_graph import (
    BoneMask,
    ContextPool,
    GraphContext,
    GraphParameter,
    ParameterType,
    Pose,
    Skeleton,
)


# =============================================================================
# COVERAGE PLAN
# =============================================================================
#
# GraphContext:
#   - Default field values:                    test_default_fields
#   - All-args constructor:                    test_custom_fields
#   - get_parameter (existing / missing / default): 3 tests
#   - get_parameter_float (existing / missing):     2 tests
#   - get_parameter_int (existing / missing):       2 tests
#   - get_parameter_bool (existing / missing):      2 tests
#   - get_bone_mask (existing / missing):           2 tests
#   - advance_time (time, tick, dt, shared refs):   2 tests
#   - with_depth (depth increment, shared refs):    2 tests
#   - _node_results preservation across both:       1 test
#
# ContextPool:
#   - acquire creates new when pool empty:   test_acquire_creates_on_empty
#   - acquire sets all supplied fields:       test_acquire_sets_all_fields
#   - release returns to pool:               test_release_returns_to_pool
#   - acquire reuses available context:      test_acquire_reuses
#   - acquire resets stale state:            test_acquire_resets_stale_state
#   - acquire resets to fresh defaults:      test_acquire_resets_to_defaults
#   - active_count inc/dec:                  test_active_count_lifecycle
#   - available_count reflects pool:         test_available_count
#   - total_created monotonic:               test_total_created_monotonic
#   - multiple cycles no allocation growth:   test_multiple_cycles_no_growth

# =============================================================================
# 1 -- GraphContext: FIELD DEFAULTS
# =============================================================================


class TestGraphContextDefaults:
    """Every field of GraphContext must have a sensible default."""

    def test_default_fields(self) -> None:
        ctx = GraphContext()
        assert ctx.parameters == {}
        assert ctx.dt == 0.0
        assert ctx.skeleton is None
        assert ctx.bone_masks == {}
        assert ctx.normalized_time == 0.0
        assert ctx.sync_group is None
        assert ctx.layer_weight == 1.0
        assert ctx.current_time == 0.0
        assert ctx.tick == 0
        assert ctx.current_node_id is None
        assert ctx.evaluation_depth == 0
        assert ctx._node_results is None

    def test_custom_fields(self) -> None:
        params = {"speed": GraphParameter.float_param("speed", 1.0)}
        skel = Skeleton()
        skel.add_bone("Root")
        masks = {"full": BoneMask.full(skel)}

        ctx = GraphContext(
            parameters=params,
            dt=0.033,
            skeleton=skel,
            bone_masks=masks,
            normalized_time=0.5,
            sync_group="locomotion",
            layer_weight=0.8,
            current_time=12.0,
            tick=60,
            current_node_id="blend_node",
            evaluation_depth=3,
        )

        assert ctx.parameters is params
        assert ctx.dt == 0.033
        assert ctx.skeleton is skel
        assert ctx.bone_masks is masks
        assert ctx.normalized_time == 0.5
        assert ctx.sync_group == "locomotion"
        assert ctx.layer_weight == 0.8
        assert ctx.current_time == 12.0
        assert ctx.tick == 60
        assert ctx.current_node_id == "blend_node"
        assert ctx.evaluation_depth == 3
        assert ctx._node_results is None


# =============================================================================
# 2 -- GraphContext: GET_PARAMETER
# =============================================================================


class TestGraphContextGetParameter:
    """Exercises get_parameter and its typed helpers."""

    def test_get_parameter_exists(self) -> None:
        p = GraphParameter.float_param("speed", 42.0)
        ctx = GraphContext(parameters={"speed": p})
        assert ctx.get_parameter("speed") == 42.0

    def test_get_parameter_missing(self) -> None:
        ctx = GraphContext()
        assert ctx.get_parameter("speed") is None

    def test_get_parameter_missing_with_default(self) -> None:
        ctx = GraphContext()
        assert ctx.get_parameter("speed", default=10.0) == 10.0

    def test_get_parameter_float_exists(self) -> None:
        p = GraphParameter.float_param("speed", 3.5)
        ctx = GraphContext(parameters={"speed": p})
        value = ctx.get_parameter_float("speed")
        assert isinstance(value, float)
        assert value == 3.5

    def test_get_parameter_float_missing(self) -> None:
        ctx = GraphContext()
        value = ctx.get_parameter_float("speed")
        assert isinstance(value, float)
        assert value == 0.0

    def test_get_parameter_float_missing_custom_default(self) -> None:
        ctx = GraphContext()
        value = ctx.get_parameter_float("speed", default=7.5)
        assert isinstance(value, float)
        assert value == 7.5

    def test_get_parameter_int_exists(self) -> None:
        p = GraphParameter.int_param("frame", 7)
        ctx = GraphContext(parameters={"frame": p})
        value = ctx.get_parameter_int("frame")
        assert isinstance(value, int)
        assert value == 7

    def test_get_parameter_int_missing(self) -> None:
        ctx = GraphContext()
        value = ctx.get_parameter_int("frame")
        assert isinstance(value, int)
        assert value == 0

    def test_get_parameter_int_missing_custom_default(self) -> None:
        ctx = GraphContext()
        value = ctx.get_parameter_int("frame", default=99)
        assert isinstance(value, int)
        assert value == 99

    def test_get_parameter_bool_exists_true(self) -> None:
        p = GraphParameter.bool_param("is_running", True)
        ctx = GraphContext(parameters={"is_running": p})
        value = ctx.get_parameter_bool("is_running")
        assert isinstance(value, bool)
        assert value is True

    def test_get_parameter_bool_exists_false(self) -> None:
        p = GraphParameter.bool_param("is_running", False)
        ctx = GraphContext(parameters={"is_running": p})
        value = ctx.get_parameter_bool("is_running")
        assert isinstance(value, bool)
        assert value is False

    def test_get_parameter_bool_missing(self) -> None:
        ctx = GraphContext()
        value = ctx.get_parameter_bool("is_running")
        assert isinstance(value, bool)
        assert value is False

    def test_get_parameter_bool_missing_custom_default(self) -> None:
        ctx = GraphContext()
        value = ctx.get_parameter_bool("is_running", default=True)
        assert isinstance(value, bool)
        assert value is True


# =============================================================================
# 3 -- GraphContext: GET_BONE_MASK
# =============================================================================


class TestGraphContextGetBoneMask:
    """Exercises get_bone_mask."""

    def test_get_bone_mask_exists(self) -> None:
        skel = Skeleton()
        skel.add_bone("Root")
        skel.add_bone("Spine")
        mask = BoneMask.full(skel, "full_body")
        ctx = GraphContext(bone_masks={"full_body": mask})
        assert ctx.get_bone_mask("full_body") is mask

    def test_get_bone_mask_missing(self) -> None:
        ctx = GraphContext()
        assert ctx.get_bone_mask("nonexistent") is None

    def test_get_bone_mask_from_empty_dict(self) -> None:
        ctx = GraphContext(bone_masks={})
        assert ctx.get_bone_mask("anything") is None


# =============================================================================
# 4 -- GraphContext: ADVANCE_TIME
# =============================================================================


class TestGraphContextAdvanceTime:
    """Exercises advance_time -- all fields, reference sharing."""

    def test_advance_time_fields(self) -> None:
        params = {"x": GraphParameter.float_param("x", 1.0)}
        skel = Skeleton()
        skel.add_bone("Root")
        masks = {"full": BoneMask.full(skel)}
        cache: dict[str, Pose] = {"n": Pose()}

        ctx = GraphContext(
            parameters=params,
            skeleton=skel,
            bone_masks=masks,
            normalized_time=0.3,
            sync_group="run",
            layer_weight=0.5,
            current_time=10.0,
            tick=5,
            current_node_id="node_a",
            evaluation_depth=2,
        )
        ctx._node_results = cache

        advanced = ctx.advance_time(0.016)

        # Fields that change
        assert advanced.current_time == 10.016
        assert advanced.tick == 6
        assert advanced.dt == 0.016

        # All other fields shared by reference
        assert advanced.parameters is params
        assert advanced.skeleton is skel
        assert advanced.bone_masks is masks
        assert advanced.normalized_time == 0.3
        assert advanced.sync_group == "run"
        assert advanced.layer_weight == 0.5
        assert advanced.current_node_id == "node_a"
        assert advanced.evaluation_depth == 2
        assert advanced._node_results is cache

    def test_advance_time_with_zero_dt(self) -> None:
        """Zero delta-time should not change current_time."""
        ctx = GraphContext(current_time=5.0, tick=10)
        advanced = ctx.advance_time(0.0)
        assert advanced.current_time == 5.0
        assert advanced.tick == 11
        assert advanced.dt == 0.0

    def test_advance_time_preserves_skeleton_reference(self) -> None:
        skel = Skeleton()
        skel.add_bone("Root")
        ctx = GraphContext(skeleton=skel)
        advanced = ctx.advance_time(0.1)
        assert advanced.skeleton is skel

    def test_advance_time_preserves_parameters_reference(self) -> None:
        params = {"p": GraphParameter.float_param("p", 1.0)}
        ctx = GraphContext(parameters=params)
        advanced = ctx.advance_time(0.1)
        assert advanced.parameters is params

    def test_advance_time_original_unchanged(self) -> None:
        """Original context is not mutated by advance_time."""
        ctx = GraphContext(current_time=10.0, tick=5, dt=0.0)
        _ = ctx.advance_time(0.033)
        assert ctx.current_time == 10.0  # unchanged
        assert ctx.tick == 5             # unchanged
        assert ctx.dt == 0.0             # unchanged


# =============================================================================
# 5 -- GraphContext: WITH_DEPTH
# =============================================================================


class TestGraphContextWithDepth:
    """Exercises with_depth -- depth increment, reference sharing."""

    def test_with_depth_increments_depth(self) -> None:
        ctx = GraphContext(evaluation_depth=0)
        deeper = ctx.with_depth()
        assert deeper.evaluation_depth == 1
        deeper2 = deeper.with_depth()
        assert deeper2.evaluation_depth == 2

    def test_with_depth_preserves_all_fields(self) -> None:
        params = {"x": GraphParameter.float_param("x", 1.0)}
        skel = Skeleton()
        skel.add_bone("Root")
        masks = {"full": BoneMask.full(skel)}
        cache: dict[str, Pose] = {"n": Pose()}

        ctx = GraphContext(
            parameters=params,
            dt=0.033,
            skeleton=skel,
            bone_masks=masks,
            normalized_time=0.5,
            sync_group="idle",
            layer_weight=0.7,
            current_time=15.0,
            tick=30,
            current_node_id="node_b",
            evaluation_depth=4,
        )
        ctx._node_results = cache

        deeper = ctx.with_depth()

        assert deeper.evaluation_depth == 5
        assert deeper.parameters is params
        assert deeper.dt == 0.033
        assert deeper.skeleton is skel
        assert deeper.bone_masks is masks
        assert deeper.normalized_time == 0.5
        assert deeper.sync_group == "idle"
        assert deeper.layer_weight == 0.7
        assert deeper.current_time == 15.0
        assert deeper.tick == 30
        assert deeper.current_node_id == "node_b"
        assert deeper._node_results is cache

    def test_with_depth_preserves_node_results(self) -> None:
        cache: dict[str, Pose] = {"a": Pose()}
        ctx = GraphContext()
        ctx._node_results = cache

        deeper = ctx.with_depth()

        assert deeper._node_results is cache

    def test_with_depth_shared_node_results_with_advance_time(self) -> None:
        """Both with_depth and advance_time share the same _node_results ref."""
        cache: dict[str, Pose] = {"n": Pose()}
        ctx = GraphContext()
        ctx._node_results = cache

        d1 = ctx.with_depth()
        d2 = ctx.advance_time(0.0)

        assert d1._node_results is cache
        assert d2._node_results is cache
        assert d1._node_results is d2._node_results

    def test_with_depth_preserves_skeleton_reference(self) -> None:
        skel = Skeleton()
        skel.add_bone("Root")
        ctx = GraphContext(skeleton=skel)
        deeper = ctx.with_depth()
        assert deeper.skeleton is skel

    def test_with_depth_preserves_parameters_reference(self) -> None:
        params = {"p": GraphParameter.float_param("p", 1.0)}
        ctx = GraphContext(parameters=params)
        deeper = ctx.with_depth()
        assert deeper.parameters is params

    def test_with_depth_original_unchanged(self) -> None:
        """Original context depth is not mutated by with_depth."""
        ctx = GraphContext(evaluation_depth=3)
        _ = ctx.with_depth()
        assert ctx.evaluation_depth == 3  # unchanged


# =============================================================================
# 6 -- CONTEXTPOOL: ACQUIRE / RELEASE / REUSE
# =============================================================================


class TestContextPoolAcquireRelease:
    """Exercises all ContextPool code paths."""

    def test_acquire_creates_on_empty(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire(dt=0.016, current_time=1.0, tick=1)

        assert isinstance(ctx, GraphContext)
        assert pool.total_created == 1
        assert pool.active_count == 1
        assert pool.available_count == 0

    def test_acquire_sets_all_supplied_fields(self) -> None:
        pool = ContextPool()
        params = {"s": GraphParameter.float_param("s", 1.0)}
        skel = Skeleton()
        skel.add_bone("Hip")
        masks = {"upper": BoneMask("upper")}

        ctx = pool.acquire(
            parameters=params,
            dt=0.033,
            skeleton=skel,
            bone_masks=masks,
            normalized_time=0.7,
            sync_group="walk",
            layer_weight=0.9,
            current_time=20.0,
            tick=100,
        )

        assert ctx.parameters is params
        assert ctx.dt == 0.033
        assert ctx.skeleton is skel
        assert ctx.bone_masks is masks
        assert ctx.normalized_time == 0.7
        assert ctx.sync_group == "walk"
        assert ctx.layer_weight == 0.9
        assert ctx.current_time == 20.0
        assert ctx.tick == 100
        # Fields reset to defaults
        assert ctx.current_node_id is None
        assert ctx.evaluation_depth == 0
        assert ctx._node_results is None

    def test_acquire_with_no_args_sets_defaults(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire()

        assert ctx.parameters == {}
        assert ctx.dt == 0.0
        assert ctx.skeleton is None
        assert ctx.bone_masks == {}
        assert ctx.normalized_time == 0.0
        assert ctx.sync_group is None
        assert ctx.layer_weight == 1.0
        assert ctx.current_time == 0.0
        assert ctx.tick == 0
        assert ctx.current_node_id is None
        assert ctx.evaluation_depth == 0
        assert ctx._node_results is None

    def test_release_returns_to_pool(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire()
        pool.release(ctx)

        assert pool.active_count == 0
        assert pool.available_count == 1
        assert pool.total_created == 1

    def test_acquire_reuses_available_context(self) -> None:
        pool = ContextPool()
        ctx1 = pool.acquire()
        pool.release(ctx1)

        ctx2 = pool.acquire()

        # Same object is reused
        assert ctx2 is ctx1
        assert pool.active_count == 1
        assert pool.available_count == 0
        assert pool.total_created == 1  # No new allocation

    def test_acquire_resets_stale_current_node_id(self) -> None:
        """Re-acquired context must not retain stale current_node_id."""
        pool = ContextPool()
        ctx = pool.acquire()

        # Simulate previous use setting fields
        ctx.current_node_id = "old_node"
        ctx.evaluation_depth = 42

        pool.release(ctx)

        ctx2 = pool.acquire()

        assert ctx2.current_node_id is None
        assert ctx2.evaluation_depth == 0
        assert ctx2._node_results is None

    def test_acquire_resets_stale_parameters(self) -> None:
        """Re-acquired context with no args must clear old parameters."""
        pool = ContextPool()
        params = {"old": GraphParameter.float_param("old", 99.0)}
        ctx = pool.acquire(parameters=params)
        pool.release(ctx)

        ctx2 = pool.acquire()  # no parameters passed

        assert ctx2.parameters == {}

    def test_acquire_overwrites_fields_with_new_values(self) -> None:
        """Re-acquired context must use the newly supplied values, not the old ones."""
        pool = ContextPool()
        params_a = {"a": GraphParameter.float_param("a", 1.0)}
        ctx = pool.acquire(parameters=params_a, dt=0.016, current_time=5.0, tick=10)
        pool.release(ctx)

        params_b = {"b": GraphParameter.float_param("b", 2.0)}
        ctx2 = pool.acquire(
            parameters=params_b,
            dt=0.033,
            current_time=15.0,
            tick=30,
            skeleton=None,
        )

        assert ctx2.parameters is params_b
        assert ctx2.dt == 0.033
        assert ctx2.current_time == 15.0
        assert ctx2.tick == 30
        assert ctx2.skeleton is None


# =============================================================================
# 7 -- CONTEXTPOOL: COUNTING INVARIANTS
# =============================================================================


class TestContextPoolCounters:
    """Exercises active_count, available_count, total_created invariants."""

    def test_active_count_lifecycle(self) -> None:
        pool = ContextPool()
        assert pool.active_count == 0

        c1 = pool.acquire()
        assert pool.active_count == 1

        c2 = pool.acquire()
        assert pool.active_count == 2

        pool.release(c1)
        assert pool.active_count == 1

        pool.release(c2)
        assert pool.active_count == 0

    def test_available_count_tracks_released_contexts(self) -> None:
        pool = ContextPool()
        assert pool.available_count == 0

        c1 = pool.acquire()
        assert pool.available_count == 0  # acquired, not available

        pool.release(c1)
        assert pool.available_count == 1

        c2 = pool.acquire()
        assert pool.available_count == 0  # reused, no longer available

        pool.release(c2)
        assert pool.available_count == 1  # one available again

    def test_total_created_only_increases_on_new_allocation(self) -> None:
        pool = ContextPool()
        assert pool.total_created == 0

        c1 = pool.acquire()
        assert pool.total_created == 1

        c2 = pool.acquire()
        assert pool.total_created == 2

        pool.release(c1)
        pool.release(c2)
        assert pool.total_created == 2  # no increase on release

        # Reuse -- no new creation
        _ = pool.acquire()
        assert pool.total_created == 2

        _ = pool.acquire()
        assert pool.total_created == 2

    def test_multiple_cycles_no_allocation_growth(self) -> None:
        """Multiple acquire/release cycles should not increase total_created."""
        pool = ContextPool()
        contexts = []

        # First wave: acquire 3
        for _ in range(3):
            contexts.append(pool.acquire())
        assert pool.total_created == 3
        assert pool.active_count == 3

        # Release all
        for c in contexts:
            pool.release(c)
        assert pool.available_count == 3
        assert pool.active_count == 0

        # Second wave: acquire 3 -- must reuse, not create new
        contexts2 = []
        for _ in range(3):
            contexts2.append(pool.acquire())
        assert pool.total_created == 3  # unchanged
        assert pool.active_count == 3

        # Third wave: release all and acquire again
        for c in contexts2:
            pool.release(c)
        contexts3 = []
        for _ in range(3):
            contexts3.append(pool.acquire())
        assert pool.total_created == 3  # still unchanged
        assert pool.active_count == 3

    def test_total_created_equals_available_plus_active(self) -> None:
        """total_created must always equal available_count + active_count."""
        pool = ContextPool()
        assert pool.total_created == pool.available_count + pool.active_count

        c1 = pool.acquire()
        c2 = pool.acquire()
        c3 = pool.acquire()
        assert pool.total_created == pool.available_count + pool.active_count
        assert pool.total_created == 3

        pool.release(c1)
        assert pool.total_created == pool.available_count + pool.active_count

        pool.release(c2)
        pool.release(c3)
        assert pool.total_created == pool.available_count + pool.active_count
        assert pool.total_created == 3


# =============================================================================
# 8 -- INTEGRATION: CONTEXTPOOL WITH WITH_DEPTH
# =============================================================================


class TestContextPoolIntegration:
    """Pool-acquired contexts work correctly with with_depth."""

    def test_pool_acquired_context_with_depth(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire(
            dt=0.016,
            current_time=10.0,
            tick=50,
            skeleton=None,
        )

        deeper = ctx.with_depth()

        assert deeper.evaluation_depth == 1
        # Preserve pool-set fields
        assert deeper.dt == 0.016
        assert deeper.current_time == 10.0
        assert deeper.tick == 50
        assert deeper._node_results is None

    def test_pool_acquired_context_advance_time(self) -> None:
        pool = ContextPool()
        ctx = pool.acquire(
            dt=0.0,
            current_time=5.0,
            tick=10,
        )

        advanced = ctx.advance_time(0.033)

        assert advanced.current_time == 5.033
        assert advanced.tick == 11
        assert advanced.dt == 0.033
