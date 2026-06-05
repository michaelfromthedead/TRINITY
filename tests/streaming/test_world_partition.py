"""
Tests for world partition streaming system (T-ENV-3.7).

Comprehensive tests covering:
- All decorator configurations
- State machine transitions
- Async loading pipeline
- Priority computation accuracy
- Cell activation/deactivation

Total: 60+ tests
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pytest

from engine.streaming.world_partition import (
    chunk,
    streamable,
    loading_priority,
    unloadable,
    ChunkConfig,
    StreamableConfig,
    LoadingPriorityConfig,
    UnloadableConfig,
    WorldChunk,
    StreamPriority,
    get_chunk_decorators,
    validate_chunk_class,
    is_fully_decorated_chunk,
)
from engine.streaming.cell_state_machine import (
    CellState,
    CellStateMachine,
    CellStateError,
    StateTransition,
    VALID_TRANSITIONS,
    get_valid_transitions_from,
    is_valid_transition,
)
from engine.streaming.async_loader import (
    LoadStage,
    LoadRequest,
    LoadResult,
    LoadError,
    TerrainLoader,
    HeightDataLoader,
    GPUUploader,
    AsyncLoadPipeline,
    LoadPipelineConfig,
    TerrainData,
    HeightData,
    GPUCellData,
    CellLoadData,
)
from engine.streaming.priority_system import (
    PriorityComputer,
    PriorityConfig,
    PriorityFactors,
    CellPriority,
    StreamingSource,
    CellActivationTrigger,
    ActivationEvent,
    ActivationType,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_chunk_config() -> ChunkConfig:
    """Basic chunk configuration."""
    return ChunkConfig(size=256.0, overlap=16.0)


@pytest.fixture
def state_machine() -> CellStateMachine:
    """Fresh state machine instance."""
    return CellStateMachine(cell_x=0, cell_y=0)


@pytest.fixture
def pipeline_config() -> LoadPipelineConfig:
    """Pipeline configuration for tests."""
    return LoadPipelineConfig(
        max_concurrent_loads=2,
        max_pending_requests=10,
        terrain_cache_enabled=True,
    )


@pytest.fixture
def priority_config() -> PriorityConfig:
    """Priority configuration for tests."""
    return PriorityConfig(
        max_priority_distance=100.0,
        min_priority_distance=2000.0,
    )


# =============================================================================
# CHUNK DECORATOR TESTS
# =============================================================================


class TestChunkDecorator:
    """Tests for @chunk decorator."""

    def test_chunk_basic_config(self) -> None:
        """@chunk applies default configuration."""
        @chunk(size=512.0)
        @dataclass
        class TestChunk:
            data: bytes = b""

        assert hasattr(TestChunk, "_chunk_config")
        assert TestChunk._chunk_config.size == 512.0
        assert TestChunk._chunk_config.overlap == 16.0  # default

    def test_chunk_with_overlap(self) -> None:
        """@chunk accepts overlap parameter."""
        @chunk(size=256.0, overlap=32.0)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._chunk_config.overlap == 32.0

    def test_chunk_with_lod_levels(self) -> None:
        """@chunk accepts LOD levels parameter."""
        @chunk(size=256.0, lod_levels=6)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._chunk_config.lod_levels == 6

    def test_chunk_with_lod_distances(self) -> None:
        """@chunk accepts custom LOD distances."""
        distances = [100.0, 300.0, 600.0]
        @chunk(size=256.0, lod_levels=3, lod_distances=distances)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._chunk_config.lod_distances == distances

    def test_chunk_invalid_size_raises(self) -> None:
        """@chunk raises ValueError for invalid size."""
        with pytest.raises(ValueError, match="positive"):
            @chunk(size=0.0)
            @dataclass
            class TestChunk:
                pass

    def test_chunk_negative_size_raises(self) -> None:
        """@chunk raises ValueError for negative size."""
        with pytest.raises(ValueError, match="positive"):
            @chunk(size=-100.0)
            @dataclass
            class TestChunk:
                pass

    def test_chunk_negative_overlap_raises(self) -> None:
        """@chunk raises ValueError for negative overlap."""
        with pytest.raises(ValueError, match="non-negative"):
            @chunk(size=256.0, overlap=-1.0)
            @dataclass
            class TestChunk:
                pass

    def test_chunk_overlap_too_large_raises(self) -> None:
        """@chunk raises ValueError when overlap >= size/2."""
        with pytest.raises(ValueError, match="less than half"):
            @chunk(size=256.0, overlap=128.0)
            @dataclass
            class TestChunk:
                pass

    def test_chunk_invalid_lod_levels_raises(self) -> None:
        """@chunk raises ValueError for LOD levels < 1."""
        with pytest.raises(ValueError, match="at least 1"):
            @chunk(size=256.0, lod_levels=0)
            @dataclass
            class TestChunk:
                pass

    def test_chunk_lod_distance_count_mismatch_raises(self) -> None:
        """@chunk raises ValueError when lod_distances count != lod_levels."""
        with pytest.raises(ValueError, match="must match"):
            @chunk(size=256.0, lod_levels=3, lod_distances=[100.0, 200.0])
            @dataclass
            class TestChunk:
                pass

    def test_chunk_provides_accessor_methods(self) -> None:
        """@chunk adds accessor methods to class."""
        @chunk(size=512.0, overlap=24.0)
        @dataclass
        class TestChunk:
            pass

        instance = TestChunk()
        assert instance.cell_size == 512.0
        assert instance.cell_overlap == 24.0


class TestStreamableDecorator:
    """Tests for @streamable decorator."""

    def test_streamable_default_priority(self) -> None:
        """@streamable uses NORMAL priority by default."""
        @streamable()
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._streamable_config.priority == StreamPriority.NORMAL

    def test_streamable_high_priority(self) -> None:
        """@streamable accepts HIGH priority."""
        @streamable(priority=StreamPriority.HIGH)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._streamable_config.priority == StreamPriority.HIGH

    def test_streamable_string_priority(self) -> None:
        """@streamable accepts string priority."""
        @streamable(priority="critical")
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._streamable_config.priority == StreamPriority.CRITICAL

    def test_streamable_keep_loaded_true(self) -> None:
        """@streamable accepts keep_loaded flag."""
        @streamable(keep_loaded=True)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._streamable_config.keep_loaded is True

    def test_streamable_preload_distance(self) -> None:
        """@streamable accepts preload_distance."""
        @streamable(preload_distance=100.0)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._streamable_config.preload_distance == 100.0

    def test_streamable_invalid_priority_string_raises(self) -> None:
        """@streamable raises ValueError for invalid priority string."""
        with pytest.raises(ValueError, match="Invalid priority"):
            @streamable(priority="invalid")
            @dataclass
            class TestChunk:
                pass

    def test_streamable_negative_preload_raises(self) -> None:
        """@streamable raises ValueError for negative preload_distance."""
        with pytest.raises(ValueError, match="non-negative"):
            @streamable(preload_distance=-1.0)
            @dataclass
            class TestChunk:
                pass

    def test_streamable_provides_accessor_methods(self) -> None:
        """@streamable adds accessor methods."""
        @streamable(priority=StreamPriority.HIGH, keep_loaded=True)
        @dataclass
        class TestChunk:
            pass

        instance = TestChunk()
        assert instance.stream_priority == StreamPriority.HIGH
        assert instance.should_keep_loaded is True


class TestLoadingPriorityDecorator:
    """Tests for @loading_priority decorator."""

    def test_loading_priority_defaults(self) -> None:
        """@loading_priority uses defaults correctly."""
        @loading_priority()
        @dataclass
        class TestChunk:
            pass

        config = TestChunk._loading_priority_config
        assert config.visibility_weight == 1.0
        assert config.player_velocity_weight == 1.0

    def test_loading_priority_custom_weights(self) -> None:
        """@loading_priority accepts custom weights."""
        @loading_priority(visibility_weight=2.0, player_velocity_weight=0.5)
        @dataclass
        class TestChunk:
            pass

        config = TestChunk._loading_priority_config
        assert config.visibility_weight == 2.0
        assert config.player_velocity_weight == 0.5

    def test_loading_priority_lod_bonus(self) -> None:
        """@loading_priority accepts lod_bonus."""
        @loading_priority(lod_bonus=1.0)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._loading_priority_config.lod_bonus == 1.0

    def test_loading_priority_negative_visibility_raises(self) -> None:
        """@loading_priority raises for negative visibility_weight."""
        with pytest.raises(ValueError, match="non-negative"):
            @loading_priority(visibility_weight=-1.0)
            @dataclass
            class TestChunk:
                pass

    def test_loading_priority_negative_velocity_raises(self) -> None:
        """@loading_priority raises for negative player_velocity_weight."""
        with pytest.raises(ValueError, match="non-negative"):
            @loading_priority(player_velocity_weight=-1.0)
            @dataclass
            class TestChunk:
                pass

    def test_loading_priority_zero_falloff_raises(self) -> None:
        """@loading_priority raises for distance_falloff <= 0."""
        with pytest.raises(ValueError, match="positive"):
            @loading_priority(distance_falloff=0.0)
            @dataclass
            class TestChunk:
                pass


class TestUnloadableDecorator:
    """Tests for @unloadable decorator."""

    def test_unloadable_defaults(self) -> None:
        """@unloadable uses defaults correctly."""
        @unloadable()
        @dataclass
        class TestChunk:
            pass

        config = TestChunk._unloadable_config
        assert config.min_age == 60.0
        assert config.save_state is True

    def test_unloadable_custom_min_age(self) -> None:
        """@unloadable accepts custom min_age."""
        @unloadable(min_age=30.0)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._unloadable_config.min_age == 30.0

    def test_unloadable_save_state_false(self) -> None:
        """@unloadable accepts save_state=False."""
        @unloadable(save_state=False)
        @dataclass
        class TestChunk:
            pass

        assert TestChunk._unloadable_config.save_state is False

    def test_unloadable_zero_min_age_raises(self) -> None:
        """@unloadable raises for min_age <= 0."""
        with pytest.raises(ValueError, match="positive"):
            @unloadable(min_age=0.0)
            @dataclass
            class TestChunk:
                pass

    def test_unloadable_negative_min_age_raises(self) -> None:
        """@unloadable raises for negative min_age."""
        with pytest.raises(ValueError, match="positive"):
            @unloadable(min_age=-10.0)
            @dataclass
            class TestChunk:
                pass


class TestDecoratorCombinations:
    """Tests for combining multiple decorators."""

    def test_all_decorators_combined(self) -> None:
        """All four decorators can be combined."""
        @chunk(size=512.0, overlap=32.0)
        @streamable(priority=StreamPriority.HIGH)
        @loading_priority(visibility_weight=2.0)
        @unloadable(min_age=45.0)
        @dataclass
        class FullChunk:
            data: bytes = b""

        assert FullChunk._chunk_config.size == 512.0
        assert FullChunk._streamable_config.priority == StreamPriority.HIGH
        assert FullChunk._loading_priority_config.visibility_weight == 2.0
        assert FullChunk._unloadable_config.min_age == 45.0

    def test_get_chunk_decorators(self) -> None:
        """get_chunk_decorators returns applied decorators."""
        @chunk(size=256.0)
        @streamable()
        @dataclass
        class PartialChunk:
            pass

        decorators = get_chunk_decorators(PartialChunk)
        assert decorators["chunk"] is True
        assert decorators["streamable"] is True
        assert decorators["loading_priority"] is False
        assert decorators["unloadable"] is False

    def test_validate_chunk_class_missing(self) -> None:
        """validate_chunk_class returns missing decorators."""
        @chunk(size=256.0)
        @dataclass
        class PartialChunk:
            pass

        missing = validate_chunk_class(PartialChunk)
        assert "streamable" in missing
        assert "loading_priority" in missing
        assert "unloadable" in missing
        assert "chunk" not in missing

    def test_is_fully_decorated_chunk(self) -> None:
        """is_fully_decorated_chunk returns correct result."""
        @chunk(size=256.0)
        @streamable()
        @loading_priority()
        @unloadable()
        @dataclass
        class FullChunk:
            pass

        @chunk(size=256.0)
        @dataclass
        class PartialChunk:
            pass

        assert is_fully_decorated_chunk(FullChunk) is True
        assert is_fully_decorated_chunk(PartialChunk) is False


# =============================================================================
# STATE MACHINE TESTS
# =============================================================================


class TestCellState:
    """Tests for CellState enum."""

    def test_state_is_in_memory(self) -> None:
        """is_in_memory returns True for LOADED and ACTIVATED."""
        assert CellState.LOADED.is_in_memory is True
        assert CellState.ACTIVATED.is_in_memory is True
        assert CellState.UNLOADED.is_in_memory is False
        assert CellState.LOADING.is_in_memory is False

    def test_state_is_transitioning(self) -> None:
        """is_transitioning returns True for LOADING and UNLOADING."""
        assert CellState.LOADING.is_transitioning is True
        assert CellState.UNLOADING.is_transitioning is True
        assert CellState.LOADED.is_transitioning is False
        assert CellState.ACTIVATED.is_transitioning is False

    def test_state_can_tick(self) -> None:
        """can_tick returns True only for ACTIVATED."""
        assert CellState.ACTIVATED.can_tick is True
        assert CellState.LOADED.can_tick is False
        assert CellState.LOADING.can_tick is False


class TestCellStateMachine:
    """Tests for CellStateMachine."""

    def test_initial_state_is_unloaded(self, state_machine: CellStateMachine) -> None:
        """State machine starts in UNLOADED state."""
        assert state_machine.state == CellState.UNLOADED

    def test_begin_load_transitions_to_loading(
        self, state_machine: CellStateMachine
    ) -> None:
        """begin_load transitions from UNLOADED to LOADING."""
        assert state_machine.begin_load() is True
        assert state_machine.state == CellState.LOADING

    def test_complete_load_transitions_to_loaded(
        self, state_machine: CellStateMachine
    ) -> None:
        """complete_load transitions from LOADING to LOADED."""
        state_machine.begin_load()
        assert state_machine.complete_load() is True
        assert state_machine.state == CellState.LOADED

    def test_activate_transitions_to_activated(
        self, state_machine: CellStateMachine
    ) -> None:
        """activate transitions from LOADED to ACTIVATED."""
        state_machine.begin_load()
        state_machine.complete_load()
        assert state_machine.activate() is True
        assert state_machine.state == CellState.ACTIVATED

    def test_deactivate_transitions_to_loaded(
        self, state_machine: CellStateMachine
    ) -> None:
        """deactivate transitions from ACTIVATED to LOADED."""
        state_machine.begin_load()
        state_machine.complete_load()
        state_machine.activate()
        assert state_machine.deactivate() is True
        assert state_machine.state == CellState.LOADED

    def test_begin_unload_from_loaded(
        self, state_machine: CellStateMachine
    ) -> None:
        """begin_unload transitions from LOADED to UNLOADING."""
        state_machine.begin_load()
        state_machine.complete_load()
        assert state_machine.begin_unload() is True
        assert state_machine.state == CellState.UNLOADING

    def test_begin_unload_from_activated(
        self, state_machine: CellStateMachine
    ) -> None:
        """begin_unload can transition from ACTIVATED to UNLOADING."""
        state_machine.begin_load()
        state_machine.complete_load()
        state_machine.activate()
        assert state_machine.begin_unload() is True
        assert state_machine.state == CellState.UNLOADING

    def test_complete_unload_transitions_to_unloaded(
        self, state_machine: CellStateMachine
    ) -> None:
        """complete_unload transitions from UNLOADING to UNLOADED."""
        state_machine.begin_load()
        state_machine.complete_load()
        state_machine.begin_unload()
        assert state_machine.complete_unload() is True
        assert state_machine.state == CellState.UNLOADED

    def test_cancel_load_transitions_to_unloaded(
        self, state_machine: CellStateMachine
    ) -> None:
        """cancel_load transitions from LOADING to UNLOADED."""
        state_machine.begin_load()
        assert state_machine.cancel_load() is True
        assert state_machine.state == CellState.UNLOADED

    def test_invalid_transition_raises_error(
        self, state_machine: CellStateMachine
    ) -> None:
        """Invalid transition raises CellStateError."""
        with pytest.raises(CellStateError):
            state_machine.complete_load()  # Can't complete_load from UNLOADED

    def test_invalid_transition_error_contains_info(
        self, state_machine: CellStateMachine
    ) -> None:
        """CellStateError contains state information."""
        try:
            state_machine.activate()  # Can't activate from UNLOADED
        except CellStateError as e:
            assert e.current_state == CellState.UNLOADED
            assert e.attempted_state == CellState.ACTIVATED

    def test_update_load_progress(
        self, state_machine: CellStateMachine
    ) -> None:
        """update_load_progress updates progress correctly."""
        state_machine.begin_load()
        state_machine.update_load_progress(0.5)
        assert state_machine.load_progress == 0.5

    def test_load_progress_clamps(
        self, state_machine: CellStateMachine
    ) -> None:
        """update_load_progress clamps to [0, 1]."""
        state_machine.begin_load()
        state_machine.update_load_progress(1.5)
        assert state_machine.load_progress == 1.0
        state_machine.update_load_progress(-0.5)
        assert state_machine.load_progress == 0.0

    def test_callbacks_on_state_enter(
        self, state_machine: CellStateMachine
    ) -> None:
        """Callbacks fire on state enter."""
        entered_states: List[CellState] = []

        def on_enter(m: CellStateMachine, old: CellState, new: CellState) -> None:
            entered_states.append(new)

        state_machine.on_state_enter(CellState.LOADING, on_enter)
        state_machine.begin_load()
        assert CellState.LOADING in entered_states

    def test_callbacks_on_state_exit(
        self, state_machine: CellStateMachine
    ) -> None:
        """Callbacks fire on state exit."""
        exited_states: List[CellState] = []

        def on_exit(m: CellStateMachine, old: CellState, new: CellState) -> None:
            exited_states.append(old)

        state_machine.on_state_exit(CellState.UNLOADED, on_exit)
        state_machine.begin_load()
        assert CellState.UNLOADED in exited_states

    def test_transition_history(
        self, state_machine: CellStateMachine
    ) -> None:
        """Transition history is recorded."""
        state_machine.begin_load()
        state_machine.complete_load()
        history = state_machine.get_transition_history()
        assert len(history) == 2
        assert history[0][1] == CellState.LOADING
        assert history[1][1] == CellState.LOADED

    def test_force_state_bypasses_validation(
        self, state_machine: CellStateMachine
    ) -> None:
        """force_state bypasses transition validation."""
        state_machine.force_state(CellState.ACTIVATED)
        assert state_machine.state == CellState.ACTIVATED

    def test_reset_returns_to_unloaded(
        self, state_machine: CellStateMachine
    ) -> None:
        """reset returns machine to initial state."""
        state_machine.begin_load()
        state_machine.complete_load()
        state_machine.reset()
        assert state_machine.state == CellState.UNLOADED
        assert state_machine.load_progress == 0.0


class TestValidTransitions:
    """Tests for transition validation functions."""

    def test_valid_transitions_count(self) -> None:
        """VALID_TRANSITIONS has expected count."""
        assert len(VALID_TRANSITIONS) == 8

    def test_is_valid_transition_unloaded_to_loading(self) -> None:
        """UNLOADED -> LOADING is valid."""
        assert is_valid_transition(CellState.UNLOADED, CellState.LOADING) is True

    def test_is_valid_transition_unloaded_to_loaded_invalid(self) -> None:
        """UNLOADED -> LOADED is invalid."""
        assert is_valid_transition(CellState.UNLOADED, CellState.LOADED) is False

    def test_get_valid_transitions_from_loaded(self) -> None:
        """get_valid_transitions_from LOADED returns correct states."""
        valid = get_valid_transitions_from(CellState.LOADED)
        assert CellState.ACTIVATED in valid
        assert CellState.UNLOADING in valid
        assert len(valid) == 2


# =============================================================================
# ASYNC LOADER TESTS
# =============================================================================


class TestLoadStage:
    """Tests for LoadStage enum."""

    def test_is_terminal(self) -> None:
        """is_terminal returns True for terminal stages."""
        assert LoadStage.COMPLETE.is_terminal is True
        assert LoadStage.FAILED.is_terminal is True
        assert LoadStage.CANCELLED.is_terminal is True
        assert LoadStage.TERRAIN.is_terminal is False

    def test_is_active(self) -> None:
        """is_active returns True for active stages."""
        assert LoadStage.TERRAIN.is_active is True
        assert LoadStage.HEIGHT_DATA.is_active is True
        assert LoadStage.GPU_UPLOAD.is_active is True
        assert LoadStage.PENDING.is_active is False
        assert LoadStage.COMPLETE.is_active is False

    def test_progress_fraction(self) -> None:
        """progress_fraction returns approximate progress."""
        assert LoadStage.PENDING.progress_fraction == 0.0
        assert LoadStage.TERRAIN.progress_fraction == pytest.approx(0.33)
        assert LoadStage.COMPLETE.progress_fraction == 1.0


class TestLoadRequest:
    """Tests for LoadRequest."""

    def test_request_creation(self) -> None:
        """LoadRequest creates with correct defaults."""
        request = LoadRequest(cell_x=5, cell_y=10)
        assert request.cell_id == (5, 10)
        assert request.stage == LoadStage.PENDING
        assert request.progress == 0.0
        assert request.cancelled is False

    def test_request_cancel(self) -> None:
        """cancel() sets cancelled flag and stage."""
        request = LoadRequest(cell_x=0, cell_y=0)
        request.cancel()
        assert request.cancelled is True
        assert request.stage == LoadStage.CANCELLED

    def test_request_update_progress(self) -> None:
        """update_progress updates stage and progress."""
        request = LoadRequest(cell_x=0, cell_y=0)
        request.update_progress(0.5, LoadStage.TERRAIN)
        assert request.progress == 0.5
        assert request.stage == LoadStage.TERRAIN


class TestLoadResult:
    """Tests for LoadResult."""

    def test_result_ok(self) -> None:
        """LoadResult.ok creates successful result."""
        result = LoadResult.ok(data="test_data", load_time_ms=100.0)
        assert result.success is True
        assert result.data == "test_data"
        assert result.error is None

    def test_result_fail(self) -> None:
        """LoadResult.fail creates failed result."""
        error = LoadError(stage=LoadStage.TERRAIN, message="Test error")
        result = LoadResult.fail(error)
        assert result.success is False
        assert result.error is error
        assert result.data is None


class TestTerrainLoader:
    """Tests for TerrainLoader."""

    @pytest.mark.asyncio
    async def test_terrain_loader_basic(self) -> None:
        """TerrainLoader loads terrain data."""
        loader = TerrainLoader(chunk_size=64)
        result = await loader.load(0, 0)
        assert result.success is True
        assert isinstance(result.data, TerrainData)
        assert result.data.heightmap_width == 64

    @pytest.mark.asyncio
    async def test_terrain_loader_cache(self) -> None:
        """TerrainLoader caches results."""
        loader = TerrainLoader(cache_enabled=True)
        await loader.load(5, 5)
        assert (5, 5) in loader._cache

    def test_terrain_loader_evict_cache(self) -> None:
        """evict_from_cache removes cached entry."""
        loader = TerrainLoader()
        loader._cache[(1, 2)] = TerrainData()
        assert loader.evict_from_cache(1, 2) is True
        assert (1, 2) not in loader._cache


class TestAsyncLoadPipeline:
    """Tests for AsyncLoadPipeline."""

    def test_pipeline_creation(self, pipeline_config: LoadPipelineConfig) -> None:
        """Pipeline creates with configuration."""
        pipeline = AsyncLoadPipeline(config=pipeline_config)
        assert pipeline.config.max_concurrent_loads == 2
        assert pipeline.pending_count == 0

    def test_pipeline_submit(self) -> None:
        """Pipeline accepts load requests."""
        pipeline = AsyncLoadPipeline()
        request = pipeline.submit(cell_x=0, cell_y=0, priority=50)
        assert request is not None
        assert pipeline.pending_count == 1

    def test_pipeline_submit_duplicate_returns_existing(self) -> None:
        """Submitting duplicate cell returns existing request."""
        pipeline = AsyncLoadPipeline()
        req1 = pipeline.submit(0, 0)
        req2 = pipeline.submit(0, 0)
        assert req1 is req2
        assert pipeline.pending_count == 1

    def test_pipeline_cancel(self) -> None:
        """Pipeline cancels pending requests."""
        pipeline = AsyncLoadPipeline()
        pipeline.submit(0, 0)
        assert pipeline.cancel(0, 0) is True
        assert pipeline.pending_count == 0

    def test_pipeline_cancel_nonexistent_returns_false(self) -> None:
        """Cancelling non-existent request returns False."""
        pipeline = AsyncLoadPipeline()
        assert pipeline.cancel(99, 99) is False

    @pytest.mark.asyncio
    async def test_pipeline_process(self) -> None:
        """Pipeline processes requests."""
        pipeline = AsyncLoadPipeline()
        pipeline.submit(0, 0)
        started = await pipeline.process()
        assert started == 1

    @pytest.mark.asyncio
    async def test_pipeline_full_load(self) -> None:
        """Pipeline completes full load cycle."""
        pipeline = AsyncLoadPipeline()
        completed: List[LoadResult] = []

        def on_complete(result: LoadResult) -> None:
            completed.append(result)

        pipeline.on_complete(on_complete)
        pipeline.submit(0, 0)
        await pipeline.process()

        # Wait for async load to complete
        await asyncio.sleep(0.1)

        assert len(completed) == 1
        assert completed[0].success is True

    def test_pipeline_stats(self) -> None:
        """Pipeline tracks statistics."""
        pipeline = AsyncLoadPipeline()
        stats = pipeline.get_stats()
        assert "total_loads" in stats
        assert "successful_loads" in stats
        assert "pending_count" in stats


# =============================================================================
# PRIORITY SYSTEM TESTS
# =============================================================================


class TestPriorityConfig:
    """Tests for PriorityConfig."""

    def test_priority_config_defaults(self) -> None:
        """PriorityConfig has sensible defaults."""
        config = PriorityConfig()
        assert config.distance_weight == 1.0
        assert config.max_priority_distance == 100.0

    def test_priority_config_invalid_max_distance_raises(self) -> None:
        """Invalid max_priority_distance raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            PriorityConfig(max_priority_distance=0.0)

    def test_priority_config_min_less_than_max_raises(self) -> None:
        """min_priority_distance <= max_priority_distance raises ValueError."""
        with pytest.raises(ValueError, match="greater than"):
            PriorityConfig(max_priority_distance=1000.0, min_priority_distance=500.0)


class TestStreamingSource:
    """Tests for StreamingSource."""

    def test_streaming_source_creation(self) -> None:
        """StreamingSource creates with defaults."""
        source = StreamingSource(source_id="player")
        assert source.position == (0.0, 0.0)
        assert source.is_active is True

    def test_streaming_source_speed(self) -> None:
        """speed property calculates correctly."""
        source = StreamingSource(velocity=(3.0, 4.0))
        assert source.speed == 5.0

    def test_streaming_source_predicted_position(self) -> None:
        """predicted_position calculates correctly."""
        source = StreamingSource(position=(10.0, 20.0), velocity=(5.0, 10.0))
        predicted = source.predicted_position(2.0)
        assert predicted == (20.0, 40.0)


class TestPriorityComputer:
    """Tests for PriorityComputer."""

    def test_compute_priority_no_sources(self) -> None:
        """Priority uses base_priority when no sources."""
        computer = PriorityComputer()
        priority = computer.compute_priority(0, 0, (128.0, 128.0))
        assert priority.priority > 0

    def test_compute_priority_with_source(self) -> None:
        """Priority increases when closer to source."""
        computer = PriorityComputer()
        computer.add_source(StreamingSource(
            position=(100.0, 100.0),
            source_id="test"
        ))

        close = computer.compute_priority(0, 0, (100.0, 100.0))
        far = computer.compute_priority(0, 0, (2000.0, 2000.0))

        assert close.priority > far.priority

    def test_compute_priority_velocity_affects_priority(self) -> None:
        """Velocity toward cell increases priority."""
        computer = PriorityComputer()
        computer.add_source(StreamingSource(
            position=(0.0, 0.0),
            velocity=(100.0, 0.0),  # Moving right
            source_id="moving"
        ))

        # Cell to the right (in direction of movement)
        ahead = computer.compute_priority(0, 0, (500.0, 0.0))
        # Cell to the left (opposite direction)
        behind = computer.compute_priority(0, 0, (-500.0, 0.0))

        assert ahead.priority > behind.priority

    def test_remove_source(self) -> None:
        """remove_source removes source by ID."""
        computer = PriorityComputer()
        computer.add_source(StreamingSource(source_id="test"))
        assert computer.remove_source("test") is True
        assert len(computer.get_sources()) == 0

    def test_update_source(self) -> None:
        """update_source updates source position."""
        computer = PriorityComputer()
        computer.add_source(StreamingSource(source_id="test", position=(0.0, 0.0)))
        computer.update_source("test", position=(100.0, 100.0))
        sources = computer.get_sources()
        assert sources[0].position == (100.0, 100.0)

    def test_compute_priorities_batch(self) -> None:
        """compute_priorities_batch sorts by priority."""
        computer = PriorityComputer()
        computer.add_source(StreamingSource(position=(0.0, 0.0), source_id="test"))

        cells = [
            (0, 0, (1000.0, 1000.0)),  # Far
            (1, 0, (100.0, 100.0)),    # Close
            (2, 0, (500.0, 500.0)),    # Medium
        ]

        priorities = computer.compute_priorities_batch(cells)
        # Should be sorted highest priority first (closest)
        assert priorities[0].cell_id == (1, 0)


class TestCellActivationTrigger:
    """Tests for CellActivationTrigger."""

    def test_trigger_activation(self) -> None:
        """trigger_activation creates events."""
        trigger = CellActivationTrigger()
        events = trigger.trigger_activation((0, 0))
        assert len(events) == len(ActivationType)
        assert trigger.pending_count > 0

    def test_trigger_clipmap_update(self) -> None:
        """trigger_clipmap_update creates clipmap event."""
        trigger = CellActivationTrigger()
        event = trigger.trigger_clipmap_update((1, 1), priority=100)
        assert event.activation_type == ActivationType.CLIPMAP_UPDATE
        assert event.cell_id == (1, 1)

    def test_trigger_foliage_merge(self) -> None:
        """trigger_foliage_merge creates foliage event."""
        trigger = CellActivationTrigger()
        event = trigger.trigger_foliage_merge((2, 2))
        assert event.activation_type == ActivationType.FOLIAGE_MERGE

    def test_process_events(self) -> None:
        """process_events processes and clears pending events."""
        trigger = CellActivationTrigger()
        processed: List[ActivationEvent] = []

        def callback(event: ActivationEvent) -> None:
            processed.append(event)

        trigger.register_callback(ActivationType.CLIPMAP_UPDATE, callback)
        trigger.trigger_clipmap_update((0, 0))

        count = trigger.process_events()
        assert count == 1
        assert len(processed) == 1

    def test_register_and_unregister_callback(self) -> None:
        """Callbacks can be registered and unregistered."""
        trigger = CellActivationTrigger()

        def callback(event: ActivationEvent) -> None:
            pass

        trigger.register_callback(ActivationType.CLIPMAP_UPDATE, callback)
        assert trigger.unregister_callback(ActivationType.CLIPMAP_UPDATE, callback) is True

    def test_get_pending_by_type(self) -> None:
        """get_pending_by_type filters by activation type."""
        trigger = CellActivationTrigger()
        trigger.trigger_clipmap_update((0, 0))
        trigger.trigger_foliage_merge((1, 1))

        clipmap_events = trigger.get_pending_by_type(ActivationType.CLIPMAP_UPDATE)
        assert len(clipmap_events) == 1

    def test_get_pending_for_cell(self) -> None:
        """get_pending_for_cell filters by cell ID."""
        trigger = CellActivationTrigger()
        trigger.trigger_activation((0, 0), [ActivationType.CLIPMAP_UPDATE])
        trigger.trigger_activation((1, 1), [ActivationType.FOLIAGE_MERGE])

        cell_events = trigger.get_pending_for_cell((0, 0))
        assert len(cell_events) == 1
        assert cell_events[0].cell_id == (0, 0)


# =============================================================================
# WORLD CHUNK TESTS
# =============================================================================


class TestWorldChunk:
    """Tests for WorldChunk class."""

    def test_world_chunk_creation(self) -> None:
        """WorldChunk creates with defaults."""
        chunk = WorldChunk(cell_x=5, cell_y=10)
        assert chunk.cell_key == (5, 10)

    def test_world_chunk_bounds(self) -> None:
        """get_world_bounds returns correct bounds."""
        chunk = WorldChunk(
            cell_x=1,
            cell_y=2,
            chunk_config=ChunkConfig(size=100.0),
        )
        (min_x, min_y), (max_x, max_y) = chunk.get_world_bounds()
        assert min_x == 100.0
        assert min_y == 200.0
        assert max_x == 200.0
        assert max_y == 300.0

    def test_world_chunk_center(self) -> None:
        """get_center returns correct center."""
        chunk = WorldChunk(
            cell_x=0,
            cell_y=0,
            chunk_config=ChunkConfig(size=100.0),
        )
        cx, cy = chunk.get_center()
        assert cx == 50.0
        assert cy == 50.0

    def test_world_chunk_distance_to_point(self) -> None:
        """distance_to_point calculates correctly."""
        chunk = WorldChunk(
            cell_x=0,
            cell_y=0,
            chunk_config=ChunkConfig(size=100.0),
        )
        # Center is at (50, 50), point at (50, 100) is 50 units away
        distance = chunk.distance_to_point(50.0, 100.0)
        assert distance == pytest.approx(50.0)

    def test_world_chunk_contains_point(self) -> None:
        """contains_point returns correct result."""
        chunk = WorldChunk(
            cell_x=0,
            cell_y=0,
            chunk_config=ChunkConfig(size=100.0),
        )
        assert chunk.contains_point(50.0, 50.0) is True
        assert chunk.contains_point(150.0, 50.0) is False

    def test_world_chunk_can_unload(self) -> None:
        """can_unload respects age and keep_loaded."""
        chunk = WorldChunk(
            load_timestamp=time.time() - 100.0,  # Loaded 100 seconds ago
            unloadable_config=UnloadableConfig(min_age=60.0),
            streamable_config=StreamableConfig(keep_loaded=False),
        )
        assert chunk.can_unload(time.time()) is True

    def test_world_chunk_cannot_unload_keep_loaded(self) -> None:
        """can_unload returns False when keep_loaded is True."""
        chunk = WorldChunk(
            load_timestamp=time.time() - 100.0,
            streamable_config=StreamableConfig(keep_loaded=True),
        )
        assert chunk.can_unload(time.time()) is False

    def test_world_chunk_cannot_unload_too_young(self) -> None:
        """can_unload returns False when age < min_age."""
        chunk = WorldChunk(
            load_timestamp=time.time() - 10.0,  # Only 10 seconds old
            unloadable_config=UnloadableConfig(min_age=60.0),
        )
        assert chunk.can_unload(time.time()) is False


# =============================================================================
# STREAM PRIORITY ENUM TESTS
# =============================================================================


class TestStreamPriority:
    """Tests for StreamPriority enum."""

    def test_priority_values(self) -> None:
        """Priority values are ordered correctly."""
        assert StreamPriority.CRITICAL.value > StreamPriority.HIGH.value
        assert StreamPriority.HIGH.value > StreamPriority.NORMAL.value
        assert StreamPriority.NORMAL.value > StreamPriority.LOW.value
        assert StreamPriority.LOW.value > StreamPriority.BACKGROUND.value

    def test_from_string_valid(self) -> None:
        """from_string parses valid strings."""
        assert StreamPriority.from_string("critical") == StreamPriority.CRITICAL
        assert StreamPriority.from_string("HIGH") == StreamPriority.HIGH
        assert StreamPriority.from_string("Normal") == StreamPriority.NORMAL

    def test_from_string_invalid_raises(self) -> None:
        """from_string raises ValueError for invalid strings."""
        with pytest.raises(ValueError, match="Invalid priority"):
            StreamPriority.from_string("invalid")
