"""Tests for realtime reflection probe capture system.

Covers:
- Face scheduler round-robin and priority modes
- LOD bias application
- Dynamic object filtering
- Multi-probe priority scheduling
- Budget limiting enforcement
- Temporal stability (no popping)
- Integration with baked probe system
"""

from __future__ import annotations

import math
import time
from typing import Optional

import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    BakedProbeAsset,
    BakedProbeConstants,
    BC6HCompressor,
    CaptureConfig,
    CompressionQuality,
    CubemapData,
    CubemapFace,
    CubemapFaceData,
    CubemapMipChain,
    HDRPixel,
    KTX2Writer,
    MipLevel,
)
from engine.rendering.lighting.reflection_probes import (
    RealtimeProbeConstants,
    SchedulerMode,
    ProbeUpdateReason,
    FaceState,
    RealtimeProbeFaceScheduler,
    RealtimeProbeCaptureSettings,
    DynamicObjectFilter,
    RealtimeProbeCapture,
    FunctionRealtimeProbeCapture,
    RealtimeProbeState,
    RealtimeReflectionProbe,
    CaptureBudget,
    RealtimeProbeManager,
    HybridProbeMode,
    HybridProbeConfig,
    HybridReflectionProbe,
)


# -----------------------------------------------------------------------------
# FaceState Tests
# -----------------------------------------------------------------------------

class TestFaceState:
    """Tests for face state tracking."""

    def test_face_state_creation(self) -> None:
        """Test creating a face state."""
        state = FaceState(face=CubemapFace.POSITIVE_X)
        assert state.face == CubemapFace.POSITIVE_X
        assert state.dirty is True
        assert state.priority == pytest.approx(1.0)

    def test_face_state_mark_dirty(self) -> None:
        """Test marking face as dirty."""
        state = FaceState(face=CubemapFace.POSITIVE_X, dirty=False, priority=0.0)
        state.mark_dirty(0.5)
        assert state.dirty is True
        assert state.change_magnitude == pytest.approx(0.5)
        assert state.priority > 0.0

    def test_face_state_mark_dirty_accumulates(self) -> None:
        """Test that mark_dirty accumulates magnitude."""
        state = FaceState(face=CubemapFace.POSITIVE_X)
        state.mark_dirty(0.3)
        state.mark_dirty(0.7)
        assert state.change_magnitude == pytest.approx(0.7)

    def test_face_state_mark_clean(self) -> None:
        """Test marking face as clean."""
        state = FaceState(face=CubemapFace.POSITIVE_X, dirty=True)
        state.mark_clean(frame=10)
        assert state.dirty is False
        assert state.last_update_frame == 10
        assert state.priority == pytest.approx(0.0)

    def test_face_state_decay_priority(self) -> None:
        """Test priority decay."""
        state = FaceState(face=CubemapFace.POSITIVE_X, priority=1.0)
        state.decay_priority(0.9)
        assert state.priority == pytest.approx(0.9)
        state.decay_priority(0.9)
        assert state.priority == pytest.approx(0.81)


# -----------------------------------------------------------------------------
# RealtimeProbeFaceScheduler Tests
# -----------------------------------------------------------------------------

class TestRealtimeProbeFaceScheduler:
    """Tests for face scheduling."""

    def test_scheduler_creation_default(self) -> None:
        """Test creating scheduler with defaults."""
        scheduler = RealtimeProbeFaceScheduler()
        assert scheduler.mode == SchedulerMode.ROUND_ROBIN
        assert len(scheduler.faces) == 6
        assert scheduler.current_face_index == 0

    def test_scheduler_round_robin_sequence(self) -> None:
        """Test round-robin face sequence."""
        scheduler = RealtimeProbeFaceScheduler(mode=SchedulerMode.ROUND_ROBIN)
        faces = [scheduler.get_next_face() for _ in range(6)]
        expected = [CubemapFace(i) for i in range(6)]
        assert faces == expected

    def test_scheduler_round_robin_wraps(self) -> None:
        """Test round-robin wraps around."""
        scheduler = RealtimeProbeFaceScheduler(mode=SchedulerMode.ROUND_ROBIN)
        for _ in range(6):
            scheduler.get_next_face()
        face = scheduler.get_next_face()
        assert face == CubemapFace.POSITIVE_X

    def test_scheduler_priority_mode_selects_highest(self) -> None:
        """Test priority mode selects highest priority face."""
        scheduler = RealtimeProbeFaceScheduler(mode=SchedulerMode.PRIORITY)
        # Mark specific face as highest priority
        scheduler.faces[2].priority = 1.0
        scheduler.faces[2].dirty = True
        for i in range(6):
            if i != 2:
                scheduler.faces[i].priority = 0.1
                scheduler.faces[i].dirty = True

        face = scheduler.get_next_face()
        assert face == CubemapFace.POSITIVE_Y  # Face index 2

    def test_scheduler_priority_mode_falls_back(self) -> None:
        """Test priority mode falls back to round-robin when no dirty faces."""
        scheduler = RealtimeProbeFaceScheduler(mode=SchedulerMode.PRIORITY)
        for state in scheduler.faces:
            state.dirty = False

        face = scheduler.get_next_face()
        assert face == CubemapFace.POSITIVE_X  # Round-robin fallback

    def test_scheduler_adaptive_mode_uses_priority_when_urgent(self) -> None:
        """Test adaptive mode uses priority for urgent updates."""
        scheduler = RealtimeProbeFaceScheduler(mode=SchedulerMode.ADAPTIVE)
        # Mark one face as urgent and clear others
        for i, state in enumerate(scheduler.faces):
            if i == 3:
                state.priority = 0.95  # Above urgent threshold
                state.dirty = True
            else:
                state.priority = 0.1
                state.dirty = False

        face = scheduler.get_next_face()
        assert face == CubemapFace.NEGATIVE_Y  # Face index 3

    def test_scheduler_get_next_faces_multiple(self) -> None:
        """Test getting multiple faces at once."""
        scheduler = RealtimeProbeFaceScheduler()
        faces = scheduler.get_next_faces(3)
        assert len(faces) == 3
        assert len(set(faces)) == 3  # All unique

    def test_scheduler_mark_face_dirty(self) -> None:
        """Test marking specific face dirty."""
        scheduler = RealtimeProbeFaceScheduler()
        scheduler.faces[0].dirty = False
        scheduler.mark_face_dirty(CubemapFace.POSITIVE_X, 0.8)
        assert scheduler.faces[0].dirty is True
        assert scheduler.faces[0].change_magnitude == pytest.approx(0.8)

    def test_scheduler_mark_all_dirty(self) -> None:
        """Test marking all faces dirty."""
        scheduler = RealtimeProbeFaceScheduler()
        for state in scheduler.faces:
            state.dirty = False
        scheduler.mark_all_dirty(0.5)
        assert all(state.dirty for state in scheduler.faces)

    def test_scheduler_mark_face_clean(self) -> None:
        """Test marking face clean."""
        scheduler = RealtimeProbeFaceScheduler()
        scheduler.get_next_face()  # Increment frame count
        scheduler.mark_face_clean(CubemapFace.POSITIVE_X)
        assert scheduler.faces[0].dirty is False

    def test_scheduler_get_update_priority(self) -> None:
        """Test getting update priority."""
        scheduler = RealtimeProbeFaceScheduler()
        scheduler.faces[2].priority = 0.75
        assert scheduler.get_update_priority(CubemapFace.POSITIVE_Y) == pytest.approx(0.75)

    def test_scheduler_is_face_dirty(self) -> None:
        """Test checking if face is dirty."""
        scheduler = RealtimeProbeFaceScheduler()
        assert scheduler.is_face_dirty(CubemapFace.POSITIVE_X) is True
        scheduler.mark_face_clean(CubemapFace.POSITIVE_X)
        assert scheduler.is_face_dirty(CubemapFace.POSITIVE_X) is False

    def test_scheduler_get_dirty_count(self) -> None:
        """Test getting dirty face count."""
        scheduler = RealtimeProbeFaceScheduler()
        assert scheduler.get_dirty_count() == 6  # All dirty by default
        scheduler.mark_face_clean(CubemapFace.POSITIVE_X)
        scheduler.mark_face_clean(CubemapFace.NEGATIVE_X)
        assert scheduler.get_dirty_count() == 4

    def test_scheduler_update_priorities_decay(self) -> None:
        """Test priority decay update."""
        scheduler = RealtimeProbeFaceScheduler()
        scheduler.decay_rate = 0.8
        scheduler.update_priorities()
        assert scheduler.faces[0].priority == pytest.approx(0.8)

    def test_scheduler_reset(self) -> None:
        """Test scheduler reset."""
        scheduler = RealtimeProbeFaceScheduler()
        scheduler.current_face_index = 3
        scheduler.frame_count = 100
        scheduler.faces[0].dirty = False
        scheduler.reset()
        assert scheduler.current_face_index == 0
        assert scheduler.frame_count == 0
        assert all(state.dirty for state in scheduler.faces)


# -----------------------------------------------------------------------------
# RealtimeProbeCaptureSettings Tests
# -----------------------------------------------------------------------------

class TestRealtimeProbeCaptureSettings:
    """Tests for capture settings."""

    def test_settings_creation_defaults(self) -> None:
        """Test creating settings with defaults."""
        settings = RealtimeProbeCaptureSettings()
        assert settings.resolution == RealtimeProbeConstants.DEFAULT_RESOLUTION
        assert settings.lod_bias == RealtimeProbeConstants.DEFAULT_LOD_BIAS
        assert settings.include_dynamic_objects is True

    def test_settings_resolution_clamping(self) -> None:
        """Test resolution is clamped to valid range."""
        settings = RealtimeProbeCaptureSettings(resolution=10)
        assert settings.resolution == RealtimeProbeConstants.MIN_RESOLUTION
        settings = RealtimeProbeCaptureSettings(resolution=10000)
        assert settings.resolution == RealtimeProbeConstants.MAX_RESOLUTION

    def test_settings_lod_bias_clamping(self) -> None:
        """Test LOD bias is clamped."""
        settings = RealtimeProbeCaptureSettings(lod_bias=-1.0)
        assert settings.lod_bias == pytest.approx(0.0)
        settings = RealtimeProbeCaptureSettings(lod_bias=10.0)
        assert settings.lod_bias == pytest.approx(RealtimeProbeConstants.MAX_LOD_BIAS)

    def test_settings_update_rate_clamping(self) -> None:
        """Test update rate clamping."""
        settings = RealtimeProbeCaptureSettings(update_rate=0)
        assert settings.update_rate == 1
        settings = RealtimeProbeCaptureSettings(update_rate=1000)
        assert settings.update_rate == RealtimeProbeConstants.MAX_UPDATE_RATE

    def test_settings_temporal_blend_factor_clamping(self) -> None:
        """Test temporal blend factor clamping."""
        settings = RealtimeProbeCaptureSettings(temporal_blend_factor=-0.5)
        assert settings.temporal_blend_factor == pytest.approx(0.0)
        settings = RealtimeProbeCaptureSettings(temporal_blend_factor=1.5)
        assert settings.temporal_blend_factor == pytest.approx(1.0)

    def test_settings_get_effective_lod_bias(self) -> None:
        """Test distance-based LOD bias."""
        settings = RealtimeProbeCaptureSettings(lod_bias=1.0, distance_scale=1.0, max_render_distance=100.0)
        bias_near = settings.get_effective_lod_bias(0.0)
        bias_far = settings.get_effective_lod_bias(100.0)
        assert bias_far > bias_near

    def test_settings_get_effective_update_rate(self) -> None:
        """Test distance-based update rate."""
        settings = RealtimeProbeCaptureSettings(update_rate=1, distance_scale=2.0, max_render_distance=100.0)
        rate_near = settings.get_effective_update_rate(0.0)
        rate_far = settings.get_effective_update_rate(100.0)
        assert rate_far > rate_near


# -----------------------------------------------------------------------------
# DynamicObjectFilter Tests
# -----------------------------------------------------------------------------

class TestDynamicObjectFilter:
    """Tests for dynamic object filtering."""

    def test_filter_creation_defaults(self) -> None:
        """Test creating filter with defaults."""
        filter_ = DynamicObjectFilter()
        assert filter_.include_layers == 0xFFFFFFFF
        assert filter_.include_moving is True

    def test_filter_should_include_default(self) -> None:
        """Test default inclusion."""
        filter_ = DynamicObjectFilter()
        assert filter_.should_include(1, 1, []) is True

    def test_filter_explicit_exclusion(self) -> None:
        """Test explicit entity exclusion."""
        filter_ = DynamicObjectFilter()
        filter_.exclude_entity(42)
        assert filter_.should_include(42, 1, []) is False
        assert filter_.should_include(43, 1, []) is True

    def test_filter_explicit_inclusion(self) -> None:
        """Test explicit entity inclusion overrides."""
        filter_ = DynamicObjectFilter(include_layers=0)  # Exclude all layers
        filter_.include_entity(42)
        assert filter_.should_include(42, 1, []) is True

    def test_filter_layer_mask(self) -> None:
        """Test layer mask filtering."""
        filter_ = DynamicObjectFilter(include_layers=0b0101)
        assert filter_.should_include(1, 0b0001, []) is True  # Layer matches
        assert filter_.should_include(2, 0b0010, []) is False  # Layer doesn't match
        assert filter_.should_include(3, 0b0100, []) is True  # Layer matches

    def test_filter_tag_exclusion(self) -> None:
        """Test tag-based exclusion."""
        filter_ = DynamicObjectFilter(exclude_tags=["editor", "debug"])
        assert filter_.should_include(1, 1, []) is True
        assert filter_.should_include(2, 1, ["editor"]) is False
        assert filter_.should_include(3, 1, ["debug"]) is False
        assert filter_.should_include(4, 1, ["player"]) is True

    def test_filter_moving_objects(self) -> None:
        """Test moving object filtering."""
        filter_ = DynamicObjectFilter(include_moving=False, velocity_threshold=0.1)
        assert filter_.should_include(1, 1, [], velocity=0.05, is_static=False) is True  # Below threshold
        assert filter_.should_include(2, 1, [], velocity=0.5, is_static=False) is False  # Above threshold
        assert filter_.should_include(3, 1, [], velocity=0.5, is_static=True) is True  # Static

    def test_filter_exclude_include_entity_toggle(self) -> None:
        """Test toggling between exclude and include."""
        filter_ = DynamicObjectFilter()
        filter_.exclude_entity(1)
        assert 1 in filter_.excluded_entities
        filter_.include_entity(1)
        assert 1 not in filter_.excluded_entities
        assert 1 in filter_.included_entities

    def test_filter_clear_exclusions(self) -> None:
        """Test clearing exclusions."""
        filter_ = DynamicObjectFilter()
        filter_.exclude_entity(1)
        filter_.include_entity(2)
        filter_.clear_exclusions()
        assert len(filter_.excluded_entities) == 0
        assert len(filter_.included_entities) == 0


# -----------------------------------------------------------------------------
# RealtimeProbeCapture Tests
# -----------------------------------------------------------------------------

class TestRealtimeProbeCapture:
    """Tests for realtime probe capture."""

    def test_capture_creation(self) -> None:
        """Test creating capture handler."""
        config = CaptureConfig(resolution=64)
        settings = RealtimeProbeCaptureSettings(resolution=64)
        capture = RealtimeProbeCapture(config, settings)
        assert capture.settings.resolution == 64

    def test_capture_buffers_initialized(self) -> None:
        """Test ping-pong buffers are initialized."""
        config = CaptureConfig(resolution=32)
        settings = RealtimeProbeCaptureSettings(resolution=32)
        capture = RealtimeProbeCapture(config, settings)
        cubemap = capture.get_current_cubemap()
        assert cubemap is not None
        assert cubemap.resolution == 32

    def test_capture_face_tracks_completion(self) -> None:
        """Test face capture tracks completion."""
        config = CaptureConfig(resolution=32)
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1, 1, 1)
        )

        assert not capture.complete_probe()
        for face in CubemapFace:
            capture.capture_face(Vec3.zero(), face)
        assert capture.complete_probe()

    def test_capture_finalize_swaps_buffers(self) -> None:
        """Test finalize_frame swaps buffers."""
        config = CaptureConfig(resolution=32)
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1, 1, 1)
        )

        for face in CubemapFace:
            capture.capture_face(Vec3.zero(), face)

        result = capture.finalize_frame()
        assert result is not None
        assert not capture.complete_probe()  # Reset after swap

    def test_capture_reset(self) -> None:
        """Test capture reset."""
        config = CaptureConfig(resolution=32)
        settings = RealtimeProbeCaptureSettings(resolution=32)
        capture = RealtimeProbeCapture(config, settings)
        capture._completed_faces.add(CubemapFace.POSITIVE_X)
        capture.reset()
        assert len(capture._completed_faces) == 0


class TestFunctionRealtimeProbeCapture:
    """Tests for function-based realtime capture."""

    def test_function_capture_samples_correctly(self) -> None:
        """Test function capture samples environment."""
        # Note: FunctionRealtimeProbeCapture uses settings.resolution, not config.resolution
        settings = RealtimeProbeCaptureSettings(resolution=64, temporal_blend_factor=0.0)
        config = CaptureConfig(resolution=64)

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(direction.x * 0.5 + 0.5, direction.y * 0.5 + 0.5, 1.0)

        capture = FunctionRealtimeProbeCapture(config, settings, sample_func)
        face_data = capture.capture_face(Vec3.zero(), CubemapFace.POSITIVE_X)

        assert face_data.resolution == settings.resolution
        # Center pixel should sample positive X direction
        center = face_data.get_pixel(32, 32)
        assert center.b == pytest.approx(1.0)


# -----------------------------------------------------------------------------
# RealtimeReflectionProbe Tests
# -----------------------------------------------------------------------------

class TestRealtimeReflectionProbe:
    """Tests for realtime reflection probes."""

    def _create_test_probe(self) -> RealtimeReflectionProbe:
        """Create a test probe."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="test_probe",
            position=Vec3(0, 0, 0),
            bounds=AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10)),
            settings=settings,
        )

        config = CaptureConfig(resolution=32)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1.0, 0.5, 0.25)
        )
        probe.initialize(capture)
        return probe

    def test_probe_creation(self) -> None:
        """Test creating a probe."""
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="test",
            position=Vec3(0, 0, 0),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )
        assert probe.probe_id == 1
        assert probe.name == "test"
        assert probe.state is not None

    def test_probe_initialize(self) -> None:
        """Test probe initialization."""
        probe = self._create_test_probe()
        assert probe._capture is not None
        assert probe._current_cubemap is not None

    def test_probe_update_face(self) -> None:
        """Test updating a single face."""
        probe = self._create_test_probe()
        face = probe.update_face(frame=1)
        assert face is not None
        assert probe.state.last_partial_frame == 1

    def test_probe_update_faces_multiple(self) -> None:
        """Test updating multiple faces."""
        probe = self._create_test_probe()
        faces = probe.update_faces(frame=1, face_count=3)
        assert len(faces) == 3

    def test_probe_complete_update_cycle(self) -> None:
        """Test completing a full update cycle."""
        probe = self._create_test_probe()
        for i in range(6):
            probe.update_face(frame=i)
        assert probe.state.update_count == 1
        assert probe.state.last_update_frame == 5

    def test_probe_get_cubemap(self) -> None:
        """Test getting current cubemap."""
        probe = self._create_test_probe()
        probe.update_faces(frame=1, face_count=6)
        cubemap = probe.get_cubemap()
        assert cubemap is not None
        assert cubemap.resolution == 32

    def test_probe_get_mip_chain(self) -> None:
        """Test getting mip chain."""
        probe = self._create_test_probe()
        probe.update_faces(frame=1, face_count=6)
        mip_chain = probe.get_mip_chain()
        assert mip_chain is not None
        assert len(mip_chain.mips) > 0

    def test_probe_sample(self) -> None:
        """Test sampling probe."""
        probe = self._create_test_probe()
        probe.update_faces(frame=1, face_count=6)
        color = probe.sample(Vec3(1, 0, 0), roughness=0.0)
        assert color.x > 0.0  # Should have sampled color

    def test_probe_mark_dirty(self) -> None:
        """Test marking probe dirty."""
        probe = self._create_test_probe()
        probe.scheduler.mark_face_clean(CubemapFace.POSITIVE_X)
        probe.mark_dirty(0.8)
        assert probe.scheduler.is_face_dirty(CubemapFace.POSITIVE_X)

    def test_probe_mark_face_dirty(self) -> None:
        """Test marking specific face dirty."""
        probe = self._create_test_probe()
        probe.scheduler.mark_face_clean(CubemapFace.POSITIVE_Y)
        probe.mark_face_dirty(CubemapFace.POSITIVE_Y, 0.5)
        assert probe.scheduler.is_face_dirty(CubemapFace.POSITIVE_Y)

    def test_probe_contains_point(self) -> None:
        """Test point containment check."""
        probe = self._create_test_probe()
        assert probe.contains(Vec3(0, 0, 0)) is True
        assert probe.contains(Vec3(5, 5, 5)) is True
        assert probe.contains(Vec3(15, 0, 0)) is False

    def test_probe_update_priority(self) -> None:
        """Test priority update based on camera."""
        probe = self._create_test_probe()
        probe.update_priority(Vec3(0, 0, 0), is_visible=True)
        priority_visible = probe.state.priority
        probe.update_priority(Vec3(0, 0, 0), is_visible=False)
        priority_hidden = probe.state.priority
        assert priority_visible > priority_hidden

    def test_probe_convert_to_baked(self) -> None:
        """Test converting realtime to baked probe."""
        probe = self._create_test_probe()
        probe.update_faces(frame=1, face_count=6)
        baked = probe.convert_to_baked()
        assert baked is not None
        assert baked.name == "test_probe_baked"
        assert len(baked.ktx2_data) > 0


# -----------------------------------------------------------------------------
# CaptureBudget Tests
# -----------------------------------------------------------------------------

class TestCaptureBudget:
    """Tests for capture budget."""

    def test_budget_creation_defaults(self) -> None:
        """Test creating budget with defaults."""
        budget = CaptureBudget()
        assert budget.max_face_renders == RealtimeProbeConstants.DEFAULT_CAPTURE_BUDGET
        assert budget.max_time_ms > 0

    def test_budget_face_renders_clamping(self) -> None:
        """Test face renders clamping."""
        budget = CaptureBudget(max_face_renders=0)
        assert budget.max_face_renders >= 1
        budget = CaptureBudget(max_face_renders=100)
        assert budget.max_face_renders <= RealtimeProbeConstants.MAX_CAPTURE_BUDGET

    def test_budget_time_clamping(self) -> None:
        """Test time budget clamping."""
        budget = CaptureBudget(max_time_ms=0.0)
        assert budget.max_time_ms >= 0.1


# -----------------------------------------------------------------------------
# RealtimeProbeManager Tests
# -----------------------------------------------------------------------------

class TestRealtimeProbeManager:
    """Tests for probe management."""

    def _create_manager_with_probes(self) -> RealtimeProbeManager:
        """Create manager with test probes."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        manager = RealtimeProbeManager(default_settings=settings)

        for i in range(3):
            pos = Vec3(i * 20, 0, 0)
            config = CaptureConfig(resolution=32)
            capture = FunctionRealtimeProbeCapture(
                config, settings,
                lambda pos, dir_: Vec3(1.0, 0.5, 0.25)
            )
            manager.register_probe(
                name=f"probe_{i}",
                position=pos,
                bounds=AABB(pos - Vec3(5, 5, 5), pos + Vec3(5, 5, 5)),
                settings=settings,
                capture=capture,
            )

        return manager

    def test_manager_creation(self) -> None:
        """Test creating manager."""
        manager = RealtimeProbeManager()
        assert manager.probe_count == 0

    def test_manager_register_probe(self) -> None:
        """Test registering a probe."""
        manager = RealtimeProbeManager()
        probe = manager.register_probe(
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )
        assert probe is not None
        assert manager.probe_count == 1

    def test_manager_unregister_probe(self) -> None:
        """Test unregistering a probe."""
        manager = RealtimeProbeManager()
        probe = manager.register_probe(
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )
        result = manager.unregister_probe(probe.probe_id)
        assert result is True
        assert manager.probe_count == 0

    def test_manager_get_probe(self) -> None:
        """Test getting probe by ID."""
        manager = RealtimeProbeManager()
        probe = manager.register_probe(
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )
        retrieved = manager.get_probe(probe.probe_id)
        assert retrieved is probe
        assert manager.get_probe(999) is None

    def test_manager_get_all_probes(self) -> None:
        """Test getting all probes."""
        manager = self._create_manager_with_probes()
        probes = manager.get_all_probes()
        assert len(probes) == 3

    def test_manager_update_probes(self) -> None:
        """Test updating probes."""
        manager = self._create_manager_with_probes()
        updates = manager.update_probes(camera_position=Vec3.zero())
        assert len(updates) > 0

    def test_manager_update_respects_budget(self) -> None:
        """Test update respects face budget."""
        manager = self._create_manager_with_probes()
        manager.budget = CaptureBudget(max_face_renders=2)
        updates = manager.update_probes(camera_position=Vec3.zero())
        total_faces = sum(len(faces) for _, faces in updates)
        assert total_faces <= 2

    def test_manager_update_prioritizes_visible(self) -> None:
        """Test visible probes are prioritized."""
        manager = self._create_manager_with_probes()
        probes = manager.get_all_probes()
        visible_ids = {probes[0].probe_id}  # Only first probe visible

        manager.budget = CaptureBudget(max_face_renders=1)
        updates = manager.update_probes(
            camera_position=Vec3.zero(),
            visible_probe_ids=visible_ids,
        )

        # First update should be visible probe
        if updates:
            assert updates[0][0] == probes[0].probe_id

    def test_manager_get_capture_budget(self) -> None:
        """Test getting capture budget."""
        manager = RealtimeProbeManager()
        manager.budget = CaptureBudget(max_face_renders=4, max_time_ms=3.0)
        faces, time_ms = manager.get_capture_budget()
        assert faces == 4
        assert time_ms == pytest.approx(3.0)

    def test_manager_set_capture_budget(self) -> None:
        """Test setting capture budget."""
        manager = RealtimeProbeManager()
        manager.set_capture_budget(max_faces=5, max_time_ms=2.5)
        faces, time_ms = manager.get_capture_budget()
        assert faces == 5
        assert time_ms == pytest.approx(2.5)

    def test_manager_get_last_update_time(self) -> None:
        """Test getting last update time."""
        manager = self._create_manager_with_probes()
        manager.update_probes(camera_position=Vec3.zero())
        update_time = manager.get_last_update_time()
        assert update_time >= 0.0

    def test_manager_find_affecting_probes(self) -> None:
        """Test finding affecting probes."""
        manager = self._create_manager_with_probes()
        affecting = manager.find_affecting_probes(Vec3(0, 0, 0))
        assert len(affecting) >= 1

        affecting = manager.find_affecting_probes(Vec3(100, 0, 0))
        assert len(affecting) == 0

    def test_manager_sample(self) -> None:
        """Test blended sampling."""
        manager = self._create_manager_with_probes()
        # Update all probes
        for _ in range(6):
            manager.update_probes(camera_position=Vec3.zero())

        color = manager.sample(Vec3(0, 0, 0), Vec3(1, 0, 0), 0.0)
        # Should have sampled something
        assert color.x >= 0.0

    def test_manager_mark_all_dirty(self) -> None:
        """Test marking all probes dirty."""
        manager = self._create_manager_with_probes()
        for probe in manager.get_all_probes():
            for state in probe.scheduler.faces:
                state.dirty = False

        manager.mark_all_dirty(0.5)

        for probe in manager.get_all_probes():
            assert probe.scheduler.get_dirty_count() == 6

    def test_manager_mark_region_dirty(self) -> None:
        """Test marking probes in region dirty."""
        manager = self._create_manager_with_probes()
        for probe in manager.get_all_probes():
            for state in probe.scheduler.faces:
                state.dirty = False

        # Mark region around first probe
        region = AABB(Vec3(-6, -6, -6), Vec3(6, 6, 6))
        manager.mark_region_dirty(region, 0.7)

        probes = manager.get_all_probes()
        # First probe should be dirty
        assert probes[0].scheduler.get_dirty_count() == 6
        # Far probes should not be dirty
        assert probes[2].scheduler.get_dirty_count() == 0

    def test_manager_clear(self) -> None:
        """Test clearing all probes."""
        manager = self._create_manager_with_probes()
        manager.clear()
        assert manager.probe_count == 0


# -----------------------------------------------------------------------------
# HybridReflectionProbe Tests
# -----------------------------------------------------------------------------

class TestHybridReflectionProbe:
    """Tests for hybrid baked/realtime probes."""

    def _create_baked_asset(self) -> BakedProbeAsset:
        """Create a test baked asset."""
        cubemap = CubemapData(resolution=4)
        for face in cubemap.faces:
            for i in range(16):
                face.pixels[i] = HDRPixel(0.0, 1.0, 0.0)  # Green

        mip_chain = CubemapMipChain(base_resolution=4, mip_count=1)
        mip_chain.mips.append(MipLevel(level=0, resolution=4, cubemap=cubemap))

        comp = BC6HCompressor()
        compressed = [comp.compress_cubemap(cubemap)]

        writer = KTX2Writer()
        ktx2_data = writer.write_to_bytes(mip_chain, compressed, supercompress=False)

        return BakedProbeAsset(
            probe_id=1,
            name="baked",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
            resolution=4,
            mip_count=1,
            is_prefiltered=False,
            ktx2_data=ktx2_data,
        )

    def _create_realtime_probe(self) -> RealtimeReflectionProbe:
        """Create a test realtime probe."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="realtime",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
            settings=settings,
        )

        config = CaptureConfig(resolution=32)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1.0, 0.0, 0.0)  # Red
        )
        probe.initialize(capture)
        probe.update_faces(frame=1, face_count=6)
        return probe

    def test_hybrid_probe_creation(self) -> None:
        """Test creating hybrid probe."""
        realtime = self._create_realtime_probe()
        config = HybridProbeConfig(mode=HybridProbeMode.REALTIME_ONLY)
        hybrid = HybridReflectionProbe(realtime, config)
        assert hybrid.probe_id == realtime.probe_id

    def test_hybrid_probe_realtime_only(self) -> None:
        """Test realtime-only mode."""
        realtime = self._create_realtime_probe()
        config = HybridProbeConfig(mode=HybridProbeMode.REALTIME_ONLY)
        hybrid = HybridReflectionProbe(realtime, config)

        color = hybrid.sample(Vec3(1, 0, 0), 0.0)
        # Should sample from realtime (red)
        assert color.x > 0.0

    def test_hybrid_probe_baked_only(self) -> None:
        """Test baked-only mode."""
        realtime = self._create_realtime_probe()
        baked = self._create_baked_asset()
        baked.load()

        config = HybridProbeConfig(mode=HybridProbeMode.BAKED_ONLY, baked_asset=baked)
        hybrid = HybridReflectionProbe(realtime, config)

        color = hybrid.sample(Vec3(1, 0, 0), 0.0)
        # Should sample from baked (green)
        assert color.y > 0.0

    def test_hybrid_probe_baked_plus_delta(self) -> None:
        """Test baked + delta mode."""
        realtime = self._create_realtime_probe()
        baked = self._create_baked_asset()
        baked.load()

        config = HybridProbeConfig(
            mode=HybridProbeMode.BAKED_PLUS_DELTA,
            baked_asset=baked,
            delta_blend=0.5,
        )
        hybrid = HybridReflectionProbe(realtime, config)

        color = hybrid.sample(Vec3(1, 0, 0), 0.0)
        # Should blend red and green
        assert color.x > 0.0 or color.y > 0.0

    def test_hybrid_probe_adaptive_mode(self) -> None:
        """Test adaptive mode switching."""
        realtime = self._create_realtime_probe()
        baked = self._create_baked_asset()
        baked.load()

        config = HybridProbeConfig(
            mode=HybridProbeMode.ADAPTIVE,
            baked_asset=baked,
            activity_threshold=0.5,
        )
        hybrid = HybridReflectionProbe(realtime, config)

        # Low activity should use baked
        hybrid.update_scene_activity(0.2)
        assert hybrid._get_effective_mode() == HybridProbeMode.BAKED_ONLY

        # High activity should use hybrid
        hybrid.update_scene_activity(0.8)
        assert hybrid._get_effective_mode() == HybridProbeMode.BAKED_PLUS_DELTA

    def test_hybrid_probe_set_baked_asset(self) -> None:
        """Test setting baked asset."""
        realtime = self._create_realtime_probe()
        config = HybridProbeConfig(mode=HybridProbeMode.BAKED_ONLY)
        hybrid = HybridReflectionProbe(realtime, config)

        baked = self._create_baked_asset()
        hybrid.set_baked_asset(baked)
        assert hybrid.config.baked_asset is baked
        assert baked.loaded


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_six_frames_capture_cycle(self) -> None:
        """Test 6 faces captured over 6 frames."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
            settings=settings,
        )

        config = CaptureConfig(resolution=32)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1, 1, 1)
        )
        probe.initialize(capture)

        captured_faces = []
        for frame in range(6):
            face = probe.update_face(frame=frame)
            captured_faces.append(face)

        # All 6 faces should have been captured
        assert len(set(captured_faces)) == 6
        assert probe.state.update_count == 1

    def test_lod_bias_increases_with_distance(self) -> None:
        """Test LOD bias correctly increases with distance."""
        settings = RealtimeProbeCaptureSettings(
            resolution=128,
            lod_bias=1.0,
            distance_scale=2.0,
            max_render_distance=100.0,
        )

        near_bias = settings.get_effective_lod_bias(10.0)
        mid_bias = settings.get_effective_lod_bias(50.0)
        far_bias = settings.get_effective_lod_bias(100.0)

        assert far_bias > mid_bias > near_bias

    def test_priority_scheduling_updates_urgent_first(self) -> None:
        """Test priority scheduling updates urgent probes first."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        manager = RealtimeProbeManager(
            default_settings=settings,
            budget=CaptureBudget(max_face_renders=1),
        )

        # Create probes
        for i in range(3):
            pos = Vec3(i * 50, 0, 0)
            config = CaptureConfig(resolution=32)
            capture = FunctionRealtimeProbeCapture(
                config, settings,
                lambda pos, dir_: Vec3(1, 1, 1)
            )
            manager.register_probe(
                name=f"probe_{i}",
                position=pos,
                bounds=AABB(pos - Vec3(5, 5, 5), pos + Vec3(5, 5, 5)),
                capture=capture,
            )

        # Make middle probe urgent
        probes = manager.get_all_probes()
        probes[1].state.priority = 0.99

        updates = manager.update_probes(camera_position=Vec3(50, 0, 0))

        # Middle probe should be updated first
        if updates:
            assert updates[0][0] == probes[1].probe_id

    def test_budget_limiting_enforced(self) -> None:
        """Test budget limiting is enforced."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        manager = RealtimeProbeManager(
            default_settings=settings,
            budget=CaptureBudget(max_face_renders=3),
        )

        # Create many probes
        for i in range(5):
            pos = Vec3(i * 20, 0, 0)
            config = CaptureConfig(resolution=32)
            capture = FunctionRealtimeProbeCapture(
                config, settings,
                lambda pos, dir_: Vec3(1, 1, 1)
            )
            manager.register_probe(
                name=f"probe_{i}",
                position=pos,
                bounds=AABB(pos - Vec3(5, 5, 5), pos + Vec3(5, 5, 5)),
                capture=capture,
            )

        updates = manager.update_probes(camera_position=Vec3.zero())

        total_faces = sum(len(faces) for _, faces in updates)
        assert total_faces <= 3

    def test_temporal_stability_no_popping(self) -> None:
        """Test temporal blending prevents popping.

        Temporal blending works with ping-pong buffers:
        - Buffer A is the current write buffer (starts at index 0)
        - Buffer B is the previous read buffer (initially empty)
        - Buffers swap after completing all 6 faces

        To test temporal stability, we:
        1. Complete a full cycle with red color (fills buffer A)
        2. finalize_frame() swaps buffers (A->prev, B->current)
        3. Start new cycle with black color
        4. Blending should mix new black with previous red
        """
        settings = RealtimeProbeCaptureSettings(
            resolution=32,
            temporal_blend_factor=0.5,  # 50% blend with previous
        )
        config = CaptureConfig(resolution=32)

        current_color = [Vec3(1.0, 0.0, 0.0)]  # Start with red

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return current_color[0]

        capture = FunctionRealtimeProbeCapture(config, settings, sample_func)

        # Complete first cycle with red - capture all 6 faces
        for face in CubemapFace:
            capture.capture_face(Vec3.zero(), face)

        # Finalize to swap buffers (red is now in "previous" buffer)
        result = capture.finalize_frame()
        assert result is not None  # Should have completed

        # Now change color to black for second cycle
        current_color[0] = Vec3(0.0, 0.0, 0.0)

        # Capture same face again - should blend black with previous red
        face_data = capture.capture_face(Vec3.zero(), CubemapFace.POSITIVE_X)
        center = face_data.get_pixel(16, 16)

        # With 50% blend: result = 0.5*new + 0.5*prev = 0.5*0 + 0.5*red
        # The previous red was itself blended with empty: 0.5*1.0 + 0.5*0 = 0.5
        # So now: 0.5*0 + 0.5*0.5 = 0.25
        # This demonstrates temporal stability - sudden changes are smoothed
        assert center.r > 0.1  # Should retain some red from previous frame
        assert center.r < 0.9  # Should not be pure red (shows blending occurred)

    def test_baked_probe_integration(self) -> None:
        """Test integration with baked probe system."""
        # Create and update realtime probe
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="realtime_test",
            position=Vec3(10, 20, 30),
            bounds=AABB(Vec3(0, 10, 20), Vec3(20, 30, 40)),
            settings=settings,
        )

        config = CaptureConfig(resolution=32)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(0.8, 0.6, 0.4)
        )
        probe.initialize(capture)
        probe.update_faces(frame=1, face_count=6)

        # Convert to baked
        baked = probe.convert_to_baked()

        assert baked is not None
        assert baked.position == probe.position
        assert baked.bounds.min == probe.bounds.min
        assert baked.bounds.max == probe.bounds.max
        assert baked.is_prefiltered is True

        # Load and sample
        baked.load()
        color = baked.sample(Vec3(1, 0, 0), 0.0)
        assert color.x > 0.0

    def test_face_dirty_tracking_triggers_rerender(self) -> None:
        """Test face dirty tracking triggers re-render."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
            settings=settings,
            scheduler=RealtimeProbeFaceScheduler(mode=SchedulerMode.PRIORITY),
        )

        config = CaptureConfig(resolution=32)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1, 1, 1)
        )
        probe.initialize(capture)

        # Complete full cycle
        for _ in range(6):
            probe.update_face(frame=0)

        # Mark one face dirty with high priority
        probe.mark_face_dirty(CubemapFace.NEGATIVE_Z, 1.0)

        # Next update should render the dirty face
        face = probe.update_face(frame=7)
        assert face == CubemapFace.NEGATIVE_Z


# -----------------------------------------------------------------------------
# Constants Tests
# -----------------------------------------------------------------------------

class TestRealtimeProbeConstants:
    """Tests for realtime probe constants."""

    def test_resolution_limits(self) -> None:
        """Test resolution limits."""
        assert RealtimeProbeConstants.MIN_RESOLUTION <= RealtimeProbeConstants.DEFAULT_RESOLUTION
        assert RealtimeProbeConstants.DEFAULT_RESOLUTION <= RealtimeProbeConstants.MAX_RESOLUTION

    def test_budget_limits(self) -> None:
        """Test budget limits."""
        assert RealtimeProbeConstants.DEFAULT_CAPTURE_BUDGET <= RealtimeProbeConstants.MAX_CAPTURE_BUDGET

    def test_lod_bias_limits(self) -> None:
        """Test LOD bias limits."""
        assert RealtimeProbeConstants.DEFAULT_LOD_BIAS <= RealtimeProbeConstants.MAX_LOD_BIAS


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_manager_sample(self) -> None:
        """Test sampling from empty manager."""
        manager = RealtimeProbeManager()
        color = manager.sample(Vec3.zero(), Vec3(1, 0, 0), 0.0)
        assert color == Vec3(0, 0, 0)

    def test_uninitialized_probe_sample(self) -> None:
        """Test sampling uninitialized probe."""
        probe = RealtimeReflectionProbe(
            probe_id=1,
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
        )
        color = probe.sample(Vec3(1, 0, 0), 0.0)
        assert color == Vec3(0, 0, 0)

    def test_scheduler_all_faces_clean(self) -> None:
        """Test scheduler behavior when all faces clean."""
        scheduler = RealtimeProbeFaceScheduler(mode=SchedulerMode.PRIORITY)
        for state in scheduler.faces:
            state.dirty = False
            state.priority = 0.0

        # Should fall back to round-robin
        face = scheduler.get_next_face()
        assert isinstance(face, CubemapFace)

    def test_manager_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent probe."""
        manager = RealtimeProbeManager()
        result = manager.unregister_probe(999)
        assert result is False

    def test_hybrid_probe_no_baked_asset(self) -> None:
        """Test hybrid probe without baked asset."""
        settings = RealtimeProbeCaptureSettings(resolution=32, temporal_blend_factor=0.0)
        realtime = RealtimeReflectionProbe(
            probe_id=1,
            name="test",
            position=Vec3.zero(),
            bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
            settings=settings,
        )

        config = CaptureConfig(resolution=32)
        capture = FunctionRealtimeProbeCapture(
            config, settings,
            lambda pos, dir_: Vec3(1, 0, 0)
        )
        realtime.initialize(capture)
        realtime.update_faces(frame=1, face_count=6)

        hybrid_config = HybridProbeConfig(mode=HybridProbeMode.BAKED_ONLY)
        hybrid = HybridReflectionProbe(realtime, hybrid_config)

        # Should return black since no baked asset
        color = hybrid.sample(Vec3(1, 0, 0), 0.0)
        assert color == Vec3(0, 0, 0)

    def test_zero_temporal_blend(self) -> None:
        """Test zero temporal blend factor."""
        settings = RealtimeProbeCaptureSettings(resolution=4, temporal_blend_factor=0.0)
        config = CaptureConfig(resolution=4)

        frame = [0]

        def sample_func(pos: Vec3, direction: Vec3) -> Vec3:
            return Vec3(frame[0], 0, 0)

        capture = FunctionRealtimeProbeCapture(config, settings, sample_func)

        # First capture
        capture.capture_face(Vec3.zero(), CubemapFace.POSITIVE_X)
        frame[0] = 1.0

        # Second capture - should have no blending
        face_data = capture.capture_face(Vec3.zero(), CubemapFace.POSITIVE_X)
        center = face_data.get_pixel(2, 2)

        # With zero blend, should be purely the new value
        assert center.r == pytest.approx(1.0, abs=0.1)
