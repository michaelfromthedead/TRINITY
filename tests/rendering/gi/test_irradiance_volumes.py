"""Tests for irradiance volume system (T-GIR-P2.8).

This module provides comprehensive tests for the irradiance volume system
including volume containment, blend weights, priority ordering, cross-fade
smoothness, and GPU buffer packing.

Test Categories:
    - VolumeType: Default priority and blend distance
    - VolumeTransition: State machine and smoothstep blending
    - IrradianceVolume: Containment, signed distance, blend weights
    - IrradianceVolumeManager: Multi-volume blending, GPU buffers
    - Integration: End-to-end volume sampling
"""

from __future__ import annotations

import math
import struct

import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.gi.irradiance_volumes import (
    IrradianceVolume,
    IrradianceVolumeManager,
    VolumeGpuData,
    VolumeState,
    VolumeTransition,
    VolumeType,
    estimate_volume_memory,
    generate_volume_sampling_wgsl,
    recommend_max_volumes,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def unit_bounds() -> AABB:
    """Unit cube centered at origin."""
    return AABB(Vec3(-1.0, -1.0, -1.0), Vec3(1.0, 1.0, 1.0))


@pytest.fixture
def offset_bounds() -> AABB:
    """10x10x10 box at position (10, 0, 0)."""
    return AABB(Vec3(5.0, -5.0, -5.0), Vec3(15.0, 5.0, 5.0))


@pytest.fixture
def local_volume(unit_bounds: AABB) -> IrradianceVolume:
    """Local volume in unit cube."""
    return IrradianceVolume(
        id=1,
        volume_type=VolumeType.LOCAL,
        bounds=unit_bounds,
        priority=3,
        blend_distance=0.5,
    )


@pytest.fixture
def global_volume() -> IrradianceVolume:
    """Global volume covering large area."""
    return IrradianceVolume(
        id=0,
        volume_type=VolumeType.GLOBAL,
        bounds=AABB(Vec3(-100.0, -10.0, -100.0), Vec3(100.0, 50.0, 100.0)),
        priority=0,
        blend_distance=0.0,
    )


@pytest.fixture
def volume_manager() -> IrradianceVolumeManager:
    """Empty volume manager."""
    return IrradianceVolumeManager(max_active_volumes=4)


# ============================================================================
# VolumeType Tests
# ============================================================================


class TestVolumeType:
    """Tests for VolumeType enum."""

    def test_default_priority_global(self):
        """GLOBAL type has lowest default priority."""
        assert VolumeType.GLOBAL.default_priority == 0

    def test_default_priority_exterior(self):
        """EXTERIOR type has priority 1."""
        assert VolumeType.EXTERIOR.default_priority == 1

    def test_default_priority_interior(self):
        """INTERIOR type has priority 2."""
        assert VolumeType.INTERIOR.default_priority == 2

    def test_default_priority_local(self):
        """LOCAL type has highest default priority."""
        assert VolumeType.LOCAL.default_priority == 3

    def test_default_blend_distance_global(self):
        """GLOBAL has no blend distance."""
        assert VolumeType.GLOBAL.default_blend_distance == 0.0

    def test_default_blend_distance_exterior(self):
        """EXTERIOR has wide blend distance."""
        assert VolumeType.EXTERIOR.default_blend_distance == 4.0

    def test_default_blend_distance_interior(self):
        """INTERIOR has medium blend distance."""
        assert VolumeType.INTERIOR.default_blend_distance == 2.0

    def test_default_blend_distance_local(self):
        """LOCAL has tight blend distance."""
        assert VolumeType.LOCAL.default_blend_distance == 1.0


# ============================================================================
# VolumeTransition Tests
# ============================================================================


class TestVolumeTransition:
    """Tests for VolumeTransition state machine."""

    def test_initial_state_inactive(self):
        """Transition starts in INACTIVE state."""
        transition = VolumeTransition()
        assert transition.state == VolumeState.INACTIVE
        assert transition.progress == 0.0

    def test_enter_starts_transition(self):
        """Activating triggers ENTERING state."""
        transition = VolumeTransition()
        transition.update(0.1, should_be_active=True)
        assert transition.state == VolumeState.ENTERING
        assert transition.progress > 0.0

    def test_enter_completes_to_active(self):
        """Full transition completes to ACTIVE state."""
        transition = VolumeTransition(rate=10.0)  # Fast transition
        transition.update(0.2, should_be_active=True)
        assert transition.state == VolumeState.ACTIVE
        assert transition.progress == 1.0

    def test_exit_starts_transition(self):
        """Deactivating triggers EXITING state."""
        transition = VolumeTransition(state=VolumeState.ACTIVE, progress=1.0)
        transition.update(0.1, should_be_active=False)
        assert transition.state == VolumeState.EXITING
        assert transition.progress < 1.0

    def test_exit_completes_to_inactive(self):
        """Full exit completes to INACTIVE state."""
        transition = VolumeTransition(state=VolumeState.ACTIVE, progress=1.0, rate=10.0)
        transition.update(0.2, should_be_active=False)
        assert transition.state == VolumeState.INACTIVE
        assert transition.progress == 0.0

    def test_blend_factor_smoothstep(self):
        """Blend factor uses smoothstep interpolation."""
        transition = VolumeTransition(progress=0.5)
        factor = transition.get_blend_factor()
        # Smoothstep at t=0.5 should be exactly 0.5
        assert abs(factor - 0.5) < 0.001

    def test_blend_factor_zero_at_start(self):
        """Blend factor is 0 at start."""
        transition = VolumeTransition(progress=0.0)
        assert transition.get_blend_factor() == 0.0

    def test_blend_factor_one_at_end(self):
        """Blend factor is 1 at end."""
        transition = VolumeTransition(progress=1.0)
        assert transition.get_blend_factor() == 1.0

    def test_is_contributing_when_active(self):
        """Volume contributes when active."""
        transition = VolumeTransition(state=VolumeState.ACTIVE)
        assert transition.is_contributing()

    def test_is_contributing_when_entering(self):
        """Volume contributes when entering."""
        transition = VolumeTransition(state=VolumeState.ENTERING)
        assert transition.is_contributing()

    def test_not_contributing_when_inactive(self):
        """Volume does not contribute when inactive."""
        transition = VolumeTransition(state=VolumeState.INACTIVE)
        assert not transition.is_contributing()


# ============================================================================
# IrradianceVolume Tests
# ============================================================================


class TestIrradianceVolume:
    """Tests for IrradianceVolume."""

    def test_contains_center(self, local_volume: IrradianceVolume):
        """Volume contains its center point."""
        center = local_volume.bounds.center
        assert local_volume.contains(center)

    def test_contains_corner(self, local_volume: IrradianceVolume):
        """Volume contains corner points."""
        corner = local_volume.bounds.min
        assert local_volume.contains(corner)

    def test_not_contains_outside(self, local_volume: IrradianceVolume):
        """Volume does not contain points outside."""
        outside = Vec3(10.0, 0.0, 0.0)
        assert not local_volume.contains(outside)

    def test_signed_distance_inside_negative(self, local_volume: IrradianceVolume):
        """Signed distance is negative inside volume."""
        center = local_volume.bounds.center
        dist = local_volume.signed_distance(center)
        assert dist < 0.0

    def test_signed_distance_outside_positive(self, local_volume: IrradianceVolume):
        """Signed distance is positive outside volume."""
        outside = Vec3(10.0, 0.0, 0.0)
        dist = local_volume.signed_distance(outside)
        assert dist > 0.0

    def test_signed_distance_on_face_zero(self, local_volume: IrradianceVolume):
        """Signed distance is zero on face."""
        on_face = Vec3(1.0, 0.0, 0.0)  # Right face
        dist = local_volume.signed_distance(on_face)
        assert abs(dist) < 0.01

    def test_blend_weight_center_is_one(self, local_volume: IrradianceVolume):
        """Blend weight is 1.0 at center (beyond blend distance from edge)."""
        center = local_volume.bounds.center
        weight = local_volume.get_blend_weight(center)
        assert weight == 1.0

    def test_blend_weight_outside_is_zero(self, local_volume: IrradianceVolume):
        """Blend weight is 0.0 outside volume."""
        outside = Vec3(10.0, 0.0, 0.0)
        weight = local_volume.get_blend_weight(outside)
        assert weight == 0.0

    def test_blend_weight_at_boundary_half(self):
        """Blend weight at boundary midpoint is 0.5."""
        # Create volume with known blend distance
        volume = IrradianceVolume(
            id=1,
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3(-5.0, -5.0, -5.0), Vec3(5.0, 5.0, 5.0)),
            blend_distance=2.0,
        )
        # Point at half blend distance from boundary (1m inside with 2m blend)
        # At x=4.0, signed_dist = -1.0, blend_distance = 2.0
        # t = 1.0/2.0 = 0.5, smoothstep(0.5) = 0.5
        point = Vec3(4.0, 0.0, 0.0)
        weight = volume.get_blend_weight(point)
        # The blend region is from edge (x=5) to blend_distance inside (x=3)
        # At x=4.0, we're 1m inside the boundary, so t = 1.0/2.0 = 0.5
        # smoothstep(0.5) = 0.5 * 0.5 * (3 - 2*0.5) = 0.25 * 2 = 0.5
        assert abs(weight - 0.5) < 0.05

    def test_blend_weight_disabled_volume(self, local_volume: IrradianceVolume):
        """Disabled volume has zero weight."""
        local_volume.enabled = False
        center = local_volume.bounds.center
        weight = local_volume.get_blend_weight(center)
        assert weight == 0.0

    def test_effective_weight_with_transition(self, local_volume: IrradianceVolume):
        """Effective weight accounts for transition state."""
        local_volume.transition.progress = 0.5
        local_volume.transition.state = VolumeState.ENTERING
        center = local_volume.bounds.center

        base_weight = local_volume.get_blend_weight(center)
        effective_weight = local_volume.get_effective_weight(center)

        assert effective_weight < base_weight
        assert effective_weight == base_weight * local_volume.transition.get_blend_factor()

    def test_priority_default_from_type(self):
        """Priority defaults to volume type's default."""
        volume = IrradianceVolume(
            id=1,
            volume_type=VolumeType.INTERIOR,
            bounds=AABB(Vec3.zero(), Vec3.one()),
        )
        assert volume.priority == VolumeType.INTERIOR.default_priority

    def test_priority_explicit_override(self):
        """Explicit priority overrides default."""
        volume = IrradianceVolume(
            id=1,
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3.zero(), Vec3.one()),
            priority=10,
        )
        assert volume.priority == 10


# ============================================================================
# IrradianceVolumeManager Tests
# ============================================================================


class TestIrradianceVolumeManager:
    """Tests for IrradianceVolumeManager."""

    def test_add_volume_assigns_id(self, volume_manager: IrradianceVolumeManager):
        """Adding volume assigns unique ID."""
        volume = IrradianceVolume(
            id=999,  # Should be overwritten
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3.zero(), Vec3.one()),
        )
        assigned_id = volume_manager.add_volume(volume)
        assert assigned_id == 0
        assert volume.id == 0

    def test_add_multiple_volumes(self, volume_manager: IrradianceVolumeManager):
        """Multiple volumes get sequential IDs."""
        ids = []
        for _ in range(3):
            volume = IrradianceVolume(
                id=0,
                volume_type=VolumeType.LOCAL,
                bounds=AABB(Vec3.zero(), Vec3.one()),
            )
            ids.append(volume_manager.add_volume(volume))

        assert ids == [0, 1, 2]

    def test_remove_volume(self, volume_manager: IrradianceVolumeManager):
        """Remove volume by ID."""
        volume = IrradianceVolume(
            id=0,
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3.zero(), Vec3.one()),
        )
        vol_id = volume_manager.add_volume(volume)
        assert volume_manager.remove_volume(vol_id)
        assert len(volume_manager.volumes) == 0

    def test_remove_nonexistent_volume(self, volume_manager: IrradianceVolumeManager):
        """Remove returns False for nonexistent ID."""
        assert not volume_manager.remove_volume(999)

    def test_get_volume_by_id(self, volume_manager: IrradianceVolumeManager):
        """Get volume by ID."""
        volume = IrradianceVolume(
            id=0,
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3.zero(), Vec3.one()),
        )
        vol_id = volume_manager.add_volume(volume)
        retrieved = volume_manager.get_volume(vol_id)
        assert retrieved is volume

    def test_get_active_volumes_sorted_by_priority(
        self, volume_manager: IrradianceVolumeManager
    ):
        """Active volumes are sorted by priority (highest first)."""
        # Create overlapping volumes with different priorities
        bounds = AABB(Vec3(-10.0, -10.0, -10.0), Vec3(10.0, 10.0, 10.0))

        low_priority = IrradianceVolume(
            id=0, volume_type=VolumeType.GLOBAL, bounds=bounds, priority=0
        )
        mid_priority = IrradianceVolume(
            id=0, volume_type=VolumeType.INTERIOR, bounds=bounds, priority=2
        )
        high_priority = IrradianceVolume(
            id=0, volume_type=VolumeType.LOCAL, bounds=bounds, priority=5
        )

        volume_manager.add_volume(low_priority)
        volume_manager.add_volume(high_priority)
        volume_manager.add_volume(mid_priority)

        active = volume_manager.get_active_volumes(Vec3.zero())

        assert len(active) == 3
        assert active[0].priority == 5
        assert active[1].priority == 2
        assert active[2].priority == 0

    def test_max_active_volumes_limit(self, volume_manager: IrradianceVolumeManager):
        """Active volumes limited to max_active_volumes."""
        volume_manager.max_active_volumes = 2
        bounds = AABB(Vec3(-10.0, -10.0, -10.0), Vec3(10.0, 10.0, 10.0))

        for i in range(5):
            volume = IrradianceVolume(
                id=0, volume_type=VolumeType.LOCAL, bounds=bounds, priority=i
            )
            volume_manager.add_volume(volume)

        active = volume_manager.get_active_volumes(Vec3.zero())
        assert len(active) == 2
        # Should keep highest priority
        assert active[0].priority == 4
        assert active[1].priority == 3

    def test_active_volumes_excludes_disabled(
        self, volume_manager: IrradianceVolumeManager
    ):
        """Disabled volumes not included in active list."""
        bounds = AABB(Vec3(-10.0, -10.0, -10.0), Vec3(10.0, 10.0, 10.0))

        enabled = IrradianceVolume(
            id=0, volume_type=VolumeType.LOCAL, bounds=bounds, enabled=True
        )
        disabled = IrradianceVolume(
            id=0, volume_type=VolumeType.LOCAL, bounds=bounds, enabled=False
        )

        volume_manager.add_volume(enabled)
        volume_manager.add_volume(disabled)

        active = volume_manager.get_active_volumes(Vec3.zero())
        assert len(active) == 1
        assert active[0] is enabled

    def test_create_volume_convenience(self, volume_manager: IrradianceVolumeManager):
        """create_volume convenience method works."""
        bounds = AABB(Vec3.zero(), Vec3.one())
        volume = volume_manager.create_volume(
            volume_type=VolumeType.INTERIOR,
            bounds=bounds,
            priority=5,
            blend_distance=3.0,
        )

        assert volume.id == 0
        assert volume.volume_type == VolumeType.INTERIOR
        assert volume.priority == 5
        assert volume.blend_distance == 3.0
        assert volume in volume_manager.volumes

    def test_sample_irradiance_empty(self, volume_manager: IrradianceVolumeManager):
        """Empty manager returns zero irradiance."""
        irradiance, confidence = volume_manager.sample_irradiance(
            Vec3.zero(), Vec3.unit_y()
        )
        assert irradiance == Vec3.zero()
        assert confidence == 0.0

    def test_sample_irradiance_single_volume(
        self, volume_manager: IrradianceVolumeManager
    ):
        """Single volume sampling works."""
        bounds = AABB(Vec3(-10.0, -10.0, -10.0), Vec3(10.0, 10.0, 10.0))
        volume = volume_manager.create_volume(VolumeType.LOCAL, bounds)
        volume.transition.state = VolumeState.ACTIVE
        volume.transition.progress = 1.0

        irradiance, confidence = volume_manager.sample_irradiance(
            Vec3.zero(), Vec3.unit_y()
        )
        # Without a grid, confidence should still reflect weight
        assert confidence >= 0.0

    def test_update_caches_active(self, volume_manager: IrradianceVolumeManager):
        """Update method caches active volumes."""
        bounds = AABB(Vec3(-10.0, -10.0, -10.0), Vec3(10.0, 10.0, 10.0))
        volume_manager.create_volume(VolumeType.LOCAL, bounds)

        volume_manager.update(Vec3.zero(), dt=0.016)

        assert len(volume_manager._active_cache) == 1

    def test_gpu_buffer_dirty_after_add(self, volume_manager: IrradianceVolumeManager):
        """GPU buffer marked dirty after adding volume."""
        bounds = AABB(Vec3.zero(), Vec3.one())
        volume_manager.create_volume(VolumeType.LOCAL, bounds)
        assert volume_manager.needs_gpu_upload()

    def test_gpu_buffer_size(self, volume_manager: IrradianceVolumeManager):
        """GPU buffer has correct size."""
        bounds = AABB(Vec3.zero(), Vec3.one())
        volume_manager.create_volume(VolumeType.LOCAL, bounds)
        volume_manager.update(Vec3.zero(), dt=0.016)

        buffer = volume_manager.build_gpu_buffer()
        # Header (16) + max_volumes * VolumeGpuData (48)
        expected_size = 16 + 4 * 48
        assert len(buffer) == expected_size

    def test_get_statistics(self, volume_manager: IrradianceVolumeManager):
        """Statistics method returns correct counts."""
        bounds = AABB(Vec3.zero(), Vec3.one())
        volume_manager.create_volume(VolumeType.LOCAL, bounds)
        volume_manager.create_volume(VolumeType.INTERIOR, bounds)
        volume_manager.update(Vec3.zero(), dt=0.016)

        stats = volume_manager.get_statistics()
        assert stats["total_volumes"] == 2
        assert stats["by_type"]["LOCAL"] == 1
        assert stats["by_type"]["INTERIOR"] == 1


# ============================================================================
# Cross-Fade Blending Tests
# ============================================================================


class TestCrossFadeBlending:
    """Tests for cross-fade blending behavior."""

    def test_smooth_transition_no_discontinuity(self):
        """Blend weight has no discontinuity at boundary."""
        volume = IrradianceVolume(
            id=1,
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3(-5.0, -5.0, -5.0), Vec3(5.0, 5.0, 5.0)),
            blend_distance=2.0,
        )

        # Sample weights across the boundary with finer steps
        # Volume edge is at x=5, blend region is x=3 to x=5
        weights = []
        step = 0.1  # Finer step size for smoother testing
        for i in range(40):  # From x=3.0 to x=7.0
            x = 3.0 + i * step
            point = Vec3(x, 0.0, 0.0)
            weights.append(volume.get_blend_weight(point))

        # Check for smoothness: no jump greater than step-appropriate threshold
        # With 0.1 step over 2.0 blend distance, max derivative is ~1.5
        # So max change per step is ~0.15
        max_diff = 0.0
        for i in range(1, len(weights)):
            diff = abs(weights[i] - weights[i - 1])
            max_diff = max(max_diff, diff)
            # Allow slightly larger jumps due to discrete sampling
            assert diff < 0.25, f"Discontinuity at step {i}: {diff}"

    def test_cross_fade_derivative_continuous(self):
        """Blend weight derivative is continuous (smoothstep property)."""
        volume = IrradianceVolume(
            id=1,
            volume_type=VolumeType.LOCAL,
            bounds=AABB(Vec3(-5.0, -5.0, -5.0), Vec3(5.0, 5.0, 5.0)),
            blend_distance=2.0,
        )

        # Sample many points and compute numerical derivative
        prev_weight = None
        prev_deriv = None

        for i in range(100):
            t = i / 99.0  # 0 to 1
            x = 3.0 + t * 4.0  # 3m to 7m (spanning blend region)
            point = Vec3(x, 0.0, 0.0)
            weight = volume.get_blend_weight(point)

            if prev_weight is not None:
                deriv = (weight - prev_weight) * 99.0 / 4.0  # Approximate derivative

                if prev_deriv is not None and i > 10 and i < 90:
                    # Derivative should change smoothly
                    deriv_change = abs(deriv - prev_deriv)
                    assert deriv_change < 0.5, f"Derivative jump at {i}"

                prev_deriv = deriv
            prev_weight = weight

    def test_interior_exterior_transition(self):
        """Interior and exterior volumes blend smoothly."""
        manager = IrradianceVolumeManager(max_active_volumes=4)

        exterior = IrradianceVolume(
            id=0,
            volume_type=VolumeType.EXTERIOR,
            bounds=AABB(Vec3(-100.0, -10.0, -100.0), Vec3(100.0, 50.0, 100.0)),
            blend_distance=5.0,
            priority=1,
        )
        interior = IrradianceVolume(
            id=1,
            volume_type=VolumeType.INTERIOR,
            bounds=AABB(Vec3(-10.0, -5.0, -10.0), Vec3(10.0, 5.0, 10.0)),
            blend_distance=2.0,
            priority=2,
        )

        manager.add_volume(exterior)
        manager.add_volume(interior)

        # At center of interior (deep inside, beyond blend_distance from edge)
        # Interior is 20x10x20, center is (0,0,0), edge at 10m in x/z, 5m in y
        # With blend_distance=2.0, need to be >2m from all edges for weight=1.0
        # Center is 10m from x/z edges and 5m from y edges, all > 2m
        interior_weight = interior.get_blend_weight(Vec3.zero())
        assert interior_weight == 1.0

        # At edge of interior (1m from edge in x), both contribute
        # Interior edge at x=10, so x=9 is 1m inside
        # With blend_distance=2.0, t = 1/2 = 0.5, smoothstep(0.5) = 0.5
        edge_point = Vec3(9.0, 0.0, 0.0)
        int_w = interior.get_blend_weight(edge_point)
        ext_w = exterior.get_blend_weight(edge_point)
        assert 0.0 < int_w < 1.0, f"Expected interior blend weight in (0,1), got {int_w}"
        assert ext_w == 1.0


# ============================================================================
# GPU Data Tests
# ============================================================================


class TestVolumeGpuData:
    """Tests for VolumeGpuData GPU buffer packing."""

    def test_gpu_data_size(self):
        """GPU data packs to correct size."""
        data = VolumeGpuData(
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(1.0, 1.0, 1.0),
            priority=1,
            blend_distance=2.0,
            grid_index=0,
            volume_type=1,
            transition_factor=1.0,
        )
        packed = data.to_bytes()
        assert len(packed) == 48

    def test_gpu_data_from_volume(self, local_volume: IrradianceVolume):
        """GPU data created from volume correctly."""
        gpu_data = VolumeGpuData.from_volume(local_volume, grid_index=2)

        assert gpu_data.bounds_min == (
            local_volume.bounds.min.x,
            local_volume.bounds.min.y,
            local_volume.bounds.min.z,
        )
        assert gpu_data.priority == local_volume.priority
        assert gpu_data.grid_index == 2
        assert gpu_data.volume_type == local_volume.volume_type.value

    def test_gpu_data_unpacks_correctly(self):
        """Packed GPU data can be unpacked."""
        data = VolumeGpuData(
            bounds_min=(1.0, 2.0, 3.0),
            bounds_max=(4.0, 5.0, 6.0),
            priority=7,
            blend_distance=8.0,
            grid_index=9,
            volume_type=10,
            transition_factor=0.5,
        )
        packed = data.to_bytes()

        # Unpack and verify
        unpacked = struct.unpack("<3fi3ffiIff", packed)
        assert unpacked[0:3] == (1.0, 2.0, 3.0)  # bounds_min
        assert unpacked[3] == 7  # priority
        assert unpacked[4:7] == (4.0, 5.0, 6.0)  # bounds_max
        assert unpacked[7] == 8.0  # blend_distance
        assert unpacked[8] == 9  # grid_index
        assert unpacked[9] == 10  # volume_type
        assert abs(unpacked[10] - 0.5) < 0.001  # transition_factor


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_recommend_max_volumes_low(self):
        """Low tier recommends 2 volumes."""
        assert recommend_max_volumes("low") == 2

    def test_recommend_max_volumes_medium(self):
        """Medium tier recommends 4 volumes."""
        assert recommend_max_volumes("medium") == 4

    def test_recommend_max_volumes_high(self):
        """High tier recommends 6 volumes."""
        assert recommend_max_volumes("high") == 6

    def test_recommend_max_volumes_ultra(self):
        """Ultra tier recommends 8 volumes."""
        assert recommend_max_volumes("ultra") == 8

    def test_recommend_max_volumes_unknown(self):
        """Unknown tier defaults to 4."""
        assert recommend_max_volumes("unknown") == 4

    def test_estimate_volume_memory(self):
        """Memory estimate is reasonable."""
        memory = estimate_volume_memory(volume_count=4, probes_per_volume=8192)
        # 4 volumes * (8192 * (192+16) + 64 + 48) + 16 header
        # = 4 * (8192 * 208 + 112) + 16
        # = 4 * 1703984 + 16 = 6,815,952
        assert memory > 6_000_000
        assert memory < 10_000_000

    def test_generate_wgsl_produces_code(self):
        """WGSL generator produces non-empty code."""
        wgsl = generate_volume_sampling_wgsl()
        assert len(wgsl) > 100
        assert "VolumeGpu" in wgsl
        assert "volume_signed_distance" in wgsl
        assert "sample_volume_irradiance" in wgsl


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete volume system."""

    def test_full_update_cycle(self):
        """Full update cycle works correctly."""
        manager = IrradianceVolumeManager(max_active_volumes=4)

        # Create several volumes
        manager.create_volume(
            VolumeType.GLOBAL,
            AABB(Vec3(-100.0, -10.0, -100.0), Vec3(100.0, 50.0, 100.0)),
        )
        manager.create_volume(
            VolumeType.INTERIOR,
            AABB(Vec3(-10.0, 0.0, -10.0), Vec3(10.0, 5.0, 10.0)),
        )
        manager.create_volume(
            VolumeType.LOCAL,
            AABB(Vec3(-2.0, 0.0, -2.0), Vec3(2.0, 3.0, 2.0)),
        )

        # Simulate frames
        camera_pos = Vec3.zero()
        for _ in range(10):
            manager.update(camera_pos, dt=0.016)

        # Check state
        stats = manager.get_statistics()
        assert stats["total_volumes"] == 3
        assert stats["active_volumes"] <= 4

    def test_camera_movement_affects_active(self):
        """Camera movement changes active volumes."""
        manager = IrradianceVolumeManager(max_active_volumes=4)

        # Volume at origin
        manager.create_volume(
            VolumeType.LOCAL,
            AABB(Vec3(-5.0, -5.0, -5.0), Vec3(5.0, 5.0, 5.0)),
        )

        # Camera at origin - volume is active
        manager.update(Vec3.zero(), dt=0.016)
        assert len(manager._active_cache) == 1

        # Camera far away - volume not active
        manager.update(Vec3(100.0, 0.0, 0.0), dt=0.016)
        assert len(manager._active_cache) == 0

    def test_overlapping_volumes_blend(self):
        """Overlapping volumes blend correctly."""
        manager = IrradianceVolumeManager(max_active_volumes=4)

        # Create two overlapping volumes
        vol1 = manager.create_volume(
            VolumeType.EXTERIOR,
            AABB(Vec3(-20.0, -5.0, -20.0), Vec3(20.0, 10.0, 20.0)),
            priority=1,
        )
        vol2 = manager.create_volume(
            VolumeType.INTERIOR,
            AABB(Vec3(-5.0, 0.0, -5.0), Vec3(5.0, 5.0, 5.0)),
            priority=2,
        )

        # Activate both fully
        vol1.transition.state = VolumeState.ACTIVE
        vol1.transition.progress = 1.0
        vol2.transition.state = VolumeState.ACTIVE
        vol2.transition.progress = 1.0

        manager.update(Vec3.zero(), dt=0.016)

        # Both should contribute
        active = manager.get_active_volumes(Vec3.zero())
        assert len(active) == 2

    def test_priority_override(self):
        """Higher priority volumes override lower priority."""
        manager = IrradianceVolumeManager(max_active_volumes=4)

        bounds = AABB(Vec3(-10.0, -10.0, -10.0), Vec3(10.0, 10.0, 10.0))

        low = manager.create_volume(VolumeType.GLOBAL, bounds, priority=0)
        high = manager.create_volume(VolumeType.LOCAL, bounds, priority=10)

        low.transition.state = VolumeState.ACTIVE
        low.transition.progress = 1.0
        high.transition.state = VolumeState.ACTIVE
        high.transition.progress = 1.0

        active = manager.get_active_volumes(Vec3.zero())

        # High priority should be first
        assert active[0].priority == 10
        assert active[1].priority == 0
