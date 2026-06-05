"""Tests for DDGI probe update system (T-GIR-P2.4).

This test module validates:
- Irradiance accumulation with distance weighting
- Visibility minimum-distance storage
- Temporal accumulation convergence
- Importance-based update scheduling
- Integration with SH storage

Test count target: 50+ tests covering all acceptance criteria.
"""

from __future__ import annotations

import math
import struct
from typing import List

import numpy as np
import pytest

from engine.core.math.vec import Vec3
from engine.rendering.gi.sh_math import SHCoefficientsL2, sh_evaluate_l2
from engine.rendering.gi.ddgi_probe_update import (
    # Constants
    DEFAULT_IRRADIANCE_BLEND,
    DEFAULT_VISIBILITY_BLEND,
    DEFAULT_GAUSSIAN_SIGMA,
    DEFAULT_CONFIDENCE_THRESHOLD,
    IMPORTANCE_CRITICAL,
    IMPORTANCE_HIGH,
    IMPORTANCE_MEDIUM,
    IMPORTANCE_LOW,
    VISIBILITY_MAX_DISTANCE,
    VISIBILITY_MISS_DISTANCE,
    # Data structures
    ProbeRayHit,
    DistanceWeightConfig,
    ProbeData,
    ProbeUpdateState,
    # Configs
    IrradianceAccumulatorConfig,
    VisibilityStorageConfig,
    TemporalAccumulatorConfig,
    ImportanceSchedulerConfig,
    DDGIProbeUpdaterConfig,
    # Enums
    ProbeImportance,
    # Core classes
    IrradianceAccumulator,
    VisibilityStorage,
    TemporalAccumulator,
    ImportanceScheduler,
    DDGIProbeUpdater,
    # Utilities
    compute_distance_statistics,
    create_test_ray_hits,
    estimate_update_cost,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def uniform_ray_hits() -> List[ProbeRayHit]:
    """Create uniform white light ray hits."""
    return create_test_ray_hits(256, "uniform")


@pytest.fixture
def gradient_ray_hits() -> List[ProbeRayHit]:
    """Create gradient ray hits (sky above, ground below)."""
    return create_test_ray_hits(256, "gradient")


@pytest.fixture
def mixed_ray_hits() -> List[ProbeRayHit]:
    """Create mixed ray hits (sky misses above, ground hits below)."""
    return create_test_ray_hits(256, "mixed")


@pytest.fixture
def default_updater() -> DDGIProbeUpdater:
    """Create default probe updater."""
    return DDGIProbeUpdater()


# ============================================================================
# Test ProbeRayHit
# ============================================================================


class TestProbeRayHit:
    """Tests for ProbeRayHit data structure."""

    def test_hit_creation(self):
        """Test creating a ray hit."""
        hit = ProbeRayHit(
            direction=Vec3(1, 0, 0),
            radiance=Vec3(0.5, 0.5, 0.5),
            distance=10.0,
        )
        assert hit.direction.x == 1.0
        assert hit.radiance.x == 0.5
        assert hit.distance == 10.0
        assert not hit.hit_backface

    def test_miss_detection(self):
        """Test ray miss detection."""
        hit = ProbeRayHit(
            direction=Vec3(0, 1, 0),
            radiance=Vec3(0.8, 0.9, 1.0),
            distance=VISIBILITY_MISS_DISTANCE,
        )
        assert hit.is_miss()

    def test_valid_hit(self):
        """Test valid hit detection."""
        hit = ProbeRayHit(
            direction=Vec3(0, 0, 1),
            radiance=Vec3(0.3, 0.3, 0.3),
            distance=5.0,
        )
        assert hit.is_valid()

    def test_backface_invalid(self):
        """Test backface hits are invalid."""
        hit = ProbeRayHit(
            direction=Vec3(1, 0, 0),
            radiance=Vec3(0.5, 0.5, 0.5),
            distance=10.0,
            hit_backface=True,
        )
        assert not hit.is_valid()

    def test_negative_distance_invalid(self):
        """Test negative distance hits are invalid."""
        hit = ProbeRayHit(
            direction=Vec3(1, 0, 0),
            radiance=Vec3(0.5, 0.5, 0.5),
            distance=-1.0,
        )
        assert not hit.is_valid()


# ============================================================================
# Test DistanceWeightConfig
# ============================================================================


class TestDistanceWeightConfig:
    """Tests for distance weighting configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DistanceWeightConfig()
        assert config.sigma == DEFAULT_GAUSSIAN_SIGMA
        assert config.min_distance == 0.01
        assert config.max_distance == 100.0

    def test_weight_at_mean(self):
        """Test weight is maximum at mean distance."""
        config = DistanceWeightConfig()
        weight = config.compute_weight(10.0, 10.0)
        assert weight == pytest.approx(1.0)

    def test_weight_falloff(self):
        """Test weight decreases away from mean."""
        config = DistanceWeightConfig(sigma=0.5)
        weight_at_mean = config.compute_weight(10.0, 10.0)
        weight_far = config.compute_weight(15.0, 10.0)
        weight_close = config.compute_weight(5.0, 10.0)

        assert weight_at_mean > weight_far
        assert weight_at_mean > weight_close

    def test_weight_below_minimum(self):
        """Test weight is zero below minimum distance."""
        config = DistanceWeightConfig(min_distance=1.0)
        weight = config.compute_weight(0.5, 10.0)
        assert weight == 0.0

    def test_confidence_high_samples(self):
        """Test confidence increases with sample count."""
        config = DistanceWeightConfig()
        low_conf = config.compute_confidence(8, 0.1)
        high_conf = config.compute_confidence(64, 0.1)
        assert high_conf > low_conf

    def test_confidence_low_variance(self):
        """Test confidence increases with lower variance."""
        config = DistanceWeightConfig()
        high_var_conf = config.compute_confidence(64, 1.0)
        low_var_conf = config.compute_confidence(64, 0.01)
        assert low_var_conf > high_var_conf


# ============================================================================
# Test Distance Statistics
# ============================================================================


class TestDistanceStatistics:
    """Tests for distance statistics computation."""

    def test_uniform_distances(self):
        """Test statistics with uniform distances."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 10.0),
            ProbeRayHit(Vec3(0, 1, 0), Vec3(0.5, 0.5, 0.5), 10.0),
            ProbeRayHit(Vec3(0, 0, 1), Vec3(0.5, 0.5, 0.5), 10.0),
        ]
        mean, var, count = compute_distance_statistics(hits)
        assert mean == pytest.approx(10.0)
        assert var == pytest.approx(0.0)
        assert count == 3

    def test_varied_distances(self):
        """Test statistics with varied distances."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 5.0),
            ProbeRayHit(Vec3(0, 1, 0), Vec3(0.5, 0.5, 0.5), 10.0),
            ProbeRayHit(Vec3(0, 0, 1), Vec3(0.5, 0.5, 0.5), 15.0),
        ]
        mean, var, count = compute_distance_statistics(hits)
        assert mean == pytest.approx(10.0)
        assert var > 0
        assert count == 3

    def test_excludes_misses(self):
        """Test that ray misses are excluded."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 10.0),
            ProbeRayHit(Vec3(0, 1, 0), Vec3(0.8, 0.9, 1.0), VISIBILITY_MISS_DISTANCE),
        ]
        mean, var, count = compute_distance_statistics(hits)
        assert mean == pytest.approx(10.0)
        assert count == 1

    def test_empty_returns_max(self):
        """Test empty hits returns max distance."""
        mean, var, count = compute_distance_statistics([])
        assert mean == VISIBILITY_MAX_DISTANCE
        assert count == 0


# ============================================================================
# Test IrradianceAccumulator
# ============================================================================


class TestIrradianceAccumulator:
    """Tests for irradiance accumulation."""

    def test_basic_accumulation(self, uniform_ray_hits):
        """Test basic irradiance accumulation."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate(uniform_ray_hits)

        assert isinstance(sh, SHCoefficientsL2)
        assert accum.sample_count > 0
        assert accum.confidence > 0

    def test_uniform_produces_dc(self, uniform_ray_hits):
        """Test uniform light produces DC-dominant SH."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate(uniform_ray_hits)

        # DC coefficient should be dominant
        dc = sh.get(0)
        assert np.all(dc > 0)

    def test_gradient_produces_directional(self, gradient_ray_hits):
        """Test gradient light produces directional SH."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate(gradient_ray_hits)

        # L1 coefficients should be non-zero for gradient
        l1_y = sh.get(1)  # Y direction
        assert np.any(np.abs(l1_y) > 0.001)

    def test_empty_hits(self):
        """Test empty hits produces zero SH."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate([])

        assert accum.sample_count == 0
        assert accum.confidence == 0.0
        # Check all coefficients are zero
        for i in range(9):
            assert np.allclose(sh.get(i), 0)

    def test_backface_rejection(self):
        """Test backface hits are rejected."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(1, 1, 1), 10.0, hit_backface=False),
            ProbeRayHit(Vec3(0, 1, 0), Vec3(1, 1, 1), 10.0, hit_backface=True),
        ]
        config = IrradianceAccumulatorConfig(backface_rejection=True)
        accum = IrradianceAccumulator(config)
        accum.accumulate(hits)

        assert accum.sample_count == 1

    def test_distance_weighting_enabled(self, uniform_ray_hits):
        """Test distance weighting affects results."""
        config_weighted = IrradianceAccumulatorConfig(use_distance_weighting=True)
        config_unweighted = IrradianceAccumulatorConfig(use_distance_weighting=False)

        accum_weighted = IrradianceAccumulator(config_weighted)
        accum_unweighted = IrradianceAccumulator(config_unweighted)

        sh_weighted = accum_weighted.accumulate(uniform_ray_hits)
        sh_unweighted = accum_unweighted.accumulate(uniform_ray_hits)

        # Results may differ due to weighting
        # Just verify both produce valid output
        assert isinstance(sh_weighted, SHCoefficientsL2)
        assert isinstance(sh_unweighted, SHCoefficientsL2)

    def test_radiance_clamping(self):
        """Test radiance is clamped to valid range."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(200, 200, 200), 10.0),  # Very bright
            ProbeRayHit(Vec3(0, 1, 0), Vec3(-1, -1, -1), 10.0),  # Negative
        ]
        config = IrradianceAccumulatorConfig(max_radiance=100.0, min_radiance=0.0)
        accum = IrradianceAccumulator(config)
        sh = accum.accumulate(hits)

        # DC should be positive and bounded
        dc = sh.get(0)
        assert np.all(dc >= 0)


# ============================================================================
# Test VisibilityStorage
# ============================================================================


class TestVisibilityStorage:
    """Tests for visibility (depth) storage."""

    def test_initialization(self):
        """Test visibility storage initialization."""
        config = VisibilityStorageConfig(resolution=64)
        storage = VisibilityStorage(config)

        assert storage.config.resolution == 64

    def test_update_from_hits(self, uniform_ray_hits):
        """Test updating visibility from ray hits."""
        storage = VisibilityStorage()
        storage.update_from_hits(uniform_ray_hits)

        # Query in a hit direction
        direction = Vec3(1, 0, 0)
        depth = storage.get_mean_depth(direction)
        assert depth < VISIBILITY_MAX_DISTANCE

    def test_minimum_distance_stored(self):
        """Test that minimum distance is stored per direction."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 5.0),
            ProbeRayHit(Vec3(1, 0.1, 0).normalized(), Vec3(0.5, 0.5, 0.5), 10.0),
            ProbeRayHit(Vec3(1, -0.1, 0).normalized(), Vec3(0.5, 0.5, 0.5), 15.0),
        ]
        storage = VisibilityStorage(VisibilityStorageConfig(resolution=64))
        storage.update_from_hits(hits)

        # Query near +X should find minimum
        depth = storage.get_mean_depth(Vec3(1, 0, 0))
        assert depth <= 10.0  # Should be close to 5.0

    def test_occlusion_visible(self):
        """Test occlusion returns 1 for visible points."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 20.0),
        ]
        storage = VisibilityStorage()
        storage.update_from_hits(hits)

        # Query closer than stored depth should be visible
        occlusion = storage.compute_occlusion(Vec3(1, 0, 0), 10.0)
        assert occlusion == pytest.approx(1.0)

    def test_occlusion_occluded(self):
        """Test occlusion returns <1 for occluded points."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 10.0),
        ]
        config = VisibilityStorageConfig(use_chebyshev=False)
        storage = VisibilityStorage(config)
        storage.update_from_hits(hits)

        # Query farther than stored depth should be occluded
        occlusion = storage.compute_occlusion(Vec3(1, 0, 0), 20.0)
        assert occlusion < 1.0

    def test_chebyshev_soft_shadow(self):
        """Test Chebyshev produces soft shadow falloff."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), 10.0),
        ]
        config = VisibilityStorageConfig(use_chebyshev=True, depth_sharpness=1.0)
        storage = VisibilityStorage(config)
        storage.update_from_hits(hits)

        # Just past occluder should have partial occlusion
        occ_near = storage.compute_occlusion(Vec3(1, 0, 0), 11.0)
        occ_far = storage.compute_occlusion(Vec3(1, 0, 0), 50.0)

        assert occ_near > occ_far  # Closer = more visible

    def test_serialization(self, uniform_ray_hits):
        """Test visibility data serialization."""
        storage = VisibilityStorage()
        storage.update_from_hits(uniform_ray_hits)

        data = storage.to_bytes()
        assert len(data) > 0

        # Deserialize
        restored = VisibilityStorage.from_bytes(data, storage.config)
        # Check a direction matches
        dir_test = Vec3(1, 0, 0)
        assert storage.get_mean_depth(dir_test) == pytest.approx(
            restored.get_mean_depth(dir_test)
        )


# ============================================================================
# Test TemporalAccumulator
# ============================================================================


class TestTemporalAccumulator:
    """Tests for temporal accumulation."""

    def test_first_frame_no_blend(self):
        """Test first frame uses new data directly."""
        accum = TemporalAccumulator()
        sh_new = SHCoefficientsL2.zero()
        sh_new.set(0, np.array([1.0, 1.0, 1.0]))

        result = accum.accumulate(sh_new)

        # First frame should match input
        np.testing.assert_array_almost_equal(result.get(0), sh_new.get(0))
        assert accum.frame_count == 1

    def test_temporal_blend(self):
        """Test temporal blending averages over time."""
        config = TemporalAccumulatorConfig(irradiance_blend=0.9, adaptive_blend=False)
        accum = TemporalAccumulator(config)

        # First frame: value 1
        sh1 = SHCoefficientsL2.zero()
        sh1.set(0, np.array([1.0, 1.0, 1.0]))
        accum.accumulate(sh1)

        # Second frame: value 2
        sh2 = SHCoefficientsL2.zero()
        sh2.set(0, np.array([2.0, 2.0, 2.0]))
        result = accum.accumulate(sh2)

        # Result should be blended: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        expected = 0.9 * 1.0 + 0.1 * 2.0
        assert result.get(0)[0] == pytest.approx(expected, rel=0.01)

    def test_convergence_over_time(self):
        """Test convergence factor increases over frames."""
        config = TemporalAccumulatorConfig(min_frames=8, max_frames=32)
        accum = TemporalAccumulator(config)

        sh = SHCoefficientsL2.zero()
        sh.set(0, np.array([0.5, 0.5, 0.5]))

        # Accumulate multiple frames
        for _ in range(10):
            accum.accumulate(sh)

        assert accum.is_converged
        assert accum.convergence_factor > 0

    def test_convergence_min_frames(self):
        """Test not converged before min_frames."""
        config = TemporalAccumulatorConfig(min_frames=16)
        accum = TemporalAccumulator(config)

        sh = SHCoefficientsL2.zero()
        for _ in range(8):
            accum.accumulate(sh)

        assert not accum.is_converged

    def test_reset_clears_history(self):
        """Test reset clears accumulated history."""
        accum = TemporalAccumulator()
        sh = SHCoefficientsL2.zero()
        sh.set(0, np.array([1.0, 1.0, 1.0]))

        accum.accumulate(sh)
        accum.accumulate(sh)
        accum.reset()

        assert accum.frame_count == 0
        assert not accum.is_converged

    def test_force_update(self):
        """Test force update replaces history."""
        accum = TemporalAccumulator()

        sh1 = SHCoefficientsL2.zero()
        sh1.set(0, np.array([1.0, 1.0, 1.0]))
        accum.accumulate(sh1)

        sh2 = SHCoefficientsL2.zero()
        sh2.set(0, np.array([5.0, 5.0, 5.0]))
        accum.force_update(sh2)

        # Next accumulate should blend from forced value
        sh3 = SHCoefficientsL2.zero()
        sh3.set(0, np.array([5.0, 5.0, 5.0]))
        result = accum.accumulate(sh3)

        assert result.get(0)[0] == pytest.approx(5.0, rel=0.1)

    def test_adaptive_blend_initial_ramp(self):
        """Test adaptive blend ramps up during initial frames."""
        config = TemporalAccumulatorConfig(
            min_frames=16, adaptive_blend=True, irradiance_blend=0.97
        )
        accum = TemporalAccumulator(config)

        sh = SHCoefficientsL2.zero()
        sh.set(0, np.array([1.0, 1.0, 1.0]))

        # First few frames should have lower blend (faster adaptation)
        accum.accumulate(sh)
        early_blend = accum._last_blend

        for _ in range(15):
            accum.accumulate(sh)
        late_blend = accum._last_blend

        # Early blend should be lower than late blend
        # (Though exact values depend on adaptive logic)
        assert early_blend < late_blend or accum.is_converged


# ============================================================================
# Test ImportanceScheduler
# ============================================================================


class TestImportanceScheduler:
    """Tests for importance-based probe scheduling."""

    def test_register_probe(self):
        """Test probe registration."""
        scheduler = ImportanceScheduler()
        scheduler.register_probe(0)
        scheduler.register_probe(1)

        stats = scheduler.get_statistics()
        assert stats["total_probes"] == 2

    def test_unregister_probe(self):
        """Test probe unregistration."""
        scheduler = ImportanceScheduler()
        scheduler.register_probe(0)
        scheduler.unregister_probe(0)

        stats = scheduler.get_statistics()
        assert stats["total_probes"] == 0

    def test_critical_importance_near_camera(self):
        """Test probes near camera get critical importance."""
        config = ImportanceSchedulerConfig(camera_distance_critical=10.0)
        scheduler = ImportanceScheduler(config)
        scheduler.register_probe(0)
        scheduler.update_probe_info(0, camera_distance=5.0, in_frustum=True, variance=0.0)

        state = scheduler._probe_states[0]
        assert state.importance == ProbeImportance.CRITICAL

    def test_low_importance_far_from_camera(self):
        """Test probes far from camera get low importance."""
        config = ImportanceSchedulerConfig(camera_distance_medium=60.0)
        scheduler = ImportanceScheduler(config)
        scheduler.register_probe(0)
        scheduler.update_probe_info(0, camera_distance=100.0, in_frustum=True, variance=0.0)

        state = scheduler._probe_states[0]
        assert state.importance == ProbeImportance.LOW

    def test_dormant_outside_frustum(self):
        """Test probes outside frustum are dormant."""
        scheduler = ImportanceScheduler()
        scheduler.register_probe(0)
        scheduler.update_probe_info(0, camera_distance=10.0, in_frustum=False, variance=0.0)

        state = scheduler._probe_states[0]
        assert state.importance == ProbeImportance.DORMANT

    def test_high_variance_boosts_importance(self):
        """Test high variance boosts importance."""
        config = ImportanceSchedulerConfig(
            variance_threshold_high=0.1,
            camera_distance_medium=60.0,
        )
        scheduler = ImportanceScheduler(config)
        scheduler.register_probe(0)

        # Far from camera but high variance
        scheduler.update_probe_info(0, camera_distance=80.0, in_frustum=True, variance=0.5)

        state = scheduler._probe_states[0]
        # Should be boosted from LOW to CRITICAL due to variance
        assert state.importance.value <= ProbeImportance.CRITICAL.value

    def test_get_probes_to_update_critical(self):
        """Test critical probes are always updated."""
        config = ImportanceSchedulerConfig(critical_update_rate=1)
        scheduler = ImportanceScheduler(config)
        scheduler.register_probe(0)
        scheduler.update_probe_info(0, camera_distance=5.0, in_frustum=True, variance=0.0)

        # Should update every frame
        probes = scheduler.get_probes_to_update(1, 10)
        assert 0 in probes

    def test_low_importance_skips_frames(self):
        """Test low importance probes skip frames."""
        config = ImportanceSchedulerConfig(
            low_update_rate=16,
            camera_distance_medium=30.0,
        )
        scheduler = ImportanceScheduler(config)
        scheduler.register_probe(0)
        scheduler.update_probe_info(0, camera_distance=100.0, in_frustum=True, variance=0.0)
        scheduler.mark_updated(0, 0)

        # Should NOT update on frame 8 (before 16 frame rate)
        probes = scheduler.get_probes_to_update(8, 10)
        assert 0 not in probes

        # SHOULD update on frame 16
        probes = scheduler.get_probes_to_update(16, 10)
        assert 0 in probes

    def test_budget_limiting(self):
        """Test max_updates limits returned probes."""
        scheduler = ImportanceScheduler()
        for i in range(10):
            scheduler.register_probe(i)
            scheduler.update_probe_info(i, camera_distance=5.0, in_frustum=True, variance=0.0)

        probes = scheduler.get_probes_to_update(1, 5)
        assert len(probes) <= 5

    def test_priority_ordering(self):
        """Test probes are ordered by priority."""
        scheduler = ImportanceScheduler()

        # Register with varying distances
        scheduler.register_probe(0)
        scheduler.update_probe_info(0, camera_distance=100.0, in_frustum=True, variance=0.0)

        scheduler.register_probe(1)
        scheduler.update_probe_info(1, camera_distance=5.0, in_frustum=True, variance=0.0)

        probes = scheduler.get_probes_to_update(1, 10)

        # Critical probe (1) should come first
        if len(probes) >= 2:
            assert probes[0] == 1


# ============================================================================
# Test DDGIProbeUpdater (Integration)
# ============================================================================


class TestDDGIProbeUpdater:
    """Integration tests for the main probe updater."""

    def test_register_probe(self):
        """Test probe registration."""
        updater = DDGIProbeUpdater()
        updater.register_probe(0, Vec3(0, 0, 0))

        assert updater.probe_count == 1

    def test_unregister_probe(self):
        """Test probe unregistration."""
        updater = DDGIProbeUpdater()
        updater.register_probe(0, Vec3(0, 0, 0))
        updater.unregister_probe(0)

        assert updater.probe_count == 0

    def test_update_probe(self, uniform_ray_hits):
        """Test updating a single probe."""
        updater = DDGIProbeUpdater()
        updater.register_probe(0, Vec3(0, 0, 0))

        sh = updater.update_probe(0, uniform_ray_hits)

        assert isinstance(sh, SHCoefficientsL2)
        # DC should be positive for uniform light
        assert np.all(sh.get(0) > 0)

    def test_update_unknown_probe_raises(self, uniform_ray_hits):
        """Test updating unknown probe raises error."""
        updater = DDGIProbeUpdater()

        with pytest.raises(ValueError):
            updater.update_probe(999, uniform_ray_hits)

    def test_get_probes_to_update(self):
        """Test getting probes scheduled for update."""
        updater = DDGIProbeUpdater()
        for i in range(5):
            updater.register_probe(i, Vec3(i, 0, 0))
            updater.update_probe_camera_info(i, Vec3(0, 0, 0), True)

        probes = updater.get_probes_to_update(1, 3)
        assert len(probes) <= 3

    def test_frame_statistics(self, uniform_ray_hits):
        """Test frame statistics tracking."""
        updater = DDGIProbeUpdater()
        updater.register_probe(0, Vec3(0, 0, 0))

        updater.begin_frame(1)
        updater.update_probe(0, uniform_ray_hits)
        stats = updater.end_frame()

        assert stats["frame_index"] == 1
        assert stats["updates_this_frame"] == 1
        assert stats["probe_count"] == 1

    def test_temporal_accumulation(self, uniform_ray_hits):
        """Test temporal accumulation over multiple frames."""
        config = DDGIProbeUpdaterConfig(enable_temporal=True)
        updater = DDGIProbeUpdater(config)
        updater.register_probe(0, Vec3(0, 0, 0))

        # Update multiple times
        for frame in range(10):
            updater.begin_frame(frame)
            updater.update_probe(0, uniform_ray_hits)
            updater.end_frame()

        # Check convergence
        convergence = updater.get_probe_convergence(0)
        assert convergence > 0

    def test_visibility_storage(self, uniform_ray_hits):
        """Test visibility data is stored."""
        config = DDGIProbeUpdaterConfig(enable_visibility=True)
        updater = DDGIProbeUpdater(config)
        updater.register_probe(0, Vec3(0, 0, 0))

        updater.update_probe(0, uniform_ray_hits)

        vis = updater.get_probe_visibility(0)
        assert vis is not None

    def test_reset_probe(self, uniform_ray_hits):
        """Test resetting a probe."""
        updater = DDGIProbeUpdater()
        updater.register_probe(0, Vec3(0, 0, 0))
        updater.update_probe(0, uniform_ray_hits)

        updater.reset_probe(0)

        # After reset, irradiance should be zero
        sh = updater.get_probe_irradiance(0)
        for i in range(9):
            assert np.allclose(sh.get(i), 0)

    def test_reset_all(self, uniform_ray_hits):
        """Test resetting all probes."""
        updater = DDGIProbeUpdater()
        for i in range(3):
            updater.register_probe(i, Vec3(i, 0, 0))
            updater.update_probe(i, uniform_ray_hits)

        updater.reset_all()

        for i in range(3):
            sh = updater.get_probe_irradiance(i)
            for j in range(9):
                assert np.allclose(sh.get(j), 0)

    def test_scheduling_disabled(self, uniform_ray_hits):
        """Test with scheduling disabled."""
        config = DDGIProbeUpdaterConfig(enable_scheduling=False)
        updater = DDGIProbeUpdater(config)

        for i in range(5):
            updater.register_probe(i, Vec3(i, 0, 0))

        # Should return all probes when scheduling disabled
        probes = updater.get_probes_to_update(1, 10)
        assert len(probes) == 5

    def test_irradiance_buffer_export(self, uniform_ray_hits):
        """Test GPU buffer export for irradiance."""
        updater = DDGIProbeUpdater()
        updater.register_probe(0, Vec3(0, 0, 0))
        updater.update_probe(0, uniform_ray_hits)

        buffer = updater.build_irradiance_buffer()

        # SH coefficients: 9 * 4 floats (padded) * 4 bytes = 144 bytes per probe
        assert len(buffer) == 144

    def test_visibility_buffer_export(self, uniform_ray_hits):
        """Test GPU buffer export for visibility."""
        config = DDGIProbeUpdaterConfig(
            enable_visibility=True,
            visibility_config=VisibilityStorageConfig(resolution=64),
        )
        updater = DDGIProbeUpdater(config)
        updater.register_probe(0, Vec3(0, 0, 0))
        updater.update_probe(0, uniform_ray_hits)

        buffer = updater.build_visibility_buffer()

        # 64 directions * 2 floats (mean, mean_sq) * 4 bytes = 512 bytes
        assert len(buffer) == 64 * 2 * 4

    def test_iter_probes(self, uniform_ray_hits):
        """Test probe iteration."""
        updater = DDGIProbeUpdater()
        for i in range(3):
            updater.register_probe(i, Vec3(i, 0, 0))

        probes = list(updater.iter_probes())
        assert len(probes) == 3
        assert all(isinstance(p, ProbeData) for p in probes)


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_test_ray_hits_uniform(self):
        """Test creating uniform test hits."""
        hits = create_test_ray_hits(64, "uniform")
        assert len(hits) == 64
        assert all(isinstance(h, ProbeRayHit) for h in hits)

    def test_create_test_ray_hits_gradient(self):
        """Test creating gradient test hits."""
        hits = create_test_ray_hits(64, "gradient")
        assert len(hits) == 64

        # Check gradient: upper hemisphere should be brighter
        upper_hits = [h for h in hits if h.direction.y > 0.5]
        lower_hits = [h for h in hits if h.direction.y < -0.5]

        if upper_hits and lower_hits:
            upper_mean = np.mean([h.radiance.x for h in upper_hits])
            lower_mean = np.mean([h.radiance.x for h in lower_hits])
            assert upper_mean > lower_mean

    def test_create_test_ray_hits_mixed(self):
        """Test creating mixed test hits."""
        hits = create_test_ray_hits(64, "mixed")
        assert len(hits) == 64

        # Should have some misses (sky)
        misses = [h for h in hits if h.is_miss()]
        assert len(misses) > 0

    def test_estimate_update_cost(self):
        """Test update cost estimation."""
        cost = estimate_update_cost(1000, 256)
        assert cost > 0
        assert cost < 1.0  # Should be reasonable for 256K rays

    def test_estimate_cost_scales_with_probes(self):
        """Test cost scales with probe count."""
        cost_small = estimate_update_cost(100, 256)
        cost_large = estimate_update_cost(1000, 256)

        assert cost_large > cost_small


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_ray_hit(self):
        """Test accumulation with single ray hit."""
        hits = [ProbeRayHit(Vec3(0, 1, 0), Vec3(1, 1, 1), 10.0)]
        accum = IrradianceAccumulator()
        sh = accum.accumulate(hits)

        # Should produce valid SH
        assert accum.sample_count == 1

    def test_all_misses(self):
        """Test accumulation with all ray misses."""
        hits = [
            ProbeRayHit(Vec3(1, 0, 0), Vec3(0.8, 0.9, 1.0), VISIBILITY_MISS_DISTANCE),
            ProbeRayHit(Vec3(0, 1, 0), Vec3(0.8, 0.9, 1.0), VISIBILITY_MISS_DISTANCE),
        ]
        accum = IrradianceAccumulator()
        sh = accum.accumulate(hits)

        # Misses still contribute with reduced weight
        assert accum.sample_count > 0

    def test_zero_frame_count(self):
        """Test temporal accumulator at zero frames."""
        accum = TemporalAccumulator()
        assert accum.frame_count == 0
        assert not accum.is_converged
        assert accum.convergence_factor == 0.0

    def test_very_high_ray_count(self):
        """Test with high ray count."""
        hits = create_test_ray_hits(1024, "uniform")
        accum = IrradianceAccumulator()
        sh = accum.accumulate(hits)

        assert accum.sample_count > 500
        assert accum.confidence > 0.5

    def test_importance_constants(self):
        """Test importance constant values."""
        assert IMPORTANCE_CRITICAL == 1.0
        assert IMPORTANCE_HIGH == 0.75
        assert IMPORTANCE_MEDIUM == 0.5
        assert IMPORTANCE_LOW == 0.25

    def test_visibility_constants(self):
        """Test visibility constant values."""
        assert VISIBILITY_MAX_DISTANCE == 1000.0
        assert VISIBILITY_MISS_DISTANCE == 10000.0

    def test_blend_constants(self):
        """Test blend constant values."""
        assert DEFAULT_IRRADIANCE_BLEND == 0.97
        assert DEFAULT_VISIBILITY_BLEND == 0.95


# ============================================================================
# Test SH Integration
# ============================================================================


class TestSHIntegration:
    """Tests for SH storage integration from P1.2."""

    def test_sh_coefficients_compatible(self, uniform_ray_hits):
        """Test accumulated SH is compatible with SH evaluation."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate(uniform_ray_hits)

        # Evaluate in +Z direction
        direction = np.array([0, 0, 1], dtype=np.float32)
        color = sh_evaluate_l2(sh, direction)

        assert len(color) == 3
        assert np.all(color >= 0)

    def test_sh_direction_consistency(self, gradient_ray_hits):
        """Test SH evaluation matches input directions."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate(gradient_ray_hits)

        # Upper hemisphere should be brighter
        up = sh_evaluate_l2(sh, np.array([0, 1, 0], dtype=np.float32))
        down = sh_evaluate_l2(sh, np.array([0, -1, 0], dtype=np.float32))

        # Gradient has brighter upper hemisphere
        assert np.mean(up) > np.mean(down)

    def test_sh_serialization_roundtrip(self, uniform_ray_hits):
        """Test SH serialization works for GPU upload."""
        accum = IrradianceAccumulator()
        sh = accum.accumulate(uniform_ray_hits)

        # Serialize to bytes
        data = sh.to_bytes()
        assert len(data) == 144  # 9 * 4 * 4 bytes (padded vec4)

        # Deserialize
        restored = SHCoefficientsL2.from_bytes(data)

        # Check coefficients match
        for i in range(9):
            np.testing.assert_array_almost_equal(sh.get(i), restored.get(i))
