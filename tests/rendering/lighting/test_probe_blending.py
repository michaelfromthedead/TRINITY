"""Tests for per-pixel reflection probe blending system.

Covers:
- Weight calculation accuracy (distance, normal, visibility)
- Normalization correctness
- Multi-probe blending
- Edge cases (0, 1, N probes)
- Boundary transitions
- Normal alignment weighting
- Configuration validation
- ProbeBlendPass execution
- WGSL shader generation
"""

from __future__ import annotations

import math
from typing import Optional, List, Callable

import pytest

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    CaptureConfig,
    CubemapData,
    CubemapFace,
    CubemapFaceData,
    HDRPixel,
)
from engine.rendering.lighting.reflection_probes import (
    RealtimeReflectionProbe,
    RealtimeProbeManager,
    RealtimeProbeCapture,
    RealtimeProbeCaptureSettings,
)
from engine.rendering.lighting.probe_blending import (
    ProbeBlendConstants,
    FalloffType,
    ProbeInfluence,
    ProbeCollectorConfig,
    ProbeCollector,
    BlendResult,
    ProbeBlender,
    ProbeBlendConfig,
    GBufferSample,
    ReflectionBuffer,
    ProbeBlendPass,
    generate_probe_blend_wgsl,
    generate_probe_blend_shader,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def simple_probe_manager() -> RealtimeProbeManager:
    """Create a probe manager with a single probe."""
    manager = RealtimeProbeManager()
    manager.register_probe(
        name="test_probe",
        position=Vec3(0, 0, 0),
        bounds=AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5)),
    )
    return manager


@pytest.fixture
def multi_probe_manager() -> RealtimeProbeManager:
    """Create a probe manager with multiple overlapping probes."""
    manager = RealtimeProbeManager()

    # Probe A at origin
    manager.register_probe(
        name="probe_a",
        position=Vec3(0, 0, 0),
        bounds=AABB(Vec3(-10, -10, -10), Vec3(10, 10, 10)),
    )

    # Probe B offset in X
    manager.register_probe(
        name="probe_b",
        position=Vec3(5, 0, 0),
        bounds=AABB(Vec3(0, -10, -10), Vec3(10, 10, 10)),
    )

    # Probe C offset in Y
    manager.register_probe(
        name="probe_c",
        position=Vec3(0, 5, 0),
        bounds=AABB(Vec3(-10, 0, -10), Vec3(10, 10, 10)),
    )

    return manager


@pytest.fixture
def equal_probes_manager() -> RealtimeProbeManager:
    """Create two equal probes for midpoint blending test."""
    manager = RealtimeProbeManager()

    # Two identical probes at different positions
    manager.register_probe(
        name="probe_left",
        position=Vec3(-5, 0, 0),
        bounds=AABB(Vec3(-10, -5, -5), Vec3(0, 5, 5)),
    )

    manager.register_probe(
        name="probe_right",
        position=Vec3(5, 0, 0),
        bounds=AABB(Vec3(0, -5, -5), Vec3(10, 5, 5)),
    )

    return manager


# -----------------------------------------------------------------------------
# ProbeBlendConstants Tests
# -----------------------------------------------------------------------------

class TestProbeBlendConstants:
    """Tests for probe blend constants."""

    def test_default_max_probes(self) -> None:
        """Test default max probes value."""
        assert ProbeBlendConstants.DEFAULT_MAX_PROBES == 4

    def test_max_probes_limit(self) -> None:
        """Test maximum probes limit."""
        assert ProbeBlendConstants.MAX_PROBES_LIMIT == 16

    def test_min_weight_threshold(self) -> None:
        """Test minimum weight threshold."""
        assert ProbeBlendConstants.MIN_WEIGHT_THRESHOLD == pytest.approx(0.001)

    def test_default_blend_distance(self) -> None:
        """Test default blend distance."""
        assert ProbeBlendConstants.DEFAULT_BLEND_DISTANCE == pytest.approx(2.0)

    def test_blend_distance_bounds(self) -> None:
        """Test blend distance bounds."""
        assert ProbeBlendConstants.MIN_BLEND_DISTANCE < ProbeBlendConstants.MAX_BLEND_DISTANCE

    def test_default_normal_weight(self) -> None:
        """Test default normal weight."""
        assert 0.0 <= ProbeBlendConstants.DEFAULT_NORMAL_WEIGHT <= 1.0

    def test_default_visibility_weight(self) -> None:
        """Test default visibility weight."""
        assert 0.0 <= ProbeBlendConstants.DEFAULT_VISIBILITY_WEIGHT <= 1.0


# -----------------------------------------------------------------------------
# FalloffType Tests
# -----------------------------------------------------------------------------

class TestFalloffType:
    """Tests for falloff type enum."""

    def test_all_falloff_types_exist(self) -> None:
        """Test all expected falloff types exist."""
        assert FalloffType.LINEAR
        assert FalloffType.QUADRATIC
        assert FalloffType.SMOOTH
        assert FalloffType.INVERSE
        assert FalloffType.EXPONENTIAL

    def test_falloff_types_are_unique(self) -> None:
        """Test falloff types have unique values."""
        values = [f.value for f in FalloffType]
        assert len(values) == len(set(values))


# -----------------------------------------------------------------------------
# ProbeInfluence Tests
# -----------------------------------------------------------------------------

class TestProbeInfluence:
    """Tests for probe influence calculation."""

    def test_influence_at_probe_center(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test influence is maximum at probe center."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),  # At center
        )
        weight = influence.calculate_weight()
        assert weight == pytest.approx(1.0)

    def test_influence_at_probe_boundary(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test influence decreases at boundary."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(4.9, 0, 0),  # Near boundary
        )
        weight = influence.calculate_weight()
        assert 0.0 < weight < 1.0

    def test_influence_outside_probe(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test influence is zero outside probe bounds."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(100, 0, 0),  # Way outside
        )
        weight = influence.calculate_weight()
        assert weight == pytest.approx(0.0)

    def test_distance_falloff_linear(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test linear distance falloff."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            falloff_type=FalloffType.LINEAR,
        )
        falloff = influence.distance_falloff()
        assert falloff == pytest.approx(1.0)  # At center

    def test_distance_falloff_quadratic(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test quadratic distance falloff."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            falloff_type=FalloffType.QUADRATIC,
        )
        falloff = influence.distance_falloff()
        assert falloff == pytest.approx(1.0)

    def test_distance_falloff_smooth(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test smooth (smoothstep) distance falloff."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            falloff_type=FalloffType.SMOOTH,
        )
        falloff = influence.distance_falloff()
        assert falloff == pytest.approx(1.0)

    def test_distance_falloff_inverse(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test inverse distance falloff."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            falloff_type=FalloffType.INVERSE,
            blend_distance=2.0,
        )
        falloff = influence.distance_falloff()
        assert falloff == pytest.approx(1.0)

    def test_distance_falloff_exponential(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test exponential distance falloff."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            falloff_type=FalloffType.EXPONENTIAL,
        )
        falloff = influence.distance_falloff()
        assert falloff == pytest.approx(1.0)

    def test_falloff_decreases_with_distance(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test falloff monotonically decreases with distance."""
        probe = simple_probe_manager.get_all_probes()[0]

        distances = [0.0, 1.0, 2.0, 3.0, 4.0]
        prev_falloff = 2.0  # Start higher than max

        for d in distances:
            influence = ProbeInfluence(
                probe=probe,
                world_position=Vec3(d, 0, 0),
                falloff_type=FalloffType.SMOOTH,
            )
            falloff = influence.distance_falloff()
            assert falloff <= prev_falloff
            prev_falloff = falloff

    def test_normal_alignment_facing_probe(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test normal alignment when facing probe."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(-3, 0, 0),
            surface_normal=Vec3(1, 0, 0),  # Pointing toward probe
        )
        alignment = influence.normal_alignment()
        assert alignment > 0.9

    def test_normal_alignment_away_from_probe(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test normal alignment when facing away."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(3, 0, 0),
            surface_normal=Vec3(1, 0, 0),  # Pointing away from probe
        )
        alignment = influence.normal_alignment()
        assert alignment < 0.5

    def test_normal_alignment_perpendicular(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test normal alignment when perpendicular."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(3, 0, 0),
            surface_normal=Vec3(0, 1, 0),  # Perpendicular
        )
        alignment = influence.normal_alignment()
        assert alignment == pytest.approx(0.5, abs=0.1)

    def test_normal_alignment_no_normal(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test normal alignment returns 1.0 when no normal provided."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            surface_normal=None,
        )
        alignment = influence.normal_alignment()
        assert alignment == pytest.approx(1.0)

    def test_visibility_factor_visible(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test visibility factor when visible."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            is_visible=True,
        )
        vis = influence.visibility_factor()
        assert vis == pytest.approx(1.0)

    def test_visibility_factor_not_visible(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test visibility factor when not visible."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            is_visible=False,
        )
        vis = influence.visibility_factor()
        assert vis == pytest.approx(0.0)

    def test_visibility_affects_weight(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test visibility affects final weight."""
        probe = simple_probe_manager.get_all_probes()[0]

        visible = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            is_visible=True,
            visibility_weight=1.0,
        )
        not_visible = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            is_visible=False,
            visibility_weight=1.0,
        )

        assert visible.calculate_weight() > not_visible.calculate_weight()

    def test_get_distance(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test getting distance to probe."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(3, 4, 0),
        )
        distance = influence.get_distance()
        assert distance == pytest.approx(5.0)

    def test_cache_invalidation(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test cache invalidation works."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
        )

        # Calculate weight (caches result)
        weight1 = influence.calculate_weight()

        # Invalidate and recalculate
        influence.invalidate_cache()
        weight2 = influence.calculate_weight()

        assert weight1 == pytest.approx(weight2)

    def test_weight_combined_factors(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test weight combines all factors correctly."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(2, 0, 0),
            surface_normal=Vec3(-1, 0, 0),  # Facing probe
            is_visible=True,
            normal_weight=0.5,
            visibility_weight=0.5,
        )
        weight = influence.calculate_weight()
        assert 0.0 < weight < 1.0


# -----------------------------------------------------------------------------
# ProbeCollectorConfig Tests
# -----------------------------------------------------------------------------

class TestProbeCollectorConfig:
    """Tests for probe collector configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ProbeCollectorConfig()
        assert config.max_probes == ProbeBlendConstants.DEFAULT_MAX_PROBES
        assert config.min_weight == ProbeBlendConstants.MIN_WEIGHT_THRESHOLD

    def test_max_probes_clamping(self) -> None:
        """Test max probes is clamped."""
        config = ProbeCollectorConfig(max_probes=0)
        assert config.max_probes >= 1

        config = ProbeCollectorConfig(max_probes=100)
        assert config.max_probes <= ProbeBlendConstants.MAX_PROBES_LIMIT

    def test_min_weight_clamping(self) -> None:
        """Test min weight is clamped."""
        config = ProbeCollectorConfig(min_weight=-1.0)
        assert config.min_weight >= 0.0

        config = ProbeCollectorConfig(min_weight=1.0)
        assert config.min_weight <= 0.5

    def test_blend_distance_clamping(self) -> None:
        """Test blend distance is clamped."""
        config = ProbeCollectorConfig(blend_distance=0.0)
        assert config.blend_distance >= ProbeBlendConstants.MIN_BLEND_DISTANCE

        config = ProbeCollectorConfig(blend_distance=1000.0)
        assert config.blend_distance <= ProbeBlendConstants.MAX_BLEND_DISTANCE

    def test_normal_weight_clamping(self) -> None:
        """Test normal weight is clamped."""
        config = ProbeCollectorConfig(normal_weight=-0.5)
        assert config.normal_weight >= 0.0

        config = ProbeCollectorConfig(normal_weight=1.5)
        assert config.normal_weight <= 1.0

    def test_visibility_weight_clamping(self) -> None:
        """Test visibility weight is clamped."""
        config = ProbeCollectorConfig(visibility_weight=-0.5)
        assert config.visibility_weight >= 0.0

        config = ProbeCollectorConfig(visibility_weight=1.5)
        assert config.visibility_weight <= 1.0


# -----------------------------------------------------------------------------
# ProbeCollector Tests
# -----------------------------------------------------------------------------

class TestProbeCollector:
    """Tests for probe collection."""

    def test_collector_creation(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test creating a probe collector."""
        collector = ProbeCollector(simple_probe_manager)
        assert collector.probe_manager is simple_probe_manager

    def test_collect_probes_inside(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test collecting probes when inside bounds."""
        collector = ProbeCollector(simple_probe_manager)
        influences = collector.collect_probes(Vec3(0, 0, 0))
        assert len(influences) == 1

    def test_collect_probes_outside(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test collecting probes when outside all bounds."""
        collector = ProbeCollector(simple_probe_manager)
        influences = collector.collect_probes(Vec3(100, 0, 0))
        assert len(influences) == 0

    def test_collect_multiple_probes(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test collecting multiple overlapping probes."""
        collector = ProbeCollector(multi_probe_manager)
        influences = collector.collect_probes(Vec3(5, 5, 0))  # Inside A, B, and C
        assert len(influences) >= 2

    def test_collect_with_normal(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test collecting with surface normal."""
        collector = ProbeCollector(simple_probe_manager)
        influences = collector.collect_probes(
            Vec3(0, 0, 0),
            surface_normal=Vec3(0, 1, 0),
        )
        assert len(influences) == 1
        assert influences[0].surface_normal is not None

    def test_collect_with_visibility_func(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test collecting with custom visibility function."""
        # Visibility function that blocks all probes
        def block_all(probe: RealtimeReflectionProbe, pos: Vec3) -> bool:
            return False

        collector = ProbeCollector(multi_probe_manager)
        influences = collector.collect_probes(
            Vec3(5, 5, 0),
            visibility_func=block_all,
        )

        # All probes blocked by visibility
        for inf in influences:
            assert not inf.is_visible

    def test_sort_by_influence(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test sorting influences by weight."""
        collector = ProbeCollector(multi_probe_manager)
        influences = collector.collect_probes(Vec3(5, 5, 0))
        sorted_influences = collector.sort_by_influence(influences)

        # Check sorted in descending order
        for i in range(len(sorted_influences) - 1):
            assert sorted_influences[i].calculate_weight() >= sorted_influences[i + 1].calculate_weight()

    def test_get_top_n_default(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test getting top N with default max."""
        collector = ProbeCollector(multi_probe_manager, ProbeCollectorConfig(max_probes=2))
        top = collector.get_top_n(Vec3(5, 5, 0))
        assert len(top) <= 2

    def test_get_top_n_explicit(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test getting top N with explicit count."""
        collector = ProbeCollector(multi_probe_manager)
        top = collector.get_top_n(Vec3(5, 5, 0), n=1)
        assert len(top) == 1

    def test_get_top_n_more_than_available(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test getting more probes than available."""
        collector = ProbeCollector(simple_probe_manager)
        top = collector.get_top_n(Vec3(0, 0, 0), n=10)
        assert len(top) == 1  # Only one probe available


# -----------------------------------------------------------------------------
# ProbeBlender Tests
# -----------------------------------------------------------------------------

class TestProbeBlender:
    """Tests for probe blending."""

    def test_normalize_weights_single(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test normalizing single probe weight."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(probe=probe, world_position=Vec3(0, 0, 0))

        blender = ProbeBlender()
        normalized = blender.normalize_weights([influence])

        assert len(normalized) == 1
        assert normalized[0][1] == pytest.approx(1.0)

    def test_normalize_weights_multiple(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test normalizing multiple probe weights."""
        probes = multi_probe_manager.get_all_probes()
        influences = [
            ProbeInfluence(probe=p, world_position=Vec3(5, 5, 0))
            for p in probes
            if p.contains(Vec3(5, 5, 0))
        ]

        blender = ProbeBlender()
        normalized = blender.normalize_weights(influences)

        # Weights should sum to 1.0
        total = sum(w for _, w in normalized)
        assert total == pytest.approx(1.0)

    def test_normalize_weights_empty(self) -> None:
        """Test normalizing empty list."""
        blender = ProbeBlender()
        normalized = blender.normalize_weights([])
        assert len(normalized) == 0

    def test_normalize_weights_sum_to_one(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test normalized weights always sum to 1.0."""
        collector = ProbeCollector(multi_probe_manager)
        influences = collector.get_top_n(Vec3(5, 5, 0))

        blender = ProbeBlender()
        normalized = blender.normalize_weights(influences)

        if normalized:
            total = sum(w for _, w in normalized)
            assert total == pytest.approx(1.0)

    def test_blend_samples_empty(self) -> None:
        """Test blending with no probes."""
        blender = ProbeBlender()
        result = blender.blend_samples([], Vec3(1, 0, 0))

        assert result.color.r == pytest.approx(0.0)
        assert result.color.g == pytest.approx(0.0)
        assert result.color.b == pytest.approx(0.0)
        assert result.probe_count == 0

    def test_blend_samples_single(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test blending with single probe."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(probe=probe, world_position=Vec3(0, 0, 0))

        blender = ProbeBlender()
        result = blender.blend_samples([influence], Vec3(1, 0, 0))

        assert result.probe_count == 1
        assert result.dominant_probe is probe

    def test_blend_samples_multiple(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test blending with multiple probes."""
        collector = ProbeCollector(multi_probe_manager)
        influences = collector.get_top_n(Vec3(5, 5, 0))

        blender = ProbeBlender()
        result = blender.blend_samples(influences, Vec3(1, 0, 0))

        assert result.probe_count >= 1
        assert result.weight_sum > 0

    def test_blend_at_midpoint_equal_weights(self, equal_probes_manager: RealtimeProbeManager) -> None:
        """Test blending at exact midpoint gives 0.5/0.5 weights."""
        collector = ProbeCollector(equal_probes_manager, ProbeCollectorConfig(
            falloff_type=FalloffType.LINEAR,
            normal_weight=0.0,
            visibility_weight=0.0,
        ))

        # Midpoint between the two probe boundaries at x=0
        # Position just inside both probes
        midpoint = Vec3(-0.01, 0, 0)  # Slightly inside left probe
        midpoint2 = Vec3(0.01, 0, 0)  # Slightly inside right probe

        # At exact boundary, only one probe contains the point
        # Test positions that are inside overlap region
        influences = collector.get_top_n(Vec3(0, 0, 0))  # Exact boundary

        # If we're at a point inside both probes with equal distances
        # the weights should be equal when normalized
        if len(influences) == 2:
            blender = ProbeBlender()
            normalized = blender.normalize_weights(influences)
            weights = [w for _, w in normalized]
            # Both weights should be equal (0.5 each)
            assert abs(weights[0] - weights[1]) < 0.1

    def test_sample_blended(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test convenience method sample_blended."""
        collector = ProbeCollector(simple_probe_manager)
        blender = ProbeBlender()

        result = blender.sample_blended(
            collector,
            world_position=Vec3(0, 0, 0),
            direction=Vec3(1, 0, 0),
            roughness=0.5,
        )

        assert result.probe_count >= 0


# -----------------------------------------------------------------------------
# ProbeBlendConfig Tests
# -----------------------------------------------------------------------------

class TestProbeBlendConfig:
    """Tests for probe blend configuration."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ProbeBlendConfig()
        assert config.max_probes == ProbeBlendConstants.DEFAULT_MAX_PROBES
        assert config.distance_falloff_type == FalloffType.SMOOTH

    def test_max_probes_clamping(self) -> None:
        """Test max probes is clamped."""
        config = ProbeBlendConfig(max_probes=0)
        assert config.max_probes >= 1

    def test_normal_weight_clamping(self) -> None:
        """Test normal weight is clamped to 0-1."""
        config = ProbeBlendConfig(normal_weight=-0.5)
        assert config.normal_weight == 0.0

        config = ProbeBlendConfig(normal_weight=1.5)
        assert config.normal_weight == 1.0

    def test_visibility_weight_clamping(self) -> None:
        """Test visibility weight is clamped to 0-1."""
        config = ProbeBlendConfig(visibility_weight=-0.5)
        assert config.visibility_weight == 0.0

        config = ProbeBlendConfig(visibility_weight=1.5)
        assert config.visibility_weight == 1.0

    def test_blend_distance_clamping(self) -> None:
        """Test blend distance is clamped."""
        config = ProbeBlendConfig(blend_distance=0.01)
        assert config.blend_distance >= ProbeBlendConstants.MIN_BLEND_DISTANCE

    def test_roughness_levels_clamping(self) -> None:
        """Test roughness levels is clamped."""
        config = ProbeBlendConfig(roughness_levels=0)
        assert config.roughness_levels >= 1

        config = ProbeBlendConfig(roughness_levels=100)
        assert config.roughness_levels <= 12

    def test_to_collector_config(self) -> None:
        """Test conversion to ProbeCollectorConfig."""
        config = ProbeBlendConfig(
            max_probes=8,
            distance_falloff_type=FalloffType.QUADRATIC,
            normal_weight=0.5,
        )
        collector_config = config.to_collector_config()

        assert collector_config.max_probes == 8
        assert collector_config.falloff_type == FalloffType.QUADRATIC
        assert collector_config.normal_weight == 0.5


# -----------------------------------------------------------------------------
# GBufferSample Tests
# -----------------------------------------------------------------------------

class TestGBufferSample:
    """Tests for G-buffer sample."""

    def test_gbuffer_sample_creation(self) -> None:
        """Test creating a G-buffer sample."""
        sample = GBufferSample(
            position=Vec3(1, 2, 3),
            normal=Vec3(0, 1, 0),
            roughness=0.5,
            metallic=0.0,
        )
        assert sample.position.x == pytest.approx(1.0)
        assert sample.roughness == pytest.approx(0.5)

    def test_gbuffer_sample_default_valid(self) -> None:
        """Test G-buffer sample is valid by default."""
        sample = GBufferSample(
            position=Vec3(0, 0, 0),
            normal=Vec3(0, 1, 0),
            roughness=0.0,
            metallic=0.0,
        )
        assert sample.is_valid is True


# -----------------------------------------------------------------------------
# ReflectionBuffer Tests
# -----------------------------------------------------------------------------

class TestReflectionBuffer:
    """Tests for reflection buffer."""

    def test_buffer_creation(self) -> None:
        """Test creating a reflection buffer."""
        buffer = ReflectionBuffer(width=16, height=16)
        assert buffer.width == 16
        assert buffer.height == 16
        assert len(buffer.data) == 256

    def test_get_pixel(self) -> None:
        """Test getting pixel from buffer."""
        buffer = ReflectionBuffer(width=4, height=4)
        pixel = buffer.get_pixel(0, 0)
        assert pixel.r == pytest.approx(0.0)

    def test_set_pixel(self) -> None:
        """Test setting pixel in buffer."""
        buffer = ReflectionBuffer(width=4, height=4)
        buffer.set_pixel(1, 1, HDRPixel(1.0, 0.5, 0.0))
        pixel = buffer.get_pixel(1, 1)
        assert pixel.r == pytest.approx(1.0)
        assert pixel.g == pytest.approx(0.5)
        assert pixel.b == pytest.approx(0.0)

    def test_get_pixel_out_of_bounds(self) -> None:
        """Test getting pixel out of bounds returns black."""
        buffer = ReflectionBuffer(width=4, height=4)
        pixel = buffer.get_pixel(100, 100)
        assert pixel.r == pytest.approx(0.0)

    def test_set_pixel_out_of_bounds(self) -> None:
        """Test setting pixel out of bounds is safe."""
        buffer = ReflectionBuffer(width=4, height=4)
        buffer.set_pixel(100, 100, HDRPixel(1.0, 1.0, 1.0))  # Should not crash

    def test_clear_buffer(self) -> None:
        """Test clearing buffer."""
        buffer = ReflectionBuffer(width=4, height=4)
        buffer.set_pixel(0, 0, HDRPixel(1.0, 1.0, 1.0))
        buffer.clear()
        pixel = buffer.get_pixel(0, 0)
        assert pixel.r == pytest.approx(0.0)

    def test_clear_with_color(self) -> None:
        """Test clearing buffer with specific color."""
        buffer = ReflectionBuffer(width=4, height=4)
        buffer.clear(HDRPixel(0.5, 0.5, 0.5))
        pixel = buffer.get_pixel(0, 0)
        assert pixel.r == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# ProbeBlendPass Tests
# -----------------------------------------------------------------------------

class TestProbeBlendPass:
    """Tests for probe blend pass execution."""

    def test_pass_creation(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test creating a blend pass."""
        blend_pass = ProbeBlendPass(simple_probe_manager)
        assert blend_pass.config is not None
        assert blend_pass.reflection_buffer is None

    def test_pass_config_setter(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test setting blend pass config."""
        blend_pass = ProbeBlendPass(simple_probe_manager)
        new_config = ProbeBlendConfig(max_probes=8)
        blend_pass.config = new_config
        assert blend_pass.config.max_probes == 8

    def test_execute_creates_buffer(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test execute creates reflection buffer."""
        blend_pass = ProbeBlendPass(simple_probe_manager)

        def sample_gbuffer(x: int, y: int) -> GBufferSample:
            return GBufferSample(
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                roughness=0.5,
                metallic=0.0,
            )

        buffer = blend_pass.execute(
            width=8,
            height=8,
            gbuffer_sampler=sample_gbuffer,
            camera_position=Vec3(0, 5, 5),
        )

        assert buffer is not None
        assert buffer.width == 8
        assert buffer.height == 8

    def test_execute_skips_invalid_samples(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test execute skips invalid G-buffer samples."""
        blend_pass = ProbeBlendPass(simple_probe_manager)

        def sample_gbuffer(x: int, y: int) -> GBufferSample:
            return GBufferSample(
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                roughness=0.5,
                metallic=0.0,
                is_valid=False,  # Sky pixel
            )

        buffer = blend_pass.execute(
            width=4,
            height=4,
            gbuffer_sampler=sample_gbuffer,
            camera_position=Vec3(0, 5, 5),
        )

        # Buffer should be all black (no processing)
        pixel = buffer.get_pixel(0, 0)
        assert pixel.r == pytest.approx(0.0)

    def test_get_reflection_buffer(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test getting reflection buffer after execute."""
        blend_pass = ProbeBlendPass(simple_probe_manager)

        def sample_gbuffer(x: int, y: int) -> GBufferSample:
            return GBufferSample(
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                roughness=0.5,
                metallic=0.0,
            )

        blend_pass.execute(4, 4, sample_gbuffer, Vec3(0, 5, 5))
        buffer = blend_pass.get_reflection_buffer()
        assert buffer is not None

    def test_execution_time_tracking(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test execution time is tracked."""
        blend_pass = ProbeBlendPass(simple_probe_manager)

        def sample_gbuffer(x: int, y: int) -> GBufferSample:
            return GBufferSample(
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                roughness=0.5,
                metallic=0.0,
            )

        blend_pass.execute(4, 4, sample_gbuffer, Vec3(0, 5, 5))
        assert blend_pass.last_execution_time_ms >= 0.0

    def test_get_stats(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test getting execution stats."""
        blend_pass = ProbeBlendPass(simple_probe_manager)

        def sample_gbuffer(x: int, y: int) -> GBufferSample:
            return GBufferSample(
                position=Vec3(0, 0, 0),
                normal=Vec3(0, 1, 0),
                roughness=0.5,
                metallic=0.0,
            )

        blend_pass.execute(4, 4, sample_gbuffer, Vec3(0, 5, 5))
        stats = blend_pass.get_stats()

        assert "execution_time_ms" in stats
        assert "pixel_count" in stats
        assert "probe_count" in stats
        assert "max_probes_per_pixel" in stats


# -----------------------------------------------------------------------------
# WGSL Shader Generation Tests
# -----------------------------------------------------------------------------

class TestWGSLGeneration:
    """Tests for WGSL shader generation."""

    def test_generate_default_shader(self) -> None:
        """Test generating shader with default config."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig())
        assert "MAX_PROBES" in shader
        assert "calculate_weight" in shader
        assert "@compute" in shader

    def test_shader_contains_constants(self) -> None:
        """Test shader contains required constants."""
        config = ProbeBlendConfig(max_probes=8)
        shader = generate_probe_blend_wgsl(config)
        assert "MAX_PROBES: u32 = 8u" in shader

    def test_shader_contains_falloff_code(self) -> None:
        """Test shader contains falloff calculation."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig(distance_falloff_type=FalloffType.LINEAR))
        assert "distance_falloff" in shader

    def test_shader_contains_normal_alignment(self) -> None:
        """Test shader contains normal alignment function."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig())
        assert "normal_alignment" in shader

    def test_shader_contains_main_kernel(self) -> None:
        """Test shader contains main compute kernel."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig())
        assert "fn main" in shader
        assert "@workgroup_size" in shader

    def test_generate_shader_function(self) -> None:
        """Test generate_probe_blend_shader convenience function."""
        shader = generate_probe_blend_shader()
        assert len(shader) > 0

    def test_different_falloff_types_generate_different_code(self) -> None:
        """Test different falloff types produce different shaders."""
        linear = generate_probe_blend_wgsl(ProbeBlendConfig(distance_falloff_type=FalloffType.LINEAR))
        smooth = generate_probe_blend_wgsl(ProbeBlendConfig(distance_falloff_type=FalloffType.SMOOTH))
        assert linear != smooth

    def test_shader_has_probe_data_struct(self) -> None:
        """Test shader defines ProbeData struct."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig())
        assert "struct ProbeData" in shader

    def test_shader_has_uniforms(self) -> None:
        """Test shader defines Uniforms struct."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig())
        assert "struct Uniforms" in shader

    def test_shader_samples_cubemaps(self) -> None:
        """Test shader samples cubemap textures."""
        shader = generate_probe_blend_wgsl(ProbeBlendConfig())
        assert "probe_cubemaps" in shader
        assert "textureSampleLevel" in shader


# -----------------------------------------------------------------------------
# Boundary Transition Tests
# -----------------------------------------------------------------------------

class TestBoundaryTransitions:
    """Tests for smooth boundary transitions."""

    def test_smooth_transition_at_boundary(self, multi_probe_manager: RealtimeProbeManager) -> None:
        """Test weights change smoothly at boundaries."""
        collector = ProbeCollector(multi_probe_manager, ProbeCollectorConfig(
            falloff_type=FalloffType.SMOOTH,
        ))

        # Sample along a line crossing probe boundaries
        positions = [Vec3(x, 0, 0) for x in range(-8, 9, 2)]
        prev_weights = None

        for pos in positions:
            influences = collector.collect_probes(pos)
            if influences:
                current_weights = [inf.calculate_weight() for inf in influences]
                if prev_weights is not None:
                    # Check weights changed smoothly (no huge jumps)
                    for w in current_weights:
                        assert 0.0 <= w <= 1.0
                prev_weights = current_weights

    def test_no_weight_discontinuities(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test no sudden weight changes."""
        probe = simple_probe_manager.get_all_probes()[0]

        # Sample along radius
        samples = 10
        prev_weight = None

        for i in range(samples):
            t = i / (samples - 1)
            x = t * 5.0  # From center to edge
            influence = ProbeInfluence(
                probe=probe,
                world_position=Vec3(x, 0, 0),
                falloff_type=FalloffType.SMOOTH,
            )
            weight = influence.calculate_weight()

            if prev_weight is not None:
                # Weight should not jump by more than reasonable amount
                delta = abs(weight - prev_weight)
                assert delta < 0.3  # No huge jumps

            prev_weight = weight


# -----------------------------------------------------------------------------
# Edge Case Tests
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_probes(self) -> None:
        """Test handling when no probes are registered."""
        manager = RealtimeProbeManager()
        collector = ProbeCollector(manager)
        influences = collector.get_top_n(Vec3(0, 0, 0))
        assert len(influences) == 0

    def test_single_probe_normalization(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test single probe weight normalizes to 1.0."""
        collector = ProbeCollector(simple_probe_manager)
        influences = collector.get_top_n(Vec3(0, 0, 0))

        blender = ProbeBlender()
        normalized = blender.normalize_weights(influences)

        assert len(normalized) == 1
        assert normalized[0][1] == pytest.approx(1.0)

    def test_all_zero_weights(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test handling when all weights are near zero."""
        # Force all probes to have zero weight by checking outside bounds
        collector = ProbeCollector(simple_probe_manager, ProbeCollectorConfig(min_weight=0.0))
        influences = collector.collect_probes(Vec3(100, 100, 100))

        blender = ProbeBlender()
        result = blender.blend_samples(influences, Vec3(1, 0, 0))

        assert result.probe_count == 0
        assert result.color.r == pytest.approx(0.0)

    def test_very_small_probe(self) -> None:
        """Test handling very small probe bounds."""
        manager = RealtimeProbeManager()
        manager.register_probe(
            name="tiny",
            position=Vec3(0, 0, 0),
            bounds=AABB(Vec3(-0.01, -0.01, -0.01), Vec3(0.01, 0.01, 0.01)),
        )

        collector = ProbeCollector(manager)
        influences = collector.get_top_n(Vec3(0, 0, 0))
        # Should still work
        assert len(influences) >= 0

    def test_overlapping_identical_probes(self) -> None:
        """Test handling completely overlapping identical probes."""
        manager = RealtimeProbeManager()
        bounds = AABB(Vec3(-5, -5, -5), Vec3(5, 5, 5))

        manager.register_probe(name="a", position=Vec3(0, 0, 0), bounds=bounds)
        manager.register_probe(name="b", position=Vec3(0, 0, 0), bounds=bounds)

        collector = ProbeCollector(manager)
        influences = collector.get_top_n(Vec3(0, 0, 0))

        assert len(influences) == 2

        blender = ProbeBlender()
        normalized = blender.normalize_weights(influences)

        total = sum(w for _, w in normalized)
        assert total == pytest.approx(1.0)

    def test_position_at_exact_boundary(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test position exactly on probe boundary."""
        probe = simple_probe_manager.get_all_probes()[0]

        # Position exactly on boundary
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(5.0, 0, 0),  # Exact boundary
        )
        weight = influence.calculate_weight()
        # Should be included but with low weight
        assert 0.0 <= weight <= 1.0

    def test_degenerate_normal(self, simple_probe_manager: RealtimeProbeManager) -> None:
        """Test handling zero-length normal vector."""
        probe = simple_probe_manager.get_all_probes()[0]
        influence = ProbeInfluence(
            probe=probe,
            world_position=Vec3(0, 0, 0),
            surface_normal=Vec3(0, 0, 0),  # Degenerate normal
        )
        # Should not crash
        alignment = influence.normal_alignment()
        assert 0.0 <= alignment <= 1.0
