"""Tests for Lumen-Lite Feasibility Study (T-GIR-P11.3).

Comprehensive test coverage for all evaluation components:
    - MeshCardEvaluator: Card generation cost estimation
    - ScreenProbeEvaluator: Coverage and stability analysis
    - SDFEvaluator: SDF memory and trace cost
    - RadianceCacheComparison: TRINITY vs Lumen cache
    - QualityMetrics: PSNR, stability, light leaks
    - CostBudgetAnalyzer: GPU budget comparison
    - FeasibilityAssessment: Go/no-go decision

Test Categories:
    - Unit tests for each evaluator class
    - Integration tests for full assessment
    - Edge case handling
    - Report generation validation
"""

import math
import pytest
from typing import List, Tuple

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3

from engine.rendering.gi.lumen_lite_study import (
    # Enums
    FeasibilityLevel,
    ComponentType,
    DecisionOutcome,
    # Mesh Card
    MeshCardEvaluator,
    MeshCardEvaluatorConfig,
    MeshCardCostEstimate,
    # Screen Probe
    ScreenProbeEvaluator,
    ScreenProbeEvaluatorConfig,
    ScreenProbeAnalysis,
    # SDF
    SDFEvaluator,
    SDFEvaluatorConfig,
    SDFEvaluationResult,
    # Radiance Cache
    RadianceCacheComparison,
    RadianceCacheComparisonResult,
    # Quality Metrics
    QualityMetrics,
    QualityMetricResult,
    # Cost Budget
    CostBudgetAnalyzer,
    CostBudgetResult,
    # Feasibility
    FeasibilityAssessment,
    ComponentAssessment,
    FeasibilityReport,
    # Convenience functions
    run_feasibility_study,
    quick_assessment,
    # Constants
    PSNR_EXCELLENT,
    PSNR_GOOD,
    BYTES_PER_MESH_CARD,
    BYTES_PER_SDF_TEXEL,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_bounds() -> AABB:
    """Default scene bounds for testing."""
    return AABB(Vec3(-50, -10, -50), Vec3(50, 40, 50))


@pytest.fixture
def small_bounds() -> AABB:
    """Small scene bounds for testing."""
    return AABB(Vec3(0, 0, 0), Vec3(10, 5, 10))


@pytest.fixture
def mesh_card_evaluator() -> MeshCardEvaluator:
    """Default mesh card evaluator."""
    return MeshCardEvaluator()


@pytest.fixture
def screen_probe_evaluator() -> ScreenProbeEvaluator:
    """Default screen probe evaluator."""
    return ScreenProbeEvaluator()


@pytest.fixture
def sdf_evaluator() -> SDFEvaluator:
    """Default SDF evaluator."""
    return SDFEvaluator()


@pytest.fixture
def cache_comparison() -> RadianceCacheComparison:
    """Default cache comparison."""
    return RadianceCacheComparison()


@pytest.fixture
def quality_metrics() -> QualityMetrics:
    """Default quality metrics."""
    return QualityMetrics()


@pytest.fixture
def cost_analyzer() -> CostBudgetAnalyzer:
    """Default cost analyzer."""
    return CostBudgetAnalyzer()


@pytest.fixture
def feasibility_assessment() -> FeasibilityAssessment:
    """Default feasibility assessment."""
    return FeasibilityAssessment()


# ============================================================================
# MeshCardEvaluator Tests
# ============================================================================


class TestMeshCardEvaluator:
    """Tests for MeshCardEvaluator."""

    def test_init_default_config(self):
        """Test default configuration initialization."""
        evaluator = MeshCardEvaluator()
        assert evaluator.config.min_card_size == 0.1
        assert evaluator.config.max_cards_per_mesh == 64
        assert evaluator.config.target_texels_per_meter == 8.0

    def test_init_custom_config(self):
        """Test custom configuration initialization."""
        config = MeshCardEvaluatorConfig(
            min_card_size=0.5,
            max_cards_per_mesh=32,
        )
        evaluator = MeshCardEvaluator(config)
        assert evaluator.config.min_card_size == 0.5
        assert evaluator.config.max_cards_per_mesh == 32

    def test_estimate_card_cost_simple_mesh(self, mesh_card_evaluator):
        """Test cost estimation for simple mesh."""
        cost = mesh_card_evaluator.estimate_card_cost(
            vertex_count=100,
            triangle_count=50,
            surface_area=5.0,
            has_uv=True,
        )

        assert isinstance(cost, MeshCardCostEstimate)
        assert cost.vertex_count == 100
        assert cost.card_count >= 1
        assert cost.generation_time_ms > 0
        assert cost.memory_bytes > 0
        assert cost.update_time_ms > 0

    def test_estimate_card_cost_complex_mesh(self, mesh_card_evaluator):
        """Test cost estimation for complex mesh."""
        cost = mesh_card_evaluator.estimate_card_cost(
            vertex_count=100000,
            triangle_count=50000,
            surface_area=500.0,
            has_uv=True,
        )

        # Complex mesh should have more cards and higher costs
        assert cost.card_count > 10
        assert cost.generation_time_ms > 0.1
        assert cost.memory_bytes > 1000

    def test_estimate_card_cost_no_uvs(self, mesh_card_evaluator):
        """Test cost estimation when mesh has no UVs."""
        cost_with_uv = mesh_card_evaluator.estimate_card_cost(
            vertex_count=1000,
            triangle_count=500,
            surface_area=10.0,
            has_uv=True,
        )

        cost_no_uv = mesh_card_evaluator.estimate_card_cost(
            vertex_count=1000,
            triangle_count=500,
            surface_area=10.0,
            has_uv=False,
        )

        # No UV should have higher generation time
        assert cost_no_uv.generation_time_ms > cost_with_uv.generation_time_ms

    def test_estimate_card_cost_max_cards_clamped(self, mesh_card_evaluator):
        """Test that card count is clamped to max."""
        cost = mesh_card_evaluator.estimate_card_cost(
            vertex_count=1000,
            triangle_count=500,
            surface_area=10000.0,  # Very large area
            has_uv=True,
        )

        assert cost.card_count <= mesh_card_evaluator.config.max_cards_per_mesh

    def test_evaluate_mesh_high_feasibility(self, mesh_card_evaluator):
        """Test mesh evaluation with high feasibility."""
        feasibility, explanation = mesh_card_evaluator.evaluate_mesh(
            vertex_count=500,
            triangle_count=200,
            surface_area=1.0,  # Small surface area to avoid hitting max cards
            has_uv=True,
        )

        # HIGH or MEDIUM is acceptable for a simple mesh
        assert feasibility in [FeasibilityLevel.HIGH, FeasibilityLevel.MEDIUM]

    def test_evaluate_mesh_medium_feasibility(self, mesh_card_evaluator):
        """Test mesh evaluation with medium feasibility."""
        feasibility, explanation = mesh_card_evaluator.evaluate_mesh(
            vertex_count=500,
            triangle_count=200,
            surface_area=1.0,  # Small surface to reduce card count issues
            has_uv=False,  # No UVs is a moderate issue
        )

        # MEDIUM or LOW is acceptable when UVs are missing
        assert feasibility in [FeasibilityLevel.MEDIUM, FeasibilityLevel.LOW]
        assert "UV" in explanation

    def test_evaluate_mesh_low_feasibility(self, mesh_card_evaluator):
        """Test mesh evaluation with low feasibility."""
        # Very complex mesh
        feasibility, explanation = mesh_card_evaluator.evaluate_mesh(
            vertex_count=10000000,  # 10M vertices
            triangle_count=5000000,
            surface_area=1000.0,
            has_uv=False,
        )

        assert feasibility in [FeasibilityLevel.LOW, FeasibilityLevel.BLOCKED]

    def test_get_memory_overhead(self, mesh_card_evaluator):
        """Test memory overhead calculation."""
        overhead = mesh_card_evaluator.get_memory_overhead(
            total_meshes=100,
            avg_cards_per_mesh=10,
        )

        expected = 100 * 10 * BYTES_PER_MESH_CARD
        assert overhead == expected

    def test_estimate_scene_cost(self, mesh_card_evaluator):
        """Test scene cost estimation."""
        mesh_infos = [
            (1000, 500, 10.0, True),
            (2000, 1000, 20.0, True),
            (500, 200, 5.0, False),
        ]

        total_gen_ms, total_memory, worst_feas = (
            mesh_card_evaluator.estimate_scene_cost(mesh_infos)
        )

        assert total_gen_ms > 0
        assert total_memory > 0
        assert isinstance(worst_feas, FeasibilityLevel)


# ============================================================================
# ScreenProbeEvaluator Tests
# ============================================================================


class TestScreenProbeEvaluator:
    """Tests for ScreenProbeEvaluator."""

    def test_init_default_config(self):
        """Test default configuration."""
        evaluator = ScreenProbeEvaluator()
        assert evaluator.config.screen_width == 1920
        assert evaluator.config.screen_height == 1080
        assert evaluator.config.probe_spacing == 16

    def test_init_custom_config(self):
        """Test custom configuration."""
        config = ScreenProbeEvaluatorConfig(
            screen_width=2560,
            screen_height=1440,
            probe_spacing=8,
        )
        evaluator = ScreenProbeEvaluator(config)
        assert evaluator.config.screen_width == 2560
        assert evaluator.config.probe_spacing == 8

    def test_compare_coverage(self, screen_probe_evaluator, default_bounds):
        """Test coverage comparison."""
        screen_cov, ddgi_cov, analysis = screen_probe_evaluator.compare_coverage(
            ddgi_probe_count=4096,
            ddgi_bounds=default_bounds,
        )

        # Screen probes always have 100% visible coverage
        assert screen_cov == 1.0
        # DDGI coverage should be positive
        assert 0 < ddgi_cov <= 1.0
        assert len(analysis) > 0

    def test_analyze_stability_static(self, screen_probe_evaluator):
        """Test stability analysis with static camera."""
        screen_stab, ddgi_stab, analysis = screen_probe_evaluator.analyze_stability(
            camera_velocity=0.0,
            scene_motion_percentage=0.0,
        )

        # Both should be stable with no motion
        assert screen_stab > 0.5
        assert ddgi_stab > 0.9
        assert len(analysis) > 0

    def test_analyze_stability_moving_camera(self, screen_probe_evaluator):
        """Test stability analysis with moving camera."""
        screen_stab, ddgi_stab, analysis = screen_probe_evaluator.analyze_stability(
            camera_velocity=10.0,
            scene_motion_percentage=0.0,
        )

        # Screen stability should decrease with camera motion
        # DDGI should remain stable
        assert ddgi_stab > screen_stab

    def test_estimate_cost(self, screen_probe_evaluator):
        """Test cost estimation."""
        screen_cost, ddgi_cost, analysis = screen_probe_evaluator.estimate_cost(
            ddgi_probe_count=4096,
            rays_per_ddgi_probe=256,
        )

        assert screen_cost > 0
        assert ddgi_cost > 0
        assert len(analysis) > 0

    def test_full_analysis(self, screen_probe_evaluator, default_bounds):
        """Test full analysis."""
        analysis = screen_probe_evaluator.full_analysis(
            ddgi_probe_count=4096,
            ddgi_bounds=default_bounds,
        )

        assert isinstance(analysis, ScreenProbeAnalysis)
        assert analysis.probe_count > 0
        assert 0 <= analysis.coverage_percentage <= 100
        assert 0 <= analysis.temporal_stability_score <= 1
        assert analysis.update_cost_ms > 0
        assert analysis.memory_bytes > 0


# ============================================================================
# SDFEvaluator Tests
# ============================================================================


class TestSDFEvaluator:
    """Tests for SDFEvaluator."""

    def test_init_default_config(self):
        """Test default configuration."""
        evaluator = SDFEvaluator()
        assert evaluator.config.mesh_sdf_resolution == 64
        assert evaluator.config.global_sdf_resolution == 256
        assert evaluator.config.use_sparse_sdf is True

    def test_estimate_sdf_memory_sparse(self, sdf_evaluator, default_bounds):
        """Test SDF memory estimation with sparse SDFs."""
        mesh_mem, global_mem = sdf_evaluator.estimate_sdf_memory(
            mesh_count=100,
            avg_mesh_extents=Vec3(2, 2, 2),
            scene_bounds=default_bounds,
        )

        assert mesh_mem > 0
        assert global_mem > 0

        # Sparse should use less than full resolution
        full_mesh_mem = 100 * (64 ** 3) * BYTES_PER_SDF_TEXEL
        assert mesh_mem < full_mesh_mem

    def test_estimate_sdf_memory_dense(self, default_bounds):
        """Test SDF memory estimation with dense SDFs."""
        config = SDFEvaluatorConfig(use_sparse_sdf=False)
        evaluator = SDFEvaluator(config)

        mesh_mem, global_mem = evaluator.estimate_sdf_memory(
            mesh_count=100,
            avg_mesh_extents=Vec3(2, 2, 2),
            scene_bounds=default_bounds,
        )

        # Dense should use full resolution
        expected_mesh = 100 * (64 ** 3) * BYTES_PER_SDF_TEXEL
        assert mesh_mem == expected_mesh

    def test_estimate_trace_cost(self, sdf_evaluator):
        """Test trace cost estimation."""
        cost = sdf_evaluator.estimate_trace_cost(
            trace_count=1_000_000,
            avg_trace_distance=50.0,
        )

        assert cost > 0

        # More traces = higher cost
        cost_more = sdf_evaluator.estimate_trace_cost(
            trace_count=2_000_000,
            avg_trace_distance=50.0,
        )
        assert cost_more > cost

    def test_evaluate_quality_sufficient_resolution(self, sdf_evaluator):
        """Test quality evaluation with sufficient resolution."""
        quality, explanation = sdf_evaluator.evaluate_quality(
            mesh_triangle_density=100.0,  # 100 tris/m^3
            sdf_resolution=64,
        )

        assert quality > 0.7
        assert len(explanation) > 0

    def test_evaluate_quality_insufficient_resolution(self, sdf_evaluator):
        """Test quality evaluation with insufficient resolution."""
        quality, explanation = sdf_evaluator.evaluate_quality(
            mesh_triangle_density=10000.0,  # Very dense
            sdf_resolution=32,  # Too low
        )

        assert quality < 0.7
        assert "insufficient" in explanation.lower()

    def test_full_evaluation(self, sdf_evaluator, default_bounds):
        """Test full SDF evaluation."""
        result = sdf_evaluator.full_evaluation(
            mesh_count=500,
            total_triangles=1_000_000,
            scene_bounds=default_bounds,
        )

        assert isinstance(result, SDFEvaluationResult)
        assert result.mesh_sdf_memory_bytes > 0
        assert result.global_sdf_memory_bytes > 0
        assert result.trace_cost_ms > 0
        assert result.construction_time_ms > 0
        assert 0 <= result.quality_score <= 1
        assert isinstance(result.feasibility, FeasibilityLevel)

    def test_full_evaluation_high_complexity(self, sdf_evaluator, default_bounds):
        """Test full evaluation with high complexity scene."""
        result = sdf_evaluator.full_evaluation(
            mesh_count=10000,
            total_triangles=50_000_000,
            scene_bounds=default_bounds,
            traces_per_frame=5_000_000,
        )

        # High complexity should have feasibility concerns
        assert result.feasibility in [
            FeasibilityLevel.LOW,
            FeasibilityLevel.BLOCKED,
            FeasibilityLevel.MEDIUM,
        ]


# ============================================================================
# RadianceCacheComparison Tests
# ============================================================================


class TestRadianceCacheComparison:
    """Tests for RadianceCacheComparison."""

    def test_compare_approaches(self, cache_comparison, default_bounds):
        """Test approach comparison."""
        result = cache_comparison.compare_approaches(
            trinity_grid_dims=(64, 64, 32),
            trinity_bounds=default_bounds,
        )

        assert isinstance(result, RadianceCacheComparisonResult)
        assert result.trinity_memory_bytes > 0
        assert result.lumen_memory_bytes > 0
        assert result.trinity_update_cost_ms > 0
        assert result.lumen_update_cost_ms > 0
        assert 0 <= result.trinity_quality_score <= 1
        assert 0 <= result.lumen_quality_score <= 1
        assert len(result.winner_at_equal_cost) > 0
        assert len(result.analysis) > 0

    def test_compare_approaches_different_resolutions(
        self, cache_comparison, default_bounds
    ):
        """Test comparison with different TRINITY resolutions."""
        result_low = cache_comparison.compare_approaches(
            trinity_grid_dims=(32, 32, 16),
            trinity_bounds=default_bounds,
        )

        result_high = cache_comparison.compare_approaches(
            trinity_grid_dims=(128, 128, 64),
            trinity_bounds=default_bounds,
        )

        # Higher resolution should use more memory
        assert result_high.trinity_memory_bytes > result_low.trinity_memory_bytes

    def test_quality_assessment(self, cache_comparison):
        """Test quality assessment with samplers."""
        test_positions = [
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
            Vec3(0, 0, 1),
        ]

        # Create simple samplers
        def ground_truth(pos: Vec3, normal: Vec3) -> Vec3:
            return Vec3(0.5, 0.5, 0.5)

        def trinity_sampler(pos: Vec3, normal: Vec3) -> Vec3:
            return Vec3(0.48, 0.52, 0.49)  # Close match

        psnr, variance = cache_comparison.quality_assessment(
            test_positions=test_positions,
            ground_truth_sampler=ground_truth,
            trinity_sampler=trinity_sampler,
            normal=Vec3(0, 1, 0),
        )

        assert psnr > PSNR_GOOD  # Should be good quality
        assert variance >= 0

    def test_quality_assessment_poor_match(self, cache_comparison):
        """Test quality assessment with poor match."""
        test_positions = [Vec3(0, 0, 0)]

        def ground_truth(pos: Vec3, normal: Vec3) -> Vec3:
            return Vec3(1.0, 1.0, 1.0)

        def trinity_sampler(pos: Vec3, normal: Vec3) -> Vec3:
            return Vec3(0.0, 0.0, 0.0)  # Complete mismatch

        psnr, variance = cache_comparison.quality_assessment(
            test_positions=test_positions,
            ground_truth_sampler=ground_truth,
            trinity_sampler=trinity_sampler,
            normal=Vec3(0, 1, 0),
        )

        assert psnr < PSNR_GOOD  # Should be poor quality

    def test_quality_assessment_empty_positions(self, cache_comparison):
        """Test quality assessment with no positions."""
        psnr, variance = cache_comparison.quality_assessment(
            test_positions=[],
            ground_truth_sampler=lambda p, n: Vec3.zero(),
            trinity_sampler=lambda p, n: Vec3.zero(),
            normal=Vec3(0, 1, 0),
        )

        assert psnr == 0.0
        assert variance == 0.0


# ============================================================================
# QualityMetrics Tests
# ============================================================================


class TestQualityMetrics:
    """Tests for QualityMetrics."""

    def test_measure_psnr_perfect_match(self, quality_metrics):
        """Test PSNR with perfect match."""
        samples = [Vec3(0.5, 0.5, 0.5), Vec3(0.3, 0.3, 0.3)]
        reference = [Vec3(0.5, 0.5, 0.5), Vec3(0.3, 0.3, 0.3)]

        psnr = quality_metrics.measure_psnr(samples, reference)
        assert psnr == 100.0  # Perfect match

    def test_measure_psnr_partial_match(self, quality_metrics):
        """Test PSNR with partial match."""
        # Use closer values for a better PSNR
        samples = [Vec3(0.50, 0.50, 0.50)]
        reference = [Vec3(0.51, 0.49, 0.50)]  # Very close match

        psnr = quality_metrics.measure_psnr(samples, reference)
        # Should be a reasonable PSNR for a close match
        assert psnr > 20.0  # Reasonable threshold for close match
        assert psnr < 100.0

    def test_measure_psnr_mismatched_lengths(self, quality_metrics):
        """Test PSNR with mismatched lengths."""
        samples = [Vec3(0.5, 0.5, 0.5)]
        reference = [Vec3(0.5, 0.5, 0.5), Vec3(0.3, 0.3, 0.3)]

        psnr = quality_metrics.measure_psnr(samples, reference)
        assert psnr == 0.0

    def test_measure_stability_perfect(self, quality_metrics):
        """Test stability with identical frames."""
        frame1 = [Vec3(0.5, 0.5, 0.5), Vec3(0.3, 0.3, 0.3)]
        frame2 = [Vec3(0.5, 0.5, 0.5), Vec3(0.3, 0.3, 0.3)]

        stability = quality_metrics.measure_stability([frame1, frame2])
        assert stability == 1.0  # Perfect stability

    def test_measure_stability_varying(self, quality_metrics):
        """Test stability with varying frames."""
        frame1 = [Vec3(0.5, 0.5, 0.5)]
        frame2 = [Vec3(0.6, 0.6, 0.6)]
        frame3 = [Vec3(0.4, 0.4, 0.4)]

        stability = quality_metrics.measure_stability([frame1, frame2, frame3])
        assert 0 < stability < 1.0

    def test_measure_stability_single_frame(self, quality_metrics):
        """Test stability with single frame."""
        frame = [Vec3(0.5, 0.5, 0.5)]
        stability = quality_metrics.measure_stability([frame])
        assert stability == 1.0

    def test_detect_light_leaks_no_leaks(self, quality_metrics):
        """Test light leak detection with no leaks."""
        samples = [
            (Vec3(0, 0, 0), Vec3(0.0, 0.0, 0.0), True),  # Occluded, dark
            (Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), False),  # Visible, lit
        ]

        count, severity = quality_metrics.detect_light_leaks(samples)
        assert count == 0
        assert severity == 0.0

    def test_detect_light_leaks_with_leaks(self, quality_metrics):
        """Test light leak detection with leaks."""
        samples = [
            (Vec3(0, 0, 0), Vec3(0.5, 0.5, 0.5), True),  # Occluded but lit = leak
            (Vec3(1, 0, 0), Vec3(0.5, 0.5, 0.5), False),  # Visible, lit
        ]

        count, severity = quality_metrics.detect_light_leaks(samples)
        assert count == 1
        assert severity > 0

    def test_full_assessment(self, quality_metrics):
        """Test full quality assessment."""
        test_samples = [Vec3(0.5, 0.5, 0.5)]
        reference_samples = [Vec3(0.52, 0.48, 0.5)]
        frame_history = [
            [Vec3(0.5, 0.5, 0.5)],
            [Vec3(0.51, 0.49, 0.5)],
        ]
        occlusion_samples = [
            (Vec3(0, 0, 0), Vec3(0.0, 0.0, 0.0), True),
        ]

        result = quality_metrics.full_assessment(
            test_samples=test_samples,
            reference_samples=reference_samples,
            frame_history=frame_history,
            occlusion_samples=occlusion_samples,
        )

        assert isinstance(result, QualityMetricResult)
        assert result.psnr_db > 0
        assert 0 <= result.temporal_stability <= 1
        assert result.light_leak_count >= 0
        assert 0 <= result.overall_score <= 1


# ============================================================================
# CostBudgetAnalyzer Tests
# ============================================================================


class TestCostBudgetAnalyzer:
    """Tests for CostBudgetAnalyzer."""

    def test_estimate_lumen_cost(self, cost_analyzer):
        """Test Lumen cost estimation."""
        total, breakdown = cost_analyzer.estimate_lumen_cost(
            mesh_count=100,
            screen_width=1920,
            screen_height=1080,
            traces_per_frame=500_000,
        )

        assert total > 0
        assert "mesh_cards" in breakdown
        assert "screen_probes" in breakdown
        assert "sdf_tracing" in breakdown
        assert "radiance_cache" in breakdown
        assert abs(sum(breakdown.values()) - total) < 0.001

    def test_estimate_ddgi_cost(self, cost_analyzer):
        """Test DDGI cost estimation."""
        total, breakdown = cost_analyzer.estimate_ddgi_cost(
            probe_count=4096,
            rays_per_probe=256,
            update_fraction=0.1,
            cache_cells=1_000_000,
        )

        assert total > 0
        assert "probe_tracing" in breakdown
        assert "probe_update" in breakdown
        assert "radiance_cache" in breakdown
        assert "sampling" in breakdown

    def test_compare_at_budget(self, cost_analyzer, default_bounds):
        """Test comparison at fixed budget."""
        result = cost_analyzer.compare_at_budget(
            budget_ms=2.0,
            scene_mesh_count=500,
            scene_bounds=default_bounds,
        )

        assert isinstance(result, CostBudgetResult)
        assert result.lumen_total_ms > 0
        assert result.ddgi_total_ms > 0
        assert len(result.lumen_breakdown) > 0
        assert len(result.ddgi_breakdown) > 0
        assert len(result.equivalent_quality_winner) > 0

    def test_compare_at_budget_different_budgets(self, cost_analyzer, default_bounds):
        """Test comparison at different budgets."""
        result_low = cost_analyzer.compare_at_budget(
            budget_ms=1.0,
            scene_mesh_count=100,
            scene_bounds=default_bounds,
        )

        result_high = cost_analyzer.compare_at_budget(
            budget_ms=5.0,
            scene_mesh_count=100,
            scene_bounds=default_bounds,
        )

        # Higher budget should allow more resources
        # (DDGI may have more probes)
        assert result_high.ddgi_total_ms >= result_low.ddgi_total_ms


# ============================================================================
# FeasibilityAssessment Tests
# ============================================================================


class TestFeasibilityAssessment:
    """Tests for FeasibilityAssessment."""

    def test_init_default(self):
        """Test default initialization."""
        assessment = FeasibilityAssessment()
        assert assessment.mesh_card_eval is not None
        assert assessment.screen_probe_eval is not None
        assert assessment.sdf_eval is not None
        assert assessment.cache_comp is not None
        assert assessment.quality is not None
        assert assessment.cost_analyzer is not None

    def test_assess_simple_scene(self, feasibility_assessment, default_bounds):
        """Test assessment of simple scene."""
        report = feasibility_assessment.assess(
            scene_mesh_count=100,
            scene_triangle_count=100_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
            budget_ms=2.0,
        )

        assert isinstance(report, FeasibilityReport)
        assert isinstance(report.decision, DecisionOutcome)
        assert len(report.rationale) > 0
        assert len(report.component_assessments) == 4  # 4 components
        assert len(report.quality_comparison) > 0
        assert isinstance(report.cost_comparison, CostBudgetResult)

    def test_assess_complex_scene(self, feasibility_assessment, default_bounds):
        """Test assessment of complex scene."""
        report = feasibility_assessment.assess(
            scene_mesh_count=10000,
            scene_triangle_count=50_000_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
            budget_ms=2.0,
        )

        # Complex scene should have more concerns
        issues_total = sum(
            len(a.issues) for a in report.component_assessments
        )
        assert issues_total > 0

    def test_assess_generates_implementation_plan_on_go(
        self, feasibility_assessment, default_bounds
    ):
        """Test that GO decision includes implementation plan."""
        # Simple scene should get GO
        report = feasibility_assessment.assess(
            scene_mesh_count=50,
            scene_triangle_count=50_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
            budget_ms=5.0,
        )

        if report.decision != DecisionOutcome.NO_GO:
            assert len(report.implementation_plan) > 0
            assert report.timeline_weeks > 0

    def test_assess_includes_risks(self, feasibility_assessment, default_bounds):
        """Test that assessment includes risks."""
        report = feasibility_assessment.assess(
            scene_mesh_count=100,
            scene_triangle_count=100_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
        )

        assert len(report.risks) > 0

    def test_generate_report(self, feasibility_assessment, default_bounds):
        """Test report generation."""
        report = feasibility_assessment.assess(
            scene_mesh_count=100,
            scene_triangle_count=100_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
        )

        text = feasibility_assessment.generate_report(report)

        assert "# Lumen-Lite Feasibility Study" in text
        assert "Executive Summary" in text
        assert "Component Analysis" in text
        assert report.decision.name in text

    def test_create_plan(self, feasibility_assessment, default_bounds):
        """Test implementation plan creation."""
        report = feasibility_assessment.assess(
            scene_mesh_count=100,
            scene_triangle_count=100_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
            budget_ms=5.0,
        )

        plan = feasibility_assessment.create_plan(report)

        if report.decision != DecisionOutcome.NO_GO:
            assert len(plan) > 0
            assert "phase" in plan[0]
            assert "name" in plan[0]
            assert "duration" in plan[0]
            assert "tasks" in plan[0]

    def test_assess_all_component_types_present(
        self, feasibility_assessment, default_bounds
    ):
        """Test that all component types are assessed."""
        report = feasibility_assessment.assess(
            scene_mesh_count=100,
            scene_triangle_count=100_000,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
        )

        component_types = {a.component for a in report.component_assessments}
        assert ComponentType.MESH_CARDS in component_types
        assert ComponentType.SCREEN_PROBES in component_types
        assert ComponentType.SOFTWARE_SDF in component_types
        assert ComponentType.RADIANCE_CACHE in component_types


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_run_feasibility_study_default(self):
        """Test run_feasibility_study with defaults."""
        report = run_feasibility_study()

        assert isinstance(report, FeasibilityReport)
        assert isinstance(report.decision, DecisionOutcome)

    def test_run_feasibility_study_custom(self):
        """Test run_feasibility_study with custom parameters."""
        bounds = AABB(Vec3(-100, 0, -100), Vec3(100, 50, 100))

        report = run_feasibility_study(
            scene_mesh_count=500,
            scene_triangle_count=2_000_000,
            scene_bounds=bounds,
            ddgi_probe_count=8192,
            trinity_cache_dims=(128, 128, 64),
            budget_ms=3.0,
        )

        assert isinstance(report, FeasibilityReport)

    def test_quick_assessment_low(self):
        """Test quick assessment for low complexity."""
        decision, summary = quick_assessment("low")

        assert isinstance(decision, DecisionOutcome)
        assert len(summary) > 0
        assert "low" in summary.lower()

    def test_quick_assessment_medium(self):
        """Test quick assessment for medium complexity."""
        decision, summary = quick_assessment("medium")

        assert isinstance(decision, DecisionOutcome)
        assert "medium" in summary.lower()

    def test_quick_assessment_high(self):
        """Test quick assessment for high complexity."""
        decision, summary = quick_assessment("high")

        assert isinstance(decision, DecisionOutcome)
        assert "high" in summary.lower()

    def test_quick_assessment_extreme(self):
        """Test quick assessment for extreme complexity."""
        decision, summary = quick_assessment("extreme")

        assert isinstance(decision, DecisionOutcome)
        assert "extreme" in summary.lower()

    def test_quick_assessment_invalid(self):
        """Test quick assessment with invalid complexity."""
        decision, summary = quick_assessment("invalid")

        # Should fall back to medium
        assert isinstance(decision, DecisionOutcome)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_mesh_count(self, feasibility_assessment, default_bounds):
        """Test with zero meshes."""
        report = feasibility_assessment.assess(
            scene_mesh_count=0,
            scene_triangle_count=0,
            scene_bounds=default_bounds,
            ddgi_probe_count=4096,
            trinity_cache_dims=(64, 64, 32),
        )

        assert isinstance(report, FeasibilityReport)

    def test_single_mesh(self, mesh_card_evaluator):
        """Test single mesh evaluation."""
        cost = mesh_card_evaluator.estimate_card_cost(
            vertex_count=3,  # Single triangle
            triangle_count=1,
            surface_area=0.001,
            has_uv=True,
        )

        assert cost.card_count >= 1

    def test_very_small_bounds(self, sdf_evaluator):
        """Test with very small scene bounds."""
        small_bounds = AABB(Vec3(0, 0, 0), Vec3(0.1, 0.1, 0.1))

        result = sdf_evaluator.full_evaluation(
            mesh_count=1,
            total_triangles=100,
            scene_bounds=small_bounds,
        )

        assert isinstance(result, SDFEvaluationResult)

    def test_very_large_bounds(self, sdf_evaluator):
        """Test with very large scene bounds."""
        large_bounds = AABB(Vec3(-1000, -100, -1000), Vec3(1000, 500, 1000))

        result = sdf_evaluator.full_evaluation(
            mesh_count=1000,
            total_triangles=10_000_000,
            scene_bounds=large_bounds,
        )

        assert isinstance(result, SDFEvaluationResult)

    def test_zero_budget(self, cost_analyzer, default_bounds):
        """Test comparison with zero budget."""
        result = cost_analyzer.compare_at_budget(
            budget_ms=0.0,
            scene_mesh_count=100,
            scene_bounds=default_bounds,
        )

        # Should still produce valid result
        assert isinstance(result, CostBudgetResult)

    def test_extreme_probe_count(self, screen_probe_evaluator, default_bounds):
        """Test with extreme probe counts."""
        # Very few DDGI probes
        _, ddgi_cov_low, _ = screen_probe_evaluator.compare_coverage(
            ddgi_probe_count=8,
            ddgi_bounds=default_bounds,
        )

        # Many DDGI probes
        _, ddgi_cov_high, _ = screen_probe_evaluator.compare_coverage(
            ddgi_probe_count=100000,
            ddgi_bounds=default_bounds,
        )

        # More probes should have better coverage
        assert ddgi_cov_high >= ddgi_cov_low


# ============================================================================
# Component Assessment Tests
# ============================================================================


class TestComponentAssessment:
    """Tests for ComponentAssessment data structure."""

    def test_component_assessment_creation(self):
        """Test creating ComponentAssessment."""
        assessment = ComponentAssessment(
            component=ComponentType.MESH_CARDS,
            feasibility=FeasibilityLevel.HIGH,
            cost_estimate=0.5,
            memory_estimate=1024,
            issues=["Minor issue"],
            recommendations=["Use caching"],
        )

        assert assessment.component == ComponentType.MESH_CARDS
        assert assessment.feasibility == FeasibilityLevel.HIGH
        assert assessment.cost_estimate == 0.5
        assert assessment.memory_estimate == 1024
        assert len(assessment.issues) == 1
        assert len(assessment.recommendations) == 1


# ============================================================================
# Feasibility Report Tests
# ============================================================================


class TestFeasibilityReport:
    """Tests for FeasibilityReport data structure."""

    def test_feasibility_report_creation(self, default_bounds):
        """Test creating FeasibilityReport."""
        assessment = ComponentAssessment(
            component=ComponentType.MESH_CARDS,
            feasibility=FeasibilityLevel.HIGH,
            cost_estimate=0.5,
            memory_estimate=1024,
            issues=[],
            recommendations=[],
        )

        cost_result = CostBudgetResult(
            lumen_total_ms=1.0,
            lumen_breakdown={"mesh_cards": 0.5},
            ddgi_total_ms=1.5,
            ddgi_breakdown={"probe_tracing": 1.0},
            equivalent_quality_winner="Lumen",
            cost_efficiency=1.5,
        )

        report = FeasibilityReport(
            decision=DecisionOutcome.GO,
            rationale="All clear",
            component_assessments=[assessment],
            quality_comparison={"psnr": 35.0},
            cost_comparison=cost_result,
            implementation_plan=["Phase 1"],
            risks=["Risk 1"],
            timeline_weeks=8,
        )

        assert report.decision == DecisionOutcome.GO
        assert len(report.component_assessments) == 1
        assert report.timeline_weeks == 8


# ============================================================================
# Decision Outcome Tests
# ============================================================================


class TestDecisionOutcome:
    """Tests for DecisionOutcome enum."""

    def test_decision_outcomes(self):
        """Test all decision outcomes exist."""
        assert DecisionOutcome.GO is not None
        assert DecisionOutcome.NO_GO is not None
        assert DecisionOutcome.CONDITIONAL_GO is not None

    def test_decision_outcome_values(self):
        """Test decision outcome values are distinct."""
        values = [d.value for d in DecisionOutcome]
        assert len(values) == len(set(values))


# ============================================================================
# Feasibility Level Tests
# ============================================================================


class TestFeasibilityLevel:
    """Tests for FeasibilityLevel enum."""

    def test_feasibility_levels(self):
        """Test all feasibility levels exist."""
        assert FeasibilityLevel.HIGH is not None
        assert FeasibilityLevel.MEDIUM is not None
        assert FeasibilityLevel.LOW is not None
        assert FeasibilityLevel.BLOCKED is not None

    def test_feasibility_level_ordering(self):
        """Test feasibility level ordering."""
        assert FeasibilityLevel.HIGH.value < FeasibilityLevel.MEDIUM.value
        assert FeasibilityLevel.MEDIUM.value < FeasibilityLevel.LOW.value
        assert FeasibilityLevel.LOW.value < FeasibilityLevel.BLOCKED.value


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_simple_scene(self):
        """Test complete pipeline for simple scene."""
        report = run_feasibility_study(
            scene_mesh_count=50,
            scene_triangle_count=25_000,
            budget_ms=3.0,
        )

        # Simple scene should be feasible
        assert report.decision in [DecisionOutcome.GO, DecisionOutcome.CONDITIONAL_GO]

        # Generate report text
        assessment = FeasibilityAssessment()
        text = assessment.generate_report(report)
        assert "Lumen-Lite" in text

        # Create plan
        plan = assessment.create_plan(report)
        assert len(plan) > 0

    def test_full_pipeline_complex_scene(self):
        """Test complete pipeline for complex scene."""
        report = run_feasibility_study(
            scene_mesh_count=5000,
            scene_triangle_count=25_000_000,
            budget_ms=1.0,  # Tight budget
        )

        # Complex scene with tight budget should have concerns
        total_issues = sum(
            len(a.issues) for a in report.component_assessments
        )
        assert total_issues > 0

    def test_pipeline_consistency(self):
        """Test that pipeline produces consistent results."""
        # Run twice with same parameters
        report1 = run_feasibility_study(scene_mesh_count=100)
        report2 = run_feasibility_study(scene_mesh_count=100)

        # Results should be identical
        assert report1.decision == report2.decision
        assert len(report1.component_assessments) == len(report2.component_assessments)
