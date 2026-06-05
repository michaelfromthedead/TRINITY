"""Lumen-Lite Feasibility Study for TRINITY (T-GIR-P11.3).

This module provides a comprehensive evaluation framework for assessing the
feasibility of implementing Lumen-style GI approaches versus TRINITY's existing
DDGI infrastructure.

Components Evaluated:
    - Mesh Cards: Surface card generation for radiance caching
    - Screen Probes: Screen-space probe placement vs world-space DDGI
    - Software SDF Tracing: Signed distance field construction and tracing
    - Radiance Cache: Comparison with TRINITY's P2.7 implementation

The study produces a go/no-go decision based on:
    - Quality comparison at equivalent GPU budgets
    - Memory overhead analysis
    - Implementation complexity assessment
    - Integration effort estimation

References:
    - Siggraph 2022: "Lumen: Real-Time Global Illumination in Unreal Engine 5"
    - Epic Games GDC 2021: "Radiance Caching for Real-Time GI"
    - DDGI Paper (JCGT 2019): Dynamic Diffuse Global Illumination
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3


# ============================================================================
# Constants
# ============================================================================

# Performance baselines (ms)
MS_PER_MILLION_RAYS = 0.5  # RT hardware typical
SDF_TRACE_MS_PER_MILLION = 0.15  # Software SDF tracing
SCREEN_PROBE_UPDATE_MS = 0.3  # Per 1080p frame

# Memory constants (bytes)
BYTES_PER_MESH_CARD = 256  # Position, normal, tangent, bounds, radiance
BYTES_PER_SDF_TEXEL = 4  # Single float distance
BYTES_PER_SCREEN_PROBE = 64  # SH L1 RGB + metadata
BYTES_PER_DDGI_PROBE = 128  # SH L2 RGB + visibility

# Quality thresholds
PSNR_EXCELLENT = 40.0  # dB
PSNR_GOOD = 35.0
PSNR_ACCEPTABLE = 30.0
PSNR_POOR = 25.0

# Feasibility thresholds
FEASIBILITY_HIGH_THRESHOLD = 0.7
FEASIBILITY_MEDIUM_THRESHOLD = 0.4


# ============================================================================
# Enums
# ============================================================================


class FeasibilityLevel(Enum):
    """Feasibility assessment level."""

    HIGH = auto()    # Straightforward implementation
    MEDIUM = auto()  # Moderate complexity
    LOW = auto()     # Significant challenges
    BLOCKED = auto()  # Fundamental blockers


class ComponentType(Enum):
    """Lumen component types."""

    MESH_CARDS = auto()
    SCREEN_PROBES = auto()
    SOFTWARE_SDF = auto()
    RADIANCE_CACHE = auto()


class DecisionOutcome(Enum):
    """Go/no-go decision outcome."""

    GO = auto()
    NO_GO = auto()
    CONDITIONAL_GO = auto()  # Go with caveats


# ============================================================================
# Mesh Card Evaluation
# ============================================================================


@dataclass
class MeshCardCostEstimate:
    """Cost estimate for mesh card generation.

    Attributes:
        vertex_count: Source mesh vertex count
        uv_chart_count: Number of UV charts
        card_count: Generated card count
        generation_time_ms: One-time generation cost
        memory_bytes: Runtime memory consumption
        update_time_ms: Per-frame update cost
    """

    vertex_count: int
    uv_chart_count: int
    card_count: int
    generation_time_ms: float
    memory_bytes: int
    update_time_ms: float


@dataclass
class MeshCardEvaluatorConfig:
    """Configuration for mesh card evaluation.

    Attributes:
        min_card_size: Minimum card size in world units
        max_cards_per_mesh: Maximum cards per mesh
        target_texels_per_meter: Target texel density
        enable_importance_sampling: Use importance-based card placement
    """

    min_card_size: float = 0.1
    max_cards_per_mesh: int = 64
    target_texels_per_meter: float = 8.0
    enable_importance_sampling: bool = True


class MeshCardEvaluator:
    """Evaluates mesh card generation complexity and cost.

    Mesh cards are simplified surface representations used by Lumen
    to cache radiance on mesh surfaces. This evaluator assesses:
        - Card generation pipeline cost
        - Memory overhead per mesh
        - Runtime update costs

    Usage:
        evaluator = MeshCardEvaluator(config)
        cost = evaluator.estimate_card_cost(mesh_info)
        feasibility = evaluator.evaluate_mesh(mesh_info)
    """

    def __init__(self, config: Optional[MeshCardEvaluatorConfig] = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or MeshCardEvaluatorConfig()
        self._cache: Dict[int, MeshCardCostEstimate] = {}

    def estimate_card_cost(
        self,
        vertex_count: int,
        triangle_count: int,
        surface_area: float,
        has_uv: bool = True,
    ) -> MeshCardCostEstimate:
        """Estimate cost for mesh card generation.

        Args:
            vertex_count: Number of vertices in mesh
            triangle_count: Number of triangles
            surface_area: Total surface area (square meters)
            has_uv: Whether mesh has UV coordinates

        Returns:
            Cost estimate for this mesh
        """
        # Estimate UV chart count
        if has_uv:
            # Assume roughly 1 chart per 100 triangles
            uv_chart_count = max(1, triangle_count // 100)
        else:
            # No UVs: need automatic UV generation
            uv_chart_count = max(1, int(math.sqrt(triangle_count)))

        # Card count based on surface area and config
        min_card_area = self.config.min_card_size ** 2
        cards_by_area = int(surface_area / min_card_area)
        card_count = min(cards_by_area, self.config.max_cards_per_mesh)
        card_count = max(1, card_count)

        # Generation time: O(V + T * log(cards))
        # UV generation dominates if needed
        uv_gen_factor = 2.0 if not has_uv else 1.0
        base_gen_ms = (vertex_count / 10000.0) * 0.1 * uv_gen_factor
        card_gen_ms = card_count * 0.01
        generation_time_ms = base_gen_ms + card_gen_ms

        # Memory: per-card storage
        memory_bytes = card_count * BYTES_PER_MESH_CARD

        # Update time: proportional to card count
        update_time_ms = card_count * 0.001

        return MeshCardCostEstimate(
            vertex_count=vertex_count,
            uv_chart_count=uv_chart_count,
            card_count=card_count,
            generation_time_ms=generation_time_ms,
            memory_bytes=memory_bytes,
            update_time_ms=update_time_ms,
        )

    def evaluate_mesh(
        self,
        vertex_count: int,
        triangle_count: int,
        surface_area: float,
        has_uv: bool = True,
    ) -> Tuple[FeasibilityLevel, str]:
        """Evaluate feasibility for a single mesh.

        Args:
            vertex_count: Vertex count
            triangle_count: Triangle count
            surface_area: Surface area
            has_uv: Has UV coordinates

        Returns:
            Tuple of (feasibility_level, explanation)
        """
        cost = self.estimate_card_cost(
            vertex_count, triangle_count, surface_area, has_uv
        )

        issues = []

        # Check generation time
        if cost.generation_time_ms > 100.0:
            issues.append(f"High generation time: {cost.generation_time_ms:.1f}ms")

        # Check memory
        if cost.memory_bytes > 64 * 1024:  # 64KB per mesh
            issues.append(f"High memory: {cost.memory_bytes / 1024:.1f}KB")

        # Check card count
        if cost.card_count >= self.config.max_cards_per_mesh:
            issues.append(f"Max cards reached: {cost.card_count}")

        # No UVs is a moderate issue
        if not has_uv:
            issues.append("No UVs: requires automatic UV generation")

        if len(issues) == 0:
            return FeasibilityLevel.HIGH, "No issues detected"
        elif len(issues) == 1:
            return FeasibilityLevel.MEDIUM, issues[0]
        elif len(issues) <= 2:
            return FeasibilityLevel.LOW, "; ".join(issues)
        else:
            return FeasibilityLevel.BLOCKED, "; ".join(issues)

    def get_memory_overhead(
        self,
        total_meshes: int,
        avg_cards_per_mesh: int,
    ) -> int:
        """Calculate total memory overhead for scene.

        Args:
            total_meshes: Number of meshes in scene
            avg_cards_per_mesh: Average cards per mesh

        Returns:
            Total memory in bytes
        """
        return total_meshes * avg_cards_per_mesh * BYTES_PER_MESH_CARD

    def estimate_scene_cost(
        self,
        mesh_infos: List[Tuple[int, int, float, bool]],
    ) -> Tuple[float, int, FeasibilityLevel]:
        """Estimate total cost for a scene.

        Args:
            mesh_infos: List of (vertex_count, triangle_count, area, has_uv)

        Returns:
            Tuple of (total_gen_ms, total_memory, feasibility)
        """
        total_gen_ms = 0.0
        total_memory = 0
        worst_feasibility = FeasibilityLevel.HIGH

        for vertex_count, triangle_count, area, has_uv in mesh_infos:
            cost = self.estimate_card_cost(
                vertex_count, triangle_count, area, has_uv
            )
            total_gen_ms += cost.generation_time_ms
            total_memory += cost.memory_bytes

            feasibility, _ = self.evaluate_mesh(
                vertex_count, triangle_count, area, has_uv
            )
            if feasibility.value > worst_feasibility.value:
                worst_feasibility = feasibility

        return total_gen_ms, total_memory, worst_feasibility


# ============================================================================
# Screen Probe Evaluation
# ============================================================================


@dataclass
class ScreenProbeAnalysis:
    """Analysis results for screen-space probe placement.

    Attributes:
        probe_count: Number of probes at target resolution
        coverage_percentage: Percentage of visible surfaces covered
        temporal_stability_score: Stability score [0, 1]
        parallax_artifacts_score: Parallax artifact severity [0, 1]
        update_cost_ms: Per-frame update cost
        memory_bytes: Total memory consumption
    """

    probe_count: int
    coverage_percentage: float
    temporal_stability_score: float
    parallax_artifacts_score: float
    update_cost_ms: float
    memory_bytes: int


@dataclass
class ScreenProbeEvaluatorConfig:
    """Configuration for screen probe evaluation.

    Attributes:
        screen_width: Target screen width
        screen_height: Target screen height
        probe_spacing: Pixels between probes
        temporal_filter_strength: Temporal accumulation strength [0, 1]
        use_importance_sampling: Use importance-based probe placement
    """

    screen_width: int = 1920
    screen_height: int = 1080
    probe_spacing: int = 16
    temporal_filter_strength: float = 0.95
    use_importance_sampling: bool = True


class ScreenProbeEvaluator:
    """Evaluates screen-space probe placement versus world-space DDGI.

    Screen probes are placed in screen space and offer advantages for
    view-dependent effects but suffer from temporal instability and
    parallax issues.

    Comparison criteria:
        - Coverage: How much of the visible scene is covered
        - Temporal stability: Consistency across frames
        - Parallax handling: Quality with camera motion
        - Cost: GPU time per frame
    """

    def __init__(self, config: Optional[ScreenProbeEvaluatorConfig] = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or ScreenProbeEvaluatorConfig()

    def compare_coverage(
        self,
        ddgi_probe_count: int,
        ddgi_bounds: AABB,
        view_distance: float = 100.0,
    ) -> Tuple[float, float, str]:
        """Compare coverage between screen probes and DDGI.

        Args:
            ddgi_probe_count: Number of DDGI probes
            ddgi_bounds: World bounds covered by DDGI
            view_distance: Maximum view distance

        Returns:
            Tuple of (screen_coverage, ddgi_coverage, analysis)
        """
        # Screen probe count
        probes_x = self.config.screen_width // self.config.probe_spacing
        probes_y = self.config.screen_height // self.config.probe_spacing
        screen_probe_count = probes_x * probes_y

        # Screen coverage: 100% of visible surfaces by design
        screen_coverage = 1.0

        # DDGI coverage: depends on probe density vs scene size
        ddgi_extent = ddgi_bounds.max - ddgi_bounds.min
        ddgi_volume = ddgi_extent.x * ddgi_extent.y * ddgi_extent.z

        # Effective coverage radius per probe (approximate)
        avg_spacing = (ddgi_volume / ddgi_probe_count) ** (1.0 / 3.0)
        # Coverage degrades at boundaries
        ddgi_coverage = min(1.0, 0.9 * (1.0 - avg_spacing / view_distance))

        analysis = (
            f"Screen: {screen_probe_count} probes, 100% visible coverage. "
            f"DDGI: {ddgi_probe_count} probes, ~{ddgi_coverage * 100:.0f}% "
            f"effective coverage (spacing: {avg_spacing:.1f}m)"
        )

        return screen_coverage, ddgi_coverage, analysis

    def analyze_stability(
        self,
        camera_velocity: float = 0.0,
        scene_motion_percentage: float = 0.0,
    ) -> Tuple[float, float, str]:
        """Analyze temporal stability comparison.

        Args:
            camera_velocity: Camera velocity (m/s)
            scene_motion_percentage: Percentage of scene in motion

        Returns:
            Tuple of (screen_stability, ddgi_stability, analysis)
        """
        # DDGI stability: world-space probes are inherently stable
        # Only affected by dynamic lighting changes
        ddgi_base_stability = 0.95
        ddgi_stability = ddgi_base_stability * (1.0 - scene_motion_percentage * 0.1)

        # Screen probe stability: affected by camera motion
        # Temporal filter helps but introduces ghosting
        screen_base_stability = 0.7
        camera_factor = 1.0 - min(1.0, camera_velocity / 10.0)
        temporal_help = self.config.temporal_filter_strength * 0.2
        screen_stability = screen_base_stability * camera_factor + temporal_help

        analysis = (
            f"Screen probes: {screen_stability:.2f} stability "
            f"(camera velocity impact: {1.0 - camera_factor:.2f}). "
            f"DDGI: {ddgi_stability:.2f} stability (scene motion impact: "
            f"{scene_motion_percentage * 0.1:.2f})"
        )

        return screen_stability, ddgi_stability, analysis

    def estimate_cost(
        self,
        ddgi_probe_count: int,
        rays_per_ddgi_probe: int = 256,
    ) -> Tuple[float, float, str]:
        """Estimate per-frame cost comparison.

        Args:
            ddgi_probe_count: DDGI probe count
            rays_per_ddgi_probe: Rays per DDGI probe

        Returns:
            Tuple of (screen_cost_ms, ddgi_cost_ms, analysis)
        """
        # Screen probe cost: fixed for resolution
        probes_x = self.config.screen_width // self.config.probe_spacing
        probes_y = self.config.screen_height // self.config.probe_spacing
        screen_probe_count = probes_x * probes_y
        rays_per_screen_probe = 64  # Typical screen probe ray count
        screen_rays = screen_probe_count * rays_per_screen_probe
        screen_cost_ms = (screen_rays / 1_000_000) * MS_PER_MILLION_RAYS

        # DDGI cost: depends on update rate
        # Assume 10% of probes update per frame
        ddgi_update_fraction = 0.1
        ddgi_rays = int(ddgi_probe_count * ddgi_update_fraction * rays_per_ddgi_probe)
        ddgi_cost_ms = (ddgi_rays / 1_000_000) * MS_PER_MILLION_RAYS

        analysis = (
            f"Screen: {screen_probe_count} probes * {rays_per_screen_probe} rays = "
            f"{screen_cost_ms:.2f}ms. "
            f"DDGI: {ddgi_probe_count} probes * {ddgi_update_fraction * 100:.0f}% "
            f"* {rays_per_ddgi_probe} rays = {ddgi_cost_ms:.2f}ms"
        )

        return screen_cost_ms, ddgi_cost_ms, analysis

    def full_analysis(
        self,
        ddgi_probe_count: int,
        ddgi_bounds: AABB,
        rays_per_ddgi_probe: int = 256,
        camera_velocity: float = 5.0,
    ) -> ScreenProbeAnalysis:
        """Perform full screen probe analysis.

        Args:
            ddgi_probe_count: DDGI comparison probe count
            ddgi_bounds: DDGI bounds
            rays_per_ddgi_probe: DDGI rays per probe
            camera_velocity: Typical camera velocity

        Returns:
            Complete analysis results
        """
        probes_x = self.config.screen_width // self.config.probe_spacing
        probes_y = self.config.screen_height // self.config.probe_spacing
        probe_count = probes_x * probes_y

        coverage, _, _ = self.compare_coverage(ddgi_probe_count, ddgi_bounds)
        stability, _, _ = self.analyze_stability(camera_velocity)
        cost_ms, _, _ = self.estimate_cost(ddgi_probe_count, rays_per_ddgi_probe)

        # Parallax artifacts: worse at higher velocities and lower probe density
        parallax_score = 1.0 - min(1.0, camera_velocity / 20.0) * (
            1.0 - self.config.probe_spacing / 32.0
        )

        memory_bytes = probe_count * BYTES_PER_SCREEN_PROBE

        return ScreenProbeAnalysis(
            probe_count=probe_count,
            coverage_percentage=coverage * 100.0,
            temporal_stability_score=stability,
            parallax_artifacts_score=parallax_score,
            update_cost_ms=cost_ms,
            memory_bytes=memory_bytes,
        )


# ============================================================================
# SDF Evaluation
# ============================================================================


@dataclass
class SDFEvaluationResult:
    """Results from SDF evaluation.

    Attributes:
        mesh_sdf_memory_bytes: Memory for mesh SDFs
        global_sdf_memory_bytes: Memory for global distance field
        trace_cost_ms: Cost per million traces
        construction_time_ms: One-time construction cost
        quality_score: Quality score [0, 1]
        feasibility: Overall feasibility
    """

    mesh_sdf_memory_bytes: int
    global_sdf_memory_bytes: int
    trace_cost_ms: float
    construction_time_ms: float
    quality_score: float
    feasibility: FeasibilityLevel


@dataclass
class SDFEvaluatorConfig:
    """Configuration for SDF evaluation.

    Attributes:
        mesh_sdf_resolution: Resolution for per-mesh SDFs
        global_sdf_resolution: Resolution for global distance field
        max_trace_distance: Maximum trace distance (meters)
        use_sparse_sdf: Use sparse SDF representation
    """

    mesh_sdf_resolution: int = 64
    global_sdf_resolution: int = 256
    max_trace_distance: float = 100.0
    use_sparse_sdf: bool = True


class SDFEvaluator:
    """Evaluates SDF construction and tracing feasibility.

    Signed Distance Fields enable efficient software ray tracing by
    providing distance-to-surface queries. Lumen uses mesh SDFs for
    detailed geometry and a global distance field for acceleration.

    Evaluation criteria:
        - Memory: SDF storage requirements
        - Construction: Build time for mesh/global SDFs
        - Tracing: Per-ray trace cost
        - Quality: Accuracy vs actual geometry
    """

    def __init__(self, config: Optional[SDFEvaluatorConfig] = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config or SDFEvaluatorConfig()

    def estimate_sdf_memory(
        self,
        mesh_count: int,
        avg_mesh_extents: Vec3,
        scene_bounds: AABB,
    ) -> Tuple[int, int]:
        """Estimate SDF memory requirements.

        Args:
            mesh_count: Number of meshes requiring SDFs
            avg_mesh_extents: Average mesh bounding box extents
            scene_bounds: Total scene bounds

        Returns:
            Tuple of (mesh_sdf_memory, global_sdf_memory)
        """
        # Per-mesh SDF memory
        res = self.config.mesh_sdf_resolution
        mesh_sdf_voxels = res ** 3

        if self.config.use_sparse_sdf:
            # Sparse: only store near-surface voxels (typically 10-20%)
            mesh_sdf_voxels = int(mesh_sdf_voxels * 0.15)

        mesh_sdf_bytes = mesh_sdf_voxels * BYTES_PER_SDF_TEXEL
        total_mesh_sdf = mesh_count * mesh_sdf_bytes

        # Global SDF memory
        global_res = self.config.global_sdf_resolution
        global_voxels = global_res ** 3

        if self.config.use_sparse_sdf:
            # Global SDF: typically sparser (5-10% occupancy)
            global_voxels = int(global_voxels * 0.08)

        global_sdf_bytes = global_voxels * BYTES_PER_SDF_TEXEL

        return total_mesh_sdf, global_sdf_bytes

    def estimate_trace_cost(
        self,
        trace_count: int,
        avg_trace_distance: float,
    ) -> float:
        """Estimate software SDF trace cost.

        Args:
            trace_count: Number of traces per frame
            avg_trace_distance: Average trace distance

        Returns:
            Cost in milliseconds
        """
        # Cost scales with trace count and distance
        distance_factor = avg_trace_distance / self.config.max_trace_distance
        traces_millions = trace_count / 1_000_000
        return traces_millions * SDF_TRACE_MS_PER_MILLION * (1.0 + distance_factor)

    def evaluate_quality(
        self,
        mesh_triangle_density: float,
        sdf_resolution: int,
    ) -> Tuple[float, str]:
        """Evaluate SDF quality vs mesh geometry.

        Args:
            mesh_triangle_density: Triangles per cubic meter
            sdf_resolution: SDF resolution

        Returns:
            Tuple of (quality_score, explanation)
        """
        # Quality depends on SDF resolution vs geometric detail
        # Higher triangle density needs higher SDF resolution
        required_res = int(math.sqrt(mesh_triangle_density) * 2)

        if sdf_resolution >= required_res * 1.5:
            quality = 0.95
            explanation = "SDF resolution exceeds requirements"
        elif sdf_resolution >= required_res:
            quality = 0.85
            explanation = "SDF resolution matches requirements"
        elif sdf_resolution >= required_res * 0.7:
            quality = 0.7
            explanation = "SDF resolution slightly below requirements"
        else:
            quality = 0.5
            explanation = f"SDF resolution insufficient (need {required_res})"

        return quality, explanation

    def full_evaluation(
        self,
        mesh_count: int,
        total_triangles: int,
        scene_bounds: AABB,
        traces_per_frame: int = 1_000_000,
    ) -> SDFEvaluationResult:
        """Perform full SDF feasibility evaluation.

        Args:
            mesh_count: Total mesh count
            total_triangles: Total triangle count
            scene_bounds: Scene bounds
            traces_per_frame: Expected traces per frame

        Returns:
            Complete evaluation results
        """
        # Calculate scene metrics
        extent = scene_bounds.max - scene_bounds.min
        scene_volume = extent.x * extent.y * extent.z
        triangle_density = total_triangles / max(scene_volume, 1.0)
        avg_extents = Vec3(extent.x / 10, extent.y / 10, extent.z / 10)

        # Memory estimation
        mesh_sdf_mem, global_sdf_mem = self.estimate_sdf_memory(
            mesh_count, avg_extents, scene_bounds
        )

        # Trace cost
        avg_trace_dist = min(extent.x, extent.y, extent.z) * 0.5
        trace_cost = self.estimate_trace_cost(traces_per_frame, avg_trace_dist)

        # Construction time (rough estimate)
        # O(mesh_count * resolution^3) for mesh SDFs
        # O(global_resolution^3) for global SDF
        mesh_construction = mesh_count * (self.config.mesh_sdf_resolution ** 3) / 1e9
        global_construction = (self.config.global_sdf_resolution ** 3) / 1e9
        construction_time_ms = (mesh_construction + global_construction) * 1000

        # Quality
        quality, _ = self.evaluate_quality(
            triangle_density, self.config.mesh_sdf_resolution
        )

        # Determine feasibility
        total_memory_mb = (mesh_sdf_mem + global_sdf_mem) / (1024 * 1024)

        if total_memory_mb > 512:
            feasibility = FeasibilityLevel.BLOCKED
        elif total_memory_mb > 256 or trace_cost > 2.0:
            feasibility = FeasibilityLevel.LOW
        elif total_memory_mb > 128 or trace_cost > 1.0:
            feasibility = FeasibilityLevel.MEDIUM
        else:
            feasibility = FeasibilityLevel.HIGH

        return SDFEvaluationResult(
            mesh_sdf_memory_bytes=mesh_sdf_mem,
            global_sdf_memory_bytes=global_sdf_mem,
            trace_cost_ms=trace_cost,
            construction_time_ms=construction_time_ms,
            quality_score=quality,
            feasibility=feasibility,
        )


# ============================================================================
# Radiance Cache Comparison
# ============================================================================


@dataclass
class RadianceCacheComparisonResult:
    """Results from radiance cache comparison.

    Attributes:
        trinity_memory_bytes: TRINITY P2.7 cache memory
        lumen_memory_bytes: Lumen-style cache memory
        trinity_update_cost_ms: TRINITY update cost
        lumen_update_cost_ms: Lumen update cost
        trinity_quality_score: TRINITY quality [0, 1]
        lumen_quality_score: Lumen quality [0, 1]
        winner_at_equal_cost: Which approach wins at equal cost
        analysis: Detailed analysis text
    """

    trinity_memory_bytes: int
    lumen_memory_bytes: int
    trinity_update_cost_ms: float
    lumen_update_cost_ms: float
    trinity_quality_score: float
    lumen_quality_score: float
    winner_at_equal_cost: str
    analysis: str


class RadianceCacheComparison:
    """Compares Lumen's radiance cache with TRINITY's P2.7 implementation.

    TRINITY's radiance cache (T-GIR-P2.7) uses a 3D grid with SH L1
    storage and temporal accumulation. Lumen uses screen-space
    radiance caching with importance sampling.

    Comparison metrics:
        - Resolution and coverage
        - Update frequency
        - Sampling patterns
        - Quality vs cost tradeoff
    """

    def __init__(self) -> None:
        """Initialize the comparison."""
        pass

    def compare_approaches(
        self,
        trinity_grid_dims: Tuple[int, int, int],
        trinity_bounds: AABB,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> RadianceCacheComparisonResult:
        """Compare TRINITY and Lumen radiance cache approaches.

        Args:
            trinity_grid_dims: TRINITY cache grid dimensions (w, h, d)
            trinity_bounds: TRINITY cache bounds
            screen_width: Screen width for Lumen
            screen_height: Screen height for Lumen

        Returns:
            Detailed comparison results
        """
        w, h, d = trinity_grid_dims

        # TRINITY memory: SH L1 (4 coeffs * 3 channels * 4 bytes) + metadata
        trinity_cell_bytes = 4 * 3 * 4 + 16  # 64 bytes
        trinity_memory = w * h * d * trinity_cell_bytes

        # Lumen memory: screen-space cache + history
        lumen_cache_cells = (screen_width // 8) * (screen_height // 8)
        lumen_cell_bytes = 32  # More compact screen-space format
        lumen_memory = lumen_cache_cells * lumen_cell_bytes * 2  # Double buffer

        # Update costs
        # TRINITY: update all cells from probes
        trinity_update_ms = (w * h * d) / 1_000_000 * 0.1

        # Lumen: screen-space update with temporal
        lumen_update_ms = (lumen_cache_cells / 1_000_000) * 0.15

        # Quality estimation
        # TRINITY: world-space provides consistent coverage
        trinity_extent = trinity_bounds.max - trinity_bounds.min
        trinity_cell_size = max(
            trinity_extent.x / w,
            trinity_extent.y / h,
            trinity_extent.z / d,
        )
        trinity_quality = min(1.0, 2.0 / trinity_cell_size)

        # Lumen: screen-space has view-dependent quality
        lumen_pixel_coverage = 8  # Pixels per cache cell
        lumen_quality = min(1.0, 4.0 / lumen_pixel_coverage)

        # Determine winner at equal cost budget
        cost_ratio = trinity_update_ms / max(lumen_update_ms, 0.001)
        quality_ratio = trinity_quality / max(lumen_quality, 0.001)

        if cost_ratio < 1.0 and quality_ratio >= 1.0:
            winner = "TRINITY (lower cost, equal/better quality)"
        elif cost_ratio > 1.0 and quality_ratio <= 1.0:
            winner = "Lumen (lower cost, equal/better quality)"
        elif quality_ratio > 1.1:
            winner = "TRINITY (better quality)"
        elif quality_ratio < 0.9:
            winner = "Lumen (better quality)"
        else:
            winner = "Tie (comparable performance)"

        analysis = (
            f"TRINITY: {w}x{h}x{d} grid, {trinity_memory / 1024:.0f}KB, "
            f"{trinity_update_ms:.2f}ms, quality={trinity_quality:.2f}. "
            f"Lumen: {lumen_cache_cells} cells, {lumen_memory / 1024:.0f}KB, "
            f"{lumen_update_ms:.2f}ms, quality={lumen_quality:.2f}"
        )

        return RadianceCacheComparisonResult(
            trinity_memory_bytes=trinity_memory,
            lumen_memory_bytes=lumen_memory,
            trinity_update_cost_ms=trinity_update_ms,
            lumen_update_cost_ms=lumen_update_ms,
            trinity_quality_score=trinity_quality,
            lumen_quality_score=lumen_quality,
            winner_at_equal_cost=winner,
            analysis=analysis,
        )

    def quality_assessment(
        self,
        test_positions: List[Vec3],
        ground_truth_sampler: Callable[[Vec3, Vec3], Vec3],
        trinity_sampler: Callable[[Vec3, Vec3], Vec3],
        normal: Vec3,
    ) -> Tuple[float, float]:
        """Assess quality via PSNR-like metric.

        Args:
            test_positions: Positions to sample
            ground_truth_sampler: Reference path-traced result
            trinity_sampler: TRINITY cache sampler
            normal: Surface normal for sampling

        Returns:
            Tuple of (trinity_quality_psnr, error_variance)
        """
        if not test_positions:
            return 0.0, 0.0

        total_error = 0.0
        errors = []

        for pos in test_positions:
            ref = ground_truth_sampler(pos, normal)
            trinity = trinity_sampler(pos, normal)

            # MSE for this sample
            error = (
                (ref.x - trinity.x) ** 2 +
                (ref.y - trinity.y) ** 2 +
                (ref.z - trinity.z) ** 2
            ) / 3.0
            errors.append(error)
            total_error += error

        mse = total_error / len(test_positions)

        # Convert to PSNR (assuming max value of 1.0)
        if mse < 1e-10:
            psnr = 100.0  # Perfect match
        else:
            psnr = 10.0 * math.log10(1.0 / mse)

        # Variance
        mean_error = sum(errors) / len(errors)
        variance = sum((e - mean_error) ** 2 for e in errors) / len(errors)

        return psnr, variance


# ============================================================================
# Quality Metrics
# ============================================================================


@dataclass
class QualityMetricResult:
    """Results from quality metric evaluation.

    Attributes:
        psnr_db: Peak signal-to-noise ratio (dB)
        temporal_stability: Temporal stability score [0, 1]
        light_leak_count: Number of detected light leaks
        light_leak_severity: Severity of leaks [0, 1]
        overall_score: Combined quality score [0, 1]
    """

    psnr_db: float
    temporal_stability: float
    light_leak_count: int
    light_leak_severity: float
    overall_score: float


class QualityMetrics:
    """Quality metrics for GI comparison.

    Provides objective quality measurements including:
        - PSNR against reference (path-traced)
        - Temporal stability analysis
        - Light leak detection
    """

    def __init__(self) -> None:
        """Initialize quality metrics."""
        self._reference_samples: List[Tuple[Vec3, Vec3]] = []

    def measure_psnr(
        self,
        test_samples: List[Vec3],
        reference_samples: List[Vec3],
    ) -> float:
        """Measure PSNR between test and reference samples.

        Args:
            test_samples: GI method output (RGB per sample)
            reference_samples: Reference path-traced output

        Returns:
            PSNR in decibels
        """
        if len(test_samples) != len(reference_samples) or not test_samples:
            return 0.0

        total_mse = 0.0

        for test, ref in zip(test_samples, reference_samples):
            mse = (
                (test.x - ref.x) ** 2 +
                (test.y - ref.y) ** 2 +
                (test.z - ref.z) ** 2
            ) / 3.0
            total_mse += mse

        mse = total_mse / len(test_samples)

        if mse < 1e-10:
            return 100.0

        return 10.0 * math.log10(1.0 / mse)

    def measure_stability(
        self,
        frame_samples: List[List[Vec3]],
    ) -> float:
        """Measure temporal stability across frames.

        Args:
            frame_samples: List of samples per frame

        Returns:
            Stability score [0, 1] (1 = perfectly stable)
        """
        if len(frame_samples) < 2:
            return 1.0

        total_variance = 0.0
        sample_count = 0

        # For each sample position, measure variance across frames
        num_samples = len(frame_samples[0])

        for i in range(num_samples):
            values = [frame[i] for frame in frame_samples if i < len(frame)]
            if len(values) < 2:
                continue

            # Compute variance for this sample
            mean = Vec3(
                sum(v.x for v in values) / len(values),
                sum(v.y for v in values) / len(values),
                sum(v.z for v in values) / len(values),
            )

            variance = sum(
                (v.x - mean.x) ** 2 + (v.y - mean.y) ** 2 + (v.z - mean.z) ** 2
                for v in values
            ) / len(values)

            total_variance += variance
            sample_count += 1

        if sample_count == 0:
            return 1.0

        avg_variance = total_variance / sample_count

        # Convert variance to stability score
        # Low variance = high stability
        return 1.0 / (1.0 + avg_variance * 10.0)

    def detect_light_leaks(
        self,
        samples: List[Tuple[Vec3, Vec3, bool]],
        threshold: float = 0.1,
    ) -> Tuple[int, float]:
        """Detect light leaks in GI results.

        Args:
            samples: List of (position, gi_value, is_occluded)
            threshold: Brightness threshold for leak detection

        Returns:
            Tuple of (leak_count, severity)
        """
        leak_count = 0
        total_severity = 0.0

        for pos, gi_value, is_occluded in samples:
            if is_occluded:
                # Should be dark but check for light
                brightness = (gi_value.x + gi_value.y + gi_value.z) / 3.0
                if brightness > threshold:
                    leak_count += 1
                    total_severity += brightness - threshold

        severity = total_severity / max(len(samples), 1)
        return leak_count, severity

    def full_assessment(
        self,
        test_samples: List[Vec3],
        reference_samples: List[Vec3],
        frame_history: List[List[Vec3]],
        occlusion_samples: List[Tuple[Vec3, Vec3, bool]],
    ) -> QualityMetricResult:
        """Perform full quality assessment.

        Args:
            test_samples: Current frame GI samples
            reference_samples: Reference samples
            frame_history: Historical frame samples
            occlusion_samples: Samples for leak detection

        Returns:
            Complete quality metric results
        """
        psnr = self.measure_psnr(test_samples, reference_samples)
        stability = self.measure_stability(frame_history)
        leak_count, leak_severity = self.detect_light_leaks(occlusion_samples)

        # Compute overall score
        psnr_score = min(1.0, psnr / PSNR_EXCELLENT)
        leak_score = 1.0 - min(1.0, leak_severity * 5.0)
        overall = (psnr_score * 0.5 + stability * 0.3 + leak_score * 0.2)

        return QualityMetricResult(
            psnr_db=psnr,
            temporal_stability=stability,
            light_leak_count=leak_count,
            light_leak_severity=leak_severity,
            overall_score=overall,
        )


# ============================================================================
# Cost Budget Analysis
# ============================================================================


@dataclass
class CostBudgetResult:
    """Results from cost budget analysis.

    Attributes:
        lumen_total_ms: Total Lumen pipeline cost
        lumen_breakdown: Per-component breakdown
        ddgi_total_ms: Total DDGI pipeline cost
        ddgi_breakdown: Per-component breakdown
        equivalent_quality_winner: Winner at equivalent cost
        cost_efficiency: Cost efficiency comparison
    """

    lumen_total_ms: float
    lumen_breakdown: Dict[str, float]
    ddgi_total_ms: float
    ddgi_breakdown: Dict[str, float]
    equivalent_quality_winner: str
    cost_efficiency: float


class CostBudgetAnalyzer:
    """Analyzes GPU time budgets for Lumen vs DDGI pipelines.

    Compares:
        - Lumen: mesh cards + screen probes + SDF tracing + cache
        - DDGI: probe tracing + probe update + sampling
    """

    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass

    def estimate_lumen_cost(
        self,
        mesh_count: int,
        screen_width: int,
        screen_height: int,
        traces_per_frame: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Estimate Lumen pipeline cost.

        Args:
            mesh_count: Number of meshes with cards
            screen_width: Screen width
            screen_height: Screen height
            traces_per_frame: SDF traces per frame

        Returns:
            Tuple of (total_ms, breakdown)
        """
        # Mesh card update (LOD selection, radiance injection)
        card_update_ms = mesh_count * 0.001

        # Screen probe update
        probe_count = (screen_width // 16) * (screen_height // 16)
        rays_per_probe = 64
        probe_ms = (probe_count * rays_per_probe / 1_000_000) * MS_PER_MILLION_RAYS

        # SDF tracing
        sdf_ms = (traces_per_frame / 1_000_000) * SDF_TRACE_MS_PER_MILLION

        # Radiance cache update
        cache_cells = (screen_width // 8) * (screen_height // 8)
        cache_ms = (cache_cells / 1_000_000) * 0.1

        breakdown = {
            "mesh_cards": card_update_ms,
            "screen_probes": probe_ms,
            "sdf_tracing": sdf_ms,
            "radiance_cache": cache_ms,
        }

        total = sum(breakdown.values())
        return total, breakdown

    def estimate_ddgi_cost(
        self,
        probe_count: int,
        rays_per_probe: int,
        update_fraction: float,
        cache_cells: int,
    ) -> Tuple[float, Dict[str, float]]:
        """Estimate DDGI pipeline cost.

        Args:
            probe_count: Total DDGI probes
            rays_per_probe: Rays per probe
            update_fraction: Fraction of probes updated per frame
            cache_cells: Radiance cache cells

        Returns:
            Tuple of (total_ms, breakdown)
        """
        # Probe tracing
        updated_probes = int(probe_count * update_fraction)
        trace_rays = updated_probes * rays_per_probe
        trace_ms = (trace_rays / 1_000_000) * MS_PER_MILLION_RAYS

        # Probe update (SH accumulation, visibility)
        update_ms = updated_probes * 0.0001

        # Cache update from probes
        cache_ms = (cache_cells / 1_000_000) * 0.1

        # Sampling (during shading, amortized)
        sample_ms = 0.1  # Fixed overhead

        breakdown = {
            "probe_tracing": trace_ms,
            "probe_update": update_ms,
            "radiance_cache": cache_ms,
            "sampling": sample_ms,
        }

        total = sum(breakdown.values())
        return total, breakdown

    def compare_at_budget(
        self,
        budget_ms: float,
        scene_mesh_count: int,
        scene_bounds: AABB,
        screen_width: int = 1920,
        screen_height: int = 1080,
    ) -> CostBudgetResult:
        """Compare approaches at a fixed GPU budget.

        Args:
            budget_ms: Target GPU budget in milliseconds
            scene_mesh_count: Number of scene meshes
            scene_bounds: Scene bounds
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            Cost budget comparison results
        """
        # Estimate Lumen costs
        traces = 500_000  # Moderate trace count
        lumen_total, lumen_breakdown = self.estimate_lumen_cost(
            scene_mesh_count, screen_width, screen_height, traces
        )

        # Calculate DDGI probe count that fits similar budget
        # Iteratively find probe count
        target_probes = 4096
        rays_per_probe = 256
        update_fraction = 0.1
        extent = scene_bounds.max - scene_bounds.min
        cache_cells = int(extent.x * extent.y * extent.z / 8)

        ddgi_total, ddgi_breakdown = self.estimate_ddgi_cost(
            target_probes, rays_per_probe, update_fraction, cache_cells
        )

        # Scale DDGI to match budget
        if ddgi_total > budget_ms and ddgi_total > 0:
            scale = budget_ms / ddgi_total
            target_probes = int(target_probes * scale)
            ddgi_total, ddgi_breakdown = self.estimate_ddgi_cost(
                max(64, target_probes), rays_per_probe, update_fraction, cache_cells
            )

        # Determine winner
        if lumen_total <= ddgi_total * 0.9:
            winner = "Lumen (lower cost)"
        elif ddgi_total <= lumen_total * 0.9:
            winner = "DDGI (lower cost)"
        else:
            winner = "Comparable cost"

        # Cost efficiency: quality per ms
        efficiency = 1.0 if lumen_total == 0 else ddgi_total / lumen_total

        return CostBudgetResult(
            lumen_total_ms=lumen_total,
            lumen_breakdown=lumen_breakdown,
            ddgi_total_ms=ddgi_total,
            ddgi_breakdown=ddgi_breakdown,
            equivalent_quality_winner=winner,
            cost_efficiency=efficiency,
        )


# ============================================================================
# Feasibility Assessment
# ============================================================================


@dataclass
class ComponentAssessment:
    """Assessment for a single component.

    Attributes:
        component: Component type
        feasibility: Feasibility level
        cost_estimate: Cost estimate
        memory_estimate: Memory estimate
        issues: List of identified issues
        recommendations: Implementation recommendations
    """

    component: ComponentType
    feasibility: FeasibilityLevel
    cost_estimate: float
    memory_estimate: int
    issues: List[str]
    recommendations: List[str]


@dataclass
class FeasibilityReport:
    """Complete feasibility study report.

    Attributes:
        decision: Go/no-go decision
        rationale: Decision rationale
        component_assessments: Per-component assessments
        quality_comparison: Quality comparison results
        cost_comparison: Cost comparison results
        implementation_plan: Implementation plan if GO
        risks: Identified risks
        timeline_weeks: Estimated timeline
    """

    decision: DecisionOutcome
    rationale: str
    component_assessments: List[ComponentAssessment]
    quality_comparison: Dict[str, float]
    cost_comparison: CostBudgetResult
    implementation_plan: List[str]
    risks: List[str]
    timeline_weeks: int


class FeasibilityAssessment:
    """Aggregates all evaluations into a final feasibility assessment.

    Produces the go/no-go decision document with:
        - Component-by-component analysis
        - Overall quality comparison
        - Implementation roadmap (if GO)
        - Risk assessment
    """

    def __init__(
        self,
        mesh_card_evaluator: Optional[MeshCardEvaluator] = None,
        screen_probe_evaluator: Optional[ScreenProbeEvaluator] = None,
        sdf_evaluator: Optional[SDFEvaluator] = None,
        cache_comparison: Optional[RadianceCacheComparison] = None,
        quality_metrics: Optional[QualityMetrics] = None,
        cost_analyzer: Optional[CostBudgetAnalyzer] = None,
    ) -> None:
        """Initialize the assessment.

        Args:
            mesh_card_evaluator: Mesh card evaluator
            screen_probe_evaluator: Screen probe evaluator
            sdf_evaluator: SDF evaluator
            cache_comparison: Cache comparison
            quality_metrics: Quality metrics
            cost_analyzer: Cost analyzer
        """
        self.mesh_card_eval = mesh_card_evaluator or MeshCardEvaluator()
        self.screen_probe_eval = screen_probe_evaluator or ScreenProbeEvaluator()
        self.sdf_eval = sdf_evaluator or SDFEvaluator()
        self.cache_comp = cache_comparison or RadianceCacheComparison()
        self.quality = quality_metrics or QualityMetrics()
        self.cost_analyzer = cost_analyzer or CostBudgetAnalyzer()

    def assess(
        self,
        scene_mesh_count: int,
        scene_triangle_count: int,
        scene_bounds: AABB,
        ddgi_probe_count: int,
        trinity_cache_dims: Tuple[int, int, int],
        budget_ms: float = 2.0,
    ) -> FeasibilityReport:
        """Perform complete feasibility assessment.

        Args:
            scene_mesh_count: Number of meshes
            scene_triangle_count: Total triangles
            scene_bounds: Scene bounds
            ddgi_probe_count: Existing DDGI probe count
            trinity_cache_dims: TRINITY cache dimensions
            budget_ms: Target GPU budget

        Returns:
            Complete feasibility report
        """
        assessments = []
        issues_count = 0
        blocked = False

        # Mesh Card Assessment
        mesh_infos = [(1000, 500, 10.0, True)] * scene_mesh_count  # Simplified
        gen_ms, mesh_mem, mesh_feas = self.mesh_card_eval.estimate_scene_cost(
            mesh_infos
        )
        mesh_issues = []
        mesh_recs = []

        if gen_ms > 1000:
            mesh_issues.append(f"High generation time: {gen_ms:.0f}ms")
        if mesh_mem > 100 * 1024 * 1024:
            mesh_issues.append(f"High memory: {mesh_mem / (1024 * 1024):.0f}MB")

        if mesh_feas == FeasibilityLevel.BLOCKED:
            blocked = True
            mesh_recs.append("Consider reducing mesh count or card density")
        else:
            mesh_recs.append("Use LOD-based card generation")
            mesh_recs.append("Cache cards for static meshes")

        assessments.append(ComponentAssessment(
            component=ComponentType.MESH_CARDS,
            feasibility=mesh_feas,
            cost_estimate=gen_ms,
            memory_estimate=mesh_mem,
            issues=mesh_issues,
            recommendations=mesh_recs,
        ))
        issues_count += len(mesh_issues)

        # Screen Probe Assessment
        probe_analysis = self.screen_probe_eval.full_analysis(
            ddgi_probe_count, scene_bounds
        )
        screen_issues = []
        screen_recs = []

        if probe_analysis.temporal_stability_score < 0.7:
            screen_issues.append(
                f"Low stability: {probe_analysis.temporal_stability_score:.2f}"
            )
        if probe_analysis.update_cost_ms > budget_ms * 0.5:
            screen_issues.append(
                f"High cost: {probe_analysis.update_cost_ms:.2f}ms"
            )

        screen_feas = FeasibilityLevel.MEDIUM
        if probe_analysis.temporal_stability_score < 0.5:
            screen_feas = FeasibilityLevel.LOW
        elif probe_analysis.temporal_stability_score > 0.8:
            screen_feas = FeasibilityLevel.HIGH

        screen_recs.append("Use temporal reprojection")
        screen_recs.append("Implement importance-based probe placement")

        assessments.append(ComponentAssessment(
            component=ComponentType.SCREEN_PROBES,
            feasibility=screen_feas,
            cost_estimate=probe_analysis.update_cost_ms,
            memory_estimate=probe_analysis.memory_bytes,
            issues=screen_issues,
            recommendations=screen_recs,
        ))
        issues_count += len(screen_issues)

        # SDF Assessment
        sdf_result = self.sdf_eval.full_evaluation(
            scene_mesh_count, scene_triangle_count, scene_bounds
        )
        sdf_issues = []
        sdf_recs = []

        total_sdf_mem = (
            sdf_result.mesh_sdf_memory_bytes +
            sdf_result.global_sdf_memory_bytes
        )
        if total_sdf_mem > 256 * 1024 * 1024:
            sdf_issues.append(
                f"High SDF memory: {total_sdf_mem / (1024 * 1024):.0f}MB"
            )
        if sdf_result.trace_cost_ms > budget_ms * 0.3:
            sdf_issues.append(
                f"High trace cost: {sdf_result.trace_cost_ms:.2f}ms"
            )
        if sdf_result.quality_score < 0.7:
            sdf_issues.append(
                f"Quality concerns: {sdf_result.quality_score:.2f}"
            )

        if sdf_result.feasibility == FeasibilityLevel.BLOCKED:
            blocked = True

        sdf_recs.append("Use sparse SDF representation")
        sdf_recs.append("Implement SDF streaming for large scenes")
        sdf_recs.append("Fall back to DDGI for distant geometry")

        assessments.append(ComponentAssessment(
            component=ComponentType.SOFTWARE_SDF,
            feasibility=sdf_result.feasibility,
            cost_estimate=sdf_result.trace_cost_ms,
            memory_estimate=total_sdf_mem,
            issues=sdf_issues,
            recommendations=sdf_recs,
        ))
        issues_count += len(sdf_issues)

        # Radiance Cache Assessment
        cache_result = self.cache_comp.compare_approaches(
            trinity_cache_dims, scene_bounds
        )
        cache_issues = []
        cache_recs = []

        if cache_result.lumen_quality_score < cache_result.trinity_quality_score:
            cache_issues.append(
                f"Lower quality than TRINITY: {cache_result.lumen_quality_score:.2f} "
                f"vs {cache_result.trinity_quality_score:.2f}"
            )

        cache_feas = FeasibilityLevel.HIGH
        if "TRINITY" in cache_result.winner_at_equal_cost:
            cache_feas = FeasibilityLevel.MEDIUM
            cache_issues.append("TRINITY cache currently more efficient")

        cache_recs.append("Consider hybrid approach")
        cache_recs.append("Use TRINITY cache for world-space, Lumen for screen-space")

        assessments.append(ComponentAssessment(
            component=ComponentType.RADIANCE_CACHE,
            feasibility=cache_feas,
            cost_estimate=cache_result.lumen_update_cost_ms,
            memory_estimate=cache_result.lumen_memory_bytes,
            issues=cache_issues,
            recommendations=cache_recs,
        ))
        issues_count += len(cache_issues)

        # Cost comparison
        cost_result = self.cost_analyzer.compare_at_budget(
            budget_ms, scene_mesh_count, scene_bounds
        )

        # Quality comparison (estimated)
        quality_comparison = {
            "lumen_estimated_psnr": 32.0,  # Estimated
            "ddgi_estimated_psnr": 34.0,   # DDGI typically better for diffuse
            "lumen_stability": probe_analysis.temporal_stability_score,
            "ddgi_stability": 0.95,
        }

        # Make decision
        if blocked:
            decision = DecisionOutcome.NO_GO
            rationale = (
                "One or more components have fundamental blockers. "
                "SDF memory or mesh card generation exceeds practical limits."
            )
        elif issues_count > 6:
            decision = DecisionOutcome.NO_GO
            rationale = (
                f"Too many issues identified ({issues_count}). "
                "Implementation risk is too high."
            )
        elif issues_count > 3:
            decision = DecisionOutcome.CONDITIONAL_GO
            rationale = (
                f"Moderate issues ({issues_count}). Proceed with caution "
                "and implement incrementally."
            )
        else:
            decision = DecisionOutcome.GO
            rationale = (
                f"Few issues ({issues_count}). Implementation is feasible "
                "within acceptable risk."
            )

        # Implementation plan
        implementation_plan = []
        if decision != DecisionOutcome.NO_GO:
            implementation_plan = [
                "Phase 1: Implement mesh card generation pipeline (2 weeks)",
                "Phase 2: Add screen-space probe system (2 weeks)",
                "Phase 3: Integrate software SDF tracing (3 weeks)",
                "Phase 4: Hybrid radiance cache (2 weeks)",
                "Phase 5: Performance optimization (1 week)",
                "Phase 6: Quality tuning and fallback paths (1 week)",
            ]

        # Risks
        risks = [
            "Temporal stability may require significant tuning",
            "SDF memory could grow unbounded with complex scenes",
            "Performance may vary significantly with scene complexity",
            "Integration with existing DDGI may be complex",
        ]

        timeline = 11 if decision != DecisionOutcome.NO_GO else 0

        return FeasibilityReport(
            decision=decision,
            rationale=rationale,
            component_assessments=assessments,
            quality_comparison=quality_comparison,
            cost_comparison=cost_result,
            implementation_plan=implementation_plan,
            risks=risks,
            timeline_weeks=timeline,
        )

    def generate_report(self, report: FeasibilityReport) -> str:
        """Generate human-readable report text.

        Args:
            report: Feasibility report

        Returns:
            Formatted report text
        """
        lines = [
            "# Lumen-Lite Feasibility Study",
            "",
            "## Executive Summary",
            "",
            f"**Decision: {report.decision.name}**",
            "",
            f"Rationale: {report.rationale}",
            "",
            "## Component Analysis",
            "",
        ]

        for assessment in report.component_assessments:
            lines.append(f"### {assessment.component.name.replace('_', ' ').title()}")
            lines.append(f"- Feasibility: **{assessment.feasibility.name}**")
            lines.append(f"- Cost: {assessment.cost_estimate:.2f}ms")
            lines.append(
                f"- Memory: {assessment.memory_estimate / 1024:.1f}KB"
            )

            if assessment.issues:
                lines.append("- Issues:")
                for issue in assessment.issues:
                    lines.append(f"  - {issue}")

            if assessment.recommendations:
                lines.append("- Recommendations:")
                for rec in assessment.recommendations:
                    lines.append(f"  - {rec}")

            lines.append("")

        lines.extend([
            "## Quality Comparison",
            "",
            f"- DDGI estimated PSNR: {report.quality_comparison.get('ddgi_estimated_psnr', 0):.1f} dB",
            f"- Lumen estimated PSNR: {report.quality_comparison.get('lumen_estimated_psnr', 0):.1f} dB",
            f"- DDGI stability: {report.quality_comparison.get('ddgi_stability', 0):.2f}",
            f"- Lumen stability: {report.quality_comparison.get('lumen_stability', 0):.2f}",
            "",
            "## Cost Comparison",
            "",
            f"- DDGI total: {report.cost_comparison.ddgi_total_ms:.2f}ms",
            f"- Lumen total: {report.cost_comparison.lumen_total_ms:.2f}ms",
            f"- Winner at equal cost: {report.cost_comparison.equivalent_quality_winner}",
            "",
        ])

        if report.decision != DecisionOutcome.NO_GO:
            lines.extend([
                "## Implementation Plan",
                "",
            ])
            for i, phase in enumerate(report.implementation_plan, 1):
                lines.append(f"{i}. {phase}")

            lines.extend([
                "",
                f"**Estimated Timeline: {report.timeline_weeks} weeks**",
                "",
            ])

        lines.extend([
            "## Risks",
            "",
        ])
        for risk in report.risks:
            lines.append(f"- {risk}")

        lines.extend([
            "",
            "## Recommendation",
            "",
        ])

        if report.decision == DecisionOutcome.GO:
            lines.append(
                "Proceed with Lumen-Lite implementation following the phased approach. "
                "Maintain DDGI as fallback during development."
            )
        elif report.decision == DecisionOutcome.CONDITIONAL_GO:
            lines.append(
                "Proceed cautiously with Phase 1 only. Evaluate results before "
                "committing to full implementation. Consider hybrid approach."
            )
        else:
            lines.append(
                "Do not proceed with Lumen-Lite implementation. Continue with "
                "DDGI optimization and enhancement instead."
            )

        return "\n".join(lines)

    def create_plan(self, report: FeasibilityReport) -> List[Dict[str, str]]:
        """Create structured implementation plan.

        Args:
            report: Feasibility report

        Returns:
            List of phase dictionaries
        """
        if report.decision == DecisionOutcome.NO_GO:
            return []

        return [
            {
                "phase": "1",
                "name": "Mesh Card Pipeline",
                "duration": "2 weeks",
                "tasks": [
                    "Implement card generation from mesh data",
                    "Add UV chart computation",
                    "Create card LOD system",
                    "Build radiance injection pipeline",
                ],
                "deliverables": ["MeshCardGenerator", "CardRadianceInjector"],
                "risks": ["UV generation quality", "Memory overhead"],
            },
            {
                "phase": "2",
                "name": "Screen Probes",
                "duration": "2 weeks",
                "tasks": [
                    "Implement screen-space probe placement",
                    "Add temporal reprojection",
                    "Build probe tracing dispatch",
                    "Integrate with frame graph",
                ],
                "deliverables": ["ScreenProbeSystem", "ProbeTracer"],
                "risks": ["Temporal stability", "Parallax artifacts"],
            },
            {
                "phase": "3",
                "name": "SDF Tracing",
                "duration": "3 weeks",
                "tasks": [
                    "Implement mesh SDF generation",
                    "Build global distance field",
                    "Create software SDF tracer",
                    "Add sparse SDF representation",
                ],
                "deliverables": ["MeshSDFGenerator", "GlobalDistanceField", "SDFTracer"],
                "risks": ["Memory usage", "Trace performance"],
            },
            {
                "phase": "4",
                "name": "Hybrid Cache",
                "duration": "2 weeks",
                "tasks": [
                    "Integrate screen-space cache with TRINITY cache",
                    "Add fallback path selection",
                    "Implement quality metrics",
                    "Build debug visualization",
                ],
                "deliverables": ["HybridRadianceCache", "CacheDebugView"],
                "risks": ["Integration complexity", "Quality parity"],
            },
            {
                "phase": "5",
                "name": "Optimization",
                "duration": "1 week",
                "tasks": [
                    "Profile and optimize hot paths",
                    "Tune temporal parameters",
                    "Add quality presets",
                    "Memory optimization",
                ],
                "deliverables": ["OptimizedPipeline", "QualityPresets"],
                "risks": ["Platform variance"],
            },
            {
                "phase": "6",
                "name": "Integration",
                "duration": "1 week",
                "tasks": [
                    "Final integration with renderer",
                    "Fallback path testing",
                    "Documentation",
                    "Performance validation",
                ],
                "deliverables": ["LumenLiteGI", "Documentation"],
                "risks": ["Regression in existing features"],
            },
        ]


# ============================================================================
# Convenience Functions
# ============================================================================


def run_feasibility_study(
    scene_mesh_count: int = 1000,
    scene_triangle_count: int = 1_000_000,
    scene_bounds: Optional[AABB] = None,
    ddgi_probe_count: int = 4096,
    trinity_cache_dims: Tuple[int, int, int] = (64, 64, 32),
    budget_ms: float = 2.0,
) -> FeasibilityReport:
    """Run complete feasibility study with default parameters.

    Args:
        scene_mesh_count: Number of scene meshes
        scene_triangle_count: Total scene triangles
        scene_bounds: Scene bounds (defaults to 100m cube)
        ddgi_probe_count: Existing DDGI probe count
        trinity_cache_dims: TRINITY cache dimensions
        budget_ms: Target GPU budget

    Returns:
        Complete feasibility report
    """
    if scene_bounds is None:
        scene_bounds = AABB(Vec3(-50, -10, -50), Vec3(50, 40, 50))

    assessment = FeasibilityAssessment()
    return assessment.assess(
        scene_mesh_count,
        scene_triangle_count,
        scene_bounds,
        ddgi_probe_count,
        trinity_cache_dims,
        budget_ms,
    )


def quick_assessment(
    scene_complexity: str = "medium",
) -> Tuple[DecisionOutcome, str]:
    """Quick assessment based on scene complexity.

    Args:
        scene_complexity: "low", "medium", "high", or "extreme"

    Returns:
        Tuple of (decision, summary)
    """
    complexity_params = {
        "low": (100, 100_000, 1.0),
        "medium": (500, 500_000, 2.0),
        "high": (2000, 2_000_000, 3.0),
        "extreme": (5000, 10_000_000, 5.0),
    }

    mesh_count, tri_count, budget = complexity_params.get(
        scene_complexity, complexity_params["medium"]
    )

    report = run_feasibility_study(
        scene_mesh_count=mesh_count,
        scene_triangle_count=tri_count,
        budget_ms=budget,
    )

    summary = (
        f"{scene_complexity.title()} complexity scene: {report.decision.name}. "
        f"{report.rationale}"
    )

    return report.decision, summary
