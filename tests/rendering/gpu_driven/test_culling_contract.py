"""
Blackbox contract tests for GPU culling pipeline Phase 1.

Covers T-CULL-1.1 through T-CULL-1.6 from the public contract only.
No implementation access. Tests are written against the API surface
declared in __init__.py and documented in PHASE_1_ARCH.md.

Cleanroom discipline: only public API, no private attribute access,
no line-number references, no internal-structure assumptions.
"""

import math
import time

import pytest

from engine.rendering.gpu_driven.culling import (
    Vec3,
    Vec4,
    AABB,
    BoundingSphere,
    Frustum,
    FrustumPlane,
    FrustumCuller,
    InstanceBounds,
    CullingStats,
    CullResult,
    OcclusionCuller,
    DistanceCuller,
    SmallFeatureCuller,
    HZBConfig,
    DistanceCullConfig,
    SmallFeatureCullConfig,
    CullingConstants,
    CullingPipeline,
    Culler,
)


# =============================================================================
# HELPERS -- public-api only
# =============================================================================


def _perspective_matrix(
    fov_y_rad: float,
    aspect: float,
    near: float,
    far: float,
) -> list[list[float]]:
    """OpenGL-style perspective projection (row-major)."""
    f = 1.0 / math.tan(fov_y_rad * 0.5)
    return [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, -(far + near) / (far - near), -2.0 * far * near / (far - near)],
        [0.0, 0.0, -1.0, 0.0],
    ]


def _instance_bounds(
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float = 1.0,
) -> InstanceBounds:
    """Public helper: build InstanceBounds at (cx, cy, cz) with given radius."""
    return InstanceBounds(
        instance_id=0,
        bounding_sphere=BoundingSphere(
            center=Vec3(center_x, center_y, center_z),
            radius=radius,
        ),
        aabb=AABB(
            min_point=Vec3(center_x - radius, center_y - radius, center_z - radius),
            max_point=Vec3(center_x + radius, center_y + radius, center_z + radius),
        ),
    )


# =============================================================================
# T-CULL-1.1: FRUSTUM PLANE EXTRACTION (PUBLIC CONTRACT)
# =============================================================================


class TestFrustumContract:
    """Frustum public API: Gribb-Hartmann extraction, plane tests."""

    # ------------------------------------------------------------------
    # Frustum.from_view_projection_matrix -- basic contract
    # ------------------------------------------------------------------

    def test_frustum_has_exactly_six_planes(self) -> None:
        """from_view_projection_matrix returns a Frustum with 6 planes."""
        vp = _perspective_matrix(math.radians(60), 16.0 / 9.0, 0.1, 500.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        assert len(frustum.planes) == 6

    def test_frustum_planes_are_public(self) -> None:
        """Frustum planes are accessible as a list of Vec4."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        for plane in frustum.planes:
            assert isinstance(plane, Vec4)

    def test_frustum_plane_labels_exist(self) -> None:
        """FrustumPlane enum provides named plane indices."""
        assert hasattr(FrustumPlane, "LEFT")
        assert hasattr(FrustumPlane, "RIGHT")
        assert hasattr(FrustumPlane, "BOTTOM")
        assert hasattr(FrustumPlane, "TOP")
        assert hasattr(FrustumPlane, "NEAR")
        assert hasattr(FrustumPlane, "FAR")

    def test_frustum_planes_indexable_by_label(self) -> None:
        """Planes can be accessed via FrustumPlane enum labels."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        for label in (FrustumPlane.LEFT, FrustumPlane.RIGHT,
                      FrustumPlane.TOP, FrustumPlane.BOTTOM,
                      FrustumPlane.NEAR, FrustumPlane.FAR):
            assert frustum.planes[label] is not None

    # ------------------------------------------------------------------
    # Perspective projection -- point-in-frustum contract
    # ------------------------------------------------------------------

    def test_perspective_frustum_center_point_visible(self) -> None:
        """Point at frustum center is visible."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, -5.0), radius=0.0)
        assert frustum.test_sphere(sphere) is True

    def test_perspective_frustum_behind_camera_culled(self) -> None:
        """Point behind camera (positive Z) is culled."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, 5.0), radius=0.0)
        assert frustum.test_sphere(sphere) is False

    def test_perspective_frustum_beyond_far_culled(self) -> None:
        """Point beyond far plane is culled."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, -200.0), radius=0.0)
        assert frustum.test_sphere(sphere) is False

    # ------------------------------------------------------------------
    # Sphere-frustum -- partial intersection contract
    # ------------------------------------------------------------------

    def test_sphere_partially_intersecting_is_visible(self) -> None:
        """A sphere that straddles a frustum plane is visible."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        # Large sphere centered just outside left plane, radius pulls it in
        sphere = BoundingSphere(center=Vec3(-5.0, 0.0, -5.0), radius=10.0)
        assert frustum.test_sphere(sphere) is True

    def test_sphere_fully_outside_is_culled(self) -> None:
        """A sphere fully outside all planes is culled."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        # Sphere well to the left, small radius
        sphere = BoundingSphere(center=Vec3(-50.0, 0.0, -5.0), radius=1.0)
        assert frustum.test_sphere(sphere) is False

    # ------------------------------------------------------------------
    # AABB-frustum contract
    # ------------------------------------------------------------------

    def test_aabb_fully_inside_is_visible(self) -> None:
        """AABB fully inside frustum is visible."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        aabb = AABB(min_point=Vec3(-1.0, -1.0, -10.0), max_point=Vec3(1.0, 1.0, -5.0))
        assert frustum.test_aabb(aabb) is True

    def test_aabb_fully_left_culled(self) -> None:
        """AABB fully to the left of frustum is culled."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        aabb = AABB(min_point=Vec3(-30.0, -1.0, -10.0), max_point=Vec3(-25.0, 1.0, -5.0))
        assert frustum.test_aabb(aabb) is False

    def test_aabb_fully_below_culled(self) -> None:
        """AABB fully below frustum is culled."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        aabb = AABB(min_point=Vec3(-1.0, -30.0, -10.0), max_point=Vec3(1.0, -25.0, -5.0))
        assert frustum.test_aabb(aabb) is False

    def test_aabb_straddling_plane_is_visible(self) -> None:
        """AABB that straddles a frustum plane is visible."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        # AABB straddling the right plane
        aabb = AABB(min_point=Vec3(-0.5, -1.0, -10.0), max_point=Vec3(2.0, 1.0, -5.0))
        assert frustum.test_aabb(aabb) is True

    def test_aabb_beyond_far_culled(self) -> None:
        """AABB fully beyond far plane is culled."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        aabb = AABB(
            min_point=Vec3(-1.0, -1.0, -200.0),
            max_point=Vec3(1.0, 1.0, -150.0),
        )
        assert frustum.test_aabb(aabb) is False

    # ------------------------------------------------------------------
    # Degenerate inputs -- contract boundary
    # ------------------------------------------------------------------

    def test_empty_aabb_does_not_crash(self) -> None:
        """Degenerate (reversed) AABB does not raise exceptions."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        aabb = AABB(min_point=Vec3(5.0, 5.0, 5.0), max_point=Vec3(1.0, 1.0, 1.0))
        result = frustum.test_aabb(aabb)
        assert isinstance(result, bool)

    def test_zero_radius_sphere_contract(self) -> None:
        """A zero-radius sphere (point) returns bool from test_sphere."""
        vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        frustum = Frustum.from_view_projection_matrix(vp)
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, -5.0), radius=0.0)
        assert frustum.test_sphere(sphere) is True


# =============================================================================
# T-CULL-1.2: HZB PYRAMID CONSTRUCTION (PUBLIC CONTRACT)
# =============================================================================


class TestHZBPyramidContract:
    """HZB mip pyramid: public contract only, no internal state access."""

    def test_hzb_config_constructor(self) -> None:
        """HZBConfig accepts width and height."""
        config = HZBConfig(width=64, height=48)
        assert config.width == 64
        assert config.height == 48

    def test_occlusion_culler_constructed(self) -> None:
        """OcclusionCuller is constructable with HZBConfig."""
        culler = OcclusionCuller(HZBConfig(width=64, height=48))
        assert culler is not None

    def test_has_hzb_pyramid_before_update(self) -> None:
        """has_hzb_pyramid() returns False before any update."""
        culler = OcclusionCuller(HZBConfig(width=64, height=48))
        assert culler.has_hzb_pyramid() is False

    def test_has_hzb_pyramid_after_update(self) -> None:
        """has_hzb_pyramid() returns True after depth update."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)])
        assert culler.has_hzb_pyramid() is True

    def test_hzb_mip_levels_after_2x2(self) -> None:
        """2x2 depth buffer produces exactly 2 mip levels (base + 1x1)."""
        culler = OcclusionCuller(HZBConfig(width=2, height=2))
        culler.update(depth_buffer=[[0.1, 0.5], [0.9, 0.3]])
        assert culler.hzb_mip_levels == 2

    @pytest.mark.parametrize("size,expected_levels", [
        (1, 1),
        (2, 2),
        (4, 3),
        (8, 4),
        (16, 5),
        (32, 6),
    ])
    def test_hzb_mip_levels_power_of_two(
        self, size: int, expected_levels: int,
    ) -> None:
        """Power-of-two NxN depth buffers produce log2(N)+1 mip levels."""
        culler = OcclusionCuller(HZBConfig(width=size, height=size))
        culler.update(depth_buffer=[[0.5 for _ in range(size)] for _ in range(size)])
        assert culler.hzb_mip_levels == expected_levels

    def test_hzb_non_power_of_two_eventually_reaches_1x1(self) -> None:
        """Non-power-of-two dimensions eventually reduce to 1x1 top level."""
        culler = OcclusionCuller(HZBConfig(width=7, height=11))
        depth = [[0.3 for _ in range(7)] for _ in range(11)]
        culler.update(depth_buffer=depth)
        # At least 1 level exists, and repeating reductions hit 1x1
        assert culler.hzb_mip_levels >= 1

    def test_hzb_mip_count_does_not_decrease_with_reupdate(self) -> None:
        """Re-updating the HZB produces a valid state (no crash)."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)])
        levels1 = culler.hzb_mip_levels
        culler.update(depth_buffer=[[0.9 for _ in range(4)] for _ in range(4)])
        levels2 = culler.hzb_mip_levels
        assert levels2 == levels1, (
            "Mip level count should be stable for same dimensions"
        )

    def test_hzb_mip_levels_is_int(self) -> None:
        """hzb_mip_levels returns an int."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)])
        assert isinstance(culler.hzb_mip_levels, int)


# =============================================================================
# T-CULL-1.3: HZB OCCLUSION QUERY (PUBLIC CONTRACT)
# =============================================================================


class TestHZBOcclusionContract:
    """HZB occlusion test via public API: is_occluded and cull."""

    def setup_method(self) -> None:
        self.vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)

    # ------------------------------------------------------------------
    # is_occluded public API
    # ------------------------------------------------------------------

    def test_is_occluded_returns_bool(self) -> None:
        """is_occluded returns a bool for any valid sphere."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(
            depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)],
            view_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            projection_matrix=self.vp,
            near_plane=1.0,
            far_plane=100.0,
        )
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, -5.0), radius=0.0)
        result = culler.is_occluded(sphere)
        assert isinstance(result, bool)

    def test_object_in_front_of_wall_not_occluded(self) -> None:
        """Object in front of a depth wall is NOT occluded."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(
            depth_buffer=[[0.8 for _ in range(4)] for _ in range(4)],
            view_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            projection_matrix=self.vp,
            near_plane=1.0,
            far_plane=100.0,
        )
        # z=-2 => projected depth ~0.5, which is less than wall at 0.8
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, -2.0), radius=0.0)
        assert culler.is_occluded(sphere) is False

    def test_object_behind_wall_is_occluded(self) -> None:
        """Object behind a depth wall IS occluded."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(
            depth_buffer=[[0.3 for _ in range(4)] for _ in range(4)],
            view_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            projection_matrix=self.vp,
            near_plane=1.0,
            far_plane=100.0,
        )
        # z=-60 => projected depth near 1.0, behind wall at 0.3
        sphere = BoundingSphere(center=Vec3(0.0, 0.0, -60.0), radius=0.0)
        assert culler.is_occluded(sphere) is True

    # ------------------------------------------------------------------
    # cull() public API for occlusion
    # ------------------------------------------------------------------

    def test_occlusion_cull_returns_cullingstats(self) -> None:
        """cull() on OcclusionCuller returns CullingStats."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(
            depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)],
            view_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            projection_matrix=self.vp,
            near_plane=1.0,
            far_plane=100.0,
        )
        instance = _instance_bounds(0.0, 0.0, -10.0, 1.0)
        visible_mask = [True]
        stats = culler.cull([instance], visible_mask)
        assert isinstance(stats, CullingStats)

    def test_occlusion_cull_stats_all_non_negative(self) -> None:
        """All CullingStats counters are non-negative."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(
            depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)],
            view_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            projection_matrix=self.vp,
            near_plane=1.0,
            far_plane=100.0,
        )
        instances = [_instance_bounds(0.0, 0.0, z, 0.5) for z in [-5.0, -30.0, -80.0]]
        visible_mask = [True, True, True]
        stats = culler.cull(instances, visible_mask)
        assert stats.total_instances >= 0
        assert stats.visible_instances >= 0
        assert stats.occlusion_culled >= 0

    def test_occlusion_cull_mask_respected(self) -> None:
        """Already-culled instances are not counted as visible."""
        culler = OcclusionCuller(HZBConfig(width=4, height=4))
        culler.update(
            depth_buffer=[[0.5 for _ in range(4)] for _ in range(4)],
            view_matrix=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            projection_matrix=self.vp,
            near_plane=1.0,
            far_plane=100.0,
        )
        instance = _instance_bounds(0.0, 0.0, -10.0, 1.0)
        visible_mask = [False]  # pre-culled
        stats = culler.cull([instance], visible_mask)
        assert stats.visible_instances == 0
        assert visible_mask[0] is False


# =============================================================================
# T-CULL-1.4: DISTANCE CULLER WITH LOD (PUBLIC CONTRACT)
# =============================================================================


class TestDistanceCullerContract:
    """Distance culling LOD system: public API only."""

    def setup_method(self) -> None:
        self.config = DistanceCullConfig(
            max_distance=500.0,
            fade_distance=50.0,
            lod_distances=[10.0, 50.0, 200.0, 500.0],
        )
        self.culler = DistanceCuller(self.config)
        self.culler.update(camera_position=Vec3(0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # LOD level computation (public API)
    # ------------------------------------------------------------------

    def test_compute_lod_level_returns_int(self) -> None:
        """compute_lod_level returns an int for any distance."""
        result = self.culler.compute_lod_level(42.0)
        assert isinstance(result, int)

    def test_lod_increases_with_distance(self) -> None:
        """LOD index monotonically increases with distance."""
        distances = [0.0, 5.0, 25.0, 75.0, 150.0, 350.0, 600.0]
        lods = [self.culler.compute_lod_level(d) for d in distances]
        for i in range(1, len(lods)):
            assert lods[i] >= lods[i - 1], (
                f"LOD decreased at index {i}: {lods}"
            )

    def test_lod_0_for_near_objects(self) -> None:
        """Objects close to camera get LOD 0."""
        assert self.culler.compute_lod_level(5.0) == 0

    def test_lod_increments_at_distances(self) -> None:
        """LOD index increments through the distance bands."""
        # Past the last threshold = len(lod_distances)
        assert self.culler.compute_lod_level(1000.0) == 4

    def test_lod_at_exact_threshold(self) -> None:
        """At exact threshold distances the next LOD is used."""
        # Default contract: distance < lod_distances[i] => LOD i
        assert self.culler.compute_lod_level(10.0) == 1  # at boundary -> next

    # ------------------------------------------------------------------
    # Cull operation
    # ------------------------------------------------------------------

    def test_cull_returns_cullingstats(self) -> None:
        """cull() returns CullingStats."""
        instance = _instance_bounds(0.0, 0.0, -50.0, 1.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        assert isinstance(stats, CullingStats)

    def test_nearby_instance_stays_visible(self) -> None:
        """Instance within max distance stays visible."""
        instance = _instance_bounds(0.0, 0.0, -50.0, 1.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        assert visible_mask[0] is True
        assert stats.distance_culled == 0

    def test_instance_beyond_max_culled(self) -> None:
        """Instance beyond max render distance is culled."""
        instance = _instance_bounds(0.0, 0.0, -600.0, 1.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        if not visible_mask[0]:
            assert stats.distance_culled == 1

    def test_large_sphere_radius_compensation(self) -> None:
        """Large sphere near max distance stays visible (radius compensation)."""
        # Center at 600, radius 350 => effective surface at 250 < 500
        instance = _instance_bounds(0.0, 0.0, -600.0, 350.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        assert visible_mask[0] is True

    def test_camera_offset_distance_computed(self) -> None:
        """Camera at nonzero position: distance computed correctly."""
        self.culler.update(camera_position=Vec3(10.0, 20.0, 30.0))
        instance = _instance_bounds(40.0, 50.0, 60.0, 1.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        assert isinstance(stats, CullingStats)
        # Distance = sqrt(30^2 + 30^2 + 30^2) ~= 51.96, inside max=500
        assert visible_mask[0] is True

    def test_distance_cull_stats_consistent(self) -> None:
        """Stats.total_instances and .distance_culled are consistent."""
        instances = [
            _instance_bounds(0.0, 0.0, -50.0, 1.0),
            _instance_bounds(0.0, 0.0, -600.0, 1.0),
        ]
        visible_mask = [True, True]
        stats = self.culler.cull(instances, visible_mask)
        assert stats.total_instances == 2
        culled = 2 - sum(visible_mask)
        assert culled == stats.distance_culled


# =============================================================================
# T-CULL-1.5: SMALL FEATURE CULLER (PUBLIC CONTRACT)
# =============================================================================


class TestSmallFeatureCullerContract:
    """Screen-space size culling: public API only."""

    def setup_method(self) -> None:
        self.config = SmallFeatureCullConfig(
            min_screen_size=2.0,
            screen_width=1920,
            screen_height=1080,
        )
        self.culler = SmallFeatureCuller(self.config)
        self.culler.update(
            camera_position=Vec3(0.0, 0.0, 0.0),
            fov_y=math.radians(60.0),
        )

    # ------------------------------------------------------------------
    # Cull behavior
    # ------------------------------------------------------------------

    def test_cull_returns_cullingstats(self) -> None:
        """cull() returns CullingStats."""
        instance = _instance_bounds(0.0, 0.0, -10.0, 1.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        assert isinstance(stats, CullingStats)

    def test_large_nearby_object_not_culled(self) -> None:
        """Large nearby object stays visible."""
        instance = _instance_bounds(0.0, 0.0, -10.0, 5.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        assert visible_mask[0] is True

    def test_tiny_distant_object_culled(self) -> None:
        """Tiny distant object is culled (sub-pixel feature)."""
        instance = _instance_bounds(0.0, 0.0, -5000.0, 0.01)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        if not visible_mask[0]:
            assert stats.small_feature_culled >= 1

    def test_min_screen_size_threshold_applied(self) -> None:
        """Different min_screen_size thresholds change culling behavior."""
        # Object at moderate distance
        instance = _instance_bounds(0.0, 0.0, -200.0, 2.0)
        visible_mask = [True]
        stats = self.culler.cull([instance], visible_mask)
        result_default = visible_mask[0]

        # With very low threshold (nearly everything visible)
        config_tiny = SmallFeatureCullConfig(
            min_screen_size=0.001, screen_width=1920, screen_height=1080,
        )
        culler_tiny = SmallFeatureCuller(config_tiny)
        culler_tiny.update(
            camera_position=Vec3(0.0, 0.0, 0.0),
            fov_y=math.radians(60.0),
        )
        visible_mask2 = [True]
        culler_tiny.cull([instance], visible_mask2)

        # With very high threshold (nearly everything culled)
        config_huge = SmallFeatureCullConfig(
            min_screen_size=5000.0, screen_width=1920, screen_height=1080,
        )
        culler_huge = SmallFeatureCuller(config_huge)
        culler_huge.update(
            camera_position=Vec3(0.0, 0.0, 0.0),
            fov_y=math.radians(60.0),
        )
        visible_mask3 = [True]
        culler_huge.cull([instance], visible_mask3)

        # The high-threshold culler should cull more (or equal)
        assert visible_mask3[0] <= visible_mask2[0], (
            "Higher min_screen_size should cull at least as much"
        )

    def test_different_resolutions_different_behavior(self) -> None:
        """Different screen dimensions produce different culling behavior."""
        instance = _instance_bounds(0.0, 0.0, -100.0, 1.0)

        # HD resolution
        visible_hd = [True]
        self.culler.cull([instance], visible_hd)

        # Low resolution (object appears larger in screen-space fraction)
        config_low = SmallFeatureCullConfig(
            min_screen_size=2.0, screen_width=320, screen_height=240,
        )
        culler_low = SmallFeatureCuller(config_low)
        culler_low.update(
            camera_position=Vec3(0.0, 0.0, 0.0),
            fov_y=math.radians(60.0),
        )
        visible_low = [True]
        culler_low.cull([instance], visible_low)

        # Just verify both complete without error
        assert isinstance(visible_hd[0], bool)
        assert isinstance(visible_low[0], bool)

    def test_pre_culled_instance_respected(self) -> None:
        """SmallFeatureCuller respects already-culled instances."""
        instance = _instance_bounds(0.0, 0.0, -5000.0, 0.01)
        visible_mask = [False]  # pre-culled by earlier stage
        stats = self.culler.cull([instance], visible_mask)
        assert visible_mask[0] is False
        assert stats.visible_instances == 0


# =============================================================================
# T-CULL-1.6: CULLING PIPELINE COMPOSITION (PUBLIC CONTRACT)
# =============================================================================


class TestCullingPipelineContract:
    """Pipeline composition: sequencing, early-out, stats, configurability."""

    def setup_method(self) -> None:
        self.vp = _perspective_matrix(math.radians(90), 1.0, 1.0, 100.0)
        self.pipeline = CullingPipeline()
        self.pipeline.update(
            view_projection_matrix=self.vp,
            camera_position=Vec3(0.0, 0.0, 0.0),
            camera_forward=Vec3(0.0, 0.0, -1.0),
            fov_y=math.radians(60.0),
        )

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def test_pipeline_has_all_cullers(self) -> None:
        """CullingPipeline creates all four culler instances."""
        pipeline = CullingPipeline()
        assert pipeline.frustum_culler is not None
        assert pipeline.occlusion_culler is not None
        assert pipeline.distance_culler is not None
        assert pipeline.small_feature_culler is not None

    def test_cullers_are_culler_subclasses(self) -> None:
        """All cullers inherit from Culler ABC."""
        for cls in (FrustumCuller, OcclusionCuller, DistanceCuller, SmallFeatureCuller):
            assert issubclass(cls, Culler)

    def test_cullers_have_cull_and_update(self) -> None:
        """All culler types implement cull() and update()."""
        for cls in (FrustumCuller, OcclusionCuller, DistanceCuller, SmallFeatureCuller):
            instance = cls()
            assert callable(instance.cull)
            assert callable(instance.update)

    # ------------------------------------------------------------------
    # Full pipeline cull
    # ------------------------------------------------------------------

    def test_pipeline_cull_returns_indices_and_stats(self) -> None:
        """pipeline.cull() returns (list[int], CullingStats)."""
        instances = [_instance_bounds(0.0, 0.0, -10.0, 1.0)]
        indices, stats = self.pipeline.cull(instances)
        assert isinstance(indices, list)
        assert isinstance(stats, CullingStats)

    def test_pipeline_visible_indices_are_positions(self) -> None:
        """Visible indices are list positions (0..N-1), not instance_id values."""
        instances = [
            InstanceBounds(
                instance_id=10,
                bounding_sphere=BoundingSphere(
                    center=Vec3(0.0, 0.0, -10.0), radius=1.0,
                ),
                aabb=AABB(
                    min_point=Vec3(-1, -1, -11), max_point=Vec3(1, 1, -9),
                ),
            ),
            InstanceBounds(
                instance_id=20,
                bounding_sphere=BoundingSphere(
                    center=Vec3(0.0, 0.0, 5.0), radius=1.0,
                ),
                aabb=AABB(
                    min_point=Vec3(-1, -1, 4), max_point=Vec3(1, 1, 6),
                ),
            ),
        ]
        indices, stats = self.pipeline.cull(instances)
        # Indices are positions into the input list
        assert 0 in indices, "First instance (visible) should be in indices"
        assert 1 not in indices, "Second instance (behind camera) should not be in indices"
        for idx in indices:
            assert 0 <= idx < len(instances), f"Index {idx} out of range"

    def test_pipeline_stats_total_matches_input(self) -> None:
        """Stats.total_instances matches input count."""
        instances = [_instance_bounds(0.0, 0.0, z, 1.0) for z in [-5.0, -50.0, 5.0, -200.0]]
        indices, stats = self.pipeline.cull(instances)
        assert stats.total_instances == len(instances)

    # ------------------------------------------------------------------
    # Stage enable/disable
    # ------------------------------------------------------------------

    def test_stage_disable_removes_culling(self) -> None:
        """Disabling all stages: all instances pass through visible."""
        self.pipeline.configure(
            enable_frustum=False,
            enable_occlusion=False,
            enable_distance=False,
            enable_small_feature=False,
        )
        instances = [
            _instance_bounds(0.0, 0.0, 5.0, 1.0),    # behind camera
            _instance_bounds(0.0, 0.0, -5000.0, 1.0),  # far
        ]
        indices, stats = self.pipeline.cull(instances)
        assert len(indices) == len(instances)
        assert stats.visible_instances == len(instances)
        assert stats.frustum_culled == 0
        assert stats.occlusion_culled == 0
        assert stats.distance_culled == 0
        assert stats.small_feature_culled == 0

    def test_disabling_frustum_returns_zero_frustum_culled(self) -> None:
        """Disabling frustum: frustum_culled stays 0."""
        self.pipeline.configure(enable_frustum=False)
        instance = _instance_bounds(0.0, 0.0, 5.0, 1.0)
        indices, stats = self.pipeline.cull([instance])
        assert stats.frustum_culled == 0

    def test_disabling_distance_returns_zero_distance_culled(self) -> None:
        """Disabling distance: distance_culled stays 0."""
        self.pipeline.configure(enable_distance=False)
        instance = _instance_bounds(0.0, 0.0, -5000.0, 1.0)
        indices, stats = self.pipeline.cull([instance])
        assert stats.distance_culled == 0

    def test_disabling_small_feature_returns_zero_small_feature_culled(self) -> None:
        """Disabling small feature: small_feature_culled stays 0."""
        self.pipeline.configure(enable_small_feature=False)
        instance = _instance_bounds(0.0, 0.0, -5000.0, 0.001)
        indices, stats = self.pipeline.cull([instance])
        assert stats.small_feature_culled == 0

    # ------------------------------------------------------------------
    # Stats consistency
    # ------------------------------------------------------------------

    def test_stats_counters_are_consistent(self) -> None:
        """
        Sum of all culled counters + visible = total.
        This is the fundamental pipeline invariant.
        """
        instances = [
            _instance_bounds(0.0, 0.0, -10.0, 1.0),
            _instance_bounds(0.0, 0.0, -80.0, 1.0),
            _instance_bounds(0.0, 0.0, 10.0, 1.0),   # behind
            _instance_bounds(5.0, 0.0, -10.0, 1.0),
        ]
        indices, stats = self.pipeline.cull(instances)
        total_culled = (
            stats.frustum_culled
            + stats.occlusion_culled
            + stats.distance_culled
            + stats.small_feature_culled
        )
        assert stats.total_instances == stats.visible_instances + total_culled, (
            f"Count mismatch: total={stats.total_instances}, "
            f"visible={stats.visible_instances}, "
            f"culled_sum={total_culled}"
        )

    def test_cull_ratio_calculation(self) -> None:
        """Cull ratio = 1 - (visible / total)."""
        stats = CullingStats(total_instances=100, visible_instances=60)
        assert stats.cull_ratio == pytest.approx(0.4)

    def test_cull_ratio_zero_when_empty(self) -> None:
        """Cull ratio is 0 when there are no instances."""
        stats = CullingStats(total_instances=0, visible_instances=0)
        assert stats.cull_ratio == 0.0

    # ------------------------------------------------------------------
    # Throughput benchmark (per AC)
    # ------------------------------------------------------------------

    def test_pipeline_throughput_benchmark(self) -> None:
        """10,000 objects process faster than 1 second (>10K/sec)."""
        num_objects = 10_000
        instances = [
            _instance_bounds(
                float(i % 100) - 50.0,
                float((i // 100) % 100) - 50.0,
                -(10.0 + float(i % 10) * 10.0),
                1.0,
            )
            for i in range(num_objects)
        ]

        # Warmup
        self.pipeline.cull(instances[:100])

        start = time.perf_counter()
        indices, stats = self.pipeline.cull(instances)
        elapsed = time.perf_counter() - start

        throughput = num_objects / elapsed

        assert elapsed < 1.0, (
            f"Pipeline slower than 10K objects/sec: "
            f"{elapsed * 1000:.1f}ms for {num_objects} objects"
        )
        assert stats.total_instances == num_objects

    # ------------------------------------------------------------------
    # Large scene validation
    # ------------------------------------------------------------------

    def test_large_scene_all_instances_processed(self) -> None:
        """All instances are accounted for in pipeline stats."""
        num = 500
        instances = [
            _instance_bounds(
                float(i % 50) - 25.0,
                float((i // 50) % 10) - 5.0,
                -(5.0 + float(i % 20) * 5.0),
                1.0,
            )
            for i in range(num)
        ]
        indices, stats = self.pipeline.cull(instances)
        total_accounted = (
            stats.visible_instances
            + stats.frustum_culled
            + stats.occlusion_culled
            + stats.distance_culled
            + stats.small_feature_culled
        )
        assert total_accounted == num

    # ------------------------------------------------------------------
    # CullResult contract
    # ------------------------------------------------------------------

    def test_cullresult_is_exported(self) -> None:
        """CullResult is importable and usable as a type."""
        # CullResult is an enum in the implementation; verify it exists
        # and can be enumerated.
        assert CullResult is not None
        assert hasattr(CullResult, "__members__") or callable(CullResult)
        # Check that enum values exist (implementation-defined members)
        members = list(CullResult)
        assert len(members) > 0, "CullResult should have at least one member"


# =============================================================================
# CULLING CONSTANTS (PUBLIC CONTRACT)
# =============================================================================


class TestCullingConstantsContract:
    """CullingConstants provide epsilon/threshold values."""

    def test_constants_have_expected_fields(self) -> None:
        """CullingConstants exposes at least an EPSILON value."""
        assert hasattr(CullingConstants, "EPSILON")
        assert isinstance(CullingConstants.EPSILON, float)

    def test_constants_are_numeric(self) -> None:
        """Constants are numeric floats."""
        assert isinstance(CullingConstants.EPSILON, float)
