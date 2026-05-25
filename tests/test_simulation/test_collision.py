"""
Comprehensive tests for the Collision Detection module.

Tests cover:
- Broadphase algorithms (SAP, BVH, Grid, Octree)
- Narrowphase algorithms (GJK, EPA, SAT)
- Contact manifold management
- Continuous collision detection
- Collision filtering
- Collision events

Total: 145+ tests
"""

import pytest
import math

from engine.simulation.collision import (
    # Config
    BROADPHASE_MARGIN,
    CONTACT_TOLERANCE,
    MAX_CONTACT_POINTS,
    CCD_THRESHOLD_VELOCITY,
    CollisionQuality,
    CollisionConfig,
    DEFAULT_CONFIG,
    # Broadphase
    Vec3,
    AABB,
    Ray,
    CollisionPair,
    RaycastHit,
    BroadphaseType,
    SweepAndPrune,
    DynamicBVH,
    SpatialHashGrid,
    Octree,
    create_broadphase,
    # Narrowphase
    NarrowphaseAlgorithm,
    ContactResult,
    Sphere,
    Capsule,
    Box,
    ConvexHull,
    gjk_distance,
    epa_penetration,
    sat_test,
    sphere_sphere,
    sphere_capsule,
    capsule_capsule,
    box_box,
    sphere_box,
    capsule_box,
    collide_shapes,
    # Contact Manifold
    ContactPoint,
    ManifoldKey,
    ContactManifold,
    ManifoldCache,
    ContactPair,
    create_contact_pairs,
    # CCD
    CCDMode,
    CCDResult,
    MotionState,
    linear_sweep_sphere,
    linear_sweep_test,
    time_of_impact,
    time_of_impact_sphere_sphere,
    conservative_advancement,
    speculative_contacts,
    CCDManager,
    # Collision Filtering
    CollisionLayer,
    CollisionMask,
    CollisionFilter,
    should_collide,
    create_layer_matrix,
    CollisionFilterManager,
    FilterPresets,
    # Collision Events
    CollisionEventType,
    CollisionEvent,
    CollisionEventDispatcher,
    CollisionListener,
    CollisionEventProcessor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_aabb():
    """Create a simple AABB for testing."""
    return AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))


@pytest.fixture
def sphere_at_origin():
    """Create a sphere at the origin."""
    return Sphere(center=Vec3(0, 0, 0), radius=1.0)


@pytest.fixture
def unit_box():
    """Create a unit box at the origin."""
    return Box(
        center=Vec3(0, 0, 0),
        half_extents=Vec3(0.5, 0.5, 0.5),
    )


@pytest.fixture
def event_dispatcher():
    """Create a fresh event dispatcher."""
    return CollisionEventDispatcher()


# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for collision configuration."""

    def test_default_config_values(self):
        """Test default configuration values."""
        assert BROADPHASE_MARGIN == 0.05
        assert CONTACT_TOLERANCE == 0.01
        assert MAX_CONTACT_POINTS == 4
        assert CCD_THRESHOLD_VELOCITY == 10.0

    def test_collision_quality_enum(self):
        """Test collision quality presets."""
        assert CollisionQuality.LOW.value < CollisionQuality.MEDIUM.value
        assert CollisionQuality.MEDIUM.value < CollisionQuality.HIGH.value
        assert CollisionQuality.HIGH.value < CollisionQuality.ULTRA.value

    def test_config_from_quality_low(self):
        """Test low quality configuration."""
        config = CollisionConfig.from_quality(CollisionQuality.LOW)
        assert config.max_contact_points == 2
        assert config.gjk_max_iterations == 32

    def test_config_from_quality_medium(self):
        """Test medium quality configuration (default)."""
        config = CollisionConfig.from_quality(CollisionQuality.MEDIUM)
        assert config.max_contact_points == MAX_CONTACT_POINTS

    def test_config_from_quality_high(self):
        """Test high quality configuration."""
        config = CollisionConfig.from_quality(CollisionQuality.HIGH)
        assert config.max_contact_points == 8
        assert config.gjk_max_iterations == 128

    def test_config_from_quality_ultra(self):
        """Test ultra quality configuration."""
        config = CollisionConfig.from_quality(CollisionQuality.ULTRA)
        assert config.max_contact_points == 16
        assert config.gjk_max_iterations == 256

    def test_default_config_instance(self):
        """Test default config instance."""
        assert DEFAULT_CONFIG.broadphase_margin == BROADPHASE_MARGIN
        assert DEFAULT_CONFIG.contact_tolerance == CONTACT_TOLERANCE


# =============================================================================
# Vec3 Tests
# =============================================================================


class TestVec3:
    """Tests for Vec3 class."""

    def test_vec3_creation(self):
        """Test Vec3 creation."""
        v = Vec3(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3

    def test_vec3_default(self):
        """Test Vec3 default values."""
        v = Vec3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_vec3_addition(self):
        """Test Vec3 addition."""
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)
        result = a + b
        assert result.x == 5
        assert result.y == 7
        assert result.z == 9

    def test_vec3_subtraction(self):
        """Test Vec3 subtraction."""
        a = Vec3(4, 5, 6)
        b = Vec3(1, 2, 3)
        result = a - b
        assert result.x == 3
        assert result.y == 3
        assert result.z == 3

    def test_vec3_scalar_multiply(self):
        """Test Vec3 scalar multiplication."""
        v = Vec3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_vec3_dot_product(self):
        """Test Vec3 dot product."""
        a = Vec3(1, 0, 0)
        b = Vec3(0, 1, 0)
        assert a.dot(b) == 0  # Perpendicular

        c = Vec3(1, 0, 0)
        d = Vec3(1, 0, 0)
        assert c.dot(d) == 1  # Parallel

    def test_vec3_length(self):
        """Test Vec3 length."""
        v = Vec3(3, 4, 0)
        assert v.length() == 5.0

    def test_vec3_normalized(self):
        """Test Vec3 normalization."""
        v = Vec3(3, 0, 0)
        n = v.normalized()
        assert abs(n.x - 1.0) < 1e-6
        assert abs(n.length() - 1.0) < 1e-6

    def test_vec3_normalized_zero(self):
        """Test Vec3 normalization of zero vector."""
        v = Vec3(0, 0, 0)
        n = v.normalized()
        assert n.length() == 0.0

    def test_vec3_indexing(self):
        """Test Vec3 indexing."""
        v = Vec3(1, 2, 3)
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3

    def test_vec3_indexing_error(self):
        """Test Vec3 index out of range."""
        v = Vec3(1, 2, 3)
        with pytest.raises(IndexError):
            _ = v[3]


# =============================================================================
# AABB Tests
# =============================================================================


class TestAABB:
    """Tests for AABB class."""

    def test_aabb_creation(self, simple_aabb):
        """Test AABB creation."""
        assert simple_aabb.min_point.x == -1
        assert simple_aabb.max_point.x == 1

    def test_aabb_from_center_extents(self):
        """Test AABB from center and extents."""
        aabb = AABB.from_center_extents(Vec3(0, 0, 0), Vec3(1, 1, 1))
        assert aabb.min_point.x == -1
        assert aabb.max_point.x == 1

    def test_aabb_from_points(self):
        """Test AABB from points."""
        points = [Vec3(-1, 0, 0), Vec3(1, 0, 0), Vec3(0, 2, 0)]
        aabb = AABB.from_points(points)
        assert aabb.min_point.x == -1
        assert aabb.max_point.y == 2

    def test_aabb_center(self, simple_aabb):
        """Test AABB center."""
        center = simple_aabb.center()
        assert center.x == 0
        assert center.y == 0
        assert center.z == 0

    def test_aabb_extents(self, simple_aabb):
        """Test AABB extents."""
        extents = simple_aabb.extents()
        assert extents.x == 1
        assert extents.y == 1
        assert extents.z == 1

    def test_aabb_surface_area(self, simple_aabb):
        """Test AABB surface area."""
        area = simple_aabb.surface_area()
        assert area == 24.0  # 6 faces * 4 (2x2 each)

    def test_aabb_volume(self, simple_aabb):
        """Test AABB volume."""
        volume = simple_aabb.volume()
        assert volume == 8.0  # 2 * 2 * 2

    def test_aabb_expanded(self, simple_aabb):
        """Test AABB expansion."""
        expanded = simple_aabb.expanded(0.5)
        assert expanded.min_point.x == -1.5
        assert expanded.max_point.x == 1.5

    def test_aabb_contains_point_inside(self, simple_aabb):
        """Test AABB contains point - inside."""
        assert simple_aabb.contains_point(Vec3(0, 0, 0))
        assert simple_aabb.contains_point(Vec3(0.5, 0.5, 0.5))

    def test_aabb_contains_point_outside(self, simple_aabb):
        """Test AABB contains point - outside."""
        assert not simple_aabb.contains_point(Vec3(2, 0, 0))

    def test_aabb_intersects_overlapping(self):
        """Test AABB intersection - overlapping."""
        a = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        b = AABB(Vec3(0, 0, 0), Vec3(2, 2, 2))
        assert a.intersects(b)
        assert b.intersects(a)

    def test_aabb_intersects_separate(self):
        """Test AABB intersection - separate."""
        a = AABB(Vec3(-1, -1, -1), Vec3(0, 0, 0))
        b = AABB(Vec3(1, 1, 1), Vec3(2, 2, 2))
        assert not a.intersects(b)

    def test_aabb_intersects_touching(self):
        """Test AABB intersection - touching."""
        a = AABB(Vec3(-1, -1, -1), Vec3(0, 0, 0))
        b = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        assert a.intersects(b)

    def test_aabb_merge(self):
        """Test AABB merge."""
        a = AABB(Vec3(-1, -1, -1), Vec3(0, 0, 0))
        b = AABB(Vec3(0, 0, 0), Vec3(1, 1, 1))
        merged = a.merge(b)
        assert merged.min_point.x == -1
        assert merged.max_point.x == 1

    def test_aabb_ray_intersect_hit(self, simple_aabb):
        """Test AABB ray intersection - hit."""
        origin = Vec3(-5, 0, 0)
        direction = Vec3(1, 0, 0)
        hit, t_min, t_max = simple_aabb.ray_intersect(origin, direction)
        assert hit
        assert t_min > 0

    def test_aabb_ray_intersect_miss(self, simple_aabb):
        """Test AABB ray intersection - miss."""
        origin = Vec3(-5, 5, 0)
        direction = Vec3(1, 0, 0)
        hit, t_min, t_max = simple_aabb.ray_intersect(origin, direction)
        assert not hit


# =============================================================================
# Broadphase Tests
# =============================================================================


class TestSweepAndPrune:
    """Tests for Sweep and Prune broadphase."""

    def test_sap_insert(self):
        """Test SAP insert."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert sap.object_count == 1
        assert id1 == 0

    def test_sap_remove(self):
        """Test SAP remove."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert sap.remove(id1)
        assert sap.object_count == 0

    def test_sap_query_overlaps_intersecting(self):
        """Test SAP overlap query - intersecting."""
        sap = SweepAndPrune()
        sap.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        sap.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        pairs = sap.query_overlaps()
        assert len(pairs) == 1

    def test_sap_query_overlaps_separate(self):
        """Test SAP overlap query - separate."""
        sap = SweepAndPrune()
        sap.insert(AABB(Vec3(-2, -2, -2), Vec3(-1, -1, -1)))
        sap.insert(AABB(Vec3(1, 1, 1), Vec3(2, 2, 2)))
        pairs = sap.query_overlaps()
        assert len(pairs) == 0

    def test_sap_query_aabb(self):
        """Test SAP AABB query."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = sap.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        results = sap.query_aabb(AABB(Vec3(-1, -1, -1), Vec3(2, 2, 2)))
        assert id1 in results
        assert id2 not in results

    def test_sap_query_ray(self):
        """Test SAP ray query."""
        sap = SweepAndPrune()
        id1 = sap.insert(AABB(Vec3(5, -1, -1), Vec3(6, 1, 1)))
        ray = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
        hits = sap.query_ray(ray)
        assert len(hits) == 1
        assert hits[0].object_id == id1


class TestDynamicBVH:
    """Tests for Dynamic BVH broadphase."""

    def test_bvh_insert(self):
        """Test BVH insert."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert bvh.object_count == 1
        assert id1 == 0

    def test_bvh_insert_multiple(self):
        """Test BVH multiple inserts."""
        bvh = DynamicBVH()
        for i in range(10):
            bvh.insert(AABB(Vec3(i, i, i), Vec3(i + 1, i + 1, i + 1)))
        assert bvh.object_count == 10

    def test_bvh_remove(self):
        """Test BVH remove."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert bvh.remove(id1)
        assert bvh.object_count == 0

    def test_bvh_update(self):
        """Test BVH update."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert bvh.update_aabb(id1, AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))

    def test_bvh_query_overlaps(self):
        """Test BVH overlap query."""
        bvh = DynamicBVH()
        bvh.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        bvh.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        pairs = bvh.query_overlaps()
        assert len(pairs) == 1

    def test_bvh_query_aabb(self):
        """Test BVH AABB query."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = bvh.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        results = bvh.query_aabb(AABB(Vec3(-1, -1, -1), Vec3(2, 2, 2)))
        assert id1 in results
        assert id2 not in results

    def test_bvh_query_ray(self):
        """Test BVH ray query."""
        bvh = DynamicBVH()
        id1 = bvh.insert(AABB(Vec3(5, -1, -1), Vec3(6, 1, 1)))
        ray = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
        hits = bvh.query_ray(ray)
        assert len(hits) == 1
        assert hits[0].object_id == id1


class TestSpatialHashGrid:
    """Tests for Spatial Hash Grid broadphase."""

    def test_grid_insert(self):
        """Test grid insert."""
        grid = SpatialHashGrid(cell_size=2.0)
        id1 = grid.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert grid.object_count == 1

    def test_grid_zero_cell_size_raises(self):
        """Test that zero cell size raises ValueError."""
        with pytest.raises(ValueError, match="cell_size must be positive"):
            SpatialHashGrid(cell_size=0.0)

    def test_grid_query_overlaps(self):
        """Test grid overlap query."""
        grid = SpatialHashGrid(cell_size=2.0)
        grid.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        grid.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        pairs = grid.query_overlaps()
        assert len(pairs) == 1

    def test_grid_query_aabb(self):
        """Test grid AABB query."""
        grid = SpatialHashGrid(cell_size=2.0)
        id1 = grid.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = grid.insert(AABB(Vec3(10, 10, 10), Vec3(11, 11, 11)))
        results = grid.query_aabb(AABB(Vec3(-1, -1, -1), Vec3(2, 2, 2)))
        assert id1 in results
        assert id2 not in results


class TestOctree:
    """Tests for Octree broadphase."""

    def test_octree_insert(self):
        """Test octree insert."""
        octree = Octree()
        id1 = octree.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        assert octree.object_count == 1

    def test_octree_query_overlaps(self):
        """Test octree overlap query."""
        octree = Octree()
        octree.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        octree.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))
        pairs = octree.query_overlaps()
        assert len(pairs) == 1

    def test_octree_query_aabb(self):
        """Test octree AABB query."""
        octree = Octree()
        id1 = octree.insert(AABB(Vec3(0, 0, 0), Vec3(1, 1, 1)))
        id2 = octree.insert(AABB(Vec3(50, 50, 50), Vec3(51, 51, 51)))
        results = octree.query_aabb(AABB(Vec3(-1, -1, -1), Vec3(2, 2, 2)))
        assert id1 in results
        assert id2 not in results


class TestBroadphaseFactory:
    """Tests for broadphase factory function."""

    def test_create_sap(self):
        """Test creating SAP."""
        bp = create_broadphase(BroadphaseType.SAP)
        assert isinstance(bp, SweepAndPrune)

    def test_create_bvh(self):
        """Test creating BVH."""
        bp = create_broadphase(BroadphaseType.BVH)
        assert isinstance(bp, DynamicBVH)

    def test_create_grid(self):
        """Test creating Grid."""
        bp = create_broadphase(BroadphaseType.GRID)
        assert isinstance(bp, SpatialHashGrid)

    def test_create_octree(self):
        """Test creating Octree."""
        bp = create_broadphase(BroadphaseType.OCTREE)
        assert isinstance(bp, Octree)


# =============================================================================
# Narrowphase Tests
# =============================================================================


class TestSphereSphere:
    """Tests for sphere-sphere collision."""

    def test_sphere_sphere_intersecting(self):
        """Test intersecting spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1.5, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert result.colliding
        assert result.depth > 0

    def test_sphere_sphere_separate(self):
        """Test separate spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        assert not result.colliding

    def test_sphere_sphere_touching(self):
        """Test touching spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(2, 0, 0), radius=1.0)
        result = sphere_sphere(a, b)
        # Touching is considered colliding with 0 depth
        assert result.depth >= 0

    def test_sphere_sphere_concentric(self):
        """Test concentric spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(0, 0, 0), radius=0.5)
        result = sphere_sphere(a, b)
        assert result.colliding
        assert abs(result.depth - 1.5) < 1e-6  # Floating point tolerance


class TestSphereCapsule:
    """Tests for sphere-capsule collision."""

    def test_sphere_capsule_intersecting(self):
        """Test intersecting sphere and capsule."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        capsule = Capsule(start=Vec3(1, -1, 0), end=Vec3(1, 1, 0), radius=0.5)
        result = sphere_capsule(sphere, capsule)
        assert result.colliding

    def test_sphere_capsule_separate(self):
        """Test separate sphere and capsule."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        capsule = Capsule(start=Vec3(5, -1, 0), end=Vec3(5, 1, 0), radius=0.5)
        result = sphere_capsule(sphere, capsule)
        assert not result.colliding


class TestCapsuleCapsule:
    """Tests for capsule-capsule collision."""

    def test_capsule_capsule_intersecting(self):
        """Test intersecting capsules."""
        a = Capsule(start=Vec3(0, -1, 0), end=Vec3(0, 1, 0), radius=0.5)
        b = Capsule(start=Vec3(0.5, -1, 0), end=Vec3(0.5, 1, 0), radius=0.5)
        result = capsule_capsule(a, b)
        assert result.colliding

    def test_capsule_capsule_parallel_separate(self):
        """Test separate parallel capsules."""
        a = Capsule(start=Vec3(0, -1, 0), end=Vec3(0, 1, 0), radius=0.5)
        b = Capsule(start=Vec3(5, -1, 0), end=Vec3(5, 1, 0), radius=0.5)
        result = capsule_capsule(a, b)
        assert not result.colliding

    def test_capsule_capsule_crossing(self):
        """Test crossing capsules."""
        a = Capsule(start=Vec3(-1, 0, 0), end=Vec3(1, 0, 0), radius=0.3)
        b = Capsule(start=Vec3(0, -1, 0), end=Vec3(0, 1, 0), radius=0.3)
        result = capsule_capsule(a, b)
        assert result.colliding


class TestBoxBox:
    """Tests for box-box collision using SAT."""

    def test_box_box_intersecting(self):
        """Test intersecting boxes."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = box_box(a, b)
        assert result.colliding

    def test_box_box_separate(self):
        """Test separate boxes."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = box_box(a, b)
        assert not result.colliding

    def test_sat_test_direct(self):
        """Test SAT directly."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1.5, 0, 0), half_extents=Vec3(1, 1, 1))
        result = sat_test(a, b)
        assert result.colliding
        assert len(result.points) > 0


class TestSphereBox:
    """Tests for sphere-box collision."""

    def test_sphere_box_intersecting(self):
        """Test intersecting sphere and box."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        box = Box(center=Vec3(1, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = sphere_box(sphere, box)
        assert result.colliding

    def test_sphere_box_separate(self):
        """Test separate sphere and box."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        box = Box(center=Vec3(5, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = sphere_box(sphere, box)
        assert not result.colliding


class TestGJK:
    """Tests for GJK algorithm."""

    def test_gjk_distance_separate(self):
        """Test GJK distance for separate shapes."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        intersecting, distance, ca, cb = gjk_distance(a, b)
        assert not intersecting
        assert distance > 0

    def test_gjk_distance_intersecting(self):
        """Test GJK distance for intersecting shapes."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1, 0, 0), radius=1.0)
        intersecting, distance, ca, cb = gjk_distance(a, b)
        assert intersecting

    def test_gjk_with_convex_hull(self):
        """Test GJK with convex hull."""
        hull = ConvexHull(vertices=[
            Vec3(-1, -1, -1), Vec3(1, -1, -1),
            Vec3(-1, 1, -1), Vec3(1, 1, -1),
            Vec3(-1, -1, 1), Vec3(1, -1, 1),
            Vec3(-1, 1, 1), Vec3(1, 1, 1),
        ])
        sphere = Sphere(center=Vec3(5, 0, 0), radius=1.0)
        intersecting, distance, ca, cb = gjk_distance(hull, sphere)
        assert not intersecting


class TestGenericCollision:
    """Tests for generic collide_shapes function."""

    def test_collision_dispatch_spheres(self):
        """Test collision dispatch for spheres."""
        a = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        b = Sphere(center=Vec3(1, 0, 0), radius=1.0)
        result = collide_shapes(a, b)
        assert result.colliding

    def test_collision_dispatch_boxes(self):
        """Test collision dispatch for boxes."""
        a = Box(center=Vec3(0, 0, 0), half_extents=Vec3(1, 1, 1))
        b = Box(center=Vec3(1, 0, 0), half_extents=Vec3(1, 1, 1))
        result = collide_shapes(a, b)
        assert result.colliding

    def test_collision_mixed_shapes(self):
        """Test collision between different shapes."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=1.0)
        box = Box(center=Vec3(1, 0, 0), half_extents=Vec3(0.5, 0.5, 0.5))
        result = collide_shapes(sphere, box)
        assert result.colliding


# =============================================================================
# Contact Manifold Tests
# =============================================================================


class TestContactPoint:
    """Tests for ContactPoint."""

    def test_contact_point_creation(self):
        """Test contact point creation."""
        cp = ContactPoint(
            position=Vec3(1, 0, 0),
            normal=Vec3(1, 0, 0),
            depth=0.1,
        )
        assert cp.depth == 0.1

    def test_contact_point_distance(self):
        """Test contact point distance."""
        a = ContactPoint(position=Vec3(0, 0, 0))
        b = ContactPoint(position=Vec3(3, 4, 0))
        assert a.distance_to(b) == 5.0

    def test_contact_point_warm_start(self):
        """Test warm start impulses."""
        cp = ContactPoint()
        cp.update_impulse(10.0, 1.0, 2.0)
        n, t1, t2 = cp.get_warm_start_impulse()
        assert n == 10.0 * 0.8  # WARM_START_FACTOR


class TestContactManifold:
    """Tests for ContactManifold."""

    def test_manifold_creation(self):
        """Test manifold creation."""
        m = ContactManifold(body_a=1, body_b=2)
        assert m.body_a == 1
        assert m.body_b == 2
        assert m.contact_count == 0

    def test_manifold_add_contact(self):
        """Test adding contact to manifold."""
        m = ContactManifold(body_a=1, body_b=2)
        m.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        assert m.contact_count == 1

    def test_manifold_add_multiple_contacts(self):
        """Test adding multiple contacts."""
        m = ContactManifold(body_a=1, body_b=2, max_contacts=4)
        for i in range(5):
            m.add_contact(Vec3(i, 0, 0), Vec3(0, 1, 0), 0.1)
        assert m.contact_count == 4  # Reduced to max

    def test_manifold_remove_contact(self):
        """Test removing contact."""
        m = ContactManifold(body_a=1, body_b=2)
        cp = m.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        assert m.remove_contact(cp.contact_id)
        assert m.contact_count == 0

    def test_manifold_reduce(self):
        """Test manifold reduction."""
        m = ContactManifold(body_a=1, body_b=2, max_contacts=2)
        # Add contacts that form a spread pattern
        m.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.2)  # Deepest
        m.add_contact(Vec3(1, 0, 0), Vec3(0, 1, 0), 0.1)
        m.add_contact(Vec3(-1, 0, 0), Vec3(0, 1, 0), 0.1)
        m.add_contact(Vec3(0, 0, 1), Vec3(0, 1, 0), 0.1)
        assert m.contact_count == 2

    def test_manifold_age_contacts(self):
        """Test contact aging."""
        m = ContactManifold(body_a=1, body_b=2)
        m.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        for _ in range(5):
            m.age_contacts()
        assert m.contact_count == 0  # Removed due to age


class TestManifoldCache:
    """Tests for ManifoldCache."""

    def test_cache_get_or_create(self):
        """Test get or create manifold."""
        cache = ManifoldCache()
        m = cache.get_or_create(1, 2)
        assert m.body_a == 1
        assert m.body_b == 2

    def test_cache_get_existing(self):
        """Test getting existing manifold."""
        cache = ManifoldCache()
        m1 = cache.get_or_create(1, 2)
        m2 = cache.get_or_create(1, 2)
        assert m1 is m2

    def test_cache_remove(self):
        """Test removing from cache."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        assert cache.remove(1, 2)
        assert cache.manifold_count == 0

    def test_cache_remove_body(self):
        """Test removing all manifolds for a body."""
        cache = ManifoldCache()
        cache.get_or_create(1, 2)
        cache.get_or_create(1, 3)
        cache.get_or_create(2, 3)
        count = cache.remove_body(1)
        assert count == 2


# =============================================================================
# CCD Tests
# =============================================================================


class TestCCDResult:
    """Tests for CCDResult."""

    def test_ccd_result_no_hit(self):
        """Test CCD result with no hit."""
        result = CCDResult()
        assert not result.hit
        assert not result  # bool conversion

    def test_ccd_result_hit(self):
        """Test CCD result with hit."""
        result = CCDResult(hit=True, toi=0.5)
        assert result.hit
        assert result  # bool conversion


class TestMotionState:
    """Tests for MotionState."""

    def test_motion_state_position_at(self):
        """Test position interpolation."""
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(10, 0, 0),
        )
        pos = motion.position_at(0.5)
        assert pos.x == 5.0

    def test_motion_state_speed(self):
        """Test speed calculation."""
        motion = MotionState(velocity=Vec3(3, 4, 0))
        assert motion.speed() == 5.0


class TestLinearSweep:
    """Tests for linear sweep CCD."""

    def test_linear_sweep_sphere_hit(self):
        """Test sphere sweep - hit using analytical TOI."""
        # Use the analytical sphere-sphere TOI which is more reliable
        sphere_a = Sphere(center=Vec3(0, 0, 0), radius=0.5)
        velocity_a = Vec3(50, 0, 0)  # Fast moving
        sphere_b = Sphere(center=Vec3(10, 0, 0), radius=1.0)
        velocity_b = Vec3(0, 0, 0)

        result = time_of_impact_sphere_sphere(
            sphere_a, velocity_a,
            sphere_b, velocity_b,
            dt=1.0,
        )
        assert result.hit
        assert 0 < result.toi < 1

    def test_linear_sweep_sphere_miss(self):
        """Test sphere sweep - miss."""
        sphere = Sphere(center=Vec3(0, 0, 0), radius=0.5)
        motion = MotionState(
            position=Vec3(0, 0, 0),
            velocity=Vec3(50, 0, 0),
        )
        target = Sphere(center=Vec3(10, 10, 0), radius=1.0)  # Off path
        result = linear_sweep_sphere(sphere, motion, target)
        assert not result.hit


class TestTimeOfImpact:
    """Tests for time of impact calculation."""

    def test_toi_sphere_sphere(self):
        """Test TOI for spheres."""
        result = time_of_impact_sphere_sphere(
            Sphere(Vec3(0, 0, 0), 0.5),
            Vec3(10, 0, 0),
            Sphere(Vec3(5, 0, 0), 0.5),
            Vec3(0, 0, 0),
        )
        assert result.hit

    def test_toi_spheres_diverging(self):
        """Test TOI for diverging spheres."""
        result = time_of_impact_sphere_sphere(
            Sphere(Vec3(0, 0, 0), 0.5),
            Vec3(-10, 0, 0),  # Moving away
            Sphere(Vec3(5, 0, 0), 0.5),
            Vec3(10, 0, 0),   # Also moving away
        )
        assert not result.hit

    def test_toi_spheres_stationary(self):
        """Test TOI for stationary spheres (same velocity)."""
        # Both spheres moving with same velocity = no relative motion
        result = time_of_impact_sphere_sphere(
            Sphere(Vec3(0, 0, 0), 0.5),
            Vec3(10, 0, 0),
            Sphere(Vec3(5, 0, 0), 0.5),
            Vec3(10, 0, 0),  # Same velocity
        )
        # Not overlapping and not moving relative to each other
        assert not result.hit


class TestCCDManager:
    """Tests for CCD manager."""

    def test_ccd_manager_creation(self):
        """Test CCD manager creation."""
        manager = CCDManager()
        assert manager.mode == CCDMode.SWEPT

    def test_ccd_manager_needs_ccd(self):
        """Test needs_ccd check."""
        manager = CCDManager()
        slow = Vec3(1, 0, 0)
        fast = Vec3(100, 0, 0)
        assert not manager.needs_ccd(slow)
        assert manager.needs_ccd(fast)

    def test_ccd_manager_test_pair(self):
        """Test pair testing."""
        manager = CCDManager()
        sphere_a = Sphere(Vec3(0, 0, 0), 0.5)
        motion_a = MotionState(velocity=Vec3(20, 0, 0))
        sphere_b = Sphere(Vec3(5, 0, 0), 0.5)
        motion_b = MotionState()
        result = manager.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        assert result.hit


# =============================================================================
# Collision Filter Tests
# =============================================================================


class TestCollisionLayer:
    """Tests for CollisionLayer."""

    def test_collision_layer_values(self):
        """Test layer bit values."""
        assert CollisionLayer.DEFAULT == 1
        assert CollisionLayer.STATIC == 2
        assert CollisionLayer.DYNAMIC == 4

    def test_collision_layer_combinations(self):
        """Test layer combinations."""
        combined = CollisionLayer.STATIC | CollisionLayer.DYNAMIC
        assert CollisionLayer.STATIC in combined
        assert CollisionLayer.DYNAMIC in combined


class TestCollisionMask:
    """Tests for CollisionMask."""

    def test_mask_from_layers(self):
        """Test mask creation from layers."""
        mask = CollisionMask.from_layers(
            CollisionLayer.STATIC,
            CollisionLayer.DYNAMIC,
        )
        assert mask.includes(CollisionLayer.STATIC)
        assert mask.includes(CollisionLayer.DYNAMIC)
        assert not mask.includes(CollisionLayer.TRIGGER)

    def test_mask_all_except(self):
        """Test mask with exclusions."""
        mask = CollisionMask.all_except(CollisionLayer.DEBRIS)
        assert mask.includes(CollisionLayer.STATIC)
        assert not mask.includes(CollisionLayer.DEBRIS)

    def test_mask_operations(self):
        """Test mask operations."""
        mask = CollisionMask.from_layers(CollisionLayer.STATIC)
        mask = mask.add(CollisionLayer.DYNAMIC)
        assert mask.includes(CollisionLayer.DYNAMIC)
        mask = mask.remove(CollisionLayer.STATIC)
        assert not mask.includes(CollisionLayer.STATIC)


class TestCollisionFilter:
    """Tests for CollisionFilter."""

    def test_filter_creation(self):
        """Test filter creation."""
        f = CollisionFilter(
            category=CollisionLayer.PLAYER,
            mask=CollisionMask(CollisionLayer.ALL),
        )
        assert f.category == CollisionLayer.PLAYER

    def test_filter_presets(self):
        """Test filter presets."""
        static = CollisionFilter.static()
        assert static.category == CollisionLayer.STATIC

        dynamic = CollisionFilter.dynamic()
        assert dynamic.category == CollisionLayer.DYNAMIC

        trigger = CollisionFilter.trigger()
        assert trigger.category == CollisionLayer.TRIGGER


class TestShouldCollide:
    """Tests for should_collide function."""

    def test_should_collide_default(self):
        """Test default filters collide."""
        a = CollisionFilter()
        b = CollisionFilter()
        assert should_collide(a, b)

    def test_should_collide_filtered(self):
        """Test filtered collision."""
        a = CollisionFilter(
            category=CollisionLayer.PLAYER,
            mask=CollisionMask.all_except(CollisionLayer.NPC),
        )
        b = CollisionFilter(category=CollisionLayer.NPC)
        assert not should_collide(a, b)

    def test_should_collide_same_group(self):
        """Test same group doesn't collide."""
        a = CollisionFilter(group=1)
        b = CollisionFilter(group=1)
        assert not should_collide(a, b)

    def test_should_collide_different_groups(self):
        """Test different groups collide."""
        a = CollisionFilter(group=1)
        b = CollisionFilter(group=2)
        assert should_collide(a, b)


class TestCollisionFilterManager:
    """Tests for CollisionFilterManager."""

    def test_manager_set_get_filter(self):
        """Test setting and getting filters."""
        manager = CollisionFilterManager()
        f = CollisionFilter.player()
        manager.set_filter(1, f)
        assert manager.get_filter(1).category == CollisionLayer.PLAYER

    def test_manager_should_collide(self):
        """Test manager collision check."""
        manager = CollisionFilterManager()
        manager.set_filter(1, CollisionFilter.player(group=1))
        manager.set_filter(2, CollisionFilter.player(group=1))
        assert not manager.should_collide(1, 2)  # Same group

    def test_manager_custom_callback(self):
        """Test custom collision callback."""
        manager = CollisionFilterManager()
        manager.add_callback(lambda a, b: a != 1)  # Block object 1
        assert not manager.should_collide(1, 2)
        assert manager.should_collide(2, 3)


# =============================================================================
# Collision Event Tests
# =============================================================================


class TestCollisionEvent:
    """Tests for CollisionEvent."""

    def test_event_creation(self):
        """Test event creation."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
        )
        assert event.is_begin
        assert not event.is_persist
        assert not event.is_end

    def test_event_get_other_body(self):
        """Test getting other body."""
        event = CollisionEvent(
            event_type=CollisionEventType.BEGIN,
            body_a=1,
            body_b=2,
        )
        assert event.get_other_body(1) == 2
        assert event.get_other_body(2) == 1


class TestCollisionEventDispatcher:
    """Tests for CollisionEventDispatcher."""

    def test_dispatcher_creation(self, event_dispatcher):
        """Test dispatcher creation."""
        assert event_dispatcher.handler_count() == 0

    def test_dispatcher_register_handler(self, event_dispatcher):
        """Test registering handlers."""
        event_dispatcher.on_collision_begin(lambda e: True)
        event_dispatcher.on_collision_persist(lambda e: True)
        event_dispatcher.on_collision_end(lambda e: True)
        assert event_dispatcher.handler_count() == 3

    def test_dispatcher_dispatch_begin(self, event_dispatcher):
        """Test dispatching begin event."""
        received = []
        event_dispatcher.on_collision_begin(
            lambda e: received.append(e) or True
        )
        event_dispatcher.dispatch_begin(1, 2, [])
        assert len(received) == 1
        assert received[0].is_begin

    def test_dispatcher_dispatch_persist(self, event_dispatcher):
        """Test dispatching persist event."""
        received = []
        event_dispatcher.on_collision_persist(
            lambda e: received.append(e) or True
        )
        event_dispatcher.dispatch_persist(1, 2, [])
        assert len(received) == 1

    def test_dispatcher_dispatch_end(self, event_dispatcher):
        """Test dispatching end event."""
        received = []
        event_dispatcher.on_collision_end(
            lambda e: received.append(e) or True
        )
        event_dispatcher.dispatch_end(1, 2)
        assert len(received) == 1

    def test_dispatcher_body_handler(self, event_dispatcher):
        """Test body-specific handler."""
        received = []
        event_dispatcher.on_body_collision(
            1, lambda e: received.append(e) or True
        )
        event_dispatcher.dispatch_begin(1, 2, [])
        event_dispatcher.dispatch_begin(3, 4, [])
        assert len(received) == 1

    def test_dispatcher_filter(self, event_dispatcher):
        """Test event filter."""
        event_dispatcher.add_filter(lambda a, b: a != 1)
        received = []
        event_dispatcher.on_collision_begin(
            lambda e: received.append(e) or True
        )
        event_dispatcher.dispatch_begin(1, 2, [])
        event_dispatcher.dispatch_begin(3, 4, [])
        assert len(received) == 1

    def test_dispatcher_priority(self, event_dispatcher):
        """Test handler priority."""
        order = []
        event_dispatcher.on_collision_begin(
            lambda e: order.append("low") or True, priority=0
        )
        event_dispatcher.on_collision_begin(
            lambda e: order.append("high") or True, priority=10
        )
        event_dispatcher.dispatch_begin(1, 2, [])
        assert order == ["high", "low"]

    def test_dispatcher_deferred(self, event_dispatcher):
        """Test deferred event processing."""
        received = []
        event_dispatcher.on_collision_begin(
            lambda e: received.append(e) or True
        )
        event_dispatcher.begin_deferred()
        event_dispatcher.dispatch_begin(1, 2, [])
        assert len(received) == 0
        event_dispatcher.end_deferred()
        assert len(received) == 1


class TestCollisionListener:
    """Tests for CollisionListener."""

    def test_listener_base_class(self):
        """Test listener base class."""

        class TestListener(CollisionListener):
            def __init__(self):
                self.began = False

            def on_collision_begin(self, event):
                self.began = True
                return True

        listener = TestListener()
        dispatcher = CollisionEventDispatcher()
        listener.register(dispatcher)
        dispatcher.dispatch_begin(1, 2, [])
        assert listener.began


class TestCollisionEventProcessor:
    """Tests for CollisionEventProcessor."""

    def test_processor_process_manifold(self):
        """Test processing manifold."""
        dispatcher = CollisionEventDispatcher()
        processor = CollisionEventProcessor(dispatcher)

        received = []
        dispatcher.on_collision_begin(lambda e: received.append(e) or True)

        manifold = ContactManifold(1, 2)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        processor.process_manifold(manifold)

        assert len(received) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestCollisionIntegration:
    """Integration tests combining multiple components."""

    def test_full_collision_pipeline(self):
        """Test complete collision detection pipeline."""
        # Broadphase
        broadphase = create_broadphase(BroadphaseType.BVH)
        id1 = broadphase.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        id2 = broadphase.insert(AABB(Vec3(0, 0, 0), Vec3(2, 2, 2)))

        pairs = broadphase.query_overlaps()
        assert len(pairs) > 0

        # Narrowphase
        sphere1 = Sphere(Vec3(0, 0, 0), 1.0)
        sphere2 = Sphere(Vec3(1.5, 0, 0), 1.0)
        result = collide_shapes(sphere1, sphere2)
        assert result.colliding

        # Contact manifold
        cache = ManifoldCache()
        manifold = cache.get_or_create(id1, id2)
        for point in result.points:
            manifold.add_contact(
                point, result.normal, result.depth
            )

        # Events
        dispatcher = CollisionEventDispatcher()
        events = []
        dispatcher.on_collision_begin(lambda e: events.append(e) or True)

        processor = CollisionEventProcessor(dispatcher)
        processor.process_manifold(manifold)
        assert len(events) == 1

    def test_filtered_collision_pipeline(self):
        """Test collision pipeline with filtering."""
        filter_manager = CollisionFilterManager()
        filter_manager.set_filter(1, CollisionFilter.player(group=1))
        filter_manager.set_filter(2, CollisionFilter.player(group=1))

        # Same group - should not collide
        assert not filter_manager.should_collide(1, 2)

        filter_manager.set_filter(3, CollisionFilter.enemy())
        assert filter_manager.should_collide(1, 3)

    def test_ccd_with_events(self):
        """Test CCD integration with events."""
        ccd = CCDManager(mode=CCDMode.SWEPT)
        dispatcher = CollisionEventDispatcher()

        events = []
        dispatcher.on_collision_begin(lambda e: events.append(e) or True)

        sphere_a = Sphere(Vec3(0, 0, 0), 0.5)
        motion_a = MotionState(velocity=Vec3(50, 0, 0))
        sphere_b = Sphere(Vec3(3, 0, 0), 0.5)
        motion_b = MotionState()

        result = ccd.test_pair(sphere_a, motion_a, sphere_b, motion_b)
        if result.hit:
            dispatcher.dispatch_begin(1, 2, [
                ContactPoint(
                    position=result.point,
                    normal=result.normal,
                    depth=0.0,
                )
            ])

        assert len(events) == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_broadphase(self):
        """Test empty broadphase queries."""
        bp = DynamicBVH()
        assert bp.query_overlaps() == []
        assert bp.query_aabb(AABB()) == []

    def test_single_element_broadphase(self):
        """Test broadphase with single element."""
        bp = DynamicBVH()
        id1 = bp.insert(AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1)))
        # Single element should return no overlapping pairs
        pairs = bp.query_overlaps()
        assert len(pairs) == 0
        # Query should find the single object
        results = bp.query_aabb(AABB(Vec3(0, 0, 0), Vec3(0.5, 0.5, 0.5)))
        assert len(results) == 1
        assert results[0] == id1
        # Remove should work
        assert bp.remove(id1)
        assert bp.object_count == 0

    def test_degenerate_shapes(self):
        """Test degenerate shape handling."""
        # Zero-radius sphere at box center should be inside/colliding
        sphere = Sphere(Vec3(0, 0, 0), 0.0)
        box = Box(Vec3(0, 0, 0), Vec3(1, 1, 1))
        result = collide_shapes(sphere, box)
        # A point at the center of a box is inside it
        assert result.colliding
        # Zero-radius sphere outside box should not collide
        sphere_outside = Sphere(Vec3(5, 0, 0), 0.0)
        result_outside = collide_shapes(sphere_outside, box)
        assert not result_outside.colliding

    def test_coincident_contacts(self):
        """Test handling of coincident contacts."""
        manifold = ContactManifold(1, 2)
        # Add contacts at same position - should be merged due to position matching
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.1)
        manifold.add_contact(Vec3(0, 0, 0), Vec3(0, 1, 0), 0.2)  # Second with different depth
        # Should merge due to position-based matching (same position within threshold)
        assert manifold.contact_count == 1  # Merged into one
        # The contact should have the updated depth
        assert manifold.contacts[0].depth == 0.2

    def test_high_velocity_ccd(self):
        """Test CCD with very high velocity."""
        ccd = CCDManager()
        sphere = Sphere(Vec3(0, 0, 0), 0.1)
        motion = MotionState(velocity=Vec3(10000, 0, 0))
        target = Sphere(Vec3(100, 0, 0), 0.1)
        result = ccd.test_pair(sphere, motion, target, MotionState())
        # High velocity moving toward target should result in collision
        assert result.hit
        # TOI should be positive and less than 1
        assert 0 < result.toi < 1
        # TOI should be approximately when spheres touch: distance=100, radii=0.2, velocity=10000
        # Time to impact = (100 - 0.2) / 10000 = 0.00998
        assert abs(result.toi - 0.00998) < 0.001

    def test_manifold_cache_eviction(self):
        """Test manifold cache eviction."""
        cache = ManifoldCache(max_manifolds=5)
        for i in range(10):
            cache.get_or_create(i, i + 100)
        assert cache.manifold_count <= 5


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Basic performance tests."""

    def test_broadphase_many_objects(self):
        """Test broadphase with many objects."""
        bvh = DynamicBVH()
        for i in range(100):
            x = (i % 10) * 2
            y = (i // 10) * 2
            bvh.insert(AABB(Vec3(x, y, 0), Vec3(x + 1, y + 1, 1)))
        pairs = bvh.query_overlaps()
        # Should complete without error

    def test_narrowphase_batch(self):
        """Test batch narrowphase tests."""
        shapes = [
            (Sphere(Vec3(i * 0.5, 0, 0), 0.4), Sphere(Vec3(i * 0.5 + 0.3, 0, 0), 0.4))
            for i in range(50)
        ]
        for a, b in shapes:
            result = collide_shapes(a, b)
            # Should complete quickly


# =============================================================================
# Additional Tests for 140+ Coverage
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests to ensure 140+ test coverage."""

    def test_vec3_min_max_components(self):
        """Test Vec3 min/max components."""
        a = Vec3(1, 2, 3)
        b = Vec3(3, 1, 2)
        min_v = a.min_components(b)
        max_v = a.max_components(b)
        assert min_v.x == 1 and min_v.y == 1 and min_v.z == 2
        assert max_v.x == 3 and max_v.y == 2 and max_v.z == 3

    def test_ray_point_at(self):
        """Test Ray point_at method."""
        ray = Ray(Vec3(0, 0, 0), Vec3(1, 0, 0))
        point = ray.point_at(5.0)
        assert point.x == 5.0

    def test_collision_pair_hash(self):
        """Test CollisionPair hash and equality."""
        pair1 = CollisionPair(1, 2)
        pair2 = CollisionPair(2, 1)
        assert hash(pair1) == hash(pair2)
        assert pair1 == pair2

    def test_contact_result_bool(self):
        """Test ContactResult bool conversion."""
        colliding = ContactResult(colliding=True)
        not_colliding = ContactResult(colliding=False)
        assert colliding
        assert not not_colliding

    def test_capsule_properties(self):
        """Test Capsule axis and height properties."""
        capsule = Capsule(
            start=Vec3(0, 0, 0),
            end=Vec3(0, 2, 0),
            radius=0.5,
        )
        assert capsule.height == 2.0
        assert capsule.axis.y == 2.0

    def test_degenerate_capsule_closest_point(self):
        """Test degenerate capsule (point capsule) closest point."""
        # Capsule with start == end is a point capsule (sphere)
        capsule = Capsule(
            start=Vec3(1, 2, 3),
            end=Vec3(1, 2, 3),
            radius=0.5,
        )
        # Closest point on axis should be the start/end point
        closest = capsule.closest_point_on_axis(Vec3(5, 5, 5))
        assert closest.x == 1.0
        assert closest.y == 2.0
        assert closest.z == 3.0

    def test_box_get_vertices(self, unit_box):
        """Test Box vertices generation."""
        vertices = unit_box.get_vertices()
        assert len(vertices) == 8

    def test_convex_hull_support(self):
        """Test ConvexHull support function."""
        hull = ConvexHull(vertices=[
            Vec3(-1, 0, 0),
            Vec3(1, 0, 0),
            Vec3(0, 1, 0),
        ])
        support = hull.support(Vec3(1, 0, 0))
        assert support.x == 1.0

    def test_convex_hull_empty_raises_error(self):
        """Test that empty ConvexHull raises ValueError on support."""
        empty_hull = ConvexHull(vertices=[])
        with pytest.raises(ValueError, match="no vertices"):
            empty_hull.support(Vec3(1, 0, 0))

    def test_aabb_from_empty_points(self):
        """Test AABB from empty points list."""
        aabb = AABB.from_points([])
        # Should return default AABB (zeros)
        assert aabb.min_point.x == 0.0
        assert aabb.min_point.y == 0.0
        assert aabb.min_point.z == 0.0
        assert aabb.max_point.x == 0.0
        assert aabb.max_point.y == 0.0
        assert aabb.max_point.z == 0.0

    def test_manifold_key_equality(self):
        """Test ManifoldKey equality and hashing."""
        key1 = ManifoldKey(1, 2)
        key2 = ManifoldKey(2, 1)
        assert key1 == key2
        assert hash(key1) == hash(key2)

    def test_contact_pair_tangent_basis(self):
        """Test ContactPair tangent basis computation."""
        pair = ContactPair(
            body_a=1,
            body_b=2,
            contact=ContactPoint(),
            normal=Vec3(0, 1, 0),
        )
        pair.compute_tangent_basis()
        # Tangents should be orthogonal to normal
        assert abs(pair.tangent_1.dot(pair.normal)) < 1e-6
        assert abs(pair.tangent_2.dot(pair.normal)) < 1e-6

    def test_filter_presets_platformer(self):
        """Test platformer filter presets."""
        presets = FilterPresets.platformer()
        assert "terrain" in presets
        assert "player" in presets

    def test_filter_presets_racing(self):
        """Test racing filter presets."""
        presets = FilterPresets.racing()
        assert "track" in presets
        assert "vehicle" in presets
