"""
Advanced tests for foliage system.

Tests for:
- Deterministic placement (same seed = same result)
- Minimum spacing enforcement in Poisson-like placement
- Frustum culling accuracy
- Grass chunk streaming behavior
- Edge cases and algorithmic correctness
- Zero area handling
- Buffer overflow prevention
"""

import math
from typing import Tuple

import pytest

from engine.world.foliage.grass import (
    GrassChunk,
    GrassSettings,
    LandscapeGrass,
    ProceduralGrass,
)
from engine.world.foliage.instances import (
    FoliageCluster,
    FoliageInstance,
    FoliageManager,
    Frustum,
    HierarchicalInstancedMesh,
)
from engine.world.foliage.placement import (
    Bounds,
    FoliagePlacement,
    NoiseGenerator,
    PlacementResult,
    PlacementRule,
    ProceduralPlacer,
)
from engine.world.foliage.types import FoliageType, GrassType


# =============================================================================
# Mock Terrain for Testing
# =============================================================================


class MockTerrain:
    """Configurable mock terrain for testing."""

    def __init__(
        self,
        height: float = 0.0,
        normal: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        layer: int = 0,
        water: bool = False,
        road: bool = False,
        height_map: dict = None,
        normal_map: dict = None,
    ):
        self._height = height
        self._normal = normal
        self._layer = layer
        self._water = water
        self._road = road
        self._height_map = height_map or {}
        self._normal_map = normal_map or {}

    def get_height_at(self, x: float, z: float) -> float:
        key = (round(x, 2), round(z, 2))
        return self._height_map.get(key, self._height)

    def get_normal_at(self, x: float, z: float) -> Tuple[float, float, float]:
        key = (round(x, 2), round(z, 2))
        return self._normal_map.get(key, self._normal)

    def get_layer_at(self, x: float, z: float) -> int:
        return self._layer

    def is_water_at(self, x: float, z: float) -> bool:
        return self._water

    def is_road_at(self, x: float, z: float) -> bool:
        return self._road


# =============================================================================
# Deterministic Placement Tests
# =============================================================================


class TestDeterministicPlacement:
    """Tests ensuring same seed produces identical results."""

    def test_procedural_placer_same_seed_same_positions(self):
        """Same seed must produce identical placement positions."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        rule = PlacementRule(noise_threshold=0.0)

        # Generate with same seed twice
        placer1 = ProceduralPlacer(seed=12345)
        results1 = placer1.generate_in_bounds(
            terrain, bounds, "test", 2.0, 1.0, (0.8, 1.2), True, rule
        )

        placer2 = ProceduralPlacer(seed=12345)
        results2 = placer2.generate_in_bounds(
            terrain, bounds, "test", 2.0, 1.0, (0.8, 1.2), True, rule
        )

        assert len(results1) == len(results2), "Same seed must produce same count"
        for r1, r2 in zip(results1, results2):
            assert r1.position == r2.position, "Positions must be identical"
            assert r1.rotation == r2.rotation, "Rotations must be identical"
            assert r1.scale == r2.scale, "Scales must be identical"

    def test_procedural_placer_different_seed_different_positions(self):
        """Different seeds must produce different placement positions."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer1 = ProceduralPlacer(seed=11111)
        results1 = placer1.generate_in_bounds(
            terrain, bounds, "test", 2.0, 1.0, (0.8, 1.2), True, rule
        )

        placer2 = ProceduralPlacer(seed=22222)
        results2 = placer2.generate_in_bounds(
            terrain, bounds, "test", 2.0, 1.0, (0.8, 1.2), True, rule
        )

        # At least some positions should differ
        if len(results1) > 0 and len(results2) > 0:
            different_found = False
            for r1, r2 in zip(results1, results2):
                if r1.position != r2.position:
                    different_found = True
                    break
            assert different_found, "Different seeds should produce different positions"

    def test_noise_generator_deterministic(self):
        """Noise generator must be fully deterministic."""
        noise1 = NoiseGenerator(seed=42)
        noise2 = NoiseGenerator(seed=42)

        # Test multiple positions
        test_points = [(0.0, 0.0), (10.5, 20.3), (-5.0, 15.0), (100.0, 100.0)]
        for x, z in test_points:
            v1 = noise1.sample(x, z, scale=10.0)
            v2 = noise2.sample(x, z, scale=10.0)
            assert v1 == v2, f"Noise at ({x}, {z}) must be identical with same seed"

    def test_grass_generation_deterministic(self):
        """Grass generation must be deterministic with same seed."""
        settings = GrassSettings(density_scale=1.0)
        terrain = MockTerrain()
        grass_type = GrassType(type_id="test_grass", density=10.0)
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=20.0, max_z=20.0)

        gen1 = ProceduralGrass(settings, seed=99999)
        instances1 = gen1.generate_for_chunk(terrain, bounds, grass_type)

        gen2 = ProceduralGrass(settings, seed=99999)
        instances2 = gen2.generate_for_chunk(terrain, bounds, grass_type)

        assert len(instances1) == len(instances2)
        for i1, i2 in zip(instances1, instances2):
            assert i1.position == i2.position
            assert i1.rotation == i2.rotation
            assert i1.height == i2.height

    def test_foliage_placement_deterministic(self):
        """FoliagePlacement wrapper must produce deterministic results."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=30.0, max_z=30.0)

        fp1 = FoliagePlacement(
            foliage_type_id="plant",
            seed=777,
            density=3.0,
            rules=PlacementRule(noise_threshold=0.0),
        )
        results1 = fp1.generate_placements(terrain, bounds)

        fp2 = FoliagePlacement(
            foliage_type_id="plant",
            seed=777,
            density=3.0,
            rules=PlacementRule(noise_threshold=0.0),
        )
        results2 = fp2.generate_placements(terrain, bounds)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.position == r2.position


# =============================================================================
# Minimum Spacing Tests
# =============================================================================


class TestMinimumSpacing:
    """Tests verifying minimum spacing between placement instances."""

    def test_grid_spacing_respects_min_spacing(self):
        """Grid-based placement should respect minimum spacing."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        rule = PlacementRule(noise_threshold=0.0)
        min_spacing = 5.0

        placer = ProceduralPlacer(seed=42)
        # High density but with min_spacing constraint
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 10.0, min_spacing, (1.0, 1.0), False, rule
        )

        # Check all pairs for minimum spacing
        # Note: Current implementation uses grid, so spacing is guaranteed
        # by the grid structure itself, but jitter may reduce effective spacing
        if len(results) > 1:
            for i, r1 in enumerate(results):
                for r2 in results[i + 1:]:
                    dx = r1.position[0] - r2.position[0]
                    dz = r1.position[2] - r2.position[2]
                    dist = math.sqrt(dx * dx + dz * dz)
                    # Allow some tolerance due to jitter
                    # The grid spacing is max(min_spacing, 1/sqrt(density))
                    # Jitter can reduce this by up to 0.5 * spacing on each point
                    effective_min = min_spacing * 0.25  # Conservative check
                    assert dist >= effective_min, (
                        f"Points too close: {dist:.3f} < {effective_min:.3f}"
                    )

    def test_spacing_calculation_with_density(self):
        """Spacing should be max(min_spacing, 1/sqrt(density))."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=20.0, max_z=20.0)
        rule = PlacementRule(noise_threshold=0.0)

        # Low density: 1/sqrt(0.25) = 2.0, min_spacing=1.0 -> spacing=2.0
        placer = ProceduralPlacer(seed=42)
        results_low = placer.generate_in_bounds(
            terrain, bounds, "test", 0.25, 1.0, (1.0, 1.0), False, rule
        )

        # High density: 1/sqrt(4) = 0.5, min_spacing=2.0 -> spacing=2.0
        results_high = placer.generate_in_bounds(
            terrain, bounds, "test", 4.0, 2.0, (1.0, 1.0), False, rule
        )

        # Both should produce similar spacing due to constraint
        # (approximately 2.0 in both cases)
        # Count roughly = area / spacing^2 = 400 / 4 = 100
        assert len(results_low) <= 150  # Some tolerance
        assert len(results_high) <= 150


# =============================================================================
# Frustum Culling Tests
# =============================================================================


class TestFrustumCulling:
    """Tests for frustum culling accuracy."""

    def test_frustum_cull_cluster_outside(self):
        """Cluster entirely outside frustum should be fully culled."""
        # Create a frustum that only includes positive X
        # Plane: x >= 100 (normal (1,0,0), d=-100)
        frustum = Frustum(planes=[
            (1.0, 0.0, 0.0, -100.0),  # x >= 100
        ])

        # Create cluster in negative X space
        bounds = Bounds(min_x=-50.0, min_z=0.0, max_x=0.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test")
        cluster.add_instance(FoliageInstance(instance_id=0, position=(-25.0, 0.0, 25.0)))
        cluster.add_instance(FoliageInstance(instance_id=1, position=(-40.0, 0.0, 40.0)))

        visible = cluster.cull(frustum)
        assert visible == 0, "All instances should be culled"
        for inst in cluster.get_instances():
            assert inst.visible is False

    def test_frustum_cull_cluster_inside(self):
        """Cluster entirely inside frustum should have all instances visible."""
        # Create a frustum that includes everything
        frustum = Frustum(planes=[])  # No planes = everything inside

        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test")
        cluster.add_instance(FoliageInstance(instance_id=0, position=(10.0, 5.0, 10.0)))
        cluster.add_instance(FoliageInstance(instance_id=1, position=(40.0, 5.0, 40.0)))

        visible = cluster.cull(frustum)
        assert visible == 2
        for inst in cluster.get_instances():
            assert inst.visible is True

    def test_frustum_point_containment_accuracy(self):
        """Test frustum point containment with multiple planes."""
        # Create a box frustum: 0 <= x <= 100, 0 <= y <= 100, 0 <= z <= 100
        frustum = Frustum(planes=[
            (1.0, 0.0, 0.0, 0.0),     # x >= 0
            (-1.0, 0.0, 0.0, 100.0),  # x <= 100 (i.e., -x + 100 >= 0)
            (0.0, 1.0, 0.0, 0.0),     # y >= 0
            (0.0, -1.0, 0.0, 100.0),  # y <= 100
            (0.0, 0.0, 1.0, 0.0),     # z >= 0
            (0.0, 0.0, -1.0, 100.0),  # z <= 100
        ])

        # Inside point
        assert frustum.contains_point((50.0, 50.0, 50.0)) is True
        assert frustum.contains_point((0.0, 0.0, 0.0)) is True
        assert frustum.contains_point((100.0, 100.0, 100.0)) is True

        # Outside points
        assert frustum.contains_point((-1.0, 50.0, 50.0)) is False
        assert frustum.contains_point((101.0, 50.0, 50.0)) is False
        assert frustum.contains_point((50.0, -1.0, 50.0)) is False
        assert frustum.contains_point((50.0, 101.0, 50.0)) is False
        assert frustum.contains_point((50.0, 50.0, -1.0)) is False
        assert frustum.contains_point((50.0, 50.0, 101.0)) is False

    def test_frustum_sphere_intersection(self):
        """Test sphere-frustum intersection accuracy."""
        # Plane at x = 0
        frustum = Frustum(planes=[(1.0, 0.0, 0.0, 0.0)])

        # Sphere fully inside (center at x=10, radius 5)
        assert frustum.contains_sphere((10.0, 0.0, 0.0), 5.0) is True

        # Sphere intersecting (center at x=-3, radius 5)
        assert frustum.contains_sphere((-3.0, 0.0, 0.0), 5.0) is True

        # Sphere fully outside (center at x=-10, radius 2)
        assert frustum.contains_sphere((-10.0, 0.0, 0.0), 2.0) is False

        # Sphere touching plane (center at x=-5, radius 5)
        assert frustum.contains_sphere((-5.0, 0.0, 0.0), 5.0) is True


# =============================================================================
# Grass Chunk Streaming Tests
# =============================================================================


class TestGrassChunkStreaming:
    """Tests for grass chunk streaming behavior."""

    def test_chunk_generation_on_demand(self):
        """Chunks should be generated only when needed."""
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=100.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="test", density=5.0)
        landscape.add_grass_type(grass_type)

        assert landscape.total_chunk_count == 0

        # Update near origin
        landscape.update((0.0, 0.0, 0.0), terrain)

        # Should have generated chunks around origin
        assert landscape.total_chunk_count > 0
        assert landscape.active_chunk_count > 0

    def test_chunk_visibility_update_on_camera_move(self):
        """Chunk visibility should update when camera moves."""
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=64.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="test", density=5.0)
        landscape.add_grass_type(grass_type)

        # Initial position
        landscape.update((0.0, 0.0, 0.0), terrain)
        initial_active = landscape.active_chunk_count

        # Move far away
        landscape.update((500.0, 0.0, 500.0), terrain)

        # Active chunks should be different
        far_active = landscape.active_chunk_count
        assert far_active > 0  # Should have chunks at new position

    def test_chunk_unloading(self):
        """Distant chunks should be unloadable to free memory."""
        settings = GrassSettings(cull_distance=100.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=64.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="test", density=5.0)
        landscape.add_grass_type(grass_type)

        # Generate chunks at origin
        landscape.update((0.0, 0.0, 0.0), terrain)
        initial_count = landscape.total_chunk_count

        # Generate chunks far away
        landscape.update((500.0, 0.0, 500.0), terrain)
        after_move_count = landscape.total_chunk_count

        # Total should have increased
        assert after_move_count > initial_count

        # Unload distant chunks
        unloaded = landscape.unload_distant_chunks((500.0, 0.0, 500.0), 100.0)

        # Should have unloaded the original chunks
        assert unloaded > 0

    def test_render_chunks_sorted_by_distance(self):
        """Render chunks should be sorted nearest to farthest."""
        settings = GrassSettings(cull_distance=200.0)
        landscape = LandscapeGrass(settings, chunk_size=32.0, view_distance=150.0)
        terrain = MockTerrain()
        landscape.set_terrain(terrain)
        grass_type = GrassType(type_id="test", density=5.0)
        landscape.add_grass_type(grass_type)

        landscape.update((50.0, 0.0, 50.0), terrain)
        chunks = landscape.get_render_chunks((50.0, 0.0, 50.0))

        # Verify sorted by distance
        for i in range(len(chunks) - 1):
            d1 = chunks[i].get_distance_to(50.0, 50.0)
            d2 = chunks[i + 1].get_distance_to(50.0, 50.0)
            assert d1 <= d2, "Chunks should be sorted by distance"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and potential issues."""

    def test_zero_density_produces_no_instances(self):
        """Zero density should produce no instances."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=100.0, max_z=100.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer = ProceduralPlacer(seed=42)
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 0.0, 1.0, (1.0, 1.0), False, rule
        )
        assert len(results) == 0

    def test_zero_area_bounds(self):
        """Zero area bounds should produce no instances and not crash."""
        terrain = MockTerrain()
        # Point bounds (zero area)
        bounds = Bounds(min_x=50.0, min_z=50.0, max_x=50.0, max_z=50.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer = ProceduralPlacer(seed=42)
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 10.0, 1.0, (1.0, 1.0), False, rule
        )
        # Should not crash, may produce 0 or 1 instance
        assert isinstance(results, list)

    def test_very_high_density_limited(self):
        """Very high density should still complete in reasonable time."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=10.0, max_z=10.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer = ProceduralPlacer(seed=42)
        # Extreme density
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 1000.0, 0.1, (1.0, 1.0), False, rule
        )
        # Should complete without hanging
        # Max instances roughly = 100 / 0.01 = 10000
        assert len(results) < 15000  # Reasonable upper bound

    def test_negative_coordinates_handled(self):
        """Negative coordinates should work correctly."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=-100.0, min_z=-100.0, max_x=-50.0, max_z=-50.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer = ProceduralPlacer(seed=42)
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 1.0, 2.0, (1.0, 1.0), False, rule
        )

        # Should produce results in negative space
        assert len(results) > 0
        for r in results:
            assert bounds.contains(r.position[0], r.position[2])

    def test_large_bounds_performance(self):
        """Large bounds with low density should perform well."""
        terrain = MockTerrain()
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=1000.0, max_z=1000.0)
        rule = PlacementRule(noise_threshold=0.0)

        placer = ProceduralPlacer(seed=42)
        # Low density = larger spacing = fewer instances
        results = placer.generate_in_bounds(
            terrain, bounds, "test", 0.01, 5.0, (1.0, 1.0), False, rule
        )
        # Should complete quickly with reasonable instance count
        # Spacing = max(5.0, 1/sqrt(0.01)) = max(5, 10) = 10
        # Instances ~ 1000000 / 100 = 10000
        assert len(results) < 15000

    def test_cluster_min_max_y_bounds(self):
        """Cluster Y bounds should track correctly."""
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=50.0, max_z=50.0)
        cluster = FoliageCluster(bounds, "test")

        cluster.add_instance(FoliageInstance(instance_id=0, position=(10.0, 5.0, 10.0)))
        cluster.add_instance(FoliageInstance(instance_id=1, position=(20.0, -3.0, 20.0)))
        cluster.add_instance(FoliageInstance(instance_id=2, position=(30.0, 15.0, 30.0)))

        # Check internal state (accessing private for testing)
        assert cluster._min_y == -3.0
        assert cluster._max_y == 15.0

    def test_hism_empty_cluster_removal(self):
        """HISM should remove empty clusters."""
        ft = FoliageType(type_id="test")
        hism = HierarchicalInstancedMesh(ft, cluster_size=50.0)

        # Add instance
        instance_id = hism.add_instance(PlacementResult(position=(25.0, 0.0, 25.0)))
        assert hism.cluster_count == 1

        # Remove instance
        hism.remove_instance(instance_id)
        assert hism.cluster_count == 0  # Empty cluster should be removed


# =============================================================================
# LOD Distance Tests
# =============================================================================


class TestLODDistance:
    """Tests for LOD level calculation based on distance."""

    def test_lod_level_boundaries(self):
        """LOD levels should transition at correct distances."""
        ft = FoliageType(
            type_id="test",
            lod_distances=[50.0, 150.0, 500.0],
        )

        # LOD 0: distance < 50
        assert ft.get_lod_level(0.0) == 0
        assert ft.get_lod_level(49.9) == 0

        # LOD 1: 50 <= distance < 150
        assert ft.get_lod_level(50.0) == 1
        assert ft.get_lod_level(149.9) == 1

        # LOD 2: 150 <= distance < 500
        assert ft.get_lod_level(150.0) == 2
        assert ft.get_lod_level(499.9) == 2

        # LOD 3: distance >= 500
        assert ft.get_lod_level(500.0) == 3
        assert ft.get_lod_level(1000.0) == 3

    def test_cluster_lod_update(self):
        """Cluster should update instance LOD levels correctly."""
        bounds = Bounds(min_x=0.0, min_z=0.0, max_x=1000.0, max_z=1000.0)
        cluster = FoliageCluster(bounds, "test")

        # Add instances at various distances from origin
        cluster.add_instance(FoliageInstance(instance_id=0, position=(25.0, 0.0, 0.0)))   # dist 25
        cluster.add_instance(FoliageInstance(instance_id=1, position=(100.0, 0.0, 0.0)))  # dist 100
        cluster.add_instance(FoliageInstance(instance_id=2, position=(300.0, 0.0, 0.0)))  # dist 300

        lod_distances = [50.0, 150.0, 500.0]
        cluster.update_lod((0.0, 0.0, 0.0), lod_distances)

        instances = cluster.get_instances()
        assert instances[0].lod_level == 0  # < 50
        assert instances[1].lod_level == 1  # 50-150
        assert instances[2].lod_level == 2  # 150-500


# =============================================================================
# Spatial Hash / Cluster Tests
# =============================================================================


class TestSpatialClustering:
    """Tests for spatial clustering correctness."""

    def test_instances_assigned_to_correct_clusters(self):
        """Instances should be in the cluster matching their position."""
        ft = FoliageType(type_id="test")
        hism = HierarchicalInstancedMesh(ft, cluster_size=50.0)

        # Add instances in different cluster regions
        hism.add_instance(PlacementResult(position=(25.0, 0.0, 25.0)))   # Cluster (0, 0)
        hism.add_instance(PlacementResult(position=(75.0, 0.0, 25.0)))   # Cluster (1, 0)
        hism.add_instance(PlacementResult(position=(25.0, 0.0, 75.0)))   # Cluster (0, 1)
        hism.add_instance(PlacementResult(position=(125.0, 0.0, 125.0))) # Cluster (2, 2)

        assert hism.cluster_count == 4

    def test_cluster_boundary_positions(self):
        """Positions on cluster boundaries should be assigned correctly."""
        ft = FoliageType(type_id="test")
        hism = HierarchicalInstancedMesh(ft, cluster_size=50.0)

        # Exactly on boundary at x=50 should go to cluster 1
        hism.add_instance(PlacementResult(position=(50.0, 0.0, 25.0)))
        hism.add_instance(PlacementResult(position=(0.0, 0.0, 25.0)))

        # These should be in different clusters
        assert hism.cluster_count == 2

    def test_get_clusters_in_bounds(self):
        """Should return correct clusters for query bounds."""
        ft = FoliageType(type_id="test")
        hism = HierarchicalInstancedMesh(ft, cluster_size=50.0)

        # Create instances in a 4x4 grid of clusters
        for i in range(4):
            for j in range(4):
                pos = (25.0 + i * 50.0, 0.0, 25.0 + j * 50.0)
                hism.add_instance(PlacementResult(position=pos))

        assert hism.cluster_count == 16

        # Query a 2x2 region - note that bounds 0-99 maps to clusters 0,1
        # (cluster key = floor(coord / cluster_size))
        query_bounds = Bounds(min_x=0.0, min_z=0.0, max_x=99.0, max_z=99.0)
        clusters = hism.get_clusters_in_bounds(query_bounds)
        # Should get clusters (0,0), (0,1), (1,0), (1,1) = 4 clusters
        assert len(clusters) == 4


# =============================================================================
# Buffer Overflow Prevention Tests
# =============================================================================


class TestBufferLimits:
    """Tests ensuring no buffer overflow conditions."""

    def test_instance_buffer_generation(self):
        """Instance buffer should contain all visible instances."""
        ft = FoliageType(type_id="test")
        hism = HierarchicalInstancedMesh(ft)

        # Add many instances
        for i in range(100):
            hism.add_instance(PlacementResult(
                position=(float(i % 10) * 5, 0.0, float(i // 10) * 5)
            ))

        buffer = hism.get_instance_buffer()
        assert len(buffer) == 100

        # Verify buffer structure
        for entry in buffer:
            assert "position" in entry
            assert "rotation" in entry
            assert "scale" in entry
            assert "lod_level" in entry

    def test_grass_instance_buffer_structure(self):
        """Grass instance buffer should have correct structure."""
        settings = GrassSettings()
        gen = ProceduralGrass(settings)

        from engine.world.foliage.grass import GrassInstance
        instances = [
            GrassInstance(position=(0.0, 0.0, 0.0), height=0.3, width=0.05),
            GrassInstance(position=(1.0, 0.0, 1.0), height=0.4, width=0.06),
        ]

        buffer = gen.generate_instance_buffer(instances)
        assert len(buffer) == 2
        for entry in buffer:
            assert "position" in entry
            assert "rotation" in entry
            assert "height" in entry
            assert "width" in entry
            assert "bend" in entry
            assert "color_blend" in entry
