"""
Tests for PCG scatter placement.

Tests cover:
- Poisson disk minimum spacing guarantee
- Grid regularity
- Cluster grouping
- Bounds containment
- Determinism
"""

import math
import pytest

from engine.world.pcg.scatter import (
    ScatterPattern,
    ScatterSettings,
    ScatterPoint,
    Bounds,
    DeterministicRandom,
    RandomScatter,
    PoissonDiskScatter,
    GridScatter,
    JitteredGridScatter,
    ClusteredScatter,
    OrganicScatter,
    ScatterSystem,
)


class TestScatterSettings:
    """Tests for ScatterSettings dataclass."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = ScatterSettings()
        assert settings.pattern == ScatterPattern.POISSON_DISK
        assert settings.density == 1.0
        assert settings.min_spacing == 1.0
        assert settings.seed == 0
        assert settings.jitter == 0.0
        assert settings.cluster_size == 5
        assert settings.cluster_radius == 10.0

    def test_custom_settings(self):
        """Test custom settings values."""
        settings = ScatterSettings(
            pattern=ScatterPattern.CLUSTERED,
            density=2.5,
            min_spacing=5.0,
            seed=42,
            jitter=0.3,
            cluster_size=10,
            cluster_radius=20.0,
        )
        assert settings.pattern == ScatterPattern.CLUSTERED
        assert settings.density == 2.5
        assert settings.min_spacing == 5.0

    def test_invalid_density(self):
        """Test validation of density."""
        with pytest.raises(ValueError, match="density must be > 0"):
            ScatterSettings(density=0)

        with pytest.raises(ValueError, match="density must be > 0"):
            ScatterSettings(density=-1)

    def test_invalid_min_spacing(self):
        """Test validation of min_spacing."""
        with pytest.raises(ValueError, match="min_spacing must be > 0"):
            ScatterSettings(min_spacing=0)

    def test_invalid_jitter(self):
        """Test validation of jitter."""
        with pytest.raises(ValueError, match="jitter must be >= 0"):
            ScatterSettings(jitter=-0.1)

    def test_invalid_cluster_size(self):
        """Test validation of cluster_size."""
        with pytest.raises(ValueError, match="cluster_size must be >= 1"):
            ScatterSettings(cluster_size=0)

    def test_invalid_cluster_radius(self):
        """Test validation of cluster_radius."""
        with pytest.raises(ValueError, match="cluster_radius must be > 0"):
            ScatterSettings(cluster_radius=0)


class TestScatterPoint:
    """Tests for ScatterPoint dataclass."""

    def test_creation(self):
        """Test basic creation."""
        point = ScatterPoint(position=(10.5, 20.3))
        assert point.position == (10.5, 20.3)
        assert point.weight == 1.0

    def test_with_weight(self):
        """Test creation with weight."""
        point = ScatterPoint(position=(10.5, 20.3), weight=0.75)
        assert point.weight == 0.75

    def test_x_y_properties(self):
        """Test x and y property accessors."""
        point = ScatterPoint(position=(10.5, 20.3))
        assert point.x == 10.5
        assert point.y == 20.3


class TestBounds:
    """Tests for Bounds dataclass."""

    def test_default_bounds(self):
        """Test default bounds values."""
        bounds = Bounds()
        assert bounds.min_x == 0.0
        assert bounds.min_y == 0.0
        assert bounds.max_x == 100.0
        assert bounds.max_y == 100.0

    def test_custom_bounds(self):
        """Test custom bounds values."""
        bounds = Bounds(min_x=10, min_y=20, max_x=50, max_y=60)
        assert bounds.width == 40
        assert bounds.height == 40
        assert bounds.area == 1600

    def test_invalid_bounds(self):
        """Test validation of bounds."""
        with pytest.raises(ValueError, match="min_x.*must be < max_x"):
            Bounds(min_x=100, max_x=50)

        with pytest.raises(ValueError, match="min_y.*must be < max_y"):
            Bounds(min_y=100, max_y=50)

    def test_contains(self):
        """Test point containment check."""
        bounds = Bounds(min_x=0, min_y=0, max_x=100, max_y=100)

        assert bounds.contains(50, 50)
        assert bounds.contains(0, 0)
        assert bounds.contains(100, 100)
        assert not bounds.contains(-1, 50)
        assert not bounds.contains(101, 50)


class TestDeterministicRandom:
    """Tests for DeterministicRandom class."""

    def test_determinism(self):
        """Test that same seed produces same sequence."""
        rng1 = DeterministicRandom(42)
        rng2 = DeterministicRandom(42)

        for _ in range(100):
            assert rng1.next_int(0, 100) == rng2.next_int(0, 100)

    def test_different_seeds_differ(self):
        """Test that different seeds produce different sequences."""
        rng1 = DeterministicRandom(42)
        rng2 = DeterministicRandom(43)

        differences = sum(
            1 for _ in range(100)
            if rng1.next_int(0, 1000) != rng2.next_int(0, 1000)
        )
        assert differences > 90

    def test_next_int_range(self):
        """Test integer range."""
        rng = DeterministicRandom(42)
        for _ in range(100):
            value = rng.next_int(10, 20)
            assert 10 <= value <= 20

    def test_next_float_range(self):
        """Test float range."""
        rng = DeterministicRandom(42)
        for _ in range(100):
            value = rng.next_float(0.5, 1.5)
            assert 0.5 <= value <= 1.5

    def test_next_point_in_bounds(self):
        """Test point generation within bounds."""
        rng = DeterministicRandom(42)
        bounds = Bounds(10, 20, 50, 60)

        for _ in range(100):
            x, y = rng.next_point_in_bounds(bounds)
            assert bounds.contains(x, y)

    def test_next_point_in_circle(self):
        """Test point generation within circle."""
        rng = DeterministicRandom(42)
        radius = 5.0

        for _ in range(100):
            x, y = rng.next_point_in_circle(radius)
            dist = math.sqrt(x * x + y * y)
            assert dist <= radius

    def test_next_point_in_annulus(self):
        """Test point generation within annulus."""
        rng = DeterministicRandom(42)
        inner = 2.0
        outer = 5.0

        for _ in range(100):
            x, y = rng.next_point_in_annulus(inner, outer)
            dist = math.sqrt(x * x + y * y)
            assert inner <= dist <= outer

    def test_shuffle_determinism(self):
        """Test that shuffle is deterministic."""
        items = list(range(10))

        rng1 = DeterministicRandom(42)
        rng2 = DeterministicRandom(42)

        shuffled1 = rng1.shuffle(items)
        shuffled2 = rng2.shuffle(items)

        assert shuffled1 == shuffled2

    def test_shuffle_completeness(self):
        """Test that shuffle contains all items."""
        rng = DeterministicRandom(42)
        items = list(range(10))
        shuffled = rng.shuffle(items)

        assert sorted(shuffled) == items

    def test_reset(self):
        """Test resetting the generator."""
        rng = DeterministicRandom(42)

        values1 = [rng.next_int(0, 100) for _ in range(10)]

        rng.reset()

        values2 = [rng.next_int(0, 100) for _ in range(10)]

        assert values1 == values2


class TestRandomScatter:
    """Tests for RandomScatter generator."""

    def test_creation(self):
        """Test basic creation."""
        settings = ScatterSettings(seed=42)
        scatter = RandomScatter(settings)
        assert scatter.settings == settings

    def test_generate_count(self):
        """Test that correct number of points generated."""
        scatter = RandomScatter(ScatterSettings(seed=42))
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds, count=50)
        assert len(points) == 50

    def test_generate_density(self):
        """Test density-based generation."""
        settings = ScatterSettings(seed=42, density=0.01)  # 0.01 per unit
        scatter = RandomScatter(settings)
        bounds = Bounds(0, 0, 100, 100)  # 10000 units

        points = scatter.generate(bounds)
        # Should be around 100 points
        assert 50 <= len(points) <= 150

    def test_bounds_containment(self):
        """Test that all points are within bounds."""
        scatter = RandomScatter(ScatterSettings(seed=42))
        bounds = Bounds(10, 20, 50, 60)

        points = scatter.generate(bounds, count=100)

        for point in points:
            assert bounds.contains(point.x, point.y), f"Point {point} outside bounds"

    def test_determinism(self):
        """Test that same seed produces same points."""
        scatter1 = RandomScatter(ScatterSettings(seed=42))
        scatter2 = RandomScatter(ScatterSettings(seed=42))
        bounds = Bounds(0, 0, 100, 100)

        points1 = scatter1.generate(bounds, count=50)
        points2 = scatter2.generate(bounds, count=50)

        for p1, p2 in zip(points1, points2):
            assert p1.position == p2.position


class TestPoissonDiskScatter:
    """Tests for PoissonDiskScatter generator."""

    def test_creation(self):
        """Test basic creation."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter = PoissonDiskScatter(settings)
        assert scatter.min_distance == 5.0

    def test_minimum_spacing_guarantee(self):
        """Test that minimum spacing is guaranteed."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter = PoissonDiskScatter(settings)
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)
        min_dist_sq = settings.min_spacing ** 2

        # Check all pairs
        for i, p1 in enumerate(points):
            for p2 in points[i + 1:]:
                dist_sq = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                assert dist_sq >= min_dist_sq * 0.99, f"Points too close: {p1}, {p2}"

    def test_bounds_containment(self):
        """Test that all points are within bounds."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter = PoissonDiskScatter(settings)
        bounds = Bounds(10, 20, 90, 80)

        points = scatter.generate(bounds)

        for point in points:
            assert bounds.contains(point.x, point.y)

    def test_determinism(self):
        """Test that same seed produces same points."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter1 = PoissonDiskScatter(settings)
        scatter2 = PoissonDiskScatter(ScatterSettings(seed=42, min_spacing=5.0))
        bounds = Bounds(0, 0, 50, 50)

        points1 = scatter1.generate(bounds)
        points2 = scatter2.generate(bounds)

        assert len(points1) == len(points2)
        for p1, p2 in zip(points1, points2):
            assert p1.position == p2.position

    def test_fills_space(self):
        """Test that space is reasonably filled."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter = PoissonDiskScatter(settings)
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)

        # With 5.0 spacing in 100x100, expect roughly (100/5)^2 * 0.7 = ~280 points
        # (Poisson disk typically achieves ~70% of grid density)
        assert len(points) >= 100, f"Too few points: {len(points)}"

    def test_different_max_attempts(self):
        """Test that max_attempts parameter works."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        scatter_low = PoissonDiskScatter(settings, max_attempts=5)
        scatter_high = PoissonDiskScatter(settings, max_attempts=50)

        assert scatter_low.max_attempts == 5
        assert scatter_high.max_attempts == 50


class TestGridScatter:
    """Tests for GridScatter generator."""

    def test_creation(self):
        """Test basic creation."""
        settings = ScatterSettings(seed=42, min_spacing=10.0)
        scatter = GridScatter(settings)
        assert scatter.spacing == 10.0

    def test_grid_regularity(self):
        """Test that points form a regular grid."""
        scatter = GridScatter(ScatterSettings(seed=42), spacing=10.0)
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)

        # Extract unique x and y coordinates
        x_coords = sorted(set(p.x for p in points))
        y_coords = sorted(set(p.y for p in points))

        # Check regularity of x spacing
        for i in range(1, len(x_coords)):
            spacing = x_coords[i] - x_coords[i - 1]
            assert spacing == pytest.approx(10.0)

        # Check regularity of y spacing
        for i in range(1, len(y_coords)):
            spacing = y_coords[i] - y_coords[i - 1]
            assert spacing == pytest.approx(10.0)

    def test_bounds_containment(self):
        """Test that all points are within bounds."""
        scatter = GridScatter(ScatterSettings(seed=42), spacing=10.0)
        bounds = Bounds(10, 20, 90, 80)

        points = scatter.generate(bounds)

        for point in points:
            assert bounds.contains(point.x, point.y)

    def test_point_count(self):
        """Test expected number of points."""
        scatter = GridScatter(ScatterSettings(seed=42), spacing=10.0)
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)

        # With 10 spacing in 100x100, expect 10x10 = 100 points
        assert len(points) == 100


class TestJitteredGridScatter:
    """Tests for JitteredGridScatter generator."""

    def test_creation(self):
        """Test basic creation."""
        scatter = JitteredGridScatter(ScatterSettings(seed=42), spacing=10.0, jitter=0.3)
        assert scatter.spacing == 10.0
        assert scatter.jitter == 0.3

    def test_invalid_jitter(self):
        """Test validation of jitter parameter."""
        with pytest.raises(ValueError, match="jitter must be in"):
            JitteredGridScatter(ScatterSettings(seed=42), jitter=0.6)

    def test_jitter_offset(self):
        """Test that points are offset from grid but nearby."""
        scatter_grid = GridScatter(ScatterSettings(seed=42), spacing=10.0)
        scatter_jitter = JitteredGridScatter(
            ScatterSettings(seed=42), spacing=10.0, jitter=0.3
        )
        bounds = Bounds(0, 0, 100, 100)

        grid_points = scatter_grid.generate(bounds)
        jitter_points = scatter_jitter.generate(bounds)

        # Same count
        assert len(grid_points) == len(jitter_points)

        # Jittered points should be within jitter distance of grid
        max_offset = 10.0 * 0.3  # spacing * jitter
        for gp, jp in zip(grid_points, jitter_points):
            dx = abs(gp.x - jp.x)
            dy = abs(gp.y - jp.y)
            # Allow some tolerance for bounds clamping
            assert dx <= max_offset + 0.1
            assert dy <= max_offset + 0.1

    def test_determinism(self):
        """Test that same seed produces same points."""
        settings = ScatterSettings(seed=42)
        scatter1 = JitteredGridScatter(settings, spacing=10.0, jitter=0.3)
        scatter2 = JitteredGridScatter(
            ScatterSettings(seed=42), spacing=10.0, jitter=0.3
        )
        bounds = Bounds(0, 0, 50, 50)

        points1 = scatter1.generate(bounds)
        points2 = scatter2.generate(bounds)

        for p1, p2 in zip(points1, points2):
            assert p1.position == p2.position


class TestClusteredScatter:
    """Tests for ClusteredScatter generator."""

    def test_creation(self):
        """Test basic creation."""
        scatter = ClusteredScatter(
            ScatterSettings(seed=42),
            cluster_count=5,
            points_per_cluster=10,
            cluster_radius=15.0,
        )
        assert scatter.cluster_count == 5
        assert scatter.points_per_cluster == 10
        assert scatter.cluster_radius == 15.0

    def test_cluster_grouping(self):
        """Test that points form clusters."""
        scatter = ClusteredScatter(
            ScatterSettings(seed=42),
            cluster_count=3,
            points_per_cluster=20,
            cluster_radius=5.0,
        )
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)

        # Points should be grouped - most points should have neighbors
        # within cluster_radius
        points_with_neighbors = 0
        for p1 in points:
            has_neighbor = False
            for p2 in points:
                if p1 is p2:
                    continue
                dist = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
                if dist <= 10.0:  # Double cluster radius
                    has_neighbor = True
                    break
            if has_neighbor:
                points_with_neighbors += 1

        # Most points should have neighbors
        assert points_with_neighbors >= len(points) * 0.8

    def test_total_point_count(self):
        """Test total number of points."""
        scatter = ClusteredScatter(
            ScatterSettings(seed=42),
            cluster_count=5,
            points_per_cluster=10,
            cluster_radius=10.0,
        )
        bounds = Bounds(0, 0, 100, 100)

        points = scatter.generate(bounds)
        assert len(points) == 50  # 5 clusters * 10 points

    def test_bounds_containment(self):
        """Test that all points are within bounds."""
        scatter = ClusteredScatter(
            ScatterSettings(seed=42),
            cluster_count=5,
            points_per_cluster=10,
            cluster_radius=10.0,
        )
        bounds = Bounds(10, 10, 90, 90)

        points = scatter.generate(bounds)

        for point in points:
            assert bounds.contains(point.x, point.y)


class TestOrganicScatter:
    """Tests for OrganicScatter generator."""

    def test_creation(self):
        """Test basic creation."""
        scatter = OrganicScatter(
            ScatterSettings(seed=42, min_spacing=5.0),
            noise_threshold=0.3,
        )
        assert scatter.noise_threshold == 0.3

    def test_without_noise_map(self):
        """Test generation without noise map."""
        scatter = OrganicScatter(ScatterSettings(seed=42, min_spacing=5.0))
        bounds = Bounds(0, 0, 50, 50)

        points = scatter.generate(bounds)

        # Should generate some points
        assert len(points) > 0

        # Should still maintain minimum spacing
        min_dist_sq = 5.0 ** 2
        for i, p1 in enumerate(points):
            for p2 in points[i + 1:]:
                dist_sq = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                assert dist_sq >= min_dist_sq * 0.99


class TestScatterSystem:
    """Tests for ScatterSystem class."""

    def test_creation(self):
        """Test basic creation."""
        system = ScatterSystem()
        assert system.settings is not None

    def test_generate_default(self):
        """Test generation with default pattern."""
        settings = ScatterSettings(seed=42, min_spacing=5.0)
        system = ScatterSystem(settings)
        bounds = Bounds(0, 0, 50, 50)

        points = system.generate(bounds)
        assert len(points) > 0

    def test_generate_specific_pattern(self):
        """Test generation with specific pattern."""
        system = ScatterSystem(ScatterSettings(seed=42))
        bounds = Bounds(0, 0, 50, 50)

        points = system.generate(bounds, pattern=ScatterPattern.GRID)

        # Grid pattern should have regular spacing
        x_coords = sorted(set(p.x for p in points))
        for i in range(1, len(x_coords)):
            spacing = x_coords[i] - x_coords[i - 1]
            assert spacing == pytest.approx(1.0)  # Default min_spacing

    def test_filter_points(self):
        """Test point filtering."""
        system = ScatterSystem(ScatterSettings(seed=42, min_spacing=5.0))
        bounds = Bounds(0, 0, 50, 50)

        points = system.generate(bounds)

        # Filter to top half
        filtered = system.filter_points(points, lambda p: p.y > 25)

        for point in filtered:
            assert point.y > 25

    def test_filter_by_bounds(self):
        """Test filtering by bounds."""
        system = ScatterSystem(ScatterSettings(seed=42, min_spacing=5.0))
        bounds = Bounds(0, 0, 100, 100)

        points = system.generate(bounds)

        # Filter to smaller bounds
        small_bounds = Bounds(25, 25, 75, 75)
        filtered = system.filter_by_bounds(points, small_bounds)

        for point in filtered:
            assert small_bounds.contains(point.x, point.y)

    def test_filter_by_weight(self):
        """Test filtering by weight."""
        points = [
            ScatterPoint((0, 0), weight=0.2),
            ScatterPoint((1, 0), weight=0.5),
            ScatterPoint((2, 0), weight=0.8),
            ScatterPoint((3, 0), weight=1.0),
        ]

        system = ScatterSystem()
        filtered = system.filter_by_weight(points, 0.4, 0.9)

        assert len(filtered) == 2
        for point in filtered:
            assert 0.4 <= point.weight <= 0.9

    def test_merge_points(self):
        """Test merging multiple point lists."""
        system = ScatterSystem()

        points1 = [ScatterPoint((0, 0)), ScatterPoint((1, 0))]
        points2 = [ScatterPoint((2, 0)), ScatterPoint((3, 0))]
        points3 = [ScatterPoint((4, 0))]

        merged = system.merge_points(points1, points2, points3)
        assert len(merged) == 5

    def test_remove_overlapping(self):
        """Test removing overlapping points."""
        system = ScatterSystem()

        points = [
            ScatterPoint((0, 0)),
            ScatterPoint((0.5, 0)),  # Too close to first
            ScatterPoint((5, 0)),
            ScatterPoint((5.5, 0)),  # Too close to third
            ScatterPoint((10, 0)),
        ]

        result = system.remove_overlapping(points, min_distance=2.0)

        # Should keep first, third, and fifth
        assert len(result) == 3
        assert result[0].position == (0, 0)
        assert result[1].position == (5, 0)
        assert result[2].position == (10, 0)
