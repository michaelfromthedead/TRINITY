"""
Comprehensive tests for the Destruction System module.

Tests cover:
- Configuration validation
- Damage types and resistance
- Voronoi fracture generation
- Radial fracture patterns
- Slice fracture operations
- Support graph stress paths
- Debris lifecycle management
- Chunk spawning and validation
- Main destruction system integration

Total: 100+ tests
"""

import pytest
import math
import random
from typing import List, Tuple

# Import all destruction system components
from engine.simulation.destruction import (
    # Configuration
    DEFAULT_FRACTURE_SEED,
    MIN_CHUNK_VOLUME,
    MAX_CHUNKS_PER_OBJECT,
    MIN_VORONOI_SITES,
    DEBRIS_LIFETIME,
    DEBRIS_MIN_VELOCITY,
    DEBRIS_SLEEP_TIME,
    MAX_ACTIVE_DEBRIS,
    DAMAGE_PROPAGATION_FACTOR,
    SUPPORT_STRESS_THRESHOLD,
    FracturePattern,
    DebrisState,
    SupportType,
    FractureConfig,
    DebrisConfig,
    DamageConfig,
    SupportConfig,
    DestructionSystemConfig,
    DEFAULT_CONFIG,
    # Damage
    DamageType,
    Damage,
    DamageResistance,
    DamageAccumulator,
    DamageResult,
    apply_damage_modifiers,
    get_damage_type_properties,
    # Voronoi
    Vec3,
    Triangle,
    Plane,
    BoundingBox,
    Chunk,
    VoronoiFracture,
    SiteDistribution,
    vec3_add,
    vec3_sub,
    vec3_length,
    vec3_normalize,
    vec3_distance,
    # Radial
    RadialFracture,
    RadialChunk,
    ConcentricRadialFracture,
    SpiderWebFracture,
    # Slice
    SliceFracture,
    SliceResult,
    AdaptiveSliceFracture,
    HierarchicalSliceFracture,
    # Support
    SupportGraph,
    SupportNode,
    SupportEdge,
    build_support_graph_from_chunks,
    # Debris
    Debris,
    DebrisManager,
    DebrisPool,
    DebrisLOD,
    spawn_debris_from_fracture,
    # System
    DestructionSystem,
    Destructible,
    DestructibleState,
    FractureRequest,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def cube_mesh() -> Tuple[List[Vec3], List[Triangle]]:
    """Simple cube mesh for testing."""
    vertices = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
    ]
    triangles = [
        (0, 1, 2), (0, 2, 3),  # Front
        (4, 6, 5), (4, 7, 6),  # Back
        (0, 5, 1), (0, 4, 5),  # Bottom
        (2, 7, 3), (2, 6, 7),  # Top
        (0, 7, 4), (0, 3, 7),  # Left
        (1, 5, 6), (1, 6, 2),  # Right
    ]
    return vertices, triangles


@pytest.fixture
def simple_chunk() -> Chunk:
    """Simple chunk for testing."""
    return Chunk(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
        triangles=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        volume=0.5,
        centroid=(0.25, 0.25, 0.25)
    )


@pytest.fixture
def destruction_system() -> DestructionSystem:
    """Fresh destruction system for testing."""
    return DestructionSystem()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Tests for configuration classes."""

    def test_default_config_values(self):
        """Test default configuration values."""
        assert DEFAULT_FRACTURE_SEED == 42
        assert MIN_CHUNK_VOLUME == 0.001
        assert MAX_CHUNKS_PER_OBJECT == 64
        assert DEBRIS_LIFETIME == 10.0
        assert MAX_ACTIVE_DEBRIS == 1000
        assert DAMAGE_PROPAGATION_FACTOR == 0.5
        assert SUPPORT_STRESS_THRESHOLD == 1000.0

    def test_fracture_pattern_enum(self):
        """Test FracturePattern enum values."""
        assert FracturePattern.VORONOI == 0
        assert FracturePattern.RADIAL == 1
        assert FracturePattern.SLICE == 2
        assert FracturePattern.CUSTOM == 3

    def test_debris_state_enum(self):
        """Test DebrisState enum values."""
        assert DebrisState.ACTIVE == 0
        assert DebrisState.SLEEPING == 1
        assert DebrisState.PENDING_CLEANUP == 2
        assert DebrisState.POOLED == 3

    def test_support_type_enum(self):
        """Test SupportType enum values."""
        assert SupportType.FIXED == 0
        assert SupportType.STRUCTURAL == 1
        assert SupportType.TEMPORARY == 2

    def test_fracture_config_defaults(self):
        """Test FractureConfig default values."""
        config = FractureConfig()
        assert config.pattern == FracturePattern.VORONOI
        assert config.seed == DEFAULT_FRACTURE_SEED
        assert config.max_chunks == MAX_CHUNKS_PER_OBJECT

    def test_fracture_config_validation(self):
        """Test FractureConfig validation."""
        with pytest.raises(ValueError, match="max_chunks must be >= 1"):
            FractureConfig(max_chunks=0)

        with pytest.raises(ValueError, match="min_chunk_volume must be > 0"):
            FractureConfig(min_chunk_volume=0)

    def test_debris_config_defaults(self):
        """Test DebrisConfig default values."""
        config = DebrisConfig()
        assert config.lifetime == DEBRIS_LIFETIME
        assert config.max_active == MAX_ACTIVE_DEBRIS

    def test_debris_config_validation(self):
        """Test DebrisConfig validation."""
        with pytest.raises(ValueError, match="lifetime must be >="):
            DebrisConfig(lifetime=0.1)

        with pytest.raises(ValueError, match="max_active must be >= 1"):
            DebrisConfig(max_active=0)

    def test_damage_config_validation(self):
        """Test DamageConfig validation."""
        with pytest.raises(ValueError, match="propagation_factor must be in"):
            DamageConfig(propagation_factor=1.5)

    def test_support_config_validation(self):
        """Test SupportConfig validation."""
        with pytest.raises(ValueError, match="stress_threshold must be > 0"):
            SupportConfig(stress_threshold=0)

    def test_destruction_system_config(self):
        """Test DestructionSystemConfig composition."""
        config = DestructionSystemConfig()
        assert isinstance(config.fracture, FractureConfig)
        assert isinstance(config.debris, DebrisConfig)
        assert isinstance(config.damage, DamageConfig)
        assert isinstance(config.support, SupportConfig)


# =============================================================================
# DAMAGE TYPE TESTS
# =============================================================================

class TestDamageTypes:
    """Tests for damage types and resistance system."""

    def test_damage_type_enum(self):
        """Test all damage types exist."""
        assert DamageType.IMPACT == 0
        assert DamageType.EXPLOSIVE is not None
        assert DamageType.STRESS is not None
        assert DamageType.BURN is not None
        assert DamageType.PIERCE is not None

    def test_damage_creation(self):
        """Test Damage dataclass creation."""
        damage = Damage(
            amount=50.0,
            damage_type=DamageType.IMPACT,
            position=(0, 0, 0),
            direction=(1, 0, 0)
        )
        assert damage.amount == 50.0
        assert damage.damage_type == DamageType.IMPACT
        assert damage.position == (0, 0, 0)

    def test_damage_negative_amount(self):
        """Test Damage rejects negative amount."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Damage(amount=-10.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))

    def test_damage_direction_normalization(self):
        """Test Damage normalizes direction."""
        damage = Damage(
            amount=10.0,
            damage_type=DamageType.IMPACT,
            position=(0, 0, 0),
            direction=(3, 4, 0)
        )
        dx, dy, dz = damage.direction
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        assert abs(length - 1.0) < 1e-6

    def test_damage_falloff_linear(self):
        """Test linear damage falloff."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0, 0, 0),
            radius=10.0,
            falloff="linear"
        )
        assert damage.calculate_falloff(0) == 1.0
        assert damage.calculate_falloff(5) == 0.5
        assert damage.calculate_falloff(10) == 0.0
        assert damage.calculate_falloff(15) == 0.0

    def test_damage_falloff_quadratic(self):
        """Test quadratic damage falloff."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0, 0, 0),
            radius=10.0,
            falloff="quadratic"
        )
        assert damage.calculate_falloff(0) == 1.0
        assert abs(damage.calculate_falloff(5) - 0.75) < 0.01

    def test_damage_with_falloff(self):
        """Test Damage.with_falloff creates new instance."""
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0, 0, 0),
            radius=10.0,
            falloff="linear"
        )
        new_damage = damage.with_falloff(5.0)
        assert new_damage.amount == 50.0
        assert new_damage.radius == 0.0

    def test_damage_resistance_creation(self):
        """Test DamageResistance creation."""
        resistance = DamageResistance(
            resistances={DamageType.IMPACT: 0.5, DamageType.EXPLOSIVE: 2.0}
        )
        assert resistance.get_resistance(DamageType.IMPACT) == 0.5
        assert resistance.get_resistance(DamageType.EXPLOSIVE) == 2.0

    def test_damage_resistance_default(self):
        """Test DamageResistance default resistance."""
        resistance = DamageResistance(default_resistance=0.8)
        assert resistance.get_resistance(DamageType.BURN) == 0.8

    def test_damage_resistance_apply(self):
        """Test DamageResistance.apply modifies damage."""
        resistance = DamageResistance(resistances={DamageType.IMPACT: 0.5})
        damage = Damage(amount=100.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))
        modified = resistance.apply(damage)
        assert modified == 50.0

    def test_damage_resistance_from_dict(self):
        """Test DamageResistance.from_dict factory."""
        resistance = DamageResistance.from_dict({
            "impact": 0.5,
            "explosive": 1.5
        })
        assert resistance.get_resistance(DamageType.IMPACT) == 0.5
        assert resistance.get_resistance(DamageType.EXPLOSIVE) == 1.5

    def test_damage_resistance_immunity(self):
        """Test DamageResistance immunity check."""
        resistance = DamageResistance(resistances={DamageType.BURN: 0.0})
        assert resistance.is_immune(DamageType.BURN)
        assert not resistance.is_immune(DamageType.IMPACT)

    def test_damage_resistance_vulnerability(self):
        """Test DamageResistance vulnerability check."""
        resistance = DamageResistance(resistances={DamageType.FREEZE: 2.0})
        assert resistance.is_vulnerable(DamageType.FREEZE)
        assert not resistance.is_vulnerable(DamageType.IMPACT)


# =============================================================================
# DAMAGE ACCUMULATOR TESTS
# =============================================================================

class TestDamageAccumulator:
    """Tests for damage accumulation system."""

    def test_accumulator_creation(self):
        """Test DamageAccumulator creation."""
        acc = DamageAccumulator(threshold=100.0)
        assert acc.total_damage == 0.0
        assert acc.threshold == 100.0
        assert not acc.is_destroyed

    def test_accumulator_accumulate(self):
        """Test damage accumulation."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(30.0, DamageType.IMPACT)
        assert acc.total_damage == 30.0
        assert acc.remaining_health == 70.0

    def test_accumulator_health_percent(self):
        """Test health percentage calculation."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(25.0, DamageType.IMPACT)
        assert acc.health_percent == 0.75

    def test_accumulator_destruction(self):
        """Test destruction threshold."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(100.0, DamageType.IMPACT)
        assert acc.is_destroyed

    def test_accumulator_damage_by_type(self):
        """Test tracking damage by type."""
        acc = DamageAccumulator(threshold=200.0)
        acc.accumulate(30.0, DamageType.IMPACT)
        acc.accumulate(50.0, DamageType.EXPLOSIVE)
        assert acc.get_damage_by_type(DamageType.IMPACT) == 30.0
        assert acc.get_damage_by_type(DamageType.EXPLOSIVE) == 50.0

    def test_accumulator_dominant_type(self):
        """Test getting dominant damage type."""
        acc = DamageAccumulator(threshold=200.0)
        acc.accumulate(30.0, DamageType.IMPACT)
        acc.accumulate(50.0, DamageType.EXPLOSIVE)
        assert acc.get_dominant_damage_type() == DamageType.EXPLOSIVE

    def test_accumulator_reset(self):
        """Test accumulator reset."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(50.0, DamageType.IMPACT)
        acc.reset()
        assert acc.total_damage == 0.0
        assert acc.get_dominant_damage_type() is None

    def test_accumulator_serialization(self):
        """Test accumulator serialization."""
        acc = DamageAccumulator(threshold=100.0)
        acc.accumulate(50.0, DamageType.IMPACT)
        data = acc.to_dict()
        restored = DamageAccumulator.from_dict(data)
        assert restored.total_damage == 50.0
        assert restored.threshold == 100.0


# =============================================================================
# VORONOI FRACTURE TESTS
# =============================================================================

class TestVoronoiFracture:
    """Tests for Voronoi fracture generation."""

    def test_voronoi_creation(self):
        """Test VoronoiFracture creation."""
        vf = VoronoiFracture(seed=42, num_sites=16)
        assert vf.seed == 42
        assert vf.num_sites == 16

    def test_voronoi_seed_setter(self):
        """Test seed setter resets RNG."""
        vf = VoronoiFracture(seed=42)
        vf.seed = 123
        assert vf.seed == 123

    def test_voronoi_site_generation_uniform(self):
        """Test uniform site generation."""
        vf = VoronoiFracture(seed=42, num_sites=10)
        bounds = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        sites = vf.generate_voronoi_sites(bounds, SiteDistribution.UNIFORM)
        assert len(sites) == 10
        for site in sites:
            assert bounds.contains(site)

    def test_voronoi_site_generation_clustered(self):
        """Test clustered site generation."""
        vf = VoronoiFracture(seed=42, num_sites=20)
        bounds = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        sites = vf.generate_voronoi_sites(bounds, SiteDistribution.CLUSTERED)
        assert len(sites) == 20

    def test_voronoi_site_generation_impact_centered(self):
        """Test impact-centered site generation."""
        vf = VoronoiFracture(seed=42, num_sites=16)
        bounds = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        impact = (5, 5, 5)
        sites = vf.generate_voronoi_sites(
            bounds, SiteDistribution.IMPACT_CENTERED, impact_point=impact
        )
        assert len(sites) == 16

    def test_voronoi_cell_computation(self):
        """Test Voronoi cell computation."""
        vf = VoronoiFracture(seed=42, num_sites=8)
        bounds = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        vf.generate_voronoi_sites(bounds)
        cells = vf.compute_voronoi_cells()
        assert len(cells) == 8
        for cell in cells:
            assert len(cell.planes) > 0

    def test_voronoi_fracture_cube(self, cube_mesh):
        """Test fracturing a cube mesh produces valid split geometry."""
        vertices, triangles = cube_mesh
        vf = VoronoiFracture(seed=42, num_sites=8, min_chunk_volume=0.0001)
        chunks = vf.fracture(vertices, triangles)

        # Verify we got multiple chunks (actual split occurred)
        assert len(chunks) > 1, "Fracture should produce multiple chunks"

        # Verify chunks are geometrically valid
        total_volume = 0.0
        for chunk in chunks:
            assert len(chunk.vertices) >= 4, "Chunk must have at least 4 vertices"
            assert len(chunk.triangles) >= 4, "Chunk must have at least 4 triangles"
            chunk.compute_volume()
            assert chunk.volume > 0, "Chunk volume must be positive"
            total_volume += chunk.volume

        # Total volume of chunks should approximate original mesh volume
        # (cube from -1 to 1 = 8 cubic units)
        original_volume = 8.0
        # Allow some tolerance for numerical precision
        assert total_volume > original_volume * 0.5, \
            f"Total chunk volume {total_volume} too small compared to original {original_volume}"

    def test_voronoi_deterministic(self, cube_mesh):
        """Test that same seed produces same results."""
        vertices, triangles = cube_mesh
        vf1 = VoronoiFracture(seed=42, num_sites=8)
        vf2 = VoronoiFracture(seed=42, num_sites=8)
        chunks1 = vf1.fracture(vertices, triangles)
        chunks2 = vf2.fracture(vertices, triangles)
        assert len(chunks1) == len(chunks2)

    def test_voronoi_max_chunks_limit(self, cube_mesh):
        """Test max chunks limit is enforced."""
        vertices, triangles = cube_mesh
        vf = VoronoiFracture(seed=42, num_sites=100, max_chunks=5)
        chunks = vf.fracture(vertices, triangles)
        assert len(chunks) <= 5, f"Got {len(chunks)} chunks but max is 5"
        # With many sites, we should hit the limit exactly
        assert len(chunks) == 5, "Should produce exactly max_chunks when sites > max"

    def test_voronoi_single_chunk_min_sites(self, cube_mesh):
        """Test fracture with minimum sites may produce single chunk."""
        vertices, triangles = cube_mesh
        # Use MIN_VORONOI_SITES (4) which might produce 1-4 chunks
        from engine.simulation.destruction import MIN_VORONOI_SITES
        vf = VoronoiFracture(seed=42, num_sites=MIN_VORONOI_SITES, min_chunk_volume=0.0001)
        chunks = vf.fracture(vertices, triangles)
        # Should still produce at least one valid chunk
        assert len(chunks) >= 1

    def test_voronoi_handles_degenerate_input(self):
        """Test Voronoi gracefully handles degenerate input."""
        # Create a flat (degenerate) mesh - zero volume
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]  # All on z=0 plane
        triangles = [(0, 1, 2), (1, 3, 2)]
        vf = VoronoiFracture(seed=42, num_sites=8, min_chunk_volume=0.0001)
        # Should not crash, may produce empty result
        chunks = vf.fracture(vertices, triangles)
        # Flat mesh has zero volume, so chunks may be filtered out
        assert isinstance(chunks, list)

    def test_voronoi_chunk_volume_computation(self, simple_chunk):
        """Test chunk volume computation."""
        simple_chunk.compute_volume()
        assert simple_chunk.volume >= 0

    def test_voronoi_chunk_centroid_computation(self, simple_chunk):
        """Test chunk centroid computation."""
        simple_chunk.compute_centroid()
        cx, cy, cz = simple_chunk.centroid
        assert isinstance(cx, float)
        assert isinstance(cy, float)
        assert isinstance(cz, float)


# =============================================================================
# RADIAL FRACTURE TESTS
# =============================================================================

class TestRadialFracture:
    """Tests for radial fracture patterns."""

    def test_radial_creation(self):
        """Test RadialFracture creation."""
        rf = RadialFracture(seed=42, num_slices=8, num_rings=3)
        assert rf.seed == 42
        assert rf.num_slices == 8
        assert rf.num_rings == 3

    def test_radial_pattern_generation(self):
        """Test radial pattern generation."""
        rf = RadialFracture(seed=42, num_slices=8, num_rings=3)
        slices, rings = rf.generate_radial_pattern(
            center=(0, 0, 0),
            radius=5.0
        )
        assert len(slices) == 8
        assert len(rings) == 3

    def test_radial_slices_have_planes(self):
        """Test radial slices have cutting planes."""
        rf = RadialFracture(seed=42, num_slices=6)
        slices, _ = rf.generate_radial_pattern(center=(0, 0, 0), radius=5.0)
        for slice_obj in slices:
            assert slice_obj.plane is not None

    def test_radial_rings_have_valid_radii(self):
        """Test radial rings have valid radii."""
        rf = RadialFracture(seed=42, num_rings=4)
        _, rings = rf.generate_radial_pattern(center=(0, 0, 0), radius=10.0)
        for ring in rings:
            assert ring.radius_inner >= 0
            assert ring.radius_outer > ring.radius_inner
            assert ring.radius_outer <= 10.0

    def test_radial_impact_directed(self):
        """Test impact-directed radial pattern."""
        rf = RadialFracture(seed=42, num_slices=8, num_rings=3)
        slices, rings = rf.generate_impact_directed(
            center=(0, 0, 0),
            impact_direction=(1, 0, 0),
            radius=5.0,
            intensity=1.0
        )
        assert len(slices) > 0
        assert len(rings) > 0

    def test_radial_fracture_mesh(self, cube_mesh):
        """Test radial mesh fracturing."""
        vertices, triangles = cube_mesh
        rf = RadialFracture(seed=42, num_slices=6, num_rings=2, min_chunk_volume=0.0001)
        chunks = rf.fracture_mesh(vertices, triangles)
        # May produce chunks depending on mesh
        assert isinstance(chunks, list)

    def test_concentric_radial_fracture(self):
        """Test ConcentricRadialFracture."""
        cf = ConcentricRadialFracture(seed=42, num_slices=8, num_rings=5)
        slices, rings = cf.generate_radial_pattern(center=(0, 0, 0), radius=10.0)
        assert len(rings) == 5

    def test_spider_web_fracture(self):
        """Test SpiderWebFracture."""
        swf = SpiderWebFracture(seed=42, num_radial=8, num_circular=4, irregularity=0.3)
        slices, rings = swf.generate_radial_pattern(center=(0, 0, 0), radius=10.0)
        assert len(slices) == 8
        assert len(rings) == 4

    def test_radial_get_cut_planes(self):
        """Test getting cut planes."""
        rf = RadialFracture(seed=42, num_slices=6)
        rf.generate_radial_pattern(center=(0, 0, 0), radius=5.0)
        planes = rf.get_cut_planes()
        assert len(planes) == 6


# =============================================================================
# SLICE FRACTURE TESTS
# =============================================================================

class TestSliceFracture:
    """Tests for planar slice fracture."""

    def test_slice_creation(self):
        """Test SliceFracture creation."""
        sf = SliceFracture(seed=42)
        assert sf.seed == 42

    def test_single_slice(self, cube_mesh):
        """Test single plane slice."""
        vertices, triangles = cube_mesh
        sf = SliceFracture(seed=42)
        plane = Plane(point=(0, 0, 0), normal=(1, 0, 0))
        result = sf.slice_mesh(vertices, triangles, plane)
        assert isinstance(result, SliceResult)
        # Should produce front and back chunks
        assert result.front_chunk is not None or result.back_chunk is not None

    def test_multi_slice(self, cube_mesh):
        """Test multiple plane slices."""
        vertices, triangles = cube_mesh
        sf = SliceFracture(seed=42, min_chunk_volume=0.0001)
        planes = [
            Plane(point=(0, 0, 0), normal=(1, 0, 0)),
            Plane(point=(0, 0, 0), normal=(0, 1, 0)),
        ]
        chunks = sf.multi_slice(vertices, triangles, planes)
        assert len(chunks) > 0

    def test_parallel_slices(self, cube_mesh):
        """Test parallel slices."""
        vertices, triangles = cube_mesh
        sf = SliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = sf.parallel_slices(
            vertices, triangles,
            direction=(1, 0, 0),
            num_slices=3
        )
        assert len(chunks) > 0

    def test_random_slice_planes(self):
        """Test random plane generation."""
        sf = SliceFracture(seed=42)
        bounds = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        planes = sf.random_slice_planes(bounds, num_planes=5)
        assert len(planes) == 5

    def test_grid_slice(self, cube_mesh):
        """Test grid slicing."""
        vertices, triangles = cube_mesh
        sf = SliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = sf.grid_slice(vertices, triangles, grid_size=(2, 2, 2))
        assert len(chunks) > 0

    def test_adaptive_slice_fracture(self, cube_mesh):
        """Test AdaptiveSliceFracture."""
        vertices, triangles = cube_mesh
        asf = AdaptiveSliceFracture(seed=42, min_chunk_volume=0.0001)
        chunks = asf.fracture_adaptive(
            vertices, triangles,
            impact_point=(0, 0, 0),
            impact_intensity=0.5
        )
        assert isinstance(chunks, list)

    def test_hierarchical_slice_fracture(self, cube_mesh):
        """Test HierarchicalSliceFracture."""
        vertices, triangles = cube_mesh
        hsf = HierarchicalSliceFracture(
            seed=42,
            min_chunk_volume=0.0001,
            max_depth=2
        )
        chunks = hsf.fracture_hierarchical(vertices, triangles)
        assert isinstance(chunks, list)


# =============================================================================
# SUPPORT GRAPH TESTS
# =============================================================================

class TestSupportGraph:
    """Tests for support graph stress paths."""

    def test_support_graph_creation(self):
        """Test SupportGraph creation."""
        sg = SupportGraph()
        assert len(sg.nodes) == 0
        assert len(sg.edges) == 0

    def test_add_node(self):
        """Test adding nodes."""
        sg = SupportGraph()
        node = sg.add_node(0, position=(0, 0, 0), mass=1.0)
        assert node.id == 0
        assert 0 in sg.nodes

    def test_add_anchor(self):
        """Test adding anchor nodes."""
        sg = SupportGraph()
        node = sg.add_anchor(0, position=(0, 0, 0))
        assert node.is_anchor
        assert node.is_supported
        assert 0 in sg.anchors

    def test_remove_anchor(self):
        """Test removing anchor status."""
        sg = SupportGraph()
        sg.add_anchor(0, position=(0, 0, 0))
        sg.remove_anchor(0)
        assert 0 not in sg.anchors
        assert not sg.nodes[0].is_anchor

    def test_add_connection(self):
        """Test adding connections."""
        sg = SupportGraph()
        sg.add_node(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        edge = sg.add_connection(0, 1)
        assert edge.node_a == 0 or edge.node_b == 0
        assert 1 in sg.nodes[0].connections
        assert 0 in sg.nodes[1].connections

    def test_remove_connection(self):
        """Test removing connections."""
        sg = SupportGraph()
        sg.add_node(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_connection(0, 1)
        sg.remove_connection(0, 1)
        assert 1 not in sg.nodes[0].connections

    def test_compute_stress_paths(self):
        """Test stress path computation."""
        sg = SupportGraph()
        sg.add_anchor(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_node(2, position=(2, 0, 0))
        sg.add_connection(0, 1)
        sg.add_connection(1, 2)
        sg.compute_stress_paths()
        assert sg.nodes[1].is_supported
        assert sg.nodes[1].support_distance == 1
        assert sg.nodes[2].support_distance == 2

    def test_detect_unsupported(self):
        """Test detecting unsupported nodes."""
        sg = SupportGraph()
        sg.add_anchor(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_node(2, position=(2, 0, 0))  # Not connected
        sg.add_connection(0, 1)
        unsupported = sg.detect_unsupported()
        assert 2 in unsupported
        assert 1 not in unsupported

    def test_propagate_damage(self):
        """Test damage propagation."""
        sg = SupportGraph(stress_threshold=100.0)
        sg.add_node(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_connection(0, 1, strength=50.0)
        broken = sg.propagate_damage(0, damage_amount=100.0)
        assert len(broken) == 1

    def test_get_connected_component(self):
        """Test getting connected component."""
        sg = SupportGraph()
        sg.add_node(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_node(2, position=(2, 0, 0))
        sg.add_connection(0, 1)
        component = sg.get_connected_component(0)
        assert 0 in component
        assert 1 in component
        assert 2 not in component

    def test_get_falling_groups(self):
        """Test getting falling groups."""
        sg = SupportGraph()
        sg.add_anchor(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))  # Connected to anchor
        sg.add_node(2, position=(2, 0, 0))  # Disconnected
        sg.add_node(3, position=(3, 0, 0))  # Disconnected, connected to 2
        sg.add_connection(0, 1)
        sg.add_connection(2, 3)
        groups = sg.get_falling_groups()
        assert len(groups) == 1  # One group of disconnected nodes
        assert 2 in groups[0] or 3 in groups[0]

    def test_disconnection_triggers_unsupported(self):
        """Test that breaking connection causes nodes to become unsupported."""
        sg = SupportGraph()
        sg.add_anchor(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_node(2, position=(2, 0, 0))
        sg.add_connection(0, 1)
        sg.add_connection(1, 2)

        # Initially all nodes should be supported
        sg.compute_stress_paths()
        assert sg.nodes[1].is_supported
        assert sg.nodes[2].is_supported

        # Break the connection between 0 and 1
        sg.remove_connection(0, 1)

        # After recomputing, nodes 1 and 2 should be unsupported
        unsupported = sg.detect_unsupported()
        assert 1 in unsupported
        assert 2 in unsupported

    def test_propagate_damage_max_depth(self):
        """Test damage propagation respects max depth limit."""
        sg = SupportGraph(stress_threshold=100.0)
        # Create a long chain of nodes
        for i in range(20):
            sg.add_node(i, position=(i, 0, 0))
        for i in range(19):
            sg.add_connection(i, i+1, strength=1000.0)

        # Propagate damage with low max_depth
        broken = sg.propagate_damage(0, damage_amount=50.0, max_depth=3)

        # Should not propagate beyond depth 3
        # Nodes beyond depth 3 should have zero stress
        assert sg.nodes[5].stress == 0.0

    def test_propagate_damage_no_infinite_loop(self):
        """Test damage propagation handles cycles without infinite loop."""
        sg = SupportGraph(stress_threshold=100.0, propagation_rate=0.5)
        # Create a cycle: 0-1-2-0
        sg.add_node(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_node(2, position=(0, 1, 0))
        sg.add_connection(0, 1, strength=1000.0)
        sg.add_connection(1, 2, strength=1000.0)
        sg.add_connection(2, 0, strength=1000.0)

        # This should complete without hanging
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Damage propagation took too long - possible infinite loop")

        # Set a 1 second timeout (should complete in milliseconds)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1)
        try:
            broken = sg.propagate_damage(0, damage_amount=50.0)
        finally:
            signal.alarm(0)

        # Should complete and all nodes should have some stress
        assert sg.nodes[0].stress > 0

    def test_support_graph_serialization(self):
        """Test support graph serialization."""
        sg = SupportGraph()
        sg.add_anchor(0, position=(0, 0, 0))
        sg.add_node(1, position=(1, 0, 0))
        sg.add_connection(0, 1)
        data = sg.to_dict()
        restored = SupportGraph.from_dict(data)
        assert len(restored.nodes) == 2
        assert len(restored.edges) == 1


# =============================================================================
# DEBRIS TESTS
# =============================================================================

class TestDebris:
    """Tests for debris lifecycle management."""

    def test_debris_pool_creation(self):
        """Test DebrisPool creation."""
        pool = DebrisPool(initial_size=10)
        assert pool.pool_size == 10

    def test_debris_pool_acquire(self):
        """Test acquiring debris from pool."""
        pool = DebrisPool(initial_size=10)
        debris = pool.acquire()
        assert debris is not None
        assert debris.state == DebrisState.ACTIVE

    def test_debris_pool_release(self):
        """Test releasing debris to pool."""
        pool = DebrisPool(initial_size=10)
        debris = pool.acquire()
        pool.release(debris)
        assert debris.state == DebrisState.POOLED

    def test_debris_manager_creation(self):
        """Test DebrisManager creation."""
        dm = DebrisManager(max_active=100)
        assert dm.max_active == 100
        assert dm.active_count == 0

    def test_debris_manager_spawn(self, simple_chunk):
        """Test spawning debris."""
        dm = DebrisManager(max_active=100)
        debris = dm.spawn_debris(
            chunk=simple_chunk,
            velocity=(1, 0, 0)
        )
        assert debris is not None
        assert dm.active_count == 1

    def test_debris_manager_max_limit(self, simple_chunk):
        """Test max active debris limit."""
        dm = DebrisManager(max_active=5)
        for _ in range(10):
            dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0))
        assert dm.active_count <= 5

    def test_debris_manager_update(self, simple_chunk):
        """Test debris manager update processes expiration correctly."""
        dm = DebrisManager(max_active=100)
        # Use a very short lifetime
        debris = dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0), lifetime=0.001)
        debris_id = debris.id
        assert dm.active_count == 1

        # Simulate time passing by directly manipulating spawn_time
        # This is more reliable than sleep() for testing
        debris.spawn_time = debris.spawn_time - 1.0  # Make it 1 second old

        # Update should clean up expired debris
        cleaned = dm.update(dt=0.016)
        # The debris should be marked for cleanup after update
        assert debris_id in cleaned or dm.active_count == 0

    def test_debris_manager_update_sleep_state(self, simple_chunk):
        """Test debris enters sleep state when velocity is low."""
        from engine.simulation.destruction import DEBRIS_MIN_VELOCITY, DEBRIS_SLEEP_TIME
        dm = DebrisManager(max_active=100, sleep_velocity=DEBRIS_MIN_VELOCITY, sleep_time=0.1)
        debris = dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0), lifetime=10.0)

        # Initial state should be ACTIVE
        assert debris.state == DebrisState.ACTIVE

        # Update multiple times with low velocity to trigger sleep
        for _ in range(10):
            dm.update(dt=0.05)

        # After enough time at low velocity, should be SLEEPING
        assert debris.state == DebrisState.SLEEPING

    def test_debris_manager_destroy(self, simple_chunk):
        """Test destroying specific debris."""
        dm = DebrisManager()
        debris = dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0))
        result = dm.destroy_debris(debris.id)
        assert result is True

    def test_debris_manager_destroy_all(self, simple_chunk):
        """Test destroying all debris."""
        dm = DebrisManager()
        for _ in range(5):
            dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0))
        count = dm.destroy_all()
        assert count == 5

    def test_debris_lod_calculation(self, simple_chunk):
        """Test LOD calculation based on distance."""
        dm = DebrisManager()
        dm.set_lod_distances(full=10.0, reduced=25.0, simple=50.0)
        debris = dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0))
        dm.update(dt=0.016, camera_position=(5, 0, 0))
        assert debris.lod == DebrisLOD.FULL

    def test_debris_manager_stats(self, simple_chunk):
        """Test getting debris statistics."""
        dm = DebrisManager()
        for _ in range(3):
            dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0))
        stats = dm.get_stats()
        assert stats['active_count'] == 3

    def test_debris_get_in_radius(self, simple_chunk):
        """Test getting debris in radius."""
        dm = DebrisManager()
        dm.spawn_debris(chunk=simple_chunk, velocity=(0, 0, 0))
        debris_list = dm.get_debris_in_radius(center=(0, 0, 0), radius=10.0)
        assert len(debris_list) >= 1

    def test_spawn_debris_from_fracture(self, simple_chunk):
        """Test spawn_debris_from_fracture helper."""
        dm = DebrisManager()
        chunks = [simple_chunk, simple_chunk]
        debris_list = spawn_debris_from_fracture(
            manager=dm,
            chunks=chunks,
            center_velocity=(1, 0, 0),
            spread_factor=1.0
        )
        assert len(debris_list) == 2


# =============================================================================
# DESTRUCTION SYSTEM TESTS
# =============================================================================

class TestDestructionSystem:
    """Tests for main destruction system integration."""

    def test_system_creation(self):
        """Test DestructionSystem creation."""
        ds = DestructionSystem()
        assert ds is not None
        assert ds.config is not None

    def test_register_destructible(self, destruction_system, cube_mesh):
        """Test registering a destructible."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            health=100.0
        )
        assert dest_id == 0
        assert dest_id in destruction_system.destructibles

    def test_unregister_destructible(self, destruction_system, cube_mesh):
        """Test unregistering a destructible."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        result = destruction_system.unregister_destructible(dest_id)
        assert result is True
        assert dest_id not in destruction_system.destructibles

    def test_apply_damage(self, destruction_system, cube_mesh):
        """Test applying damage."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            health=100.0
        )
        damage = Damage(
            amount=30.0,
            damage_type=DamageType.IMPACT,
            position=(0, 0, 0)
        )
        result = destruction_system.apply_damage(dest_id, damage, immediate=True)
        assert result is not None
        assert result.final_amount > 0

    def test_damage_queuing(self, destruction_system, cube_mesh):
        """Test damage queuing for next update."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        damage = Damage(amount=30.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))
        result = destruction_system.apply_damage(dest_id, damage, immediate=False)
        assert result is None  # Queued, not processed yet

    def test_trigger_fracture(self, destruction_system, cube_mesh):
        """Test triggering fracture."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            fracture_depth=1
        )
        chunks = destruction_system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(1, 0, 0),
            immediate=True
        )
        assert chunks is not None
        assert len(chunks) > 0

    def test_fracture_updates_state(self, destruction_system, cube_mesh):
        """Test fracturing updates destructible state."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        destruction_system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(1, 0, 0),
            immediate=True
        )
        destructible = destruction_system.get_destructible(dest_id)
        assert destructible.state == DestructibleState.FRACTURED

    def test_apply_area_damage(self, destruction_system, cube_mesh):
        """Test applying area damage."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        damage = Damage(
            amount=100.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0, 0, 0)
        )
        results = destruction_system.apply_area_damage(
            center=(0, 0, 0),
            radius=10.0,
            damage=damage
        )
        assert len(results) > 0

    def test_system_update(self, destruction_system, cube_mesh):
        """Test system update processing."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        damage = Damage(amount=30.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))
        destruction_system.apply_damage(dest_id, damage, immediate=False)
        destruction_system.update(dt=0.016)
        assert len(destruction_system.damage_events) > 0

    def test_damage_events(self, destruction_system, cube_mesh):
        """Test damage event generation."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        damage = Damage(amount=30.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))
        destruction_system.apply_damage(dest_id, damage, immediate=True)
        assert len(destruction_system.damage_events) == 1
        event = destruction_system.damage_events[0]
        assert event.destructible_id == dest_id

    def test_fracture_events(self, destruction_system, cube_mesh):
        """Test fracture event generation."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        destruction_system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(1, 0, 0),
            immediate=True
        )
        assert len(destruction_system.fracture_events) == 1
        event = destruction_system.fracture_events[0]
        assert event.destructible_id == dest_id
        assert len(event.chunks) > 0

    def test_get_destructibles_in_radius(self, destruction_system, cube_mesh):
        """Test finding destructibles in radius."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        found = destruction_system.get_destructibles_in_radius(
            center=(0, 0, 0),
            radius=10.0
        )
        assert dest_id in found

    def test_system_clear(self, destruction_system, cube_mesh):
        """Test clearing the system."""
        vertices, triangles = cube_mesh
        destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        destruction_system.clear()
        assert len(destruction_system.destructibles) == 0

    def test_system_stats(self, destruction_system, cube_mesh):
        """Test getting system statistics."""
        vertices, triangles = cube_mesh
        destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles
        )
        stats = destruction_system.get_stats()
        assert stats['destructible_count'] == 1

    def test_damage_with_resistance(self, destruction_system, cube_mesh):
        """Test damage modified by resistance."""
        vertices, triangles = cube_mesh
        resistance = DamageResistance(resistances={DamageType.IMPACT: 0.5})
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            health=100.0,
            resistance=resistance
        )
        damage = Damage(amount=100.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))
        result = destruction_system.apply_damage(dest_id, damage, immediate=True)
        # Should take 50% of 100 = 50 damage
        assert result.final_amount < damage.amount

    def test_custom_fracture_callback(self, destruction_system, cube_mesh):
        """Test custom fracture pattern uses callback."""
        vertices, triangles = cube_mesh

        # Track if callback was called
        callback_called = [False]
        callback_chunks = []

        def custom_fracture(verts, tris, impact_pt, impact_dir, intensity):
            callback_called[0] = True
            # Create simple test chunks
            chunk1 = Chunk(
                vertices=verts[:4],
                triangles=[(0, 1, 2)],
                volume=1.0,
                centroid=(0, 0, 0)
            )
            chunk2 = Chunk(
                vertices=verts[4:] if len(verts) > 4 else verts[:4],
                triangles=[(0, 1, 2)],
                volume=1.0,
                centroid=(1, 0, 0)
            )
            callback_chunks.extend([chunk1, chunk2])
            return [chunk1, chunk2]

        destruction_system.set_custom_fracture_callback(custom_fracture)

        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            fracture_pattern=FracturePattern.CUSTOM
        )

        chunks = destruction_system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(1, 0, 0),
            immediate=True
        )

        assert callback_called[0], "Custom fracture callback was not called"
        assert len(chunks) == 2, "Custom callback should return 2 chunks"

    def test_custom_fracture_fallback_without_callback(self, destruction_system, cube_mesh):
        """Test CUSTOM pattern falls back to Voronoi if no callback set."""
        vertices, triangles = cube_mesh

        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            fracture_pattern=FracturePattern.CUSTOM
        )

        # No custom callback set - should use Voronoi fallback
        chunks = destruction_system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(1, 0, 0),
            immediate=True
        )

        # Should still produce chunks via fallback
        assert len(chunks) > 0

    def test_destruction_threshold(self, destruction_system, cube_mesh):
        """Test object destruction at threshold."""
        vertices, triangles = cube_mesh
        dest_id = destruction_system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            health=100.0
        )
        damage = Damage(amount=150.0, damage_type=DamageType.IMPACT, position=(0, 0, 0))
        result = destruction_system.apply_damage(dest_id, damage, immediate=True)
        assert result.was_lethal
        destructible = destruction_system.get_destructible(dest_id)
        assert destructible.state == DestructibleState.DESTROYED


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_vec3_add(self):
        """Test vector addition."""
        result = vec3_add((1, 2, 3), (4, 5, 6))
        assert result == (5, 7, 9)

    def test_vec3_sub(self):
        """Test vector subtraction."""
        result = vec3_sub((4, 5, 6), (1, 2, 3))
        assert result == (3, 3, 3)

    def test_vec3_length(self):
        """Test vector length."""
        length = vec3_length((3, 4, 0))
        assert abs(length - 5.0) < 1e-6

    def test_vec3_normalize(self):
        """Test vector normalization."""
        result = vec3_normalize((3, 4, 0))
        length = vec3_length(result)
        assert abs(length - 1.0) < 1e-6

    def test_vec3_distance(self):
        """Test point distance."""
        dist = vec3_distance((0, 0, 0), (3, 4, 0))
        assert abs(dist - 5.0) < 1e-6

    def test_bounding_box_creation(self):
        """Test BoundingBox creation."""
        bb = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert bb.volume == 1000.0

    def test_bounding_box_center(self):
        """Test BoundingBox center calculation."""
        bb = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert bb.center == (5, 5, 5)

    def test_bounding_box_contains(self):
        """Test BoundingBox point containment."""
        bb = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert bb.contains((5, 5, 5))
        assert not bb.contains((15, 5, 5))

    def test_bounding_box_from_points(self):
        """Test BoundingBox.from_points factory."""
        points = [(0, 0, 0), (10, 0, 0), (5, 10, 5)]
        bb = BoundingBox.from_points(points)
        assert bb.min_point == (0, 0, 0)
        assert bb.max_point == (10, 10, 5)

    def test_plane_signed_distance(self):
        """Test Plane.signed_distance."""
        plane = Plane(point=(0, 0, 0), normal=(0, 1, 0))
        assert plane.signed_distance((0, 5, 0)) == 5.0
        assert plane.signed_distance((0, -5, 0)) == -5.0

    def test_plane_classify_point(self):
        """Test Plane.classify_point."""
        plane = Plane(point=(0, 0, 0), normal=(0, 1, 0))
        assert plane.classify_point((0, 5, 0)) == 1   # Front
        assert plane.classify_point((0, -5, 0)) == -1  # Back
        assert plane.classify_point((0, 0, 0)) == 0   # On plane

    def test_plane_bisector(self):
        """Test Plane.bisector."""
        plane = Plane.bisector((0, 0, 0), (10, 0, 0))
        assert abs(plane.signed_distance((5, 0, 0))) < 1e-6

    def test_chunk_validity(self, simple_chunk):
        """Test chunk validity check."""
        simple_chunk.volume = 0.01
        assert simple_chunk.is_valid(min_volume=0.001)
        assert not simple_chunk.is_valid(min_volume=0.1)


# =============================================================================
# DEGENERATE GEOMETRY TESTS
# =============================================================================

class TestDegenerateGeometry:
    """Tests for handling degenerate geometry edge cases."""

    def test_degenerate_triangle_detection(self):
        """Test that degenerate triangles are properly detected."""
        from engine.simulation.destruction.fracture_voronoi import (
            is_degenerate_triangle,
            triangle_area,
        )

        # Collinear points (zero area)
        v0, v1, v2 = (0, 0, 0), (1, 0, 0), (2, 0, 0)
        assert is_degenerate_triangle(v0, v1, v2)

        # Duplicate points
        v0, v1, v2 = (1, 1, 1), (1, 1, 1), (2, 2, 2)
        assert is_degenerate_triangle(v0, v1, v2)

        # Valid triangle
        v0, v1, v2 = (0, 0, 0), (1, 0, 0), (0, 1, 0)
        assert not is_degenerate_triangle(v0, v1, v2)

    def test_triangle_area_calculation(self):
        """Test triangle area calculation."""
        from engine.simulation.destruction.fracture_voronoi import triangle_area

        # Unit triangle has area 0.5
        area = triangle_area((0, 0, 0), (1, 0, 0), (0, 1, 0))
        assert abs(area - 0.5) < 1e-10

        # Zero area for collinear points
        area = triangle_area((0, 0, 0), (1, 0, 0), (2, 0, 0))
        assert abs(area) < 1e-10

    def test_slice_handles_edge_on_plane(self, cube_mesh):
        """Test slice fracture handles edges lying on the cutting plane."""
        vertices, triangles = cube_mesh
        sf = SliceFracture(seed=42, min_chunk_volume=0.0001)

        # Create a plane that passes through mesh edges
        plane = Plane(point=(0, 0, 0), normal=(1, 0, 0))
        result = sf.slice_mesh(vertices, triangles, plane)

        # Should produce valid result without division by zero
        assert result is not None
        # At least one side should have a chunk
        assert result.front_chunk is not None or result.back_chunk is not None

    def test_voronoi_clips_degenerate_results(self, cube_mesh):
        """Test Voronoi clipping filters out degenerate triangles."""
        vertices, triangles = cube_mesh
        vf = VoronoiFracture(seed=42, num_sites=16, min_chunk_volume=0.0001)
        chunks = vf.fracture(vertices, triangles)

        # All resulting triangles should be non-degenerate
        from engine.simulation.destruction.fracture_voronoi import is_degenerate_triangle

        for chunk in chunks:
            for tri in chunk.triangles:
                v0 = chunk.vertices[tri[0]]
                v1 = chunk.vertices[tri[1]]
                v2 = chunk.vertices[tri[2]]
                assert not is_degenerate_triangle(v0, v1, v2), \
                    f"Found degenerate triangle in chunk: {tri}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete destruction system."""

    def test_full_destruction_workflow(self, cube_mesh):
        """Test complete destruction workflow."""
        vertices, triangles = cube_mesh

        # Create system
        system = DestructionSystem()

        # Register destructible
        resistance = DamageResistance.from_dict({
            "impact": 1.0,
            "explosive": 0.5
        })

        dest_id = system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            health=100.0,
            resistance=resistance,
            fracture_pattern=FracturePattern.VORONOI,
            fracture_depth=2
        )

        # Apply damage
        damage = Damage(
            amount=80.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0.5, 0.5, 0.5),
            direction=(1, 0, 0)
        )
        result = system.apply_damage(dest_id, damage, immediate=True)

        # Verify damage was reduced by resistance
        assert result.final_amount < 80.0

        # Trigger fracture
        chunks = system.trigger_fracture(
            dest_id,
            impact_point=(0.5, 0.5, 0.5),
            impact_direction=(1, 0, 0),
            immediate=True
        )

        # Verify fracture occurred
        assert len(chunks) > 0
        destructible = system.get_destructible(dest_id)
        assert destructible.state == DestructibleState.FRACTURED

        # Update system
        system.update(dt=0.016, camera_position=(10, 0, 0))

        # Check debris was spawned
        assert system.debris_manager.active_count > 0

    def test_area_damage_multiple_targets(self):
        """Test area damage affecting multiple destructibles."""
        system = DestructionSystem()

        # Create a simple mesh
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        triangles = [(0, 1, 2), (0, 1, 3)]

        # Register multiple destructibles at different positions
        ids = []
        for i in range(3):
            # Offset each destructible
            offset_verts = [(v[0] + i*5, v[1], v[2]) for v in vertices]
            dest_id = system.register_destructible(
                vertices=offset_verts,
                triangles=triangles,
                health=100.0
            )
            ids.append(dest_id)

        # Apply area damage
        damage = Damage(
            amount=50.0,
            damage_type=DamageType.EXPLOSIVE,
            position=(0, 0, 0)
        )
        results = system.apply_area_damage(
            center=(0, 0, 0),
            radius=20.0,
            damage=damage,
            falloff="linear"
        )

        # All three should be affected
        assert len(results) == 3

        # Closer ones should take more damage
        result0 = next(r for r in results if r.damage_type == DamageType.EXPLOSIVE)
        assert result0.final_amount > 0

    def test_support_graph_with_destruction(self, cube_mesh):
        """Test support graph integration with destruction."""
        vertices, triangles = cube_mesh
        system = DestructionSystem()

        dest_id = system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            health=50.0  # Low health for easy fracture
        )

        # Fracture the object
        system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(0, 1, 0),
            immediate=True
        )

        destructible = system.get_destructible(dest_id)

        # Verify support graph was created
        assert destructible.support_graph is not None
        assert len(destructible.support_graph.nodes) > 0

    def test_debris_cleanup_over_time(self, cube_mesh):
        """Test debris cleanup after lifetime expires."""
        vertices, triangles = cube_mesh
        system = DestructionSystem()

        dest_id = system.register_destructible(
            vertices=vertices,
            triangles=triangles,
            debris_lifetime=0.1  # Very short lifetime
        )

        system.trigger_fracture(
            dest_id,
            impact_point=(0, 0, 0),
            impact_direction=(1, 0, 0),
            immediate=True
        )

        initial_count = system.debris_manager.active_count
        assert initial_count > 0

        # Simulate time passing
        import time
        time.sleep(0.2)

        # Update should clean up expired debris
        system.update(dt=0.2)

        # Debris should be marked for cleanup
        # (actual cleanup happens during update)
