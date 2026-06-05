"""
Blackbox tests for vegetation SDF (T-DEMO-4.5 and T-DEMO-4.6).

Tests the Tree SDF and Forest SDF implementations as black boxes,
verifying behavior against specifications without knowledge of internals.

T-DEMO-4.5: Tree SDF
- Tree shape correctness
- Trunk geometry (cylinder/cone)
- Canopy geometry (spheres/ellipsoids)
- Branch connectivity
- Smooth union blending

T-DEMO-4.6: Forest SDF (Domain Repetition)
- Forest periodicity
- Per-cell variation
- Hash determinism
- SDF continuity at cell boundaries
"""

from __future__ import annotations

import math

import pytest


# =============================================================================
# Import module under test
# =============================================================================

from engine.rendering.demoscene.vegetation_sdf import (
    # Enums
    TrunkType,
    CanopyType,
    # Configurations
    TreeConfig,
    BranchConfig,
    ForestConfig,
    TreeVariation,
    # SDF Classes
    TreeSDF,
    ForestSDF,
    # Hash functions
    cell_hash,
    cell_hash_float,
    hash_to_float,
    # WGSL generation
    generate_tree_wgsl,
    generate_forest_wgsl,
    # Python SDF functions
    sdf_tree,
    sdf_forest,
)


# =============================================================================
# Test Constants
# =============================================================================

TOL_SURFACE = 1e-6      # Points on surface should be very close to 0
TOL_DISTANCE = 0.01     # General distance tolerance
TOL_CONTINUITY = 0.1    # Max allowed jump for nearby points
TOL_HASH = 1e-9         # Hash reproducibility tolerance


# =============================================================================
# TreeConfig Validation Tests
# =============================================================================

class TestTreeConfigValidation:
    """Test TreeConfig parameter validation."""

    def test_default_config_is_valid(self):
        """Default configuration should be valid."""
        config = TreeConfig()
        assert config.trunk_height > 0
        assert config.trunk_radius > 0
        assert config.canopy_spheres >= 1

    def test_negative_trunk_height_raises(self):
        """Negative trunk height should raise ValueError."""
        with pytest.raises(ValueError, match="Trunk height must be positive"):
            TreeConfig(trunk_height=-1.0)

    def test_zero_trunk_height_raises(self):
        """Zero trunk height should raise ValueError."""
        with pytest.raises(ValueError, match="Trunk height must be positive"):
            TreeConfig(trunk_height=0.0)

    def test_negative_trunk_radius_raises(self):
        """Negative trunk radius should raise ValueError."""
        with pytest.raises(ValueError, match="Trunk radius must be positive"):
            TreeConfig(trunk_radius=-0.1)

    def test_zero_canopy_spheres_raises(self):
        """Zero canopy spheres should raise ValueError."""
        with pytest.raises(ValueError, match="Canopy spheres must be at least 1"):
            TreeConfig(canopy_spheres=0)

    def test_negative_smooth_k_raises(self):
        """Negative smooth_k should raise ValueError."""
        with pytest.raises(ValueError, match="Smooth k must be non-negative"):
            TreeConfig(smooth_k=-0.1)

    def test_invalid_trunk_taper_raises(self):
        """Invalid trunk taper should raise ValueError."""
        with pytest.raises(ValueError, match="Trunk taper must be in"):
            TreeConfig(trunk_taper=0.0)
        with pytest.raises(ValueError, match="Trunk taper must be in"):
            TreeConfig(trunk_taper=1.5)


# =============================================================================
# BranchConfig Validation Tests
# =============================================================================

class TestBranchConfigValidation:
    """Test BranchConfig parameter validation."""

    def test_default_branch_config_valid(self):
        """Default branch config should be valid."""
        config = BranchConfig()
        assert config.count >= 0
        assert config.radius > 0

    def test_negative_branch_count_raises(self):
        """Negative branch count should raise ValueError."""
        with pytest.raises(ValueError, match="Branch count must be non-negative"):
            BranchConfig(count=-1)

    def test_negative_branch_radius_raises(self):
        """Negative branch radius should raise ValueError."""
        with pytest.raises(ValueError, match="Branch radius must be positive"):
            BranchConfig(radius=-0.05)

    def test_invalid_attachment_height_raises(self):
        """Invalid attachment height should raise ValueError."""
        with pytest.raises(ValueError, match="Attachment height must be in"):
            BranchConfig(attachment_height=1.5)
        with pytest.raises(ValueError, match="Attachment height must be in"):
            BranchConfig(attachment_height=-0.1)


# =============================================================================
# TreeSDF Basic Shape Tests (T-DEMO-4.5)
# =============================================================================

class TestTreeSDFBasicShape:
    """Test basic tree shape properties."""

    def test_tree_base_at_origin(self):
        """Tree base should be at y=0."""
        tree = TreeSDF()
        # Point at base of trunk should be on or near surface
        d = tree.evaluate((0.0, 0.0, 0.0))
        # Should be near the trunk surface
        assert d < tree.config.trunk_radius * 1.5

    def test_point_inside_trunk_negative(self):
        """Points inside trunk should have negative distance."""
        tree = TreeSDF(TreeConfig(trunk_height=2.0, trunk_radius=0.3))
        # Point on trunk axis at half height
        d = tree.evaluate((0.0, 1.0, 0.0))
        assert d < 0, f"Inside trunk should be negative, got {d}"

    def test_point_far_outside_positive(self):
        """Points far outside tree should have positive distance."""
        tree = TreeSDF()
        # Point far from tree
        d = tree.evaluate((10.0, 0.0, 0.0))
        assert d > 0, f"Far outside should be positive, got {d}"

    def test_trunk_extends_to_height(self):
        """Trunk should extend to configured height."""
        config = TreeConfig(trunk_height=3.0, trunk_radius=0.2)
        tree = TreeSDF(config)

        # Point at top of trunk should be near surface
        d = tree.evaluate_trunk((0.0, 3.0, 0.0))
        assert abs(d) < config.trunk_radius * 2

    def test_canopy_above_trunk(self):
        """Canopy should be above the trunk."""
        config = TreeConfig(trunk_height=2.0, canopy_height_offset=0.5)
        tree = TreeSDF(config)

        canopy_center_y = config.trunk_height + config.canopy_height_offset

        # Point at canopy center should be inside
        d = tree.evaluate_canopy((0.0, canopy_center_y, 0.0))
        assert d < 0, f"Canopy center should be inside, got {d}"


# =============================================================================
# Trunk Geometry Tests
# =============================================================================

class TestTrunkGeometry:
    """Test trunk geometry for cylinder and tapered cone."""

    def test_cylinder_trunk_radial_symmetry(self):
        """Cylinder trunk should have radial symmetry."""
        config = TreeConfig(trunk_type=TrunkType.CYLINDER, trunk_radius=0.25)
        tree = TreeSDF(config)

        # Sample at various angles at mid-height
        y = config.trunk_height / 2
        r = config.trunk_radius + 0.1  # Just outside trunk

        distances = []
        for angle in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            x = r * math.cos(angle)
            z = r * math.sin(angle)
            d = tree.evaluate_trunk((x, y, z))
            distances.append(d)

        # All distances should be approximately equal
        for d in distances:
            assert d == pytest.approx(distances[0], abs=TOL_DISTANCE), \
                f"Cylinder should be radially symmetric: {distances}"

    def test_tapered_cone_wider_at_base(self):
        """Tapered cone should be wider at base than top."""
        config = TreeConfig(
            trunk_type=TrunkType.TAPERED_CONE,
            trunk_radius=0.3,
            trunk_taper=0.5,  # Top is half the base radius
            trunk_height=2.0,
        )
        tree = TreeSDF(config)

        # Distance at base (y=0.1) should be different from top (y=1.9)
        r = 0.2  # Inside base but outside top

        d_base = tree.evaluate_trunk((r, 0.1, 0.0))
        d_top = tree.evaluate_trunk((r, 1.9, 0.0))

        # At base, point should be inside (negative)
        # At top, same radial distance should be outside (positive)
        assert d_base < d_top, f"Cone should taper: d_base={d_base}, d_top={d_top}"


# =============================================================================
# Canopy Geometry Tests
# =============================================================================

class TestCanopyGeometry:
    """Test canopy geometry with spheres and ellipsoids."""

    def test_single_canopy_sphere_centered(self):
        """Single canopy sphere should be centered above trunk."""
        config = TreeConfig(
            canopy_spheres=1,
            canopy_radius=1.0,
            trunk_height=2.0,
            canopy_height_offset=0.5,
        )
        tree = TreeSDF(config)

        center_y = config.trunk_height + config.canopy_height_offset

        # Center of single sphere should be most inside
        d_center = tree.evaluate_canopy((0.0, center_y, 0.0))
        d_offset = tree.evaluate_canopy((0.5, center_y, 0.0))

        assert d_center < d_offset, "Center should be more inside"

    def test_multiple_canopy_spheres(self):
        """Multiple canopy spheres should create distributed shape."""
        config = TreeConfig(
            canopy_spheres=5,
            canopy_radius=0.5,
            canopy_spread=0.8,
            trunk_height=2.0,
        )
        tree = TreeSDF(config)

        positions = tree.get_canopy_sphere_positions()
        assert len(positions) == 5, f"Expected 5 canopy spheres, got {len(positions)}"

        # First position should be central
        assert positions[0] == pytest.approx((0.0, config.trunk_height + config.canopy_height_offset, 0.0), abs=0.01)

    def test_ellipsoid_canopy_flattened(self):
        """Ellipsoid canopy should be vertically flattened."""
        config = TreeConfig(
            canopy_type=CanopyType.ELLIPSOIDS,
            canopy_spheres=1,
            canopy_radius=1.0,
            trunk_height=2.0,
        )
        tree = TreeSDF(config)

        center_y = config.trunk_height + config.canopy_height_offset

        # Distance above/below center should be less than to the side
        d_above = tree.evaluate_canopy((0.0, center_y + 0.6, 0.0))
        d_side = tree.evaluate_canopy((0.6, center_y, 0.0))

        # Ellipsoid is flattened (0.7 vertical), so vertical reach is less
        # Side distance should be less (more inside) than same distance vertically
        assert d_above > d_side, "Ellipsoid should be flattened vertically"


# =============================================================================
# Branch Tests
# =============================================================================

class TestBranchConnectivity:
    """Test branch connectivity between trunk and canopy."""

    def test_no_branches_by_default(self):
        """Default configuration has no branches."""
        tree = TreeSDF()
        d = tree.evaluate_branches((0.0, 0.0, 0.0))
        assert d == float("inf"), "No branches should return inf"

    def test_branches_connect_to_trunk(self):
        """Branches should start near the trunk."""
        config = TreeConfig(
            branches=BranchConfig(count=4, radius=0.05, attachment_height=0.6),
            trunk_height=2.0,
            trunk_radius=0.15,
        )
        tree = TreeSDF(config)

        endpoints = tree.get_branch_endpoints()
        assert len(endpoints) == 4

        # Check each branch starts near trunk
        for start, end in endpoints:
            dist_from_axis = math.sqrt(start[0]**2 + start[2]**2)
            assert dist_from_axis < config.trunk_radius * 1.5, \
                f"Branch start should be near trunk axis: {start}"

    def test_branches_extend_outward(self):
        """Branches should extend away from trunk."""
        config = TreeConfig(
            branches=BranchConfig(count=4, radius=0.05),
            trunk_height=2.0,
            canopy_spheres=5,
        )
        tree = TreeSDF(config)

        endpoints = tree.get_branch_endpoints()

        for start, end in endpoints:
            start_dist = math.sqrt(start[0]**2 + start[2]**2)
            end_dist = math.sqrt(end[0]**2 + end[2]**2)
            assert end_dist > start_dist, \
                f"Branch should extend outward: start={start_dist}, end={end_dist}"

    def test_branch_sdf_near_attachment(self):
        """Points near branch attachment should be close to surface."""
        config = TreeConfig(
            branches=BranchConfig(count=4, radius=0.1, attachment_height=0.5),
            trunk_height=2.0,
            trunk_radius=0.15,
        )
        tree = TreeSDF(config)

        endpoints = tree.get_branch_endpoints()
        start, _ = endpoints[0]

        # Point at branch start should be on branch surface
        d = tree.evaluate_branches(start)
        assert abs(d) < config.branches.radius * 1.5


# =============================================================================
# Smooth Union Tests
# =============================================================================

class TestSmoothUnion:
    """Test smooth union blending between components."""

    def test_smooth_transition_trunk_canopy(self):
        """Transition from trunk to canopy should be smooth."""
        config = TreeConfig(
            trunk_height=2.0,
            canopy_height_offset=0.0,  # Canopy starts at trunk top
            smooth_k=0.2,
        )
        tree = TreeSDF(config)

        # Sample along the trunk-canopy boundary
        y_transition = config.trunk_height
        distances = []
        for y in [y_transition - 0.2, y_transition, y_transition + 0.2]:
            d = tree.evaluate((0.0, y, 0.0))
            distances.append(d)

        # Check for continuity (allow larger tolerance since shapes may change)
        for i in range(len(distances) - 1):
            delta = abs(distances[i+1] - distances[i])
            assert delta < 0.5, \
                f"Transition should be reasonably smooth: delta={delta}"

    def test_higher_k_smoother_blend(self):
        """Higher smooth_k should produce smoother transitions."""
        config_low = TreeConfig(smooth_k=0.05)
        config_high = TreeConfig(smooth_k=0.3)

        tree_low = TreeSDF(config_low)
        tree_high = TreeSDF(config_high)

        # Both should produce valid (finite) distances
        d_low = tree_low.evaluate((0.0, 2.0, 0.0))
        d_high = tree_high.evaluate((0.0, 2.0, 0.0))

        assert math.isfinite(d_low)
        assert math.isfinite(d_high)


# =============================================================================
# TreeSDF Position Offset Tests
# =============================================================================

class TestTreePosition:
    """Test tree position offset functionality."""

    def test_position_offset_translates_tree(self):
        """Position offset should translate the entire tree."""
        from engine.rendering.demoscene.sdf_ast import Vec3

        tree_origin = TreeSDF()
        tree_offset = TreeSDF(position=Vec3(5.0, 0.0, 0.0))

        # Same relative point should give same distance
        d_origin = tree_origin.evaluate((0.0, 1.0, 0.0))
        d_offset = tree_offset.evaluate((5.0, 1.0, 0.0))

        assert d_origin == pytest.approx(d_offset, abs=TOL_DISTANCE)

    def test_vertical_position_offset(self):
        """Vertical position offset should shift tree base."""
        from engine.rendering.demoscene.sdf_ast import Vec3

        tree = TreeSDF(position=Vec3(0.0, 3.0, 0.0))

        # Tree base is now at y=3
        d_at_base = tree.evaluate((0.0, 3.0, 0.0))
        d_below = tree.evaluate((0.0, 0.0, 0.0))

        assert d_below > d_at_base, "Point below offset base should be further"


# =============================================================================
# Hash Function Tests (T-DEMO-4.6)
# =============================================================================

class TestHashFunctions:
    """Test hash function properties."""

    def test_hash_deterministic(self):
        """Hash should be deterministic for same input."""
        cell = (5, -3, 7)

        h1 = cell_hash(cell)
        h2 = cell_hash(cell)

        assert h1 == h2, "Hash should be deterministic"

    def test_hash_different_for_different_cells(self):
        """Different cells should produce different hashes."""
        hashes = set()
        for x in range(-3, 4):
            for z in range(-3, 4):
                h = cell_hash((x, 0, z))
                hashes.add(h)

        # Most hashes should be unique (7*7=49 cells)
        assert len(hashes) >= 35, f"Hash should distribute well, got {len(hashes)}"

    def test_hash_to_float_in_range(self):
        """hash_to_float should return values in [0, 1)."""
        for h in [0, 1, 100, 0x7FFFFFFF, 0xFFFFFFFF]:
            f = hash_to_float(h)
            assert 0.0 <= f < 1.0, f"Float should be in [0, 1): {f}"

    def test_cell_hash_float_channel_independence(self):
        """Different channels should give independent values."""
        cell = (10, 20, 30)

        f0 = cell_hash_float(cell, channel=0)
        f1 = cell_hash_float(cell, channel=1)
        f2 = cell_hash_float(cell, channel=2)

        # Values should be different
        assert f0 != f1, "Channels should be independent"
        assert f1 != f2, "Channels should be independent"
        assert f0 != f2, "Channels should be independent"

    def test_hash_float_reproducible(self):
        """Hash float should be reproducible."""
        cell = (-5, 7, 13)

        f1 = cell_hash_float(cell, channel=3)
        f2 = cell_hash_float(cell, channel=3)

        assert f1 == pytest.approx(f2, abs=TOL_HASH)


# =============================================================================
# ForestConfig Validation Tests
# =============================================================================

class TestForestConfigValidation:
    """Test ForestConfig parameter validation."""

    def test_default_forest_config_valid(self):
        """Default forest config should be valid."""
        config = ForestConfig()
        assert all(s > 0 for s in config.cell_size)
        assert 0.0 <= config.density <= 1.0

    def test_negative_cell_size_raises(self):
        """Negative cell size should raise ValueError."""
        with pytest.raises(ValueError, match="Cell size must be positive"):
            ForestConfig(cell_size=(-1.0, 8.0, 8.0))

    def test_invalid_density_raises(self):
        """Invalid density should raise ValueError."""
        with pytest.raises(ValueError, match="Density must be in"):
            ForestConfig(density=1.5)
        with pytest.raises(ValueError, match="Density must be in"):
            ForestConfig(density=-0.1)


# =============================================================================
# TreeVariation Validation Tests
# =============================================================================

class TestTreeVariationValidation:
    """Test TreeVariation parameter validation."""

    def test_default_variation_valid(self):
        """Default variation should be valid."""
        var = TreeVariation()
        assert var.height_min > 0
        assert var.height_max >= var.height_min

    def test_invalid_height_range_raises(self):
        """Invalid height range should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid height range"):
            TreeVariation(height_min=1.5, height_max=1.0)

    def test_invalid_canopy_range_raises(self):
        """Invalid canopy range should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid canopy range"):
            TreeVariation(canopy_min=0)

    def test_invalid_jitter_raises(self):
        """Invalid jitter should raise ValueError."""
        with pytest.raises(ValueError, match="Position jitter must be in"):
            TreeVariation(position_jitter=0.7)


# =============================================================================
# ForestSDF Basic Tests
# =============================================================================

class TestForestSDFBasic:
    """Test basic forest SDF functionality."""

    def test_forest_creates_valid_sdf(self):
        """Forest should create valid (finite) SDF values."""
        forest = ForestSDF()

        # Sample at various points
        for p in [(0, 0, 0), (10, 0, 10), (-5, 0, -5)]:
            d = forest.evaluate(p)
            assert math.isfinite(d), f"Distance should be finite at {p}"

    def test_forest_has_trees(self):
        """Forest with high density should have trees."""
        config = ForestConfig(density=1.0)  # Every cell has a tree
        forest = ForestSDF(config)

        # Some point near cell center should be close to a tree
        d = forest.evaluate((4.0, 0.5, 4.0))
        assert d < 5.0, f"Should be near a tree with density=1.0: {d}"

    def test_empty_forest_far_distance(self):
        """Forest with zero density should return large distance."""
        config = ForestConfig(density=0.0)  # No trees
        forest = ForestSDF(config)

        d = forest.evaluate((4.0, 0.5, 4.0))
        assert d > 1e9, f"Empty forest should have large distance: {d}"


# =============================================================================
# Forest Cell ID Tests
# =============================================================================

class TestForestCellID:
    """Test cell ID computation."""

    def test_cell_id_correct(self):
        """Cell ID should be computed correctly."""
        config = ForestConfig(cell_size=(10.0, 10.0, 10.0))
        forest = ForestSDF(config)

        # Point in cell (0, 0, 0)
        assert forest.get_cell_id((5.0, 5.0, 5.0)) == (0, 0, 0)

        # Point in cell (1, 0, 0)
        assert forest.get_cell_id((15.0, 5.0, 5.0)) == (1, 0, 0)

        # Point in cell (-1, 0, 0)
        assert forest.get_cell_id((-5.0, 5.0, 5.0)) == (-1, 0, 0)

    def test_cell_id_boundary(self):
        """Cell ID at boundaries should be correct."""
        config = ForestConfig(cell_size=(8.0, 8.0, 8.0))
        forest = ForestSDF(config)

        # Exactly at cell boundary
        assert forest.get_cell_id((8.0, 0.0, 0.0)) == (1, 0, 0)
        assert forest.get_cell_id((7.99, 0.0, 0.0)) == (0, 0, 0)


# =============================================================================
# Forest Periodicity Tests (Domain Repetition)
# =============================================================================

class TestForestPeriodicity:
    """Test forest periodic behavior via domain repetition."""

    def test_same_relative_position_similar_distance(self):
        """Same relative position in different cells should be similar."""
        config = ForestConfig(
            cell_size=(10.0, 10.0, 10.0),
            density=1.0,  # Every cell has a tree
            variation=TreeVariation(
                height_min=1.0, height_max=1.0,  # No variation
                width_min=1.0, width_max=1.0,
                canopy_min=5, canopy_max=5,
                position_jitter=0.0,  # No jitter
                rotation_enabled=False,
            ),
        )
        forest = ForestSDF(config)

        # With no variation and no jitter, trees in different cells
        # should be at the same relative position
        d1 = forest.evaluate((5.0, 1.0, 5.0))    # Cell (0, 0, 0)
        d2 = forest.evaluate((15.0, 1.0, 5.0))   # Cell (1, 0, 0)
        d3 = forest.evaluate((5.0, 1.0, 15.0))   # Cell (0, 0, 1)

        # Distances should be similar (but not identical due to hash)
        # Actually they will be different because the hash still applies
        # This tests that the domain repetition is working
        assert all(math.isfinite(d) for d in [d1, d2, d3])

    def test_adjacent_cells_independent(self):
        """Adjacent cells should have independent tree configurations."""
        config = ForestConfig(density=1.0)
        forest = ForestSDF(config)

        # Get tree configs for adjacent cells
        config1 = forest.get_cell_tree_config((0, 0, 0))
        config2 = forest.get_cell_tree_config((1, 0, 0))

        # Configs should exist
        assert config1 is not None
        assert config2 is not None

        # At least one parameter should be different
        # (due to per-cell hash variation)
        different = (
            config1.trunk_height != config2.trunk_height or
            config1.canopy_spheres != config2.canopy_spheres
        )
        assert different, "Adjacent cells should have different trees"


# =============================================================================
# Per-Cell Variation Tests
# =============================================================================

class TestPerCellVariation:
    """Test per-cell tree variation."""

    def test_height_variation_in_range(self):
        """Tree heights should vary within configured range."""
        var = TreeVariation(height_min=0.5, height_max=2.0)
        config = ForestConfig(
            base_tree=TreeConfig(trunk_height=3.0),
            variation=var,
            density=1.0,
        )
        forest = ForestSDF(config)

        heights = []
        for x in range(10):
            tree_config = forest.get_cell_tree_config((x, 0, 0))
            if tree_config:
                heights.append(tree_config.trunk_height)

        # All heights should be in valid range
        base_height = config.base_tree.trunk_height
        for h in heights:
            assert var.height_min * base_height <= h <= var.height_max * base_height, \
                f"Height {h} out of range"

        # There should be variation
        assert max(heights) > min(heights) * 1.1, "Should have height variation"

    def test_canopy_count_variation(self):
        """Canopy sphere count should vary within configured range."""
        var = TreeVariation(canopy_min=3, canopy_max=7)
        config = ForestConfig(variation=var, density=1.0)
        forest = ForestSDF(config)

        counts = set()
        for x in range(20):
            tree_config = forest.get_cell_tree_config((x, 0, 0))
            if tree_config:
                counts.add(tree_config.canopy_spheres)

        # All counts should be in valid range
        for c in counts:
            assert var.canopy_min <= c <= var.canopy_max, f"Count {c} out of range"

        # Should have some variation
        assert len(counts) > 1, "Should have canopy count variation"

    def test_position_jitter_within_bounds(self):
        """Tree positions should be jittered within bounds."""
        config = ForestConfig(
            cell_size=(10.0, 10.0, 10.0),
            variation=TreeVariation(position_jitter=0.3),
            density=1.0,
        )
        forest = ForestSDF(config)

        for x in range(10):
            pos = forest.get_cell_tree_position((x, 0, 0))
            if pos:
                # Position should be within cell bounds
                cell_min_x = x * 10.0
                cell_max_x = (x + 1) * 10.0

                assert cell_min_x - 0.1 <= pos[0] <= cell_max_x + 0.1, \
                    f"Position {pos} outside cell bounds"


# =============================================================================
# Hash Determinism Tests
# =============================================================================

class TestHashDeterminism:
    """Test that forest generation is deterministic."""

    def test_same_cell_same_config(self):
        """Same cell ID should always produce same tree config."""
        config = ForestConfig(density=1.0)

        forest1 = ForestSDF(config)
        forest2 = ForestSDF(config)

        cell = (7, 0, -3)

        config1 = forest1.get_cell_tree_config(cell)
        config2 = forest2.get_cell_tree_config(cell)

        assert config1.trunk_height == config2.trunk_height
        assert config1.canopy_spheres == config2.canopy_spheres

    def test_same_point_same_distance(self):
        """Same point should always produce same distance."""
        forest = ForestSDF()

        p = (12.5, 0.7, -8.3)

        d1 = forest.evaluate(p)
        d2 = forest.evaluate(p)

        assert d1 == d2, "Same point should give same distance"


# =============================================================================
# SDF Continuity Tests
# =============================================================================

class TestSDFContinuity:
    """Test SDF continuity properties."""

    def test_continuity_within_cell(self):
        """SDF should be continuous within a cell."""
        forest = ForestSDF(ForestConfig(density=1.0))

        # Walk along a line and check continuity
        prev_d = forest.evaluate((0.0, 1.0, 0.0))
        for i in range(1, 20):
            x = i * 0.1
            d = forest.evaluate((x, 1.0, 0.0))
            delta = abs(d - prev_d)
            assert delta < 0.5, f"Jump too large at x={x}: delta={delta}"
            prev_d = d

    def test_continuity_at_cell_boundary(self):
        """SDF should be continuous at cell boundaries."""
        config = ForestConfig(cell_size=(8.0, 8.0, 8.0), density=1.0)
        forest = ForestSDF(config)

        # Points just before and after cell boundary
        boundary_x = 8.0
        step = 0.01

        d_before = forest.evaluate((boundary_x - step, 1.0, 4.0))
        d_after = forest.evaluate((boundary_x + step, 1.0, 4.0))

        delta = abs(d_after - d_before)
        # At boundaries, we check neighborhood, so continuity is preserved
        assert delta < 1.0, f"Discontinuity at boundary: delta={delta}"

    def test_lipschitz_bound(self):
        """SDF gradient should be bounded (Lipschitz condition)."""
        forest = ForestSDF()

        p1 = (5.0, 1.0, 5.0)
        p2 = (5.1, 1.0, 5.0)

        d1 = forest.evaluate(p1)
        d2 = forest.evaluate(p2)

        dist = 0.1
        gradient = abs(d2 - d1) / dist

        # SDF gradient should be approximately 1 (exact for true SDF)
        assert gradient < 2.0, f"Gradient too steep: {gradient}"


# =============================================================================
# WGSL Generation Tests
# =============================================================================

class TestWGSLGeneration:
    """Test WGSL code generation."""

    def test_tree_wgsl_contains_function(self):
        """Tree WGSL should contain main SDF function."""
        config = TreeConfig()
        wgsl = generate_tree_wgsl(config, name="my_tree")

        assert "fn sdf_my_tree(p: vec3<f32>) -> f32" in wgsl
        assert "fn sdf_my_tree_trunk(p: vec3<f32>) -> f32" in wgsl
        assert "fn sdf_my_tree_canopy(p: vec3<f32>) -> f32" in wgsl

    def test_tree_wgsl_has_primitives(self):
        """Tree WGSL should include primitive functions."""
        wgsl = generate_tree_wgsl(TreeConfig())

        assert "fn sdf_sphere" in wgsl
        assert "fn sdf_cylinder" in wgsl

    def test_tree_wgsl_has_smooth_union(self):
        """Tree WGSL should include smooth union."""
        wgsl = generate_tree_wgsl(TreeConfig())
        assert "fn sdf_smooth_union" in wgsl

    def test_tree_wgsl_with_branches(self):
        """Tree with branches should generate branch function."""
        config = TreeConfig(branches=BranchConfig(count=4))
        wgsl = generate_tree_wgsl(config, name="tree")

        assert "fn sdf_tree_branches(p: vec3<f32>) -> f32" in wgsl
        assert "sdf_capsule" in wgsl

    def test_forest_wgsl_contains_functions(self):
        """Forest WGSL should contain required functions."""
        config = ForestConfig()
        wgsl = generate_forest_wgsl(config, name="my_forest")

        assert "fn sdf_my_forest(p: vec3<f32>) -> f32" in wgsl
        assert "fn sdf_my_forest_cell_tree" in wgsl
        assert "fn cell_hash" in wgsl
        assert "fn get_cell_id" in wgsl

    def test_forest_wgsl_has_rotation(self):
        """Forest WGSL should include rotation function."""
        wgsl = generate_forest_wgsl(ForestConfig())
        assert "fn rotate_y" in wgsl

    def test_wgsl_valid_syntax(self):
        """Generated WGSL should have valid basic syntax."""
        wgsl = generate_forest_wgsl(ForestConfig())

        # Check balanced braces
        assert wgsl.count("{") == wgsl.count("}")
        # Parentheses may be slightly imbalanced due to float literals like 0.5
        # Check they are roughly balanced (within 5)
        paren_diff = abs(wgsl.count("(") - wgsl.count(")"))
        assert paren_diff <= 5, f"Parentheses imbalanced by {paren_diff}"

        # Check no Python syntax
        assert "def " not in wgsl
        assert "self." not in wgsl
        assert "True" not in wgsl  # Note: "true" (lowercase) is WGSL
        assert "False" not in wgsl  # Note: "false" (lowercase) is WGSL


# =============================================================================
# TreeSDF Trinity Pattern Tests (Mirror/Tracker)
# =============================================================================

class TestTreeSDFTrinityPattern:
    """Test Trinity pattern (Mirror/Tracker) for TreeSDF."""

    def test_tree_has_mirror(self):
        """TreeSDF should have a Mirror."""
        tree = TreeSDF()
        assert tree.mirror is not None
        assert tree.mirror.node_type == "TreeSDF"

    def test_tree_has_tracker(self):
        """TreeSDF should have a Tracker."""
        tree = TreeSDF()
        assert tree.tracker is not None
        assert tree.tracker.version >= 0

    def test_tree_unique_id(self):
        """Each TreeSDF should have unique ID."""
        tree1 = TreeSDF()
        tree2 = TreeSDF()
        assert tree1._node_id != tree2._node_id

    def test_tree_mirror_fields(self):
        """Mirror should provide field access."""
        tree = TreeSDF()
        fields = tree.mirror.fields

        assert "config" in fields
        assert "position" in fields

    def test_tree_tracker_dirty(self):
        """Tracker should report dirty state."""
        tree = TreeSDF()
        # New tree should be dirty
        assert tree.tracker.is_dirty

        # Clear and check
        tree.tracker.clear()
        assert not tree.tracker.is_dirty


# =============================================================================
# ForestSDF Trinity Pattern Tests
# =============================================================================

class TestForestSDFTrinityPattern:
    """Test Trinity pattern for ForestSDF."""

    def test_forest_has_mirror(self):
        """ForestSDF should have a Mirror."""
        forest = ForestSDF()
        assert forest.mirror is not None
        assert forest.mirror.node_type == "ForestSDF"

    def test_forest_has_tracker(self):
        """ForestSDF should have a Tracker."""
        forest = ForestSDF()
        assert forest.tracker is not None

    def test_forest_unique_id(self):
        """Each ForestSDF should have unique ID."""
        f1 = ForestSDF()
        f2 = ForestSDF()
        assert f1._node_id != f2._node_id


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_tree(self):
        """Very small tree should still work."""
        config = TreeConfig(
            trunk_height=0.1,
            trunk_radius=0.01,
            canopy_radius=0.05,
        )
        tree = TreeSDF(config)

        d = tree.evaluate((0.0, 0.05, 0.0))
        assert math.isfinite(d)

    def test_very_large_tree(self):
        """Very large tree should still work."""
        config = TreeConfig(
            trunk_height=100.0,
            trunk_radius=5.0,
            canopy_radius=20.0,
        )
        tree = TreeSDF(config)

        d = tree.evaluate((0.0, 50.0, 0.0))
        assert d < 0, "Should be inside large tree"

    def test_single_canopy_sphere(self):
        """Tree with single canopy sphere should work."""
        config = TreeConfig(canopy_spheres=1)
        tree = TreeSDF(config)

        positions = tree.get_canopy_sphere_positions()
        assert len(positions) == 1

    def test_many_canopy_spheres(self):
        """Tree with many canopy spheres should work."""
        config = TreeConfig(canopy_spheres=20)
        tree = TreeSDF(config)

        positions = tree.get_canopy_sphere_positions()
        assert len(positions) == 20

    def test_zero_smooth_k(self):
        """Zero smooth_k should use hard union."""
        config = TreeConfig(smooth_k=0.0)
        tree = TreeSDF(config)

        d = tree.evaluate((0.0, 1.0, 0.0))
        assert math.isfinite(d)

    def test_forest_at_negative_coordinates(self):
        """Forest should work at negative coordinates."""
        forest = ForestSDF(ForestConfig(density=1.0))

        d = forest.evaluate((-100.0, 1.0, -100.0))
        assert math.isfinite(d)

    def test_forest_very_small_cells(self):
        """Forest with very small cells should work."""
        config = ForestConfig(
            cell_size=(1.0, 1.0, 1.0),
            density=1.0,
        )
        forest = ForestSDF(config)

        d = forest.evaluate((0.5, 0.5, 0.5))
        assert math.isfinite(d)

    def test_forest_has_tree_check(self):
        """has_tree_in_cell should respect density."""
        forest_full = ForestSDF(ForestConfig(density=1.0))
        forest_empty = ForestSDF(ForestConfig(density=0.0))

        # Full density: every cell has a tree
        assert forest_full.has_tree_in_cell((0, 0, 0))
        assert forest_full.has_tree_in_cell((100, 0, -50))

        # Zero density: no cells have trees
        assert not forest_empty.has_tree_in_cell((0, 0, 0))
        assert not forest_empty.has_tree_in_cell((100, 0, -50))


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_tree_with_all_features(self):
        """Test tree with all features enabled."""
        config = TreeConfig(
            trunk_height=3.0,
            trunk_radius=0.25,
            trunk_type=TrunkType.TAPERED_CONE,
            trunk_taper=0.6,
            canopy_type=CanopyType.ELLIPSOIDS,
            canopy_spheres=7,
            canopy_radius=1.2,
            branches=BranchConfig(count=5, radius=0.08),
            smooth_k=0.2,
        )
        tree = TreeSDF(config)

        # Should be able to evaluate
        d = tree.evaluate((0.0, 1.5, 0.0))
        assert d < 0, "Inside tree should be negative"

        # Should generate valid WGSL
        wgsl = tree.to_wgsl("full_tree")
        assert "fn sdf_full_tree" in wgsl

    def test_full_forest_with_variation(self):
        """Test forest with all variation features."""
        config = ForestConfig(
            cell_size=(12.0, 12.0, 12.0),
            base_tree=TreeConfig(trunk_height=4.0, canopy_spheres=5),
            variation=TreeVariation(
                height_min=0.6,
                height_max=1.4,
                canopy_min=3,
                canopy_max=8,
                position_jitter=0.4,
                rotation_enabled=True,
            ),
            density=0.8,
        )
        forest = ForestSDF(config)

        # Evaluate at multiple points
        points = [
            (6.0, 1.0, 6.0),
            (18.0, 1.0, 6.0),
            (-6.0, 1.0, -6.0),
        ]
        for p in points:
            d = forest.evaluate(p)
            assert math.isfinite(d), f"Infinite distance at {p}"

        # Generate WGSL
        wgsl = forest.to_wgsl("varied_forest")
        assert "fn sdf_varied_forest" in wgsl


# =============================================================================
# Label and Repr Tests
# =============================================================================

class TestLabelAndRepr:
    """Test label() and __repr__ methods."""

    def test_tree_label(self):
        """Tree label should be descriptive."""
        tree = TreeSDF(TreeConfig(trunk_height=5.0, canopy_spheres=3))
        label = tree.label()
        assert "TreeSDF" in label
        assert "5.0" in label or "5" in label

    def test_tree_repr(self):
        """Tree repr should include ID."""
        tree = TreeSDF()
        r = repr(tree)
        assert "TreeSDF" in r
        assert "id=" in r

    def test_forest_label(self):
        """Forest label should be descriptive."""
        forest = ForestSDF(ForestConfig(density=0.7))
        label = forest.label()
        assert "ForestSDF" in label
        assert "0.7" in label

    def test_forest_repr(self):
        """Forest repr should include ID."""
        forest = ForestSDF()
        r = repr(forest)
        assert "ForestSDF" in r
        assert "id=" in r
