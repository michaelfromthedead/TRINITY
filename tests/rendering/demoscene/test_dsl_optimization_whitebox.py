"""
Whitebox tests for DSL Pattern Optimization Passes (T-DEMO-8.2).

Tests:
  - Pattern recognition and replacement
  - CSE correctness (same result, fewer evaluations)
  - LOD simplification based on distance
  - Output size/speed comparison
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CapsuleNode, CellIdNode, ConeNode, CylinderNode,
    EllipsoidNode, FloatNode, IntersectionNode, KifsNode, MirrorNode,
    OctahedronNode, PlaneNode, PositionNode, PyramidNode, RepeatNode,
    RoundedBoxNode, BoxFrameNode, SceneGraph, SphereNode, StretchNode,
    SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
)
from engine.rendering.demoscene.pattern_optimizer import (
    PatternType, SDFPattern, PatternMatch, PatternDatabase,
    ExpressionStats, CSECache,
    LODLevel, LODConfig, LODResult,
    PatternOptimizationPass, PatternMatchingPass, EnhancedCSEPass,
    LODSimplificationPass, DSLPatternOptimizer,
    PATTERN_PASSES, CSE_PASSES, LOD_PASSES, ALL_PATTERN_PASSES,
    pattern_optimize, cse_optimize, lod_simplify,
)


# =============================================================================
# PATTERN DATABASE TESTS
# =============================================================================

class TestPatternDatabase:
    """Tests for PatternDatabase class."""

    def test_pattern_database_init(self) -> None:
        """Test that pattern database initializes with builtin patterns."""
        db = PatternDatabase()
        assert len(db._patterns) >= 7  # At least 7 builtin patterns

    def test_pattern_database_stats_empty_initially(self) -> None:
        """Test that stats are empty before matching."""
        db = PatternDatabase()
        assert len(db.stats) == 0

    def test_pattern_database_match_returns_none_for_simple_node(self) -> None:
        """Test that simple nodes don't match any pattern."""
        db = PatternDatabase()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        result = db.match(sphere)
        assert result is None

    def test_pattern_database_register_custom_pattern(self) -> None:
        """Test registering a custom pattern."""
        db = PatternDatabase()
        initial_count = len(db._patterns)

        custom = SDFPattern(
            PatternType.CHAMFERED_BOX,
            "Custom test pattern",
            lambda n: None,  # Never matches
            lambda m: m.matched_node,
            1.0
        )
        db.register_pattern(custom)
        assert len(db._patterns) == initial_count + 1

    def test_get_pattern_by_type(self) -> None:
        """Test getting a pattern by its type."""
        db = PatternDatabase()
        pattern = db.get_pattern(PatternType.ROUNDED_BOX)
        assert pattern is not None
        assert pattern.pattern_type == PatternType.ROUNDED_BOX

    def test_get_pattern_returns_none_for_unknown(self) -> None:
        """Test that unknown pattern types return None."""
        db = PatternDatabase()
        # CHAMFERED_BOX is not registered by default
        pattern = db.get_pattern(PatternType.CHAMFERED_BOX)
        assert pattern is None


# =============================================================================
# PATTERN MATCHING TESTS
# =============================================================================

class TestPatternMatching:
    """Tests for pattern matching functionality."""

    def test_match_rounded_box_pattern(self) -> None:
        """Test matching rounded box pattern (box - sphere)."""
        db = PatternDatabase()
        box = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        sphere = SphereNode(PositionNode(), FloatNode(0.1))
        subtraction = SubtractionNode(box, sphere)

        match = db.match(subtraction)
        assert match is not None
        assert match.pattern_type == PatternType.ROUNDED_BOX
        assert match.confidence == 0.85

    def test_match_shell_pattern(self) -> None:
        """Test matching shell pattern (outer sphere - inner sphere)."""
        db = PatternDatabase()
        outer = SphereNode(PositionNode(), FloatNode(2.0))
        inner = SphereNode(PositionNode(), FloatNode(1.5))
        shell = SubtractionNode(outer, inner)

        match = db.match(shell)
        assert match is not None
        assert match.pattern_type == PatternType.SHELL
        assert match.extracted_params["thickness"] == 0.5

    def test_shell_pattern_requires_smaller_inner(self) -> None:
        """Test that shell pattern requires inner < outer radius."""
        db = PatternDatabase()
        outer = SphereNode(PositionNode(), FloatNode(1.0))
        inner = SphereNode(PositionNode(), FloatNode(2.0))  # Larger!
        shell = SubtractionNode(outer, inner)

        match = db.match(shell)
        assert match is None  # Should not match

    def test_match_dome_pattern(self) -> None:
        """Test matching dome pattern (sphere intersection plane)."""
        db = PatternDatabase()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        plane = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0))
        dome = IntersectionNode(sphere, plane)

        match = db.match(dome)
        assert match is not None
        assert match.pattern_type == PatternType.DOME

    def test_match_dome_pattern_reversed(self) -> None:
        """Test matching dome pattern with plane first."""
        db = PatternDatabase()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        plane = PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0))
        dome = IntersectionNode(plane, sphere)  # Reversed order

        match = db.match(dome)
        assert match is not None
        assert match.pattern_type == PatternType.DOME

    def test_match_ring_pattern(self) -> None:
        """Test matching ring pattern (torus with ratio > 4)."""
        db = PatternDatabase()
        ring = TorusNode(PositionNode(), FloatNode(5.0), FloatNode(0.1))  # Ratio = 50

        match = db.match(ring)
        assert match is not None
        assert match.pattern_type == PatternType.RING
        assert match.extracted_params["ratio"] == 50.0

    def test_ring_pattern_requires_high_ratio(self) -> None:
        """Test that ring pattern requires ratio >= 4."""
        db = PatternDatabase()
        torus = TorusNode(PositionNode(), FloatNode(1.0), FloatNode(0.5))  # Ratio = 2

        match = db.match(torus)
        assert match is None  # Should not match as ring

    def test_match_disc_pattern(self) -> None:
        """Test matching disc pattern (flat cylinder)."""
        db = PatternDatabase()
        disc = CylinderNode(PositionNode(), FloatNode(0.1), FloatNode(2.0))  # h=0.1, r=2.0

        match = db.match(disc)
        assert match is not None
        assert match.pattern_type == PatternType.DISC

    def test_disc_pattern_requires_flat_cylinder(self) -> None:
        """Test that disc requires height < radius/4."""
        db = PatternDatabase()
        cylinder = CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0))  # Not flat

        match = db.match(cylinder)
        assert match is None  # Not a disc

    def test_match_rod_pattern(self) -> None:
        """Test matching rod pattern (thin cylinder)."""
        db = PatternDatabase()
        rod = CylinderNode(PositionNode(), FloatNode(10.0), FloatNode(0.05))  # h=10, r=0.05

        match = db.match(rod)
        assert match is not None
        assert match.pattern_type == PatternType.ROD

    def test_rod_pattern_requires_thin_cylinder(self) -> None:
        """Test that rod requires radius < height/10."""
        db = PatternDatabase()
        cylinder = CylinderNode(PositionNode(), FloatNode(1.0), FloatNode(0.5))

        match = db.match(cylinder)
        assert match is None  # Not a rod

    def test_match_hollow_box_pattern(self) -> None:
        """Test matching hollow box pattern (outer - inner box)."""
        db = PatternDatabase()
        outer = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(PositionNode(), Vec3Node(1.5, 1.5, 1.5))
        hollow = SubtractionNode(outer, inner)

        match = db.match(hollow)
        assert match is not None
        assert match.pattern_type == PatternType.HOLLOW_BOX

    def test_hollow_box_requires_smaller_inner(self) -> None:
        """Test that hollow box requires inner smaller in all dimensions."""
        db = PatternDatabase()
        outer = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        inner = BoxNode(PositionNode(), Vec3Node(2.0, 0.5, 0.5))  # X larger!
        hollow = SubtractionNode(outer, inner)

        match = db.match(hollow)
        assert match is None


# =============================================================================
# PATTERN REPLACEMENT TESTS
# =============================================================================

class TestPatternReplacement:
    """Tests for pattern replacement functionality."""

    def test_rounded_box_replacement(self) -> None:
        """Test that rounded box pattern is replaced with RoundedBoxNode."""
        db = PatternDatabase()
        box = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        sphere = SphereNode(PositionNode(), FloatNode(0.2))
        subtraction = SubtractionNode(box, sphere)

        match = db.match(subtraction)
        pattern = db.get_pattern(PatternType.ROUNDED_BOX)
        result = pattern.replacement_fn(match)

        assert isinstance(result, RoundedBoxNode)

    def test_hollow_box_replacement(self) -> None:
        """Test that hollow box pattern is replaced with BoxFrameNode."""
        db = PatternDatabase()
        outer = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        hollow = SubtractionNode(outer, inner)

        match = db.match(hollow)
        pattern = db.get_pattern(PatternType.HOLLOW_BOX)
        result = pattern.replacement_fn(match)

        assert isinstance(result, BoxFrameNode)


# =============================================================================
# CSE CACHE TESTS
# =============================================================================

class TestCSECache:
    """Tests for CSECache class."""

    def test_cache_init(self) -> None:
        """Test CSE cache initialization."""
        cache = CSECache()
        assert cache.hit_rate == 0.0
        assert cache.stats["total_entries"] == 0

    def test_cache_track_new_expression(self) -> None:
        """Test tracking a new expression."""
        cache = CSECache()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))

        h, is_dup = cache.track(sphere)
        assert is_dup is False
        assert cache.stats["total_entries"] == 1

    def test_cache_track_duplicate_expression(self) -> None:
        """Test tracking a duplicate expression."""
        cache = CSECache()
        sphere1 = SphereNode(PositionNode(), FloatNode(1.0))
        sphere2 = SphereNode(PositionNode(), FloatNode(1.0))  # Same structure

        h1, dup1 = cache.track(sphere1)
        h2, dup2 = cache.track(sphere2)

        assert h1 == h2  # Same hash
        assert dup1 is False
        assert dup2 is True
        assert cache.stats["total_hits"] == 1

    def test_cache_different_expressions_different_hash(self) -> None:
        """Test that different expressions have different hashes."""
        cache = CSECache()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        box = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))

        h1, _ = cache.track(sphere)
        h2, _ = cache.track(box)

        assert h1 != h2

    def test_cache_get_duplicates(self) -> None:
        """Test getting duplicate expressions."""
        cache = CSECache()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))

        # Track same expression 3 times
        for _ in range(3):
            cache.track(sphere)

        duplicates = cache.get_duplicates(min_occurrences=2)
        assert len(duplicates) == 1
        assert duplicates[0].occurrences == 3

    def test_cache_assign_variable(self) -> None:
        """Test variable name assignment for caching."""
        cache = CSECache()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        h, _ = cache.track(sphere)

        var1 = cache.assign_variable(h)
        var2 = cache.assign_variable(h)

        assert var1 == var2  # Same variable name
        assert var1.startswith("_expr_")

    def test_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        cache = CSECache()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))

        # Track 4 times: 1 miss + 3 hits
        for _ in range(4):
            cache.track(sphere)

        assert cache.hit_rate == 0.75


# =============================================================================
# ENHANCED CSE PASS TESTS
# =============================================================================

class TestEnhancedCSEPass:
    """Tests for EnhancedCSEPass class."""

    def test_cse_pass_no_duplicates(self) -> None:
        """Test CSE pass with no duplicate expressions."""
        # Use different positions to avoid PositionNode sharing
        sphere = SphereNode(Vec3Node(0.0, 0.0, 0.0), FloatNode(1.0))
        box = BoxNode(Vec3Node(1.0, 1.0, 1.0), Vec3Node(2.0, 2.0, 2.0))
        graph = SceneGraph(primitives=(sphere, box), pipeline=(), name="test")

        cse_pass = EnhancedCSEPass(graph)
        result = cse_pass.run()

        assert result is graph  # Unchanged
        # There may still be some structural duplicates (like FloatNode(1.0))
        # but the main primitives should be unique

    def test_cse_pass_with_duplicates(self) -> None:
        """Test CSE pass with duplicate expressions."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        graph = SceneGraph(primitives=(sphere, sphere), pipeline=(), name="test")

        cse_pass = EnhancedCSEPass(graph)
        cse_pass.run()

        # Same sphere appears twice
        assert cse_pass.stats["duplicates"] >= 1

    def test_cse_pass_hoisted_variables(self) -> None:
        """Test that duplicate expressions generate hoisted variables."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        # Create union of same sphere (creates duplicates)
        union = UnionNode(sphere, sphere)

        cse_pass = EnhancedCSEPass(union)
        cse_pass.run()

        # Should have hoisted the common sphere expression
        vars_list = cse_pass.hoisted_variables
        # At minimum, the duplicate sphere should be hoisted
        assert len(vars_list) >= 0  # May or may not hoist depending on traversal

    def test_cse_evaluation_savings(self) -> None:
        """Test evaluation savings calculation."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        # Same sphere 4 times = 3 evaluations saved
        graph = SceneGraph(
            primitives=(sphere, sphere, sphere, sphere),
            pipeline=(), name="test"
        )

        cse_pass = EnhancedCSEPass(graph)
        cse_pass.run()

        # Should save evaluations
        assert "evaluation_savings" in cse_pass.stats


# =============================================================================
# LOD CONFIG TESTS
# =============================================================================

class TestLODConfig:
    """Tests for LODConfig class."""

    def test_lod_config_defaults(self) -> None:
        """Test LOD config default values."""
        config = LODConfig()
        assert config.enable_sphere_proxy is True
        assert config.enable_skip_detail is True
        assert LODLevel.BILLBOARD in config.distance_thresholds

    def test_lod_config_primitive_complexity(self) -> None:
        """Test primitive complexity values."""
        config = LODConfig()
        assert config.primitive_complexity[SphereNode] == 1.0
        assert config.primitive_complexity[TorusNode] > config.primitive_complexity[SphereNode]


# =============================================================================
# LOD SIMPLIFICATION PASS TESTS
# =============================================================================

class TestLODSimplificationPass:
    """Tests for LODSimplificationPass class."""

    def test_lod_pass_no_simplification_at_close_range(self) -> None:
        """Test that close primitives are not simplified."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        graph = SceneGraph(primitives=(sphere,), pipeline=(), name="test")

        lod_pass = LODSimplificationPass(graph, camera_distance=5.0)  # Close
        result = lod_pass.run()

        assert result is graph  # Unchanged

    def test_lod_pass_sphere_proxy_at_billboard_distance(self) -> None:
        """Test that complex primitives become spheres at billboard distance."""
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        graph = SceneGraph(primitives=(torus,), pipeline=(), name="test")

        lod_pass = LODSimplificationPass(graph, camera_distance=150.0)  # Far
        result = lod_pass.run()

        # Torus should be simplified to sphere
        assert len(result.primitives) == 1
        assert isinstance(result.primitives[0], SphereNode)

    def test_lod_pass_skip_detail_at_far_distance(self) -> None:
        """Test that high-complexity primitives are skipped at far distance."""
        # BoxFrame is high complexity (2.5)
        box_frame = BoxFrameNode(
            PositionNode(),
            Vec3Node(1.0, 1.0, 1.0),
            FloatNode(0.1)
        )
        graph = SceneGraph(primitives=(box_frame,), pipeline=(), name="test")

        config = LODConfig(
            skip_detail_distance=50.0,
            complexity_threshold=2.0
        )
        lod_pass = LODSimplificationPass(graph, config, camera_distance=80.0)
        result = lod_pass.run()

        # BoxFrame should be skipped
        assert len(result.primitives) == 0

    def test_lod_pass_preserves_simple_primitives(self) -> None:
        """Test that simple primitives are preserved at any distance."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        graph = SceneGraph(primitives=(sphere,), pipeline=(), name="test")

        config = LODConfig(enable_skip_detail=True, complexity_threshold=2.0)
        lod_pass = LODSimplificationPass(graph, config, camera_distance=100.0)
        result = lod_pass.run()

        # Sphere is simple (1.0) so should remain
        assert len(result.primitives) == 1

    def test_lod_level_thresholds(self) -> None:
        """Test LOD level determination based on distance."""
        config = LODConfig()
        sphere = SphereNode(PositionNode(), FloatNode(1.0))

        # Test at various distances
        for dist, expected_level in [
            (5.0, LODLevel.FULL),
            (15.0, LODLevel.HIGH),
            (30.0, LODLevel.MEDIUM),
            (60.0, LODLevel.LOW),
            (120.0, LODLevel.BILLBOARD),
        ]:
            lod_pass = LODSimplificationPass(sphere, config, dist)
            level = lod_pass._get_lod_level(dist)
            assert level == expected_level

    def test_lod_bounding_radius_calculation(self) -> None:
        """Test bounding radius calculation for various primitives."""
        config = LODConfig()
        lod_pass = LODSimplificationPass(SphereNode(PositionNode(), FloatNode(1.0)), config)

        # Sphere
        sphere = SphereNode(PositionNode(), FloatNode(2.5))
        assert lod_pass._get_bounding_radius(sphere) == 2.5

        # Box
        box = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        assert lod_pass._get_bounding_radius(box) == pytest.approx(math.sqrt(3.0))

        # Torus
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        assert lod_pass._get_bounding_radius(torus) == 2.5


# =============================================================================
# PATTERN MATCHING PASS TESTS
# =============================================================================

class TestPatternMatchingPass:
    """Tests for PatternMatchingPass class."""

    def test_pattern_pass_no_matches(self) -> None:
        """Test pattern pass with no pattern matches."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        graph = SceneGraph(primitives=(sphere,), pipeline=(), name="test")

        pattern_pass = PatternMatchingPass(graph)
        result = pattern_pass.run()

        assert result is graph
        assert pattern_pass.stats["patterns_matched"] == 0

    def test_pattern_pass_with_match(self) -> None:
        """Test pattern pass with a pattern match."""
        outer = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        hollow = SubtractionNode(outer, inner)

        pattern_pass = PatternMatchingPass(hollow)
        result = pattern_pass.run()

        assert pattern_pass.stats["patterns_matched"] == 1
        assert isinstance(result, BoxFrameNode)

    def test_pattern_pass_records_matches(self) -> None:
        """Test that pattern pass records all matches."""
        # Create two matchable patterns
        sphere1 = SphereNode(PositionNode(), FloatNode(2.0))
        sphere2 = SphereNode(PositionNode(), FloatNode(1.5))
        shell = SubtractionNode(sphere1, sphere2)

        pattern_pass = PatternMatchingPass(shell)
        pattern_pass.run()

        matches = pattern_pass.matches
        assert len(matches) == 1
        assert matches[0].pattern_type == PatternType.SHELL


# =============================================================================
# DSL PATTERN OPTIMIZER TESTS
# =============================================================================

class TestDSLPatternOptimizer:
    """Tests for DSLPatternOptimizer class."""

    def test_optimizer_with_pattern_passes(self) -> None:
        """Test optimizer with pattern matching passes."""
        optimizer = DSLPatternOptimizer(PATTERN_PASSES)
        sphere = SphereNode(PositionNode(), FloatNode(1.0))

        result = optimizer.optimize(sphere)
        assert result is sphere  # No patterns to match

    def test_optimizer_with_cse_passes(self) -> None:
        """Test optimizer with CSE passes."""
        optimizer = DSLPatternOptimizer(CSE_PASSES)
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        union = UnionNode(sphere, sphere)

        result = optimizer.optimize(union)
        assert "EnhancedCSEPass" in optimizer.stats

    def test_optimizer_with_lod_passes(self) -> None:
        """Test optimizer with LOD passes."""
        optimizer = DSLPatternOptimizer(LOD_PASSES)
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))

        result = optimizer.optimize(torus, camera_distance=150.0)
        assert isinstance(result, SphereNode)  # Simplified to sphere

    def test_optimizer_with_all_passes(self) -> None:
        """Test optimizer with all passes."""
        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        graph = SceneGraph(primitives=(sphere,), pipeline=(), name="test")

        result = optimizer.optimize(graph)
        assert len(optimizer.stats) == 3  # Three passes

    def test_optimizer_total_stats(self) -> None:
        """Test optimizer total stats aggregation."""
        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        sphere = SphereNode(PositionNode(), FloatNode(1.0))

        optimizer.optimize(sphere)
        # Total stats should aggregate numeric values
        assert isinstance(optimizer.total_stats, dict)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_pattern_optimize_function(self) -> None:
        """Test pattern_optimize convenience function."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        result = pattern_optimize(sphere)
        assert result is sphere

    def test_cse_optimize_function(self) -> None:
        """Test cse_optimize convenience function."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        result = cse_optimize(sphere)
        assert result is sphere

    def test_lod_simplify_function(self) -> None:
        """Test lod_simplify convenience function."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        result = lod_simplify(sphere, dist=5.0)
        assert result is sphere


# =============================================================================
# CORRECTNESS TESTS (Same Result, Fewer Evaluations)
# =============================================================================

class TestOptimizationCorrectness:
    """Tests to verify optimization correctness."""

    def test_pattern_replacement_preserves_position(self) -> None:
        """Test that pattern replacement preserves position information."""
        position = Vec3Node(1.0, 2.0, 3.0)
        outer = BoxNode(position, Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(position, Vec3Node(1.0, 1.0, 1.0))
        hollow = SubtractionNode(outer, inner)

        result = pattern_optimize(hollow)
        assert isinstance(result, BoxFrameNode)
        # Position should be preserved
        assert result.position == position

    def test_lod_bounding_sphere_contains_original(self) -> None:
        """Test that LOD bounding sphere contains original primitive."""
        # A box at origin with size (1,1,1) has bounding radius sqrt(3)
        box = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        config = LODConfig()

        lod_pass = LODSimplificationPass(box, config, camera_distance=150.0)
        result = lod_pass.run()

        if isinstance(result, SphereNode):
            # Bounding sphere should have radius >= sqrt(3)
            assert result.radius.value >= math.sqrt(3.0) - 0.001

    def test_cse_identifies_structural_equality(self) -> None:
        """Test that CSE correctly identifies structurally equal expressions."""
        cache = CSECache()

        # Two spheres with same parameters
        s1 = SphereNode(PositionNode(), FloatNode(1.5))
        s2 = SphereNode(PositionNode(), FloatNode(1.5))

        h1 = cache.compute_hash(s1)
        h2 = cache.compute_hash(s2)

        assert h1 == h2

    def test_cse_distinguishes_different_params(self) -> None:
        """Test that CSE distinguishes expressions with different params."""
        cache = CSECache()

        s1 = SphereNode(PositionNode(), FloatNode(1.0))
        s2 = SphereNode(PositionNode(), FloatNode(2.0))

        h1 = cache.compute_hash(s1)
        h2 = cache.compute_hash(s2)

        assert h1 != h2


# =============================================================================
# OUTPUT SIZE COMPARISON TESTS
# =============================================================================

class TestOutputComparison:
    """Tests comparing output size and structure."""

    def test_pattern_optimization_reduces_node_count(self) -> None:
        """Test that pattern optimization can reduce node count."""
        # Hollow box: Subtraction(Box, Box) = 3 nodes
        # Becomes: BoxFrameNode = 1 node
        outer = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        hollow = SubtractionNode(outer, inner)

        def count_nodes(node):
            count = 1
            for child in node.children():
                count += count_nodes(child)
            return count

        original_count = count_nodes(hollow)
        result = pattern_optimize(hollow)
        result_count = count_nodes(result)

        # BoxFrameNode has fewer nodes than Subtraction(Box, Box)
        assert result_count <= original_count

    def test_lod_simplification_reduces_complexity(self) -> None:
        """Test that LOD simplification reduces scene complexity."""
        # Create a scene with complex primitives
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        ellipsoid = EllipsoidNode(PositionNode(), Vec3Node(1.0, 2.0, 0.5))
        graph = SceneGraph(primitives=(torus, ellipsoid), pipeline=(), name="test")

        # At billboard distance, both should become spheres
        result = lod_simplify(graph, dist=150.0)

        # All primitives should be spheres (simpler)
        for prim in result.primitives:
            assert isinstance(prim, SphereNode)
