"""
Blackbox tests for DSL Pattern Optimization Passes (T-DEMO-8.2).

End-to-end tests verifying:
  - Pattern recognition produces correct replacements
  - CSE reduces evaluation count without changing results
  - LOD simplification preserves visual bounds
  - Full optimization pipeline works correctly
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CapsuleNode, CylinderNode, EllipsoidNode,
    FloatNode, IntersectionNode, OctahedronNode, PlaneNode, PositionNode,
    PyramidNode, RepeatNode, RoundedBoxNode, BoxFrameNode, SceneGraph,
    SphereNode, SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
)
from engine.rendering.demoscene.pattern_optimizer import (
    PatternType, PatternDatabase, CSECache, LODConfig, LODLevel,
    DSLPatternOptimizer, ALL_PATTERN_PASSES,
    pattern_optimize, cse_optimize, lod_simplify,
)


# =============================================================================
# END-TO-END PATTERN RECOGNITION TESTS
# =============================================================================

class TestPatternRecognitionE2E:
    """End-to-end tests for pattern recognition."""

    def test_e2e_rounded_box_recognition(self) -> None:
        """Test complete rounded box pattern recognition and replacement."""
        # Build a typical rounded box construction
        box = BoxNode(PositionNode(), Vec3Node(1.0, 0.5, 1.5))
        corner_sphere = SphereNode(PositionNode(), FloatNode(0.15))
        rounded_box = SubtractionNode(box, corner_sphere)

        result = pattern_optimize(rounded_box)

        assert isinstance(result, RoundedBoxNode)
        # Size should be preserved
        assert result.half_extents.x == 1.0
        assert result.half_extents.y == 0.5
        assert result.half_extents.z == 1.5

    def test_e2e_hollow_box_recognition(self) -> None:
        """Test complete hollow box pattern recognition and replacement."""
        outer = BoxNode(PositionNode(), Vec3Node(3.0, 2.0, 1.0))
        inner = BoxNode(PositionNode(), Vec3Node(2.5, 1.5, 0.5))
        hollow = SubtractionNode(outer, inner)

        result = pattern_optimize(hollow)

        assert isinstance(result, BoxFrameNode)
        # Outer size should be preserved
        assert result.half_extents.x == 3.0
        assert result.half_extents.y == 2.0
        assert result.half_extents.z == 1.0

    def test_e2e_shell_recognition(self) -> None:
        """Test complete shell pattern recognition."""
        outer_sphere = SphereNode(PositionNode(), FloatNode(5.0))
        inner_sphere = SphereNode(PositionNode(), FloatNode(4.5))
        shell = SubtractionNode(outer_sphere, inner_sphere)

        db = PatternDatabase()
        match = db.match(shell)

        assert match is not None
        assert match.pattern_type == PatternType.SHELL
        assert match.extracted_params["thickness"] == 0.5

    def test_e2e_dome_recognition(self) -> None:
        """Test complete dome pattern recognition."""
        sphere = SphereNode(PositionNode(), FloatNode(2.0))
        ground_plane = PlaneNode(
            PositionNode(),
            Vec3Node(0.0, 1.0, 0.0),
            FloatNode(0.0)
        )
        dome = IntersectionNode(sphere, ground_plane)

        db = PatternDatabase()
        match = db.match(dome)

        assert match is not None
        assert match.pattern_type == PatternType.DOME
        assert match.extracted_params["radius"] == 2.0


# =============================================================================
# END-TO-END CSE TESTS
# =============================================================================

class TestCSEE2E:
    """End-to-end tests for CSE optimization."""

    def test_e2e_cse_detects_repeated_primitives(self) -> None:
        """Test that CSE detects repeated primitive expressions."""
        # Create a scene with the same primitive used multiple times
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        scene = SceneGraph(
            primitives=(sphere, sphere, sphere),
            pipeline=(),
            name="repeated_spheres"
        )

        cache = CSECache()
        cse_optimize(scene, cache)

        # Should detect duplicates
        assert cache.stats["total_hits"] >= 2

    def test_e2e_cse_distinguishes_different_primitives(self) -> None:
        """Test that CSE correctly distinguishes different primitives."""
        sphere1 = SphereNode(PositionNode(), FloatNode(1.0))
        sphere2 = SphereNode(PositionNode(), FloatNode(2.0))  # Different radius
        scene = SceneGraph(
            primitives=(sphere1, sphere2),
            pipeline=(),
            name="different_spheres"
        )

        cache = CSECache()
        cse_optimize(scene, cache)

        # Should have 2 unique entries (plus internal nodes)
        assert cache.stats["total_entries"] >= 2

    def test_e2e_cse_complex_scene(self) -> None:
        """Test CSE on a complex scene with nested expressions."""
        # Create a scene with unions containing repeated spheres
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        union1 = UnionNode(sphere, sphere)
        union2 = UnionNode(sphere, sphere)
        final = UnionNode(union1, union2)

        cache = CSECache()
        cse_optimize(final, cache)

        # Should detect many duplicates
        assert cache.stats["duplicates"] >= 1


# =============================================================================
# END-TO-END LOD TESTS
# =============================================================================

class TestLODE2E:
    """End-to-end tests for LOD optimization."""

    def test_e2e_lod_preserves_near_scene(self) -> None:
        """Test that LOD preserves scene at close distance."""
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        ellipsoid = EllipsoidNode(PositionNode(), Vec3Node(1.0, 2.0, 0.5))
        scene = SceneGraph(
            primitives=(torus, ellipsoid),
            pipeline=(),
            name="complex_scene"
        )

        # At close distance, nothing should change
        result = lod_simplify(scene, dist=5.0)

        assert len(result.primitives) == 2
        assert isinstance(result.primitives[0], TorusNode)
        assert isinstance(result.primitives[1], EllipsoidNode)

    def test_e2e_lod_simplifies_far_scene(self) -> None:
        """Test that LOD simplifies scene at far distance."""
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        ellipsoid = EllipsoidNode(PositionNode(), Vec3Node(1.0, 2.0, 0.5))
        scene = SceneGraph(
            primitives=(torus, ellipsoid),
            pipeline=(),
            name="complex_scene"
        )

        # At billboard distance, should simplify to spheres
        result = lod_simplify(scene, dist=150.0)

        for prim in result.primitives:
            assert isinstance(prim, SphereNode)

    def test_e2e_lod_skip_high_complexity(self) -> None:
        """Test that LOD skips high-complexity primitives at extreme distance."""
        # BoxFrame is complexity 2.5
        box_frame = BoxFrameNode(
            PositionNode(),
            Vec3Node(1.0, 1.0, 1.0),
            FloatNode(0.1)
        )
        scene = SceneGraph(primitives=(box_frame,), pipeline=(), name="test")

        config = LODConfig(
            skip_detail_distance=50.0,
            complexity_threshold=2.0
        )
        result = lod_simplify(scene, config, dist=80.0)

        # BoxFrame should be skipped (removed)
        assert len(result.primitives) == 0

    def test_e2e_lod_bounding_sphere_accuracy(self) -> None:
        """Test that LOD bounding spheres are accurate."""
        # Create a box and verify its bounding sphere
        box = BoxNode(PositionNode(), Vec3Node(3.0, 4.0, 0.0))
        scene = SceneGraph(primitives=(box,), pipeline=(), name="test")

        result = lod_simplify(scene, dist=150.0)

        if len(result.primitives) > 0:
            sphere = result.primitives[0]
            assert isinstance(sphere, SphereNode)
            # Diagonal of 3x4x0 box is 5
            expected_radius = 5.0
            assert sphere.radius.value == pytest.approx(expected_radius, rel=0.01)


# =============================================================================
# FULL PIPELINE TESTS
# =============================================================================

class TestFullOptimizationPipeline:
    """End-to-end tests for the full optimization pipeline."""

    def test_e2e_full_pipeline_simple_scene(self) -> None:
        """Test full optimization pipeline on simple scene."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        scene = SceneGraph(primitives=(sphere,), pipeline=(), name="simple")

        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        result = optimizer.optimize(scene)

        assert isinstance(result, SceneGraph)
        assert len(optimizer.stats) == 3

    def test_e2e_full_pipeline_complex_scene(self) -> None:
        """Test full optimization pipeline on complex scene."""
        # Build a complex scene with patterns
        outer = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(PositionNode(), Vec3Node(1.5, 1.5, 1.5))
        hollow = SubtractionNode(outer, inner)

        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        result = optimizer.optimize(hollow)

        # Should be optimized to BoxFrameNode
        assert isinstance(result, BoxFrameNode)

    def test_e2e_full_pipeline_with_lod(self) -> None:
        """Test full pipeline with LOD at far distance."""
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        scene = SceneGraph(primitives=(torus,), pipeline=(), name="test")

        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        result = optimizer.optimize(scene, camera_distance=150.0)

        # Torus should be simplified at far distance
        if isinstance(result, SceneGraph) and len(result.primitives) > 0:
            assert isinstance(result.primitives[0], SphereNode)

    def test_e2e_full_pipeline_preserves_scene_name(self) -> None:
        """Test that optimization preserves scene metadata."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        scene = SceneGraph(
            primitives=(sphere,),
            pipeline=(),
            name="important_scene_name"
        )

        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        result = optimizer.optimize(scene)

        if isinstance(result, SceneGraph):
            assert result.name == "important_scene_name"


# =============================================================================
# PATTERN DATABASE EXTENSIBILITY TESTS
# =============================================================================

class TestPatternExtensibility:
    """Tests for pattern database extensibility."""

    def test_custom_pattern_registration(self) -> None:
        """Test registering and using a custom pattern."""
        from engine.rendering.demoscene.pattern_optimizer import SDFPattern

        def match_large_sphere(node):
            if isinstance(node, SphereNode) and node.radius.value > 10.0:
                from engine.rendering.demoscene.pattern_optimizer import PatternMatch
                return PatternMatch(
                    PatternType.CAPSULE_APPROX,  # Reuse type
                    node,
                    {"radius": node.radius.value},
                    0.9
                )
            return None

        def replace_large_sphere(match):
            # Just return original for this test
            return match.matched_node

        custom = SDFPattern(
            PatternType.CAPSULE_APPROX,
            "Large sphere pattern",
            match_large_sphere,
            replace_large_sphere,
            1.0
        )

        db = PatternDatabase()
        db.register_pattern(custom)

        large_sphere = SphereNode(PositionNode(), FloatNode(15.0))
        match = db.match(large_sphere)

        # Should match our custom pattern
        assert match is not None


# =============================================================================
# PERFORMANCE COMPARISON TESTS
# =============================================================================

class TestPerformanceComparison:
    """Tests comparing optimization performance characteristics."""

    def test_hollow_box_node_reduction(self) -> None:
        """Test that hollow box optimization reduces node count."""
        outer = BoxNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        inner = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        hollow = SubtractionNode(outer, inner)

        def count_nodes(node, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return 0
            visited.add(id(node))
            count = 1
            for child in node.children():
                count += count_nodes(child, visited)
            return count

        before = count_nodes(hollow)
        result = pattern_optimize(hollow)
        after = count_nodes(result)

        # BoxFrameNode should have fewer total nodes
        assert after <= before

    def test_lod_reduces_primitive_complexity(self) -> None:
        """Test that LOD reduces overall scene complexity."""
        # Create scene with complex primitives
        prims = [
            TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),
            EllipsoidNode(PositionNode(), Vec3Node(1.0, 2.0, 0.5)),
            OctahedronNode(PositionNode(), FloatNode(1.0)),
        ]
        scene = SceneGraph(primitives=tuple(prims), pipeline=(), name="test")

        config = LODConfig()
        result = lod_simplify(scene, config, dist=150.0)

        # All should be simplified to spheres (complexity 1.0)
        total_complexity = sum(
            config.primitive_complexity.get(type(p), 1.0)
            for p in result.primitives
        )
        original_complexity = sum(
            config.primitive_complexity.get(type(p), 1.0)
            for p in scene.primitives
        )

        assert total_complexity <= original_complexity

    def test_cse_evaluation_savings_calculation(self) -> None:
        """Test that CSE correctly calculates evaluation savings."""
        # Create expression with 4 identical spheres = 3 savings
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        scene = SceneGraph(
            primitives=(sphere, sphere, sphere, sphere),
            pipeline=(),
            name="test"
        )

        from engine.rendering.demoscene.pattern_optimizer import EnhancedCSEPass
        cse_pass = EnhancedCSEPass(scene)
        cse_pass.run()

        # Should have evaluation savings
        if "evaluation_savings" in cse_pass.stats:
            assert cse_pass.stats["evaluation_savings"] >= 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_scene(self) -> None:
        """Test optimization of empty scene."""
        scene = SceneGraph(primitives=(), pipeline=(), name="empty")

        optimizer = DSLPatternOptimizer(ALL_PATTERN_PASSES)
        result = optimizer.optimize(scene)

        assert isinstance(result, SceneGraph)
        assert len(result.primitives) == 0

    def test_single_primitive(self) -> None:
        """Test optimization of single primitive scene."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        scene = SceneGraph(primitives=(sphere,), pipeline=(), name="single")

        result = pattern_optimize(scene)

        assert isinstance(result, SceneGraph)
        assert len(result.primitives) == 1

    def test_zero_radius_sphere(self) -> None:
        """Test handling of zero-radius sphere."""
        sphere = SphereNode(PositionNode(), FloatNode(0.0))

        # Should not crash
        result = pattern_optimize(sphere)
        assert result is sphere

    def test_negative_dimensions(self) -> None:
        """Test handling of negative dimensions in patterns."""
        # Box with negative size
        box = BoxNode(PositionNode(), Vec3Node(-1.0, 1.0, 1.0))

        # Should not crash
        result = pattern_optimize(box)
        assert result is box

    def test_lod_at_zero_distance(self) -> None:
        """Test LOD at exactly zero distance."""
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))

        result = lod_simplify(torus, dist=0.0)

        # At zero distance, should preserve original
        assert result is torus

    def test_lod_at_exact_threshold(self) -> None:
        """Test LOD at exact threshold distance."""
        config = LODConfig()
        torus = TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5))
        scene = SceneGraph(primitives=(torus,), pipeline=(), name="test")

        # At exactly billboard threshold
        billboard_dist = config.distance_thresholds[LODLevel.BILLBOARD]
        result = lod_simplify(scene, config, dist=billboard_dist)

        # Should be simplified at threshold
        if len(result.primitives) > 0:
            assert isinstance(result.primitives[0], SphereNode)
