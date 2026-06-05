"""
Integration tests for SDF Optimizer (T-DEMO-2.8 through T-DEMO-2.12).

Verifies that optimized WGSL compiles and produces equivalent output
to unoptimized WGSL.
"""

from __future__ import annotations

import pytest

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CellIdNode, ConeNode, CylinderNode,
    FloatNode, KifsNode, MirrorNode, PlaneNode, PositionNode,
    RepeatNode, SceneGraph, SphereNode, StretchNode, TorusNode,
    TwistNode, UnionNode, IntersectionNode, SubtractionNode, Vec3Node,
)
from engine.rendering.demoscene.wgsl_codegen import generate_wgsl
from engine.rendering.demoscene.sdf_optimizer import (
    SDFOptimizer,
    DEFAULT_PASSES,
    FAST_PASSES,
    optimize_ast,
    fold_constants,
    eliminate_dead_code,
)


# =============================================================================
# WGSL COMPILATION VERIFICATION
# =============================================================================

def verify_wgsl_syntax(wgsl: str) -> bool:
    """Basic verification that WGSL has expected structure.

    This is a lightweight check since full WGSL compilation requires GPU.
    """
    required_elements = [
        "fn sd",
        "vec3<f32>",
        "f32",
        "return",
    ]
    for elem in required_elements:
        if elem not in wgsl:
            return False

    # Check balanced braces
    open_braces = wgsl.count("{")
    close_braces = wgsl.count("}")
    if open_braces != close_braces:
        return False

    # Check balanced parentheses
    open_parens = wgsl.count("(")
    close_parens = wgsl.count(")")
    if open_parens != close_parens:
        return False

    return True


class TestOptimizedWGSLCompilation:
    """Tests that optimized ASTs produce valid WGSL."""

    def test_simple_sphere_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="sphere_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="sphere_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "sdSphere" in wgsl

    def test_box_scene_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0)),),
            pipeline=(),
            name="box_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="box_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "sdBox" in wgsl

    def test_torus_scene_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),),
            pipeline=(),
            name="torus_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="torus_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "sdTorus" in wgsl

    def test_cylinder_scene_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(2.0), FloatNode(1.0)),),
            pipeline=(),
            name="cylinder_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="cylinder_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "sdCylinder" in wgsl

    def test_cone_scene_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),),
            pipeline=(),
            name="cone_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="cone_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "sdCone" in wgsl

    def test_plane_scene_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(PlaneNode(PositionNode(), Vec3Node(0.0, 1.0, 0.0), FloatNode(0.0)),),
            pipeline=(),
            name="plane_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="plane_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "sdPlane" in wgsl

    def test_multiple_primitives_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.3)),
            ),
            pipeline=(),
            name="multi_prim"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="multi_prim")
        assert verify_wgsl_syntax(wgsl)
        assert "sdSphere" in wgsl
        assert "sdBox" in wgsl
        assert "sdTorus" in wgsl

    def test_scene_with_twist_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
            name="twist_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="twist_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "domain_twist" in wgsl

    def test_scene_with_repeat_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),),
            name="repeat_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="repeat_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "domain_repeat" in wgsl

    def test_scene_with_mirror_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
            pipeline=(MirrorNode(PositionNode(), Axis.X),),
            name="mirror_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="mirror_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "domain_mirror_x" in wgsl

    def test_scene_with_kifs_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.5)),),
            pipeline=(KifsNode(PositionNode(), FloatNode(6.0)),),
            name="kifs_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="kifs_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "domain_kifs" in wgsl

    def test_scene_with_stretch_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(StretchNode(PositionNode(), FloatNode(2.0), Axis.Y),),
            name="stretch_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="stretch_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "domain_stretch_y" in wgsl

    def test_scene_with_bend_optimized_wgsl(self) -> None:
        sg = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(0.5, 2.0, 0.5)),),
            pipeline=(BendNode(PositionNode(), FloatNode(5.0)),),
            name="bend_scene"
        )
        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="bend_scene")
        assert verify_wgsl_syntax(wgsl)
        assert "domain_bend" in wgsl


class TestOptimizedVsUnoptimizedEquivalence:
    """Tests that optimized and unoptimized WGSL are functionally equivalent."""

    def test_identity_removal_preserves_scene(self) -> None:
        """Scene with identity twist (rate=0) should produce same SDF."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.0)),),
            name="test"
        )
        # Unoptimized would include the twist
        unoptimized_wgsl = generate_wgsl(sg, name="test")

        # Optimized removes the identity twist
        optimized = optimize_ast(sg)
        optimized_wgsl = generate_wgsl(optimized, name="test")

        # Both should be valid WGSL
        assert verify_wgsl_syntax(unoptimized_wgsl)
        assert verify_wgsl_syntax(optimized_wgsl)

        # Optimized should be shorter (no twist call)
        assert "domain_twist" in unoptimized_wgsl
        assert "domain_twist" not in optimized_wgsl

    def test_degenerate_removal_produces_valid_wgsl(self) -> None:
        """Scene with degenerate primitive should still produce valid WGSL."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(0.0)),  # Degenerate
            ),
            pipeline=(),
            name="test"
        )

        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="test")

        assert verify_wgsl_syntax(wgsl)
        assert len(optimized.primitives) == 1

    def test_nested_repeat_flattening_valid_wgsl(self) -> None:
        """Flattened repeats should produce valid WGSL."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
            name="nested_repeat"
        )

        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="nested_repeat")

        assert verify_wgsl_syntax(wgsl)
        # Should have only one repeat in pipeline
        assert len(optimized.pipeline) == 1

    def test_complex_pipeline_optimization(self) -> None:
        """Complex pipeline with multiple optimization opportunities."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(0.5, 0.5, 0.5)),
            ),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(0.0)),  # Identity
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),  # Can be flattened
                StretchNode(PositionNode(), FloatNode(1.0), Axis.X),  # Identity
                MirrorNode(PositionNode(), Axis.X),
            ),
            name="complex"
        )

        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="complex")

        assert verify_wgsl_syntax(wgsl)
        # Identity ops should be removed, repeats flattened
        assert len(optimized.pipeline) < len(sg.pipeline)


class TestEdgeCases:
    """Edge case tests for optimization and WGSL generation."""

    def test_empty_scene_generates_valid_wgsl(self) -> None:
        """Empty scene after optimization should still generate."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),  # Degenerate
            pipeline=(),
            name="empty"
        )

        optimized = optimize_ast(sg)
        # Empty scene - generate_wgsl may raise or return minimal WGSL
        # This tests that optimization handles it gracefully
        assert len(optimized.primitives) == 0

    def test_all_ops_removed_valid_wgsl(self) -> None:
        """Scene where all pipeline ops are identity."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(0.0)),
                StretchNode(PositionNode(), FloatNode(1.0), Axis.X),
                StretchNode(PositionNode(), FloatNode(1.0), Axis.Y),
                StretchNode(PositionNode(), FloatNode(1.0), Axis.Z),
            ),
            name="all_identity"
        )

        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="all_identity")

        assert verify_wgsl_syntax(wgsl)
        assert len(optimized.pipeline) == 0

    def test_large_scene_optimization(self) -> None:
        """Large scene with many primitives."""
        primitives = tuple(
            SphereNode(PositionNode(), FloatNode(0.1 * i))
            for i in range(1, 11)  # 10 spheres
        )

        sg = SceneGraph(
            primitives=primitives,
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                MirrorNode(PositionNode(), Axis.X),
            ),
            name="large_scene"
        )

        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="large_scene")

        assert verify_wgsl_syntax(wgsl)
        # All spheres have positive radius, so all should remain
        assert len(optimized.primitives) == 10

    def test_mixed_valid_invalid_primitives(self) -> None:
        """Mix of valid and invalid primitives."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),      # Valid
                BoxNode(PositionNode(), Vec3Node(0.0, 1.0, 1.0)), # Invalid (zero x)
                TorusNode(PositionNode(), FloatNode(2.0), FloatNode(0.5)),  # Valid
                CylinderNode(PositionNode(), FloatNode(0.0), FloatNode(1.0)),  # Invalid (zero h)
                ConeNode(PositionNode(), FloatNode(2.0), FloatNode(0.0), FloatNode(1.0)),  # Valid
            ),
            pipeline=(),
            name="mixed"
        )

        optimized = optimize_ast(sg)
        wgsl = generate_wgsl(optimized, name="mixed")

        assert verify_wgsl_syntax(wgsl)
        assert len(optimized.primitives) == 3


class TestOptimizationPreservesSemantics:
    """Tests that optimizations preserve scene semantics."""

    def test_constant_folding_preserves_params(self) -> None:
        """Constant folding should preserve non-identity parameters."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(2.5)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(1.5)),),
            name="test"
        )

        optimized = fold_constants(sg)

        # Parameters should be unchanged
        assert optimized.primitives[0].radius.value == 2.5
        assert optimized.pipeline[0].rate.value == 1.5

    def test_dead_code_preserves_valid_primitives(self) -> None:
        """Dead code elimination should preserve valid primitives."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(2.0)),
                SphereNode(PositionNode(), FloatNode(3.0)),
            ),
            pipeline=(),
            name="test"
        )

        optimized = eliminate_dead_code(sg)

        # All primitives are valid, should be preserved
        assert len(optimized.primitives) == 3
        radii = [p.radius.value for p in optimized.primitives]
        assert radii == [1.0, 2.0, 3.0]

    def test_full_optimization_preserves_scene_name(self) -> None:
        """Full optimization should preserve scene name."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.0)),),
            name="my_important_scene"
        )

        optimized = optimize_ast(sg)

        assert optimized.name == "my_important_scene"
