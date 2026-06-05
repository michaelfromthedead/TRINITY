"""
Whitebox tests for SDF Optimizer (T-DEMO-2.8 through T-DEMO-2.12).

Tests optimization passes:
  - T-DEMO-2.8: Constant Folding
  - T-DEMO-2.9: Dead Code Elimination
  - T-DEMO-2.10: Common Sub-expression Elimination (CSE)
  - T-DEMO-2.11: Domain Repetition Flattening
  - T-DEMO-2.12: Material Merging

Each test verifies optimization correctness (input AST -> expected output AST).
"""

from __future__ import annotations

import math
import pytest

from engine.rendering.demoscene.ast_nodes import (
    Axis, BendNode, BoxNode, CellIdNode, ConeNode, CylinderNode,
    FloatNode, KifsNode, MirrorNode, PlaneNode, PositionNode,
    RepeatNode, SceneGraph, SphereNode, StretchNode, TorusNode,
    TwistNode, UnionNode, IntersectionNode, SubtractionNode, Vec3Node,
)
from engine.rendering.demoscene.sdf_optimizer import (
    OptimizationPass,
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass,
    DomainRepetitionFlatteningPass,
    MaterialMergingPass,
    SDFOptimizer,
    DEFAULT_PASSES,
    FAST_PASSES,
    AGGRESSIVE_PASSES,
    optimize_ast,
    fold_constants,
    eliminate_dead_code,
    eliminate_common_subexpressions,
    flatten_repeats,
    merge_materials,
    ast_hash,
    ast_equal,
)


# =============================================================================
# AST HASH AND EQUALITY TESTS
# =============================================================================

class TestAstHash:
    """Tests for ast_hash function."""

    def test_float_node_hash_deterministic(self) -> None:
        f1 = FloatNode(1.5)
        f2 = FloatNode(1.5)
        assert ast_hash(f1) == ast_hash(f2)

    def test_float_node_hash_different_values(self) -> None:
        f1 = FloatNode(1.5)
        f2 = FloatNode(2.5)
        assert ast_hash(f1) != ast_hash(f2)

    def test_vec3_node_hash_deterministic(self) -> None:
        v1 = Vec3Node(1.0, 2.0, 3.0)
        v2 = Vec3Node(1.0, 2.0, 3.0)
        assert ast_hash(v1) == ast_hash(v2)

    def test_vec3_node_hash_different_values(self) -> None:
        v1 = Vec3Node(1.0, 2.0, 3.0)
        v2 = Vec3Node(1.0, 2.0, 4.0)
        assert ast_hash(v1) != ast_hash(v2)

    def test_position_node_hash_consistent(self) -> None:
        p1 = PositionNode()
        p2 = PositionNode()
        assert ast_hash(p1) == ast_hash(p2)

    def test_sphere_node_hash_deterministic(self) -> None:
        s1 = SphereNode(PositionNode(), FloatNode(1.0))
        s2 = SphereNode(PositionNode(), FloatNode(1.0))
        assert ast_hash(s1) == ast_hash(s2)

    def test_sphere_node_hash_different_radius(self) -> None:
        s1 = SphereNode(PositionNode(), FloatNode(1.0))
        s2 = SphereNode(PositionNode(), FloatNode(2.0))
        assert ast_hash(s1) != ast_hash(s2)

    def test_box_node_hash_deterministic(self) -> None:
        b1 = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))
        b2 = BoxNode(PositionNode(), Vec3Node(1.0, 2.0, 3.0))
        assert ast_hash(b1) == ast_hash(b2)

    def test_repeat_node_hash_deterministic(self) -> None:
        r1 = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        r2 = RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0))
        assert ast_hash(r1) == ast_hash(r2)

    def test_scene_graph_hash_deterministic(self) -> None:
        sg1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        sg2 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        assert ast_hash(sg1) == ast_hash(sg2)


class TestAstEqual:
    """Tests for ast_equal function."""

    def test_equal_float_nodes(self) -> None:
        f1 = FloatNode(1.5)
        f2 = FloatNode(1.5)
        assert ast_equal(f1, f2)

    def test_unequal_float_nodes(self) -> None:
        f1 = FloatNode(1.5)
        f2 = FloatNode(2.5)
        assert not ast_equal(f1, f2)

    def test_equal_spheres(self) -> None:
        s1 = SphereNode(PositionNode(), FloatNode(1.0))
        s2 = SphereNode(PositionNode(), FloatNode(1.0))
        assert ast_equal(s1, s2)

    def test_different_types_not_equal(self) -> None:
        s = SphereNode(PositionNode(), FloatNode(1.0))
        b = BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        assert not ast_equal(s, b)


# =============================================================================
# T-DEMO-2.8: CONSTANT FOLDING TESTS
# =============================================================================

class TestConstantFolding:
    """Tests for ConstantFoldingPass (T-DEMO-2.8)."""

    def test_float_node_preserved(self) -> None:
        f = FloatNode(3.14)
        result = fold_constants(f)
        assert isinstance(result, FloatNode)
        assert result.value == 3.14

    def test_nan_folded_to_zero(self) -> None:
        f = FloatNode(float('nan'))
        result = fold_constants(f)
        assert isinstance(result, FloatNode)
        assert result.value == 0.0

    def test_inf_folded_to_large_value(self) -> None:
        f = FloatNode(float('inf'))
        result = fold_constants(f)
        assert isinstance(result, FloatNode)
        assert result.value == 1e10

    def test_negative_inf_folded(self) -> None:
        f = FloatNode(float('-inf'))
        result = fold_constants(f)
        assert isinstance(result, FloatNode)
        assert result.value == -1e10

    def test_vec3_preserved(self) -> None:
        v = Vec3Node(1.0, 2.0, 3.0)
        result = fold_constants(v)
        assert isinstance(result, Vec3Node)
        assert result.as_tuple() == (1.0, 2.0, 3.0)

    def test_sphere_radius_preserved(self) -> None:
        s = SphereNode(PositionNode(), FloatNode(2.0))
        result = fold_constants(s)
        assert isinstance(result, SphereNode)
        assert result.radius.value == 2.0

    def test_identity_twist_removed(self) -> None:
        """Twist with rate=0 is identity, should be removed."""
        twist = TwistNode(PositionNode(), FloatNode(0.0))
        result = fold_constants(twist)
        assert isinstance(result, PositionNode)

    def test_non_identity_twist_preserved(self) -> None:
        twist = TwistNode(PositionNode(), FloatNode(1.5))
        result = fold_constants(twist)
        assert isinstance(result, TwistNode)
        assert result.rate.value == 1.5

    def test_identity_stretch_removed(self) -> None:
        """Stretch with factor=1 is identity, should be removed."""
        stretch = StretchNode(PositionNode(), FloatNode(1.0), Axis.X)
        result = fold_constants(stretch)
        assert isinstance(result, PositionNode)

    def test_non_identity_stretch_preserved(self) -> None:
        stretch = StretchNode(PositionNode(), FloatNode(2.0), Axis.X)
        result = fold_constants(stretch)
        assert isinstance(result, StretchNode)

    def test_identity_kifs_removed(self) -> None:
        """KIFS with folds<=1 is identity, should be removed."""
        kifs = KifsNode(PositionNode(), FloatNode(1.0))
        result = fold_constants(kifs)
        assert isinstance(result, PositionNode)

    def test_non_identity_kifs_preserved(self) -> None:
        kifs = KifsNode(PositionNode(), FloatNode(6.0))
        result = fold_constants(kifs)
        assert isinstance(result, KifsNode)

    def test_identity_bend_removed_zero(self) -> None:
        """Bend with radius=0 is identity."""
        bend = BendNode(PositionNode(), FloatNode(0.0))
        result = fold_constants(bend)
        assert isinstance(result, PositionNode)

    def test_identity_bend_removed_infinite(self) -> None:
        """Bend with very large radius is essentially identity."""
        bend = BendNode(PositionNode(), FloatNode(1e9))
        result = fold_constants(bend)
        assert isinstance(result, PositionNode)

    def test_zero_cell_repeat_removed(self) -> None:
        """Repeat with cell_size=(0,0,0) is identity (no repetition)."""
        repeat = RepeatNode(PositionNode(), Vec3Node(0.0, 0.0, 0.0))
        result = fold_constants(repeat)
        assert isinstance(result, PositionNode)

    def test_scene_graph_pipeline_folded(self) -> None:
        """Scene graph with identity ops in pipeline."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.0)),),
            name="test"
        )
        result = fold_constants(sg)
        assert isinstance(result, SceneGraph)
        assert len(result.pipeline) == 0

    def test_scene_graph_non_identity_preserved(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
            name="test"
        )
        result = fold_constants(sg)
        assert isinstance(result, SceneGraph)
        assert len(result.pipeline) == 1

    def test_union_children_folded(self) -> None:
        union = UnionNode(
            SphereNode(PositionNode(), FloatNode(float('nan'))),
            BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0))
        )
        result = fold_constants(union)
        assert isinstance(result, UnionNode)
        assert result.left.radius.value == 0.0


# =============================================================================
# T-DEMO-2.9: DEAD CODE ELIMINATION TESTS
# =============================================================================

class TestDeadCodeElimination:
    """Tests for DeadCodeEliminationPass (T-DEMO-2.9)."""

    def test_empty_scene_preserved(self) -> None:
        sg = SceneGraph(primitives=(), pipeline=(), name="empty")
        result = eliminate_dead_code(sg)
        assert isinstance(result, SceneGraph)
        assert len(result.primitives) == 0

    def test_valid_primitive_preserved(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 1

    def test_degenerate_sphere_removed(self) -> None:
        """Sphere with radius <= 0 is degenerate."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_negative_radius_sphere_removed(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(-1.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_degenerate_box_removed(self) -> None:
        """Box with zero dimension is degenerate."""
        sg = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 0.0, 1.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_degenerate_cylinder_removed(self) -> None:
        sg = SceneGraph(
            primitives=(CylinderNode(PositionNode(), FloatNode(0.0), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_degenerate_torus_removed(self) -> None:
        sg = SceneGraph(
            primitives=(TorusNode(PositionNode(), FloatNode(0.0), FloatNode(0.5)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_degenerate_cone_removed(self) -> None:
        sg = SceneGraph(
            primitives=(ConeNode(PositionNode(), FloatNode(0.0), FloatNode(0.0), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_valid_primitives_kept_degenerate_removed(self) -> None:
        """Mix of valid and degenerate primitives."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(0.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 2

    def test_duplicate_mirror_ops_removed(self) -> None:
        """Two consecutive mirrors on same axis cancel out."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                MirrorNode(PositionNode(), Axis.X),
                MirrorNode(PositionNode(), Axis.X),
            ),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.pipeline) == 1

    def test_different_axis_mirrors_preserved(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                MirrorNode(PositionNode(), Axis.X),
                MirrorNode(PositionNode(), Axis.Y),
            ),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.pipeline) == 2

    def test_duplicate_repeat_ops_removed(self) -> None:
        """Duplicate repeat operations with same cell size."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
            ),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.pipeline) == 1


# =============================================================================
# T-DEMO-2.10: COMMON SUB-EXPRESSION ELIMINATION TESTS
# =============================================================================

class TestCommonSubexpressionElimination:
    """Tests for CommonSubexpressionEliminationPass (T-DEMO-2.10)."""

    def test_single_expression_no_change(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_common_subexpressions(sg)
        assert isinstance(result, SceneGraph)

    def test_duplicate_primitives_detected(self) -> None:
        """Two identical sphere nodes should be detected as duplicates."""
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        sg = SceneGraph(
            primitives=(sphere, sphere),
            pipeline=(),
            name="test"
        )
        cse_pass = CommonSubexpressionEliminationPass(sg)
        result = cse_pass.run()
        assert len(cse_pass.hoisted_variables) > 0

    def test_different_primitives_position_hoisted(self) -> None:
        """Different primitives share PositionNode, which gets hoisted."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(),
            name="test"
        )
        cse_pass = CommonSubexpressionEliminationPass(sg)
        cse_pass.run()
        # PositionNode appears twice, so it gets detected for hoisting
        assert len(cse_pass.hoisted_variables) >= 1

    def test_stats_track_hoisted_count(self) -> None:
        sphere = SphereNode(PositionNode(), FloatNode(1.0))
        sg = SceneGraph(
            primitives=(sphere, sphere, sphere),
            pipeline=(),
            name="test"
        )
        cse_pass = CommonSubexpressionEliminationPass(sg)
        cse_pass.run()
        assert cse_pass.stats.get("expressions_hoisted", 0) > 0


# =============================================================================
# T-DEMO-2.11: DOMAIN REPETITION FLATTENING TESTS
# =============================================================================

class TestDomainRepetitionFlattening:
    """Tests for DomainRepetitionFlatteningPass (T-DEMO-2.11)."""

    def test_single_repeat_preserved(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),),
            name="test"
        )
        result = flatten_repeats(sg)
        assert len(result.pipeline) == 1

    def test_nested_repeats_combined(self) -> None:
        """Two nested repeats should be combined into one."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
            name="test"
        )
        result = flatten_repeats(sg)
        assert len(result.pipeline) == 1

    def test_nested_repeats_gcd_cell_size(self) -> None:
        """Combined cell size should be GCD-based."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
            name="test"
        )
        result = flatten_repeats(sg)
        repeat = result.pipeline[0]
        assert isinstance(repeat, RepeatNode)
        assert repeat.cell_size.x == 2.0

    def test_non_repeat_ops_preserved(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(1.0)),
                MirrorNode(PositionNode(), Axis.X),
            ),
            name="test"
        )
        result = flatten_repeats(sg)
        assert len(result.pipeline) == 2

    def test_repeat_with_other_ops_partial_flatten(self) -> None:
        """Repeat followed by non-repeat, then repeat."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                TwistNode(PositionNode(), FloatNode(1.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
            name="test"
        )
        result = flatten_repeats(sg)
        assert len(result.pipeline) == 3

    def test_stats_track_flattened_count(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
            name="test"
        )
        pass_instance = DomainRepetitionFlatteningPass(sg)
        pass_instance.run()
        assert pass_instance.stats.get("repeats_flattened", 0) > 0


# =============================================================================
# T-DEMO-2.12: MATERIAL MERGING TESTS
# =============================================================================

class TestMaterialMerging:
    """Tests for MaterialMergingPass (T-DEMO-2.12)."""

    def test_single_primitive_no_change(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        result = merge_materials(sg)
        assert len(result.primitives) == 1

    def test_multiple_primitives_grouped(self) -> None:
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
                SphereNode(PositionNode(), FloatNode(2.0)),
            ),
            pipeline=(),
            name="test"
        )
        pass_instance = MaterialMergingPass(sg)
        pass_instance.run()
        assert len(pass_instance.material_groups) == 1

    def test_stats_track_single_material(self) -> None:
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(),
            name="test"
        )
        pass_instance = MaterialMergingPass(sg)
        pass_instance.run()
        assert pass_instance.stats.get("single_material_scene", 0) == 1


# =============================================================================
# SDF OPTIMIZER ORCHESTRATION TESTS
# =============================================================================

class TestSDFOptimizer:
    """Tests for SDFOptimizer class."""

    def test_empty_passes_returns_unchanged(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        optimizer = SDFOptimizer([])
        result = optimizer.optimize(sg)
        assert ast_equal(result, sg)

    def test_single_pass_applied(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),
            pipeline=(),
            name="test"
        )
        optimizer = SDFOptimizer([DeadCodeEliminationPass])
        result = optimizer.optimize(sg)
        assert len(result.primitives) == 0

    def test_multiple_passes_applied_in_order(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.0)),),
            name="test"
        )
        optimizer = SDFOptimizer([ConstantFoldingPass, DeadCodeEliminationPass])
        result = optimizer.optimize(sg)
        assert len(result.pipeline) == 0

    def test_stats_collected_per_pass(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(0.0)),),
            name="test"
        )
        optimizer = SDFOptimizer([ConstantFoldingPass])
        optimizer.optimize(sg)
        assert "ConstantFoldingPass" in optimizer.stats

    def test_total_stats_aggregated(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(0.0)),
                StretchNode(PositionNode(), FloatNode(1.0), Axis.X),
            ),
            name="test"
        )
        optimizer = SDFOptimizer([ConstantFoldingPass])
        optimizer.optimize(sg)
        assert "identity_twist_removed" in optimizer.total_stats

    def test_optimize_multiple_asts(self) -> None:
        sg1 = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test1"
        )
        sg2 = SceneGraph(
            primitives=(BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),),
            pipeline=(),
            name="test2"
        )
        optimizer = SDFOptimizer(DEFAULT_PASSES)
        results = optimizer.optimize_multiple([sg1, sg2])
        assert len(results) == 2


class TestDefaultPasses:
    """Tests for default pass configurations."""

    def test_default_passes_exist(self) -> None:
        assert len(DEFAULT_PASSES) == 5

    def test_fast_passes_subset(self) -> None:
        assert len(FAST_PASSES) == 2
        assert ConstantFoldingPass in FAST_PASSES
        assert DeadCodeEliminationPass in FAST_PASSES

    def test_aggressive_passes_has_double_folding(self) -> None:
        """Aggressive passes run constant folding twice."""
        count = sum(1 for p in AGGRESSIVE_PASSES if p == ConstantFoldingPass)
        assert count == 2


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_optimize_ast_with_defaults(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="test"
        )
        result = optimize_ast(sg)
        assert isinstance(result, SceneGraph)

    def test_optimize_ast_with_custom_passes(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),
            pipeline=(),
            name="test"
        )
        result = optimize_ast(sg, passes=[DeadCodeEliminationPass])
        assert len(result.primitives) == 0

    def test_fold_constants_convenience(self) -> None:
        twist = TwistNode(PositionNode(), FloatNode(0.0))
        result = fold_constants(twist)
        assert isinstance(result, PositionNode)

    def test_eliminate_dead_code_convenience(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(0.0)),),
            pipeline=(),
            name="test"
        )
        result = eliminate_dead_code(sg)
        assert len(result.primitives) == 0

    def test_flatten_repeats_convenience(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
            ),
            name="test"
        )
        result = flatten_repeats(sg)
        assert len(result.pipeline) == 1

    def test_merge_materials_convenience(self) -> None:
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(),
            name="test"
        )
        result = merge_materials(sg)
        assert len(result.primitives) == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for full optimization pipeline."""

    def test_complex_scene_optimized(self) -> None:
        """Complex scene with multiple optimization opportunities."""
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(1.0)),
                SphereNode(PositionNode(), FloatNode(0.0)),
                BoxNode(PositionNode(), Vec3Node(1.0, 1.0, 1.0)),
            ),
            pipeline=(
                TwistNode(PositionNode(), FloatNode(0.0)),
                RepeatNode(PositionNode(), Vec3Node(2.0, 2.0, 2.0)),
                RepeatNode(PositionNode(), Vec3Node(4.0, 4.0, 4.0)),
                MirrorNode(PositionNode(), Axis.X),
                MirrorNode(PositionNode(), Axis.X),
            ),
            name="complex"
        )
        result = optimize_ast(sg)
        assert len(result.primitives) == 2
        assert len(result.pipeline) < 5

    def test_optimization_preserves_name(self) -> None:
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(),
            name="my_scene"
        )
        result = optimize_ast(sg)
        assert result.name == "my_scene"

    def test_idempotent_optimization(self) -> None:
        """Running optimization twice should give same result."""
        sg = SceneGraph(
            primitives=(SphereNode(PositionNode(), FloatNode(1.0)),),
            pipeline=(TwistNode(PositionNode(), FloatNode(2.0)),),
            name="test"
        )
        result1 = optimize_ast(sg)
        result2 = optimize_ast(result1)
        assert ast_hash(result1) == ast_hash(result2)

    def test_empty_scene_optimization(self) -> None:
        sg = SceneGraph(primitives=(), pipeline=(), name="empty")
        result = optimize_ast(sg)
        assert len(result.primitives) == 0
        assert len(result.pipeline) == 0

    def test_all_primitives_removed_yields_empty(self) -> None:
        sg = SceneGraph(
            primitives=(
                SphereNode(PositionNode(), FloatNode(0.0)),
                BoxNode(PositionNode(), Vec3Node(0.0, 1.0, 1.0)),
            ),
            pipeline=(),
            name="test"
        )
        result = optimize_ast(sg)
        assert len(result.primitives) == 0
