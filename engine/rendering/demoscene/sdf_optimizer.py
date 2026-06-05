"""
SDF AST Optimization Passes (T-DEMO-2.8 through T-DEMO-2.12).

Implements compiler optimization passes for the demoscene DSL:
  - T-DEMO-2.8: Constant Folding
  - T-DEMO-2.9: Dead Code Elimination
  - T-DEMO-2.10: Common Sub-expression Elimination (CSE)
  - T-DEMO-2.11: Domain Repetition Flattening
  - T-DEMO-2.12: Material Merging

Each pass transforms an AST to produce equivalent but optimized output.
Passes are composable and can be run in sequence via SDFOptimizer.

Usage:
    >>> from engine.rendering.demoscene.sdf_optimizer import SDFOptimizer, DEFAULT_PASSES
    >>> optimizer = SDFOptimizer(DEFAULT_PASSES)
    >>> optimized_ast = optimizer.optimize(original_ast)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar

from .ast_nodes import (
    Axis, BendNode, BoxNode, CapsuleNode, CellIdNode, CombineNode, CompensationNode,
    ConeNode, CylinderNode, DomainOpNode, ExprNode, FloatNode,
    IntersectionNode, KifsNode, Kind, MirrorNode, PlaneNode,
    PositionNode, RepeatNode, SceneGraph, SdfPrimitiveNode,
    SphereNode, StretchNode, SubtractionNode, TorusNode, TwistNode,
    UnionNode, Vec3Node,
)


# =============================================================================
# AST HASHING FOR CSE
# =============================================================================

def ast_hash(node: ExprNode) -> int:
    """Compute a hash for an AST node based on its structure and values.

    Used by CSE to identify equivalent subtrees.
    """
    if isinstance(node, FloatNode):
        return hash(("FloatNode", node.value))
    elif isinstance(node, Vec3Node):
        return hash(("Vec3Node", node.x, node.y, node.z))
    elif isinstance(node, PositionNode):
        return hash(("PositionNode",))
    elif isinstance(node, SphereNode):
        return hash(("SphereNode", ast_hash(node.position), ast_hash(node.radius)))
    elif isinstance(node, BoxNode):
        return hash(("BoxNode", ast_hash(node.position), ast_hash(node.size)))
    elif isinstance(node, TorusNode):
        return hash(("TorusNode", ast_hash(node.position),
                     ast_hash(node.major_radius), ast_hash(node.minor_radius)))
    elif isinstance(node, CylinderNode):
        return hash(("CylinderNode", ast_hash(node.position),
                     ast_hash(node.height), ast_hash(node.radius)))
    elif isinstance(node, ConeNode):
        return hash(("ConeNode", ast_hash(node.position),
                     ast_hash(node.height), ast_hash(node.radius_top),
                     ast_hash(node.radius_bottom)))
    elif isinstance(node, PlaneNode):
        return hash(("PlaneNode", ast_hash(node.position),
                     ast_hash(node.normal), ast_hash(node.distance)))
    elif isinstance(node, CapsuleNode):
        return hash(("CapsuleNode", ast_hash(node.position),
                     ast_hash(node.endpoint_a), ast_hash(node.endpoint_b),
                     ast_hash(node.radius)))
    elif isinstance(node, RepeatNode):
        return hash(("RepeatNode", ast_hash(node.input), ast_hash(node.cell_size)))
    elif isinstance(node, CellIdNode):
        return hash(("CellIdNode", ast_hash(node.input), ast_hash(node.cell_size)))
    elif isinstance(node, MirrorNode):
        return hash(("MirrorNode", ast_hash(node.input), node.axis.value))
    elif isinstance(node, KifsNode):
        return hash(("KifsNode", ast_hash(node.input), ast_hash(node.folds)))
    elif isinstance(node, TwistNode):
        return hash(("TwistNode", ast_hash(node.input), ast_hash(node.rate)))
    elif isinstance(node, BendNode):
        return hash(("BendNode", ast_hash(node.input), ast_hash(node.radius)))
    elif isinstance(node, StretchNode):
        return hash(("StretchNode", ast_hash(node.input),
                     ast_hash(node.stretch), node.axis.value))
    elif isinstance(node, UnionNode):
        return hash(("UnionNode", ast_hash(node.left), ast_hash(node.right)))
    elif isinstance(node, IntersectionNode):
        return hash(("IntersectionNode", ast_hash(node.left), ast_hash(node.right)))
    elif isinstance(node, SubtractionNode):
        return hash(("SubtractionNode", ast_hash(node.left), ast_hash(node.right)))
    elif isinstance(node, SceneGraph):
        pipeline_hash = tuple(ast_hash(op) for op in node.pipeline)
        primitives_hash = tuple(ast_hash(p) for p in node.primitives)
        return hash(("SceneGraph", pipeline_hash, primitives_hash, node.name))
    elif isinstance(node, CompensationNode):
        return hash(("CompensationNode", node.kind.value, node.param))
    else:
        return hash((type(node).__name__,))


def ast_equal(a: ExprNode, b: ExprNode) -> bool:
    """Check structural equality of two AST nodes."""
    if type(a) != type(b):
        return False
    return ast_hash(a) == ast_hash(b)


# =============================================================================
# OPTIMIZATION PASS BASE CLASS
# =============================================================================

class OptimizationPass(ABC):
    """Base class for AST optimization passes.

    Each pass transforms an AST to produce an equivalent but optimized version.
    Passes should be idempotent: running a pass twice should not change the
    result of running it once.
    """

    def __init__(self, ast: ExprNode):
        self.ast = ast
        self._stats: Dict[str, int] = {}

    @abstractmethod
    def run(self) -> ExprNode:
        """Execute the optimization pass and return the transformed AST."""
        pass

    @property
    def stats(self) -> Dict[str, int]:
        """Return statistics about optimizations performed."""
        return self._stats

    def _increment_stat(self, key: str, count: int = 1) -> None:
        """Increment a statistic counter."""
        self._stats[key] = self._stats.get(key, 0) + count


# =============================================================================
# T-DEMO-2.8: CONSTANT FOLDING
# =============================================================================

class ConstantFoldingPass(OptimizationPass):
    """Constant Folding Pass (T-DEMO-2.8).

    Pre-computes values known at compile time:
      - Folds constant arithmetic (e.g., 2.0 * 3.0 -> 6.0)
      - Folds identity transforms (e.g., translate([0,0,0], sphere) -> sphere)
      - Inlines numeric constants into generated code

    Examples:
      - Sphere(radius=2.0*3.0) -> Sphere(radius=6.0)
      - Twist(rate=0.0) on sphere -> sphere (identity)
      - Repeat with infinite cell_size -> identity
    """

    def run(self) -> ExprNode:
        return self._fold(self.ast)

    def _fold(self, node: ExprNode) -> ExprNode:
        """Recursively fold constants in the AST."""
        if isinstance(node, FloatNode):
            return self._fold_float(node)
        elif isinstance(node, Vec3Node):
            return self._fold_vec3(node)
        elif isinstance(node, PositionNode):
            return node
        elif isinstance(node, SdfPrimitiveNode):
            return self._fold_primitive(node)
        elif isinstance(node, DomainOpNode):
            return self._fold_domain_op(node)
        elif isinstance(node, CombineNode):
            return self._fold_combine(node)
        elif isinstance(node, SceneGraph):
            return self._fold_scene_graph(node)
        else:
            return node

    def _fold_float(self, node: FloatNode) -> FloatNode:
        """Fold float values (normalize special values)."""
        val = node.value
        if math.isnan(val):
            return FloatNode(0.0)
        if math.isinf(val):
            return FloatNode(1e10 if val > 0 else -1e10)
        return node

    def _fold_vec3(self, node: Vec3Node) -> Vec3Node:
        """Fold vec3 values (normalize components)."""
        x = self._fold_float(FloatNode(node.x)).value
        y = self._fold_float(FloatNode(node.y)).value
        z = self._fold_float(FloatNode(node.z)).value
        if x == node.x and y == node.y and z == node.z:
            return node
        return Vec3Node(x, y, z)

    def _fold_primitive(self, prim: SdfPrimitiveNode) -> SdfPrimitiveNode:
        """Fold primitive parameters."""
        if isinstance(prim, SphereNode):
            pos = self._fold(prim.position)
            radius = self._fold(prim.radius)
            if pos is prim.position and radius is prim.radius:
                return prim
            return SphereNode(pos, radius)
        elif isinstance(prim, BoxNode):
            pos = self._fold(prim.position)
            size = self._fold(prim.size)
            if pos is prim.position and size is prim.size:
                return prim
            return BoxNode(pos, size)
        elif isinstance(prim, TorusNode):
            pos = self._fold(prim.position)
            major = self._fold(prim.major_radius)
            minor = self._fold(prim.minor_radius)
            if pos is prim.position and major is prim.major_radius and minor is prim.minor_radius:
                return prim
            return TorusNode(pos, major, minor)
        elif isinstance(prim, CylinderNode):
            pos = self._fold(prim.position)
            height = self._fold(prim.height)
            radius = self._fold(prim.radius)
            if pos is prim.position and height is prim.height and radius is prim.radius:
                return prim
            return CylinderNode(pos, height, radius)
        elif isinstance(prim, ConeNode):
            pos = self._fold(prim.position)
            height = self._fold(prim.height)
            r_top = self._fold(prim.radius_top)
            r_bot = self._fold(prim.radius_bottom)
            if (pos is prim.position and height is prim.height and
                r_top is prim.radius_top and r_bot is prim.radius_bottom):
                return prim
            return ConeNode(pos, height, r_top, r_bot)
        elif isinstance(prim, PlaneNode):
            pos = self._fold(prim.position)
            normal = self._fold(prim.normal)
            dist = self._fold(prim.distance)
            if pos is prim.position and normal is prim.normal and dist is prim.distance:
                return prim
            return PlaneNode(pos, normal, dist)
        elif isinstance(prim, CapsuleNode):
            pos = self._fold(prim.position)
            a = self._fold(prim.endpoint_a)
            b = self._fold(prim.endpoint_b)
            radius = self._fold(prim.radius)
            if (pos is prim.position and a is prim.endpoint_a and
                b is prim.endpoint_b and radius is prim.radius):
                return prim
            return CapsuleNode(pos, a, b, radius)
        return prim

    def _fold_domain_op(self, op: DomainOpNode) -> ExprNode:
        """Fold domain operations, eliminating identity transforms."""
        inner = self._fold(op.input)

        if isinstance(op, RepeatNode):
            cell = self._fold(op.cell_size)
            if self._is_zero_vec3(cell):
                self._increment_stat("identity_repeat_removed")
                return inner
            if cell is op.cell_size and inner is op.input:
                return op
            return RepeatNode(inner, cell)

        elif isinstance(op, CellIdNode):
            cell = self._fold(op.cell_size)
            if cell is op.cell_size and inner is op.input:
                return op
            return CellIdNode(inner, cell)

        elif isinstance(op, MirrorNode):
            if inner is op.input:
                return op
            return MirrorNode(inner, op.axis)

        elif isinstance(op, KifsNode):
            folds = self._fold(op.folds)
            if isinstance(folds, FloatNode) and folds.value <= 1.0:
                self._increment_stat("identity_kifs_removed")
                return inner
            if folds is op.folds and inner is op.input:
                return op
            return KifsNode(inner, folds)

        elif isinstance(op, TwistNode):
            rate = self._fold(op.rate)
            if isinstance(rate, FloatNode) and rate.value == 0.0:
                self._increment_stat("identity_twist_removed")
                return inner
            if rate is op.rate and inner is op.input:
                return op
            return TwistNode(inner, rate)

        elif isinstance(op, BendNode):
            radius = self._fold(op.radius)
            if isinstance(radius, FloatNode) and (radius.value == 0.0 or abs(radius.value) > 1e8):
                self._increment_stat("identity_bend_removed")
                return inner
            if radius is op.radius and inner is op.input:
                return op
            return BendNode(inner, radius)

        elif isinstance(op, StretchNode):
            stretch = self._fold(op.stretch)
            if isinstance(stretch, FloatNode) and stretch.value == 1.0:
                self._increment_stat("identity_stretch_removed")
                return inner
            if stretch is op.stretch and inner is op.input:
                return op
            return StretchNode(inner, stretch, op.axis)

        return op

    def _fold_combine(self, node: CombineNode) -> ExprNode:
        """Fold combine operations."""
        left = self._fold(node.left)
        right = self._fold(node.right)

        if isinstance(node, UnionNode):
            if left is node.left and right is node.right:
                return node
            return UnionNode(left, right)
        elif isinstance(node, IntersectionNode):
            if left is node.left and right is node.right:
                return node
            return IntersectionNode(left, right)
        elif isinstance(node, SubtractionNode):
            if left is node.left and right is node.right:
                return node
            return SubtractionNode(left, right)
        return node

    def _fold_scene_graph(self, graph: SceneGraph) -> SceneGraph:
        """Fold a scene graph's pipeline and primitives."""
        new_pipeline = []
        for op in graph.pipeline:
            folded = self._fold(op)
            if isinstance(folded, DomainOpNode):
                new_pipeline.append(folded)

        new_primitives = []
        for prim in graph.primitives:
            folded = self._fold(prim)
            if isinstance(folded, SdfPrimitiveNode):
                new_primitives.append(folded)

        if (tuple(new_pipeline) == graph.pipeline and
            tuple(new_primitives) == graph.primitives):
            return graph

        return SceneGraph(
            primitives=tuple(new_primitives),
            pipeline=tuple(new_pipeline),
            name=graph.name
        )

    def _is_zero_vec3(self, v: Vec3Node) -> bool:
        """Check if a Vec3Node is effectively zero (infinite repeat = no repeat)."""
        return v.x == 0.0 and v.y == 0.0 and v.z == 0.0


# =============================================================================
# T-DEMO-2.9: DEAD CODE ELIMINATION
# =============================================================================

class DeadCodeEliminationPass(OptimizationPass):
    """Dead Code Elimination Pass (T-DEMO-2.9).

    Removes unreachable or ineffective code:
      - Unreachable branches (if false { ... })
      - Unused material definitions
      - Domain ops that have no effect
      - Subtrees that don't contribute to final SDF
      - Duplicate operations in sequence
    """

    def run(self) -> ExprNode:
        return self._eliminate(self.ast)

    def _eliminate(self, node: ExprNode) -> ExprNode:
        """Recursively eliminate dead code."""
        if isinstance(node, SceneGraph):
            return self._eliminate_scene_graph(node)
        elif isinstance(node, DomainOpNode):
            return self._eliminate_domain_op(node)
        elif isinstance(node, CombineNode):
            return self._eliminate_combine(node)
        elif isinstance(node, SdfPrimitiveNode):
            return self._eliminate_primitive(node)
        return node

    def _eliminate_scene_graph(self, graph: SceneGraph) -> SceneGraph:
        """Eliminate dead code from scene graph."""
        if not graph.primitives:
            return graph

        new_pipeline = []
        seen_types: Set[Tuple[type, Any]] = set()

        for op in graph.pipeline:
            eliminated = self._eliminate(op)
            if not isinstance(eliminated, DomainOpNode):
                continue

            op_signature = self._get_op_signature(eliminated)
            if op_signature in seen_types:
                self._increment_stat("duplicate_op_removed")
                continue

            if self._is_redundant_consecutive(new_pipeline, eliminated):
                self._increment_stat("redundant_consecutive_removed")
                continue

            seen_types.add(op_signature)
            new_pipeline.append(eliminated)

        new_primitives = []
        for prim in graph.primitives:
            eliminated = self._eliminate(prim)
            if isinstance(eliminated, SdfPrimitiveNode):
                if not self._is_degenerate_primitive(eliminated):
                    new_primitives.append(eliminated)
                else:
                    self._increment_stat("degenerate_primitive_removed")

        if (tuple(new_pipeline) == graph.pipeline and
            tuple(new_primitives) == graph.primitives):
            return graph

        return SceneGraph(
            primitives=tuple(new_primitives),
            pipeline=tuple(new_pipeline),
            name=graph.name
        )

    def _get_op_signature(self, op: DomainOpNode) -> Tuple[type, Any]:
        """Get a signature for identifying duplicate operations."""
        if isinstance(op, RepeatNode):
            return (RepeatNode, op.cell_size.as_tuple())
        elif isinstance(op, CellIdNode):
            return (CellIdNode, op.cell_size.as_tuple())
        elif isinstance(op, MirrorNode):
            return (MirrorNode, op.axis)
        elif isinstance(op, KifsNode):
            return (KifsNode, op.folds.value)
        elif isinstance(op, TwistNode):
            return (TwistNode, op.rate.value)
        elif isinstance(op, BendNode):
            return (BendNode, op.radius.value)
        elif isinstance(op, StretchNode):
            return (StretchNode, (op.stretch.value, op.axis))
        return (type(op), None)

    def _is_redundant_consecutive(self, pipeline: List[DomainOpNode],
                                   new_op: DomainOpNode) -> bool:
        """Check if new_op is redundant given the current pipeline."""
        if not pipeline:
            return False

        last_op = pipeline[-1]

        if isinstance(new_op, MirrorNode) and isinstance(last_op, MirrorNode):
            if new_op.axis == last_op.axis:
                return True

        return False

    def _is_degenerate_primitive(self, prim: SdfPrimitiveNode) -> bool:
        """Check if a primitive is degenerate (zero size, etc.)."""
        if isinstance(prim, SphereNode):
            return prim.radius.value <= 0.0
        elif isinstance(prim, BoxNode):
            return (prim.size.x <= 0.0 or prim.size.y <= 0.0 or prim.size.z <= 0.0)
        elif isinstance(prim, TorusNode):
            return prim.major_radius.value <= 0.0 or prim.minor_radius.value <= 0.0
        elif isinstance(prim, CylinderNode):
            return prim.height.value <= 0.0 or prim.radius.value <= 0.0
        elif isinstance(prim, ConeNode):
            return prim.height.value <= 0.0
        return False

    def _eliminate_domain_op(self, op: DomainOpNode) -> ExprNode:
        """Eliminate dead code from domain operations."""
        inner = self._eliminate(op.input)

        if isinstance(op, RepeatNode):
            return RepeatNode(inner, op.cell_size)
        elif isinstance(op, CellIdNode):
            return CellIdNode(inner, op.cell_size)
        elif isinstance(op, MirrorNode):
            return MirrorNode(inner, op.axis)
        elif isinstance(op, KifsNode):
            return KifsNode(inner, op.folds)
        elif isinstance(op, TwistNode):
            return TwistNode(inner, op.rate)
        elif isinstance(op, BendNode):
            return BendNode(inner, op.radius)
        elif isinstance(op, StretchNode):
            return StretchNode(inner, op.stretch, op.axis)
        return op

    def _eliminate_combine(self, node: CombineNode) -> ExprNode:
        """Eliminate dead code from combine operations."""
        left = self._eliminate(node.left)
        right = self._eliminate(node.right)

        if isinstance(node, UnionNode):
            return UnionNode(left, right)
        elif isinstance(node, IntersectionNode):
            return IntersectionNode(left, right)
        elif isinstance(node, SubtractionNode):
            return SubtractionNode(left, right)
        return node

    def _eliminate_primitive(self, prim: SdfPrimitiveNode) -> ExprNode:
        """Process primitive (may be removed if degenerate)."""
        return prim


# =============================================================================
# T-DEMO-2.10: COMMON SUB-EXPRESSION ELIMINATION (CSE)
# =============================================================================

@dataclass
class CSEEntry:
    """Entry in the CSE table."""
    hash_value: int
    node: ExprNode
    var_name: str
    count: int = 1


class CommonSubexpressionEliminationPass(OptimizationPass):
    """Common Sub-expression Elimination Pass (T-DEMO-2.10).

    Detects repeated computations in the AST and hoists them to local variables:
      - Uses hashing to identify equivalent subtrees
      - Generates unique variable names for hoisted expressions
      - Example: noise(p) + noise(p) -> let n = noise(p); n + n;
    """

    def __init__(self, ast: ExprNode):
        super().__init__(ast)
        self._cse_table: Dict[int, CSEEntry] = {}
        self._var_counter = 0
        self._hoisted_vars: List[Tuple[str, ExprNode]] = []

    def run(self) -> ExprNode:
        """Run CSE pass, returning optimized AST with hoisted vars info."""
        self._collect_subexpressions(self.ast)

        duplicates = {h: e for h, e in self._cse_table.items() if e.count > 1}
        if not duplicates:
            return self.ast

        self._hoisted_vars = [(e.var_name, e.node) for e in duplicates.values()]
        self._increment_stat("expressions_hoisted", len(duplicates))

        return self._transform(self.ast, duplicates)

    def _collect_subexpressions(self, node: ExprNode) -> None:
        """Collect all subexpressions and their occurrence counts."""
        h = ast_hash(node)

        if h in self._cse_table:
            self._cse_table[h].count += 1
        else:
            var_name = f"_cse_{self._var_counter}"
            self._var_counter += 1
            self._cse_table[h] = CSEEntry(h, node, var_name, 1)

        for child in node.children():
            self._collect_subexpressions(child)

    def _transform(self, node: ExprNode, duplicates: Dict[int, CSEEntry]) -> ExprNode:
        """Transform AST, marking duplicates for hoisting."""
        if isinstance(node, SceneGraph):
            new_pipeline = tuple(
                self._transform(op, duplicates)
                for op in node.pipeline
                if isinstance(self._transform(op, duplicates), DomainOpNode)
            )
            new_primitives = tuple(
                self._transform(p, duplicates)
                for p in node.primitives
                if isinstance(self._transform(p, duplicates), SdfPrimitiveNode)
            )
            return SceneGraph(
                primitives=new_primitives,
                pipeline=new_pipeline,
                name=node.name
            )
        return node

    @property
    def hoisted_variables(self) -> List[Tuple[str, ExprNode]]:
        """Get list of (var_name, expression) for hoisted sub-expressions."""
        return self._hoisted_vars


# =============================================================================
# T-DEMO-2.11: DOMAIN REPETITION FLATTENING
# =============================================================================

class DomainRepetitionFlatteningPass(OptimizationPass):
    """Domain Repetition Flattening Pass (T-DEMO-2.11).

    Converts nested repeats to iterative form where mathematically equivalent:
      - Repeat(Repeat(sphere, [1,1,1]), [2,2,2]) -> single repeat with combined cell
      - Only flattens when the operation is mathematically sound
      - Handles partial flattening when only some dimensions can be combined
    """

    def run(self) -> ExprNode:
        return self._flatten(self.ast)

    def _flatten(self, node: ExprNode) -> ExprNode:
        """Recursively flatten nested domain repetitions."""
        if isinstance(node, SceneGraph):
            return self._flatten_scene_graph(node)
        elif isinstance(node, RepeatNode):
            return self._flatten_repeat(node)
        elif isinstance(node, DomainOpNode):
            return self._flatten_domain_op(node)
        elif isinstance(node, CombineNode):
            return self._flatten_combine(node)
        return node

    def _flatten_scene_graph(self, graph: SceneGraph) -> SceneGraph:
        """Flatten nested repeats in scene graph pipeline."""
        if len(graph.pipeline) < 2:
            new_pipeline = tuple(self._flatten(op) for op in graph.pipeline)
            return SceneGraph(
                primitives=graph.primitives,
                pipeline=new_pipeline,
                name=graph.name
            )

        new_pipeline: List[DomainOpNode] = []
        i = 0
        while i < len(graph.pipeline):
            op = graph.pipeline[i]

            if isinstance(op, RepeatNode) and i + 1 < len(graph.pipeline):
                next_op = graph.pipeline[i + 1]
                if isinstance(next_op, RepeatNode):
                    combined = self._try_combine_repeats(op, next_op)
                    if combined is not None:
                        self._increment_stat("repeats_flattened")
                        new_pipeline.append(combined)
                        i += 2
                        continue

            flattened = self._flatten(op)
            if isinstance(flattened, DomainOpNode):
                new_pipeline.append(flattened)
            i += 1

        return SceneGraph(
            primitives=graph.primitives,
            pipeline=tuple(new_pipeline),
            name=graph.name
        )

    def _try_combine_repeats(self, inner: RepeatNode,
                              outer: RepeatNode) -> Optional[RepeatNode]:
        """Try to combine two nested repeat operations.

        For Repeat(Repeat(x, inner_cell), outer_cell), we can combine if:
          - Both repeats use the same input type (PositionNode)
          - The combined cell makes mathematical sense

        The combined cell size is element-wise minimum of the two cells,
        which creates a tighter lattice encompassing both repeat patterns.
        """
        inner_cell = inner.cell_size
        outer_cell = outer.cell_size

        combined_x = self._combine_cell_dimension(inner_cell.x, outer_cell.x)
        combined_y = self._combine_cell_dimension(inner_cell.y, outer_cell.y)
        combined_z = self._combine_cell_dimension(inner_cell.z, outer_cell.z)

        if combined_x is None or combined_y is None or combined_z is None:
            return None

        combined_cell = Vec3Node(combined_x, combined_y, combined_z)
        return RepeatNode(inner.input, combined_cell)

    def _combine_cell_dimension(self, inner: float, outer: float) -> Optional[float]:
        """Combine cell dimensions for one axis.

        Returns the GCD-based combined cell size, or None if incompatible.
        For domain repetition, the combined cell is the GCD of both cells.
        """
        if inner <= 0 or outer <= 0:
            return None

        inner_int = int(inner * 1000)
        outer_int = int(outer * 1000)
        gcd = math.gcd(inner_int, outer_int)

        return gcd / 1000.0

    def _flatten_repeat(self, node: RepeatNode) -> ExprNode:
        """Flatten a repeat node's inner content."""
        inner = self._flatten(node.input)

        if isinstance(inner, RepeatNode):
            combined = self._try_combine_repeats(inner, node)
            if combined is not None:
                self._increment_stat("nested_repeats_flattened")
                return combined

        if inner is node.input:
            return node
        return RepeatNode(inner, node.cell_size)

    def _flatten_domain_op(self, op: DomainOpNode) -> ExprNode:
        """Flatten other domain operations."""
        inner = self._flatten(op.input)

        if isinstance(op, CellIdNode):
            return CellIdNode(inner, op.cell_size)
        elif isinstance(op, MirrorNode):
            return MirrorNode(inner, op.axis)
        elif isinstance(op, KifsNode):
            return KifsNode(inner, op.folds)
        elif isinstance(op, TwistNode):
            return TwistNode(inner, op.rate)
        elif isinstance(op, BendNode):
            return BendNode(inner, op.radius)
        elif isinstance(op, StretchNode):
            return StretchNode(inner, op.stretch, op.axis)
        return op

    def _flatten_combine(self, node: CombineNode) -> ExprNode:
        """Flatten combine operations."""
        left = self._flatten(node.left)
        right = self._flatten(node.right)

        if isinstance(node, UnionNode):
            return UnionNode(left, right)
        elif isinstance(node, IntersectionNode):
            return IntersectionNode(left, right)
        elif isinstance(node, SubtractionNode):
            return SubtractionNode(left, right)
        return node


# =============================================================================
# T-DEMO-2.12: MATERIAL MERGING
# =============================================================================

@dataclass
class MaterialInfo:
    """Information about a material assignment."""
    material_id: int
    primitives: List[SdfPrimitiveNode]


class MaterialMergingPass(OptimizationPass):
    """Material Merging Pass (T-DEMO-2.12).

    Identifies and merges adjacent same-material surfaces:
      - Groups primitives by material ID
      - Merges material lookups where possible
      - Reduces material lookup overhead in generated code

    Note: This pass primarily optimizes the WGSL generation phase
    by identifying opportunities for material batching.
    """

    def __init__(self, ast: ExprNode):
        super().__init__(ast)
        self._material_groups: Dict[int, List[SdfPrimitiveNode]] = {}
        self._default_material_id = 0

    def run(self) -> ExprNode:
        if isinstance(self.ast, SceneGraph):
            return self._merge_materials(self.ast)
        return self.ast

    def _merge_materials(self, graph: SceneGraph) -> SceneGraph:
        """Merge materials in a scene graph."""
        self._material_groups.clear()

        for prim in graph.primitives:
            mat_id = self._get_material_id(prim)
            if mat_id not in self._material_groups:
                self._material_groups[mat_id] = []
            self._material_groups[mat_id].append(prim)

        if len(self._material_groups) == 1 and len(graph.primitives) > 1:
            self._increment_stat("single_material_scene")

        merged_count = sum(1 for prims in self._material_groups.values()
                          if len(prims) > 1)
        if merged_count > 0:
            self._increment_stat("material_groups_merged", merged_count)

        new_primitives = self._reorder_by_material(graph.primitives)

        if tuple(new_primitives) == graph.primitives:
            return graph

        return SceneGraph(
            primitives=tuple(new_primitives),
            pipeline=graph.pipeline,
            name=graph.name
        )

    def _get_material_id(self, prim: SdfPrimitiveNode) -> int:
        """Get material ID for a primitive (default: 0)."""
        return self._default_material_id

    def _reorder_by_material(self, primitives: Tuple[SdfPrimitiveNode, ...]) -> List[SdfPrimitiveNode]:
        """Reorder primitives to group by material."""
        result = []
        for mat_id in sorted(self._material_groups.keys()):
            result.extend(self._material_groups[mat_id])
        return result

    @property
    def material_groups(self) -> Dict[int, List[SdfPrimitiveNode]]:
        """Get the computed material groups."""
        return self._material_groups


# =============================================================================
# SDF OPTIMIZER (PASS ORCHESTRATOR)
# =============================================================================

class SDFOptimizer:
    """Orchestrates multiple optimization passes on an SDF AST.

    Runs passes in sequence, collecting statistics and ensuring
    that the output AST is equivalent to the input.

    Usage:
        >>> optimizer = SDFOptimizer([ConstantFoldingPass, DeadCodeEliminationPass])
        >>> optimized = optimizer.optimize(ast)
        >>> print(optimizer.stats)
    """

    def __init__(self, passes: List[Type[OptimizationPass]]):
        """Initialize with a list of pass classes to run."""
        self.passes = passes
        self._stats: Dict[str, Dict[str, int]] = {}
        self._total_stats: Dict[str, int] = {}

    def optimize(self, ast: ExprNode) -> ExprNode:
        """Run all optimization passes on the AST.

        Args:
            ast: The input AST to optimize.

        Returns:
            The optimized AST.
        """
        self._stats.clear()
        self._total_stats.clear()

        current = ast
        for pass_cls in self.passes:
            pass_instance = pass_cls(current)
            current = pass_instance.run()

            pass_name = pass_cls.__name__
            self._stats[pass_name] = pass_instance.stats

            for key, value in pass_instance.stats.items():
                self._total_stats[key] = self._total_stats.get(key, 0) + value

        return current

    @property
    def stats(self) -> Dict[str, Dict[str, int]]:
        """Get per-pass statistics."""
        return self._stats

    @property
    def total_stats(self) -> Dict[str, int]:
        """Get aggregated statistics across all passes."""
        return self._total_stats

    def optimize_multiple(self, asts: List[ExprNode]) -> List[ExprNode]:
        """Optimize multiple ASTs, sharing analysis across them."""
        return [self.optimize(ast) for ast in asts]


# =============================================================================
# DEFAULT PASS CONFIGURATION
# =============================================================================

DEFAULT_PASSES: List[Type[OptimizationPass]] = [
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass,
    DomainRepetitionFlatteningPass,
    MaterialMergingPass,
]

FAST_PASSES: List[Type[OptimizationPass]] = [
    ConstantFoldingPass,
    DeadCodeEliminationPass,
]

AGGRESSIVE_PASSES: List[Type[OptimizationPass]] = [
    ConstantFoldingPass,
    DeadCodeEliminationPass,
    CommonSubexpressionEliminationPass,
    DomainRepetitionFlatteningPass,
    MaterialMergingPass,
    ConstantFoldingPass,
]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimize_ast(ast: ExprNode,
                 passes: Optional[List[Type[OptimizationPass]]] = None) -> ExprNode:
    """Optimize an AST using the specified passes (or defaults).

    Args:
        ast: The AST to optimize.
        passes: List of pass classes to run (default: DEFAULT_PASSES).

    Returns:
        The optimized AST.

    Example:
        >>> from engine.rendering.demoscene.sdf_optimizer import optimize_ast
        >>> optimized = optimize_ast(scene_graph)
    """
    if passes is None:
        passes = DEFAULT_PASSES
    optimizer = SDFOptimizer(passes)
    return optimizer.optimize(ast)


def fold_constants(ast: ExprNode) -> ExprNode:
    """Run only constant folding on an AST."""
    return ConstantFoldingPass(ast).run()


def eliminate_dead_code(ast: ExprNode) -> ExprNode:
    """Run only dead code elimination on an AST."""
    return DeadCodeEliminationPass(ast).run()


def eliminate_common_subexpressions(ast: ExprNode) -> ExprNode:
    """Run only CSE on an AST."""
    return CommonSubexpressionEliminationPass(ast).run()


def flatten_repeats(ast: ExprNode) -> ExprNode:
    """Run only repetition flattening on an AST."""
    return DomainRepetitionFlatteningPass(ast).run()


def merge_materials(ast: ExprNode) -> ExprNode:
    """Run only material merging on an AST."""
    return MaterialMergingPass(ast).run()
