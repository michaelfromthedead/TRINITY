"""
DSL Pattern Optimization Passes (T-DEMO-8.2).

Advanced optimization passes for the demoscene DSL:
  - Pattern Matching Optimization
  - Enhanced Common Subexpression Elimination (CSE)
  - Automatic LOD for Distant Rays
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .ast_nodes import (
    Axis, BendNode, BoxNode, CapsuleNode, CellIdNode, CombineNode,
    ConeNode, CylinderNode, DomainOpNode, EllipsoidNode,
    ExprNode, FloatNode, IntersectionNode, KifsNode, MirrorNode,
    OctahedronNode, PlaneNode, PositionNode, PyramidNode, RepeatNode,
    RoundedBoxNode, BoxFrameNode, SceneGraph, SdfPrimitiveNode, SphereNode,
    StretchNode, SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
)


__all__ = [
    "PatternMatchingPass", "EnhancedCSEPass", "LODSimplificationPass",
    "DSLPatternOptimizer", "PATTERN_PASSES", "LOD_PASSES", "ALL_PATTERN_PASSES",
    "pattern_optimize", "cse_optimize", "lod_simplify",
    "PatternDatabase", "SDFPattern", "PatternMatch",
    "LODLevel", "LODConfig", "CSECache", "ExpressionStats",
]


class PatternType(Enum):
    """Types of recognized SDF patterns."""
    ROUNDED_BOX = auto()
    HOLLOW_BOX = auto()
    CHAMFERED_BOX = auto()
    RING = auto()
    DISC = auto()
    ROD = auto()
    DOME = auto()
    SHELL = auto()
    CAPSULE_APPROX = auto()


@dataclass(frozen=True)
class SDFPattern:
    """Definition of an SDF pattern to match."""
    pattern_type: PatternType
    description: str
    match_fn: Callable[[ExprNode], Optional["PatternMatch"]]
    replacement_fn: Callable[["PatternMatch"], ExprNode]
    speedup_estimate: float = 1.0


@dataclass
class PatternMatch:
    """Result of a successful pattern match."""
    pattern_type: PatternType
    matched_node: ExprNode
    extracted_params: Dict[str, Any]
    confidence: float = 1.0


class PatternDatabase:
    """Database of known SDF patterns for optimization."""

    def __init__(self) -> None:
        self._patterns: List[SDFPattern] = []
        self._stats: Dict[PatternType, int] = {}
        self._register_builtin_patterns()

    def _register_builtin_patterns(self) -> None:
        """Register built-in SDF patterns."""
        self._patterns.append(SDFPattern(
            PatternType.ROUNDED_BOX,
            "Box with rounded corners via sphere subtraction",
            self._match_rounded_box, self._replace_rounded_box, 1.5))
        self._patterns.append(SDFPattern(
            PatternType.SHELL,
            "Hollow sphere via subtraction of smaller sphere",
            self._match_shell, self._replace_shell, 1.8))
        self._patterns.append(SDFPattern(
            PatternType.DOME,
            "Half-sphere via intersection with plane",
            self._match_dome, self._replace_dome, 1.3))
        self._patterns.append(SDFPattern(
            PatternType.RING,
            "Ring shape (torus with ratio > 4:1)",
            self._match_ring, self._replace_ring, 1.2))
        self._patterns.append(SDFPattern(
            PatternType.DISC,
            "Flat disc (cylinder with height < radius/4)",
            self._match_disc, self._replace_disc, 1.4))
        self._patterns.append(SDFPattern(
            PatternType.ROD,
            "Thin rod (cylinder with radius < height/10)",
            self._match_rod, self._replace_rod, 1.3))
        self._patterns.append(SDFPattern(
            PatternType.HOLLOW_BOX,
            "Hollow box via inner box subtraction",
            self._match_hollow_box, self._replace_hollow_box, 1.6))

    def register_pattern(self, pattern: SDFPattern) -> None:
        """Register a custom pattern."""
        self._patterns.append(pattern)

    def match(self, node: ExprNode) -> Optional[PatternMatch]:
        """Try to match the node against all registered patterns."""
        for pattern in self._patterns:
            m = pattern.match_fn(node)
            if m is not None:
                self._stats[m.pattern_type] = self._stats.get(m.pattern_type, 0) + 1
                return m
        return None

    def get_pattern(self, pt: PatternType) -> Optional[SDFPattern]:
        """Get a specific pattern by type."""
        for p in self._patterns:
            if p.pattern_type == pt:
                return p
        return None

    @property
    def stats(self) -> Dict[PatternType, int]:
        return self._stats.copy()

    def _match_rounded_box(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, SubtractionNode):
            return None
        if not isinstance(node.left, BoxNode):
            return None
        if not isinstance(node.right, (SphereNode, UnionNode)):
            return None
        params = {"position": node.left.position, "size": node.left.size}
        if isinstance(node.right, SphereNode):
            params["corner_radius"] = node.right.radius.value
        return PatternMatch(PatternType.ROUNDED_BOX, node, params, 0.85)

    def _replace_rounded_box(self, match: PatternMatch) -> ExprNode:
        p = match.extracted_params
        return RoundedBoxNode(p["position"], p["size"],
                              FloatNode(p.get("corner_radius", 0.1)))

    def _match_shell(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, SubtractionNode):
            return None
        if not isinstance(node.left, SphereNode) or not isinstance(node.right, SphereNode):
            return None
        if node.right.radius.value >= node.left.radius.value:
            return None
        thickness = node.left.radius.value - node.right.radius.value
        return PatternMatch(PatternType.SHELL, node, {
            "position": node.left.position,
            "outer_radius": node.left.radius.value,
            "inner_radius": node.right.radius.value,
            "thickness": thickness}, 0.95)

    def _replace_shell(self, match: PatternMatch) -> ExprNode:
        return match.matched_node

    def _match_dome(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, IntersectionNode):
            return None
        sphere, plane = None, None
        if isinstance(node.left, SphereNode) and isinstance(node.right, PlaneNode):
            sphere, plane = node.left, node.right
        elif isinstance(node.left, PlaneNode) and isinstance(node.right, SphereNode):
            plane, sphere = node.left, node.right
        else:
            return None
        return PatternMatch(PatternType.DOME, node, {
            "position": sphere.position, "radius": sphere.radius.value,
            "plane_normal": plane.normal, "plane_distance": plane.distance.value}, 0.9)

    def _replace_dome(self, match: PatternMatch) -> ExprNode:
        return match.matched_node

    def _match_ring(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, TorusNode):
            return None
        if node.minor_radius.value <= 0:
            return None
        ratio = node.major_radius.value / node.minor_radius.value
        if ratio < 4.0:
            return None
        return PatternMatch(PatternType.RING, node, {
            "position": node.position, "major": node.major_radius.value,
            "minor": node.minor_radius.value, "ratio": ratio}, 0.9)

    def _replace_ring(self, match: PatternMatch) -> ExprNode:
        return match.matched_node

    def _match_disc(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, CylinderNode):
            return None
        if node.radius.value <= 0 or node.height.value >= node.radius.value / 4:
            return None
        return PatternMatch(PatternType.DISC, node, {
            "position": node.position, "radius": node.radius.value,
            "height": node.height.value}, 0.95)

    def _replace_disc(self, match: PatternMatch) -> ExprNode:
        return match.matched_node

    def _match_rod(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, CylinderNode):
            return None
        if node.height.value <= 0 or node.radius.value >= node.height.value / 10:
            return None
        return PatternMatch(PatternType.ROD, node, {
            "position": node.position, "radius": node.radius.value,
            "height": node.height.value}, 0.9)

    def _replace_rod(self, match: PatternMatch) -> ExprNode:
        return match.matched_node

    def _match_hollow_box(self, node: ExprNode) -> Optional[PatternMatch]:
        if not isinstance(node, SubtractionNode):
            return None
        if not isinstance(node.left, BoxNode) or not isinstance(node.right, BoxNode):
            return None
        outer, inner = node.left, node.right
        if not isinstance(outer.size, Vec3Node) or not isinstance(inner.size, Vec3Node):
            return None
        if (inner.size.x >= outer.size.x or inner.size.y >= outer.size.y or
                inner.size.z >= outer.size.z):
            return None
        thickness = min(outer.size.x - inner.size.x, outer.size.y - inner.size.y,
                        outer.size.z - inner.size.z) / 2.0
        return PatternMatch(PatternType.HOLLOW_BOX, node, {
            "position": outer.position, "outer_size": outer.size,
            "inner_size": inner.size, "thickness": thickness}, 0.9)

    def _replace_hollow_box(self, match: PatternMatch) -> ExprNode:
        p = match.extracted_params
        return BoxFrameNode(p["position"], p["outer_size"], FloatNode(p["thickness"]))


@dataclass
class ExpressionStats:
    """Statistics for a tracked expression."""
    hash_value: int
    node: ExprNode
    occurrences: int = 1
    cached_var_name: Optional[str] = None


class CSECache:
    """Cache for Common Subexpression Elimination with memoization."""

    def __init__(self) -> None:
        self._cache: Dict[int, ExpressionStats] = {}
        self._var_counter: int = 0
        self._total_hits: int = 0
        self._total_misses: int = 0

    def compute_hash(self, node: ExprNode) -> int:
        return self._hash_node(node)

    def _hash_node(self, node: ExprNode) -> int:
        hasher = hashlib.md5(usedforsecurity=False)
        hasher.update(type(node).__name__.encode())
        if isinstance(node, FloatNode):
            hasher.update(str(node.value).encode())
        elif isinstance(node, Vec3Node):
            hasher.update(f"{node.x},{node.y},{node.z}".encode())
        elif isinstance(node, PositionNode):
            hasher.update(b"position")
        elif isinstance(node, SphereNode):
            hasher.update(str(self._hash_node(node.position)).encode())
            hasher.update(str(self._hash_node(node.radius)).encode())
        elif isinstance(node, BoxNode):
            hasher.update(str(self._hash_node(node.position)).encode())
            hasher.update(str(self._hash_node(node.size)).encode())
        elif isinstance(node, CombineNode):
            hasher.update(node.kind.encode())
            hasher.update(str(self._hash_node(node.left)).encode())
            hasher.update(str(self._hash_node(node.right)).encode())
        elif isinstance(node, DomainOpNode):
            hasher.update(str(self._hash_node(node.input)).encode())
        elif isinstance(node, SceneGraph):
            for p in node.primitives:
                hasher.update(str(self._hash_node(p)).encode())
        return int(hasher.hexdigest(), 16) & ((1 << 64) - 1)

    def track(self, node: ExprNode) -> Tuple[int, bool]:
        h = self.compute_hash(node)
        if h in self._cache:
            self._cache[h].occurrences += 1
            self._total_hits += 1
            return h, True
        self._cache[h] = ExpressionStats(h, node, 1)
        self._total_misses += 1
        return h, False

    def get_duplicates(self, min_occurrences: int = 2) -> List[ExpressionStats]:
        return [s for s in self._cache.values() if s.occurrences >= min_occurrences]

    def assign_variable(self, h: int) -> str:
        if h in self._cache and self._cache[h].cached_var_name is None:
            self._cache[h].cached_var_name = f"_expr_{self._var_counter}"
            self._var_counter += 1
        return self._cache[h].cached_var_name if h in self._cache else f"_expr_{h}"

    @property
    def hit_rate(self) -> float:
        total = self._total_hits + self._total_misses
        return self._total_hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        return {"total_entries": len(self._cache), "total_hits": self._total_hits,
                "total_misses": self._total_misses, "hit_rate": self.hit_rate,
                "duplicates": len(self.get_duplicates())}


class LODLevel(Enum):
    """Level of detail for SDF primitives."""
    FULL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BILLBOARD = 4


@dataclass
class LODConfig:
    """Configuration for automatic LOD."""
    distance_thresholds: Dict[LODLevel, float] = field(default_factory=lambda: {
        LODLevel.FULL: 0.0, LODLevel.HIGH: 10.0, LODLevel.MEDIUM: 25.0,
        LODLevel.LOW: 50.0, LODLevel.BILLBOARD: 100.0})
    primitive_complexity: Dict[type, float] = field(default_factory=lambda: {
        SphereNode: 1.0, BoxNode: 1.2, PlaneNode: 0.8, CylinderNode: 1.5,
        ConeNode: 1.6, TorusNode: 2.0, CapsuleNode: 1.4, EllipsoidNode: 1.8,
        BoxFrameNode: 2.5, RoundedBoxNode: 1.8, OctahedronNode: 2.0, PyramidNode: 1.7})
    enable_sphere_proxy: bool = True
    enable_skip_detail: bool = True
    skip_detail_distance: float = 75.0
    complexity_threshold: float = 2.0


@dataclass
class LODResult:
    """Result of LOD simplification."""
    original_node: ExprNode
    simplified_node: Optional[ExprNode]
    lod_level: LODLevel
    complexity_reduction: float


class PatternOptimizationPass(ABC):
    """Base class for pattern optimization passes."""

    def __init__(self, ast: ExprNode) -> None:
        self.ast = ast
        self._stats: Dict[str, Any] = {}

    @abstractmethod
    def run(self) -> ExprNode:
        pass

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def _increment_stat(self, key: str, count: int = 1) -> None:
        self._stats[key] = self._stats.get(key, 0) + count


class PatternMatchingPass(PatternOptimizationPass):
    """Pattern Matching Optimization Pass."""

    def __init__(self, ast: ExprNode, pattern_db: Optional[PatternDatabase] = None):
        super().__init__(ast)
        self._pattern_db = pattern_db or PatternDatabase()
        self._matches: List[PatternMatch] = []

    def run(self) -> ExprNode:
        result = self._optimize(self.ast)
        self._stats["patterns_matched"] = len(self._matches)
        self._stats["pattern_types"] = dict(self._pattern_db.stats)
        return result

    def _optimize(self, node: ExprNode) -> ExprNode:
        m = self._pattern_db.match(node)
        if m is not None:
            self._matches.append(m)
            p = self._pattern_db.get_pattern(m.pattern_type)
            if p is not None:
                return p.replacement_fn(m)
        if isinstance(node, SceneGraph):
            return self._optimize_scene_graph(node)
        elif isinstance(node, CombineNode):
            return self._optimize_combine(node)
        elif isinstance(node, DomainOpNode):
            return self._optimize_domain_op(node)
        return node

    def _optimize_scene_graph(self, graph: SceneGraph) -> SceneGraph:
        new_prims = tuple(self._optimize(p) for p in graph.primitives
                         if isinstance(self._optimize(p), SdfPrimitiveNode))
        new_pipe = tuple(self._optimize(o) for o in graph.pipeline
                        if isinstance(self._optimize(o), DomainOpNode))
        if new_prims == graph.primitives and new_pipe == graph.pipeline:
            return graph
        return SceneGraph(new_prims, new_pipe, graph.name)

    def _optimize_combine(self, node: CombineNode) -> ExprNode:
        left, right = self._optimize(node.left), self._optimize(node.right)
        if left is node.left and right is node.right:
            return node
        if isinstance(node, UnionNode):
            return UnionNode(left, right)
        elif isinstance(node, IntersectionNode):
            return IntersectionNode(left, right)
        elif isinstance(node, SubtractionNode):
            return SubtractionNode(left, right)
        return node

    def _optimize_domain_op(self, node: DomainOpNode) -> ExprNode:
        inner = self._optimize(node.input)
        if inner is node.input:
            return node
        if isinstance(node, RepeatNode):
            return RepeatNode(inner, node.cell_size)
        elif isinstance(node, MirrorNode):
            return MirrorNode(inner, node.axis)
        elif isinstance(node, TwistNode):
            return TwistNode(inner, node.rate)
        elif isinstance(node, BendNode):
            return BendNode(inner, node.radius)
        elif isinstance(node, StretchNode):
            return StretchNode(inner, node.stretch, node.axis)
        elif isinstance(node, KifsNode):
            return KifsNode(inner, node.folds)
        elif isinstance(node, CellIdNode):
            return CellIdNode(inner, node.cell_size)
        return node

    @property
    def matches(self) -> List[PatternMatch]:
        return self._matches.copy()


class EnhancedCSEPass(PatternOptimizationPass):
    """Enhanced Common Subexpression Elimination Pass."""

    def __init__(self, ast: ExprNode, cache: Optional[CSECache] = None):
        super().__init__(ast)
        self._cache = cache or CSECache()
        self._hoisted_vars: List[Tuple[str, ExprNode]] = []

    def run(self) -> ExprNode:
        self._collect_expressions(self.ast)
        duplicates = self._cache.get_duplicates(2)
        if not duplicates:
            self._stats.update(self._cache.stats)
            return self.ast
        for dup in duplicates:
            var = self._cache.assign_variable(dup.hash_value)
            self._hoisted_vars.append((var, dup.node))
        self._stats.update(self._cache.stats)
        self._stats["hoisted_expressions"] = len(self._hoisted_vars)
        self._stats["evaluation_savings"] = sum(d.occurrences - 1 for d in duplicates)
        return self.ast

    def _collect_expressions(self, node: ExprNode) -> None:
        self._cache.track(node)
        for child in node.children():
            self._collect_expressions(child)

    @property
    def hoisted_variables(self) -> List[Tuple[str, ExprNode]]:
        return self._hoisted_vars.copy()

    @property
    def cache(self) -> CSECache:
        return self._cache


class LODSimplificationPass(PatternOptimizationPass):
    """LOD Simplification Pass."""

    def __init__(self, ast: ExprNode, config: Optional[LODConfig] = None,
                 camera_distance: float = 0.0):
        super().__init__(ast)
        self._config = config or LODConfig()
        self._camera_distance = camera_distance
        self._lod_results: List[LODResult] = []

    def run(self) -> ExprNode:
        result = self._simplify(self.ast)
        self._stats["primitives_simplified"] = len(self._lod_results)
        self._stats["total_complexity_reduction"] = sum(
            r.complexity_reduction for r in self._lod_results)
        return result

    def _simplify(self, node: ExprNode) -> Optional[ExprNode]:
        if isinstance(node, SceneGraph):
            return self._simplify_scene_graph(node)
        elif isinstance(node, SdfPrimitiveNode):
            return self._simplify_primitive(node)
        elif isinstance(node, CombineNode):
            return self._simplify_combine(node)
        elif isinstance(node, DomainOpNode):
            return self._simplify_domain_op(node)
        return node

    def _simplify_scene_graph(self, graph: SceneGraph) -> SceneGraph:
        new_prims = []
        for prim in graph.primitives:
            s = self._simplify_primitive(prim)
            if s is not None and isinstance(s, SdfPrimitiveNode):
                new_prims.append(s)
        if tuple(new_prims) == graph.primitives:
            return graph
        return SceneGraph(tuple(new_prims), graph.pipeline, graph.name)

    def _simplify_primitive(self, prim: SdfPrimitiveNode) -> Optional[ExprNode]:
        lod = self._get_lod_level(self._camera_distance)
        complexity = self._config.primitive_complexity.get(type(prim), 1.5)
        if (self._config.enable_skip_detail and
            self._camera_distance > self._config.skip_detail_distance and
                complexity > self._config.complexity_threshold):
            self._lod_results.append(LODResult(prim, None, LODLevel.BILLBOARD, complexity))
            return None
        if (self._config.enable_sphere_proxy and lod == LODLevel.BILLBOARD and
                not isinstance(prim, SphereNode)):
            r = self._get_bounding_radius(prim)
            simplified = SphereNode(prim.position, FloatNode(r))
            self._lod_results.append(LODResult(prim, simplified, lod, complexity - 1.0))
            return simplified
        return prim

    def _simplify_combine(self, node: CombineNode) -> Optional[ExprNode]:
        left, right = self._simplify(node.left), self._simplify(node.right)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        if left is node.left and right is node.right:
            return node
        if isinstance(node, UnionNode):
            return UnionNode(left, right)
        elif isinstance(node, IntersectionNode):
            return IntersectionNode(left, right)
        elif isinstance(node, SubtractionNode):
            return SubtractionNode(left, right)
        return node

    def _simplify_domain_op(self, node: DomainOpNode) -> Optional[ExprNode]:
        inner = self._simplify(node.input)
        if inner is None:
            return None
        if inner is node.input:
            return node
        if isinstance(node, RepeatNode):
            return RepeatNode(inner, node.cell_size)
        elif isinstance(node, MirrorNode):
            return MirrorNode(inner, node.axis)
        elif isinstance(node, TwistNode):
            return TwistNode(inner, node.rate)
        elif isinstance(node, BendNode):
            return BendNode(inner, node.radius)
        elif isinstance(node, StretchNode):
            return StretchNode(inner, node.stretch, node.axis)
        elif isinstance(node, KifsNode):
            return KifsNode(inner, node.folds)
        elif isinstance(node, CellIdNode):
            return CellIdNode(inner, node.cell_size)
        return node

    def _get_lod_level(self, dist: float) -> LODLevel:
        t = self._config.distance_thresholds
        if dist >= t[LODLevel.BILLBOARD]:
            return LODLevel.BILLBOARD
        if dist >= t[LODLevel.LOW]:
            return LODLevel.LOW
        if dist >= t[LODLevel.MEDIUM]:
            return LODLevel.MEDIUM
        if dist >= t[LODLevel.HIGH]:
            return LODLevel.HIGH
        return LODLevel.FULL

    def _get_bounding_radius(self, prim: SdfPrimitiveNode) -> float:
        if isinstance(prim, SphereNode):
            return prim.radius.value
        elif isinstance(prim, BoxNode):
            s = prim.size
            return math.sqrt(s.x ** 2 + s.y ** 2 + s.z ** 2)
        elif isinstance(prim, CylinderNode):
            return math.sqrt(prim.radius.value ** 2 + (prim.height.value / 2) ** 2)
        elif isinstance(prim, TorusNode):
            return prim.major_radius.value + prim.minor_radius.value
        elif isinstance(prim, EllipsoidNode):
            return max(prim.radii.x, prim.radii.y, prim.radii.z)
        elif isinstance(prim, RoundedBoxNode):
            s = prim.half_extents
            return math.sqrt(s.x ** 2 + s.y ** 2 + s.z ** 2)
        elif isinstance(prim, BoxFrameNode):
            s = prim.half_extents
            return math.sqrt(s.x ** 2 + s.y ** 2 + s.z ** 2)
        elif isinstance(prim, OctahedronNode):
            return prim.scale.value
        elif isinstance(prim, PyramidNode):
            return prim.height.value
        return 1.0

    @property
    def lod_results(self) -> List[LODResult]:
        return self._lod_results.copy()


class DSLPatternOptimizer:
    """Orchestrates pattern optimization passes."""

    def __init__(self, passes: List[Type[PatternOptimizationPass]]) -> None:
        self.passes = passes
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._total_stats: Dict[str, Any] = {}

    def optimize(self, ast: ExprNode, pattern_db: Optional[PatternDatabase] = None,
                 cse_cache: Optional[CSECache] = None,
                 lod_config: Optional[LODConfig] = None,
                 camera_distance: float = 0.0) -> ExprNode:
        self._stats.clear()
        self._total_stats.clear()
        current = ast
        for pass_cls in self.passes:
            if pass_cls == PatternMatchingPass:
                inst = pass_cls(current, pattern_db)
            elif pass_cls == EnhancedCSEPass:
                inst = pass_cls(current, cse_cache)
            elif pass_cls == LODSimplificationPass:
                inst = pass_cls(current, lod_config, camera_distance)
            else:
                inst = pass_cls(current)
            current = inst.run()
            self._stats[pass_cls.__name__] = inst.stats
            for k, v in inst.stats.items():
                if isinstance(v, (int, float)):
                    self._total_stats[k] = self._total_stats.get(k, 0) + v
        return current

    @property
    def stats(self) -> Dict[str, Dict[str, Any]]:
        return self._stats

    @property
    def total_stats(self) -> Dict[str, Any]:
        return self._total_stats


PATTERN_PASSES: List[Type[PatternOptimizationPass]] = [PatternMatchingPass]
CSE_PASSES: List[Type[PatternOptimizationPass]] = [EnhancedCSEPass]
LOD_PASSES: List[Type[PatternOptimizationPass]] = [LODSimplificationPass]
ALL_PATTERN_PASSES: List[Type[PatternOptimizationPass]] = [
    PatternMatchingPass, EnhancedCSEPass, LODSimplificationPass]


def pattern_optimize(ast: ExprNode, db: Optional[PatternDatabase] = None) -> ExprNode:
    return PatternMatchingPass(ast, db).run()


def cse_optimize(ast: ExprNode, cache: Optional[CSECache] = None) -> ExprNode:
    return EnhancedCSEPass(ast, cache).run()


def lod_simplify(ast: ExprNode, cfg: Optional[LODConfig] = None,
                 dist: float = 0.0) -> ExprNode:
    return LODSimplificationPass(ast, cfg, dist).run()
