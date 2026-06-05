from __future__ import annotations
import ast, inspect, textwrap
from typing import Any, Callable, Optional, Sequence, Union
from .ast_nodes import (
    Axis, BendNode, BoxNode, CapsuleNode, CellIdNode, CombineNode, CompensationNode,
    ConeNode, CylinderNode, DomainOpNode, ExprNode, FloatNode,
    IntersectionNode, KifsNode, MirrorNode, PlaneNode,
    PositionNode, RepeatNode, SceneGraph, SdfPrimitiveNode, SphereNode,
    StretchNode, SubtractionNode, TorusNode, TwistNode, UnionNode, Vec3Node,
)

def _to_float(v):
    if isinstance(v, FloatNode): return v
    return FloatNode(float(v))

def _to_vec3(v):
    if isinstance(v, Vec3Node): return v
    if isinstance(v, (tuple, list)) and len(v) == 3:
        return Vec3Node(float(v[0]), float(v[1]), float(v[2]))
    raise TypeError(f"Cannot coerce {v!r} to Vec3Node")

def _to_axis(v):
    if isinstance(v, Axis): return v
    if isinstance(v, str): return Axis(v.lower())
    raise TypeError(f"Cannot coerce {v!r} to Axis")

_COMPOSITION_DISPATCH = {
    "domain_repeat": lambda p, cell_size: RepeatNode(PositionNode() if isinstance(p, PositionNode) else p, _to_vec3(cell_size)),
    "domain_cell_id": lambda p, cell_size: CellIdNode(PositionNode() if isinstance(p, PositionNode) else p, _to_vec3(cell_size)),
    "domain_mirror_x": lambda p: MirrorNode(PositionNode() if isinstance(p, PositionNode) else p, Axis.X),
    "domain_mirror_y": lambda p: MirrorNode(PositionNode() if isinstance(p, PositionNode) else p, Axis.Y),
    "domain_mirror_z": lambda p: MirrorNode(PositionNode() if isinstance(p, PositionNode) else p, Axis.Z),
    "domain_kifs": lambda p, folds=6.0: KifsNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(folds)),
    "domain_twist": lambda p, rate=1.0: TwistNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(rate)),
    "domain_bend": lambda p, radius=5.0: BendNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(radius)),
    "domain_stretch_x": lambda p, factor=2.0: StretchNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(factor), Axis.X),
    "domain_stretch_y": lambda p, factor=2.0: StretchNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(factor), Axis.Y),
    "domain_stretch_z": lambda p, factor=2.0: StretchNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(factor), Axis.Z),
}

_PRIMITIVE_DISPATCH = {
    "sdSphere": lambda p, r=1.0: SphereNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(r)),
    "sdBox": lambda p, sx=1.0, sy=1.0, sz=1.0: BoxNode(PositionNode() if isinstance(p, PositionNode) else p, Vec3Node(_to_float(sx).value, _to_float(sy).value, _to_float(sz).value)),
    "sdTorus": lambda p, major=2.0, minor=0.5: TorusNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(major), _to_float(minor)),
    "sdCylinder": lambda p, h=2.0, r=1.0: CylinderNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(h), _to_float(r)),
    "sdCone": lambda p, h=2.0, r1=0.0, r2=1.0: ConeNode(PositionNode() if isinstance(p, PositionNode) else p, _to_float(h), _to_float(r1), _to_float(r2)),
    "sdPlane": lambda p, nx=0.0, ny=1.0, nz=0.0, d=0.0: PlaneNode(PositionNode() if isinstance(p, PositionNode) else p, Vec3Node(_to_float(nx).value, _to_float(ny).value, _to_float(nz).value), _to_float(d)),
    "sdCapsule": lambda p, a=(0.0, -1.0, 0.0), b=(0.0, 1.0, 0.0), r=0.5: CapsuleNode(PositionNode() if isinstance(p, PositionNode) else p, _to_vec3(a), _to_vec3(b), _to_float(r)),
}

_MARKER_DISPATCH = {
    "repeat": lambda **kw: RepeatNode(kw.get("input", PositionNode()), _to_vec3(kw.get("cell_size", (2,2,2)))),
    "cell_id": lambda **kw: CellIdNode(kw.get("input", PositionNode()), _to_vec3(kw.get("cell_size", (4,4,1)))),
    "mirror": lambda **kw: MirrorNode(kw.get("input", PositionNode()), _to_axis(kw.get("axis", "x"))),
    "kifs": lambda **kw: KifsNode(kw.get("input", PositionNode()), _to_float(kw.get("folds", 6.0))),
    "twist": lambda **kw: TwistNode(kw.get("input", PositionNode()), _to_float(kw.get("rate", 1.0))),
    "bend": lambda **kw: BendNode(kw.get("input", PositionNode()), _to_float(kw.get("radius", 5.0))),
    "stretch": lambda **kw: StretchNode(kw.get("input", PositionNode()), _to_float(kw.get("stretch", 2.0)), _to_axis(kw.get("axis", "x"))),
    "sphere": lambda **kw: SphereNode(kw.get("position", PositionNode()), _to_float(kw.get("radius", 1.0))),
    "box": lambda **kw: BoxNode(kw.get("position", PositionNode()), _to_vec3(kw.get("size", (1,1,1)))),
    "torus": lambda **kw: TorusNode(kw.get("position", PositionNode()), _to_float(kw.get("major_radius", 2.0)), _to_float(kw.get("minor_radius", 0.5))),
    "cylinder": lambda **kw: CylinderNode(kw.get("position", PositionNode()), _to_float(kw.get("height", 2.0)), _to_float(kw.get("radius", 1.0))),
    "cone": lambda **kw: ConeNode(kw.get("position", PositionNode()), _to_float(kw.get("height", 2.0)), _to_float(kw.get("radius_top", 0.0)), _to_float(kw.get("radius_bottom", 1.0))),
    "plane": lambda **kw: PlaneNode(kw.get("position", PositionNode()), _to_vec3(kw.get("normal", (0,1,0))), _to_float(kw.get("distance", 0.0))),
    "capsule": lambda **kw: CapsuleNode(kw.get("position", PositionNode()), _to_vec3(kw.get("endpoint_a", (0,-1,0))), _to_vec3(kw.get("endpoint_b", (0,1,0))), _to_float(kw.get("radius", 0.5))),
}

_COMPOSITE_DISPATCH = {**_COMPOSITION_DISPATCH, **_PRIMITIVE_DISPATCH}

def _ast_arg_to_node(arg_expr):
    if isinstance(arg_expr, ast.Constant):
        val = arg_expr.value
        if isinstance(val, (int, float)): return FloatNode(float(val))
        return val
    if isinstance(arg_expr, ast.UnaryOp) and isinstance(arg_expr.op, ast.USub):
        inner = _ast_arg_to_node(arg_expr.operand)
        if isinstance(inner, FloatNode): return FloatNode(-inner.value)
        return -inner
    if isinstance(arg_expr, ast.Tuple):
        elts = [_ast_arg_to_node(e) for e in arg_expr.elts]
        if all(isinstance(e, FloatNode) for e in elts):
            return Vec3Node(*[e.value for e in elts])
        return tuple(elts)
    if isinstance(arg_expr, ast.List):
        return [_ast_arg_to_node(e) for e in arg_expr.elts]
    if isinstance(arg_expr, ast.Name) and arg_expr.id == "p":
        return PositionNode()
    return None

def _build_ast_from_call(call_expr):
    func_name = ""
    if isinstance(call_expr.func, ast.Name):
        func_name = call_expr.func.id
    elif isinstance(call_expr.func, ast.Attribute):
        func_name = call_expr.func.attr
    else:
        return None
    if func_name not in _COMPOSITE_DISPATCH:
        return None
    constructor = _COMPOSITE_DISPATCH[func_name]
    resolved_args = []
    for a in call_expr.args:
        if isinstance(a, ast.Call):
            child = _build_ast_from_call(a)
            resolved_args.append(child if child is not None else _ast_arg_to_node(a))
        else:
            resolved_args.append(_ast_arg_to_node(a))
    kw_args = {}
    for kw in call_expr.keywords:
        kw_args[kw.arg] = _ast_arg_to_node(kw.value)
    if kw_args:
        return constructor(*resolved_args, **kw_args)
    return constructor(*resolved_args) if resolved_args else constructor()

def _disassemble_lambda(fn):
    try:
        source = inspect.getsource(fn)
    except (OSError, TypeError):
        raise ValueError(f"Cannot inspect source of {fn!r}")
    source = textwrap.dedent(source).strip()
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda) and isinstance(node.body, ast.Call):
            return _build_ast_from_call(node.body)
    return None

class AstBuilder:
    @classmethod
    def walk(cls, obj):
        return cls._walk(obj)
    @classmethod
    def _walk(cls, obj):
        if isinstance(obj, dict): return cls._walk_dict(obj)
        if isinstance(obj, list): return cls._walk_list(obj)
        if isinstance(obj, ExprNode): return obj
        if callable(obj): return cls._walk_callable(obj)
        return cls._walk_dsl_object(obj)
    @classmethod
    def _walk_dict(cls, d):
        ty = d.get("type", "").lower()
        if ty in _MARKER_DISPATCH:
            processed = {}
            for k, v in d.items():
                if k == "type": continue
                if isinstance(v, dict): processed[k] = cls._walk_dict(v)
                elif isinstance(v, list): processed[k] = cls._walk_list(v)
                else: processed[k] = v
            return _MARKER_DISPATCH[ty](**processed)
        if "pipeline" in d:
            return cls._build_scene(d)
        return {k: cls._walk(v) for k, v in d.items() if k != "type"}
    @classmethod
    def _walk_list(cls, lst):
        return [cls._walk(item) for item in lst]
    @classmethod
    def _walk_callable(cls, fn):
        return walk_composition(fn)
    @classmethod
    def _walk_dsl_object(cls, obj):
        node_type = getattr(obj, "_node_type", None)
        if node_type and node_type in _MARKER_DISPATCH:
            kwargs = {}
            for attr in ("repeat_cell", "mirror_axis", "folds", "rate", "radius", "stretch_factor", "axis", "input", "position", "height", "radius_top", "radius_bottom", "normal", "distance"):
                if hasattr(obj, attr): kwargs[attr] = getattr(obj, attr)
            return _MARKER_DISPATCH[node_type](**kwargs)
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {}
    @classmethod
    def _build_scene(cls, d):
        pipeline = []
        for item in d.get("pipeline", []):
            node = cls._walk(item)
            if isinstance(node, DomainOpNode): pipeline.append(node)
        primitives = []
        for item in d.get("primitives", []):
            node = cls._walk(item)
            if isinstance(node, SdfPrimitiveNode): primitives.append(node)
        return SceneGraph(primitives=tuple(primitives), pipeline=tuple(pipeline), name=d.get("name", ""))

def walk_composition(fn, primitives=None):
    root = _disassemble_lambda(fn)
    if root is None:
        raise ValueError("No function calls found in composition")
    pipeline = []
    resolved_primitives = []
    def _walk_tree(node):
        if isinstance(node, DomainOpNode):
            _walk_tree(node.input)
            pipeline.append(node)
        elif isinstance(node, SdfPrimitiveNode):
            pos = getattr(node, 'position', None)
            if isinstance(pos, DomainOpNode):
                _walk_tree(pos)
            resolved_primitives.append(node)
    _walk_tree(root)
    if not resolved_primitives and primitives:
        for p in primitives:
            if isinstance(p, dict):
                node = AstBuilder._walk(p)
                if isinstance(node, SdfPrimitiveNode): resolved_primitives.append(node)
            elif isinstance(p, SdfPrimitiveNode):
                resolved_primitives.append(p)
    return SceneGraph(primitives=tuple(resolved_primitives), pipeline=tuple(pipeline), name=getattr(fn, "__name__", str(fn)))

build_from_composition = walk_composition
