# engine/rendering/demoscene Investigation

**Lines**: 1,130 (wgsl_codegen.py: 645, ast_nodes.py: 244, ast_builder.py: 241)
**Classification**: REAL

## File Analysis

### 1. wgsl_codegen.py (645 lines) - REAL
**Purpose**: WGSL shader code generation from SceneGraph AST

**Key algorithms/classes**:
- `WgslCodeGen` class: Full AST walker that emits valid WGSL source code
- Complete SDF primitive function templates: `sdSphere`, `sdBox`, `sdTorus`, `sdCylinder`, `sdCone`, `sdPlane`, `sdCapsule`
- Domain operation chain emission with distance compensation for non-isometric ops (KIFS, Stretch)
- Material system with PBR struct and `scene_material()` switch function
- Scene entry point generation: `sd_scene(p: vec3<f32>) -> vec2<f32>`
- Pipeline expression builder for domain transformations
- Compensation factor calculation for sphere tracing accuracy

**Notable**: Uses Inigo Quilez SDF naming conventions, implements correct distance compensation math

### 2. ast_nodes.py (244 lines) - REAL
**Purpose**: AST node definitions for SDF scene representation

**Key algorithms/classes**:
- Base `ExprNode` with `walk()`, `children()`, `pretty()`, `label()` methods
- Primitive nodes: `FloatNode`, `Vec3Node`, `PositionNode`
- Domain operation nodes: `RepeatNode`, `CellIdNode`, `MirrorNode`, `KifsNode`, `TwistNode`, `BendNode`, `StretchNode`
- SDF primitive nodes: `SphereNode`, `BoxNode`, `TorusNode`, `CylinderNode`, `ConeNode`, `PlaneNode`, `CapsuleNode`
- CSG combine nodes: `UnionNode`, `IntersectionNode`, `SubtractionNode`
- `MaterialNode` with PBR properties (albedo, roughness, metallic, emissive, ambient_occlusion)
- `SceneGraph` root node containing primitives, pipeline, materials

**Notable**: All nodes are frozen dataclasses (immutable), proper tree traversal infrastructure

### 3. ast_builder.py (241 lines) - REAL
**Purpose**: Build AST from dict descriptions, lambdas, or DSL objects

**Key algorithms/classes**:
- `AstBuilder.walk()`: Multi-dispatch walker handling dicts, lists, ExprNodes, callables, DSL objects
- `_COMPOSITION_DISPATCH`: Maps domain operation names to node constructors
- `_PRIMITIVE_DISPATCH`: Maps SDF function names to node constructors
- `_MARKER_DISPATCH`: Maps type strings to node builders
- `walk_composition()`: Disassembles Python lambdas via `ast` module to extract SDF compositions
- `_build_ast_from_call()`: Recursively builds AST from Python AST Call nodes
- `_disassemble_lambda()`: Introspects Python source to parse SDF DSL expressions

**Notable**: Real metaprogramming - parses Python lambdas to extract SDF scene descriptions

## Key Findings

This subsystem implements a complete **WGSL shader code generator for signed distance field (SDF) rendering**:

1. **Full AST-to-WGSL compilation pipeline**: Python scene description -> AST nodes -> valid WGSL shader code
2. **Complete SDF primitive library**: 7 primitives with mathematically correct distance functions
3. **Domain operations with distance compensation**: Handles non-isometric transformations (KIFS, stretch) with proper sphere tracing correction factors
4. **PBR material system**: Full material struct with albedo, roughness, metallic, emissive, AO
5. **CSG support**: Union, intersection, subtraction nodes for combining primitives
6. **Python DSL introspection**: Can parse Python lambdas like `lambda p: sdSphere(domain_twist(p, 2.0), 1.0)` into AST

This is a **production-quality demoscene/raymarching toolchain**, following Inigo Quilez conventions used in Shadertoy and professional SDF rendering.

## Evidence

### REAL: Complete SDF math implementation (wgsl_codegen.py lines 67-69)
```python
_SDF_SPHERE_FN = """\
fn sdSphere(p: vec3<f32>, r: f32) -> f32 {
    return length(p) - r;
}
"""
```

### REAL: Distance compensation for non-isometric ops (wgsl_codegen.py lines 394-409)
```python
def _build_kifs_compensation(self, op: KifsNode) -> str:
    return f"""\
fn domain_kifs_compensation(folds: f32) -> f32 {{
    let safe_folds = max(abs(folds), 1.0);
    let angle = 6.283185307179586 / safe_folds;
    let half_angle = angle * 0.5;
    let per_fold = cos(half_angle);
    var comp: f32 = 1.0;
    for (var i = 0u; i < u32(safe_folds); i = i + 1u) {{
        comp *= per_fold;
    }}
    return max(comp, 1e-8);
}}
"""
```

### REAL: Python AST introspection for DSL (ast_builder.py lines 142-155)
```python
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
```

### REAL: Full scene graph with traversal (ast_nodes.py lines 200-223)
```python
@dataclass(frozen=True)
class SceneGraph(ExprNode):
    primitives: tuple[SdfPrimitiveNode, ...]
    pipeline: tuple[DomainOpNode, ...] = ()
    materials: tuple[MaterialNode, ...] = ()
    name: str = ""
    def children(self):
        return (*self.pipeline, *self.primitives)
    def deep_label(self):
        lines = [f"SceneGraph: {self.name or '(unnamed)'}"]
        if self.pipeline:
            lines.append("  Pipeline:")
            for op in self.pipeline:
                lines.append(f"    {op.label()}")
        # ... full pretty-printing
```

**No stubs detected**: Zero `pass`, `NotImplementedError`, or placeholder implementations found.
