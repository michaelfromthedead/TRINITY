# CLARIFICATION: engine/rendering/demoscene

## Philosophical Framing

### What This System Is

This is a **demoscene/raymarching toolchain** that bridges Python scene descriptions to GPU-executable WGSL shaders. The system embodies the mathematical elegance of signed distance fields (SDFs), where complex 3D geometry emerges from simple mathematical functions composed through domain operations.

### Why This Architecture

**AST-Based Design**: Rather than string concatenation or templating, the system uses a proper abstract syntax tree. This enables:
- Optimization passes before code emission
- Validation of scene structure
- Pretty-printing for debugging
- Clean separation between representation and emission

**Immutable Nodes**: Frozen dataclasses prevent accidental mutation during traversal and enable potential caching/memoization of computed values.

**Multi-Dispatch Builder**: The walker pattern with dispatch tables allows extensibility - new node types or DSL constructs can be added by extending the dispatch tables rather than modifying core walker logic.

## Design Rationale

### Inigo Quilez Conventions

The SDF primitive naming (`sdSphere`, `sdBox`, etc.) follows Inigo Quilez's Shadertoy conventions. This is intentional:
- Demoscene practitioners recognize the API immediately
- Existing Shadertoy shaders can inform scene construction
- Mathematical formulations are well-documented in IQ's articles

### Distance Compensation

Non-isometric domain operations (KIFS, Stretch) distort the distance field in ways that break sphere tracing accuracy. The compensation factor calculation addresses this by:
- Computing the worst-case distance scaling
- Applying correction to maintain safe stepping distances
- Preventing artifacts from over-stepping

### Lambda Introspection

The ability to parse Python lambdas like `lambda p: sdSphere(domain_twist(p, 2.0), 1.0)` into AST is powerful metaprogramming that:
- Allows natural Python syntax for scene description
- Avoids verbose dict construction for simple scenes
- Leverages Python's own parser via `ast` module

This requires source availability at runtime, which is a trade-off accepted for the ergonomic benefits.

## Component Responsibilities

### wgsl_codegen.py

The **emitter** - walks the AST and produces valid WGSL source. Responsibilities:
- SDF function templates for all primitives
- Domain operation chain emission
- Compensation factor injection for non-isometric ops
- Material system code generation
- Scene entry point construction

### ast_nodes.py

The **representation** - defines the vocabulary of scene elements. Responsibilities:
- Base ExprNode interface with traversal methods
- Primitive nodes (geometric shapes)
- Domain operation nodes (space transformations)
- CSG combine nodes (boolean operations)
- Material and SceneGraph containers

### ast_builder.py

The **parser** - transforms input representations into AST. Responsibilities:
- Multi-dispatch walking of heterogeneous inputs
- Dict-to-node mapping via dispatch tables
- Lambda disassembly via Python AST inspection
- DSL object handling

## Integration Points

### Upstream

Scene descriptions arrive as:
- Dict structures describing scene hierarchy
- Python lambdas encoding SDF expressions
- DSL objects from higher-level scene authoring tools

### Downstream

Generated WGSL code feeds into:
- GPU shader compilation
- Renderer pipeline binding
- Hot-reload systems for live coding

## PBR Material Model

The material system uses physically-based rendering properties:
- **albedo**: Base color (RGB)
- **roughness**: Surface micro-facet distribution
- **metallic**: Conductor vs dielectric behavior
- **emissive**: Self-illumination
- **ambient_occlusion**: Pre-computed local shadowing

This aligns with modern rendering standards (glTF 2.0, USD, etc.) while keeping the demoscene aesthetic.
