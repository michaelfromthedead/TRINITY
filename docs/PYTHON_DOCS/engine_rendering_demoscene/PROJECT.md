# PROJECT: engine/rendering/demoscene

## Overview

**Classification**: REAL (production-quality code)
**Lines**: 1,130 total
- wgsl_codegen.py: 645 lines
- ast_nodes.py: 244 lines
- ast_builder.py: 241 lines

## Scope

This subsystem implements a complete WGSL shader code generator for signed distance field (SDF) rendering. It provides:

1. **AST-to-WGSL Compilation Pipeline**: Python scene description to valid WGSL shader code
2. **SDF Primitive Library**: 7 primitives with mathematically correct distance functions
3. **Domain Operations with Distance Compensation**: Non-isometric transformations with sphere tracing correction
4. **PBR Material System**: Full material struct with albedo, roughness, metallic, emissive, AO
5. **CSG Support**: Union, intersection, subtraction nodes for combining primitives
6. **Python DSL Introspection**: Parse Python lambdas into AST representation

## Goals

1. Maintain production-quality demoscene/raymarching toolchain
2. Follow Inigo Quilez SDF conventions (Shadertoy/professional standard)
3. Support complete AST-based scene graph manipulation
4. Enable both dict-based and lambda-based scene descriptions
5. Generate valid, optimized WGSL shader code

## Constraints

- All AST nodes are frozen dataclasses (immutable)
- Distance compensation required for non-isometric domain operations (KIFS, Stretch)
- Must support proper tree traversal infrastructure
- Lambda introspection requires source availability (no compiled-only code)

## Files

| File | Lines | Purpose |
|------|-------|---------|
| wgsl_codegen.py | 645 | WGSL shader code generation from SceneGraph AST |
| ast_nodes.py | 244 | AST node definitions for SDF scene representation |
| ast_builder.py | 241 | Build AST from dict descriptions, lambdas, or DSL objects |

## Acceptance Criteria

### Code Generation
- [ ] All 7 SDF primitives emit correct WGSL functions
- [ ] Domain operations emit with distance compensation where required
- [ ] Materials generate PBR struct and scene_material() switch
- [ ] Scene entry point sd_scene() generates correctly
- [ ] Pipeline expression builder chains transformations properly

### AST System
- [ ] All nodes support walk(), children(), pretty(), label() methods
- [ ] Frozen dataclasses maintain immutability
- [ ] SceneGraph root properly contains primitives, pipeline, materials

### Builder
- [ ] Multi-dispatch walker handles dicts, lists, ExprNodes, callables, DSL objects
- [ ] Lambda disassembly extracts SDF compositions correctly
- [ ] All dispatchers map operations to correct node constructors

## Quality Status

**No stubs detected**: Zero `pass`, `NotImplementedError`, or placeholder implementations found.

This is production-quality code following professional SDF rendering standards.
