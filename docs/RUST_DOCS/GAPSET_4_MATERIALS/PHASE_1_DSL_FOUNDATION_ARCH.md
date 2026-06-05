# Phase 1: DSL Foundation — Architecture

## Status: STUB ONLY

The DSL system exists as a scaffold in `trinity/materials/`. No functional compilation pipeline exists.

## Current Architecture

```
trinity/materials/
├── __init__.py       # Re-exports Material, surface
├── dsl.py            # Material base class, SurfaceContext, SurfaceOutput, surface decorator
└── compiler.py       # MaterialCompiler stub (returns placeholder WGSL)
```

## Components

### `dsl.py` — Material Base Classes

```python
class SurfaceOutput:
    # Fields: albedo, metallic, roughness, emission, alpha, normal
    # No WGSL generation

class Material:
    # Base class. Override surface(ctx, out)
    def surface(self, ctx, out): pass

class SurfaceContext:
    # Stub methods: sample(), noise(), texture()
    # All bodies are `...` (ellipsis)

def surface(func):
    # Decorator: marks func._is_surface = True
```

### `compiler.py` — MaterialCompiler Stub

```python
class MaterialCompiler:
    TYPE_MAP = {float: "f32", int: "i32", bool: "bool", str: "str"}
    
    def compile(self, material_class) -> str:
        # Calls inspect.getsource(), ast.parse(), _walk()
        # Returns "// WGSL surface body placeholder"
    
    def _walk(self, node) -> str:
        # Stub — no AST node type support
```

## Missing for Functional Implementation

1. **MaterialMeta metaclass** — `__init_subclass__` hook to intercept class creation, extract `surface()` source, invoke AST walker
2. **AST->WGSL translator** — 15 node types mapping to WGSL strings in `compiler.py`
3. **PBR template** — WGSL template wrapping translated body with bindless MaterialTable pattern
4. **Builtins library** — WGSL noise, math, color functions
5. **Texture binding descriptors** — Texture2D/TextureCube Python classes generating bindings
6. **SurfaceContext WGSL generation** — `sample()`, `world_position()`, `time()` etc. generating WGSL

## Cross-References

- `GAP_4_SUMMARY.md` → Phase 1 task verification
- `GAPSET_3_BRIDGE/PHASE_8_COMMAND_CHANNEL_MATERIAL_DSL_ARCH.md` — Original DSL architecture plan
- `CLARIFICATION.md` — DSL vs node-based divergence
- `crates/renderer-backend/shaders/pbr.frag.wgsl` — Target template for DSL output
