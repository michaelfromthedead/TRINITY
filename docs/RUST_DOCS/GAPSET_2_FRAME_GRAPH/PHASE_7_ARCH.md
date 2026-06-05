# Phase 7: Bridge + Emit -- Architecture

## Overview
The bridge layer connecting Python pass declarations to Rust compiler execution, plus the final compilation output. Defines the protocol by which Python and Rust communicate and the structure of the compiled result. Covers S1-G3 (Bridge Channel Protocol).

## Current Bridge: JSON Serialization

### Python Side (`FrameGraph.serialize()`, frame_graph.py lines 778-870)
Produces a JSON dict with `passes` and `resources` arrays:

```json
{
  "passes": [
    {
      "name": "GBuffer",
      "pass_type": "Graphics",
      "color_attachments": ["gbuffer_albedo"],
      "depth_attachment": "gbuffer_depth",
      "compute_shader": null,
      "workgroup_size": null,
      "reads": [],
      "writes": ["gbuffer_albedo", "gbuffer_depth"]
    }
  ],
  "resources": [
    {
      "name": "gbuffer_albedo",
      "resource_type": "Texture2D",
      "width": 0, "height": 0, "depth": 1,
      "format": "R8G8B8A8_UNORM",
      "is_transient": true
    }
  ]
}
```

### Rust Side (`deserialize_from_json()`, mod.rs lines 1761-1991)
Parses JSON into `(Vec<IrPass>, Vec<IrResource>)`:
- Resource handles are assigned sequentially (index in array)
- Pass types: Graphics, Compute, Copy, RayTracing
- Color/depth attachments resolved from resource names
- Workgroup size parsed for compute passes

### Rust Side (`execute()`, mod.rs lines 1998-2009)
Accepts passes+resources, compiles via all 6 phases, returns JSON statistics:
```json
{
  "success": true,
  "num_passes": 10,
  "num_resources": 15,
  "num_edges": 22,
  "num_barriers": 31,
  "execution_order": [0, 3, 1, 2, ...]
}
```

## TODO-Specified Bridge (Not Implemented)

### Type Channel (T-FG-7.1)
- `type_register(component_id, component_name, field_layouts, flags)` PyO3 function
- `TypeRegistry` stores ECS component type metadata
- Called from Python `ComponentMeta.__new__()`
- **Status: [-] Not implemented**

### Data Channel (T-FG-7.3, 7.4)
- `component_read(entity_id, component_id)` -> rows by offset
- `component_write(entity_id, component_id, data)` -> writes at offset
- `world_spawn()` / `world_despawn()` / `world_query()` ECS lifecycle
- **Status: [-] Not implemented**

### Command Channel (T-FG-7.5)
- `crossbeam::bounded::<RendererCommand>(16)` SPSC queue
- `renderer_resize(width, height)`, `renderer_screenshot()`, `renderer_recompile_materials()`, `renderer_shutdown()`
- Commands drained at frame start by render thread
- **Status: [-] Not implemented**

## CompiledFrameGraph (mod.rs lines 1496-1513)

### Fields
| Field | Type | Description |
|-------|------|-------------|
| `passes` | `Vec<IrPass>` | All passes (dead removed) |
| `resources` | `Vec<IrResource>` | All resources |
| `edges` | `Vec<IrEdge>` | Dependency edges |
| `order` | `Vec<PassIndex>` | Topological execution order |
| `barriers` | `Vec<(PassIndex, PassIndex, ResourceState, ResourceState)>` | Pipeline barriers |
| `async_passes` | `Vec<(PassIndex, String)>` | Async-eligible passes |
| `eliminated_passes` | `Vec<PassIndex>` | Dead passes removed |

### Compile Pipeline
```
CompiledFrameGraph::compile(passes, resources)
  -> build_dag()           // Phase 2: DAG construction
  -> topological_sort()     // Phase 2b: topological sort
  -> compute_lifetimes()    // Phase 3: resource lifetimes
  -> compute_barriers()     // Phase 4: barrier scheduling
  -> async_schedule()       // Phase 5: async scheduling
  -> eliminate_dead_passes() // Phase 6: dead pass elimination
  -> Ok(CompiledFrameGraph)  // Final assembly
```

## Python CompilationResult (frame_graph.py lines 51-73)

| Field | Type | Description |
|-------|------|-------------|
| `success` | `bool` | Whether compilation succeeded |
| `error_message` | `Optional[str]` | Error message if failed |
| `execution_order` | `list[str]` | Pass names in execution order |
| `culled_passes` | `list[str]` | Names of culled passes |
| `barrier_count` | `int` | Total barriers generated |
| `alias_group_count` | `int` | Resource alias group count |
| `async_pass_count` | `int` | Async-eligible pass count |

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-7.1 | [-] | Not implemented |
| T-FG-7.2 | [~] | JSON bridge exists (serialize/deserialize_from_json/execute) |
| T-FG-7.3 | [-] | Not implemented |
| T-FG-7.4 | [-] | Not implemented |
| T-FG-7.5 | [-] | Not implemented |
| T-FG-7.6 | [x] | mod.rs:1496-1559 |
| T-FG-7.7 | [~] | frame_graph.py:51-73 (no memory_savings_percent or errors) |
| T-FG-7.8 | [-] | Not implemented |
| T-FG-7.9 | [~] | Debug on types, no Display for CompiledFG, no CLI flag |
| T-FG-7.10 | [-] | Not implemented |

## Gaps & Risks

1. **No bridge channels exist** -- All 3 channels (Type, Data, Command) specified in the TODO are absent. Only JSON serialization exists
2. **No PyO3 integration** -- The JSON bridge works but requires serialization/deserialization overhead. No zero-copy Python-Rust interop
3. **No ArcSwap** -- Recompilation requires a full stop-the-world rebuild. No atomic hot-swap for in-flight frames
4. **No serialization** -- `CompiledFrameGraph` derives `Debug` but not `Serialize`. Golden file output not possible
5. **No Display for CompiledFrameGraph** -- Pass order, barrier count, and alias groups cannot be formatted for debug output
6. **CompilationResult is Python-only** -- Not wired to the Rust compiler's output. When Rust compiles via `execute()`, the result stays in Rust
