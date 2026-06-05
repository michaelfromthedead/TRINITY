# Phase 4: Barrier Insertion -- Architecture

## Overview
Automatic GPU pipeline barrier insertion between passes. Tracks resource states and generates transition barriers when a pass requires a resource in a different state than it was left in by the previous pass. Covers S1-G6 (Automatic Barrier Insertion).

## Key Type: Resource State (`mod.rs` lines 971-1001)

13 states tracking the GPU pipeline location of each resource:

| State | When Used |
|-------|-----------|
| `Uninitialized` | Before first write (transients only) |
| `VertexBuffer` | Vertex input stage |
| `IndexBuffer` | Index input stage |
| `IndirectArgument` | Indirect draw/dispatch buffer |
| `ColorAttachment` | Render target output |
| `DepthStencilAttachment` | Depth/stencil test (writeable) |
| `DepthStencilReadOnly` | Depth/stencil test (read-only) |
| `ShaderRead` | Sampled image / uniform texel buffer |
| `ShaderReadWrite` | Storage image / storage buffer (UAV) |
| `TransferSrc` | Copy source |
| `TransferDst` | Copy destination |
| `AccelerationStructure` | Ray tracing BLAS/TLAS |
| `Present` | Swap chain presentation |

## Algorithm: Barrier Computation (`compute_barriers`, mod.rs lines 1425-1475)

### Input
- `order: &[PassIndex]` -- topological execution order
- `passes: &[IrPass]` -- passes with access sets
- `edges: &[IrEdge]` -- dependency edges

### State Mapping Functions

**`state_required_by_pass(pass, resource)`** maps pass access patterns to required states:
| Access Pattern | Required State |
|----------------|----------------|
| Color attachment write | `ColorAttachment` |
| Depth-stencil write | `DepthStencilAttachment` |
| Depth-stencil read-only | `DepthStencilReadOnly` |
| Shader read | `ShaderRead` |
| Shader write (ReadWrite) | `ShaderReadWrite` |
| Texture sampled (view_type) | `ShaderRead` |
| Storage buffer | `ShaderReadWrite` |

**`state_left_by_pass(pass, resource)`** maps pass completion states:
| Access Pattern | Left State |
|----------------|------------|
| Wrote color attachment | `ColorAttachment` |
| Wrote depth-stencil | `DepthStencilAttachment` |
| Read only | `ShaderRead` (preserved) |
| ReadWrite | `ShaderReadWrite` |

### Algorithm
1. Build `pass_map` and `ordered_set` from execution order
2. For each edge where both endpoints are in the execution order:
   - Compute `before = state_left_by_pass(from_pass, resource)`
   - Compute `after = state_required_by_pass(to_pass, resource)`
   - If `before != after`, emit barrier `(from, to, before, after)`
3. Deduplicate via `HashSet<(PassIndex, PassIndex, ResourceHandle)>`

### Python Implementation (`barrier_manager.py`)
- `ResourceStateTracker` maintains `_current_states: dict[str, ResourceState]`
- `analyze_passes()` generates per-pass `BarrierBatch` with pipeline stage mapping
- `_needs_barrier()` skips same-state and `Uninitialized->X` transitions
- `_check_uav_hazards()` for UAV-to-UAV barriers
- `_create_transition_barrier()` maps states to pipeline stages:
  - `_STATE_TO_STAGE` mapping (14 pipeline stages, 18 access flags)
  - `_TRANSITION_MAP` (valid state transition pairs)

## Pipeline Stage Mapping (Python)

| TRINITY State | PipelineStage | AccessFlags |
|---------------|---------------|-------------|
| UNDEFINED | TOP_OF_PIPE | (empty) |
| RENDER_TARGET | COLOR_ATTACHMENT_OUTPUT | COLOR_ATTACHMENT_WRITE |
| DEPTH_WRITE | EARLY_FRAGMENT_TEST | DEPTH_STENCIL_ATTACHMENT_WRITE |
| SHADER_RESOURCE | FRAGMENT_SHADER | TEXTURE_SHADER_READ |
| UNORDERED_ACCESS | COMPUTE_SHADER | TEXTURE_STORAGE_READ_WRITE |
| COPY_SOURCE | TRANSFER | TRANSFER_READ |
| COPY_DEST | TRANSFER | TRANSFER_WRITE |
| PRESENT | BOTTOM_OF_PIPE | (empty) |

## Implementation Status

| Task | Status | Location |
|------|--------|----------|
| T-FG-4.1 | [x] | Python barrier_manager.py, Rust mod.rs:1425-1475 |
| T-FG-4.2 | [x] | Rust compute_barriers(), Python analyze_passes() |
| T-FG-4.3 | [x] | Python BarrierBatch, Rust deduplication |
| T-FG-4.4 | [~] | Python skips same-state only; no A->B->A elimination |
| T-FG-4.5 | [-] | Not implemented (wgpu command generation) |
| T-FG-4.6 | [~] | Python pre-barriers only; no post-barriers |
| T-FG-4.7 | [~] | 2 Rust barrier tests; no Python barrier tests |
| T-FG-4.8 | [-] | Not implemented |

## Gaps & Risks

1. **No wgpu command generation** -- This is the critical gap. Barrier records exist but cannot drive actual `wgpu::CommandEncoder` calls. The Python `_execute_barriers()` and `_prepare_for_present()` are pass-through stubs
2. **No redundant barrier elimination** -- A->B->A and B->A->B transitions are not detected or removed
3. **No per-pass post-barriers** -- Barriers are only emitted before passes, not after. Some hardware requires post-pass barriers for cache flushing
4. **Pipeline stage mapping is Python-only** -- The Rust compiler has no stage mapping; barriers are bare `(from, to, before, after)` tuples
5. **State coverage incomplete** -- Only common state transitions are mapped; edge cases (e.g., Present <-> any non-Transfer) may produce incorrect barriers
