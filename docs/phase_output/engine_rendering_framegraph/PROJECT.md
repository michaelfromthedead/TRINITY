# PROJECT: engine/rendering/framegraph

## Scope

Complete the frame graph rendering subsystem by implementing the RHI execution layer. The Python frame graph provides production-quality scheduling, aliasing, barrier management, and async compute algorithms. The remaining work is connecting these to concrete GPU backends (wgpu/Vulkan/D3D12).

## Current State

- **Lines**: 3,524 (3,312 in main files + 61 in config.py + ~150 in __init__.py)
- **Classification**: REAL (with execution stubbed at RHI boundary)
- **Completion**: ~70% — all algorithms implemented, execution stubbed

### Files

| File | Lines | Status |
|------|-------|--------|
| frame_graph.py | 879 | REAL — core orchestrator |
| pass_node.py | 726 | REAL — pass type hierarchy |
| resource_manager.py | 654 | REAL — aliasing, lifetimes |
| barrier_manager.py | 574 | REAL — state machine barriers |
| async_scheduler.py | 479 | REAL — multi-queue scheduling |
| config.py | 61 | REAL — dataclass configs |

## Goals

1. **Define concrete RHI context interface** replacing `context: Any`
2. **Implement GPU resource allocation** for transient/history/external resources
3. **Wire barrier execution** to wgpu/Vulkan backend
4. **Enable async compute execution** with real fence synchronization
5. **Validate Rust bridge serialization** against `IrPass`/`IrResource` types

## Constraints

- Python 3.13 compatibility required (engine embeds statically-linked 3.13)
- PyO3 bridge must serialize to JSON matching Rust IR types
- No GPU vendor lock-in — backend must abstract wgpu/Vulkan/D3D12
- Maintain existing method chaining API for pass construction
- Keep all configuration via dataclasses (config.py pattern)

## Acceptance Criteria

1. Frame graph `execute()` calls concrete RHI methods instead of logging
2. `compute_aliasing()` results drive actual GPU memory allocation
3. Barrier batches execute via `context.execute_barriers()` or equivalent
4. Cross-queue sync points translate to real GPU fence wait/signal
5. All existing unit tests pass (scheduling, aliasing, barrier algorithms)
6. Serialization round-trips through Rust IR without data loss

## Dependencies

- engine/rendering/rhi — concrete backend implementation (wgpu bindings)
- crates/renderer-backend — Rust-side `IrPass`/`IrResource` types
- PyO3 — Python-Rust FFI for serialization path

## Architecture Reference

Matches modern GPU rendering patterns:
- Frostbite Frame Graph (GDC 2017)
- Unreal Engine 4 RDG (Render Dependency Graph)
- Unity HDRP (High Definition Render Pipeline)
