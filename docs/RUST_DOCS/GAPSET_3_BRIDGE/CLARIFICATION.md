# CLARIFICATION -- GAPSET_3_BRIDGE

**Purpose:** Conceptual framing, decision rationales, and pedagogical context for the bridge architecture.

**Relationship to other docs:**
- `PROJECT.md` -- the what (project overview and scope)
- `PHASE_*_ARCH.md` -- the how per phase (11 architecture documents)
- `PHASE_N_TODO.md` -- the do (39 tasks across 11 phases, corrected status)
- `GAP_3_SUMMARY.md` -- the was (deep codebase investigation results)
- This document -- the "why it looks this way"

---

## 1. Why Python/Rust split

Python drives ergonomics and iteration speed. The engine's editor tooling, asset pipelines, ECS logic, and configuration are all in Python -- 1,985 files of it. Rust drives performance, safety, and GPU access. The bridge architecture allows gradual migration: Python code continues working at every step while Rust components are activated feature by feature behind PyO3 gates. An editor scripter can set `transform.position.x = 5` without knowing whether that field resolves to `__dict__`, a Rust SoA column, or a GPU buffer -- the descriptor chain abstracts the storage backend.

The split is not arbitrary. Python owns everything that changes rapidly or benefits from introspection (editor UI, material graphs, REPL, debug overlays, asset import pipelines). Rust owns everything that needs GPU access or deterministic performance (ECS storage for networked state, math for physics determinism, wgpu for rendering).

---

## 2. Why omega is a separate crate

The `omega` crate (`/home/user/dev/USER/PROJECTS_VOID/TRINITY/omega/`) is a standalone deterministic math library with zero GPU dependencies. It was designed as a separate compilation unit for three reasons:

1. **CI without a GPU**: Omega compiles and tests on any machine -- no wgpu, no Vulkan drivers, no GPU needed. This means math tests run in CI on every commit.
2. **Shared across crates**: Other TRINITY crates (renderer-backend, and potentially physics or networking crates) all need the same math types. A shared crate prevents type duplication.
3. **Separation of concerns**: Math types (Vec3, Mat4, Quat, Fixed32) are a different concern from GPU management (wgpu, buffers, pipelines). Mixing them in one crate would violate the single-responsibility principle.

The bytemuck `Pod`/`Zeroable` derive is the bridge between them: omega types can be cast to byte slices for wgpu buffer upload without omega knowing anything about GPUs.

---

## 3. Why wgpu over Vulkan

wgpu is the Rust-native GPU abstraction that maps to Vulkan (Linux/Windows), Metal (macOS/iOS), DX12 (Windows), and WebGPU (Web). It was chosen over raw Vulkan because:

1. **Rust-native API** -- wgpu's builder pattern and type system integrate naturally with Rust's ownership model. Raw Vulkan requires manual handle management and unsafe code.
2. **Cross-platform** -- one API surface covers all target backends. No platform-specific rendering paths.
3. **Matches TRINITY's Rust stack** -- the engine already uses Rust for safety-critical components. wgpu extends this to GPU code without introducing C++ interop.
4. **WGSL shader language** -- wgpu's native shading language is WGSL, which compiles to SPIR-V (Vulkan), MSL (Metal), and DXIL (DX12). Shaders are written once and run on any backend.

The trade-off is that wgpu abstracts away some GPU features (async compute queues on some hardware, vendor-specific extensions). For TRINITY's use cases (forward+ rendering, GPU particles, DDGI), wgpu's feature set is sufficient.

---

## 4. Why the implementation diverged from the original plan

The original PHASE_N_TODO.md described a PyO3-first approach: build Rust components, then wire Python to them. The actual implementation took a Python-native path, building all engine subsystems in Python first, then constructing Rust components independently where they provided clear value (omega math lib, frame graph IR, GPU-driven buffer staging). The PyO3 bridge was never built.

This divergence is valid and architecturally sound. The descriptor chain model (see section 5) supports gradual activation -- adding PyO3 bindings to omega would instantly activate the Rust storage path across the entire engine. The Python implementations are not waste; they are the reference design and fallback runtime.

The divergence was likely pragmatic: building a working engine in Python first allowed rapid iteration on the game-side API (ECS, rendering pipeline, editor) while the Rust components incubated separately. The Rust work that exists (omega math, frame graph IR, GPU-driven buffers) is high-quality and structurally important.

---

## 5. The descriptor chain model

ComponentMeta installs a stack of descriptors per field, each wrapping the next:

```
[ValidationDescriptor] -> [ConversionDescriptor] -> [SerializationDescriptor] -> [StorageDescriptor/RustStorageDescriptor]
                                                           [innermost: RustStorageDescriptor when _omega available]
```

`RustStorageDescriptor` (`trinity/descriptors/rust_storage.py`) sits at the innermost position. When `_omega` is importable, field reads/writes route through `component_read(entity_id, component_id, offset, field_type)` which fetches data from the Rust SoA ComponentStore. When `_omega` is absent (current state), it falls back to `__dict__` via `_dict_get`/`_dict_set`.

This is the activation model: PyO3 bindings are the single gate. Flip the compile flag, rebuild omega with `pyo3` feature, and every component field access in the engine routes through Rust storage. No Python code changes needed. The architecture is ready; only the bindings are missing.

---

## 6. What "bridge" means in this context

The bridge is three logical channels, not a single API:

| Channel | Direction | Protocol | Status |
|---------|-----------|----------|--------|
| **Type Channel** | Python -> Rust | Component schemas (field names, types, offsets) via `type_register()` | ✅ LIVE — 14 PyO3 functions, `_omega.so` ABI3 |
| **Data Channel** | Bidirectional | Entity component data via `component_read/write/delete()` | ✅ LIVE — SoA ComponentStore, World dual-writes |
| **Command Channel** | Python -> Rust -> GPU | Frame graph execution via `frame_graph_execute()` | ✅ LIVE — 6-phase compiler, 8 WGSL shaders, wgpu renderer |

All three channels are now LIVE. `_omega.so` provides 14 PyO3 functions covering type registration, component read/write/delete, renderer lifecycle, material compilation, frame graph execution, and editor entity listing. The descriptor chain activates Rust storage transparently when `_omega` is importable.

A fourth, supplementary channel exists via `foundation/bridge.py`:

| Channel | Direction | Protocol | Status |
|---------|-----------|----------|--------|
| **ShellLang Bridge** | Python -> ShellLang | `TrinityWorldAdapter` syncs instances to ShellLang entities | Working |

This is a separate bridge for AI/REPL integration, not part of the three-channel GPU bridge model.
