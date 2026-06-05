# CLARIFICATION.md — GAPSET_13_TOOLING Clarification Requests

## Overview

The RDC investigation of GAPSET_13_TOOLING revealed a three-tier architecture (Python tools, Rust bridge, Engine integration) where Tier 1 and Tier 2 are largely complete but Tier 3 is unimplemented. Several architectural decisions remain unclear and need resolution before proceeding with implementation.

---

## C-1: Python Tooling <> Rust Editor Boundary

**Question:** What is the intended relationship between the existing Python tooling layer and the Rust/egui editor panels?

**Context:** The investigation found that `engine/tooling/` contains ~130 Python files across 20 subsystems implementing full editor functionality (hierarchy, inspector, debug, profiling, etc.) with their own UI rendering. The TODO assumes these need to be re-implemented in Rust/egui.

**Options:**
- (A) **Replace**: Rust/egui panels replace Python UI entirely. Python tools become backend-only data providers.
- (B) **Wrap**: Rust/egui panels embed Python-rendered content via the bridge protocol (texture sharing).
- (C) **Hybrid**: Core panels (Inspector, Hierarchy) in Rust/egui; complex tools (Material Editor, Visual Scripting) remain Python with Rust integration points.

**Recommendation:** Option (A) for core panels, (C) for complex tools. The Python tools already have their own UI framework; rewriting them in egui would duplicate thousands of lines. A hybrid approach where simple panels use egui natively and complex tools bridge to Python-rendered content would be faster.

---

## C-2: Frontend Strategy — Tauri WebView vs egui Native

**Question:** Should the editor UI be rendered via egui (native Rust immediate-mode GUI) or via the Tauri WebView (TypeScript frontend)?

**Context:** The current architecture has a Tauri desktop shell with WebView capabilities (plugins: dialog, fs, shell) and a Rust bridge protocol. The TODO assumes egui for all panels. However, the Tauri WebView could render a TypeScript/React frontend that communicates via the existing bridge protocol.

**Options:**
- (A) **egui native**: All panels rendered as egui widgets in the Rust process. Tight wgpu integration, minimal IPC overhead.
- (B) **Tauri WebView**: Panels rendered in TypeScript/React in the WebView. Uses existing bridge protocol. Slower but richer UI components.
- (C) **Mixed**: Performance-critical panels (viewport, profiler) in egui; rich panels (material editor, node graph) in WebView.

**Recommendation:** Option (C). The viewport MUST be egui (wgpu surface sharing, minimal latency). UI-heavy tools can use the WebView with the existing bridge protocol. This avoids duplicating the TypeScript frontend work while getting native GPU performance where it matters.

---

## C-3: EguiUIContext Protocol Design

**Question:** What is the exact design of the Python `UIContext` protocol that bridges Python tool calls to egui?

**Context:** T-TL-1.3 requires an `EguiUIContext` adapter that maps Python UI calls to `egui::Ui`. The Python tools in `engine/tooling/` use their own UI abstractions. The bridge.rs stub mentions PyO3 integration but is empty.

**Open questions:**
- What Python types/classes constitute the `UIContext` protocol?
- Is this a PyO3 bridge (Rust functions callable from Python), a JSON-RPC bridge (Python sends UI commands over the bridge protocol), or a texture-sharing bridge (Python renders to a texture, Rust composites)?
- What is the minimum viable set of Python UI calls that need to be mapped?

**Recommendation:** A JSON-RPC approach is simplest: the Python tool process sends UI commands (e.g., `{method: "ui.label", params: {text: "Hello"}}`) over the existing sidecar protocol, and the Rust process renders them via egui. This avoids PyO3 complexity and uses the already-working sidecar infrastructure.

---

## C-4: Bridge Protocol Ownership

**Question:** Is the bridge protocol in `bridge_protocol.rs` the canonical schema, or was it auto-generated from the TODO requirements?

**Context:** The bridge protocol (2188 lines, 22 endpoints, ~1300 lines of tests) is impressively complete. However, there are no corresponding TypeScript type exports at `flowforge/packages/core/bridge/types.ts` or Python mock implementations at `tests/integration/_omega_mock.py` as documented in the module header. It is unclear if the protocol was designed against real tooling requirements or theoretical ones.

**Clarification needed:** Have the 22 endpoints been validated against actual Python tooling needs? Specifically:
- `type.*` endpoints map to `engine/tooling/` type systems — which subsystems actually use these?
- `command.*` endpoints (spawn, despawn, query) — do these map to the ComponentStore API in `crates/renderer-backend/src/component_store.rs`?
- `data.*` endpoints — are these consumed by Python tools via the sidecar, or by the TypeScript frontend?

**Recommendation:** Map each endpoint to a real Python tool subsystem to validate completeness. Likely gaps: no material DSL compilation endpoint, no debug visualization switch endpoint, no profiler data streaming endpoint.

---

## C-5: PyO3 vs Sidecar Architecture

**Question:** What is the long-term Python-Rust integration strategy? There are two competing mechanisms:

- **PyO3 bridge** (`crates/renderer-backend/src/bridge.rs`): Stub only. Would embed Python in the Rust process or Rust in Python, enabling direct function calls without serialization overhead. Better for hot paths (per-frame component reads).
- **JSON-RPC sidecar** (`flowforge/src-tauri/src/sidecar/mod.rs`): Fully implemented. Python runs as a child process; communication over stdin/stdout JSON-RPC 2.0. Works today but has serialization overhead.

**Context:** The current codebase has both mechanisms: a working sidecar and a stub PyO3 bridge. The bridge.rs TODO mentions `type_register`, `component_read`, `component_write`, `component_delete` — which overlap with the sidecar's `type.*` and `data.*` endpoints.

**Options:**
- (A) **Sidecar only**: Drop PyO3. All Python-Rust communication goes through JSON-RPC. Simpler, already works.
- (B) **PyO3 only**: Replace sidecar with embedded Python. Better performance but adds PyO3 dependency complexity.
- (C) **Both**: Sidecar for tooling/editor operations (low frequency), PyO3 for hot-path ECS data access (per-frame).

**Recommendation:** Option (C). The sidecar is proven for tooling operations. PyO3 should be used only for hot-path component data access where JSON-RPC overhead is prohibitive. However, validate whether the hot path actually needs PyO3 — if batch_read/write over JSON-RPC meets performance targets (1M reads in <100ms per the protocol spec), PyO3 may be unnecessary.

---

## C-6: GPU Pipeline Dependency (S1-S9, S10+, S15)

**Question:** What is the actual state of the GPU pipeline dependencies (S1-S9, S10+, S15) that Phase 3, Phase 6, and Phase 8 are blocked on?

**Context:** The TODO marks Phases 3, 6, and 8 as blocked on external gap sets (S1-S9 GPU Pipeline, S10+ Frame Graph, S15 Core ECS). However, `crates/renderer-backend/src/` contains modules for `gpu_driven`, `frame_graph`, `renderer`, `ddgi`, `particles`, `post_process`, etc. The renderer may be further along than the TODO assumes.

**Clarification needed:** Can we determine the real status of S1-S9, S10+, and S15 to unblock Phase 3 timeline estimation? Specifically:
- Does `gpu_driven` module provide the wgpu pipeline needed for viewport rendering?
- Does `frame_graph` module provide the frame graph IR needed for debug passes?
- What is missing that prevents wgpu-egui integration?

---

## C-7: FlowForge vs egui Editor

**Question:** What is the relationship between the FlowForge application (Tauri desktop) and the egui-based editor panels?

**Context:** FlowForge is described as "a domain-agnostic visual programming environment" (main.rs doc comment). The Tauri application registers commands for workflow execution, node definitions, file I/O, code generation, and Trinity introspection. The TODO describes editor panels (Inspector, Hierarchy, Material Editor, etc.) as egui widgets in the renderer-backend crate.

**Options:**
- (A) **FlowForge is the editor shell**: FlowForge Tauri app hosts the egui viewport and panels as native Rust widgets alongside the WebView.
- (B) **FlowForge is separate**: FlowForge is a node-based visual scripting tool; the editor panels are a separate application.
- (C) **FlowForge hosts everything**: All tooling panels are part of FlowForge, either as egui widgets or WebView content.

**Recommendation:** Need architectural guidance. Option (C) makes the most sense architecturally — one application with a unified bridge protocol serving both TypeScript WebView panels and Rust egui panels.

---

## C-8: Testing Strategy for Rust/egui Integration

**Question:** What is the testing strategy for the Rust/egui integration layer?

**Context:** The Python tooling layer has comprehensive tests (~150 test files across 20 subsystems). The Rust bridge protocol has extensive serialization tests (~1300 lines). The Rust Editor struct has unit tests. However, egui panels are notoriously difficult to test because they require a running GPU context.

**Open questions:**
- Should egui panels have snapshot tests (render to image, compare)?
- Should panel logic be separated from rendering (Model-View pattern) for unit testability?
- Is there a plan for integration tests that exercise the full Python-sidecar-bridge-egui pipeline?

**Recommendation:** Separate panel business logic (Model) from egui rendering (View) to enable unit testing of panel behavior without GPU context. Use egui's `Id::new("test")` and `Ui::__test` for basic rendering tests. Full integration tests can use the sidecar test infrastructure already in `sidecar_tests.rs` and `blackbox_bridge_contract.rs`.

---

## Summary of Pending Decisions

| ID | Decision Needed | Impact | Recommended |
|----|----------------|--------|-------------|
| C-1 | Python tooling <> Rust boundary | Architecture of ALL phases | Hybrid: core panels in Rust, complex tools bridged |
| C-2 | egui vs Tauri WebView | UI rendering strategy | Mixed: viewport in egui, rich UI in WebView |
| C-3 | EguiUIContext protocol design | T-TL-1.3 implementation | JSON-RPC bridge to Python |
| C-4 | Bridge protocol completeness | All bridge-dependent tasks | Map endpoints to real subsystems |
| C-5 | PyO3 vs sidecar architecture | Long-term perf strategy | Both: sidecar for tooling, PyO3 for hot path |
| C-6 | GPU pipeline (S1-S9) status | Phase 3, 6, 8 timeline | Investigate renderer-backend modules |
| C-7 | FlowForge <> egui relationship | Application architecture | FlowForge as unified editor shell |
| C-8 | Testing strategy for egui | QA approach | Model-View separation for testability |
