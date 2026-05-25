# PHASE 4: Command Channel -- Triangle in wgpu

**Scope:** Implement the minimal wgpu renderer skeleton -- Instance, Adapter, Device, Surface, a triangle vertex/fragment shader, a render pipeline, and a winit event loop -- to prove the GPU pipeline end-to-end.
**Depends on:** Phase 0 (renderer-backend crate skeleton with wgpu dep)
**Produces:** renderer.rs (wgpu renderer), window.rs (winit), upload.rs (persistently-mapped ring buffer), crossbeam command channel in bridge.rs
**Status:** NOT STARTED -- No wgpu renderer exists. The frame graph IR (1,681 lines) and GPU-driven buffer staging (777 lines) provide foundation but are not connected to any wgpu runtime.

## 1. Overview

Phase 4 is the largest single gap in GAPSET_3_BRIDGE. Of 27 checkboxes in the original TODO, only 1 is real (the Python window module) and 4 are partial (the buffer staging subsystem in gpu_driven/buffers.rs serves a related but different purpose). A wgpu Instance has never been created, no adapter has been requested, no triangle has been drawn. The renderer-backend crate has wgpu 24 as a dependency (added via the corrected Cargo.toml), and the frame graph IR defines the pass/resource model, but the runtime orchestration layer that connects them does not exist.

## 2. Architectural decisions

- **Frame graph IR as the scheduling layer, not direct wgpu calls**: The intended architecture is that Python submits a compiled frame graph, Rust deserializes it into `IrPass`/`IrResource`/`IrEdge` objects, the DAG scheduler topologically sorts them, barrier insertion resolves resource transitions, and the executor issues wgpu commands. A "triangle" in this model is a single Graphics pass with one color attachment.
- **Crossbeam SPSC for the command channel**: Python writes render commands into a crossbeam channel; Rust's render loop reads and executes them. This decouples Python's frame rate from the GPU's.
- **Buffer staging should use gpu_driven/buffers.rs**: The existing triple-buffered `BufferRegistry` (777 lines, 13 unit tests) with `SlotState` machine (`Idle -> Acquired -> Submitted -> Ready`) covers the staging need. A separate `MappedRingBuffer` (as described in the original TODO) may be redundant.
- **Window management stays in Python for now**: `engine/platform/window/window.py` handles window creation, input events, and surface configuration. A Rust `window.rs` with winit would replace this later.

## 3. Constraints specific to this phase

- wgpu 24 requires a valid `Surface` from a window handle. On Linux, this means an X11 or Wayland connection via `raw-window-handle`.
- WGSL shaders must be compilable by naga (the WGSL validator in dev-dependencies). The renderer-backend crate has naga 24 in dev-dependencies for test-time validation.
- The first triangle should use hardcoded vertex data, not the ECS. Proving the GPU pipeline works is the milestone.
- All GPU-uploaded data must be bytemuck Pod. Omega's Vec/Mat/Quat types satisfy this.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `renderer.rs` | wgpu Instance/Adapter/Device/Surface/Queue setup | DOES NOT EXIST |
| `wgsl/triangle.vert.wgsl` | Vertex shader (full-screen quad or triangle) | DOES NOT EXIST |
| `wgsl/triangle.frag.wgsl` | Fragment shader (solid color) | DOES NOT EXIST |
| `window.rs` | winit event loop, surface creation | DOES NOT EXIST |
| `upload.rs` | Persistently-mapped ring buffer for transforms | DOES NOT EXIST (alternative exists: see below) |
| `gpu_driven/buffers.rs` (777 lines) | BufferRegistry with triple-buffered staging | EXISTS -- covers staging, not mapped ring |
| `bridge.rs` | Crossbeam command channel + PyO3 renderer_resize/screenshot/shutdown | STUB only |
| `engine/platform/window/window.py` | Python window module | EXISTS |
| `frame_graph/mod.rs` | Frame graph IR (pass types, resource handles) | EXISTS -- provides execution model |
| `Cargo.toml` (renderer-backend) | wgpu 24, bytemuck, crossbeam, parking_lot deps | EXISTS -- correct deps now present |

## 5. Testing strategy

- Smoke: `wgpu::Instance::any` succeeds, `Adapter::request_device` returns a device.
- Shader: naga validates the WGSL triangle shaders at compile time (via include_str! in tests).
- Integration: Spawn a winit window, create a wgpu surface, submit a render pass with one triangle, write to a swapchain texture.
- Command channel: Test crossbeam channel round-trip with a `RenderCommand::DrawTriangle` variant.

## 6. Open questions

- Should the first triangle be drawn via the frame graph (requiring the DAG builder to exist first) or via raw wgpu directly (simpler, proves the GPU works)? A direct wgpu approach is recommended for Phase 4, with frame graph integration deferred to Phase 7.
- Does the existing `BufferRegistry` (777 lines, triple-buffered) satisfy the staging requirement, or is a wgpu `MappedRingBuffer` needed? The BufferRegistry uses `wgpu::Buffer` with `COPY_DST | MAP_READ` flags -- this may not support persistently-mapped writes. Benchmark needed.
- Should `window.rs` use `winit` directly or consume the Python window's raw handle via `raw-window-handle`? The Python window module is proven cross-platform; wrapping its handle avoids maintaining a separate windowing stack.

## 7. References

- Phase 5 (Scene Rendering) depends on Phase 4's renderer for mesh drawing.
- Phase 7 (Frame Graph) consumes Phase 4's wgpu device to execute compiled passes.
- Phase 0 provides the crate skeleton with wgpu dependency.
- GAP_3_SUMMARY.md section "Phase 4: Triangle in wgpu" (corrected status: largest gap, 22/27 absent).
