# CLARIFICATION — GAPSET_18_UI_XR

**Purpose:** Architectural philosophy and divergence analysis.

---

## 1. The TODO vs Reality divergence

The PHASE_N_TODO.md was written as a greenfield plan: 68 tasks, all `[ ]`, ~31-45 weeks estimated. The codebase tells a different story: ~79,000 lines of Python across 131 files implementing nearly everything the TODO describes. The divergence occurred because the UI/XR systems were built Python-first (like the rest of TRINITY), while the TODO assumed a Rust/WGSL implementation path.

## 2. Architecture: Python framework, optional GPU backend

The UI system uses a standard retained-mode widget hierarchy:
```
Widget (base) → Container → Layout → Styling → Rendering
```
All framework code is Python. GPU rendering (WGSL shaders for text, SDF fonts, batch rendering) is a planned optimization, not a prerequisite.

## 3. XR: OpenXR-first, WebXR fallback

The XR runtime supports both native OpenXR (via the Khronos loader) and WebXR (for browser-based experiences). The runtime abstraction layer (`xr_runtime.py`, 720 lines) provides a unified API.

## 4. Foundation integration status

The UI/XR codebase uses its own decorator system where Foundation decorators were planned. `@focusable`, `@ui_layer`, `@anchor` exist as concepts but may not be wired to Foundation's Registry/Tracker. This is the same pattern seen across other gapsets (AUDIO, NETWORKING, PHYSICS, etc.).

## 5. Remaining work

- Foundation decorator wiring (cross-cutting, shared across all gapsets)
- WGSL GPU rendering backend (SDF fonts, batch rendering)
- Integration tests
