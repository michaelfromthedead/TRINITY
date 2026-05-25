# GAPSET_18_UI_XR — UI Framework + XR Support

**Owner:** Michael
**Status:** MOSTLY IMPLEMENTED (~80-85% complete in Python)
**RDC run:** 2026-05-22

## 1. Goal

Build the UI framework (widget system, layout, styling, animation, accessibility) and XR support (OpenXR, WebXR, spatial interaction, avatars) for the TRINITY engine.

## 2. Why

Every engine needs a UI system for editor tooling, game HUDs, menus, and debug overlays. XR support enables VR/AR applications with spatial interaction and immersive rendering.

## 3. Hardware / Environment constraints

- UI: CPU rendering (Python), optional GPU acceleration (WGSL planned)
- XR: OpenXR runtime (Khronos standard), WebXR fallback
- Cross-platform: Windows/Linux (desktop VR), Web (WebXR)

## 4. Non-goals

- Not a replacement for OS-native UI (file dialogs, system menus)
- Not a WYSIWYG UI editor (that's GAP 13 Tooling)
- Not a full game engine UI framework (no UMG/WPF-level visual designer)

## 5. Phase overview

| Phase | Name | Status |
|-------|------|--------|
| U1 | UI Foundation | ✅ COMPLETE — Widget, Events, Coordinate, Focus, decorators |
| U2 | UI Widgets | ✅ COMPLETE — Container, primitives (Text/Image/Border/Spacer), display, game, input widgets |
| U3 | UI Layout + Styling | ✅ COMPLETE — Layout managers, theme/style system, data binding |
| U4 | UI Animation + Screens | ✅ COMPLETE — UI animation, screen management, accessibility |
| X1 | XR Foundation | ✅ COMPLETE — OpenXR, WebXR, Session, Capabilities |
| X2 | XR Interaction | ✅ COMPLETE — Direct, Gaze, Ray interactors, locomotion |
| X3 | XR Rendering + Avatars | ✅ COMPLETE — Stereo rendering, avatars, spatial UI |
| X4 | GPU Backend | [ ] NOT STARTED — WGSL UI rendering, SDF fonts |

## 6. Key reference documents

- [GAP_18_SUMMARY.md](GAP_18_SUMMARY.md) — Source-code verification
- [PHASE_N_TODO.md](PHASE_N_TODO.md) — Task list (68 tasks)
- [CLARIFICATION.md](CLARIFICATION.md) — Architectural philosophy
