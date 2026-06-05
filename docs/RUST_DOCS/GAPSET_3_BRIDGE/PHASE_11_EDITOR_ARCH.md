# PHASE 11: Editor Integration

**Scope:** Embed the Rust renderer in an editor context -- provide a viewport for the wgpu swapchain, gizmo rendering, scene selection, and a REPL connected to the live ECS.
**Depends on:** Phase 4 (wgpu renderer for viewport rendering)
**Produces:** egui-wgpu integration in Rust, REPL-to-live-runtime connection
**Status:** DIVERTED -- The original plan specified egui-wgpu for immediate-mode GUI over the 3D viewport. The actual implementation is a Python-native editor (`engine/tooling/editor/`) with viewport, gizmos, modes, plugins, selection, and preferences. A debug console with REPL exists (`engine/debug/console/`). The ShellLang bridge (`foundation/bridge.py`) provides an alternative integration pathway via `TrinityWorldAdapter`.

## 1. Overview

Phase 11 covers how developers interact with the engine at runtime. Two integration points matter: the editor viewport (what the user sees when running the engine with `--editor`) and the REPL (what the user types to inspect/modify the live world). The original plan's egui-wgpu integration was never started. Instead, the project built a Python-native editor framework with the same capabilities: viewport rendering, gizmo-based manipulation, mode switching, plugin loading, and a command console. The foundation/bridge.py module connects this to the ShellLang system for AI-assisted editing.

## 2. Architectural decisions

- **Python-native editor instead of egui-wgpu**: The editor (`engine/tooling/editor/`) is implemented in Python with modules for the main shell (`app_shell.py`), 3D viewport (`viewport.py`), command dispatch (`commands.py`), gizmo rendering (`gizmos.py`), interaction modes (`modes.py`), plugin loading (`plugins.py`), entity selection (`selection.py`), user preferences (`preferences.py`), and keyboard shortcuts (`shortcuts.py`). This is a complete editor framework, not a stub.
- **ShellLang bridge for AI and REPL integration**: `foundation/bridge.py` provides `TrinityWorldAdapter` that syncs Trinity component instances with ShellLang entities. This enables bidirectional querying and modification from the ShellLang AI interface (`AIInterface`) and REPL (`Shell`).
- **Debug console with ECS integration potential**: `engine/debug/console/` provides console.py (REPL loop), commands.py (command dispatch), cvar.py (console variables), scripting.py (script execution). The integration with the live ECS component store for entity CRUD is documented but not demonstrated.
- **No Rust GUI**: The egui-wgpu path (Rust immediate-mode GUI over the wgpu viewport) was never started. The Python editor handles all UI responsibilities.

## 3. Constraints specific to this phase

- The Python editor runs in the same process as the engine -- no IPC or RPC for editor commands.
- Gizmo rendering requires immediate-mode draw calls. In the current architecture, these would need to be injected into the frame graph as overlay passes.
- REPL commands must be thread-safe: the REPL runs on a separate thread but modifies shared ECS state. The ECS `_lock` (threading.Lock) in ComponentMeta protects type registration but not entity-level mutations.

## 4. Component breakdown

| File/Component | Role | Status |
|----------------|------|--------|
| `egui_integration.rs` | egui-wgpu immediate-mode GUI | DOES NOT EXIST |
| `engine/tooling/editor/app_shell.py` | Editor main shell | EXISTS (Python) |
| `engine/tooling/editor/viewport.py` | 3D viewport rendering | EXISTS (Python) |
| `engine/tooling/editor/commands.py` | Command dispatch | EXISTS (Python) |
| `engine/tooling/editor/gizmos.py` | Transform gizmo rendering | EXISTS (Python) |
| `engine/tooling/editor/modes.py` | Interaction modes (select/translate/rotate/scale) | EXISTS (Python) |
| `engine/tooling/editor/plugins.py` | Plugin loading | EXISTS (Python) |
| `engine/tooling/editor/selection.py` | Entity selection | EXISTS (Python) |
| `engine/tooling/editor/preferences.py` | User preferences | EXISTS (Python) |
| `engine/tooling/editor/shortcuts.py` | Keyboard shortcuts | EXISTS (Python) |
| `engine/debug/console/console.py` | REPL loop | EXISTS |
| `engine/debug/console/commands.py` | Console command registry | EXISTS |
| `engine/debug/console/cvar.py` | Console variables | EXISTS |
| `engine/debug/console/scripting.py` | Script execution | EXISTS |
| `foundation/bridge.py` | TrinityWorldAdapter -- ShellLang sync | EXISTS |
| IPython REPL with component CRUD | Live entity create/destroy/modify | PARTIAL -- console exists but ECS integration not demonstrated |

## 5. Testing strategy

- Unit: Editor command dispatch -- register a command, invoke by name, verify execution.
- Unit: Gizmo hit-testing -- click at screen coordinate, verify correct gizmo axis is selected.
- Integration: REPL loop -- start console, type `spawn(Position(1,2,3), Velocity(0,0,0))`, verify entity appears in ECS World.
- Integration: TrinityWorldAdapter -- create a Trinity component instance, add to ShellLang world, verify bidirectional entity mapping.
- Integration: Editor viewport -- render a scene, verify viewport displays the correct camera view.

## 6. Open questions

- Should egui-wgpu be added for the final production editor, or is the Python editor sufficient? The Python editor is functional but adds frame latency (Python->C FFI for every draw call). An egui-wgpu viewport overlay would be more performant for gizmo rendering.
- The REPL-to-ECS integration is partial. Should the existing `console.py` be extended to import the ECS World and expose entity CRUD, or should a separate "live inspector" be built? Extending the existing console is less work.
- The foundation/bridge.py ShellLang integration is an alternative to the three-channel bridge model. Should it be documented as the primary editor integration path, or as a supplementary tool?

## 7. References

- Phase 4 (wgpu Renderer) provides the viewport surface for editor rendering.
- Phase 2 (Component Store) provides the ECS that the REPL and editor manipulate.
- GAP_3_SUMMARY.md section "Phase 11: Editor Integration" (4 real, 2 partial, 6 absent).
