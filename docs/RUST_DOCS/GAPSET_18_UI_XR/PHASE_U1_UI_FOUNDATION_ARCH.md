# Phase U1: UI Foundation — Architecture

**Tasks:** T-UX-1.1 through T-UX-1.5 (5 tasks)
**Effort:** 12-16 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase U1 establishes the core UI framework primitives: the base Widget component, event system, coordinate transforms, focus management, and foundational decorators.

---

## 2. Component Architecture

### 2.1 Widget Base (`framework/widget.py`)

```
Widget
├── id: ImmutableDescriptor[str]           # Unique widget identifier
├── parent: TrackedDescriptor[Widget?]     # Parent reference (None for root)
├── children: TrackedDescriptor[list]      # Child widget list
├── local_x/y: TrackedDescriptor[float]    # Local position relative to parent
├── width/height: TrackedDescriptor[float] # Widget dimensions
├── visible: TrackedDescriptor[bool]       # Visibility flag
├── enabled: TrackedDescriptor[bool]       # Interaction enabled flag
└── global_x/y: ComputedDescriptor[float]  # Computed world position
```

**Descriptor chain:** `TrackedDescriptor` for mutable fields enables Foundation Tracker integration for undo/redo and change propagation.

### 2.2 Event System (`framework/events.py`)

| Event Type | Payload | Source |
|------------|---------|--------|
| ClickEvent | position, button, modifiers | Pointer input |
| KeyEvent | key, modifiers, repeat | Keyboard input |
| FocusEvent | gained/lost, previous/next widget | Focus manager |
| HoverEvent | enter/leave, position | Pointer tracking |
| DragEvent | start/move/end, delta, payload | Drag system |

Events use `EventMeta` and integrate with Foundation `EventLog` for recording and replay.

### 2.3 Coordinate System (`framework/coordinate.py`)

- **Local-to-global transform:** Walk parent chain, accumulate offsets
- **Anchor points:** top-left, center, stretch (affects offset calculation)
- **World/screen conversion:** For 3D UI panels (XR Phase X8)

### 2.4 Focus Management (`framework/focus.py`)

- Focus manager tracks single focused widget
- Tab-order navigation via `@focusable` decorator
- Arrow key navigation within containers
- FocusEvent dispatch on focus change

---

## 3. Decorators

| Decorator | Purpose | Protocol |
|-----------|---------|----------|
| `@focusable` | Marks widget as tab-navigable | TAG + REGISTER |
| `@ui_layer` | Assigns widget to render layer | TAG + REGISTER |
| `@anchor` | Configures anchor point | TAG + REGISTER |

---

## 4. Dependencies

- Foundation: Registry, Tracker, EventLog, Mirror
- Trinity: ComponentMeta, TrackedDescriptor, ComputedDescriptor, ImmutableDescriptor, TransientDescriptor

---

## 5. File Inventory

| File | Purpose | Lines |
|------|---------|-------|
| `engine/ui/framework/widget.py` | Base Widget component | ~200 |
| `engine/ui/framework/events.py` | Event types | ~150 |
| `engine/ui/framework/coordinate.py` | Coordinate transforms | ~100 |
| `engine/ui/framework/focus.py` | Focus management | ~150 |
| `trinity/decorators/ui.py` | UI decorators | ~100 |
