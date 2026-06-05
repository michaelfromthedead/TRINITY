# Phase U5: UI Integration — Architecture

**Tasks:** T-UX-5.1 through T-UX-5.5 (5 tasks)
**Effort:** 8-13 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase U5 wires the UI system to Foundation runtime primitives: Tracker (undo/redo), EventLog (input recording), Mirror (inspection), and descriptor change propagation.

---

## 2. Layout Invalidation Wiring (T-UX-5.1)

### Change Propagation
```
Widget.width.set(new_value)
    ↓
TrackedDescriptor.notify()
    ↓
Parent.layout_dirty = True
    ↓
Layout recalculation (bounded)
    ↓
Children repositioned
```

**Dirty flag propagation:** Only invalidated subtrees recalculate. No infinite loops.

---

## 3. Observable Re-render Wiring (T-UX-5.2)

### Model → Widget → Render
```
PlayerStats.health = 50
    ↓
ObservableDescriptor.notify()
    ↓
HealthBar.value.set(50)
    ↓
TrackedDescriptor.notify()
    ↓
HealthBar.render()
```

---

## 4. Undo/Redo Integration (T-UX-5.3)

Foundation `Tracker` records widget state changes in frame-grouped batches.

| Operation | Tracker Action |
|-----------|----------------|
| Property change | Record old/new value |
| Widget add | Record parent + index |
| Widget remove | Record widget state |
| Layout change | Record position/size |

`Tracker.undo()` restores previous state. `Tracker.redo()` reapplies.

---

## 5. Mirror Inspection (T-UX-5.4)

Foundation `Mirror` enables runtime widget inspection:

```
Widget Tree
├── MainMenu (Screen)
│   ├── Title (Label)
│   │   ├── text: "TRINITY" (TrackedDescriptor)
│   │   ├── font_size: 48 (RangeDescriptor)
│   │   └── binding: None
│   └── PlayButton (Button)
│       ├── state: "idle" (TrackedDescriptor)
│       └── on_click: bound to start_game()
```

Output is human-readable for editor tooling.

---

## 6. EventLog Integration (T-UX-5.5)

All UI input events recorded in Foundation `EventLog`:

| Event | Recorded Data |
|-------|---------------|
| Click | widget_id, position, button, timestamp |
| Key | widget_id, key, modifiers, timestamp |
| Focus | widget_id, previous_id, timestamp |
| Hover | widget_id, enter/leave, timestamp |
| Drag | widget_id, start/move/end, delta, timestamp |

`EventLog.replay()` reproduces input sequence for testing/debugging.

---

## 7. Dependencies

- Phase U1-U4: All UI phases
- Foundation: Tracker, EventLog, Mirror
