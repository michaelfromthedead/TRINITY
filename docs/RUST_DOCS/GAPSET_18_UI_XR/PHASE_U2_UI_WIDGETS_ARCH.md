# Phase U2: UI Widgets — Architecture

**Tasks:** T-UX-2.1 through T-UX-2.8 (8 tasks)
**Effort:** 24-33 days
**Status:** ✅ COMPLETE (per PROJECT.md verification)

---

## 1. Overview

Phase U2 implements the widget library: containers, primitive widgets (text, image, border), input widgets (button, slider, checkbox), display widgets, and game-specific widgets (health bar, minimap, inventory).

---

## 2. Widget Hierarchy

```
Widget (base)
├── Container
│   ├── HBox, VBox
│   ├── Grid
│   └── ScrollView
├── Primitives
│   ├── Image
│   ├── Text
│   ├── Border
│   └── Spacer
├── Input
│   ├── Button
│   ├── Checkbox
│   ├── Slider
│   ├── TextInput
│   └── Dropdown
├── Display
│   ├── Label
│   ├── ProgressBar
│   └── Icon
└── Game
    ├── HealthBar
    ├── Minimap
    ├── InventorySlot
    ├── DamageNumbers
    └── Tooltip
```

---

## 3. Container System (`framework/container.py`)

Containers manage children layout with configurable direction, gap, and padding.

| Layout Mode | Description |
|-------------|-------------|
| HBox | Horizontal stack |
| VBox | Vertical stack |
| Grid | Row/column grid |
| Absolute | Manual positioning |

`@layout` decorator configures: `direction`, `gap`, `padding`, `alignment`.

---

## 4. Input Widgets

### Button States
```
IDLE → HOVERED → PRESSED → RELEASED → IDLE
```

### Slider
- Draggable thumb
- `RangeDescriptor` for value clamping (min/max)
- Step snapping optional

### TextInput
- Cursor position, selection range
- IME composition support
- Keyboard capture while focused

---

## 5. Game Widgets

| Widget | Key Features |
|--------|--------------|
| HealthBar | Animated damage flash, gradient fill |
| Minimap | Top-down world render, fog of war |
| InventorySlot | Drag-and-drop via `@draggable`/`@droppable` |
| DamageNumbers | Float animation, fade out |
| Tooltip | Hover delay, positioning |

---

## 6. Decorators

| Decorator | Purpose |
|-----------|---------|
| `@draggable` | Enables drag source |
| `@droppable` | Enables drop target |
| `@scrollable` | Enables scroll behavior |
| `@tooltip` | Configures tooltip text/delay |
| `@responsive` | Breakpoint-based styling |

---

## 7. Decorator Stacks (`trinity/decorators/builtin_stacks/ui.py`)

| Stack | Composed Decorators |
|-------|---------------------|
| `interactive_widget` | focusable + draggable + tooltip |
| `data_bound_widget` | tracked + observable + computed |
| `game_hud_element` | ui_layer + responsive |
| `inventory_slot` | draggable + droppable + tooltip |
| `accessible_widget` | focusable + aria_label |

---

## 8. Dependencies

- Phase U1: Widget base, events, coordinate, focus
- Foundation: Tracker, EventLog
