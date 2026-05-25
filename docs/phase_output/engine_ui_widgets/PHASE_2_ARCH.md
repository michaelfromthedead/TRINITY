# PHASE 2 ARCHITECTURE: Focus Management System

## Problem Statement

Widgets have focus-related state (`_is_focused`) but no system to coordinate focus across the widget tree. There is no tracking of which widget is currently focused, no Tab navigation, no focus trapping for modals, and no focus restoration.

## Architectural Decision

### Focus Coordinator

Single coordinator tracks focus state for the entire widget tree:

```python
class FocusCoordinator:
    _focused_widget: Widget | None
    _focus_chain: list[Widget]
    _trap_stack: list[Widget]  # modal focus traps
    _restore_stack: list[Widget]  # focus to restore on trap pop
```

### Focusability

Widgets declare focusability:
```python
def is_focusable(self) -> bool:
    return self._enabled and self._visible
```

Interactive widgets (Button, TextInput, Slider, etc.) return True by default.
Display widgets (Label, ProgressBar) return False by default.

### Focus Chain

The focus chain is an ordered list of all focusable widgets in tab order:
1. Built by traversing widget tree depth-first
2. Respects `tab_index` property if set
3. Rebuilt when widget tree changes

### Focus Navigation

| Key | Action |
|-----|--------|
| Tab | Focus next in chain |
| Shift+Tab | Focus previous in chain |
| Arrow keys | Navigate within widget groups (radio buttons, etc.) |
| Escape | Close modal / remove focus trap |

### Focus Trapping

Modals push a focus trap:
```python
coordinator.push_focus_trap(modal_container)
# Tab only cycles within modal_container descendants
# Original focus saved to restore_stack

coordinator.pop_focus_trap()
# Focus restored to saved widget
```

### Focus Events

Widgets receive focus events:
```python
def on_focus(self) -> None:
    self._is_focused = True
    self._update_visual_state()
    self._dirty = True

def on_blur(self) -> None:
    self._is_focused = False
    self._update_visual_state()
    self._dirty = True
```

## Component Diagram

```
+------------------+
| FocusCoordinator |
+------------------+
        |
        | manages
        v
+-------+-------+
|  Focus Chain  |  <- ordered focusable widgets
+---------------+
   |   |   |
+--v-+ +-v-+ +-v--+
|Btn | |Inp| |Btn |  <- widgets with focus state
+----+ +---+ +----+
```

## Integration Points

### With Input Router (Phase 3)

Focus determines keyboard event target:
- Focused widget receives keyboard events
- Tab/Shift+Tab intercepted by coordinator

### With Layout (Phase 1)

Focus chain rebuilt when layout changes:
- Layout add/remove triggers focus chain rebuild
- Tab index respects layout order

### With Existing Widgets

Widgets already have `_is_focused` flag. Coordinator calls:
```python
widget.on_focus()
widget.on_blur()
```

## State Machine

```
[No Focus] --click--> [Widget A Focused]
                           |
                         Tab
                           |
                           v
                     [Widget B Focused]
                           |
                      push_trap
                           |
                           v
                     [Trap: Modal Focused]
                           |
                       pop_trap
                           |
                           v
                     [Widget B Focused] (restored)
```

## Dependencies

- Widget `is_focusable()` method
- Widget `on_focus()` / `on_blur()` handlers
- Widget tree traversal (for building focus chain)

## Risks

1. **Focus chain invalidation**: Frequent widget tree changes may cause performance issues. Mitigation: batch changes, rebuild once per frame.

2. **Complex tab order**: Custom tab_index values may conflict. Mitigation: stable sort by (tab_index, tree_order).

3. **Modal stacking**: Nested modals complicate trap stack. Mitigation: explicit push/pop API, no implicit behavior.
