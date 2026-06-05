# PHASE 2 TODO: Focus Management System

## Prerequisites
- Phase 1 (Layout Engine) complete or layout-independent widget tree
- Widgets have `_is_focused` state property

---

## Task 2.1: Focusability Protocol

**File**: `engine/ui/widgets/base.py`

**Description**: Add focusability methods to widget base.

**Acceptance Criteria**:
- [ ] `is_focusable() -> bool` method returns `_enabled and _visible` by default
- [ ] `tab_index: int | None` property (None = natural order)
- [ ] Interactive widgets (Button, TextInput, etc.) are focusable
- [ ] Display widgets (Label, ProgressBar) are not focusable
- [ ] Disabled or invisible widgets are not focusable

**Evidence of Completion**: `button.is_focusable()` returns True; `label.is_focusable()` returns False.

---

## Task 2.2: Focus Event Handlers

**File**: `engine/ui/widgets/base.py`

**Description**: Add focus/blur event handlers to widget base.

**Acceptance Criteria**:
- [ ] `on_focus()` sets `_is_focused = True`, calls `_update_visual_state()`, sets `_dirty = True`
- [ ] `on_blur()` sets `_is_focused = False`, calls `_update_visual_state()`, sets `_dirty = True`
- [ ] Widgets can override for custom focus behavior
- [ ] Focus callbacks can be subscribed/unsubscribed

**Evidence of Completion**: `button.on_focus()` triggers visual state update and dirty flag.

---

## Task 2.3: Focus Coordinator Core

**File**: `engine/ui/focus/coordinator.py`

**Description**: Create focus coordinator class.

**Acceptance Criteria**:
- [ ] `FocusCoordinator` class with `_focused_widget: Widget | None`
- [ ] `focus(widget)` calls `on_blur()` on current, `on_focus()` on new
- [ ] `blur()` removes focus from current widget
- [ ] `get_focused() -> Widget | None` returns current focused widget
- [ ] Single coordinator instance per widget tree

**Evidence of Completion**: `coordinator.focus(button)` triggers `button.on_focus()`.

---

## Task 2.4: Focus Chain Builder

**File**: `engine/ui/focus/chain.py`

**Description**: Build ordered list of focusable widgets.

**Acceptance Criteria**:
- [ ] `build_focus_chain(root) -> list[Widget]` traverses widget tree depth-first
- [ ] Only includes widgets where `is_focusable()` is True
- [ ] Sorts by `tab_index` (None last, then stable by tree order)
- [ ] Returns empty list if no focusable widgets

**Evidence of Completion**: Tree with 5 widgets, 3 focusable, returns ordered list of 3.

---

## Task 2.5: Tab Navigation

**File**: `engine/ui/focus/coordinator.py`

**Description**: Implement Tab and Shift+Tab navigation.

**Acceptance Criteria**:
- [ ] `focus_next()` moves focus to next widget in chain
- [ ] `focus_previous()` moves focus to previous widget in chain
- [ ] Wraps around at ends of chain
- [ ] No-op if chain is empty
- [ ] Skips widgets that become non-focusable

**Evidence of Completion**: Tab from first widget to second to third and wraps to first.

---

## Task 2.6: Focus Trapping

**File**: `engine/ui/focus/coordinator.py`

**Description**: Implement modal focus trapping.

**Acceptance Criteria**:
- [ ] `push_focus_trap(container)` restricts Tab to container descendants
- [ ] Saves current focused widget to restore stack
- [ ] Automatically focuses first focusable in trap
- [ ] `pop_focus_trap()` removes trap, restores saved focus
- [ ] Nested traps supported (stack-based)

**Evidence of Completion**: Modal push restricts Tab; pop restores original focus.

---

## Task 2.7: Programmatic Focus

**File**: `engine/ui/focus/coordinator.py`

**Description**: API for programmatic focus control.

**Acceptance Criteria**:
- [ ] `focus_first()` focuses first widget in chain
- [ ] `focus_last()` focuses last widget in chain
- [ ] `focus_by_id(widget_id)` focuses widget by ID if focusable
- [ ] Returns success boolean
- [ ] Respects focus traps (only focuses within trap if active)

**Evidence of Completion**: `coordinator.focus_first()` focuses first button in dialog.

---

## Task 2.8: Focus Chain Invalidation

**File**: `engine/ui/focus/coordinator.py`

**Description**: Rebuild focus chain when widget tree changes.

**Acceptance Criteria**:
- [ ] `invalidate_focus_chain()` marks chain for rebuild
- [ ] Chain rebuilt lazily on next navigation
- [ ] Widget add/remove triggers invalidation
- [ ] Widget enable/disable triggers invalidation
- [ ] Batched invalidation (one rebuild per frame max)

**Evidence of Completion**: Adding widget, then Tab, uses updated chain.

---

## Task 2.9: Focus Module Exports

**File**: `engine/ui/focus/__init__.py`

**Description**: Export focus classes for public API.

**Acceptance Criteria**:
- [ ] Exports: FocusCoordinator
- [ ] Exports: build_focus_chain (if needed externally)
- [ ] No internal implementation details exposed

**Evidence of Completion**: `from engine.ui.focus import FocusCoordinator` works.

---

## Summary

| Task | Effort | Priority |
|------|--------|----------|
| 2.1 Focusability Protocol | Small | P0 |
| 2.2 Focus Event Handlers | Small | P0 |
| 2.3 Coordinator Core | Medium | P0 |
| 2.4 Focus Chain Builder | Medium | P0 |
| 2.5 Tab Navigation | Medium | P0 |
| 2.6 Focus Trapping | Medium | P1 |
| 2.7 Programmatic Focus | Small | P1 |
| 2.8 Chain Invalidation | Medium | P1 |
| 2.9 Module Exports | Small | P0 |

**Total Tasks**: 9
**Critical Path**: 2.1 -> 2.2 -> 2.3 -> 2.4 -> 2.5 -> 2.9
