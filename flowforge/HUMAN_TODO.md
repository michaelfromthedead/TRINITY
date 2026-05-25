# FlowForge Human Verification Checklist

**What to test by hand before release**

---

## Quick Start

```bash
cd apps/desktop
bun run dev          # Start dev server
bun run tauri dev    # Start Tauri app (in another terminal)
```

Test file: `apps/desktop/e2e/fixtures/test_file.py` (21 Trinity classes)

---

## 1. App Launches Correctly

- [ ] Window appears within 3 seconds
- [ ] Dark theme loads (no white flash)
- [ ] Canvas visible in center
- [ ] Sidebar icons on left
- [ ] Window title shows "FlowForge"
- [ ] No error toasts on startup
- [ ] Console has no red errors

---

## 2. Open a Python File

- [ ] Press **Ctrl+O**
- [ ] Native file dialog opens
- [ ] Navigate to `e2e/fixtures/test_file.py`
- [ ] Select and open
- [ ] Nodes appear on canvas
- [ ] Window title updates with filename
- [ ] Different colored nodes visible:
  - [ ] Blue = Components
  - [ ] Green = Systems
  - [ ] Purple = Resources
  - [ ] Orange = Events
- [ ] Fields listed inside each node
- [x] Edges connect related nodes

---

## 3. Canvas Navigation

- [ ] **Mouse wheel** zooms in/out
- [ ] **Click + drag** on empty space pans
- [ ] **Ctrl+1** fits all nodes in view
- [ ] **Ctrl+0** resets zoom to 100%
- [ ] Canvas stays responsive, no lag

---

## 4. Node Selection

- [ ] **Click** a node to select it
- [ ] Selected node has highlight/border
- [ ] **Ctrl+A** selects all nodes
- [ ] **Click empty space** deselects all
- [ ] **Delete key** removes selected node
- [ ] Confirmation dialog appears before delete

---

## 5. Node Search (Ctrl+F)

- [ ] **Ctrl+F** opens search panel
- [ ] Input is auto-focused
- [ ] Type "Player" - results filter
- [ ] Matching nodes highlight amber on canvas
- [ ] **Arrow keys** navigate results
- [ ] **Enter** selects and centers on node
- [ ] **Escape** closes search
- [ ] Type filter dropdown works

---

## 6. Type Filter (Sidebar)

- [ ] Click Component toggle - components hide
- [ ] Click again - components reappear
- [ ] Same for System, Resource, Event
- [ ] Hidden nodes don't appear in search

---

## 7. Source Navigation

- [ ] Click a node
- [ ] Source location shows (file:line)
- [ ] Copy button works (check clipboard)
- [ ] **Double-click** node opens external editor
- [ ] Editor opens at correct line (if VS Code/Cursor installed)

---

## 8. Edit a Node

### 8.1 Add Field
- [ ] **Right-click** a Component node
- [ ] Click "Add Field"
- [ ] Dialog opens
- [ ] Enter: name=`health`, type=`int`, default=`100`
- [ ] Click Add
- [ ] New field appears on node

### 8.2 Edit Field
- [ ] **Double-click** the field name you just added
- [ ] Rename to `max_health`
- [ ] Press Enter
- [ ] Name updates on node

### 8.3 Delete Field
- [x] **Right-click** the field
- [x] Click "Delete Field"
- [x] Confirm
- [x] Field removed

### 8.4 Delete Node
- [ ] Select a node
- [ ] Press **Delete**
- [ ] Confirm deletion
- [ ] Node and edges removed

**
RIGHT CLICK AND RENAME NODE makes all nodes disappear
---

## 9. Undo/Redo

- [ ] After deleting node: **Ctrl+Z**
- [ ] Node reappears
- [x] **Ctrl+Shift+Z** redoes the delete
- [x] Node disappears again
THIS DOES NOT WORK CORRECTLY.
CTRL SHIFT Z DESTROYS ENTIRE GRAPH STILL
---

## 10. Save Changes

- [x ] Make an edit (add a field)
- [x ] Title bar shows modified indicator (*)
- [x ] Press **Ctrl+S**
- [x ] Diff preview dialog appears
- [x ] Shows your changes in green
- [x ] Click "Apply"
- [x ] File saved
- [x ] Modified indicator clears
- [x ] Open file in text editor - changes are there

---

## 11. File Conflict Detection

- [x ] Open a file in FlowForge
- [x ] Edit the same file in a text editor externally
- [x ] Save the external edit
- [x ] FlowForge shows conflict dialog
- [x ] "Reload" loads external changes
- [x ] (Or test "Overwrite" to keep FlowForge version)

---

## 12. Error Handling

- [x ] Try to open a non-Python file → error message, no crash
- [x ] Try to open Python with syntax error → error with line number
- [x ] Close the Python sidecar manually (kill process) → app shows error, attempts recovery

---

## 13. Keyboard Shortcuts Work

| Shortcut | Action | ✓ |
|----------|--------|---|
| Ctrl+O | Open file | [ ] |
| Ctrl+S | Save | [ ] |
| Ctrl+Shift+S | Save As | [ ] |
| Ctrl+N | New graph | [ ] |
| Ctrl+F | Search nodes | [ ] |
| Ctrl+A | Select all | [ ] |
| Ctrl+Z | Undo | [ ] |
| Ctrl+Shift+Z | Redo | [ ] |
| Ctrl+1 | Zoom to fit | [ ] |
| Ctrl+0 | Reset view | [ ] |
| Delete | Delete selected | [ ] |
| Escape | Close dialogs | [ ] |

---

## Sign-Off

**Tester:** ______________________

**Date:** ______________________

**Status:**
- [ ] All checks pass - Ready for release
- [ ] Issues found (list below)

**Issues:**
1.
2.
3.
