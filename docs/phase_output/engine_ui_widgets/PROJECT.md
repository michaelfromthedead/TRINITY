# PROJECT: engine/ui/widgets

## Overview

The UI widgets module is a production-quality widget framework with 14,943 lines of Python code across 24 files. The framework provides comprehensive state management, input handling (mouse/keyboard), event systems with callbacks, styling configurations, and dirty-tracking for efficient rendering. Widgets prepare geometry data for an external renderer rather than performing GPU rendering directly.

## Scope

### In Scope

1. **Existing Widget Categories (60+ classes)**
   - Input Widgets: Button, Checkbox, Slider, TextInput, Dropdown
   - Display Widgets: Label, ProgressBar, Icon, Badge
   - Game Widgets: HealthBar, Minimap, InventorySlot, DamageNumbers, Tooltip, RichTooltip
   - Primitives: Text, Image, Border, Spacer

2. **Current Capabilities**
   - Complete state machines for all interactive widgets
   - Event system with subscription/unsubscription callbacks
   - Input handling (mouse click/move/drag, keyboard navigation, clipboard)
   - Dirty-tracking for render optimization
   - Serialization (to_dict/from_dict) for persistence
   - Accessibility support (screen reader text, ARIA roles)
   - Animation support (easing, interpolation)

3. **Missing Systems to Implement**
   - Layout engine (widgets use absolute positioning only)
   - GPU rendering code (widgets prepare data, do not draw)
   - Focus management system (widgets have focus state but no focus coordinator)
   - Input dispatch system (no central input router)

### Out of Scope

- Complete UI framework rewrite
- External UI library integration (imgui, Qt)
- Widget design changes beyond integration requirements

## Goals

1. Complete the widget framework with missing infrastructure systems
2. Integrate widget geometry output with Rust renderer backend
3. Provide layout primitives (flex, grid) for widget positioning
4. Implement centralized focus and input management

## Constraints

- Must maintain compatibility with existing 24 widget files
- Widgets produce geometry data for external Rust renderer (no Python-side GPU rendering)
- Must preserve existing event subscription/callback patterns
- Accessibility support must be maintained
- Animation and dirty-tracking systems must remain functional

## Acceptance Criteria

### Phase 1: Layout Engine
- [ ] Flex layout container implemented
- [ ] Grid layout container implemented
- [ ] Layout integrates with existing widget bounds (x, y, width, height)
- [ ] Widgets can be positioned via layout or absolute coordinates
- [ ] Layout recalculation on widget resize/add/remove

### Phase 2: Focus Management
- [ ] Focus coordinator tracks currently focused widget
- [ ] Focus navigation (Tab, Shift+Tab, arrow keys)
- [ ] Focus visual states propagate through dirty-tracking
- [ ] Focus trapping for modal dialogs
- [ ] Focus restoration on dialog close

### Phase 3: Input Dispatch
- [ ] Central input router receives all input events
- [ ] Input events dispatched to appropriate widget based on position/focus
- [ ] Event propagation (bubbling, capturing) supported
- [ ] Input modifiers (shift, ctrl, alt) properly forwarded
- [ ] Drag-and-drop coordination across widgets

### Phase 4: Renderer Integration
- [ ] Widget geometry data formatted for Rust renderer consumption
- [ ] Batch geometry updates when widgets dirty
- [ ] Nine-slice, UV coords, and scale modes passed to renderer
- [ ] Animation frame data synchronized with render loop
- [ ] Accessibility tree exposed to renderer for platform integration

## File Inventory

| File | Lines | Category | Status |
|------|-------|----------|--------|
| __init__.py | 227 | Core | Complete |
| constants.py | 234 | Core | Complete |
| button.py | 682 | Input | Complete |
| checkbox.py | 564 | Input | Complete |
| slider.py | 806 | Input | Complete |
| text_input.py | 1,394 | Input | Complete |
| dropdown.py | 1,208 | Input | Complete |
| label.py | 676 | Display | Complete |
| progress_bar.py | 1,039 | Display | Complete |
| icon.py | 824 | Display | Complete |
| badge.py | 763 | Display | Complete |
| health_bar.py | 672 | Game | Complete |
| minimap.py | 885 | Game | Complete |
| inventory_slot.py | 826 | Game | Complete |
| damage_numbers.py | 634 | Game | Complete |
| tooltip.py | 1,103 | Game | Complete |
| text.py | 940 | Primitives | Complete |
| image.py | 673 | Primitives | Complete |
| border.py | 258 | Primitives | Complete |
| spacer.py | 253 | Primitives | Complete |

**Total**: 14,943 lines across 24 files (20 listed above + 4 additional subdirectory files)
