# PHASE 2 ARCHITECTURE: Screens Module

---

## Overview

The Screens Module provides screen lifecycle management, navigation, and transition animations. It comprises 3 files (~2,659 lines) implementing the screen abstraction, stack-based navigation, and 22+ easing-based transitions.

---

## Component Architecture

### Screen Base (screen.py — 642 lines)

**Purpose**: Base screen class defining lifecycle and state machine.

**Classes**:
- `Screen` — Base class for all screens
- `ScreenParams` — Parameters passed to screen on entry
- `ScreenResult` — Result returned when screen exits
- `ScreenState` — Enumeration of screen states

**State Machine**:
```
INACTIVE ──> ENTERING ──> ACTIVE ──> EXITING ──> INACTIVE
                │           │
                │           v
                │        PAUSED
                │           │
                └───────────┘
                  (on_resume)
```

**Lifecycle Methods**:
| Method | When Called | Purpose |
|--------|-------------|---------|
| `on_enter()` | Transition into screen starts | Initialize resources |
| `on_exit()` | Transition out starts | Release resources |
| `on_pause()` | Screen obscured by modal | Pause updates |
| `on_resume()` | Modal dismissed | Resume updates |
| `on_back_pressed()` | Back navigation triggered | Return `True` to consume |

**Parameter Passing**:
```python
# Navigate with parameters
stack.push(SettingsScreen, params=ScreenParams(theme="dark"))

# Receive in screen
class SettingsScreen(Screen):
    def on_enter(self):
        theme = self.params.get("theme", "light")
```

**Result Handling**:
```python
# Exit with result
self.exit_with_result(ScreenResult(selected_item=item))

# Receive in previous screen
def on_result(self, result: ScreenResult):
    if result.selected_item:
        self.apply_selection(result.selected_item)
```

---

### Screen Stack (screen_stack.py — 994 lines)

**Purpose**: Stack-based navigation with history tracking.

**Classes**:
- `ScreenStack` — Main navigation controller
- `ScreenCache` — LRU cache for screen instances
- `NavigationHistory` — Back/forward history tracking

**Stack Operations**:
| Operation | Effect | Use Case |
|-----------|--------|----------|
| `push(Screen)` | Add to top | Normal forward navigation |
| `pop()` | Remove top | Back navigation |
| `replace(Screen)` | Replace top | Login -> Home (no back) |
| `pop_to_root()` | Clear to first | Return to home |
| `pop_until(predicate)` | Pop until match | Navigate to specific screen |

**Navigation History**:
```
History: [A] -> [A, B] -> [A, B, C]
                              |
Back:    [A, B, C] -> [A, B]  |  Forward: [A, B] -> [A, B, C]
```

**Screen Cache**:
- LRU eviction policy (`_cache_eviction()`)
- Configurable max size
- Screens cached by type + params hash
- Evicted screens have `on_exit()` called

**Modal Support**:
```python
# Push modal (previous screen paused, not exited)
stack.push_modal(ConfirmDialog)

# Underlying screen receives on_pause()
# When modal pops, underlying receives on_resume()
```

**Deep Linking**:
```python
# Parse deep link
stack.navigate_to_deep_link("/settings/account?tab=security")

# Resolves to:
# push(SettingsScreen)
# push(AccountScreen, params={tab: "security"})
```

---

### Transitions (transitions.py — 1,023 lines)

**Purpose**: Animated transitions between screens.

**Classes**:
- `Transition` — Base transition class
- `FadeTransition` — Opacity fade
- `SlideTransition` — Directional slide
- `ZoomTransition` — Scale zoom
- `CompositeTransition` — Combine transitions

**22 Easing Functions**:

| Family | In | Out | InOut |
|--------|-----|-----|-------|
| Linear | `linear` | — | — |
| Quad | `ease_in_quad` | `ease_out_quad` | `ease_in_out_quad` |
| Cubic | `ease_in_cubic` | `ease_out_cubic` | `ease_in_out_cubic` |
| Quart | `ease_in_quart` | `ease_out_quart` | `ease_in_out_quart` |
| Quint | `ease_in_quint` | `ease_out_quint` | `ease_in_out_quint` |
| Sine | `ease_in_sine` | `ease_out_sine` | `ease_in_out_sine` |
| Expo | `ease_in_expo` | `ease_out_expo` | `ease_in_out_expo` |
| Back | `ease_in_back` | `ease_out_back` | `ease_in_out_back` |
| Bounce | `ease_in_bounce` | `ease_out_bounce` | `ease_in_out_bounce` |

**Easing Formulas**:
```python
# Quadratic
ease_in_quad = t * t
ease_out_quad = t * (2 - t)
ease_in_out_quad = 2*t*t if t < 0.5 else -1 + (4-2*t)*t

# Bounce (complex piecewise)
def ease_out_bounce(t):
    if t < 1/2.75:
        return 7.5625 * t * t
    elif t < 2/2.75:
        t -= 1.5/2.75
        return 7.5625 * t * t + 0.75
    # ... additional segments
```

**Transition Composition**:
```python
# Parallel: fade and slide simultaneously
parallel = CompositeTransition.parallel([
    FadeTransition(duration=0.3),
    SlideTransition(direction="left", duration=0.3)
])

# Sequential: fade out, then slide in
sequential = CompositeTransition.sequential([
    FadeTransition(target="outgoing", duration=0.15),
    SlideTransition(target="incoming", duration=0.15)
])
```

**Transition Flow**:
```
_apply_transition(outgoing, incoming, transition)
    -> transition.start()
    -> per frame: transition.update(delta_time)
    -> transition.on_progress(t)  # t in [0, 1], eased
    -> transition.finish()
```

---

## Module Dependencies

```
screen.py       --(standalone)
screen_stack.py --> screen.py (imports Screen, ScreenState)
transitions.py  --(standalone, may import for type hints)
```

---

## Integration Points

1. **Layout Module** — Screens contain layouts for their content
2. **Styling Module** — Screens may have associated styles
3. **Input System** — Back button events routed to `on_back_pressed()`
4. **Render Loop** — Transitions need frame updates

---

## Data Structures

### Screen Stack
```
Stack: [Screen, Screen, Screen, ...]
        ^bottom             ^top (visible)
```

### Navigation History
```
History: {
    entries: [(ScreenType, ScreenParams), ...],
    current_index: int
}
```

### Screen Cache (LRU)
```
Cache: {
    (ScreenType, params_hash): Screen,
    ...
}
access_order: [key, key, key, ...]  # most recent at end
max_size: int
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Push | O(1) | Append to stack |
| Pop | O(1) | Remove from stack |
| Cache lookup | O(1) | Hash map |
| Cache eviction | O(1) | LRU head removal |
| History back/forward | O(1) | Index move |
| Deep link parse | O(n) | n = path segments |

---

## Design Decisions

1. **Stack-Based Navigation** — Matches mobile app paradigms (iOS/Android)
2. **LRU Cache** — Balances memory with navigation speed
3. **Modal Support** — Explicit pause/resume for underlying screens
4. **22 Easing Functions** — Comprehensive animation vocabulary
5. **Composite Transitions** — Flexible animation composition
6. **Deep Linking** — URL-based navigation for external triggers
