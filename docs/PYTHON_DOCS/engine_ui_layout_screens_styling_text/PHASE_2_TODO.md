# PHASE 2 TODO: Screens Module

---

## Screen Lifecycle Tasks

### T-2.1: Verify Screen State Transitions

**File**: `engine/ui/screens/screen.py`

**Description**: Ensure state machine transitions are valid and complete.

**Acceptance Criteria**:
- [ ] INACTIVE -> ENTERING transition works
- [ ] ENTERING -> ACTIVE transition works
- [ ] ACTIVE -> EXITING transition works
- [ ] EXITING -> INACTIVE transition works
- [ ] ACTIVE -> PAUSED transition works
- [ ] PAUSED -> ACTIVE (resume) transition works
- [ ] Invalid transitions are rejected or logged

---

### T-2.2: Verify Lifecycle Method Calls

**File**: `engine/ui/screens/screen.py`

**Description**: Ensure lifecycle methods are called at correct times.

**Acceptance Criteria**:
- [ ] `on_enter()` called when state becomes ENTERING
- [ ] `on_exit()` called when state becomes EXITING
- [ ] `on_pause()` called when state becomes PAUSED
- [ ] `on_resume()` called when returning from PAUSED to ACTIVE
- [ ] `on_back_pressed()` called on back navigation

---

### T-2.3: Verify Screen Parameters

**File**: `engine/ui/screens/screen.py`

**Description**: Ensure `ScreenParams` are passed correctly.

**Acceptance Criteria**:
- [ ] Params available in `on_enter()` via `self.params`
- [ ] Params support get with default value
- [ ] Params are immutable after creation
- [ ] Params are hashable for cache key generation

---

### T-2.4: Verify Screen Results

**File**: `engine/ui/screens/screen.py`

**Description**: Ensure `ScreenResult` is returned to previous screen.

**Acceptance Criteria**:
- [ ] `exit_with_result()` triggers exit transition
- [ ] Result is delivered to previous screen's `on_result()`
- [ ] Result is `None` if screen exits without result
- [ ] Result contains expected data types

---

## Screen Stack Tasks

### T-2.5: Verify Push Operation

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure `push()` correctly adds screen to stack.

**Acceptance Criteria**:
- [ ] New screen becomes top of stack
- [ ] Previous screen's `on_exit()` is called (non-modal)
- [ ] Transition animation plays between screens
- [ ] Stack size increases by 1

---

### T-2.6: Verify Pop Operation

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure `pop()` correctly removes screen from stack.

**Acceptance Criteria**:
- [ ] Top screen is removed from stack
- [ ] Screen's `on_exit()` is called
- [ ] Previous screen's `on_enter()` or `on_resume()` is called
- [ ] Transition animation plays (reverse direction)
- [ ] Returns result if screen exited with result

---

### T-2.7: Verify Replace Operation

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure `replace()` swaps top screen.

**Acceptance Criteria**:
- [ ] Old top screen is removed
- [ ] New screen becomes top
- [ ] Stack size remains same
- [ ] Back navigation does not return to replaced screen

---

### T-2.8: Verify Pop To Root

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure `pop_to_root()` clears stack to first screen.

**Acceptance Criteria**:
- [ ] All screens except root are removed
- [ ] Each removed screen's `on_exit()` is called
- [ ] Root screen becomes visible
- [ ] Stack size is 1 after operation

---

### T-2.9: Verify Modal Push

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure modal screens pause underlying screen.

**Acceptance Criteria**:
- [ ] Underlying screen receives `on_pause()`, not `on_exit()`
- [ ] Modal screen receives `on_enter()`
- [ ] Underlying screen remains in stack
- [ ] Modal renders on top of underlying screen

---

### T-2.10: Verify Modal Pop

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure popping modal resumes underlying screen.

**Acceptance Criteria**:
- [ ] Modal screen receives `on_exit()`
- [ ] Underlying screen receives `on_resume()`
- [ ] Underlying screen becomes visible
- [ ] Result from modal delivered to underlying screen

---

### T-2.11: Verify Screen Cache

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure LRU cache works correctly.

**Acceptance Criteria**:
- [ ] Cached screens are reused on subsequent push
- [ ] Cache key includes screen type and params hash
- [ ] LRU eviction removes least recently used first
- [ ] Evicted screens have `on_exit()` called
- [ ] Cache respects max size limit

---

### T-2.12: Verify Navigation History

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure back/forward navigation works.

**Acceptance Criteria**:
- [ ] `can_go_back()` returns `True` when history has previous entries
- [ ] `can_go_forward()` returns `True` after going back
- [ ] `go_back()` navigates to previous screen
- [ ] `go_forward()` navigates to next screen in history
- [ ] History updates on push/pop operations

---

### T-2.13: Verify Deep Linking

**File**: `engine/ui/screens/screen_stack.py`

**Description**: Ensure `navigate_to_deep_link()` parses and navigates correctly.

**Acceptance Criteria**:
- [ ] URL path segments resolve to screen types
- [ ] Query parameters become screen params
- [ ] Nested paths create stack of screens
- [ ] Invalid paths handled gracefully

---

## Transition Tasks

### T-2.14: Verify Fade Transition

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure fade transition animates opacity correctly.

**Acceptance Criteria**:
- [ ] Outgoing screen fades from 1.0 to 0.0 opacity
- [ ] Incoming screen fades from 0.0 to 1.0 opacity
- [ ] Crossfade mode overlaps both screens
- [ ] Duration is respected

---

### T-2.15: Verify Slide Transition

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure slide transition moves screens correctly.

**Acceptance Criteria**:
- [ ] `direction="left"` slides outgoing left, incoming from right
- [ ] `direction="right"` slides outgoing right, incoming from left
- [ ] `direction="up"` slides outgoing up, incoming from bottom
- [ ] `direction="down"` slides outgoing down, incoming from top
- [ ] Positions interpolate smoothly

---

### T-2.16: Verify Zoom Transition

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure zoom transition scales screens correctly.

**Acceptance Criteria**:
- [ ] Outgoing screen scales down (zoom out)
- [ ] Incoming screen scales up (zoom in)
- [ ] Scale origin is configurable (center, corner)
- [ ] Combined with fade for smooth effect

---

### T-2.17: Verify All 22 Easing Functions

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure each easing function produces correct curves.

**Acceptance Criteria**:
- [ ] `linear(t)` returns `t`
- [ ] `ease_in_quad(t)` returns `t * t`
- [ ] `ease_out_quad(t)` returns `t * (2 - t)`
- [ ] `ease_in_out_quad(t)` is symmetric
- [ ] Cubic, Quart, Quint follow power law patterns
- [ ] Sine functions use trigonometric formulas
- [ ] Expo functions use exponential formulas
- [ ] Back functions overshoot target
- [ ] Bounce functions simulate bouncing physics

---

### T-2.18: Verify Composite Transitions (Parallel)

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure parallel composition runs transitions simultaneously.

**Acceptance Criteria**:
- [ ] All child transitions start at same time
- [ ] Progress is synchronized across children
- [ ] Composite completes when longest child completes
- [ ] Effects combine (fade + slide at once)

---

### T-2.19: Verify Composite Transitions (Sequential)

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure sequential composition runs transitions in order.

**Acceptance Criteria**:
- [ ] Second transition starts after first completes
- [ ] Total duration is sum of child durations
- [ ] Progress splits across children
- [ ] Effects chain (fade out, then slide in)

---

### T-2.20: Verify Transition Timing

**File**: `engine/ui/screens/transitions.py`

**Description**: Ensure transitions respect duration and complete correctly.

**Acceptance Criteria**:
- [ ] `transition.start()` initializes state
- [ ] `transition.update(dt)` advances progress
- [ ] Progress reaches 1.0 after duration elapsed
- [ ] `transition.finish()` cleans up state
- [ ] Interrupted transitions handle gracefully
