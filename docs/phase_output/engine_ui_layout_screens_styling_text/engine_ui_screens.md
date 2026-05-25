# Investigation: engine/ui/screens

## Summary
The `engine/ui/screens` module is a fully-realized screen management system with complete lifecycle management, navigation stack operations, LRU caching, and a comprehensive transition system with 22 easing functions and multiple transition types (fade, slide, zoom, composite, custom). This is production-quality code, not stubs.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 87 | COMPLETE | Clean re-exports of 27 symbols |
| `screen.py` | 643 | COMPLETE | Full Screen base class with lifecycle, state, params, widgets |
| `screen_stack.py` | 995 | COMPLETE | Full navigation stack with push/pop/replace/clear/modal/overlay |
| `transitions.py` | 1024 | COMPLETE | 22 easing functions, 6 transition types, factory pattern |

**Total: 2,749 lines of implementation**

## Screen Components

### Screen Base Class (`screen.py`)
- `ScreenState` enum: ENTERING, ACTIVE, PAUSED, EXITING, DESTROYED
- `ScreenParams` dataclass: Data passing between screens with `data`, `source_screen`, `transition_override`
- `ScreenResult` dataclass: Return values from screens with `success` and `data`
- `Screen` abstract base class:
  - Lifecycle methods: `on_create`, `on_destroy`, `on_enter`, `on_exit`, `on_pause`, `on_resume`
  - Widget tree management: `add_widget`, `get_widget`, `remove_widget`, `clear_widgets`
  - Callback registration: `add_on_enter`, `add_on_exit`, `add_on_pause`, `add_on_resume`, `add_on_state_change`
  - Configuration flags: `is_modal`, `blocks_input`, `is_overlay`, `can_go_back`, `pause_below`
  - Per-frame `update(delta_time)` hook

### Screen Stack (`screen_stack.py`)
- `StackOperation` enum: PUSH, POP, REPLACE, CLEAR, SWAP
- `HistoryEntry` dataclass: Navigation history tracking with timestamps
- `ScreenCache` class: LRU cache (configurable max size, default 10)
- `ScreenStack` class:
  - Factory pattern: `register_factory(name, factory)` for lazy screen instantiation
  - Navigation ops: `push`, `pop`, `replace`, `clear`, `swap`, `pop_to`, `pop_to_root`
  - Modal support: `push_modal`, `push_overlay`
  - Query ops: `get`, `get_by_name`, `contains`, `index_of`, `get_screens_above`, `get_screens_below`
  - Back navigation: `back()` with screen consumption
  - History: up to 100 entries with trimming

### Transitions (`transitions.py`)
- **22 Easing Functions**: LINEAR, EASE_IN/OUT/IN_OUT, QUAD, CUBIC, EXPO, BACK (overshoot), ELASTIC, BOUNCE
- **Transition Interface** (`ITransition`): `start`, `update`, `reset`, `get_exiting_transform`, `get_entering_transform`
- **Transition Types**:
  - `FadeTransition`: Alpha blend with crossfade option
  - `SlideTransition`: LEFT/RIGHT/UP/DOWN with push mode
  - `ZoomTransition`: Scale in/out with configurable min/max scale
  - `InstantTransition`: No animation
  - `CompositeTransition`: Combine multiple effects
  - `CustomTransition`: User-provided transform functions
- **TransitionFactory**: Static factory methods for common transitions

## Implementation
- Real screen management? **YES** - Full lifecycle with state machine, callbacks, widget tree
- Real transitions? **YES** - 6 transition types, 22 easing curves, time-based updates
- Real HUD system? **PARTIAL** - Overlay screens supported via `push_overlay`, but no dedicated HUD class

## Verdict
**REAL IMPLEMENTATION** - Production-quality screen management system

## Evidence

### Screen Lifecycle State Machine
```python
class ScreenState(Enum):
    ENTERING = auto()      # Screen is transitioning in
    ACTIVE = auto()        # Screen is fully active and interactive
    PAUSED = auto()        # Screen is paused (e.g., another screen on top)
    EXITING = auto()       # Screen is transitioning out
    DESTROYED = auto()     # Screen has been removed and cleaned up
```

### Stack Operations with Transition Integration
```python
def push(
    self,
    name_or_screen: str | Screen,
    params: Optional[ScreenParams] = None,
    transition: Optional["ITransition"] = None,
    use_cache: bool = True,
) -> Optional[Screen]:
    # Get or create screen
    if isinstance(name_or_screen, str):
        screen = self._get_or_create_screen(name_or_screen, params, use_cache)
        if screen is None:
            return None
    else:
        screen = name_or_screen
    
    # Pause the current top screen
    if old_top and self._auto_pause_below and screen.pause_below:
        old_top._pause()
    
    # Add to stack
    self._stack.append(screen)
    screen._stack = self
    
    # Record history
    self._record_history(screen.name, params, StackOperation.PUSH)
    
    # Start enter transition
    screen._enter(params)
```

### Easing Function Implementation (Bounce Example)
```python
def _ease_out_bounce(t: float) -> float:
    """Bounce ease out."""
    n1 = BOUNCE_AMPLITUDE  # 7.5625
    d1 = BOUNCE_DIVISOR    # 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    elif t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375
```

### LRU Screen Cache
```python
class ScreenCache:
    def _evict_if_needed(self) -> None:
        """Evict least recently used screens if cache is full."""
        while len(self._cache) > self._max_size:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
```

## Notes
- No separate HUD or menu classes found - these would be implemented as custom Screen subclasses
- Transition system outputs transform dictionaries (alpha, x, y, scale) that would need a renderer to apply
- Widget tree management is type-agnostic (`Any`) - integrates with whatever widget system is used
- Memory leak prevention built in: `_cleanup_internal()` clears callbacks and references on destroy
