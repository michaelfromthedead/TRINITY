# Archaeological Investigation: engine/ui/accessibility, animation, binding, framework

**Date**: 2026-05-22  
**Investigator**: Research Agent  
**Scope**: 19 files across 4 modules, ~15,949 lines  

---

## Executive Summary

**Classification**: **REAL** (all 19 files)

All four modules contain production-quality implementations with complete algorithms, proper error handling, and comprehensive documentation. No stubs or placeholder code detected.

---

## Module Classifications

| Module | Files | Lines | Classification |
|--------|-------|-------|----------------|
| accessibility | 5 | ~3,449 | REAL |
| animation | 5 | ~4,233 | REAL |
| binding | 4 | ~3,526 | REAL |
| framework | 5 | ~3,924 | REAL |

---

## Detailed File Analysis

### 1. engine/ui/accessibility (5 files, ~3,449 lines)

#### keyboard_nav.py (839 lines) - REAL

**Purpose**: Keyboard navigation system for UI widgets

**Key Classes**:
- `KeyboardNavigator`: Main controller with focus management
- `TabOrder`: Manages sequential tab navigation order
- `TabStop`: Individual focusable element wrapper
- `NavigationGroup`: Groups related navigation targets
- `SkipLink`: Accessibility skip navigation support
- `KeyboardShortcut`: Hotkey binding system

**Evidence of REAL Implementation**:
- Complete spatial analysis algorithm for nearest focus target detection
- Arrow key navigation with proper direction handling
- Tab order calculation with dynamic reordering
- Skip link activation for accessibility compliance

**Key Algorithm** (spatial navigation):
```python
# Finds nearest widget in a direction using bounding box analysis
def _find_nearest_in_direction(self, current, direction):
    # Calculates distances and angles to all candidates
    # Returns closest matching widget based on direction vector
```

---

#### scale.py (696 lines) - REAL

**Purpose**: DPI awareness and UI scaling

**Key Classes**:
- `ScaleManager`: Singleton managing global scale factors
- `ScaleConfig`: Configuration for scale behavior
- `MonitorInfo`: Per-monitor DPI information
- `TouchTargetSize`: WCAG 2.5.5 compliant sizing

**Evidence of REAL Implementation**:
- WCAG 2.5.5 touch target requirements (44x44 CSS pixels minimum)
- Font scaling presets with accessibility levels
- Multi-monitor DPI awareness
- Zoom level management with bounds

**Key Algorithm** (touch target validation):
```python
MINIMUM_TARGET_SIZE = 44  # WCAG 2.5.5 requirement
def validate_touch_target(self, width: float, height: float) -> bool:
    return width >= self.MINIMUM_TARGET_SIZE and height >= self.MINIMUM_TARGET_SIZE
```

---

#### motion.py (664 lines) - REAL

**Purpose**: Reduced motion preference system

**Key Classes**:
- `MotionManager`: Singleton for motion preferences
- `MotionConfig`: Configuration container
- `AnimationPreference`: User preference levels

**Evidence of REAL Implementation**:
- System preference detection via platform APIs
- Category-based animation control (essential vs decorative)
- Duration multipliers for slowing animations
- Integration with OS accessibility settings

**Key Algorithm** (animation filtering):
```python
def should_animate(self, category: AnimationCategory) -> bool:
    if self._preference == AnimationPreference.NO_PREFERENCE:
        return True
    if self._preference == AnimationPreference.REDUCE:
        return category in self._allowed_categories
    return False
```

---

#### screen_reader.py (642 lines) - REAL

**Purpose**: ARIA support for screen readers

**Key Classes**:
- `AccessibilityManager`: Singleton managing announcements
- `AriaRole`: 100+ ARIA roles enumeration
- `AriaProperty`: ARIA properties (describedby, labelledby, etc.)
- `AriaState`: ARIA states (expanded, selected, checked, etc.)
- `AriaLiveRegion`: Live region with politeness levels

**Evidence of REAL Implementation**:
- Complete ARIA role taxonomy
- Focus change announcements
- State change announcements with delta detection
- Live region support (polite, assertive, off)
- Platform-specific screen reader integration hooks

**Key Algorithm** (announcement queueing):
```python
def announce(self, message: str, priority: Priority = Priority.POLITE):
    if priority == Priority.ASSERTIVE:
        self._queue.insert(0, message)  # Assertive goes to front
    else:
        self._queue.append(message)
```

---

#### high_contrast.py (608 lines) - REAL

**Purpose**: High contrast mode and colorblind support

**Key Classes**:
- `HighContrastManager`: Theme switching and contrast calculation
- `Color`: RGBA color with conversion utilities
- `HighContrastTheme`: Predefined contrast themes
- `FocusIndicator`: High-visibility focus ring configuration

**Evidence of REAL Implementation**:
- WCAG 2.1 contrast ratio calculation with sRGB linearization
- Colorblind simulation using Brettel, Vienot, Mollon transformation matrices
- Theme switching with smooth transitions
- Focus indicator customization

**Key Algorithm** (WCAG contrast ratio):
```python
def contrast_ratio(self, color1: Color, color2: Color) -> float:
    l1 = self._relative_luminance(color1)
    l2 = self._relative_luminance(color2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def _relative_luminance(self, color: Color) -> float:
    r = self._linearize(color.r / 255)
    g = self._linearize(color.g / 255)
    b = self._linearize(color.b / 255)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def _linearize(self, value: float) -> float:
    if value <= 0.03928:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4
```

**Key Algorithm** (colorblind simulation - Brettel coefficients):
```python
# Protanopia, deuteranopia, tritanopia transformation matrices
BRETTEL_PROTAN = [
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998]
]
```

---

### 2. engine/ui/animation (5 files, ~4,233 lines)

#### animator.py (928 lines) - REAL

**Purpose**: State machine-based animation system

**Key Classes**:
- `Animator`: Main state machine controller
- `AnimatorState`: Individual animation state
- `AnimationState`: Runtime state with progress tracking
- `AnimationTransition`: Transition between states
- `AnimationLayer`: Layered animation blending

**Evidence of REAL Implementation**:
- Complete state machine with transitions
- Layer blending modes (override, additive, multiply, average)
- Transition callbacks and conditions
- Parameter-driven transitions

**Key Algorithm** (layer blending):
```python
def _blend_layers(self) -> dict:
    result = {}
    for layer in sorted(self._layers, key=lambda l: l.priority):
        layer_values = layer.get_current_values()
        if layer.blend_mode == BlendMode.OVERRIDE:
            result.update(layer_values)
        elif layer.blend_mode == BlendMode.ADDITIVE:
            for k, v in layer_values.items():
                result[k] = result.get(k, 0) + v * layer.weight
```

---

#### tween.py (899 lines) - REAL

**Purpose**: Property tweening with interpolation

**Key Classes**:
- `Tween[T]`: Generic property interpolator
- `TweenSequence`: Sequential tween chain
- `TweenGroup`: Parallel tween execution
- `TweenManager`: Global tween lifecycle

**Evidence of REAL Implementation**:
- Generic type support for interpolation
- Yoyo and repeat modes
- Delay and duration controls
- Value interpolation for numeric, tuple, dict types
- Callback hooks (on_start, on_update, on_complete)

**Key Algorithm** (value interpolation):
```python
def _interpolate(self, start: T, end: T, t: float) -> T:
    if isinstance(start, (int, float)):
        return start + (end - start) * t
    if isinstance(start, tuple):
        return tuple(s + (e - s) * t for s, e in zip(start, end))
    if isinstance(start, dict):
        return {k: self._interpolate(start[k], end[k], t) for k in start}
```

---

#### keyframe.py (890 lines) - REAL

**Purpose**: Keyframe animation system

**Key Classes**:
- `Keyframe`: Single keyframe with value and easing
- `KeyframeTrack`: Sequence of keyframes for a property
- `KeyframeAnimation`: Multi-track animation
- `KeyframeAnimationManager`: Lifecycle management

**Evidence of REAL Implementation**:
- Loop modes (ONCE, LOOP, PING_PONG)
- Multi-track animations targeting different properties
- Seek and progress control
- Keyframe insertion and removal

**Key Algorithm** (keyframe interpolation):
```python
def sample(self, time: float) -> Any:
    # Find surrounding keyframes
    for i, kf in enumerate(self._keyframes):
        if kf.time >= time:
            if i == 0:
                return kf.value
            prev = self._keyframes[i - 1]
            t = (time - prev.time) / (kf.time - prev.time)
            eased = kf.easing(t)
            return self._interpolate(prev.value, kf.value, eased)
```

---

#### triggers.py (885 lines) - REAL

**Purpose**: Animation trigger system

**Key Classes**:
- `TriggerBase`: Abstract trigger interface
- `StateTrigger`: Widget state monitoring
- `EventTrigger`: Event-based activation
- `PropertyTrigger`: Property value watching
- `DataTrigger`: Data binding triggers
- `MultiTrigger`: Composite triggers with logic

**Evidence of REAL Implementation**:
- Widget state monitoring (hover, pressed, focused)
- Event subscription and cleanup
- Property change detection
- AND/OR/XOR/NAND/NOR logic operators

**Key Algorithm** (multi-trigger logic):
```python
def evaluate(self) -> bool:
    results = [t.evaluate() for t in self._triggers]
    if self._operator == LogicOp.AND:
        return all(results)
    if self._operator == LogicOp.OR:
        return any(results)
    if self._operator == LogicOp.XOR:
        return sum(results) == 1
```

---

#### easing.py (631 lines) - REAL

**Purpose**: Comprehensive easing function library

**Key Functions** (30+):
- Linear, Quad, Cubic, Quart, Quint (in/out/in-out)
- Sine, Expo, Circ (in/out/in-out)
- Elastic, Back, Bounce (in/out/in-out)
- `CubicBezier` class for custom curves

**Evidence of REAL Implementation**:
- Mathematical implementations of all standard easing curves
- CubicBezier with Newton-Raphson iteration for x-to-y mapping
- Elastic and bounce physics with proper damping

**Key Algorithm** (cubic bezier):
```python
class CubicBezier:
    def __call__(self, t: float) -> float:
        # Newton-Raphson to find x parameter
        x = t
        for _ in range(8):
            x_at_t = self._sample_x(x)
            if abs(x_at_t - t) < 1e-6:
                break
            dx = self._sample_dx(x)
            if abs(dx) < 1e-6:
                break
            x -= (x_at_t - t) / dx
        return self._sample_y(x)
```

---

### 3. engine/ui/binding (4 files, ~3,526 lines)

#### binding.py (1281 lines) - REAL

**Purpose**: MVVM data binding core

**Key Classes**:
- `Binding`: Single property binding
- `MultiBinding`: Multiple sources to one target
- `BindingGroup`: Managed binding collection
- `PropertyPath`: Dot-notation path navigation
- `BindingContext`: Hierarchical data context
- `BindingExpression`: Runtime binding state

**Evidence of REAL Implementation**:
- ONE_WAY, TWO_WAY, ONE_TIME, ONE_WAY_TO_SOURCE modes
- Nested property paths with indexer support (`items[0].name`)
- Weak reference cleanup
- Change notification subscription

**Key Algorithm** (property path navigation):
```python
def get_value(self, source: Any) -> Any:
    current = source
    for segment in self._segments:
        if segment.is_indexer:
            current = current[segment.index]
        else:
            current = getattr(current, segment.name)
    return current
```

---

#### validation.py (850 lines) - REAL

**Purpose**: Input validation system

**Key Classes**:
- `IValidator`: Validator interface
- `IAsyncValidator`: Async validation interface
- `ValidationContext`: Validation state container
- Validators: Required, Range, Regex, Email, Url, Composite

**Evidence of REAL Implementation**:
- Trigger modes (ON_CHANGE, ON_BLUR, ON_SUBMIT)
- Async validation support with cancellation
- Composite validators with AND/OR logic
- Localized error messages

**Key Algorithm** (async validation):
```python
async def validate_async(self, value: Any, context: ValidationContext) -> ValidationResult:
    tasks = [v.validate_async(value, context) for v in self._validators]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r.error for r in results if not r.is_valid]
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

---

#### converter.py (752 lines) - REAL

**Purpose**: Value converter system

**Key Classes**:
- `IConverter`: Converter interface
- `BoolToVisibilityConverter`: Bool to visible/hidden
- `NumberFormatConverter`: Number formatting
- `ColorConverter`: Color string parsing
- `ChainedConverter`: Sequential conversion
- `LambdaConverter`: Function-based conversion
- `CachedConverter`: Memoized conversion
- `MultiValueConverter`: Multiple inputs to one output

**Evidence of REAL Implementation**:
- Bidirectional conversion (convert/convert_back)
- Caching with weak key references
- Async converter support
- Chained converter composition

**Key Algorithm** (chained conversion):
```python
def convert(self, value: Any, target_type: type, parameter: Any) -> Any:
    current = value
    for converter in self._converters:
        current = converter.convert(current, target_type, parameter)
    return current

def convert_back(self, value: Any, target_type: type, parameter: Any) -> Any:
    current = value
    for converter in reversed(self._converters):
        current = converter.convert_back(current, target_type, parameter)
    return current
```

---

#### observable.py (643 lines) - REAL

**Purpose**: Observable collections

**Key Classes**:
- `ObservableList`: List with change notifications
- `ObservableDict`: Dict with change notifications
- `VirtualizedListView`: Viewport-based virtualization
- `CollectionChangeEvent`: Change event data

**Evidence of REAL Implementation**:
- Add/remove/replace/move/reset notifications
- Batch change operations with single notification
- Virtualization with widget recycling
- Weak listener references

**Key Algorithm** (virtualization):
```python
def _update_visible_range(self):
    first = int(self._scroll_offset / self._item_height)
    visible_count = int(self._viewport_height / self._item_height) + 2
    last = min(first + visible_count, len(self._source))
    
    # Recycle widgets outside range
    for i in list(self._active_widgets.keys()):
        if i < first or i >= last:
            self._recycle_widget(i)
    
    # Create widgets for newly visible items
    for i in range(first, last):
        if i not in self._active_widgets:
            self._create_widget(i)
```

---

### 4. engine/ui/framework (5 files, ~3,924 lines)

#### widget.py (1031 lines) - REAL

**Purpose**: Base widget class for UI hierarchy

**Key Classes**:
- `Widget`: Base class for all UI elements
- `TrackedDescriptor`: Property descriptor with dirty tracking
- `WidgetStyle`: Style properties container
- `LayoutConstraints`: Min/max/preferred size constraints

**Evidence of REAL Implementation**:
- Hierarchy management (parent, children, traversal)
- Event handler registration and dispatch
- Lifecycle hooks (on_mount, on_unmount, on_update, on_render)
- Dirty tracking for efficient updates
- Hit testing for event routing

**Key Algorithm** (hit testing):
```python
def hit_test(self, x: float, y: float) -> Optional["Widget"]:
    if not self.contains_point(x, y):
        return None
    # Check children in reverse (top to bottom)
    for child in reversed(self._children):
        local_x = x - child.x
        local_y = y - child.y
        result = child.hit_test(local_x, local_y)
        if result:
            return result
    return self if self.is_interactive else None
```

---

#### coordinate.py (770 lines) - REAL

**Purpose**: Coordinate system utilities

**Key Classes**:
- `Point`: 2D point with operations
- `Size`: Width/height dimensions
- `Rect`: Rectangle with position and size
- `Margins`: Top/right/bottom/left spacing
- `Transform2D`: 2D affine transformation matrix
- `CoordinateConverter`: Space conversion utilities

**Evidence of REAL Implementation**:
- Full vector math (add, subtract, scale, dot product)
- Transform composition (translate, rotate, scale)
- Anchor-based positioning
- Coordinate space conversion (local to global, global to local)

**Key Algorithm** (transform composition):
```python
def compose(self, other: "Transform2D") -> "Transform2D":
    # Matrix multiplication for affine transforms
    return Transform2D(
        a=self.a * other.a + self.c * other.b,
        b=self.b * other.a + self.d * other.b,
        c=self.a * other.c + self.c * other.d,
        d=self.b * other.c + self.d * other.d,
        tx=self.a * other.tx + self.c * other.ty + self.tx,
        ty=self.b * other.tx + self.d * other.ty + self.ty
    )
```

---

#### focus.py (753 lines) - REAL

**Purpose**: Focus management system

**Key Classes**:
- `FocusManager`: Singleton managing global focus
- `FocusGroup`: Group of related focusable widgets
- `FocusTrap`: Modal focus containment

**Evidence of REAL Implementation**:
- Tab navigation with wrap-around
- Spatial navigation (arrow keys)
- Focus history for restoration
- Focus trapping for modals and dialogs

**Key Algorithm** (focus trap):
```python
def trap_focus(self, container: Widget):
    self._trap_stack.append(FocusTrap(container))
    focusables = self._get_focusable_descendants(container)
    if focusables:
        self.set_focus(focusables[0])

def release_trap(self):
    if self._trap_stack:
        trap = self._trap_stack.pop()
        if trap.previous_focus and trap.previous_focus.is_focusable:
            self.set_focus(trap.previous_focus)
```

---

#### container.py (708 lines) - REAL

**Purpose**: Container widgets for layout

**Key Classes**:
- `Container`: Base container with children
- `HBox`: Horizontal layout
- `VBox`: Vertical layout
- `Stack`: Z-order stacking layout
- `ScrollContainer`: Scrollable viewport
- `LayoutConfig`: Layout configuration options

**Evidence of REAL Implementation**:
- Horizontal/vertical layout with spacing and alignment
- Main axis and cross axis alignment options
- Scroll with viewport clipping
- Flexible and fixed sizing

**Key Algorithm** (HBox layout):
```python
def _layout_children(self):
    x = self._padding.left
    for child in self._children:
        child.x = x
        child.y = self._calculate_cross_position(child)
        x += child.width + self._spacing
```

---

#### events.py (662 lines) - REAL

**Purpose**: Event system following W3C model

**Key Classes**:
- `UIEvent`: Base event class
- `MouseEvent`: Click, move, scroll events
- `KeyboardEvent`: Key down, up, char input
- `FocusEvent`: Focus in/out events
- `DragEvent`: Drag and drop events
- `EventDispatcher`: Event routing engine

**Evidence of REAL Implementation**:
- W3C event model (capture, target, bubble phases)
- Event stopping and default prevention
- Event cloning for re-dispatch
- Modifier key tracking (shift, ctrl, alt, meta)

**Key Algorithm** (W3C dispatch):
```python
def dispatch(event: UIEvent, target: Widget) -> bool:
    path = []
    current = target
    while current:
        path.insert(0, current)
        current = current.parent

    # Capture phase (root to target, excluding target)
    event.phase = EventPhase.CAPTURE
    for widget in path[:-1]:
        if event.is_stopped:
            break
        widget._dispatch_to_handlers(event, capture=True)

    # Target phase
    if not event.is_stopped:
        event.phase = EventPhase.TARGET
        target._dispatch_to_handlers(event, capture=True)
        if not event.is_stopped_immediate:
            target._dispatch_to_handlers(event, capture=False)

    # Bubble phase (target to root, excluding target)
    if event.bubbles and not event.is_stopped:
        event.phase = EventPhase.BUBBLE
        for widget in reversed(path[:-1]):
            if event.is_stopped:
                break
            widget._dispatch_to_handlers(event, capture=False)

    return not event.is_default_prevented
```

---

## Evidence Summary: Why All Files Are REAL

### Positive Indicators Present

1. **Complete Algorithm Implementations**: Every file contains full working algorithms, not placeholder code
2. **Mathematical Correctness**: WCAG contrast calculation, Brettel colorblind matrices, bezier curves all mathematically accurate
3. **Error Handling**: Proper exception handling, input validation, edge case coverage
4. **Type Annotations**: Comprehensive Python typing throughout
5. **Documentation**: Detailed docstrings explaining purpose and usage
6. **Design Patterns**: Proper use of singleton, observer, strategy, composite patterns
7. **Integration Points**: Clean interfaces between modules

### Negative Indicators Absent

1. **No `raise NotImplementedError`**: No unimplemented methods
2. **No `pass` statements**: No empty method bodies
3. **No `TODO` comments**: No incomplete work markers
4. **No hardcoded test values**: Real parameterized implementations
5. **No circular imports**: Clean dependency graph

---

## Key Algorithms Found

| Algorithm | File | Purpose |
|-----------|------|---------|
| WCAG 2.1 Contrast Ratio | high_contrast.py | sRGB linearization + luminance calculation |
| Brettel Colorblind Simulation | high_contrast.py | Protanopia/deuteranopia/tritanopia matrices |
| Newton-Raphson Bezier | easing.py | x-to-y mapping for cubic bezier curves |
| Spatial Navigation | keyboard_nav.py | Nearest widget in direction |
| W3C Event Dispatch | events.py | Capture/target/bubble phases |
| Virtualized List | observable.py | Widget recycling for large lists |
| Transform Composition | coordinate.py | Affine matrix multiplication |
| State Machine Blending | animator.py | Layer weight interpolation |

---

## Conclusion

All 19 files across engine/ui/accessibility, animation, binding, and framework are **REAL** implementations. The codebase demonstrates professional-quality Python with complete WCAG accessibility support, comprehensive animation systems, MVVM-style data binding, and a W3C-compliant event model. No stubs, placeholders, or incomplete implementations were found.
