# PHASE 4 ARCHITECTURE: Animation, Audio & Presentation

## Scope

Animation, audio, and presentation decorators:
- `ik_procedural.py` (288 lines)
- `animation.py` (160 lines)
- `cinematics.py` (155 lines)
- `audio.py` (218 lines)
- `ui.py` (165 lines)
- `localization.py` (227 lines)
- `accessibility.py` (120 lines)

## Architecture Pattern: Enum-Like Validation

These files use extensive VALID_* constants for domain-specific enums:

```python
VALID_IK_SOLVERS = frozenset({"fabrik", "ccd", "jacobian", "analytical"})
VALID_EASING = frozenset({"linear", "ease_in", "ease_out", "ease_in_out", ...})
VALID_FALLOFF = frozenset({"linear", "logarithmic", "inverse_square"})
```

## Component: IK & Procedural Animation

**File**: `trinity/decorators/ik_procedural.py` (288 lines)

**Decorators**: `@ik_chain`, `@ik_goal`, `@procedural_bone`, `@motion_matching`, `@ragdoll`

**Architecture**:
- IK solver enum validation
- Chain configuration with joint limits
- Motion matching database references
- Ragdoll physics parameters

**Solver Enums**:
```python
VALID_IK_SOLVERS = frozenset({"fabrik", "ccd", "jacobian", "analytical"})
VALID_MOTION_MATCHING_FEATURES = frozenset({"position", "velocity", "trajectory", "pose"})
```

## Component: Animation Decorators

**File**: `trinity/decorators/animation.py` (160 lines)

**Decorators**: `@tween`, `@blend_tree`

**Architecture**:
- Easing function validation
- Blend tree node configuration
- Animation curve support

**Easing Functions**:
```python
VALID_EASING = frozenset({
    "linear",
    "ease_in", "ease_out", "ease_in_out",
    "ease_in_quad", "ease_out_quad", "ease_in_out_quad",
    "ease_in_cubic", "ease_out_cubic", "ease_in_out_cubic",
    "ease_in_elastic", "ease_out_elastic", "ease_in_out_elastic",
    "ease_in_bounce", "ease_out_bounce", "ease_in_out_bounce",
})
```

## Component: Cinematics Decorators

**File**: `trinity/decorators/cinematics.py` (155 lines)

**Decorators**: `@cutscene`, `@camera_track`

**Architecture**:
- Camera blend validation
- Track sequencing
- Cutscene timeline management

**Blend Types**:
```python
VALID_CAMERA_BLENDS = frozenset({"cut", "ease", "smooth", "spring"})
```

## Component: Audio Decorators

**File**: `trinity/decorators/audio.py` (218 lines)

**Decorators**: `@sound`, `@audio_bus`, `@spatial_audio`

**Architecture**:
- Falloff curve validation
- Bus routing configuration
- 3D spatial audio settings

**Falloff Curves**:
```python
VALID_FALLOFF = frozenset({"linear", "logarithmic", "inverse_square", "custom"})
```

## Component: UI Decorators

**File**: `trinity/decorators/ui.py` (165 lines)

**Decorators**: `@widget`, `@layout`

**Architecture**:
- Layout direction validation
- Widget hierarchy management
- Style binding

**Layout Directions**:
```python
VALID_DIRECTIONS = frozenset({"horizontal", "vertical", "grid", "flow"})
```

## Component: Localization Decorators

**File**: `trinity/decorators/localization.py` (227 lines)

**Decorators**: `@localized`, `@plural`, `@rtl_aware`, `@text_overflow`

**Architecture**:
- Target type validation (text fields)
- Pluralization rules
- RTL language support
- Text overflow handling

**Pattern**:
```python
def _validate_target_type(target, decorator_name):
    """Validate target is a text-containing field."""
    if not hasattr(target, '__text__'):
        raise TypeError(f"{decorator_name} can only be applied to text fields")
```

## Component: Accessibility Decorators

**File**: `trinity/decorators/accessibility.py` (120 lines)

**Decorators**: `@accessible`

**Architecture**:
- ARIA role validation
- Screen reader hints
- Keyboard navigation support

**ARIA Roles**:
```python
VALID_ROLES = frozenset({
    "button", "checkbox", "dialog", "grid", "heading",
    "img", "link", "list", "listitem", "menu", "menuitem",
    "navigation", "progressbar", "radiogroup", "slider",
    "spinbutton", "tab", "tablist", "tabpanel", "textbox",
    "tree", "treeitem",
})
```

## Op Types Used

| Op | Purpose | Files |
|----|---------|-------|
| `Op.TAG` | Store animation/audio metadata | All |
| `Op.REGISTER` | Register in "animation", "audio", "ui" | All |
| `Op.DESCRIBE` | Generate documentation | localization.py |
| `Op.VALIDATE` | Runtime validation | accessibility.py |

## Key Decisions

1. **Extensive enums**: Presentation systems have many valid options, all as frozensets
2. **Target type validation**: Some decorators (localization) validate target type, not just parameters
3. **Accessibility compliance**: ARIA roles follow W3C specifications
4. **Audio falloff**: Physics-based falloff curves for realistic spatial audio
