# PHASE 1 ARCHITECTURE: Core Infrastructure Decorators

## Scope

Core infrastructure decorators that provide foundational capabilities:
- `lifecycle.py` (189 lines) - Tier 7
- `debug_safety.py` (278 lines) - Tier 10-11
- `composition.py` (138 lines)
- `stacks.py` (122 lines)
- `introspection.py` (99 lines)

## Architecture Pattern

All decorators follow a 6-part structural pattern:

```
1. VALID_* constants (frozenset)
2. Validator function (_validate_X)
3. Step builder function (_X_steps)
4. After-apply function (_after_X)
5. make_decorator() call
6. Registry registration
```

## Component: Lifecycle Decorators

**File**: `trinity/decorators/lifecycle.py` (189 lines, Tier 7)

**Decorators**: `@on_add`, `@on_remove`, `@on_change`, `@on_spawn`, `@on_despawn`

**Architecture**:
- Uses `Op.HOOK` to register lifecycle callbacks
- Hooks fire at component lifecycle boundaries
- After-apply sets `_lifecycle_hooks` attribute on target

**Pattern**:
```python
def _on_add_steps(params):
    return [
        Step(Op.HOOK, {"event": "add", "handler": params["handler"]}),
        Step(Op.TAG, {"key": "has_lifecycle_hooks", "value": True}),
    ]
```

## Component: Debug Safety Decorators

**File**: `trinity/decorators/debug_safety.py` (278 lines, Tier 10-11)

**Decorators**: `@reads`, `@writes`, `@trace_stack`, `@track_changes`

**Architecture**:
- **Mixed pattern**: Manual (`@reads`, `@writes`) + make_decorator (`@trace_stack`, `@track_changes`)
- Manual decorators call `run_steps` directly for fine-grained control
- Uses `Op.TRACK` for change tracking
- Uses `Op.TAG` for read/write dependency markers

**Manual Pattern**:
```python
def reads(*fields):
    def decorator(target):
        steps = [Step(Op.TAG, {"key": "reads", "value": set(fields)})]
        run_steps(target, steps)
        target._reads = set(fields)
        return target
    return decorator
```

## Component: Composition

**File**: `trinity/decorators/composition.py` (138 lines)

**Decorators**: `@composite`, `@alias`

**Architecture**:
- `@composite`: Combines multiple decorators into one
- `@alias`: Creates named alias for existing decorator
- Both validate callable targets
- Uses `Op.TAG` to mark composition metadata

## Component: Stacks

**File**: `trinity/decorators/stacks.py` (122 lines)

**Architecture**:
- Not a decorator file - provides Stack composition utility
- `Stack` class for combining decorators
- `stack()` function for inline composition
- `parameterized_stack()` for runtime parameterization

**Anti-Pattern Detection**:
```python
def _validate_stack_combination(decorators):
    # Hard errors - contradictory combinations
    if "parallel" in names and "exclusive" in names:
        raise ValueError("contradictory")
    
    # Warnings - likely mistakes
    if "networked" in names and "track_changes" not in names:
        warnings.warn("delta sync requires change tracking")
```

## Component: Introspection

**File**: `trinity/decorators/introspection.py` (99 lines)

**Architecture**:
- Pure query functions, no decorators
- Reads `_steps` and `_decorators` attributes set by other decorators

**API**:
| Function | Purpose |
|----------|---------|
| `primitives(cls, field)` | Get Steps list |
| `composites(cls, field)` | Get decorator names |
| `chain(cls, field)` | Human-readable chain |
| `find_decorators(cls, name)` | Search by name |
| `compose(*steps)` | Create anonymous decorator |

## Op Types Used

| Op | Purpose | Files |
|----|---------|-------|
| `Op.TAG` | Store metadata on target | All |
| `Op.HOOK` | Register lifecycle callbacks | lifecycle.py |
| `Op.TRACK` | Enable change tracking | debug_safety.py |
| `Op.REGISTER` | Register in named registry | All |

## Dependencies

- Lifecycle decorators have no dependencies (Tier 7)
- debug_safety depends on lifecycle (Tier 10-11)
- composition depends on make_decorator infrastructure
- stacks depends on composition
- introspection depends on all (read-only queries)

## Key Decisions

1. **Manual vs make_decorator**: debug_safety uses both patterns - manual for fine-grained control, make_decorator for standard cases
2. **Stack validation**: Proactive anti-pattern detection prevents subtle bugs
3. **Introspection as query-only**: No side effects, pure reflection
4. **Tier ordering**: Lifecycle at tier 7 ensures it loads before all dependent decorators
