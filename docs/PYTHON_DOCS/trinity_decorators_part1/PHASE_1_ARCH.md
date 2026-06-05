# PHASE 1 ARCHITECTURE: Core Decorator Infrastructure

## Phase Scope

Foundation layer: `ops.py`, `registry.py`, `base.py`, `__init__.py`

## Architecture Decisions

### ADR-DEC-001: 7 Primitive Operations

**Context**: Need a minimal set of primitives that compose into all decorator behaviors.

**Decision**: Implement exactly 7 Ops as an Enum:
- TAG - Attach queryable metadata
- HOOK - Wire lifecycle callbacks
- REGISTER - Add to named registry
- DESCRIBE - Extract schema from annotations
- TRACK - Enable change monitoring
- VALIDATE - Enforce constraints
- INTERCEPT - Wrap field access

**Consequences**:
- All decorators reduce to Op combinations
- New decorator behaviors require new Ops (rare)
- Step execution order is deterministic

### ADR-DEC-002: make_decorator() Factory

**Context**: Need uniform decorator construction.

**Decision**: Single factory function with signature:
```python
def make_decorator(
    name: str,
    steps: Union[list[Step], Callable[..., list[Step]]],
    doc: str = "",
    validate: Optional[Callable[..., None]] = None,
    after_steps: Optional[Callable[[Any, dict[str, Any]], Any]] = None,
)
```

**Consequences**:
- Deferred step generation (callable) supports dynamic decorators
- `after_steps` hook enables post-processing (flyweight registry setup, etc.)
- Validation runs before step execution

### ADR-DEC-003: 54-Tier Registry

**Context**: Decorators have dependencies; load order matters.

**Decision**: IntEnum with 54 tiers from COMPILATION (0) to BRIDGES_CACHING (53).

**Consequences**:
- Lower tiers always available when higher tiers register
- Circular dependency is a registration-time error
- Tier gaps allow future interpolation

### ADR-DEC-004: Thread-Safe Singleton Registry

**Context**: Multi-threaded engine init, hot-reload scenarios.

**Decision**: Module-level registry protected by `threading.RLock()`.

**Consequences**:
- All registration atomic
- Queries during registration are safe
- No locking needed for read-only access after init

## Component Diagram

```
+-------------+
|   ops.py    |  7 Ops enum, Step dataclass, make_decorator()
+------+------+
       |
       v
+------+------+
| registry.py |  54-tier IntEnum, thread-safe registry singleton
+------+------+
       |
       v
+------+------+
|   base.py   |  Decorator tracking, attribute attachment, validation
+------+------+
       |
       v
+------+------+
| __init__.py |  Re-exports ~150 symbols from all modules
+-------------+
```

## Data Structures

### Step
```python
@dataclass
class Step:
    op: Op
    params: dict[str, Any]
```

### Registry Entry
```python
@dataclass
class RegistryEntry:
    name: str
    tier: Tier
    target: Any
    metadata: dict[str, Any]
```

## Thread Safety Guarantees

| Operation | Lock Required | Notes |
|-----------|---------------|-------|
| Register decorator | RLock | Atomic write |
| Query tier | None | Immutable after init |
| Iterate registry | None | Snapshot iteration |
| Hot-reload | RLock | Full reregistration |

## Validation Patterns

All parameter validation uses the pattern:
```python
def _validate_<decorator>(**kwargs: Any) -> None:
    if <invalid_condition>:
        raise ValueError(f"@{decorator}: {descriptive_message}")
```
