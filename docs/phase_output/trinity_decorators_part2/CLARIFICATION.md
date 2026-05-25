# CLARIFICATION: Trinity Decorators Part 2 - Design Philosophy

## Philosophical Framing

### Ops-Based Architecture as Declarative Intent

The decorator framework embodies a declarative philosophy: decorators express *what* should happen, not *how*. The `Step(Op.X, params)` pattern separates intent from execution:

- **TAG** - "This target has this property"
- **REGISTER** - "This target belongs to this registry"
- **HOOK** - "This lifecycle event should trigger this action"
- **TRACK** - "Changes to this target should be observed"
- **VALIDATE** - "This constraint must hold at runtime"
- **DESCRIBE** - "This documentation should be generated"

This separation enables:
1. Static analysis of decorator effects
2. Composition without execution ordering conflicts
3. Tooling support (IDE hints, linters, generators)

### Validation as First-Class Citizen

Every decorator implements validation *before* step generation. This is not defensive programming but a design principle: **invalid configurations should fail fast with actionable messages**.

```python
# Pattern: Specific, actionable error messages
if stacking not in VALID_STACKING:
    raise ValueError(
        f"@buff: invalid stacking '{stacking}'. "
        f"Valid stacking modes: {sorted(VALID_STACKING)}"
    )
```

The error message pattern:
1. Name the decorator (`@buff:`)
2. State what's wrong (`invalid stacking 'X'`)
3. Provide valid options (`Valid stacking modes: [...]`)

### make_decorator as Factory Pattern

The `make_decorator` function enforces structural consistency while allowing behavioral variation. Every decorator follows the same construction:

```python
decorator = make_decorator(
    name="...",           # Identity
    steps=...,            # Step builder function
    validate=...,         # Parameter validator
    after_steps=...,      # Post-apply hook
    doc="...",            # Documentation
)
```

This pattern ensures:
- All decorators have consistent introspection attributes
- Registry can enumerate all decorators uniformly
- Tooling knows exact decorator structure

## Design Rationale

### Why Steps Instead of Direct Mutation?

Direct attribute mutation creates ordering dependencies:
```python
# Problem: Which decorator ran first matters
target._networked = True
target._tracked = True  # Might need _networked to be set
```

Steps create an intermediate representation:
```python
# Solution: Steps are order-independent data
[Step(Op.TAG, {"key": "networked"}), Step(Op.TAG, {"key": "tracked"})]
```

The step executor can:
1. Analyze dependencies
2. Reorder if needed
3. Detect conflicts early

### Why Frozen Valid Constants?

```python
VALID_STACKING = frozenset({"replace", "stack", "refresh", "unique"})
```

Frozen sets are:
1. Immutable (cannot be accidentally modified)
2. Hashable (can be used in validation caches)
3. O(1) membership tests
4. Self-documenting (all valid options visible at module level)

### Why Config Dataclasses?

Several files use frozen dataclasses for complex configurations:

```python
@dataclass(frozen=True)
class InterestConfig:
    radius: float
    priority: int
    relevance_curve: str
```

Benefits:
1. Type safety (IDE autocomplete, type checkers)
2. Immutability (cannot mutate after creation)
3. Equality (two configs with same values are equal)
4. Hashing (can be dict keys or set members)

### Stack Validation Philosophy

The `stacks.py` module implements **proactive anti-pattern detection**:

- **Hard errors**: Contradictory combinations (`@parallel` + `@exclusive`)
- **Warnings**: Likely mistakes (`@networked` without `@track_changes`)

This philosophy: better to fail at decoration time than debug subtle runtime bugs.

### Introspection API Design

The `introspection.py` module provides query primitives:

| Function | Purpose |
|----------|---------|
| `primitives(cls, field)` | Get raw Steps |
| `composites(cls, field)` | Get decorator names |
| `chain(cls, field)` | Human-readable chain |
| `find_decorators(cls, name)` | Search by name |
| `compose(*steps)` | Create anonymous decorator |

Design principle: **decorators should be queryable after application**. This enables:
1. Runtime reflection ("what decorators are on this class?")
2. Validation ("does this class have required decorators?")
3. Documentation generation

## Tier Distribution Rationale

The tier system (7-51) reflects decorator dependencies:

- **Low tiers (7-11)**: Core infrastructure (lifecycle, debug_safety, assets)
- **Mid tiers (32-40)**: Platform and system decorators
- **High tiers (41-51)**: Game-specific systems (depends on lower tiers)

This ensures decorators are loaded in dependency order.

## Special Cases Explained

### debug_safety.py Mixed Pattern

This file uses both patterns:
- **Manual**: `@reads`, `@writes` (need fine-grained control)
- **make_decorator**: `@trace_stack`, `@track_changes` (standard pattern)

The manual pattern allows calling `run_steps` directly with custom step lists.

### stacks.py as Composition Layer

Not a decorator file but a **meta-layer** for combining decorators:
- `Stack` class holds decorator sequence
- `stack()` function creates inline stacks
- Validation catches anti-patterns before decoration

### introspection.py as Query Layer

Pure query functions, no decorators defined. Provides the reflection API that other tools use to analyze decorated code.
