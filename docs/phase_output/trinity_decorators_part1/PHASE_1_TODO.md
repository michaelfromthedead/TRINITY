# PHASE 1 TODO: Core Decorator Infrastructure

## Summary
Implement foundation layer: Ops enum, make_decorator factory, tier registry, base utilities.

---

## T-DEC-1.1: Implement 7-Op Enum

**File**: `trinity/decorators/ops.py`

**Task**: Define Op enum with exactly 7 values.

**Acceptance Criteria**:
- [ ] `Op.TAG` - Attach queryable metadata
- [ ] `Op.HOOK` - Wire lifecycle callbacks
- [ ] `Op.REGISTER` - Add to named registry
- [ ] `Op.DESCRIBE` - Extract schema from annotations
- [ ] `Op.TRACK` - Enable change monitoring
- [ ] `Op.VALIDATE` - Enforce constraints
- [ ] `Op.INTERCEPT` - Wrap field access
- [ ] Enum inherits from `str, Enum` for serialization

---

## T-DEC-1.2: Implement Step Dataclass

**File**: `trinity/decorators/ops.py`

**Task**: Define Step dataclass for operation sequencing.

**Acceptance Criteria**:
- [ ] Fields: `op: Op`, `params: dict[str, Any]`
- [ ] Frozen dataclass (immutable)
- [ ] `__slots__` for memory efficiency

---

## T-DEC-1.3: Implement make_decorator Factory

**File**: `trinity/decorators/ops.py`

**Task**: Factory function that constructs decorators from Steps.

**Acceptance Criteria**:
- [ ] Signature matches: `make_decorator(name, steps, doc, validate, after_steps)`
- [ ] `steps` can be list or callable (deferred generation)
- [ ] `validate` runs before step execution
- [ ] `after_steps` runs after all steps complete
- [ ] Returns callable decorator
- [ ] Preserves `__name__`, `__doc__` via `functools.wraps`

---

## T-DEC-1.4: Implement 54-Tier IntEnum

**File**: `trinity/decorators/registry.py`

**Task**: Define Tier enum with 54 values.

**Acceptance Criteria**:
- [ ] Tier 0: COMPILATION
- [ ] Tier 1: ECS_CORE
- [ ] Tier 2: MEMORY
- [ ] Tier 3: SCHEDULING
- [ ] Tier 4: DATA_FLOW
- [ ] Tier 5: GPU
- [ ] Tier 6: DEV
- [ ] ... through Tier 53: BRIDGES_CACHING
- [ ] All tiers from investigation documented

---

## T-DEC-1.5: Implement Thread-Safe Registry

**File**: `trinity/decorators/registry.py`

**Task**: Singleton registry with RLock protection.

**Acceptance Criteria**:
- [ ] Module-level `_registry: dict[int, RegistryEntry]`
- [ ] Module-level `_lock: threading.RLock`
- [ ] `register(target, tier, metadata)` - atomic insert
- [ ] `get(tier)` - returns entries for tier
- [ ] `all()` - returns all entries
- [ ] Thread-safe iteration (snapshot copy)

---

## T-DEC-1.6: Implement Base Utilities

**File**: `trinity/decorators/base.py`

**Task**: Decorator tracking and attribute attachment.

**Acceptance Criteria**:
- [ ] `attach_metadata(target, key, value)` - sets `target._<key> = value`
- [ ] `get_metadata(target, key, default=None)` - gets `target._<key>`
- [ ] `has_decorator(target, decorator_name)` - checks if decorated
- [ ] `list_decorators(target)` - returns applied decorator names
- [ ] Validation utilities for common patterns

---

## T-DEC-1.7: Implement Re-Export Init

**File**: `trinity/decorators/__init__.py`

**Task**: Re-export all public symbols.

**Acceptance Criteria**:
- [ ] Imports from ops, registry, base
- [ ] Imports from all domain modules (gpu, memory, scheduling, etc.)
- [ ] `__all__` list with ~150 symbols
- [ ] No circular import errors
- [ ] Lazy imports for heavy modules if needed

---

## Dependencies

```
T-DEC-1.1 ─┬─> T-DEC-1.2 ─┬─> T-DEC-1.3
           │              │
           v              v
T-DEC-1.4 ────────────> T-DEC-1.5 ──> T-DEC-1.6 ──> T-DEC-1.7
```

## Estimated Effort

| Task | Lines | Complexity |
|------|-------|------------|
| T-DEC-1.1 | ~20 | Low |
| T-DEC-1.2 | ~15 | Low |
| T-DEC-1.3 | ~80 | Medium |
| T-DEC-1.4 | ~60 | Low |
| T-DEC-1.5 | ~100 | Medium |
| T-DEC-1.6 | ~80 | Low |
| T-DEC-1.7 | ~50 | Low |
