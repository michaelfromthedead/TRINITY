# PHASE 1 TODO: Core Infrastructure Decorators

## Overview

Validate and document core infrastructure decorators: lifecycle, debug_safety, composition, stacks, introspection.

---

## T1.1: Validate Lifecycle Decorators

**File**: `trinity/decorators/lifecycle.py`

**Tasks**:
- [ ] Verify `@on_add` produces `Op.HOOK` with event="add"
- [ ] Verify `@on_remove` produces `Op.HOOK` with event="remove"
- [ ] Verify `@on_change` produces `Op.HOOK` with event="change"
- [ ] Verify `@on_spawn` produces `Op.HOOK` with event="spawn"
- [ ] Verify `@on_despawn` produces `Op.HOOK` with event="despawn"
- [ ] Verify all set `_lifecycle_hooks` attribute on target
- [ ] Verify registry registration to "lifecycle"

**Acceptance Criteria**:
- All 5 decorators produce correct Op.HOOK steps
- After-apply functions set expected attributes
- Registry contains all lifecycle decorators

---

## T1.2: Validate Debug Safety Decorators (Manual Pattern)

**File**: `trinity/decorators/debug_safety.py`

**Tasks**:
- [ ] Verify `@reads(*fields)` sets `_reads` attribute
- [ ] Verify `@writes(*fields)` sets `_writes` attribute
- [ ] Verify manual decorators call `run_steps` directly
- [ ] Verify Op.TAG with key="reads" and key="writes"

**Acceptance Criteria**:
- `@reads` and `@writes` correctly tag targets
- Manual pattern bypasses make_decorator correctly
- Attributes readable via introspection

---

## T1.3: Validate Debug Safety Decorators (make_decorator Pattern)

**File**: `trinity/decorators/debug_safety.py`

**Tasks**:
- [ ] Verify `@trace_stack` uses make_decorator
- [ ] Verify `@track_changes` uses make_decorator
- [ ] Verify `@track_changes` produces `Op.TRACK` step
- [ ] Verify both have proper validators
- [ ] Verify both register in "debug" registry

**Acceptance Criteria**:
- Both decorators follow 6-part pattern
- Op.TRACK enables change tracking
- Coexists with manual decorators in same file

---

## T1.4: Validate Composition Decorators

**File**: `trinity/decorators/composition.py`

**Tasks**:
- [ ] Verify `@composite` accepts multiple decorators
- [ ] Verify `@composite` applies decorators in order
- [ ] Verify `@alias` creates named reference
- [ ] Verify callable validation on both
- [ ] Verify Op.TAG marks composition metadata

**Acceptance Criteria**:
- `@composite` correctly chains decorators
- `@alias` creates working alias
- Callable validation raises TypeError on non-callables

---

## T1.5: Validate Stack Utilities

**File**: `trinity/decorators/stacks.py`

**Tasks**:
- [ ] Verify `Stack` class stores decorator sequence
- [ ] Verify `Stack.__call__` applies decorators
- [ ] Verify `stack()` function creates inline Stack
- [ ] Verify `parameterized_stack()` supports runtime params
- [ ] Verify `_validate_stack_combination` detects anti-patterns

**Anti-Pattern Tests**:
- [ ] `@parallel` + `@exclusive` raises ValueError
- [ ] `@networked` without `@track_changes` raises UserWarning

**Acceptance Criteria**:
- Stack applies decorators in sequence
- Anti-pattern detection prevents contradictory combinations
- Warnings issued for likely mistakes

---

## T1.6: Validate Introspection API

**File**: `trinity/decorators/introspection.py`

**Tasks**:
- [ ] Verify `primitives(cls, field)` returns Step list
- [ ] Verify `composites(cls, field)` returns decorator names
- [ ] Verify `chain(cls, field)` returns human-readable string
- [ ] Verify `find_decorators(cls, name)` searches by name
- [ ] Verify `compose(*steps)` creates anonymous decorator
- [ ] Verify all functions are pure (no side effects)

**Acceptance Criteria**:
- All 5+ functions return expected data types
- No side effects on target classes
- Works with all decorator patterns (manual and make_decorator)

---

## T1.7: Document Tier Dependencies

**Tasks**:
- [ ] Confirm lifecycle.py is Tier 7
- [ ] Confirm debug_safety.py is Tier 10-11
- [ ] Document dependency chain: lifecycle -> debug_safety -> composition -> stacks -> introspection
- [ ] Verify load order in decorator registry

**Acceptance Criteria**:
- Tier assignments documented
- Dependency chain validated
- No circular dependencies

---

## Summary

| Task | File | Decorators/Functions |
|------|------|---------------------|
| T1.1 | lifecycle.py | 5 decorators |
| T1.2 | debug_safety.py | 2 manual decorators |
| T1.3 | debug_safety.py | 2 make_decorator decorators |
| T1.4 | composition.py | 2 decorators |
| T1.5 | stacks.py | 3 functions |
| T1.6 | introspection.py | 5+ functions |
| T1.7 | N/A | Tier documentation |
