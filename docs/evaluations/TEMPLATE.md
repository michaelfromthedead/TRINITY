# Evaluation: <MODULE_NAME>

**Directory:** `<path/to/module>/`
**Files:** <N>
**Lines of Code:** <N>
**Evaluator:** <agent-type or human>
**Date:** <YYYY-MM-DD>

---

## Summary

<2-3 sentences: overall health assessment, main concerns, production readiness>

---

## Completeness

**Status:** <COMPLETE | MOSTLY_COMPLETE | PARTIAL | STUB>

### Stubs / NotImplementedError
| File | Line | Description |
|------|------|-------------|
| `example.py` | 42 | `calculate_physics()` raises NotImplementedError |

### TODO/FIXME Comments
| File | Line | Comment |
|------|------|---------|
| `example.py` | 15 | `# TODO: implement caching` |

### Empty / Pass-only Functions
| File | Function | Notes |
|------|----------|-------|
| `example.py` | `cleanup()` | Empty body, may be intentional |

---

## Correctness Concerns

### Logic Issues
| Severity | File | Line | Issue |
|----------|------|------|-------|
| HIGH | `example.py` | 88 | Off-by-one in loop bounds |
| MEDIUM | `other.py` | 22 | Potential division by zero |

### Type Issues
| File | Line | Issue |
|------|------|-------|
| `example.py` | 50 | Returns `None` but type hint says `int` |

### Dead Code
| File | Line | Description |
|------|------|-------------|
| `old.py` | 1-200 | Entire file appears unused |

---

## Architecture

### Dependencies (this module imports)
```
foundation.serializer
trinity.base
engine.core.constants
```

### Dependents (other modules import this)
```
engine.gameplay.combat
engine.tooling.editor
```

### Layering Assessment
- [ ] Clean layering (no circular deps, correct direction)
- [ ] Minor violations: <describe>
- [ ] Major violations: <describe>

### Design Patterns Used
- <Pattern>: <where and how>

### Coupling Assessment
- **Tight coupling:** <list tightly coupled files>
- **Loose coupling:** <well-isolated components>

---

## Test Coverage

**Test Directory:** `tests/<corresponding>/`
**Test Files:** <N>
**Estimated Coverage:** <HIGH | MEDIUM | LOW | NONE>

### Missing Test Coverage
| File/Function | Why It Matters |
|---------------|----------------|
| `physics_solver.py:step()` | Core simulation, needs regression tests |

### Test Quality Notes
- <observation about test quality>

---

## Integration Points

### With trinity/
- <how this module uses trinity patterns>

### With foundation/
- <how this module uses foundation systems>

### With Rust Backend
- <FFI calls, data structures shared, or "no direct integration">

---

## Recommendations

### Critical (blocks production)
1. <actionable item>

### Important (should fix)
1. <actionable item>

### Nice-to-have
1. <actionable item>

---

## File Inventory

| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 45 | COMPLETE | Exports correct |
| `core.py` | 320 | COMPLETE | Main implementation |
| `helpers.py` | 150 | PARTIAL | 2 stubs |
| `experimental.py` | 80 | STUB | Not ready |

**Legend:**
- COMPLETE: Fully implemented, no stubs
- PARTIAL: Mostly implemented, some gaps
- STUB: Placeholder only
- EMPTY: No meaningful code

---

## Raw Metrics

```
Total files: <N>
Total lines: <N>
Blank lines: <N>
Comment lines: <N>
Code lines: <N>
Functions: <N>
Classes: <N>
```

---

*Evaluation complete. See docs/PYTHON_EVALUATION_TODO.md for next steps.*
