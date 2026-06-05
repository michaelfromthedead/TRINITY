# PHASE 3 ARCHITECTURE: Development Tools

## Phase Scope

The tools directory (`trinity/tools/`) providing development utilities for introspection and validation.

## Components

### 3.1 step_trace.py (74 lines)

Introspection tool showing all Steps on a class grouped by layer.

**Layers**:
1. Decorator - Steps from `@` decorators
2. Descriptor - Steps from field descriptors
3. Metaclass - Steps from metaclass processing

**Output Format**:
```
=== Step Trace: PlayerHealth ===
[Decorator] (2 steps)
  TRACK(field='health')
  VALIDATE(constraint='positive')
[Descriptor] (3 steps)
  health.tracked: TRACK(field='health')
  health.range: VALIDATE(constraint='range', min=0, max=100)
[Metaclass] (4 steps)
  TAG(key='component_id', value=42)
  REGISTER(registry='component_registry')
```

### 3.2 lint.py (73 lines)

Validation hook for composition rules.

**Features**:
- Validates classes against composition rules
- Import-time hook via `install_lint_hook`
- Warns on validation errors
- Clean uninstall via `uninstall_lint_hook`

**Usage**:
```python
from trinity.tools import install_lint_hook

# Enable validation at import time
install_lint_hook()

# Later, disable if needed
uninstall_lint_hook()
```

### 3.3 op_coverage.py (52 lines)

Coverage analysis for Op usage.

**Features**:
- Counts Op usage across all registered classes
- Tracks classes with zero steps
- Builds coverage map (Op -> classes using it)

**Output**:
```python
{
    Op.TRACK: ['PlayerHealth', 'Enemy', 'Projectile'],
    Op.VALIDATE: ['PlayerHealth', 'Projectile'],
    Op.REGISTER: ['PlayerHealth', 'Enemy', 'Projectile'],
    # ...
}
```

### 3.4 doctor.py (41 lines)

Health check for Trinity classes.

**Features**:
- Validates all registered Trinity classes
- Returns pass/fail counts
- Collects per-class error messages

**Output**:
```python
{
    'passed': 42,
    'failed': 3,
    'errors': {
        'BrokenComponent': ['Missing _component_id'],
        # ...
    }
}
```

## Architecture Decisions

### AD-3.1: Layer-Based Step Organization

step_trace groups steps by source layer (Decorator, Descriptor, Metaclass).

Rationale: When debugging, developers need to know WHERE a step came from, not just WHAT it is. Layer grouping provides this context.

### AD-3.2: Import-Time Hook

lint uses an import hook rather than runtime validation.

Rationale: Catching errors at import time (when classes are defined) is earlier than catching them at instantiation time. Earlier detection means faster feedback.

### AD-3.3: Coverage Map Inversion

op_coverage builds Op -> [classes] rather than class -> [Ops].

Rationale: The common question is "which classes use this Op?" not "which Ops does this class use?" The inverted map answers the common question directly.

### AD-3.4: Minimal Dependencies

Tools depend only on:
- Standard library
- Trinity's Step/Op types
- Trinity's registries

Rationale: Tools should be usable even when other parts of Trinity are broken. Minimal dependencies ensure tools remain functional for debugging.

## Data Sources

Tools introspect these class attributes:

| Attribute | Set By | Content |
|-----------|--------|---------|
| `_metaclass_steps` | EngineMeta | List of Step objects from metaclass |
| `_decorator_steps` | Decorators | List of Step objects from decorators |
| `_field_descriptors` | ComponentMeta | Dict of field name -> descriptor |

Each descriptor also has `_steps` containing its Step objects.

## Integration Points

### With Metaclasses

Tools read `_metaclass_steps` set by metaclasses:
- step_trace displays them
- lint validates them
- op_coverage counts them
- doctor checks them

### With Descriptors

Tools read `_field_descriptors` and descriptor `_steps`:
- step_trace shows per-field steps
- lint validates composition rules
- op_coverage counts descriptor ops

### With Decorators

Tools read `_decorator_steps`:
- step_trace shows decorator actions
- lint validates decorator usage
- op_coverage counts decorator ops

## Tool Composition

Tools can be combined:

```python
# Full health check with detailed trace on failures
from trinity.tools import doctor, step_trace

results = doctor.check_all()
for cls_name, errors in results['errors'].items():
    cls = get_class_by_name(cls_name)
    print(step_trace(cls))
    for error in errors:
        print(f"  ERROR: {error}")
```
