# PHASE 3 TODO: Development Tools

## Objective

Verify and validate the development tools implementation.

---

## T-3.1: step_trace

**File**: `trinity/tools/step_trace.py`

### Tasks
- [ ] Test trace output for class with decorator steps
- [ ] Test trace output for class with descriptor steps
- [ ] Test trace output for class with metaclass steps
- [ ] Test trace output for class with all three layers
- [ ] Verify layer grouping is correct (Decorator, Descriptor, Metaclass)
- [ ] Test trace for class with no steps
- [ ] Verify step details are correctly formatted
- [ ] Test chain traversal for composed descriptors

### Acceptance Criteria
- All steps displayed grouped by layer
- Step details (Op type, parameters) accurately shown
- Descriptor chain steps include field name prefix

---

## T-3.2: lint

**File**: `trinity/tools/lint.py`

### Tasks
- [ ] Test `install_lint_hook` activates validation
- [ ] Test validation passes for compliant class
- [ ] Test validation fails for class with descriptor exclusion violation
- [ ] Test validation fails for class with accepts_inner violation
- [ ] Test validation fails for class with accepts_outer violation
- [ ] Verify warning messages are descriptive
- [ ] Test `uninstall_lint_hook` deactivates validation
- [ ] Test re-installation after uninstall

### Acceptance Criteria
- Hook catches violations at import time
- Warnings clearly identify the problem
- Hook can be cleanly enabled/disabled

---

## T-3.3: op_coverage

**File**: `trinity/tools/op_coverage.py`

### Tasks
- [ ] Test Op counting across multiple classes
- [ ] Verify coverage map structure (Op -> [class names])
- [ ] Test tracking of zero-step classes
- [ ] Test coverage with no registered classes
- [ ] Test coverage with classes using same Op multiple times
- [ ] Verify all Op types from trinity.constants are tracked

### Acceptance Criteria
- Coverage map accurately reflects Op usage
- Zero-step classes identified
- Multiple uses of same Op counted correctly

---

## T-3.4: doctor

**File**: `trinity/tools/doctor.py`

### Tasks
- [ ] Test health check with all valid classes
- [ ] Test health check with invalid class (missing _component_id)
- [ ] Test health check with invalid class (composition error)
- [ ] Verify pass count is accurate
- [ ] Verify fail count is accurate
- [ ] Test error messages are per-class
- [ ] Test with empty registry
- [ ] Test error collection for multiple issues on one class

### Acceptance Criteria
- Pass/fail counts match reality
- Error messages identify specific problems
- All registered classes checked

---

## T-3.5: Tool Integration

### Tasks
- [ ] Test step_trace after doctor identifies failure
- [ ] Test lint output matches doctor findings
- [ ] Test op_coverage includes classes found by doctor
- [ ] Verify tools work with cleared registries
- [ ] Test tools on dynamically created classes

### Acceptance Criteria
- Tools provide complementary information
- Tools work correctly in test isolation (cleared registries)
- Dynamic class creation supported

---

## T-3.6: Tool Documentation

### Tasks
- [ ] Verify docstrings present on all public functions
- [ ] Test example usage from docstrings
- [ ] Verify return type hints accurate

### Acceptance Criteria
- Public API fully documented
- Examples work as shown
