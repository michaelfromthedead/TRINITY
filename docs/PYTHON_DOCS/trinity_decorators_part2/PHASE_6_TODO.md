# PHASE 6 TODO: Debug, Security & Platform Systems

## Overview

Validate debug, security, error handling, time, build/deploy, platform, and input decorators.

---

## T6.1: Validate Debug Extended Decorators

**File**: `trinity/decorators/debug_extended.py`

**Tasks**:
- [ ] Verify `@network_debug` creates NetworkDebugConfig
- [ ] Verify NetworkDebugConfig fields: latency_sim, packet_loss, bandwidth_limit
- [ ] Verify `@automation_test` creates AutomationTestConfig
- [ ] Verify AutomationTestConfig fields: timeout, retry_count, tags
- [ ] Verify both configs are frozen dataclasses

**Acceptance Criteria**:
- Both decorators create config dataclasses
- Configs are frozen and hashable
- Both register in "debug" registry

---

## T6.2: Validate Debug Cheat Decorators

**File**: `trinity/decorators/debug_cheat.py`

**Tasks**:
- [ ] Verify `@cheat` registers cheat command
- [ ] Verify `@debug_draw` enables debug visualization
- [ ] Verify `@inspector` validates min/max range
- [ ] Verify range validation: min_val <= max_val

**Range Validation Test**:
```python
@inspector(min_val=10, max_val=5)  # Should raise:
# ValueError: @inspector: min_val (10) cannot be greater than max_val (5)
```

**Acceptance Criteria**:
- All 3 decorators follow 6-part pattern
- Range validation catches invalid min/max
- All register in "debug" registry

---

## T6.3: Validate Security Decorators

**File**: `trinity/decorators/security.py`

**Tasks**:
- [ ] Verify `@server_authoritative` marks server authority
- [ ] Verify `@validated` enforces input validation
- [ ] Verify `@rate_limited` validates scope against VALID_RATE_SCOPES
- [ ] Verify `@encrypted` marks encrypted data

**Rate Scopes**:
- [ ] "global" - global rate limit
- [ ] "per_user" - per user limit
- [ ] "per_ip" - per IP limit
- [ ] "per_session" - per session limit

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Rate scope validation produces actionable error
- All register in "security" registry

---

## T6.4: Validate Error Handling Decorators

**File**: `trinity/decorators/error_handling.py`

**Tasks**:
- [ ] Verify `@crash_safe` enables crash recovery
- [ ] Verify `@recoverable` validates strategy against VALID_RECOVERY_STRATEGIES
- [ ] Verify `@error_boundary` sets error boundary
- [ ] Verify `@bug_report` configures bug reporting

**Recovery Strategies**:
- [ ] "retry" - retry operation
- [ ] "fallback" - use fallback value
- [ ] "ignore" - ignore error
- [ ] "escalate" - escalate to parent

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Recovery strategy validation produces actionable error
- All register in "error" registry

---

## T6.5: Validate Time Decorators

**File**: `trinity/decorators/time.py`

**Tasks**:
- [ ] Verify `@time_scale` configures time scaling
- [ ] Verify `@pausable` enables pause/resume
- [ ] Verify `@rewindable` validates interpolation
- [ ] Verify `@deterministic` marks deterministic simulation

**Interpolation Types**:
- [ ] "none" - no interpolation
- [ ] "linear" - linear interpolation
- [ ] "smoothstep" - smooth interpolation
- [ ] "cubic" - cubic interpolation

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Interpolation validation produces actionable error
- All register in "time" registry

---

## T6.6: Validate Build Deploy Decorators

**File**: `trinity/decorators/build_deploy.py`

**Tasks**:
- [ ] Verify `@build_only` marks build-time only code
- [ ] Verify `@build_only` accepts configurations set
- [ ] Verify `@strip_in_release` marks for release stripping
- [ ] Verify `@asset_bundle` configures asset bundling
- [ ] Verify `@feature_flag` integrates feature flags

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Build configurations properly stored
- All register in "build" registry

---

## T6.7: Validate Platform Specifics Decorator

**File**: `trinity/decorators/platform_specifics.py`

**Tasks**:
- [ ] Verify `@battery_aware` validates mode against VALID_BATTERY_MODES
- [ ] Verify battery mode handling

**Battery Modes**:
- [ ] "normal" - normal operation
- [ ] "low_power" - low power mode
- [ ] "critical" - critical battery
- [ ] "charging" - charging mode

**Acceptance Criteria**:
- Battery mode validation produces actionable error
- Registers in "platform" registry

---

## T6.8: Validate Input Decorators

**File**: `trinity/decorators/input.py`

**Tasks**:
- [ ] Verify `@input_action` requires at least one binding
- [ ] Verify `@input_axis` configures axis mapping
- [ ] Verify binding validation

**Binding Validation Test**:
```python
@input_action(bindings=[])  # Should raise:
# ValueError: @input_action: at least one binding is required
```

**Acceptance Criteria**:
- Both decorators follow 6-part pattern
- Binding validation enforced
- All register in "input" registry

---

## Summary

| Task | File | Decorators | Lines |
|------|------|------------|-------|
| T6.1 | debug_extended.py | 2 | 212 |
| T6.2 | debug_cheat.py | 3 | 191 |
| T6.3 | security.py | 4 | 196 |
| T6.4 | error_handling.py | 4 | 216 |
| T6.5 | time.py | 4 | 224 |
| T6.6 | build_deploy.py | 4 | 216 |
| T6.7 | platform_specifics.py | 1 | 111 |
| T6.8 | input.py | 2 | 159 |

**Total**: 24 decorators, 1,525 lines
