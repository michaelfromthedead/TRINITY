# PHASE 6 ARCHITECTURE: Debug, Security & Platform Systems

## Scope

Debug, security, error handling, time, and platform decorators:
- `debug_extended.py` (212 lines)
- `debug_cheat.py` (191 lines)
- `security.py` (196 lines)
- `error_handling.py` (216 lines)
- `time.py` (224 lines)
- `build_deploy.py` (216 lines)
- `platform_specifics.py` (111 lines)
- `input.py` (159 lines)

## Architecture Pattern: Config Dataclasses for Complex Systems

Debug and security files use frozen dataclasses for complex configurations:

```python
@dataclass(frozen=True)
class NetworkDebugConfig:
    latency_sim: float = 0.0
    packet_loss: float = 0.0
    bandwidth_limit: Optional[int] = None
```

## Component: Debug Extended Decorators

**File**: `trinity/decorators/debug_extended.py` (212 lines)

**Decorators**: `@network_debug`, `@automation_test`

**Architecture**:
- Config dataclasses for debug settings
- NetworkDebugConfig, AutomationTestConfig
- Network latency/packet loss simulation
- Automation test markers

**Config Dataclasses**:
```python
@dataclass(frozen=True)
class NetworkDebugConfig:
    latency_sim: float = 0.0
    packet_loss: float = 0.0
    bandwidth_limit: Optional[int] = None

@dataclass(frozen=True)
class AutomationTestConfig:
    timeout: float = 30.0
    retry_count: int = 0
    tags: tuple[str, ...] = ()
```

## Component: Debug Cheat Decorators

**File**: `trinity/decorators/debug_cheat.py` (191 lines)

**Decorators**: `@cheat`, `@debug_draw`, `@inspector`

**Architecture**:
- Range validation for debug parameters
- Cheat command registration
- Debug visualization
- Inspector widget generation

**Range Validation**:
```python
def _validate_inspector(min_val=None, max_val=None, **_):
    if min_val is not None and max_val is not None and min_val > max_val:
        raise ValueError(
            f"@inspector: min_val ({min_val}) cannot be greater than max_val ({max_val})"
        )
```

## Component: Security Decorators

**File**: `trinity/decorators/security.py` (196 lines)

**Decorators**: `@server_authoritative`, `@validated`, `@rate_limited`, `@encrypted`

**Architecture**:
- Server authority markers
- Input validation enforcement
- Rate limiting with scope validation
- Encryption markers

**Rate Scopes**:
```python
VALID_RATE_SCOPES = frozenset({"global", "per_user", "per_ip", "per_session"})
```

## Component: Error Handling Decorators

**File**: `trinity/decorators/error_handling.py` (216 lines)

**Decorators**: `@crash_safe`, `@recoverable`, `@error_boundary`, `@bug_report`

**Architecture**:
- Recovery strategy validation
- Error boundary configuration
- Crash reporting integration
- Bug report metadata

**Recovery Strategies**:
```python
VALID_RECOVERY_STRATEGIES = frozenset({"retry", "fallback", "ignore", "escalate"})
```

## Component: Time Decorators

**File**: `trinity/decorators/time.py` (224 lines)

**Decorators**: `@time_scale`, `@pausable`, `@rewindable`, `@deterministic`

**Architecture**:
- Time scale configuration
- Pause/resume support
- Rewind mechanics
- Deterministic simulation markers

**Interpolation Enums**:
```python
VALID_INTERPOLATION = frozenset({"none", "linear", "smoothstep", "cubic"})
```

## Component: Build Deploy Decorators

**File**: `trinity/decorators/build_deploy.py` (216 lines)

**Decorators**: `@build_only`, `@strip_in_release`, `@asset_bundle`, `@feature_flag`

**Architecture**:
- Build configuration markers
- Release stripping
- Asset bundling
- Feature flag integration

**Pattern**:
```python
def _build_only_steps(params):
    return [
        Step(Op.TAG, {"key": "build_only", "value": True}),
        Step(Op.TAG, {"key": "build_configurations", "value": params.get("configs", {"debug"})}),
        Step(Op.REGISTER, {"registry": "build"}),
    ]
```

## Component: Platform Specifics Decorators

**File**: `trinity/decorators/platform_specifics.py` (111 lines)

**Decorators**: `@battery_aware`

**Architecture**:
- Battery mode validation
- Platform-specific behavior

**Battery Modes**:
```python
VALID_BATTERY_MODES = frozenset({"normal", "low_power", "critical", "charging"})
```

## Component: Input Decorators

**File**: `trinity/decorators/input.py` (159 lines)

**Decorators**: `@input_action`, `@input_axis`

**Architecture**:
- Input binding validation
- Action and axis mapping

**Binding Validation**:
```python
def _validate_input_action(bindings=[], **_):
    if not bindings:
        raise ValueError("@input_action: at least one binding is required")
```

## Op Types Used

| Op | Purpose | Files |
|----|---------|-------|
| `Op.TAG` | Store debug/security metadata | All |
| `Op.REGISTER` | Register in "debug", "security", "build" | All |
| `Op.VALIDATE` | Runtime validation | security.py |
| `Op.DESCRIBE` | Documentation generation | debug_cheat.py |

## Key Decisions

1. **Config dataclasses**: Complex debug settings use frozen dataclasses
2. **Range validation**: Debug parameters validate min/max ranges
3. **Rate scope validation**: Security decorators validate rate limiting scope
4. **Recovery strategies**: Error handling has explicit recovery options
5. **Build stripping**: build_deploy decorators enable release optimization
