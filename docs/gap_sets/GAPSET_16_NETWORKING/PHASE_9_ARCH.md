# Phase 9 Architecture -- Anti-Cheat & Security

> **Cluster**: GAPSET_16_NETWORKING
> **Module**: `engine/networking/security/`

---

## Overview

The security module provides a comprehensive anti-cheat and security pipeline with field-level authority validation, input sanity checking, rate limiting, statistical anomaly detection, and a configurable escalation response system.

---

## File Map

| File | LOC | Role |
|------|-----|------|
| `authority_validator.py` | 388 | Field-level write permissions with thread-safe RLock |
| `input_validator.py` | 524 | Player state validation: speed, teleport, rotation, sequence |
| `rate_limiter.py` | 478 | Token bucket with adaptive rate limiting |
| `anomaly_detector.py` | 687 | 10 anomaly types, 4 severity levels, statistical + heuristic |
| `response.py` | 597 | 5-tier escalation, HWID/IP/shadow/manual bans |
| `config.py` | 208 | 7 frozen dataclass configs |
| `__init__.py` | ~60 | Comprehensive exports |

---

## Architecture

### Security Pipeline

```
[Incoming Action]
    |
    v
AuthorityValidator -- field-level write permission check
    |
    v
InputValidator -- sanity bounds, speed/teleport detection
    |
    v
RateLimiter -- per-action token bucket, sliding window
    |
    v
AnomalyDetector -- statistical + heuristic threat detection
    |
    v
ResponseManager -- severity-based escalation
    |
    v
[Allow / Warn / Kick / Ban]
```

### AuthorityValidator (authority_validator.py)

**Field-Level Authority**: Grants/denies write access at the individual field level. Uses 4-tuple: (Caller, Entity, Authority, Field).

**Thread Safety**: Uses `threading.RLock` around all write and validate operations.

**Key Classes**:
- `Caller`: Represents the requesting client/user
- `Entity`: The game entity being accessed
- `Authority`: Enum of authority levels (OWNER, SERVER, SIMULATED, etc.)
- `EntityAuthority`: Entity-level authority container
- `FieldAuthority`: Per-field permission rules

### InputValidator (input_validator.py)

**ValidationResult** (8 result types): VALID, SPEED_HACK, TELEPORT, FLY_HACK, INVALID_ROTATION, INVALID_SEQUENCE, RATE_EXCEEDED, INVALID_ACTION.

**Detection Capabilities**:
- Speed hack: Distance/time > max speed
- Teleport: Distance between updates > max distance
- Fly hack: Position.Y changes without ground contact
- Rotation clamping: Invalid Euler angles
- Sequence gaps: Missing sequence numbers
- Action rate: Per-action type frequency limits

**PlayerState**: Tracks per-player position, velocity, rotation, sequence, and action timestamps for validation.

### RateLimiter (rate_limiter.py)

**TokenBucket**: Classic token bucket per client/per action. Configurable:
- `capacity`: Max accumulated tokens
- `fill_rate`: Tokens per second
- `initial`: Starting tokens

**RateLimiter**: Wraps token bucket with action-type awareness. Supports multiple cost values per action type.

**AdaptiveRateLimiter**: Adjusts rate limits based on server load:
- Detects overload conditions
- Reduces limits proportionally
- Recovers when load normalizes
- Configurable min/max range

**RateLimitResult**: Contains allowed/rejected status, current tokens, wait time, and adaptive level info.

### AnomalyDetector (anomaly_detector.py)

**10 AnomalyType Values**:
| Type | Detection Method |
|------|-----------------|
| SPEED_HACK | Speed exceeding threshold with configurable margin |
| TELEPORT | Position delta exceeding max |
| FLY_HACK | Y position changes without support |
| AIMBOT | View angle snap-to-target pattern |
| NO_SPREAD | Weapon spread pattern deviation |
| NO_RECOIL | Recoil pattern deviation |
| RAPID_FIRE | Fire rate exceeding weapon cap |
| INPUT_BURST | Input frequency burst detection |
| TIMING_ANOMALY | Unusual packet timing patterns |
| CUSTOM | Custom detector function |

**4 Severity Levels**: LOW, MEDIUM, HIGH, CRITICAL.

**Detection Methods**:
- **Statistical**: Moving average, standard deviation, z-score
- **Heuristic**: Rule-based threshold checks
- **Custom**: User-provided detector functions

**PlayerStats**: Per-player rolling statistics for anomaly scoring.

**AnomalyReport**: Contains player ID, anomaly type, severity, confidence (0-1), evidence data, and timestamp.

### Response Manager (response.py)

**ResponseSeverity** (5 tiers):
| Severity | Action |
|----------|--------|
| WARN | Log warning, notify observers |
| KICK | Disconnect player |
| BAN_1H | 1-hour temp ban |
| BAN_24H | 24-hour temp ban |
| BAN_PERMANENT | Permanent ban with optional shadow ban |

**EscalationRule**: Maps (anomaly_type, severity, repeat_count) -> ResponseSeverity. Configurable from config.

**BanRecord**: Contains player ID, HWID hash, IP, severity, reason, timestamp, expiry, and overlapping ban chain.

**Ban Types**:
- HWID ban: Hardware identifier hash
- IP ban: Address-based
- Shadow ban: Player sees normal world but is isolated
- Manual ban: Operator-triggered

### Config (config.py)

7 frozen dataclass configs:
- `AuthorityConfig`: Permission rules
- `InputValidationConfig`: Bounds and thresholds
- `RateLimitConfig`: Token bucket parameters
- `AnomalyConfig`: Detection thresholds per type
- `ResponseConfig`: Escalation rules and ban durations
- `AntiCheatConfig`: Master config containing all above
- `SecurityConfig`: Global security settings

---

## Test Coverage

**`tests/test_networking/test_security.py`** (920 lines, 16 test classes):
- Unit tests for individual components
- Adversarial tests for bypass attempts
- Thread-safety tests (concurrent access)
- Integration tests for full pipeline
- Bypass tests for common cheat vectors

This is the **only test file** in the entire networking layer.

---

## Reality Status

- AuthorityValidator (field-level, thread-safe): **[x]** Complete
- InputValidator (speed/teleport/rotation/sequence): **[x]** Complete
- RateLimiter (token bucket, adaptive): **[x]** Complete
- AnomalyDetector (10 types, 4 severities): **[x]** Complete
- ResponseManager (5-tier escalation, HWID/IP/shadow): **[x]** Complete
- Config (7 frozen dataclass configs): **[x]** Complete
- Tests (920 lines, 16 classes): **[x]** Complete

---

*End of PHASE_9_ARCH.md*
