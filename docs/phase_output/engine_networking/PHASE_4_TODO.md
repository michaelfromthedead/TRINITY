# PHASE 4 TODO: RPC and Security Systems

## Overview

Phase 4 implements the RPC system and comprehensive security layer. All tasks assume the existing implementation is production-ready; these TODOs focus on testing, verification, and identified gaps.

---

## 1. RPC Manager Tasks

### 1.1 Unit Tests: RPC Registration

**File**: `tests/blackbox_rpc_manager.py`

**Acceptance Criteria**:
- [ ] `register_rpc()` adds function to registry
- [ ] `get_rpc_info()` returns RPCInfo for registered RPC
- [ ] `unregister_rpc()` removes from registry
- [ ] Hash collision detection (different names, same hash)
- [ ] Decorator `@rpc` registers with correct metadata
- [ ] Rate limit parameter stored correctly

---

### 1.2 Unit Tests: RPC Dispatch

**File**: `tests/blackbox_rpc_dispatch.py`

**Acceptance Criteria**:
- [ ] `call_rpc()` serializes and invokes callback
- [ ] Sequence number incremented per call
- [ ] Reliable calls tracked in pending_rpcs
- [ ] `receive_rpc()` deserializes and executes handler
- [ ] Deduplication prevents double execution
- [ ] Handler exceptions caught and logged

---

### 1.3 Unit Tests: RPC Serialization

**File**: `tests/blackbox_rpc_serialization.py`

**Acceptance Criteria**:
- [ ] Header serialization roundtrip (12 bytes)
- [ ] RPC hash matches MD5 truncation
- [ ] Pickle args roundtrip for basic types
- [ ] Pickle args roundtrip for complex types
- [ ] Max payload size enforced

---

## 2. RPC Channel Tasks

### 2.1 Unit Tests: Channel State

**File**: `tests/blackbox_rpc_channel.py`

**Acceptance Criteria**:
- [ ] Channel starts in initial state
- [ ] Sequence numbers increment on send
- [ ] Pending ACK tracking on reliable send
- [ ] ACK received removes from pending
- [ ] Retransmit triggered after timeout
- [ ] Out-of-order reliable messages buffered
- [ ] Buffer drained when gaps filled

---

### 2.2 Unit Tests: Channel Manager

**File**: `tests/blackbox_rpc_channel_manager.py`

**Acceptance Criteria**:
- [ ] `get_or_create_channel()` creates on first access
- [ ] `get_or_create_channel()` returns existing on second access
- [ ] `broadcast_rpc()` sends to all channels
- [ ] `broadcast_rpc()` respects exclude set
- [ ] `close_channel()` initiates graceful close
- [ ] `remove_channel()` forces immediate removal
- [ ] `cleanup_closed_channels()` removes CLOSED channels

---

### 2.3 Unit Tests: ACK/NACK

**File**: `tests/blackbox_rpc_ack.py`

**Acceptance Criteria**:
- [ ] ACK message serialization roundtrip
- [ ] NACK message serialization with reason
- [ ] ACK processing removes from pending
- [ ] NACK processing removes from pending
- [ ] Batch message serialization roundtrip

---

## 3. RPC Validation Tasks

### 3.1 Unit Tests: Authority Validation

**File**: `tests/blackbox_rpc_authority.py`

**Acceptance Criteria**:
- [ ] SERVER authority: only server can invoke
- [ ] CLIENT authority: only client can invoke
- [ ] OWNER authority: only owner can invoke
- [ ] OWNER authority with wrong owner: rejected
- [ ] MULTICAST authority: only server can invoke
- [ ] ValidationError raised on violation

---

### 3.2 Unit Tests: Rate Limiting

**File**: `tests/blackbox_rpc_rate_limit.py`

**Acceptance Criteria**:
- [ ] Calls within limit allowed
- [ ] Calls exceeding limit rejected
- [ ] Burst allowance consumed first
- [ ] Window sliding removes old calls
- [ ] Per-caller, per-RPC tracking
- [ ] `get_remaining_calls()` returns correct count

---

### 3.3 Unit Tests: Parameter Validation

**File**: `tests/blackbox_rpc_param_validation.py`

**Acceptance Criteria**:
- [ ] `validate_param_range()` accepts in-range values
- [ ] `validate_param_range()` rejects out-of-range values
- [ ] `validate_param_type()` accepts correct types
- [ ] `validate_param_type()` rejects wrong types
- [ ] `validate_param_length()` accepts valid lengths
- [ ] `validate_param_length()` rejects invalid lengths
- [ ] ValidationError includes parameter name

---

## 4. Security Authority Tasks

### 4.1 Unit Tests: Field Authority

**File**: `tests/blackbox_authority_validator.py`

**Acceptance Criteria**:
- [ ] SERVER authority can write any field
- [ ] CLIENT authority cannot write server-only fields
- [ ] OWNER authority can write owner-writable fields
- [ ] Custom validator invoked for custom rules
- [ ] Unconfigured fields default to server-only

---

### 4.2 Unit Tests: Spawn/Destroy Authority

**File**: `tests/blackbox_spawn_authority.py`

**Acceptance Criteria**:
- [ ] `validate_spawn()` allows authorized spawns
- [ ] `validate_spawn()` rejects unauthorized spawns
- [ ] `validate_destroy()` allows owner to destroy owned entity
- [ ] `validate_destroy()` allows server to destroy any entity
- [ ] `validate_destroy()` rejects non-owner client

---

## 5. Input Validation Tasks

### 5.1 Unit Tests: Movement Validation

**File**: `tests/blackbox_input_validator.py`

**Acceptance Criteria**:
- [ ] Movement within MAX_SPEED accepted
- [ ] Movement exceeding MAX_SPEED rejected
- [ ] Teleport within MAX_TELEPORT_DISTANCE accepted
- [ ] Teleport exceeding MAX_TELEPORT_DISTANCE rejected
- [ ] Position within WORLD_BOUNDS accepted
- [ ] Position outside WORLD_BOUNDS rejected
- [ ] TOLERANCE_MULTIPLIER applied correctly

---

### 5.2 Unit Tests: Rotation Validation

**File**: `tests/blackbox_rotation_validation.py`

**Acceptance Criteria**:
- [ ] Rotation within MAX_ROTATION_RATE accepted
- [ ] Rotation exceeding MAX_ROTATION_RATE rejected
- [ ] Delta time considered in rate calculation

---

### 5.3 Unit Tests: Sequence Validation

**File**: `tests/blackbox_sequence_validation.py`

**Acceptance Criteria**:
- [ ] Sequential packets accepted
- [ ] Out-of-order packets flagged
- [ ] Duplicate sequence rejected
- [ ] Large sequence gap flagged
- [ ] Wraparound handled correctly

---

### 5.4 Unit Tests: Player State Tracking

**File**: `tests/blackbox_player_state.py`

**Acceptance Criteria**:
- [ ] New player state created on first input
- [ ] State updated after validation
- [ ] Violation count incremented on invalid input
- [ ] MAX_PLAYER_STATE_ENTRIES limit enforced
- [ ] Thread-safe access with Lock

---

## 6. Rate Limiter Tasks

### 6.1 Unit Tests: Token Bucket

**File**: `tests/blackbox_token_bucket.py`

**Acceptance Criteria**:
- [ ] Initial tokens equal burst size
- [ ] `consume()` deducts tokens
- [ ] `consume()` returns False when insufficient
- [ ] Tokens refill over time at rate
- [ ] Tokens capped at burst size
- [ ] Warning threshold detected

---

### 6.2 Unit Tests: Adaptive Rate Limiter

**File**: `tests/blackbox_adaptive_rate_limiter.py`

**Acceptance Criteria**:
- [ ] Normal load: base limits applied
- [ ] High load: limits reduced by factor
- [ ] `set_server_load()` updates current load
- [ ] Load clamped to [0, 1]

---

## 7. Anomaly Detection Tasks

### 7.1 Unit Tests: Event Recording

**File**: `tests/blackbox_anomaly_events.py`

**Acceptance Criteria**:
- [ ] Shot events recorded with hit/headshot/distance
- [ ] Kill events recorded with victim/weapon
- [ ] Movement events recorded with speed/delta
- [ ] Reaction events recorded with time
- [ ] Damage events recorded with dealt/expected
- [ ] Event count capped at MAX_EVENTS_PER_PLAYER

---

### 7.2 Unit Tests: Aimbot Detection

**File**: `tests/blackbox_aimbot_detection.py`

**Acceptance Criteria**:
- [ ] Accuracy below threshold: no anomaly
- [ ] Accuracy above threshold: AIMBOT anomaly
- [ ] Headshot rate below threshold: no anomaly
- [ ] Headshot rate above threshold: AIMBOT anomaly
- [ ] Minimum sample size enforced

---

### 7.3 Unit Tests: Speed Hack Detection

**File**: `tests/blackbox_speed_hack_detection.py`

**Acceptance Criteria**:
- [ ] Variable speed: no anomaly
- [ ] Constant high speed + low variance: SPEED_HACK anomaly
- [ ] Speed below minimum: not flagged

---

### 7.4 Unit Tests: Reaction Time Detection

**File**: `tests/blackbox_reaction_detection.py`

**Acceptance Criteria**:
- [ ] Normal reaction times: no anomaly
- [ ] Sub-100ms reactions: flagged
- [ ] High ratio of fast reactions: IMPOSSIBLE_REACTION anomaly

---

### 7.5 Unit Tests: Risk Scoring

**File**: `tests/blackbox_risk_scoring.py`

**Acceptance Criteria**:
- [ ] No anomalies: score 0.0
- [ ] Single low-severity: small score
- [ ] Single critical: large score
- [ ] Time decay reduces score
- [ ] Score capped at 1.0

---

## 8. Response Manager Tasks

### 8.1 Unit Tests: Escalation

**File**: `tests/blackbox_response_escalation.py`

**Acceptance Criteria**:
- [ ] First violations: warnings
- [ ] After WARNINGS_BEFORE_KICK: kick
- [ ] After KICKS_BEFORE_TEMP_BAN: temp ban
- [ ] First temp ban: 1 hour
- [ ] Second temp ban: 24 hours
- [ ] After TEMP_BANS_BEFORE_PERMANENT: permanent ban

---

### 8.2 Unit Tests: Ban Records

**File**: `tests/blackbox_ban_records.py`

**Acceptance Criteria**:
- [ ] Ban record created with all fields
- [ ] HWID hash stored correctly
- [ ] IP address stored correctly
- [ ] `is_banned()` returns True for banned player
- [ ] `is_banned()` returns False for unbanned player
- [ ] Temp ban expires correctly

---

### 8.3 Unit Tests: HWID Banning

**File**: `tests/blackbox_hwid_banning.py`

**Acceptance Criteria**:
- [ ] HWID hash generated consistently
- [ ] HWID ban blocks new accounts with same HWID
- [ ] Different HWID not blocked
- [ ] Salt used in hash

---

### 8.4 Unit Tests: Appeal Tracking

**File**: `tests/blackbox_appeal_tracking.py`

**Acceptance Criteria**:
- [ ] Appeal recorded on ban record
- [ ] Appeal count incremented
- [ ] Multiple appeals tracked

---

## 9. Integration Tests

### 9.1 RPC End-to-End

**File**: `tests/integration_rpc.py`

**Acceptance Criteria**:
- [ ] Client calls RPC, server receives
- [ ] Server calls RPC, client receives
- [ ] Owner RPC only accepted from owner
- [ ] Multicast reaches all clients
- [ ] Rate limiting enforced across network

---

### 9.2 Security Pipeline

**File**: `tests/integration_security.py`

**Acceptance Criteria**:
- [ ] Valid input passes all checks
- [ ] Speed hack detected and client warned
- [ ] Repeated violations escalate to kick
- [ ] Aimbot pattern triggers ban
- [ ] HWID ban blocks reconnection

---

## 10. Gap Tasks

### 10.1 Gap: Replace Pickle Serialization

**File**: `engine/networking/rpc/rpc_manager.py` (modify)

**Background**: Pickle is a security risk for untrusted data.

**Acceptance Criteria**:
- [ ] Replace pickle with MessagePack or custom protocol
- [ ] Whitelist allowed types for deserialization
- [ ] Performance: < 10% overhead vs pickle
- [ ] Backward compatibility migration path

---

### 10.2 Gap: RPC Encryption

**File**: `engine/networking/rpc/rpc_channel.py` (modify)

**Background**: RPC payloads transmitted in cleartext.

**Acceptance Criteria**:
- [ ] Optional encryption flag per RPC
- [ ] Key exchange during connection
- [ ] AES-GCM for payload encryption
- [ ] Nonce derived from sequence

---

### 10.3 Gap: Custom Anomaly Detectors

**File**: `engine/networking/security/anomaly_detector.py` (modify)

**Background**: API for custom detectors exists but needs documentation.

**Acceptance Criteria**:
- [ ] Document custom detector registration
- [ ] Provide example detectors
- [ ] Allow game-specific cheat patterns

---

### 10.4 Gap: Shadow Ban Implementation

**File**: `engine/networking/security/response.py` (modify)

**Background**: SHADOW_BAN type defined but not implemented.

**Acceptance Criteria**:
- [ ] Shadow-banned players matched only with other cheaters
- [ ] Matchmaking integration
- [ ] No visible indication to banned player

---

## 11. Performance Tasks

### 11.1 Benchmark: RPC Throughput

**File**: `benchmarks/rpc_throughput.py`

**Acceptance Criteria**:
- [ ] Reliable RPC: > 5,000 calls/second
- [ ] Unreliable RPC: > 20,000 calls/second
- [ ] Multicast to 64 clients: > 1,000 broadcasts/second

---

### 11.2 Benchmark: Security Validation

**File**: `benchmarks/security_validation.py`

**Acceptance Criteria**:
- [ ] Input validation: > 100,000 validations/second
- [ ] Rate limit check: > 500,000 checks/second
- [ ] Anomaly analysis: > 10,000 analyses/second

---

### 11.3 Profile Lock Contention

**Acceptance Criteria**:
- [ ] Profile security subsystem under concurrent load
- [ ] Lock contention < 5% of validation time
- [ ] No deadlocks detected

---

## 12. Documentation Tasks

### 12.1 RPC Usage Guide

**Acceptance Criteria**:
- [ ] Decorator usage examples
- [ ] Authority selection guidelines
- [ ] Reliability mode selection
- [ ] Rate limiting configuration

---

### 12.2 Security Configuration Guide

**Acceptance Criteria**:
- [ ] Threshold tuning recommendations
- [ ] False positive reduction strategies
- [ ] Escalation policy customization
- [ ] HWID hash component selection

---

### 12.3 Anti-Cheat Developer Guide

**Acceptance Criteria**:
- [ ] Detection algorithm explanations
- [ ] Custom detector implementation
- [ ] Testing with simulated cheaters
- [ ] Response integration
