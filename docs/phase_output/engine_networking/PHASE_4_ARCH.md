# PHASE 4 ARCHITECTURE: RPC and Security Systems

## Phase Overview

Phase 4 implements the Remote Procedure Call (RPC) system for cross-network method invocation and the comprehensive security system for anti-cheat, input validation, and ban management.

---

## 1. RPC System Architecture

### 1.1 Component Overview

```
RPC System
    |
    +-- RPCManager (core registration and dispatch)
    |       - register_rpc(func, authority, reliability, rate_limit)
    |       - call_rpc(name, args, target, caller_id)
    |       - receive_rpc(data, caller, caller_id)
    |       - Hash-based routing (4-byte MD5)
    |
    +-- RPCChannel (per-connection transport)
    |       - Sequence numbers
    |       - Pending ACK tracking
    |       - Retransmission
    |       - Out-of-order buffering
    |
    +-- RPCValidator (authority and rate limiting)
    |       - Authority checking
    |       - Rate limiting per-caller per-RPC
    |       - Parameter validation helpers
    |
    +-- RPCChannelManager (multi-connection management)
            - get_or_create_channel(connection_id)
            - broadcast_rpc(rpc_info, args, exclude)
            - cleanup_closed_channels()
```

### 1.2 Authority Model

| Authority | Direction | Who Can Invoke |
|-----------|-----------|----------------|
| SERVER | Server -> Client | Server process only |
| CLIENT | Client -> Server | Client process only |
| OWNER | Bidirectional | Entity owner only |
| MULTICAST | Server -> All | Server broadcasts to all |

**Authority Validation**:

```python
def validate_authority(caller_id: int, rpc_info: RPCInfo, is_server: bool, owner_id: Optional[int]) -> bool:
    match rpc_info.authority:
        case RPCAuthority.SERVER:
            return is_server
        case RPCAuthority.CLIENT:
            return not is_server
        case RPCAuthority.OWNER:
            if owner_id is None:
                raise ValidationError("OWNER authority requires owner_id")
            return caller_id == owner_id
        case RPCAuthority.MULTICAST:
            return is_server
```

### 1.3 Reliability Modes

| Mode | Guarantee | Use Case |
|------|-----------|----------|
| RELIABLE | Guaranteed, ordered | State changes, critical actions |
| UNRELIABLE | Fire-and-forget | Position updates, ephemeral data |
| RELIABLE_UNORDERED | Guaranteed, any order | Independent batched operations |

### 1.4 Rate Limiting

**Sliding Window with Burst**:

```python
@dataclass
class RateLimitConfig:
    max_calls: int           # Calls per window
    window_seconds: float    # Window duration
    burst_allowance: int     # Extra calls allowed in burst

class RateLimiter:
    def check_rate_limit(self, caller_id: int, rpc_name: str) -> bool:
        key = (caller_id, rpc_name)
        now = time.time()
        
        # Get call history for this caller/RPC
        history = self._call_history.get(key, [])
        
        # Prune old entries
        history = [t for t in history if now - t < self._config.window_seconds]
        
        # Check limit (base + burst)
        max_allowed = self._config.max_calls + self._config.burst_allowance
        if len(history) >= max_allowed:
            return False
        
        # Record this call
        history.append(now)
        self._call_history[key] = history
        return True
```

### 1.5 RPC Message Format

**Channel Message** (12 bytes header + payload):

```
+-------------+------------+-------+------------+-----------+
| msg_type(1) | rpc_hash(4)| seq(4)| flags(1)   | len(2)    |
+-------------+------------+-------+------------+-----------+
| payload (pickle-serialized args)                          |
+-----------------------------------------------------------+
```

**Message Types**:

| Type | Value | Purpose |
|------|-------|---------|
| RPC_MSG_CALL | 0x01 | RPC invocation |
| RPC_MSG_ACK | 0x02 | Acknowledgment |
| RPC_MSG_NACK | 0x03 | Negative acknowledgment (error) |
| RPC_MSG_BATCH | 0x04 | Multiple RPCs in one packet |

### 1.6 Decorator Interface

```python
@rpc(authority=RPCAuthority.CLIENT, reliable=True, rate_limit=10.0)
def request_spawn(self, spawn_point_id: int, loadout: dict):
    """Client requests to spawn at a spawn point."""
    pass

@rpc(authority=RPCAuthority.SERVER, reliable=True)
def on_damage(self, damage: float, source_id: int, hit_location: tuple):
    """Server notifies client of damage taken."""
    pass

@rpc(authority=RPCAuthority.MULTICAST, reliable=True)
def announce_victory(self, winner_team: int, score: dict):
    """Server announces game victory to all clients."""
    pass
```

---

## 2. Security System Architecture

### 2.1 Component Overview

```
Security System
    |
    +-- AuthorityValidator
    |       - Field write validation
    |       - Entity spawn validation
    |       - Entity destroy validation
    |       - Custom validators
    |
    +-- InputValidator
    |       - Movement speed limits
    |       - Rotation rate limits
    |       - Action rate limits
    |       - Packet sequence validation
    |       - World bounds checking
    |
    +-- RateLimiter
    |       - Token bucket algorithm
    |       - Per-action type limits
    |       - Adaptive server load adjustment
    |
    +-- AnomalyDetector
    |       - 10 anomaly types
    |       - Statistical analysis
    |       - Risk scoring
    |       - Custom detectors
    |
    +-- ResponseManager
            - Escalating penalties
            - Ban records (player/HWID/IP)
            - Appeal tracking
```

### 2.2 Authority Validation

**Authority Levels**:

| Level | Description |
|-------|-------------|
| SERVER | Full authority - can do anything |
| CLIENT | Limited authority - needs validation |
| OWNER | Ownership authority - can modify owned entities |

**Field Write Validation**:

```python
def validate_write(self, entity: Entity, field_name: str, caller: Caller) -> bool:
    rule = self._field_rules.get(field_name)
    if not rule:
        return caller.authority == Authority.SERVER  # Default: server only
    
    if caller.authority == Authority.SERVER:
        return True
    
    if rule.owner_can_write and self._is_owner(entity, caller):
        return True
    
    if rule.custom_validator:
        return rule.custom_validator(entity, field_name, caller)
    
    return False
```

### 2.3 Input Validation

**Validation Bounds**:

| Parameter | Limit | Purpose |
|-----------|-------|---------|
| MAX_SPEED | 600 units/sec | Prevent speed hacks |
| MAX_ROTATION_RATE | 720 deg/sec | Prevent snap aiming |
| MAX_ACTION_RATE | 20/sec | Prevent macro abuse |
| MAX_TELEPORT_DISTANCE | 100 units | Detect teleport hacks |
| WORLD_BOUNDS | +/-100000 units | Prevent out-of-bounds |

**Movement Validation**:

```python
def validate_movement(self, player_id: int, new_position: Vector3, delta_time: float) -> ValidationReport:
    state = self._player_states.get(player_id)
    if not state:
        state = self._create_player_state(player_id, new_position)
        return ValidationReport(result=ValidationResult.VALID)
    
    # Speed check
    distance = euclidean_distance(state.position, new_position)
    speed = distance / delta_time
    max_speed = MAX_SPEED * TOLERANCE_MULTIPLIER
    
    if speed > max_speed:
        return ValidationReport(
            result=ValidationResult.INVALID_SPEED,
            expected=max_speed,
            actual=speed
        )
    
    # Teleport check
    if distance > MAX_TELEPORT_DISTANCE:
        return ValidationReport(
            result=ValidationResult.INVALID_TELEPORT,
            expected=MAX_TELEPORT_DISTANCE,
            actual=distance
        )
    
    # World bounds check
    if not self._in_world_bounds(new_position):
        return ValidationReport(
            result=ValidationResult.INVALID_BOUNDS,
            actual=new_position
        )
    
    # Update state
    state.position = new_position
    return ValidationReport(result=ValidationResult.VALID)
```

### 2.4 Rate Limiting

**Token Bucket Implementation**:

```python
class TokenBucket:
    def __init__(self, rate: float, burst: int):
        self._rate = rate        # Tokens per second
        self._burst = burst      # Maximum tokens
        self._tokens = burst     # Start full
        self._last_update = time.time()
    
    def consume(self, count: int = 1) -> bool:
        self._refill()
        if self._tokens >= count:
            self._tokens -= count
            return True
        return False
    
    def _refill(self):
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_update = now
```

**Default Rate Limits**:

| Action Type | Rate/sec | Burst |
|-------------|----------|-------|
| Input | 60 | 10 |
| RPC | 10 | 5 |
| Chat | 1 | 5 |
| Spawn | 2 | 3 |
| Item Use | 10 | 5 |
| Voice | 50 | 20 |

**Adaptive Rate Limiting**:

```python
class AdaptiveRateLimiter(RateLimiter):
    def __init__(self, base_limits: dict, load_threshold: float = 0.8, reduction_factor: float = 0.5):
        self._base_limits = base_limits
        self._load_threshold = load_threshold
        self._reduction_factor = reduction_factor
        self._current_load = 0.0
    
    def get_effective_limit(self, action_type: str) -> float:
        base = self._base_limits[action_type]
        if self._current_load > self._load_threshold:
            return base * self._reduction_factor
        return base
    
    def set_server_load(self, load: float):
        self._current_load = clamp(load, 0.0, 1.0)
```

### 2.5 Anomaly Detection

**Anomaly Types**:

| Type | Detection Method | Severity |
|------|------------------|----------|
| AIMBOT | High accuracy + headshot rate | CRITICAL |
| SPEED_HACK | Low speed variance + high speed | HIGH |
| TELEPORT | Large position jumps | HIGH |
| IMPOSSIBLE_REACTION | Sub-100ms reaction times | CRITICAL |
| WALL_HACK_SUSPECT | Hitting non-visible targets | MEDIUM |
| DAMAGE_HACK | Damage exceeds expected | CRITICAL |
| RAPID_FIRE | Attack rate exceeds limits | HIGH |
| NO_RECOIL | Low recoil variance | MEDIUM |
| GOD_MODE | Taking no/reduced damage | CRITICAL |
| RESOURCE_HACK | Impossible resource gains | HIGH |

**Detection Thresholds**:

```python
@dataclass
class AnomalyDetectionConfig:
    # Aimbot
    accuracy_threshold: float = 0.95       # 95% accuracy suspicious
    headshot_rate_threshold: float = 0.80  # 80% headshots suspicious
    accuracy_sample_size: int = 50         # Minimum shots to analyze
    
    # Reaction time
    min_reaction_time_ms: float = 100.0    # Sub-100ms impossible
    impossible_reaction_ratio: float = 0.5 # >50% fast reactions
    
    # Movement
    speed_variance_threshold: float = 0.1  # Constant speed = hack
    speed_hack_min_speed: float = 100.0    # Only flag if fast
    
    # Damage
    damage_multiplier_threshold: float = 1.5  # 150% expected damage
```

**Risk Scoring**:

```python
def get_player_risk_score(self, player_id: int) -> float:
    anomalies = self._anomaly_history.get(player_id, [])
    if not anomalies:
        return 0.0
    
    score = 0.0
    now = time.time()
    
    for anomaly in anomalies:
        # Severity weight
        severity_weight = {
            AnomalySeverity.LOW: 0.1,
            AnomalySeverity.MEDIUM: 0.25,
            AnomalySeverity.HIGH: 0.5,
            AnomalySeverity.CRITICAL: 1.0
        }[anomaly.severity]
        
        # Time decay (older anomalies count less)
        hours_ago = (now - anomaly.timestamp) / 3600
        time_weight = max(0.0, 1.0 - hours_ago / DECAY_HOURS)
        
        score += severity_weight * time_weight
    
    return min(1.0, score)
```

### 2.6 Response/Ban Management

**Escalation Chain**:

```
Violation -> Warning -> Warning -> Warning -> Kick ->
Kick -> 1hr Ban -> 24hr Ban -> Permanent Ban
```

**Escalation Logic**:

```python
def escalate(self, player_id: int, reason: str) -> ResponseAction:
    record = self._records.get(player_id, PlayerRecord())
    
    if record.warnings < WARNINGS_BEFORE_KICK:
        record.warnings += 1
        return ResponseAction(type=ResponseType.WARNING, reason=reason)
    
    if record.kicks < KICKS_BEFORE_TEMP_BAN:
        record.kicks += 1
        return ResponseAction(type=ResponseType.KICK, reason=reason)
    
    if record.temp_bans < TEMP_BANS_BEFORE_LONGER:
        duration = FIRST_TEMP_BAN_DURATION if record.temp_bans == 0 else SECOND_TEMP_BAN_DURATION
        record.temp_bans += 1
        return ResponseAction(type=ResponseType.TEMP_BAN, duration=duration, reason=reason)
    
    if record.temp_bans < TEMP_BANS_BEFORE_PERMANENT:
        record.temp_bans += 1
        return ResponseAction(type=ResponseType.TEMP_BAN, duration=LONGER_BAN_DURATION, reason=reason)
    
    return ResponseAction(type=ResponseType.PERMANENT_BAN, reason=reason)
```

**Ban Types**:

| Type | Description |
|------|-------------|
| WARNING | Message to player |
| KICK | Remove from session |
| TEMP_BAN | Timed ban (1hr, 24hr) |
| PERMANENT_BAN | Indefinite |
| SHADOW_BAN | Play with other cheaters only |

**Ban Records**:

```python
@dataclass
class BanRecord:
    player_id: int
    ban_type: BanType
    reason: str
    issued_at: float
    expires_at: Optional[float]  # None for permanent
    issued_by: str
    hwid_hash: Optional[str]     # Hardware ID hash
    ip_address: Optional[str]
    appeal_count: int = 0
```

**HWID Hashing**:

```python
def generate_hwid_hash(components: list[str], salt: str) -> str:
    """Generate hardware fingerprint hash."""
    combined = '|'.join(components) + salt
    return hashlib.sha256(combined.encode()).hexdigest()
```

---

## 3. Integration Flow

### 3.1 RPC Call Flow

```
Client calls RPC
        |
        v
RPCManager.call_rpc(name, args, target, caller_id)
        |
        +-- Validate authority (must be CLIENT or OWNER)
        +-- Check rate limit
        +-- Serialize: rpc_hash + sequence + flags + args
        |
        v
RPCChannel.queue(serialized_data, reliable)
        |
        +-- If reliable: track in pending_ack
        |
        v
Transport.send(data, connection)
        |
        v
Server receives
        |
        v
RPCManager.receive_rpc(data, caller, caller_id)
        |
        +-- Deserialize
        +-- Validate authority (caller matches expected)
        +-- Check rate limit (server-side)
        +-- Check deduplication
        |
        v
Execute handler
        |
        v
RPCChannel.send_ack(sequence)
```

### 3.2 Security Validation Flow

```
Input received from client
        |
        v
AuthorityValidator.validate_write(entity, field, caller)
        |
        +-- Reject if unauthorized
        |
        v
RateLimiter.check(player_id, action_type)
        |
        +-- Reject if rate exceeded
        |
        v
InputValidator.validate(player_id, input_data)
        |
        +-- Reject if invalid (speed, teleport, bounds)
        |
        v
AnomalyDetector.record_event(player_id, event_type, data)
        |
        v
AnomalyDetector.analyze_player(player_id)
        |
        +-- If anomalies detected:
        |       |
        |       v
        |   ResponseManager.escalate(player_id, reason)
        |       |
        |       v
        |   Execute response (warn/kick/ban)
        |
        v
Accept input and apply to game state
```

---

## 4. Configuration

### 4.1 RPC Configuration

```python
DEFAULT_RETRANSMIT_TIMEOUT = 1.0      # Seconds before retransmit
CALL_HISTORY_MAX_AGE = 60.0           # Deduplication window
DEFAULT_RPC_BATCH_SIZE = 10           # Max RPCs per batch packet
DEFAULT_MAX_OUTGOING_DATA_SIZE = 4096 # Max RPC payload
```

### 4.2 Security Configuration

```python
# Input validation
INPUT_VALIDATION = InputValidationConfig(
    max_speed=600.0,
    max_rotation_rate=720.0,
    max_action_rate=20.0,
    max_teleport_distance=100.0,
    tolerance_multiplier=1.5,
    world_min=(-100000, -10000, -100000),
    world_max=(100000, 50000, 100000),
)

# Rate limits
RATE_LIMIT_DEFAULTS = {
    'input': (60.0, 10),
    'rpc': (10.0, 5),
    'chat': (1.0, 5),
    'spawn': (2.0, 3),
    'item_use': (10.0, 5),
    'voice': (50.0, 20),
}

# Anomaly detection
ANOMALY_DETECTION = AnomalyDetectionConfig()

# Response escalation
RESPONSE = ResponseConfig(
    warnings_before_kick=3,
    kicks_before_temp_ban=2,
    temp_bans_before_longer=2,
    temp_bans_before_permanent=4,
    first_temp_ban_duration=3600.0,
    second_temp_ban_duration=86400.0,
)
```

---

## 5. Thread Safety

**Thread-Safe Components**:

- `InputValidator`: Uses `Lock` for player state access
- `RateLimiter`: Uses `Lock` for bucket state
- `AnomalyDetector`: Uses `RLock` for player stats
- `ResponseManager`: Uses `Lock` for ban records

**Lock Usage Pattern**:

```python
class InputValidator:
    def __init__(self):
        self._lock = threading.Lock()
        self._player_states: dict[int, PlayerState] = {}
    
    def validate_movement(self, player_id: int, position: Vector3) -> ValidationReport:
        with self._lock:
            state = self._player_states.get(player_id)
            # ... validation logic ...
            self._player_states[player_id] = updated_state
            return report
```

---

## 6. Public API Summary

### RPC Module

```python
from engine.networking.rpc import (
    # Manager
    RPCManager, RPCInfo, RPCAuthority, RPCReliability, rpc,
    # Channel
    RPCChannel, RPCChannelManager, RPCMessage,
    # Validation
    RPCValidator, RateLimiter, ValidationError,
)
```

### Security Module

```python
from engine.networking.security import (
    # Authority
    AuthorityValidator, Authority, Caller, Entity,
    # Input
    InputValidator, ValidationResult, Vector3,
    # Rate Limiting
    RateLimiter, AdaptiveRateLimiter, TokenBucket,
    # Anomaly Detection
    AnomalyDetector, AnomalyType, AnomalySeverity,
    # Response
    ResponseManager, ResponseSeverity, BanRecord,
)
```
