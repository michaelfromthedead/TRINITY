# CLARIFICATION: Engine Networking System

## Philosophical Framing

The engine networking system embodies a **server-authoritative architecture** that balances player agency with cheat prevention. This design acknowledges the fundamental tension in real-time multiplayer games: players demand responsive controls while security requires server validation of all state changes.

The solution implements a "trust but verify" model where clients predict their own state locally for immediate feedback, but the server remains the sole source of truth. When predictions diverge from server state, reconciliation mechanisms hide the correction from players through smoothing algorithms.

---

## Design Rationale

### Why UDP Over TCP?

TCP's guaranteed delivery and ordering impose latency costs unacceptable for real-time gameplay. A lost packet triggers retransmission and blocks all subsequent data. UDP allows the application layer to implement selective reliability: critical game events (damage, spawns) use reliable channels while frequent updates (positions) tolerate loss.

The multi-channel architecture (unreliable, reliable, reliable-ordered, sequenced) lets each data type use appropriate guarantees:

| Data Type | Channel | Rationale |
|-----------|---------|-----------|
| Position updates | Sequenced | Only latest matters; old data is obsolete |
| Damage events | Reliable-ordered | Must arrive and in correct order |
| Chat messages | Reliable-unordered | Must arrive but order less critical |
| Cosmetic effects | Unreliable | Loss is acceptable |

### Why Client-Side Prediction?

Without prediction, players would experience input lag equal to their round-trip time (50-200ms). The prediction system simulates physics locally, applying inputs immediately. When server state arrives, the system compares predictions against truth.

Small errors (within tolerance) are silently corrected through smoothing. Large errors trigger rollback: reset to server state, replay all unconfirmed inputs. The visual smoother separates display position from simulation position, preventing jarring teleportation.

### Why Lag Compensation?

In FPS games, the server must decide if a shot hit. But the server sees the world 50-100ms behind what the shooter saw. Without compensation, players would have to lead their targets by their ping.

The lag compensation system records world state history and rewinds to the shooter's view time when validating hits. This creates "shooter's advantage" but feels fair because hits register where the crosshair was placed. The anti-cheat validator catches clients claiming impossible view times.

### Why Property-Level Replication?

Entity state often changes partially. Sending full state every tick wastes bandwidth. Property-level dirty tracking only transmits changed fields. Combined with delta compression against acknowledged baselines, typical entity updates shrink from 200 bytes to 40 bytes.

The relevancy system further reduces bandwidth by only sending data players can perceive. An entity 1000 meters away doesn't need position updates at 60Hz. Grid-based spatial hashing makes relevancy checks O(1) rather than O(n) distance comparisons.

### Why Smallest-Three Quaternion Encoding?

Quaternions require 4 components but unit quaternions satisfy x^2 + y^2 + z^2 + w^2 = 1. Given three components, the fourth is computable (with a sign ambiguity resolved by convention). This reduces rotation from 16 bytes (4 floats) to 4 bytes (32 bits total: 2 bits for dropped component index, 10 bits each for three components).

### Why Token Bucket Rate Limiting?

Fixed-window rate limits allow bursts at window boundaries. Token bucket algorithms smooth traffic by continuously refilling "tokens" that requests consume. Burst capacity allows temporary spikes while sustained rate limits prevent abuse.

Adaptive rate limiting extends this by reducing limits when server load is high, prioritizing existing players over new requests.

---

## Architectural Decisions

### Centralized Configuration

All magic numbers live in `config.py` files with typed constants. This allows:

1. Single source of truth for tuning
2. Clear documentation of defaults
3. Easy per-deployment customization
4. Type checking via `Final` annotations

### Callback-Based Event System

Rather than tight coupling, subsystems communicate through callbacks:

```python
def set_on_match_found(self, callback: Callable[[MatchResult], None]) -> None:
    self._on_match_found = callback
```

This enables composition and testing. The transport layer doesn't know about matchmaking; it just calls registered callbacks when connections arrive.

### Thread Safety Where Required

Most networking code runs single-threaded in the game loop. However:

- **NetGUIDManager**: Shared across server threads, uses `Lock`
- **Security managers**: May process reports from multiple sources, use `RLock`
- **Rate limiters**: Per-player state accessed concurrently, use per-entry locks

The rest intentionally avoids locking overhead for single-threaded hot paths.

### Dataclass-Heavy Design

Heavy use of `@dataclass` with `slots=True` and `frozen=True` where appropriate:

- Immutable snapshots prevent accidental mutation of historical state
- Slots reduce memory overhead for high-frequency objects (packets, snapshots)
- Generated `__eq__` and `__hash__` simplify state comparison

---

## Security Philosophy

The security system implements **defense in depth**:

1. **Input validation**: Reject physically impossible inputs (speed > max)
2. **Rate limiting**: Prevent packet flooding
3. **Authority checking**: Server-side validation of all state changes
4. **Anomaly detection**: Statistical analysis of player behavior
5. **Escalating responses**: Graduated penalties from warning to permanent ban

The anti-cheat system prioritizes **low false positives** over catching all cheaters. A 95% accuracy threshold over 50 shots is high enough that legitimate players won't trigger it, even if some subtle aimbots evade detection.

The system also tracks **multiple identifiers** (player ID, HWID hash, IP address) to prevent ban evasion through alt accounts.

---

## Trade-offs and Limitations

### Pickle Serialization Risk

RPC arguments use Python's `pickle` for serialization. This is a security risk if untrusted data is deserialized. Mitigation: validate caller authority before deserializing, or migrate to MessagePack/custom protocol.

### No Encryption

The transport layer defines `ENCRYPTED` flag but doesn't implement encryption. Games with sensitive data should add DTLS or a custom encryption layer.

### IPv4 Only

Socket binding currently assumes `AF_INET`. IPv6 support requires minimal changes but isn't implemented.

### No Congestion Control

Rate limiting prevents overwhelming the server but doesn't adapt to network congestion. For WAN deployment, consider TCP-friendly rate control algorithms.

---

## Relationship to Game Systems

The networking layer is **transport-agnostic** regarding game logic. It provides:

- Reliable/unreliable data channels
- Property replication primitives
- Prediction/reconciliation framework
- Security validation hooks

Game-specific logic (combat rules, scoring, win conditions) lives above this layer. The networking system doesn't know what "damage" means; it just replicates the `health` property and validates that only the server modifies it.

---

## Evolution Path

The current implementation handles typical multiplayer game requirements. Future enhancements might include:

1. **QUIC transport**: Modern reliable-UDP with built-in encryption
2. **Interest management improvements**: Octree for non-uniform entity distributions
3. **Snapshot interpolation**: Buffered state playback for spectators
4. **Rollback netcode**: Full client-server rollback for fighting games
5. **Relay servers**: NAT traversal and latency hiding
6. **Cross-play**: Platform-independent serialization format
