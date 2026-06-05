# Investigation: engine/networking/rpc/

## Classification: REAL (Production-Ready Implementation)

All four files contain fully functional, production-grade code with complete implementations, not stubs.

---

## File Summary

| File | Lines | Classification | Purpose |
|------|-------|----------------|---------|
| `rpc_manager.py` | 615 | REAL | Core RPC registration, dispatch, serialization |
| `rpc_channel.py` | 593 | REAL | Network transmission, ordering, reliability |
| `rpc_validation.py` | 538 | REAL | Authority checking, rate limiting, parameter validation |
| `__init__.py` | 67 | REAL | Public API exports |

---

## 1. RPC Routing

### Registration System (`RPCManager`)
- `register_rpc(func, authority, reliability, rate_limit, name)` registers callable with metadata
- `unregister_rpc(name)` removes RPC from registry
- `get_rpc_info(name)` retrieves `RPCInfo` dataclass for a registered RPC

### Hash-Based Lookup
- `RPCInfo.get_hash()` computes 4-byte MD5 hash of RPC name for network identification
- Deserialization scans `_registered_rpcs` to map hash back to name

### Call Flow
1. `call_rpc(name, args, target, caller_id)` validates authority, checks rate limit
2. Assigns monotonic sequence number
3. Serializes via `_serialize_rpc()` (pickle for args)
4. Invokes `_on_rpc_callback` for transmission
5. Reliable calls tracked in `_pending_rpcs`

### Receive Flow
1. `receive_rpc(data, caller, caller_id)` deserializes
2. Validates authority and rate limit
3. Checks deduplication via `_call_history`
4. Executes handler from `_rpc_handlers`

### Decorator Support
```python
@rpc(authority=RPCAuthority.CLIENT, reliable=True, rate_limit=10.0)
def request_action(self, action_id: int):
    ...
```

---

## 2. Validation System

### Authority Types (`RPCAuthority` enum)
| Authority | Direction | Who Can Invoke |
|-----------|-----------|----------------|
| `SERVER` | Server -> Client | Only server |
| `CLIENT` | Client -> Server | Only client |
| `OWNER` | Bidirectional | Only entity owner |
| `MULTICAST` | Server -> All | Only server broadcasts |

### Authority Validation (`rpc_validation.py`)
- `validate_authority(caller_id, rpc_info, is_server, owner_id)` enforces rules
- `OWNER` requires `owner_id` parameter; raises `ValidationError` if caller != owner
- `MULTICAST` raises `ValidationError` if caller is not server

### Rate Limiting
- `RateLimiter` class implements sliding window algorithm with burst support
- `RateLimitConfig` dataclass: `max_calls`, `window_seconds`, `burst_allowance`
- Per-caller, per-RPC tracking via `_call_history: dict[tuple[int, str], list[float]]`
- `check_rate_limit()` returns `False` when exceeded (consumes burst first)
- `get_remaining_calls()` reports quota status

### Parameter Validation Helpers
- `validate_param_range(value, min_val, max_val, param_name)`
- `validate_param_type(value, expected_type, param_name)`
- `validate_param_length(value, min_len, max_len, param_name)`
- All raise `ValidationError` on failure

### Custom Validators
- `RPCValidator.register_custom_validator(rpc_name, fn)` installs `(caller_id, entity_id, args) -> bool`
- Entity ownership tracked via `_entity_owners: dict[int, int]`

---

## 3. Reliability Modes

### Reliability Enum (`RPCReliability`)
| Mode | Guarantee | Use Case |
|------|-----------|----------|
| `RELIABLE` | Guaranteed, ordered | State changes, critical actions |
| `UNRELIABLE` | Fire-and-forget | Position updates, ephemeral data |
| `RELIABLE_UNORDERED` | Guaranteed, may reorder | Independent batched operations |

### Pending RPC Tracking
- `PendingRPC` dataclass: `rpc_info`, `args`, `target`, `sequence`, `timestamp`, `retries`
- `_pending_rpcs: dict[int, PendingRPC]` maps sequence to pending
- `acknowledge_rpc(sequence)` removes from pending
- `get_pending_retransmits(timeout)` returns RPCs past timeout, increments `retries`

### Channel-Level Reliability (`RPCChannel`)
- Sequence numbers: `_send_sequence`, `_recv_sequence`, `_acked_sequence`
- Outgoing queue: `_outgoing: deque[RPCMessage]`
- Pending ACKs: `_pending_ack: dict[int, RPCMessage]`
- Incoming buffer: `_incoming_buffer: dict[int, RPCMessage]` for out-of-order reliable messages
- `_process_buffered_messages()` drains buffer when gaps filled

### ACK/NACK Protocol
- Message types from config: `RPC_MSG_CALL=0x01`, `RPC_MSG_ACK=0x02`, `RPC_MSG_NACK=0x03`, `RPC_MSG_BATCH=0x04`
- `create_ack(sequence)` returns serialized ACK
- `create_nack(sequence, reason)` returns serialized NACK with reason payload
- `_handle_ack()` calls `acknowledge()` removing from `_pending_ack`
- `_handle_nack()` removes from pending (RPC rejected)

### Retransmission
- `get_retransmit_data()` scans `_pending_ack` for messages past `_retransmit_timeout`
- Updates `timestamp` on each retransmit

---

## 4. Multicast

### Authority-Based
- `RPCAuthority.MULTICAST` restricts invocation to server only
- Validation rejects client attempts to multicast

### Broadcast Implementation (`RPCChannelManager`)
```python
def broadcast_rpc(
    self,
    rpc_info: RPCInfo,
    args: bytes,
    exclude: Optional[set[int]] = None
) -> dict[int, int]:
```
- Iterates all open channels in `_channels`
- Skips connections in `exclude` set
- Returns `{connection_id: sequence}` map for sent messages

### Channel Management
- `get_or_create_channel(connection_id)` auto-creates on demand
- `close_channel(connection_id)` initiates graceful close (waits for pending ACKs)
- `remove_channel(connection_id)` forces immediate removal
- `cleanup_closed_channels()` removes channels in `CLOSED` state

---

## 5. Serialization Format

### RPC Message Header (10 bytes + payload length + payload)
```
msg_type:   1 byte  (0x01=CALL, 0x02=ACK, 0x03=NACK, 0x04=BATCH)
rpc_hash:   4 bytes (little-endian, MD5-derived)
sequence:   4 bytes (little-endian)
flags:      1 byte  (bit 0 = reliable)
payload_len: 2 bytes (little-endian)
payload:    variable (pickle-serialized args)
```

### Manager-Level Serialization (slightly different)
```
name_hash:  4 bytes
sequence:   4 bytes
flags:      1 byte (bit 0 = reliable, bit 1 = ordered)
args_len:   2 bytes
args:       pickle
```

---

## 6. Configuration Dependencies

Uses `get_config()` from `..config` for:
- `DEFAULT_RETRANSMIT_TIMEOUT`
- `CALL_HISTORY_MAX_AGE`
- `RATE_LIMITER_MAX_AGE`
- `DEFAULT_RPC_BATCH_SIZE`
- `DEFAULT_MAX_OUTGOING_DATA_SIZE`
- `MAX_NACK_REASON_LENGTH`
- `RPC_MSG_CALL`, `RPC_MSG_ACK`, `RPC_MSG_NACK`, `RPC_MSG_BATCH`
- `DEFAULT_RATE_LIMIT_MAX_CALLS`, `DEFAULT_RATE_LIMIT_WINDOW_SECONDS`, `DEFAULT_RATE_LIMIT_BURST_ALLOWANCE`

---

## 7. Public API Exports (`__init__.py`)

### RPC Manager
- `RPCManager`, `RPCInfo`, `RPCAuthority`, `RPCReliability`, `RPCCallResult`, `PendingRPC`, `rpc`

### RPC Channel
- `RPCChannel`, `RPCChannelManager`, `RPCChannelState`, `RPCMessage`

### Validation
- `RPCValidator`, `RateLimiter`, `RateLimitConfig`, `ValidationError`
- `validate_authority`, `validate_rate_limit`, `validate_param_range`, `validate_param_type`, `validate_param_length`

---

## 8. Code Quality Assessment

### Strengths
- Clean separation: manager (logic), channel (transport), validation (security)
- Proper dataclass usage with `slots=True` and `frozen=True` where appropriate
- Comprehensive docstrings with Args/Returns/Raises
- Match/case for authority routing
- Type hints throughout
- Exception hierarchy (`ValidationError`)

### Potential Concerns
- **Pickle serialization**: Security risk if untrusted data received; consider MessagePack or custom protocol
- **Hash collisions**: 4-byte MD5 truncation could collide with many RPCs; acceptable for typical game scales
- **No encryption**: Transport-layer responsibility, but worth noting
- **Deduplication unbounded**: `_call_history` could grow without `cleanup_history()` calls

---

## 9. Integration Points

- Imports `get_config()` from `..config`
- Designed for use with networking transport layer (connection abstraction)
- Handler execution isolated with try/except for `TypeError`, `ValueError`, `AttributeError`, general `Exception`
- Callback-based transmission via `set_rpc_callback()`

---

## Conclusion

The RPC subsystem is a complete, well-architected implementation covering:
1. Registration and dispatch with hash-based routing
2. Four authority modes (SERVER, CLIENT, OWNER, MULTICAST)
3. Three reliability modes (RELIABLE, UNRELIABLE, RELIABLE_UNORDERED)
4. Sliding-window rate limiting with burst allowance
5. Channel-based multicast via `broadcast_rpc()`
6. Ordered/unordered delivery with retransmission support

No stubs detected. Ready for integration testing against the transport layer.
