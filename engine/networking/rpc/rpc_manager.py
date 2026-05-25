"""RPC (Remote Procedure Call) system for networked game actions.

Handles registration, invocation, and dispatch of remote procedure calls
between server and clients.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar

from ..config import get_config

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class RPCAuthority(Enum):
    """Authority determining who can invoke the RPC."""
    SERVER = auto()    # Only server can invoke (server -> client)
    CLIENT = auto()    # Only client can invoke (client -> server)
    OWNER = auto()     # Only entity owner can invoke
    MULTICAST = auto() # Server broadcasts to all clients


class RPCReliability(Enum):
    """Delivery reliability for RPC."""
    RELIABLE = auto()      # Guaranteed delivery, ordered
    UNRELIABLE = auto()    # No guarantee, fire and forget
    RELIABLE_UNORDERED = auto()  # Guaranteed but may arrive out of order


@dataclass(frozen=True, slots=True)
class RPCInfo:
    """Metadata about a registered RPC.

    Attributes:
        name: Unique RPC identifier
        authority: Who can invoke this RPC
        reliability: Delivery reliability
        ordered: Whether calls must be processed in order
        rate_limit: Max calls per second (0 = unlimited)
        validate_params: Whether to validate parameters
    """
    name: str
    authority: RPCAuthority = RPCAuthority.SERVER
    reliability: RPCReliability = RPCReliability.RELIABLE
    ordered: bool = True
    rate_limit: float = 0.0
    validate_params: bool = True

    @property
    def reliable(self) -> bool:
        """Check if RPC uses reliable delivery."""
        return self.reliability != RPCReliability.UNRELIABLE

    def get_hash(self) -> int:
        """Get a hash for network identification."""
        return int.from_bytes(
            hashlib.md5(self.name.encode()).digest()[:4],
            'little'
        )


@dataclass
class PendingRPC:
    """An RPC awaiting acknowledgment.

    Attributes:
        rpc_info: The RPC metadata
        args: Call arguments
        target: Target entity/connection
        sequence: Sequence number
        timestamp: When RPC was sent
        retries: Number of retry attempts
    """
    rpc_info: RPCInfo
    args: tuple[Any, ...]
    target: Any
    sequence: int
    timestamp: float = field(default_factory=time.time)
    retries: int = 0


@dataclass
class RPCCallResult:
    """Result of an RPC call attempt.

    Attributes:
        success: Whether call was accepted
        sequence: Assigned sequence number (if accepted)
        error: Error message (if rejected)
    """
    success: bool
    sequence: int = 0
    error: str = ""


T = TypeVar('T')


class RPCManager:
    """Central manager for RPC registration and dispatch.

    Handles registration, invocation, validation, and queuing of
    remote procedure calls.

    Attributes:
        is_server: Whether this manager is on server side
    """
    __slots__ = (
        '_is_server', '_registered_rpcs', '_rpc_handlers',
        '_pending_rpcs', '_next_sequence', '_rate_limiters',
        '_call_history', '_on_rpc_callback'
    )

    def __init__(self, is_server: bool = True):
        """Initialize the RPC manager.

        Args:
            is_server: Whether running on server
        """
        self._is_server = is_server

        # RPC registration
        self._registered_rpcs: dict[str, RPCInfo] = {}
        self._rpc_handlers: dict[str, Callable[..., Any]] = {}

        # Pending reliable RPCs
        self._pending_rpcs: dict[int, PendingRPC] = {}  # sequence -> pending
        self._next_sequence = 1

        # Rate limiting per caller
        # (caller_id, rpc_name) -> list of call timestamps
        self._rate_limiters: dict[tuple[int, str], list[float]] = {}

        # Call history for deduplication
        # (caller_id, rpc_name, sequence) -> timestamp
        self._call_history: dict[tuple[int, str, int], float] = {}

        # Callback for sending RPCs
        self._on_rpc_callback: Optional[Callable[[bytes, Any, bool], None]] = None

    @property
    def is_server(self) -> bool:
        """Check if running on server."""
        return self._is_server

    def register_rpc(
        self,
        func: Callable[..., Any],
        authority: RPCAuthority = RPCAuthority.SERVER,
        reliability: RPCReliability = RPCReliability.RELIABLE,
        rate_limit: float = 0.0,
        name: Optional[str] = None
    ) -> RPCInfo:
        """Register an RPC function.

        Args:
            func: The function to register
            authority: Who can invoke this RPC
            reliability: Delivery reliability
            rate_limit: Max calls per second (0 = unlimited)
            name: RPC name (defaults to function name)

        Returns:
            RPCInfo for the registered RPC
        """
        rpc_name = name or func.__name__

        info = RPCInfo(
            name=rpc_name,
            authority=authority,
            reliability=reliability,
            rate_limit=rate_limit
        )

        self._registered_rpcs[rpc_name] = info
        self._rpc_handlers[rpc_name] = func

        return info

    def unregister_rpc(self, name: str) -> bool:
        """Unregister an RPC.

        Args:
            name: RPC name to unregister

        Returns:
            True if RPC was unregistered
        """
        if name in self._registered_rpcs:
            del self._registered_rpcs[name]
            self._rpc_handlers.pop(name, None)
            return True
        return False

    def get_rpc_info(self, name: str) -> Optional[RPCInfo]:
        """Get info for a registered RPC.

        Args:
            name: RPC name

        Returns:
            RPCInfo or None
        """
        return self._registered_rpcs.get(name)

    def call_rpc(
        self,
        name: str,
        args: tuple[Any, ...],
        target: Any,
        caller_id: Optional[int] = None
    ) -> RPCCallResult:
        """Call an RPC.

        Args:
            name: RPC name
            args: Arguments to pass
            target: Target entity/connection
            caller_id: ID of the caller (for validation)

        Returns:
            RPCCallResult indicating success/failure
        """
        # Get RPC info
        info = self._registered_rpcs.get(name)
        if info is None:
            return RPCCallResult(success=False, error=f"Unknown RPC: {name}")

        # Validate authority
        if not self._validate_authority(info, caller_id):
            return RPCCallResult(
                success=False,
                error=f"Authority denied for RPC: {name}"
            )

        # Check rate limit
        if caller_id is not None and info.rate_limit > 0:
            if not self._check_rate_limit(caller_id, name, info.rate_limit):
                return RPCCallResult(
                    success=False,
                    error=f"Rate limit exceeded for RPC: {name}"
                )

        # Assign sequence number
        sequence = self._next_sequence
        self._next_sequence += 1

        # Serialize and send
        data = self._serialize_rpc(info, sequence, args)

        if self._on_rpc_callback:
            self._on_rpc_callback(data, target, info.reliable)

        # Track if reliable
        if info.reliable:
            pending = PendingRPC(
                rpc_info=info,
                args=args,
                target=target,
                sequence=sequence
            )
            self._pending_rpcs[sequence] = pending

        return RPCCallResult(success=True, sequence=sequence)

    def receive_rpc(
        self,
        data: bytes,
        caller: Any,
        caller_id: int
    ) -> tuple[bool, Any]:
        """Process a received RPC.

        Args:
            data: Serialized RPC data
            caller: The caller object
            caller_id: Caller identifier

        Returns:
            Tuple of (success, result)
        """
        # Deserialize
        rpc_name, sequence, args, consumed = self._deserialize_rpc(data)

        # Get RPC info
        info = self._registered_rpcs.get(rpc_name)
        if info is None:
            return False, f"Unknown RPC: {rpc_name}"

        # Validate authority
        if not self._validate_authority(info, caller_id, incoming=True):
            return False, f"Authority denied for RPC: {rpc_name}"

        # Check rate limit
        if info.rate_limit > 0:
            if not self._check_rate_limit(caller_id, rpc_name, info.rate_limit):
                return False, f"Rate limit exceeded"

        # Check for duplicate (deduplication)
        history_key = (caller_id, rpc_name, sequence)
        if history_key in self._call_history:
            # Already processed, this is a retransmit
            return True, None

        # Record in history
        self._call_history[history_key] = time.time()

        # Execute handler
        handler = self._rpc_handlers.get(rpc_name)
        if handler is None:
            return False, f"No handler for RPC: {rpc_name}"

        try:
            result = handler(caller, *args)
            return True, result
        except (TypeError, ValueError, AttributeError) as e:
            _logger.warning("RPC handler error for %s: %s", rpc_name, e)
            return False, str(e)
        except Exception as e:
            _logger.error("Unexpected RPC handler error for %s: %s", rpc_name, e)
            return False, str(e)

    def acknowledge_rpc(self, sequence: int) -> bool:
        """Acknowledge a sent RPC.

        Args:
            sequence: Sequence number to acknowledge

        Returns:
            True if RPC was pending
        """
        return self._pending_rpcs.pop(sequence, None) is not None

    def get_pending_retransmits(self, timeout: float = _config.DEFAULT_RETRANSMIT_TIMEOUT) -> list[PendingRPC]:
        """Get RPCs that need retransmission.

        Args:
            timeout: Time after which to retransmit

        Returns:
            List of pending RPCs to retransmit
        """
        now = time.time()
        retransmit = []

        for pending in list(self._pending_rpcs.values()):
            if now - pending.timestamp > timeout:
                pending.timestamp = now
                pending.retries += 1
                retransmit.append(pending)

        return retransmit

    def set_rpc_callback(
        self,
        callback: Callable[[bytes, Any, bool], None]
    ) -> None:
        """Set callback for sending RPCs.

        Args:
            callback: Function (data, target, reliable) for sending
        """
        self._on_rpc_callback = callback

    def cleanup_history(self, max_age: float = _config.CALL_HISTORY_MAX_AGE) -> int:
        """Clean up old call history entries.

        Args:
            max_age: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        now = time.time()
        old_keys = [
            k for k, ts in self._call_history.items()
            if now - ts > max_age
        ]
        for key in old_keys:
            del self._call_history[key]
        return len(old_keys)

    def cleanup_rate_limiters(self, max_age: float = _config.RATE_LIMITER_MAX_AGE) -> int:
        """Clean up old rate limiter entries.

        Args:
            max_age: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        now = time.time()
        count = 0

        for key in list(self._rate_limiters.keys()):
            timestamps = self._rate_limiters[key]
            # Remove old timestamps
            self._rate_limiters[key] = [
                ts for ts in timestamps if now - ts < max_age
            ]
            if not self._rate_limiters[key]:
                del self._rate_limiters[key]
                count += 1

        return count

    def _validate_authority(
        self,
        info: RPCInfo,
        caller_id: Optional[int],
        incoming: bool = False
    ) -> bool:
        """Validate RPC authority.

        Args:
            info: RPC info
            caller_id: Caller ID
            incoming: Whether this is an incoming RPC

        Returns:
            True if authorized
        """
        match info.authority:
            case RPCAuthority.SERVER:
                # Server RPCs can only be sent from server
                if incoming:
                    # Incoming = we're receiving, so caller should be server
                    return not self._is_server
                else:
                    # Outgoing = we're sending, so we should be server
                    return self._is_server

            case RPCAuthority.CLIENT:
                # Client RPCs can only be sent from client
                if incoming:
                    # Incoming = we're receiving, so caller should be client
                    return self._is_server
                else:
                    # Outgoing = we're sending, so we should be client
                    return not self._is_server

            case RPCAuthority.OWNER:
                # Owner validation requires additional context
                # For now, allow if caller_id is provided
                return caller_id is not None

            case RPCAuthority.MULTICAST:
                # Only server can multicast
                if incoming:
                    return not self._is_server
                else:
                    return self._is_server

        return False

    def _check_rate_limit(
        self,
        caller_id: int,
        rpc_name: str,
        max_rate: float
    ) -> bool:
        """Check rate limit for an RPC call.

        Args:
            caller_id: Caller identifier
            rpc_name: RPC name
            max_rate: Maximum calls per second

        Returns:
            True if within rate limit
        """
        key = (caller_id, rpc_name)
        now = time.time()

        if key not in self._rate_limiters:
            self._rate_limiters[key] = []

        timestamps = self._rate_limiters[key]

        # Remove old timestamps (outside 1 second window)
        timestamps = [ts for ts in timestamps if now - ts < 1.0]
        self._rate_limiters[key] = timestamps

        # Check rate
        if len(timestamps) >= max_rate:
            return False

        # Record this call
        timestamps.append(now)
        return True

    def _serialize_rpc(
        self,
        info: RPCInfo,
        sequence: int,
        args: tuple[Any, ...]
    ) -> bytes:
        """Serialize an RPC for transmission.

        Args:
            info: RPC info
            sequence: Sequence number
            args: Arguments

        Returns:
            Serialized RPC data
        """
        import pickle

        # Header: name_hash(4) + sequence(4) + flags(1)
        name_hash = info.get_hash()
        flags = (
            (0x01 if info.reliable else 0x00) |
            (0x02 if info.ordered else 0x00)
        )

        header = struct.pack('<IIB', name_hash, sequence, flags)

        # Serialize args with pickle (could use custom serialization)
        args_data = pickle.dumps(args)
        args_len = struct.pack('<H', len(args_data))

        return header + args_len + args_data

    def _deserialize_rpc(
        self,
        data: bytes
    ) -> tuple[str, int, tuple[Any, ...], int]:
        """Deserialize an RPC from network data.

        Args:
            data: Serialized RPC data

        Returns:
            Tuple of (name, sequence, args, bytes_consumed)
        """
        import pickle

        offset = 0

        # Header
        name_hash, sequence, flags = struct.unpack('<IIB', data[offset:offset+9])
        offset += 9

        # Find RPC by hash
        rpc_name = None
        for name, info in self._registered_rpcs.items():
            if info.get_hash() == name_hash:
                rpc_name = name
                break

        if rpc_name is None:
            # Unknown RPC, try to extract args anyway
            rpc_name = f"unknown_{name_hash:08x}"

        # Args
        args_len = struct.unpack('<H', data[offset:offset+2])[0]
        offset += 2

        args_data = data[offset:offset+args_len]
        offset += args_len

        try:
            args = pickle.loads(args_data)
        except (pickle.UnpicklingError, ValueError, TypeError, EOFError) as e:
            _logger.warning("Failed to deserialize RPC args: %s", e)
            args = ()

        return rpc_name, sequence, args, offset


def rpc(
    authority: RPCAuthority = RPCAuthority.SERVER,
    reliable: bool = True,
    rate_limit: float = 0.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to mark a method as an RPC.

    Usage:
        @rpc(authority=RPCAuthority.CLIENT, reliable=True)
        def request_action(self, action_id: int):
            ...

    Args:
        authority: Who can invoke this RPC
        reliable: Whether to use reliable delivery
        rate_limit: Max calls per second

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Store RPC metadata on the function
        func._rpc_info = RPCInfo(
            name=func.__name__,
            authority=authority,
            reliability=(
                RPCReliability.RELIABLE if reliable
                else RPCReliability.UNRELIABLE
            ),
            rate_limit=rate_limit
        )
        return func

    return decorator
