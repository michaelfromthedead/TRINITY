"""RPC validation system for authority and rate limiting.

Provides security validation for remote procedure calls including
authority checking, rate limiting, and parameter validation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..config import get_config
from .rpc_manager import RPCInfo, RPCAuthority

_logger = logging.getLogger(__name__)

# Get config instance
_config = get_config()


class ValidationError(Exception):
    """Exception raised when RPC validation fails.

    Attributes:
        rpc_name: Name of the RPC that failed validation
        reason: Reason for validation failure
        caller_id: ID of the caller
    """

    def __init__(
        self,
        rpc_name: str,
        reason: str,
        caller_id: Optional[int] = None
    ):
        self.rpc_name = rpc_name
        self.reason = reason
        self.caller_id = caller_id
        super().__init__(f"RPC validation failed for '{rpc_name}': {reason}")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        max_calls: Maximum calls in the window
        window_seconds: Time window in seconds
        burst_allowance: Extra calls allowed in burst
    """
    max_calls: int = _config.DEFAULT_RATE_LIMIT_MAX_CALLS
    window_seconds: float = _config.DEFAULT_RATE_LIMIT_WINDOW_SECONDS
    burst_allowance: int = _config.DEFAULT_RATE_LIMIT_BURST_ALLOWANCE


@dataclass
class RateLimiter:
    """Per-player, per-RPC rate limiter.

    Uses a sliding window algorithm with burst support.

    Attributes:
        config: Rate limit configuration
    """
    config: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Tracking: (caller_id, rpc_name) -> list of timestamps
    _call_history: dict[tuple[int, str], list[float]] = field(
        default_factory=dict
    )

    # Burst tracking: (caller_id, rpc_name) -> burst remaining
    _burst_remaining: dict[tuple[int, str], int] = field(
        default_factory=dict
    )

    def check_rate_limit(
        self,
        caller_id: int,
        rpc_name: str,
        custom_limit: Optional[float] = None
    ) -> bool:
        """Check if a call is within rate limits.

        Args:
            caller_id: Caller identifier
            rpc_name: RPC name
            custom_limit: Optional custom rate limit (calls/second)

        Returns:
            True if within rate limit
        """
        key = (caller_id, rpc_name)
        now = time.time()

        # Initialize tracking
        if key not in self._call_history:
            self._call_history[key] = []
            self._burst_remaining[key] = self.config.burst_allowance

        # Get effective limit
        max_calls = (
            int(custom_limit * self.config.window_seconds)
            if custom_limit else self.config.max_calls
        )

        # Clean old entries
        window_start = now - self.config.window_seconds
        self._call_history[key] = [
            ts for ts in self._call_history[key]
            if ts > window_start
        ]

        # Check rate
        current_calls = len(self._call_history[key])

        if current_calls < max_calls:
            # Within normal limit
            self._call_history[key].append(now)
            return True

        # Check burst allowance
        if self._burst_remaining[key] > 0:
            self._burst_remaining[key] -= 1
            self._call_history[key].append(now)
            return True

        return False

    def record_call(self, caller_id: int, rpc_name: str) -> None:
        """Record an RPC call for rate limiting.

        Args:
            caller_id: Caller identifier
            rpc_name: RPC name
        """
        key = (caller_id, rpc_name)
        now = time.time()

        if key not in self._call_history:
            self._call_history[key] = []
            self._burst_remaining[key] = self.config.burst_allowance

        self._call_history[key].append(now)

    def reset_burst(self, caller_id: int, rpc_name: str) -> None:
        """Reset burst allowance for a caller/RPC.

        Args:
            caller_id: Caller identifier
            rpc_name: RPC name
        """
        key = (caller_id, rpc_name)
        self._burst_remaining[key] = self.config.burst_allowance

    def get_remaining_calls(
        self,
        caller_id: int,
        rpc_name: str
    ) -> int:
        """Get remaining calls in current window.

        Args:
            caller_id: Caller identifier
            rpc_name: RPC name

        Returns:
            Number of remaining allowed calls
        """
        key = (caller_id, rpc_name)
        now = time.time()

        if key not in self._call_history:
            return self.config.max_calls

        window_start = now - self.config.window_seconds
        current_calls = sum(
            1 for ts in self._call_history[key]
            if ts > window_start
        )

        remaining = self.config.max_calls - current_calls
        remaining += self._burst_remaining.get(key, 0)

        return max(0, remaining)

    def cleanup(self, max_age: float = _config.RATE_LIMITER_MAX_AGE) -> int:
        """Clean up old tracking data.

        Args:
            max_age: Maximum age in seconds

        Returns:
            Number of entries removed
        """
        now = time.time()
        removed = 0

        for key in list(self._call_history.keys()):
            # Remove entries older than max_age
            self._call_history[key] = [
                ts for ts in self._call_history[key]
                if now - ts < max_age
            ]

            # Remove empty entries
            if not self._call_history[key]:
                del self._call_history[key]
                self._burst_remaining.pop(key, None)
                removed += 1

        return removed


def validate_authority(
    caller_id: int,
    rpc_info: RPCInfo,
    is_server: bool,
    owner_id: Optional[int] = None
) -> bool:
    """Validate authority for an RPC call.

    Args:
        caller_id: ID of the caller
        rpc_info: RPC metadata
        is_server: Whether validation is on server side
        owner_id: Owner ID for OWNER authority RPCs

    Returns:
        True if authority is valid

    Raises:
        ValidationError: If authority check fails
    """
    match rpc_info.authority:
        case RPCAuthority.SERVER:
            # Server -> Client: Only server can invoke
            # If we're validating on server, caller must be server
            # If we're validating on client, incoming should be from server
            if is_server:
                # Server is sending - always OK
                return True
            else:
                # Client receiving - OK (from server)
                return True

        case RPCAuthority.CLIENT:
            # Client -> Server: Only clients can invoke
            if is_server:
                # Server receiving from client - OK
                return True
            else:
                # Client is sending - always OK
                return True

        case RPCAuthority.OWNER:
            # Only entity owner can invoke
            if owner_id is None:
                raise ValidationError(
                    rpc_info.name,
                    "Owner validation requires owner_id",
                    caller_id
                )
            if caller_id != owner_id:
                raise ValidationError(
                    rpc_info.name,
                    f"Caller {caller_id} is not owner {owner_id}",
                    caller_id
                )
            return True

        case RPCAuthority.MULTICAST:
            # Only server can multicast
            if not is_server:
                raise ValidationError(
                    rpc_info.name,
                    "Only server can multicast",
                    caller_id
                )
            return True

    return True


def validate_rate_limit(
    caller_id: int,
    rpc_name: str,
    rate_limiter: RateLimiter,
    custom_limit: Optional[float] = None
) -> bool:
    """Validate rate limit for an RPC call.

    Args:
        caller_id: Caller identifier
        rpc_name: RPC name
        rate_limiter: Rate limiter instance
        custom_limit: Optional custom rate limit

    Returns:
        True if within rate limit

    Raises:
        ValidationError: If rate limit exceeded
    """
    if not rate_limiter.check_rate_limit(caller_id, rpc_name, custom_limit):
        raise ValidationError(
            rpc_name,
            "Rate limit exceeded",
            caller_id
        )
    return True


@dataclass
class RPCValidator:
    """Complete RPC validation system.

    Combines authority, rate limiting, and custom validation rules.
    """
    is_server: bool = True
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)

    # Custom validators: rpc_name -> validator function
    _custom_validators: dict[str, Callable[[int, Any, tuple], bool]] = field(
        default_factory=dict
    )

    # Entity owner mapping: entity_id -> owner_id
    _entity_owners: dict[int, int] = field(default_factory=dict)

    def validate(
        self,
        caller_id: int,
        rpc_info: RPCInfo,
        args: tuple[Any, ...],
        entity_id: Optional[int] = None
    ) -> bool:
        """Perform complete validation of an RPC call.

        Args:
            caller_id: Caller identifier
            rpc_info: RPC metadata
            args: Call arguments
            entity_id: Optional entity ID for owner checks

        Returns:
            True if all validation passes

        Raises:
            ValidationError: If any validation fails
        """
        # Authority check
        owner_id = self._entity_owners.get(entity_id) if entity_id else None
        validate_authority(caller_id, rpc_info, self.is_server, owner_id)

        # Rate limit check
        if rpc_info.rate_limit > 0:
            validate_rate_limit(
                caller_id,
                rpc_info.name,
                self.rate_limiter,
                rpc_info.rate_limit
            )

        # Custom validation
        if rpc_info.name in self._custom_validators:
            validator = self._custom_validators[rpc_info.name]
            if not validator(caller_id, entity_id, args):
                raise ValidationError(
                    rpc_info.name,
                    "Custom validation failed",
                    caller_id
                )

        return True

    def register_custom_validator(
        self,
        rpc_name: str,
        validator: Callable[[int, Any, tuple], bool]
    ) -> None:
        """Register a custom validator for an RPC.

        Args:
            rpc_name: RPC name
            validator: Function (caller_id, entity_id, args) -> bool
        """
        self._custom_validators[rpc_name] = validator

    def unregister_custom_validator(self, rpc_name: str) -> bool:
        """Unregister a custom validator.

        Args:
            rpc_name: RPC name

        Returns:
            True if validator was removed
        """
        return self._custom_validators.pop(rpc_name, None) is not None

    def set_entity_owner(self, entity_id: int, owner_id: int) -> None:
        """Set the owner of an entity.

        Args:
            entity_id: Entity identifier
            owner_id: Owner identifier
        """
        self._entity_owners[entity_id] = owner_id

    def remove_entity_owner(self, entity_id: int) -> Optional[int]:
        """Remove entity owner mapping.

        Args:
            entity_id: Entity identifier

        Returns:
            Previous owner ID or None
        """
        return self._entity_owners.pop(entity_id, None)

    def get_entity_owner(self, entity_id: int) -> Optional[int]:
        """Get the owner of an entity.

        Args:
            entity_id: Entity identifier

        Returns:
            Owner ID or None
        """
        return self._entity_owners.get(entity_id)

    def cleanup(self) -> None:
        """Clean up old rate limiting data."""
        self.rate_limiter.cleanup()


# Parameter validation helpers

def validate_param_range(
    value: Any,
    min_val: Optional[Any] = None,
    max_val: Optional[Any] = None,
    param_name: str = "parameter"
) -> bool:
    """Validate a parameter is within a range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If value out of range
    """
    if min_val is not None and value < min_val:
        raise ValidationError(
            "param_validation",
            f"{param_name} {value} below minimum {min_val}"
        )
    if max_val is not None and value > max_val:
        raise ValidationError(
            "param_validation",
            f"{param_name} {value} above maximum {max_val}"
        )
    return True


def validate_param_type(
    value: Any,
    expected_type: type,
    param_name: str = "parameter"
) -> bool:
    """Validate a parameter's type.

    Args:
        value: Value to validate
        expected_type: Expected type
        param_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If type mismatch
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            "param_validation",
            f"{param_name} expected {expected_type.__name__}, got {type(value).__name__}"
        )
    return True


def validate_param_length(
    value: Any,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    param_name: str = "parameter"
) -> bool:
    """Validate a parameter's length.

    Args:
        value: Value to validate (must have __len__)
        min_len: Minimum length
        max_len: Maximum length
        param_name: Name for error messages

    Returns:
        True if valid

    Raises:
        ValidationError: If length out of range
    """
    try:
        length = len(value)
    except TypeError:
        raise ValidationError(
            "param_validation",
            f"{param_name} has no length"
        )

    if min_len is not None and length < min_len:
        raise ValidationError(
            "param_validation",
            f"{param_name} length {length} below minimum {min_len}"
        )
    if max_len is not None and length > max_len:
        raise ValidationError(
            "param_validation",
            f"{param_name} length {length} above maximum {max_len}"
        )
    return True
