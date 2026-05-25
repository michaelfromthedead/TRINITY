"""
Rate limiting system for network security.

This module implements token bucket rate limiting to prevent abuse
and ensure fair resource usage across players.

Thread-safety: All classes in this module are thread-safe.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional
import threading
import time

from engine.networking.security.config import (
    RATE_LIMIT_DEFAULTS,
    ADAPTIVE_RATE_LIMIT,
    VALIDATION_LIMITS,
)


class RateLimitResult(Enum):
    """Result of a rate limit check."""
    ALLOWED = auto()
    DENIED = auto()
    WARNED = auto()  # Near limit


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.

    All defaults are loaded from security config to avoid magic numbers.

    Attributes:
        requests_per_second: Maximum sustained request rate
        burst_size: Maximum burst capacity
        refill_rate: Tokens added per second (defaults to requests_per_second)
        warning_threshold: Fraction of tokens remaining to trigger warning
    """
    requests_per_second: float = RATE_LIMIT_DEFAULTS.INPUT_REQUESTS_PER_SECOND
    burst_size: int = RATE_LIMIT_DEFAULTS.INPUT_BURST_SIZE
    refill_rate: Optional[float] = None
    warning_threshold: float = RATE_LIMIT_DEFAULTS.WARNING_THRESHOLD

    def __post_init__(self):
        # Validate inputs to prevent abuse
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if self.burst_size <= 0:
            raise ValueError("burst_size must be positive")
        if self.burst_size > VALIDATION_LIMITS.MAX_TOKENS_PER_REQUEST * 10:
            raise ValueError(f"burst_size exceeds maximum allowed")
        if not (0.0 <= self.warning_threshold <= 1.0):
            raise ValueError("warning_threshold must be between 0.0 and 1.0")

        if self.refill_rate is None:
            self.refill_rate = self.requests_per_second
        elif self.refill_rate <= 0:
            raise ValueError("refill_rate must be positive")


@dataclass
class RateLimitStats:
    """Statistics for rate limiting."""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    warned_requests: int = 0
    last_denied_time: float = 0.0


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Thread-safe implementation using locks.
    """

    def __init__(self, config: RateLimitConfig):
        """
        Initialize the token bucket.

        Args:
            config: Rate limit configuration
        """
        self._config = config
        self._tokens = float(config.burst_size)
        self._last_refill_time = time.time()
        self._lock = threading.Lock()
        self._stats = RateLimitStats()

    @property
    def tokens(self) -> float:
        """Get current token count (after refill)."""
        self._refill()
        return self._tokens

    @property
    def stats(self) -> RateLimitStats:
        """Get rate limit statistics."""
        return self._stats

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        current_time = time.time()
        time_passed = current_time - self._last_refill_time

        tokens_to_add = time_passed * self._config.refill_rate
        self._tokens = min(self._tokens + tokens_to_add, float(self._config.burst_size))
        self._last_refill_time = current_time

    def try_consume(self, tokens: int = 1) -> RateLimitResult:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            RateLimitResult indicating if the request was allowed

        Raises:
            ValueError: If tokens is invalid
        """
        # Validate tokens to prevent abuse
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError("tokens must be a positive integer")
        if tokens > VALIDATION_LIMITS.MAX_TOKENS_PER_REQUEST:
            raise ValueError(
                f"tokens exceeds maximum ({VALIDATION_LIMITS.MAX_TOKENS_PER_REQUEST})"
            )

        with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens

                # Check if near limit
                threshold = self._config.burst_size * self._config.warning_threshold
                if self._tokens <= threshold:
                    self._stats.warned_requests += 1
                    self._stats.allowed_requests += 1
                    return RateLimitResult.WARNED

                self._stats.allowed_requests += 1
                return RateLimitResult.ALLOWED
            else:
                self._stats.denied_requests += 1
                self._stats.last_denied_time = time.time()
                return RateLimitResult.DENIED

    def get_remaining_tokens(self) -> int:
        """Get the number of remaining tokens."""
        with self._lock:
            self._refill()
            return int(self._tokens)

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        with self._lock:
            self._tokens = float(self._config.burst_size)
            self._last_refill_time = time.time()

    def time_until_refill(self, tokens: int = 1) -> float:
        """
        Calculate time until specified tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Time in seconds until tokens are available (0 if already available)
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                return 0.0

            tokens_needed = tokens - self._tokens
            return tokens_needed / self._config.refill_rate


# Default rate limit configurations for common operations (from config)
DEFAULT_LIMITS = {
    "input": RateLimitConfig(
        requests_per_second=RATE_LIMIT_DEFAULTS.INPUT_REQUESTS_PER_SECOND,
        burst_size=RATE_LIMIT_DEFAULTS.INPUT_BURST_SIZE
    ),
    "rpc": RateLimitConfig(
        requests_per_second=RATE_LIMIT_DEFAULTS.RPC_REQUESTS_PER_SECOND,
        burst_size=RATE_LIMIT_DEFAULTS.RPC_BURST_SIZE
    ),
    "chat": RateLimitConfig(
        requests_per_second=RATE_LIMIT_DEFAULTS.CHAT_REQUESTS_PER_SECOND,
        burst_size=RATE_LIMIT_DEFAULTS.CHAT_BURST_SIZE
    ),
    "spawn": RateLimitConfig(
        requests_per_second=RATE_LIMIT_DEFAULTS.SPAWN_REQUESTS_PER_SECOND,
        burst_size=RATE_LIMIT_DEFAULTS.SPAWN_BURST_SIZE
    ),
    "item_use": RateLimitConfig(
        requests_per_second=RATE_LIMIT_DEFAULTS.ITEM_USE_REQUESTS_PER_SECOND,
        burst_size=RATE_LIMIT_DEFAULTS.ITEM_USE_BURST_SIZE
    ),
    "voice": RateLimitConfig(
        requests_per_second=RATE_LIMIT_DEFAULTS.VOICE_REQUESTS_PER_SECOND,
        burst_size=RATE_LIMIT_DEFAULTS.VOICE_BURST_SIZE
    ),
}


class RateLimiter:
    """
    Rate limiter managing per-player rate limits.

    Thread-safe implementation for concurrent access.
    """

    def __init__(
        self,
        default_configs: Optional[Dict[str, RateLimitConfig]] = None
    ):
        """
        Initialize the rate limiter.

        Args:
            default_configs: Default configurations for action types
        """
        self._default_configs = default_configs or DEFAULT_LIMITS.copy()
        self._player_limiters: Dict[str, Dict[str, TokenBucket]] = {}
        self._global_stats: Dict[str, RateLimitStats] = {}
        self._lock = threading.RLock()

    def set_default_config(self, action: str, config: RateLimitConfig) -> None:
        """
        Set default configuration for an action type.

        Args:
            action: The action type
            config: The rate limit configuration
        """
        with self._lock:
            self._default_configs[action] = config

    def _get_bucket(self, player_id: str, action: str) -> TokenBucket:
        """Get or create a token bucket for a player/action combination."""
        with self._lock:
            if player_id not in self._player_limiters:
                self._player_limiters[player_id] = {}

            if action not in self._player_limiters[player_id]:
                config = self._default_configs.get(
                    action,
                    RateLimitConfig()  # Default config if action unknown
                )
                self._player_limiters[player_id][action] = TokenBucket(config)

            return self._player_limiters[player_id][action]

    def check_rate_limit(
        self,
        player_id: str,
        action: str,
        tokens: int = 1
    ) -> RateLimitResult:
        """
        Check if an action is within rate limits.

        Args:
            player_id: The player's unique identifier
            action: The type of action being rate limited
            tokens: Number of tokens to consume

        Returns:
            RateLimitResult indicating if the action is allowed
        """
        bucket = self._get_bucket(player_id, action)
        result = bucket.try_consume(tokens)

        # Update global stats
        with self._lock:
            if action not in self._global_stats:
                self._global_stats[action] = RateLimitStats()

            self._global_stats[action].total_requests += 1
            if result == RateLimitResult.ALLOWED:
                self._global_stats[action].allowed_requests += 1
            elif result == RateLimitResult.WARNED:
                self._global_stats[action].warned_requests += 1
                self._global_stats[action].allowed_requests += 1
            else:
                self._global_stats[action].denied_requests += 1
                self._global_stats[action].last_denied_time = time.time()

        return result

    def get_remaining_tokens(self, player_id: str, action: str) -> int:
        """
        Get remaining tokens for a player/action.

        Args:
            player_id: The player's unique identifier
            action: The action type

        Returns:
            Number of remaining tokens
        """
        bucket = self._get_bucket(player_id, action)
        return bucket.get_remaining_tokens()

    def reset_player_limits(self, player_id: str) -> None:
        """
        Reset all rate limits for a player.

        Args:
            player_id: The player's unique identifier
        """
        with self._lock:
            if player_id in self._player_limiters:
                for bucket in self._player_limiters[player_id].values():
                    bucket.reset()

    def remove_player(self, player_id: str) -> None:
        """
        Remove a player from rate limiting tracking.

        Args:
            player_id: The player's unique identifier
        """
        with self._lock:
            self._player_limiters.pop(player_id, None)

    def get_player_stats(self, player_id: str) -> Dict[str, RateLimitStats]:
        """
        Get rate limit statistics for a player.

        Args:
            player_id: The player's unique identifier

        Returns:
            Dictionary mapping action types to their stats
        """
        with self._lock:
            if player_id not in self._player_limiters:
                return {}

            return {
                action: bucket.stats
                for action, bucket in self._player_limiters[player_id].items()
            }

    def get_global_stats(self) -> Dict[str, RateLimitStats]:
        """Get global rate limit statistics."""
        with self._lock:
            return dict(self._global_stats)

    def time_until_allowed(
        self,
        player_id: str,
        action: str,
        tokens: int = 1
    ) -> float:
        """
        Get time until an action will be allowed.

        Args:
            player_id: The player's unique identifier
            action: The action type
            tokens: Number of tokens needed

        Returns:
            Time in seconds until allowed (0 if already allowed)
        """
        bucket = self._get_bucket(player_id, action)
        return bucket.time_until_refill(tokens)

    def is_player_limited(self, player_id: str, action: str) -> bool:
        """
        Check if a player is currently rate limited for an action.

        Args:
            player_id: The player's unique identifier
            action: The action type

        Returns:
            True if the player would be denied
        """
        remaining = self.get_remaining_tokens(player_id, action)
        return remaining < 1

    def get_all_player_ids(self) -> list:
        """Get list of all tracked player IDs."""
        with self._lock:
            return list(self._player_limiters.keys())


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts limits based on server load and player behavior.
    """

    def __init__(
        self,
        default_configs: Optional[Dict[str, RateLimitConfig]] = None,
        load_threshold: float = ADAPTIVE_RATE_LIMIT.LOAD_THRESHOLD,
        reduction_factor: float = ADAPTIVE_RATE_LIMIT.REDUCTION_FACTOR
    ):
        """
        Initialize the adaptive rate limiter.

        Args:
            default_configs: Default configurations for action types
            load_threshold: Server load threshold to trigger reduction
            reduction_factor: Factor to reduce limits by when overloaded

        Raises:
            ValueError: If thresholds are invalid
        """
        super().__init__(default_configs)

        # Validate configuration
        if not (0.0 <= load_threshold <= 1.0):
            raise ValueError("load_threshold must be between 0.0 and 1.0")
        if not (0.0 < reduction_factor <= 1.0):
            raise ValueError("reduction_factor must be between 0.0 and 1.0 (exclusive/inclusive)")

        self._load_threshold = load_threshold
        self._reduction_factor = reduction_factor
        self._current_load = 0.0
        self._overloaded = False

    def update_server_load(self, load: float) -> None:
        """
        Update the current server load.

        Args:
            load: Current server load (0.0 to 1.0)
        """
        with self._lock:
            self._current_load = max(
                ADAPTIVE_RATE_LIMIT.MIN_LOAD,
                min(ADAPTIVE_RATE_LIMIT.MAX_LOAD, load)
            )
            self._overloaded = self._current_load >= self._load_threshold

    def check_rate_limit(
        self,
        player_id: str,
        action: str,
        tokens: int = 1
    ) -> RateLimitResult:
        """
        Check rate limit with adaptive adjustment.

        When server is overloaded, effective limits are reduced.
        """
        # When overloaded, require more tokens effectively
        if self._overloaded:
            effective_tokens = int(tokens / self._reduction_factor)
            tokens = max(1, effective_tokens)

        return super().check_rate_limit(player_id, action, tokens)

    @property
    def is_overloaded(self) -> bool:
        """Check if the server is currently overloaded."""
        return self._overloaded

    @property
    def current_load(self) -> float:
        """Get the current server load."""
        return self._current_load
