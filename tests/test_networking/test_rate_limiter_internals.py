"""
White-box tests for rate limiter internals.

Tests TokenBucket, RateLimiter, and AdaptiveRateLimiter edge cases
including validation, thread safety, and overload handling.
"""

from __future__ import annotations

import threading
import time
from unittest import mock
import pytest

from engine.networking.security.rate_limiter import (
    RateLimitConfig, RateLimitResult, RateLimitStats,
    TokenBucket, RateLimiter, AdaptiveRateLimiter,
)
from engine.networking.security.config import (
    VALIDATION_LIMITS, RATE_LIMIT_DEFAULTS, ADAPTIVE_RATE_LIMIT,
)


class TestRateLimitConfigValidation:
    """RateLimitConfig validation edge cases."""

    def test_zero_requests_per_second_raises(self):
        with pytest.raises(ValueError, match="requests_per_second.*positive"):
            RateLimitConfig(requests_per_second=0)

    def test_negative_requests_per_second_raises(self):
        with pytest.raises(ValueError, match="requests_per_second.*positive"):
            RateLimitConfig(requests_per_second=-1)

    def test_zero_burst_raises(self):
        with pytest.raises(ValueError, match="burst_size.*positive"):
            RateLimitConfig(requests_per_second=10, burst_size=0)

    def test_burst_exceeds_max_raises(self):
        with pytest.raises(ValueError, match="burst_size.*maximum"):
            RateLimitConfig(requests_per_second=10, burst_size=VALIDATION_LIMITS.MAX_TOKENS_PER_REQUEST * 10 + 1)

    def test_warning_threshold_low_bound(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10, warning_threshold=0.0)
        assert config.warning_threshold == 0.0

    def test_warning_threshold_high_bound(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10, warning_threshold=1.0)
        assert config.warning_threshold == 1.0

    def test_warning_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="warning_threshold"):
            RateLimitConfig(requests_per_second=10, warning_threshold=-0.1)

    def test_warning_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="warning_threshold"):
            RateLimitConfig(requests_per_second=10, warning_threshold=1.1)

    def test_refill_rate_defaults_to_rps(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        assert config.refill_rate == 10.0

    def test_zero_refill_rate_raises(self):
        with pytest.raises(ValueError, match="refill_rate.*positive"):
            RateLimitConfig(requests_per_second=10, burst_size=10, refill_rate=0)

    def test_negative_refill_rate_raises(self):
        with pytest.raises(ValueError, match="refill_rate.*positive"):
            RateLimitConfig(requests_per_second=10, burst_size=10, refill_rate=-1)


class TestTokenBucketEdgeCases:
    """TokenBucket edge case whitebox tests."""

    def test_initial_tokens_equals_burst(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=20)
        bucket = TokenBucket(config)
        assert bucket.tokens == 20.0

    def test_consume_zero_raises(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        with pytest.raises(ValueError, match="positive integer"):
            bucket.try_consume(0)

    def test_consume_negative_raises(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        with pytest.raises(ValueError, match="positive integer"):
            bucket.try_consume(-1)

    def test_consume_non_integer_raises(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        with pytest.raises(ValueError, match="positive integer"):
            bucket.try_consume(1.5)

    def test_consume_exceeds_max_tokens_raises(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        with pytest.raises(ValueError, match="tokens exceeds maximum"):
            bucket.try_consume(VALIDATION_LIMITS.MAX_TOKENS_PER_REQUEST + 1)

    def test_consume_denied_when_depleted(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=1)
        bucket = TokenBucket(config)
        result1 = bucket.try_consume(1)
        assert result1 in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)
        result2 = bucket.try_consume(1)
        assert result2 == RateLimitResult.DENIED

    def test_consume_warned_near_threshold(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10, warning_threshold=0.2)
        bucket = TokenBucket(config)
        for _ in range(8):
            bucket.try_consume(1)
        result = bucket.try_consume(1)
        assert result == RateLimitResult.WARNED

    def test_reset_restores_burst(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        bucket.try_consume(10)
        assert bucket.try_consume(1) == RateLimitResult.DENIED
        bucket.reset()
        assert bucket.tokens == 10.0

    def test_time_until_refill_zero_when_available(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        assert bucket.time_until_refill(1) == 0.0

    def test_time_until_refill_positive_when_empty(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=1)
        bucket = TokenBucket(config)
        bucket.try_consume(1)
        t = bucket.time_until_refill(1)
        assert t > 0.0

    def test_tokens_never_exceed_burst(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=5)
        bucket = TokenBucket(config)
        with mock.patch('time.time', return_value=1000.0):
            bucket._last_refill_time = 500.0
            tokens = bucket.tokens
        assert tokens == 5.0

    def test_get_remaining_tokens(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=10)
        bucket = TokenBucket(config)
        bucket.try_consume(3)
        remaining = bucket.get_remaining_tokens()
        assert remaining == 7

    def test_concurrent_thread_safety(self):
        """TokenBucket handles concurrent access."""
        config = RateLimitConfig(requests_per_second=1000, burst_size=100)
        bucket = TokenBucket(config)
        errors = []
        def consume():
            try:
                bucket.try_consume(1)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=consume) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestRateLimiterEdgeCases:
    """RateLimiter edge case whitebox tests."""

    def test_unknown_player_returns_result(self):
        limiter = RateLimiter()
        result = limiter.check_rate_limit("unknown_player", "input")
        assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_unknown_action_uses_default_config(self):
        limiter = RateLimiter()
        result = limiter.check_rate_limit("player1", "nonexistent_action")
        assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_set_default_config(self):
        limiter = RateLimiter()
        config = RateLimitConfig(requests_per_second=1, burst_size=1)
        limiter.set_default_config("custom_action", config)
        assert limiter._default_configs["custom_action"] is config

    def test_remove_player_then_recreate(self):
        limiter = RateLimiter()
        limiter.check_rate_limit("player1", "input")
        limiter.remove_player("player1")
        result = limiter.check_rate_limit("player1", "input")
        assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_reset_player_limits(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=2)
        limiter = RateLimiter({None: config})
        limiter._default_configs["test"] = config
        limiter.check_rate_limit("p1", "test")
        limiter.check_rate_limit("p1", "test")
        limiter.reset_player_limits("p1")
        result = limiter.check_rate_limit("p1", "test")
        assert result in (RateLimitResult.ALLOWED, RateLimitResult.WARNED)

    def test_get_player_stats_empty_for_unknown(self):
        limiter = RateLimiter()
        stats = limiter.get_player_stats("unknown")
        assert stats == {}

    def test_get_all_player_ids(self):
        limiter = RateLimiter()
        limiter.check_rate_limit("p1", "input")
        limiter.check_rate_limit("p2", "input")
        ids = limiter.get_all_player_ids()
        assert "p1" in ids
        assert "p2" in ids

    def test_global_stats(self):
        limiter = RateLimiter()
        limiter.check_rate_limit("p1", "input")
        stats = limiter.get_global_stats()
        assert "input" in stats

    def test_is_player_limited(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=1)
        limiter = RateLimiter({"test": config})
        limiter.check_rate_limit("p1", "test")
        assert limiter.is_player_limited("p1", "test") is True

    def test_time_until_allowed(self):
        config = RateLimitConfig(requests_per_second=10, burst_size=1)
        limiter = RateLimiter({"test": config})
        limiter.check_rate_limit("p1", "test")
        t = limiter.time_until_allowed("p1", "test")
        assert t > 0.0


class TestAdaptiveRateLimiter:
    """AdaptiveRateLimiter edge case tests."""

    def test_initial_not_overloaded(self):
        arl = AdaptiveRateLimiter()
        assert arl.is_overloaded is False
        assert arl.current_load == 0.0

    def test_update_server_load_normal(self):
        arl = AdaptiveRateLimiter()
        arl.update_server_load(0.5)
        assert arl.current_load == 0.5
        assert arl.is_overloaded is False

    def test_update_server_load_overloaded(self):
        arl = AdaptiveRateLimiter(load_threshold=0.8)
        arl.update_server_load(0.9)
        assert arl.is_overloaded is True

    def test_update_server_load_clamp_min(self):
        arl = AdaptiveRateLimiter()
        arl.update_server_load(-0.5)
        assert arl.current_load == 0.0

    def test_update_server_load_clamp_max(self):
        arl = AdaptiveRateLimiter()
        arl.update_server_load(2.0)
        assert arl.current_load == 1.0

    def test_invalid_load_threshold_negative(self):
        with pytest.raises(ValueError, match="load_threshold"):
            AdaptiveRateLimiter(load_threshold=-0.1)

    def test_invalid_load_threshold_over_one(self):
        with pytest.raises(ValueError, match="load_threshold"):
            AdaptiveRateLimiter(load_threshold=1.5)

    def test_invalid_reduction_factor_zero(self):
        with pytest.raises(ValueError, match="reduction_factor"):
            AdaptiveRateLimiter(reduction_factor=0.0)

    def test_invalid_reduction_factor_over_one(self):
        with pytest.raises(ValueError, match="reduction_factor"):
            AdaptiveRateLimiter(reduction_factor=2.0)

    def test_overloaded_reduces_capacity(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=10)
        arl = AdaptiveRateLimiter(
            default_configs={"test": config},
            load_threshold=0.5,
            reduction_factor=0.5
        )
        arl.update_server_load(0.8)
        # First request consumes 2 tokens (ceil(1/0.5))
        arl.check_rate_limit("p1", "test")
        remaining = arl.get_remaining_tokens("p1", "test")
        # 10 - 2 = 8 (or less due to time passing)
        assert remaining < 10

    def test_borderline_load_threshold(self):
        arl = AdaptiveRateLimiter(load_threshold=0.8)
        arl.update_server_load(0.8)
        assert arl.is_overloaded is True


class TestRateLimitStats:
    """RateLimitStats integrity tests."""

    def test_increment_on_allow(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=10)
        bucket = TokenBucket(config)
        bucket.try_consume(1)
        assert bucket.stats.total_requests == 1
        assert bucket.stats.allowed_requests == 1

    def test_increment_on_deny(self):
        config = RateLimitConfig(requests_per_second=100, burst_size=1)
        bucket = TokenBucket(config)
        bucket.try_consume(1)
        bucket.try_consume(1)
        assert bucket.stats.total_requests == 2
        assert bucket.stats.denied_requests == 1

    def test_global_stats_aggregate(self):
        limiter = RateLimiter()
        limiter.check_rate_limit("p1", "input")
        limiter.check_rate_limit("p2", "input")
        stats = limiter.get_global_stats()
        assert stats["input"].total_requests >= 2
