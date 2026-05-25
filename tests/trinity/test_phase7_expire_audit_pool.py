"""Tests for ExpiringDescriptor, AuditDescriptor, and PooledDescriptor."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from trinity.decorators.ops import Op, Step
from trinity.descriptors.expiring import ExpiringDescriptor
from trinity.descriptors.audit import AuditDescriptor, get_audit_log, clear_audit_log
from trinity.descriptors.pooled_field import PooledDescriptor, acquire


# =============================================================================
# ExpiringDescriptor
# =============================================================================


class Expirable:
    token = ExpiringDescriptor(ttl=1.0, default="expired")


class TestExpiringDescriptor:
    def test_value_returned_before_expiry(self):
        obj = Expirable()
        obj.token = "abc123"
        assert obj.token == "abc123"

    def test_default_after_expiry(self):
        obj = Expirable()
        with patch("trinity.descriptors.expiring.time") as mock_time:
            mock_time.time.return_value = 1000.0
            obj.token = "abc123"
            # After TTL
            mock_time.time.return_value = 1002.0
            assert obj.token == "expired"

    def test_re_setting_resets_ttl(self):
        obj = Expirable()
        with patch("trinity.descriptors.expiring.time") as mock_time:
            mock_time.time.return_value = 1000.0
            obj.token = "first"
            # Almost expired
            mock_time.time.return_value = 1000.9
            assert obj.token == "first"
            # Re-set resets the TTL
            obj.token = "second"
            # 0.9s after re-set, still valid
            mock_time.time.return_value = 1001.8
            assert obj.token == "second"
            # Past new TTL
            mock_time.time.return_value = 1002.1
            assert obj.token == "expired"

    def test_descriptor_steps(self):
        desc = ExpiringDescriptor(ttl=5.0)
        steps = desc.descriptor_steps
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops
        assert Op.TAG in ops

    def test_get_metadata(self):
        desc = ExpiringDescriptor(ttl=3.0, default="none")
        meta = desc.get_metadata()
        assert isinstance(meta, dict)
        assert meta["ttl"] == 3.0
        assert meta["default"] == "none"


# =============================================================================
# AuditDescriptor
# =============================================================================


class Audited:
    value = AuditDescriptor(max_entries=5)


class AuditedReads:
    value = AuditDescriptor(max_entries=5, log_reads=True)


class TestAuditDescriptor:
    def test_post_set_logs(self):
        obj = Audited()
        obj.value = 10
        obj.value = 20
        log = get_audit_log(obj, "value")
        assert len(log) == 2
        assert log[0][1] == "set"
        assert log[0][2] is None  # old
        assert log[0][3] == 10    # new
        assert log[1][1] == "set"
        assert log[1][2] == 10    # old
        assert log[1][3] == 20    # new

    def test_log_reads_logs_gets(self):
        obj = AuditedReads()
        obj.value = 42
        _ = obj.value
        log = get_audit_log(obj, "value")
        # 1 set + 1 get
        assert len(log) == 2
        assert log[1][1] == "get"
        assert log[1][2] == 42

    def test_get_audit_log_with_limit(self):
        obj = Audited()
        for i in range(5):
            obj.value = i
        log = get_audit_log(obj, "value", limit=2)
        assert len(log) == 2
        assert log[-1][3] == 4

    def test_clear_audit_log(self):
        obj = Audited()
        obj.value = 1
        clear_audit_log(obj, "value")
        assert get_audit_log(obj, "value") == []

    def test_max_entries_trims(self):
        obj = Audited()
        for i in range(10):
            obj.value = i
        log = get_audit_log(obj, "value")
        assert len(log) == 5

    def test_descriptor_steps(self):
        desc = AuditDescriptor(max_entries=10)
        steps = desc.descriptor_steps
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops
        assert Op.TAG in ops

    def test_get_metadata(self):
        desc = AuditDescriptor(max_entries=10, log_reads=True)
        meta = desc.get_metadata()
        assert isinstance(meta, dict)
        assert meta["max_entries"] == 10
        assert meta["log_reads"] is True


# =============================================================================
# PooledDescriptor
# =============================================================================


_factory_calls = 0


def _make_buffer():
    global _factory_calls
    _factory_calls += 1
    return bytearray(1024)


class Pooled:
    buf = PooledDescriptor(pool_factory=_make_buffer, max_pool_size=3)


class TestPooledDescriptor:
    def setup_method(self):
        global _factory_calls
        _factory_calls = 0
        # Clear pool between tests
        Pooled._pool_buf = []

    def test_acquire_from_factory(self):
        val = acquire(Pooled, "buf")
        assert isinstance(val, bytearray)
        assert len(val) == 1024
        assert _factory_calls == 1

    def test_acquire_from_pool(self):
        obj = Pooled()
        original = bytearray(b"hello" + bytes(1019))
        obj.buf = original
        # Setting a new value returns old to pool
        obj.buf = bytearray(1024)
        # Now acquire should get from pool, not factory
        before = _factory_calls
        pooled_val = acquire(Pooled, "buf")
        assert _factory_calls == before  # no new factory call
        assert pooled_val is original

    def test_set_returns_old_to_pool(self):
        obj = Pooled()
        obj.buf = bytearray(1024)
        obj.buf = bytearray(1024)
        pool = Pooled._pool_buf
        assert len(pool) == 1

    def test_max_pool_size(self):
        obj = Pooled()
        for i in range(5):
            obj.buf = bytearray(1024)
        pool = Pooled._pool_buf
        assert len(pool) == 3

    def test_descriptor_steps(self):
        desc = PooledDescriptor(pool_factory=_make_buffer, max_pool_size=3)
        steps = desc.descriptor_steps
        assert isinstance(steps, list)
        assert all(isinstance(s, Step) for s in steps)
        ops = [s.op for s in steps]
        assert Op.INTERCEPT in ops
        assert Op.TAG in ops

    def test_get_metadata(self):
        desc = PooledDescriptor(pool_factory=_make_buffer, max_pool_size=5)
        meta = desc.get_metadata()
        assert isinstance(meta, dict)
        assert meta["max_pool_size"] == 5
