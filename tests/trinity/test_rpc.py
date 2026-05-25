"""
Tests for RPC decorators (rpc.py).

Tests the @rpc decorator built on Ops.

Each test verifies:
1. Steps are applied (decompose works, _applied_steps populated)
2. Domain attributes are set correctly
3. Validation rejects invalid params
4. Introspection works
"""

import pytest

from trinity.decorators.ops import Op, decompose
from trinity.decorators.registry import Tier, registry
from trinity.decorators.rpc import (
    VALID_AUTHORITIES,
    rpc,
)


# =============================================================================
# @rpc
# =============================================================================


class TestRpc:
    def test_default_params(self):
        @rpc()
        def shoot():
            pass

        assert shoot._rpc is True
        assert shoot._rpc_authority == "server"
        assert shoot._rpc_reliable is True

    def test_custom_authority(self):
        @rpc(authority="client")
        def request_spawn():
            pass

        assert request_spawn._rpc_authority == "client"

    def test_owner_authority(self):
        @rpc(authority="owner")
        def set_name():
            pass

        assert set_name._rpc_authority == "owner"

    def test_unreliable(self):
        @rpc(reliable=False)
        def move_input():
            pass

        assert move_input._rpc_reliable is False

    def test_custom_both(self):
        @rpc(authority="client", reliable=False)
        def ping():
            pass

        assert ping._rpc_authority == "client"
        assert ping._rpc_reliable is False

    def test_invalid_authority(self):
        with pytest.raises(ValueError, match="invalid authority"):

            @rpc(authority="anyone")
            def bad():
                pass

    def test_applied_decorators(self):
        @rpc()
        def f():
            pass

        assert "rpc" in f._applied_decorators

    def test_steps_recorded(self):
        @rpc()
        def f():
            pass

        assert len(f._applied_steps) >= 3
        ops = [s.op for s in f._applied_steps]
        assert Op.TAG in ops
        assert Op.REGISTER in ops

    def test_tags_set(self):
        @rpc(authority="owner", reliable=False)
        def f():
            pass

        assert f._tags["rpc"] is True
        assert f._tags["rpc_authority"] == "owner"
        assert f._tags["rpc_reliable"] is False

    def test_registry_entry(self):
        @rpc()
        def f():
            pass

        assert "rpc" in f._registries

    def test_decompose(self):
        steps = decompose(rpc)
        assert len(steps) >= 3
        tag_steps = [s for s in steps if s.op == Op.TAG]
        assert len(tag_steps) >= 2

    def test_decorator_name(self):
        assert rpc.__name__ == "rpc"

    def test_is_decorator(self):
        assert rpc._is_decorator is True

    def test_no_parens(self):
        @rpc
        def f():
            pass

        assert f._rpc is True
        assert f._rpc_authority == "server"
        assert f._rpc_reliable is True

    def test_all_valid_authorities(self):
        for auth in VALID_AUTHORITIES:

            @rpc(authority=auth)
            def f():
                pass

            assert f._rpc_authority == auth

    def test_function_still_callable(self):
        @rpc()
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_registry_registered(self):
        spec = registry.get("rpc")
        assert spec is not None
        assert spec.tier == Tier.NETWORK_RPC
        assert spec.target_types == ("function",)

    def test_server_reliable_tags(self):
        @rpc(authority="server", reliable=True)
        def f():
            pass

        assert f._tags["rpc_authority"] == "server"
        assert f._tags["rpc_reliable"] is True

    def test_client_unreliable_tags(self):
        @rpc(authority="client", reliable=False)
        def f():
            pass

        assert f._tags["rpc_authority"] == "client"
        assert f._tags["rpc_reliable"] is False


# =============================================================================
# COMPOSITION
# =============================================================================


class TestRpcComposition:
    def test_multiple_rpc_allowed(self):
        # unique=False
        @rpc(authority="server")
        @rpc(authority="client")
        def f():
            pass

        assert f._rpc is True

    def test_rpc_with_other_attrs(self):
        @rpc(authority="owner", reliable=False)
        def f():
            pass

        assert f._rpc is True
        assert f._rpc_authority == "owner"
        assert f._rpc_reliable is False
        assert "rpc" in f._applied_decorators
