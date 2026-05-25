"""Tests for Tier 31 — SECURITY decorators."""

import pytest

from trinity.decorators.security import (
    VALID_RATE_SCOPES,
    encrypted,
    rate_limited,
    server_authoritative,
    validated,
)
from trinity.decorators.registry import Tier, registry


# =========================================================================
# @server_authoritative
# =========================================================================


class TestServerAuthoritative:
    def test_with_parens(self):
        @server_authoritative()
        def move():
            pass

        assert move._server_authoritative is True

    def test_without_parens(self):
        @server_authoritative
        def move():
            pass

        assert move._server_authoritative is True

    def test_applied_decorators(self):
        @server_authoritative
        def f():
            pass

        assert "server_authoritative" in f._applied_decorators

    def test_tags(self):
        @server_authoritative
        def f():
            pass

        assert f._tags["server_authoritative"] is True

    def test_registries(self):
        @server_authoritative
        def f():
            pass

        assert "security" in f._registries

    def test_registry_spec(self):
        spec = registry.get("server_authoritative")
        assert spec is not None
        assert spec.tier == Tier.SECURITY
        assert spec.target_types == ("function",)


# =========================================================================
# @validated
# =========================================================================


class TestValidated:
    def test_no_rules(self):
        @validated()
        class C:
            pass

        assert C._validated is True
        assert C._validation_rules == []

    def test_without_parens(self):
        @validated
        class C:
            pass

        assert C._validated is True
        assert C._validation_rules == []

    def test_with_rules(self):
        r1 = lambda x: x > 0
        r2 = lambda x: x < 100

        @validated(rules=[r1, r2])
        class C:
            pass

        assert C._validated is True
        assert len(C._validation_rules) == 2
        assert C._validation_rules[0] is r1
        assert C._validation_rules[1] is r2

    def test_non_callable_rule_raises(self):
        with pytest.raises(ValueError, match="callable"):
            @validated(rules=[42])
            class C:
                pass

    def test_mixed_callable_non_callable_raises(self):
        with pytest.raises(ValueError, match="callable"):
            @validated(rules=[lambda x: x, "not_callable"])
            class C:
                pass

    def test_applied_decorators(self):
        @validated()
        class C:
            pass

        assert "validated" in C._applied_decorators

    def test_tags(self):
        @validated()
        class C:
            pass

        assert C._tags["validated"] is True

    def test_registry_spec(self):
        spec = registry.get("validated")
        assert spec is not None
        assert spec.tier == Tier.SECURITY
        assert spec.target_types == ("class",)


# =========================================================================
# @rate_limited
# =========================================================================


class TestRateLimited:
    def test_basic(self):
        @rate_limited(max_per_second=10.0)
        def action():
            pass

        assert action._rate_limited is True
        assert action._rate_limit_max == 10.0
        assert action._rate_limit_per == "player"

    def test_global_scope(self):
        @rate_limited(max_per_second=5, per="global")
        def action():
            pass

        assert action._rate_limit_per == "global"

    def test_int_max(self):
        @rate_limited(max_per_second=100)
        def action():
            pass

        assert action._rate_limit_max == 100

    def test_zero_max_raises(self):
        with pytest.raises(ValueError, match="max_per_second"):
            @rate_limited(max_per_second=0)
            def action():
                pass

    def test_negative_max_raises(self):
        with pytest.raises(ValueError, match="max_per_second"):
            @rate_limited(max_per_second=-5)
            def action():
                pass

    def test_invalid_scope_raises(self):
        with pytest.raises(ValueError, match="scope"):
            @rate_limited(max_per_second=10, per="team")
            def action():
                pass

    def test_valid_scopes(self):
        assert VALID_RATE_SCOPES == frozenset({"player", "global"})

    def test_applied_decorators(self):
        @rate_limited(max_per_second=1)
        def f():
            pass

        assert "rate_limited" in f._applied_decorators

    def test_tags(self):
        @rate_limited(max_per_second=5, per="global")
        def f():
            pass

        assert f._tags["rate_limited"] is True
        assert f._tags["rate_limit_max"] == 5
        assert f._tags["rate_limit_per"] == "global"

    def test_registry_spec(self):
        spec = registry.get("rate_limited")
        assert spec is not None
        assert spec.tier == Tier.SECURITY
        assert spec.target_types == ("function",)


# =========================================================================
# @encrypted
# =========================================================================


class TestEncrypted:
    def test_with_parens(self):
        @encrypted()
        class Secret:
            pass

        assert Secret._encrypted is True

    def test_without_parens(self):
        @encrypted
        class Secret:
            pass

        assert Secret._encrypted is True

    def test_applied_decorators(self):
        @encrypted
        class C:
            pass

        assert "encrypted" in C._applied_decorators

    def test_tags(self):
        @encrypted
        class C:
            pass

        assert C._tags["encrypted"] is True

    def test_registries(self):
        @encrypted
        class C:
            pass

        assert "security" in C._registries

    def test_registry_spec(self):
        spec = registry.get("encrypted")
        assert spec is not None
        assert spec.tier == Tier.SECURITY
        assert spec.target_types == ("class",)


# =========================================================================
# Combination tests
# =========================================================================


class TestSecurityCombinations:
    def test_validated_and_encrypted(self):
        @encrypted()
        @validated(rules=[lambda x: x is not None])
        class Secure:
            pass

        assert Secure._validated is True
        assert Secure._encrypted is True
        assert len(Secure._validation_rules) == 1

    def test_server_auth_and_rate_limited(self):
        @rate_limited(max_per_second=30)
        @server_authoritative
        def rpc_call():
            pass

        assert rpc_call._server_authoritative is True
        assert rpc_call._rate_limited is True
        assert rpc_call._rate_limit_max == 30

    def test_all_function_decorators(self):
        @rate_limited(max_per_second=10, per="global")
        @server_authoritative
        def critical():
            pass

        assert critical._server_authoritative is True
        assert critical._rate_limited is True
        assert critical._rate_limit_per == "global"

    def test_all_class_decorators(self):
        @encrypted
        @validated(rules=[])
        class Data:
            pass

        assert Data._validated is True
        assert Data._encrypted is True


# =========================================================================
# Registry tier check
# =========================================================================


class TestSecurityRegistry:
    def test_all_security_decorators_registered(self):
        names = {"server_authoritative", "validated", "rate_limited", "encrypted"}
        for name in names:
            spec = registry.get(name)
            assert spec is not None, f"{name} not registered"
            assert spec.tier == Tier.SECURITY

    def test_all_unique(self):
        for name in ("server_authoritative", "validated", "rate_limited", "encrypted"):
            spec = registry.get(name)
            assert spec.unique is True
