"""
Comprehensive unit tests for the Capability Security system.
Tests capability flags, immutable sets, context management, and SecureShell.
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from foundation.capabilities import (
    Capability,
    CapabilitySet,
    CapabilityError,
    SecureContext,
    require_capability,
    with_capabilities,
    check_capability,
    assert_capability,
    get_current_capabilities,
    CAPS_NONE,
    CAPS_READONLY,
    CAPS_READWRITE,
    CAPS_FULL,
)
from foundation.secure_shell import (
    SecureShell,
    create_readonly_shell,
    create_sandbox_shell,
    create_full_shell,
)


class TestCapabilityFlags:
    """Tests for Capability flag combinations."""

    def test_capability_none(self):
        """NONE should be zero."""
        assert Capability.NONE.value == 0

    def test_capability_individual_flags(self):
        """Individual flags should be distinct powers of 2."""
        flags = [
            Capability.READ,
            Capability.WRITE,
            Capability.CREATE,
            Capability.DELETE,
            Capability.EXECUTE,
            Capability.SPAWN,
            Capability.NETWORK,
            Capability.FILESYSTEM,
        ]
        # Each should be non-zero
        for f in flags:
            assert f != 0

        # Each should be distinct
        for i, f1 in enumerate(flags):
            for j, f2 in enumerate(flags):
                if i != j:
                    assert f1 != f2

    def test_capability_combinations(self):
        """Capability combinations should work with bitwise OR."""
        rw = Capability.READ | Capability.WRITE
        assert Capability.READ in rw
        assert Capability.WRITE in rw
        assert Capability.DELETE not in rw

    def test_readonly_preset(self):
        """READONLY should equal READ."""
        assert Capability.READONLY == Capability.READ

    def test_readwrite_preset(self):
        """READWRITE should combine READ and WRITE."""
        assert Capability.READWRITE == (Capability.READ | Capability.WRITE)

    def test_full_preset(self):
        """FULL should include all capabilities."""
        full = Capability.FULL
        assert Capability.READ in full
        assert Capability.WRITE in full
        assert Capability.CREATE in full
        assert Capability.DELETE in full
        assert Capability.EXECUTE in full
        assert Capability.SPAWN in full
        assert Capability.NETWORK in full
        assert Capability.FILESYSTEM in full


class TestCapabilitySetImmutable:
    """Tests for CapabilitySet immutability."""

    def test_capability_set_is_frozen(self):
        """CapabilitySet should be a frozen dataclass."""
        caps = CapabilitySet(Capability.READ)
        with pytest.raises(Exception):  # FrozenInstanceError
            caps.capabilities = Capability.WRITE

    def test_grant_returns_new_set(self):
        """grant() should return a new set, not modify original."""
        original = CapabilitySet(Capability.READ)
        new_set = original.grant(Capability.WRITE)

        assert new_set is not original
        assert original.has(Capability.READ)
        assert not original.has(Capability.WRITE)
        assert new_set.has(Capability.READ)
        assert new_set.has(Capability.WRITE)

    def test_revoke_returns_new_set(self):
        """revoke() should return a new set, not modify original."""
        original = CapabilitySet(Capability.READ | Capability.WRITE)
        new_set = original.revoke(Capability.WRITE)

        assert new_set is not original
        assert original.has(Capability.WRITE)
        assert not new_set.has(Capability.WRITE)
        assert new_set.has(Capability.READ)


class TestCapabilitySetHas:
    """Tests for CapabilitySet.has() method."""

    def test_has_single_capability(self):
        """has() should correctly check single capabilities."""
        caps = CapabilitySet(Capability.READ | Capability.WRITE)
        assert caps.has(Capability.READ)
        assert caps.has(Capability.WRITE)
        assert not caps.has(Capability.DELETE)

    def test_has_compound_capability(self):
        """has() with compound capability requires ALL flags."""
        caps = CapabilitySet(Capability.READ | Capability.WRITE)
        assert caps.has(Capability.READ | Capability.WRITE)
        assert not caps.has(Capability.READ | Capability.DELETE)

    def test_has_none(self):
        """has(NONE) should always be True."""
        caps = CapabilitySet(Capability.READ)
        assert caps.has(Capability.NONE)

    def test_has_any_single(self):
        """has_any() should return True if any flag matches."""
        caps = CapabilitySet(Capability.READ)
        assert caps.has_any(Capability.READ | Capability.WRITE)
        assert not caps.has_any(Capability.DELETE | Capability.CREATE)

    def test_contains_operator(self):
        """'in' operator should work via __contains__."""
        caps = CapabilitySet(Capability.READ | Capability.WRITE)
        assert Capability.READ in caps
        assert Capability.DELETE not in caps


class TestCapabilitySetGrantRevoke:
    """Tests for grant() and revoke() operations."""

    def test_grant_single(self):
        """grant() should add a single capability."""
        caps = CapabilitySet(Capability.READ)
        new_caps = caps.grant(Capability.WRITE)
        assert new_caps.has(Capability.WRITE)

    def test_grant_multiple(self):
        """grant() should add multiple capabilities at once."""
        caps = CapabilitySet(Capability.NONE)
        new_caps = caps.grant(Capability.READ | Capability.WRITE)
        assert new_caps.has(Capability.READ)
        assert new_caps.has(Capability.WRITE)

    def test_grant_idempotent(self):
        """Granting already-present capability should be idempotent."""
        caps = CapabilitySet(Capability.READ)
        new_caps = caps.grant(Capability.READ)
        assert new_caps.capabilities == caps.capabilities

    def test_revoke_single(self):
        """revoke() should remove a single capability."""
        caps = CapabilitySet(Capability.READ | Capability.WRITE)
        new_caps = caps.revoke(Capability.WRITE)
        assert new_caps.has(Capability.READ)
        assert not new_caps.has(Capability.WRITE)

    def test_revoke_absent(self):
        """Revoking absent capability should be safe."""
        caps = CapabilitySet(Capability.READ)
        new_caps = caps.revoke(Capability.WRITE)
        assert new_caps.capabilities == caps.capabilities

    def test_or_operator(self):
        """CapabilitySets should support | operator for union."""
        caps1 = CapabilitySet(Capability.READ)
        caps2 = CapabilitySet(Capability.WRITE)
        combined = caps1 | caps2
        assert combined.has(Capability.READ)
        assert combined.has(Capability.WRITE)

    def test_and_operator(self):
        """CapabilitySets should support & operator for intersection."""
        caps1 = CapabilitySet(Capability.READ | Capability.WRITE)
        caps2 = CapabilitySet(Capability.WRITE | Capability.DELETE)
        intersect = caps1 & caps2
        assert intersect.has(Capability.WRITE)
        assert not intersect.has(Capability.READ)
        assert not intersect.has(Capability.DELETE)


class TestRequireCapabilityDecorator:
    """Tests for @require_capability decorator."""

    def test_decorator_allows_with_capability(self):
        """Decorated function should run when capability is present."""
        @require_capability(Capability.READ)
        def read_data():
            return "data"

        with SecureContext(CapabilitySet(Capability.READ)):
            result = read_data()
            assert result == "data"

    def test_decorator_blocks_without_capability(self):
        """Decorated function should raise when capability is missing."""
        @require_capability(Capability.WRITE)
        def write_data():
            return "written"

        with SecureContext(CapabilitySet(Capability.READ)):
            with pytest.raises(CapabilityError) as exc_info:
                write_data()
            assert exc_info.value.required == Capability.WRITE

    def test_decorator_blocks_without_context(self):
        """Decorated function should raise when no context is active."""
        @require_capability(Capability.READ)
        def read_data():
            return "data"

        with pytest.raises(CapabilityError):
            read_data()

    def test_decorator_preserves_function_metadata(self):
        """Decorator should preserve function name and docstring."""
        @require_capability(Capability.READ)
        def documented_function():
            """This is a documented function."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert "documented" in documented_function.__doc__


class TestSecureContext:
    """Tests for SecureContext context manager."""

    def test_context_sets_capabilities(self):
        """SecureContext should set capabilities for the block."""
        assert get_current_capabilities() is None

        with SecureContext(CapabilitySet(Capability.READ)):
            caps = get_current_capabilities()
            assert caps is not None
            assert caps.has(Capability.READ)

        assert get_current_capabilities() is None

    def test_context_restores_on_exit(self):
        """SecureContext should restore previous state on exit."""
        with SecureContext(CapabilitySet(Capability.READ)):
            with SecureContext(CapabilitySet(Capability.WRITE)):
                caps = get_current_capabilities()
                # Inner context restricts to intersection
                assert not caps.has(Capability.WRITE)  # Not in outer
            # Outer context restored
            caps = get_current_capabilities()
            assert caps.has(Capability.READ)

    def test_nested_contexts_restrict(self):
        """Nested contexts should restrict capabilities by default."""
        outer = CapabilitySet(Capability.READ | Capability.WRITE)
        inner = CapabilitySet(Capability.READ | Capability.DELETE)

        with SecureContext(outer):
            assert check_capability(Capability.WRITE)
            with SecureContext(inner):
                # Can only have intersection: READ
                assert check_capability(Capability.READ)
                assert not check_capability(Capability.WRITE)  # Not requested
                assert not check_capability(Capability.DELETE)  # Not in outer
            # Outer restored
            assert check_capability(Capability.WRITE)

    def test_context_on_exception(self):
        """SecureContext should restore capabilities even on exception."""
        try:
            with SecureContext(CapabilitySet(Capability.READ)):
                raise ValueError("test error")
        except ValueError:
            pass

        assert get_current_capabilities() is None

    def test_check_capability_function(self):
        """check_capability should work within context."""
        with SecureContext(CapabilitySet(Capability.READ | Capability.WRITE)):
            assert check_capability(Capability.READ)
            assert check_capability(Capability.WRITE)
            assert not check_capability(Capability.DELETE)

    def test_check_capability_outside_context(self):
        """check_capability should return False outside context."""
        assert not check_capability(Capability.READ)


class TestSecureShellExecute:
    """Tests for SecureShell execution capabilities."""

    def test_execute_allowed_with_capability(self):
        """execute() should work when EXECUTE capability is present."""
        shell = SecureShell(CapabilitySet(Capability.EXECUTE))
        result = shell.execute("1 + 1")
        assert result.success
        assert result.value == 2

    def test_execute_denied_without_capability(self):
        """execute() should fail when EXECUTE capability is missing."""
        shell = SecureShell(CapabilitySet(Capability.READ))
        result = shell.execute("1 + 1")
        assert not result.success
        assert "EXECUTE" in result.error
        assert result.error_type == "CapabilityError"

    def test_execute_with_readonly_shell(self):
        """create_readonly_shell() should not allow execution."""
        shell = create_readonly_shell()
        result = shell.execute("1 + 1")
        assert not result.success

    def test_execute_with_sandbox_shell(self):
        """create_sandbox_shell() should allow execution."""
        shell = create_sandbox_shell()
        result = shell.execute("2 * 3")
        assert result.success
        assert result.value == 6

    def test_execute_with_full_shell(self):
        """create_full_shell() should allow all operations."""
        shell = create_full_shell()
        result = shell.execute("'hello'.upper()")
        assert result.success
        assert result.value == "HELLO"


class TestSecureShellWriteDenied:
    """Tests for SecureShell write restrictions."""

    def test_write_field_denied_without_capability(self):
        """write_field() should fail without WRITE capability."""
        shell = SecureShell(CapabilitySet(Capability.READ))

        class Obj:
            value = 10

        obj = Obj()
        with pytest.raises(CapabilityError) as exc_info:
            shell.write_field(obj, "value", 20)
        assert exc_info.value.required == Capability.WRITE

    def test_write_field_allowed_with_capability(self):
        """write_field() should work with WRITE capability."""
        shell = SecureShell(CapabilitySet(Capability.WRITE))

        class Obj:
            value = 10

        obj = Obj()
        shell.write_field(obj, "value", 20)
        assert obj.value == 20

    def test_read_field_denied_without_capability(self):
        """read_field() should fail without READ capability."""
        shell = SecureShell(CapabilitySet(Capability.WRITE))

        class Obj:
            value = 10

        obj = Obj()
        with pytest.raises(CapabilityError) as exc_info:
            shell.read_field(obj, "value")
        assert exc_info.value.required == Capability.READ

    def test_read_field_allowed_with_capability(self):
        """read_field() should work with READ capability."""
        shell = SecureShell(CapabilitySet(Capability.READ))

        class Obj:
            value = 42

        obj = Obj()
        result = shell.read_field(obj, "value")
        assert result == 42


class TestNestedSecureContexts:
    """Tests for nested secure context behavior."""

    def test_deep_nesting(self):
        """Multiple levels of nesting should work correctly."""
        caps1 = CapabilitySet(Capability.READ | Capability.WRITE | Capability.DELETE)
        caps2 = CapabilitySet(Capability.READ | Capability.WRITE)
        caps3 = CapabilitySet(Capability.READ)

        with SecureContext(caps1):
            assert check_capability(Capability.DELETE)
            with SecureContext(caps2):
                assert not check_capability(Capability.DELETE)
                assert check_capability(Capability.WRITE)
                with SecureContext(caps3):
                    assert not check_capability(Capability.WRITE)
                    assert check_capability(Capability.READ)
                # caps2 level restored
                assert check_capability(Capability.WRITE)
            # caps1 level restored
            assert check_capability(Capability.DELETE)

    def test_same_capability_nesting(self):
        """Nesting with same capabilities should work."""
        caps = CapabilitySet(Capability.READ | Capability.WRITE)

        with SecureContext(caps):
            with SecureContext(caps):
                with SecureContext(caps):
                    assert check_capability(Capability.READ)
                    assert check_capability(Capability.WRITE)


class TestCapabilitySetRepr:
    """Tests for CapabilitySet string representation."""

    def test_repr_none(self):
        """NONE capability should have clear repr."""
        caps = CapabilitySet(Capability.NONE)
        assert "NONE" in repr(caps)

    def test_repr_single(self):
        """Single capability should show in repr."""
        caps = CapabilitySet(Capability.READ)
        assert "READ" in repr(caps)

    def test_repr_multiple(self):
        """Multiple capabilities should show in repr."""
        caps = CapabilitySet(Capability.READ | Capability.WRITE)
        r = repr(caps)
        assert "READ" in r
        assert "WRITE" in r


class TestCapabilityError:
    """Tests for CapabilityError exception."""

    def test_error_has_required(self):
        """CapabilityError should track required capability."""
        error = CapabilityError(
            "Missing WRITE",
            required=Capability.WRITE,
            available=CapabilitySet(Capability.READ)
        )
        assert error.required == Capability.WRITE
        assert error.available.has(Capability.READ)

    def test_error_message(self):
        """CapabilityError should have message."""
        error = CapabilityError("Custom message")
        assert "Custom message" in str(error)


class TestWithCapabilitiesDecorator:
    """Tests for @with_capabilities decorator."""

    def test_decorator_sets_context(self):
        """Decorator should set capability context for function."""
        @with_capabilities(CapabilitySet(Capability.READ))
        def check_caps():
            return check_capability(Capability.READ)

        result = check_caps()
        assert result is True

    def test_decorator_restores_context(self):
        """Decorator should restore context after function returns."""
        @with_capabilities(CapabilitySet(Capability.READ))
        def inner():
            return check_capability(Capability.READ)

        # No context before
        assert get_current_capabilities() is None
        inner()
        # No context after
        assert get_current_capabilities() is None


class TestPresetCapabilitySets:
    """Tests for pre-defined capability sets."""

    def test_caps_none(self):
        """CAPS_NONE should have no capabilities."""
        assert not CAPS_NONE.has(Capability.READ)
        assert not CAPS_NONE.has(Capability.WRITE)

    def test_caps_readonly(self):
        """CAPS_READONLY should have only READ."""
        assert CAPS_READONLY.has(Capability.READ)
        assert not CAPS_READONLY.has(Capability.WRITE)

    def test_caps_readwrite(self):
        """CAPS_READWRITE should have READ and WRITE."""
        assert CAPS_READWRITE.has(Capability.READ)
        assert CAPS_READWRITE.has(Capability.WRITE)
        assert not CAPS_READWRITE.has(Capability.DELETE)

    def test_caps_full(self):
        """CAPS_FULL should have all capabilities."""
        assert CAPS_FULL.has(Capability.READ)
        assert CAPS_FULL.has(Capability.WRITE)
        assert CAPS_FULL.has(Capability.CREATE)
        assert CAPS_FULL.has(Capability.DELETE)
        assert CAPS_FULL.has(Capability.EXECUTE)
        assert CAPS_FULL.has(Capability.SPAWN)
        assert CAPS_FULL.has(Capability.NETWORK)
        assert CAPS_FULL.has(Capability.FILESYSTEM)


class TestSecureShellAdditionalMethods:
    """Tests for additional SecureShell methods."""

    def test_create_entity_denied(self):
        """create_entity() should fail without CREATE capability."""
        shell = SecureShell(CapabilitySet(Capability.READ))

        with pytest.raises(CapabilityError) as exc_info:
            shell.create_entity(dict, x=1)
        assert exc_info.value.required == Capability.CREATE

    def test_create_entity_allowed(self):
        """create_entity() should work with CREATE capability."""
        shell = SecureShell(CapabilitySet(Capability.CREATE))
        result = shell.create_entity(dict)
        assert isinstance(result, dict)

    def test_delete_entity_denied(self):
        """delete_entity() should fail without DELETE capability."""
        shell = SecureShell(CapabilitySet(Capability.READ))

        with pytest.raises(CapabilityError) as exc_info:
            shell.delete_entity(object())
        assert exc_info.value.required == Capability.DELETE

    def test_delete_entity_raises_not_implemented(self):
        """delete_entity() raises NotImplementedError after capability check."""
        shell = SecureShell(CapabilitySet(Capability.DELETE))

        with pytest.raises(NotImplementedError) as exc_info:
            shell.delete_entity(object())
        assert "entity manager" in str(exc_info.value).lower()

    def test_with_capabilities_returns_new_shell(self):
        """with_capabilities() should return new shell."""
        shell = SecureShell(CapabilitySet(Capability.READ))
        new_shell = shell.with_capabilities(CapabilitySet(Capability.WRITE))

        assert new_shell is not shell
        assert new_shell.capabilities.has(Capability.WRITE)
        assert not shell.capabilities.has(Capability.WRITE)

    def test_restrict_to_reduces_capabilities(self):
        """restrict_to() should reduce capabilities."""
        shell = SecureShell(CapabilitySet(Capability.READ | Capability.WRITE))
        restricted = shell.restrict_to(Capability.READ)

        assert restricted.capabilities.has(Capability.READ)
        assert not restricted.capabilities.has(Capability.WRITE)
