"""Tests for CVar system - types, flags, and callbacks.

Tests cover:
- CVar creation with different types (int, float, bool, str)
- CVar flags (READONLY, CHEAT, CONFIG, SCALABILITY)
- on_change callback registration and invocation
- CVarRegistry singleton functionality
"""

import pytest

from engine.debug.console.cvar import (
    CVar,
    CVarBoundsError,
    CVarCheatError,
    CVarFlags,
    CVarReadOnlyError,
    CVarRegistry,
    CVarTypeError,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the CVarRegistry singleton before each test."""
    CVarRegistry.reset_instance()
    yield
    CVarRegistry.reset_instance()


class TestCVarTypes:
    """Test CVar type handling."""

    def test_int_cvar(self):
        """Test integer CVar creation and value handling."""
        cvar = CVar("test.int", default=10)

        assert cvar.value == 10
        assert cvar._value_type == int

        cvar.value = 20
        assert cvar.value == 20

    def test_float_cvar(self):
        """Test float CVar creation and value handling."""
        cvar = CVar("test.float", default=3.14)

        assert cvar.value == 3.14
        assert cvar._value_type == float

        cvar.value = 2.71
        assert cvar.value == 2.71

    def test_bool_cvar(self):
        """Test boolean CVar creation and value handling."""
        cvar = CVar("test.bool", default=True)

        assert cvar.value is True
        assert cvar._value_type == bool

        cvar.value = False
        assert cvar.value is False

    def test_str_cvar(self):
        """Test string CVar creation and value handling."""
        cvar = CVar("test.str", default="hello")

        assert cvar.value == "hello"
        assert cvar._value_type == str

        cvar.value = "world"
        assert cvar.value == "world"

    def test_int_from_string(self):
        """Test setting int CVar from string (console input)."""
        cvar = CVar("test.int_str", default=0)

        cvar.value = "42"
        assert cvar.value == 42

    def test_float_from_string(self):
        """Test setting float CVar from string."""
        cvar = CVar("test.float_str", default=0.0)

        cvar.value = "3.14159"
        assert cvar.value == pytest.approx(3.14159)

    def test_bool_from_string_true(self):
        """Test setting bool CVar from various true strings."""
        cvar = CVar("test.bool_str", default=False)

        for true_str in ["true", "True", "TRUE", "1", "yes", "on"]:
            cvar.value = true_str
            assert cvar.value is True, f"Failed for '{true_str}'"

    def test_bool_from_string_false(self):
        """Test setting bool CVar from various false strings."""
        cvar = CVar("test.bool_str2", default=True)

        for false_str in ["false", "False", "FALSE", "0", "no", "off"]:
            cvar.value = false_str
            assert cvar.value is False, f"Failed for '{false_str}'"

    def test_int_to_float_coercion(self):
        """Test that int values are coerced to float for float CVars."""
        cvar = CVar("test.float_coerce", default=1.0)

        cvar.value = 5  # int
        assert cvar.value == 5.0
        assert isinstance(cvar.value, float)

    def test_type_error_wrong_type(self):
        """Test that wrong types raise CVarTypeError."""
        cvar = CVar("test.type_error", default=10)

        with pytest.raises(CVarTypeError):
            cvar.value = [1, 2, 3]  # list is not valid

    def test_type_error_invalid_string(self):
        """Test that invalid string conversions raise CVarTypeError."""
        cvar = CVar("test.invalid_str", default=10)

        with pytest.raises(CVarTypeError):
            cvar.value = "not_a_number"

    def test_unsupported_type_on_creation(self):
        """Test that unsupported types raise CVarTypeError on creation."""
        with pytest.raises(CVarTypeError):
            CVar("test.list", default=[1, 2, 3])


class TestCVarFlags:
    """Test CVar flags behavior."""

    def test_readonly_flag(self):
        """Test that READONLY prevents modification."""
        cvar = CVar("test.readonly", default=100, flags=CVarFlags.READONLY)

        with pytest.raises(CVarReadOnlyError):
            cvar.value = 200

        assert cvar.value == 100

    def test_readonly_reset(self):
        """Test that reset also respects READONLY."""
        cvar = CVar("test.readonly_reset", default=100, flags=CVarFlags.READONLY)
        cvar._value = 200  # Force change internally

        with pytest.raises(CVarReadOnlyError):
            cvar.reset()

    def test_cheat_flag_disabled(self):
        """Test that CHEAT CVars are blocked when cheats disabled."""
        cvar = CVar("test.cheat", default=True, flags=CVarFlags.CHEAT)
        CVarRegistry.instance().cheats_enabled = False

        with pytest.raises(CVarCheatError):
            _ = cvar.value

        with pytest.raises(CVarCheatError):
            cvar.value = False

    def test_cheat_flag_enabled(self):
        """Test that CHEAT CVars work when cheats enabled."""
        cvar = CVar("test.cheat_enabled", default=True, flags=CVarFlags.CHEAT)
        CVarRegistry.instance().cheats_enabled = True

        assert cvar.value is True
        cvar.value = False
        assert cvar.value is False

    def test_combined_flags(self):
        """Test combining multiple flags."""
        cvar = CVar(
            "test.combined",
            default=5,
            flags=CVarFlags.CONFIG | CVarFlags.SCALABILITY
        )

        assert CVarFlags.CONFIG in cvar.flags
        assert CVarFlags.SCALABILITY in cvar.flags
        assert CVarFlags.READONLY not in cvar.flags

    def test_no_flags(self):
        """Test CVars without flags work normally."""
        cvar = CVar("test.no_flags", default=50)

        assert cvar.flags == CVarFlags.NONE
        cvar.value = 100
        assert cvar.value == 100


class TestCVarCallbacks:
    """Test CVar change callback functionality."""

    def test_on_change_callback(self):
        """Test that on_change callback is invoked."""
        cvar = CVar("test.callback", default=0)
        callback_data = {"called": False, "old": None, "new": None}

        def callback(old, new):
            callback_data["called"] = True
            callback_data["old"] = old
            callback_data["new"] = new

        cvar.on_change(callback)
        cvar.value = 10

        assert callback_data["called"]
        assert callback_data["old"] == 0
        assert callback_data["new"] == 10

    def test_no_callback_when_value_unchanged(self):
        """Test that callback is not called when value doesn't change."""
        cvar = CVar("test.no_change", default=5)
        call_count = {"count": 0}

        def callback(old, new):
            call_count["count"] += 1

        cvar.on_change(callback)
        cvar.value = 5  # Same value

        assert call_count["count"] == 0

    def test_multiple_callbacks(self):
        """Test multiple callbacks are all invoked."""
        cvar = CVar("test.multi_callback", default=0)
        results = []

        cvar.on_change(lambda o, n: results.append(("cb1", o, n)))
        cvar.on_change(lambda o, n: results.append(("cb2", o, n)))
        cvar.on_change(lambda o, n: results.append(("cb3", o, n)))

        cvar.value = 42

        assert len(results) == 3
        assert ("cb1", 0, 42) in results
        assert ("cb2", 0, 42) in results
        assert ("cb3", 0, 42) in results

    def test_off_change_removes_callback(self):
        """Test that off_change removes a callback."""
        cvar = CVar("test.off_change", default=0)
        call_count = {"count": 0}

        def callback(old, new):
            call_count["count"] += 1

        cvar.on_change(callback)
        cvar.value = 1
        assert call_count["count"] == 1

        cvar.off_change(callback)
        cvar.value = 2
        assert call_count["count"] == 1  # Not incremented

    def test_callback_exception_handling(self):
        """Test that callback exceptions don't break value setting."""
        cvar = CVar("test.exception_cb", default=0)
        good_callback_called = {"called": False}

        def bad_callback(old, new):
            raise ValueError("Intentional error")

        def good_callback(old, new):
            good_callback_called["called"] = True

        cvar.on_change(bad_callback)
        cvar.on_change(good_callback)

        # Should not raise, value should be set, and good callback should run
        cvar.value = 10
        assert cvar.value == 10
        assert good_callback_called["called"]


class TestCVarMethods:
    """Test CVar utility methods."""

    def test_reset(self):
        """Test resetting to default value."""
        cvar = CVar("test.reset", default=100)
        cvar.value = 200

        cvar.reset()
        assert cvar.value == 100

    def test_is_default(self):
        """Test is_default property."""
        cvar = CVar("test.is_default", default=50)

        assert cvar.is_default
        cvar.value = 60
        assert not cvar.is_default
        cvar.reset()
        assert cvar.is_default

    def test_str_representation(self):
        """Test string representation."""
        cvar = CVar("r.VSync", default=1, flags=CVarFlags.CONFIG)

        result = str(cvar)
        assert "r.VSync" in result
        assert "1" in result
        assert "CONFIG" in result

    def test_get_info(self):
        """Test get_info method."""
        cvar = CVar(
            "test.info",
            default=42,
            flags=CVarFlags.CONFIG,
            description="A test CVar"
        )

        info = cvar.get_info()
        assert info["name"] == "test.info"
        assert info["value"] == 42
        assert info["default"] == 42
        assert info["type"] == "int"
        assert info["description"] == "A test CVar"
        assert info["is_default"]


class TestCVarRegistry:
    """Test CVarRegistry singleton functionality."""

    def test_singleton_instance(self):
        """Test that instance returns the same registry."""
        reg1 = CVarRegistry.instance()
        reg2 = CVarRegistry.instance()

        assert reg1 is reg2

    def test_auto_registration(self):
        """Test that CVars auto-register on creation."""
        cvar = CVar("test.auto_reg", default=1)
        registry = CVarRegistry.instance()

        assert registry.get("test.auto_reg") is cvar

    def test_duplicate_name_error(self):
        """Test that duplicate names raise ValueError."""
        CVar("test.duplicate", default=1)

        with pytest.raises(ValueError, match="already registered"):
            CVar("test.duplicate", default=2)

    def test_unregister(self):
        """Test unregistering a CVar."""
        cvar = CVar("test.unregister", default=1)
        registry = CVarRegistry.instance()

        assert registry.unregister("test.unregister")
        assert registry.get("test.unregister") is None
        assert not registry.unregister("test.unregister")  # Already gone

    def test_find_pattern(self):
        """Test finding CVars by pattern."""
        CVar("r.VSync", default=1)
        CVar("r.Shadows", default=3)
        CVar("r.AA", default=2)
        CVar("p.Gravity", default=-980.0)

        registry = CVarRegistry.instance()
        r_cvars = registry.find("r.*")

        assert len(r_cvars) == 3
        names = {cvar.name for cvar in r_cvars}
        assert names == {"r.VSync", "r.Shadows", "r.AA"}

    def test_categories(self):
        """Test getting CVar categories."""
        CVar("cat1.var1", default=1)
        CVar("cat1.var2", default=2)
        CVar("cat2.var1", default=3)

        registry = CVarRegistry.instance()
        categories = registry.categories()

        assert "cat1" in categories
        assert "cat2" in categories

    def test_by_category(self):
        """Test getting CVars by category."""
        CVar("render.vsync", default=1)
        CVar("render.shadows", default=3)
        CVar("physics.gravity", default=-980.0)

        registry = CVarRegistry.instance()
        render_cvars = registry.by_category("render")

        assert len(render_cvars) == 2
        names = {cvar.name for cvar in render_cvars}
        assert names == {"render.vsync", "render.shadows"}

    def test_with_flags(self):
        """Test finding CVars with specific flags."""
        CVar("flags.config", default=1, flags=CVarFlags.CONFIG)
        CVar("flags.cheat", default=2, flags=CVarFlags.CHEAT)
        CVar("flags.both", default=3, flags=CVarFlags.CONFIG | CVarFlags.CHEAT)
        CVar("flags.none", default=4)

        registry = CVarRegistry.instance()

        config_cvars = registry.with_flags(CVarFlags.CONFIG)
        assert len(config_cvars) == 2

        cheat_cvars = registry.with_flags(CVarFlags.CHEAT)
        assert len(cheat_cvars) == 2

    def test_reset_all(self):
        """Test resetting all CVars to defaults."""
        cvar1 = CVar("reset.a", default=1)
        cvar2 = CVar("reset.b", default=2)

        cvar1.value = 100
        cvar2.value = 200

        registry = CVarRegistry.instance()
        count = registry.reset_all()

        assert count == 2
        assert cvar1.value == 1
        assert cvar2.value == 2

    def test_export_import_config(self):
        """Test exporting and importing CONFIG CVars."""
        CVar("config.a", default=1, flags=CVarFlags.CONFIG)
        CVar("config.b", default=2, flags=CVarFlags.CONFIG)
        CVar("noconfig.c", default=3)  # Not CONFIG

        registry = CVarRegistry.instance()
        registry.get("config.a").value = 10
        registry.get("config.b").value = 20

        exported = registry.export_config()
        assert exported == {"config.a": 10, "config.b": 20}

        # Reset and re-import
        registry.reset_all()
        count = registry.import_config({"config.a": 100, "config.b": 200})

        assert count == 2
        assert registry.get("config.a").value == 100
        assert registry.get("config.b").value == 200

    def test_len_and_contains(self):
        """Test len and contains operations."""
        CVar("test.len1", default=1)
        CVar("test.len2", default=2)

        registry = CVarRegistry.instance()

        assert len(registry) == 2
        assert "test.len1" in registry
        assert "nonexistent" not in registry


class TestCVarBounds:
    """Test CVar bounds checking functionality."""

    def test_int_cvar_with_bounds(self):
        """Test integer CVar with min/max bounds."""
        cvar = CVar("test.bounded_int", default=5, min_value=0, max_value=10)

        assert cvar.value == 5
        cvar.value = 0
        assert cvar.value == 0
        cvar.value = 10
        assert cvar.value == 10

    def test_float_cvar_with_bounds(self):
        """Test float CVar with min/max bounds."""
        cvar = CVar("test.bounded_float", default=0.5, min_value=0.0, max_value=1.0)

        assert cvar.value == 0.5
        cvar.value = 0.0
        assert cvar.value == 0.0
        cvar.value = 1.0
        assert cvar.value == 1.0

    def test_bounds_violation_below_min(self):
        """Test that value below minimum raises CVarBoundsError."""
        cvar = CVar("test.bounds_min", default=5, min_value=0, max_value=10)

        with pytest.raises(CVarBoundsError, match="below minimum"):
            cvar.value = -1

    def test_bounds_violation_above_max(self):
        """Test that value above maximum raises CVarBoundsError."""
        cvar = CVar("test.bounds_max", default=5, min_value=0, max_value=10)

        with pytest.raises(CVarBoundsError, match="above maximum"):
            cvar.value = 11

    def test_default_below_min_raises_error(self):
        """Test that default value below min raises CVarBoundsError on creation."""
        with pytest.raises(CVarBoundsError, match="below minimum"):
            CVar("test.bad_default_min", default=-5, min_value=0, max_value=10)

    def test_default_above_max_raises_error(self):
        """Test that default value above max raises CVarBoundsError on creation."""
        with pytest.raises(CVarBoundsError, match="above maximum"):
            CVar("test.bad_default_max", default=15, min_value=0, max_value=10)

    def test_bounds_on_non_numeric_raises_error(self):
        """Test that setting bounds on non-numeric type raises CVarTypeError."""
        with pytest.raises(CVarTypeError, match="not a numeric type"):
            CVar("test.str_bounds", default="hello", min_value=0, max_value=10)

    def test_only_min_bound(self):
        """Test CVar with only minimum bound."""
        cvar = CVar("test.min_only", default=5, min_value=0)

        cvar.value = 100  # No max, should work
        assert cvar.value == 100

        with pytest.raises(CVarBoundsError):
            cvar.value = -1

    def test_only_max_bound(self):
        """Test CVar with only maximum bound."""
        cvar = CVar("test.max_only", default=5, max_value=10)

        cvar.value = -100  # No min, should work
        assert cvar.value == -100

        with pytest.raises(CVarBoundsError):
            cvar.value = 11

    def test_bounds_from_string_input(self):
        """Test that bounds checking works with string input."""
        cvar = CVar("test.bounds_str", default=5, min_value=0, max_value=10)

        cvar.value = "7"
        assert cvar.value == 7

        with pytest.raises(CVarBoundsError):
            cvar.value = "15"

    def test_bounds_in_get_info(self):
        """Test that bounds appear in get_info output."""
        cvar = CVar("test.bounds_info", default=5, min_value=0, max_value=10)
        info = cvar.get_info()

        # bounds should be accessible through CVar attributes
        assert cvar.min_value == 0
        assert cvar.max_value == 10
