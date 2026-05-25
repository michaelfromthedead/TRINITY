"""Tests for the CVar (Configuration Variable) system.

Tests all CVar types, validation, callbacks, and registry functionality.
"""

import pytest
import threading
from pathlib import Path
from enum import Enum
import tempfile

from engine.tooling.console.cvar_system import (
    CVar,
    CVarRegistry,
    CVarType,
    CVarFlags,
    IntCVar,
    FloatCVar,
    BoolCVar,
    StringCVar,
    EnumCVar,
    CVarChangeEvent,
)


class TestIntCVar:
    """Tests for IntCVar."""

    def test_basic_creation(self):
        cvar = IntCVar("test_int", 42, "Test integer CVar")
        assert cvar.name == "test_int"
        assert cvar.value == 42
        assert cvar.default == 42
        assert cvar.description == "Test integer CVar"
        assert cvar.cvar_type == CVarType.INT

    def test_set_value(self):
        cvar = IntCVar("test_int", 0)
        cvar.value = 100
        assert cvar.value == 100

    def test_range_validation(self):
        cvar = IntCVar("test_int", 5, min_value=0, max_value=10)
        cvar.value = 10
        assert cvar.value == 10

        with pytest.raises(ValueError, match="above maximum"):
            cvar.value = 11

        with pytest.raises(ValueError, match="below minimum"):
            cvar.value = -1

    def test_string_parsing(self):
        cvar = IntCVar("test_int", 0)
        parsed = cvar.parse("42")
        assert parsed == 42

    def test_float_conversion(self):
        cvar = IntCVar("test_int", 0)
        cvar.value = 3.7  # Should truncate to 3
        assert cvar.value == 3

    def test_boolean_rejected(self):
        cvar = IntCVar("test_int", 0)
        with pytest.raises(ValueError, match="Boolean not allowed"):
            cvar.value = True

    def test_invalid_string(self):
        cvar = IntCVar("test_int", 0)
        with pytest.raises(ValueError, match="Cannot convert"):
            cvar.parse("not_a_number")

    def test_reset_to_default(self):
        cvar = IntCVar("test_int", 42)
        cvar.value = 100
        cvar.reset()
        assert cvar.value == 42

    def test_min_max_properties(self):
        cvar = IntCVar("test", 5, min_value=0, max_value=10)
        assert cvar.min_value == 0
        assert cvar.max_value == 10

    def test_invalid_range_creation(self):
        with pytest.raises(ValueError, match="cannot be greater"):
            IntCVar("test", 5, min_value=10, max_value=5)


class TestFloatCVar:
    """Tests for FloatCVar."""

    def test_basic_creation(self):
        cvar = FloatCVar("test_float", 3.14)
        assert cvar.value == pytest.approx(3.14)
        assert cvar.cvar_type == CVarType.FLOAT

    def test_precision(self):
        cvar = FloatCVar("test_float", 0.0, precision=2)
        cvar.value = 3.14159
        assert cvar.value == pytest.approx(3.14)

    def test_range_validation(self):
        cvar = FloatCVar("test_float", 0.5, min_value=0.0, max_value=1.0)
        cvar.value = 1.0
        assert cvar.value == 1.0

        with pytest.raises(ValueError):
            cvar.value = 1.5

    def test_string_parsing(self):
        cvar = FloatCVar("test_float", 0.0)
        parsed = cvar.parse("2.718")
        assert parsed == pytest.approx(2.718)

    def test_int_conversion(self):
        cvar = FloatCVar("test_float", 0.0)
        cvar.value = 5
        assert cvar.value == 5.0


class TestBoolCVar:
    """Tests for BoolCVar."""

    def test_basic_creation(self):
        cvar = BoolCVar("test_bool", True)
        assert cvar.value is True
        assert cvar.cvar_type == CVarType.BOOL

    def test_true_string_values(self):
        cvar = BoolCVar("test_bool", False)
        for value in ["true", "1", "yes", "on", "enabled"]:
            cvar.value = value
            assert cvar.value is True

    def test_false_string_values(self):
        cvar = BoolCVar("test_bool", True)
        for value in ["false", "0", "no", "off", "disabled"]:
            cvar.value = value
            assert cvar.value is False

    def test_toggle(self):
        cvar = BoolCVar("test_bool", False)
        result = cvar.toggle()
        assert result is True
        assert cvar.value is True

        result = cvar.toggle()
        assert result is False
        assert cvar.value is False

    def test_invalid_string(self):
        cvar = BoolCVar("test_bool", False)
        with pytest.raises(ValueError, match="Cannot convert"):
            cvar.parse("maybe")

    def test_int_conversion(self):
        cvar = BoolCVar("test_bool", False)
        cvar.value = 1
        assert cvar.value is True
        cvar.value = 0
        assert cvar.value is False


class TestStringCVar:
    """Tests for StringCVar."""

    def test_basic_creation(self):
        cvar = StringCVar("test_string", "hello")
        assert cvar.value == "hello"
        assert cvar.cvar_type == CVarType.STRING

    def test_max_length(self):
        cvar = StringCVar("test_string", "", max_length=10)
        cvar.value = "short"
        assert cvar.value == "short"

        with pytest.raises(ValueError, match="exceeds maximum"):
            cvar.value = "this is too long"

    def test_allowed_values(self):
        cvar = StringCVar("test_string", "a", allowed_values=["a", "b", "c"])
        cvar.value = "b"
        assert cvar.value == "b"

        with pytest.raises(ValueError, match="not in allowed values"):
            cvar.value = "d"

    def test_pattern_validation(self):
        cvar = StringCVar("test_string", "abc", pattern=r"^[a-z]+$")
        cvar.value = "xyz"
        assert cvar.value == "xyz"

        with pytest.raises(ValueError, match="does not match pattern"):
            cvar.value = "ABC123"

    def test_any_type_conversion(self):
        cvar = StringCVar("test_string", "")
        cvar.value = 123
        assert cvar.value == "123"


class TestEnumCVar:
    """Tests for EnumCVar."""

    class SampleEnum(Enum):
        LOW = 1
        MEDIUM = 2
        HIGH = 3

    def test_basic_creation(self):
        cvar = EnumCVar("test_enum", "LOW", self.SampleEnum)
        assert cvar.value == "LOW"
        assert cvar.cvar_type == CVarType.ENUM

    def test_enum_member_default(self):
        cvar = EnumCVar("test_enum", self.SampleEnum.MEDIUM, self.SampleEnum)
        assert cvar.value == "MEDIUM"

    def test_get_enum_value(self):
        cvar = EnumCVar("test_enum", "HIGH", self.SampleEnum)
        assert cvar.get_enum_value() == self.SampleEnum.HIGH

    def test_set_by_string(self):
        cvar = EnumCVar("test_enum", "LOW", self.SampleEnum)
        cvar.value = "HIGH"
        assert cvar.value == "HIGH"

    def test_set_by_enum_member(self):
        cvar = EnumCVar("test_enum", "LOW", self.SampleEnum)
        cvar.value = self.SampleEnum.MEDIUM
        assert cvar.value == "MEDIUM"

    def test_case_insensitive(self):
        cvar = EnumCVar("test_enum", "LOW", self.SampleEnum)
        cvar.value = "high"
        assert cvar.value == "HIGH"

    def test_invalid_value(self):
        cvar = EnumCVar("test_enum", "LOW", self.SampleEnum)
        with pytest.raises(ValueError, match="not a valid member"):
            cvar.value = "INVALID"

    def test_enum_values_property(self):
        cvar = EnumCVar("test_enum", "LOW", self.SampleEnum)
        assert cvar.enum_values == frozenset({"LOW", "MEDIUM", "HIGH"})


class TestCVarFlags:
    """Tests for CVar flags."""

    def test_readonly_flag(self):
        cvar = IntCVar("readonly", 42, flags=CVarFlags.READONLY)
        with pytest.raises(PermissionError, match="readonly"):
            cvar.value = 100

    def test_readonly_reset(self):
        cvar = IntCVar("readonly", 42, flags=CVarFlags.READONLY)
        cvar.reset()  # Should not raise, but also not change
        assert cvar.value == 42

    def test_flag_combinations(self):
        flags = CVarFlags.ARCHIVE | CVarFlags.REPLICATED
        cvar = IntCVar("combined", 0, flags=flags)
        assert CVarFlags.ARCHIVE in cvar.flags
        assert CVarFlags.REPLICATED in cvar.flags
        assert CVarFlags.CHEAT not in cvar.flags


class TestCVarCallbacks:
    """Tests for CVar change callbacks."""

    def test_callback_on_change(self):
        events = []

        def callback(event: CVarChangeEvent):
            events.append(event)

        cvar = IntCVar("test", 0)
        cvar.add_callback(callback)
        cvar.value = 42

        assert len(events) == 1
        assert events[0].old_value == 0
        assert events[0].new_value == 42
        assert events[0].cvar_name == "test"

    def test_callback_not_called_on_same_value(self):
        events = []

        def callback(event):
            events.append(event)

        cvar = IntCVar("test", 42)
        cvar.add_callback(callback)
        cvar.value = 42  # Same value

        assert len(events) == 0

    def test_remove_callback(self):
        events = []

        def callback(event):
            events.append(event)

        cvar = IntCVar("test", 0)
        cvar.add_callback(callback)
        cvar.remove_callback(callback)
        cvar.value = 42

        assert len(events) == 0

    def test_callback_source(self):
        events = []

        def callback(event):
            events.append(event)

        cvar = IntCVar("test", 0)
        cvar.add_callback(callback)
        cvar.set(42, source="console")

        assert events[0].source == "console"

    def test_callback_exception_handled(self):
        def bad_callback(event):
            raise Exception("Callback error")

        cvar = IntCVar("test", 0)
        cvar.add_callback(bad_callback)

        # Should not raise
        cvar.value = 42
        assert cvar.value == 42


class TestCVarRegistry:
    """Tests for CVarRegistry."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset singleton before and after each test."""
        CVarRegistry.reset_instance()
        yield
        CVarRegistry.reset_instance()

    def test_register_and_get(self):
        registry = CVarRegistry()
        cvar = IntCVar("test_var", 42)
        registry.register(cvar)

        retrieved = registry.get("test_var")
        assert retrieved is cvar

    def test_duplicate_registration_raises(self):
        registry = CVarRegistry()
        cvar1 = IntCVar("test_var", 42)
        cvar2 = IntCVar("test_var", 100)

        registry.register(cvar1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(cvar2)

    def test_unregister(self):
        registry = CVarRegistry()
        cvar = IntCVar("test_var", 42)
        registry.register(cvar)

        removed = registry.unregister("test_var")
        assert removed is cvar
        assert registry.get("test_var") is None

    def test_get_value(self):
        registry = CVarRegistry()
        cvar = IntCVar("test_var", 42)
        registry.register(cvar)

        assert registry.get_value("test_var") == 42
        assert registry.get_value("nonexistent", 99) == 99

    def test_set_value(self):
        registry = CVarRegistry()
        cvar = IntCVar("test_var", 42)
        registry.register(cvar)

        registry.set_value("test_var", 100)
        assert cvar.value == 100

    def test_set_value_not_found(self):
        registry = CVarRegistry()
        with pytest.raises(KeyError):
            registry.set_value("nonexistent", 42)

    def test_all_cvars(self):
        registry = CVarRegistry()
        cvar1 = IntCVar("var1", 1)
        cvar2 = IntCVar("var2", 2)
        registry.register(cvar1)
        registry.register(cvar2)

        all_cvars = registry.all_cvars()
        assert len(all_cvars) == 2
        assert cvar1 in all_cvars
        assert cvar2 in all_cvars

    def test_by_category(self):
        registry = CVarRegistry()
        cvar1 = IntCVar("var1", 1, category="graphics")
        cvar2 = IntCVar("var2", 2, category="graphics")
        cvar3 = IntCVar("var3", 3, category="audio")

        registry.register(cvar1)
        registry.register(cvar2)
        registry.register(cvar3)

        graphics = registry.by_category("graphics")
        assert len(graphics) == 2
        assert cvar1 in graphics
        assert cvar2 in graphics
        assert cvar3 not in graphics

    def test_categories(self):
        registry = CVarRegistry()
        cvar1 = IntCVar("var1", 1, category="graphics")
        cvar2 = IntCVar("var2", 2, category="audio")

        registry.register(cvar1)
        registry.register(cvar2)

        categories = registry.categories()
        assert "graphics" in categories
        assert "audio" in categories

    def test_find_pattern(self):
        registry = CVarRegistry()
        cvar1 = IntCVar("r_shadows", 1)
        cvar2 = IntCVar("r_lighting", 2)
        cvar3 = IntCVar("s_volume", 3)

        registry.register(cvar1)
        registry.register(cvar2)
        registry.register(cvar3)

        matches = registry.find("r_*")
        assert len(matches) == 2
        assert cvar1 in matches
        assert cvar2 in matches

    def test_singleton(self):
        instance1 = CVarRegistry.get_instance()
        instance2 = CVarRegistry.get_instance()
        assert instance1 is instance2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cvars.json"
            registry = CVarRegistry(config_path=path)

            cvar1 = IntCVar("archived", 42, flags=CVarFlags.ARCHIVE)
            cvar2 = IntCVar("not_archived", 100)

            registry.register(cvar1)
            registry.register(cvar2)
            registry.save()

            # Create new registry and load
            registry2 = CVarRegistry(config_path=path)
            cvar1_new = IntCVar("archived", 0, flags=CVarFlags.ARCHIVE)
            cvar2_new = IntCVar("not_archived", 0)
            registry2.register(cvar1_new)
            registry2.register(cvar2_new)

            count = registry2.load()
            assert count == 1  # Only archived was saved
            assert cvar1_new.value == 42
            assert cvar2_new.value == 0  # Not loaded

    def test_reset_all(self):
        registry = CVarRegistry()
        cvar1 = IntCVar("var1", 10)
        cvar2 = IntCVar("var2", 20)

        registry.register(cvar1)
        registry.register(cvar2)

        cvar1.value = 100
        cvar2.value = 200

        registry.reset_all()

        assert cvar1.value == 10
        assert cvar2.value == 20

    def test_clear(self):
        registry = CVarRegistry()
        cvar = IntCVar("test", 42)
        registry.register(cvar)
        registry.clear()

        assert registry.get("test") is None
        assert len(registry.all_cvars()) == 0


class TestCVarThreadSafety:
    """Tests for thread-safe CVar operations."""

    def test_concurrent_value_changes(self):
        cvar = IntCVar("test", 0)
        errors = []

        def increment():
            try:
                for _ in range(100):
                    old = cvar.value
                    cvar.value = old + 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Value might not be exactly 1000 due to race conditions in increment,
        # but no crashes should occur

    def test_concurrent_registry_access(self):
        CVarRegistry.reset_instance()
        registry = CVarRegistry.get_instance()
        errors = []

        def register_cvars(start):
            try:
                for i in range(10):
                    cvar = IntCVar(f"var_{start}_{i}", i)
                    registry.register(cvar)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_cvars, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.all_cvars()) == 50


class TestCVarStringRepresentation:
    """Tests for CVar string representations."""

    def test_str(self):
        cvar = IntCVar("test_var", 42)
        assert str(cvar) == "test_var = 42"

    def test_repr(self):
        cvar = IntCVar("test_var", 42)
        repr_str = repr(cvar)
        assert "IntCVar" in repr_str
        assert "test_var" in repr_str
        assert "42" in repr_str


class TestCVarEmptyName:
    """Tests for CVar name validation."""

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            IntCVar("", 42)

    def test_whitespace_name_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            IntCVar("   ", 42)

    def test_name_stripped(self):
        cvar = IntCVar("  test  ", 42)
        assert cvar.name == "test"
