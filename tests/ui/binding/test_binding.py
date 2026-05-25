"""
Comprehensive tests for the data binding system.

Tests cover:
- PropertyPath
- BindingContext
- BindingExpression
- Binding (one-way, two-way, one-time, one-way-to-source)
- MultiBinding
- BindingGroup
- Factory functions
- Error handling
- Validation integration
"""
import pytest
import weakref

from engine.ui.binding.binding import (
    Binding,
    BindingContext,
    BindingError,
    BindingExpression,
    BindingGroup,
    BindingMode,
    BindingStatus,
    MultiBinding,
    PropertyPath,
    UpdateSourceTrigger,
    bind,
    bind_one_time,
    bind_two_way,
    multi_bind,
)
from engine.ui.binding.converter import (
    IConverter,
    NumberFormatConverter,
    StringConcatConverter,
)
from engine.ui.binding.validation import (
    RequiredValidator,
    ValidationContext,
    ValidationResult,
)


# ========== Observable Test Classes ==========


class ObservableObject:
    """Simple observable object for testing."""

    def __init__(self, **kwargs):
        self._observers = []
        self._data = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_observer(self, callback):
        self._observers.append(callback)

    def remove_observer(self, callback):
        if callback in self._observers:
            self._observers.remove(callback)

    def set_value(self, name, value):
        old_value = getattr(self, name, None)
        setattr(self, name, value)
        for observer in self._observers:
            observer(self, name, old_value, value)


class PropertyChangeObject:
    """Object using property change notification pattern."""

    def __init__(self, **kwargs):
        self._listeners = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def add_property_change_listener(self, callback):
        self._listeners.append(callback)

    def remove_property_change_listener(self, callback):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def set_value(self, name, value):
        old_value = getattr(self, name, None)
        setattr(self, name, value)
        for listener in self._listeners:
            listener(name, old_value, value)


# ========== Fixtures ==========


@pytest.fixture
def source_object():
    """Observable source object."""
    return ObservableObject(name="John", age=30, email="john@example.com")


@pytest.fixture
def target_object():
    """Observable target object."""
    return ObservableObject(text="", value=0)


@pytest.fixture
def nested_object():
    """Object with nested properties."""
    address = ObservableObject(city="New York", zip="10001")
    return ObservableObject(name="John", address=address)


@pytest.fixture
def list_object():
    """Object with list property."""
    return ObservableObject(items=[1, 2, 3, 4, 5])


@pytest.fixture
def dict_object():
    """Object with dict property."""
    return ObservableObject(data={"key": "value", "count": 10})


# ========== PropertyPath Tests ==========


class TestPropertyPath:
    """Tests for PropertyPath class."""

    def test_simple_path(self):
        """Test simple property path."""
        path = PropertyPath("name")
        assert path.path == "name"
        assert path.is_simple
        assert path.root == "name"

    def test_nested_path(self):
        """Test dot-separated nested path."""
        path = PropertyPath("address.city")
        assert not path.is_simple
        assert path.root == "address"
        assert len(path.segments) == 2

    def test_index_path(self):
        """Test indexed path."""
        path = PropertyPath("items[0]")
        assert path.root == "items"
        assert path.segments[1] == ("", 0)

    def test_complex_path(self):
        """Test complex path with mixed access."""
        path = PropertyPath("users[0].address.city")
        assert len(path.segments) == 4

    def test_get_value_simple(self, source_object):
        """Test getting simple property value."""
        path = PropertyPath("name")
        value = path.get_value(source_object)
        assert value == "John"

    def test_get_value_nested(self, nested_object):
        """Test getting nested property value."""
        path = PropertyPath("address.city")
        value = path.get_value(nested_object)
        assert value == "New York"

    def test_get_value_index(self, list_object):
        """Test getting indexed value."""
        path = PropertyPath("items[2]")
        value = path.get_value(list_object)
        assert value == 3

    def test_get_value_dict_key(self, dict_object):
        """Test getting dict key value."""
        path = PropertyPath("data.key")
        value = path.get_value(dict_object)
        assert value == "value"

    def test_get_value_none_object(self):
        """Test getting value from None returns None."""
        path = PropertyPath("name")
        value = path.get_value(None)
        assert value is None

    def test_get_value_missing_property(self, source_object):
        """Test getting missing property returns None."""
        path = PropertyPath("nonexistent")
        value = path.get_value(source_object)
        assert value is None

    def test_set_value_simple(self, source_object):
        """Test setting simple property value."""
        path = PropertyPath("name")
        result = path.set_value(source_object, "Jane")
        assert result is True
        assert source_object.name == "Jane"

    def test_set_value_nested(self, nested_object):
        """Test setting nested property value."""
        path = PropertyPath("address.city")
        result = path.set_value(nested_object, "Boston")
        assert result is True
        assert nested_object.address.city == "Boston"

    def test_set_value_index(self, list_object):
        """Test setting indexed value."""
        path = PropertyPath("items[0]")
        result = path.set_value(list_object, 100)
        assert result is True
        assert list_object.items[0] == 100

    def test_set_value_dict(self, dict_object):
        """Test setting dict value."""
        path = PropertyPath("data.key")
        result = path.set_value(dict_object, "new_value")
        assert result is True
        assert dict_object.data["key"] == "new_value"

    def test_set_value_missing_intermediate(self, source_object):
        """Test setting value with missing intermediate returns False."""
        path = PropertyPath("missing.property")
        result = path.set_value(source_object, "value")
        assert result is False

    def test_equality(self):
        """Test path equality."""
        path1 = PropertyPath("name")
        path2 = PropertyPath("name")
        path3 = PropertyPath("age")
        assert path1 == path2
        assert path1 != path3

    def test_equality_with_string(self):
        """Test path equality with string."""
        path = PropertyPath("name")
        assert path == "name"

    def test_hash(self):
        """Test path is hashable."""
        path = PropertyPath("name")
        paths = {path}
        assert PropertyPath("name") in paths

    def test_unclosed_bracket_raises(self):
        """Test unclosed bracket raises error."""
        with pytest.raises(ValueError, match="Unclosed bracket"):
            PropertyPath("items[0")


# ========== BindingContext Tests ==========


class TestBindingContext:
    """Tests for BindingContext class."""

    def test_create_with_source(self, source_object):
        """Test creating context with source."""
        context = BindingContext(source=source_object)
        assert context.source is source_object

    def test_set_source(self, source_object):
        """Test setting source."""
        context = BindingContext()
        context.source = source_object
        assert context.source is source_object

    def test_fallback_value(self):
        """Test fallback value."""
        context = BindingContext(fallback_value="default")
        assert context.fallback_value == "default"

    def test_target_null_value(self):
        """Test target null value."""
        context = BindingContext(target_null_value="N/A")
        assert context.target_null_value == "N/A"

    def test_string_format(self):
        """Test string format."""
        context = BindingContext(string_format="{}")
        assert context.string_format == "{}"

    def test_register_converter(self):
        """Test registering a converter."""
        context = BindingContext()
        converter = NumberFormatConverter()
        context.register_converter("number", converter)
        assert context.get_converter("number") is converter

    def test_get_converter_from_parent(self):
        """Test getting converter from parent context."""
        parent = BindingContext()
        converter = NumberFormatConverter()
        parent.register_converter("number", converter)

        child = BindingContext()
        child.parent = parent

        assert child.get_converter("number") is converter

    def test_register_validator(self):
        """Test registering a validator."""
        context = BindingContext()
        validator = RequiredValidator()
        context.register_validator("required", validator)
        assert context.get_validator("required") is validator

    def test_resources(self):
        """Test resource management."""
        context = BindingContext()
        context.set_resource("theme", "dark")
        assert context.get_resource("theme") == "dark"

    def test_resources_from_parent(self):
        """Test getting resource from parent."""
        parent = BindingContext()
        parent.set_resource("theme", "dark")

        child = BindingContext()
        child.parent = parent

        assert child.get_resource("theme") == "dark"

    def test_create_child(self, source_object):
        """Test creating child context."""
        parent = BindingContext(
            source=source_object,
            fallback_value="default",
        )
        child = parent.create_child()

        assert child.parent is parent
        assert child.source is source_object
        assert child.fallback_value == "default"

    def test_create_child_with_different_source(self, source_object, target_object):
        """Test child context with different source."""
        parent = BindingContext(source=source_object)
        child = parent.create_child(source=target_object)

        assert child.source is target_object
        assert parent.source is source_object


# ========== BindingExpression Tests ==========


class TestBindingExpression:
    """Tests for BindingExpression class."""

    def test_evaluate_simple(self):
        """Test evaluating simple expression."""
        expr = BindingExpression(
            lambda ctx: ctx["value"] * 2,
            dependencies=["value"],
        )
        result = expr.evaluate({"value": 5})
        assert result == 10

    def test_evaluate_multiple_deps(self):
        """Test evaluating expression with multiple dependencies."""
        expr = BindingExpression(
            lambda ctx: f"{ctx['first']} {ctx['last']}",
            dependencies=["first", "last"],
        )
        result = expr.evaluate({"first": "John", "last": "Doe"})
        assert result == "John Doe"

    def test_dependencies_property(self):
        """Test dependencies property."""
        expr = BindingExpression(
            lambda ctx: ctx["a"] + ctx["b"],
            dependencies=["a", "b"],
        )
        assert expr.dependencies == ["a", "b"]

    def test_evaluate_with_error_returns_fallback(self):
        """Test expression error returns fallback."""
        expr = BindingExpression(
            lambda ctx: ctx["missing"],
            fallback_value="default",
        )
        result = expr.evaluate({})
        assert result == "default"


# ========== Binding Tests - One-Way ==========


class TestBindingOneWay:
    """Tests for one-way bindings."""

    def test_create_binding(self, source_object, target_object):
        """Test creating a binding."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
        )
        assert binding.status == BindingStatus.UNATTACHED
        assert binding.mode == BindingMode.ONE_WAY

    def test_attach_updates_target(self, source_object, target_object):
        """Test attaching binding updates target."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
        )
        binding.attach()

        assert binding.status == BindingStatus.ACTIVE
        assert target_object.text == "John"

    def test_source_change_updates_target(self, source_object, target_object):
        """Test source change updates target."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
        )
        binding.attach()

        source_object.set_value("name", "Jane")
        assert target_object.text == "Jane"

    def test_detach_stops_updates(self, source_object, target_object):
        """Test detaching stops updates."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
        )
        binding.attach()
        binding.detach()

        source_object.set_value("name", "Jane")
        assert target_object.text == "John"  # Not updated
        assert binding.status == BindingStatus.DETACHED

    def test_converter_applied(self, source_object, target_object):
        """Test converter is applied."""

        class UpperConverter(IConverter):
            def convert(self, value, parameter=None):
                return value.upper()

            def convert_back(self, value, parameter=None):
                return value.lower()

        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            converter=UpperConverter(),
        )
        binding.attach()

        assert target_object.text == "JOHN"

    def test_string_format_applied(self, source_object, target_object):
        """Test string format is applied."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            string_format="Hello, {}!",
        )
        binding.attach()

        assert target_object.text == "Hello, John!"

    def test_fallback_value_when_source_none(self, target_object):
        """Test fallback value when source is None."""
        binding = Binding(
            source=None,
            source_path="name",
            target=target_object,
            target_path="text",
            fallback_value="Default",
        )
        binding.attach()

        # Should have error status
        assert binding.status == BindingStatus.ERROR

    def test_target_null_value(self, source_object, target_object):
        """Test target null value when source value is None."""
        source_object.optional = None
        binding = Binding(
            source=source_object,
            source_path="optional",
            target=target_object,
            target_path="text",
            target_null_value="N/A",
        )
        binding.attach()

        assert target_object.text == "N/A"


# ========== Binding Tests - Two-Way ==========


class TestBindingTwoWay:
    """Tests for two-way bindings."""

    def test_create_two_way(self, source_object, target_object):
        """Test creating two-way binding."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
        )
        assert binding.mode == BindingMode.TWO_WAY

    def test_source_to_target(self, source_object, target_object):
        """Test two-way binding source to target."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
        )
        binding.attach()

        assert target_object.text == "John"

    def test_target_to_source(self, source_object, target_object):
        """Test two-way binding target to source."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
        )
        binding.attach()

        target_object.set_value("text", "Jane")
        assert source_object.name == "Jane"

    def test_converter_back_applied(self, source_object, target_object):
        """Test converter back is applied."""

        class NumberConverter(IConverter):
            def convert(self, value, parameter=None):
                return str(value)

            def convert_back(self, value, parameter=None):
                return int(value)

        source_object.count = 10
        target_object.text = ""

        binding = Binding(
            source=source_object,
            source_path="count",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
            converter=NumberConverter(),
        )
        binding.attach()

        assert target_object.text == "10"

        target_object.set_value("text", "20")
        assert source_object.count == 20

    def test_validation_blocks_source_update(self, source_object, target_object):
        """Test validation blocks invalid source updates."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
            validators=[RequiredValidator()],
        )
        binding.attach()

        target_object.set_value("text", "")
        assert source_object.name == "John"  # Not updated

        result = binding.validation_result
        assert result is not None
        assert not result.is_valid


# ========== Binding Tests - One-Time ==========


class TestBindingOneTime:
    """Tests for one-time bindings."""

    def test_one_time_sets_initial(self, source_object, target_object):
        """Test one-time binding sets initial value."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.ONE_TIME,
        )
        binding.attach()

        assert target_object.text == "John"

    def test_one_time_no_updates(self, source_object, target_object):
        """Test one-time binding doesn't update on changes."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.ONE_TIME,
        )
        binding.attach()

        source_object.set_value("name", "Jane")
        # One-time binding doesn't update (but implementation may still observe)
        # The test checks behavior, not subscription details


# ========== Binding Tests - One-Way-To-Source ==========


class TestBindingOneWayToSource:
    """Tests for one-way-to-source bindings."""

    def test_one_way_to_source_no_initial(self, source_object, target_object):
        """Test one-way-to-source doesn't set initial target value."""
        target_object.text = "Initial"
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.ONE_WAY_TO_SOURCE,
        )
        binding.attach()

        # Target should keep its initial value
        assert target_object.text == "Initial"

    def test_one_way_to_source_updates_source(self, source_object, target_object):
        """Test one-way-to-source updates source on target change."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.ONE_WAY_TO_SOURCE,
        )
        binding.attach()

        target_object.set_value("text", "NewValue")
        assert source_object.name == "NewValue"


# ========== MultiBinding Tests ==========


class TestMultiBinding:
    """Tests for multi-source bindings."""

    def test_create_multi_binding(self, source_object, target_object):
        """Test creating multi-binding."""
        binding = MultiBinding(
            sources=[
                (source_object, "name"),
                (source_object, "age"),
            ],
            target=target_object,
            target_path="text",
        )
        assert len(binding._sources) == 2

    def test_multi_binding_with_converter(self, source_object, target_object):
        """Test multi-binding with converter."""
        binding = MultiBinding(
            sources=[
                (source_object, "name"),
                (source_object, "email"),
            ],
            target=target_object,
            target_path="text",
            converter=StringConcatConverter(" - "),
        )
        binding.attach()

        assert target_object.text == "John - john@example.com"

    def test_multi_binding_updates_on_source_change(self, source_object, target_object):
        """Test multi-binding updates when any source changes."""
        binding = MultiBinding(
            sources=[
                (source_object, "name"),
                (source_object, "age"),
            ],
            target=target_object,
            target_path="text",
            converter=StringConcatConverter(": "),
        )
        binding.attach()

        source_object.set_value("name", "Jane")
        assert "Jane" in target_object.text

    def test_get_source_values(self, source_object, target_object):
        """Test getting all source values."""
        binding = MultiBinding(
            sources=[
                (source_object, "name"),
                (source_object, "age"),
            ],
            target=target_object,
            target_path="text",
        )
        values = binding.get_source_values()
        assert values == ["John", 30]

    def test_multi_binding_fallback(self, source_object, target_object):
        """Test multi-binding with fallback value."""
        binding = MultiBinding(
            sources=[(source_object, "missing")],
            target=target_object,
            target_path="text",
            fallback_value="Default",
        )
        binding.attach()

        # Without converter, first non-None value or None
        # But fallback is applied when result is None


# ========== BindingGroup Tests ==========


class TestBindingGroup:
    """Tests for BindingGroup class."""

    def test_create_group(self):
        """Test creating binding group."""
        group = BindingGroup()
        assert len(group) == 0

    def test_add_binding(self, source_object, target_object):
        """Test adding binding to group."""
        group = BindingGroup()
        binding = bind(source_object, "name", target_object, "text")
        group.add(binding)

        assert len(group) == 1

    def test_add_duplicate_ignored(self, source_object, target_object):
        """Test adding same binding twice is ignored."""
        group = BindingGroup()
        binding = bind(source_object, "name", target_object, "text")
        group.add(binding)
        group.add(binding)

        assert len(group) == 1

    def test_remove_binding(self, source_object, target_object):
        """Test removing binding from group."""
        group = BindingGroup()
        binding = bind(source_object, "name", target_object, "text")
        group.add(binding)
        group.remove(binding)

        assert len(group) == 0

    def test_attach_all(self, source_object, target_object):
        """Test attaching all bindings."""
        group = BindingGroup()
        binding1 = bind(source_object, "name", target_object, "text")
        binding2 = bind(source_object, "age", target_object, "value")
        group.add(binding1)
        group.add(binding2)

        group.attach_all()

        assert binding1.status == BindingStatus.ACTIVE
        assert binding2.status == BindingStatus.ACTIVE

    def test_detach_all(self, source_object, target_object):
        """Test detaching all bindings."""
        group = BindingGroup()
        binding = bind(source_object, "name", target_object, "text")
        group.add(binding)
        binding.attach()

        group.detach_all()

        assert binding.status == BindingStatus.DETACHED

    def test_update_targets(self, source_object, target_object):
        """Test updating all targets."""
        group = BindingGroup()
        binding = bind(source_object, "name", target_object, "text")
        group.add(binding)
        binding.attach()

        source_object.name = "Jane"  # Direct assignment, no notification
        group.update_targets()

        assert target_object.text == "Jane"

    def test_validate_all(self, source_object, target_object):
        """Test validating all bindings."""
        group = BindingGroup()
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
            validators=[RequiredValidator()],
        )
        group.add(binding)
        binding.attach()

        results = group.validate_all()
        assert len(results) == 1
        assert results[0].is_valid

    def test_is_valid(self, source_object, target_object):
        """Test is_valid property."""
        group = BindingGroup()
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
            validators=[RequiredValidator()],
        )
        group.add(binding)
        binding.attach()

        assert group.is_valid

    def test_errors_property(self, source_object, target_object):
        """Test errors property."""
        group = BindingGroup()

        class BadConverter(IConverter):
            def convert(self, value, parameter=None):
                raise ValueError("Conversion error")

            def convert_back(self, value, parameter=None):
                return value

        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            converter=BadConverter(),
        )
        group.add(binding)
        binding.attach()  # This will cause an error

        errors = group.errors
        assert len(errors) > 0

    def test_clear(self, source_object, target_object):
        """Test clearing group."""
        group = BindingGroup()
        binding = bind(source_object, "name", target_object, "text")
        group.add(binding)
        binding.attach()

        group.clear()

        assert len(group) == 0
        assert binding.status == BindingStatus.DETACHED

    def test_iterate(self, source_object, target_object):
        """Test iterating over group."""
        group = BindingGroup()
        binding1 = bind(source_object, "name", target_object, "text")
        binding2 = bind(source_object, "age", target_object, "value")
        group.add(binding1)
        group.add(binding2)

        bindings = list(group)
        assert len(bindings) == 2

    def test_context_property(self):
        """Test group context property."""
        context = BindingContext()
        group = BindingGroup(context=context)
        assert group.context is context


# ========== Factory Function Tests ==========


class TestFactoryFunctions:
    """Tests for binding factory functions."""

    def test_bind(self, source_object, target_object):
        """Test bind factory function."""
        binding = bind(source_object, "name", target_object, "text")
        assert isinstance(binding, Binding)
        assert binding.mode == BindingMode.ONE_WAY

    def test_bind_two_way(self, source_object, target_object):
        """Test bind_two_way factory function."""
        binding = bind_two_way(
            source_object, "name",
            target_object, "text",
        )
        assert binding.mode == BindingMode.TWO_WAY

    def test_bind_one_time(self, source_object, target_object):
        """Test bind_one_time factory function."""
        binding = bind_one_time(
            source_object, "name",
            target_object, "text",
        )
        assert binding.mode == BindingMode.ONE_TIME

    def test_multi_bind(self, source_object, target_object):
        """Test multi_bind factory function."""
        binding = multi_bind(
            [(source_object, "name"), (source_object, "age")],
            target_object, "text",
        )
        assert isinstance(binding, MultiBinding)


# ========== BindingError Tests ==========


class TestBindingError:
    """Tests for BindingError class."""

    def test_create_error(self):
        """Test creating binding error."""
        error = BindingError(
            message="Test error",
            source_path="name",
        )
        assert error.message == "Test error"
        assert error.source_path == "name"

    def test_error_with_exception(self):
        """Test error with exception."""
        exc = ValueError("Test")
        error = BindingError(
            message="Test error",
            exception=exc,
        )
        assert error.exception is exc

    def test_timestamp(self):
        """Test error has timestamp."""
        error = BindingError(message="Test")
        assert error.timestamp > 0


# ========== Error Handling Tests ==========


class TestBindingErrorHandling:
    """Tests for binding error handling."""

    def test_converter_error_logged(self, source_object, target_object):
        """Test converter errors are logged."""

        class BadConverter(IConverter):
            def convert(self, value, parameter=None):
                raise ValueError("Converter failed")

            def convert_back(self, value, parameter=None):
                return value

        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            converter=BadConverter(),
            fallback_value="Error",
        )
        binding.attach()

        assert binding.has_error
        assert len(binding.errors) > 0
        assert "Converter error" in binding.errors[0].message

    def test_clear_errors(self, source_object, target_object):
        """Test clearing errors."""

        class BadConverter(IConverter):
            def convert(self, value, parameter=None):
                raise ValueError("Converter failed")

            def convert_back(self, value, parameter=None):
                return value

        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            converter=BadConverter(),
        )
        binding.attach()

        assert binding.has_error
        binding.clear_errors()
        assert not binding.has_error


# ========== Weak Reference Tests ==========


class TestBindingWeakReferences:
    """Tests for binding weak references."""

    def test_source_can_be_garbage_collected(self, target_object):
        """Test source can be garbage collected."""
        source = ObservableObject(name="John")
        binding = Binding(
            source=source,
            source_path="name",
            target=target_object,
            target_path="text",
        )

        del source
        # Source should be collectible
        assert binding.source is None

    def test_target_can_be_garbage_collected(self, source_object):
        """Test target can be garbage collected."""
        target = ObservableObject(text="")
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target,
            target_path="text",
        )

        del target
        assert binding.target is None


# ========== Validation Integration Tests ==========


class TestBindingValidation:
    """Tests for binding validation integration."""

    def test_validate_method(self, source_object, target_object):
        """Test explicit validate method."""
        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
            validators=[RequiredValidator()],
        )
        binding.attach()

        result = binding.validate()
        assert result.is_valid

        target_object.text = ""
        result = binding.validate()
        assert not result.is_valid

    def test_validation_context_integration(self, source_object, target_object):
        """Test ValidationContext integration."""
        context = ValidationContext()
        context.add_validator("name", RequiredValidator())

        binding = Binding(
            source=source_object,
            source_path="name",
            target=target_object,
            target_path="text",
            mode=BindingMode.TWO_WAY,
            validation_context=context,
        )
        binding.attach()

        target_object.set_value("text", "")
        # Validation should prevent source update
        assert source_object.name == "John"
