"""Tests for DataBoundDescriptor for runtime data binding (T-CC-1.8)."""
import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from engine.core.data_binding import (
    BindingContext,
    BindingError,
    BindingManager,
    BindingMode,
    BindingState,
    ComputedBinding,
    DataBoundDescriptor,
    DataSource,
    DataSourceRegistry,
    DictDataSource,
    JsonFileSource,
    bound,
    computed,
    with_bindings,
)


class TestBindingMode:
    """Tests for BindingMode enum."""

    def test_all_modes_exist(self):
        modes = [
            BindingMode.ONE_WAY,
            BindingMode.TWO_WAY,
            BindingMode.ONE_TIME,
        ]
        assert len(modes) == 3


class TestBindingState:
    """Tests for BindingState enum."""

    def test_all_states_exist(self):
        states = [
            BindingState.UNBOUND,
            BindingState.BOUND,
            BindingState.ERROR,
            BindingState.STALE,
        ]
        assert len(states) == 4


class TestBindingContext:
    """Tests for BindingContext dataclass."""

    def test_default_values(self):
        ctx = BindingContext(source_path="config", key_path="app.name")
        assert ctx.source_path == "config"
        assert ctx.key_path == "app.name"
        assert ctx.mode == BindingMode.ONE_WAY
        assert ctx.default is None

    def test_custom_values(self):
        ctx = BindingContext(
            source_path="settings",
            key_path="debug",
            mode=BindingMode.TWO_WAY,
            default=False,
        )
        assert ctx.mode == BindingMode.TWO_WAY
        assert ctx.default is False


class TestBindingError:
    """Tests for BindingError dataclass."""

    def test_error_str(self):
        error = BindingError(source="config", key="name", message="Not found")
        assert str(error) == "[config:name] Not found"


class TestDictDataSource:
    """Tests for DictDataSource."""

    def test_init_empty(self):
        source = DictDataSource()
        assert source.data == {}

    def test_init_with_data(self):
        source = DictDataSource({"key": "value"})
        assert source.data["key"] == "value"

    def test_get_simple(self):
        source = DictDataSource({"name": "test"})
        value, found = source.get("name")
        assert found
        assert value == "test"

    def test_get_nested(self):
        source = DictDataSource({"app": {"config": {"debug": True}}})
        value, found = source.get("app.config.debug")
        assert found
        assert value is True

    def test_get_not_found(self):
        source = DictDataSource({})
        value, found = source.get("missing")
        assert not found
        assert value is None

    def test_get_list_index(self):
        source = DictDataSource({"items": ["a", "b", "c"]})
        value, found = source.get("items.1")
        assert found
        assert value == "b"

    def test_get_list_index_out_of_range(self):
        source = DictDataSource({"items": ["a"]})
        value, found = source.get("items.5")
        assert not found

    def test_set_simple(self):
        source = DictDataSource()
        result = source.set("name", "test")
        assert result
        value, found = source.get("name")
        assert found
        assert value == "test"

    def test_set_nested(self):
        source = DictDataSource()
        result = source.set("app.config.debug", True)
        assert result
        value, found = source.get("app.config.debug")
        assert found
        assert value is True

    def test_set_empty_path(self):
        source = DictDataSource()
        result = source.set("", "value")
        assert not result

    def test_watch_callback(self):
        source = DictDataSource({"name": "old"})
        received = []

        def callback(value):
            received.append(value)

        source.watch("name", callback)
        source.set("name", "new")

        assert len(received) == 1
        assert received[0] == "new"

    def test_unwatch(self):
        source = DictDataSource({"name": "old"})
        received = []

        def callback(value):
            received.append(value)

        source.watch("name", callback)
        source.unwatch("name", callback)
        source.set("name", "new")

        assert len(received) == 0

    def test_update(self):
        source = DictDataSource({"a": 1})
        received = []

        def callback(value):
            received.append(value)

        source.watch("a", callback)
        source.update({"a": 2, "b": 3})

        assert source.data["a"] == 2
        assert source.data["b"] == 3
        assert len(received) == 1


class TestJsonFileSource:
    """Tests for JsonFileSource."""

    def test_load_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{"name": "test", "value": 42}')

            source = JsonFileSource(path)
            result = source.load()

            assert result
            value, found = source.get("name")
            assert found
            assert value == "test"

    def test_load_missing(self):
        source = JsonFileSource("/nonexistent.json")
        result = source.load()
        assert not result

    def test_load_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("{invalid}")

            source = JsonFileSource(path)
            result = source.load()
            assert not result

    def test_get_auto_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{"key": "value"}')

            source = JsonFileSource(path)
            value, found = source.get("key")

            assert found
            assert value == "value"

    def test_set_and_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{}')

            source = JsonFileSource(path)
            source.load()
            source.set("name", "updated")

            # Verify saved
            with open(path) as f:
                data = json.load(f)
            assert data["name"] == "updated"

    def test_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text('{"v": 1}')

            source = JsonFileSource(path)
            source.load()

            received = []
            source.watch("v", lambda x: received.append(x))

            path.write_text('{"v": 2}')
            source.reload()

            assert len(received) == 1
            assert received[0] == 2


class TestDataSourceRegistry:
    """Tests for DataSourceRegistry singleton."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_singleton(self):
        r1 = DataSourceRegistry.get_instance()
        r2 = DataSourceRegistry.get_instance()
        assert r1 is r2

    def test_register_and_get(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"key": "value"})

        registry.register("test", source)
        result = registry.get("test")

        assert result is source

    def test_get_nonexistent(self):
        registry = DataSourceRegistry.get_instance()
        result = registry.get("missing")
        assert result is None

    def test_unregister(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource()
        registry.register("test", source)

        result = registry.unregister("test")
        assert result

        assert registry.get("test") is None

    def test_unregister_nonexistent(self):
        registry = DataSourceRegistry.get_instance()
        result = registry.unregister("missing")
        assert not result

    def test_list_sources(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("a", DictDataSource())
        registry.register("b", DictDataSource())

        sources = registry.list_sources()
        assert "a" in sources
        assert "b" in sources

    def test_clear(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("test", DictDataSource())
        registry.clear()

        assert len(registry.list_sources()) == 0


class TestDataBoundDescriptor:
    """Tests for DataBoundDescriptor."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_basic_binding(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"name": "test"}))

        class MyClass:
            name = DataBoundDescriptor("config", "name", default="default")

        obj = MyClass()
        assert obj.name == "test"

    def test_default_when_not_found(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({}))

        class MyClass:
            name = DataBoundDescriptor("config", "name", default="default")

        obj = MyClass()
        assert obj.name == "default"

    def test_default_when_no_source(self):
        class MyClass:
            name = DataBoundDescriptor("missing", "name", default="default")

        obj = MyClass()
        assert obj.name == "default"

    def test_transform(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"value": "42"}))

        class MyClass:
            value = DataBoundDescriptor("config", "value", transform=int)

        obj = MyClass()
        assert obj.value == 42
        assert isinstance(obj.value, int)

    def test_one_way_updates(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"name": "initial"})
        registry.register("config", source)

        class MyClass:
            name = DataBoundDescriptor(
                "config", "name", mode=BindingMode.ONE_WAY
            )

        obj = MyClass()
        assert obj.name == "initial"

        source.set("name", "updated")
        assert obj.name == "updated"

    def test_two_way_write_back(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"name": "initial"})
        registry.register("config", source)

        class MyClass:
            name = DataBoundDescriptor(
                "config", "name", mode=BindingMode.TWO_WAY
            )

        obj = MyClass()
        obj.name = "changed"

        value, _ = source.get("name")
        assert value == "changed"

    def test_one_time_no_update(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"name": "initial"})
        registry.register("config", source)

        class MyClass:
            name = DataBoundDescriptor(
                "config", "name", mode=BindingMode.ONE_TIME
            )

        obj = MyClass()
        obj.name = "changed"  # Should be ignored

        assert obj.name == "initial"

    def test_validation_pass(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"age": 25}))

        class MyClass:
            age = DataBoundDescriptor(
                "config", "age",
                mode=BindingMode.TWO_WAY,
                validate=lambda x: x >= 0
            )

        obj = MyClass()
        obj.age = 30
        assert obj.age == 30

    def test_validation_fail(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"age": 25}))

        class MyClass:
            age = DataBoundDescriptor(
                "config", "age",
                mode=BindingMode.TWO_WAY,
                validate=lambda x: x >= 0
            )

        obj = MyClass()
        with pytest.raises(ValueError):
            obj.age = -5

    def test_inverse_transform(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"value": 10})
        registry.register("config", source)

        class MyClass:
            value = DataBoundDescriptor(
                "config", "value",
                mode=BindingMode.TWO_WAY,
                transform=lambda x: x * 2,  # Source 10 -> 20
                inverse_transform=lambda x: x // 2,  # Write 30 -> 15
            )

        obj = MyClass()
        assert obj.value == 20

        obj.value = 30
        raw, _ = source.get("value")
        assert raw == 15

    def test_get_state(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"name": "test"}))

        class MyClass:
            name = DataBoundDescriptor("config", "name")

        desc = MyClass.__dict__["name"]
        obj = MyClass()
        _ = obj.name  # Trigger binding

        state = desc.get_state(obj)
        assert state == BindingState.BOUND

    def test_refresh(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"value": 1})
        registry.register("config", source)

        class MyClass:
            value = DataBoundDescriptor("config", "value")

        desc = MyClass.__dict__["value"]
        obj = MyClass()
        assert obj.value == 1

        source._data["value"] = 99  # Direct modify without notify

        desc.refresh(obj)
        assert obj.value == 99

    def test_multiple_instances(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"value": 100})
        registry.register("config", source)

        class MyClass:
            value = DataBoundDescriptor(
                "config", "value", mode=BindingMode.TWO_WAY
            )

        obj1 = MyClass()
        obj2 = MyClass()

        assert obj1.value == 100
        assert obj2.value == 100

        obj1.value = 200
        # Both should see update since source changed
        assert obj2.value == 200


class TestBoundHelper:
    """Tests for bound() helper function."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_bound_creates_descriptor(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"name": "test"}))

        class MyClass:
            name = bound("config", "name", default="default")

        obj = MyClass()
        assert obj.name == "test"

    def test_bound_with_options(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"value": "42"}))

        class MyClass:
            value = bound(
                "config", "value",
                mode=BindingMode.TWO_WAY,
                transform=int,
                inverse_transform=str,
            )

        obj = MyClass()
        assert obj.value == 42


class TestBindingManager:
    """Tests for BindingManager."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_discover_bindings(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"a": 1, "b": 2}))

        class MyClass:
            a = bound("config", "a")
            b = bound("config", "b")
            c = "not a binding"

        obj = MyClass()
        manager = BindingManager(obj)

        assert "a" in manager.bindings
        assert "b" in manager.bindings
        assert "c" not in manager.bindings

    def test_refresh_all(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"a": 1, "b": 2})
        registry.register("config", source)

        class MyClass:
            a = bound("config", "a")
            b = bound("config", "b")

        obj = MyClass()
        manager = BindingManager(obj)

        source._data["a"] = 10
        source._data["b"] = 20

        results = manager.refresh_all()
        assert results["a"]
        assert results["b"]
        assert obj.a == 10
        assert obj.b == 20

    def test_get_states(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"a": 1}))

        class MyClass:
            a = bound("config", "a")
            b = bound("config", "missing")

        obj = MyClass()
        _ = obj.a
        _ = obj.b

        manager = BindingManager(obj)
        states = manager.get_states()

        assert states["a"] == BindingState.BOUND
        assert states["b"] == BindingState.ERROR


class TestWithBindingsDecorator:
    """Tests for @with_bindings decorator."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_adds_binding_manager(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"name": "test"}))

        @with_bindings
        class Config:
            name = bound("config", "name")

        obj = Config()
        assert hasattr(obj, 'bindings')
        assert isinstance(obj.bindings, BindingManager)

    def test_preserves_init(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"name": "test"}))

        @with_bindings
        class Config:
            name = bound("config", "name")

            def __init__(self, extra: str = "default"):
                self.extra = extra

        obj = Config("custom")
        assert obj.extra == "custom"
        assert obj.name == "test"


class TestComputedBinding:
    """Tests for ComputedBinding."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_computed_from_bindings(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"width": 10, "height": 5}))

        width_desc = DataBoundDescriptor("config", "width")
        height_desc = DataBoundDescriptor("config", "height")

        class Rectangle:
            width = width_desc
            height = height_desc
            area = ComputedBinding(lambda w, h: w * h, width_desc, height_desc)

        rect = Rectangle()
        assert rect.area == 50

    def test_computed_updates(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"a": 2, "b": 3})
        registry.register("config", source)

        a_desc = DataBoundDescriptor("config", "a")
        b_desc = DataBoundDescriptor("config", "b")

        class Math:
            a = a_desc
            b = b_desc
            product = ComputedBinding(lambda x, y: x * y, a_desc, b_desc)

        obj = Math()
        assert obj.product == 6

        source.set("a", 5)
        assert obj.product == 15

    def test_computed_with_default(self):
        class Calc:
            result = ComputedBinding(
                lambda: 1/0,  # Will raise
                default=0,
            )

        obj = Calc()
        assert obj.result == 0

    def test_computed_is_readonly(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"x": 5}))

        x_desc = DataBoundDescriptor("config", "x")

        class Values:
            x = x_desc
            double = ComputedBinding(lambda v: v * 2, x_desc)

        obj = Values()
        obj.double = 100  # Should be ignored
        assert obj.double == 10


class TestComputedDecorator:
    """Tests for @computed decorator."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_computed_decorator(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"first": "John", "last": "Doe"}))

        first_desc = DataBoundDescriptor("config", "first")
        last_desc = DataBoundDescriptor("config", "last")

        class Person:
            first = first_desc
            last = last_desc

            @computed(first_desc, last_desc)
            def full_name(first, last):
                return f"{first} {last}"

        person = Person()
        assert person.full_name == "John Doe"


class TestConcurrency:
    """Thread safety tests."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_concurrent_access(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"counter": 0})
        registry.register("config", source)

        class Counter:
            value = bound("config", "counter", mode=BindingMode.TWO_WAY)

        obj = Counter()
        errors = []

        def increment():
            try:
                for _ in range(100):
                    current = obj.value
                    obj.value = current + 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        DataSourceRegistry.get_instance().clear()

    def test_descriptor_on_class_access(self):
        registry = DataSourceRegistry.get_instance()
        registry.register("config", DictDataSource({"name": "test"}))

        class MyClass:
            name = bound("config", "name")

        desc = MyClass.name
        assert isinstance(desc, DataBoundDescriptor)

    def test_empty_key_path(self):
        source = DictDataSource({"root": "value"})
        value, found = source.get("")
        assert found
        assert value == {"root": "value"}

    def test_deeply_nested_path(self):
        source = DictDataSource({
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        })
        value, found = source.get("level1.level2.level3.value")
        assert found
        assert value == "deep"

    def test_source_removed_during_use(self):
        registry = DataSourceRegistry.get_instance()
        source = DictDataSource({"name": "test"})
        registry.register("config", source)

        class MyClass:
            name = bound("config", "name", mode=BindingMode.TWO_WAY)

        obj = MyClass()
        assert obj.name == "test"

        registry.unregister("config")

        # Should still have cached value
        assert obj.name == "test"
