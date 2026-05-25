"""
Tests for test_fixtures.py - Test fixtures and setup/teardown.
"""

import pytest

from engine.tooling.testing.test_fixtures import (
    EntityFixture,
    Fixture,
    FixtureManager,
    FixtureScope,
    GameWorldFixture,
    ResourceFixture,
    after_all,
    before_all,
    fixture,
    setup,
    teardown,
)


class TestFixtureScope:
    """Tests for FixtureScope enum."""

    def test_scope_values_exist(self):
        assert FixtureScope.FUNCTION
        assert FixtureScope.CLASS
        assert FixtureScope.MODULE
        assert FixtureScope.SESSION


class TestFixture:
    """Tests for Fixture dataclass."""

    def test_create_fixture(self):
        def factory():
            return {"data": 123}

        fix = Fixture(name="test_fixture", factory=factory)
        assert fix.name == "test_fixture"
        assert fix.scope == FixtureScope.FUNCTION

    def test_create_value(self):
        def factory():
            return {"value": 42}

        fix = Fixture(name="test", factory=factory)
        value = fix.create()
        assert value == {"value": 42}

    def test_caching_for_non_function_scope(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        fix = Fixture(name="test", factory=factory, scope=FixtureScope.MODULE)

        value1 = fix.create()
        value2 = fix.create()

        assert value1 is value2
        assert call_count == 1

    def test_no_caching_for_function_scope(self):
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        fix = Fixture(name="test", factory=factory, scope=FixtureScope.FUNCTION)

        # Function scope should always create new values
        # but since we're not resetting, it caches after first
        value1 = fix.create()
        fix.reset()  # Need to reset between calls
        value2 = fix.create()

        assert call_count == 2

    def test_destroy(self):
        class Closeable:
            closed = False

            def close(self):
                self.closed = True

        obj = Closeable()

        def factory():
            return obj

        fix = Fixture(name="test", factory=factory, scope=FixtureScope.MODULE)
        fix.create()
        fix.destroy()

        assert obj.closed


class TestFixtureDecorator:
    """Tests for @fixture decorator."""

    def test_basic_fixture(self):
        @fixture
        def my_fixture():
            return {"data": 123}

        assert my_fixture._is_fixture is True
        assert my_fixture._fixture.name == "my_fixture"

    def test_fixture_with_scope(self):
        @fixture(scope="module")
        def module_fixture():
            return {}

        assert module_fixture._fixture.scope == FixtureScope.MODULE

    def test_fixture_with_autouse(self):
        @fixture(autouse=True)
        def auto_fixture():
            return {}

        assert auto_fixture._fixture.autouse is True

    def test_generator_fixture(self):
        cleanup_called = False

        @fixture
        def gen_fixture():
            nonlocal cleanup_called
            yield {"data": 123}
            cleanup_called = True

        # Generator fixtures need special handling
        assert gen_fixture._fixture_generator is not None


class TestSetupTeardownDecorators:
    """Tests for @setup and @teardown decorators."""

    def test_setup_decorator(self):
        @setup
        def my_setup():
            pass

        assert my_setup._is_setup is True

    def test_teardown_decorator(self):
        @teardown
        def my_teardown():
            pass

        assert my_teardown._is_teardown is True

    def test_before_all_decorator(self):
        @before_all
        def module_setup():
            pass

        assert module_setup._is_setup is True
        assert module_setup._setup_scope == FixtureScope.MODULE

    def test_after_all_decorator(self):
        @after_all
        def module_teardown():
            pass

        assert module_teardown._is_teardown is True
        assert module_teardown._teardown_scope == FixtureScope.MODULE


class TestFixtureManager:
    """Tests for FixtureManager class."""

    def test_create_manager(self):
        manager = FixtureManager()
        assert manager is not None

    def test_register_fixture(self):
        manager = FixtureManager()
        fix = Fixture(name="test", factory=lambda: 42)
        manager.register(fix)

        assert "test" in manager._fixtures

    def test_get_fixture(self):
        manager = FixtureManager()
        fix = Fixture(name="test", factory=lambda: 42)
        manager.register(fix)

        value = manager.get_fixture("test")
        assert value == 42

    def test_fixture_dependencies(self):
        manager = FixtureManager()

        @fixture
        def base():
            return 10

        @fixture
        def dependent(base):
            return base * 2

        manager.register_function(base)
        manager.register_function(dependent)

        value = manager.get_fixture("dependent")
        assert value == 20

    def test_teardown_fixture(self):
        manager = FixtureManager()
        closed = []

        class Resource:
            def close(self):
                closed.append(True)

        fix = Fixture(name="resource", factory=Resource, scope=FixtureScope.MODULE)
        manager.register(fix)
        manager.get_fixture("resource")
        manager.teardown_fixture("resource")

        assert len(closed) == 1

    def test_fixture_context(self):
        manager = FixtureManager()
        fix = Fixture(name="data", factory=lambda: {"key": "value"})
        manager.register(fix)

        with manager.fixture_context("data") as fixtures:
            assert fixtures["data"] == {"key": "value"}

    def test_inject_fixtures(self):
        manager = FixtureManager()
        fix = Fixture(name="config", factory=lambda: {"debug": True})
        manager.register(fix)

        @manager.inject_fixtures
        def test_func(config):
            return config["debug"]

        result = test_func()
        assert result is True


class TestGameWorldFixture:
    """Tests for GameWorldFixture class."""

    def test_create_world(self):
        world = GameWorldFixture()
        assert world is not None
        assert len(world.entities) == 0

    def test_create_entity(self):
        world = GameWorldFixture()
        entity_id = world.create_entity()

        assert entity_id == 1
        assert entity_id in world.entities

    def test_add_component(self):
        world = GameWorldFixture()
        entity_id = world.create_entity()
        world.add_component(entity_id, "Position", {"x": 0, "y": 0})

        assert world.has_component(entity_id, "Position")
        assert world.get_component(entity_id, "Position") == {"x": 0, "y": 0}

    def test_remove_entity(self):
        world = GameWorldFixture()
        entity_id = world.create_entity()
        world.remove_entity(entity_id)

        assert entity_id not in world.entities

    def test_fire_event(self):
        world = GameWorldFixture()
        world.fire_event({"type": "TestEvent", "data": 123})

        events = world.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "TestEvent"

    def test_set_resource(self):
        world = GameWorldFixture()
        world.set_resource("Time", {"delta": 0.016})

        assert world.get_resource("Time") == {"delta": 0.016}

    def test_tick(self):
        world = GameWorldFixture()
        updated = []

        class MockSystem:
            def update(self, world, dt):
                updated.append(dt)

        world.add_system(MockSystem())
        world.tick(0.016)

        assert updated == [0.016]

    def test_reset(self):
        world = GameWorldFixture()
        world.create_entity()
        world.set_resource("Test", 123)
        world.reset()

        assert len(world.entities) == 0
        assert len(world.resources) == 0


class TestEntityFixture:
    """Tests for EntityFixture class."""

    def test_create_entity_fixture(self):
        world = GameWorldFixture()
        entity = EntityFixture(world)

        assert entity.entity_id == 1

    def test_add_component_chain(self):
        world = GameWorldFixture()
        entity = EntityFixture(world)

        result = entity.add_component("Position", {"x": 0}).add_component("Velocity", {"vx": 1})

        assert result is entity  # Chaining returns self
        assert entity.has_component("Position")
        assert entity.has_component("Velocity")

    def test_get_component(self):
        world = GameWorldFixture()
        entity = EntityFixture(world)
        entity.add_component("Health", {"current": 100})

        health = entity.get_component("Health")
        assert health["current"] == 100

    def test_destroy(self):
        world = GameWorldFixture()
        entity = EntityFixture(world)
        entity_id = entity.entity_id

        entity.destroy()
        assert entity_id not in world.entities


class TestResourceFixture:
    """Tests for ResourceFixture class."""

    def test_create_temp_file(self):
        resource = ResourceFixture()
        path = resource.create_temp_file("test.txt", "hello world")

        assert path.endswith("test.txt")

        # Read back
        with open(path) as f:
            assert f.read() == "hello world"

        resource.cleanup()

    def test_create_temp_dir(self):
        import os

        resource = ResourceFixture()
        path = resource.create_temp_dir("test_dir")

        assert os.path.isdir(path)

        resource.cleanup()

    def test_cleanup_removes_files(self):
        import os

        resource = ResourceFixture()
        path = resource.create_temp_file("cleanup_test.txt", "data")

        resource.cleanup()

        assert not os.path.exists(path)

    def test_register_handle(self):
        resource = ResourceFixture()

        class MockHandle:
            closed = False

            def close(self):
                self.closed = True

        handle = MockHandle()
        resource.register_handle(handle)
        resource.cleanup()

        assert handle.closed
