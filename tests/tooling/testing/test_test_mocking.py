"""
Tests for test_mocking.py - Mocking framework for game objects.
"""

import pytest

from engine.tooling.testing.test_mocking import (
    CallRecord,
    Mock,
    MockComponent,
    MockContext,
    MockEntity,
    MockResource,
    MockSystem,
    MockWorld,
    create_mock,
    patch,
    spy,
    stub,
)


class TestCallRecord:
    """Tests for CallRecord dataclass."""

    def test_create_record(self):
        record = CallRecord(args=(1, 2), kwargs={"x": 3})
        assert record.args == (1, 2)
        assert record.kwargs == {"x": 3}

    def test_matches_exact(self):
        record = CallRecord(args=(1, 2), kwargs={"x": 3})
        assert record.matches(1, 2, x=3) is True

    def test_matches_partial_args(self):
        record = CallRecord(args=(1, 2, 3), kwargs={})
        assert record.matches(1, 2, 3) is True

    def test_matches_partial_kwargs(self):
        record = CallRecord(args=(), kwargs={"a": 1, "b": 2})
        assert record.matches(a=1) is True


class TestMock:
    """Tests for Mock class."""

    def test_create_mock(self):
        mock = Mock(name="TestMock")
        assert mock._name == "TestMock"

    def test_configure_return(self):
        mock = Mock()
        mock.configure_return("get_value", 42)

        assert mock.get_value() == 42

    def test_was_called(self):
        mock = Mock()
        mock.some_method()

        assert mock.was_called("some_method") is True
        assert mock.was_called("other_method") is False

    def test_call_count(self):
        mock = Mock()
        mock.method()
        mock.method()
        mock.method()

        assert mock.call_count("method") == 3

    def test_last_call(self):
        mock = Mock()
        mock.method(1, 2)
        mock.method(3, 4)

        last = mock.last_call("method")
        assert last.args == (3, 4)

    def test_assert_called(self):
        mock = Mock()
        mock.method()

        mock.assert_called("method")  # Should not raise

    def test_assert_called_fail(self):
        mock = Mock()

        with pytest.raises(AssertionError):
            mock.assert_called("never_called")

    def test_assert_called_with(self):
        mock = Mock()
        mock.method(1, key="value")

        mock.assert_called_with("method", 1, key="value")

    def test_assert_called_with_fail(self):
        mock = Mock()
        mock.method(1)

        with pytest.raises(AssertionError):
            mock.assert_called_with("method", 2)

    def test_configure_side_effect_callable(self):
        mock = Mock()
        mock.configure_side_effect("double", lambda x: x * 2)

        assert mock.double(5) == 10

    def test_configure_side_effect_exception(self):
        mock = Mock()
        mock.configure_side_effect("error", ValueError("test error"))

        with pytest.raises(ValueError):
            mock.error()

    def test_configure_side_effect_list(self):
        mock = Mock()
        mock.configure_side_effect("sequence", [1, 2, 3])

        assert mock.sequence() == 1
        assert mock.sequence() == 2
        assert mock.sequence() == 3

    def test_reset(self):
        mock = Mock()
        mock.method()
        mock.reset()

        assert mock.call_count("method") == 0

    def test_attribute_access(self):
        mock = Mock(value=42)
        assert mock.value == 42

    def test_attribute_set(self):
        mock = Mock()
        mock.value = 100
        assert mock.value == 100


class TestMockEntity:
    """Tests for MockEntity class."""

    def test_create_entity(self):
        entity = MockEntity(id=1, name="Player")
        assert entity.id == 1
        assert entity.name == "Player"

    def test_add_component(self):
        entity = MockEntity(id=1)
        entity.add_component("Position", {"x": 0, "y": 0})

        assert entity.has_component("Position")

    def test_get_component(self):
        entity = MockEntity(id=1)
        entity.add_component("Health", {"current": 100})

        health = entity.get_component("Health")
        assert health["current"] == 100

    def test_remove_component(self):
        entity = MockEntity(id=1)
        entity.add_component("Position", {})
        entity.remove_component("Position")

        assert entity.has_component("Position") is False

    def test_tags(self):
        entity = MockEntity(id=1)
        entity.add_tag("player")
        entity.add_tag("alive")

        assert entity.has_tag("player") is True
        assert entity.has_tag("enemy") is False

        entity.remove_tag("alive")
        assert entity.has_tag("alive") is False

    def test_activate_deactivate(self):
        entity = MockEntity(id=1)
        assert entity.active is True

        entity.deactivate()
        assert entity.active is False

        entity.activate()
        assert entity.active is True


class TestMockComponent:
    """Tests for MockComponent class."""

    def test_create_component(self):
        comp = MockComponent("Health", current=100, max=100)
        assert comp.type_name == "Health"
        assert comp.current == 100
        assert comp.max == 100

    def test_modify_component(self):
        comp = MockComponent("Health", current=100)
        assert comp.dirty is False

        comp.current = 50
        assert comp.dirty is True
        assert comp.current == 50

    def test_mark_clean(self):
        comp = MockComponent("Health", current=100)
        comp.current = 50
        comp.mark_clean()

        assert comp.dirty is False

    def test_to_dict(self):
        comp = MockComponent("Position", x=10, y=20)
        data = comp.to_dict()

        assert data == {"x": 10, "y": 20}


class TestMockSystem:
    """Tests for MockSystem class."""

    def test_create_system(self):
        system = MockSystem("MovementSystem", phase="update", priority=10)
        assert system.name == "MovementSystem"
        assert system.phase == "update"
        assert system.priority == 10

    def test_update(self):
        system = MockSystem("TestSystem")
        world = MockWorld()

        system.update(world, 0.016)
        assert system.update_count == 1

    def test_enable_disable(self):
        system = MockSystem("TestSystem")
        world = MockWorld()

        system.disable()
        system.update(world, 0.016)
        assert system.update_count == 0

        system.enable()
        system.update(world, 0.016)
        assert system.update_count == 1

    def test_configure_update(self):
        system = MockSystem("TestSystem")
        results = []

        def custom_update(world, dt):
            results.append(dt)

        system.configure_update(custom_update)

        world = MockWorld()
        system.update(world, 0.016)

        assert results == [0.016]

    def test_reset(self):
        system = MockSystem("TestSystem")
        world = MockWorld()

        system.update(world, 0.016)
        system.update(world, 0.016)
        system.reset()

        assert system.update_count == 0


class TestMockWorld:
    """Tests for MockWorld class."""

    def test_create_world(self):
        world = MockWorld()
        assert world.entity_count == 0

    def test_create_entity(self):
        world = MockWorld()
        entity_id = world.create_entity()

        assert entity_id == 1
        assert world.entity_count == 1

    def test_destroy_entity(self):
        world = MockWorld()
        entity_id = world.create_entity()
        world.destroy_entity(entity_id)

        assert world.entity_count == 0

    def test_components(self):
        world = MockWorld()
        entity_id = world.create_entity()
        world.add_component(entity_id, "Position", {"x": 0, "y": 0})

        assert world.has_component(entity_id, "Position")
        assert world.get_component(entity_id, "Position") == {"x": 0, "y": 0}

        world.remove_component(entity_id, "Position")
        assert world.has_component(entity_id, "Position") is False

    def test_resources(self):
        world = MockWorld()
        world.set_resource("Time", {"delta": 0.016})

        assert world.get_resource("Time") == {"delta": 0.016}

    def test_events(self):
        world = MockWorld()
        world.fire_event({"type": "TestEvent"})

        events = world.get_events()
        assert len(events) == 1

        world.clear_events()
        assert len(world.get_events()) == 0

    def test_update(self):
        world = MockWorld()
        system = MockSystem("TestSystem")
        world.add_system(system)

        world.update(0.016)
        assert system.update_count == 1

    def test_query(self):
        world = MockWorld()

        e1 = world.create_entity()
        world.add_component(e1, "Position", {})
        world.add_component(e1, "Velocity", {})

        e2 = world.create_entity()
        world.add_component(e2, "Position", {})

        result = world.query("Position", "Velocity")
        assert result == [e1]


class TestMockResource:
    """Tests for MockResource class."""

    def test_create_resource(self):
        resource = MockResource("Time", delta=0.016, elapsed=0.0)
        assert resource.name == "Time"
        assert resource.delta == 0.016
        assert resource.elapsed == 0.0

    def test_modify_resource(self):
        resource = MockResource("Time", delta=0.016)
        resource.delta = 0.033

        assert resource.delta == 0.033


class TestMockContext:
    """Tests for MockContext class."""

    def test_create_context(self):
        with MockContext() as ctx:
            assert ctx is not None

    def test_create_mock_in_context(self):
        with MockContext() as ctx:
            player = ctx.mock("player", health=100)
            assert player.health == 100

    def test_get_mock(self):
        with MockContext() as ctx:
            player = ctx.mock("player")
            retrieved = ctx.get_mock("player")
            assert player is retrieved

    def test_reset_all(self):
        with MockContext() as ctx:
            mock1 = ctx.mock("mock1")
            mock2 = ctx.mock("mock2")

            mock1.method()
            mock2.method()

            ctx.reset_all()

            assert mock1.call_count("method") == 0
            assert mock2.call_count("method") == 0


class TestSpyFunction:
    """Tests for spy function."""

    def test_spy_tracks_calls(self):
        class Calculator:
            def add(self, a, b):
                return a + b

        calc = Calculator()
        spy_mock = spy(calc, "add")

        result = calc.add(1, 2)

        assert result == 3  # Original method still works
        assert spy_mock.was_called("add")


class TestStubDecorator:
    """Tests for stub decorator."""

    def test_stub_returns_value(self):
        @stub(return_value=42)
        def complex_calculation():
            raise RuntimeError("Should not be called")

        assert complex_calculation() == 42


class TestCreateMock:
    """Tests for create_mock function."""

    def test_create_simple_mock(self):
        mock = create_mock(health=100)
        assert mock.health == 100

    def test_create_mock_with_spec(self):
        class Player:
            def attack(self):
                pass

            def defend(self):
                pass

        mock = create_mock(Player, health=100)
        assert mock.health == 100
        assert hasattr(mock, "attack")
