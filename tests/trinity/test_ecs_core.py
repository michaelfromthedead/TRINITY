"""
Tests for ECS Core decorators (ecs_core.py).

Tests the 9 ECS decorators built on Ops:
    @component, @tag, @resource, @event, @system,
    @query, @bundle, @relation, @derived

Each test verifies:
1. Steps are applied (_applied_steps populated)
2. Domain attributes are set correctly
3. Metaclass registration happens
4. Introspection works
"""

import warnings

import pytest

from trinity.decorators.ecs_core import (
    bundle,
    component,
    derived,
    event,
    query,
    relation,
    resource,
    system,
    tag,
)
from trinity.decorators.ops import Op, decompose, expand

# =============================================================================
# @component
# =============================================================================


class TestComponent:
    def test_no_parens(self):
        @component
        class Health:
            current: float = 100.0

        assert Health._component is True
        assert Health._component_name == "Health"

    def test_with_name(self):
        @component(name="PlayerHP")
        class Health2:
            current: float = 100.0

        assert Health2._component_name == "PlayerHP"

    def test_with_empty_parens(self):
        @component()
        class Vel:
            x: float = 0.0

        assert Vel._component is True

    def test_has_component_id(self):
        @component
        class Pos:
            x: float = 0.0

        assert hasattr(Pos, "_component_id")
        assert isinstance(Pos._component_id, int)

    def test_applied_decorators(self):
        @component
        class C:
            x: int = 0

        assert "component" in C._applied_decorators

    def test_steps_recorded(self):
        @component
        class C2:
            x: int = 0

        assert hasattr(C2, "_applied_steps")
        ops_used = {s.op for s in C2._applied_steps}
        assert Op.TAG in ops_used
        assert Op.REGISTER in ops_used


# =============================================================================
# @tag
# =============================================================================


class TestTag:
    def test_basic(self):
        @tag
        class Player:
            pass

        assert Player._component is True
        assert Player._tag is True
        assert Player._field_types == {}

    def test_has_component_id(self):
        @tag
        class Enemy:
            pass

        assert hasattr(Enemy, "_component_id")

    def test_applied_decorators(self):
        @tag
        class Static:
            pass

        assert "tag" in Static._applied_decorators

    def test_steps_recorded(self):
        @tag
        class Flying:
            pass

        ops_used = {s.op for s in Flying._applied_steps}
        assert Op.TAG in ops_used


# =============================================================================
# @resource
# =============================================================================


class TestResource:
    def test_no_parens(self):
        @resource
        class Time:
            delta: float = 0.016

        assert Time._resource is True
        assert Time._resource_name == "Time"

    def test_with_name(self):
        @resource(name="PhysicsConfig")
        class PhysSettings:
            gravity: float = -9.81

        assert PhysSettings._resource_name == "PhysicsConfig"

    def test_has_resource_id(self):
        @resource
        class Input:
            keys: dict = None

        assert hasattr(Input, "_resource_id")

    def test_applied_decorators(self):
        @resource
        class Audio:
            volume: float = 1.0

        assert "resource" in Audio._applied_decorators

    def test_steps_recorded(self):
        @resource
        class R:
            pass

        ops_used = {s.op for s in R._applied_steps}
        assert Op.TAG in ops_used


# =============================================================================
# @event
# =============================================================================


class TestEvent:
    def test_basic(self):
        @event
        class DamageEvent:
            amount: float
            source: int

        assert DamageEvent._event is True
        assert "DamageEvent" in DamageEvent._event_name

    def test_has_event_id(self):
        @event
        class SpawnEvent:
            entity_id: int

        assert hasattr(SpawnEvent, "_event_id")

    def test_event_fields_extracted(self):
        @event
        class MoveEvent:
            x: float
            y: float

        assert "x" in MoveEvent._event_fields
        assert "y" in MoveEvent._event_fields

    def test_applied_decorators(self):
        @event
        class E:
            pass

        assert "event" in E._applied_decorators


# =============================================================================
# @system
# =============================================================================


class TestSystem:
    def test_default_phase(self):
        @system()
        def update_positions():
            pass

        assert update_positions._system is True
        assert update_positions._system_phase == "update"

    def test_custom_phase(self):
        @system(phase="physics")
        def gravity():
            pass

        assert gravity._system_phase == "physics"

    def test_applied_decorators(self):
        @system()
        def render():
            pass

        assert "system" in render._applied_decorators

    def test_steps_recorded(self):
        @system()
        def tick():
            pass

        ops_used = {s.op for s in tick._applied_steps}
        assert Op.TAG in ops_used

    def test_defaults_set(self):
        @system()
        def fn():
            pass

        assert fn._reads == ()
        assert fn._writes == ()
        assert fn._exclusive is False
        assert fn._priority == 0


# =============================================================================
# @query
# =============================================================================


class TestQuery:
    def test_basic(self):
        class Transform:
            pass

        class Velocity:
            pass

        @query(components=(Transform, Velocity))
        def move_query():
            pass

        assert move_query._query is True
        assert move_query._query_components == (Transform, Velocity)

    def test_filters(self):
        class Player:
            pass

        class Dead:
            pass

        @query(components=(), with_=(Player,), without=(Dead,))
        def alive_players():
            pass

        assert alive_players._query_with == (Player,)
        assert alive_players._query_without == (Dead,)

    def test_maybe(self):
        class Accel:
            pass

        @query(components=(), maybe=(Accel,))
        def q():
            pass

        assert q._query_maybe == (Accel,)

    def test_applied_decorators(self):
        @query(components=())
        def q():
            pass

        assert "query" in q._applied_decorators


# =============================================================================
# @bundle
# =============================================================================


class TestBundle:
    def test_basic(self):
        @bundle
        class PlayerBundle:
            health: float
            speed: float

        assert PlayerBundle._bundle is True
        assert "health" in PlayerBundle._bundle_components
        assert "speed" in PlayerBundle._bundle_components

    def test_applied_decorators(self):
        @bundle
        class B:
            x: int

        assert "bundle" in B._applied_decorators

    def test_steps_recorded(self):
        @bundle
        class B2:
            x: int

        ops_used = {s.op for s in B2._applied_steps}
        assert Op.TAG in ops_used
        assert Op.DESCRIBE in ops_used


# =============================================================================
# @relation
# =============================================================================


class TestRelation:
    def test_basic(self):
        @component
        @relation(kind="one_to_many")
        class ChildOf:
            parent: int = 0

        assert ChildOf._relation is True
        assert ChildOf._relation_kind == "one_to_many"
        assert ChildOf._relation_exclusive is False

    def test_exclusive(self):
        @component
        @relation(kind="one_to_one", exclusive=True)
        class TargetedBy:
            targeter: int = 0

        assert TargetedBy._relation_exclusive is True
        assert TargetedBy._relation_kind == "one_to_one"

    def test_auto_component(self):
        """@relation auto-applies @component if missing."""

        @relation(kind="one_to_many")
        class Owns:
            item: int = 0

        assert Owns._component is True
        assert Owns._relation is True

    def test_applied_decorators(self):
        @relation(kind="one_to_many")
        class R:
            x: int = 0

        assert "relation" in R._applied_decorators

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="invalid kind"):

            @relation(kind="many_to_all")
            class Bad:
                pass


# =============================================================================
# @derived
# =============================================================================


class TestDerived:
    def test_basic(self):
        class LocalTransform:
            pass

        @component
        @derived(from_components=(LocalTransform,), cache=True)
        class WorldTransform:
            @staticmethod
            def compute(local):
                return local

        assert WorldTransform._derived is True
        assert WorldTransform._derived_from == (LocalTransform,)
        assert WorldTransform._derived_cache is True

    def test_auto_component(self):
        class Src:
            pass

        @derived(from_components=(Src,))
        class D:
            @staticmethod
            def compute(s):
                return s

        assert D._component is True
        assert D._derived is True

    def test_missing_compute_warns(self):
        class Src2:
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @derived(from_components=(Src2,))
            class NoCompute:
                pass

            assert any("compute" in str(warning.message) for warning in w)

    def test_applied_decorators(self):
        class S:
            pass

        @derived(from_components=(S,))
        class D2:
            @staticmethod
            def compute(s):
                return s

        assert "derived" in D2._applied_decorators

    def test_missing_from_components(self):
        with pytest.raises(ValueError, match="from_components"):

            @derived(from_components=())
            class Bad:
                pass


# =============================================================================
# INTROSPECTION
# =============================================================================


class TestEcsCoreIntrospection:
    @pytest.mark.parametrize(
        "dec",
        [component, tag, resource, event, system, query, bundle, relation, derived],
    )
    def test_decompose_returns_list(self, dec):
        steps = decompose(dec)
        assert isinstance(steps, list)

    @pytest.mark.parametrize(
        "dec",
        [component, tag, resource, event, system, query, bundle, relation, derived],
    )
    def test_expand_returns_string(self, dec):
        result = expand(dec)
        assert isinstance(result, str)

    def test_all_register_ecs_core(self):
        """Every ECS core decorator should have a REGISTER step for 'ecs_core'."""
        for dec in [
            component,
            tag,
            resource,
            event,
            system,
            query,
            bundle,
            relation,
            derived,
        ]:
            steps = decompose(dec)
            reg_steps = [s for s in steps if s.op is Op.REGISTER]
            assert any(s.args.get("registry") == "ecs_core" for s in reg_steps), (
                f"{dec.__name__} missing REGISTER(ecs_core) step"
            )
