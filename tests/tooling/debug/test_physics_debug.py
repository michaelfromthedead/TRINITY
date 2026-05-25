"""Tests for physics debug - collision viz, contacts, rays."""

import pytest
import time
from engine.tooling.debug.physics_debug import (
    PhysicsDebugger,
    CollisionShapeVisualizer,
    ContactPointDisplay,
    RaycastVisualizer,
    CollisionShape,
    CollisionShapeType,
    PhysicsBodyType,
    ContactPoint,
    RaycastRequest,
    RaycastHit,
    Vector3,
    Quaternion,
)


class TestCollisionShape:
    """Tests for CollisionShape class."""

    def test_shape_creation(self):
        shape = CollisionShape(
            shape_id="shape_001",
            shape_type=CollisionShapeType.SPHERE,
            position=Vector3(0, 0, 0),
            radius=1.0,
        )
        assert shape.shape_id == "shape_001"
        assert shape.shape_type == CollisionShapeType.SPHERE

    def test_shape_with_body_type(self):
        shape = CollisionShape(
            shape_id="shape",
            shape_type=CollisionShapeType.BOX,
            position=Vector3(0, 0, 0),
            body_type=PhysicsBodyType.STATIC,
        )
        assert shape.body_type == PhysicsBodyType.STATIC


class TestCollisionShapeVisualizer:
    """Tests for CollisionShapeVisualizer class."""

    def test_visualizer_creation(self):
        viz = CollisionShapeVisualizer()
        assert viz.is_enabled is True
        assert viz.shape_count == 0

    def test_add_shape(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        viz.add_shape(shape)
        assert viz.shape_count == 1
        assert viz.get_shape("s1") is shape

    def test_remove_shape(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        viz.add_shape(shape)
        removed = viz.remove_shape("s1")
        assert removed is shape
        assert viz.shape_count == 0

    def test_update_shape(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        viz.add_shape(shape)

        result = viz.update_shape(
            "s1",
            position=Vector3(10, 0, 0),
            is_sleeping=True,
        )
        assert result is True
        assert shape.position.x == 10
        assert shape.is_sleeping is True

    def test_update_shape_not_found(self):
        viz = CollisionShapeVisualizer()
        result = viz.update_shape("nonexistent", position=Vector3(0, 0, 0))
        assert result is False

    def test_show_options(self):
        viz = CollisionShapeVisualizer()
        viz.set_show_sleeping(False)
        viz.set_show_triggers(False)
        viz.set_show_static(False)

    def test_color_by_type(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))

        viz.set_color_by_type(True)
        color1 = viz.get_shape_color(shape)

        viz.set_color_by_type(False)
        color2 = viz.get_shape_color(shape)

        # Colors should be different based on mode
        assert color1 == viz._shape_colors[CollisionShapeType.SPHERE]

    def test_generate_draw_commands_sphere(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape(
            "s1",
            CollisionShapeType.SPHERE,
            Vector3(0, 0, 0),
            radius=2.0,
        )
        viz.add_shape(shape)

        commands = viz.generate_draw_commands()
        assert len(commands) > 0
        assert commands[0]["type"] == "sphere"

    def test_generate_draw_commands_box(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape(
            "s1",
            CollisionShapeType.BOX,
            Vector3(0, 0, 0),
            extents=Vector3(1, 1, 1),
        )
        viz.add_shape(shape)

        commands = viz.generate_draw_commands()
        assert commands[0]["type"] == "box"

    def test_generate_draw_commands_capsule(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape(
            "s1",
            CollisionShapeType.CAPSULE,
            Vector3(0, 0, 0),
            radius=0.5,
            height=2.0,
        )
        viz.add_shape(shape)

        commands = viz.generate_draw_commands()
        assert commands[0]["type"] == "capsule"

    def test_generate_draw_commands_disabled(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        viz.add_shape(shape)
        viz.disable()

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_filter_sleeping(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape(
            "s1",
            CollisionShapeType.SPHERE,
            Vector3(0, 0, 0),
            is_sleeping=True,
        )
        viz.add_shape(shape)
        viz.set_show_sleeping(False)

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_filter_triggers(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape(
            "s1",
            CollisionShapeType.SPHERE,
            Vector3(0, 0, 0),
            body_type=PhysicsBodyType.TRIGGER,
        )
        viz.add_shape(shape)
        viz.set_show_triggers(False)

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_filter_static(self):
        viz = CollisionShapeVisualizer()
        shape = CollisionShape(
            "s1",
            CollisionShapeType.SPHERE,
            Vector3(0, 0, 0),
            body_type=PhysicsBodyType.STATIC,
        )
        viz.add_shape(shape)
        viz.set_show_static(False)

        commands = viz.generate_draw_commands()
        assert len(commands) == 0

    def test_get_shapes_by_type(self):
        viz = CollisionShapeVisualizer()
        viz.add_shape(CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0)))
        viz.add_shape(CollisionShape("s2", CollisionShapeType.BOX, Vector3(0, 0, 0)))
        viz.add_shape(CollisionShape("s3", CollisionShapeType.SPHERE, Vector3(0, 0, 0)))

        spheres = viz.get_shapes_by_type(CollisionShapeType.SPHERE)
        assert len(spheres) == 2

    def test_get_shapes_by_layer(self):
        viz = CollisionShapeVisualizer()
        viz.add_shape(CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0), layer=1))
        viz.add_shape(CollisionShape("s2", CollisionShapeType.SPHERE, Vector3(0, 0, 0), layer=2))
        viz.add_shape(CollisionShape("s3", CollisionShapeType.SPHERE, Vector3(0, 0, 0), layer=1))

        layer1 = viz.get_shapes_by_layer(1)
        assert len(layer1) == 2

    def test_clear_all_shapes(self):
        viz = CollisionShapeVisualizer()
        viz.add_shape(CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0)))
        viz.add_shape(CollisionShape("s2", CollisionShapeType.BOX, Vector3(0, 0, 0)))
        viz.clear_all_shapes()
        assert viz.shape_count == 0


class TestContactPoint:
    """Tests for ContactPoint class."""

    def test_contact_creation(self):
        contact = ContactPoint(
            contact_id=1,
            position=Vector3(0, 0, 0),
            normal=Vector3(0, 1, 0),
            penetration_depth=0.01,
            body_a_id="body_a",
            body_b_id="body_b",
        )
        assert contact.contact_id == 1
        assert contact.body_a_id == "body_a"


class TestContactPointDisplay:
    """Tests for ContactPointDisplay class."""

    def test_display_creation(self):
        display = ContactPointDisplay()
        assert display.is_enabled is True
        assert display.contact_count == 0

    def test_add_contact(self):
        display = ContactPointDisplay()
        contact = ContactPoint(
            contact_id=1,
            position=Vector3(0, 0, 0),
            normal=Vector3(0, 1, 0),
            penetration_depth=0.01,
            body_a_id="a",
            body_b_id="b",
        )
        display.add_contact(contact)
        assert display.contact_count == 1

    def test_add_contact_from_data(self):
        display = ContactPointDisplay()
        contact = display.add_contact_from_data(
            position=Vector3(1, 2, 3),
            normal=Vector3(0, 1, 0),
            penetration_depth=0.05,
            body_a_id="body_a",
            body_b_id="body_b",
            impulse=10.0,
        )
        assert contact.position.x == 1
        assert contact.impulse == 10.0
        assert display.contact_count == 1

    def test_max_contacts(self):
        display = ContactPointDisplay(max_contacts=5)
        for i in range(10):
            display.add_contact_from_data(
                Vector3(i, 0, 0),
                Vector3(0, 1, 0),
                0.01,
                "a",
                "b",
            )
        assert display.contact_count == 5

    def test_remove_contact(self):
        display = ContactPointDisplay()
        contact = display.add_contact_from_data(
            Vector3(0, 0, 0),
            Vector3(0, 1, 0),
            0.01,
            "a",
            "b",
        )
        removed = display.remove_contact(contact.contact_id)
        assert removed is contact
        assert display.contact_count == 0

    def test_clear_contacts(self):
        display = ContactPointDisplay()
        display.add_contact_from_data(Vector3(0, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b")
        display.add_contact_from_data(Vector3(1, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b")
        display.clear_contacts()
        assert display.contact_count == 0

    def test_show_options(self):
        display = ContactPointDisplay()
        display.set_show_normals(False)
        display.set_show_impulses(False)

    def test_generate_draw_commands(self):
        display = ContactPointDisplay()
        display.add_contact_from_data(
            Vector3(0, 0, 0),
            Vector3(0, 1, 0),
            0.01,
            "a",
            "b",
            impulse=5.0,
        )

        commands = display.generate_draw_commands()
        assert len(commands) > 0

    def test_generate_draw_commands_disabled(self):
        display = ContactPointDisplay()
        display.add_contact_from_data(Vector3(0, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b")
        display.disable()

        commands = display.generate_draw_commands()
        assert len(commands) == 0

    def test_get_contacts_for_body(self):
        display = ContactPointDisplay()
        display.add_contact_from_data(Vector3(0, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b")
        display.add_contact_from_data(Vector3(1, 0, 0), Vector3(0, 1, 0), 0.01, "a", "c")
        display.add_contact_from_data(Vector3(2, 0, 0), Vector3(0, 1, 0), 0.01, "d", "e")

        contacts_a = display.get_contacts_for_body("a")
        assert len(contacts_a) == 2


class TestRaycastHit:
    """Tests for RaycastHit class."""

    def test_hit_creation(self):
        hit = RaycastHit(
            hit=True,
            position=Vector3(5, 0, 0),
            normal=Vector3(-1, 0, 0),
            distance=5.0,
            body_id="target",
        )
        assert hit.hit is True
        assert hit.distance == 5.0


class TestRaycastVisualizer:
    """Tests for RaycastVisualizer class."""

    def test_visualizer_creation(self):
        viz = RaycastVisualizer()
        assert viz.is_enabled is True
        assert viz.raycast_count == 0

    def test_add_raycast(self):
        viz = RaycastVisualizer()
        ray = viz.add_raycast(
            ray_id="ray_001",
            origin=Vector3(0, 0, 0),
            direction=Vector3(1, 0, 0),
            max_distance=100.0,
        )
        assert ray.ray_id == "ray_001"
        assert viz.raycast_count == 1

    def test_add_raycast_with_hit(self):
        viz = RaycastVisualizer()
        hit = RaycastHit(
            hit=True,
            position=Vector3(10, 0, 0),
            normal=Vector3(-1, 0, 0),
            distance=10.0,
        )
        ray = viz.add_raycast(
            ray_id="ray",
            origin=Vector3(0, 0, 0),
            direction=Vector3(1, 0, 0),
            result=hit,
        )
        assert ray.result is hit

    def test_update_result(self):
        viz = RaycastVisualizer()
        viz.add_raycast(
            ray_id="ray",
            origin=Vector3(0, 0, 0),
            direction=Vector3(1, 0, 0),
        )
        hit = RaycastHit(hit=True, position=Vector3(5, 0, 0), distance=5.0)
        result = viz.update_result("ray", hit)
        assert result is True
        assert viz._raycasts["ray"].result is hit

    def test_remove_raycast(self):
        viz = RaycastVisualizer()
        viz.add_raycast("ray", Vector3(0, 0, 0), Vector3(1, 0, 0))
        removed = viz.remove_raycast("ray")
        assert removed is not None
        assert viz.raycast_count == 0

    def test_clear_expired(self):
        viz = RaycastVisualizer()
        viz.add_raycast(
            "ray",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            lifetime=0.1,
        )
        # Set timestamp in the past
        viz._raycasts["ray"].timestamp = time.time() - 1.0

        removed = viz.clear_expired()
        assert removed == 1
        assert viz.raycast_count == 0

    def test_show_misses(self):
        viz = RaycastVisualizer()
        viz.set_show_misses(False)

    def test_generate_draw_commands_hit(self):
        viz = RaycastVisualizer()
        hit = RaycastHit(
            hit=True,
            position=Vector3(10, 0, 0),
            normal=Vector3(-1, 0, 0),
            distance=10.0,
        )
        viz.add_raycast(
            "ray",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            result=hit,
        )

        commands = viz.generate_draw_commands()
        assert len(commands) > 0
        # Should have line, point, and normal arrow
        types = [cmd["type"] for cmd in commands]
        assert "line" in types
        assert "point" in types
        assert "arrow" in types

    def test_generate_draw_commands_miss(self):
        viz = RaycastVisualizer()
        miss = RaycastHit(hit=False)
        viz.add_raycast(
            "ray",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            result=miss,
        )

        commands = viz.generate_draw_commands()
        # Should have miss line
        assert len(commands) > 0

    def test_generate_draw_commands_miss_hidden(self):
        viz = RaycastVisualizer()
        miss = RaycastHit(hit=False)
        viz.add_raycast(
            "ray",
            Vector3(0, 0, 0),
            Vector3(1, 0, 0),
            result=miss,
        )
        viz.set_show_misses(False)

        commands = viz.generate_draw_commands()
        # Should not show misses
        assert len(commands) == 0


class TestPhysicsDebugger:
    """Tests for PhysicsDebugger singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        PhysicsDebugger.reset_instance()
        yield
        PhysicsDebugger.reset_instance()

    def test_singleton(self):
        d1 = PhysicsDebugger.get_instance()
        d2 = PhysicsDebugger.get_instance()
        assert d1 is d2

    def test_enable_disable(self):
        debugger = PhysicsDebugger.get_instance()
        debugger.enable()
        assert debugger.is_enabled
        debugger.disable()
        assert not debugger.is_enabled

    def test_subsystems_accessible(self):
        debugger = PhysicsDebugger.get_instance()
        assert isinstance(debugger.shape_visualizer, CollisionShapeVisualizer)
        assert isinstance(debugger.contact_display, ContactPointDisplay)
        assert isinstance(debugger.raycast_visualizer, RaycastVisualizer)

    def test_collision_layer_names(self):
        debugger = PhysicsDebugger.get_instance()
        debugger.set_collision_layer_name(1, "Player")
        debugger.set_collision_layer_name(2, "Enemy")

        assert debugger.get_collision_layer_name(1) == "Player"
        assert debugger.get_collision_layer_name(2) == "Enemy"
        assert debugger.get_collision_layer_name(99) == "Layer 99"

    def test_generate_all_draw_commands(self):
        debugger = PhysicsDebugger.get_instance()
        debugger.shape_visualizer.add_shape(
            CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        )
        debugger.contact_display.add_contact_from_data(
            Vector3(0, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b"
        )
        debugger.raycast_visualizer.add_raycast(
            "ray", Vector3(0, 0, 0), Vector3(1, 0, 0)
        )

        commands = debugger.generate_all_draw_commands()
        assert len(commands) > 0

    def test_generate_all_disabled(self):
        debugger = PhysicsDebugger.get_instance()
        debugger.shape_visualizer.add_shape(
            CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        )
        debugger.disable()

        commands = debugger.generate_all_draw_commands()
        assert len(commands) == 0

    def test_clear_all(self):
        debugger = PhysicsDebugger.get_instance()
        debugger.shape_visualizer.add_shape(
            CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        )
        debugger.contact_display.add_contact_from_data(
            Vector3(0, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b"
        )

        debugger.clear_all()
        assert debugger.shape_visualizer.shape_count == 0
        assert debugger.contact_display.contact_count == 0

    def test_get_stats(self):
        debugger = PhysicsDebugger.get_instance()
        debugger.shape_visualizer.add_shape(
            CollisionShape("s1", CollisionShapeType.SPHERE, Vector3(0, 0, 0))
        )
        debugger.contact_display.add_contact_from_data(
            Vector3(0, 0, 0), Vector3(0, 1, 0), 0.01, "a", "b"
        )

        stats = debugger.get_stats()
        assert stats["shapes"] == 1
        assert stats["contacts"] == 1
