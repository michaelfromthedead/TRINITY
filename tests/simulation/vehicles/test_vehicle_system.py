"""Tests for vehicle system management.

Tests cover:
- Vector3 math operations
- Transform dataclass
- VehicleSystem registration and lifecycle
- Vehicle groups
- Collision callbacks
- Statistics and monitoring
"""

import pytest

from engine.simulation.vehicles.vehicle_system import (
    VehicleType,
    VehicleState,
    Vector3,
    Transform,
    VehicleGroup,
    CollisionInfo,
    VehicleSystem,
    generate_vehicle_id,
)


# =============================================================================
# Vector3 Tests
# =============================================================================


class TestVector3:
    """Tests for Vector3 math operations."""

    def test_default_zero(self):
        """Default Vector3 should be zero."""
        v = Vector3()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0

    def test_custom_values(self):
        """Vector3 should accept custom values."""
        v = Vector3(1.0, 2.0, 3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_addition(self):
        """Vector addition should work correctly."""
        a = Vector3(1, 2, 3)
        b = Vector3(4, 5, 6)
        c = a + b
        assert c.x == 5
        assert c.y == 7
        assert c.z == 9

    def test_subtraction(self):
        """Vector subtraction should work correctly."""
        a = Vector3(5, 7, 9)
        b = Vector3(1, 2, 3)
        c = a - b
        assert c.x == 4
        assert c.y == 5
        assert c.z == 6

    def test_scalar_multiplication(self):
        """Scalar multiplication should work correctly."""
        v = Vector3(1, 2, 3)
        result = v * 2
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_scalar_rmul(self):
        """Right multiplication should work."""
        v = Vector3(1, 2, 3)
        result = 2 * v
        assert result.x == 2
        assert result.y == 4
        assert result.z == 6

    def test_scalar_division(self):
        """Scalar division should work correctly."""
        v = Vector3(2, 4, 6)
        result = v / 2
        assert result.x == 1
        assert result.y == 2
        assert result.z == 3

    def test_division_by_zero(self):
        """Division by zero should raise error."""
        v = Vector3(1, 2, 3)
        with pytest.raises(ZeroDivisionError):
            _ = v / 0

    def test_negation(self):
        """Negation should work correctly."""
        v = Vector3(1, 2, 3)
        neg = -v
        assert neg.x == -1
        assert neg.y == -2
        assert neg.z == -3

    def test_dot_product(self):
        """Dot product should be correct."""
        a = Vector3(1, 0, 0)
        b = Vector3(0, 1, 0)
        assert a.dot(b) == 0  # Perpendicular

        c = Vector3(1, 2, 3)
        d = Vector3(4, 5, 6)
        assert c.dot(d) == 1*4 + 2*5 + 3*6  # = 32

    def test_cross_product(self):
        """Cross product should be correct."""
        i = Vector3(1, 0, 0)
        j = Vector3(0, 1, 0)
        k = i.cross(j)
        assert k.x == 0
        assert k.y == 0
        assert k.z == 1  # i x j = k

    def test_magnitude(self):
        """Magnitude should be correct."""
        v = Vector3(3, 4, 0)
        assert v.magnitude() == 5  # 3-4-5 triangle

    def test_magnitude_squared(self):
        """Magnitude squared should avoid sqrt."""
        v = Vector3(3, 4, 0)
        assert v.magnitude_squared() == 25

    def test_normalized(self):
        """Normalized should return unit vector."""
        v = Vector3(3, 4, 0)
        n = v.normalized()
        assert abs(n.magnitude() - 1.0) < 0.0001
        assert n.x == 3/5
        assert n.y == 4/5

    def test_normalized_zero_vector(self):
        """Normalizing zero vector should return zero."""
        v = Vector3(0, 0, 0)
        n = v.normalized()
        assert n.magnitude() == 0

    def test_copy(self):
        """Copy should create independent copy."""
        v = Vector3(1, 2, 3)
        c = v.copy()
        c.x = 10
        assert v.x == 1  # Original unchanged

    def test_class_methods(self):
        """Class factory methods should work."""
        assert Vector3.zero().magnitude() == 0
        assert Vector3.up().y == 1
        assert Vector3.forward().z == 1
        assert Vector3.right().x == 1


# =============================================================================
# Transform Tests
# =============================================================================


class TestTransform:
    """Tests for Transform dataclass."""

    def test_default_values(self):
        """Transform should have identity defaults."""
        t = Transform()
        assert t.position.magnitude() == 0
        assert t.rotation.magnitude() == 0
        assert t.scale.x == 1 and t.scale.y == 1 and t.scale.z == 1

    def test_copy(self):
        """Copy should be deep copy."""
        t = Transform()
        t.position = Vector3(1, 2, 3)
        c = t.copy()
        c.position.x = 10
        assert t.position.x == 1  # Original unchanged


# =============================================================================
# VehicleGroup Tests
# =============================================================================


class TestVehicleGroup:
    """Tests for VehicleGroup."""

    def test_creation(self):
        """Group should initialize correctly."""
        group = VehicleGroup(name="test-group")
        assert group.name == "test-group"
        assert len(group.vehicle_ids) == 0
        assert group.enabled

    def test_add_vehicle(self):
        """Should add vehicle to group."""
        group = VehicleGroup(name="test")
        group.add("vehicle-1")
        assert "vehicle-1" in group.vehicle_ids

    def test_remove_vehicle(self):
        """Should remove vehicle from group."""
        group = VehicleGroup(name="test")
        group.add("vehicle-1")
        group.remove("vehicle-1")
        assert "vehicle-1" not in group.vehicle_ids

    def test_remove_nonexistent(self):
        """Removing non-existent should not error."""
        group = VehicleGroup(name="test")
        group.remove("nonexistent")  # Should not raise


# =============================================================================
# CollisionInfo Tests
# =============================================================================


class TestCollisionInfo:
    """Tests for CollisionInfo dataclass."""

    def test_vehicle_vehicle_collision(self):
        """Should store vehicle-vehicle collision."""
        info = CollisionInfo(
            vehicle_a_id="car-1",
            vehicle_b_id="car-2",
            contact_point=Vector3(0, 0, 0),
            contact_normal=Vector3(0, 1, 0),
            penetration_depth=0.01,
            relative_velocity=10.0,
        )
        assert info.vehicle_a_id == "car-1"
        assert info.vehicle_b_id == "car-2"

    def test_vehicle_static_collision(self):
        """Should handle vehicle-static collision."""
        info = CollisionInfo(
            vehicle_a_id="car-1",
            vehicle_b_id=None,  # Static geometry
            contact_point=Vector3(0, 0, 0),
            contact_normal=Vector3(0, 1, 0),
            penetration_depth=0.05,
            relative_velocity=5.0,
        )
        assert info.vehicle_b_id is None


# =============================================================================
# VehicleSystem Tests
# =============================================================================


class TestVehicleSystem:
    """Tests for VehicleSystem manager."""

    @pytest.fixture
    def system(self):
        """Create a vehicle system."""
        return VehicleSystem()

    @pytest.fixture
    def mock_vehicle(self):
        """Create a mock vehicle for testing."""
        class MockVehicle:
            def __init__(self, vehicle_id=None):
                self.vehicle_id = vehicle_id or generate_vehicle_id()
                self.vehicle_type = VehicleType.WHEELED
                self.state = VehicleState.ACTIVE
                self.transform = Transform()
                self.velocity = Vector3.zero()
                self.angular_velocity = Vector3.zero()
                self.mass = 1500.0

            def update(self, dt):
                pass

            def apply_force(self, force, position=None):
                pass

            def apply_torque(self, torque):
                pass

            def reset(self):
                pass

        return MockVehicle

    def test_initialization(self, system):
        """System should initialize correctly."""
        assert system.vehicle_count == 0
        assert system.gravity > 0

    def test_gravity_setter(self, system):
        """Gravity should be settable."""
        system.gravity = 10.0
        assert system.gravity == 10.0

    def test_gravity_rejects_negative(self, system):
        """Negative gravity should be rejected."""
        with pytest.raises(ValueError):
            system.gravity = -1.0

    def test_substeps_setter(self, system):
        """Substeps should be settable."""
        system.substeps = 8
        assert system.substeps == 8

    def test_substeps_rejects_zero(self, system):
        """Zero substeps should be rejected."""
        with pytest.raises(ValueError):
            system.substeps = 0

    def test_register_vehicle(self, system, mock_vehicle):
        """Should register vehicle."""
        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)
        assert vehicle_id == vehicle.vehicle_id
        assert system.vehicle_count == 1

    def test_register_duplicate_fails(self, system, mock_vehicle):
        """Registering duplicate ID should fail."""
        vehicle = mock_vehicle(vehicle_id="unique-id")
        system.register_vehicle(vehicle)

        vehicle2 = mock_vehicle(vehicle_id="unique-id")
        with pytest.raises(ValueError, match="already registered"):
            system.register_vehicle(vehicle2)

    def test_unregister_vehicle(self, system, mock_vehicle):
        """Should unregister vehicle."""
        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)

        result = system.unregister_vehicle(vehicle_id)
        assert result
        assert system.vehicle_count == 0

    def test_unregister_nonexistent(self, system):
        """Unregistering non-existent should return False."""
        result = system.unregister_vehicle("nonexistent")
        assert not result

    def test_get_vehicle(self, system, mock_vehicle):
        """Should retrieve vehicle by ID."""
        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)

        retrieved = system.get_vehicle(vehicle_id)
        assert retrieved is vehicle

    def test_get_vehicle_not_found(self, system):
        """Should return None for unknown ID."""
        result = system.get_vehicle("nonexistent")
        assert result is None

    def test_get_vehicles_by_type(self, system, mock_vehicle):
        """Should filter vehicles by type."""
        v1 = mock_vehicle()
        v1.vehicle_type = VehicleType.WHEELED
        v2 = mock_vehicle()
        v2.vehicle_type = VehicleType.TRACKED
        v3 = mock_vehicle()
        v3.vehicle_type = VehicleType.WHEELED

        system.register_vehicle(v1)
        system.register_vehicle(v2)
        system.register_vehicle(v3)

        wheeled = system.get_vehicles_by_type(VehicleType.WHEELED)
        assert len(wheeled) == 2

    def test_create_group(self, system):
        """Should create vehicle group."""
        group = system.create_group("test-group")
        assert group.name == "test-group"

    def test_create_duplicate_group_fails(self, system):
        """Creating duplicate group should fail."""
        system.create_group("test-group")
        with pytest.raises(ValueError, match="already exists"):
            system.create_group("test-group")

    def test_add_to_group(self, system, mock_vehicle):
        """Should add vehicle to group."""
        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)
        system.create_group("test-group")

        result = system.add_to_group(vehicle_id, "test-group")
        assert result

        vehicles_in_group = system.get_vehicles_in_group("test-group")
        assert len(vehicles_in_group) == 1

    def test_add_to_nonexistent_group(self, system, mock_vehicle):
        """Adding to non-existent group should fail."""
        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)

        result = system.add_to_group(vehicle_id, "nonexistent")
        assert not result

    def test_remove_from_group(self, system, mock_vehicle):
        """Should remove vehicle from group."""
        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)
        system.create_group("test-group")
        system.add_to_group(vehicle_id, "test-group")

        result = system.remove_from_group(vehicle_id, "test-group")
        assert result

        vehicles_in_group = system.get_vehicles_in_group("test-group")
        assert len(vehicles_in_group) == 0

    def test_update(self, system, mock_vehicle):
        """Update should process all vehicles."""
        v1 = mock_vehicle()
        v2 = mock_vehicle()
        system.register_vehicle(v1)
        system.register_vehicle(v2)

        system.update(dt=0.016)
        # Should not crash

    def test_update_zero_dt(self, system, mock_vehicle):
        """Zero dt should return early."""
        vehicle = mock_vehicle()
        system.register_vehicle(vehicle)

        system.update(dt=0.0)
        # Should not crash

    def test_collision_callback(self, system):
        """Should call collision callbacks."""
        collisions = []

        def on_collision(info):
            collisions.append(info)

        system.register_collision_callback(on_collision)

        collision = CollisionInfo(
            vehicle_a_id="car-1",
            vehicle_b_id="car-2",
            contact_point=Vector3.zero(),
            contact_normal=Vector3.up(),
            penetration_depth=0.01,
            relative_velocity=5.0,
        )
        system.notify_collision(collision)

        assert len(collisions) == 1

    def test_vehicle_added_callback(self, system, mock_vehicle):
        """Should call added callbacks."""
        added_ids = []

        def on_added(vehicle_id):
            added_ids.append(vehicle_id)

        system.on_vehicle_added(on_added)

        vehicle = mock_vehicle()
        system.register_vehicle(vehicle)

        assert vehicle.vehicle_id in added_ids

    def test_vehicle_removed_callback(self, system, mock_vehicle):
        """Should call removed callbacks."""
        removed_ids = []

        def on_removed(vehicle_id):
            removed_ids.append(vehicle_id)

        system.on_vehicle_removed(on_removed)

        vehicle = mock_vehicle()
        vehicle_id = system.register_vehicle(vehicle)
        system.unregister_vehicle(vehicle_id)

        assert vehicle_id in removed_ids

    def test_iter_vehicles(self, system, mock_vehicle):
        """Should iterate over all vehicles."""
        v1 = mock_vehicle()
        v2 = mock_vehicle()
        system.register_vehicle(v1)
        system.register_vehicle(v2)

        vehicles = list(system.iter_vehicles())
        assert len(vehicles) == 2

    def test_iter_active_vehicles(self, system, mock_vehicle):
        """Should iterate only active vehicles."""
        v1 = mock_vehicle()
        v2 = mock_vehicle()
        v2.state = VehicleState.SLEEPING
        system.register_vehicle(v1)
        system.register_vehicle(v2)

        active = list(system.iter_active_vehicles())
        assert len(active) == 1

    def test_clear(self, system, mock_vehicle):
        """Clear should remove all vehicles."""
        v1 = mock_vehicle()
        v2 = mock_vehicle()
        system.register_vehicle(v1)
        system.register_vehicle(v2)

        system.clear()
        assert system.vehicle_count == 0

    def test_get_stats(self, system, mock_vehicle):
        """Should return system statistics."""
        v1 = mock_vehicle()
        system.register_vehicle(v1)

        stats = system.get_stats()
        assert "total_vehicles" in stats
        assert stats["total_vehicles"] == 1


# =============================================================================
# generate_vehicle_id Tests
# =============================================================================


class TestGenerateVehicleId:
    """Tests for vehicle ID generation."""

    def test_generates_string(self):
        """Should generate string ID."""
        id1 = generate_vehicle_id()
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_unique_ids(self):
        """Should generate unique IDs."""
        ids = [generate_vehicle_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique


# =============================================================================
# Sleeping Tests
# =============================================================================


class TestSleeping:
    """Tests for vehicle sleeping behavior."""

    @pytest.fixture
    def system(self):
        """Create system with sleeping enabled."""
        return VehicleSystem(enable_sleeping=True)

    @pytest.fixture
    def mock_vehicle(self):
        """Create mock vehicle."""
        class MockVehicle:
            def __init__(self):
                self.vehicle_id = generate_vehicle_id()
                self.vehicle_type = VehicleType.WHEELED
                self.state = VehicleState.ACTIVE
                self.transform = Transform()
                self.velocity = Vector3.zero()
                self.angular_velocity = Vector3.zero()
                self.mass = 1500.0
                self.force_applied = False

            def update(self, dt):
                pass

            def apply_force(self, force, position=None):
                self.force_applied = True

            def apply_torque(self, torque):
                pass

            def reset(self):
                pass

        return MockVehicle

    def test_wake_vehicle(self, system, mock_vehicle):
        """Should wake sleeping vehicle."""
        vehicle = mock_vehicle()
        vehicle.state = VehicleState.SLEEPING
        system.register_vehicle(vehicle)

        result = system.wake_vehicle(vehicle.vehicle_id)
        assert result
        assert vehicle.state == VehicleState.ACTIVE

    def test_wake_already_active(self, system, mock_vehicle):
        """Waking active vehicle should return False."""
        vehicle = mock_vehicle()
        system.register_vehicle(vehicle)

        result = system.wake_vehicle(vehicle.vehicle_id)
        assert not result

    def test_wake_nonexistent(self, system):
        """Waking non-existent should return False."""
        result = system.wake_vehicle("nonexistent")
        assert not result
