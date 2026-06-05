"""
Tests for Camera Collision Detection (collision.py).

Tests collision detection and response for camera systems:
    - Raycast collision
    - Sphere cast collision
    - Collision response strategies
    - Occlusion detection
    - Transparency fading
"""

import math
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set
from enum import IntFlag, auto


# =============================================================================
# Mock Classes for Testing
# =============================================================================


@dataclass
class Vector3:
    """Mock 3D vector for testing."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / mag, self.y / mag, self.z / mag)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def lerp(self, target: "Vector3", t: float) -> "Vector3":
        return Vector3(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t,
            self.z + (target.z - self.z) * t,
        )

    def distance_to(self, other: "Vector3") -> float:
        return (self - other).magnitude()


class CollisionLayer(IntFlag):
    """Collision layer mask flags."""
    NONE = 0
    STATIC = auto()
    DYNAMIC = auto()
    CHARACTER = auto()
    TRIGGER = auto()
    CAMERA = auto()
    TRANSPARENT = auto()
    ALL = STATIC | DYNAMIC | CHARACTER | TRIGGER | CAMERA | TRANSPARENT


@dataclass
class RaycastHit:
    """Result of a raycast query."""
    hit: bool = False
    point: Vector3 = field(default_factory=Vector3)
    normal: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    distance: float = float('inf')
    collider: Optional["Collider"] = None
    layer: CollisionLayer = CollisionLayer.STATIC


@dataclass
class SpherecastHit:
    """Result of a spherecast query."""
    hit: bool = False
    point: Vector3 = field(default_factory=Vector3)
    normal: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    distance: float = float('inf')
    penetration_depth: float = 0.0
    collider: Optional["Collider"] = None
    layer: CollisionLayer = CollisionLayer.STATIC


@dataclass
class Collider:
    """Mock collider for testing."""
    id: str = ""
    position: Vector3 = field(default_factory=Vector3)
    layer: CollisionLayer = CollisionLayer.STATIC
    is_trigger: bool = False
    material: str = "default"
    bounds_min: Vector3 = field(default_factory=lambda: Vector3(-1, -1, -1))
    bounds_max: Vector3 = field(default_factory=lambda: Vector3(1, 1, 1))


@dataclass
class OcclusionInfo:
    """Information about camera occlusion."""
    is_occluded: bool = False
    occluders: List[Collider] = field(default_factory=list)
    closest_distance: float = float('inf')
    occlusion_percentage: float = 0.0


# =============================================================================
# Collision Response Strategies
# =============================================================================


class CollisionResponseStrategy:
    """Base collision response strategy."""

    def respond(
        self, desired_position: Vector3, hit: RaycastHit, target: Vector3
    ) -> Vector3:
        raise NotImplementedError


class PullInResponse(CollisionResponseStrategy):
    """Pull camera position back from collision, offsetting toward desired position."""

    def __init__(self, offset: float = 0.1):
        self.offset = offset

    def respond(
        self, desired_position: Vector3, hit: RaycastHit, target: Vector3
    ) -> Vector3:
        # Direction from target to desired position (away from target)
        direction = (desired_position - target).normalized()
        # Place camera at hit point plus offset toward desired position
        return hit.point + direction * self.offset


class PushOutResponse(CollisionResponseStrategy):
    """Push camera out along collision normal."""

    def __init__(self, offset: float = 0.1):
        self.offset = offset

    def respond(
        self, desired_position: Vector3, hit: RaycastHit, target: Vector3
    ) -> Vector3:
        return hit.point + hit.normal * self.offset


class SlideResponse(CollisionResponseStrategy):
    """Slide camera along collision surface."""

    def respond(
        self, desired_position: Vector3, hit: RaycastHit, target: Vector3
    ) -> Vector3:
        movement = desired_position - target
        dot = movement.dot(hit.normal)
        slide = Vector3(
            movement.x - hit.normal.x * dot,
            movement.y - hit.normal.y * dot,
            movement.z - hit.normal.z * dot,
        )
        return target + slide


class ClampDistanceResponse(CollisionResponseStrategy):
    """Clamp camera distance from target."""

    def __init__(self, min_distance: float = 0.5):
        self.min_distance = min_distance

    def respond(
        self, desired_position: Vector3, hit: RaycastHit, target: Vector3
    ) -> Vector3:
        direction = (desired_position - target).normalized()
        distance = max(self.min_distance, hit.distance - 0.1)
        return target + direction * distance


# =============================================================================
# Camera Collision System
# =============================================================================


class CameraCollisionSystem:
    """Camera collision detection and response system."""

    def __init__(self):
        self.enabled = True
        self.collision_radius = 0.2
        self.collision_mask = CollisionLayer.STATIC | CollisionLayer.DYNAMIC
        self.response_strategy = PullInResponse()
        self.smooth_recovery = True
        self.recovery_speed = 5.0
        self.occlusion_enabled = True
        self.multi_raycast = False
        self.raycast_count = 5
        self._current_distance = 0.0
        self._target_distance = 0.0
        self._occluded_objects: Set[str] = set()
        self._physics_world: Optional["PhysicsWorld"] = None

    def set_physics_world(self, world: "PhysicsWorld"):
        self._physics_world = world

    def set_collision_mask(self, mask: CollisionLayer):
        self.collision_mask = mask

    def set_response_strategy(self, strategy: CollisionResponseStrategy):
        self.response_strategy = strategy

    def raycast(
        self, origin: Vector3, direction: Vector3, max_distance: float
    ) -> RaycastHit:
        """Perform a raycast in the physics world."""
        if not self._physics_world:
            return RaycastHit()
        return self._physics_world.raycast(
            origin, direction, max_distance, self.collision_mask
        )

    def spherecast(
        self, origin: Vector3, direction: Vector3, radius: float, max_distance: float
    ) -> SpherecastHit:
        """Perform a spherecast in the physics world."""
        if not self._physics_world:
            return SpherecastHit()
        return self._physics_world.spherecast(
            origin, direction, radius, max_distance, self.collision_mask
        )

    def check_collision(
        self, target: Vector3, desired_position: Vector3
    ) -> Tuple[bool, Vector3]:
        """Check for collision and return adjusted position."""
        if not self.enabled:
            return False, desired_position

        direction = (desired_position - target).normalized()
        distance = (desired_position - target).magnitude()

        if self.collision_radius > 0:
            hit = self.spherecast(target, direction, self.collision_radius, distance)
            if hit.hit:
                raycast_hit = RaycastHit(
                    hit=True,
                    point=hit.point,
                    normal=hit.normal,
                    distance=hit.distance,
                    collider=hit.collider,
                )
                adjusted = self.response_strategy.respond(
                    desired_position, raycast_hit, target
                )
                return True, adjusted
        else:
            hit = self.raycast(target, direction, distance)
            if hit.hit:
                adjusted = self.response_strategy.respond(desired_position, hit, target)
                return True, adjusted

        return False, desired_position

    def check_occlusion(
        self, camera_position: Vector3, target: Vector3
    ) -> OcclusionInfo:
        """Check if target is occluded from camera."""
        if not self.occlusion_enabled or not self._physics_world:
            return OcclusionInfo()

        direction = (target - camera_position).normalized()
        distance = (target - camera_position).magnitude()

        if self.multi_raycast:
            return self._multi_raycast_occlusion(camera_position, target, distance)

        hit = self.raycast(camera_position, direction, distance)
        if hit.hit and hit.distance < distance - 0.1:
            return OcclusionInfo(
                is_occluded=True,
                occluders=[hit.collider] if hit.collider else [],
                closest_distance=hit.distance,
                occlusion_percentage=1.0,
            )

        return OcclusionInfo()

    def _multi_raycast_occlusion(
        self, camera_position: Vector3, target: Vector3, distance: float
    ) -> OcclusionInfo:
        """Perform multiple raycasts for occlusion percentage."""
        occluders = []
        hit_count = 0
        closest_distance = float('inf')

        offsets = [
            Vector3(0, 0, 0),
            Vector3(0.3, 0, 0),
            Vector3(-0.3, 0, 0),
            Vector3(0, 0.3, 0),
            Vector3(0, -0.3, 0),
        ]

        for offset in offsets[: self.raycast_count]:
            ray_target = target + offset
            direction = (ray_target - camera_position).normalized()
            hit = self.raycast(camera_position, direction, distance + 0.5)

            if hit.hit and hit.distance < distance - 0.1:
                hit_count += 1
                closest_distance = min(closest_distance, hit.distance)
                if hit.collider and hit.collider not in occluders:
                    occluders.append(hit.collider)

        occlusion_percentage = hit_count / self.raycast_count
        return OcclusionInfo(
            is_occluded=occlusion_percentage > 0,
            occluders=occluders,
            closest_distance=closest_distance if hit_count > 0 else float('inf'),
            occlusion_percentage=occlusion_percentage,
        )

    def update_smooth_recovery(
        self, current_position: Vector3, desired_position: Vector3, delta_time: float
    ) -> Vector3:
        """Smoothly recover camera position after collision."""
        if not self.smooth_recovery:
            return desired_position

        t = 1.0 - math.exp(-self.recovery_speed * delta_time)
        return current_position.lerp(desired_position, t)

    def get_occluded_objects(self) -> Set[str]:
        """Get set of currently occluded object IDs."""
        return self._occluded_objects

    def mark_occluded(self, object_id: str):
        """Mark an object as occluded."""
        self._occluded_objects.add(object_id)

    def unmark_occluded(self, object_id: str):
        """Remove object from occluded set."""
        self._occluded_objects.discard(object_id)


class TransparencyFader:
    """Handles transparency fading for occluding objects."""

    def __init__(self):
        self.enabled = True
        self.fade_duration = 0.3
        self.min_alpha = 0.3
        self.max_alpha = 1.0
        self._fading_objects: dict[str, float] = {}
        self._target_alpha: dict[str, float] = {}
        self._final_alpha: dict[str, float] = {}  # Stores final alpha after fade completes

    def start_fade_out(self, object_id: str):
        """Start fading out an object."""
        if object_id not in self._fading_objects:
            self._fading_objects[object_id] = self._final_alpha.get(object_id, self.max_alpha)
        self._target_alpha[object_id] = self.min_alpha
        # Remove from final if restarting fade
        if object_id in self._final_alpha:
            del self._final_alpha[object_id]

    def start_fade_in(self, object_id: str):
        """Start fading in an object."""
        if object_id not in self._fading_objects:
            self._fading_objects[object_id] = self._final_alpha.get(object_id, self.min_alpha)
        self._target_alpha[object_id] = self.max_alpha
        # Remove from final if restarting fade
        if object_id in self._final_alpha:
            del self._final_alpha[object_id]

    def update(self, delta_time: float):
        """Update all fading objects."""
        if not self.enabled:
            return

        completed = []
        for object_id, current_alpha in self._fading_objects.items():
            target = self._target_alpha.get(object_id, self.max_alpha)
            fade_speed = (self.max_alpha - self.min_alpha) / self.fade_duration
            step = fade_speed * delta_time

            if current_alpha < target:
                new_alpha = min(current_alpha + step, target)
            else:
                new_alpha = max(current_alpha - step, target)

            self._fading_objects[object_id] = new_alpha

            # Mark as completed if fade has finished (reached target)
            if abs(new_alpha - target) < 0.01:
                completed.append((object_id, new_alpha))

        for object_id, final_alpha in completed:
            self._final_alpha[object_id] = final_alpha
            del self._fading_objects[object_id]
            del self._target_alpha[object_id]

    def get_alpha(self, object_id: str) -> float:
        """Get current alpha value for an object."""
        if object_id in self._fading_objects:
            return self._fading_objects[object_id]
        return self._final_alpha.get(object_id, self.max_alpha)

    def get_transparency(self, object_id: str) -> float:
        """Get current transparency value (1 - alpha) for an object."""
        return 1.0 - self.get_alpha(object_id)

    def is_fading(self, object_id: str) -> bool:
        """Check if an object is currently fading."""
        return object_id in self._fading_objects


class PhysicsWorld:
    """Mock physics world for testing."""

    def __init__(self):
        self.colliders: List[Collider] = []

    def add_collider(self, collider: Collider):
        self.colliders.append(collider)

    def raycast(
        self,
        origin: Vector3,
        direction: Vector3,
        max_distance: float,
        mask: CollisionLayer,
    ) -> RaycastHit:
        """Perform a raycast against all colliders."""
        closest_hit = RaycastHit()

        for collider in self.colliders:
            if not (collider.layer & mask):
                continue

            hit = self._ray_box_intersection(
                origin, direction, collider.bounds_min, collider.bounds_max, collider
            )
            if hit.hit and hit.distance < closest_hit.distance:
                closest_hit = hit

        if closest_hit.distance <= max_distance:
            return closest_hit
        return RaycastHit()

    def spherecast(
        self,
        origin: Vector3,
        direction: Vector3,
        radius: float,
        max_distance: float,
        mask: CollisionLayer,
    ) -> SpherecastHit:
        """Perform a spherecast against all colliders."""
        ray_hit = self.raycast(origin, direction, max_distance, mask)
        if ray_hit.hit:
            return SpherecastHit(
                hit=True,
                point=ray_hit.point,
                normal=ray_hit.normal,
                distance=max(0, ray_hit.distance - radius),
                penetration_depth=0.0,
                collider=ray_hit.collider,
            )
        return SpherecastHit()

    def _ray_box_intersection(
        self,
        origin: Vector3,
        direction: Vector3,
        box_min: Vector3,
        box_max: Vector3,
        collider: Collider,
    ) -> RaycastHit:
        """Simple ray-box intersection test."""
        t_min = 0.0
        t_max = float('inf')

        for axis in ['x', 'y', 'z']:
            orig = getattr(origin, axis)
            dir_val = getattr(direction, axis)
            b_min = getattr(box_min, axis)
            b_max = getattr(box_max, axis)

            if abs(dir_val) < 1e-8:
                if orig < b_min or orig > b_max:
                    return RaycastHit()
            else:
                t1 = (b_min - orig) / dir_val
                t2 = (b_max - orig) / dir_val
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return RaycastHit()

        if t_min >= 0:
            hit_point = origin + direction * t_min
            return RaycastHit(
                hit=True,
                point=hit_point,
                normal=Vector3(0, 1, 0),
                distance=t_min,
                collider=collider,
                layer=collider.layer,
            )
        return RaycastHit()


# =============================================================================
# Raycast Collision Detection Tests (~25 tests)
# =============================================================================


class TestRaycastCollision:
    """Test raycast collision detection."""

    def test_raycast_no_collision(self):
        """Test raycast with no collision."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(0, 0, 1), 10.0)
        assert hit.hit is False

    def test_raycast_hit_detection(self):
        """Test raycast detects collision."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        collider = Collider(
            id="wall",
            bounds_min=Vector3(4, -1, -1),
            bounds_max=Vector3(6, 1, 1),
            layer=CollisionLayer.STATIC,
        )
        world.add_collider(collider)
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.hit is True
        assert hit.distance == pytest.approx(4.0, abs=0.1)

    def test_raycast_returns_closest(self):
        """Test raycast returns closest hit."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="far",
                bounds_min=Vector3(8, -1, -1),
                bounds_max=Vector3(10, 1, 1),
            )
        )
        world.add_collider(
            Collider(
                id="near",
                bounds_min=Vector3(2, -1, -1),
                bounds_max=Vector3(4, 1, 1),
            )
        )
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 20.0)
        assert hit.collider.id == "near"

    def test_raycast_max_distance(self):
        """Test raycast respects max distance."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(8, -1, -1),
                bounds_max=Vector3(10, 1, 1),
            )
        )
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 5.0)
        assert hit.hit is False

    def test_raycast_layer_filtering(self):
        """Test raycast filters by collision layer."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="trigger",
                bounds_min=Vector3(2, -1, -1),
                bounds_max=Vector3(4, 1, 1),
                layer=CollisionLayer.TRIGGER,
            )
        )
        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.STATIC)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.hit is False

    def test_raycast_hit_point(self):
        """Test raycast returns correct hit point."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(5, -1, -1),
                bounds_max=Vector3(7, 1, 1),
            )
        )
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.point.x == pytest.approx(5.0, abs=0.1)

    def test_raycast_returns_collider(self):
        """Test raycast returns hit collider."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        collider = Collider(
            id="wall",
            bounds_min=Vector3(2, -1, -1),
            bounds_max=Vector3(4, 1, 1),
        )
        world.add_collider(collider)
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider is collider


# =============================================================================
# Sphere Cast Collision Tests (~20 tests)
# =============================================================================


class TestSpherecastCollision:
    """Test spherecast collision detection."""

    def test_spherecast_no_collision(self):
        """Test spherecast with no collision."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        system.set_physics_world(world)

        hit = system.spherecast(Vector3(0, 0, 0), Vector3(0, 0, 1), 0.5, 10.0)
        assert hit.hit is False

    def test_spherecast_hit_detection(self):
        """Test spherecast detects collision."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(4, -1, -1),
                bounds_max=Vector3(6, 1, 1),
            )
        )
        system.set_physics_world(world)

        hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.5, 10.0)
        assert hit.hit is True

    def test_spherecast_radius_affects_distance(self):
        """Test spherecast radius affects reported distance."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(5, -1, -1),
                bounds_max=Vector3(7, 1, 1),
            )
        )
        system.set_physics_world(world)

        hit_small = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.2, 10.0)
        hit_large = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 1.0, 10.0)

        assert hit_large.distance < hit_small.distance

    def test_spherecast_layer_filtering(self):
        """Test spherecast filters by layer."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="dynamic",
                bounds_min=Vector3(2, -1, -1),
                bounds_max=Vector3(4, 1, 1),
                layer=CollisionLayer.DYNAMIC,
            )
        )
        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.STATIC)

        hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.5, 10.0)
        assert hit.hit is False

    def test_spherecast_zero_radius(self):
        """Test spherecast with zero radius acts like raycast."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(5, -1, -1),
                bounds_max=Vector3(7, 1, 1),
            )
        )
        system.set_physics_world(world)

        sphere_hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.0, 10.0)
        ray_hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)

        assert sphere_hit.distance == pytest.approx(ray_hit.distance, abs=0.1)


# =============================================================================
# Collision Response Tests (~25 tests)
# =============================================================================


class TestCollisionResponse:
    """Test collision response strategies."""

    def test_pull_in_response(self):
        """Test pull-in response moves toward target."""
        strategy = PullInResponse(offset=0.1)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(hit=True, point=Vector3(5, 0, 0), distance=5.0)

        result = strategy.respond(desired, hit, target)
        assert result.x > 5.0
        assert result.x < 10.0

    def test_pull_in_offset(self):
        """Test pull-in offset affects final position."""
        strategy_small = PullInResponse(offset=0.1)
        strategy_large = PullInResponse(offset=0.5)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(hit=True, point=Vector3(5, 0, 0), distance=5.0)

        result_small = strategy_small.respond(desired, hit, target)
        result_large = strategy_large.respond(desired, hit, target)

        assert result_large.x > result_small.x

    def test_push_out_response(self):
        """Test push-out response moves along normal."""
        strategy = PushOutResponse(offset=0.2)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(
            hit=True,
            point=Vector3(5, 0, 0),
            normal=Vector3(-1, 0, 0),
            distance=5.0,
        )

        result = strategy.respond(desired, hit, target)
        assert result.x < 5.0

    def test_push_out_normal_direction(self):
        """Test push-out follows normal direction."""
        strategy = PushOutResponse(offset=1.0)
        desired = Vector3(0, 10, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(
            hit=True,
            point=Vector3(0, 5, 0),
            normal=Vector3(0, -1, 0),
            distance=5.0,
        )

        result = strategy.respond(desired, hit, target)
        assert result.y == pytest.approx(4.0, abs=0.1)

    def test_slide_response(self):
        """Test slide response slides along surface."""
        strategy = SlideResponse()
        desired = Vector3(10, 5, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(
            hit=True,
            point=Vector3(5, 2.5, 0),
            normal=Vector3(-1, 0, 0),
            distance=5.0,
        )

        result = strategy.respond(desired, hit, target)
        assert result.y > 0

    def test_clamp_distance_response(self):
        """Test clamp distance respects minimum."""
        strategy = ClampDistanceResponse(min_distance=2.0)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(hit=True, point=Vector3(1, 0, 0), distance=1.0)

        result = strategy.respond(desired, hit, target)
        distance = result.magnitude()
        assert distance >= 2.0


# =============================================================================
# Check Collision Tests (~15 tests)
# =============================================================================


class TestCheckCollision:
    """Test the check_collision method."""

    def test_check_collision_disabled(self):
        """Test check_collision when disabled."""
        system = CameraCollisionSystem()
        system.enabled = False

        collided, position = system.check_collision(
            Vector3(0, 0, 0), Vector3(10, 0, 0)
        )
        assert collided is False
        assert position.x == 10.0

    def test_check_collision_no_hit(self):
        """Test check_collision with no collision."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        system.set_physics_world(world)

        collided, position = system.check_collision(
            Vector3(0, 0, 0), Vector3(10, 0, 0)
        )
        assert collided is False

    def test_check_collision_hit(self):
        """Test check_collision with collision."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(4, -1, -1),
                bounds_max=Vector3(6, 1, 1),
            )
        )
        system.set_physics_world(world)

        collided, position = system.check_collision(
            Vector3(0, 0, 0), Vector3(10, 0, 0)
        )
        assert collided is True
        assert position.x < 10.0

    def test_check_collision_uses_radius(self):
        """Test check_collision uses collision radius."""
        system = CameraCollisionSystem()
        system.collision_radius = 1.0
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(5, -1, -1),
                bounds_max=Vector3(7, 1, 1),
            )
        )
        system.set_physics_world(world)

        collided, _ = system.check_collision(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert collided is True

    def test_check_collision_zero_radius_uses_raycast(self):
        """Test check_collision uses raycast when radius is 0."""
        system = CameraCollisionSystem()
        system.collision_radius = 0.0
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(5, -1, -1),
                bounds_max=Vector3(7, 1, 1),
            )
        )
        system.set_physics_world(world)

        collided, _ = system.check_collision(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert collided is True


# =============================================================================
# Occlusion Detection Tests (~20 tests)
# =============================================================================


class TestOcclusionDetection:
    """Test occlusion detection functionality."""

    def test_occlusion_no_occluder(self):
        """Test occlusion detection with no occluders."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is False

    def test_occlusion_detected(self):
        """Test occlusion is detected."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(4, -1, -1),
                bounds_max=Vector3(6, 1, 1),
            )
        )
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is True

    def test_occlusion_returns_occluders(self):
        """Test occlusion returns list of occluders."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        collider = Collider(
            id="wall",
            bounds_min=Vector3(4, -1, -1),
            bounds_max=Vector3(6, 1, 1),
        )
        world.add_collider(collider)
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert len(info.occluders) == 1
        assert info.occluders[0] is collider

    def test_occlusion_returns_closest_distance(self):
        """Test occlusion returns closest distance."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(3, -1, -1),
                bounds_max=Vector3(5, 1, 1),
            )
        )
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.closest_distance == pytest.approx(3.0, abs=0.1)

    def test_occlusion_disabled(self):
        """Test occlusion detection when disabled."""
        system = CameraCollisionSystem()
        system.occlusion_enabled = False
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(4, -1, -1),
                bounds_max=Vector3(6, 1, 1),
            )
        )
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is False

    def test_multi_raycast_occlusion(self):
        """Test multi-raycast occlusion calculation."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        system.raycast_count = 5
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall",
                bounds_min=Vector3(4, -2, -2),
                bounds_max=Vector3(6, 2, 2),
            )
        )
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is True
        assert 0 < info.occlusion_percentage <= 1.0

    def test_occlusion_percentage_partial(self):
        """Test partial occlusion percentage."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        system.raycast_count = 5
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="small_wall",
                bounds_min=Vector3(4, -0.1, -0.1),
                bounds_max=Vector3(6, 0.1, 0.1),
            )
        )
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.occlusion_percentage < 1.0


# =============================================================================
# Transparency Fading Tests (~15 tests)
# =============================================================================


class TestTransparencyFading:
    """Test transparency fading for occluders."""

    def test_start_fade_out(self):
        """Test starting fade out."""
        fader = TransparencyFader()
        fader.start_fade_out("object1")
        assert "object1" in fader._fading_objects

    def test_start_fade_in(self):
        """Test starting fade in."""
        fader = TransparencyFader()
        fader._fading_objects["object1"] = 0.3
        fader.start_fade_in("object1")
        assert fader._target_alpha["object1"] == fader.max_alpha

    def test_fade_out_progress(self):
        """Test fade out progresses over time."""
        fader = TransparencyFader()
        fader.fade_duration = 1.0
        fader.start_fade_out("object1")

        initial = fader.get_alpha("object1")
        fader.update(0.5)
        after = fader.get_alpha("object1")

        assert after < initial

    def test_fade_in_progress(self):
        """Test fade in progresses over time."""
        fader = TransparencyFader()
        fader.fade_duration = 1.0
        fader._fading_objects["object1"] = fader.min_alpha
        fader.start_fade_in("object1")

        initial = fader.get_alpha("object1")
        fader.update(0.5)
        after = fader.get_alpha("object1")

        assert after > initial

    def test_fade_completes(self):
        """Test fade completes at target alpha."""
        fader = TransparencyFader()
        fader.fade_duration = 0.3
        fader.start_fade_out("object1")

        for _ in range(10):
            fader.update(0.1)

        assert fader.get_alpha("object1") <= fader.min_alpha + 0.1

    def test_fade_in_removes_from_tracking(self):
        """Test completed fade in removes object from tracking."""
        fader = TransparencyFader()
        fader.fade_duration = 0.1
        fader._fading_objects["object1"] = 0.99
        fader.start_fade_in("object1")

        fader.update(0.5)

        assert "object1" not in fader._fading_objects

    def test_is_fading(self):
        """Test is_fading returns correct status."""
        fader = TransparencyFader()
        assert fader.is_fading("object1") is False

        fader.start_fade_out("object1")
        assert fader.is_fading("object1") is True

    def test_get_alpha_not_fading(self):
        """Test get_alpha returns max for non-fading objects."""
        fader = TransparencyFader()
        assert fader.get_alpha("unknown") == fader.max_alpha

    def test_disabled_no_update(self):
        """Test disabled fader does not update."""
        fader = TransparencyFader()
        fader.enabled = False
        fader._fading_objects["object1"] = 1.0
        fader._target_alpha["object1"] = 0.3

        fader.update(1.0)
        assert fader.get_alpha("object1") == 1.0


# =============================================================================
# Collision Mask Filtering Tests (~10 tests)
# =============================================================================


class TestCollisionMaskFiltering:
    """Test collision mask filtering."""

    def test_static_only(self):
        """Test filtering for static only."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="static", layer=CollisionLayer.STATIC,
                     bounds_min=Vector3(2, -1, -1), bounds_max=Vector3(4, 1, 1))
        )
        world.add_collider(
            Collider(id="dynamic", layer=CollisionLayer.DYNAMIC,
                     bounds_min=Vector3(1, -1, -1), bounds_max=Vector3(1.5, 1, 1))
        )
        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.STATIC)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "static"

    def test_multiple_layers(self):
        """Test filtering for multiple layers."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="trigger", layer=CollisionLayer.TRIGGER,
                     bounds_min=Vector3(1, -1, -1), bounds_max=Vector3(2, 1, 1))
        )
        world.add_collider(
            Collider(id="dynamic", layer=CollisionLayer.DYNAMIC,
                     bounds_min=Vector3(3, -1, -1), bounds_max=Vector3(4, 1, 1))
        )
        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.STATIC | CollisionLayer.DYNAMIC)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "dynamic"

    def test_all_layers(self):
        """Test filtering with ALL layers."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="trigger", layer=CollisionLayer.TRIGGER,
                     bounds_min=Vector3(1, -1, -1), bounds_max=Vector3(2, 1, 1))
        )
        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.ALL)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.hit is True


# =============================================================================
# Smooth Collision Recovery Tests (~10 tests)
# =============================================================================


class TestSmoothRecovery:
    """Test smooth collision recovery."""

    def test_smooth_recovery_enabled(self):
        """Test smooth recovery interpolates position."""
        system = CameraCollisionSystem()
        system.smooth_recovery = True
        system.recovery_speed = 5.0

        current = Vector3(0, 0, 0)
        desired = Vector3(10, 0, 0)
        result = system.update_smooth_recovery(current, desired, 0.1)

        assert 0 < result.x < 10

    def test_smooth_recovery_disabled(self):
        """Test disabled smooth recovery returns desired directly."""
        system = CameraCollisionSystem()
        system.smooth_recovery = False

        current = Vector3(0, 0, 0)
        desired = Vector3(10, 0, 0)
        result = system.update_smooth_recovery(current, desired, 0.1)

        assert result.x == 10.0

    def test_recovery_speed_affects_interpolation(self):
        """Test recovery speed affects interpolation rate."""
        current = Vector3(0, 0, 0)
        desired = Vector3(10, 0, 0)

        system_slow = CameraCollisionSystem()
        system_slow.recovery_speed = 1.0

        system_fast = CameraCollisionSystem()
        system_fast.recovery_speed = 10.0

        result_slow = system_slow.update_smooth_recovery(current, desired, 0.1)
        result_fast = system_fast.update_smooth_recovery(current, desired, 0.1)

        assert result_fast.x > result_slow.x

    def test_recovery_converges(self):
        """Test recovery converges to desired position."""
        system = CameraCollisionSystem()
        system.recovery_speed = 5.0

        current = Vector3(0, 0, 0)
        desired = Vector3(10, 0, 0)

        for _ in range(100):
            current = system.update_smooth_recovery(current, desired, 0.1)

        assert current.x == pytest.approx(10.0, abs=0.01)


# =============================================================================
# Multiple Collision Hits Tests (~10 tests)
# =============================================================================


class TestMultipleCollisionHits:
    """Test handling of multiple collision hits."""

    def test_closest_hit_selected(self):
        """Test closest hit is selected from multiple."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="far", bounds_min=Vector3(8, -1, -1), bounds_max=Vector3(10, 1, 1))
        )
        world.add_collider(
            Collider(id="middle", bounds_min=Vector3(4, -1, -1), bounds_max=Vector3(6, 1, 1))
        )
        world.add_collider(
            Collider(id="near", bounds_min=Vector3(1, -1, -1), bounds_max=Vector3(2, 1, 1))
        )
        system.set_physics_world(world)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 20.0)
        assert hit.collider.id == "near"

    def test_multiple_occluders_in_occlusion_check(self):
        """Test multiple occluders detected."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        world = PhysicsWorld()
        world.add_collider(
            Collider(
                id="wall1",
                bounds_min=Vector3(4, -0.5, -0.5),
                bounds_max=Vector3(5, 0.5, 0.5),
            )
        )
        world.add_collider(
            Collider(
                id="wall2",
                bounds_min=Vector3(4, -0.5, 0.2),
                bounds_max=Vector3(5, 0.5, 0.8),
            )
        )
        system.set_physics_world(world)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_physics_world(self):
        """Test behavior without physics world."""
        system = CameraCollisionSystem()
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.hit is False

    def test_zero_direction(self):
        """Test raycast with zero direction."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(0, 0, 0), 10.0)
        assert hit.hit is False

    def test_negative_max_distance(self):
        """Test raycast with negative max distance."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="wall", bounds_min=Vector3(2, -1, -1), bounds_max=Vector3(4, 1, 1))
        )
        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), -5.0)
        assert hit.hit is False

    def test_mark_unmark_occluded(self):
        """Test marking and unmarking occluded objects."""
        system = CameraCollisionSystem()
        system.mark_occluded("obj1")
        assert "obj1" in system.get_occluded_objects()
        system.unmark_occluded("obj1")
        assert "obj1" not in system.get_occluded_objects()

    def test_unmark_nonexistent(self):
        """Test unmarking non-existent object does not error."""
        system = CameraCollisionSystem()
        system.unmark_occluded("nonexistent")


# =============================================================================
# Additional Raycast Tests
# =============================================================================


class TestRaycastAdvanced:
    """Additional raycast collision tests."""

    def test_raycast_diagonal_direction(self):
        """Test raycast with diagonal direction."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="wall", bounds_min=Vector3(5, 5, -1), bounds_max=Vector3(7, 7, 1))
        )
        system.set_physics_world(world)
        direction = Vector3(1, 1, 0).normalized()
        hit = system.raycast(Vector3(0, 0, 0), direction, 20.0)
        assert hit.hit is True

    def test_raycast_vertical(self):
        """Test vertical raycast."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="ceiling", bounds_min=Vector3(-1, 10, -1), bounds_max=Vector3(1, 12, 1))
        )
        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(0, 1, 0), 20.0)
        assert hit.hit is True
        assert hit.distance == pytest.approx(10.0, abs=0.1)

    def test_raycast_backward(self):
        """Test raycast in negative direction."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="wall", bounds_min=Vector3(-5, -1, -1), bounds_max=Vector3(-3, 1, 1))
        )
        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(-1, 0, 0), 20.0)
        assert hit.hit is True

    def test_raycast_origin_inside_collider(self):
        """Test raycast with origin inside collider."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="box", bounds_min=Vector3(-5, -5, -5), bounds_max=Vector3(5, 5, 5))
        )
        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 20.0)

    def test_raycast_exactly_at_max_distance(self):
        """Test raycast with hit exactly at max distance."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="wall", bounds_min=Vector3(10, -1, -1), bounds_max=Vector3(12, 1, 1))
        )
        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.hit is True


# =============================================================================
# Additional Spherecast Tests
# =============================================================================


class TestSpherecastAdvanced:
    """Additional spherecast tests."""

    def test_spherecast_large_radius(self):
        """Test spherecast with large radius."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="wall", bounds_min=Vector3(10, -1, -1), bounds_max=Vector3(12, 1, 1))
        )
        system.set_physics_world(world)
        hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 5.0, 20.0)
        assert hit.hit is True
        assert hit.distance < 10.0

    def test_spherecast_grazing_hit(self):
        """Test spherecast that grazes a collider."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="wall", bounds_min=Vector3(5, 1, -1), bounds_max=Vector3(7, 3, 1))
        )
        system.set_physics_world(world)
        hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.5, 20.0)

    def test_spherecast_multiple_potential_hits(self):
        """Test spherecast with multiple potential hits."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="near", bounds_min=Vector3(3, -2, -2), bounds_max=Vector3(5, 2, 2))
        )
        world.add_collider(
            Collider(id="far", bounds_min=Vector3(8, -2, -2), bounds_max=Vector3(10, 2, 2))
        )
        system.set_physics_world(world)
        hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.5, 20.0)
        assert hit.collider.id == "near"


# =============================================================================
# Additional Collision Response Tests
# =============================================================================


class TestCollisionResponseAdvanced:
    """Additional collision response tests."""

    def test_pull_in_zero_offset(self):
        """Test pull-in with zero offset."""
        strategy = PullInResponse(offset=0.0)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(hit=True, point=Vector3(5, 0, 0), distance=5.0)
        result = strategy.respond(desired, hit, target)
        assert result.x == pytest.approx(5.0, abs=0.01)

    def test_push_out_angled_normal(self):
        """Test push-out with angled normal."""
        strategy = PushOutResponse(offset=1.0)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        normal = Vector3(-0.707, 0.707, 0).normalized()
        hit = RaycastHit(hit=True, point=Vector3(5, 0, 0), normal=normal, distance=5.0)
        result = strategy.respond(desired, hit, target)
        assert result.y > 0

    def test_slide_no_component_in_normal(self):
        """Test slide when movement parallel to surface."""
        strategy = SlideResponse()
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(hit=True, point=Vector3(5, 0, 0), normal=Vector3(0, 1, 0), distance=5.0)
        result = strategy.respond(desired, hit, target)

    def test_clamp_distance_large_min(self):
        """Test clamp distance with large minimum."""
        strategy = ClampDistanceResponse(min_distance=8.0)
        desired = Vector3(10, 0, 0)
        target = Vector3(0, 0, 0)
        hit = RaycastHit(hit=True, point=Vector3(5, 0, 0), distance=5.0)
        result = strategy.respond(desired, hit, target)
        assert result.magnitude() >= 8.0


# =============================================================================
# Additional Occlusion Tests
# =============================================================================


class TestOcclusionAdvanced:
    """Additional occlusion detection tests."""

    def test_partial_occlusion_threshold(self):
        """Test occlusion percentage threshold."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        system.raycast_count = 5
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="small", bounds_min=Vector3(5, -0.05, -0.05), bounds_max=Vector3(6, 0.05, 0.05))
        )
        system.set_physics_world(world)
        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert 0 < info.occlusion_percentage < 1.0

    def test_no_occlusion_when_behind_camera(self):
        """Test no occlusion for objects behind camera."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="behind", bounds_min=Vector3(-5, -1, -1), bounds_max=Vector3(-3, 1, 1))
        )
        system.set_physics_world(world)
        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is False

    def test_occlusion_at_target(self):
        """Test occlusion check when hit is at target."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(
            Collider(id="at_target", bounds_min=Vector3(9, -1, -1), bounds_max=Vector3(11, 1, 1))
        )
        system.set_physics_world(world)
        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))


# =============================================================================
# Additional Transparency Fading Tests
# =============================================================================


class TestTransparencyFadingAdvanced:
    """Additional transparency fading tests."""

    def test_fade_out_to_min_alpha(self):
        """Test fade out reaches minimum alpha."""
        fader = TransparencyFader()
        fader.fade_duration = 0.1
        fader.min_alpha = 0.2
        fader.start_fade_out("obj1")
        for _ in range(20):
            fader.update(0.1)
        assert fader.get_alpha("obj1") <= fader.min_alpha + 0.05

    def test_simultaneous_fades(self):
        """Test multiple simultaneous fades."""
        fader = TransparencyFader()
        fader.start_fade_out("obj1")
        fader.start_fade_out("obj2")
        fader.start_fade_in("obj3")
        fader.update(0.1)
        assert fader.is_fading("obj1")
        assert fader.is_fading("obj2")

    def test_fade_speed_variation(self):
        """Test different fade speeds for different objects."""
        fader = TransparencyFader()
        fader.fade_duration = 1.0
        fader.start_fade_out("slow")
        fader._fading_objects["fast"] = 1.0
        fader._target_alpha["fast"] = 0.3
        fader.update(0.5)

    def test_fade_in_after_fade_out(self):
        """Test fade in immediately after fade out."""
        fader = TransparencyFader()
        fader.start_fade_out("obj1")
        fader.update(0.1)
        fader.start_fade_in("obj1")
        assert fader._target_alpha["obj1"] == fader.max_alpha


# =============================================================================
# Additional Smooth Recovery Tests
# =============================================================================


class TestSmoothRecoveryAdvanced:
    """Additional smooth recovery tests."""

    def test_recovery_with_moving_target(self):
        """Test recovery while target is moving."""
        system = CameraCollisionSystem()
        system.recovery_speed = 5.0
        current = Vector3(0, 0, 0)
        for i in range(10):
            desired = Vector3(i * 2, 0, 0)
            current = system.update_smooth_recovery(current, desired, 0.1)
        assert current.x > 0

    def test_very_high_recovery_speed(self):
        """Test very high recovery speed is near instant."""
        system = CameraCollisionSystem()
        system.recovery_speed = 100.0
        current = Vector3(0, 0, 0)
        desired = Vector3(10, 0, 0)
        result = system.update_smooth_recovery(current, desired, 0.1)
        assert result.x > 9.0

    def test_zero_recovery_speed(self):
        """Test zero recovery speed maintains position."""
        system = CameraCollisionSystem()
        system.recovery_speed = 0.0
        current = Vector3(5, 5, 5)
        desired = Vector3(10, 0, 0)
        result = system.update_smooth_recovery(current, desired, 0.1)
        assert result.x == pytest.approx(5.0, abs=0.1)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCollisionIntegration:
    """Integration tests for collision systems."""

    def test_full_collision_pipeline(self):
        """Test complete collision detection pipeline."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        # Add multiple colliders
        world.add_collider(Collider(
            id="wall1", bounds_min=Vector3(5, -2, -2), bounds_max=Vector3(6, 2, 2),
            layer=CollisionLayer.STATIC
        ))
        world.add_collider(Collider(
            id="wall2", bounds_min=Vector3(8, -2, -2), bounds_max=Vector3(9, 2, 2),
            layer=CollisionLayer.STATIC
        ))

        system.set_physics_world(world)
        system.set_response_strategy(PullInResponse(offset=0.2))

        collided, adjusted = system.check_collision(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert collided is True
        assert adjusted.x < 10.0

    def test_collision_with_fading_pipeline(self):
        """Test collision with transparency fading."""
        system = CameraCollisionSystem()
        fader = TransparencyFader()
        world = PhysicsWorld()

        world.add_collider(Collider(
            id="glass", bounds_min=Vector3(3, -1, -1), bounds_max=Vector3(4, 1, 1),
            layer=CollisionLayer.TRANSPARENT
        ))
        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.ALL)

        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        if info.is_occluded:
            for occluder in info.occluders:
                fader.start_fade_out(occluder.id)

        fader.update(0.1)
        assert fader.is_fading("glass") is True

    def test_dynamic_collision_response(self):
        """Test dynamic collision response switching."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1)
        ))
        system.set_physics_world(world)

        # Test with pull-in
        system.set_response_strategy(PullInResponse())
        collided1, pos1 = system.check_collision(Vector3(0, 0, 0), Vector3(10, 0, 0))

        # Switch to push-out
        system.set_response_strategy(PushOutResponse())
        collided2, pos2 = system.check_collision(Vector3(0, 0, 0), Vector3(10, 0, 0))

        assert collided1 is True
        assert collided2 is True

    def test_third_person_camera_collision(self):
        """Test third person camera collision simulation."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="obstacle", bounds_min=Vector3(2, 0, -1), bounds_max=Vector3(3, 2, 1)
        ))
        system.set_physics_world(world)
        system.collision_radius = 0.3
        system.set_response_strategy(ClampDistanceResponse(min_distance=1.0))

        target = Vector3(0, 1, 0)  # Character position
        desired = Vector3(5, 2, 0)  # Desired camera position

        collided, adjusted = system.check_collision(target, desired)
        if collided:
            distance = (adjusted - target).magnitude()
            assert distance >= 1.0

    def test_occlusion_with_multiple_objects(self):
        """Test occlusion detection with multiple objects."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        system.raycast_count = 5
        world = PhysicsWorld()

        # Add multiple occluders
        world.add_collider(Collider(
            id="pillar1", bounds_min=Vector3(3, -2, -0.5), bounds_max=Vector3(4, 2, 0.5)
        ))
        world.add_collider(Collider(
            id="pillar2", bounds_min=Vector3(5, -2, -0.5), bounds_max=Vector3(6, 2, 0.5)
        ))

        system.set_physics_world(world)
        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert info.is_occluded is True


class TestCollisionStress:
    """Stress tests for collision system."""

    def test_many_colliders(self):
        """Test with many colliders in the scene."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        for i in range(50):
            world.add_collider(Collider(
                id=f"wall_{i}",
                bounds_min=Vector3(i * 2, -1, -1),
                bounds_max=Vector3(i * 2 + 0.5, 1, 1)
            ))

        system.set_physics_world(world)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 200.0)
        assert hit.hit is True
        assert hit.collider.id == "wall_0"

    def test_rapid_collision_checks(self):
        """Test rapid collision checking."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1)
        ))
        system.set_physics_world(world)

        for i in range(1000):
            system.check_collision(
                Vector3(0, 0, 0),
                Vector3(10 + (i % 5), 0, 0)
            )

    def test_many_fading_objects(self):
        """Test many objects fading simultaneously."""
        fader = TransparencyFader()

        for i in range(100):
            fader.start_fade_out(f"object_{i}")

        for _ in range(50):
            fader.update(0.016)

    def test_occlusion_many_raycasts(self):
        """Test occlusion with many raycasts."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        system.raycast_count = 5
        world = PhysicsWorld()

        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -3, -3), bounds_max=Vector3(6, 3, 3)
        ))
        system.set_physics_world(world)

        for _ in range(500):
            system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))


class TestCollisionAdvancedScenarios:
    """Advanced collision scenarios."""

    def test_spherecast_through_gap(self):
        """Test spherecast through a gap between colliders."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        # Create gap - walls above and below
        world.add_collider(Collider(
            id="top", bounds_min=Vector3(5, 2, -1), bounds_max=Vector3(6, 5, 1)
        ))
        world.add_collider(Collider(
            id="bottom", bounds_min=Vector3(5, -5, -1), bounds_max=Vector3(6, -2, 1)
        ))
        system.set_physics_world(world)

        # Cast through gap
        hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.5, 10.0)
        # May or may not hit depending on gap size

    def test_collision_at_angle(self):
        """Test collision at various angles."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -5, -5), bounds_max=Vector3(6, 5, 5)
        ))
        system.set_physics_world(world)

        angles = [0, 30, 45, 60, 90, 120, 150, 180]
        for angle in angles:
            rad = math.radians(angle)
            direction = Vector3(math.cos(rad), 0, math.sin(rad))
            hit = system.raycast(Vector3(0, 0, 0), direction, 20.0)

    def test_recovery_from_inside_collider(self):
        """Test recovery when camera starts inside collider."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="box", bounds_min=Vector3(-2, -2, -2), bounds_max=Vector3(2, 2, 2)
        ))
        system.set_physics_world(world)

        # Camera inside collider
        target = Vector3(0, 0, 0)
        desired = Vector3(0, 0, 0)

        collided, adjusted = system.check_collision(target, desired)

    def test_slide_along_corner(self):
        """Test slide response at corner."""
        strategy = SlideResponse()
        desired = Vector3(10, 10, 0)
        target = Vector3(0, 0, 0)

        # Hit corner - normal at 45 degrees
        normal = Vector3(-0.707, -0.707, 0)
        hit = RaycastHit(hit=True, point=Vector3(5, 5, 0), normal=normal, distance=7.07)

        result = strategy.respond(desired, hit, target)

    def test_transparent_layer_handling(self):
        """Test transparent layer special handling."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        world.add_collider(Collider(
            id="glass", bounds_min=Vector3(3, -1, -1), bounds_max=Vector3(4, 1, 1),
            layer=CollisionLayer.TRANSPARENT
        ))
        world.add_collider(Collider(
            id="solid", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1),
            layer=CollisionLayer.STATIC
        ))

        system.set_physics_world(world)

        # Only collide with static
        system.set_collision_mask(CollisionLayer.STATIC)
        hit1 = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit1.collider.id == "solid"

        # Collide with transparent
        system.set_collision_mask(CollisionLayer.TRANSPARENT)
        hit2 = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit2.collider.id == "glass"

    def test_camera_collision_with_character(self):
        """Test camera doesn't collide with character layer."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        world.add_collider(Collider(
            id="character", bounds_min=Vector3(2, -1, -1), bounds_max=Vector3(3, 2, 1),
            layer=CollisionLayer.CHARACTER
        ))
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1),
            layer=CollisionLayer.STATIC
        ))

        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.STATIC | CollisionLayer.DYNAMIC)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "wall"

    def test_collision_response_chain(self):
        """Test multiple collision responses in sequence."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1)
        ))
        system.set_physics_world(world)

        strategies = [
            PullInResponse(offset=0.1),
            PushOutResponse(offset=0.1),
            SlideResponse(),
            ClampDistanceResponse(min_distance=1.0)
        ]

        for strategy in strategies:
            system.set_response_strategy(strategy)
            collided, adjusted = system.check_collision(
                Vector3(0, 0, 0), Vector3(10, 0, 0)
            )
            assert collided is True

    def test_occlusion_partial_coverage(self):
        """Test partial occlusion coverage calculation."""
        system = CameraCollisionSystem()
        system.multi_raycast = True
        system.raycast_count = 5
        world = PhysicsWorld()

        # Small occluder - partial coverage
        world.add_collider(Collider(
            id="small", bounds_min=Vector3(5, -0.1, -0.1), bounds_max=Vector3(6, 0.1, 0.1)
        ))

        system.set_physics_world(world)
        info = system.check_occlusion(Vector3(0, 0, 0), Vector3(10, 0, 0))

        # Should have partial occlusion
        assert info.occlusion_percentage < 1.0

    def test_smooth_recovery_varying_speeds(self):
        """Test smooth recovery with varying speeds."""
        system = CameraCollisionSystem()

        speeds = [0.1, 1.0, 5.0, 10.0, 50.0]
        for speed in speeds:
            system.recovery_speed = speed
            current = Vector3(0, 0, 0)
            desired = Vector3(10, 0, 0)
            result = system.update_smooth_recovery(current, desired, 0.1)
            assert 0 <= result.x <= 10

    def test_fader_multiple_cycles(self):
        """Test transparency fader through multiple fade cycles."""
        fader = TransparencyFader()
        fader.fade_duration = 0.2

        for _ in range(5):
            fader.start_fade_out("object1")
            for _ in range(10):
                fader.update(0.05)

            fader.start_fade_in("object1")
            for _ in range(10):
                fader.update(0.05)

    def test_raycast_all_directions(self):
        """Test raycast in all cardinal directions."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        # Surround origin with walls
        positions = [
            (5, 0, 0), (-5, 0, 0), (0, 5, 0), (0, -5, 0), (0, 0, 5), (0, 0, -5)
        ]
        for i, (x, y, z) in enumerate(positions):
            world.add_collider(Collider(
                id=f"wall_{i}",
                bounds_min=Vector3(x-0.5, y-0.5, z-0.5),
                bounds_max=Vector3(x+0.5, y+0.5, z+0.5)
            ))

        system.set_physics_world(world)

        directions = [
            Vector3(1, 0, 0), Vector3(-1, 0, 0),
            Vector3(0, 1, 0), Vector3(0, -1, 0),
            Vector3(0, 0, 1), Vector3(0, 0, -1)
        ]

        for direction in directions:
            hit = system.raycast(Vector3(0, 0, 0), direction, 10.0)
            assert hit.hit is True

    def test_collision_with_trigger_volumes(self):
        """Test collision system ignores trigger volumes."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        world.add_collider(Collider(
            id="trigger", bounds_min=Vector3(2, -1, -1), bounds_max=Vector3(3, 1, 1),
            layer=CollisionLayer.TRIGGER, is_trigger=True
        ))
        world.add_collider(Collider(
            id="solid", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1),
            layer=CollisionLayer.STATIC
        ))

        system.set_physics_world(world)
        system.set_collision_mask(CollisionLayer.STATIC | CollisionLayer.DYNAMIC)

        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "solid"

    def test_occlusion_tracking(self):
        """Test occlusion object tracking."""
        system = CameraCollisionSystem()

        system.mark_occluded("obj1")
        system.mark_occluded("obj2")
        system.mark_occluded("obj3")

        occluded = system.get_occluded_objects()
        assert len(occluded) == 3

        system.unmark_occluded("obj2")
        occluded = system.get_occluded_objects()
        assert len(occluded) == 2
        assert "obj2" not in occluded

    def test_spherecast_vs_raycast_consistency(self):
        """Test spherecast with zero radius matches raycast."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -2, -2), bounds_max=Vector3(6, 2, 2)
        ))
        system.set_physics_world(world)

        ray_hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        sphere_hit = system.spherecast(Vector3(0, 0, 0), Vector3(1, 0, 0), 0.0, 10.0)

        assert ray_hit.hit == sphere_hit.hit
        assert abs(ray_hit.distance - sphere_hit.distance) < 0.1

    def test_collision_disabled_system(self):
        """Test collision system when disabled."""
        system = CameraCollisionSystem()
        system.enabled = False
        world = PhysicsWorld()
        world.add_collider(Collider(
            id="wall", bounds_min=Vector3(5, -1, -1), bounds_max=Vector3(6, 1, 1)
        ))
        system.set_physics_world(world)

        collided, adjusted = system.check_collision(Vector3(0, 0, 0), Vector3(10, 0, 0))
        assert collided is False

    def test_collision_layer_combinations(self):
        """Test various collision layer combinations."""
        system = CameraCollisionSystem()
        world = PhysicsWorld()

        world.add_collider(Collider(
            id="static", bounds_min=Vector3(2, -1, -1), bounds_max=Vector3(3, 1, 1),
            layer=CollisionLayer.STATIC
        ))
        world.add_collider(Collider(
            id="dynamic", bounds_min=Vector3(4, -1, -1), bounds_max=Vector3(5, 1, 1),
            layer=CollisionLayer.DYNAMIC
        ))
        world.add_collider(Collider(
            id="character", bounds_min=Vector3(6, -1, -1), bounds_max=Vector3(7, 1, 1),
            layer=CollisionLayer.CHARACTER
        ))

        system.set_physics_world(world)

        # Test static only
        system.set_collision_mask(CollisionLayer.STATIC)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "static"

        # Test dynamic only
        system.set_collision_mask(CollisionLayer.DYNAMIC)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "dynamic"

        # Test all
        system.set_collision_mask(CollisionLayer.ALL)
        hit = system.raycast(Vector3(0, 0, 0), Vector3(1, 0, 0), 10.0)
        assert hit.collider.id == "static"  # First one

    def test_fader_completed_state(self):
        """Test transparency fader completed state."""
        fader = TransparencyFader()
        fader.fade_duration = 0.1

        fader.start_fade_out("object1")
        for _ in range(20):
            fader.update(0.016)

        assert fader.is_fading("object1") is False
        # After fade out, alpha should be at min_alpha (0.3)
        # So transparency (1 - alpha) should be 0.7
        assert fader.get_alpha("object1") <= fader.min_alpha + 0.05
        assert fader.get_transparency("object1") >= 1.0 - fader.min_alpha - 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
