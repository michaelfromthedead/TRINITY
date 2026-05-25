"""Camera Collision System - Collision detection and response for cameras.

This module provides collision detection between the camera and world geometry,
with various response modes including pull-in, push-out, and transparency fading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import math

from engine.core.math.vec import Vec3
from engine.core.math.mat import Mat4

from engine.gameplay.camera.constants import (
    COLLISION_PROBE_RADIUS,
    MIN_COLLISION_DISTANCE,
    COLLISION_PULL_IN_SPEED,
    COLLISION_PUSH_OUT_SPEED,
    COLLISION_INTERP_SPEED,
    DEFAULT_COLLISION_MASK,
    MAX_COLLISION_PROBES,
    CAMERA_EPSILON,
    MIN_DELTA_TIME,
    MAX_DELTA_TIME,
    DEFAULT_FADE_DISTANCE,
    DEFAULT_OCCLUSION_FADE_IN_TIME,
    DEFAULT_OCCLUSION_FADE_OUT_TIME,
    MIN_OCCLUSION_ALPHA,
    BLEND_RESPONSE_MIN_ALPHA,
    BLEND_RESPONSE_ALPHA_RANGE,
    PULL_IN_WEIGHT_THRESHOLD,
)

if TYPE_CHECKING:
    from engine.simulation.physics import PhysicsWorld


class CollisionResponse(Enum):
    """How the camera should respond to collision."""
    PULL_IN = auto()      # Pull camera toward target (most common)
    PUSH_OUT = auto()     # Push camera away from obstacle
    FADE = auto()         # Fade occluding objects instead of moving camera
    CLIP = auto()         # Allow camera to clip through (no response)
    BLEND = auto()        # Blend between pull-in and fade based on distance


@dataclass(slots=True)
class CollisionHit:
    """Result of a collision check."""
    hit: bool = False
    hit_point: Vec3 = field(default_factory=Vec3.zero)
    hit_normal: Vec3 = field(default_factory=Vec3.up)
    hit_distance: float = 0.0
    hit_entity: Optional[Any] = None
    safe_position: Vec3 = field(default_factory=Vec3.zero)


@dataclass(slots=True)
class CollisionSettings:
    """Configuration for camera collision behavior."""
    enabled: bool = True
    response_mode: CollisionResponse = CollisionResponse.PULL_IN
    probe_radius: float = COLLISION_PROBE_RADIUS
    collision_mask: int = DEFAULT_COLLISION_MASK
    min_distance: float = MIN_COLLISION_DISTANCE
    pull_in_speed: float = COLLISION_PULL_IN_SPEED
    push_out_speed: float = COLLISION_PUSH_OUT_SPEED
    interpolation_speed: float = COLLISION_INTERP_SPEED
    fade_distance: float = DEFAULT_FADE_DISTANCE  # Distance at which fading starts
    soft_collision: bool = True  # Smooth interpolation vs instant snap


class CameraCollision:
    """
    Handles camera collision detection and response.

    Features:
    - Raycast and spherecast collision detection
    - Multiple response modes (pull-in, push-out, fade)
    - Collision layer masking
    - Smooth collision interpolation
    - Integration with physics system
    """

    __slots__ = (
        "_settings",
        "_current_safe_distance",
        "_target_safe_distance",
        "_last_collision_point",
        "_is_colliding",
        "_collision_alpha",  # For fade mode
        "_occluding_entities",
        "_physics_world",
        "_custom_raycast",
        "_last_safe_pos",  # For smooth collision interpolation
    )

    def __init__(
        self,
        settings: Optional[CollisionSettings] = None,
        physics_world: Optional[PhysicsWorld] = None,
    ) -> None:
        """
        Initialize camera collision system.

        Args:
            settings: Collision configuration
            physics_world: Optional physics world for raycasting
        """
        self._settings = settings if settings is not None else CollisionSettings()
        self._current_safe_distance = float("inf")
        self._target_safe_distance = float("inf")
        self._last_collision_point = Vec3.zero()
        self._is_colliding = False
        self._collision_alpha = 1.0
        self._occluding_entities: Set[Any] = set()
        self._physics_world = physics_world
        self._custom_raycast: Optional[Callable[[Vec3, Vec3, int], CollisionHit]] = None
        self._last_safe_pos: Optional[Vec3] = None  # Properly initialized for smooth interpolation

    @property
    def settings(self) -> CollisionSettings:
        """Get collision settings."""
        return self._settings

    @property
    def is_colliding(self) -> bool:
        """Check if camera is currently in collision state."""
        return self._is_colliding

    @property
    def collision_alpha(self) -> float:
        """Get fade alpha (1.0 = fully visible, 0.0 = fully faded)."""
        return self._collision_alpha

    @property
    def occluding_entities(self) -> Set[Any]:
        """Get set of currently occluding entities."""
        return self._occluding_entities

    def set_physics_world(self, physics_world: PhysicsWorld) -> None:
        """Set the physics world for collision queries."""
        self._physics_world = physics_world

    def set_custom_raycast(
        self,
        raycast_func: Callable[[Vec3, Vec3, int], CollisionHit]
    ) -> None:
        """
        Set a custom raycast function.

        Args:
            raycast_func: Function taking (start, end, mask) returning CollisionHit
        """
        self._custom_raycast = raycast_func

    def raycast_check(
        self,
        start: Vec3,
        end: Vec3,
        mask: Optional[int] = None,
    ) -> CollisionHit:
        """
        Perform a raycast collision check.

        Args:
            start: Ray start position
            end: Ray end position
            mask: Optional collision layer mask

        Returns:
            CollisionHit with results
        """
        if mask is None:
            mask = self._settings.collision_mask

        # Use custom raycast if provided
        if self._custom_raycast is not None:
            return self._custom_raycast(start, end, mask)

        # Use physics world if available
        if self._physics_world is not None:
            return self._physics_raycast(start, end, mask)

        # No collision system available
        return CollisionHit(hit=False, safe_position=end)

    def _physics_raycast(
        self,
        start: Vec3,
        end: Vec3,
        mask: int,
    ) -> CollisionHit:
        """Perform raycast using physics world."""
        # This would integrate with the actual physics system
        # For now, return no hit
        direction = end - start
        distance = direction.length()

        if distance < CAMERA_EPSILON:
            return CollisionHit(hit=False, safe_position=end)

        # Placeholder - actual implementation would use physics world
        return CollisionHit(hit=False, safe_position=end)

    def sphere_cast_check(
        self,
        start: Vec3,
        end: Vec3,
        radius: Optional[float] = None,
        mask: Optional[int] = None,
    ) -> CollisionHit:
        """
        Perform a sphere cast collision check.

        Args:
            start: Sphere start position
            end: Sphere end position
            radius: Optional sphere radius (uses probe_radius if not specified)
            mask: Optional collision layer mask

        Returns:
            CollisionHit with results
        """
        if radius is None:
            radius = self._settings.probe_radius
        if mask is None:
            mask = self._settings.collision_mask

        # Calculate direction and distance
        direction = end - start
        distance = direction.length()

        if distance < CAMERA_EPSILON:
            return CollisionHit(hit=False, safe_position=end)

        direction_norm = direction / distance

        # Perform multiple ray checks to approximate sphere cast
        hit_result = CollisionHit(hit=False, safe_position=end)
        min_hit_distance = float("inf")

        # Center ray
        center_hit = self.raycast_check(start, end, mask)
        if center_hit.hit and center_hit.hit_distance < min_hit_distance:
            min_hit_distance = center_hit.hit_distance
            hit_result = center_hit

        # Calculate perpendicular axes for offset rays
        world_up = Vec3.up()
        if abs(direction_norm.dot(world_up)) > 0.99:
            world_up = Vec3.forward()

        right = direction_norm.cross(world_up).normalized()
        up = right.cross(direction_norm).normalized()

        # Sample offset rays around the sphere
        offsets = [
            right * radius,
            right * -radius,
            up * radius,
            up * -radius,
            (right + up).normalized() * radius,
            (right - up).normalized() * radius,
            (-right + up).normalized() * radius,
            (-right - up).normalized() * radius,
        ]

        num_probes = min(len(offsets), MAX_COLLISION_PROBES - 1)
        for i in range(num_probes):
            offset = offsets[i]
            offset_start = start + offset
            offset_end = end + offset

            offset_hit = self.raycast_check(offset_start, offset_end, mask)
            if offset_hit.hit and offset_hit.hit_distance < min_hit_distance:
                min_hit_distance = offset_hit.hit_distance
                hit_result = offset_hit

        # Adjust safe position to account for radius
        if hit_result.hit:
            safe_distance = max(0.0, min_hit_distance - radius - self._settings.min_distance)
            hit_result.safe_position = start + direction_norm * safe_distance

        return hit_result

    def resolve_collision(
        self,
        desired_pos: Vec3,
        target_pos: Vec3,
        delta_time: float = 0.016,
    ) -> Vec3:
        """
        Resolve collision and return safe camera position.

        Args:
            desired_pos: Where the camera wants to be
            target_pos: What the camera is looking at (character position)
            delta_time: Time since last update

        Returns:
            Safe camera position after collision resolution
        """
        if not self._settings.enabled:
            return desired_pos

        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Perform collision check
        hit = self.sphere_cast_check(target_pos, desired_pos)

        # Track collision state
        was_colliding = self._is_colliding
        self._is_colliding = hit.hit

        if hit.hit:
            self._last_collision_point = hit.hit_point
            self._target_safe_distance = hit.hit_distance

            # Track occluding entity
            if hit.hit_entity is not None:
                self._occluding_entities.add(hit.hit_entity)
        else:
            self._target_safe_distance = float("inf")
            self._occluding_entities.clear()

        # Handle response based on mode
        response_mode = self._settings.response_mode

        if response_mode == CollisionResponse.CLIP:
            return desired_pos

        elif response_mode == CollisionResponse.FADE:
            return self._handle_fade_response(desired_pos, target_pos, hit, delta_time)

        elif response_mode == CollisionResponse.PULL_IN:
            return self._handle_pull_in_response(desired_pos, target_pos, hit, delta_time)

        elif response_mode == CollisionResponse.PUSH_OUT:
            return self._handle_push_out_response(desired_pos, target_pos, hit, delta_time)

        elif response_mode == CollisionResponse.BLEND:
            return self._handle_blend_response(desired_pos, target_pos, hit, delta_time)

        return desired_pos

    def _handle_pull_in_response(
        self,
        desired_pos: Vec3,
        target_pos: Vec3,
        hit: CollisionHit,
        delta_time: float,
    ) -> Vec3:
        """Handle pull-in collision response."""
        from engine.gameplay.camera.constants import COLLISION_INTERP_FACTOR

        if not hit.hit:
            # No collision - smoothly return to desired position
            if self._settings.soft_collision and self._last_safe_pos is not None:
                interp_speed = self._settings.interpolation_speed
                factor = 1.0 - math.exp(-interp_speed * delta_time)
                result = self._last_safe_pos.lerp(desired_pos, factor)
                self._last_safe_pos = result
                return result
            self._last_safe_pos = desired_pos
            return desired_pos

        # Collision detected - pull toward target
        safe_pos = hit.safe_position

        if self._settings.soft_collision:
            # Smoothly interpolate to safe position
            interp_speed = self._settings.pull_in_speed
            factor = 1.0 - math.exp(-interp_speed * delta_time * COLLISION_INTERP_FACTOR)

            current_pos = self._last_safe_pos if self._last_safe_pos is not None else desired_pos
            result = current_pos.lerp(safe_pos, factor)
            self._last_safe_pos = result
            return result
        else:
            self._last_safe_pos = safe_pos
            return safe_pos

    def _handle_push_out_response(
        self,
        desired_pos: Vec3,
        target_pos: Vec3,
        hit: CollisionHit,
        delta_time: float,
    ) -> Vec3:
        """Handle push-out collision response."""
        from engine.gameplay.camera.constants import COLLISION_INTERP_FACTOR

        if not hit.hit:
            return desired_pos

        # Push camera along collision normal
        push_distance = self._settings.probe_radius + self._settings.min_distance
        pushed_pos = hit.hit_point + hit.hit_normal * push_distance

        if self._settings.soft_collision:
            interp_speed = self._settings.push_out_speed
            factor = 1.0 - math.exp(-interp_speed * delta_time * COLLISION_INTERP_FACTOR)
            current_pos = self._last_safe_pos if self._last_safe_pos is not None else desired_pos
            result = current_pos.lerp(pushed_pos, factor)
            self._last_safe_pos = result
            return result

        self._last_safe_pos = pushed_pos
        return pushed_pos

    def _handle_fade_response(
        self,
        desired_pos: Vec3,
        target_pos: Vec3,
        hit: CollisionHit,
        delta_time: float,
    ) -> Vec3:
        """Handle fade collision response (don't move camera, fade occluders)."""
        if hit.hit:
            # Calculate fade based on distance
            total_distance = (desired_pos - target_pos).length()
            hit_ratio = hit.hit_distance / total_distance if total_distance > 0 else 1.0

            # Fade starts at fade_distance ratio
            fade_threshold = self._settings.fade_distance / total_distance if total_distance > 0 else 0.5
            if hit_ratio < fade_threshold:
                self._collision_alpha = hit_ratio / fade_threshold
            else:
                self._collision_alpha = 1.0
        else:
            self._collision_alpha = 1.0

        return desired_pos  # Don't move camera in fade mode

    def _handle_blend_response(
        self,
        desired_pos: Vec3,
        target_pos: Vec3,
        hit: CollisionHit,
        delta_time: float,
    ) -> Vec3:
        """Handle blended response (mix of pull-in and fade)."""
        if not hit.hit:
            self._collision_alpha = 1.0
            return desired_pos

        # Calculate blend factor based on collision severity
        total_distance = (desired_pos - target_pos).length()
        severity = 1.0 - (hit.hit_distance / total_distance) if total_distance > 0 else 1.0

        # For minor collisions, use fade
        # For severe collisions, use pull-in
        fade_weight = max(0.0, 1.0 - severity * 2.0)
        pull_in_weight = 1.0 - fade_weight

        # Calculate fade alpha
        self._collision_alpha = BLEND_RESPONSE_MIN_ALPHA + BLEND_RESPONSE_ALPHA_RANGE * (1.0 - severity)

        # Calculate position blend
        if pull_in_weight > PULL_IN_WEIGHT_THRESHOLD:
            pull_in_pos = self._handle_pull_in_response(desired_pos, target_pos, hit, delta_time)
            return desired_pos.lerp(pull_in_pos, pull_in_weight)

        return desired_pos


class OcclusionDetector:
    """
    Detects objects between camera and target for transparency handling.

    Features:
    - Detect occluding objects
    - Track occlusion state over time
    - Configurable detection parameters
    - Entity filtering
    """

    __slots__ = (
        "_occluding_objects",
        "_detection_radius",
        "_detection_mask",
        "_ignore_entities",
        "_fade_in_time",
        "_fade_out_time",
        "_fade_states",
        "_camera_collision",
    )

    def __init__(
        self,
        detection_radius: float = COLLISION_PROBE_RADIUS * 2.0,
        collision_mask: int = DEFAULT_COLLISION_MASK,
        camera_collision: Optional[CameraCollision] = None,
    ) -> None:
        """
        Initialize occlusion detector.

        Args:
            detection_radius: Radius of detection cone
            collision_mask: Collision layers to check
            camera_collision: Optional camera collision for raycasting
        """
        self._occluding_objects: Set[Any] = set()
        self._detection_radius = detection_radius
        self._detection_mask = collision_mask
        self._ignore_entities: Set[Any] = set()
        self._fade_in_time = DEFAULT_OCCLUSION_FADE_IN_TIME  # Time to fade in when no longer occluding
        self._fade_out_time = DEFAULT_OCCLUSION_FADE_OUT_TIME  # Time to fade out when occluding
        self._fade_states: Dict[Any, float] = {}  # entity -> alpha
        self._camera_collision = camera_collision

    @property
    def occluding_objects(self) -> Set[Any]:
        """Get currently occluding objects."""
        return self._occluding_objects

    def add_ignore_entity(self, entity: Any) -> None:
        """Add entity to ignore list (e.g., the player character)."""
        self._ignore_entities.add(entity)

    def remove_ignore_entity(self, entity: Any) -> None:
        """Remove entity from ignore list."""
        self._ignore_entities.discard(entity)

    def get_fade_alpha(self, entity: Any) -> float:
        """Get current fade alpha for an entity (1.0 = visible, 0.0 = faded)."""
        return self._fade_states.get(entity, 1.0)

    def detect_occluders(
        self,
        camera_pos: Vec3,
        target_pos: Vec3,
    ) -> Set[Any]:
        """
        Detect objects occluding the view.

        Args:
            camera_pos: Camera position
            target_pos: Target position

        Returns:
            Set of occluding entities
        """
        if self._camera_collision is None:
            return set()

        new_occluders: Set[Any] = set()

        # Cast from target to camera
        hit = self._camera_collision.sphere_cast_check(
            target_pos,
            camera_pos,
            self._detection_radius,
            self._detection_mask,
        )

        if hit.hit and hit.hit_entity is not None:
            if hit.hit_entity not in self._ignore_entities:
                new_occluders.add(hit.hit_entity)

        self._occluding_objects = new_occluders
        return new_occluders

    def update_fade_states(self, delta_time: float) -> Dict[Any, float]:
        """
        Update fade states for all tracked entities.

        Args:
            delta_time: Time since last update

        Returns:
            Dictionary of entity -> alpha values
        """
        delta_time = max(MIN_DELTA_TIME, min(MAX_DELTA_TIME, delta_time))

        # Update fade for occluding objects
        for entity in self._occluding_objects:
            if entity in self._fade_states:
                # Fade out
                current = self._fade_states[entity]
                target = MIN_OCCLUSION_ALPHA  # Don't go fully invisible
                rate = 1.0 / self._fade_out_time if self._fade_out_time > 0 else 10.0
                self._fade_states[entity] = max(target, current - rate * delta_time)
            else:
                self._fade_states[entity] = 1.0

        # Update fade for non-occluding objects
        to_remove = []
        for entity, alpha in self._fade_states.items():
            if entity not in self._occluding_objects:
                # Fade in
                rate = 1.0 / self._fade_in_time if self._fade_in_time > 0 else 10.0
                new_alpha = min(1.0, alpha + rate * delta_time)
                if new_alpha >= 1.0:
                    to_remove.append(entity)
                else:
                    self._fade_states[entity] = new_alpha

        for entity in to_remove:
            del self._fade_states[entity]

        return self._fade_states.copy()


class TransparencyManager:
    """
    Manages transparency of occluding objects.

    Features:
    - Apply transparency to materials
    - Restore original materials
    - Smooth fade transitions
    - Material caching
    """

    __slots__ = (
        "_occlusion_detector",
        "_original_materials",
        "_fade_material",
        "_apply_transparency_callback",
        "_restore_material_callback",
    )

    def __init__(
        self,
        occlusion_detector: OcclusionDetector,
        fade_material: Optional[Any] = None,
    ) -> None:
        """
        Initialize transparency manager.

        Args:
            occlusion_detector: Occlusion detector for tracking occluders
            fade_material: Optional material to use for faded objects
        """
        self._occlusion_detector = occlusion_detector
        self._original_materials: Dict[Any, Any] = {}  # entity -> original material
        self._fade_material = fade_material
        self._apply_transparency_callback: Optional[Callable[[Any, float], None]] = None
        self._restore_material_callback: Optional[Callable[[Any], None]] = None

    def set_transparency_callbacks(
        self,
        apply_callback: Callable[[Any, float], None],
        restore_callback: Callable[[Any], None],
    ) -> None:
        """
        Set callbacks for applying and restoring transparency.

        Args:
            apply_callback: Function(entity, alpha) to apply transparency
            restore_callback: Function(entity) to restore original material
        """
        self._apply_transparency_callback = apply_callback
        self._restore_material_callback = restore_callback

    def update(self, camera_pos: Vec3, target_pos: Vec3, delta_time: float) -> None:
        """
        Update transparency for all tracked objects.

        Args:
            camera_pos: Current camera position
            target_pos: Current target position
            delta_time: Time since last update
        """
        # Detect occluders
        self._occlusion_detector.detect_occluders(camera_pos, target_pos)

        # Update fade states
        fade_states = self._occlusion_detector.update_fade_states(delta_time)

        # Apply transparency
        for entity, alpha in fade_states.items():
            if self._apply_transparency_callback is not None:
                self._apply_transparency_callback(entity, alpha)

        # Restore fully visible objects
        current_occluders = self._occlusion_detector.occluding_objects
        restored = []
        for entity in self._original_materials:
            if entity not in current_occluders and entity not in fade_states:
                if self._restore_material_callback is not None:
                    self._restore_material_callback(entity)
                restored.append(entity)

        for entity in restored:
            del self._original_materials[entity]

    def cleanup(self) -> None:
        """Restore all materials to original state."""
        if self._restore_material_callback is not None:
            for entity in list(self._original_materials.keys()):
                self._restore_material_callback(entity)
        self._original_materials.clear()


__all__ = [
    "CollisionResponse",
    "CollisionHit",
    "CollisionSettings",
    "CameraCollision",
    "OcclusionDetector",
    "TransparencyManager",
]
