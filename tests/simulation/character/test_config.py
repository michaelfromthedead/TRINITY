"""
Whitebox tests for engine/simulation/character/config.py

Tests configuration constants, enums, and collision masks.
"""

import pytest
from engine.simulation.character.config import (
    # Capsule
    DEFAULT_CAPSULE_HEIGHT,
    DEFAULT_CAPSULE_RADIUS,
    DEFAULT_CROUCHED_HEIGHT,
    DEFAULT_PRONE_HEIGHT,
    # Step/Slope
    DEFAULT_STEP_HEIGHT,
    MAX_SLOPE_ANGLE,
    MIN_SLOPE_ANGLE,
    STEEP_SLOPE_ANGLE,
    # Collision
    GROUND_PROBE_DISTANCE,
    GROUND_SPHERE_PROBE_RADIUS,
    MAX_COLLISION_ITERATIONS,
    MAX_DEPENETRATION_VELOCITY,
    MIN_MOVE_DISTANCE,
    SKIN_WIDTH,
    # Movement
    AIR_CONTROL,
    DEFAULT_GRAVITY,
    DEFAULT_JUMP_VELOCITY,
    MAX_FALL_VELOCITY,
    # Ground Detection
    COYOTE_TIME_MS,
    JUMP_BUFFER_TIME_MS,
    LEDGE_DETECTION_HEIGHT,
    LEDGE_GRAB_DISTANCE,
    # Movement Speeds
    CLIMBING_SPEED,
    CROUCHING_SPEED,
    FLYING_SPEED,
    PRONE_SPEED,
    RUNNING_SPEED,
    SPRINTING_SPEED,
    SWIMMING_SPEED,
    WALKING_SPEED,
    MovementSpeed,
    # Acceleration
    AIR_ACCELERATION,
    AIR_DECELERATION,
    GROUND_ACCELERATION,
    GROUND_DECELERATION,
    TURN_ACCELERATION,
    # Platform
    MAX_PLATFORM_VELOCITY,
    PLATFORM_DETACH_THRESHOLD,
    PLATFORM_STICK_FORCE,
    # Ragdoll
    RAGDOLL_BLEND_TIME_MS,
    RAGDOLL_MIN_VELOCITY,
    RAGDOLL_RECOVERY_TIME_MS,
    RAGDOLL_SETTLED_TIME_MS,
    # Active Ragdoll
    BALANCE_THRESHOLD,
    DEFAULT_PD_KD,
    DEFAULT_PD_KP,
    MAX_TORQUE,
    # Blend
    BLEND_ADDITIVE,
    BLEND_CHAIN,
    BLEND_POSE,
    DEFAULT_BLEND_WEIGHT,
    HIT_REACTION_BLEND_IN_MS,
    HIT_REACTION_BLEND_OUT_MS,
    BlendMode,
    # Interaction
    CARRY_MASS_LIMIT,
    CLIMB_MAX_HEIGHT,
    GRAB_DISTANCE,
    PUSH_FORCE,
    THROW_FORCE_MULTIPLIER,
    VAULT_MAX_HEIGHT,
    # Collision Layers
    CollisionLayer,
    LAYER_CHARACTER,
    LAYER_DEFAULT,
    LAYER_DYNAMIC,
    LAYER_PLATFORM,
    LAYER_RAGDOLL,
    LAYER_STATIC,
    MASK_CHARACTER_MOVEMENT,
    MASK_GROUND_DETECTION,
    MASK_RAGDOLL,
    # Materials
    FRICTION_CONCRETE,
    FRICTION_DEFAULT,
    FRICTION_GRASS,
    FRICTION_ICE,
    FRICTION_METAL,
    FRICTION_MUD,
    FRICTION_SAND,
    FRICTION_WOOD,
    SURFACE_FRICTION,
    SurfaceMaterial,
    # Performance
    MAX_ACTIVE_RAGDOLLS,
    MAX_CHARACTERS_PER_FRAME,
    MAX_PLATFORMS_PER_CHARACTER,
    MAX_RAGDOLL_BODIES,
)


class TestCapsuleConfiguration:
    """Tests for capsule dimension constants."""

    def test_capsule_height_positive(self):
        """Capsule height must be positive."""
        assert DEFAULT_CAPSULE_HEIGHT > 0

    def test_capsule_radius_positive(self):
        """Capsule radius must be positive."""
        assert DEFAULT_CAPSULE_RADIUS > 0

    def test_capsule_height_greater_than_radius(self):
        """Height should exceed twice the radius for valid capsule."""
        assert DEFAULT_CAPSULE_HEIGHT > DEFAULT_CAPSULE_RADIUS * 2

    def test_crouched_height_less_than_standing(self):
        """Crouched height must be less than standing height."""
        assert DEFAULT_CROUCHED_HEIGHT < DEFAULT_CAPSULE_HEIGHT

    def test_prone_height_less_than_crouched(self):
        """Prone height must be less than crouched height."""
        assert DEFAULT_PRONE_HEIGHT < DEFAULT_CROUCHED_HEIGHT

    def test_prone_height_positive(self):
        """Prone height must still be positive."""
        assert DEFAULT_PRONE_HEIGHT > 0


class TestSlopeConfiguration:
    """Tests for slope and step constants."""

    def test_step_height_positive(self):
        """Step height must be positive."""
        assert DEFAULT_STEP_HEIGHT > 0

    def test_step_height_reasonable(self):
        """Step height should not exceed typical stair height."""
        assert DEFAULT_STEP_HEIGHT <= 0.5

    def test_slope_angle_order(self):
        """Slope angles should be in ascending order."""
        assert MIN_SLOPE_ANGLE < MAX_SLOPE_ANGLE < STEEP_SLOPE_ANGLE

    def test_max_slope_angle_range(self):
        """Max slope angle should be in valid range."""
        assert 0 < MAX_SLOPE_ANGLE < 90

    def test_steep_slope_angle_range(self):
        """Steep slope angle should be between max and vertical."""
        assert MAX_SLOPE_ANGLE < STEEP_SLOPE_ANGLE < 90


class TestCollisionConfiguration:
    """Tests for collision detection constants."""

    def test_skin_width_small_positive(self):
        """Skin width should be small but positive."""
        assert 0 < SKIN_WIDTH < 0.1

    def test_ground_probe_distance_positive(self):
        """Ground probe distance must be positive."""
        assert GROUND_PROBE_DISTANCE > 0

    def test_ground_sphere_probe_radius_positive(self):
        """Ground sphere probe radius must be positive."""
        assert GROUND_SPHERE_PROBE_RADIUS > 0

    def test_max_collision_iterations_reasonable(self):
        """Collision iterations should be bounded."""
        assert 1 <= MAX_COLLISION_ITERATIONS <= 10

    def test_min_move_distance_very_small(self):
        """Min move distance should be very small."""
        assert 0 < MIN_MOVE_DISTANCE < 0.01

    def test_max_depenetration_velocity_positive(self):
        """Max depenetration velocity must be positive."""
        assert MAX_DEPENETRATION_VELOCITY > 0


class TestMovementConfiguration:
    """Tests for movement physics constants."""

    def test_gravity_negative(self):
        """Gravity should be negative (downward)."""
        assert DEFAULT_GRAVITY < 0

    def test_gravity_earth_like(self):
        """Gravity should be roughly Earth-like."""
        assert -15 < DEFAULT_GRAVITY < -5

    def test_jump_velocity_positive(self):
        """Jump velocity should be positive (upward)."""
        assert DEFAULT_JUMP_VELOCITY > 0

    def test_max_fall_velocity_positive(self):
        """Max fall velocity should be positive (magnitude)."""
        assert MAX_FALL_VELOCITY > 0

    def test_air_control_normalized(self):
        """Air control should be between 0 and 1."""
        assert 0 <= AIR_CONTROL <= 1


class TestGroundDetectionConfiguration:
    """Tests for ground detection timing constants."""

    def test_coyote_time_positive(self):
        """Coyote time must be positive."""
        assert COYOTE_TIME_MS > 0

    def test_coyote_time_reasonable(self):
        """Coyote time should not be too long."""
        assert COYOTE_TIME_MS <= 500

    def test_jump_buffer_time_positive(self):
        """Jump buffer time must be positive."""
        assert JUMP_BUFFER_TIME_MS > 0

    def test_ledge_grab_distance_positive(self):
        """Ledge grab distance must be positive."""
        assert LEDGE_GRAB_DISTANCE > 0

    def test_ledge_detection_height_greater_than_standing(self):
        """Ledge detection height should account for reach."""
        assert LEDGE_DETECTION_HEIGHT > DEFAULT_CAPSULE_HEIGHT


class TestMovementSpeedEnum:
    """Tests for MovementSpeed enum."""

    def test_walking_speed_positive(self):
        """Walking speed must be positive."""
        assert MovementSpeed.WALKING.value > 0

    def test_running_faster_than_walking(self):
        """Running should be faster than walking."""
        assert MovementSpeed.RUNNING.value > MovementSpeed.WALKING.value

    def test_sprinting_faster_than_running(self):
        """Sprinting should be faster than running."""
        assert MovementSpeed.SPRINTING.value > MovementSpeed.RUNNING.value

    def test_crouching_slower_than_walking(self):
        """Crouching should be slower than walking."""
        assert MovementSpeed.CROUCHING.value < MovementSpeed.WALKING.value

    def test_prone_slowest(self):
        """Prone should be slowest standard movement."""
        assert MovementSpeed.PRONE.value < MovementSpeed.CROUCHING.value

    def test_swimming_reasonable(self):
        """Swimming speed should be reasonable."""
        assert 0 < MovementSpeed.SWIMMING.value < MovementSpeed.RUNNING.value

    def test_speed_constants_match_enum(self):
        """Speed constants should match enum values."""
        assert WALKING_SPEED == MovementSpeed.WALKING.value
        assert RUNNING_SPEED == MovementSpeed.RUNNING.value
        assert SPRINTING_SPEED == MovementSpeed.SPRINTING.value
        assert CROUCHING_SPEED == MovementSpeed.CROUCHING.value
        assert PRONE_SPEED == MovementSpeed.PRONE.value
        assert SWIMMING_SPEED == MovementSpeed.SWIMMING.value
        assert CLIMBING_SPEED == MovementSpeed.CLIMBING.value
        assert FLYING_SPEED == MovementSpeed.FLYING.value


class TestAccelerationConfiguration:
    """Tests for acceleration constants."""

    def test_ground_acceleration_positive(self):
        """Ground acceleration must be positive."""
        assert GROUND_ACCELERATION > 0

    def test_ground_deceleration_positive(self):
        """Ground deceleration must be positive."""
        assert GROUND_DECELERATION > 0

    def test_air_acceleration_less_than_ground(self):
        """Air acceleration should be less than ground."""
        assert AIR_ACCELERATION < GROUND_ACCELERATION

    def test_turn_acceleration_positive(self):
        """Turn acceleration must be positive."""
        assert TURN_ACCELERATION > 0


class TestPlatformConfiguration:
    """Tests for platform handling constants."""

    def test_platform_stick_force_positive(self):
        """Platform stick force must be positive."""
        assert PLATFORM_STICK_FORCE > 0

    def test_platform_detach_threshold_positive(self):
        """Platform detach threshold must be positive."""
        assert PLATFORM_DETACH_THRESHOLD > 0

    def test_max_platform_velocity_positive(self):
        """Max platform velocity must be positive."""
        assert MAX_PLATFORM_VELOCITY > 0


class TestRagdollConfiguration:
    """Tests for ragdoll physics constants."""

    def test_blend_time_positive(self):
        """Ragdoll blend time must be positive."""
        assert RAGDOLL_BLEND_TIME_MS > 0

    def test_recovery_time_greater_than_blend(self):
        """Recovery time should be at least as long as blend time."""
        assert RAGDOLL_RECOVERY_TIME_MS >= RAGDOLL_BLEND_TIME_MS

    def test_min_velocity_positive(self):
        """Minimum velocity threshold must be positive."""
        assert RAGDOLL_MIN_VELOCITY > 0

    def test_settled_time_positive(self):
        """Settled time must be positive."""
        assert RAGDOLL_SETTLED_TIME_MS > 0


class TestActiveRagdollConfiguration:
    """Tests for active ragdoll PD controller constants."""

    def test_pd_kp_positive(self):
        """PD proportional gain must be positive."""
        assert DEFAULT_PD_KP > 0

    def test_pd_kd_positive(self):
        """PD derivative gain must be positive."""
        assert DEFAULT_PD_KD > 0

    def test_pd_kd_less_than_kp(self):
        """Derivative gain typically less than proportional."""
        assert DEFAULT_PD_KD < DEFAULT_PD_KP

    def test_max_torque_positive(self):
        """Max torque must be positive."""
        assert MAX_TORQUE > 0

    def test_balance_threshold_positive(self):
        """Balance threshold must be positive."""
        assert BALANCE_THRESHOLD > 0


class TestBlendModeEnum:
    """Tests for BlendMode enum."""

    def test_blend_mode_values(self):
        """BlendMode enum should have expected values."""
        assert BlendMode.POSE.value == "pose"
        assert BlendMode.ADDITIVE.value == "additive"
        assert BlendMode.CHAIN.value == "chain"

    def test_blend_constants_match_enum(self):
        """Blend constants should match enum values."""
        assert BLEND_POSE == BlendMode.POSE.value
        assert BLEND_ADDITIVE == BlendMode.ADDITIVE.value
        assert BLEND_CHAIN == BlendMode.CHAIN.value

    def test_default_blend_weight_normalized(self):
        """Default blend weight should be 0-1."""
        assert 0 <= DEFAULT_BLEND_WEIGHT <= 1

    def test_hit_reaction_blend_times_positive(self):
        """Hit reaction blend times must be positive."""
        assert HIT_REACTION_BLEND_IN_MS > 0
        assert HIT_REACTION_BLEND_OUT_MS > 0

    def test_blend_out_longer_than_in(self):
        """Blend out typically longer than blend in."""
        assert HIT_REACTION_BLEND_OUT_MS >= HIT_REACTION_BLEND_IN_MS


class TestInteractionConfiguration:
    """Tests for character interaction constants."""

    def test_push_force_positive(self):
        """Push force must be positive."""
        assert PUSH_FORCE > 0

    def test_grab_distance_positive(self):
        """Grab distance must be positive."""
        assert GRAB_DISTANCE > 0

    def test_carry_mass_limit_positive(self):
        """Carry mass limit must be positive."""
        assert CARRY_MASS_LIMIT > 0

    def test_throw_force_multiplier_positive(self):
        """Throw force multiplier must be positive."""
        assert THROW_FORCE_MULTIPLIER > 0

    def test_vault_max_height_positive(self):
        """Vault max height must be positive."""
        assert VAULT_MAX_HEIGHT > 0

    def test_climb_max_height_greater_than_vault(self):
        """Climb height should be greater than vault height."""
        assert CLIMB_MAX_HEIGHT > VAULT_MAX_HEIGHT


class TestCollisionLayerEnum:
    """Tests for CollisionLayer enum."""

    def test_layer_values_unique(self):
        """All collision layer values should be unique."""
        values = [layer.value for layer in CollisionLayer]
        assert len(values) == len(set(values))

    def test_layer_values_valid_bits(self):
        """Layer values should be valid bit indices."""
        for layer in CollisionLayer:
            assert 0 <= layer.value < 32

    def test_layer_constants_match_enum(self):
        """Layer constants should match enum values."""
        assert LAYER_DEFAULT == CollisionLayer.DEFAULT
        assert LAYER_STATIC == CollisionLayer.STATIC
        assert LAYER_DYNAMIC == CollisionLayer.DYNAMIC
        assert LAYER_CHARACTER == CollisionLayer.CHARACTER
        assert LAYER_PLATFORM == CollisionLayer.PLATFORM
        assert LAYER_RAGDOLL == CollisionLayer.RAGDOLL


class TestCollisionMasks:
    """Tests for collision masks."""

    def test_character_movement_mask_includes_static(self):
        """Character movement should collide with static objects."""
        static_bit = 1 << CollisionLayer.STATIC
        assert (MASK_CHARACTER_MOVEMENT & static_bit) != 0

    def test_character_movement_mask_includes_dynamic(self):
        """Character movement should collide with dynamic objects."""
        dynamic_bit = 1 << CollisionLayer.DYNAMIC
        assert (MASK_CHARACTER_MOVEMENT & dynamic_bit) != 0

    def test_character_movement_mask_includes_platform(self):
        """Character movement should collide with platforms."""
        platform_bit = 1 << CollisionLayer.PLATFORM
        assert (MASK_CHARACTER_MOVEMENT & platform_bit) != 0

    def test_ground_detection_mask_same_as_movement(self):
        """Ground detection should have same mask as movement."""
        assert MASK_GROUND_DETECTION == MASK_CHARACTER_MOVEMENT

    def test_ragdoll_mask_includes_ragdoll(self):
        """Ragdoll mask should include ragdoll layer for self-collision."""
        ragdoll_bit = 1 << CollisionLayer.RAGDOLL
        assert (MASK_RAGDOLL & ragdoll_bit) != 0


class TestSurfaceMaterialEnum:
    """Tests for SurfaceMaterial enum."""

    def test_surface_material_values(self):
        """SurfaceMaterial enum should have expected values."""
        assert SurfaceMaterial.DEFAULT.value == "default"
        assert SurfaceMaterial.CONCRETE.value == "concrete"
        assert SurfaceMaterial.ICE.value == "ice"

    def test_all_materials_have_friction(self):
        """All surface materials should have friction defined."""
        for material in SurfaceMaterial:
            if material != SurfaceMaterial.WATER:  # Water may not have friction
                assert material.value in SURFACE_FRICTION

    def test_ice_has_lowest_friction(self):
        """Ice should have the lowest friction."""
        assert FRICTION_ICE < FRICTION_MUD
        assert FRICTION_ICE < FRICTION_SAND
        assert FRICTION_ICE < FRICTION_METAL

    def test_concrete_has_highest_friction(self):
        """Concrete should have highest friction."""
        assert FRICTION_CONCRETE >= FRICTION_DEFAULT
        assert FRICTION_CONCRETE >= FRICTION_WOOD
        assert FRICTION_CONCRETE >= FRICTION_GRASS

    def test_friction_values_normalized(self):
        """All friction values should be 0-1."""
        for friction in SURFACE_FRICTION.values():
            assert 0 <= friction <= 1


class TestPerformanceConfiguration:
    """Tests for performance limit constants."""

    def test_max_characters_reasonable(self):
        """Max characters per frame should be reasonable."""
        assert 1 <= MAX_CHARACTERS_PER_FRAME <= 256

    def test_max_ragdoll_bodies_reasonable(self):
        """Max ragdoll bodies should be reasonable."""
        assert 1 <= MAX_RAGDOLL_BODIES <= 64

    def test_max_active_ragdolls_reasonable(self):
        """Max active ragdolls should be reasonable."""
        assert 1 <= MAX_ACTIVE_RAGDOLLS <= MAX_CHARACTERS_PER_FRAME

    def test_max_platforms_per_character_reasonable(self):
        """Max platforms per character should be small."""
        assert 1 <= MAX_PLATFORMS_PER_CHARACTER <= 8
