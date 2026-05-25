"""
Tests for the volumes system (volumes.py).

Tests all volume types, bounds calculations, blending,
and the VolumeManager.
"""

import math
import pytest

from engine.world.environment.volumes import (
    # Enums
    VolumeType,
    VolumeShape,
    OverlapState,
    # Bounds
    BoundingBox,
    SphereBounds,
    CapsuleBounds,
    # Base
    BaseVolume,
    # Physics volumes
    PhysicsVolume,
    WaterVolume,
    PainVolume,
    KillVolume,
    # Gameplay volumes
    TriggerVolume,
    BlockingVolume,
    CameraVolume,
    SpawnVolume,
    # Visual volumes
    PostProcessVolume,
    FogVolume,
    ReflectionVolume,
    LightmassVolume,
    # Audio volumes
    ReverbVolume,
    AmbientVolume,
    # Navigation volumes
    NavModifierVolume,
    NavLinkVolume,
    NavExcludeVolume,
    # Manager
    VolumeManager,
)


# =============================================================================
# BoundingBox Tests
# =============================================================================


class TestBoundingBox:
    def test_default_creation(self):
        box = BoundingBox()
        assert box.min_point == (0.0, 0.0, 0.0)
        assert box.max_point == (1.0, 1.0, 1.0)

    def test_custom_bounds(self):
        box = BoundingBox(min_point=(-5, -5, -5), max_point=(5, 5, 5))
        assert box.min_point == (-5, -5, -5)
        assert box.max_point == (5, 5, 5)

    def test_center_calculation(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert box.center == (5.0, 5.0, 5.0)

    def test_extents_calculation(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 20, 30))
        assert box.extents == (5.0, 10.0, 15.0)

    def test_contains_point_inside(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert box.contains_point((5, 5, 5)) is True

    def test_contains_point_on_edge(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert box.contains_point((0, 5, 5)) is True
        assert box.contains_point((10, 5, 5)) is True

    def test_contains_point_outside(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        assert box.contains_point((15, 5, 5)) is False
        assert box.contains_point((-1, 5, 5)) is False

    def test_distance_to_edge_inside(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        dist = box.distance_to_edge((5, 5, 5))
        assert dist == 5.0  # Distance to nearest face

    def test_distance_to_edge_at_corner(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        dist = box.distance_to_edge((0, 0, 0))
        assert dist == 0.0

    def test_distance_to_edge_outside(self):
        box = BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10))
        dist = box.distance_to_edge((15, 5, 5))
        assert dist < 0


# =============================================================================
# SphereBounds Tests
# =============================================================================


class TestSphereBounds:
    def test_default_creation(self):
        sphere = SphereBounds()
        assert sphere.center == (0.0, 0.0, 0.0)
        assert sphere.radius == 1.0

    def test_contains_point_inside(self):
        sphere = SphereBounds(center=(0, 0, 0), radius=10)
        assert sphere.contains_point((5, 0, 0)) is True
        assert sphere.contains_point((0, 5, 0)) is True

    def test_contains_point_on_surface(self):
        sphere = SphereBounds(center=(0, 0, 0), radius=10)
        assert sphere.contains_point((10, 0, 0)) is True

    def test_contains_point_outside(self):
        sphere = SphereBounds(center=(0, 0, 0), radius=10)
        assert sphere.contains_point((15, 0, 0)) is False

    def test_distance_to_edge_center(self):
        sphere = SphereBounds(center=(0, 0, 0), radius=10)
        dist = sphere.distance_to_edge((0, 0, 0))
        assert dist == 10.0

    def test_distance_to_edge_surface(self):
        sphere = SphereBounds(center=(0, 0, 0), radius=10)
        dist = sphere.distance_to_edge((10, 0, 0))
        assert abs(dist) < 0.001

    def test_distance_to_edge_outside(self):
        sphere = SphereBounds(center=(0, 0, 0), radius=10)
        dist = sphere.distance_to_edge((15, 0, 0))
        assert dist == -5.0


# =============================================================================
# CapsuleBounds Tests
# =============================================================================


class TestCapsuleBounds:
    def test_default_creation(self):
        capsule = CapsuleBounds()
        assert capsule.point_a == (0.0, 0.0, 0.0)
        assert capsule.point_b == (0.0, 1.0, 0.0)
        assert capsule.radius == 0.5

    def test_contains_point_at_end(self):
        capsule = CapsuleBounds(point_a=(0, 0, 0), point_b=(0, 10, 0), radius=2)
        assert capsule.contains_point((0, 0, 0)) is True
        assert capsule.contains_point((0, 10, 0)) is True

    def test_contains_point_along_axis(self):
        capsule = CapsuleBounds(point_a=(0, 0, 0), point_b=(0, 10, 0), radius=2)
        assert capsule.contains_point((0, 5, 0)) is True

    def test_contains_point_offset(self):
        capsule = CapsuleBounds(point_a=(0, 0, 0), point_b=(0, 10, 0), radius=2)
        assert capsule.contains_point((1, 5, 0)) is True  # Within radius
        assert capsule.contains_point((3, 5, 0)) is False  # Outside radius

    def test_distance_to_edge(self):
        capsule = CapsuleBounds(point_a=(0, 0, 0), point_b=(0, 10, 0), radius=2)
        dist = capsule.distance_to_edge((0, 5, 0))
        assert dist == 2.0  # At axis, distance is radius


# =============================================================================
# PhysicsVolume Tests
# =============================================================================


class TestPhysicsVolume:
    def test_default_creation(self):
        volume = PhysicsVolume()
        assert volume.volume_type == VolumeType.GRAVITY
        assert volume.gravity_strength == 9.81

    def test_gravity_direction_normalized(self):
        volume = PhysicsVolume(gravity_direction=(0, -2, 0))
        assert volume.gravity_direction == (0.0, -1.0, 0.0)

    def test_gravity_vector(self):
        volume = PhysicsVolume(gravity_direction=(0, -1, 0), gravity_strength=10.0)
        grav = volume.get_gravity_vector()
        assert grav == (0.0, -10.0, 0.0)

    def test_fluid_friction(self):
        volume = PhysicsVolume(fluid_friction=0.5)
        assert volume.fluid_friction == 0.5

    def test_settings_dict(self):
        volume = PhysicsVolume()
        settings = volume.get_settings_dict()
        assert "gravity_direction" in settings
        assert "gravity_strength" in settings
        assert "fluid_friction" in settings


# =============================================================================
# WaterVolume Tests
# =============================================================================


class TestWaterVolume:
    def test_default_creation(self):
        volume = WaterVolume()
        assert volume.volume_type == VolumeType.WATER
        assert volume.buoyancy == 1.0

    def test_water_height(self):
        volume = WaterVolume(water_height=10.0)
        assert volume.water_height == 10.0

    def test_submersion_depth(self):
        volume = WaterVolume(water_height=10.0)
        depth = volume.get_submersion_depth((0, 5, 0))
        assert depth == 5.0  # 5 units below surface

    def test_current_vector(self):
        volume = WaterVolume(current_direction=(1, 0, 0), current_strength=5.0)
        current = volume.get_current_vector()
        assert current == (5.0, 0.0, 0.0)

    def test_drag_clamped(self):
        volume = WaterVolume(drag=2.0)
        assert volume.drag == 1.0  # Clamped to max


# =============================================================================
# PainVolume Tests
# =============================================================================


class TestPainVolume:
    def test_default_creation(self):
        volume = PainVolume()
        assert volume.volume_type == VolumeType.PAIN
        assert volume.damage_per_second == 10.0

    def test_damage_calculation(self):
        volume = PainVolume(damage_per_second=20.0)
        damage = volume.calculate_damage(0.5)  # 0.5 seconds
        assert damage == 10.0

    def test_damage_type(self):
        volume = PainVolume(damage_type="fire")
        assert volume.damage_type == "fire"


# =============================================================================
# KillVolume Tests
# =============================================================================


class TestKillVolume:
    def test_default_creation(self):
        volume = KillVolume()
        assert volume.volume_type == VolumeType.KILL
        assert "Fell out" in volume.kill_message

    def test_respawn_point(self):
        volume = KillVolume(respawn_point=(0, 10, 0))
        assert volume.respawn_point == (0, 10, 0)


# =============================================================================
# TriggerVolume Tests
# =============================================================================


class TestTriggerVolume:
    def test_default_creation(self):
        volume = TriggerVolume()
        assert volume.volume_type == VolumeType.TRIGGER
        assert len(volume.overlapping_actors) == 0

    def test_check_overlap_entered(self):
        volume = TriggerVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        state = volume.check_overlap("actor1", (5, 5, 5))
        assert state == OverlapState.ENTERED
        assert "actor1" in volume.overlapping_actors

    def test_check_overlap_inside(self):
        volume = TriggerVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        volume.check_overlap("actor1", (5, 5, 5))  # Enter
        state = volume.check_overlap("actor1", (6, 6, 6))  # Still inside
        assert state == OverlapState.INSIDE

    def test_check_overlap_exited(self):
        volume = TriggerVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        volume.check_overlap("actor1", (5, 5, 5))  # Enter
        state = volume.check_overlap("actor1", (15, 15, 15))  # Exit
        assert state == OverlapState.EXITED
        assert "actor1" not in volume.overlapping_actors

    def test_filter_tags(self):
        volume = TriggerVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            filter_tags=["player"],
        )
        # Actor with matching tag
        state = volume.check_overlap("actor1", (5, 5, 5), tags=["player"])
        assert state == OverlapState.ENTERED

        # Actor without matching tag
        state = volume.check_overlap("actor2", (5, 5, 5), tags=["enemy"])
        assert state == OverlapState.OUTSIDE

    def test_callbacks(self):
        volume = TriggerVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        entered = []
        exited = []
        volume.add_on_enter(lambda a: entered.append(a))
        volume.add_on_exit(lambda a: exited.append(a))

        # Trigger callbacks
        state = volume.check_overlap("actor1", (5, 5, 5))
        volume.trigger_callbacks(state, "actor1")
        assert "actor1" in entered

        state = volume.check_overlap("actor1", (15, 15, 15))
        volume.trigger_callbacks(state, "actor1")
        assert "actor1" in exited


# =============================================================================
# PostProcessVolume Tests
# =============================================================================


class TestPostProcessVolume:
    def test_default_creation(self):
        volume = PostProcessVolume()
        assert volume.volume_type == VolumeType.POST_PROCESS
        assert volume.exposure == 1.0

    def test_exposure_clamped(self):
        volume = PostProcessVolume()
        volume.exposure = 0.001
        assert volume.exposure == 0.01  # Min clamp

    def test_saturation(self):
        volume = PostProcessVolume(saturation=1.5)
        assert volume.saturation == 1.5

    def test_vignette_clamped(self):
        volume = PostProcessVolume()
        volume.vignette_intensity = 2.0
        assert volume.vignette_intensity == 1.0  # Max clamp

    def test_settings_dict(self):
        volume = PostProcessVolume(bloom_intensity=0.5)
        settings = volume.get_settings_dict()
        assert settings["bloom_intensity"] == 0.5


# =============================================================================
# FogVolume Tests
# =============================================================================


class TestFogVolume:
    def test_default_creation(self):
        volume = FogVolume()
        assert volume.volume_type == VolumeType.FOG
        assert volume.fog_density == 0.02

    def test_fog_color(self):
        volume = FogVolume(fog_color=(1.0, 0.5, 0.0))
        assert volume.fog_color == (1.0, 0.5, 0.0)

    def test_fog_color_clamped(self):
        volume = FogVolume()
        volume.fog_color = (2.0, -0.5, 0.5)
        assert volume.fog_color == (1.0, 0.0, 0.5)


# =============================================================================
# ReverbVolume Tests
# =============================================================================


class TestReverbVolume:
    def test_default_creation(self):
        volume = ReverbVolume()
        assert volume.volume_type == VolumeType.REVERB
        assert volume.reverb_preset == "default"

    def test_room_size_clamped(self):
        volume = ReverbVolume()
        volume.room_size = 2.0
        assert volume.room_size == 1.0

    def test_wet_dry_mix(self):
        volume = ReverbVolume(wet_dry_mix=0.7)
        assert volume.wet_dry_mix == 0.7


# =============================================================================
# SpawnVolume Tests
# =============================================================================


class TestSpawnVolume:
    def test_default_creation(self):
        volume = SpawnVolume()
        assert volume.volume_type == VolumeType.SPAWN

    def test_random_point_in_box(self):
        volume = SpawnVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        point = volume.get_random_point()
        assert 0 <= point[0] <= 10
        assert 0 <= point[1] <= 10
        assert 0 <= point[2] <= 10

    def test_spawn_team(self):
        volume = SpawnVolume(spawn_team="red")
        assert volume.spawn_team == "red"


# =============================================================================
# Navigation Volumes Tests
# =============================================================================


class TestNavVolumes:
    def test_nav_modifier_volume(self):
        volume = NavModifierVolume(area_class="water", cost_modifier=2.0)
        assert volume.volume_type == VolumeType.NAV_MODIFIER
        assert volume.area_class == "water"
        assert volume.cost_modifier == 2.0

    def test_nav_link_volume(self):
        volume = NavLinkVolume(link_type="ladder", target_point=(0, 10, 0))
        assert volume.volume_type == VolumeType.NAV_LINK
        assert volume.link_type == "ladder"
        assert volume.target_point == (0, 10, 0)

    def test_nav_exclude_volume(self):
        volume = NavExcludeVolume()
        assert volume.volume_type == VolumeType.NAV_EXCLUDE


# =============================================================================
# BaseVolume Tests
# =============================================================================


class TestBaseVolume:
    def test_blend_weight_inside(self):
        volume = PhysicsVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            blend_radius=2.0,
        )
        # At center - fully inside
        weight = volume.get_blend_weight((5, 5, 5))
        assert weight == 1.0

    def test_blend_weight_at_edge(self):
        volume = PhysicsVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            blend_radius=2.0,
        )
        # At edge
        weight = volume.get_blend_weight((0, 5, 5))
        assert weight == 0.0

    def test_blend_weight_in_blend_zone(self):
        volume = PhysicsVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            blend_radius=2.0,
        )
        # 1 unit from edge (in blend zone)
        weight = volume.get_blend_weight((1, 5, 5))
        assert 0 < weight < 1

    def test_inactive_volume(self):
        volume = TriggerVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            is_active=False,
        )
        assert volume.contains_point(5, 5, 5) is False
        assert volume.get_blend_weight((5, 5, 5)) == 0.0

    def test_observer_notification(self):
        volume = PhysicsVolume()
        changes = []
        volume.add_observer("gravity_strength", lambda v, f, o, n: changes.append((f, o, n)))
        volume.gravity_strength = 15.0
        assert len(changes) == 1
        assert changes[0] == ("gravity_strength", 9.81, 15.0)


# =============================================================================
# VolumeManager Tests
# =============================================================================


class TestVolumeManager:
    def test_add_volume(self):
        manager = VolumeManager()
        volume = TriggerVolume()
        manager.add_volume(volume)
        assert manager.get_volume(volume.volume_id) is volume

    def test_remove_volume(self):
        manager = VolumeManager()
        volume = TriggerVolume()
        manager.add_volume(volume)
        removed = manager.remove_volume(volume.volume_id)
        assert removed is volume
        assert manager.get_volume(volume.volume_id) is None

    def test_get_volumes_at_point(self):
        manager = VolumeManager()
        v1 = TriggerVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            priority=1,
        )
        v2 = TriggerVolume(
            bounds=BoundingBox(min_point=(5, 5, 5), max_point=(15, 15, 15)),
            priority=2,
        )
        manager.add_volume(v1)
        manager.add_volume(v2)

        # Point inside both
        volumes = manager.get_volumes_at_point((7, 7, 7))
        assert len(volumes) == 2
        assert volumes[0].priority > volumes[1].priority  # Sorted by priority

    def test_get_volumes_by_type(self):
        manager = VolumeManager()
        trigger = TriggerVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        fog = FogVolume(bounds=BoundingBox(
            min_point=(0, 0, 0), max_point=(10, 10, 10)
        ))
        manager.add_volume(trigger)
        manager.add_volume(fog)

        volumes = manager.get_volumes_at_point((5, 5, 5), VolumeType.FOG)
        assert len(volumes) == 1
        assert volumes[0] is fog

    def test_get_blended_settings(self):
        manager = VolumeManager()
        v1 = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            exposure=1.0,
            priority=1,
        )
        v2 = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(10, 10, 10)),
            exposure=2.0,
            priority=2,
        )
        manager.add_volume(v1)
        manager.add_volume(v2)

        settings = manager.get_blended_settings((5, 5, 5), VolumeType.POST_PROCESS)
        assert "exposure" in settings
        # Should be weighted average

    def test_clear(self):
        manager = VolumeManager()
        manager.add_volume(TriggerVolume())
        manager.add_volume(FogVolume())
        manager.clear()
        assert len(manager.get_all_volumes()) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestVolumeIntegration:
    def test_overlapping_volumes_priority(self):
        manager = VolumeManager()

        # Create overlapping post-process volumes with different priorities
        v1 = PostProcessVolume(
            bounds=BoundingBox(min_point=(0, 0, 0), max_point=(20, 20, 20)),
            exposure=1.0,
            saturation=1.0,
            priority=1,
        )
        v2 = PostProcessVolume(
            bounds=BoundingBox(min_point=(5, 5, 5), max_point=(15, 15, 15)),
            exposure=2.0,
            saturation=0.5,
            priority=10,  # Higher priority
        )
        manager.add_volume(v1)
        manager.add_volume(v2)

        # Get blended settings at overlapping point
        settings = manager.get_blended_settings((10, 10, 10), VolumeType.POST_PROCESS)

        # Higher priority volume should have more influence
        assert settings["exposure"] > 1.0

    def test_volume_with_sphere_bounds(self):
        volume = PhysicsVolume(
            bounds=SphereBounds(center=(5, 5, 5), radius=5),
            shape=VolumeShape.SPHERE,
            gravity_strength=0.0,
        )

        assert volume.contains_point(5, 5, 5) is True
        assert volume.contains_point(0, 5, 5) is True  # At edge
        assert volume.contains_point(-5, 5, 5) is False  # Outside
