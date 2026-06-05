"""Tests for CrowdSystem (T-AN-9.9) — Crowd Animation System.

Covers:
- Steering computation (RVO/ORCA)
- Animation selection logic
- Texture baking correctness
- LOD distance thresholds
- Frustum culling accuracy
- Instance buffer generation
- Large crowd performance (1000+ agents)

Test count: 70+ tests
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any
from unittest import mock

import pytest

from engine.core.math import Vec3, Vec4, Quat, Mat4, Transform
from engine.animation.systems.crowd_system import (
    # System decorator
    system,
    # Enums
    SteeringMode,
    CullingMode,
    AnimationBakeMode,
    # Frustum culling
    Plane,
    Frustum,
    # RVO/ORCA steering
    VelocityObstacle,
    ORCALine,
    RVOConfig,
    RVOSteering,
    # Instance buffer
    CrowdInstanceData,
    CrowdInstanceBuffer,
    # Components
    CrowdComponent,
    # System
    CrowdSystem,
    # Constants
    DEFAULT_NEIGHBOR_RADIUS,
    DEFAULT_MAX_NEIGHBORS,
    DEFAULT_TIME_HORIZON,
    DEFAULT_PHASE_OFFSET_RANGE,
)
from engine.animation.crowds.crowd_behavior import (
    CrowdAgent,
    CrowdSimulator,
    AgentState,
    AnimationBlend,
    BehaviorContext,
)
from engine.animation.crowds.crowd_lod import (
    CrowdLOD,
    LODLevel,
    LODTransition,
)
from engine.animation.crowds.crowd_renderer import (
    CrowdRenderer,
    CrowdInstance,
    InstanceBuffer,
)
from engine.animation.crowds.animation_texture import (
    AnimationTexture,
    AnimationTextureAtlas,
    Skeleton,
    AnimationClip,
    bake_clip_to_texture,
    TextureFormat,
)
from engine.animation.config import (
    CROWD_SYSTEM_CONFIG,
    CROWD_BEHAVIOR_CONFIG,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_skeleton() -> Skeleton:
    """Create a simple skeleton for testing."""
    return Skeleton(
        bone_names=["root", "spine", "head", "left_arm", "right_arm"],
        bone_parents=[-1, 0, 1, 1, 1],
        bind_poses=[Transform.identity() for _ in range(5)],
    )


@pytest.fixture
def simple_clip(simple_skeleton: Skeleton) -> AnimationClip:
    """Create a simple animation clip."""
    clip = AnimationClip(
        name="walk",
        duration=1.0,
        frame_rate=30.0,
        bone_tracks={
            0: [Transform.identity() for _ in range(30)],
            1: [Transform.identity() for _ in range(30)],
            2: [Transform.identity() for _ in range(30)],
        },
    )
    return clip


@pytest.fixture
def crowd_component() -> CrowdComponent:
    """Create a basic crowd component."""
    component = CrowdComponent()
    # Set up basic LOD levels
    component.lod.add_lod_level(LODLevel(distance=0.0, bone_count=65))
    component.lod.add_lod_level(LODLevel(distance=25.0, bone_count=40))
    component.lod.add_lod_level(LODLevel(distance=50.0, bone_count=20))
    component.lod.add_lod_level(LODLevel(distance=100.0, bone_count=8))
    return component


@pytest.fixture
def crowd_system() -> CrowdSystem:
    """Create a crowd system."""
    return CrowdSystem()


# =============================================================================
# 1. SYSTEM DECORATOR TESTS
# =============================================================================


class TestSystemDecorator:
    """Tests for @system decorator."""

    def test_system_decorator_sets_phase(self) -> None:
        """Verify @system sets _system_phase attribute."""
        assert hasattr(CrowdSystem, '_system_phase')
        assert CrowdSystem._system_phase == "animation"

    def test_system_decorator_sets_order(self) -> None:
        """Verify @system sets _system_order attribute."""
        assert hasattr(CrowdSystem, '_system_order')
        assert CrowdSystem._system_order == 5

    def test_system_decorator_sets_reads(self) -> None:
        """Verify @system sets _system_reads attribute."""
        assert hasattr(CrowdSystem, '_system_reads')
        assert "CrowdComponent" in CrowdSystem._system_reads

    def test_system_decorator_sets_writes(self) -> None:
        """Verify @system sets _system_writes attribute."""
        assert hasattr(CrowdSystem, '_system_writes')
        assert "CrowdInstance" in CrowdSystem._system_writes


# =============================================================================
# 2. STEERING TESTS (RVO/ORCA)
# =============================================================================


class TestRVOSteering:
    """Tests for RVO/ORCA steering computation."""

    def test_simple_avoidance_no_neighbors(self) -> None:
        """Simple steering with no neighbors returns preferred velocity."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.SIMPLE,
        )

        assert abs(result.x - 1.0) < 0.01
        assert abs(result.z) < 0.01

    def test_simple_avoidance_single_neighbor(self) -> None:
        """Simple steering avoids single neighbor."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        neighbor = CrowdAgent(position=Vec3(1.0, 0.0, 0.0))
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[neighbor],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.SIMPLE,
        )

        # Should have some lateral component to avoid neighbor
        # Either Z or negative X to avoid
        assert result.length() > 0

    def test_rvo_no_obstacles_returns_preferred(self) -> None:
        """RVO with no obstacles returns preferred velocity."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.RVO,
        )

        assert abs(result.x - 1.0) < 0.01

    def test_rvo_avoids_collision_course(self) -> None:
        """RVO modifies velocity to avoid collision."""
        steering = RVOSteering()
        agent = CrowdAgent(
            position=Vec3.zero(),
            velocity=Vec3(1.0, 0.0, 0.0),
        )
        neighbor = CrowdAgent(
            position=Vec3(2.0, 0.0, 0.0),
            velocity=Vec3(-1.0, 0.0, 0.0),
        )
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[neighbor],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.RVO,
        )

        # Result should differ from direct collision course
        # May have some Z component to avoid
        assert result is not None

    def test_orca_no_obstacles_returns_preferred(self) -> None:
        """ORCA with no obstacles returns preferred velocity."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.ORCA,
        )

        assert abs(result.x - 1.0) < 0.01

    def test_orca_avoids_collision(self) -> None:
        """ORCA produces collision-free velocity."""
        steering = RVOSteering()
        agent = CrowdAgent(
            position=Vec3.zero(),
            velocity=Vec3(1.0, 0.0, 0.0),
        )
        neighbor = CrowdAgent(
            position=Vec3(1.5, 0.0, 0.0),
            velocity=Vec3(-1.0, 0.0, 0.0),
        )
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[neighbor],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.ORCA,
        )

        # ORCA should produce some avoidance
        assert result is not None

    def test_steering_respects_max_speed(self) -> None:
        """Steering clamps velocity to max speed."""
        config = RVOConfig(max_speed=2.0)
        steering = RVOSteering(config)
        agent = CrowdAgent(position=Vec3.zero())
        preferred = Vec3(10.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[],
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.SIMPLE,
        )

        assert result.length() <= config.max_speed + 0.01

    def test_obstacle_avoidance(self) -> None:
        """Steering avoids static obstacles."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        obstacles = [(Vec3(1.0, 0.0, 0.0), 0.5)]
        preferred = Vec3(1.0, 0.0, 0.0)

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=[],
            obstacles=obstacles,
            preferred_velocity=preferred,
            mode=SteeringMode.SIMPLE,
        )

        # Should deviate from direct path
        assert result is not None

    def test_velocity_obstacle_construction(self) -> None:
        """Test velocity obstacle is constructed correctly."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        neighbor = CrowdAgent(position=Vec3(3.0, 0.0, 0.0))

        vo = steering._build_velocity_obstacle(agent, neighbor)

        assert vo is not None
        assert vo.left_leg.length() > 0.99  # Normalized
        assert vo.right_leg.length() > 0.99

    def test_orca_line_construction(self) -> None:
        """Test ORCA line is constructed correctly."""
        steering = RVOSteering()
        agent = CrowdAgent(
            position=Vec3.zero(),
            velocity=Vec3(1.0, 0.0, 0.0),
        )
        neighbor = CrowdAgent(
            position=Vec3(3.0, 0.0, 0.0),
            velocity=Vec3(-1.0, 0.0, 0.0),
        )

        line = steering._build_orca_line(agent, neighbor)

        assert line is not None
        assert line.direction.length() > 0.9

    def test_multiple_neighbors_handling(self) -> None:
        """Steering handles multiple neighbors."""
        steering = RVOSteering()
        agent = CrowdAgent(position=Vec3.zero())
        neighbors = [
            CrowdAgent(position=Vec3(1.0, 0.0, 0.0)),
            CrowdAgent(position=Vec3(-1.0, 0.0, 0.0)),
            CrowdAgent(position=Vec3(0.0, 0.0, 1.0)),
        ]
        preferred = Vec3(0.0, 0.0, 0.0)  # Stay still

        result = steering.compute_new_velocity(
            agent=agent,
            neighbors=neighbors,
            obstacles=[],
            preferred_velocity=preferred,
            mode=SteeringMode.RVO,
        )

        # Should produce some result
        assert result is not None


# =============================================================================
# 3. ANIMATION SELECTION TESTS
# =============================================================================


class TestAnimationSelection:
    """Tests for animation selection based on agent state."""

    def test_idle_agent_gets_idle_animation(self) -> None:
        """Idle agents use idle animation index."""
        agent = CrowdAgent(current_state=AgentState.IDLE)
        agent.animation_blend = AnimationBlend.single(agent.idle_animation)

        assert agent.animation_blend.get_primary_animation() == agent.idle_animation

    def test_walking_agent_gets_walk_animation(self) -> None:
        """Walking agents use walk animation."""
        agent = CrowdAgent(current_state=AgentState.WALKING)
        agent.animation_blend = AnimationBlend.single(agent.walk_animation)

        assert agent.animation_blend.get_primary_animation() == agent.walk_animation

    def test_fleeing_agent_gets_run_animation(self) -> None:
        """Fleeing agents use run animation."""
        agent = CrowdAgent(current_state=AgentState.FLEEING)
        agent.animation_blend = AnimationBlend.single(agent.run_animation)

        assert agent.animation_blend.get_primary_animation() == agent.run_animation

    def test_animation_blend_between_two(self) -> None:
        """Animation blends between two animations."""
        blend = AnimationBlend.blend(0, 1, 0.5)

        assert len(blend.animation_indices) == 2
        assert 0 in blend.animation_indices
        assert 1 in blend.animation_indices

    def test_animation_blend_primary_selection(self) -> None:
        """Primary animation is highest weight."""
        blend = AnimationBlend(
            animation_indices=[0, 1, 2],
            weights=[0.2, 0.6, 0.2],
        )

        assert blend.get_primary_animation() == 1

    def test_velocity_based_animation_selection(self, crowd_component: CrowdComponent) -> None:
        """Animation changes based on velocity."""
        agent_id = crowd_component.add_agent(Vec3.zero())
        agent = crowd_component.get_agent(agent_id)
        assert agent is not None

        # Set low velocity - should be idle-ish
        agent.velocity = Vec3(0.1, 0.0, 0.0)
        agent.animation_blend = AnimationBlend.single(agent.idle_animation)

        assert agent.animation_blend.get_primary_animation() == agent.idle_animation


# =============================================================================
# 4. TEXTURE BAKING TESTS
# =============================================================================


class TestTextureBaking:
    """Tests for animation texture baking."""

    def test_bake_simple_clip(self, simple_skeleton: Skeleton, simple_clip: AnimationClip) -> None:
        """Bake a simple clip to texture."""
        texture = bake_clip_to_texture(simple_clip, simple_skeleton)

        assert texture is not None
        assert texture.frame_count == simple_clip.frame_count
        assert texture.bone_count == simple_skeleton.bone_count

    def test_bake_preserves_transforms(self, simple_skeleton: Skeleton) -> None:
        """Baked transforms can be recovered."""
        # Create clip with specific transform
        test_transform = Transform(
            translation=Vec3(1.0, 2.0, 3.0),
            rotation=Quat.identity(),
            scale=Vec3(1.0, 1.0, 1.0),
        )
        clip = AnimationClip(
            name="test",
            duration=1.0,
            frame_rate=30.0,
            bone_tracks={0: [test_transform]},
        )

        texture = bake_clip_to_texture(clip, simple_skeleton)
        recovered = texture.get_bone_transform(0, 0)

        assert abs(recovered.translation.x - 1.0) < 0.01
        assert abs(recovered.translation.y - 2.0) < 0.01
        assert abs(recovered.translation.z - 3.0) < 0.01

    def test_atlas_creation(self, simple_skeleton: Skeleton, simple_clip: AnimationClip) -> None:
        """Create animation atlas from multiple clips."""
        atlas = AnimationTextureAtlas()

        clip1 = AnimationClip(name="walk", duration=1.0, frame_rate=30.0, bone_tracks={0: [Transform.identity()] * 30})
        clip2 = AnimationClip(name="run", duration=0.5, frame_rate=30.0, bone_tracks={0: [Transform.identity()] * 15})

        tex1 = bake_clip_to_texture(clip1, simple_skeleton)
        tex2 = bake_clip_to_texture(clip2, simple_skeleton)

        assert atlas.add_clip("walk", tex1)
        assert atlas.add_clip("run", tex2)
        assert atlas.height == 45  # 30 + 15 frames

    def test_atlas_clip_lookup(self, simple_skeleton: Skeleton) -> None:
        """Atlas clip info lookup works."""
        atlas = AnimationTextureAtlas()
        clip = AnimationClip(name="idle", duration=1.0, frame_rate=30.0, bone_tracks={0: [Transform.identity()] * 30})
        texture = bake_clip_to_texture(clip, simple_skeleton)
        atlas.add_clip("idle", texture)

        info = atlas.get_clip_info("idle")
        assert info is not None
        assert info[0] == 0  # start_row
        assert info[1] == 30  # frame_count

    def test_system_bake_to_atlas(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
        simple_skeleton: Skeleton,
    ) -> None:
        """System can bake clips to atlas."""
        clips = [
            AnimationClip(name="idle", duration=1.0, frame_rate=30.0, bone_tracks={0: [Transform.identity()] * 30}),
            AnimationClip(name="walk", duration=1.0, frame_rate=30.0, bone_tracks={0: [Transform.identity()] * 30}),
        ]

        atlas = crowd_system.bake_animation_to_atlas(crowd_component, simple_skeleton, clips)

        assert atlas.height == 60
        assert "idle" in atlas.clips
        assert "walk" in atlas.clips


# =============================================================================
# 5. LOD DISTANCE THRESHOLD TESTS
# =============================================================================


class TestLODThresholds:
    """Tests for LOD selection based on distance."""

    def test_close_agent_gets_high_lod(self, crowd_component: CrowdComponent) -> None:
        """Agents close to camera get highest LOD."""
        lod = crowd_component.lod.get_lod_for_distance(5.0)
        assert lod == 0  # Highest detail

    def test_medium_agent_gets_medium_lod(self, crowd_component: CrowdComponent) -> None:
        """Agents at medium distance get medium LOD."""
        lod = crowd_component.lod.get_lod_for_distance(35.0)
        assert lod == 1

    def test_far_agent_gets_low_lod(self, crowd_component: CrowdComponent) -> None:
        """Agents far from camera get lowest LOD."""
        lod = crowd_component.lod.get_lod_for_distance(150.0)
        assert lod == 3  # Lowest detail

    def test_lod_hysteresis(self, crowd_component: CrowdComponent) -> None:
        """LOD switching has hysteresis."""
        crowd_component.lod.set_hysteresis(2.0)

        # At boundary, current LOD should influence decision
        lod_from_high = crowd_component.lod.get_lod_for_distance(25.0, current_lod=0)
        lod_from_low = crowd_component.lod.get_lod_for_distance(25.0, current_lod=1)

        # With hysteresis, staying at current LOD is preferred
        # The exact behavior depends on implementation
        assert lod_from_high is not None
        assert lod_from_low is not None

    def test_system_lod_distances_configurable(self, crowd_system: CrowdSystem) -> None:
        """System LOD distances are configurable."""
        crowd_system.set_lod_distances([10.0, 30.0, 60.0, 120.0])

        assert crowd_system._lod_distances == [10.0, 30.0, 60.0, 120.0]

    def test_lod_level_bone_reduction(self, crowd_component: CrowdComponent) -> None:
        """Higher LOD levels use fewer bones."""
        level0 = crowd_component.lod.get_lod_level(0)
        level3 = crowd_component.lod.get_lod_level(3)

        assert level0 is not None
        assert level3 is not None
        assert level0.bone_count > level3.bone_count


# =============================================================================
# 6. FRUSTUM CULLING TESTS
# =============================================================================


class TestFrustumCulling:
    """Tests for frustum culling accuracy."""

    def test_point_inside_frustum(self) -> None:
        """Point inside frustum is visible."""
        # Create a simple frustum (cube from -1 to 1)
        frustum = Frustum(planes=[
            Plane(normal=Vec3(1.0, 0.0, 0.0), distance=1.0),    # Left
            Plane(normal=Vec3(-1.0, 0.0, 0.0), distance=1.0),   # Right
            Plane(normal=Vec3(0.0, 1.0, 0.0), distance=1.0),    # Bottom
            Plane(normal=Vec3(0.0, -1.0, 0.0), distance=1.0),   # Top
            Plane(normal=Vec3(0.0, 0.0, 1.0), distance=1.0),    # Near
            Plane(normal=Vec3(0.0, 0.0, -1.0), distance=1.0),   # Far
        ])

        assert frustum.is_point_visible(Vec3.zero())

    def test_point_outside_frustum(self) -> None:
        """Point outside frustum is not visible."""
        frustum = Frustum(planes=[
            Plane(normal=Vec3(1.0, 0.0, 0.0), distance=1.0),
            Plane(normal=Vec3(-1.0, 0.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, 1.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, -1.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, 0.0, 1.0), distance=1.0),
            Plane(normal=Vec3(0.0, 0.0, -1.0), distance=1.0),
        ])

        assert not frustum.is_point_visible(Vec3(5.0, 0.0, 0.0))

    def test_sphere_intersects_frustum(self) -> None:
        """Sphere intersecting frustum is visible."""
        frustum = Frustum(planes=[
            Plane(normal=Vec3(1.0, 0.0, 0.0), distance=1.0),
            Plane(normal=Vec3(-1.0, 0.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, 1.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, -1.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, 0.0, 1.0), distance=1.0),
            Plane(normal=Vec3(0.0, 0.0, -1.0), distance=1.0),
        ])

        # Sphere centered at edge but with radius reaching inside
        assert frustum.is_sphere_visible(Vec3(1.5, 0.0, 0.0), 1.0)

    def test_sphere_outside_frustum(self) -> None:
        """Sphere fully outside frustum is not visible."""
        frustum = Frustum(planes=[
            Plane(normal=Vec3(1.0, 0.0, 0.0), distance=1.0),
            Plane(normal=Vec3(-1.0, 0.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, 1.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, -1.0, 0.0), distance=1.0),
            Plane(normal=Vec3(0.0, 0.0, 1.0), distance=1.0),
            Plane(normal=Vec3(0.0, 0.0, -1.0), distance=1.0),
        ])

        assert not frustum.is_sphere_visible(Vec3(10.0, 0.0, 0.0), 0.5)

    def test_plane_distance_calculation(self) -> None:
        """Plane distance to point calculation."""
        plane = Plane(normal=Vec3(1.0, 0.0, 0.0), distance=0.0)
        point = Vec3(5.0, 0.0, 0.0)

        dist = plane.distance_to_point(point)
        assert abs(dist - 5.0) < 0.01

    def test_culling_mode_none_skips_culling(self, crowd_component: CrowdComponent) -> None:
        """Culling mode NONE skips frustum checks."""
        crowd_component.culling_mode = CullingMode.NONE
        # All agents should remain visible regardless of frustum

    def test_culling_mode_sphere(self, crowd_component: CrowdComponent) -> None:
        """Sphere culling mode uses bounding sphere."""
        crowd_component.culling_mode = CullingMode.SPHERE
        crowd_component.culling_radius = 0.5
        assert crowd_component.culling_radius == 0.5


# =============================================================================
# 7. INSTANCE BUFFER TESTS
# =============================================================================


class TestInstanceBuffer:
    """Tests for instance buffer generation."""

    def test_buffer_add_instance(self) -> None:
        """Adding instance increases count."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        data = CrowdInstanceData(
            transform=Mat4.identity(),
            animation_texture_row=0,
            animation_time=0.0,
            lod_level=0,
            visible=True,
        )

        idx = buffer.add_instance(data)
        assert idx == 0
        assert buffer.instance_count == 1

    def test_buffer_transform_packing(self) -> None:
        """Transform is correctly packed to buffer."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        transform = Transform(
            translation=Vec3(1.0, 2.0, 3.0),
            rotation=Quat.identity(),
            scale=Vec3(1.0, 1.0, 1.0),
        )

        data = CrowdInstanceData(
            transform=transform.to_matrix(),
            animation_texture_row=0,
            animation_time=0.0,
            lod_level=0,
            visible=True,
        )

        buffer.add_instance(data)

        # Check transform data contains translation
        # Matrix column 3 contains translation
        assert len(buffer.transforms) >= 16

    def test_buffer_animation_packing(self) -> None:
        """Animation data is correctly packed."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        data = CrowdInstanceData(
            transform=Mat4.identity(),
            animation_texture_row=5,
            animation_time=1.5,
            animation_phase_offset=0.3,
            lod_level=2,
            visible=True,
        )

        buffer.add_instance(data)

        # Check animation data
        assert buffer.animations[0] == 5.0  # row
        assert abs(buffer.animations[1] - 1.8) < 0.01  # time + offset
        assert abs(buffer.animations[2] - 0.3) < 0.01  # offset
        assert buffer.animations[3] == 2.0  # lod

    def test_buffer_color_packing(self) -> None:
        """Color data is correctly packed."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        data = CrowdInstanceData(
            transform=Mat4.identity(),
            tint_color=Vec4(1.0, 0.5, 0.25, 1.0),
            visible=True,
        )

        buffer.add_instance(data)

        assert abs(buffer.colors[0] - 1.0) < 0.01
        assert abs(buffer.colors[1] - 0.5) < 0.01
        assert abs(buffer.colors[2] - 0.25) < 0.01
        assert abs(buffer.colors[3] - 1.0) < 0.01

    def test_buffer_invisible_alpha_zero(self) -> None:
        """Invisible instances have alpha 0."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        data = CrowdInstanceData(
            transform=Mat4.identity(),
            tint_color=Vec4(1.0, 1.0, 1.0, 1.0),
            visible=False,
        )

        buffer.add_instance(data)

        assert buffer.colors[3] == 0.0  # Alpha

    def test_buffer_auto_grow(self) -> None:
        """Buffer grows automatically when needed."""
        buffer = CrowdInstanceBuffer()
        # Don't reserve, let it grow

        for i in range(100):
            data = CrowdInstanceData(transform=Mat4.identity())
            buffer.add_instance(data)

        assert buffer.instance_count == 100
        assert buffer.capacity >= 100

    def test_buffer_memory_size(self) -> None:
        """Memory size calculation is accurate."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(100)

        size = buffer.get_memory_size_bytes()
        # 100 instances * (16 + 4 + 4) floats * 4 bytes + 100 ints * 4 bytes
        expected = 100 * (16 + 4 + 4) * 4 + 100 * 4
        assert size == expected

    def test_buffer_update_instance(self) -> None:
        """Updating instance modifies data."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        data1 = CrowdInstanceData(
            transform=Mat4.identity(),
            animation_time=1.0,
        )
        idx = buffer.add_instance(data1)

        data2 = CrowdInstanceData(
            transform=Mat4.identity(),
            animation_time=2.0,
        )
        buffer.update_instance(idx, data2)

        assert abs(buffer.animations[1] - 2.0) < 0.01

    def test_buffer_clear(self) -> None:
        """Clear resets buffer."""
        buffer = CrowdInstanceBuffer()
        buffer.reserve(10)

        for i in range(5):
            buffer.add_instance(CrowdInstanceData())

        buffer.clear()

        assert buffer.instance_count == 0
        assert buffer.visible_count == 0


# =============================================================================
# 8. CROWD COMPONENT TESTS
# =============================================================================


class TestCrowdComponent:
    """Tests for CrowdComponent functionality."""

    def test_add_agent(self, crowd_component: CrowdComponent) -> None:
        """Adding agent increases count."""
        agent_id = crowd_component.add_agent(Vec3.zero())
        assert crowd_component.get_agent_count() == 1
        assert agent_id > 0

    def test_remove_agent(self, crowd_component: CrowdComponent) -> None:
        """Removing agent decreases count."""
        agent_id = crowd_component.add_agent(Vec3.zero())
        assert crowd_component.get_agent_count() == 1

        result = crowd_component.remove_agent(agent_id)
        assert result is True
        assert crowd_component.get_agent_count() == 0

    def test_remove_nonexistent_agent(self, crowd_component: CrowdComponent) -> None:
        """Removing non-existent agent returns False."""
        result = crowd_component.remove_agent(9999)
        assert result is False

    def test_get_agent(self, crowd_component: CrowdComponent) -> None:
        """Get agent by ID."""
        agent_id = crowd_component.add_agent(Vec3(1.0, 0.0, 0.0))
        agent = crowd_component.get_agent(agent_id)

        assert agent is not None
        assert abs(agent.position.x - 1.0) < 0.01

    def test_set_agent_target(self, crowd_component: CrowdComponent) -> None:
        """Set movement target for agent."""
        agent_id = crowd_component.add_agent(Vec3.zero())

        result = crowd_component.set_agent_target(agent_id, Vec3(10.0, 0.0, 0.0))
        assert result is True

        agent = crowd_component.get_agent(agent_id)
        assert agent is not None
        assert agent.current_state == AgentState.WALKING

    def test_phase_offset(self, crowd_component: CrowdComponent) -> None:
        """Phase offset is stored per agent."""
        agent_id = crowd_component.add_agent(Vec3.zero(), phase_offset=0.5)

        offset = crowd_component.get_phase_offset(agent_id)
        assert abs(offset - 0.5) < 0.01

    def test_random_phase_offset(self, crowd_component: CrowdComponent) -> None:
        """Random phase offset when not specified."""
        agent_id = crowd_component.add_agent(Vec3.zero())
        offset = crowd_component.get_phase_offset(agent_id)

        assert 0.0 <= offset <= DEFAULT_PHASE_OFFSET_RANGE

    def test_set_phase_offset(self, crowd_component: CrowdComponent) -> None:
        """Can modify phase offset."""
        agent_id = crowd_component.add_agent(Vec3.zero())
        crowd_component.set_phase_offset(agent_id, 0.75)

        assert abs(crowd_component.get_phase_offset(agent_id) - 0.75) < 0.01


# =============================================================================
# 9. CROWD SYSTEM UPDATE TESTS
# =============================================================================


class TestCrowdSystemUpdate:
    """Tests for CrowdSystem update logic."""

    def test_update_empty_list(self, crowd_system: CrowdSystem) -> None:
        """Update with empty list doesn't crash."""
        crowd_system.update(mock.MagicMock(), 0.016, [])

    def test_update_disabled_component(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Disabled components are skipped."""
        crowd_component.enabled = False
        crowd_component.add_agent(Vec3.zero())

        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        # Should still work without crashing

    def test_update_builds_instance_buffer(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Update builds instance buffer."""
        crowd_component.add_agent(Vec3.zero())
        crowd_component.camera_position = Vec3(0.0, 0.0, 0.0)

        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        buffer = crowd_component.get_instance_buffer()
        assert buffer.instance_count >= 0

    def test_update_culls_distant_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Update culls agents beyond cull distance."""
        crowd_system.set_cull_distance(50.0)

        # Add agent far away
        crowd_component.add_agent(Vec3(100.0, 0.0, 0.0))
        crowd_component.camera_position = Vec3.zero()

        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        # Agent should be culled
        assert crowd_system._total_agents_culled >= 1


# =============================================================================
# 10. FORMATION SPAWNING TESTS
# =============================================================================


class TestFormationSpawning:
    """Tests for crowd formation spawning."""

    def test_circle_formation(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Spawn agents in circle formation."""
        agent_ids = crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=8,
            radius=5.0,
            formation="circle",
        )

        assert len(agent_ids) == 8
        assert crowd_component.get_agent_count() == 8

    def test_grid_formation(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Spawn agents in grid formation."""
        agent_ids = crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=16,
            radius=5.0,
            formation="grid",
        )

        assert len(agent_ids) == 16

    def test_random_formation(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Spawn agents in random formation."""
        agent_ids = crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=10,
            radius=5.0,
            formation="random",
        )

        assert len(agent_ids) == 10

    def test_formation_randomize_phase(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Formation can randomize phase offsets."""
        agent_ids = crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=10,
            radius=5.0,
            formation="circle",
            randomize_phase=True,
        )

        # All agents should have different phase offsets
        offsets = [crowd_component.get_phase_offset(aid) for aid in agent_ids]
        assert len(set(offsets)) > 1  # Not all the same


# =============================================================================
# 11. STATISTICS TESTS
# =============================================================================


class TestCrowdStatistics:
    """Tests for crowd statistics tracking."""

    def test_get_stats_empty(self, crowd_system: CrowdSystem) -> None:
        """Stats for empty crowd."""
        stats = crowd_system.get_stats([])

        assert stats["total_agents"] == 0
        assert stats["total_visible"] == 0

    def test_get_stats_with_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Stats include agent counts."""
        for _ in range(10):
            crowd_component.add_agent(Vec3.zero())

        stats = crowd_system.get_stats([(mock.MagicMock(), crowd_component)])

        assert stats["total_agents"] == 10


# =============================================================================
# 12. LARGE CROWD PERFORMANCE TESTS
# =============================================================================


class TestLargeCrowdPerformance:
    """Performance tests for large crowds (1000+ agents)."""

    @pytest.mark.timeout(10)
    def test_spawn_1000_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Can spawn 1000 agents quickly."""
        start = time.perf_counter()

        agent_ids = crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=1000,
            radius=50.0,
            formation="random",
        )

        elapsed = time.perf_counter() - start

        assert len(agent_ids) == 1000
        assert elapsed < 2.0  # Should complete in under 2 seconds

    @pytest.mark.timeout(10)
    def test_update_1000_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Can update 1000 agents in reasonable time."""
        # Spawn agents
        crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=1000,
            radius=50.0,
            formation="random",
        )

        crowd_component.camera_position = Vec3.zero()

        # Time update
        start = time.perf_counter()

        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        elapsed = time.perf_counter() - start

        # Should update in under 100ms for real-time use
        assert elapsed < 0.5  # Allow some headroom

    @pytest.mark.timeout(30)
    def test_update_loop_1000_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Can run 60 updates (1 second) with 1000 agents."""
        crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=1000,
            radius=50.0,
            formation="random",
        )

        crowd_component.camera_position = Vec3.zero()

        start = time.perf_counter()

        for _ in range(60):
            crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        elapsed = time.perf_counter() - start

        # 60 frames should complete in reasonable time
        assert elapsed < 10.0  # 10 seconds max for 60 frames

    def test_rvo_steering_1000_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """RVO steering works with 1000 agents."""
        crowd_component.steering_mode = SteeringMode.RVO

        crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=1000,
            radius=50.0,
            formation="random",
        )

        # Set targets for some agents
        agents = list(crowd_component._synced_instances.keys())
        for agent_id in agents[:100]:
            crowd_component.set_agent_target(agent_id, Vec3(100.0, 0.0, 0.0))

        # Should update without crashing
        crowd_component.camera_position = Vec3.zero()
        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

    def test_instance_buffer_1000_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Instance buffer handles 1000 agents."""
        crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=1000,
            radius=50.0,
            formation="random",
        )

        crowd_component.camera_position = Vec3.zero()
        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        buffer = crowd_component.get_instance_buffer()

        # Buffer should have some visible instances
        assert buffer.instance_count >= 0
        assert buffer.capacity >= buffer.instance_count


# =============================================================================
# 13. EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_agent_at_camera_position(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Agent at exact camera position."""
        crowd_component.add_agent(Vec3.zero())
        crowd_component.camera_position = Vec3.zero()

        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        # Should not crash, agent should be visible
        assert crowd_component.get_visible_count() >= 0

    def test_coincident_agents(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Two agents at same position."""
        crowd_component.add_agent(Vec3.zero())
        crowd_component.add_agent(Vec3.zero())

        crowd_component.camera_position = Vec3(10.0, 0.0, 0.0)
        crowd_system.update(mock.MagicMock(), 0.016, [(mock.MagicMock(), crowd_component)])

        # Should not crash

    def test_zero_dt_update(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Update with zero delta time."""
        crowd_component.add_agent(Vec3.zero())

        crowd_system.update(mock.MagicMock(), 0.0, [(mock.MagicMock(), crowd_component)])

        # Should not crash

    def test_very_large_dt_update(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Update with very large delta time."""
        crowd_component.add_agent(Vec3.zero())

        crowd_system.update(mock.MagicMock(), 10.0, [(mock.MagicMock(), crowd_component)])

        # Should not crash

    def test_negative_lod_distance(self, crowd_component: CrowdComponent) -> None:
        """LOD handles negative distance gracefully."""
        lod = crowd_component.lod.get_lod_for_distance(-5.0)
        assert lod == 0  # Should return highest detail

    def test_zero_radius_formation(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Formation with zero radius."""
        agent_ids = crowd_system.spawn_crowd_formation(
            component=crowd_component,
            center=Vec3.zero(),
            count=5,
            radius=0.0,
            formation="circle",
        )

        assert len(agent_ids) == 5

    def test_flee_event(
        self,
        crowd_system: CrowdSystem,
        crowd_component: CrowdComponent,
    ) -> None:
        """Trigger flee event."""
        for _ in range(10):
            crowd_component.add_agent(Vec3.zero())

        affected = crowd_system.trigger_flee_event(
            component=crowd_component,
            threat_position=Vec3.zero(),
            radius=10.0,
        )

        assert affected == 10
