"""
Whitebox tests for engine/simulation/character/physics_animation_blend.py

Tests PhysicsAnimationBlender, BonePose, SkeletonPose, and blend operations.
"""

import math
import pytest
from engine.simulation.character.physics_animation_blend import (
    BlendLayer,
    BonePose,
    HitReaction,
    PhysicsAnimationBlender,
    SkeletonPose,
)
from engine.simulation.character.config import BlendMode
from engine.simulation.character.character_controller import (
    Quaternion,
    Vector3,
)


class TestBonePose:
    """Tests for BonePose dataclass."""

    def test_default_construction(self):
        """Default BonePose should have identity values."""
        pose = BonePose()
        assert pose.position.x == 0.0
        assert pose.position.y == 0.0
        assert pose.position.z == 0.0
        assert pose.rotation.w == 1.0
        assert pose.scale.x == 1.0
        assert pose.scale.y == 1.0
        assert pose.scale.z == 1.0

    def test_custom_construction(self):
        """BonePose should accept custom values."""
        pose = BonePose(
            position=Vector3(1.0, 2.0, 3.0),
            rotation=Quaternion(0.0, 0.707, 0.0, 0.707),
            scale=Vector3(2.0, 2.0, 2.0),
        )
        assert pose.position.x == 1.0
        assert pose.rotation.y == 0.707
        assert pose.scale.x == 2.0

    def test_lerp_at_zero(self):
        """lerp at t=0 should return first pose."""
        a = BonePose(position=Vector3(0.0, 0.0, 0.0))
        b = BonePose(position=Vector3(10.0, 10.0, 10.0))
        result = a.lerp(b, 0.0)
        assert result.position.x == pytest.approx(0.0)
        assert result.position.y == pytest.approx(0.0)

    def test_lerp_at_one(self):
        """lerp at t=1 should return second pose."""
        a = BonePose(position=Vector3(0.0, 0.0, 0.0))
        b = BonePose(position=Vector3(10.0, 10.0, 10.0))
        result = a.lerp(b, 1.0)
        assert result.position.x == pytest.approx(10.0)

    def test_lerp_midpoint(self):
        """lerp at t=0.5 should return midpoint."""
        a = BonePose(position=Vector3(0.0, 0.0, 0.0))
        b = BonePose(position=Vector3(10.0, 10.0, 10.0))
        result = a.lerp(b, 0.5)
        assert result.position.x == pytest.approx(5.0)

    def test_slerp_identity(self):
        """slerp between identical quaternions should return same."""
        pose = BonePose()
        a = Quaternion.identity()
        b = Quaternion.identity()
        result = pose._slerp(a, b, 0.5)
        assert result.w == pytest.approx(1.0, abs=0.01)

    def test_slerp_interpolates(self):
        """slerp should interpolate rotations."""
        pose = BonePose()
        a = Quaternion.identity()
        b = Quaternion(0.0, 0.707, 0.0, 0.707)  # 90 deg Y
        result = pose._slerp(a, b, 0.5)
        # Should be ~45 degree rotation
        assert 0 < result.y < 0.707


class TestSkeletonPose:
    """Tests for SkeletonPose dataclass."""

    def test_default_construction(self):
        """Default SkeletonPose should have empty bones."""
        pose = SkeletonPose()
        assert len(pose.bones) == 0

    def test_add_bones(self):
        """SkeletonPose should store bones."""
        pose = SkeletonPose()
        pose.bones["Head"] = BonePose(position=Vector3(0.0, 1.8, 0.0))
        pose.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        assert len(pose.bones) == 2
        assert pose.bones["Head"].position.y == 1.8

    def test_root_motion(self):
        """SkeletonPose should store root motion."""
        pose = SkeletonPose()
        pose.root_motion = Vector3(1.0, 0.0, 0.0)
        assert pose.root_motion.x == 1.0


class TestBlendLayer:
    """Tests for BlendLayer dataclass."""

    def test_default_construction(self):
        """Default BlendLayer should have default values."""
        layer = BlendLayer()
        assert layer.name == ""
        assert layer.mode == BlendMode.POSE
        assert layer.weight == 1.0
        assert layer.mask is None
        assert layer.source is None
        assert layer.enabled is True

    def test_custom_construction(self):
        """BlendLayer should accept custom values."""
        layer = BlendLayer(
            name="physics",
            mode=BlendMode.ADDITIVE,
            weight=0.5,
            enabled=False,
        )
        assert layer.name == "physics"
        assert layer.mode == BlendMode.ADDITIVE
        assert layer.weight == 0.5
        assert layer.enabled is False


class TestHitReaction:
    """Tests for HitReaction dataclass."""

    def test_default_construction(self):
        """Default HitReaction should have zero values."""
        reaction = HitReaction()
        assert reaction.hit_force == 0.0
        assert reaction.blend_weight == 0.0
        assert len(reaction.affected_bones) == 0

    def test_custom_construction(self):
        """HitReaction should accept custom values."""
        reaction = HitReaction(
            hit_point=Vector3(0.0, 1.5, 0.0),
            hit_direction=Vector3(1.0, 0.0, 0.0),
            hit_force=100.0,
            affected_bones=["Spine", "Chest"],
            start_time=0.0,
        )
        assert reaction.hit_force == 100.0
        assert len(reaction.affected_bones) == 2


class TestPhysicsAnimationBlender:
    """Tests for PhysicsAnimationBlender class."""

    def test_construction(self):
        """PhysicsAnimationBlender should be constructible."""
        blender = PhysicsAnimationBlender()
        assert blender is not None

    def test_add_layer(self):
        """add_layer should add a blend layer."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics", BlendMode.POSE, 1.0)
        assert len(blender._layers) == 1
        assert blender._layers[0].name == "physics"

    def test_add_layer_with_mask(self):
        """add_layer should accept mask."""
        blender = PhysicsAnimationBlender()
        mask = {"Spine": 1.0, "Head": 0.5}
        blender.add_layer("physics", mask=mask)
        assert blender._layers[0].mask == mask

    def test_remove_layer(self):
        """remove_layer should remove layer by name."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics")
        result = blender.remove_layer("physics")
        assert result is True
        assert len(blender._layers) == 0

    def test_remove_layer_not_found(self):
        """remove_layer should return False if not found."""
        blender = PhysicsAnimationBlender()
        result = blender.remove_layer("nonexistent")
        assert result is False

    def test_set_layer_weight(self):
        """set_layer_weight should update layer weight."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics")
        blender.set_layer_weight("physics", 0.5)
        assert blender._layers[0].weight == 0.5

    def test_set_layer_weight_clamped(self):
        """set_layer_weight should clamp to 0-1."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics")
        blender.set_layer_weight("physics", 2.0)
        assert blender._layers[0].weight == 1.0
        blender.set_layer_weight("physics", -1.0)
        assert blender._layers[0].weight == 0.0

    def test_set_layer_enabled(self):
        """set_layer_enabled should update layer enabled state."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics")
        blender.set_layer_enabled("physics", False)
        assert blender._layers[0].enabled is False

    def test_set_layer_pose(self):
        """set_layer_pose should update layer source."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics")
        pose = SkeletonPose()
        blender.set_layer_pose("physics", pose)
        assert blender._layers[0].source is pose

    def test_per_limb_blend_weights_property(self):
        """per_limb_blend_weights should return copy of weights."""
        blender = PhysicsAnimationBlender()
        blender.set_bone_weight("Head", 0.5)
        weights = blender.per_limb_blend_weights
        assert weights["Head"] == 0.5
        # Modify should not affect original
        weights["Head"] = 1.0
        assert blender._limb_weights["Head"] == 0.5

    def test_set_bone_weight(self):
        """set_bone_weight should update bone weight."""
        blender = PhysicsAnimationBlender()
        blender.set_bone_weight("Spine", 0.7)
        assert blender._limb_weights["Spine"] == 0.7

    def test_set_bone_weight_clamped(self):
        """set_bone_weight should clamp to 0-1."""
        blender = PhysicsAnimationBlender()
        blender.set_bone_weight("Spine", 2.0)
        assert blender._limb_weights["Spine"] == 1.0

    def test_set_limb_weight(self):
        """set_limb_weight should update multiple bones."""
        blender = PhysicsAnimationBlender()
        blender.set_limb_weight(["Arm_L", "Hand_L"], 0.5)
        assert blender._limb_weights["Arm_L"] == 0.5
        assert blender._limb_weights["Hand_L"] == 0.5

    def test_clear_bone_weights(self):
        """clear_bone_weights should remove all weights."""
        blender = PhysicsAnimationBlender()
        blender.set_bone_weight("Spine", 0.5)
        blender.clear_bone_weights()
        assert len(blender._limb_weights) == 0

    def test_get_bone_weight_existing(self):
        """get_bone_weight should return existing weight."""
        blender = PhysicsAnimationBlender()
        blender.set_bone_weight("Spine", 0.3)
        assert blender.get_bone_weight("Spine") == 0.3

    def test_get_bone_weight_default(self):
        """get_bone_weight should return default for unknown bones."""
        blender = PhysicsAnimationBlender()
        weight = blender.get_bone_weight("Unknown")
        assert weight == blender._default_weight


class TestBoneHierarchy:
    """Tests for bone hierarchy management."""

    def test_set_bone_hierarchy(self):
        """set_bone_hierarchy should store hierarchy."""
        blender = PhysicsAnimationBlender()
        hierarchy = {
            "Spine": ["Spine1", "Shoulder_L", "Shoulder_R"],
            "Spine1": ["Neck"],
            "Neck": ["Head"],
        }
        parents = {
            "Head": "Neck",
            "Neck": "Spine1",
            "Spine1": "Spine",
        }
        blender.set_bone_hierarchy(hierarchy, parents)
        assert "Spine" in blender._bone_hierarchy
        assert blender._bone_parent["Head"] == "Neck"

    def test_get_bone_chain(self):
        """get_bone_chain should return chain of bones."""
        blender = PhysicsAnimationBlender()
        blender._bone_parent = {
            "Head": "Neck",
            "Neck": "Spine1",
            "Spine1": "Spine",
        }
        chain = blender.get_bone_chain("Spine", "Head")
        assert "Spine" in chain
        assert "Head" in chain
        assert chain[0] == "Spine"
        assert chain[-1] == "Head"


class TestPoseBlending:
    """Tests for pose blending operations."""

    def test_blend_poses_at_zero(self):
        """blend_poses at weight 0 should return anim pose."""
        blender = PhysicsAnimationBlender()
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        result = blender.blend_poses(anim, physics, 0.0)
        assert result.bones["Spine"].position.y == pytest.approx(1.0)

    def test_blend_poses_at_one(self):
        """blend_poses at weight 1 should return physics pose."""
        blender = PhysicsAnimationBlender()
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        result = blender.blend_poses(anim, physics, 1.0)
        assert result.bones["Spine"].position.y == pytest.approx(2.0)

    def test_blend_poses_midpoint(self):
        """blend_poses at weight 0.5 should interpolate."""
        blender = PhysicsAnimationBlender()
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 0.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        result = blender.blend_poses(anim, physics, 0.5)
        assert result.bones["Spine"].position.y == pytest.approx(1.0)

    def test_blend_poses_per_bone_weight(self):
        """blend_poses should respect per-bone weights."""
        blender = PhysicsAnimationBlender()
        blender.set_bone_weight("Spine", 0.0)  # No physics on spine
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        result = blender.blend_poses(anim, physics, 1.0)
        # Despite weight=1, per-bone weight=0 means no physics
        assert result.bones["Spine"].position.y == pytest.approx(1.0)

    def test_blend_poses_combines_all_bones(self):
        """blend_poses should include bones from both poses."""
        blender = PhysicsAnimationBlender()
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose()
        physics = SkeletonPose()
        physics.bones["Head"] = BonePose()
        result = blender.blend_poses(anim, physics, 0.5)
        assert "Spine" in result.bones
        assert "Head" in result.bones

    def test_blend_poses_root_motion(self):
        """blend_poses should blend root motion."""
        blender = PhysicsAnimationBlender()
        anim = SkeletonPose()
        anim.root_motion = Vector3(1.0, 0.0, 0.0)
        physics = SkeletonPose()
        physics.root_motion = Vector3(3.0, 0.0, 0.0)
        result = blender.blend_poses(anim, physics, 0.5)
        assert result.root_motion.x == pytest.approx(2.0)


class TestAdditiveBlending:
    """Tests for additive physics blending."""

    def test_additive_physics_zero_weight(self):
        """additive_physics at weight 0 should return base."""
        blender = PhysicsAnimationBlender()
        base = SkeletonPose()
        base.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        delta = SkeletonPose()
        delta.bones["Spine"] = BonePose(position=Vector3(0.0, 0.5, 0.0))
        result = blender.additive_physics(base, delta, 0.0)
        assert result.bones["Spine"].position.y == pytest.approx(1.0)

    def test_additive_physics_full_weight(self):
        """additive_physics at weight 1 should add full delta."""
        blender = PhysicsAnimationBlender()
        base = SkeletonPose()
        base.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        delta = SkeletonPose()
        delta.bones["Spine"] = BonePose(position=Vector3(0.0, 0.5, 0.0))
        result = blender.additive_physics(base, delta, 1.0)
        assert result.bones["Spine"].position.y == pytest.approx(1.5)

    def test_additive_physics_partial_weight(self):
        """additive_physics should scale delta by weight."""
        blender = PhysicsAnimationBlender()
        base = SkeletonPose()
        base.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        delta = SkeletonPose()
        delta.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        result = blender.additive_physics(base, delta, 0.5)
        assert result.bones["Spine"].position.y == pytest.approx(1.5)


class TestChainBlending:
    """Tests for chain blending."""

    def test_blend_chain(self):
        """blend_chain should blend specified bone chain."""
        blender = PhysicsAnimationBlender()
        blender._bone_hierarchy = {
            "Spine": ["Head"],
            "Head": [],
        }
        base = SkeletonPose()
        base.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        base.bones["Head"] = BonePose(position=Vector3(0.0, 1.8, 0.0))
        base.bones["Leg"] = BonePose(position=Vector3(0.0, 0.5, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 1.1, 0.0))
        physics.bones["Head"] = BonePose(position=Vector3(0.0, 1.9, 0.0))
        physics.bones["Leg"] = BonePose(position=Vector3(0.0, 0.6, 0.0))
        result = blender.blend_chain(base, physics, "Spine", 1.0)
        # Spine and Head should be affected, Leg should not
        assert result.bones["Spine"].position.y == pytest.approx(1.1)
        assert result.bones["Head"].position.y == pytest.approx(1.9)
        assert result.bones["Leg"].position.y == pytest.approx(0.5)


class TestHitReactions:
    """Tests for hit reaction system."""

    def test_add_hit_reaction(self):
        """add_hit_reaction should add reaction."""
        blender = PhysicsAnimationBlender()
        blender.add_hit_reaction(
            hit_point=Vector3(0.0, 1.5, 0.0),
            hit_direction=Vector3(1.0, 0.0, 0.0),
            hit_force=100.0,
            affected_bones=["Spine", "Chest"],
            current_time=0.0,
        )
        assert len(blender._hit_reactions) == 1

    def test_add_hit_reaction_max_limit(self):
        """add_hit_reaction should remove oldest when at limit."""
        blender = PhysicsAnimationBlender()
        blender._max_hit_reactions = 2
        for i in range(3):
            blender.add_hit_reaction(
                hit_point=Vector3.zero(),
                hit_direction=Vector3.zero(),
                hit_force=float(i),
                affected_bones=[],
                current_time=float(i),
            )
        assert len(blender._hit_reactions) == 2
        # First should be removed
        assert blender._hit_reactions[0].hit_force == 1.0

    def test_clear_hit_reactions(self):
        """clear_hit_reactions should remove all reactions."""
        blender = PhysicsAnimationBlender()
        blender.add_hit_reaction(
            Vector3.zero(), Vector3.zero(), 100.0, [], 0.0
        )
        blender.clear_hit_reactions()
        assert len(blender._hit_reactions) == 0

    def test_update_hit_reactions_blend_in(self):
        """update_hit_reactions should blend in reactions."""
        blender = PhysicsAnimationBlender()
        blender.add_hit_reaction(
            hit_point=Vector3(0.0, 1.0, 0.0),
            hit_direction=Vector3(1.0, 0.0, 0.0),
            hit_force=100.0,
            affected_bones=["Spine"],
            current_time=0.0,
        )
        pose = SkeletonPose()
        pose.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        result = blender.update_hit_reactions(pose, 0.025, 0.016)  # 25ms into reaction
        # Should have partial blend
        assert blender._hit_reactions[0].blend_weight > 0

    def test_update_hit_reactions_removes_expired(self):
        """update_hit_reactions should remove expired reactions."""
        blender = PhysicsAnimationBlender()
        blender.add_hit_reaction(
            hit_point=Vector3.zero(),
            hit_direction=Vector3.zero(),
            hit_force=100.0,
            affected_bones=[],
            current_time=0.0,
        )
        pose = SkeletonPose()
        # Update well past reaction duration (350ms = 50ms in + 300ms out)
        blender.update_hit_reactions(pose, 1.0, 0.016)  # 1000ms later
        assert len(blender._hit_reactions) == 0


class TestProcessPipeline:
    """Tests for full process pipeline."""

    def test_process_no_layers(self):
        """process with no layers should return anim pose."""
        blender = PhysicsAnimationBlender()
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        result = blender.process(anim, None, 0.0, 0.016)
        assert result.bones["Spine"].position.y == 1.0

    def test_process_with_pose_layer(self):
        """process with POSE layer should blend."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics", BlendMode.POSE, 0.5)
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        blender.set_layer_pose("physics", physics)
        result = blender.process(anim, None, 0.0, 0.016)
        assert result.bones["Spine"].position.y == pytest.approx(1.5)

    def test_process_with_additive_layer(self):
        """process with ADDITIVE layer should add."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics", BlendMode.ADDITIVE, 1.0)
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        delta = SkeletonPose()
        delta.bones["Spine"] = BonePose(position=Vector3(0.0, 0.5, 0.0))
        blender.set_layer_pose("physics", delta)
        result = blender.process(anim, None, 0.0, 0.016)
        assert result.bones["Spine"].position.y == pytest.approx(1.5)

    def test_process_disabled_layer_ignored(self):
        """process should ignore disabled layers."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics", BlendMode.POSE, 1.0)
        blender.set_layer_enabled("physics", False)
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        blender.set_layer_pose("physics", physics)
        result = blender.process(anim, None, 0.0, 0.016)
        assert result.bones["Spine"].position.y == pytest.approx(1.0)

    def test_process_zero_weight_layer_ignored(self):
        """process should ignore zero weight layers."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics", BlendMode.POSE, 0.0)
        anim = SkeletonPose()
        anim.bones["Spine"] = BonePose(position=Vector3(0.0, 1.0, 0.0))
        physics = SkeletonPose()
        physics.bones["Spine"] = BonePose(position=Vector3(0.0, 2.0, 0.0))
        blender.set_layer_pose("physics", physics)
        result = blender.process(anim, None, 0.0, 0.016)
        assert result.bones["Spine"].position.y == pytest.approx(1.0)


class TestDebugInfo:
    """Tests for debug info."""

    def test_get_debug_info(self):
        """get_debug_info should return debug dictionary."""
        blender = PhysicsAnimationBlender()
        blender.add_layer("physics")
        blender.set_bone_weight("Spine", 0.5)
        blender.add_hit_reaction(
            Vector3.zero(), Vector3.zero(), 100.0, [], 0.0
        )
        info = blender.get_debug_info()
        assert "layer_count" in info
        assert info["layer_count"] == 1
        assert "layers" in info
        assert "limb_weights" in info
        assert "hit_reaction_count" in info
        assert info["hit_reaction_count"] == 1
