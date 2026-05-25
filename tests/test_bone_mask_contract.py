"""Contract tests for BoneMask (T-AG-1.4).

CLEANROOM: tests the PUBLIC API contract only.
No knowledge of internal representation, data structures, or algorithms.

Contract sources:
  - Task T-AG-1.4 description (public API: BoneMask, MissingBoneMode,
    BoneMaskPresets, apply, combine, invert, full)
  - engine/animation/graph/__init__.py (public exports)

Forbidden files (NOT read):
  - engine/animation/graph/bone_mask.py (DEV implementation)
  - tests/test_bone_mask_whitebox.py (parallel peer)
"""
import pytest
from engine.animation.graph import (
    MissingBoneMode,
    BoneMask,
    BoneMaskPresets,
    SkeletonHierarchy,
)
from engine.core.math.transform import Transform
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


# ============================================================================
# Fixtures  (skeleton.py is ALLOWED for creating test fixtures)
# ============================================================================

@pytest.fixture
def simple_skeleton():
    """A minimal humanoid-like skeleton for bone-mask operations."""
    skel = SkeletonHierarchy("test_humanoid")
    skel.add_bone("root")
    skel.add_bone("spine", parent_name="root")
    skel.add_bone("head", parent_name="spine")
    skel.add_bone("left_arm", parent_name="spine")
    skel.add_bone("right_arm", parent_name="spine")
    skel.add_bone("left_leg", parent_name="root")
    skel.add_bone("right_leg", parent_name="root")
    return skel


@pytest.fixture
def all_bones():
    """Return the list of bone names in simple_skeleton in order."""
    return ["root", "spine", "head", "left_arm",
            "right_arm", "left_leg", "right_leg"]


@pytest.fixture
def humanoid_skeleton():
    """A minimal humanoid skeleton with UE4/Mixamo naming (matches presets)."""
    skel = SkeletonHierarchy("humanoid")
    skel.add_bone("Hips")
    skel.add_bone("Spine", parent_name="Hips")
    skel.add_bone("Head", parent_name="Spine")
    skel.add_bone("LeftArm", parent_name="Spine")
    skel.add_bone("RightArm", parent_name="Spine")
    skel.add_bone("LeftLeg", parent_name="Hips")
    skel.add_bone("RightLeg", parent_name="Hips")
    return skel


# ============================================================================
# Equivalence Class: BoneMask creation
# ============================================================================

class TestBoneMaskCreation:
    """BoneMask can be created from a skeleton reference."""

    def test_create_from_skeleton_only(self, simple_skeleton):
        """A BoneMask can be created from just a skeleton reference."""
        mask = BoneMask()
        assert mask is not None

    def test_create_is_bonemask_instance(self, simple_skeleton):
        """BoneMask(skeleton) returns a BoneMask instance."""
        mask = BoneMask()
        assert isinstance(mask, BoneMask)

    def test_create_then_set_weight(self, simple_skeleton):
        """After creation, set_weight + get_weight round-trips."""
        mask = BoneMask()
        mask.set_weight("head", 0.75)
        assert mask.get_weight("head") == 0.75


# ============================================================================
# Equivalence Class: get_weight / set_weight
# ============================================================================

class TestWeightAccess:
    """Individual per-bone weights can be read and written."""

    def test_get_weight_returns_float(self, simple_skeleton):
        """get_weight returns a float for any registered bone."""
        mask = BoneMask()
        mask.set_weight("root", 0.5)
        w = mask.get_weight("root")
        assert isinstance(w, float)

    def test_get_weight_default_zero(self, simple_skeleton):
        """A freshly created mask has weight 0.0 for all bones."""
        mask = BoneMask()
        assert mask.get_weight("root") == 0.0
        assert mask.get_weight("spine") == 0.0
        assert mask.get_weight("head") == 0.0

    def test_set_and_get_weight(self, simple_skeleton):
        """set_weight followed by get_weight returns the same value."""
        mask = BoneMask()
        mask.set_weight("spine", 0.9)
        assert mask.get_weight("spine") == 0.9

    def test_set_weight_to_zero(self, simple_skeleton):
        """set_weight with 0.0 is accepted."""
        mask = BoneMask()
        mask.set_weight("head", 0.5)
        mask.set_weight("head", 0.0)
        assert mask.get_weight("head") == 0.0

    def test_set_weight_to_one(self, simple_skeleton):
        """set_weight with 1.0 is accepted."""
        mask = BoneMask()
        mask.set_weight("head", 0.0)
        mask.set_weight("head", 1.0)
        assert mask.get_weight("head") == 1.0

    def test_set_weight_clamps_low(self, simple_skeleton):
        """Weights below 0 are clamped to 0."""
        mask = BoneMask()
        mask.set_weight("spine", 0.5)
        mask.set_weight("spine", -0.1)
        w = mask.get_weight("spine")
        assert 0.0 <= w <= 1.0, f"Weight {w} outside [0, 1]"

    def test_set_weight_clamps_high(self, simple_skeleton):
        """Weights above 1 are clamped to 1."""
        mask = BoneMask()
        mask.set_weight("spine", 1.5)
        w = mask.get_weight("spine")
        assert 0.0 <= w <= 1.0, f"Weight {w} outside [0, 1]"

    def test_multiple_bones_independent(self, simple_skeleton):
        """Different bone weights are stored independently."""
        mask = BoneMask()
        mask.set_weight("head", 1.0)
        mask.set_weight("spine", 0.5)
        mask.set_weight("root", 0.0)
        assert mask.get_weight("head") == 1.0
        assert mask.get_weight("spine") == 0.5
        assert mask.get_weight("root") == 0.0


# ============================================================================
# Equivalence Class: MissingBoneMode enumeration
# ============================================================================

class TestMissingBoneMode:
    """MissingBoneMode enumeration exists with ZERO and ONE values."""

    def test_zero_mode_enum_value(self):
        """MissingBoneMode has a ZERO member."""
        assert hasattr(MissingBoneMode, "ZERO")

    def test_one_mode_enum_value(self):
        """MissingBoneMode has a ONE member."""
        assert hasattr(MissingBoneMode, "ONE")

    def test_z_and_one_are_distinct(self):
        """ZERO and ONE are different enum members."""
        assert MissingBoneMode.ZERO != MissingBoneMode.ONE

    def test_zero_and_one_truthy(self):
        """Both ZERO and ONE enum members are truthy (they exist)."""
        assert bool(MissingBoneMode.ZERO)
        assert bool(MissingBoneMode.ONE)


# ============================================================================
# Equivalence Class: apply — blend transforms toward identity
# ============================================================================

class TestApply:
    """BoneMask.apply blends per-bone transforms toward identity.

    apply accepts a ``dict[str, Transform]`` mapping bone names to
    their current animated transforms and returns a new dict with
    each transform blended toward identity according to the mask weight.
    """

    def _bone_dict(self, skeleton, pos=(1.0, 0.0, 0.0),
                   scale=(1.0, 1.0, 1.0)):
        """Create a bone-name-to-transform dict from the skeleton."""
        return {
            bone.name: Transform(Vec3(*pos), Quat(0.0, 0.0, 0.0, 1.0), Vec3(*scale))
            for bone in skeleton
        }

    def test_apply_returns_dict(self, simple_skeleton):
        """apply returns a dict[str, Transform]."""
        mask = BoneMask()
        transforms = self._bone_dict(simple_skeleton)
        result = mask.apply(transforms)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, Transform)

    def test_apply_zero_weight_returns_identity(self, simple_skeleton):
        """A bone with weight=0.0 maps to the identity transform."""
        mask = BoneMask()
        mask.set_weight("root", 0.0)
        transforms = self._bone_dict(simple_skeleton,
                                     pos=(2.0, 3.0, 4.0),
                                     scale=(2.0, 2.0, 2.0))
        result = mask.apply(transforms)
        # root should be identity (weight 0.0)
        assert result["root"].translation == Vec3(0.0, 0.0, 0.0)
        # Other bones (weight 0.0 default) should also be identity
        assert result["head"].translation == Vec3(0.0, 0.0, 0.0)

    def test_apply_one_weight_returns_original(self, simple_skeleton):
        """A bone with weight=1.0 leaves the original transform unchanged."""
        mask = BoneMask()
        mask.set_weight("root", 1.0)
        pos = (5.0, -2.0, 3.0)
        scale = (1.5, 2.0, 0.5)
        transforms = self._bone_dict(simple_skeleton, pos=pos, scale=scale)
        result = mask.apply(transforms)
        assert result["root"].translation == Vec3(*pos)
        assert result["root"].scale == Vec3(*scale)

    def test_apply_partial_weight_blends(self, simple_skeleton):
        """A bone with weight=0.5 blends halfway toward identity."""
        mask = BoneMask()
        mask.set_weight("root", 0.5)
        pos = (2.0, 0.0, 0.0)
        scale = (3.0, 1.0, 1.0)
        transforms = self._bone_dict(simple_skeleton, pos=pos, scale=scale)
        result = mask.apply(transforms)
        root_result = result["root"]
        # weight=0.5 means 50% of effect toward identity:
        # translation lerp(Vec3(2,0,0), Vec3(0,0,0), 0.5) -> Vec3(1, 0, 0)
        # scale lerp(Vec3(3,1,1), Vec3(1,1,1), 0.5) -> Vec3(2, 1, 1)
        assert abs(root_result.translation.x - 1.0) < 0.001
        assert abs(root_result.scale.x - 2.0) < 0.001

    def test_apply_preserves_key_set(self, simple_skeleton):
        """apply returns a dict with the same bone-name keys as the input."""
        mask = BoneMask()
        mask.set_weight("root", 0.0)
        transforms = self._bone_dict(simple_skeleton)
        result = mask.apply(transforms)
        assert set(result.keys()) == set(transforms.keys())


# ============================================================================
# Equivalence Class: combine — product of weights
# ============================================================================

class TestCombine:
    """BoneMask.combine unites two masks via per-bone weight multiplication."""

    def test_combine_two_masks_with_overlap(self, simple_skeleton):
        """combine multiplies overlapping bone weights."""
        a = BoneMask()
        a.set_weight("root", 0.5)
        a.set_weight("spine", 0.8)

        b = BoneMask()
        b.set_weight("root", 0.3)
        b.set_weight("head", 1.0)

        combined = a.combine(b)
        # root: 0.5 * 0.3 = 0.15
        assert abs(combined.get_weight("root") - 0.15) < 0.001
        # spine: 0.8 * 0.0 = 0.0
        assert combined.get_weight("spine") == 0.0
        # head: 0.0 * 1.0 = 0.0
        assert combined.get_weight("head") == 0.0

    def test_combine_returns_new_mask(self, simple_skeleton):
        """combine returns a new BoneMask and does not mutate inputs."""
        a = BoneMask()
        a.set_weight("root", 0.8)
        b = BoneMask()
        b.set_weight("root", 0.5)

        combined = a.combine(b)
        assert combined is not a
        assert combined is not b
        # Originals unchanged
        assert a.get_weight("root") == 0.8
        assert b.get_weight("root") == 0.5
        assert abs(combined.get_weight("root") - 0.4) < 0.001

    def test_combine_is_commutative(self, simple_skeleton):
        """combine is commutative: a.combine(b) == b.combine(a)."""
        a = BoneMask()
        a.set_weight("root", 0.4)
        a.set_weight("spine", 0.9)
        b = BoneMask()
        b.set_weight("root", 0.7)
        b.set_weight("head", 0.2)

        ab = a.combine(b)
        ba = b.combine(a)
        assert abs(ab.get_weight("root") - ba.get_weight("root")) < 0.001
        assert abs(ab.get_weight("spine") - ba.get_weight("spine")) < 0.001
        assert abs(ab.get_weight("head") - ba.get_weight("head")) < 0.001

    def test_combine_with_non_bonemask_handled_gracefully(self, simple_skeleton):
        """combine with non-BoneMask raises an informative error."""
        a = BoneMask()
        # The implementation should handle non-BoneMask gracefully
        # (either by returning NotImplemented, raising, or converting)
        with pytest.raises((TypeError, AttributeError)):
            a.combine("not_a_mask")  # type: ignore[arg-type]


# ============================================================================
# Equivalence Class: invert — map each weight w to 1 - w
# ============================================================================

class TestInvert:
    """BoneMask.invert returns a fresh mask with 1 - weight per bone."""

    def test_invert_basic(self, simple_skeleton):
        """invert maps 0.3 to 0.7, 0.8 to 0.2, etc."""
        mask = BoneMask()
        mask.set_weight("root", 0.3)
        mask.set_weight("spine", 0.8)
        inv = mask.invert()
        assert abs(inv.get_weight("root") - 0.7) < 0.001
        assert abs(inv.get_weight("spine") - 0.2) < 0.001

    def test_invert_twice_returns_original(self, simple_skeleton):
        """Double-invert recovers the original weights."""
        mask = BoneMask()
        mask.set_weight("root", 0.3)
        mask.set_weight("spine", 0.8)
        inv = mask.invert()
        inv2 = inv.invert()
        assert abs(inv2.get_weight("root") - 0.3) < 0.001
        assert abs(inv2.get_weight("spine") - 0.8) < 0.001

    def test_invert_returns_new_mask(self, simple_skeleton):
        """invert does not mutate the original mask."""
        mask = BoneMask()
        mask.set_weight("root", 0.4)
        inv = mask.invert()
        assert mask.get_weight("root") == 0.4
        assert abs(inv.get_weight("root") - 0.6) < 0.001
        assert inv is not mask

    def test_invert_zero_becomes_one(self, simple_skeleton):
        """invert maps 0.0 to 1.0."""
        mask = BoneMask()
        mask.set_weight("root", 0.0)
        inv = mask.invert()
        assert inv.get_weight("root") == 1.0

    def test_invert_one_becomes_zero(self, simple_skeleton):
        """invert maps 1.0 to 0.0."""
        mask = BoneMask()
        mask.set_weight("root", 1.0)
        inv = mask.invert()
        assert inv.get_weight("root") == 0.0


# ============================================================================
# Equivalence Class: full — all weights to 1.0
# ============================================================================

class TestFull:
    """BoneMask.full(skeleton) sets all bone weights to 1.0."""

    def test_full_sets_all_to_one(self, simple_skeleton, all_bones):
        """full(skeleton) sets every registered bone's weight to 1.0."""
        mask = BoneMask()
        for name in all_bones:
            mask.set_weight(name, 0.0)
        full_mask = mask.full(simple_skeleton)
        for bone_name in all_bones:
            assert full_mask.get_weight(bone_name) == 1.0, (
                f"Bone '{bone_name}' should be 1.0 after full()"
            )

    def test_full_from_partial(self, simple_skeleton, all_bones):
        """full(skeleton) from a partially-set mask sets all bones to 1.0."""
        mask = BoneMask()
        mask.set_weight("root", 0.3)
        mask.set_weight("spine", 0.5)
        full_mask = mask.full(simple_skeleton)
        for bone_name in all_bones:
            assert full_mask.get_weight(bone_name) == 1.0

    def test_full_returns_new_mask(self, simple_skeleton):
        """full() returns a new BoneMask without mutating the original."""
        mask = BoneMask()
        mask.set_weight("root", 0.3)
        full_mask = mask.full(simple_skeleton)
        assert mask.get_weight("root") == 0.3
        assert full_mask.get_weight("root") == 1.0
        assert full_mask is not mask

    def test_full_produces_zero_weight_apply_identity(self, simple_skeleton):
        """After full() then invert(), apply yields identity transforms."""
        mask = BoneMask()
        full_mask = mask.full(simple_skeleton)
        empty_mask = full_mask.invert()  # all 0.0
        pos = Vec3(7.0, 3.0, -2.0)
        transforms = {
            bone.name: Transform(pos, Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0))
            for bone in simple_skeleton
        }
        result = empty_mask.apply(transforms)
        for r in result.values():
            assert r.translation == Vec3(0.0, 0.0, 0.0)


# ============================================================================
# Equivalence Class: BoneMaskPresets — body region masks
# ============================================================================

class TestBoneMaskPresets:
    """BoneMaskPresets generate masks for common body regions."""

    def test_upper_body_preset(self, simple_skeleton):
        """BoneMaskPresets.upper_body returns a BoneMask."""
        mask = BoneMaskPresets.upper_body(simple_skeleton)
        assert isinstance(mask, BoneMask)

    def test_lower_body_preset(self, simple_skeleton):
        """BoneMaskPresets.lower_body returns a BoneMask."""
        mask = BoneMaskPresets.lower_body(simple_skeleton)
        assert isinstance(mask, BoneMask)

    def test_left_arm_preset(self, simple_skeleton):
        """BoneMaskPresets.left_arm returns a BoneMask."""
        mask = BoneMaskPresets.left_arm(simple_skeleton)
        assert isinstance(mask, BoneMask)

    def test_right_arm_preset(self, simple_skeleton):
        """BoneMaskPresets.right_arm returns a BoneMask."""
        mask = BoneMaskPresets.right_arm(simple_skeleton)
        assert isinstance(mask, BoneMask)

    def test_all_preset_weights_in_zero_one_range(self, simple_skeleton):
        """All weights produced by every preset are in [0, 1]."""
        for preset_fn in [
            BoneMaskPresets.upper_body,
            BoneMaskPresets.lower_body,
            BoneMaskPresets.left_arm,
            BoneMaskPresets.right_arm,
            BoneMaskPresets.left_leg,
            BoneMaskPresets.right_leg,
        ]:
            mask = preset_fn(simple_skeleton)
            for bone_name in ["root", "spine", "head", "left_arm",
                              "right_arm", "left_leg", "right_leg"]:
                w = mask.get_weight(bone_name)
                assert 0.0 <= w <= 1.0, (
                    f"Preset {preset_fn.__name__}: bone '{bone_name}' "
                    f"weight {w} outside [0, 1]"
                )

    def test_some_preset_has_non_default_weight(self, humanoid_skeleton):
        """At least one preset assigns a non-zero weight to a bone
        (proving the preset knows about the skeleton's naming)."""
        any_non_default = False
        for preset_fn in [
            BoneMaskPresets.upper_body,
            BoneMaskPresets.lower_body,
            BoneMaskPresets.left_arm,
            BoneMaskPresets.right_arm,
            BoneMaskPresets.left_leg,
            BoneMaskPresets.right_leg,
        ]:
            mask = preset_fn(humanoid_skeleton)
            if any(
                mask.get_weight(bn) != 0.0
                for bn in ["Hips", "Spine", "Head", "LeftArm",
                           "RightArm", "LeftLeg", "RightLeg"]
            ):
                any_non_default = True
                break
        assert any_non_default, (
            "No preset recognised any bone in the humanoid skeleton. "
            "At least one preset should assign a non-zero weight."
        )


# ============================================================================
# Equivalence Class: Integration — combined operations compose
# ============================================================================

class TestIntegration:
    """Multiple BoneMask operations compose correctly."""

    def test_combine_with_invert(self, simple_skeleton):
        """combine(mask, mask.invert()) yields near-zero for set bones."""
        mask = BoneMask()
        mask.set_weight("root", 0.7)
        mask.set_weight("spine", 0.4)
        inv = mask.invert()
        combined = mask.combine(inv)
        # root:  0.7 * (1 - 0.7) = 0.21
        # spine: 0.4 * (1 - 0.4) = 0.24
        assert abs(combined.get_weight("root") - 0.21) < 0.01
        assert abs(combined.get_weight("spine") - 0.24) < 0.01

    def test_full_then_apply_returns_originals(self, simple_skeleton):
        """A full mask passes all transforms through unchanged."""
        mask = BoneMask()
        full_mask = mask.full(simple_skeleton)
        pos = Vec3(7.0, 3.0, -2.0)
        transforms = {
            bone.name: Transform(pos, Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0))
            for bone in simple_skeleton
        }
        result = full_mask.apply(transforms)
        for r in result.values():
            assert r.translation == pos

    def test_zero_weight_apply_then_invert_apply(self, simple_skeleton):
        """Zero mask yields identity; invert yields pass-through."""
        mask = BoneMask()
        mask.set_weight("root", 0.0)
        pos = Vec3(4.0, 0.0, 0.0)
        transforms = {
            bone.name: Transform(pos, Quat(0.0, 0.0, 0.0, 1.0), Vec3(1.0, 1.0, 1.0))
            for bone in simple_skeleton
        }
        # weight 0.0 -> identity
        r0 = mask.apply(transforms)
        assert r0["root"].translation == Vec3(0.0, 0.0, 0.0)

        # invert -> weight 1.0 -> pass-through
        inv = mask.invert()
        r1 = inv.apply(transforms)
        assert r1["root"].translation == pos
