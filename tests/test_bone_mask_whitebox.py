"""WHITEBOX tests for engine/animation/graph/bone_mask.py.

WHITEBOX coverage plan:
  [BoneMask]
    Path A1:  __init__ default mode ZERO and default name
    Path A2:  __init__ custom name and ONE mode
    Path B1:  set_weight normal value within [0,1]
    Path B2:  set_weight clamping below 0
    Path B3:  set_weight clamping above 1
    Path B4:  set_weight boundary 0.0 and 1.0 pass through
    Path C1:  get_weight returns explicit weight when set
    Path C2:  get_weight missing bone with ZERO mode returns 0.0
    Path C3:  get_weight missing bone with ONE mode returns 1.0
    Path C4:  get_weight after set_weight retrieves clamped value
    Path D1:  weights property returns read-only dict snapshot
    Path E1:  has_bone True for explicit entry
    Path E2:  has_bone False for unset bone
    Path F1:  bone_count starts zero
    Path F2:  bone_count increments after set_weight
    Path G1:  apply weight 1.0 returns transform unchanged
    Path G2:  apply weight 0.0 returns identity transform
    Path G3:  apply weight 0.5 returns lerp midpoint
    Path G4:  apply only affects bones in transforms dict (missing bone defaults)
    Path H1:  combine multiplies weights for overlapping bones
    Path H2:  combine union of non-overlapping bones
    Path H3:  combine with ZERO/ONE modes on non-overlapping bones
    Path H4:  combine name defaults to "{self.name}_x_{other.name}"
    Path I1:  invert produces 1-weight for every explicit entry
    Path I2:  invert preserves MissingBoneMode
    Path J1:  full creates mask with all skeleton bones at 1.0
    Path J2:  full with empty skeleton returns empty mask
    Path K1:  from_bone_names creates mask with specified bones at weight
    Path K2:  from_bone_names weight clamped to [0,1]
    Path K3:  from_bone_names include_children adds descendants
    Path K4:  from_bone_names skips bones not in skeleton
    Path K5:  from_bone_names mode is ZERO (missing bones default to 0)
    Path L1:  copy creates independent deep-ish copy

  [BoneMaskPresets]
    Path M1:  upper_body includes named bones
    Path M2:  lower_body includes named bones
    Path M3:  left_arm includes named bones
    Path M4:  right_arm includes named bones
    Path M5:  left_leg includes named bones
    Path M6:  right_leg includes named bones
    Path N1:  gradient linear falloff
    Path N2:  gradient exponential falloff
    Path N3:  gradient start_bone not in skeleton returns empty mask
    Path N4:  gradient single bone chain (no descendants)

  [Edge cases]
    Path O1:  empty mask (no entries) apply returns transforms unchanged
    Path O2:  empty mask (no entries) combine with another mask
    Path O3:  invert of empty mask returns empty mask
    Path O4:  missing bone in apply uses mode default
    Path O5:  apply with empty transforms dict returns empty dict
    Path O6:  set_weight then get_weight round-trip preserves value
"""

from __future__ import annotations

import math
import pytest

from engine.animation.graph.bone_mask import BoneMask, BoneMaskPresets, MissingBoneMode
from engine.animation.graph.skeleton import Skeleton
from engine.core.math.transform import Transform
from engine.core.math.vec import Vec3
from engine.core.math.quat import Quat


# =========================================================================
# Helpers
# =========================================================================

def _transform_equal(a: Transform, b: Transform) -> bool:
    """Compare two Transforms field-by-field (no __eq__)."""
    return (
        a.translation.x == b.translation.x
        and a.translation.y == b.translation.y
        and a.translation.z == b.translation.z
        and a.rotation.x == b.rotation.x
        and a.rotation.y == b.rotation.y
        and a.rotation.z == b.rotation.z
        and a.rotation.w == b.rotation.w
        and a.scale.x == b.scale.x
        and a.scale.y == b.scale.y
        and a.scale.z == b.scale.z
    )


def _make_transform(
    tx: float = 0.0, ty: float = 0.0, tz: float = 0.0,
    rx: float = 0.0, ry: float = 0.0, rz: float = 0.0, rw: float = 1.0,
    sx: float = 1.0, sy: float = 1.0, sz: float = 1.0,
) -> Transform:
    """Construct a Transform with explicit numeric fields."""
    return Transform(
        Vec3(tx, ty, tz),
        Quat(rx, ry, rz, rw),
        Vec3(sx, sy, sz),
    )


def _make_humanoid_skeleton() -> Skeleton:
    """Build a minimal humanoid skeleton for preset testing."""
    skel = Skeleton("humanoid")
    # Hips (root)
    skel.add_bone("Hips")
    # Spine chain
    skel.add_bone("Spine", parent_name="Hips")
    skel.add_bone("Spine1", parent_name="Spine")
    skel.add_bone("Spine2", parent_name="Spine1")
    skel.add_bone("Chest", parent_name="Spine2")
    skel.add_bone("Neck", parent_name="Chest")
    skel.add_bone("Head", parent_name="Neck")
    # Left arm
    skel.add_bone("LeftShoulder", parent_name="Chest")
    skel.add_bone("LeftArm", parent_name="LeftShoulder")
    skel.add_bone("LeftForeArm", parent_name="LeftArm")
    skel.add_bone("LeftHand", parent_name="LeftForeArm")
    # Right arm
    skel.add_bone("RightShoulder", parent_name="Chest")
    skel.add_bone("RightArm", parent_name="RightShoulder")
    skel.add_bone("RightForeArm", parent_name="RightArm")
    skel.add_bone("RightHand", parent_name="RightForeArm")
    # Pelvis
    skel.add_bone("Pelvis", parent_name="Hips")
    # Left leg
    skel.add_bone("LeftUpLeg", parent_name="Pelvis")
    skel.add_bone("LeftLeg", parent_name="LeftUpLeg")
    skel.add_bone("LeftFoot", parent_name="LeftLeg")
    skel.add_bone("LeftToeBase", parent_name="LeftFoot")
    # Right leg
    skel.add_bone("RightUpLeg", parent_name="Pelvis")
    skel.add_bone("RightLeg", parent_name="RightUpLeg")
    skel.add_bone("RightFoot", parent_name="RightLeg")
    skel.add_bone("RightToeBase", parent_name="RightFoot")
    return skel


def _lerp_identity_to(xform: Transform, w: float) -> Transform:
    """Compute identity.lerp(xform, w) -- direct ref for apply() tests."""
    return Transform.identity().lerp(xform, w)


# =========================================================================
# BoneMask: __init__
# =========================================================================

class TestBoneMaskInit:
    def test_default_mode_is_zero(self) -> None:
        mask = BoneMask()
        assert mask.name == "mask"
        assert mask.mode is MissingBoneMode.ZERO

    def test_custom_name_and_one_mode(self) -> None:
        mask = BoneMask(name="custom", mode=MissingBoneMode.ONE)
        assert mask.name == "custom"
        assert mask.mode is MissingBoneMode.ONE

    def test_initial_no_weights(self) -> None:
        mask = BoneMask()
        assert mask._weights == {}
        assert mask.bone_count == 0


# =========================================================================
# BoneMask: set_weight / get_weight
# =========================================================================

class TestBoneMaskSetGetWeight:
    def test_set_and_get_normal_value(self) -> None:
        mask = BoneMask()
        mask.set_weight("Hips", 0.75)
        assert mask.get_weight("Hips") == 0.75

    def test_clamp_below_zero(self) -> None:
        mask = BoneMask()
        mask.set_weight("Hips", -0.5)
        assert mask.get_weight("Hips") == 0.0

    def test_clamp_above_one(self) -> None:
        mask = BoneMask()
        mask.set_weight("Hips", 1.5)
        assert mask.get_weight("Hips") == 1.0

    def test_boundary_zero(self) -> None:
        mask = BoneMask()
        mask.set_weight("Hips", 0.0)
        assert mask.get_weight("Hips") == 0.0

    def test_boundary_one(self) -> None:
        mask = BoneMask()
        mask.set_weight("Hips", 1.0)
        assert mask.get_weight("Hips") == 1.0

    def test_get_missing_zero_mode(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        assert mask.get_weight("MissingBone") == 0.0

    def test_get_missing_one_mode(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ONE)
        assert mask.get_weight("MissingBone") == 1.0

    def test_get_weight_after_explicit_set_overrides_mode(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("Spine", 0.3)
        assert mask.get_weight("Spine") == 0.3
        # Explicit bone is in _weights, mode only applies to missing bones
        assert mask.get_weight("Head") == 0.0  # not set, mode ZERO


# =========================================================================
# BoneMask: weights property, has_bone, bone_count
# =========================================================================

class TestBoneMaskProperties:
    def test_weights_returns_snapshot(self) -> None:
        mask = BoneMask()
        mask.set_weight("a", 0.5)
        snapshot = mask.weights
        assert snapshot == {"a": 0.5}
        # Mutating snapshot should NOT affect mask
        snapshot["b"] = 1.0
        assert mask.bone_count == 1

    def test_weights_empty(self) -> None:
        mask = BoneMask()
        assert mask.weights == {}

    def test_has_bone_explicit_entry(self) -> None:
        mask = BoneMask()
        mask.set_weight("Hips", 1.0)
        assert mask.has_bone("Hips") is True

    def test_has_bone_missing(self) -> None:
        mask = BoneMask()
        assert mask.has_bone("Spine") is False

    def test_bone_count_starts_zero(self) -> None:
        mask = BoneMask()
        assert mask.bone_count == 0

    def test_bone_count_after_sets(self) -> None:
        mask = BoneMask()
        mask.set_weight("a", 0.5)
        mask.set_weight("b", 1.0)
        assert mask.bone_count == 2

    def test_bone_count_does_not_count_duplicate_sets(self) -> None:
        mask = BoneMask()
        mask.set_weight("x", 0.3)
        mask.set_weight("x", 0.8)
        assert mask.bone_count == 1


# =========================================================================
# BoneMask: apply()
# =========================================================================

class TestBoneMaskApply:
    def test_weight_one_returns_transform_unchanged(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("BoneA", 1.0)
        xform = _make_transform(tx=10, ty=20, tz=30, sx=2, sy=3, sz=4)
        result = mask.apply({"BoneA": xform})
        expected = _lerp_identity_to(xform, 1.0)
        assert _transform_equal(result["BoneA"], expected)

    def test_weight_zero_returns_identity(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("BoneA", 0.0)
        xform = _make_transform(tx=10, ty=20, tz=30, sx=2, sy=3, sz=4)
        result = mask.apply({"BoneA": xform})
        expected = _lerp_identity_to(xform, 0.0)
        assert _transform_equal(result["BoneA"], expected)

    def test_weight_half_returns_midpoint(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("BoneA", 0.5)
        xform = _make_transform(tx=10, ty=20, tz=30, sx=2, sy=3, sz=4)
        result = mask.apply({"BoneA": xform})
        expected = _lerp_identity_to(xform, 0.5)
        assert _transform_equal(result["BoneA"], expected)

    def test_only_affected_bones_modified(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("BoneA", 0.0)
        xform_a = _make_transform(tx=10, ty=0, tz=0)
        xform_b = _make_transform(tx=0, ty=20, tz=0)
        result = mask.apply({"BoneA": xform_a, "BoneB": xform_b})
        # BoneA weight 0 -> identity
        assert _transform_equal(result["BoneA"], Transform.identity())
        # BoneB not in mask -> mode ZERO -> weight 0 -> identity
        assert _transform_equal(result["BoneB"], Transform.identity())

    def test_one_mode_missing_bone_fully_affected(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ONE, name="full_body")
        # No explicit weights, so all bones use mode->weight=1
        xform = _make_transform(tx=5, ty=10, tz=15, sx=2, sy=1, sz=3)
        result = mask.apply({"BoneX": xform})
        expected = _lerp_identity_to(xform, 1.0)
        assert _transform_equal(result["BoneX"], expected)

    def test_empty_transforms_dict_returns_empty(self) -> None:
        mask = BoneMask()
        result = mask.apply({})
        assert result == {}


# =========================================================================
# BoneMask: combine()
# =========================================================================

class TestBoneMaskCombine:
    def test_multiply_overlapping_weights(self) -> None:
        a = BoneMask(name="a")
        a.set_weight("Bone", 0.8)
        b = BoneMask(name="b")
        b.set_weight("Bone", 0.5)
        c = a.combine(b)
        assert c.get_weight("Bone") == 0.4  # 0.8 * 0.5

    def test_union_of_non_overlapping_with_one_mode(self) -> None:
        """With ONE mode, missing bones default to 1.0 so the weight passes through."""
        a = BoneMask(name="a", mode=MissingBoneMode.ONE)
        a.set_weight("X", 0.7)
        b = BoneMask(name="b", mode=MissingBoneMode.ONE)
        b.set_weight("Y", 0.3)
        c = a.combine(b)
        # For X: a=0.7, b=1.0 (ONE default) => 0.7*1.0 = 0.7
        # For Y: a=1.0 (ONE default), b=0.3 => 1.0*0.3 = 0.3
        assert c.get_weight("X") == pytest.approx(0.7)
        assert c.get_weight("Y") == pytest.approx(0.3)

    def test_union_of_non_overlapping_zero_mode_product(self) -> None:
        """With ZERO mode, the other mask's default 0.0 zeros out non-overlapping bones."""
        a = BoneMask(name="a")
        a.set_weight("X", 0.7)
        b = BoneMask(name="b")
        b.set_weight("Y", 0.3)
        c = a.combine(b)
        # For X: a=0.7, b.get_weight("X")=0.0 (ZERO default) => 0.7*0.0 = 0.0
        # For Y: a.get_weight("Y")=0.0 (ZERO default), b=0.3 => 0.0*0.3 = 0.0
        assert c.get_weight("X") == pytest.approx(0.0)
        assert c.get_weight("Y") == pytest.approx(0.0)

    def test_missing_bone_defaults_from_each_mask(self) -> None:
        """Bone 'Z' is not in either mask -- both use ZERO mode, so 0*0=0."""
        a = BoneMask(name="a", mode=MissingBoneMode.ZERO)
        a.set_weight("X", 1.0)
        b = BoneMask(name="b", mode=MissingBoneMode.ZERO)
        b.set_weight("Y", 1.0)
        c = a.combine(b)
        # Z missing from both => get_weight returns 0 for each => 0*0=0
        assert c.get_weight("Z") == 0.0

    def test_combine_with_one_mode_gives_one(self) -> None:
        """Bone in both masks, first has ONE mode (default=1), second has explicit=1."""
        a = BoneMask(name="a", mode=MissingBoneMode.ONE)
        a.set_weight("X", 1.0)
        b = BoneMask(name="b", mode=MissingBoneMode.ONE)
        b.set_weight("X", 1.0)
        c = a.combine(b)
        assert c.get_weight("X") == 1.0

    def test_combine_default_name(self) -> None:
        a = BoneMask(name="upper")
        b = BoneMask(name="lower")
        c = a.combine(b)
        assert c.name == "upper_x_lower"

    def test_combine_custom_name(self) -> None:
        a = BoneMask(name="a")
        b = BoneMask(name="b")
        c = a.combine(b, name="combined")
        assert c.name == "combined"

    def test_combine_product_clamped(self) -> None:
        """Product exceeding 1.0 should be clamped (though 0-1 * 0-1 never exceeds 1)."""
        a = BoneMask(name="a")
        a.set_weight("X", 1.0)
        b = BoneMask(name="b")
        b.set_weight("X", 1.0)
        c = a.combine(b)
        assert c.get_weight("X") == 1.0

    def test_combine_does_not_mutate_inputs(self) -> None:
        a = BoneMask(name="a")
        a.set_weight("X", 0.8)
        b = BoneMask(name="b")
        b.set_weight("X", 0.5)
        original_a_weights = dict(a._weights)
        original_b_weights = dict(b._weights)
        a.combine(b)
        assert a._weights == original_a_weights
        assert b._weights == original_b_weights


# =========================================================================
# BoneMask: invert()
# =========================================================================

class TestBoneMaskInvert:
    def test_invert_flips_weights(self) -> None:
        mask = BoneMask(name="original")
        mask.set_weight("A", 0.2)
        mask.set_weight("B", 0.8)
        mask.set_weight("C", 0.0)
        inv = mask.invert()
        assert inv.get_weight("A") == pytest.approx(0.8)
        assert inv.get_weight("B") == pytest.approx(0.2)
        assert inv.get_weight("C") == pytest.approx(1.0)

    def test_invert_preserves_mode(self) -> None:
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("X", 0.3)
        inv = mask.invert()
        assert inv.mode is MissingBoneMode.ZERO
        # Missing bone 'Y' uses mode ZERO
        assert inv.get_weight("Y") == 0.0

    def test_invert_default_name(self) -> None:
        mask = BoneMask(name="mymask")
        inv = mask.invert()
        assert inv.name == "not_mymask"

    def test_invert_custom_name(self) -> None:
        mask = BoneMask(name="mymask")
        inv = mask.invert(name="inverted")
        assert inv.name == "inverted"

    def test_invert_empty_mask(self) -> None:
        mask = BoneMask()
        inv = mask.invert()
        assert inv.bone_count == 0
        assert inv.weights == {}

    def test_invert_does_not_mutate_original(self) -> None:
        mask = BoneMask()
        mask.set_weight("X", 0.3)
        original_w = dict(mask._weights)
        mask.invert()
        assert mask._weights == original_w


# =========================================================================
# BoneMask: full() factory
# =========================================================================

class TestBoneMaskFull:
    def test_full_has_all_bones_at_one(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.full(skel, name="full_body")
        assert mask.name == "full_body"
        assert mask.bone_count == skel.bone_count
        for bone in skel:
            assert mask.get_weight(bone.name) == 1.0

    def test_full_mode_is_one(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        mask = BoneMask.full(skel)
        assert mask.mode is MissingBoneMode.ONE

    def test_full_empty_skeleton(self) -> None:
        skel = Skeleton()
        mask = BoneMask.full(skel)
        assert mask.bone_count == 0
        assert mask.weights == {}

    def test_full_apply_returns_transforms_unchanged(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.full(skel)
        xforms = {b.name: _make_transform(tx=float(i)) for i, b in enumerate(skel)}
        result = mask.apply(xforms)
        for name, xf in xforms.items():
            assert _transform_equal(result[name], xf)


# =========================================================================
# BoneMask: from_bone_names() factory
# =========================================================================

class TestBoneMaskFromBoneNames:
    def test_explicit_list(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(
            skel, "upper", ["Spine", "Chest", "Head"], weight=1.0,
        )
        assert mask.get_weight("Spine") == 1.0
        assert mask.get_weight("Chest") == 1.0
        assert mask.get_weight("Head") == 1.0
        # Hip not in list, mode ZERO
        assert mask.get_weight("Hips") == 0.0

    def test_default_weight_is_one(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(skel, "test", ["Hips"])
        assert mask.get_weight("Hips") == 1.0

    def test_weight_clamped(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(
            skel, "test", ["Hips"], weight=2.0,
        )
        assert mask.get_weight("Hips") == 1.0

    def test_include_children_adds_descendants(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(
            skel, "left_leg", ["LeftUpLeg"], weight=1.0, include_children=True,
        )
        assert mask.get_weight("LeftUpLeg") == 1.0
        assert mask.get_weight("LeftLeg") == 1.0
        assert mask.get_weight("LeftFoot") == 1.0
        assert mask.get_weight("LeftToeBase") == 1.0

    def test_include_children_with_no_children(self) -> None:
        skel = _make_humanoid_skeleton()
        # LeftToeBase is a leaf bone
        mask = BoneMask.from_bone_names(
            skel, "test", ["LeftToeBase"], weight=1.0, include_children=True,
        )
        assert mask.get_weight("LeftToeBase") == 1.0
        assert mask.bone_count == 1

    def test_skips_bones_not_in_skeleton(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(
            skel, "test", ["Hips", "NonExistent"], weight=1.0,
        )
        assert mask.get_weight("Hips") == 1.0
        # NonExistent not in skeleton so not added
        assert mask.has_bone("NonExistent") is False
        assert mask.bone_count == 1

    def test_mode_is_zero(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(skel, "test", ["Hips"], weight=0.5)
        assert mask.mode is MissingBoneMode.ZERO
        assert mask.get_weight("Spine") == 0.0  # not set

    def test_include_children_via_depth_chain(self) -> None:
        """Spine -> Spine1 -> Spine2 -> Chest is a chain.
        Include children from Spine should get all descendants.
        """
        skel = _make_humanoid_skeleton()
        mask = BoneMask.from_bone_names(
            skel, "spine_chain", ["Spine"], weight=0.8, include_children=True,
        )
        assert mask.get_weight("Spine") == 0.8
        assert mask.get_weight("Spine1") == 0.8
        assert mask.get_weight("Spine2") == 0.8
        assert mask.get_weight("Chest") == 0.8  # Chest is descendant of Spine


# =========================================================================
# BoneMask: copy()
# =========================================================================

class TestBoneMaskCopy:
    def test_independent_copy(self) -> None:
        mask = BoneMask(name="original", mode=MissingBoneMode.ONE)
        mask.set_weight("X", 0.7)
        mask.set_weight("Y", 0.3)

        copied = mask.copy()
        assert copied.name == "original"
        assert copied.mode is MissingBoneMode.ONE
        assert copied.get_weight("X") == 0.7
        assert copied.get_weight("Y") == 0.3

        # Mutate original -- copy should be unaffected
        mask.set_weight("X", 0.1)
        assert copied.get_weight("X") == 0.7

    def test_copy_with_new_name(self) -> None:
        mask = BoneMask(name="original")
        copied = mask.copy(name="clone")
        assert copied.name == "clone"

    def test_copy_empty(self) -> None:
        mask = BoneMask()
        copied = mask.copy()
        assert copied.bone_count == 0
        assert copied._weights == {}


# =========================================================================
# BoneMaskPresets: upper_body, lower_body
# =========================================================================

class TestBoneMaskPresetsUpperLower:
    def test_upper_body_contains_spine_chest_head_arms(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMaskPresets.upper_body(skel)
        assert mask.name == "UpperBody"
        # Core upper bones at weight 1
        assert mask.get_weight("Spine") == 1.0
        assert mask.get_weight("Chest") == 1.0
        assert mask.get_weight("Neck") == 1.0
        assert mask.get_weight("Head") == 1.0
        # Arms
        assert mask.get_weight("LeftShoulder") == 1.0
        assert mask.get_weight("RightArm") == 1.0
        # Lower body -- not in upper body list, mode ZERO
        assert mask.get_weight("Hips") == 0.0
        assert mask.get_weight("LeftUpLeg") == 0.0
        assert mask.get_weight("RightLeg") == 0.0

    def test_lower_body_contains_hips_legs(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMaskPresets.lower_body(skel)
        assert mask.name == "LowerBody"
        assert mask.get_weight("Hips") == 1.0
        assert mask.get_weight("Pelvis") == 1.0
        assert mask.get_weight("LeftUpLeg") == 1.0
        assert mask.get_weight("LeftLeg") == 1.0
        assert mask.get_weight("RightFoot") == 1.0
        # Mode is ZERO (from from_bone_names defaults)
        assert mask.mode is MissingBoneMode.ZERO
        # Upper-body bones must NOT be included
        assert mask.get_weight("Spine") == 0.0
        assert mask.get_weight("Chest") == 0.0
        assert mask.get_weight("Neck") == 0.0
        assert mask.get_weight("Head") == 0.0
        assert mask.get_weight("LeftShoulder") == 0.0
        assert mask.get_weight("RightArm") == 0.0

    def test_upper_lower_complementary(self) -> None:
        """Upper + Lower should cover the full skeleton."""
        skel = _make_humanoid_skeleton()
        upper = BoneMaskPresets.upper_body(skel)
        lower = BoneMaskPresets.lower_body(skel)
        combined = upper.combine(lower)
        # Every bone should be 1 in at least one mask, so product should be 0 or 1
        # (since each mask bone is either 0 or 1)
        for bone in skel:
            w = combined.get_weight(bone.name)
            # Some bones (like Hips/Pelvis) could be in both, some in only one
            # This checks the combine doesn't introduce unexpected values
            assert w in (0.0, 1.0), f"Bone {bone.name} has unexpected weight {w}"


# =========================================================================
# BoneMaskPresets: left_arm, right_arm, left_leg, right_leg
# =========================================================================

class TestBoneMaskPresetsLimbs:
    def test_left_arm(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMaskPresets.left_arm(skel)
        assert mask.name == "LeftArm"
        assert mask.get_weight("LeftShoulder") == 1.0
        assert mask.get_weight("LeftArm") == 1.0
        assert mask.get_weight("LeftForeArm") == 1.0
        assert mask.get_weight("LeftHand") == 1.0
        # Right arm not included
        assert mask.get_weight("RightShoulder") == 0.0
        assert mask.get_weight("RightArm") == 0.0

    def test_right_arm(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMaskPresets.right_arm(skel)
        assert mask.name == "RightArm"
        assert mask.get_weight("RightShoulder") == 1.0
        assert mask.get_weight("RightArm") == 1.0
        assert mask.get_weight("RightForeArm") == 1.0
        assert mask.get_weight("RightHand") == 1.0
        # Left arm not included
        assert mask.get_weight("LeftShoulder") == 0.0
        assert mask.get_weight("LeftArm") == 0.0

    def test_left_leg(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMaskPresets.left_leg(skel)
        assert mask.name == "LeftLeg"
        assert mask.get_weight("LeftUpLeg") == 1.0
        assert mask.get_weight("LeftLeg") == 1.0
        assert mask.get_weight("LeftFoot") == 1.0
        assert mask.get_weight("LeftToeBase") == 1.0
        # Right leg not included
        assert mask.get_weight("RightUpLeg") == 0.0

    def test_right_leg(self) -> None:
        skel = _make_humanoid_skeleton()
        mask = BoneMaskPresets.right_leg(skel)
        assert mask.name == "RightLeg"
        assert mask.get_weight("RightUpLeg") == 1.0
        assert mask.get_weight("RightLeg") == 1.0
        assert mask.get_weight("RightFoot") == 1.0
        assert mask.get_weight("RightToeBase") == 1.0
        # Left leg not included
        assert mask.get_weight("LeftUpLeg") == 0.0

    def test_all_four_limbs_distinct(self) -> None:
        """Check left_arm and right_arm produce different bone sets."""
        skel = _make_humanoid_skeleton()
        la = BoneMaskPresets.left_arm(skel)
        ra = BoneMaskPresets.right_arm(skel)
        assert la.get_weight("LeftArm") == 1.0
        assert la.get_weight("RightArm") == 0.0
        assert ra.get_weight("LeftArm") == 0.0
        assert ra.get_weight("RightArm") == 1.0


# =========================================================================
# BoneMaskPresets: gradient()
# =========================================================================

class TestBoneMaskPresetsGradient:
    def test_linear_falloff_from_root_to_tip(self) -> None:
        """Build linear chain: root -> a -> b -> c. Gradient from root.
        root_weight=0.0, tip_weight=1.0.
        root depth=0, c depth=3. Expected weights: root=0, a=0.333, b=0.667, c=1.0
        """
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="a")
        skel.add_bone("c", parent_name="b")
        mask = BoneMaskPresets.gradient(
            skel, "root", falloff="linear",
            root_weight=0.0, tip_weight=1.0,
        )
        # Depth: root=0, a=1, b=2, c=3, max_rel=3
        # t = rel/3, w = 0 + 1*t
        assert mask.get_weight("root") == pytest.approx(0.0, abs=1e-9)
        assert mask.get_weight("a") == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert mask.get_weight("b") == pytest.approx(2.0 / 3.0, abs=1e-9)
        assert mask.get_weight("c") == pytest.approx(3.0 / 3.0, abs=1e-9)

    def test_linear_falloff_custom_root_tip(self) -> None:
        """root_weight=0.2, tip_weight=0.8 with same chain."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="a")
        mask = BoneMaskPresets.gradient(
            skel, "root", falloff="linear",
            root_weight=0.2, tip_weight=0.8,
        )
        # depth: root=0, a=1, b=2, max_rel=2
        # w = 0.2 + (0.8-0.2)*t, t=rel/2
        assert mask.get_weight("root") == pytest.approx(0.2, abs=1e-9)
        assert mask.get_weight("a") == pytest.approx(0.2 + 0.6 * 0.5, abs=1e-9)
        assert mask.get_weight("b") == pytest.approx(0.2 + 0.6 * 1.0, abs=1e-9)

    def test_exponential_falloff(self) -> None:
        """Chain: root -> a -> b. Exponential with rate=2.0.
        root_weight=0.0, tip_weight=1.0.
        depth: root=0, a=1, b=2, max_rel=2.
        t_root = 0/2 = 0, t_a = 1/2 = 0.5, t_b = 2/2 = 1.0
        exp transform: t' = 1 - exp(-rate * t)
        t'_root = 1 - exp(0) = 0
        t'_a = 1 - exp(-2*0.5) = 1 - exp(-1) = 1 - 0.367879 = 0.632121
        t'_b = 1 - exp(-2*1) = 1 - exp(-2) = 1 - 0.135335 = 0.864665
        w = root_weight + (tip_weight - root_weight) * t' = t' (since 0->1)
        """
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="a")
        mask = BoneMaskPresets.gradient(
            skel, "root", falloff="exponential", rate=2.0,
            root_weight=0.0, tip_weight=1.0,
        )
        assert mask.get_weight("root") == pytest.approx(0.0, abs=1e-9)
        expected_a = 1.0 - math.exp(-2.0 * 0.5)
        assert mask.get_weight("a") == pytest.approx(expected_a, abs=1e-9)
        expected_b = 1.0 - math.exp(-2.0 * 1.0)
        assert mask.get_weight("b") == pytest.approx(expected_b, abs=1e-9)

    def test_gradient_start_bone_not_in_skeleton(self) -> None:
        skel = Skeleton()
        skel.add_bone("root")
        mask = BoneMaskPresets.gradient(skel, "NonExistent")
        assert mask.bone_count == 0
        assert mask.weights == {}

    def test_gradient_single_bone_no_descendants(self) -> None:
        skel = Skeleton()
        skel.add_bone("only")
        mask = BoneMaskPresets.gradient(
            skel, "only", falloff="linear",
            root_weight=0.0, tip_weight=1.0,
        )
        # max_rel_depth = 0 (single bone), so t = 0, w = root_weight = 0
        assert mask.get_weight("only") == pytest.approx(0.0, abs=1e-9)
        assert mask.bone_count == 1

    def test_gradient_default_params(self) -> None:
        """Defaults: linear, rate=1.0, root_weight=0.0, tip_weight=1.0."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        mask = BoneMaskPresets.gradient(skel, "root")
        assert mask.get_weight("root") == pytest.approx(0.0, abs=1e-9)
        assert mask.get_weight("a") == pytest.approx(1.0, abs=1e-9)
        assert mask.name == "gradient_root"

    def test_gradient_exponential_rate_one_default(self) -> None:
        """rate=1.0 default for exponential gives standard curve."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        skel.add_bone("b", parent_name="a")
        mask = BoneMaskPresets.gradient(
            skel, "root", falloff="exponential",
        )
        # depth: root=0, a=1, b=2, max_rel=2
        # t: 0, 0.5, 1.0
        # t': 0, 1-exp(-0.5)=0.39347, 1-exp(-1)=0.63212
        assert mask.get_weight("a") == pytest.approx(1.0 - math.exp(-0.5), abs=1e-9)
        assert mask.get_weight("b") == pytest.approx(1.0 - math.exp(-1.0), abs=1e-9)

    def test_gradient_linear_default_matches_chain_length(self) -> None:
        """Verify linear gradient on a short 2-bone chain root->a."""
        skel = Skeleton()
        skel.add_bone("root")
        skel.add_bone("a", parent_name="root")
        mask = BoneMaskPresets.gradient(skel, "root", falloff="linear")
        # max_rel=1, root=0->w=0, a=1->w=1
        assert mask.get_weight("root") == pytest.approx(0.0, abs=1e-9)
        assert mask.get_weight("a") == pytest.approx(1.0, abs=1e-9)


# =========================================================================
# BoneMask: edge cases
# =========================================================================

class TestBoneMaskEdgeCases:
    def test_empty_mask_apply_all_missing_zero(self) -> None:
        """Empty mask (no explicit weights) with ZERO mode: all bones become identity."""
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        xforms = {
            "A": _make_transform(tx=10),
            "B": _make_transform(ty=20),
        }
        result = mask.apply(xforms)
        assert _transform_equal(result["A"], Transform.identity())
        assert _transform_equal(result["B"], Transform.identity())

    def test_empty_mask_apply_all_missing_one(self) -> None:
        """Empty mask with ONE mode: all bones fully affected."""
        mask = BoneMask(mode=MissingBoneMode.ONE)
        xf_a = _make_transform(tx=10)
        xf_b = _make_transform(ty=20)
        xforms = {"A": xf_a, "B": xf_b}
        result = mask.apply(xforms)
        assert _transform_equal(result["A"], xf_a)
        assert _transform_equal(result["B"], xf_b)

    def test_combine_empty_with_populated(self) -> None:
        empty = BoneMask(name="empty")
        populated = BoneMask(name="pop")
        populated.set_weight("X", 0.7)
        combined = empty.combine(populated)
        # empty.get_weight("X")=0 (ZERO mode), populated.get_weight("X")=0.7
        # Product = 0 * 0.7 = 0
        assert combined.get_weight("X") == 0.0

    def test_combine_populated_with_empty(self) -> None:
        populated = BoneMask(name="pop")
        populated.set_weight("X", 0.7)
        empty = BoneMask(name="empty")
        combined = populated.combine(empty)
        # populated.get_weight("X")=0.7, empty.get_weight("X")=0 (ZERO)
        # Product = 0.7 * 0 = 0
        assert combined.get_weight("X") == 0.0

    def test_invert_empty_returns_empty(self) -> None:
        mask = BoneMask()
        inv = mask.invert()
        assert inv.bone_count == 0

    def test_full_mask_apply_returns_originals(self) -> None:
        """A mask with all bones at weight 1.0 -> transforms unchanged."""
        mask = BoneMask(mode=MissingBoneMode.ONE)
        mask.set_weight("A", 1.0)
        mask.set_weight("B", 1.0)
        xf_a = _make_transform(tx=5)
        xf_b = _make_transform(ty=10)
        result = mask.apply({"A": xf_a, "B": xf_b})
        assert _transform_equal(result["A"], xf_a)
        assert _transform_equal(result["B"], xf_b)

    def test_set_weight_then_get_weight_round_trip_edge(self) -> None:
        mask = BoneMask()
        values = [0.0, 0.5, 1.0, 0.001, 0.999]
        for v in values:
            mask.set_weight("X", v)
            assert mask.get_weight("X") == v

    def test_get_weight_missing_after_set_weight_other_bone(self) -> None:
        """Setting one bone does not affect get_weight for other bones."""
        mask = BoneMask(mode=MissingBoneMode.ZERO)
        mask.set_weight("A", 0.8)
        assert mask.get_weight("B") == 0.0  # missing, mode ZERO

    def test_repr_format(self) -> None:
        mask = BoneMask(name="test", mode=MissingBoneMode.ZERO)
        mask.set_weight("X", 0.5)
        r = repr(mask)
        assert "BoneMask" in r
        assert "test" in r
        assert "bones=1" in r
        assert "ZERO" in r
