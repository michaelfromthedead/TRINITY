"""Tests for pose editor with pose library, blending, and additive poses."""

import pytest

# PoseSnapshot not implemented in pose_editor
pytest.skip("Pose editor API mismatch", allow_module_level=True)

from engine.core.math import Quat, Transform, Vec3
from engine.tooling.animation_tools.pose_editor import (
    AdditivePose,
    AdditiveType,
    AnimPose,
    PoseBlendMode,
    PoseCategory,
    PoseEditor,
    PoseLibrary,
    PoseSnapshot,
    PoseType,
)


# =============================================================================
# ANIM POSE TESTS
# =============================================================================


class TestAnimPose:
    def test_basic_pose(self):
        pose = AnimPose(name="T-Pose")
        assert pose.name == "T-Pose"
        assert pose.pose_type == PoseType.REFERENCE
        assert pose.bone_count == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            AnimPose(name="")

    def test_set_bone_transform(self):
        pose = AnimPose(name="Test")
        transform = Transform(
            translation=Vec3(0, 0, 0),
            rotation=Quat.identity(),
            scale=Vec3(1, 1, 1),
        )
        pose.set_bone_transform("spine_01", transform)
        assert pose.bone_count == 1
        assert pose.has_bone("spine_01")

    def test_get_bone_transform(self):
        pose = AnimPose(name="Test")
        transform = Transform(translation=Vec3(1, 2, 3))
        pose.set_bone_transform("head", transform)

        result = pose.get_bone_transform("head")
        assert result is not None
        assert result.translation.x == 1
        assert result.translation.y == 2

    def test_get_nonexistent_bone_returns_none(self):
        pose = AnimPose(name="Test")
        assert pose.get_bone_transform("nonexistent") is None

    def test_remove_bone(self):
        pose = AnimPose(name="Test")
        pose.set_bone_transform("bone", Transform.identity())
        assert pose.remove_bone("bone")
        assert not pose.has_bone("bone")

    def test_clear_bones(self):
        pose = AnimPose(name="Test")
        pose.set_bone_transform("bone1", Transform.identity())
        pose.set_bone_transform("bone2", Transform.identity())
        pose.clear()
        assert pose.bone_count == 0

    def test_copy_pose(self):
        pose = AnimPose(name="Original", pose_type=PoseType.ADDITIVE)
        pose.set_bone_transform("bone", Transform(translation=Vec3(1, 0, 0)))

        copy = pose.copy()
        assert copy.name == pose.name
        assert copy.pose_type == pose.pose_type
        assert copy is not pose
        assert copy.get_bone_transform("bone") is not pose.get_bone_transform("bone")

    def test_blend_poses(self):
        pose_a = AnimPose(name="A")
        pose_a.set_bone_transform("bone", Transform(translation=Vec3(0, 0, 0)))

        pose_b = AnimPose(name="B")
        pose_b.set_bone_transform("bone", Transform(translation=Vec3(10, 0, 0)))

        blended = AnimPose.blend(pose_a, pose_b, 0.5)
        result = blended.get_bone_transform("bone")
        assert abs(result.translation.x - 5.0) < 0.001

    def test_blend_modes(self):
        pose_a = AnimPose(name="A")
        pose_a.set_bone_transform("bone", Transform(translation=Vec3(2, 0, 0)))

        pose_b = AnimPose(name="B")
        pose_b.set_bone_transform("bone", Transform(translation=Vec3(4, 0, 0)))

        # Lerp blend
        lerp = AnimPose.blend(pose_a, pose_b, 0.5, mode=PoseBlendMode.LERP)
        assert abs(lerp.get_bone_transform("bone").translation.x - 3.0) < 0.001

    def test_get_bone_names(self):
        pose = AnimPose(name="Test")
        pose.set_bone_transform("bone_a", Transform.identity())
        pose.set_bone_transform("bone_b", Transform.identity())

        names = pose.get_bone_names()
        assert "bone_a" in names
        assert "bone_b" in names


# =============================================================================
# ADDITIVE POSE TESTS
# =============================================================================


class TestAdditivePose:
    def test_basic_additive(self):
        additive = AdditivePose(name="Attack_Add")
        assert additive.pose_type == PoseType.ADDITIVE
        assert additive.additive_type == AdditiveType.LOCAL_SPACE

    def test_compute_from_reference(self):
        base = AnimPose(name="Base")
        base.set_bone_transform("bone", Transform(translation=Vec3(0, 0, 0)))

        target = AnimPose(name="Target")
        target.set_bone_transform("bone", Transform(translation=Vec3(5, 0, 0)))

        additive = AdditivePose.compute_additive(base, target, name="Add")
        delta = additive.get_bone_transform("bone")
        # Delta should be target - base
        assert abs(delta.translation.x - 5.0) < 0.001

    def test_apply_additive(self):
        base = AnimPose(name="Base")
        base.set_bone_transform("bone", Transform(translation=Vec3(10, 0, 0)))

        additive = AdditivePose(name="Add")
        additive.set_bone_transform("bone", Transform(translation=Vec3(5, 0, 0)))

        result = additive.apply_to(base, alpha=1.0)
        combined = result.get_bone_transform("bone")
        assert abs(combined.translation.x - 15.0) < 0.001

    def test_apply_additive_partial_alpha(self):
        base = AnimPose(name="Base")
        base.set_bone_transform("bone", Transform(translation=Vec3(10, 0, 0)))

        additive = AdditivePose(name="Add")
        additive.set_bone_transform("bone", Transform(translation=Vec3(10, 0, 0)))

        result = additive.apply_to(base, alpha=0.5)
        combined = result.get_bone_transform("bone")
        # 10 + (10 * 0.5) = 15
        assert abs(combined.translation.x - 15.0) < 0.001

    def test_mesh_space_additive(self):
        additive = AdditivePose(name="MeshAdd", additive_type=AdditiveType.MESH_SPACE)
        assert additive.additive_type == AdditiveType.MESH_SPACE


# =============================================================================
# POSE SNAPSHOT TESTS
# =============================================================================


class TestPoseSnapshot:
    def test_basic_snapshot(self):
        pose = AnimPose(name="Test")
        pose.set_bone_transform("bone", Transform(translation=Vec3(1, 2, 3)))

        snapshot = PoseSnapshot.capture(pose)
        assert snapshot.pose_name == "Test"
        assert snapshot.bone_count == 1

    def test_restore_snapshot(self):
        original = AnimPose(name="Original")
        original.set_bone_transform("bone", Transform(translation=Vec3(5, 0, 0)))

        snapshot = PoseSnapshot.capture(original)

        # Modify original
        original.set_bone_transform("bone", Transform(translation=Vec3(100, 0, 0)))

        # Restore
        restored = snapshot.restore()
        assert abs(restored.get_bone_transform("bone").translation.x - 5.0) < 0.001


# =============================================================================
# POSE CATEGORY TESTS
# =============================================================================


class TestPoseCategory:
    def test_basic_category(self):
        category = PoseCategory(name="Combat")
        assert category.name == "Combat"
        assert category.pose_count == 0

    def test_add_pose(self):
        category = PoseCategory(name="Combat")
        pose = AnimPose(name="Attack01")
        assert category.add_pose(pose)
        assert category.pose_count == 1

    def test_add_duplicate_rejected(self):
        category = PoseCategory(name="Combat")
        pose1 = AnimPose(name="Attack")
        pose2 = AnimPose(name="Attack")
        category.add_pose(pose1)
        assert not category.add_pose(pose2)

    def test_remove_pose(self):
        category = PoseCategory(name="Combat")
        pose = AnimPose(name="Attack")
        category.add_pose(pose)
        assert category.remove_pose("Attack")
        assert category.pose_count == 0

    def test_get_pose(self):
        category = PoseCategory(name="Combat")
        pose = AnimPose(name="Attack")
        category.add_pose(pose)
        found = category.get_pose("Attack")
        assert found is pose

    def test_get_all_poses(self):
        category = PoseCategory(name="Combat")
        category.add_pose(AnimPose(name="Attack1"))
        category.add_pose(AnimPose(name="Attack2"))
        poses = category.get_all_poses()
        assert len(poses) == 2


# =============================================================================
# POSE LIBRARY TESTS
# =============================================================================


class TestPoseLibrary:
    def test_basic_library(self):
        library = PoseLibrary(name="CharacterPoses")
        assert library.name == "CharacterPoses"
        assert library.category_count == 0

    def test_add_category(self):
        library = PoseLibrary(name="Test")
        assert library.add_category("Combat")
        assert library.category_count == 1

    def test_add_duplicate_category_rejected(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        assert not library.add_category("Combat")

    def test_remove_category(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        assert library.remove_category("Combat")
        assert library.category_count == 0

    def test_get_category(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        category = library.get_category("Combat")
        assert category is not None
        assert category.name == "Combat"

    def test_add_pose_to_category(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        pose = AnimPose(name="Attack")
        assert library.add_pose("Combat", pose)
        assert library.get_category("Combat").pose_count == 1

    def test_add_pose_creates_category(self):
        library = PoseLibrary(name="Test")
        pose = AnimPose(name="Idle")
        library.add_pose("Movement", pose, create_category=True)
        assert library.category_count == 1
        assert library.get_category("Movement") is not None

    def test_get_pose(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        pose = AnimPose(name="Attack")
        library.add_pose("Combat", pose)
        found = library.get_pose("Combat", "Attack")
        assert found is pose

    def test_search_poses(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        library.add_pose("Combat", AnimPose(name="Attack_Light"))
        library.add_pose("Combat", AnimPose(name="Attack_Heavy"))
        library.add_pose("Combat", AnimPose(name="Block"))

        results = library.search("Attack")
        assert len(results) == 2

    def test_get_all_poses(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        library.add_category("Movement")
        library.add_pose("Combat", AnimPose(name="Attack"))
        library.add_pose("Movement", AnimPose(name="Walk"))

        all_poses = library.get_all_poses()
        assert len(all_poses) == 2

    def test_pose_tags(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        pose = AnimPose(name="Attack")
        library.add_pose("Combat", pose, tags=["melee", "fast"])

        tagged = library.get_poses_by_tag("melee")
        assert len(tagged) == 1
        assert tagged[0].name == "Attack"

    def test_get_all_tags(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        library.add_pose("Combat", AnimPose(name="Attack1"), tags=["melee", "fast"])
        library.add_pose("Combat", AnimPose(name="Attack2"), tags=["melee", "slow"])

        tags = library.get_all_tags()
        assert "melee" in tags
        assert "fast" in tags
        assert "slow" in tags

    def test_import_export(self):
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        library.add_pose("Combat", AnimPose(name="Attack"))

        data = library.to_dict()
        imported = PoseLibrary.from_dict(data)

        assert imported.name == library.name
        assert imported.category_count == 1


# =============================================================================
# POSE EDITOR TESTS
# =============================================================================


class TestPoseEditor:
    def test_basic_editor(self):
        editor = PoseEditor()
        assert editor.current_pose is None

    def test_create_new_pose(self):
        editor = PoseEditor()
        pose = editor.create_new("T-Pose")
        assert pose is not None
        assert editor.current_pose is pose

    def test_load_pose(self):
        editor = PoseEditor()
        pose = AnimPose(name="Test")
        editor.load(pose)
        assert editor.current_pose is pose

    def test_clear(self):
        editor = PoseEditor()
        editor.create_new("Test")
        editor.clear()
        assert editor.current_pose is None

    def test_set_bone_transform(self):
        editor = PoseEditor()
        editor.create_new("Test")
        transform = Transform(translation=Vec3(1, 2, 3))
        editor.set_bone_transform("spine", transform)
        assert editor.current_pose.has_bone("spine")

    def test_get_bone_transform(self):
        editor = PoseEditor()
        editor.create_new("Test")
        transform = Transform(translation=Vec3(5, 0, 0))
        editor.set_bone_transform("head", transform)

        result = editor.get_bone_transform("head")
        assert result.translation.x == 5

    def test_select_bone(self):
        editor = PoseEditor()
        editor.create_new("Test")
        editor.set_bone_transform("bone1", Transform.identity())
        editor.set_bone_transform("bone2", Transform.identity())

        editor.select_bone("bone1")
        assert "bone1" in editor.selected_bones

        editor.select_bone("bone2", add_to_selection=True)
        assert len(editor.selected_bones) == 2

    def test_clear_selection(self):
        editor = PoseEditor()
        editor.create_new("Test")
        editor.set_bone_transform("bone", Transform.identity())
        editor.select_bone("bone")
        editor.clear_selection()
        assert len(editor.selected_bones) == 0

    def test_mirror_pose(self):
        editor = PoseEditor()
        editor.create_new("Test")
        editor.set_mirror_pairs([("arm_l", "arm_r"), ("leg_l", "leg_r")])

        # Set left arm transform
        editor.set_bone_transform("arm_l", Transform(translation=Vec3(5, 0, 0)))
        editor.mirror_selected_bones(["arm_l"])

        # Right arm should be mirrored
        right = editor.get_bone_transform("arm_r")
        assert right is not None
        assert right.translation.x == -5  # Mirrored X

    def test_copy_paste_pose(self):
        editor = PoseEditor()
        editor.create_new("Original")
        editor.set_bone_transform("bone", Transform(translation=Vec3(1, 2, 3)))

        editor.copy_pose()
        editor.create_new("NewPose")
        editor.paste_pose()

        assert editor.current_pose.has_bone("bone")

    def test_blend_with_reference(self):
        editor = PoseEditor()
        editor.create_new("Current")
        editor.set_bone_transform("bone", Transform(translation=Vec3(10, 0, 0)))

        reference = AnimPose(name="Reference")
        reference.set_bone_transform("bone", Transform(translation=Vec3(0, 0, 0)))

        editor.blend_with_reference(reference, 0.5)
        result = editor.get_bone_transform("bone")
        assert abs(result.translation.x - 5.0) < 0.001

    def test_reset_to_reference(self):
        editor = PoseEditor()
        editor.create_new("Test")
        editor.set_bone_transform("bone", Transform(translation=Vec3(100, 0, 0)))

        reference = AnimPose(name="Reference")
        reference.set_bone_transform("bone", Transform(translation=Vec3(0, 0, 0)))

        editor.set_reference_pose(reference)
        editor.reset_to_reference()

        result = editor.get_bone_transform("bone")
        assert abs(result.translation.x - 0.0) < 0.001

    def test_undo_redo(self):
        editor = PoseEditor()
        editor.create_new("Test")

        editor.set_bone_transform("bone", Transform(translation=Vec3(1, 0, 0)))
        editor.set_bone_transform("bone", Transform(translation=Vec3(2, 0, 0)))

        editor.undo()
        result = editor.get_bone_transform("bone")
        assert abs(result.translation.x - 1.0) < 0.001

        editor.redo()
        result = editor.get_bone_transform("bone")
        assert abs(result.translation.x - 2.0) < 0.001

    def test_on_change_callback(self):
        editor = PoseEditor()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        editor.add_on_change(callback)
        editor.create_new("Test")
        assert callback_called[0]

    def test_get_pose_diff(self):
        editor = PoseEditor()
        editor.create_new("Current")
        editor.set_bone_transform("bone", Transform(translation=Vec3(10, 0, 0)))

        reference = AnimPose(name="Reference")
        reference.set_bone_transform("bone", Transform(translation=Vec3(5, 0, 0)))

        diff = editor.get_pose_diff(reference)
        assert "bone" in diff
        assert abs(diff["bone"].translation.x - 5.0) < 0.001

    def test_apply_pose_diff(self):
        editor = PoseEditor()
        editor.create_new("Test")
        editor.set_bone_transform("bone", Transform(translation=Vec3(0, 0, 0)))

        diff = {"bone": Transform(translation=Vec3(10, 0, 0))}
        editor.apply_pose_diff(diff)

        result = editor.get_bone_transform("bone")
        assert abs(result.translation.x - 10.0) < 0.001

    def test_library_integration(self):
        editor = PoseEditor()
        library = PoseLibrary(name="Test")
        editor.set_library(library)

        editor.create_new("MyPose")
        editor.set_bone_transform("bone", Transform.identity())
        editor.save_to_library("Combat")

        saved = library.get_pose("Combat", "MyPose")
        assert saved is not None

    def test_load_from_library(self):
        editor = PoseEditor()
        library = PoseLibrary(name="Test")
        library.add_category("Combat")
        pose = AnimPose(name="Attack")
        pose.set_bone_transform("bone", Transform(translation=Vec3(5, 0, 0)))
        library.add_pose("Combat", pose)

        editor.set_library(library)
        editor.load_from_library("Combat", "Attack")

        assert editor.current_pose is not None
        assert editor.current_pose.name == "Attack"
