"""Tests for skeleton hierarchy editing and retargeting setup."""

import pytest

from engine.core.math import Quat, Transform, Vec3
from engine.tooling.animation_tools.skeleton_editor import (
    BoneEditMode,
    BoneMirrorPair,
    BoneSelection,
    RetargetMapping,
    RetargetSource,
    SkeletonEditor,
    SkeletonPreview,
    Socket,
    SocketAttachment,
    VirtualBone,
    VirtualBoneType,
)


# =============================================================================
# BONE SELECTION TESTS
# =============================================================================


class TestBoneSelection:
    def test_basic_selection(self):
        sel = BoneSelection()
        assert not sel.has_selection
        assert sel.bone_count == 0

    def test_select_bone(self):
        sel = BoneSelection()
        sel.select_bone(0)
        assert sel.has_selection
        assert 0 in sel.selected_bones
        assert sel.primary_bone == 0

    def test_select_multiple(self):
        sel = BoneSelection()
        sel.select_bone(0)
        sel.select_bone(1, add_to_selection=True)
        assert sel.bone_count == 2
        assert sel.primary_bone == 1

    def test_deselect_bone(self):
        sel = BoneSelection()
        sel.select_bone(0)
        sel.select_bone(1, add_to_selection=True)
        sel.deselect_bone(0)
        assert 0 not in sel.selected_bones
        assert sel.primary_bone == 1

    def test_toggle_bone(self):
        sel = BoneSelection()
        sel.toggle_bone(0)
        assert 0 in sel.selected_bones
        sel.toggle_bone(0)
        assert 0 not in sel.selected_bones

    def test_clear_selection(self):
        sel = BoneSelection()
        sel.select_bone(0)
        sel.selected_sockets.add("socket1")
        sel.clear()
        assert not sel.has_selection
        assert sel.primary_bone == -1


# =============================================================================
# SOCKET TESTS
# =============================================================================


class TestSocket:
    def test_basic_socket(self):
        socket = Socket(name="weapon", bone_name="hand_r")
        assert socket.name == "weapon"
        assert socket.bone_name == "hand_r"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            Socket(name="", bone_name="hand")

    def test_empty_bone_raises(self):
        with pytest.raises(ValueError, match="Bone name cannot be empty"):
            Socket(name="socket", bone_name="")

    def test_get_world_transform(self):
        socket = Socket(
            name="weapon",
            bone_name="hand_r",
            relative_transform=Transform(
                translation=Vec3(1, 0, 0),
            ),
        )
        bone_transform = Transform(translation=Vec3(5, 5, 5))
        world = socket.get_world_transform(bone_transform)
        assert world.translation.x == 6  # 5 + 1

    def test_socket_copy(self):
        socket = Socket(
            name="weapon",
            bone_name="hand_r",
            preview_mesh="/meshes/sword.mesh",
        )
        copy = socket.copy()
        assert copy.name == socket.name
        assert copy is not socket


class TestSocketAttachment:
    def test_basic_attachment(self):
        attachment = SocketAttachment(
            socket_name="weapon",
            asset_path="/weapons/sword.mesh",
        )
        assert attachment.socket_name == "weapon"
        assert attachment.asset_path == "/weapons/sword.mesh"


# =============================================================================
# VIRTUAL BONE TESTS
# =============================================================================


class TestVirtualBone:
    def test_basic_virtual_bone(self):
        vbone = VirtualBone(
            name="weapon_target",
            source_bone="hand_r",
            bone_type=VirtualBoneType.COPY,
        )
        assert vbone.name == "weapon_target"
        assert vbone.source_bone == "hand_r"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name cannot be empty"):
            VirtualBone(name="", source_bone="bone")

    def test_midpoint_requires_target(self):
        with pytest.raises(ValueError, match="requires target bone"):
            VirtualBone(
                name="mid",
                source_bone="bone1",
                bone_type=VirtualBoneType.MIDPOINT,
            )

    def test_copy_transform(self):
        vbone = VirtualBone(
            name="copy",
            source_bone="bone",
            bone_type=VirtualBoneType.COPY,
        )
        source = Transform(translation=Vec3(1, 2, 3))
        result = vbone.compute_transform(source)
        assert result.translation.x == 1
        assert result.translation.y == 2

    def test_midpoint_transform(self):
        vbone = VirtualBone(
            name="mid",
            source_bone="bone1",
            target_bone="bone2",
            bone_type=VirtualBoneType.MIDPOINT,
            blend_factor=0.5,
        )
        source = Transform(translation=Vec3(0, 0, 0))
        target = Transform(translation=Vec3(10, 10, 10))
        result = vbone.compute_transform(source, target)
        assert abs(result.translation.x - 5.0) < 0.001

    def test_distance_transform(self):
        vbone = VirtualBone(
            name="offset",
            source_bone="bone",
            bone_type=VirtualBoneType.DISTANCE,
            blend_factor=5.0,  # distance
            look_axis=Vec3(1, 0, 0),
        )
        source = Transform(translation=Vec3(0, 0, 0))
        result = vbone.compute_transform(source)
        assert abs(result.translation.x - 5.0) < 0.001


# =============================================================================
# RETARGETING TESTS
# =============================================================================


class TestRetargetSource:
    def test_basic_source(self):
        source = RetargetSource(
            name="humanoid",
            skeleton_path="/skeletons/humanoid.skel",
        )
        assert source.name == "humanoid"


class TestBoneMirrorPair:
    def test_basic_pair(self):
        pair = BoneMirrorPair(left_bone="hand_l", right_bone="hand_r")
        assert pair.get_mirror("hand_l") == "hand_r"
        assert pair.get_mirror("hand_r") == "hand_l"
        assert pair.get_mirror("spine") is None


class TestRetargetMapping:
    def test_basic_mapping(self):
        mapping = RetargetMapping(
            source_bone="hand_l",
            target_bone="hand_l",
        )
        assert mapping.source_bone == "hand_l"
        assert mapping.target_bone == "hand_l"

    def test_apply_retarget(self):
        mapping = RetargetMapping(
            source_bone="bone",
            target_bone="bone",
            translation_mode="retarget",
        )
        source_transform = Transform(translation=Vec3(5, 5, 5))
        source_ref = Transform(translation=Vec3(0, 0, 0))
        target_ref = Transform(translation=Vec3(10, 10, 10))

        result = mapping.apply(source_transform, source_ref, target_ref)
        # Should be target_ref + source_delta
        assert result.translation.x == 15  # 10 + 5


# =============================================================================
# SKELETON PREVIEW TESTS
# =============================================================================


class TestSkeletonPreview:
    def test_basic_preview(self):
        preview = SkeletonPreview()
        assert preview.show_bones
        assert preview.show_sockets

    def test_camera_controls(self):
        preview = SkeletonPreview()
        preview.set_camera_distance(5.0)
        assert preview._camera_distance == 5.0

        preview.rotate_camera(0.1, 0.1)
        # Camera should have rotated

    def test_reset_camera(self):
        preview = SkeletonPreview()
        preview.set_camera_distance(10.0)
        preview.reset_camera()
        assert preview._camera_distance == 3.0


# =============================================================================
# SKELETON EDITOR TESTS
# =============================================================================


class TestSkeletonEditor:
    @pytest.fixture
    def sample_skeleton_data(self):
        return [
            {"name": "root", "parent_index": -1, "transform": Transform.identity()},
            {"name": "spine_01", "parent_index": 0, "transform": Transform.identity()},
            {"name": "spine_02", "parent_index": 1, "transform": Transform.identity()},
            {"name": "head", "parent_index": 2, "transform": Transform.identity()},
            {"name": "arm_l", "parent_index": 2, "transform": Transform.identity()},
            {"name": "arm_r", "parent_index": 2, "transform": Transform.identity()},
        ]

    def test_basic_editor(self):
        editor = SkeletonEditor()
        assert editor.bone_count == 0

    def test_load_skeleton(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        assert editor.bone_count == 6
        assert "root" in editor.bone_names

    def test_get_bone_index(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        assert editor.get_bone_index("spine_01") == 1
        assert editor.get_bone_index("nonexistent") == -1

    def test_get_bone_parent(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        assert editor.get_bone_parent(0) == -1  # root
        assert editor.get_bone_parent(1) == 0  # spine_01 -> root

    def test_get_bone_children(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        children = editor.get_bone_children(2)  # spine_02
        assert 3 in children  # head
        assert 4 in children  # arm_l
        assert 5 in children  # arm_r

    def test_get_root_bones(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        roots = editor.get_root_bones()
        assert 0 in roots

    def test_get_bone_depth(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        assert editor.get_bone_depth(0) == 0  # root
        assert editor.get_bone_depth(3) == 3  # head

    def test_bone_chain(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        chain = editor.get_bone_chain(0, 3)  # root to head
        assert len(chain) == 4
        assert chain[0] == 0
        assert chain[-1] == 3

    def test_bone_transform(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        transform = editor.get_bone_transform(0)
        assert transform is not None
        new_transform = Transform(translation=Vec3(1, 2, 3))
        assert editor.set_bone_transform(0, new_transform)

    # Socket tests
    def test_add_socket(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        socket = Socket(name="weapon", bone_name="arm_r")
        assert editor.add_socket(socket)
        assert len(editor.get_sockets()) == 1

    def test_duplicate_socket_rejected(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        socket1 = Socket(name="weapon", bone_name="arm_r")
        socket2 = Socket(name="weapon", bone_name="arm_l")
        editor.add_socket(socket1)
        assert not editor.add_socket(socket2)

    def test_socket_invalid_bone_rejected(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        socket = Socket(name="weapon", bone_name="nonexistent")
        assert not editor.add_socket(socket)

    def test_remove_socket(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        socket = Socket(name="weapon", bone_name="arm_r")
        editor.add_socket(socket)
        assert editor.remove_socket("weapon")
        assert len(editor.get_sockets()) == 0

    def test_get_sockets_on_bone(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        editor.add_socket(Socket(name="socket1", bone_name="arm_r"))
        editor.add_socket(Socket(name="socket2", bone_name="arm_r"))
        editor.add_socket(Socket(name="socket3", bone_name="arm_l"))
        sockets = editor.get_sockets_on_bone("arm_r")
        assert len(sockets) == 2

    def test_rename_socket(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        editor.add_socket(Socket(name="old_name", bone_name="arm_r"))
        assert editor.rename_socket("old_name", "new_name")
        assert editor.get_socket("new_name") is not None
        assert editor.get_socket("old_name") is None

    # Virtual bone tests
    def test_add_virtual_bone(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        vbone = VirtualBone(
            name="target",
            source_bone="head",
            bone_type=VirtualBoneType.COPY,
        )
        assert editor.add_virtual_bone(vbone)
        assert len(editor.get_virtual_bones()) == 1

    def test_remove_virtual_bone(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        vbone = VirtualBone(name="target", source_bone="head", bone_type=VirtualBoneType.COPY)
        editor.add_virtual_bone(vbone)
        assert editor.remove_virtual_bone("target")
        assert len(editor.get_virtual_bones()) == 0

    # Mirror pair tests
    def test_add_mirror_pair(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        pair = BoneMirrorPair(left_bone="arm_l", right_bone="arm_r")
        assert editor.add_mirror_pair(pair)
        assert editor.get_mirror_bone("arm_l") == "arm_r"

    def test_auto_detect_mirror_pairs(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        detected = editor.auto_detect_mirror_pairs("_l", "_r")
        assert detected == 1  # arm_l/arm_r

    # Retargeting tests
    def test_add_retarget_source(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        source = RetargetSource(name="source", skeleton_path="/skel.skel")
        assert editor.add_retarget_source(source)
        assert len(editor.get_retarget_sources()) == 1

    def test_add_retarget_mapping(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        mapping = RetargetMapping(source_bone="spine", target_bone="spine_01")
        editor.add_retarget_mapping(mapping)
        assert len(editor.get_retarget_mappings()) == 1

    # Selection tests
    def test_select_bone(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        editor.select_bone(0)
        assert 0 in editor.selection.selected_bones

    def test_select_bone_chain(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        editor.select_bone_chain(0, 3)
        assert len(editor.selection.selected_bones) == 4

    def test_select_children(self, sample_skeleton_data):
        editor = SkeletonEditor()
        editor.load_skeleton(sample_skeleton_data)
        editor.select_children(2)  # spine_02
        # Should select spine_02, head, arm_l, arm_r
        assert len(editor.selection.selected_bones) == 4

    def test_edit_mode(self, sample_skeleton_data):
        editor = SkeletonEditor()
        assert editor.edit_mode == BoneEditMode.SELECT
        editor.edit_mode = BoneEditMode.ROTATE
        assert editor.edit_mode == BoneEditMode.ROTATE

    def test_on_change_callback(self, sample_skeleton_data):
        editor = SkeletonEditor()
        callback_called = [False]

        def callback():
            callback_called[0] = True

        editor.add_on_change(callback)
        editor.load_skeleton(sample_skeleton_data)
        assert callback_called[0]
