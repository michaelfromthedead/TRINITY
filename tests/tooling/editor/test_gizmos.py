"""
Comprehensive tests for the Gizmo system.

Tests cover:
- Gizmo types (translate, rotate, scale, universal)
- Transform constraints (axis, snap, limits)
- Transform spaces (world, local, view, parent)
- Drag operations (begin, update, end, cancel)
- Gizmo manager operations
"""
import pytest
import sys

sys.path.insert(0, '/home/user/dev/AI_GAME_ENGINE')

from engine.tooling.editor.gizmos import (
    GizmoManager,
    Gizmo,
    GizmoType,
    GizmoSpace,
    GizmoAxis,
    GizmoConstraint,
    TranslateGizmo,
    RotateGizmo,
    ScaleGizmo,
    UniversalGizmo,
)


class TestGizmoConstraint:
    """Tests for GizmoConstraint class."""

    def test_constraint_creation(self):
        """Constraint should be created with defaults."""
        constraint = GizmoConstraint()
        assert constraint.axis == GizmoAxis.XYZ
        assert constraint.snap_enabled is False
        assert constraint.limit_enabled is False

    def test_constraint_snap(self):
        """Snap should round to grid."""
        constraint = GizmoConstraint()
        constraint.snap_enabled = True
        constraint.snap_value = 1.0

        assert constraint.apply_snap(0.3) == 0.0
        assert constraint.apply_snap(0.7) == 1.0
        assert constraint.apply_snap(2.4) == 2.0

    def test_constraint_snap_custom_value(self):
        """Snap works with custom values."""
        constraint = GizmoConstraint()
        constraint.snap_enabled = True
        constraint.snap_value = 0.25

        assert constraint.apply_snap(0.3) == 0.25
        assert constraint.apply_snap(0.4) == 0.5
        assert constraint.apply_snap(0.9) == 1.0

    def test_constraint_limit(self):
        """Limit should clamp values."""
        constraint = GizmoConstraint()
        constraint.limit_enabled = True
        constraint.limit_min = -10.0
        constraint.limit_max = 10.0

        assert constraint.apply_limit(-15.0) == -10.0
        assert constraint.apply_limit(15.0) == 10.0
        assert constraint.apply_limit(5.0) == 5.0

    def test_constraint_apply_both(self):
        """Apply should do both snap and limit."""
        constraint = GizmoConstraint()
        constraint.snap_enabled = True
        constraint.snap_value = 5.0
        constraint.limit_enabled = True
        constraint.limit_min = 0.0
        constraint.limit_max = 20.0

        # Snap then limit
        assert constraint.apply(23.0) == 20.0  # Snaps to 25, limited to 20
        assert constraint.apply(-3.0) == 0.0   # Snaps to -5, limited to 0

    def test_constraint_axis_check(self):
        """Axis enabled check."""
        constraint = GizmoConstraint(GizmoAxis.X | GizmoAxis.Y)

        assert constraint.is_axis_enabled(GizmoAxis.X) is True
        assert constraint.is_axis_enabled(GizmoAxis.Y) is True
        assert constraint.is_axis_enabled(GizmoAxis.Z) is False


class TestGizmoAxis:
    """Tests for GizmoAxis flags."""

    def test_axis_combinations(self):
        """Axis flags can be combined."""
        xy = GizmoAxis.X | GizmoAxis.Y

        assert GizmoAxis.X in xy
        assert GizmoAxis.Y in xy
        assert GizmoAxis.Z not in xy

    def test_axis_presets(self):
        """Axis presets exist."""
        assert GizmoAxis.XY == (GizmoAxis.X | GizmoAxis.Y)
        assert GizmoAxis.XZ == (GizmoAxis.X | GizmoAxis.Z)
        assert GizmoAxis.YZ == (GizmoAxis.Y | GizmoAxis.Z)
        assert GizmoAxis.XYZ == (GizmoAxis.X | GizmoAxis.Y | GizmoAxis.Z)


class TestGizmo:
    """Tests for base Gizmo class."""

    def test_gizmo_creation(self):
        """Gizmo should be created with defaults."""
        gizmo = TranslateGizmo()  # Using concrete class
        assert gizmo.visible is True
        assert gizmo.enabled is True
        assert gizmo.space == GizmoSpace.WORLD
        assert gizmo.is_dragging is False

    def test_gizmo_position(self):
        """Gizmo position can be set."""
        gizmo = TranslateGizmo()
        gizmo.set_position(10, 20, 30)

        assert gizmo.position == (10, 20, 30)

    def test_gizmo_rotation(self):
        """Gizmo rotation can be set."""
        gizmo = TranslateGizmo()
        gizmo.set_rotation(45, 90, 0)

        assert gizmo.rotation == (45, 90, 0)

    def test_gizmo_space(self):
        """Gizmo space can be changed."""
        gizmo = TranslateGizmo()
        gizmo.set_space(GizmoSpace.LOCAL)

        assert gizmo.space == GizmoSpace.LOCAL

    def test_gizmo_snap_enable(self):
        """Snap can be enabled."""
        gizmo = TranslateGizmo()
        gizmo.enable_snap(True, 2.0)

        assert gizmo.constraint.snap_enabled is True
        assert gizmo.constraint.snap_value == 2.0

    def test_gizmo_drag_lifecycle(self):
        """Drag has begin/update/end lifecycle."""
        gizmo = TranslateGizmo()
        callbacks = []
        gizmo.on_transform_start = lambda axis: callbacks.append(("start", axis))
        gizmo.on_transform_end = lambda: callbacks.append(("end",))

        gizmo.begin_drag(GizmoAxis.X, 100, 100)
        assert gizmo.is_dragging is True
        assert gizmo.selected_axis == GizmoAxis.X
        assert ("start", GizmoAxis.X) in callbacks

        gizmo.end_drag()
        assert gizmo.is_dragging is False
        assert gizmo.selected_axis == GizmoAxis.NONE
        assert ("end",) in callbacks

    def test_gizmo_drag_cancel(self):
        """Drag can be cancelled."""
        gizmo = TranslateGizmo()
        gizmo.begin_drag(GizmoAxis.X, 100, 100)

        gizmo.cancel_drag()
        assert gizmo.is_dragging is False

    def test_gizmo_drag_disabled(self):
        """Disabled gizmo cannot be dragged."""
        gizmo = TranslateGizmo()
        gizmo.enabled = False

        gizmo.begin_drag(GizmoAxis.X, 100, 100)
        assert gizmo.is_dragging is False

    def test_gizmo_drag_invisible(self):
        """Invisible gizmo cannot be dragged."""
        gizmo = TranslateGizmo()
        gizmo.visible = False

        gizmo.begin_drag(GizmoAxis.X, 100, 100)
        assert gizmo.is_dragging is False


class TestTranslateGizmo:
    """Tests for TranslateGizmo class."""

    def test_translate_creation(self):
        """TranslateGizmo should be created properly."""
        gizmo = TranslateGizmo()
        assert gizmo.gizmo_type == GizmoType.TRANSLATE
        assert gizmo.arrow_length == 1.0
        assert gizmo.sensitivity == 0.01

    def test_translate_delta_calculation(self):
        """TranslateGizmo calculates delta."""
        gizmo = TranslateGizmo()
        gizmo.begin_drag(GizmoAxis.X, 100, 100)

        delta = gizmo.update_drag(150, 100, (0, 0, 10))

        assert delta is not None
        assert delta[0] != 0  # X should have changed
        assert delta[1] == 0  # Y should be 0
        assert delta[2] == 0  # Z should be 0

    def test_translate_no_delta_without_drag(self):
        """TranslateGizmo returns None when not dragging."""
        gizmo = TranslateGizmo()

        delta = gizmo.update_drag(150, 100, (0, 0, 10))
        assert delta is None


class TestRotateGizmo:
    """Tests for RotateGizmo class."""

    def test_rotate_creation(self):
        """RotateGizmo should be created properly."""
        gizmo = RotateGizmo()
        assert gizmo.gizmo_type == GizmoType.ROTATE
        assert gizmo.radius == 1.0
        assert gizmo.sensitivity == 0.5

    def test_rotate_delta_calculation(self):
        """RotateGizmo calculates rotation delta."""
        gizmo = RotateGizmo()
        gizmo.begin_drag(GizmoAxis.Y, 100, 100)

        delta = gizmo.update_drag(150, 100, (0, 0, 10))

        assert delta is not None
        assert delta[0] == 0  # X rotation should be 0
        assert delta[1] != 0  # Y rotation should have changed
        assert delta[2] == 0  # Z rotation should be 0


class TestScaleGizmo:
    """Tests for ScaleGizmo class."""

    def test_scale_creation(self):
        """ScaleGizmo should be created properly."""
        gizmo = ScaleGizmo()
        assert gizmo.gizmo_type == GizmoType.SCALE
        assert gizmo.handle_length == 1.0
        assert gizmo.sensitivity == 0.01

    def test_scale_delta_calculation_single_axis(self):
        """ScaleGizmo calculates scale delta on single axis."""
        gizmo = ScaleGizmo()
        gizmo.begin_drag(GizmoAxis.X, 100, 100)

        delta = gizmo.update_drag(150, 100, (0, 0, 10))

        assert delta is not None
        assert delta[0] != 1.0  # X scale changed
        assert delta[1] == 1.0  # Y scale unchanged
        assert delta[2] == 1.0  # Z scale unchanged

    def test_scale_delta_calculation_uniform(self):
        """ScaleGizmo calculates uniform scale."""
        gizmo = ScaleGizmo()
        gizmo.begin_drag(GizmoAxis.XYZ, 100, 100)

        delta = gizmo.update_drag(150, 100, (0, 0, 10))

        assert delta is not None
        # All axes should scale together
        assert delta[0] == delta[1] == delta[2]


class TestUniversalGizmo:
    """Tests for UniversalGizmo class."""

    def test_universal_creation(self):
        """UniversalGizmo should be created properly."""
        gizmo = UniversalGizmo()
        assert gizmo.gizmo_type == GizmoType.UNIVERSAL
        assert gizmo.active_mode == GizmoType.TRANSLATE

    def test_universal_has_sub_gizmos(self):
        """UniversalGizmo has sub-gizmos."""
        gizmo = UniversalGizmo()

        assert gizmo.translate is not None
        assert gizmo.rotate is not None
        assert gizmo.scale_gizmo is not None

    def test_universal_set_mode(self):
        """UniversalGizmo mode can be changed."""
        gizmo = UniversalGizmo()

        gizmo.set_mode(GizmoType.ROTATE)
        assert gizmo.active_mode == GizmoType.ROTATE

        gizmo.set_mode(GizmoType.SCALE)
        assert gizmo.active_mode == GizmoType.SCALE

    def test_universal_cycle_mode(self):
        """UniversalGizmo mode can be cycled."""
        gizmo = UniversalGizmo()
        gizmo.active_mode = GizmoType.TRANSLATE

        next_mode = gizmo.cycle_mode()
        assert next_mode == GizmoType.ROTATE

        next_mode = gizmo.cycle_mode()
        assert next_mode == GizmoType.SCALE

        next_mode = gizmo.cycle_mode()
        assert next_mode == GizmoType.TRANSLATE

    def test_universal_position_propagates(self):
        """UniversalGizmo position propagates to sub-gizmos."""
        gizmo = UniversalGizmo()
        gizmo.set_position(10, 20, 30)

        assert gizmo.translate.position == (10, 20, 30)
        assert gizmo.rotate.position == (10, 20, 30)
        assert gizmo.scale_gizmo.position == (10, 20, 30)

    def test_universal_space_propagates(self):
        """UniversalGizmo space propagates to sub-gizmos."""
        gizmo = UniversalGizmo()
        gizmo.set_space(GizmoSpace.LOCAL)

        assert gizmo.translate.space == GizmoSpace.LOCAL
        assert gizmo.rotate.space == GizmoSpace.LOCAL
        assert gizmo.scale_gizmo.space == GizmoSpace.LOCAL


class TestGizmoManager:
    """Tests for GizmoManager class."""

    def test_manager_creation(self):
        """GizmoManager should be created with translate active."""
        manager = GizmoManager()
        assert manager.active_type == GizmoType.TRANSLATE
        assert manager.active_gizmo is not None
        assert manager.space == GizmoSpace.WORLD

    def test_manager_set_gizmo_type(self):
        """Manager can change gizmo type."""
        manager = GizmoManager()

        manager.set_gizmo_type(GizmoType.ROTATE)
        assert manager.active_type == GizmoType.ROTATE
        assert manager.active_gizmo.gizmo_type == GizmoType.ROTATE

    def test_manager_set_gizmo_none(self):
        """Manager can disable gizmo."""
        manager = GizmoManager()

        manager.set_gizmo_type(GizmoType.NONE)
        assert manager.active_type == GizmoType.NONE
        assert manager.active_gizmo is None

    def test_manager_cycle_gizmo(self):
        """Manager can cycle through gizmo types."""
        manager = GizmoManager()
        manager.set_gizmo_type(GizmoType.TRANSLATE)

        next_type = manager.cycle_gizmo_type()
        assert next_type == GizmoType.ROTATE

        next_type = manager.cycle_gizmo_type()
        assert next_type == GizmoType.SCALE

        next_type = manager.cycle_gizmo_type()
        assert next_type == GizmoType.TRANSLATE

    def test_manager_set_space(self):
        """Manager can change transform space."""
        manager = GizmoManager()

        manager.set_space(GizmoSpace.LOCAL)
        assert manager.space == GizmoSpace.LOCAL

    def test_manager_cycle_space(self):
        """Manager can cycle through spaces."""
        manager = GizmoManager()
        manager.space = GizmoSpace.WORLD

        next_space = manager.cycle_space()
        assert next_space == GizmoSpace.LOCAL

    def test_manager_gizmo_type_callback(self):
        """Manager triggers callback on gizmo type change."""
        manager = GizmoManager()
        changes = []
        manager.on_gizmo_changed = lambda t: changes.append(t)

        manager.set_gizmo_type(GizmoType.ROTATE)
        assert len(changes) == 1
        assert changes[0] == GizmoType.ROTATE

    def test_manager_update_transform(self):
        """Manager updates gizmo transform."""
        manager = GizmoManager()

        manager.update_gizmo_transform((10, 20, 30), (45, 90, 0))

        assert manager.active_gizmo.position == (10, 20, 30)

    def test_manager_begin_transform(self):
        """Manager can begin transform."""
        manager = GizmoManager()

        assert manager.begin_transform(GizmoAxis.X, 100, 100) is True
        assert manager.is_transforming is True

    def test_manager_begin_transform_no_gizmo(self):
        """Manager returns False when no gizmo."""
        manager = GizmoManager()
        manager.set_gizmo_type(GizmoType.NONE)

        assert manager.begin_transform(GizmoAxis.X, 100, 100) is False

    def test_manager_update_transform_during_drag(self):
        """Manager can update transform during drag."""
        manager = GizmoManager()
        manager.begin_transform(GizmoAxis.X, 100, 100)

        delta = manager.update_transform(150, 100, (0, 0, 10))
        assert delta is not None

    def test_manager_end_transform(self):
        """Manager can end transform."""
        manager = GizmoManager()
        manager.begin_transform(GizmoAxis.X, 100, 100)

        manager.end_transform()
        assert manager.is_transforming is False

    def test_manager_cancel_transform(self):
        """Manager can cancel transform."""
        manager = GizmoManager()
        manager.begin_transform(GizmoAxis.X, 100, 100)

        manager.cancel_transform()
        assert manager.is_transforming is False

    def test_manager_snap_settings(self):
        """Manager stores snap settings."""
        manager = GizmoManager()
        manager.snap_translate = True
        manager.snap_translate_value = 0.5
        manager.snap_rotate = True
        manager.snap_rotate_value = 15.0
        manager.snap_scale = True
        manager.snap_scale_value = 0.1

        assert manager.snap_translate is True
        assert manager.snap_translate_value == 0.5

    def test_manager_get_gizmo(self):
        """Manager can get specific gizmo by type."""
        manager = GizmoManager()

        translate = manager.get_gizmo(GizmoType.TRANSLATE)
        rotate = manager.get_gizmo(GizmoType.ROTATE)
        scale = manager.get_gizmo(GizmoType.SCALE)

        assert translate is not None
        assert translate.gizmo_type == GizmoType.TRANSLATE
        assert rotate.gizmo_type == GizmoType.ROTATE
        assert scale.gizmo_type == GizmoType.SCALE


class TestGizmoSpace:
    """Tests for GizmoSpace enumeration."""

    def test_all_spaces_exist(self):
        """All required spaces exist."""
        spaces = [
            GizmoSpace.WORLD,
            GizmoSpace.LOCAL,
            GizmoSpace.VIEW,
            GizmoSpace.PARENT,
        ]

        for space in spaces:
            assert isinstance(space, GizmoSpace)


class TestGizmoType:
    """Tests for GizmoType enumeration."""

    def test_all_types_exist(self):
        """All required types exist."""
        types = [
            GizmoType.NONE,
            GizmoType.TRANSLATE,
            GizmoType.ROTATE,
            GizmoType.SCALE,
            GizmoType.UNIVERSAL,
        ]

        for gtype in types:
            assert isinstance(gtype, GizmoType)
