"""
Tests for XR UI Panel (panel.py).

Tests the XRUIPanel component and related classes:
    XRUIPanel, XRPanelConfig, @xr_ui_panel decorator, UIInteractionManager

Each test verifies:
1. Panel creation and configuration
2. Decorator application and metadata
3. Interaction modes (ray, poke, gaze)
4. Coordinate transformations
"""

import sys
from pathlib import Path

import pytest

# Add engine to path for direct imports
engine_path = Path(__file__).parents[3]
if str(engine_path) not in sys.path:
    sys.path.insert(0, str(engine_path))

from engine.xr.ui.panel import (
    XRUIPanel,
    XRPanelConfig,
    XRPanelType,
    XRInteractionMode,
    xr_ui_panel,
    UIInteractionManager,
    RaycastHit,
    PokeInteraction,
    GazeInteraction,
)


# =============================================================================
# XRPanelConfig
# =============================================================================


class TestXRPanelConfig:
    def test_default_values(self):
        config = XRPanelConfig()
        assert config.width == 1.0
        assert config.height == 0.75
        assert config.pixels_per_meter == 1000.0
        assert config.curved is False
        assert config.billboard is False

    def test_custom_values(self):
        config = XRPanelConfig(
            width=2.0,
            height=1.5,
            pixels_per_meter=500.0,
            curved=True,
            curve_radius=3.0,
        )
        assert config.width == 2.0
        assert config.height == 1.5
        assert config.curved is True
        assert config.curve_radius == 3.0

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="width must be positive"):
            XRPanelConfig(width=0)

    def test_invalid_height(self):
        with pytest.raises(ValueError, match="height must be positive"):
            XRPanelConfig(height=-1)

    def test_invalid_pixels_per_meter(self):
        with pytest.raises(ValueError, match="Pixels per meter must be positive"):
            XRPanelConfig(pixels_per_meter=0)

    def test_invalid_curve_radius(self):
        with pytest.raises(ValueError, match="Curve radius must be positive"):
            XRPanelConfig(curved=True, curve_radius=0)

    def test_interaction_modes(self):
        config = XRPanelConfig(
            interaction_modes=(XRInteractionMode.RAY, XRInteractionMode.POKE)
        )
        assert XRInteractionMode.RAY in config.interaction_modes
        assert XRInteractionMode.POKE in config.interaction_modes


# =============================================================================
# XRUIPanel
# =============================================================================


class TestXRUIPanel:
    def test_default_creation(self):
        panel = XRUIPanel()
        assert panel.panel_type == XRPanelType.WORLD
        assert panel.is_visible is True
        assert panel.is_interactable is True
        assert panel.is_hovered is False

    def test_panel_types(self):
        for panel_type in XRPanelType:
            panel = XRUIPanel(panel_type=panel_type)
            assert panel.panel_type == panel_type

    def test_position_and_orientation(self):
        panel = XRUIPanel(
            position=(1.0, 2.0, 3.0),
            orientation=(0.0, 0.707, 0.0, 0.707)
        )
        assert panel.position == (1.0, 2.0, 3.0)
        assert panel.orientation == (0.0, 0.707, 0.0, 0.707)

    def test_set_position(self):
        panel = XRUIPanel()
        panel.set_position(5.0, 6.0, 7.0)
        assert panel.position == (5.0, 6.0, 7.0)
        assert panel._dirty is True

    def test_pixel_dimensions(self):
        panel = XRUIPanel(config=XRPanelConfig(
            width=0.5,
            height=0.3,
            pixels_per_meter=1000.0
        ))
        assert panel.pixel_width == 500
        assert panel.pixel_height == 300

    def test_visibility_toggle(self):
        panel = XRUIPanel()
        assert panel.is_visible is True

        panel.hide()
        assert panel.is_visible is False

        panel.show()
        assert panel.is_visible is True

        panel.toggle()
        assert panel.is_visible is False

    def test_supports_interaction_mode(self):
        config = XRPanelConfig(
            interaction_modes=(XRInteractionMode.RAY, XRInteractionMode.GAZE)
        )
        panel = XRUIPanel(config=config)

        assert panel.supports_interaction_mode(XRInteractionMode.RAY) is True
        assert panel.supports_interaction_mode(XRInteractionMode.GAZE) is True
        assert panel.supports_interaction_mode(XRInteractionMode.POKE) is False

    def test_add_child(self):
        panel = XRUIPanel()
        child = object()

        panel.add_child(child)
        assert child in panel._children
        assert panel._dirty is True

    def test_remove_child(self):
        panel = XRUIPanel()
        child = object()

        panel.add_child(child)
        result = panel.remove_child(child)

        assert result is True
        assert child not in panel._children

    def test_remove_nonexistent_child(self):
        panel = XRUIPanel()
        result = panel.remove_child(object())
        assert result is False

    def test_world_to_panel_hit(self):
        panel = XRUIPanel(
            position=(0.0, 1.0, 0.0),
            config=XRPanelConfig(width=1.0, height=1.0)
        )

        # Point at center of panel
        uv = panel.world_to_panel((0.0, 1.0, 0.0))
        assert uv is not None
        assert uv == (0.5, 0.5)

    def test_world_to_panel_miss(self):
        panel = XRUIPanel(
            position=(0.0, 1.0, 0.0),
            config=XRPanelConfig(width=1.0, height=1.0)
        )

        # Point outside panel
        uv = panel.world_to_panel((10.0, 1.0, 0.0))
        assert uv is None

    def test_panel_to_world(self):
        panel = XRUIPanel(
            position=(0.0, 1.0, 0.0),
            config=XRPanelConfig(width=1.0, height=1.0)
        )

        # Center UV
        world_pos = panel.panel_to_world((0.5, 0.5))
        assert world_pos == (0.0, 1.0, 0.0)


# =============================================================================
# @xr_ui_panel decorator
# =============================================================================


class TestXRUIPanelDecorator:
    def test_basic_application(self):
        @xr_ui_panel()
        class TestPanel:
            pass

        assert TestPanel._xr_ui_panel is True
        assert TestPanel._panel_type == XRPanelType.WORLD
        assert TestPanel._interaction_mode == XRInteractionMode.RAY

    def test_custom_panel_type(self):
        @xr_ui_panel(panel_type="head_locked")
        class TestPanel:
            pass

        assert TestPanel._panel_type == XRPanelType.HEAD_LOCKED

    def test_all_panel_types(self):
        types = ["world", "head_locked", "hand_attached", "wrist"]
        expected = [
            XRPanelType.WORLD,
            XRPanelType.HEAD_LOCKED,
            XRPanelType.HAND_ATTACHED,
            XRPanelType.WRIST,
        ]

        for type_str, expected_type in zip(types, expected):
            @xr_ui_panel(panel_type=type_str)
            class TestPanel:
                pass

            assert TestPanel._panel_type == expected_type

    def test_custom_interaction_mode(self):
        @xr_ui_panel(interaction_mode="poke")
        class TestPanel:
            pass

        assert TestPanel._interaction_mode == XRInteractionMode.POKE

    def test_all_interaction_modes(self):
        modes = ["ray", "poke", "gaze", "grab"]
        expected = [
            XRInteractionMode.RAY,
            XRInteractionMode.POKE,
            XRInteractionMode.GAZE,
            XRInteractionMode.GRAB,
        ]

        for mode_str, expected_mode in zip(modes, expected):
            @xr_ui_panel(interaction_mode=mode_str)
            class TestPanel:
                pass

            assert TestPanel._interaction_mode == expected_mode

    def test_custom_dimensions(self):
        @xr_ui_panel(width=2.0, height=1.5)
        class TestPanel:
            pass

        assert TestPanel._panel_width == 2.0
        assert TestPanel._panel_height == 1.5

    def test_curved_billboard(self):
        @xr_ui_panel(curved=True, billboard=True)
        class TestPanel:
            pass

        assert TestPanel._panel_curved is True
        assert TestPanel._panel_billboard is True

    def test_invalid_panel_type(self):
        with pytest.raises(ValueError, match="Invalid panel_type"):
            @xr_ui_panel(panel_type="invalid")
            class TestPanel:
                pass

    def test_invalid_interaction_mode(self):
        with pytest.raises(ValueError, match="Invalid interaction_mode"):
            @xr_ui_panel(interaction_mode="invalid")
            class TestPanel:
                pass

    def test_invalid_width(self):
        with pytest.raises(ValueError, match="Width must be positive"):
            @xr_ui_panel(width=0)
            class TestPanel:
                pass

    def test_invalid_height(self):
        with pytest.raises(ValueError, match="Height must be positive"):
            @xr_ui_panel(height=-1)
            class TestPanel:
                pass

    def test_tags(self):
        @xr_ui_panel(panel_type="wrist", interaction_mode="gaze")
        class TestPanel:
            pass

        assert TestPanel._tags["xr_ui_panel"] is True
        assert TestPanel._tags["panel_type"] == "wrist"
        assert TestPanel._tags["interaction_mode"] == "gaze"

    def test_applied_decorators(self):
        @xr_ui_panel()
        class TestPanel:
            pass

        assert "xr_ui_panel" in TestPanel._applied_decorators

    def test_registries(self):
        @xr_ui_panel()
        class TestPanel:
            pass

        assert "xr" in TestPanel._registries


# =============================================================================
# UIInteractionManager
# =============================================================================


class TestUIInteractionManager:
    def test_creation(self):
        manager = UIInteractionManager()
        assert manager._dwell_threshold == 1.0

    def test_custom_dwell_threshold(self):
        manager = UIInteractionManager(dwell_threshold=2.0)
        assert manager._dwell_threshold == 2.0

    def test_register_panel(self):
        manager = UIInteractionManager()
        panel = XRUIPanel()

        manager.register_panel(panel)
        assert panel in manager._panels

    def test_unregister_panel(self):
        manager = UIInteractionManager()
        panel = XRUIPanel()

        manager.register_panel(panel)
        manager.unregister_panel(panel)
        assert panel not in manager._panels

    def test_raycast_hit(self):
        manager = UIInteractionManager()
        panel = XRUIPanel(
            position=(0.0, 1.0, -2.0),
            config=XRPanelConfig(
                width=1.0,
                height=1.0,
                interaction_modes=(XRInteractionMode.RAY,)
            )
        )
        manager.register_panel(panel)

        # Ray pointing at panel
        hit = manager.raycast(
            origin=(0.0, 1.0, 0.0),
            direction=(0.0, 0.0, -1.0),
            interactor_id=1
        )

        assert hit is not None
        assert hit.panel == panel
        assert panel.is_hovered is True

    def test_raycast_miss(self):
        manager = UIInteractionManager()
        panel = XRUIPanel(
            position=(0.0, 1.0, -2.0),
            config=XRPanelConfig(
                width=1.0,
                height=1.0,
                interaction_modes=(XRInteractionMode.RAY,)
            )
        )
        manager.register_panel(panel)

        # Ray pointing away from panel
        hit = manager.raycast(
            origin=(0.0, 1.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            interactor_id=1
        )

        assert hit is None

    def test_raycast_invisible_panel(self):
        manager = UIInteractionManager()
        panel = XRUIPanel(
            position=(0.0, 1.0, -2.0),
            is_visible=False,
        )
        manager.register_panel(panel)

        hit = manager.raycast(
            origin=(0.0, 1.0, 0.0),
            direction=(0.0, 0.0, -1.0),
            interactor_id=1
        )

        assert hit is None

    def test_poke_interaction(self):
        manager = UIInteractionManager()
        panel = XRUIPanel(
            position=(0.0, 1.0, 0.0),
            config=XRPanelConfig(
                width=1.0,
                height=1.0,
                interaction_modes=(XRInteractionMode.POKE,)
            )
        )
        manager.register_panel(panel)

        # Finger touching panel
        poke = manager.poke(
            finger_position=(0.0, 1.0, 0.01),
            finger_id=1
        )

        assert poke is not None
        assert poke.panel == panel

    def test_gaze_interaction(self):
        manager = UIInteractionManager(dwell_threshold=1.0)
        panel = XRUIPanel(
            position=(0.0, 1.0, -2.0),
            config=XRPanelConfig(
                width=1.0,
                height=1.0,
                interaction_modes=(XRInteractionMode.GAZE,)
            )
        )
        manager.register_panel(panel)

        # Gaze at panel
        gaze = manager.gaze(
            gaze_origin=(0.0, 1.0, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            delta_time=0.5,
            user_id=1
        )

        assert gaze is not None
        assert gaze.panel == panel
        assert gaze.dwell_time == 0.5
        assert gaze.is_fixating is False

    def test_gaze_dwell_accumulation(self):
        manager = UIInteractionManager(dwell_threshold=1.0)
        panel = XRUIPanel(
            position=(0.0, 1.0, -2.0),
            config=XRPanelConfig(
                width=1.0,
                height=1.0,
                interaction_modes=(XRInteractionMode.GAZE,)
            )
        )
        manager.register_panel(panel)

        # First gaze
        manager.gaze(
            gaze_origin=(0.0, 1.0, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            delta_time=0.5,
            user_id=1
        )

        # Second gaze - should accumulate
        gaze = manager.gaze(
            gaze_origin=(0.0, 1.0, 0.0),
            gaze_direction=(0.0, 0.0, -1.0),
            delta_time=0.6,
            user_id=1
        )

        assert gaze.dwell_time == 1.1
        assert gaze.is_fixating is True

    def test_clear(self):
        manager = UIInteractionManager()
        panel = XRUIPanel()
        manager.register_panel(panel)

        # Create some state
        manager._active_rays[1] = RaycastHit(
            panel=panel,
            hit_point=(0, 0, 0),
            uv=(0.5, 0.5),
            distance=1.0
        )
        panel.is_hovered = True

        manager.clear()

        assert len(manager._active_rays) == 0
        assert panel.is_hovered is False
