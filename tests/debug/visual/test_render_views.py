"""
Tests for the debug render view mode system.

Tests cover:
- RenderViewMode enum
- set_view_mode/get_view_mode convenience functions
- RenderViewManager class
- Mode change callbacks
- View mode configuration
"""

import pytest


class TestRenderViewMode:
    """Tests for RenderViewMode enum."""

    def test_basic_modes_exist(self):
        """Test that basic render view modes exist."""
        from engine.debug.visual import RenderViewMode

        assert RenderViewMode.NORMAL
        assert RenderViewMode.WIREFRAME
        assert RenderViewMode.UNLIT
        assert RenderViewMode.BASE_COLOR
        assert RenderViewMode.NORMALS
        assert RenderViewMode.ROUGHNESS
        assert RenderViewMode.METALLIC
        assert RenderViewMode.AO
        assert RenderViewMode.OVERDRAW
        assert RenderViewMode.SHADER_COMPLEXITY

    def test_additional_modes_exist(self):
        """Test that additional render view modes exist."""
        from engine.debug.visual import RenderViewMode

        assert RenderViewMode.EMISSIVE
        assert RenderViewMode.SPECULAR
        assert RenderViewMode.DIFFUSE
        assert RenderViewMode.DEPTH
        assert RenderViewMode.STENCIL
        assert RenderViewMode.MOTION_VECTORS
        assert RenderViewMode.LIGHTMAP
        assert RenderViewMode.UV_CHECKER
        assert RenderViewMode.VERTEX_COLORS
        assert RenderViewMode.LOD_COLORING
        assert RenderViewMode.MIPMAP_LEVEL


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset RenderViewManager state before and after each test."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.reset()
        yield
        RenderViewManager.reset()

    def test_set_get_view_mode(self):
        """Test set_view_mode and get_view_mode."""
        from engine.debug.visual import RenderViewMode, get_view_mode, set_view_mode

        set_view_mode(RenderViewMode.WIREFRAME)
        assert get_view_mode() == RenderViewMode.WIREFRAME

    def test_toggle_view_mode(self):
        """Test toggle_view_mode function."""
        from engine.debug.visual import RenderViewMode, get_view_mode, toggle_view_mode

        # Toggle on
        result = toggle_view_mode(RenderViewMode.UNLIT)
        assert result is True
        assert get_view_mode() == RenderViewMode.UNLIT

        # Toggle off (back to NORMAL)
        result = toggle_view_mode(RenderViewMode.UNLIT)
        assert result is False
        assert get_view_mode() == RenderViewMode.NORMAL

    def test_cycle_view_mode(self):
        """Test cycle_view_mode function."""
        from engine.debug.visual import RenderViewMode, cycle_view_mode, get_view_mode

        # Start at NORMAL
        assert get_view_mode() == RenderViewMode.NORMAL

        # Cycle forward
        new_mode = cycle_view_mode(forward=True)
        assert new_mode == RenderViewMode.WIREFRAME
        assert get_view_mode() == RenderViewMode.WIREFRAME

        # Cycle backward
        new_mode = cycle_view_mode(forward=False)
        assert new_mode == RenderViewMode.NORMAL


class TestRenderViewManager:
    """Tests for RenderViewManager class."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset RenderViewManager state before and after each test."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.reset()
        yield
        RenderViewManager.reset()

    def test_default_mode(self):
        """Test default render view mode is NORMAL."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        assert RenderViewManager.get_mode() == RenderViewMode.NORMAL

    def test_set_mode(self):
        """Test setting render view mode."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_mode(RenderViewMode.BASE_COLOR)
        assert RenderViewManager.get_mode() == RenderViewMode.BASE_COLOR

    def test_previous_mode(self):
        """Test tracking previous mode."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_mode(RenderViewMode.NORMALS)
        RenderViewManager.set_mode(RenderViewMode.ROUGHNESS)

        assert RenderViewManager.get_previous_mode() == RenderViewMode.NORMALS

    def test_toggle_mode(self):
        """Test toggle between mode and NORMAL."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        # Toggle on
        result = RenderViewManager.toggle_mode(RenderViewMode.OVERDRAW)
        assert result is True
        assert RenderViewManager.get_mode() == RenderViewMode.OVERDRAW

        # Toggle off
        result = RenderViewManager.toggle_mode(RenderViewMode.OVERDRAW)
        assert result is False
        assert RenderViewManager.get_mode() == RenderViewMode.NORMAL

    def test_cycle_mode_forward(self):
        """Test cycling modes forward."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        modes = list(RenderViewMode)
        RenderViewManager.set_mode(modes[0])

        for i in range(1, 5):
            RenderViewManager.cycle_mode(forward=True)
            assert RenderViewManager.get_mode() == modes[i]

    def test_cycle_mode_backward(self):
        """Test cycling modes backward."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        modes = list(RenderViewMode)
        RenderViewManager.set_mode(modes[3])

        RenderViewManager.cycle_mode(forward=False)
        assert RenderViewManager.get_mode() == modes[2]

    def test_cycle_mode_wraps(self):
        """Test that cycle_mode wraps around."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        modes = list(RenderViewMode)

        # Set to last mode
        RenderViewManager.set_mode(modes[-1])

        # Cycle forward should wrap to first
        RenderViewManager.cycle_mode(forward=True)
        assert RenderViewManager.get_mode() == modes[0]

        # Cycle backward from first should wrap to last
        RenderViewManager.cycle_mode(forward=False)
        assert RenderViewManager.get_mode() == modes[-1]


class TestRenderViewManagerEnable:
    """Tests for RenderViewManager enable/disable."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset RenderViewManager state before and after each test."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.reset()
        yield
        RenderViewManager.reset()

    def test_disabled_forces_normal(self):
        """Test that disabling forces NORMAL mode."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)
        RenderViewManager.set_enabled(False)

        assert RenderViewManager.get_mode() == RenderViewMode.NORMAL

    def test_disabled_prevents_mode_change(self):
        """Test that disabled state prevents mode changes."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_enabled(False)
        RenderViewManager.set_mode(RenderViewMode.NORMALS)

        # Should still be NORMAL
        assert RenderViewManager.get_mode() == RenderViewMode.NORMAL

    def test_reenable_allows_mode_change(self):
        """Test that re-enabling allows mode changes."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_enabled(False)
        RenderViewManager.set_enabled(True)
        RenderViewManager.set_mode(RenderViewMode.METALLIC)

        assert RenderViewManager.get_mode() == RenderViewMode.METALLIC

    def test_is_enabled(self):
        """Test is_enabled check."""
        from engine.debug.visual import RenderViewManager

        assert RenderViewManager.is_enabled() is True

        RenderViewManager.set_enabled(False)
        assert RenderViewManager.is_enabled() is False


class TestRenderViewManagerOverlay:
    """Tests for overlay strength setting."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset RenderViewManager state before and after each test."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.reset()
        yield
        RenderViewManager.reset()

    def test_overlay_strength_default(self):
        """Test default overlay strength."""
        from engine.debug.visual import RenderViewManager

        assert RenderViewManager.get_overlay_strength() == 1.0

    def test_set_overlay_strength(self):
        """Test setting overlay strength."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.set_overlay_strength(0.5)
        assert RenderViewManager.get_overlay_strength() == 0.5

    def test_overlay_strength_validation(self):
        """Test overlay strength validation."""
        from engine.debug.visual import RenderViewManager

        with pytest.raises(ValueError):
            RenderViewManager.set_overlay_strength(1.5)

        with pytest.raises(ValueError):
            RenderViewManager.set_overlay_strength(-0.1)


class TestRenderViewManagerCallbacks:
    """Tests for mode change callbacks."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset RenderViewManager state before and after each test."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.reset()
        yield
        RenderViewManager.reset()

    def test_register_callback(self):
        """Test registering a mode change callback."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        calls = []

        def callback(old_mode, new_mode):
            calls.append((old_mode, new_mode))

        RenderViewManager.register_callback(callback)
        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)

        assert len(calls) == 1
        assert calls[0] == (RenderViewMode.NORMAL, RenderViewMode.WIREFRAME)

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        calls = []

        def callback(old_mode, new_mode):
            calls.append(1)

        RenderViewManager.register_callback(callback)
        result = RenderViewManager.unregister_callback(callback)
        assert result is True

        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)
        assert len(calls) == 0

    def test_unregister_nonexistent_callback(self):
        """Test unregistering callback that doesn't exist."""
        from engine.debug.visual import RenderViewManager

        def callback(old_mode, new_mode):
            pass

        result = RenderViewManager.unregister_callback(callback)
        assert result is False

    def test_callback_not_called_when_same_mode(self):
        """Test that callback isn't called when mode doesn't change."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        calls = []

        def callback(old_mode, new_mode):
            calls.append(1)

        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)
        RenderViewManager.register_callback(callback)
        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)  # Same mode

        assert len(calls) == 0

    def test_callback_error_handling(self):
        """Test that callback errors don't crash the system."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        calls = []

        def bad_callback(old_mode, new_mode):
            raise RuntimeError("Callback error")

        def good_callback(old_mode, new_mode):
            calls.append(1)

        RenderViewManager.register_callback(bad_callback)
        RenderViewManager.register_callback(good_callback)

        # Should not raise
        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)

        # Good callback should still have been called
        assert len(calls) == 1


class TestRenderViewConfig:
    """Tests for RenderViewConfig and related methods."""

    @pytest.fixture(autouse=True)
    def reset_manager(self):
        """Reset RenderViewManager state before and after each test."""
        from engine.debug.visual import RenderViewManager

        RenderViewManager.reset()
        yield
        RenderViewManager.reset()

    def test_get_config(self):
        """Test getting config for a mode."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        config = RenderViewManager.get_config(RenderViewMode.WIREFRAME)

        assert config.mode == RenderViewMode.WIREFRAME
        assert config.name == "Wireframe"
        assert config.category == "Geometry"
        assert len(config.description) > 0

    def test_get_all_configs(self):
        """Test getting all configs."""
        from engine.debug.visual import RenderViewManager

        configs = RenderViewManager.get_all_configs()

        # Should have config for each mode
        assert len(configs) >= 10

        # Each config should have required fields
        for config in configs:
            assert config.mode is not None
            assert config.name is not None
            assert config.category is not None

    def test_get_modes_by_category(self):
        """Test getting modes by category."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        geometry_modes = RenderViewManager.get_modes_by_category("Geometry")
        assert RenderViewMode.WIREFRAME in geometry_modes

        material_modes = RenderViewManager.get_modes_by_category("Material")
        assert RenderViewMode.BASE_COLOR in material_modes
        assert RenderViewMode.NORMALS in material_modes

    def test_get_categories(self):
        """Test getting all categories."""
        from engine.debug.visual import RenderViewManager

        categories = RenderViewManager.get_categories()

        assert "Standard" in categories
        assert "Geometry" in categories
        assert "Material" in categories
        assert "Lighting" in categories
        assert "Performance" in categories

    def test_is_gbuffer_required(self):
        """Test checking if mode requires GBuffer."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        # NORMAL doesn't require GBuffer
        assert RenderViewManager.is_gbuffer_required(RenderViewMode.NORMAL) is False

        # Material modes require GBuffer
        assert RenderViewManager.is_gbuffer_required(RenderViewMode.NORMALS) is True
        assert RenderViewManager.is_gbuffer_required(RenderViewMode.BASE_COLOR) is True

    def test_is_gbuffer_required_current_mode(self):
        """Test checking if current mode requires GBuffer."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_mode(RenderViewMode.ROUGHNESS)
        assert RenderViewManager.is_gbuffer_required() is True

        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)
        assert RenderViewManager.is_gbuffer_required() is False

    def test_get_mode_info(self):
        """Test getting mode info."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        RenderViewManager.set_mode(RenderViewMode.OVERDRAW)
        info = RenderViewManager.get_mode_info()

        assert info["mode"] == "OVERDRAW"
        assert info["name"] == "Overdraw"
        assert info["category"] == "Performance"
        assert info["is_current"] is True

    def test_get_mode_info_specific_mode(self):
        """Test getting info for specific mode."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        info = RenderViewManager.get_mode_info(RenderViewMode.DEPTH)

        assert info["mode"] == "DEPTH"
        assert "description" in info
        assert "requires_gbuffer" in info


class TestRenderViewManagerReset:
    """Tests for reset functionality."""

    def test_reset(self):
        """Test resetting to default state."""
        from engine.debug.visual import RenderViewManager, RenderViewMode

        # Change various settings
        RenderViewManager.set_mode(RenderViewMode.WIREFRAME)
        RenderViewManager.set_overlay_strength(0.3)
        RenderViewManager.set_enabled(False)
        RenderViewManager.register_callback(lambda o, n: None)

        # Reset
        RenderViewManager.reset()

        # Verify defaults restored
        assert RenderViewManager.get_mode() == RenderViewMode.NORMAL
        assert RenderViewManager.get_overlay_strength() == 1.0
        assert RenderViewManager.is_enabled() is True
