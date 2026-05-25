"""
Tests for the debug overlay system.

Tests cover:
- OverlayType enum and combinations
- Enable/disable overlay types
- Overlay callbacks registration
- Per-overlay settings
- Global enable/disable
"""

import pytest


class TestOverlayType:
    """Tests for OverlayType enum."""

    def test_overlay_types_exist(self):
        """Test that all expected overlay types exist."""
        from engine.debug.visual import OverlayType

        assert OverlayType.PHYSICS
        assert OverlayType.NAVIGATION
        assert OverlayType.RENDERING
        assert OverlayType.AI
        assert OverlayType.AUDIO
        assert OverlayType.NETWORK
        assert OverlayType.ANIMATION
        assert OverlayType.PARTICLES

    def test_overlay_type_combinations(self):
        """Test that overlay types can be combined with bitwise OR."""
        from engine.debug.visual import OverlayType

        combined = OverlayType.PHYSICS | OverlayType.AI
        assert OverlayType.PHYSICS in combined
        assert OverlayType.AI in combined
        assert OverlayType.AUDIO not in combined

    def test_overlay_type_all(self):
        """Test ALL combination contains all types."""
        from engine.debug.visual import OverlayType

        assert OverlayType.PHYSICS in OverlayType.ALL
        assert OverlayType.NAVIGATION in OverlayType.ALL
        assert OverlayType.RENDERING in OverlayType.ALL
        assert OverlayType.AI in OverlayType.ALL
        assert OverlayType.AUDIO in OverlayType.ALL
        assert OverlayType.NETWORK in OverlayType.ALL

    def test_overlay_type_gameplay(self):
        """Test GAMEPLAY convenience combination."""
        from engine.debug.visual import OverlayType

        assert OverlayType.PHYSICS in OverlayType.GAMEPLAY
        assert OverlayType.AI in OverlayType.GAMEPLAY
        assert OverlayType.ANIMATION in OverlayType.GAMEPLAY

    def test_overlay_type_graphics(self):
        """Test GRAPHICS convenience combination."""
        from engine.debug.visual import OverlayType

        assert OverlayType.RENDERING in OverlayType.GRAPHICS
        assert OverlayType.PARTICLES in OverlayType.GRAPHICS
        assert OverlayType.CULLING in OverlayType.GRAPHICS
        assert OverlayType.LOD in OverlayType.GRAPHICS


class TestDebugOverlay:
    """Tests for DebugOverlay static class."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_enable_overlay(self):
        """Test enabling an overlay type."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS)
        assert DebugOverlay.is_enabled(OverlayType.PHYSICS)

    def test_disable_overlay(self):
        """Test disabling an overlay type."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.disable(OverlayType.PHYSICS)
        assert not DebugOverlay.is_enabled(OverlayType.PHYSICS)

    def test_toggle_overlay(self):
        """Test toggling an overlay type."""
        from engine.debug.visual import DebugOverlay, OverlayType

        assert not DebugOverlay.is_enabled(OverlayType.AI)

        result = DebugOverlay.toggle(OverlayType.AI)
        assert result is True
        assert DebugOverlay.is_enabled(OverlayType.AI)

        result = DebugOverlay.toggle(OverlayType.AI)
        assert result is False
        assert not DebugOverlay.is_enabled(OverlayType.AI)

    def test_enable_multiple_overlays(self):
        """Test enabling multiple overlay types at once."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS | OverlayType.AI)
        assert DebugOverlay.is_enabled(OverlayType.PHYSICS)
        assert DebugOverlay.is_enabled(OverlayType.AI)

    def test_get_enabled_overlays(self):
        """Test getting set of enabled overlays."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.enable(OverlayType.NAVIGATION)

        enabled = DebugOverlay.get_enabled_overlays()
        assert OverlayType.PHYSICS in enabled
        assert OverlayType.NAVIGATION in enabled
        assert OverlayType.AI not in enabled

    def test_enable_all(self):
        """Test enabling all overlay types."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable_all()

        for overlay_type in [
            OverlayType.PHYSICS,
            OverlayType.NAVIGATION,
            OverlayType.RENDERING,
            OverlayType.AI,
            OverlayType.AUDIO,
            OverlayType.NETWORK,
        ]:
            assert DebugOverlay.is_enabled(overlay_type)

    def test_disable_all(self):
        """Test disabling all overlay types."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable_all()
        DebugOverlay.disable_all()

        assert len(DebugOverlay.get_enabled_overlays()) == 0


class TestDebugOverlayGlobal:
    """Tests for global overlay enable/disable."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_global_disable(self):
        """Test that global disable prevents all overlays."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.set_global_enabled(False)

        # Individual overlay is enabled, but global is disabled
        assert not DebugOverlay.is_enabled(OverlayType.PHYSICS)

    def test_global_reenable(self):
        """Test that global re-enable restores overlays."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.set_global_enabled(False)
        DebugOverlay.set_global_enabled(True)

        assert DebugOverlay.is_enabled(OverlayType.PHYSICS)

    def test_global_enabled_status(self):
        """Test global enabled status check."""
        from engine.debug.visual import DebugOverlay

        assert DebugOverlay.is_global_enabled()

        DebugOverlay.set_global_enabled(False)
        assert not DebugOverlay.is_global_enabled()

    def test_get_enabled_when_global_disabled(self):
        """Test get_enabled_overlays returns empty when global disabled."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.set_global_enabled(False)

        enabled = DebugOverlay.get_enabled_overlays()
        assert len(enabled) == 0


class TestDebugOverlayOpacity:
    """Tests for overlay opacity settings."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_set_overlay_opacity(self):
        """Test setting per-overlay opacity."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.set_opacity(OverlayType.PHYSICS, 0.5)
        opacity = DebugOverlay.get_opacity(OverlayType.PHYSICS)
        assert opacity == 0.5

    def test_set_global_opacity(self):
        """Test setting global opacity multiplier."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.set_global_opacity(0.5)
        assert DebugOverlay.get_global_opacity() == 0.5

        # Per-overlay opacity is multiplied by global
        DebugOverlay.set_opacity(OverlayType.PHYSICS, 0.8)
        effective = DebugOverlay.get_opacity(OverlayType.PHYSICS)
        assert abs(effective - 0.4) < 0.001  # 0.8 * 0.5 = 0.4

    def test_opacity_validation(self):
        """Test opacity value validation."""
        from engine.debug.visual import DebugOverlay, OverlayType

        with pytest.raises(ValueError):
            DebugOverlay.set_opacity(OverlayType.PHYSICS, 1.5)

        with pytest.raises(ValueError):
            DebugOverlay.set_opacity(OverlayType.PHYSICS, -0.1)

        with pytest.raises(ValueError):
            DebugOverlay.set_global_opacity(2.0)


class TestDebugOverlayCallbacks:
    """Tests for overlay render callbacks."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_register_callback(self):
        """Test registering a render callback."""
        from engine.debug.visual import DebugOverlay, OverlayType

        calls = []

        def callback(world, camera, dt):
            calls.append((world, camera, dt))

        DebugOverlay.register_callback(OverlayType.PHYSICS, callback)
        DebugOverlay.enable(OverlayType.PHYSICS)
        DebugOverlay.render("world", "camera", 0.016)

        assert len(calls) == 1
        assert calls[0] == ("world", "camera", 0.016)

    def test_unregister_callback(self):
        """Test unregistering a callback."""
        from engine.debug.visual import DebugOverlay, OverlayType

        calls = []

        def callback(world, camera, dt):
            calls.append(dt)

        DebugOverlay.register_callback(OverlayType.PHYSICS, callback)
        DebugOverlay.enable(OverlayType.PHYSICS)

        result = DebugOverlay.unregister_callback(OverlayType.PHYSICS, callback)
        assert result is True

        DebugOverlay.render("world", "camera", 0.016)
        assert len(calls) == 0

    def test_unregister_nonexistent_callback(self):
        """Test unregistering a callback that doesn't exist."""
        from engine.debug.visual import DebugOverlay, OverlayType

        def callback(world, camera, dt):
            pass

        result = DebugOverlay.unregister_callback(OverlayType.PHYSICS, callback)
        assert result is False

    def test_clear_callbacks(self):
        """Test clearing all callbacks for an overlay."""
        from engine.debug.visual import DebugOverlay, OverlayType

        calls = []

        DebugOverlay.register_callback(OverlayType.PHYSICS, lambda w, c, d: calls.append(1))
        DebugOverlay.register_callback(OverlayType.PHYSICS, lambda w, c, d: calls.append(2))
        DebugOverlay.enable(OverlayType.PHYSICS)

        DebugOverlay.clear_callbacks(OverlayType.PHYSICS)
        DebugOverlay.render("world", "camera", 0.016)

        assert len(calls) == 0

    def test_callback_not_called_when_disabled(self):
        """Test that callbacks are not called for disabled overlays."""
        from engine.debug.visual import DebugOverlay, OverlayType

        calls = []

        def callback(world, camera, dt):
            calls.append(dt)

        DebugOverlay.register_callback(OverlayType.PHYSICS, callback)
        # Don't enable the overlay
        DebugOverlay.render("world", "camera", 0.016)

        assert len(calls) == 0

    def test_multiple_callbacks_same_overlay(self):
        """Test multiple callbacks on same overlay."""
        from engine.debug.visual import DebugOverlay, OverlayType

        calls = []

        DebugOverlay.register_callback(OverlayType.AI, lambda w, c, d: calls.append("a"))
        DebugOverlay.register_callback(OverlayType.AI, lambda w, c, d: calls.append("b"))
        DebugOverlay.enable(OverlayType.AI)
        DebugOverlay.render("world", "camera", 0.016)

        assert "a" in calls
        assert "b" in calls

    def test_callback_error_handling(self):
        """Test that callback errors don't crash the system."""
        from engine.debug.visual import DebugOverlay, OverlayType

        calls = []

        def bad_callback(world, camera, dt):
            raise RuntimeError("Callback error")

        def good_callback(world, camera, dt):
            calls.append("good")

        DebugOverlay.register_callback(OverlayType.PHYSICS, bad_callback)
        DebugOverlay.register_callback(OverlayType.PHYSICS, good_callback)
        DebugOverlay.enable(OverlayType.PHYSICS)

        # Should not raise, should continue to good callback
        DebugOverlay.render("world", "camera", 0.016)
        assert "good" in calls


class TestDebugOverlayPriority:
    """Tests for overlay render priority."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_set_priority(self):
        """Test setting overlay priority."""
        from engine.debug.visual import DebugOverlay, OverlayType

        DebugOverlay.set_priority(OverlayType.PHYSICS, 10)
        info = DebugOverlay.get_overlay_info(OverlayType.PHYSICS)
        assert info["priority"] == 10

    def test_priority_affects_render_order(self):
        """Test that priority affects render order."""
        from engine.debug.visual import DebugOverlay, OverlayType

        order = []

        DebugOverlay.register_callback(OverlayType.PHYSICS, lambda w, c, d: order.append("physics"))
        DebugOverlay.register_callback(OverlayType.AI, lambda w, c, d: order.append("ai"))

        DebugOverlay.set_priority(OverlayType.PHYSICS, 10)  # High priority
        DebugOverlay.set_priority(OverlayType.AI, 5)  # Low priority

        DebugOverlay.enable(OverlayType.PHYSICS | OverlayType.AI)
        DebugOverlay.render("world", "camera", 0.016)

        # AI should be rendered first (lower priority), then Physics
        assert order == ["ai", "physics"]


class TestOverlaySettings:
    """Tests for per-overlay-type settings."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_physics_settings(self):
        """Test physics overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_physics_settings()
        assert hasattr(settings, "show_collision_shapes")
        assert hasattr(settings, "show_contact_points")
        assert hasattr(settings, "show_joints")
        assert hasattr(settings, "show_raycasts")
        assert hasattr(settings, "show_velocities")

    def test_navigation_settings(self):
        """Test navigation overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_navigation_settings()
        assert hasattr(settings, "show_navmesh")
        assert hasattr(settings, "show_paths")
        assert hasattr(settings, "show_off_mesh_links")
        assert hasattr(settings, "show_agents")

    def test_ai_settings(self):
        """Test AI overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_ai_settings()
        assert hasattr(settings, "show_perception_cones")
        assert hasattr(settings, "show_behavior_tree")
        assert hasattr(settings, "show_blackboard")

    def test_rendering_settings(self):
        """Test rendering overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_rendering_settings()
        assert hasattr(settings, "show_bounds")
        assert hasattr(settings, "show_wireframe")
        assert hasattr(settings, "show_normals")

    def test_audio_settings(self):
        """Test audio overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_audio_settings()
        assert hasattr(settings, "show_sound_positions")
        assert hasattr(settings, "show_attenuation_spheres")
        assert hasattr(settings, "show_active_voices")

    def test_network_settings(self):
        """Test network overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_network_settings()
        assert hasattr(settings, "show_replication_status")
        assert hasattr(settings, "show_ownership")
        assert hasattr(settings, "show_bandwidth")

    def test_modify_settings(self):
        """Test modifying overlay settings."""
        from engine.debug.visual import DebugOverlay

        settings = DebugOverlay.get_physics_settings()
        original = settings.show_velocities
        settings.show_velocities = not original

        # Verify the change persists
        settings2 = DebugOverlay.get_physics_settings()
        assert settings2.show_velocities == (not original)


class TestOverlayInfo:
    """Tests for overlay information retrieval."""

    @pytest.fixture(autouse=True)
    def reset_overlay(self):
        """Reset DebugOverlay state before and after each test."""
        from engine.debug.visual import DebugOverlay

        DebugOverlay.reset()
        yield
        DebugOverlay.reset()

    def test_get_overlay_info(self):
        """Test getting info about an overlay type."""
        from engine.debug.visual import DebugOverlay, OverlayType

        info = DebugOverlay.get_overlay_info(OverlayType.PHYSICS)

        assert "type" in info
        assert "enabled" in info
        assert "opacity" in info
        assert "priority" in info
        assert "category" in info
        assert "description" in info
        assert "callback_count" in info

        assert info["type"] == "PHYSICS"
        assert info["category"] == "Simulation"

    def test_get_all_overlay_info(self):
        """Test getting info about all overlay types."""
        from engine.debug.visual import DebugOverlay

        all_info = DebugOverlay.get_all_overlay_info()

        # Should have info for each individual overlay type
        # (excluding NONE, ALL, GAMEPLAY, GRAPHICS combinations)
        assert len(all_info) >= 10  # At least the base types

        # Each entry should have required fields
        for info in all_info:
            assert "type" in info
            assert "enabled" in info
            assert "category" in info
