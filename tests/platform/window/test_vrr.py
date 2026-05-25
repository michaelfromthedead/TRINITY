"""Tests for Variable Refresh Rate (VRR) support."""

import pytest

from engine.platform.window import VariableRefresh, VRRType, RefreshRange


class TestVRRSupport:
    """Tests for VRR support detection."""

    def test_vrr_not_supported_by_default(self):
        """Test that VRR is not supported by default in headless mode."""
        vrr = VariableRefresh()
        assert vrr.is_supported() is False

    def test_vrr_manager_creation(self):
        """Test creating VRR manager."""
        vrr = VariableRefresh()
        assert vrr is not None

    def test_vrr_simulation(self):
        """Test VRR manager with simulation."""
        vrr = VariableRefresh(simulate_vrr=True)
        assert vrr.supported is True


class TestVRRProperties:
    """Tests for VRR properties."""

    def test_default_vrr_disabled(self):
        """Test that VRR is disabled by default."""
        vrr = VariableRefresh(simulate_vrr=True)
        assert vrr.enabled is False

    def test_vrr_type_without_support(self):
        """Test VRR type when not supported."""
        vrr = VariableRefresh(simulate_vrr=False)
        assert vrr.vrr_type == VRRType.NONE

    def test_vrr_type_with_support(self):
        """Test VRR type when supported."""
        vrr = VariableRefresh(simulate_vrr=True)
        assert vrr.vrr_type in [
            VRRType.FREESYNC,
            VRRType.GSYNC,
            VRRType.GSYNC_COMPATIBLE,
            VRRType.HDMI_VRR,
            VRRType.ADAPTIVE_SYNC,
        ]

    def test_supported_property(self):
        """Test supported property."""
        vrr_unsupported = VariableRefresh(simulate_vrr=False)
        assert vrr_unsupported.supported is False

        vrr_supported = VariableRefresh(simulate_vrr=True)
        assert vrr_supported.supported is True


class TestVRREnable:
    """Tests for enabling/disabling VRR."""

    def test_enable_vrr_with_support(self):
        """Test enabling VRR when supported."""
        vrr = VariableRefresh(simulate_vrr=True)
        result = vrr.enable(True)
        assert result is True
        assert vrr.enabled is True

    def test_disable_vrr(self):
        """Test disabling VRR."""
        vrr = VariableRefresh(simulate_vrr=True)
        vrr.enable(True)
        result = vrr.enable(False)
        assert result is True
        assert vrr.enabled is False

    def test_enable_vrr_without_support(self):
        """Test that VRR cannot be enabled without support."""
        vrr = VariableRefresh(simulate_vrr=False)
        result = vrr.enable(True)
        assert result is False
        assert vrr.enabled is False

    def test_disable_vrr_without_support(self):
        """Test that VRR can be disabled even without support."""
        vrr = VariableRefresh(simulate_vrr=False)
        result = vrr.enable(False)
        assert result is True
        assert vrr.enabled is False

    def test_toggle_vrr(self):
        """Test toggling VRR state."""
        vrr = VariableRefresh(simulate_vrr=True)
        vrr.enable(True)
        assert vrr.enabled is True
        vrr.enable(False)
        assert vrr.enabled is False
        vrr.enable(True)
        assert vrr.enabled is True


class TestRefreshRange:
    """Tests for refresh rate range."""

    def test_get_range_without_support(self):
        """Test getting refresh range without VRR support."""
        vrr = VariableRefresh(simulate_vrr=False)
        range_info = vrr.get_range()
        assert isinstance(range_info, RefreshRange)
        assert range_info.min_hz == range_info.max_hz  # Fixed rate

    def test_get_range_with_support(self):
        """Test getting refresh range with VRR support."""
        vrr = VariableRefresh(simulate_vrr=True)
        range_info = vrr.get_range()
        assert isinstance(range_info, RefreshRange)
        assert range_info.min_hz > 0
        assert range_info.max_hz > range_info.min_hz

    def test_refresh_range_validity(self):
        """Test that refresh range values are valid."""
        vrr = VariableRefresh(simulate_vrr=True)
        range_info = vrr.get_range()
        assert range_info.min_hz >= 30  # Reasonable minimum
        assert range_info.max_hz <= 360  # Reasonable maximum
        assert range_info.min_hz < range_info.max_hz


class TestRefreshRangeDataclass:
    """Tests for RefreshRange dataclass."""

    def test_create_refresh_range(self):
        """Test creating refresh range."""
        range_info = RefreshRange(min_hz=48, max_hz=144)
        assert range_info.min_hz == 48
        assert range_info.max_hz == 144

    def test_refresh_range_values(self):
        """Test various refresh range values."""
        # Standard FreeSync range
        freesync = RefreshRange(min_hz=48, max_hz=144)
        assert freesync.min_hz == 48
        assert freesync.max_hz == 144

        # Wide range
        wide = RefreshRange(min_hz=30, max_hz=240)
        assert wide.min_hz == 30
        assert wide.max_hz == 240

        # Fixed rate (no VRR)
        fixed = RefreshRange(min_hz=60, max_hz=60)
        assert fixed.min_hz == fixed.max_hz


class TestVRRTypeEnum:
    """Tests for VRRType enum."""

    def test_vrr_enable_disable_workflow(self):
        """Test the full VRR enable/disable/query workflow."""
        # Supported display
        vrr_supported = VariableRefresh(simulate_vrr=True)

        # Initially disabled
        assert vrr_supported.enabled is False
        assert vrr_supported.vrr_type != VRRType.NONE

        # Enable VRR
        result = vrr_supported.enable(True)
        assert result is True
        assert vrr_supported.enabled is True

        # Query range while enabled
        range_info = vrr_supported.get_range()
        assert range_info.max_hz > range_info.min_hz

        # Disable VRR
        result = vrr_supported.enable(False)
        assert result is True
        assert vrr_supported.enabled is False

        # Unsupported display
        vrr_unsupported = VariableRefresh(simulate_vrr=False)

        # Should be NONE type
        assert vrr_unsupported.vrr_type == VRRType.NONE

        # Cannot enable
        result = vrr_unsupported.enable(True)
        assert result is False
        assert vrr_unsupported.enabled is False

        # Can disable (no-op)
        result = vrr_unsupported.enable(False)
        assert result is True

    def test_vrr_type_consistency(self):
        """Test that VRR type is consistent with support status."""
        vrr_supported = VariableRefresh(simulate_vrr=True)
        vrr_unsupported = VariableRefresh(simulate_vrr=False)

        # Supported should have a valid VRR type
        assert vrr_supported.vrr_type in [
            VRRType.FREESYNC,
            VRRType.GSYNC,
            VRRType.GSYNC_COMPATIBLE,
            VRRType.HDMI_VRR,
            VRRType.ADAPTIVE_SYNC,
        ]

        # Unsupported should be NONE
        assert vrr_unsupported.vrr_type == VRRType.NONE


class TestVRRScenarios:
    """Tests for common VRR usage scenarios."""

    def test_check_and_enable_vrr(self):
        """Test checking support before enabling."""
        vrr = VariableRefresh(simulate_vrr=True)
        if vrr.supported:
            result = vrr.enable(True)
            assert result is True
            assert vrr.enabled is True

    def test_get_capabilities(self):
        """Test getting VRR capabilities."""
        vrr = VariableRefresh(simulate_vrr=True)
        if vrr.supported:
            vrr_type = vrr.vrr_type
            range_info = vrr.get_range()
            assert vrr_type != VRRType.NONE
            assert range_info.max_hz > range_info.min_hz

    def test_vrr_state_persistence(self):
        """Test that VRR state persists across queries."""
        vrr = VariableRefresh(simulate_vrr=True)
        vrr.enable(True)

        # Query properties multiple times
        assert vrr.enabled is True
        assert vrr.enabled is True
        range1 = vrr.get_range()
        range2 = vrr.get_range()
        assert range1.min_hz == range2.min_hz
        assert range1.max_hz == range2.max_hz
