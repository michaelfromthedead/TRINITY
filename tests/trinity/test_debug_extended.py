"""
Tests for Tier 51: DEBUG_EXTENDED decorators.
"""

import pytest

from trinity.decorators.debug_extended import (
    AutomationTestConfig,
    NetworkDebugConfig,
    automation_test,
    network_debug,
)
from trinity.decorators.registry import Tier, registry


class TestNetworkDebug:
    """Test @network_debug decorator."""

    def test_basic_network_debug(self):
        """Test basic network debug without options."""

        @network_debug()
        class TestClass:
            pass

        assert hasattr(TestClass, "_network_debug")
        assert TestClass._network_debug is True
        assert TestClass._network_debug_log_packets is False
        assert TestClass._network_debug_simulate_latency == 0.0
        assert TestClass._network_debug_simulate_loss == 0.0

    def test_log_packets(self):
        """Test network debug with packet logging."""

        @network_debug(log_packets=True)
        class TestClass:
            pass

        assert TestClass._network_debug_log_packets is True

    def test_simulate_latency(self):
        """Test network debug with latency simulation."""

        @network_debug(simulate_latency=50.0)
        class TestClass:
            pass

        assert TestClass._network_debug_simulate_latency == 50.0

    def test_simulate_loss(self):
        """Test network debug with packet loss simulation."""

        @network_debug(simulate_loss=0.1)
        class TestClass:
            pass

        assert TestClass._network_debug_simulate_loss == 0.1

    def test_all_options(self):
        """Test network debug with all options."""

        @network_debug(log_packets=True, simulate_latency=100.0, simulate_loss=0.05)
        class TestClass:
            pass

        assert TestClass._network_debug_log_packets is True
        assert TestClass._network_debug_simulate_latency == 100.0
        assert TestClass._network_debug_simulate_loss == 0.05

        # Check config
        config = TestClass._network_debug_config
        assert isinstance(config, NetworkDebugConfig)
        assert config.log_packets is True
        assert config.simulate_latency == 100.0
        assert config.simulate_loss == 0.05

    def test_invalid_latency(self):
        """Test invalid latency values."""
        with pytest.raises(ValueError, match="simulate_latency must be >= 0"):

            @network_debug(simulate_latency=-10.0)
            class TestClass:
                pass

    def test_invalid_loss_below_zero(self):
        """Test invalid loss below 0."""
        with pytest.raises(ValueError, match="simulate_loss must be between 0 and 1"):

            @network_debug(simulate_loss=-0.1)
            class TestClass:
                pass

    def test_invalid_loss_above_one(self):
        """Test invalid loss above 1."""
        with pytest.raises(ValueError, match="simulate_loss must be between 0 and 1"):

            @network_debug(simulate_loss=1.5)
            class TestClass:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set."""

        @network_debug()
        class TestClass:
            pass

        assert hasattr(TestClass, "_tags")
        assert TestClass._tags.get("network_debug") is True

        assert hasattr(TestClass, "_registries")
        assert "debug_extended" in TestClass._registries

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("network_debug")
        assert spec is not None
        assert spec.name == "network_debug"
        assert spec.tier == Tier.DEBUG_EXTENDED


class TestAutomationTest:
    """Test @automation_test decorator."""

    def test_basic_automation_test(self):
        """Test basic automation test."""

        @automation_test(category="integration")
        class TestCase:
            pass

        assert hasattr(TestCase, "_automation_test")
        assert TestCase._automation_test is True
        assert TestCase._automation_test_category == "integration"
        assert TestCase._automation_test_timeout_seconds == 30.0
        assert TestCase._automation_test_required_features == frozenset()

    def test_custom_timeout(self):
        """Test automation test with custom timeout."""

        @automation_test(category="performance", timeout_seconds=60.0)
        class TestCase:
            pass

        assert TestCase._automation_test_timeout_seconds == 60.0

    def test_required_features(self):
        """Test automation test with required features."""

        @automation_test(
            category="ui", required_features={"rendering", "input", "audio"}
        )
        class TestCase:
            pass

        assert TestCase._automation_test_required_features == frozenset(
            {"rendering", "input", "audio"}
        )

        # Check config
        config = TestCase._automation_test_config
        assert isinstance(config, AutomationTestConfig)
        assert config.category == "ui"
        assert config.timeout_seconds == 30.0
        assert config.required_features == frozenset({"rendering", "input", "audio"})

    def test_all_options(self):
        """Test automation test with all options."""

        @automation_test(
            category="stress",
            timeout_seconds=120.0,
            required_features={"networking", "physics"},
        )
        class TestCase:
            pass

        assert TestCase._automation_test_category == "stress"
        assert TestCase._automation_test_timeout_seconds == 120.0
        assert TestCase._automation_test_required_features == frozenset(
            {"networking", "physics"}
        )

    def test_empty_category(self):
        """Test that empty category is rejected."""
        with pytest.raises(ValueError, match="category must be a non-empty string"):

            @automation_test(category="")
            class TestCase:
                pass

    def test_missing_category(self):
        """Test that missing category is rejected."""
        with pytest.raises(ValueError, match="category must be a non-empty string"):

            @automation_test()
            class TestCase:
                pass

    def test_invalid_timeout(self):
        """Test invalid timeout values."""
        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):

            @automation_test(category="unit", timeout_seconds=0)
            class TestCase:
                pass

        with pytest.raises(ValueError, match="timeout_seconds must be > 0"):

            @automation_test(category="unit", timeout_seconds=-10.0)
            class TestCase:
                pass

    def test_tags_and_registry(self):
        """Test that tags and registry are set."""

        @automation_test(category="smoke")
        class TestCase:
            pass

        assert hasattr(TestCase, "_tags")
        assert TestCase._tags.get("automation_test") is True

        assert hasattr(TestCase, "_registries")
        assert "debug_extended" in TestCase._registries

    def test_registry_entry(self):
        """Test that decorator is registered."""
        spec = registry.get("automation_test")
        assert spec is not None
        assert spec.name == "automation_test"
        assert spec.tier == Tier.DEBUG_EXTENDED


class TestDecoratorComposition:
    """Test decorator composition."""

    def test_both_decorators(self):
        """Test applying both debug_extended decorators."""

        @automation_test(category="network", timeout_seconds=45.0)
        @network_debug(log_packets=True, simulate_latency=25.0)
        class NetworkTest:
            pass

        assert NetworkTest._network_debug is True
        assert NetworkTest._automation_test is True

        assert NetworkTest._network_debug_log_packets is True
        assert NetworkTest._network_debug_simulate_latency == 25.0

        assert NetworkTest._automation_test_category == "network"
        assert NetworkTest._automation_test_timeout_seconds == 45.0

    def test_applied_decorators_tracking(self):
        """Test that applied decorators are tracked."""

        @network_debug()
        @automation_test(category="unit")
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_decorators")
        assert "network_debug" in TestClass._applied_decorators
        assert "automation_test" in TestClass._applied_decorators


class TestRegistryIntegration:
    """Test integration with decorator registry."""

    def test_all_decorators_registered(self):
        """Test that all debug_extended decorators are registered."""
        tier_decorators = registry.by_tier(Tier.DEBUG_EXTENDED)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "network_debug" in decorator_names
        assert "automation_test" in decorator_names

    def test_tier_ordering(self):
        """Test that DEBUG_EXTENDED has correct tier value."""
        assert Tier.DEBUG_EXTENDED == 51
