"""
Tests for analytics decorators (analytics.py).

Tests the 3 analytics decorators built on Ops:
    @telemetry, @funnel, @heatmap
"""

import pytest

from trinity.decorators.analytics import (
    VALID_CONSENT_LEVELS,
    funnel,
    heatmap,
    telemetry,
)
from trinity.decorators.ops import Op
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @telemetry
# =============================================================================


class TestTelemetry:
    def test_basic_application(self):
        @telemetry(event_name="player_died")
        def on_death():
            pass

        assert on_death._telemetry is True
        assert on_death._telemetry_event == "player_died"

    def test_default_pii_false(self):
        @telemetry(event_name="e1")
        def f():
            pass

        assert f._telemetry_pii is False

    def test_pii_true(self):
        @telemetry(event_name="e1", pii=True)
        def f():
            pass

        assert f._telemetry_pii is True

    def test_default_consent_analytics(self):
        @telemetry(event_name="e1")
        def f():
            pass

        assert f._telemetry_consent == "analytics"

    def test_all_valid_consent_levels(self):
        for level in VALID_CONSENT_LEVELS:

            @telemetry(event_name="e1", required_consent=level)
            def f():
                pass

            assert f._telemetry_consent == level

    def test_invalid_consent_raises(self):
        with pytest.raises(ValueError, match="invalid consent level"):

            @telemetry(event_name="e1", required_consent="partial")
            def f():
                pass

    def test_empty_event_name_raises(self):
        with pytest.raises(ValueError, match="'event_name' parameter is required"):

            @telemetry(event_name="")
            def f():
                pass

    def test_missing_event_name_raises(self):
        with pytest.raises(ValueError, match="'event_name' parameter is required"):
            @telemetry()
            def f():
                pass

    def test_tags_set(self):
        @telemetry(event_name="e1", pii=True, required_consent="full")
        def f():
            pass

        assert f._tags["telemetry"] is True
        assert f._tags["telemetry_event"] == "e1"
        assert f._tags["telemetry_pii"] is True
        assert f._tags["telemetry_consent"] == "full"

    def test_registered_in_analytics_registry(self):
        @telemetry(event_name="e1")
        def f():
            pass

        assert "analytics" in f._registries


# =============================================================================
# @funnel
# =============================================================================


class TestFunnel:
    def test_basic_application(self):
        @funnel(name="onboarding", step=1)
        def step_one():
            pass

        assert step_one._funnel is True
        assert step_one._funnel_name == "onboarding"
        assert step_one._funnel_step == 1

    def test_higher_step(self):
        @funnel(name="purchase", step=5)
        def checkout():
            pass

        assert checkout._funnel_step == 5

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="'name' parameter is required"):

            @funnel(name="", step=1)
            def f():
                pass

    def test_zero_step_raises(self):
        with pytest.raises(ValueError, match="'step' must be a positive integer"):

            @funnel(name="f1", step=0)
            def f():
                pass

    def test_negative_step_raises(self):
        with pytest.raises(ValueError, match="'step' must be a positive integer"):

            @funnel(name="f1", step=-1)
            def f():
                pass

    def test_float_step_raises(self):
        with pytest.raises(ValueError, match="'step' must be a positive integer"):

            @funnel(name="f1", step=1.5)
            def f():
                pass

    def test_tags_set(self):
        @funnel(name="f1", step=3)
        def f():
            pass

        assert f._tags["funnel"] is True
        assert f._tags["funnel_name"] == "f1"
        assert f._tags["funnel_step"] == 3

    def test_registered_in_analytics_registry(self):
        @funnel(name="f1", step=1)
        def f():
            pass

        assert "analytics" in f._registries


# =============================================================================
# @heatmap
# =============================================================================


class TestHeatmap:
    def test_basic_application(self):
        @heatmap()
        def track_position():
            pass

        assert track_position._heatmap is True

    def test_default_resolution(self):
        @heatmap()
        def f():
            pass

        assert f._heatmap_resolution == 1.0

    def test_custom_resolution(self):
        @heatmap(resolution=0.5)
        def f():
            pass

        assert f._heatmap_resolution == 0.5

    def test_int_resolution(self):
        @heatmap(resolution=2)
        def f():
            pass

        assert f._heatmap_resolution == 2

    def test_zero_resolution_raises(self):
        with pytest.raises(ValueError, match="'resolution' must be a positive number"):

            @heatmap(resolution=0)
            def f():
                pass

    def test_negative_resolution_raises(self):
        with pytest.raises(ValueError, match="'resolution' must be a positive number"):

            @heatmap(resolution=-1.0)
            def f():
                pass

    def test_tags_set(self):
        @heatmap(resolution=2.5)
        def f():
            pass

        assert f._tags["heatmap"] is True
        assert f._tags["heatmap_resolution"] == 2.5

    def test_registered_in_analytics_registry(self):
        @heatmap()
        def f():
            pass

        assert "analytics" in f._registries


# =============================================================================
# Registry
# =============================================================================


class TestAnalyticsRegistry:
    def test_telemetry_registered(self):
        assert "telemetry" in registry._decorators
        spec = registry._decorators["telemetry"]
        assert spec.tier == Tier.ANALYTICS

    def test_funnel_registered(self):
        assert "funnel" in registry._decorators
        spec = registry._decorators["funnel"]
        assert spec.tier == Tier.ANALYTICS

    def test_heatmap_registered(self):
        assert "heatmap" in registry._decorators
        spec = registry._decorators["heatmap"]
        assert spec.tier == Tier.ANALYTICS

    def test_tier_has_three_entries(self):
        specs = registry._by_tier[Tier.ANALYTICS]
        names = {s.name for s in specs}
        assert {"telemetry", "funnel", "heatmap"} <= names

    def test_telemetry_target_types(self):
        spec = registry._decorators["telemetry"]
        assert "function" in spec.target_types

    def test_funnel_target_types(self):
        spec = registry._decorators["funnel"]
        assert "function" in spec.target_types

    def test_heatmap_target_types(self):
        spec = registry._decorators["heatmap"]
        assert "function" in spec.target_types
