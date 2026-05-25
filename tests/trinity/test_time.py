"""Tests for Time decorators (Tier 25)."""

import pytest

from trinity.decorators.time import (
    VALID_INTERPOLATIONS,
    deterministic,
    pausable,
    rewindable,
    time_scale,
)
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @time_scale
# =============================================================================


class TestTimeScale:
    """Tests for the @time_scale decorator."""

    def test_basic_defaults(self):
        @time_scale()
        def physics():
            pass

        assert physics._time_scale is True
        assert physics._time_layer == "gameplay"
        assert physics._time_min_scale == 0.0
        assert physics._time_max_scale == 10.0

    def test_custom_layer(self):
        @time_scale(layer="ui")
        def update():
            pass

        assert update._time_layer == "ui"

    def test_custom_scales(self):
        @time_scale(min_scale=0.5, max_scale=2.0)
        def sim():
            pass

        assert sim._time_min_scale == 0.5
        assert sim._time_max_scale == 2.0

    def test_all_params(self):
        @time_scale(layer="physics", min_scale=0.1, max_scale=5.0)
        def step():
            pass

        assert step._time_layer == "physics"
        assert step._time_min_scale == 0.1
        assert step._time_max_scale == 5.0

    def test_tags(self):
        @time_scale(layer="audio")
        def audio_tick():
            pass

        assert audio_tick._tags["time_scale"] is True
        assert audio_tick._tags["time_layer"] == "audio"

    def test_applied_decorators(self):
        @time_scale()
        def tick():
            pass

        assert "time_scale" in tick._applied_decorators

    def test_registered_in_time(self):
        @time_scale()
        def sys():
            pass

        assert "time" in sys._registries

    def test_no_args(self):
        @time_scale
        def auto():
            pass

        assert auto._time_scale is True
        assert auto._time_layer == "gameplay"

    # Validation

    def test_empty_layer(self):
        with pytest.raises(ValueError, match="layer"):
            time_scale(layer="")

    def test_min_scale_negative(self):
        with pytest.raises(ValueError, match="min_scale"):
            time_scale(min_scale=-1.0)

    def test_max_scale_zero(self):
        with pytest.raises(ValueError, match="max_scale"):
            time_scale(max_scale=0.0)

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_scale.*max_scale"):
            time_scale(min_scale=5.0, max_scale=2.0)

    def test_min_equals_max(self):
        @time_scale(min_scale=1.0, max_scale=1.0)
        def fixed():
            pass

        assert fixed._time_min_scale == 1.0
        assert fixed._time_max_scale == 1.0


# =============================================================================
# @pausable
# =============================================================================


class TestPausable:
    """Tests for the @pausable decorator."""

    def test_basic_defaults(self):
        @pausable()
        def game_loop():
            pass

        assert game_loop._pausable is True
        assert game_loop._pause_layers == {"gameplay"}

    def test_custom_layers(self):
        @pausable(pause_layers={"gameplay", "physics"})
        def sim():
            pass

        assert sim._pause_layers == {"gameplay", "physics"}

    def test_single_layer(self):
        @pausable(pause_layers={"ui"})
        def ui_tick():
            pass

        assert ui_tick._pause_layers == {"ui"}

    def test_none_defaults_to_gameplay(self):
        @pausable(pause_layers=None)
        def fn():
            pass

        assert fn._pause_layers == {"gameplay"}

    def test_tags(self):
        @pausable()
        def fn():
            pass

        assert fn._tags["pausable"] is True
        assert fn._tags["pause_layers"] == {"gameplay"}

    def test_applied_decorators(self):
        @pausable()
        def fn():
            pass

        assert "pausable" in fn._applied_decorators

    def test_no_args(self):
        @pausable
        def auto():
            pass

        assert auto._pausable is True
        assert auto._pause_layers == {"gameplay"}


# =============================================================================
# @rewindable
# =============================================================================


class TestRewindable:
    """Tests for the @rewindable decorator."""

    def test_basic_defaults(self):
        @rewindable()
        class Transform:
            pass

        assert Transform._rewindable is True
        assert Transform._rewind_history == 5.0
        assert Transform._rewind_interpolation == "linear"

    def test_custom_history(self):
        @rewindable(history_seconds=10.0)
        class State:
            pass

        assert State._rewind_history == 10.0

    def test_custom_interpolation(self):
        @rewindable(interpolation="hermite")
        class Smooth:
            pass

        assert Smooth._rewind_interpolation == "hermite"

    def test_none_interpolation(self):
        @rewindable(interpolation="none")
        class Snap:
            pass

        assert Snap._rewind_interpolation == "none"

    def test_all_valid_interpolations(self):
        for interp in VALID_INTERPOLATIONS:

            @rewindable(interpolation=interp)
            class C:
                pass

            assert C._rewind_interpolation == interp

    def test_tags(self):
        @rewindable(history_seconds=3.0)
        class T:
            pass

        assert T._tags["rewindable"] is True
        assert T._tags["rewind_history"] == 3.0

    def test_applied_decorators(self):
        @rewindable()
        class A:
            pass

        assert "rewindable" in A._applied_decorators

    def test_no_args(self):
        @rewindable
        class Auto:
            pass

        assert Auto._rewindable is True
        assert Auto._rewind_history == 5.0

    # Validation

    def test_history_zero(self):
        with pytest.raises(ValueError, match="history_seconds"):
            rewindable(history_seconds=0.0)

    def test_history_negative(self):
        with pytest.raises(ValueError, match="history_seconds"):
            rewindable(history_seconds=-1.0)

    def test_invalid_interpolation(self):
        with pytest.raises(ValueError, match="interpolation"):
            rewindable(interpolation="cubic")


# =============================================================================
# @deterministic
# =============================================================================


class TestDeterministic:
    """Tests for the @deterministic decorator."""

    def test_basic(self):
        @deterministic
        def physics_step():
            pass

        assert physics_step._deterministic is True

    def test_with_parens(self):
        @deterministic()
        def step():
            pass

        assert step._deterministic is True

    def test_tags(self):
        @deterministic
        def fn():
            pass

        assert fn._tags["deterministic"] is True

    def test_applied_decorators(self):
        @deterministic
        def fn():
            pass

        assert "deterministic" in fn._applied_decorators

    def test_registered_in_time(self):
        @deterministic
        def fn():
            pass

        assert "time" in fn._registries

    def test_marker_no_params(self):
        # deterministic is a marker; calling with no args on a function works
        @deterministic
        def a():
            pass

        @deterministic()
        def b():
            pass

        assert a._deterministic is True
        assert b._deterministic is True


# =============================================================================
# Registry
# =============================================================================


class TestTimeRegistry:
    """Registry tests for Tier 25 decorators."""

    def test_time_scale_registered(self):
        spec = registry.get("time_scale")
        assert spec is not None
        assert spec.tier == Tier.TIME

    def test_pausable_registered(self):
        spec = registry.get("pausable")
        assert spec is not None
        assert spec.tier == Tier.TIME

    def test_rewindable_registered(self):
        spec = registry.get("rewindable")
        assert spec is not None
        assert spec.tier == Tier.TIME

    def test_deterministic_registered(self):
        spec = registry.get("deterministic")
        assert spec is not None
        assert spec.tier == Tier.TIME

    def test_all_in_tier(self):
        specs = registry.by_tier(Tier.TIME)
        names = {s.name for s in specs}
        assert {"time_scale", "pausable", "rewindable", "deterministic"} <= names

    def test_target_types(self):
        assert registry.get("time_scale").target_types == ("function",)
        assert registry.get("pausable").target_types == ("function",)
        assert registry.get("rewindable").target_types == ("class",)
        assert registry.get("deterministic").target_types == ("function",)
