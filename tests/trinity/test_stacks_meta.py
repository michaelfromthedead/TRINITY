"""Tests for meta-composition stacks (builtin_stacks/meta.py)."""
import pytest
from trinity.decorators.stacks import Stack
from trinity.decorators.builtin_stacks.meta import (
    production_multiplayer_game,
    open_world_mmo,
    competitive_esports,
    moddable_singleplayer,
    mobile_optimized,
)


class TestProductionMultiplayerGame:
    """production_multiplayer_game — full multiplayer profile."""

    def test_default_returns_stack(self):
        s = production_multiplayer_game()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = production_multiplayer_game()
        assert len(s.decorators) == 31, (
            f"Expected 31 decorators, got {len(s.decorators)}"
        )

    def test_contains_resolvable_names(self):
        names = production_multiplayer_game().expand()
        assert "track_changes" in names, f"Missing track_changes in {names}"
        assert "component" in names, f"Missing component in {names}"

    def test_custom_params(self):
        s = production_multiplayer_game(pool_size=128, history_frames=60, max_retry=10)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 31, "Custom params must not change structure"

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            production_multiplayer_game(nonexistent=True)


class TestOpenWorldMmo:
    """open_world_mmo — streaming, networking, persistence."""

    def test_default_returns_stack(self):
        s = open_world_mmo()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = open_world_mmo()
        assert len(s.decorators) == 41, (
            f"Expected 41 decorators, got {len(s.decorators)}"
        )

    def test_custom_params(self):
        s = open_world_mmo(pool_size=20000, chunk_size=(200, 200, 200), cache_size=1000)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 41

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            open_world_mmo(nonexistent=True)


class TestCompetitiveEsports:
    """competitive_esports — determinism, replay, events."""

    def test_default_returns_stack(self):
        s = competitive_esports()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = competitive_esports()
        assert len(s.decorators) == 48, (
            f"Expected 48 decorators, got {len(s.decorators)}"
        )

    def test_custom_params(self):
        s = competitive_esports(pool_size=256, history_frames=1200)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 48

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            competitive_esports(nonexistent=True)


class TestModdableSingleplayer:
    """moddable_singleplayer — streaming, saving, modding, UI."""

    def test_default_returns_stack(self):
        s = moddable_singleplayer()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = moddable_singleplayer()
        assert len(s.decorators) == 47, (
            f"Expected 47 decorators, got {len(s.decorators)}"
        )

    def test_custom_params(self):
        s = moddable_singleplayer(
            pool_size=5000, chunk_size=(50, 50, 50), namespace="weapons"
        )
        assert isinstance(s, Stack)
        assert len(s.decorators) == 47

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            moddable_singleplayer(nonexistent=True)


class TestMobileOptimized:
    """mobile_optimized — strict budgets, streaming, cloud."""

    def test_default_returns_stack(self):
        s = mobile_optimized()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = mobile_optimized()
        assert len(s.decorators) == 31, (
            f"Expected 31 decorators, got {len(s.decorators)}"
        )

    def test_custom_params(self):
        s = mobile_optimized(pool_size=256, cache_ttl=30.0, batch_delay_ms=16.0)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 31

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            mobile_optimized(nonexistent=True)


class TestMetaStackComposition:
    """Meta stacks are composable with each other."""

    def test_two_meta_stacks_combine(self):
        a = production_multiplayer_game()
        b = mobile_optimized()
        combined = a + b
        assert isinstance(combined, Stack)
        assert len(combined.decorators) == len(a.decorators) + len(b.decorators)

    def test_meta_repr_includes_count(self):
        s = production_multiplayer_game()
        r = repr(s)
        assert "31" in r, f"repr should include decorator count, got {r}"
