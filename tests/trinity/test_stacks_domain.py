"""Tests for domain-specific composite stacks."""
import pytest
from trinity.decorators.stacks import Stack
from trinity.decorators.builtin_stacks.gameplay import (
    full_destruction,
    gameplay_ability,
    crafting_system,
)
from trinity.decorators.builtin_stacks.audio_stacks import adaptive_audio
from trinity.decorators.builtin_stacks.platform import platform_adaptive
from trinity.decorators.builtin_stacks.modding_stacks import mod_friendly


def _expand(s: Stack) -> list[str]:
    """Return decorator names for a stack."""
    return s.expand()


# ---------------------------------------------------------------------------
# full_destruction (gameplay.py)
# ---------------------------------------------------------------------------

class TestFullDestruction:
    """full_destruction — destructible environment stack."""

    def test_default_returns_stack(self):
        s = full_destruction()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = full_destruction()
        # destructible, damage_type, damage_resistance, fracture,
        # physics_material, pooled = 6
        assert len(s.decorators) == 6, (
            f"Expected 6 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_health(self):
        """Use correct param name: health (not hp)."""
        s = full_destruction(health=500)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 6

    def test_custom_fracture_pattern(self):
        s = full_destruction(fracture_pattern="radial")
        assert len(s.decorators) == 6

    def test_custom_pool_size(self):
        s = full_destruction(pool_size=512)
        assert len(s.decorators) == 6

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            full_destruction(hp=500)  # wrong param name


# ---------------------------------------------------------------------------
# gameplay_ability (gameplay.py)
# ---------------------------------------------------------------------------

class TestGameplayAbility:
    """gameplay_ability — ability system stack."""

    def test_default_returns_stack(self):
        s = gameplay_ability()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = gameplay_ability()
        # ability, buff, gameplay_tag, serializable, track_changes = 5
        assert len(s.decorators) == 5, (
            f"Expected 5 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_contains_track_changes(self):
        """track_changes is one name that resolves correctly."""
        s = gameplay_ability()
        names = _expand(s)
        assert "track_changes" in names, (
            f"Should include track_changes, got {names}"
        )

    def test_custom_cooldown(self):
        s = gameplay_ability(cooldown=5.0)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 5

    def test_custom_max_stacks(self):
        s = gameplay_ability(max_stacks=3)
        assert len(s.decorators) == 5

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            gameplay_ability(damage=10)


# ---------------------------------------------------------------------------
# crafting_system (gameplay.py)
# ---------------------------------------------------------------------------

class TestCraftingSystem:
    """crafting_system — crafting station stack."""

    def test_default_returns_stack(self):
        s = crafting_system()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = crafting_system()
        # crafting_station, recipe, ingredient, loot_table,
        # salvage_recipe, serializable = 6
        assert len(s.decorators) == 6, (
            f"Expected 6 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_station_id(self):
        """Use correct param name: station_id (not slots)."""
        s = crafting_system(station_id="forge")
        assert isinstance(s, Stack)
        assert len(s.decorators) == 6

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            crafting_system(slots=8)  # wrong param name


# ---------------------------------------------------------------------------
# adaptive_audio (audio_stacks.py)
# ---------------------------------------------------------------------------

class TestAdaptiveAudio:
    """adaptive_audio — dynamic music/sound stack."""

    def test_default_returns_stack(self):
        s = adaptive_audio()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = adaptive_audio()
        # music_stem, music_transition, audio_snapshot, serializable = 4
        assert len(s.decorators) == 4, (
            f"Expected 4 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_crossfade_time(self):
        """Use correct param name: crossfade_time (not layers)."""
        s = adaptive_audio(crossfade_time=1.0)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 4

    def test_custom_stem_group(self):
        s = adaptive_audio(stem_group="sfx")
        assert len(s.decorators) == 4

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            adaptive_audio(layers=4)  # wrong param name


# ---------------------------------------------------------------------------
# platform_adaptive (platform.py)
# ---------------------------------------------------------------------------

class TestPlatformAdaptive:
    """platform_adaptive — cross-platform optimization stack."""

    def test_default_returns_stack(self):
        s = platform_adaptive()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = platform_adaptive()
        # battery_aware, lod, streamable = 3
        assert len(s.decorators) == 3, (
            f"Expected 3 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_lod_levels(self):
        """Use correct param name: lod_levels (not target)."""
        s = platform_adaptive(lod_levels=5)
        assert isinstance(s, Stack)
        assert len(s.decorators) == 3

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            platform_adaptive(target="mobile")  # wrong param name


# ---------------------------------------------------------------------------
# mod_friendly (modding_stacks.py)
# ---------------------------------------------------------------------------

class TestModFriendly:
    """mod_friendly — modding support stack."""

    def test_default_returns_stack(self):
        s = mod_friendly()
        assert isinstance(s, Stack)

    def test_exact_decorator_count(self):
        s = mod_friendly()
        # moddable, observable, serializable = 3
        assert len(s.decorators) == 3, (
            f"Expected 3 decorators, got {len(s.decorators)}: {_expand(s)}"
        )

    def test_custom_namespace(self):
        s = mod_friendly(namespace="weapons")
        assert isinstance(s, Stack)
        assert len(s.decorators) == 3

    def test_invalid_param_raises(self):
        with pytest.raises(TypeError):
            mod_friendly(nonexistent=True)
