"""
Tests for achievement decorators (achievements.py).

Tests the 3 achievement decorators built on Ops:
    @achievement, @progress, @stat
"""

import pytest

from trinity.decorators.achievements import (
    VALID_AGGREGATIONS,
    achievement,
    progress,
    stat,
)
from trinity.decorators.ops import Op
from trinity.decorators.registry import Tier, registry


# =============================================================================
# @achievement
# =============================================================================


class TestAchievement:
    def test_basic_application_function(self):
        @achievement(id="first_kill")
        def unlock():
            pass

        assert unlock._achievement is True
        assert unlock._achievement_id == "first_kill"

    def test_basic_application_class(self):
        @achievement(id="win_game")
        class WinGame:
            pass

        assert WinGame._achievement is True
        assert WinGame._achievement_id == "win_game"

    def test_default_platform_ids(self):
        @achievement(id="a1")
        def f():
            pass

        assert f._achievement_platform_ids == {}

    def test_custom_platform_ids(self):
        ids = {"steam": "ACH_01", "xbox": "1234"}

        @achievement(id="a1", platform_ids=ids)
        def f():
            pass

        assert f._achievement_platform_ids == ids

    def test_secret_default_false(self):
        @achievement(id="a1")
        def f():
            pass

        assert f._achievement_secret is False

    def test_secret_true(self):
        @achievement(id="a1", secret=True)
        def f():
            pass

        assert f._achievement_secret is True

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="'id' parameter is required"):

            @achievement(id="")
            def f():
                pass

    def test_missing_id_raises(self):
        with pytest.raises(ValueError, match="'id' parameter is required"):
            @achievement()
            def f():
                pass

    def test_tags_set(self):
        @achievement(id="a1")
        def f():
            pass

        assert f._tags["achievement"] is True
        assert f._tags["achievement_id"] == "a1"

    def test_platform_ids_none_becomes_empty_dict(self):
        @achievement(id="a1", platform_ids=None)
        def f():
            pass

        assert f._achievement_platform_ids == {}

    def test_registered_in_achievements_registry(self):
        @achievement(id="reg_test")
        def f():
            pass

        assert "achievements" in f._registries


# =============================================================================
# @progress
# =============================================================================


class TestProgress:
    def test_basic_application(self):
        @progress(id="kills", target=100)
        class KillTracker:
            pass

        assert KillTracker._progress is True
        assert KillTracker._progress_id == "kills"
        assert KillTracker._progress_target == 100

    def test_persistent_default_true(self):
        @progress(id="p1", target=10)
        class P:
            pass

        assert P._progress_persistent is True

    def test_persistent_false(self):
        @progress(id="p1", target=10, persistent=False)
        class P:
            pass

        assert P._progress_persistent is False

    def test_float_target(self):
        @progress(id="p1", target=99.5)
        class P:
            pass

        assert P._progress_target == 99.5

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="'id' parameter is required"):

            @progress(id="", target=10)
            class P:
                pass

    def test_zero_target_raises(self):
        with pytest.raises(ValueError, match="'target' must be a positive number"):

            @progress(id="p1", target=0)
            class P:
                pass

    def test_negative_target_raises(self):
        with pytest.raises(ValueError, match="'target' must be a positive number"):

            @progress(id="p1", target=-5)
            class P:
                pass

    def test_tags_set(self):
        @progress(id="p1", target=50)
        class P:
            pass

        assert P._tags["progress"] is True
        assert P._tags["progress_id"] == "p1"
        assert P._tags["progress_target"] == 50

    def test_registered_in_achievements_registry(self):
        @progress(id="p1", target=10)
        class P:
            pass

        assert "achievements" in P._registries


# =============================================================================
# @stat
# =============================================================================


class TestStat:
    def test_basic_application_class(self):
        @stat(id="total_kills")
        class KillStat:
            pass

        assert KillStat._stat is True
        assert KillStat._stat_id == "total_kills"

    def test_basic_application_function(self):
        @stat(id="score")
        def track_score():
            pass

        assert track_score._stat is True
        assert track_score._stat_id == "score"

    def test_default_aggregation_sum(self):
        @stat(id="s1")
        class S:
            pass

        assert S._stat_aggregation == "sum"

    def test_all_valid_aggregations(self):
        for agg in VALID_AGGREGATIONS:

            @stat(id="s1", aggregation=agg)
            class S:
                pass

            assert S._stat_aggregation == agg

    def test_invalid_aggregation_raises(self):
        with pytest.raises(ValueError, match="invalid aggregation"):

            @stat(id="s1", aggregation="median")
            class S:
                pass

    def test_empty_id_raises(self):
        with pytest.raises(ValueError, match="'id' parameter is required"):

            @stat(id="")
            class S:
                pass

    def test_tags_set(self):
        @stat(id="s1", aggregation="max")
        class S:
            pass

        assert S._tags["stat"] is True
        assert S._tags["stat_id"] == "s1"
        assert S._tags["stat_aggregation"] == "max"

    def test_registered_in_achievements_registry(self):
        @stat(id="s1")
        class S:
            pass

        assert "achievements" in S._registries


# =============================================================================
# Registry
# =============================================================================


class TestAchievementsRegistry:
    def test_achievement_registered(self):
        assert "achievement" in registry._decorators
        spec = registry._decorators["achievement"]
        assert spec.tier == Tier.ACHIEVEMENTS

    def test_progress_registered(self):
        assert "progress" in registry._decorators
        spec = registry._decorators["progress"]
        assert spec.tier == Tier.ACHIEVEMENTS

    def test_stat_registered(self):
        assert "stat" in registry._decorators
        spec = registry._decorators["stat"]
        assert spec.tier == Tier.ACHIEVEMENTS

    def test_tier_has_three_entries(self):
        specs = registry._by_tier[Tier.ACHIEVEMENTS]
        names = {s.name for s in specs}
        assert {"achievement", "progress", "stat"} <= names

    def test_achievement_target_types(self):
        spec = registry._decorators["achievement"]
        assert "function" in spec.target_types
        assert "class" in spec.target_types

    def test_progress_target_types(self):
        spec = registry._decorators["progress"]
        assert "class" in spec.target_types

    def test_stat_target_types(self):
        spec = registry._decorators["stat"]
        assert "class" in spec.target_types
        assert "function" in spec.target_types
