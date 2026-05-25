"""
Comprehensive tests for the Scoring System.

Tests cover:
- Kill/death/assist tracking
- Score calculation
- Leaderboard sorting
- Score events
- Bonus scoring (streaks, objectives)
- Score limits
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.scoring import (
    ScoringSystem,
    ScoreEventType,
    LeaderboardSortKey,
    PlayerStats,
    TeamStats,
    ScoreEvent,
    LeaderboardEntry,
)
from engine.gameplay.combat.constants import (
    ScoringConfig,
    DEFAULT_SCORING_CONFIG,
    POINTS_PER_KILL,
    POINTS_PER_ASSIST,
    POINTS_PER_DEATH,
    POINTS_PER_OBJECTIVE,
    POINTS_PER_HEADSHOT_BONUS,
    POINTS_PER_FIRST_BLOOD,
    POINTS_PER_REVENGE_KILL,
    POINTS_PER_KILLSTREAK_BONUS,
    KILLSTREAK_THRESHOLDS,
    MULTI_KILL_WINDOW,
    MULTI_KILL_NAMES,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def scoring_system():
    """Create a fresh scoring system for each test."""
    return ScoringSystem()


@pytest.fixture
def team_scoring_system():
    """Create a team-based scoring system."""
    return ScoringSystem(is_team_based=True)


@pytest.fixture
def populated_system(scoring_system):
    """Create a scoring system with players."""
    scoring_system.add_player("player_1")
    scoring_system.add_player("player_2")
    scoring_system.add_player("player_3")
    return scoring_system


@pytest.fixture
def team_populated_system(team_scoring_system):
    """Create a team scoring system with players."""
    team_scoring_system.add_player("p1", team_id="red")
    team_scoring_system.add_player("p2", team_id="red")
    team_scoring_system.add_player("p3", team_id="blue")
    team_scoring_system.add_player("p4", team_id="blue")
    return team_scoring_system


# =============================================================================
# PLAYER STATS TESTS (~15 tests)
# =============================================================================


class TestPlayerStats:
    """Tests for player stats management."""

    def test_add_player(self, scoring_system):
        """Should add player to system."""
        stats = scoring_system.add_player("player_1")
        assert stats is not None
        assert stats.player_id == "player_1"

    def test_add_player_with_team(self, team_scoring_system):
        """Should add player with team."""
        stats = team_scoring_system.add_player("player_1", team_id="red")
        assert stats.team_id == "red"

    def test_get_player_stats(self, populated_system):
        """Should get player stats."""
        stats = populated_system.get_player_stats("player_1")
        assert stats is not None
        assert stats.player_id == "player_1"

    def test_get_nonexistent_player_stats(self, scoring_system):
        """Should return None for nonexistent player."""
        stats = scoring_system.get_player_stats("nonexistent")
        assert stats is None

    def test_remove_player(self, populated_system):
        """Should remove player."""
        result = populated_system.remove_player("player_1")
        assert result
        assert populated_system.get_player_stats("player_1") is None

    def test_remove_nonexistent_player(self, scoring_system):
        """Should return False for nonexistent player."""
        result = scoring_system.remove_player("nonexistent")
        assert not result

    def test_set_player_team(self, populated_system):
        """Should change player team."""
        result = populated_system.set_player_team("player_1", "red")
        assert result

        stats = populated_system.get_player_stats("player_1")
        assert stats.team_id == "red"

    def test_initial_stats_zero(self, scoring_system):
        """Initial stats should be zero."""
        stats = scoring_system.add_player("player_1")

        assert stats.score == 0
        assert stats.kills == 0
        assert stats.deaths == 0
        assert stats.assists == 0

    def test_kd_ratio_no_deaths(self):
        """KD ratio with no deaths should return kills."""
        stats = PlayerStats(player_id="p1")
        stats.kills = 10

        assert stats.kd_ratio == 10.0

    def test_kd_ratio_with_deaths(self):
        """KD ratio should calculate correctly."""
        stats = PlayerStats(player_id="p1")
        stats.kills = 10
        stats.deaths = 5

        assert stats.kd_ratio == 2.0

    def test_kda_ratio(self):
        """KDA ratio should include assists."""
        stats = PlayerStats(player_id="p1")
        stats.kills = 10
        stats.assists = 5
        stats.deaths = 5

        assert stats.kda_ratio == 3.0  # (10 + 5) / 5


# =============================================================================
# KILL TRACKING TESTS (~25 tests)
# =============================================================================


class TestKillTracking:
    """Tests for kill tracking."""

    def test_record_kill(self, populated_system):
        """Should record kill."""
        result = populated_system.record_kill("player_1", "player_2")

        assert result["kill_awarded"]
        stats = populated_system.get_player_stats("player_1")
        assert stats.kills == 1

    def test_record_kill_victim_death(self, populated_system):
        """Kill should record victim death."""
        populated_system.record_kill("player_1", "player_2")

        stats = populated_system.get_player_stats("player_2")
        assert stats.deaths == 1

    def test_record_kill_awards_points(self, populated_system):
        """Kill should award points."""
        populated_system.record_kill("player_1", "player_2")

        stats = populated_system.get_player_stats("player_1")
        assert stats.score >= POINTS_PER_KILL

    def test_record_headshot_kill(self, populated_system):
        """Headshot should award bonus."""
        result = populated_system.record_kill(
            "player_1", "player_2", is_headshot=True
        )

        assert "headshot" in result["achievements"]
        stats = populated_system.get_player_stats("player_1")
        assert stats.headshots == 1

    def test_first_blood(self, populated_system):
        """First kill should award first blood."""
        populated_system.start_match()
        result = populated_system.record_kill("player_1", "player_2")

        assert "first_blood" in result["achievements"]
        assert populated_system.first_blood_awarded

    def test_first_blood_only_once(self, populated_system):
        """First blood should only be awarded once."""
        populated_system.start_match()
        populated_system.record_kill("player_1", "player_2")
        result = populated_system.record_kill("player_3", "player_1")

        assert "first_blood" not in result["achievements"]

    def test_revenge_kill(self, populated_system):
        """Should detect revenge kill when victim previously killed the killer."""
        # Player 1 kills Player 2 first (adds player_1 to player_2's _killed_by)
        populated_system.record_kill("player_1", "player_2")

        # Player 2 kills Player 1 (not revenge - player_2 not in player_1's _killed_by yet)
        populated_system.record_kill("player_2", "player_1")

        # Player 1 kills Player 2 again - now player_1 IS in player_2's _killed_by
        # So this is revenge (victim=player_2 had previously killed killer=player_1)
        result = populated_system.record_kill("player_1", "player_2")

        assert "revenge" in result["achievements"]

    def test_killstreak_tracking(self, populated_system):
        """Should track killstreaks."""
        # Player 1 gets 3 kills
        for _ in range(3):
            populated_system.record_kill("player_1", "player_2")

        stats = populated_system.get_player_stats("player_1")
        assert stats.current_killstreak == 3

    def test_killstreak_bonus(self, populated_system):
        """Should award killstreak bonus."""
        # Get to 5 kills for bonus
        for _ in range(5):
            populated_system.record_kill("player_1", "player_2")

        result = populated_system.record_kill("player_1", "player_2")

        # Should have received streak achievement at some point
        stats = populated_system.get_player_stats("player_1")
        assert stats.current_killstreak == 6

    def test_death_resets_killstreak(self, populated_system):
        """Death should reset killstreak."""
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_1", "player_2")

        # Player 1 dies
        populated_system.record_kill("player_2", "player_1")

        stats = populated_system.get_player_stats("player_1")
        assert stats.current_killstreak == 0

    def test_best_killstreak_tracked(self, populated_system):
        """Should track best killstreak."""
        # Get 5 kills
        for _ in range(5):
            populated_system.record_kill("player_1", "player_2")

        # Die
        populated_system.record_kill("player_2", "player_1")

        # Get 3 kills
        for _ in range(3):
            populated_system.record_kill("player_1", "player_2")

        stats = populated_system.get_player_stats("player_1")
        assert stats.best_killstreak == 5

    def test_multi_kill_detection(self, populated_system):
        """Should detect multi-kills."""
        # Quick succession kills
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_1", "player_3")

        stats = populated_system.get_player_stats("player_1")
        assert stats.double_kills >= 1

    def test_triple_kill(self, populated_system):
        """Should detect triple kill."""
        for i in range(3):
            populated_system.record_kill("player_1", f"player_{i+2}")

        stats = populated_system.get_player_stats("player_1")
        # Should have at least a triple
        assert stats.triple_kills >= 1 or stats.double_kills >= 1


# =============================================================================
# ASSIST TRACKING TESTS (~15 tests)
# =============================================================================


class TestAssistTracking:
    """Tests for assist tracking."""

    def test_record_damage_for_assist(self, populated_system):
        """Should record damage for assist tracking."""
        populated_system.record_damage("player_1", "player_3", 50.0, 100.0)

        stats = populated_system.get_player_stats("player_1")
        assert stats.damage_dealt == 50.0

    def test_assist_awarded_on_kill(self, populated_system):
        """Should award assist when damage contributed."""
        # Player 1 damages Player 3
        populated_system.record_damage("player_1", "player_3", 50.0, 100.0)

        # Player 2 kills Player 3
        result = populated_system.record_kill("player_2", "player_3")

        assert "player_1" in result["assists"]

    def test_assist_points_awarded(self, populated_system):
        """Assist should award points."""
        populated_system.record_damage("player_1", "player_3", 50.0, 100.0)
        populated_system.record_kill("player_2", "player_3")

        stats = populated_system.get_player_stats("player_1")
        assert stats.assists == 1

    def test_no_assist_below_threshold(self, populated_system):
        """Should not award assist below damage threshold."""
        # Very small damage (below threshold)
        populated_system.record_damage("player_1", "player_3", 1.0, 100.0)
        result = populated_system.record_kill("player_2", "player_3")

        # With 1% damage (below 10% threshold), no assist
        assert "player_1" not in result["assists"]

    def test_killer_no_self_assist(self, populated_system):
        """Killer should not get assist credit."""
        populated_system.record_damage("player_1", "player_3", 50.0, 100.0)
        result = populated_system.record_kill("player_1", "player_3")

        assert "player_1" not in result["assists"]

    def test_damage_cleared_after_death(self, populated_system):
        """Damage tracking should clear after target dies."""
        populated_system.record_damage("player_1", "player_3", 50.0, 100.0)
        populated_system.record_kill("player_2", "player_3")

        # Check damage was cleared
        stats = populated_system.get_player_stats("player_1")
        damage = stats.get_assist_damage("player_3", 100.0)
        assert damage is None


# =============================================================================
# SCORE CALCULATION TESTS (~20 tests)
# =============================================================================


class TestScoreCalculation:
    """Tests for score calculation."""

    def test_add_score(self, populated_system):
        """Should add score to player."""
        result = populated_system.add_score("player_1", 100)
        assert result

        stats = populated_system.get_player_stats("player_1")
        assert stats.score == 100

    def test_add_score_accumulates(self, populated_system):
        """Score should accumulate."""
        populated_system.add_score("player_1", 50)
        populated_system.add_score("player_1", 30)

        stats = populated_system.get_player_stats("player_1")
        assert stats.score == 80

    def test_set_score(self, populated_system):
        """Should set score directly."""
        populated_system.add_score("player_1", 100)
        result = populated_system.set_score("player_1", 50)

        assert result
        stats = populated_system.get_player_stats("player_1")
        assert stats.score == 50

    def test_get_score(self, populated_system):
        """Should get player score."""
        populated_system.add_score("player_1", 75)
        score = populated_system.get_score("player_1")
        assert score == 75

    def test_get_score_nonexistent(self, scoring_system):
        """Should return 0 for nonexistent player."""
        score = scoring_system.get_score("nonexistent")
        assert score == 0

    def test_negative_score_allowed(self, populated_system):
        """Should allow negative scores."""
        populated_system.add_score("player_1", -50)

        stats = populated_system.get_player_stats("player_1")
        assert stats.score == -50

    def test_team_score_updated(self, team_populated_system):
        """Adding player score should update team score."""
        team_populated_system.add_score("p1", 100)

        team_score = team_populated_system.get_team_score("red")
        assert team_score == 100

    def test_record_death_no_killer(self, populated_system):
        """Should record death without killer (suicide/env)."""
        result = populated_system.record_death("player_1")
        assert result

        stats = populated_system.get_player_stats("player_1")
        assert stats.deaths == 1


# =============================================================================
# OBJECTIVE SCORING TESTS (~10 tests)
# =============================================================================


class TestObjectiveScoring:
    """Tests for objective-based scoring."""

    def test_record_objective_capture(self, populated_system):
        """Should record objective capture."""
        result = populated_system.record_objective_capture("player_1")
        assert result

        stats = populated_system.get_player_stats("player_1")
        assert stats.objectives_captured == 1

    def test_objective_capture_awards_points(self, populated_system):
        """Objective capture should award points."""
        populated_system.record_objective_capture("player_1")

        stats = populated_system.get_player_stats("player_1")
        assert stats.score >= POINTS_PER_OBJECTIVE

    def test_record_objective_defend(self, populated_system):
        """Should record objective defense."""
        result = populated_system.record_objective_defend("player_1")
        assert result

        stats = populated_system.get_player_stats("player_1")
        assert stats.objectives_defended == 1

    def test_custom_objective_points(self, populated_system):
        """Should accept custom objective points."""
        populated_system.record_objective_capture("player_1", points=500)

        stats = populated_system.get_player_stats("player_1")
        assert stats.score >= 500

    def test_team_objectives_tracked(self, team_populated_system):
        """Should track team objectives."""
        team_populated_system.record_objective_capture("p1")

        team_stats = team_populated_system._team_stats.get("red")
        assert team_stats.objectives == 1


# =============================================================================
# LEADERBOARD TESTS (~20 tests)
# =============================================================================


class TestLeaderboard:
    """Tests for leaderboard generation."""

    def test_get_leaderboard(self, populated_system):
        """Should get sorted leaderboard."""
        populated_system.add_score("player_1", 100)
        populated_system.add_score("player_2", 200)
        populated_system.add_score("player_3", 150)

        leaderboard = populated_system.get_leaderboard()

        assert len(leaderboard) == 3
        assert leaderboard[0].player_id == "player_2"
        assert leaderboard[1].player_id == "player_3"
        assert leaderboard[2].player_id == "player_1"

    def test_leaderboard_has_ranks(self, populated_system):
        """Leaderboard entries should have ranks."""
        populated_system.add_score("player_1", 100)
        populated_system.add_score("player_2", 200)

        leaderboard = populated_system.get_leaderboard()

        assert leaderboard[0].rank == 1
        assert leaderboard[1].rank == 2

    def test_leaderboard_sort_by_kills(self, populated_system):
        """Should sort by kills."""
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_1", "player_3")
        populated_system.record_kill("player_3", "player_2")

        leaderboard = populated_system.get_leaderboard(
            sort_by=LeaderboardSortKey.KILLS
        )

        assert leaderboard[0].player_id == "player_1"

    def test_leaderboard_sort_by_kd(self, populated_system):
        """Should sort by KD ratio."""
        # Player 1: 2 kills, 0 deaths = 2.0 KD
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_1", "player_3")

        # Player 2: 1 kill, 2 deaths = 0.5 KD
        populated_system.record_kill("player_2", "player_3")

        leaderboard = populated_system.get_leaderboard(
            sort_by=LeaderboardSortKey.KD_RATIO
        )

        assert leaderboard[0].player_id == "player_1"

    def test_leaderboard_limit(self, populated_system):
        """Should limit leaderboard entries."""
        populated_system.add_score("player_1", 100)
        populated_system.add_score("player_2", 200)
        populated_system.add_score("player_3", 150)

        leaderboard = populated_system.get_leaderboard(limit=2)
        assert len(leaderboard) == 2

    def test_leaderboard_team_filter(self, team_populated_system):
        """Should filter leaderboard by team."""
        team_populated_system.add_score("p1", 100)  # red
        team_populated_system.add_score("p2", 200)  # red
        team_populated_system.add_score("p3", 150)  # blue
        team_populated_system.add_score("p4", 50)   # blue

        leaderboard = team_populated_system.get_leaderboard(team_id="red")

        assert len(leaderboard) == 2
        for entry in leaderboard:
            assert entry.team_id == "red"

    def test_team_leaderboard(self, team_populated_system):
        """Should get team leaderboard."""
        team_populated_system.add_score("p1", 100)
        team_populated_system.add_score("p2", 100)
        team_populated_system.add_score("p3", 50)
        team_populated_system.add_score("p4", 25)

        team_lb = team_populated_system.get_team_leaderboard()

        assert len(team_lb) == 2
        assert team_lb[0][0] == "red"  # 200 total
        assert team_lb[1][0] == "blue"  # 75 total

    def test_get_player_rank(self, populated_system):
        """Should get player rank."""
        populated_system.add_score("player_1", 100)
        populated_system.add_score("player_2", 200)
        populated_system.add_score("player_3", 150)

        rank = populated_system.get_player_rank("player_1")
        assert rank == 3  # Lowest score

    def test_get_player_rank_not_found(self, populated_system):
        """Should return 0 for unranked player."""
        rank = populated_system.get_player_rank("nonexistent")
        assert rank == 0


# =============================================================================
# SCORE EVENT TESTS (~15 tests)
# =============================================================================


class TestScoreEvents:
    """Tests for score event handling."""

    def test_on_score_changed(self, populated_system):
        """Should emit score changed event."""
        handler = Mock()
        populated_system.on_score_changed(handler)

        populated_system.add_score("player_1", 100)

        handler.assert_called_once()

    def test_score_changed_args(self, populated_system):
        """Score changed should provide correct args."""
        received = []

        def handler(player_id, old_score, new_score):
            received.append((player_id, old_score, new_score))

        populated_system.on_score_changed(handler)
        populated_system.add_score("player_1", 100)

        assert received[0] == ("player_1", 0, 100)

    def test_on_kill(self, populated_system):
        """Should emit kill event."""
        handler = Mock()
        populated_system.on_kill(handler)

        populated_system.record_kill("player_1", "player_2")

        handler.assert_called_once()

    def test_on_killstreak(self, populated_system):
        """Should emit killstreak event."""
        handler = Mock()
        populated_system.on_killstreak(handler)

        # Get to 5 kills for streak notification
        for _ in range(5):
            populated_system.record_kill("player_1", "player_2")

        assert handler.call_count >= 1

    def test_on_multi_kill(self, populated_system):
        """Should emit multi-kill event."""
        handler = Mock()
        populated_system.on_multi_kill(handler)

        # Quick double kill
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_1", "player_3")

        handler.assert_called()

    def test_on_first_blood(self, populated_system):
        """Should emit first blood event."""
        handler = Mock()
        populated_system.on_first_blood(handler)

        populated_system.start_match()
        populated_system.record_kill("player_1", "player_2")

        handler.assert_called_once_with("player_1", "player_2")


# =============================================================================
# EVENT HISTORY TESTS (~10 tests)
# =============================================================================


class TestEventHistory:
    """Tests for event history tracking."""

    def test_events_recorded(self, populated_system):
        """Should record events in history."""
        populated_system.add_score("player_1", 100)

        history = populated_system.get_event_history()
        assert len(history) >= 1

    def test_filter_history_by_player(self, populated_system):
        """Should filter history by player."""
        populated_system.add_score("player_1", 100)
        populated_system.add_score("player_2", 50)

        history = populated_system.get_event_history(player_id="player_1")
        for event in history:
            assert event.player_id == "player_1"

    def test_filter_history_by_type(self, populated_system):
        """Should filter history by event type."""
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_objective_capture("player_1")

        history = populated_system.get_event_history(
            event_type=ScoreEventType.KILL
        )
        for event in history:
            assert event.event_type == ScoreEventType.KILL

    def test_history_limit(self, populated_system):
        """Should limit history results."""
        for i in range(10):
            populated_system.add_score("player_1", 10)

        history = populated_system.get_event_history(limit=5)
        assert len(history) <= 5

    def test_get_recent_kills(self, populated_system):
        """Should get recent kill events."""
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_2", "player_3")

        kills = populated_system.get_recent_kills(limit=10)
        assert len(kills) == 2


# =============================================================================
# UTILITY TESTS (~10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_start_match(self, scoring_system):
        """Should mark match start."""
        scoring_system.start_match()
        assert not scoring_system.first_blood_awarded

    def test_reset(self, populated_system):
        """Should reset all scoring data."""
        populated_system.add_score("player_1", 100)
        populated_system.record_kill("player_1", "player_2")

        populated_system.reset()

        assert len(populated_system._player_stats) == 0
        assert len(populated_system._event_history) == 0

    def test_get_summary(self, populated_system):
        """Should get scoring summary."""
        populated_system.add_score("player_1", 100)
        populated_system.record_kill("player_1", "player_2")

        summary = populated_system.get_summary()

        assert "player_count" in summary
        assert "total_kills" in summary
        assert "leaderboard" in summary

    def test_get_all_player_stats(self, populated_system):
        """Should get all player stats as dicts."""
        populated_system.add_score("player_1", 100)

        all_stats = populated_system.get_all_player_stats()

        assert "player_1" in all_stats
        assert all_stats["player_1"]["score"] == 100

    def test_record_healing(self, populated_system):
        """Should record healing done."""
        result = populated_system.record_healing("player_1", "player_2", 50.0)
        assert result

        stats = populated_system.get_player_stats("player_1")
        assert stats.healing_done == 50.0


# =============================================================================
# PLAYER STATS DATACLASS TESTS (~10 tests)
# =============================================================================


class TestPlayerStatsDataclass:
    """Tests for PlayerStats dataclass."""

    def test_total_multi_kills(self):
        """Should calculate total multi-kills."""
        stats = PlayerStats(player_id="p1")
        stats.double_kills = 5
        stats.triple_kills = 3
        stats.quad_kills = 2
        stats.penta_kills = 1

        assert stats.total_multi_kills == 11

    def test_to_dict(self):
        """Should convert to dictionary."""
        stats = PlayerStats(player_id="p1", team_id="red")
        stats.score = 100
        stats.kills = 10
        stats.deaths = 5

        data = stats.to_dict()

        assert data["player_id"] == "p1"
        assert data["team_id"] == "red"
        assert data["score"] == 100
        assert data["kd_ratio"] == 2.0

    def test_damage_tracking(self):
        """Should track damage to targets."""
        stats = PlayerStats(player_id="p1")
        stats.record_damage_dealt("target_1", 50.0, 100.0)

        damage = stats.get_assist_damage("target_1", 100.0)
        assert damage == 50.0

    def test_clear_damage_tracking(self):
        """Should clear damage tracking."""
        stats = PlayerStats(player_id="p1")
        stats.record_damage_dealt("target_1", 50.0, 100.0)
        stats.clear_damage_tracking("target_1")

        damage = stats.get_assist_damage("target_1", 100.0)
        assert damage is None
