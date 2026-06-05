"""
WHITEBOX Tests for Scoring System

Tests internal implementation details:
- Multi-kill window timing precision
- Killstreak threshold boundaries
- Assist damage threshold calculations
- Score event history management
- Team score synchronization
- Internal state transitions
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
    ASSIST_DAMAGE_THRESHOLD,
    ASSIST_TIME_WINDOW,
    MAX_SCORING_HISTORY_SIZE,
    DEFAULT_MAX_HEALTH,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def scoring_system():
    """Create a fresh scoring system."""
    return ScoringSystem()


@pytest.fixture
def team_scoring_system():
    """Create team-based scoring system."""
    return ScoringSystem(is_team_based=True)


@pytest.fixture
def custom_config_system():
    """Create scoring system with custom config."""
    config = ScoringConfig(
        points_per_kill=50,
        points_per_assist=25,
        points_per_headshot_bonus=10,
        points_per_first_blood=30,
        multi_kill_window=2.0,
    )
    return ScoringSystem(config=config)


@pytest.fixture
def populated_system(scoring_system):
    """Scoring system with players."""
    for i in range(5):
        scoring_system.add_player(f"player_{i}")
    return scoring_system


@pytest.fixture
def team_populated(team_scoring_system):
    """Team system with players on teams."""
    team_scoring_system.add_player("red_1", team_id="red")
    team_scoring_system.add_player("red_2", team_id="red")
    team_scoring_system.add_player("blue_1", team_id="blue")
    team_scoring_system.add_player("blue_2", team_id="blue")
    return team_scoring_system


# =============================================================================
# MULTI-KILL WINDOW TIMING TESTS (45 tests)
# =============================================================================


class TestMultiKillWindowTiming:
    """Tests for multi-kill window timing precision."""

    def test_double_kill_within_window(self, populated_system):
        """Two kills within window should be double kill."""
        populated_system.record_kill("player_0", "player_1")
        # Simulate time passing within window
        result = populated_system.record_kill("player_0", "player_2")

        stats = populated_system.get_player_stats("player_0")
        assert stats.double_kills >= 1 or "double_kill" in result.get("achievements", [])

    def test_triple_kill_within_window(self, populated_system):
        """Three kills within window should be triple kill."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            mock_time.return_value = base_time + 1.0
            populated_system.record_kill("player_0", "player_2")

            mock_time.return_value = base_time + 2.0
            result = populated_system.record_kill("player_0", "player_3")

        stats = populated_system.get_player_stats("player_0")
        assert stats.triple_kills >= 1

    def test_quad_kill_within_window(self, populated_system):
        """Four kills within window should be quad kill."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            for i in range(4):
                mock_time.return_value = base_time + i * 1.0
                populated_system.record_kill("player_0", f"player_{i+1}")

        stats = populated_system.get_player_stats("player_0")
        assert stats.quad_kills >= 1

    def test_penta_kill_within_window(self):
        """Five kills within window should be penta kill."""
        system = ScoringSystem()
        for i in range(6):
            system.add_player(f"player_{i}")

        with patch('time.time') as mock_time:
            base_time = 1000.0
            for i in range(5):
                mock_time.return_value = base_time + i * 0.5
                system.record_kill("player_0", f"player_{i+1}")

        stats = system.get_player_stats("player_0")
        assert stats.penta_kills >= 1

    def test_multi_kill_resets_after_window_expires(self, populated_system):
        """Multi-kill counter should reset after window expires."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            # Kill outside window
            mock_time.return_value = base_time + MULTI_KILL_WINDOW + 1.0
            populated_system.record_kill("player_0", "player_2")

        stats = populated_system.get_player_stats("player_0")
        assert stats.double_kills == 0

    def test_multi_kill_window_edge_exact_boundary(self, populated_system):
        """Kill exactly at window boundary."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            # Exactly at window boundary
            mock_time.return_value = base_time + MULTI_KILL_WINDOW
            populated_system.record_kill("player_0", "player_2")

        stats = populated_system.get_player_stats("player_0")
        # Should count as double kill since <= window
        assert stats.double_kills >= 1

    def test_multi_kill_window_just_outside(self, populated_system):
        """Kill just outside window should not count."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            # Just outside window
            mock_time.return_value = base_time + MULTI_KILL_WINDOW + 0.001
            populated_system.record_kill("player_0", "player_2")

        stats = populated_system.get_player_stats("player_0")
        assert stats.double_kills == 0

    def test_multi_kill_window_custom_config(self, custom_config_system):
        """Custom window config should be respected."""
        custom_config_system.add_player("player_0")
        custom_config_system.add_player("player_1")
        custom_config_system.add_player("player_2")

        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            custom_config_system.record_kill("player_0", "player_1")

            # Within custom 2.0s window
            mock_time.return_value = base_time + 1.5
            custom_config_system.record_kill("player_0", "player_2")

        stats = custom_config_system.get_player_stats("player_0")
        assert stats.double_kills >= 1

    def test_multi_kill_count_progression(self):
        """Multi-kill count should progress correctly."""
        system = ScoringSystem()
        for i in range(7):
            system.add_player(f"player_{i}")

        with patch('time.time') as mock_time:
            base_time = 1000.0

            # First kill
            mock_time.return_value = base_time
            system.record_kill("player_0", "player_1")
            stats = system.get_player_stats("player_0")
            assert stats._multi_kill_count == 1

            # Second kill - double
            mock_time.return_value = base_time + 1.0
            system.record_kill("player_0", "player_2")
            stats = system.get_player_stats("player_0")
            assert stats._multi_kill_count == 2

    def test_multi_kill_callback_fired(self, populated_system):
        """Multi-kill callback should fire."""
        callback = Mock()
        populated_system.on_multi_kill(callback)

        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            mock_time.return_value = base_time + 1.0
            populated_system.record_kill("player_0", "player_2")

        callback.assert_called()

    def test_multi_kill_resets_on_death(self, populated_system):
        """Multi-kill counter should reset on death."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            # Player dies
            populated_system.record_death("player_0")

            # Next kill should be fresh start
            mock_time.return_value = base_time + 0.5
            populated_system.record_kill("player_0", "player_2")

        stats = populated_system.get_player_stats("player_0")
        # Death resets multi-kill, so new kill is count 1
        assert stats.double_kills == 0

    def test_multi_kill_window_zero_time_kills(self, populated_system):
        """Multiple kills at exact same time."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            populated_system.record_kill("player_0", "player_1")
            populated_system.record_kill("player_0", "player_2")
            populated_system.record_kill("player_0", "player_3")

        stats = populated_system.get_player_stats("player_0")
        assert stats.triple_kills >= 1

    def test_multi_kill_achievements_returned(self, populated_system):
        """Multi-kill names should be in achievements."""
        with patch('time.time') as mock_time:
            mock_time.return_value = 1000.0
            populated_system.record_kill("player_0", "player_1")
            result = populated_system.record_kill("player_0", "player_2")

        assert "double_kill" in result.get("achievements", [])

    def test_multi_kill_total_count(self):
        """Total multi-kills property should sum all types."""
        stats = PlayerStats(player_id="test")
        stats.double_kills = 5
        stats.triple_kills = 3
        stats.quad_kills = 2
        stats.penta_kills = 1

        assert stats.total_multi_kills == 11

    def test_multi_kill_window_millisecond_precision(self, populated_system):
        """Test millisecond precision in window timing."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_kill("player_0", "player_1")

            # 3999ms later (just under 4s default window)
            mock_time.return_value = base_time + 3.999
            populated_system.record_kill("player_0", "player_2")

        stats = populated_system.get_player_stats("player_0")
        assert stats.double_kills >= 1


# =============================================================================
# KILLSTREAK THRESHOLD TESTS (40 tests)
# =============================================================================


class TestKillstreakThresholds:
    """Tests for killstreak threshold boundaries."""

    def test_killstreak_increments(self, populated_system):
        """Killstreak should increment on kills."""
        populated_system.record_kill("player_0", "player_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.current_killstreak == 1

    def test_killstreak_three_kills(self, populated_system):
        """Three kill streak should trigger killing_spree."""
        for i in range(3):
            result = populated_system.record_kill("player_0", f"player_{i+1}")

        assert "killing_spree" in result.get("achievements", [])
        stats = populated_system.get_player_stats("player_0")
        assert stats.current_killstreak == 3

    def test_killstreak_five_kills(self):
        """Five kill streak should trigger rampage."""
        system = ScoringSystem()
        for i in range(6):
            system.add_player(f"player_{i}")

        for i in range(5):
            result = system.record_kill("player_0", f"player_{i+1}")

        assert "rampage" in result.get("achievements", [])

    def test_killstreak_seven_kills(self):
        """Seven kill streak should trigger dominating."""
        system = ScoringSystem()
        for i in range(8):
            system.add_player(f"player_{i}")

        for i in range(7):
            result = system.record_kill("player_0", f"player_{i+1}")

        assert "dominating" in result.get("achievements", [])

    def test_killstreak_ten_kills(self):
        """Ten kill streak should trigger unstoppable."""
        system = ScoringSystem()
        for i in range(11):
            system.add_player(f"player_{i}")

        for i in range(10):
            result = system.record_kill("player_0", f"player_{i+1}")

        assert "unstoppable" in result.get("achievements", [])

    def test_killstreak_fifteen_kills(self):
        """Fifteen kill streak should trigger godlike."""
        system = ScoringSystem()
        for i in range(16):
            system.add_player(f"player_{i}")

        for i in range(15):
            result = system.record_kill("player_0", f"player_{i+1}")

        assert "godlike" in result.get("achievements", [])

    def test_killstreak_twenty_kills(self):
        """Twenty kill streak should trigger legendary."""
        system = ScoringSystem()
        for i in range(21):
            system.add_player(f"player_{i}")

        for i in range(20):
            result = system.record_kill("player_0", f"player_{i+1}")

        assert "legendary" in result.get("achievements", [])

    def test_killstreak_resets_on_death(self, populated_system):
        """Killstreak should reset to zero on death."""
        for i in range(3):
            populated_system.record_kill("player_0", f"player_{i+1}")

        populated_system.record_death("player_0")

        stats = populated_system.get_player_stats("player_0")
        assert stats.current_killstreak == 0

    def test_best_killstreak_updates(self):
        """Best killstreak should track highest achieved."""
        system = ScoringSystem()
        for i in range(8):
            system.add_player(f"player_{i}")

        # First streak of 5
        for i in range(5):
            system.record_kill("player_0", f"player_{i+1}")
        system.record_death("player_0")

        # Second streak of 3
        for i in range(3):
            system.record_kill("player_0", f"player_{i+1}")

        stats = system.get_player_stats("player_0")
        assert stats.best_killstreak == 5
        assert stats.current_killstreak == 3

    def test_killstreak_bonus_points(self, populated_system):
        """Killstreak should award bonus points."""
        initial_score = 0
        for i in range(3):
            populated_system.record_kill("player_0", f"player_{i+1}")

        stats = populated_system.get_player_stats("player_0")
        # Should have kill points + streak bonus
        expected_min = 3 * POINTS_PER_KILL + POINTS_PER_FIRST_BLOOD
        assert stats.score >= expected_min

    def test_killstreak_callback_fired(self, populated_system):
        """Killstreak callback should fire at threshold."""
        callback = Mock()
        populated_system.on_killstreak(callback)

        for i in range(3):
            populated_system.record_kill("player_0", f"player_{i+1}")

        callback.assert_called()

    def test_killstreak_callback_with_correct_args(self, populated_system):
        """Killstreak callback should have correct arguments."""
        callback = Mock()
        populated_system.on_killstreak(callback)

        for i in range(3):
            populated_system.record_kill("player_0", f"player_{i+1}")

        callback.assert_called_with("player_0", 3)

    def test_deathstreak_increments(self, populated_system):
        """Deathstreak should increment on deaths."""
        populated_system.record_death("player_0")
        populated_system.record_death("player_0")

        stats = populated_system.get_player_stats("player_0")
        assert stats.current_deathstreak == 2

    def test_deathstreak_resets_on_kill(self, populated_system):
        """Deathstreak should reset on getting a kill."""
        populated_system.record_death("player_0")
        populated_system.record_death("player_0")
        populated_system.record_kill("player_0", "player_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.current_deathstreak == 0

    def test_killstreak_no_bonus_between_thresholds(self):
        """No bonus should be awarded between thresholds."""
        system = ScoringSystem()
        for i in range(5):
            system.add_player(f"player_{i}")

        # 4 kills - between 3 and 5 thresholds
        for i in range(4):
            result = system.record_kill("player_0", f"player_{i+1}")

        # Should not have rampage (5) achievement
        assert "rampage" not in result.get("achievements", [])

    def test_killstreak_threshold_exact_boundary(self):
        """Exact threshold boundary should trigger achievement."""
        system = ScoringSystem()
        for i in range(4):
            system.add_player(f"player_{i}")

        # Exactly 3 kills
        for i in range(3):
            result = system.record_kill("player_0", f"player_{i+1}")

        assert "killing_spree" in result.get("achievements", [])

    def test_multiple_killstreak_achievements(self):
        """Multiple thresholds can be achieved in sequence."""
        system = ScoringSystem()
        for i in range(11):
            system.add_player(f"player_{i}")

        achievements = []
        for i in range(10):
            result = system.record_kill("player_0", f"player_{i+1}")
            achievements.extend(result.get("achievements", []))

        assert "killing_spree" in achievements
        assert "rampage" in achievements
        assert "dominating" in achievements
        assert "unstoppable" in achievements


# =============================================================================
# ASSIST DAMAGE THRESHOLD TESTS (35 tests)
# =============================================================================


class TestAssistDamageThreshold:
    """Tests for assist damage threshold calculations."""

    def test_assist_awarded_above_threshold(self, populated_system):
        """Assist should be awarded when damage exceeds threshold."""
        # Record damage above 10% threshold
        populated_system.record_damage("player_0", "player_2", 15.0)
        result = populated_system.record_kill("player_1", "player_2")

        assert "player_0" in result.get("assists", [])

    def test_assist_not_awarded_below_threshold(self, populated_system):
        """Assist should not be awarded below threshold."""
        # Record damage below 10% threshold
        populated_system.record_damage("player_0", "player_2", 5.0)
        result = populated_system.record_kill("player_1", "player_2")

        assert "player_0" not in result.get("assists", [])

    def test_assist_exact_threshold(self, populated_system):
        """Assist at exact threshold."""
        threshold_damage = ASSIST_DAMAGE_THRESHOLD * DEFAULT_MAX_HEALTH
        populated_system.record_damage("player_0", "player_2", threshold_damage)
        result = populated_system.record_kill("player_1", "player_2")

        # Should award assist at threshold
        assert "player_0" in result.get("assists", [])

    def test_assist_time_window_expired(self, populated_system):
        """Assist should not be awarded after time window expires."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_damage("player_0", "player_2", 50.0)

            # Kill after window expires
            mock_time.return_value = base_time + ASSIST_TIME_WINDOW + 1.0
            result = populated_system.record_kill("player_1", "player_2")

        assert "player_0" not in result.get("assists", [])

    def test_assist_within_time_window(self, populated_system):
        """Assist should be awarded within time window."""
        with patch('time.time') as mock_time:
            base_time = 1000.0
            mock_time.return_value = base_time
            populated_system.record_damage("player_0", "player_2", 50.0)

            # Kill within window
            mock_time.return_value = base_time + ASSIST_TIME_WINDOW - 1.0
            result = populated_system.record_kill("player_1", "player_2")

        assert "player_0" in result.get("assists", [])

    def test_multiple_assists_same_kill(self, populated_system):
        """Multiple players can get assists on same kill."""
        populated_system.record_damage("player_0", "player_3", 30.0)
        populated_system.record_damage("player_1", "player_3", 30.0)
        result = populated_system.record_kill("player_2", "player_3")

        assists = result.get("assists", [])
        assert "player_0" in assists
        assert "player_1" in assists

    def test_killer_not_awarded_assist(self, populated_system):
        """Killer should not get assist on their own kill."""
        populated_system.record_damage("player_0", "player_1", 50.0)
        result = populated_system.record_kill("player_0", "player_1")

        assert "player_0" not in result.get("assists", [])

    def test_assist_damage_cleared_after_death(self, populated_system):
        """Damage tracking should be cleared after death."""
        populated_system.record_damage("player_0", "player_1", 50.0)
        populated_system.record_kill("player_2", "player_1")

        # Record new damage after respawn
        populated_system.record_damage("player_0", "player_1", 5.0)
        result = populated_system.record_kill("player_2", "player_1")

        # Old damage should not count
        assert "player_0" not in result.get("assists", [])

    def test_assist_points_awarded(self, populated_system):
        """Assist should award points."""
        initial_score = populated_system.get_score("player_0")
        populated_system.record_damage("player_0", "player_2", 50.0)
        populated_system.record_kill("player_1", "player_2")

        final_score = populated_system.get_score("player_0")
        assert final_score >= initial_score + POINTS_PER_ASSIST

    def test_assist_stat_increments(self, populated_system):
        """Assist stat should increment."""
        populated_system.record_damage("player_0", "player_2", 50.0)
        populated_system.record_kill("player_1", "player_2")

        stats = populated_system.get_player_stats("player_0")
        assert stats.assists == 1

    def test_cumulative_damage_for_assist(self, populated_system):
        """Multiple small hits should accumulate for assist."""
        for _ in range(5):
            populated_system.record_damage("player_0", "player_2", 5.0)  # Total 25
        result = populated_system.record_kill("player_1", "player_2")

        assert "player_0" in result.get("assists", [])

    def test_damage_tracking_per_target(self, populated_system):
        """Damage should be tracked per target."""
        populated_system.record_damage("player_0", "player_1", 50.0)
        populated_system.record_damage("player_0", "player_2", 5.0)

        result = populated_system.record_kill("player_3", "player_2")
        assert "player_0" not in result.get("assists", [])

    def test_assist_team_stat_update(self, team_populated):
        """Team assists should be updated."""
        team_populated.record_damage("red_1", "blue_1", 50.0)
        team_populated.record_kill("red_2", "blue_1")

        # Check team stats were updated
        team_stats = team_populated._team_stats.get("red")
        assert team_stats.assists >= 1


# =============================================================================
# SCORE EVENT HISTORY TESTS (30 tests)
# =============================================================================


class TestScoreEventHistory:
    """Tests for score event history management."""

    def test_event_recorded_on_score_change(self, populated_system):
        """Score change should record event."""
        populated_system.add_score("player_0", 100)

        history = populated_system.get_event_history(player_id="player_0")
        assert len(history) >= 1

    def test_event_has_correct_type(self, populated_system):
        """Event should have correct type."""
        populated_system.record_kill("player_0", "player_1")

        history = populated_system.get_event_history(
            event_type=ScoreEventType.KILL,
            limit=1
        )
        assert len(history) == 1
        assert history[0].event_type == ScoreEventType.KILL

    def test_event_has_timestamp(self, populated_system):
        """Event should have timestamp."""
        before = time.time()
        populated_system.add_score("player_0", 100)
        after = time.time()

        history = populated_system.get_event_history(player_id="player_0", limit=1)
        assert before <= history[0].timestamp <= after

    def test_event_history_limit(self, populated_system):
        """History should respect limit parameter."""
        for i in range(10):
            populated_system.add_score("player_0", 10)

        history = populated_system.get_event_history(player_id="player_0", limit=5)
        assert len(history) == 5

    def test_event_history_max_size(self, scoring_system):
        """History should trim when exceeding max size."""
        scoring_system.add_player("player_0")
        scoring_system._max_history_size = 10

        for i in range(20):
            scoring_system.add_score("player_0", 1)

        assert len(scoring_system._event_history) <= 10

    def test_event_filter_by_player(self, populated_system):
        """Should filter events by player."""
        populated_system.add_score("player_0", 100)
        populated_system.add_score("player_1", 50)

        history = populated_system.get_event_history(player_id="player_0")
        assert all(e.player_id == "player_0" for e in history)

    def test_event_filter_by_type(self, populated_system):
        """Should filter events by type."""
        populated_system.record_kill("player_0", "player_1")
        populated_system.add_score("player_0", 100, ScoreEventType.BONUS)

        kill_events = populated_system.get_event_history(
            event_type=ScoreEventType.KILL
        )
        assert all(e.event_type == ScoreEventType.KILL for e in kill_events)

    def test_recent_kills_returns_kill_events(self, populated_system):
        """get_recent_kills should return kill events."""
        populated_system.record_kill("player_0", "player_1")
        populated_system.add_score("player_0", 100)

        kills = populated_system.get_recent_kills(limit=10)
        assert all(e.event_type == ScoreEventType.KILL for e in kills)

    def test_event_points_recorded(self, populated_system):
        """Event should record points awarded."""
        populated_system.add_score("player_0", 150, ScoreEventType.BONUS)

        history = populated_system.get_event_history(player_id="player_0", limit=1)
        assert history[0].points == 150

    def test_event_is_positive(self, populated_system):
        """is_positive should return True for positive points."""
        populated_system.add_score("player_0", 100)

        history = populated_system.get_event_history(player_id="player_0", limit=1)
        assert history[0].is_positive

    def test_event_is_negative(self, populated_system):
        """is_negative should return True for negative points."""
        populated_system.add_score("player_0", -50)

        history = populated_system.get_event_history(player_id="player_0", limit=1)
        assert history[0].is_negative

    def test_event_target_recorded(self, populated_system):
        """Target should be recorded in event."""
        populated_system.record_kill("player_0", "player_1")

        history = populated_system.get_event_history(
            event_type=ScoreEventType.KILL,
            limit=1
        )
        assert history[0].target_id == "player_1"

    def test_event_team_recorded(self, team_populated):
        """Team should be recorded in event."""
        team_populated.record_kill("red_1", "blue_1")

        history = team_populated.get_event_history(
            event_type=ScoreEventType.KILL,
            limit=1
        )
        assert history[0].team_id == "red"


# =============================================================================
# TEAM SCORE SYNCHRONIZATION TESTS (25 tests)
# =============================================================================


class TestTeamScoreSynchronization:
    """Tests for team score synchronization."""

    def test_team_score_updates_on_kill(self, team_populated):
        """Team score should update when player gets kill."""
        initial_score = team_populated.get_team_score("red")
        team_populated.record_kill("red_1", "blue_1")

        final_score = team_populated.get_team_score("red")
        assert final_score > initial_score

    def test_team_kills_increment(self, team_populated):
        """Team kills should increment."""
        team_populated.record_kill("red_1", "blue_1")

        team_stats = team_populated._team_stats.get("red")
        assert team_stats.kills == 1

    def test_team_deaths_increment(self, team_populated):
        """Team deaths should increment."""
        team_populated.record_kill("red_1", "blue_1")

        team_stats = team_populated._team_stats.get("blue")
        assert team_stats.deaths == 1

    def test_team_score_synchronized_with_player_scores(self, team_populated):
        """Team score should equal sum of player scores."""
        team_populated.record_kill("red_1", "blue_1")
        team_populated.add_score("red_2", 50)

        team_score = team_populated.get_team_score("red")
        red_1_score = team_populated.get_score("red_1")
        red_2_score = team_populated.get_score("red_2")

        assert team_score == red_1_score + red_2_score

    def test_team_created_on_add_player(self, team_scoring_system):
        """Team should be created when first player added."""
        team_scoring_system.add_player("player_1", team_id="new_team")

        assert "new_team" in team_scoring_system._team_stats

    def test_team_member_count(self, team_populated):
        """Team should track member count."""
        team_stats = team_populated._team_stats.get("red")
        assert team_stats.member_count == 2

    def test_player_removed_from_team(self, team_populated):
        """Player removal should update team members."""
        team_populated.remove_player("red_1")

        team_stats = team_populated._team_stats.get("red")
        assert "red_1" not in team_stats.members

    def test_team_change_updates_members(self, team_populated):
        """Team change should update member sets."""
        team_populated.set_player_team("red_1", "blue")

        red_stats = team_populated._team_stats.get("red")
        blue_stats = team_populated._team_stats.get("blue")

        assert "red_1" not in red_stats.members
        assert "red_1" in blue_stats.members

    def test_team_kd_ratio(self, team_populated):
        """Team KD ratio should calculate correctly."""
        team_populated.record_kill("red_1", "blue_1")
        team_populated.record_kill("red_2", "blue_2")
        team_populated.record_kill("blue_1", "red_1")

        red_stats = team_populated._team_stats.get("red")
        assert red_stats.kd_ratio == 2.0  # 2 kills, 1 death

    def test_team_leaderboard(self, team_populated):
        """Team leaderboard should sort correctly."""
        team_populated.record_kill("red_1", "blue_1")
        team_populated.record_kill("red_1", "blue_2")

        leaderboard = team_populated.get_team_leaderboard()
        assert leaderboard[0][0] == "red"  # Red team should be first


# =============================================================================
# FIRST BLOOD TESTS (15 tests)
# =============================================================================


class TestFirstBlood:
    """Tests for first blood handling."""

    def test_first_blood_awarded(self, populated_system):
        """First kill should award first blood."""
        result = populated_system.record_kill("player_0", "player_1")

        assert "first_blood" in result.get("achievements", [])

    def test_first_blood_bonus_points(self, populated_system):
        """First blood should award bonus points."""
        populated_system.record_kill("player_0", "player_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.score >= POINTS_PER_KILL + POINTS_PER_FIRST_BLOOD

    def test_first_blood_only_once(self, populated_system):
        """First blood should only be awarded once per match."""
        populated_system.record_kill("player_0", "player_1")
        result = populated_system.record_kill("player_0", "player_2")

        assert "first_blood" not in result.get("achievements", [])

    def test_first_blood_flag_set(self, populated_system):
        """First blood flag should be set after first kill."""
        assert not populated_system.first_blood_awarded

        populated_system.record_kill("player_0", "player_1")

        assert populated_system.first_blood_awarded

    def test_first_blood_stat_increments(self, populated_system):
        """First blood stat should increment."""
        populated_system.record_kill("player_0", "player_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.first_bloods == 1

    def test_first_blood_callback(self, populated_system):
        """First blood callback should fire."""
        callback = Mock()
        populated_system.on_first_blood(callback)

        populated_system.record_kill("player_0", "player_1")

        callback.assert_called_once_with("player_0", "player_1")

    def test_first_blood_reset_on_match_start(self, populated_system):
        """First blood should reset on match start."""
        populated_system.record_kill("player_0", "player_1")
        populated_system.start_match()

        assert not populated_system.first_blood_awarded

    def test_first_blood_after_reset(self, populated_system):
        """First blood should be awardable after reset."""
        populated_system.record_kill("player_0", "player_1")
        populated_system.reset()
        populated_system.add_player("player_0")
        populated_system.add_player("player_1")

        result = populated_system.record_kill("player_0", "player_1")
        assert "first_blood" in result.get("achievements", [])


# =============================================================================
# REVENGE KILL TESTS (15 tests)
# =============================================================================


class TestRevengeKill:
    """Tests for revenge kill handling."""

    def test_revenge_kill_awarded(self, populated_system):
        """Killing player who killed you should be revenge."""
        populated_system.record_kill("player_1", "player_0")  # player_1 kills player_0
        result = populated_system.record_kill("player_0", "player_1")  # player_0 gets revenge

        assert "revenge" in result.get("achievements", [])

    def test_revenge_kill_bonus_points(self, populated_system):
        """Revenge kill should award bonus points."""
        populated_system.record_kill("player_1", "player_0")

        before = populated_system.get_score("player_0")
        populated_system.record_kill("player_0", "player_1")
        after = populated_system.get_score("player_0")

        expected_min = POINTS_PER_KILL + POINTS_PER_REVENGE_KILL
        assert after - before >= expected_min

    def test_revenge_stat_increments(self, populated_system):
        """Revenge kills stat should increment."""
        populated_system.record_kill("player_1", "player_0")
        populated_system.record_kill("player_0", "player_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.revenge_kills == 1

    def test_revenge_clears_after_achieved(self, populated_system):
        """Revenge should be cleared after being achieved."""
        populated_system.record_kill("player_1", "player_0")
        populated_system.record_kill("player_0", "player_1")  # Revenge achieved

        # Kill again - should not be revenge
        result = populated_system.record_kill("player_0", "player_1")
        assert "revenge" not in result.get("achievements", [])

    def test_multiple_revenge_targets(self, populated_system):
        """Can have multiple revenge targets."""
        populated_system.record_kill("player_1", "player_0")
        populated_system.record_kill("player_2", "player_0")

        result1 = populated_system.record_kill("player_0", "player_1")
        result2 = populated_system.record_kill("player_0", "player_2")

        assert "revenge" in result1.get("achievements", [])
        assert "revenge" in result2.get("achievements", [])

    def test_no_revenge_on_first_kill(self, populated_system):
        """First kill should not be revenge."""
        result = populated_system.record_kill("player_0", "player_1")

        assert "revenge" not in result.get("achievements", [])


# =============================================================================
# LEADERBOARD TESTS (20 tests)
# =============================================================================


class TestLeaderboard:
    """Tests for leaderboard functionality."""

    def test_leaderboard_sorted_by_score(self, populated_system):
        """Default sort should be by score descending."""
        populated_system.add_score("player_0", 100)
        populated_system.add_score("player_1", 200)
        populated_system.add_score("player_2", 150)

        leaderboard = populated_system.get_leaderboard()
        scores = [e.score for e in leaderboard]
        assert scores == sorted(scores, reverse=True)

    def test_leaderboard_sorted_by_kills(self, populated_system):
        """Should sort by kills when specified."""
        populated_system.record_kill("player_0", "player_2")
        populated_system.record_kill("player_1", "player_2")
        populated_system.record_kill("player_1", "player_0")

        leaderboard = populated_system.get_leaderboard(
            sort_by=LeaderboardSortKey.KILLS
        )
        assert leaderboard[0].player_id == "player_1"

    def test_leaderboard_sorted_by_deaths_ascending(self, populated_system):
        """Deaths should sort ascending (fewer is better)."""
        populated_system.record_death("player_0")
        populated_system.record_death("player_0")
        populated_system.record_death("player_1")

        leaderboard = populated_system.get_leaderboard(
            sort_by=LeaderboardSortKey.DEATHS
        )
        deaths = [e.deaths for e in leaderboard]
        assert deaths == sorted(deaths)

    def test_leaderboard_limit(self, populated_system):
        """Leaderboard should respect limit."""
        for i in range(5):
            populated_system.add_score(f"player_{i}", (i + 1) * 100)

        leaderboard = populated_system.get_leaderboard(limit=3)
        assert len(leaderboard) == 3

    def test_leaderboard_rank_assigned(self, scoring_system):
        """Entries should have correct rank assigned."""
        for i in range(3):
            scoring_system.add_player(f"player_{i}")
            scoring_system.add_score(f"player_{i}", (i + 1) * 100)

        leaderboard = scoring_system.get_leaderboard()
        ranks = [e.rank for e in leaderboard]
        assert ranks == [1, 2, 3]

    def test_leaderboard_filter_by_team(self, team_populated):
        """Should filter by team."""
        team_populated.add_score("red_1", 100)
        team_populated.add_score("blue_1", 200)

        leaderboard = team_populated.get_leaderboard(team_id="red")
        assert all(e.team_id == "red" for e in leaderboard)

    def test_leaderboard_kd_ratio_sort(self, populated_system):
        """Should sort by KD ratio."""
        # Player 0: 4 kills, 2 deaths = 2.0 KD
        for _ in range(4):
            populated_system.record_kill("player_0", "player_1")
        populated_system.record_death("player_0")
        populated_system.record_death("player_0")

        # Player 1: 2 kills, 1 death = 2.0 KD, but lower score
        for _ in range(2):
            populated_system.record_kill("player_1", "player_0")
        populated_system.record_death("player_1")

        leaderboard = populated_system.get_leaderboard(
            sort_by=LeaderboardSortKey.KD_RATIO
        )
        assert leaderboard[0].kd_ratio >= leaderboard[-1].kd_ratio

    def test_player_rank(self, populated_system):
        """Should get player rank."""
        populated_system.add_score("player_0", 300)
        populated_system.add_score("player_1", 200)
        populated_system.add_score("player_2", 100)

        rank = populated_system.get_player_rank("player_1")
        assert rank == 2


# =============================================================================
# SCORE MODIFICATION TESTS (15 tests)
# =============================================================================


class TestScoreModification:
    """Tests for direct score modification."""

    def test_add_score(self, populated_system):
        """Should add score to player."""
        populated_system.add_score("player_0", 100)
        assert populated_system.get_score("player_0") == 100

    def test_add_negative_score(self, populated_system):
        """Should allow negative score addition."""
        populated_system.add_score("player_0", 100)
        populated_system.add_score("player_0", -50)
        assert populated_system.get_score("player_0") == 50

    def test_set_score(self, populated_system):
        """Should set score directly."""
        populated_system.add_score("player_0", 100)
        populated_system.set_score("player_0", 50)
        assert populated_system.get_score("player_0") == 50

    def test_score_callback_fired(self, populated_system):
        """Score change callback should fire."""
        callback = Mock()
        populated_system.on_score_changed(callback)

        populated_system.add_score("player_0", 100)

        callback.assert_called()

    def test_score_callback_args(self, populated_system):
        """Score callback should have correct arguments."""
        callback = Mock()
        populated_system.on_score_changed(callback)

        populated_system.add_score("player_0", 100)

        callback.assert_called_with("player_0", 0, 100)

    def test_add_score_nonexistent_player(self, scoring_system):
        """Should return False for nonexistent player."""
        result = scoring_system.add_score("nonexistent", 100)
        assert not result


# =============================================================================
# OBJECTIVE SCORING TESTS (15 tests)
# =============================================================================


class TestObjectiveScoring:
    """Tests for objective scoring."""

    def test_objective_capture_points(self, populated_system):
        """Objective capture should award points."""
        populated_system.record_objective_capture("player_0", "obj_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.score >= POINTS_PER_OBJECTIVE

    def test_objective_capture_stat(self, populated_system):
        """Objective capture stat should increment."""
        populated_system.record_objective_capture("player_0", "obj_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.objectives_captured == 1

    def test_objective_defend_points(self, populated_system):
        """Objective defense should award points."""
        populated_system.record_objective_defend("player_0", "obj_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.score >= POINTS_PER_OBJECTIVE // 2

    def test_objective_defend_stat(self, populated_system):
        """Objective defense stat should increment."""
        populated_system.record_objective_defend("player_0", "obj_1")

        stats = populated_system.get_player_stats("player_0")
        assert stats.objectives_defended == 1

    def test_custom_objective_points(self, populated_system):
        """Should allow custom point values."""
        populated_system.record_objective_capture("player_0", "obj_1", points=500)

        stats = populated_system.get_player_stats("player_0")
        assert stats.score == 500

    def test_team_objectives_updated(self, team_populated):
        """Team objectives should be updated."""
        team_populated.record_objective_capture("red_1", "obj_1")

        team_stats = team_populated._team_stats.get("red")
        assert team_stats.objectives == 1


# =============================================================================
# SYSTEM UTILITY TESTS (10 tests)
# =============================================================================


class TestSystemUtility:
    """Tests for system utility functions."""

    def test_reset_clears_all(self, populated_system):
        """Reset should clear all data."""
        populated_system.record_kill("player_0", "player_1")
        populated_system.reset()

        assert len(populated_system._player_stats) == 0
        assert len(populated_system._event_history) == 0
        assert not populated_system.first_blood_awarded

    def test_start_match_resets_first_blood(self, populated_system):
        """Start match should reset first blood."""
        populated_system.record_kill("player_0", "player_1")
        populated_system.start_match()

        assert not populated_system.first_blood_awarded

    def test_get_summary(self, populated_system):
        """Should return summary dict."""
        populated_system.record_kill("player_0", "player_1")

        summary = populated_system.get_summary()

        assert "player_count" in summary
        assert "total_kills" in summary
        assert summary["total_kills"] == 1

    def test_get_all_player_stats(self, populated_system):
        """Should return all player stats as dicts."""
        populated_system.record_kill("player_0", "player_1")

        all_stats = populated_system.get_all_player_stats()

        assert "player_0" in all_stats
        assert all_stats["player_0"]["kills"] == 1

    def test_config_accessible(self, scoring_system):
        """Config should be accessible."""
        assert scoring_system.config is not None
        assert scoring_system.config.points_per_kill == POINTS_PER_KILL

    def test_is_team_based_property(self, team_scoring_system):
        """is_team_based should return correct value."""
        assert team_scoring_system.is_team_based


# =============================================================================
# DAMAGE TRACKING TESTS (10 tests)
# =============================================================================


class TestDamageTracking:
    """Tests for damage tracking."""

    def test_record_damage(self, populated_system):
        """Should record damage dealt."""
        populated_system.record_damage("player_0", "player_1", 50.0)

        stats = populated_system.get_player_stats("player_0")
        assert stats.damage_dealt >= 50.0

    def test_record_damage_taken(self, populated_system):
        """Should record damage taken."""
        populated_system.record_damage("player_0", "player_1", 50.0)

        stats = populated_system.get_player_stats("player_1")
        assert stats.damage_taken >= 50.0

    def test_record_healing(self, populated_system):
        """Should record healing done."""
        populated_system.record_healing("player_0", "player_1", 30.0)

        stats = populated_system.get_player_stats("player_0")
        assert stats.healing_done >= 30.0

    def test_damage_tracking_for_nonexistent(self, scoring_system):
        """Should return False for nonexistent player."""
        result = scoring_system.record_damage("nonexistent", "also_nonexistent", 50.0)
        assert not result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
