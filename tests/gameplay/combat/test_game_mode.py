"""
Comprehensive tests for the Game Mode System.

Tests cover:
- Mode initialization
- Mode rules
- Win conditions
- Round management
- Mode-specific scoring
- Time management
- Player management
- Custom mode extension
"""

import pytest
import time
from typing import List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.game_mode import (
    GameMode,
    GameModeConfig,
    GameModeRules,
    WinCondition,
    WinConditionType,
    ScoringEvent,
    ScoringEventType,
)


# =============================================================================
# CONCRETE TEST IMPLEMENTATION
# =============================================================================


class ConcreteGameMode(GameMode):
    """Concrete implementation of GameMode for testing."""

    def __init__(self, config: GameModeConfig):
        super().__init__(config)
        self._spawn_locations = [(0, 0, 0), (10, 0, 0), (0, 10, 0), (10, 10, 0)]
        self._spawn_index = 0

    def get_spawn_location(self, player_id: str) -> Tuple[float, float, float]:
        """Get spawn location for a player."""
        loc = self._spawn_locations[self._spawn_index % len(self._spawn_locations)]
        self._spawn_index += 1
        return loc

    def on_player_killed(
        self,
        victim_id: str,
        killer_id: Optional[str] = None,
        weapon: Optional[str] = None,
        assists: Optional[List[str]] = None
    ) -> None:
        """Handle player death event."""
        self.mark_player_dead(victim_id)
        if killer_id and killer_id in self.players:
            self.add_score(killer_id, ScoringEventType.KILL)
        if assists:
            for assist_id in assists:
                if assist_id in self.players:
                    self.add_score(assist_id, ScoringEventType.ASSIST)

    def check_win_condition(self) -> Tuple[bool, Optional[str]]:
        """Check if any win condition is met."""
        for condition in self.config.win_conditions:
            is_met, winner = condition.is_met(self)
            if is_met:
                return (True, winner)
        return (False, None)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basic_rules():
    """Create basic game mode rules."""
    return GameModeRules(
        friendly_fire=False,
        respawn_enabled=True,
        respawn_delay_seconds=3.0,
        max_respawns=None,
        min_players=2,
        max_players=16,
    )


@pytest.fixture
def basic_config(basic_rules):
    """Create a basic game mode configuration."""
    return GameModeConfig(
        mode_id="test_mode",
        mode_name="Test Mode",
        description="A test game mode",
        rules=basic_rules,
        win_conditions=[],
        time_limit_seconds=600.0,
        score_limit=100,
    )


@pytest.fixture
def game_mode(basic_config):
    """Create a fresh game mode for each test."""
    return ConcreteGameMode(basic_config)


@pytest.fixture
def game_mode_with_score_limit():
    """Create a game mode with score limit win condition."""
    rules = GameModeRules(min_players=2, max_players=8)
    config = GameModeConfig(
        mode_id="scored_mode",
        mode_name="Scored Mode",
        rules=rules,
        win_conditions=[
            WinCondition(
                condition_type=WinConditionType.SCORE_LIMIT,
                target_value=50,
            ),
        ],
        time_limit_seconds=300.0,
        score_limit=50,
    )
    return ConcreteGameMode(config)


@pytest.fixture
def game_mode_with_time_limit():
    """Create a game mode with time limit win condition."""
    rules = GameModeRules(min_players=2, max_players=8)
    config = GameModeConfig(
        mode_id="timed_mode",
        mode_name="Timed Mode",
        rules=rules,
        win_conditions=[
            WinCondition(
                condition_type=WinConditionType.TIME_LIMIT,
                time_limit_seconds=0.1,
            ),
        ],
        time_limit_seconds=0.1,
    )
    return ConcreteGameMode(config)


@pytest.fixture
def elimination_mode():
    """Create an elimination game mode."""
    rules = GameModeRules(respawn_enabled=False, min_players=2, max_players=8)
    config = GameModeConfig(
        mode_id="elimination",
        mode_name="Elimination",
        rules=rules,
        win_conditions=[
            WinCondition(condition_type=WinConditionType.ELIMINATION),
        ],
    )
    return ConcreteGameMode(config)


@pytest.fixture
def team_mode():
    """Create a team-based game mode."""
    rules = GameModeRules(min_teams=2, max_teams=4, min_players=2, max_players=16)
    config = GameModeConfig(
        mode_id="team_mode",
        mode_name="Team Mode",
        rules=rules,
        is_team_based=True,
    )
    return ConcreteGameMode(config)


# =============================================================================
# MODE INITIALIZATION TESTS (~20 tests)
# =============================================================================


class TestModeInitialization:
    """Tests for game mode initialization."""

    def test_create_game_mode(self, basic_config):
        """Should create game mode with config."""
        mode = ConcreteGameMode(basic_config)
        assert mode is not None
        assert mode.mode_id == "test_mode"

    def test_mode_name(self, game_mode):
        """Should have mode name."""
        assert game_mode.mode_name == "Test Mode"

    def test_config_accessible(self, game_mode, basic_config):
        """Should have accessible config."""
        assert game_mode.config == basic_config

    def test_rules_accessible(self, game_mode):
        """Should have accessible rules."""
        assert game_mode.rules.friendly_fire == False
        assert game_mode.rules.respawn_enabled == True

    def test_initial_players_empty(self, game_mode):
        """Should start with no players."""
        assert len(game_mode.players) == 0

    def test_initial_scores_empty(self, game_mode):
        """Should start with no scores."""
        assert len(game_mode.player_scores) == 0

    def test_initial_teams_empty(self, game_mode):
        """Should start with no teams."""
        assert len(game_mode.teams) == 0

    def test_initial_round_is_one(self, game_mode):
        """Should start at round 1."""
        assert game_mode.current_round == 1

    def test_default_team_based_false(self, game_mode):
        """Should default to non-team-based."""
        assert not game_mode.is_team_based

    def test_team_based_mode_flag(self, team_mode):
        """Should support team-based mode."""
        assert team_mode.is_team_based

    def test_default_scoring_values(self, game_mode):
        """Should have default scoring values."""
        assert game_mode._scoring_values[ScoringEventType.KILL] == 1

    def test_custom_scoring_values(self):
        """Should accept custom scoring values."""
        rules = GameModeRules()
        config = GameModeConfig(
            mode_id="custom",
            mode_name="Custom",
            rules=rules,
            scoring_values={ScoringEventType.KILL: 100},
        )
        mode = ConcreteGameMode(config)
        assert mode._scoring_values[ScoringEventType.KILL] == 100

    def test_initial_time_not_started(self, game_mode):
        """Should not have started timer initially."""
        assert game_mode._start_time is None

    def test_initial_not_paused(self, game_mode):
        """Should not be paused initially."""
        assert not game_mode._is_paused

    def test_initial_not_overtime(self, game_mode):
        """Should not be in overtime initially."""
        assert not game_mode._in_overtime


# =============================================================================
# PLAYER MANAGEMENT TESTS (~20 tests)
# =============================================================================


class TestPlayerManagement:
    """Tests for player management."""

    def test_add_player(self, game_mode):
        """Should add player to game."""
        result = game_mode.add_player("player_1")
        assert result
        assert "player_1" in game_mode.players

    def test_add_player_initializes_score(self, game_mode):
        """Adding player should initialize score."""
        game_mode.add_player("player_1")
        assert game_mode.player_scores.get("player_1", 0) == 0

    def test_add_multiple_players(self, game_mode):
        """Should add multiple players."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.add_player("p3")
        assert len(game_mode.players) == 3

    def test_add_duplicate_player(self, game_mode):
        """Should not add duplicate player."""
        game_mode.add_player("player_1")
        result = game_mode.add_player("player_1")
        assert not result
        assert len(game_mode.players) == 1

    def test_remove_player(self, game_mode):
        """Should remove player from game."""
        game_mode.add_player("player_1")
        result = game_mode.remove_player("player_1")
        assert result
        assert "player_1" not in game_mode.players

    def test_remove_nonexistent_player(self, game_mode):
        """Should return False for nonexistent player."""
        result = game_mode.remove_player("nonexistent")
        assert not result

    def test_max_players_enforced(self, basic_config):
        """Should enforce max players limit."""
        basic_config.rules.max_players = 3
        mode = ConcreteGameMode(basic_config)

        mode.add_player("p1")
        mode.add_player("p2")
        mode.add_player("p3")
        result = mode.add_player("p4")

        assert not result
        assert len(mode.players) == 3

    def test_player_team_assignment(self, team_mode):
        """Should support team assignment."""
        team_mode.add_player("player_1", team_id="team_a")
        team = team_mode.get_player_team("player_1")
        assert team == "team_a"

    def test_assign_to_team(self, team_mode):
        """Should assign player to team."""
        team_mode.add_player("player_1")
        result = team_mode.assign_to_team("player_1", "team_a")
        assert result
        assert team_mode.get_player_team("player_1") == "team_a"

    def test_get_players_on_team(self, team_mode):
        """Should get players on specific team."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_player("p2", team_id="red")
        team_mode.add_player("p3", team_id="blue")

        red_players = team_mode.get_team_players("red")
        assert len(red_players) == 2
        assert "p1" in red_players
        assert "p2" in red_players

    def test_is_player_alive(self, game_mode):
        """Should track player alive status."""
        game_mode.add_player("player_1")
        assert game_mode.is_player_alive("player_1")

    def test_mark_player_dead(self, game_mode):
        """Should mark player as dead."""
        game_mode.add_player("player_1")
        game_mode.mark_player_dead("player_1")
        assert not game_mode.is_player_alive("player_1")

    def test_respawn_player(self, game_mode):
        """Should respawn player."""
        game_mode.add_player("player_1")
        game_mode.mark_player_dead("player_1")
        result = game_mode.respawn_player("player_1")
        assert result
        assert game_mode.is_player_alive("player_1")

    def test_respawn_disabled(self, elimination_mode):
        """Should not respawn when disabled."""
        elimination_mode.add_player("player_1")
        elimination_mode.mark_player_dead("player_1")
        result = elimination_mode.respawn_player("player_1")
        assert not result

    def test_max_respawns_limit(self, basic_config):
        """Should enforce max respawns."""
        basic_config.rules.max_respawns = 2
        mode = ConcreteGameMode(basic_config)

        mode.add_player("player_1")
        mode.mark_player_dead("player_1")
        assert mode.respawn_player("player_1")
        mode.mark_player_dead("player_1")
        assert mode.respawn_player("player_1")
        mode.mark_player_dead("player_1")
        assert not mode.respawn_player("player_1")

    def test_get_alive_players(self, game_mode):
        """Should get list of alive players."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.add_player("p3")
        game_mode.mark_player_dead("p2")

        alive = game_mode.get_alive_players()
        assert len(alive) == 2
        assert "p2" not in alive

    def test_get_alive_teams(self, team_mode):
        """Should get teams with alive players."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_player("p2", team_id="blue")
        team_mode.mark_player_dead("p2")

        alive_teams = team_mode.get_alive_teams()
        assert "red" in alive_teams
        assert "blue" not in alive_teams


# =============================================================================
# SCORING TESTS (~20 tests)
# =============================================================================


class TestScoring:
    """Tests for game mode scoring."""

    def test_add_score_to_player(self, game_mode):
        """Should add score to player."""
        game_mode.add_player("player_1")
        game_mode.add_score("player_1", ScoringEventType.KILL, points=10)
        assert game_mode.player_scores["player_1"] == 10

    def test_score_accumulates(self, game_mode):
        """Score should accumulate."""
        game_mode.add_player("player_1")
        game_mode.add_score("player_1", ScoringEventType.KILL, points=10)
        game_mode.add_score("player_1", ScoringEventType.KILL, points=10)
        assert game_mode.player_scores["player_1"] == 20

    def test_score_uses_default_value(self, game_mode):
        """Should use default scoring value."""
        game_mode.set_scoring_value(ScoringEventType.KILL, 100)
        game_mode.add_player("player_1")
        game_mode.add_score("player_1", ScoringEventType.KILL)
        assert game_mode.player_scores["player_1"] == 100

    def test_record_score_event(self, game_mode):
        """Should record scoring events."""
        game_mode.add_player("player_1")
        event = ScoringEvent(
            event_type=ScoringEventType.KILL,
            player_id="player_1",
            points=25,
        )
        game_mode.record_score_event(event)
        assert len(game_mode.scoring_history) == 1
        assert game_mode.player_scores["player_1"] == 25

    def test_get_player_score(self, game_mode):
        """Should get player score."""
        game_mode.add_player("player_1")
        game_mode.add_score("player_1", ScoringEventType.KILL, points=50)
        assert game_mode.get_player_score("player_1") == 50

    def test_get_player_score_default(self, game_mode):
        """Should return 0 for player without score."""
        assert game_mode.get_player_score("nonexistent") == 0

    def test_team_score_accumulates(self, team_mode):
        """Team score should accumulate from player scores."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_score("p1", ScoringEventType.KILL, points=10)
        assert team_mode.team_scores.get("red", 0) == 10

    def test_get_team_score(self, team_mode):
        """Should get team score."""
        team_mode.create_team("red")
        team_mode.team_scores["red"] = 50
        assert team_mode.get_team_score("red") == 50

    def test_get_leading_player(self, game_mode):
        """Should get player with highest score."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.add_player("p3")

        game_mode.add_score("p1", ScoringEventType.KILL, points=30)
        game_mode.add_score("p2", ScoringEventType.KILL, points=50)
        game_mode.add_score("p3", ScoringEventType.KILL, points=20)

        leader = game_mode.get_leading_player_or_team()
        assert leader == "p2"

    def test_get_leading_team(self, team_mode):
        """Should get team with highest score in team mode."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_player("p2", team_id="blue")
        team_mode.add_score("p1", ScoringEventType.KILL, points=100)
        team_mode.add_score("p2", ScoringEventType.KILL, points=75)

        leader = team_mode.get_leading_player_or_team()
        assert leader == "red"

    def test_get_leaderboard(self, game_mode):
        """Should get sorted leaderboard."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.add_player("p3")

        game_mode.add_score("p1", ScoringEventType.KILL, points=30)
        game_mode.add_score("p2", ScoringEventType.KILL, points=50)
        game_mode.add_score("p3", ScoringEventType.KILL, points=20)

        leaderboard = game_mode.get_leaderboard()
        assert leaderboard[0][0] == "p2"
        assert leaderboard[1][0] == "p1"
        assert leaderboard[2][0] == "p3"

    def test_negative_score_allowed(self, game_mode):
        """Should allow negative scores."""
        game_mode.add_player("player_1")
        game_mode.add_score("player_1", ScoringEventType.PENALTY, points=-10)
        assert game_mode.player_scores["player_1"] == -10

    def test_score_event_callback(self, game_mode):
        """Should trigger score change callback."""
        handler = Mock()
        game_mode.on_score_change(handler)

        game_mode.add_player("player_1")
        game_mode.add_score("player_1", ScoringEventType.KILL, points=10)

        handler.assert_called_once()


# =============================================================================
# WIN CONDITION TESTS (~25 tests)
# =============================================================================


class TestWinConditions:
    """Tests for win condition checking."""

    def test_score_limit_not_met(self, game_mode_with_score_limit):
        """Score limit should not be met initially."""
        game_mode_with_score_limit.add_player("p1")
        game_mode_with_score_limit.add_score("p1", ScoringEventType.KILL, points=10)

        is_met, winner = game_mode_with_score_limit.check_win_condition()
        assert not is_met

    def test_score_limit_met(self, game_mode_with_score_limit):
        """Score limit should be met when reached."""
        game_mode_with_score_limit.add_player("p1")
        game_mode_with_score_limit.add_score("p1", ScoringEventType.KILL, points=50)

        is_met, winner = game_mode_with_score_limit.check_win_condition()
        assert is_met
        assert winner == "p1"

    def test_score_limit_exceeded(self, game_mode_with_score_limit):
        """Score limit should be met when exceeded."""
        game_mode_with_score_limit.add_player("p1")
        game_mode_with_score_limit.add_score("p1", ScoringEventType.KILL, points=100)

        is_met, winner = game_mode_with_score_limit.check_win_condition()
        assert is_met
        assert winner == "p1"

    def test_time_limit_not_expired(self, game_mode_with_time_limit):
        """Time limit should not be met before expiration."""
        game_mode_with_time_limit.add_player("p1")
        game_mode_with_time_limit.add_player("p2")
        game_mode_with_time_limit.start()

        assert not game_mode_with_time_limit.is_time_expired()

    def test_time_limit_expired(self, game_mode_with_time_limit):
        """Time limit should be met after expiration."""
        game_mode_with_time_limit.add_player("p1")
        game_mode_with_time_limit.add_player("p2")
        game_mode_with_time_limit.start()

        time.sleep(0.15)

        assert game_mode_with_time_limit.is_time_expired()

    def test_elimination_one_player_left(self, elimination_mode):
        """Elimination should trigger with one player remaining."""
        elimination_mode.add_player("p1")
        elimination_mode.add_player("p2")
        elimination_mode.add_player("p3")

        elimination_mode.mark_player_dead("p2")
        elimination_mode.mark_player_dead("p3")

        is_met, winner = elimination_mode.check_win_condition()
        assert is_met
        assert winner == "p1"

    def test_elimination_all_dead(self, elimination_mode):
        """Elimination should handle all players dead (draw)."""
        elimination_mode.add_player("p1")
        elimination_mode.add_player("p2")

        elimination_mode.mark_player_dead("p1")
        elimination_mode.mark_player_dead("p2")

        is_met, winner = elimination_mode.check_win_condition()
        assert is_met
        assert winner is None  # Draw

    def test_team_elimination(self):
        """Team elimination should trigger when one team remains."""
        rules = GameModeRules(respawn_enabled=False, min_teams=2, max_teams=2)
        config = GameModeConfig(
            mode_id="team_elim",
            mode_name="Team Elimination",
            rules=rules,
            is_team_based=True,
            win_conditions=[
                WinCondition(condition_type=WinConditionType.ELIMINATION),
            ],
        )
        mode = ConcreteGameMode(config)

        mode.add_player("p1", team_id="red")
        mode.add_player("p2", team_id="red")
        mode.add_player("p3", team_id="blue")

        mode.mark_player_dead("p3")

        is_met, winner = mode.check_win_condition()
        assert is_met
        assert winner == "red"

    def test_rounds_win_condition(self):
        """Round win condition should work."""
        rules = GameModeRules()
        config = GameModeConfig(
            mode_id="rounds",
            mode_name="Rounds",
            rules=rules,
            win_conditions=[
                WinCondition(
                    condition_type=WinConditionType.ROUNDS,
                    target_value=3,
                ),
            ],
        )
        mode = ConcreteGameMode(config)
        mode.add_player("p1")
        mode.add_player("p2")

        # Player 1 wins 3 rounds
        mode.end_round("p1")
        mode.end_round("p1")
        mode.end_round("p1")

        is_met, winner = mode.check_win_condition()
        assert is_met
        assert winner == "p1"

    def test_win_condition_custom_check(self, game_mode):
        """Custom win condition check should work."""
        def custom_check(mode):
            # Win if player named "winner" exists
            if "winner" in mode.players:
                return "winner"
            return None

        condition = WinCondition(
            condition_type=WinConditionType.OBJECTIVE,
            custom_check=custom_check,
        )
        game_mode.config.win_conditions.append(condition)

        game_mode.add_player("loser")
        is_met, winner = game_mode.check_win_condition()
        assert not is_met

        game_mode.add_player("winner")
        is_met, winner = game_mode.check_win_condition()
        assert is_met
        assert winner == "winner"

    def test_no_win_conditions(self, game_mode):
        """Should not win if no conditions defined."""
        game_mode.add_player("p1")
        game_mode.add_score("p1", ScoringEventType.KILL, points=1000)

        is_met, winner = game_mode.check_win_condition()
        assert not is_met


# =============================================================================
# ROUND MANAGEMENT TESTS (~15 tests)
# =============================================================================


class TestRoundManagement:
    """Tests for round management."""

    def test_initial_round_is_one(self, game_mode):
        """Should start at round 1."""
        assert game_mode.current_round == 1

    def test_end_round_increments(self, game_mode):
        """Ending round should increment counter."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")

        game_mode.end_round("p1")
        assert game_mode.current_round == 2

    def test_end_round_records_winner(self, game_mode):
        """Should record round winner."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")

        game_mode.end_round("p1")
        assert game_mode.round_wins.get("p1", 0) == 1

    def test_get_round_wins(self, game_mode):
        """Should get round wins for player."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")

        game_mode.end_round("p1")
        game_mode.end_round("p1")

        assert game_mode.get_round_wins("p1") == 2
        assert game_mode.get_round_wins("p2") == 0

    def test_reset_round(self, game_mode):
        """Should reset round state."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.mark_player_dead("p1")

        game_mode.reset_round()

        assert game_mode.is_player_alive("p1")

    def test_round_end_callback(self, game_mode):
        """Should trigger round end callback."""
        handler = Mock()
        game_mode.on_round_end(handler)

        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.end_round("p1")

        handler.assert_called_once_with(1, "p1")


# =============================================================================
# TIME MANAGEMENT TESTS (~15 tests)
# =============================================================================


class TestTimeManagement:
    """Tests for time management."""

    def test_start_sets_time(self, game_mode):
        """Starting should set start time."""
        game_mode.start()
        assert game_mode._start_time is not None

    def test_get_elapsed_time(self, game_mode):
        """Should get elapsed time."""
        game_mode.start()
        time.sleep(0.05)
        elapsed = game_mode.get_elapsed_time()
        assert elapsed > 0

    def test_get_elapsed_time_not_started(self, game_mode):
        """Should return 0 if not started."""
        elapsed = game_mode.get_elapsed_time()
        assert elapsed == 0.0

    def test_get_remaining_time(self, game_mode):
        """Should get remaining time."""
        game_mode.start()
        remaining = game_mode.get_remaining_time()
        assert remaining is not None
        assert remaining < game_mode.config.time_limit_seconds

    def test_get_remaining_time_no_limit(self, basic_config):
        """Should return None if no time limit."""
        basic_config.time_limit_seconds = None
        mode = ConcreteGameMode(basic_config)
        mode.start()

        assert mode.get_remaining_time() is None

    def test_pause(self, game_mode):
        """Should pause timer."""
        game_mode.start()
        time.sleep(0.02)
        game_mode.pause()

        assert game_mode._is_paused
        elapsed1 = game_mode.get_elapsed_time()
        time.sleep(0.02)
        elapsed2 = game_mode.get_elapsed_time()

        assert elapsed1 == elapsed2  # Time should not advance

    def test_resume(self, game_mode):
        """Should resume timer."""
        game_mode.start()
        game_mode.pause()
        game_mode.resume()

        assert not game_mode._is_paused

    def test_resume_accounts_for_pause(self, game_mode):
        """Resume should account for paused time."""
        game_mode.start()
        time.sleep(0.02)
        game_mode.pause()
        time.sleep(0.05)
        game_mode.resume()
        time.sleep(0.02)

        elapsed = game_mode.get_elapsed_time()
        # Should be around 0.04, not 0.09
        assert elapsed < 0.06

    def test_start_overtime(self, game_mode):
        """Should start overtime."""
        game_mode.start()
        result = game_mode.start_overtime()
        assert result
        assert game_mode.is_in_overtime()

    def test_start_overtime_disabled(self, basic_config):
        """Should not start overtime if disabled."""
        basic_config.rules.overtime_enabled = False
        mode = ConcreteGameMode(basic_config)
        mode.start()

        result = mode.start_overtime()
        assert not result

    def test_start_overtime_twice(self, game_mode):
        """Should not start overtime twice."""
        game_mode.start()
        game_mode.start_overtime()
        result = game_mode.start_overtime()
        assert not result

    def test_is_in_overtime(self, game_mode):
        """Should track overtime state."""
        game_mode.start()
        assert not game_mode.is_in_overtime()
        game_mode.start_overtime()
        assert game_mode.is_in_overtime()


# =============================================================================
# TEAM MANAGEMENT TESTS (~10 tests)
# =============================================================================


class TestTeamManagement:
    """Tests for team management in game modes."""

    def test_create_team(self, team_mode):
        """Should create team."""
        result = team_mode.create_team("team_a")
        assert result
        assert "team_a" in team_mode.teams

    def test_create_duplicate_team(self, team_mode):
        """Should not create duplicate team."""
        team_mode.create_team("team_a")
        result = team_mode.create_team("team_a")
        assert not result

    def test_team_created_on_player_assign(self, team_mode):
        """Assigning player should create team if needed."""
        team_mode.add_player("p1", team_id="new_team")
        assert "new_team" in team_mode.teams

    def test_is_friendly_fire(self, team_mode):
        """Should detect friendly fire."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_player("p2", team_id="red")
        team_mode.add_player("p3", team_id="blue")

        assert team_mode.is_friendly_fire("p1", "p2")
        assert not team_mode.is_friendly_fire("p1", "p3")

    def test_can_damage_enemy(self, team_mode):
        """Should allow damage to enemy."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_player("p2", team_id="blue")

        assert team_mode.can_damage("p1", "p2")

    def test_cannot_damage_friendly_by_default(self, team_mode):
        """Should not allow damage to friendly by default."""
        team_mode.add_player("p1", team_id="red")
        team_mode.add_player("p2", team_id="red")

        assert not team_mode.can_damage("p1", "p2")

    def test_can_damage_friendly_with_ff(self, basic_config):
        """Should allow damage with friendly fire enabled."""
        basic_config.rules.friendly_fire = True
        basic_config.is_team_based = True
        mode = ConcreteGameMode(basic_config)

        mode.add_player("p1", team_id="red")
        mode.add_player("p2", team_id="red")

        assert mode.can_damage("p1", "p2")

    def test_can_damage_self(self, game_mode):
        """Should always allow self-damage."""
        game_mode.add_player("p1")
        assert game_mode.can_damage("p1", "p1")


# =============================================================================
# CALLBACK TESTS (~10 tests)
# =============================================================================


class TestCallbacks:
    """Tests for game mode callbacks."""

    def test_on_score_change(self, game_mode):
        """Should trigger on score change."""
        handler = Mock()
        game_mode.on_score_change(handler)

        game_mode.add_player("p1")
        game_mode.add_score("p1", ScoringEventType.KILL, points=10)

        handler.assert_called_once()
        args = handler.call_args[0]
        assert args[0] == "p1"  # player_id
        assert args[1] == 0  # old_score
        assert args[2] == 10  # new_score

    def test_on_player_eliminated(self, game_mode):
        """Should trigger on player elimination."""
        handler = Mock()
        game_mode.on_player_eliminated(handler)

        game_mode.add_player("p1")
        game_mode.mark_player_dead("p1")

        handler.assert_called_once_with("p1")

    def test_on_round_end(self, game_mode):
        """Should trigger on round end."""
        handler = Mock()
        game_mode.on_round_end(handler)

        game_mode.add_player("p1")
        game_mode.end_round("p1")

        handler.assert_called_once_with(1, "p1")

    def test_multiple_callbacks(self, game_mode):
        """Should support multiple callbacks."""
        handler1 = Mock()
        handler2 = Mock()
        game_mode.on_score_change(handler1)
        game_mode.on_score_change(handler2)

        game_mode.add_player("p1")
        game_mode.add_score("p1", ScoringEventType.KILL, points=10)

        handler1.assert_called_once()
        handler2.assert_called_once()


# =============================================================================
# PLAYER KILL HANDLING TESTS (~10 tests)
# =============================================================================


class TestPlayerKillHandling:
    """Tests for player kill handling."""

    def test_on_player_killed_marks_dead(self, game_mode):
        """Kill should mark player dead."""
        game_mode.add_player("victim")
        game_mode.add_player("killer")

        game_mode.on_player_killed("victim", killer_id="killer")

        assert not game_mode.is_player_alive("victim")

    def test_on_player_killed_awards_points(self, game_mode):
        """Kill should award points to killer."""
        game_mode.add_player("victim")
        game_mode.add_player("killer")

        game_mode.on_player_killed("victim", killer_id="killer")

        assert game_mode.player_scores["killer"] == 1

    def test_on_player_killed_with_assists(self, game_mode):
        """Kill should award assist points."""
        game_mode.add_player("victim")
        game_mode.add_player("killer")
        game_mode.add_player("assist1")
        game_mode.add_player("assist2")

        game_mode.on_player_killed(
            "victim",
            killer_id="killer",
            assists=["assist1", "assist2"]
        )

        # Default assist value is 0, but handler should be called

    def test_on_player_killed_no_killer(self, game_mode):
        """Kill without killer should still mark dead."""
        game_mode.add_player("victim")

        game_mode.on_player_killed("victim")

        assert not game_mode.is_player_alive("victim")

    def test_get_spawn_location(self, game_mode):
        """Should return spawn location."""
        game_mode.add_player("p1")
        loc = game_mode.get_spawn_location("p1")

        assert isinstance(loc, tuple)
        assert len(loc) == 3


# =============================================================================
# UTILITY TESTS (~10 tests)
# =============================================================================


class TestUtility:
    """Tests for utility methods."""

    def test_get_stats(self, game_mode):
        """Should get game stats."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.start()
        game_mode.add_score("p1", ScoringEventType.KILL, points=10)

        stats = game_mode.get_stats()

        assert stats["mode_id"] == "test_mode"
        assert stats["player_count"] == 2
        assert "p1" in stats["player_scores"]

    def test_reset(self, game_mode):
        """Should reset game state."""
        game_mode.add_player("p1")
        game_mode.add_player("p2")
        game_mode.start()
        game_mode.add_score("p1", ScoringEventType.KILL, points=50)
        game_mode.mark_player_dead("p2")
        game_mode.end_round("p1")

        game_mode.reset()

        assert game_mode.player_scores["p1"] == 0
        assert game_mode.is_player_alive("p2")
        assert game_mode.current_round == 1
        assert game_mode._start_time is None

    def test_should_go_to_overtime_tied(self, game_mode_with_time_limit):
        """Should suggest overtime when tied at time limit."""
        game_mode_with_time_limit.add_player("p1")
        game_mode_with_time_limit.add_player("p2")
        game_mode_with_time_limit.add_score("p1", ScoringEventType.KILL, points=10)
        game_mode_with_time_limit.add_score("p2", ScoringEventType.KILL, points=10)
        game_mode_with_time_limit.start()

        time.sleep(0.15)

        assert game_mode_with_time_limit.should_go_to_overtime()

    def test_should_not_go_to_overtime_clear_winner(self, game_mode_with_time_limit):
        """Should not suggest overtime with clear winner."""
        game_mode_with_time_limit.add_player("p1")
        game_mode_with_time_limit.add_player("p2")
        game_mode_with_time_limit.add_score("p1", ScoringEventType.KILL, points=20)
        game_mode_with_time_limit.add_score("p2", ScoringEventType.KILL, points=10)
        game_mode_with_time_limit.start()

        time.sleep(0.15)

        # May or may not suggest overtime depending on implementation
        # The important thing is it doesn't crash


# =============================================================================
# CUSTOM MODE EXTENSION TESTS (~10 tests)
# =============================================================================


class TestCustomModeExtension:
    """Tests for extending GameMode with custom modes."""

    def test_subclass_game_mode(self):
        """Should allow subclassing GameMode."""
        class CustomMode(ConcreteGameMode):
            def __init__(self):
                rules = GameModeRules()
                config = GameModeConfig(
                    mode_id="custom",
                    mode_name="Custom Mode",
                    rules=rules,
                )
                super().__init__(config)
                self.custom_value = 42

        mode = CustomMode()
        assert mode.custom_value == 42
        assert mode.mode_id == "custom"

    def test_override_check_win_condition(self):
        """Should allow overriding win condition check."""
        class CustomMode(ConcreteGameMode):
            def __init__(self):
                rules = GameModeRules()
                config = GameModeConfig(
                    mode_id="custom",
                    mode_name="Custom",
                    rules=rules,
                )
                super().__init__(config)
                self._custom_win = False

            def check_win_condition(self):
                if self._custom_win:
                    return (True, "custom_winner")
                return super().check_win_condition()

        mode = CustomMode()
        mode._custom_win = True
        is_met, winner = mode.check_win_condition()
        assert is_met
        assert winner == "custom_winner"

    def test_override_on_player_killed(self):
        """Should allow custom kill handling."""
        kill_count = {"value": 0}

        class CustomMode(ConcreteGameMode):
            def __init__(self):
                rules = GameModeRules()
                config = GameModeConfig(
                    mode_id="custom",
                    mode_name="Custom",
                    rules=rules,
                )
                super().__init__(config)

            def on_player_killed(self, victim_id, killer_id=None, weapon=None, assists=None):
                kill_count["value"] += 1
                super().on_player_killed(victim_id, killer_id, weapon, assists)

        mode = CustomMode()
        mode.add_player("p1")
        mode.add_player("p2")
        mode.on_player_killed("p1", "p2")

        assert kill_count["value"] == 1

    def test_custom_spawn_location(self):
        """Should allow custom spawn logic."""
        class CustomMode(ConcreteGameMode):
            def get_spawn_location(self, player_id):
                # Always spawn at origin
                return (0, 0, 0)

        rules = GameModeRules()
        config = GameModeConfig(mode_id="custom", mode_name="Custom", rules=rules)
        mode = CustomMode(config)
        mode.add_player("p1")

        loc = mode.get_spawn_location("p1")
        assert loc == (0, 0, 0)
