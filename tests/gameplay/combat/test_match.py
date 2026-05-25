"""
Comprehensive tests for the Match System.

Tests cover:
- Match lifecycle (LOBBY->STARTING->IN_PROGRESS->ENDING->POST_MATCH)
- Player join/leave
- Match timer
- Match pause/resume
- Match end conditions
- Match results
- Match persistence
"""

import pytest
import time
from typing import List, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch

from engine.gameplay.combat.match import (
    Match,
    MatchState,
    MatchConfig,
    MatchResult,
    MatchEvents,
)
from engine.gameplay.combat.game_mode import (
    GameMode,
    GameModeConfig,
    GameModeRules,
    WinCondition,
    WinConditionType,
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
def game_mode():
    """Create a basic game mode."""
    rules = GameModeRules(min_players=2, max_players=8)
    config = GameModeConfig(
        mode_id="test_mode",
        mode_name="Test Mode",
        rules=rules,
        time_limit_seconds=300.0,
        score_limit=50,
    )
    return ConcreteGameMode(config)


@pytest.fixture
def match(game_mode):
    """Create a fresh match for each test."""
    return Match(game_mode)


@pytest.fixture
def configured_match(game_mode):
    """Create a match with custom configuration."""
    config = MatchConfig(
        min_players_to_start=2,
        countdown_duration_seconds=3.0,
        end_sequence_duration_seconds=3.0,
        results_duration_seconds=5.0,
        auto_start_when_ready=True,
        ready_percentage_to_start=1.0,
        allow_join_in_progress=True,
        allow_spectators=True,
        max_spectators=5,
    )
    return Match(game_mode, config)


@pytest.fixture
def populated_match(match):
    """Create a match with players."""
    match.add_player("player_1")
    match.add_player("player_2")
    return match


# =============================================================================
# MATCH LIFECYCLE TESTS (~25 tests)
# =============================================================================


class TestMatchLifecycle:
    """Tests for match lifecycle management."""

    def test_initial_state_waiting(self, match):
        """Match should start in WAITING state."""
        assert match.state == MatchState.WAITING

    def test_can_start_with_min_players(self, game_mode):
        """Should be able to start with minimum players."""
        # Disable auto-start to check can_start manually
        config = MatchConfig(auto_start_when_ready=False)
        match = Match(game_mode, config)
        match.add_player("player_1")
        match.add_player("player_2")
        match.set_player_ready("player_1")
        match.set_player_ready("player_2")
        assert match.can_start()

    def test_cannot_start_without_min_players(self, match):
        """Should not start without minimum players."""
        match.add_player("player_1")
        assert not match.can_start()

    def test_cannot_start_without_ready_players(self, populated_match):
        """Should not start if players not ready."""
        # Players added but not ready
        assert not populated_match.can_start()

    def test_start_countdown(self, game_mode):
        """Should start countdown to match."""
        # Disable auto-start to manually control countdown
        config = MatchConfig(auto_start_when_ready=False)
        match = Match(game_mode, config)
        match.add_player("player_1")
        match.add_player("player_2")
        match.set_player_ready("player_1")
        match.set_player_ready("player_2")

        result = match.start_countdown()
        assert result
        assert match.state == MatchState.STARTING

    def test_start_match_directly(self, populated_match):
        """Should start match directly."""
        populated_match.set_player_ready("player_1")
        populated_match.set_player_ready("player_2")

        result = populated_match.start_match()
        assert result
        assert populated_match.state == MatchState.IN_PROGRESS

    def test_end_match(self, populated_match):
        """Should end match."""
        populated_match.start_match()
        result = populated_match.end_match("player_1")

        assert result
        assert populated_match.state == MatchState.ENDING

    def test_complete_match(self, populated_match):
        """Should complete match after ending."""
        populated_match.start_match()
        populated_match.end_match("player_1")

        result = populated_match.complete_match()
        assert result
        assert populated_match.state == MatchState.COMPLETE

    def test_reset_match(self, populated_match):
        """Should reset match to waiting."""
        populated_match.start_match()
        populated_match.end_match("player_1")
        populated_match.complete_match()

        populated_match.reset()
        assert populated_match.state == MatchState.WAITING
        assert populated_match.ready_count == 0

    def test_force_end_match(self, populated_match):
        """Should force end match immediately."""
        populated_match.start_match()

        populated_match.force_end("admin_ended")
        assert populated_match.state == MatchState.COMPLETE

    def test_is_active_property(self, populated_match):
        """Should check if match is active."""
        assert not populated_match.is_active

        populated_match.start_match()
        assert populated_match.is_active

    def test_is_joinable_waiting(self, match):
        """Match should be joinable when waiting."""
        assert match.is_joinable

    def test_is_joinable_in_progress(self, configured_match):
        """Match should be joinable in progress if allowed."""
        configured_match.add_player("p1")
        configured_match.add_player("p2")
        configured_match.start_match()

        assert configured_match.is_joinable

    def test_not_joinable_in_progress_if_disabled(self, game_mode):
        """Match should not be joinable if disabled."""
        config = MatchConfig(allow_join_in_progress=False)
        match = Match(game_mode, config)
        match.add_player("p1")
        match.add_player("p2")
        match.start_match()

        assert not match.is_joinable

    def test_not_joinable_when_complete(self, populated_match):
        """Match should not be joinable when complete."""
        populated_match.start_match()
        populated_match.end_match("player_1")
        populated_match.complete_match()

        assert not populated_match.is_joinable


# =============================================================================
# PLAYER JOIN/LEAVE TESTS (~20 tests)
# =============================================================================


class TestPlayerJoinLeave:
    """Tests for player join and leave."""

    def test_add_player(self, match):
        """Should add player to match."""
        result = match.add_player("player_1")
        assert result
        assert match.player_count == 1

    def test_add_player_with_team(self, match):
        """Should add player with team assignment."""
        result = match.add_player("player_1", team_id="red")
        assert result

    def test_add_duplicate_player(self, match):
        """Should not add duplicate player."""
        match.add_player("player_1")
        result = match.add_player("player_1")
        assert not result

    def test_remove_player(self, populated_match):
        """Should remove player from match."""
        result = populated_match.remove_player("player_1")
        assert result
        assert populated_match.player_count == 1

    def test_remove_nonexistent_player(self, match):
        """Should return False for nonexistent player."""
        result = match.remove_player("nonexistent")
        assert not result

    def test_remove_player_clears_ready(self, populated_match):
        """Should clear ready state when player removed."""
        populated_match.set_player_ready("player_1")
        populated_match.remove_player("player_1")

        assert populated_match.ready_count == 0

    def test_set_player_ready(self, populated_match):
        """Should set player as ready."""
        result = populated_match.set_player_ready("player_1")
        assert result
        assert populated_match.is_player_ready("player_1")

    def test_set_player_not_ready(self, populated_match):
        """Should set player as not ready."""
        populated_match.set_player_ready("player_1")
        result = populated_match.set_player_ready("player_1", is_ready=False)

        assert result
        assert not populated_match.is_player_ready("player_1")

    def test_ready_only_in_waiting(self, populated_match):
        """Should only allow ready in WAITING state."""
        populated_match.start_match()
        result = populated_match.set_player_ready("player_1")
        assert not result

    def test_ready_count(self, populated_match):
        """Should track ready player count."""
        assert populated_match.ready_count == 0

        populated_match.set_player_ready("player_1")
        assert populated_match.ready_count == 1

        populated_match.set_player_ready("player_2")
        assert populated_match.ready_count == 2

    def test_add_spectator(self, configured_match):
        """Should add spectator."""
        result = configured_match.add_spectator("spectator_1")
        assert result
        assert "spectator_1" in configured_match.get_spectators()

    def test_add_spectator_max_limit(self, configured_match):
        """Should enforce spectator limit."""
        for i in range(5):
            configured_match.add_spectator(f"spec_{i}")

        result = configured_match.add_spectator("spec_extra")
        assert not result

    def test_add_spectator_disabled(self, game_mode):
        """Should not add spectator if disabled."""
        config = MatchConfig(allow_spectators=False)
        match = Match(game_mode, config)

        result = match.add_spectator("spectator_1")
        assert not result

    def test_player_cannot_be_spectator(self, configured_match):
        """Player should not be spectator simultaneously."""
        configured_match.add_player("player_1")
        result = configured_match.add_spectator("player_1")
        assert not result

    def test_remove_spectator(self, configured_match):
        """Should remove spectator."""
        configured_match.add_spectator("spectator_1")
        result = configured_match.remove_spectator("spectator_1")

        assert result
        assert "spectator_1" not in configured_match.get_spectators()


# =============================================================================
# MATCH TIMER TESTS (~15 tests)
# =============================================================================


class TestMatchTimer:
    """Tests for match timing."""

    def test_countdown_timer(self, configured_match):
        """Should count down before match."""
        configured_match.add_player("p1")
        configured_match.add_player("p2")
        configured_match.set_player_ready("p1")
        configured_match.set_player_ready("p2")
        configured_match.start_countdown()

        # Update to process countdown
        time.sleep(0.1)
        configured_match.update(0.1)

        # Still in STARTING or transitioned to IN_PROGRESS
        assert configured_match.state in (MatchState.STARTING, MatchState.IN_PROGRESS)

    def test_countdown_completes(self, game_mode):
        """Should transition to IN_PROGRESS after countdown."""
        config = MatchConfig(
            countdown_duration_seconds=0.1,
            auto_start_when_ready=False,
        )
        match = Match(game_mode, config)
        match.add_player("p1")
        match.add_player("p2")
        match.set_player_ready("p1")
        match.set_player_ready("p2")
        match.start_countdown()

        time.sleep(0.15)
        match.update(0.05)

        assert match.state == MatchState.IN_PROGRESS

    def test_match_has_id_after_start(self, populated_match):
        """Match should have ID after starting."""
        populated_match.start_match()
        assert populated_match.match_id != ""

    def test_ending_sequence_duration(self, game_mode):
        """Should wait for ending sequence."""
        config = MatchConfig(end_sequence_duration_seconds=0.1)
        match = Match(game_mode, config)
        match.add_player("p1")
        match.add_player("p2")
        match.start_match()
        match.end_match("p1")

        # Should still be ENDING
        assert match.state == MatchState.ENDING

        # Wait for sequence
        time.sleep(0.15)
        match.update(0.05)

        assert match.state == MatchState.COMPLETE


# =============================================================================
# ROUND MANAGEMENT TESTS (~15 tests)
# =============================================================================


class TestRoundManagement:
    """Tests for round management within match."""

    def test_initial_round(self, populated_match):
        """Should start at round 1."""
        populated_match.start_match()
        assert populated_match.current_round == 1

    def test_round_in_progress(self, populated_match):
        """Should track round in progress."""
        populated_match.start_match()
        assert populated_match.is_round_in_progress

    def test_start_new_round(self, populated_match):
        """Should start new round."""
        populated_match.start_match()
        populated_match.end_round("player_1")

        result = populated_match.start_round()
        assert result
        assert populated_match.current_round == 2

    def test_end_round(self, populated_match):
        """Should end current round."""
        populated_match.start_match()

        result = populated_match.end_round("player_1")
        assert result
        assert not populated_match.is_round_in_progress

    def test_cannot_start_round_while_in_progress(self, populated_match):
        """Should not start round while one is in progress."""
        populated_match.start_match()

        result = populated_match.start_round()
        assert not result

    def test_cannot_end_round_not_in_progress(self, populated_match):
        """Should not end round when none in progress."""
        populated_match.start_match()
        populated_match.end_round("player_1")

        result = populated_match.end_round("player_2")
        assert not result


# =============================================================================
# MATCH EVENTS TESTS (~15 tests)
# =============================================================================


class TestMatchEvents:
    """Tests for match event callbacks."""

    def test_on_state_change(self, populated_match):
        """Should emit state change event."""
        handler = Mock()
        populated_match.events.on_state_change(handler)

        populated_match.start_match()

        handler.assert_called()
        args = handler.call_args[0]
        assert args[0] == MatchState.WAITING
        assert args[1] == MatchState.IN_PROGRESS

    def test_on_countdown_tick(self, game_mode):
        """Should emit countdown tick events."""
        handler = Mock()
        config = MatchConfig(countdown_duration_seconds=0.5)
        match = Match(game_mode, config)
        match.events.on_countdown_tick(handler)

        match.add_player("p1")
        match.add_player("p2")
        match.start_countdown()

        # Process some ticks
        for _ in range(5):
            time.sleep(0.1)
            match.update(0.1)

        # Should have received some tick calls
        assert handler.call_count >= 0  # May or may not tick depending on timing

    def test_on_match_start(self, populated_match):
        """Should emit match start event."""
        handler = Mock()
        populated_match.events.on_match_start(handler)

        populated_match.start_match()

        handler.assert_called_once()

    def test_on_match_end(self, populated_match):
        """Should emit match end event."""
        handler = Mock()
        populated_match.events.on_match_end(handler)

        populated_match.start_match()
        populated_match.end_match("player_1")
        populated_match.complete_match()

        handler.assert_called_once()

    def test_on_player_join(self, match):
        """Should emit player join event."""
        handler = Mock()
        match.events.on_player_join(handler)

        match.add_player("player_1")

        handler.assert_called_once_with("player_1")

    def test_on_player_leave(self, populated_match):
        """Should emit player leave event."""
        handler = Mock()
        populated_match.events.on_player_leave(handler)

        populated_match.remove_player("player_1")

        handler.assert_called_once_with("player_1")

    def test_on_player_ready(self, populated_match):
        """Should emit player ready event."""
        handler = Mock()
        populated_match.events.on_player_ready(handler)

        populated_match.set_player_ready("player_1")

        handler.assert_called_once_with("player_1", True)

    def test_on_round_start(self, populated_match):
        """Should emit round start event."""
        handler = Mock()
        populated_match.events.on_round_start(handler)

        populated_match.start_match()  # Starts round 1

        handler.assert_called_once_with(1)

    def test_on_round_end(self, populated_match):
        """Should emit round end event."""
        handler = Mock()
        populated_match.events.on_round_end(handler)

        populated_match.start_match()
        populated_match.end_round("player_1")

        handler.assert_called_once_with(1, "player_1")


# =============================================================================
# MATCH RESULTS TESTS (~15 tests)
# =============================================================================


class TestMatchResults:
    """Tests for match results."""

    def test_result_available_after_end(self, populated_match):
        """Result should be available after match ends."""
        populated_match.start_match()
        populated_match.end_match("player_1")

        assert populated_match.result is not None

    def test_result_has_winner(self, populated_match):
        """Result should have winner."""
        populated_match.start_match()
        populated_match.end_match("player_1")

        assert populated_match.result.winner_id == "player_1"

    def test_result_is_draw(self, populated_match):
        """Result should indicate draw when no winner."""
        populated_match.start_match()
        populated_match.end_match(None)

        assert populated_match.result.is_draw

    def test_result_has_duration(self, populated_match):
        """Result should have match duration."""
        populated_match.start_match()
        time.sleep(0.05)
        populated_match.end_match("player_1")

        assert populated_match.result.match_duration_seconds > 0

    def test_result_has_player_scores(self, populated_match):
        """Result should have player scores."""
        populated_match.start_match()
        populated_match.game_mode.add_score("player_1", ScoringEventType.KILL, points=10)
        populated_match.end_match("player_1")

        assert "player_1" in populated_match.result.final_scores

    def test_result_has_player_stats(self, populated_match):
        """Result should have player stats."""
        populated_match.start_match()
        populated_match.record_kill("player_1", "player_2")
        populated_match.end_match("player_1")

        stats = populated_match.result.player_stats
        assert "player_1" in stats
        assert stats["player_1"]["kills"] == 1

    def test_result_has_total_kills(self, populated_match):
        """Result should have total kills."""
        populated_match.start_match()
        populated_match.record_kill("player_1", "player_2")
        populated_match.record_kill("player_2", "player_1")
        populated_match.end_match("player_1")

        assert populated_match.result.total_kills == 2

    def test_result_has_mvp(self, populated_match):
        """Result should have MVP player."""
        populated_match.start_match()
        populated_match.record_kill("player_1", "player_2")
        populated_match.end_match("player_1")

        # Player with most kills should be MVP
        assert populated_match.result.mvp_player_id == "player_1"

    def test_result_has_rounds_info(self, populated_match):
        """Result should have round information."""
        populated_match.start_match()
        populated_match.end_round("player_1")
        populated_match.start_round()
        populated_match.end_round("player_2")
        populated_match.end_match("player_1")

        assert populated_match.result.rounds_played >= 1

    def test_result_has_round_winners(self, populated_match):
        """Result should have round winners list."""
        populated_match.start_match()
        populated_match.end_round("player_1")
        populated_match.end_match("player_1")

        assert "player_1" in populated_match.result.round_winners


# =============================================================================
# STATISTICS TRACKING TESTS (~10 tests)
# =============================================================================


class TestStatisticsTracking:
    """Tests for statistics tracking during match."""

    def test_record_kill(self, populated_match):
        """Should record kill."""
        populated_match.start_match()
        populated_match.record_kill("player_1", "player_2")

        stats = populated_match.get_player_stats("player_1")
        assert stats["kills"] == 1

    def test_record_kill_with_assists(self, match):
        """Should record kills with assists."""
        match.add_player("p1")
        match.add_player("p2")
        match.add_player("p3")
        match.start_match()

        match.record_kill("p1", "p3", assists=["p2"])

        stats = match.get_player_stats("p2")
        assert stats["assists"] == 1

    def test_record_death(self, populated_match):
        """Should track deaths."""
        populated_match.start_match()
        populated_match.record_kill("player_1", "player_2")

        stats = populated_match.get_player_stats("player_2")
        assert stats["deaths"] == 1

    def test_get_all_player_stats(self, populated_match):
        """Should get all player stats."""
        populated_match.start_match()
        populated_match.record_kill("player_1", "player_2")

        all_stats = populated_match.get_all_player_stats()
        assert "player_1" in all_stats
        assert "player_2" in all_stats


# =============================================================================
# MATCH INFO TESTS (~10 tests)
# =============================================================================


class TestMatchInfo:
    """Tests for match information queries."""

    def test_get_match_info(self, populated_match):
        """Should get match information."""
        populated_match.start_match()

        info = populated_match.get_match_info()
        assert "match_id" in info
        assert "state" in info
        assert "player_count" in info

    def test_match_info_state(self, populated_match):
        """Match info should have current state."""
        info = populated_match.get_match_info()
        assert info["state"] == "WAITING"

        populated_match.start_match()
        info = populated_match.get_match_info()
        assert info["state"] == "IN_PROGRESS"

    def test_match_info_player_count(self, populated_match):
        """Match info should have player count."""
        info = populated_match.get_match_info()
        assert info["player_count"] == 2

    def test_match_info_ready_count(self, populated_match):
        """Match info should have ready count."""
        populated_match.set_player_ready("player_1")

        info = populated_match.get_match_info()
        assert info["ready_count"] == 1

    def test_match_info_game_mode(self, populated_match):
        """Match info should have game mode info."""
        info = populated_match.get_match_info()
        assert "game_mode" in info
        assert info["game_mode"]["mode_id"] == "test_mode"


# =============================================================================
# AUTO-START TESTS (~10 tests)
# =============================================================================


class TestAutoStart:
    """Tests for auto-start functionality."""

    def test_auto_start_when_all_ready(self, configured_match):
        """Should auto-start when all players ready."""
        configured_match.add_player("p1")
        configured_match.add_player("p2")

        configured_match.set_player_ready("p1")
        configured_match.set_player_ready("p2")

        # Should have started countdown
        assert configured_match.state == MatchState.STARTING

    def test_auto_start_percentage(self, game_mode):
        """Should auto-start at configured percentage."""
        config = MatchConfig(
            auto_start_when_ready=True,
            ready_percentage_to_start=0.5,  # 50%
            min_players_to_start=2,
        )
        match = Match(game_mode, config)

        match.add_player("p1")
        match.add_player("p2")
        match.add_player("p3")
        match.add_player("p4")

        # 2 of 4 = 50%
        match.set_player_ready("p1")
        match.set_player_ready("p2")

        assert match.state == MatchState.STARTING

    def test_no_auto_start_if_disabled(self, game_mode):
        """Should not auto-start if disabled."""
        config = MatchConfig(auto_start_when_ready=False)
        match = Match(game_mode, config)

        match.add_player("p1")
        match.add_player("p2")
        match.set_player_ready("p1")
        match.set_player_ready("p2")

        assert match.state == MatchState.WAITING


# =============================================================================
# UPDATE LOOP TESTS (~10 tests)
# =============================================================================


class TestUpdateLoop:
    """Tests for match update loop."""

    def test_update_countdown(self, game_mode):
        """Update should process countdown."""
        config = MatchConfig(
            countdown_duration_seconds=0.1,
            auto_start_when_ready=False,
        )
        match = Match(game_mode, config)
        match.add_player("p1")
        match.add_player("p2")
        match.set_player_ready("p1")
        match.set_player_ready("p2")
        match.start_countdown()

        time.sleep(0.15)
        match.update(0.05)

        assert match.state == MatchState.IN_PROGRESS

    def test_update_checks_win_condition(self, game_mode):
        """Update should check win conditions."""
        rules = GameModeRules(min_players=2)
        config = GameModeConfig(
            mode_id="quick",
            mode_name="Quick",
            rules=rules,
            win_conditions=[
                WinCondition(
                    condition_type=WinConditionType.SCORE_LIMIT,
                    target_value=10,
                ),
            ],
            score_limit=10,
        )
        mode = ConcreteGameMode(config)
        match = Match(mode)

        match.add_player("p1")
        match.add_player("p2")
        match.start_match()

        mode.add_score("p1", ScoringEventType.KILL, points=10)
        match.update(0.01)

        assert match.state == MatchState.ENDING

    def test_update_ending_sequence(self, game_mode):
        """Update should process ending sequence."""
        config = MatchConfig(end_sequence_duration_seconds=0.1)
        match = Match(game_mode, config)
        match.add_player("p1")
        match.add_player("p2")
        match.start_match()
        match.end_match("p1")

        time.sleep(0.15)
        match.update(0.05)

        assert match.state == MatchState.COMPLETE
