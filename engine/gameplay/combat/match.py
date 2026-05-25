"""
Match lifecycle management for game modes.

Handles the complete match lifecycle:
- Waiting: Lobby state, waiting for players
- Starting: Countdown before match begins
- InProgress: Active gameplay
- Ending: Match ending sequence
- Complete: Match finished, showing results

Also manages:
- Player ready states
- Match events and callbacks
- Match results and statistics
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set
import time

from engine.gameplay.combat.game_mode import GameMode
from engine.gameplay.combat.constants import DEFAULT_MAX_SPECTATORS


class MatchState(Enum):
    """Match lifecycle states."""
    WAITING = auto()      # Lobby, waiting for players
    STARTING = auto()     # Countdown before match
    IN_PROGRESS = auto()  # Active gameplay
    ENDING = auto()       # Match ending sequence
    COMPLETE = auto()     # Match finished


@dataclass
class MatchConfig:
    """Configuration for match behavior."""
    min_players_to_start: int = 2
    countdown_duration_seconds: float = 5.0
    end_sequence_duration_seconds: float = 5.0
    results_duration_seconds: float = 10.0
    auto_start_when_ready: bool = True
    ready_percentage_to_start: float = 1.0  # 100% of players must be ready
    allow_join_in_progress: bool = True
    allow_spectators: bool = True
    max_spectators: int = DEFAULT_MAX_SPECTATORS
    warmup_enabled: bool = False
    warmup_duration_seconds: float = 30.0


@dataclass
class MatchResult:
    """Result of a completed match."""
    winner_id: Optional[str] = None  # Player or team ID
    is_draw: bool = False
    final_scores: Dict[str, int] = field(default_factory=dict)
    team_scores: Dict[str, int] = field(default_factory=dict)
    mvp_player_id: Optional[str] = None
    match_duration_seconds: float = 0.0
    total_kills: int = 0
    player_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rounds_played: int = 1
    round_winners: List[Optional[str]] = field(default_factory=list)


class MatchEvents:
    """Events fired during match lifecycle."""

    def __init__(self):
        self._on_state_change: List[Callable[[MatchState, MatchState], None]] = []
        self._on_countdown_tick: List[Callable[[int], None]] = []
        self._on_match_start: List[Callable[[], None]] = []
        self._on_match_end: List[Callable[[MatchResult], None]] = []
        self._on_player_join: List[Callable[[str], None]] = []
        self._on_player_leave: List[Callable[[str], None]] = []
        self._on_player_ready: List[Callable[[str, bool], None]] = []
        self._on_round_start: List[Callable[[int], None]] = []
        self._on_round_end: List[Callable[[int, Optional[str]], None]] = []

    def on_state_change(self, callback: Callable[[MatchState, MatchState], None]) -> None:
        """Register callback for state changes (old_state, new_state)."""
        self._on_state_change.append(callback)

    def on_countdown_tick(self, callback: Callable[[int], None]) -> None:
        """Register callback for countdown ticks (seconds_remaining)."""
        self._on_countdown_tick.append(callback)

    def on_match_start(self, callback: Callable[[], None]) -> None:
        """Register callback for match start."""
        self._on_match_start.append(callback)

    def on_match_end(self, callback: Callable[[MatchResult], None]) -> None:
        """Register callback for match end."""
        self._on_match_end.append(callback)

    def on_player_join(self, callback: Callable[[str], None]) -> None:
        """Register callback for player join."""
        self._on_player_join.append(callback)

    def on_player_leave(self, callback: Callable[[str], None]) -> None:
        """Register callback for player leave."""
        self._on_player_leave.append(callback)

    def on_player_ready(self, callback: Callable[[str, bool], None]) -> None:
        """Register callback for player ready state change."""
        self._on_player_ready.append(callback)

    def on_round_start(self, callback: Callable[[int], None]) -> None:
        """Register callback for round start."""
        self._on_round_start.append(callback)

    def on_round_end(self, callback: Callable[[int, Optional[str]], None]) -> None:
        """Register callback for round end."""
        self._on_round_end.append(callback)

    def _emit_state_change(self, old_state: MatchState, new_state: MatchState) -> None:
        for callback in self._on_state_change:
            callback(old_state, new_state)

    def _emit_countdown_tick(self, seconds: int) -> None:
        for callback in self._on_countdown_tick:
            callback(seconds)

    def _emit_match_start(self) -> None:
        for callback in self._on_match_start:
            callback()

    def _emit_match_end(self, result: MatchResult) -> None:
        for callback in self._on_match_end:
            callback(result)

    def _emit_player_join(self, player_id: str) -> None:
        for callback in self._on_player_join:
            callback(player_id)

    def _emit_player_leave(self, player_id: str) -> None:
        for callback in self._on_player_leave:
            callback(player_id)

    def _emit_player_ready(self, player_id: str, is_ready: bool) -> None:
        for callback in self._on_player_ready:
            callback(player_id, is_ready)

    def _emit_round_start(self, round_number: int) -> None:
        for callback in self._on_round_start:
            callback(round_number)

    def _emit_round_end(self, round_number: int, winner_id: Optional[str]) -> None:
        for callback in self._on_round_end:
            callback(round_number, winner_id)


class Match:
    """
    Manages match lifecycle for a game mode.

    Handles the flow from lobby through gameplay to results,
    including player ready states, countdown, and match completion.
    """

    def __init__(self, game_mode: GameMode, config: Optional[MatchConfig] = None):
        """
        Initialize match with game mode and configuration.

        Args:
            game_mode: The game mode instance to run
            config: Match configuration (uses defaults if not provided)
        """
        self.game_mode = game_mode
        self.config = config or MatchConfig()
        self.events = MatchEvents()

        # State
        self._state = MatchState.WAITING
        self._ready_players: Set[str] = set()
        self._spectators: Set[str] = set()

        # Timing
        self._state_start_time: Optional[float] = None
        self._countdown_seconds: int = 0
        self._last_countdown_tick: int = -1

        # Match tracking
        self._match_id: str = ""
        self._match_start_time: Optional[float] = None
        self._match_end_time: Optional[float] = None
        self._result: Optional[MatchResult] = None

        # Player stats tracking
        self._player_stats: Dict[str, Dict[str, Any]] = {}
        self._total_kills: int = 0
        self._round_winners: List[Optional[str]] = []

        # Round tracking
        self._current_round: int = 1
        self._round_in_progress: bool = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def state(self) -> MatchState:
        """Get current match state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if match is in an active state (not waiting or complete)."""
        return self._state in (MatchState.STARTING, MatchState.IN_PROGRESS, MatchState.ENDING)

    @property
    def is_joinable(self) -> bool:
        """Check if players can join the match."""
        if self._state == MatchState.WAITING:
            return True
        if self._state == MatchState.IN_PROGRESS:
            return self.config.allow_join_in_progress
        return False

    @property
    def player_count(self) -> int:
        """Get number of players in match."""
        return len(self.game_mode.players)

    @property
    def ready_count(self) -> int:
        """Get number of ready players."""
        return len(self._ready_players)

    @property
    def match_id(self) -> str:
        """Get unique match identifier."""
        return self._match_id

    @property
    def result(self) -> Optional[MatchResult]:
        """Get match result (only available after completion)."""
        return self._result

    # =========================================================================
    # State Management
    # =========================================================================

    def _set_state(self, new_state: MatchState) -> None:
        """Set match state and emit event."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._state_start_time = time.time()
            self.events._emit_state_change(old_state, new_state)

    def can_start(self) -> bool:
        """Check if match can be started."""
        if self._state != MatchState.WAITING:
            return False

        player_count = len(self.game_mode.players)
        if player_count < self.config.min_players_to_start:
            return False

        if self.config.auto_start_when_ready:
            ready_ratio = len(self._ready_players) / max(1, player_count)
            return ready_ratio >= self.config.ready_percentage_to_start

        return True

    def start_countdown(self) -> bool:
        """
        Start the countdown to match start.

        Returns:
            True if countdown started successfully
        """
        if not self.can_start():
            return False

        self._set_state(MatchState.STARTING)
        self._countdown_seconds = int(self.config.countdown_duration_seconds)
        self._last_countdown_tick = self._countdown_seconds + 1
        return True

    def start_match(self) -> bool:
        """
        Start the match immediately.

        Returns:
            True if match started successfully
        """
        if self._state not in (MatchState.WAITING, MatchState.STARTING):
            return False

        self._set_state(MatchState.IN_PROGRESS)
        self._match_start_time = time.time()
        self._match_id = f"match_{int(self._match_start_time * 1000)}"
        self._current_round = 1
        self._round_in_progress = True

        # Initialize player stats
        for player_id in self.game_mode.players:
            self._player_stats[player_id] = {
                "kills": 0,
                "deaths": 0,
                "assists": 0,
                "score": 0,
                "time_alive": 0.0,
            }

        # Start the game mode timer
        self.game_mode.start()

        self.events._emit_match_start()
        self.events._emit_round_start(self._current_round)
        return True

    def end_match(self, winner_id: Optional[str] = None) -> bool:
        """
        End the match and start ending sequence.

        Args:
            winner_id: The winner (player or team ID)

        Returns:
            True if match ended successfully
        """
        if self._state != MatchState.IN_PROGRESS:
            return False

        self._set_state(MatchState.ENDING)
        self._match_end_time = time.time()

        # Build result
        self._result = self._build_result(winner_id)

        return True

    def complete_match(self) -> bool:
        """
        Complete the match and move to results state.

        Returns:
            True if completed successfully
        """
        if self._state != MatchState.ENDING:
            return False

        self._set_state(MatchState.COMPLETE)
        self.events._emit_match_end(self._result)
        return True

    def force_end(self, reason: str = "forced") -> None:
        """Force the match to end immediately."""
        if self._state in (MatchState.WAITING, MatchState.COMPLETE):
            return

        self._match_end_time = time.time()
        self._result = self._build_result(None)
        self._result.player_stats["_force_end_reason"] = {"reason": reason}
        self._set_state(MatchState.COMPLETE)
        self.events._emit_match_end(self._result)

    def reset(self) -> None:
        """Reset match to waiting state."""
        self._set_state(MatchState.WAITING)
        self._ready_players.clear()
        self._spectators.clear()
        self._match_id = ""
        self._match_start_time = None
        self._match_end_time = None
        self._result = None
        self._player_stats.clear()
        self._total_kills = 0
        self._round_winners.clear()
        self._current_round = 1
        self._round_in_progress = False
        self._countdown_seconds = 0
        self._last_countdown_tick = -1
        self.game_mode.reset()

    # =========================================================================
    # Player Management
    # =========================================================================

    def add_player(self, player_id: str, team_id: Optional[str] = None) -> bool:
        """
        Add a player to the match.

        Args:
            player_id: Unique player identifier
            team_id: Team to assign player to

        Returns:
            True if player was added
        """
        if not self.is_joinable:
            return False

        if self.game_mode.add_player(player_id, team_id):
            self._player_stats[player_id] = {
                "kills": 0,
                "deaths": 0,
                "assists": 0,
                "score": 0,
                "time_alive": 0.0,
            }
            self.events._emit_player_join(player_id)
            return True
        return False

    def remove_player(self, player_id: str) -> bool:
        """
        Remove a player from the match.

        Args:
            player_id: Player to remove

        Returns:
            True if player was removed
        """
        if player_id not in self.game_mode.players:
            return False

        self._ready_players.discard(player_id)
        if self.game_mode.remove_player(player_id):
            self.events._emit_player_leave(player_id)
            return True
        return False

    def set_player_ready(self, player_id: str, is_ready: bool = True) -> bool:
        """
        Set player ready state.

        Args:
            player_id: Player to update
            is_ready: Ready state

        Returns:
            True if state was updated
        """
        if player_id not in self.game_mode.players:
            return False

        if self._state != MatchState.WAITING:
            return False

        if is_ready:
            self._ready_players.add(player_id)
        else:
            self._ready_players.discard(player_id)

        self.events._emit_player_ready(player_id, is_ready)

        # Auto-start check
        if self.config.auto_start_when_ready and self.can_start():
            self.start_countdown()

        return True

    def is_player_ready(self, player_id: str) -> bool:
        """Check if player is ready."""
        return player_id in self._ready_players

    def add_spectator(self, spectator_id: str) -> bool:
        """Add a spectator to the match."""
        if not self.config.allow_spectators:
            return False
        if len(self._spectators) >= self.config.max_spectators:
            return False
        if spectator_id in self.game_mode.players:
            return False

        self._spectators.add(spectator_id)
        return True

    def remove_spectator(self, spectator_id: str) -> bool:
        """Remove a spectator from the match."""
        if spectator_id not in self._spectators:
            return False
        self._spectators.discard(spectator_id)
        return True

    def get_spectators(self) -> Set[str]:
        """Get set of spectator IDs."""
        return self._spectators.copy()

    # =========================================================================
    # Round Management
    # =========================================================================

    def start_round(self) -> bool:
        """Start a new round."""
        if self._state != MatchState.IN_PROGRESS:
            return False
        if self._round_in_progress:
            return False

        self._round_in_progress = True
        self.game_mode.reset_round()
        self.events._emit_round_start(self._current_round)
        return True

    def end_round(self, winner_id: Optional[str] = None) -> bool:
        """End the current round."""
        if not self._round_in_progress:
            return False

        self._round_in_progress = False
        self._round_winners.append(winner_id)
        self.game_mode.end_round(winner_id)
        self.events._emit_round_end(self._current_round, winner_id)

        self._current_round += 1
        return True

    @property
    def current_round(self) -> int:
        """Get current round number."""
        return self._current_round

    @property
    def is_round_in_progress(self) -> bool:
        """Check if a round is currently in progress."""
        return self._round_in_progress

    # =========================================================================
    # Update and Tick
    # =========================================================================

    def update(self, delta_time: float) -> None:
        """
        Update match state.

        Should be called every frame to handle state transitions.

        Args:
            delta_time: Time since last update in seconds
        """
        if self._state == MatchState.STARTING:
            self._update_countdown()

        elif self._state == MatchState.IN_PROGRESS:
            self._update_gameplay(delta_time)

        elif self._state == MatchState.ENDING:
            self._update_ending()

    def _update_countdown(self) -> None:
        """Update countdown state."""
        if self._state_start_time is None:
            return

        elapsed = time.time() - self._state_start_time
        remaining = self.config.countdown_duration_seconds - elapsed

        if remaining <= 0:
            self.start_match()
            return

        # Emit countdown ticks
        current_second = int(remaining) + 1
        if current_second != self._last_countdown_tick:
            self._last_countdown_tick = current_second
            self.events._emit_countdown_tick(current_second)

    def _update_gameplay(self, delta_time: float) -> None:
        """Update during active gameplay."""
        # Check win condition
        game_over, winner = self.game_mode.check_win_condition()

        if game_over:
            # Check for overtime
            if winner is None and self.game_mode.should_go_to_overtime():
                self.game_mode.start_overtime()
            else:
                self.end_match(winner)
        elif self.game_mode.is_time_expired():
            if self.game_mode.should_go_to_overtime():
                self.game_mode.start_overtime()
            else:
                leader = self.game_mode.get_leading_player_or_team()
                self.end_match(leader)

    def _update_ending(self) -> None:
        """Update during ending sequence."""
        if self._state_start_time is None:
            return

        elapsed = time.time() - self._state_start_time
        if elapsed >= self.config.end_sequence_duration_seconds:
            self.complete_match()

    # =========================================================================
    # Statistics and Results
    # =========================================================================

    def record_kill(self, killer_id: str, victim_id: str, assists: Optional[List[str]] = None) -> None:
        """
        Record a kill for statistics.

        Args:
            killer_id: The player who got the kill
            victim_id: The player who died
            assists: List of players who assisted
        """
        self._total_kills += 1

        if killer_id in self._player_stats:
            self._player_stats[killer_id]["kills"] += 1
        if victim_id in self._player_stats:
            self._player_stats[victim_id]["deaths"] += 1

        if assists:
            for assist_id in assists:
                if assist_id in self._player_stats:
                    self._player_stats[assist_id]["assists"] += 1

    def get_player_stats(self, player_id: str) -> Dict[str, Any]:
        """Get statistics for a player."""
        return self._player_stats.get(player_id, {}).copy()

    def get_all_player_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all players."""
        return {pid: stats.copy() for pid, stats in self._player_stats.items()}

    def _build_result(self, winner_id: Optional[str]) -> MatchResult:
        """Build match result object."""
        duration = 0.0
        if self._match_start_time and self._match_end_time:
            duration = self._match_end_time - self._match_start_time

        # Determine MVP (player with highest score)
        mvp = None
        if self._player_stats:
            scores = {pid: stats.get("kills", 0) for pid, stats in self._player_stats.items()}
            if scores:
                mvp = max(scores, key=scores.get)

        # Check for draw
        is_draw = winner_id is None and len(self.game_mode.players) > 1

        return MatchResult(
            winner_id=winner_id,
            is_draw=is_draw,
            final_scores=dict(self.game_mode.player_scores),
            team_scores=dict(self.game_mode.team_scores),
            mvp_player_id=mvp,
            match_duration_seconds=duration,
            total_kills=self._total_kills,
            player_stats=self.get_all_player_stats(),
            rounds_played=self._current_round - 1 if self._current_round > 1 else 1,
            round_winners=self._round_winners.copy(),
        )

    def get_match_info(self) -> Dict[str, Any]:
        """Get current match information."""
        return {
            "match_id": self._match_id,
            "state": self._state.name,
            "player_count": len(self.game_mode.players),
            "ready_count": len(self._ready_players),
            "spectator_count": len(self._spectators),
            "current_round": self._current_round,
            "is_joinable": self.is_joinable,
            "game_mode": {
                "mode_id": self.game_mode.mode_id,
                "mode_name": self.game_mode.mode_name,
                "elapsed_time": self.game_mode.get_elapsed_time(),
                "remaining_time": self.game_mode.get_remaining_time(),
                "in_overtime": self.game_mode.is_in_overtime(),
            },
        }
