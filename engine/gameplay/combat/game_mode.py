"""
GameMode base class with rules, win conditions, spawning, and scoring.

Provides the foundation for all game modes including:
- Game rules configuration
- Win condition evaluation
- Spawn logic delegation
- Scoring system
- Time limit management
- Overtime handling
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import time


class WinConditionType(Enum):
    """Types of win conditions for game modes."""
    SCORE_LIMIT = auto()      # First to reach score limit wins
    TIME_LIMIT = auto()       # Highest score when time expires
    ELIMINATION = auto()      # Last player/team standing
    OBJECTIVE = auto()        # Complete objective (capture, hold, etc.)
    SURVIVAL = auto()         # Survive until end (battle royale)
    ROUNDS = auto()           # Win X rounds
    COMBINED = auto()         # Multiple conditions (score OR time)


class ScoringEventType(Enum):
    """Types of scoring events."""
    KILL = auto()
    DEATH = auto()
    ASSIST = auto()
    OBJECTIVE_CAPTURE = auto()
    OBJECTIVE_DEFEND = auto()
    ZONE_TICK = auto()
    FLAG_CAPTURE = auto()
    FLAG_RETURN = auto()
    SURVIVAL_TICK = auto()
    ROUND_WIN = auto()
    BONUS = auto()
    PENALTY = auto()


@dataclass
class ScoringEvent:
    """Represents a scoring event in the game."""
    event_type: ScoringEventType
    player_id: str
    team_id: Optional[str] = None
    points: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WinCondition:
    """Definition of a win condition."""
    condition_type: WinConditionType
    target_value: Optional[int] = None  # Score limit, rounds to win, etc.
    time_limit_seconds: Optional[float] = None
    custom_check: Optional[Callable[['GameMode'], Optional[str]]] = None  # Returns winner ID
    description: str = ""

    def is_met(self, game_mode: 'GameMode') -> Tuple[bool, Optional[str]]:
        """
        Check if the win condition is met.

        Returns:
            Tuple of (is_met, winner_id/team_id or None)
        """
        if self.custom_check:
            winner = self.custom_check(game_mode)
            return (winner is not None, winner)

        if self.condition_type == WinConditionType.SCORE_LIMIT:
            if self.target_value is None:
                return (False, None)
            for player_id, score in game_mode.player_scores.items():
                if score >= self.target_value:
                    return (True, player_id)
            for team_id, score in game_mode.team_scores.items():
                if score >= self.target_value:
                    return (True, team_id)

        elif self.condition_type == WinConditionType.TIME_LIMIT:
            if self.time_limit_seconds is not None:
                elapsed = game_mode.get_elapsed_time()
                if elapsed >= self.time_limit_seconds:
                    # Find highest scorer
                    winner = game_mode.get_leading_player_or_team()
                    return (True, winner)

        elif self.condition_type == WinConditionType.ELIMINATION:
            alive_players = game_mode.get_alive_players()
            alive_teams = game_mode.get_alive_teams()

            if game_mode.is_team_based:
                if len(alive_teams) == 1:
                    return (True, list(alive_teams)[0])
                elif len(alive_teams) == 0:
                    return (True, None)  # Draw
            else:
                if len(alive_players) == 1:
                    return (True, list(alive_players)[0])
                elif len(alive_players) == 0:
                    return (True, None)  # Draw

        elif self.condition_type == WinConditionType.ROUNDS:
            if self.target_value is None:
                return (False, None)
            for team_id, wins in game_mode.round_wins.items():
                if wins >= self.target_value:
                    return (True, team_id)
            for player_id, wins in game_mode.round_wins.items():
                if wins >= self.target_value:
                    return (True, player_id)

        return (False, None)


@dataclass
class GameModeRules:
    """Configuration for game mode rules."""
    friendly_fire: bool = False
    respawn_enabled: bool = True
    respawn_delay_seconds: float = 3.0
    max_respawns: Optional[int] = None  # None = unlimited
    allow_team_switching: bool = False
    allow_late_join: bool = True
    min_players: int = 2
    max_players: int = 32
    min_teams: int = 0  # 0 = no teams (FFA)
    max_teams: int = 0
    auto_balance_teams: bool = True
    loadout_selection: bool = True
    killstreak_enabled: bool = False
    overtime_enabled: bool = True
    overtime_duration_seconds: float = 60.0


@dataclass
class GameModeConfig:
    """Complete game mode configuration."""
    mode_id: str
    mode_name: str
    description: str = ""
    rules: GameModeRules = field(default_factory=GameModeRules)
    win_conditions: List[WinCondition] = field(default_factory=list)
    scoring_values: Dict[ScoringEventType, int] = field(default_factory=dict)
    time_limit_seconds: Optional[float] = None
    score_limit: Optional[int] = None
    round_limit: Optional[int] = None
    is_team_based: bool = False
    default_team_count: int = 2
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class GameMode(ABC):
    """
    Abstract base class for game modes.

    Defines the interface for game mode rules, win conditions,
    spawn logic, scoring, and time management.
    """

    def __init__(self, config: GameModeConfig):
        """Initialize the game mode with configuration."""
        self.config = config
        self.mode_id = config.mode_id
        self.mode_name = config.mode_name
        self.rules = config.rules
        self.is_team_based = config.is_team_based

        # Scoring
        self.player_scores: Dict[str, int] = {}
        self.team_scores: Dict[str, int] = {}
        self.scoring_history: List[ScoringEvent] = []

        # Players and teams
        self.players: Set[str] = set()
        self.teams: Dict[str, Set[str]] = {}  # team_id -> set of player_ids
        self.player_teams: Dict[str, str] = {}  # player_id -> team_id
        self.alive_players: Set[str] = set()
        self.player_deaths: Dict[str, int] = {}
        self.player_respawns: Dict[str, int] = {}

        # Round tracking
        self.current_round: int = 1
        self.round_wins: Dict[str, int] = {}  # player/team_id -> wins

        # Time tracking
        self._start_time: Optional[float] = None
        self._pause_time: Optional[float] = None
        self._total_paused_time: float = 0.0
        self._is_paused: bool = False
        self._in_overtime: bool = False
        self._overtime_start: Optional[float] = None

        # Default scoring values
        self._scoring_values = {
            ScoringEventType.KILL: 1,
            ScoringEventType.DEATH: 0,
            ScoringEventType.ASSIST: 0,
            ScoringEventType.OBJECTIVE_CAPTURE: 1,
            ScoringEventType.OBJECTIVE_DEFEND: 1,
            ScoringEventType.ZONE_TICK: 1,
            ScoringEventType.FLAG_CAPTURE: 1,
            ScoringEventType.FLAG_RETURN: 0,
            ScoringEventType.SURVIVAL_TICK: 1,
            ScoringEventType.ROUND_WIN: 1,
            ScoringEventType.BONUS: 1,
            ScoringEventType.PENALTY: -1,
        }
        self._scoring_values.update(config.scoring_values)

        # Callbacks
        self._on_score_change: List[Callable[[str, int, int], None]] = []
        self._on_player_eliminated: List[Callable[[str], None]] = []
        self._on_round_end: List[Callable[[int, Optional[str]], None]] = []

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    def get_spawn_location(self, player_id: str) -> Tuple[float, float, float]:
        """
        Get spawn location for a player.

        Args:
            player_id: The player to spawn

        Returns:
            Tuple of (x, y, z) coordinates
        """
        pass

    @abstractmethod
    def on_player_killed(
        self,
        victim_id: str,
        killer_id: Optional[str] = None,
        weapon: Optional[str] = None,
        assists: Optional[List[str]] = None
    ) -> None:
        """
        Handle player death event.

        Args:
            victim_id: The player who died
            killer_id: The killer (None for environmental/suicide)
            weapon: Weapon used for the kill
            assists: List of player IDs who assisted
        """
        pass

    @abstractmethod
    def check_win_condition(self) -> Tuple[bool, Optional[str]]:
        """
        Check if any win condition is met.

        Returns:
            Tuple of (game_over, winner_id/team_id or None)
        """
        pass

    # =========================================================================
    # Player Management
    # =========================================================================

    def add_player(self, player_id: str, team_id: Optional[str] = None) -> bool:
        """
        Add a player to the game.

        Args:
            player_id: Unique player identifier
            team_id: Team to assign player to (for team modes)

        Returns:
            True if player was added successfully
        """
        if len(self.players) >= self.rules.max_players:
            return False

        if player_id in self.players:
            return False

        self.players.add(player_id)
        self.player_scores[player_id] = 0
        self.player_deaths[player_id] = 0
        self.player_respawns[player_id] = 0
        self.alive_players.add(player_id)

        if self.is_team_based and team_id:
            self.assign_to_team(player_id, team_id)

        return True

    def remove_player(self, player_id: str) -> bool:
        """Remove a player from the game."""
        if player_id not in self.players:
            return False

        self.players.discard(player_id)
        self.alive_players.discard(player_id)
        self.player_scores.pop(player_id, None)
        self.player_deaths.pop(player_id, None)
        self.player_respawns.pop(player_id, None)

        # Remove from team
        team_id = self.player_teams.pop(player_id, None)
        if team_id and team_id in self.teams:
            self.teams[team_id].discard(player_id)

        return True

    def assign_to_team(self, player_id: str, team_id: str) -> bool:
        """Assign a player to a team."""
        if player_id not in self.players:
            return False

        # Remove from current team
        old_team = self.player_teams.get(player_id)
        if old_team and old_team in self.teams:
            self.teams[old_team].discard(player_id)

        # Add to new team
        if team_id not in self.teams:
            self.teams[team_id] = set()
            self.team_scores[team_id] = 0
            self.round_wins[team_id] = 0

        self.teams[team_id].add(player_id)
        self.player_teams[player_id] = team_id
        return True

    def create_team(self, team_id: str) -> bool:
        """Create a new team."""
        if team_id in self.teams:
            return False
        if len(self.teams) >= self.rules.max_teams > 0:
            return False

        self.teams[team_id] = set()
        self.team_scores[team_id] = 0
        self.round_wins[team_id] = 0
        return True

    def get_player_team(self, player_id: str) -> Optional[str]:
        """Get the team ID for a player."""
        return self.player_teams.get(player_id)

    def get_team_players(self, team_id: str) -> Set[str]:
        """Get all players in a team."""
        return self.teams.get(team_id, set()).copy()

    def get_alive_players(self) -> Set[str]:
        """Get all alive players."""
        return self.alive_players.copy()

    def get_alive_teams(self) -> Set[str]:
        """Get teams with at least one alive player."""
        alive_teams = set()
        for team_id, players in self.teams.items():
            if any(p in self.alive_players for p in players):
                alive_teams.add(team_id)
        return alive_teams

    def is_player_alive(self, player_id: str) -> bool:
        """Check if a player is alive."""
        return player_id in self.alive_players

    def mark_player_dead(self, player_id: str) -> None:
        """Mark a player as dead."""
        self.alive_players.discard(player_id)
        self.player_deaths[player_id] = self.player_deaths.get(player_id, 0) + 1

        for callback in self._on_player_eliminated:
            callback(player_id)

    def respawn_player(self, player_id: str) -> bool:
        """
        Respawn a player if allowed.

        Returns:
            True if respawn was allowed
        """
        if not self.rules.respawn_enabled:
            return False

        if self.rules.max_respawns is not None:
            current_respawns = self.player_respawns.get(player_id, 0)
            if current_respawns >= self.rules.max_respawns:
                return False

        self.alive_players.add(player_id)
        self.player_respawns[player_id] = self.player_respawns.get(player_id, 0) + 1
        return True

    # =========================================================================
    # Scoring
    # =========================================================================

    def record_score_event(self, event: ScoringEvent) -> None:
        """Record a scoring event and update scores."""
        points = event.points
        if points == 0:
            points = self._scoring_values.get(event.event_type, 0)

        event.points = points
        self.scoring_history.append(event)

        old_score = self.player_scores.get(event.player_id, 0)
        new_score = old_score + points
        self.player_scores[event.player_id] = new_score

        # Update team score
        if event.team_id:
            self.team_scores[event.team_id] = self.team_scores.get(event.team_id, 0) + points
        elif event.player_id in self.player_teams:
            team_id = self.player_teams[event.player_id]
            self.team_scores[team_id] = self.team_scores.get(team_id, 0) + points

        for callback in self._on_score_change:
            callback(event.player_id, old_score, new_score)

    def add_score(
        self,
        player_id: str,
        event_type: ScoringEventType,
        points: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add score for a player."""
        event = ScoringEvent(
            event_type=event_type,
            player_id=player_id,
            team_id=self.player_teams.get(player_id),
            points=points or self._scoring_values.get(event_type, 0),
            metadata=metadata or {}
        )
        self.record_score_event(event)

    def get_player_score(self, player_id: str) -> int:
        """Get a player's current score."""
        return self.player_scores.get(player_id, 0)

    def get_team_score(self, team_id: str) -> int:
        """Get a team's current score."""
        return self.team_scores.get(team_id, 0)

    def get_leading_player_or_team(self) -> Optional[str]:
        """Get the ID of the leading player or team."""
        if self.is_team_based:
            if not self.team_scores:
                return None
            return max(self.team_scores, key=self.team_scores.get)
        else:
            if not self.player_scores:
                return None
            return max(self.player_scores, key=self.player_scores.get)

    def get_leaderboard(self) -> List[Tuple[str, int]]:
        """Get sorted leaderboard (player/team, score)."""
        if self.is_team_based:
            return sorted(self.team_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted(self.player_scores.items(), key=lambda x: x[1], reverse=True)

    def set_scoring_value(self, event_type: ScoringEventType, value: int) -> None:
        """Set the point value for a scoring event type."""
        self._scoring_values[event_type] = value

    # =========================================================================
    # Time Management
    # =========================================================================

    def start(self) -> None:
        """Start the game mode timer."""
        self._start_time = time.time()
        self._is_paused = False
        self._total_paused_time = 0.0

    def pause(self) -> None:
        """Pause the game mode timer."""
        if not self._is_paused:
            self._pause_time = time.time()
            self._is_paused = True

    def resume(self) -> None:
        """Resume the game mode timer."""
        if self._is_paused and self._pause_time is not None:
            self._total_paused_time += time.time() - self._pause_time
            self._is_paused = False
            self._pause_time = None

    def get_elapsed_time(self) -> float:
        """Get elapsed game time in seconds (excluding paused time)."""
        if self._start_time is None:
            return 0.0

        current = time.time() if not self._is_paused else self._pause_time
        return current - self._start_time - self._total_paused_time

    def get_remaining_time(self) -> Optional[float]:
        """Get remaining time if time limit is set."""
        if self.config.time_limit_seconds is None:
            return None

        remaining = self.config.time_limit_seconds - self.get_elapsed_time()

        if self._in_overtime and self._overtime_start:
            overtime_elapsed = time.time() - self._overtime_start
            remaining = self.rules.overtime_duration_seconds - overtime_elapsed

        return max(0.0, remaining)

    def is_time_expired(self) -> bool:
        """Check if time limit has been reached."""
        remaining = self.get_remaining_time()
        return remaining is not None and remaining <= 0

    def start_overtime(self) -> bool:
        """Start overtime period."""
        if not self.rules.overtime_enabled:
            return False
        if self._in_overtime:
            return False

        self._in_overtime = True
        self._overtime_start = time.time()
        return True

    def is_in_overtime(self) -> bool:
        """Check if game is in overtime."""
        return self._in_overtime

    def should_go_to_overtime(self) -> bool:
        """Check if game should enter overtime (tie at time limit)."""
        if not self.rules.overtime_enabled:
            return False
        if not self.is_time_expired():
            return False

        # Check for tie
        if self.is_team_based:
            if len(set(self.team_scores.values())) < len(self.team_scores):
                # At least two teams have same score
                return True
        else:
            if len(set(self.player_scores.values())) < len(self.player_scores):
                return True

        return False

    # =========================================================================
    # Round Management
    # =========================================================================

    def end_round(self, winner_id: Optional[str] = None) -> None:
        """End the current round."""
        if winner_id:
            self.round_wins[winner_id] = self.round_wins.get(winner_id, 0) + 1

        for callback in self._on_round_end:
            callback(self.current_round, winner_id)

        self.current_round += 1

    def reset_round(self) -> None:
        """Reset state for a new round."""
        self.alive_players = self.players.copy()
        # Subclasses can override for additional reset logic

    def get_round_wins(self, player_or_team_id: str) -> int:
        """Get number of rounds won."""
        return self.round_wins.get(player_or_team_id, 0)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_score_change(self, callback: Callable[[str, int, int], None]) -> None:
        """Register callback for score changes (player_id, old_score, new_score)."""
        self._on_score_change.append(callback)

    def on_player_eliminated(self, callback: Callable[[str], None]) -> None:
        """Register callback for player elimination."""
        self._on_player_eliminated.append(callback)

    def on_round_end(self, callback: Callable[[int, Optional[str]], None]) -> None:
        """Register callback for round end (round_number, winner_id)."""
        self._on_round_end.append(callback)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_friendly_fire(self, player_a: str, player_b: str) -> bool:
        """Check if damage between two players would be friendly fire."""
        if not self.is_team_based:
            return False

        team_a = self.player_teams.get(player_a)
        team_b = self.player_teams.get(player_b)

        return team_a is not None and team_a == team_b

    def can_damage(self, attacker: str, target: str) -> bool:
        """Check if attacker can damage target."""
        if attacker == target:
            return True  # Self-damage always allowed

        if self.is_friendly_fire(attacker, target):
            return self.rules.friendly_fire

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get current game mode statistics."""
        return {
            "mode_id": self.mode_id,
            "mode_name": self.mode_name,
            "player_count": len(self.players),
            "team_count": len(self.teams),
            "alive_count": len(self.alive_players),
            "elapsed_time": self.get_elapsed_time(),
            "remaining_time": self.get_remaining_time(),
            "current_round": self.current_round,
            "in_overtime": self._in_overtime,
            "is_paused": self._is_paused,
            "player_scores": dict(self.player_scores),
            "team_scores": dict(self.team_scores),
            "leaderboard": self.get_leaderboard(),
        }

    def reset(self) -> None:
        """Reset the game mode to initial state."""
        self.player_scores = {p: 0 for p in self.players}
        self.team_scores = {t: 0 for t in self.teams}
        self.scoring_history.clear()
        self.alive_players = self.players.copy()
        self.player_deaths = {p: 0 for p in self.players}
        self.player_respawns = {p: 0 for p in self.players}
        self.current_round = 1
        self.round_wins = {t: 0 for t in self.teams}
        self._start_time = None
        self._pause_time = None
        self._total_paused_time = 0.0
        self._is_paused = False
        self._in_overtime = False
        self._overtime_start = None
