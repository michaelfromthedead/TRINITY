"""
Combat System - Scoring Module

Provides comprehensive score tracking for combat systems:
- Kill/death/assist tracking
- Score calculation with configurable point values
- Leaderboard management
- Killstreaks and multi-kills
- Bonus scoring (objectives, streaks)
- Score events and callbacks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import time

from .constants import (
    POINTS_PER_KILL,
    POINTS_PER_ASSIST,
    POINTS_PER_DEATH,
    POINTS_PER_OBJECTIVE,
    POINTS_PER_HEADSHOT_BONUS,
    POINTS_PER_FIRST_BLOOD,
    POINTS_PER_REVENGE_KILL,
    POINTS_PER_KILLSTREAK_BONUS,
    ASSIST_DAMAGE_THRESHOLD,
    ASSIST_TIME_WINDOW,
    KILLSTREAK_THRESHOLDS,
    MULTI_KILL_WINDOW,
    MULTI_KILL_NAMES,
    ScoringConfig,
    DEFAULT_SCORING_CONFIG,
    MAX_SCORING_HISTORY_SIZE,
    DEFAULT_MAX_HEALTH,
)


# =============================================================================
# ENUMS
# =============================================================================


class ScoreEventType(Enum):
    """Types of scoring events."""

    KILL = auto()
    DEATH = auto()
    ASSIST = auto()
    HEADSHOT = auto()
    FIRST_BLOOD = auto()
    REVENGE = auto()
    KILLSTREAK = auto()
    KILLSTREAK_ENDED = auto()
    MULTI_KILL = auto()
    OBJECTIVE_CAPTURE = auto()
    OBJECTIVE_DEFEND = auto()
    OBJECTIVE_PROGRESS = auto()
    BONUS = auto()
    PENALTY = auto()
    TEAM_BONUS = auto()
    ROUND_WIN = auto()
    MATCH_WIN = auto()


class LeaderboardSortKey(Enum):
    """Keys for sorting leaderboards."""

    SCORE = auto()
    KILLS = auto()
    DEATHS = auto()
    ASSISTS = auto()
    KD_RATIO = auto()
    KDA_RATIO = auto()
    DAMAGE_DEALT = auto()
    DAMAGE_TAKEN = auto()
    OBJECTIVES = auto()


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PlayerStats:
    """Complete statistics for a player."""

    player_id: str
    team_id: Optional[str] = None

    # Core stats
    score: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0

    # Combat stats
    damage_dealt: float = 0.0
    damage_taken: float = 0.0
    healing_done: float = 0.0
    headshots: int = 0

    # Streaks
    current_killstreak: int = 0
    best_killstreak: int = 0
    current_deathstreak: int = 0

    # Multi-kills
    double_kills: int = 0
    triple_kills: int = 0
    quad_kills: int = 0
    penta_kills: int = 0

    # Special achievements
    first_bloods: int = 0
    revenge_kills: int = 0

    # Objective stats
    objectives_captured: int = 0
    objectives_defended: int = 0
    objective_time: float = 0.0

    # Time tracking
    time_alive: float = 0.0
    time_dead: float = 0.0
    join_time: float = field(default_factory=time.time)

    # Damage tracking for assists
    _damage_to_targets: Dict[str, Tuple[float, float]] = field(
        default_factory=dict
    )  # target_id -> (damage, timestamp)

    # Kill tracking for revenge
    _killed_by: Set[str] = field(default_factory=set)
    _last_kill_time: float = 0.0
    _multi_kill_count: int = 0

    @property
    def kd_ratio(self) -> float:
        """Kill/death ratio."""
        if self.deaths == 0:
            return float(self.kills)
        return self.kills / self.deaths

    @property
    def kda_ratio(self) -> float:
        """(Kills + Assists) / Deaths ratio."""
        if self.deaths == 0:
            return float(self.kills + self.assists)
        return (self.kills + self.assists) / self.deaths

    @property
    def total_multi_kills(self) -> int:
        """Total number of multi-kills."""
        return (
            self.double_kills +
            self.triple_kills +
            self.quad_kills +
            self.penta_kills
        )

    def record_damage_dealt(
        self,
        target_id: str,
        amount: float,
        max_health: float,
    ) -> None:
        """Record damage dealt for assist tracking."""
        current = self._damage_to_targets.get(target_id, (0.0, 0.0))
        self._damage_to_targets[target_id] = (
            current[0] + amount,
            time.time(),
        )
        self.damage_dealt += amount

    def get_assist_damage(
        self,
        target_id: str,
        max_health: float,
        time_window: float = ASSIST_TIME_WINDOW,
    ) -> Optional[float]:
        """Get damage dealt to target within assist window."""
        if target_id not in self._damage_to_targets:
            return None

        damage, timestamp = self._damage_to_targets[target_id]
        if time.time() - timestamp > time_window:
            return None

        return damage

    def clear_damage_tracking(self, target_id: str) -> None:
        """Clear damage tracking for a target (after they die)."""
        self._damage_to_targets.pop(target_id, None)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "player_id": self.player_id,
            "team_id": self.team_id,
            "score": self.score,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "damage_dealt": self.damage_dealt,
            "damage_taken": self.damage_taken,
            "healing_done": self.healing_done,
            "headshots": self.headshots,
            "current_killstreak": self.current_killstreak,
            "best_killstreak": self.best_killstreak,
            "double_kills": self.double_kills,
            "triple_kills": self.triple_kills,
            "quad_kills": self.quad_kills,
            "penta_kills": self.penta_kills,
            "first_bloods": self.first_bloods,
            "revenge_kills": self.revenge_kills,
            "objectives_captured": self.objectives_captured,
            "objectives_defended": self.objectives_defended,
            "objective_time": self.objective_time,
            "kd_ratio": self.kd_ratio,
            "kda_ratio": self.kda_ratio,
        }


@dataclass
class TeamStats:
    """Statistics for a team."""

    team_id: str
    score: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    objectives: int = 0
    rounds_won: int = 0
    members: Set[str] = field(default_factory=set)

    @property
    def member_count(self) -> int:
        """Number of team members."""
        return len(self.members)

    @property
    def kd_ratio(self) -> float:
        """Team kill/death ratio."""
        if self.deaths == 0:
            return float(self.kills)
        return self.kills / self.deaths


@dataclass
class ScoreEvent:
    """A scoring event that occurred."""

    event_type: ScoreEventType
    player_id: str
    points: int
    timestamp: float = field(default_factory=time.time)
    team_id: Optional[str] = None
    target_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_positive(self) -> bool:
        """Whether this event added points."""
        return self.points > 0

    @property
    def is_negative(self) -> bool:
        """Whether this event removed points."""
        return self.points < 0


@dataclass
class LeaderboardEntry:
    """An entry in a leaderboard."""

    rank: int
    player_id: str
    score: int
    kills: int
    deaths: int
    assists: int
    team_id: Optional[str] = None

    @property
    def kd_ratio(self) -> float:
        """Kill/death ratio."""
        if self.deaths == 0:
            return float(self.kills)
        return self.kills / self.deaths


# =============================================================================
# SCORING SYSTEM
# =============================================================================


class ScoringSystem:
    """
    Comprehensive scoring system for combat.

    Features:
    - Player and team score tracking
    - Kill/death/assist attribution
    - Killstreak and multi-kill detection
    - Configurable point values
    - Leaderboard generation
    - Score events and history
    """

    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        is_team_based: bool = False,
    ) -> None:
        """
        Initialize the scoring system.

        Args:
            config: Scoring configuration
            is_team_based: Whether this is a team-based game
        """
        self._config = config or DEFAULT_SCORING_CONFIG
        self._is_team_based = is_team_based

        # Player stats
        self._player_stats: Dict[str, PlayerStats] = {}

        # Team stats
        self._team_stats: Dict[str, TeamStats] = {}

        # Event history
        self._event_history: List[ScoreEvent] = []
        self._max_history_size: int = MAX_SCORING_HISTORY_SIZE

        # Game state
        self._first_blood_awarded: bool = False
        self._match_start_time: Optional[float] = None

        # Event handlers
        self._on_score_changed: List[Callable[[str, int, int], None]] = []
        self._on_kill: List[Callable[[str, str, Dict[str, Any]], None]] = []
        self._on_killstreak: List[Callable[[str, int], None]] = []
        self._on_multi_kill: List[Callable[[str, int], None]] = []
        self._on_first_blood: List[Callable[[str, str], None]] = []

    @property
    def config(self) -> ScoringConfig:
        """Get scoring configuration."""
        return self._config

    @property
    def is_team_based(self) -> bool:
        """Whether this is a team-based game."""
        return self._is_team_based

    @property
    def first_blood_awarded(self) -> bool:
        """Whether first blood has been awarded."""
        return self._first_blood_awarded

    # =========================================================================
    # PLAYER MANAGEMENT
    # =========================================================================

    def add_player(
        self,
        player_id: str,
        team_id: Optional[str] = None,
    ) -> PlayerStats:
        """
        Add a player to the scoring system.

        Args:
            player_id: Unique player identifier
            team_id: Optional team assignment

        Returns:
            PlayerStats for the new player
        """
        if player_id in self._player_stats:
            return self._player_stats[player_id]

        stats = PlayerStats(player_id=player_id, team_id=team_id)
        self._player_stats[player_id] = stats

        # Add to team if specified
        if team_id:
            self._ensure_team(team_id)
            self._team_stats[team_id].members.add(player_id)

        return stats

    def remove_player(self, player_id: str) -> bool:
        """
        Remove a player from the scoring system.

        Args:
            player_id: Player to remove

        Returns:
            True if player was removed
        """
        stats = self._player_stats.pop(player_id, None)
        if stats:
            # Remove from team
            if stats.team_id and stats.team_id in self._team_stats:
                self._team_stats[stats.team_id].members.discard(player_id)
            return True
        return False

    def get_player_stats(self, player_id: str) -> Optional[PlayerStats]:
        """Get stats for a player."""
        return self._player_stats.get(player_id)

    def set_player_team(self, player_id: str, team_id: Optional[str]) -> bool:
        """
        Set or change a player's team.

        Args:
            player_id: Player to update
            team_id: New team (None for no team)

        Returns:
            True if team was set
        """
        stats = self._player_stats.get(player_id)
        if not stats:
            return False

        # Remove from old team
        if stats.team_id and stats.team_id in self._team_stats:
            self._team_stats[stats.team_id].members.discard(player_id)

        # Add to new team
        stats.team_id = team_id
        if team_id:
            self._ensure_team(team_id)
            self._team_stats[team_id].members.add(player_id)

        return True

    def _ensure_team(self, team_id: str) -> TeamStats:
        """Ensure team exists and return its stats."""
        if team_id not in self._team_stats:
            self._team_stats[team_id] = TeamStats(team_id=team_id)
        return self._team_stats[team_id]

    # =========================================================================
    # SCORING
    # =========================================================================

    def add_score(
        self,
        player_id: str,
        points: int,
        event_type: ScoreEventType = ScoreEventType.BONUS,
        target_id: Optional[str] = None,
        **metadata: Any,
    ) -> bool:
        """
        Add score to a player.

        Args:
            player_id: Player to score
            points: Points to add (can be negative)
            event_type: Type of scoring event
            target_id: Target of the action (if applicable)
            **metadata: Additional event metadata

        Returns:
            True if score was added
        """
        stats = self._player_stats.get(player_id)
        if not stats:
            return False

        old_score = stats.score
        stats.score += points

        # Update team score
        if stats.team_id and stats.team_id in self._team_stats:
            self._team_stats[stats.team_id].score += points

        # Record event
        event = ScoreEvent(
            event_type=event_type,
            player_id=player_id,
            points=points,
            team_id=stats.team_id,
            target_id=target_id,
            metadata=metadata,
        )
        self._record_event(event)

        # Emit callback
        for handler in self._on_score_changed:
            try:
                handler(player_id, old_score, stats.score)
            except Exception:
                pass

        return True

    def set_score(self, player_id: str, score: int) -> bool:
        """
        Set a player's score directly.

        Args:
            player_id: Player to update
            score: New score value

        Returns:
            True if score was set
        """
        stats = self._player_stats.get(player_id)
        if not stats:
            return False

        old_score = stats.score
        diff = score - old_score
        stats.score = score

        # Update team score
        if stats.team_id and stats.team_id in self._team_stats:
            self._team_stats[stats.team_id].score += diff

        return True

    def get_score(self, player_id: str) -> int:
        """Get a player's score."""
        stats = self._player_stats.get(player_id)
        return stats.score if stats else 0

    def get_team_score(self, team_id: str) -> int:
        """Get a team's score."""
        stats = self._team_stats.get(team_id)
        return stats.score if stats else 0

    # =========================================================================
    # KILL TRACKING
    # =========================================================================

    def record_kill(
        self,
        killer_id: str,
        victim_id: str,
        is_headshot: bool = False,
        weapon: Optional[str] = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        """
        Record a kill and calculate all resulting score.

        Args:
            killer_id: Player who got the kill
            victim_id: Player who died
            is_headshot: Whether it was a headshot
            weapon: Weapon used
            **metadata: Additional metadata

        Returns:
            Dictionary with kill details and awards
        """
        result = {
            "kill_awarded": False,
            "assists": [],
            "points_awarded": {},
            "achievements": [],
        }

        killer_stats = self._player_stats.get(killer_id)
        victim_stats = self._player_stats.get(victim_id)

        if not killer_stats or not victim_stats:
            return result

        # Award kill points
        kill_points = self._config.points_per_kill
        killer_stats.kills += 1
        result["kill_awarded"] = True
        result["points_awarded"][killer_id] = kill_points

        # Update team kills
        if killer_stats.team_id and killer_stats.team_id in self._team_stats:
            self._team_stats[killer_stats.team_id].kills += 1

        # Record death
        victim_stats.deaths += 1
        victim_stats.current_killstreak = 0
        victim_stats.current_deathstreak += 1

        # Update team deaths
        if victim_stats.team_id and victim_stats.team_id in self._team_stats:
            self._team_stats[victim_stats.team_id].deaths += 1

        # Check for first blood
        if not self._first_blood_awarded:
            self._first_blood_awarded = True
            kill_points += self._config.points_per_first_blood
            killer_stats.first_bloods += 1
            result["achievements"].append("first_blood")
            for handler in self._on_first_blood:
                try:
                    handler(killer_id, victim_id)
                except Exception:
                    pass

        # Check for headshot
        if is_headshot:
            kill_points += self._config.points_per_headshot_bonus
            killer_stats.headshots += 1
            result["achievements"].append("headshot")

        # Check for revenge kill
        # Revenge = victim killed killer previously, now killer avenges
        if victim_id in killer_stats._killed_by:
            kill_points += self._config.points_per_revenge_kill
            killer_stats.revenge_kills += 1
            killer_stats._killed_by.discard(victim_id)
            result["achievements"].append("revenge")

        # Record who killed the victim
        victim_stats._killed_by.add(killer_id)

        # Update killstreak
        killer_stats.current_killstreak += 1
        killer_stats.current_deathstreak = 0

        if killer_stats.current_killstreak > killer_stats.best_killstreak:
            killer_stats.best_killstreak = killer_stats.current_killstreak

        # Check for killstreak bonus
        streak = killer_stats.current_killstreak
        if streak in KILLSTREAK_THRESHOLDS:
            bonus = self._config.points_per_killstreak_bonus * streak
            kill_points += bonus
            result["achievements"].append(KILLSTREAK_THRESHOLDS[streak])

            for handler in self._on_killstreak:
                try:
                    handler(killer_id, streak)
                except Exception:
                    pass

        # Check for multi-kill
        current_time = time.time()
        time_since_last = current_time - killer_stats._last_kill_time

        if time_since_last <= self._config.multi_kill_window:
            killer_stats._multi_kill_count += 1
            multi = killer_stats._multi_kill_count

            # Update multi-kill counters
            if multi == 2:
                killer_stats.double_kills += 1
            elif multi == 3:
                killer_stats.triple_kills += 1
            elif multi == 4:
                killer_stats.quad_kills += 1
            elif multi >= 5:
                killer_stats.penta_kills += 1

            if multi in MULTI_KILL_NAMES:
                result["achievements"].append(MULTI_KILL_NAMES[multi])

                for handler in self._on_multi_kill:
                    try:
                        handler(killer_id, multi)
                    except Exception:
                        pass
        else:
            killer_stats._multi_kill_count = 1

        killer_stats._last_kill_time = current_time

        # Process assists
        assists = self._calculate_assists(
            killer_id, victim_id, victim_stats
        )
        for assist_id, assist_damage in assists:
            assist_stats = self._player_stats.get(assist_id)
            if assist_stats:
                assist_stats.assists += 1
                assist_points = self._config.points_per_assist
                self.add_score(
                    assist_id,
                    assist_points,
                    ScoreEventType.ASSIST,
                    target_id=victim_id,
                )
                result["assists"].append(assist_id)
                result["points_awarded"][assist_id] = assist_points

                # Update team assists
                if assist_stats.team_id in self._team_stats:
                    self._team_stats[assist_stats.team_id].assists += 1

        # Clear damage tracking for victim
        for stats in self._player_stats.values():
            stats.clear_damage_tracking(victim_id)

        # Award kill score
        self.add_score(
            killer_id,
            kill_points,
            ScoreEventType.KILL,
            target_id=victim_id,
            headshot=is_headshot,
            weapon=weapon,
            **metadata,
        )

        result["points_awarded"][killer_id] = kill_points

        # Emit kill callback
        for handler in self._on_kill:
            try:
                handler(killer_id, victim_id, result)
            except Exception:
                pass

        return result

    def record_death(
        self,
        victim_id: str,
        death_points: Optional[int] = None,
        **metadata: Any,
    ) -> bool:
        """
        Record a death (without a killer - suicide/environment).

        Args:
            victim_id: Player who died
            death_points: Points to deduct (uses config default if None)
            **metadata: Additional metadata

        Returns:
            True if death was recorded
        """
        stats = self._player_stats.get(victim_id)
        if not stats:
            return False

        stats.deaths += 1
        stats.current_killstreak = 0
        stats.current_deathstreak += 1
        stats._multi_kill_count = 0

        # Update team deaths
        if stats.team_id and stats.team_id in self._team_stats:
            self._team_stats[stats.team_id].deaths += 1

        # Apply death penalty
        points = death_points if death_points is not None else self._config.points_per_death
        if points != 0:
            self.add_score(
                victim_id,
                points,
                ScoreEventType.DEATH,
                **metadata,
            )

        # Clear damage tracking
        for player_stats in self._player_stats.values():
            player_stats.clear_damage_tracking(victim_id)

        return True

    def _calculate_assists(
        self,
        killer_id: str,
        victim_id: str,
        victim_stats: PlayerStats,
    ) -> List[Tuple[str, float]]:
        """Calculate assist credits for a kill."""
        assists = []
        threshold = self._config.assist_damage_threshold

        for player_id, stats in self._player_stats.items():
            if player_id == killer_id:
                continue

            damage = stats.get_assist_damage(
                victim_id,
                max_health=DEFAULT_MAX_HEALTH,
                time_window=self._config.assist_time_window,
            )

            if damage and damage >= threshold * 100.0:  # threshold is percentage
                assists.append((player_id, damage))

        return assists

    # =========================================================================
    # OBJECTIVES
    # =========================================================================

    def record_objective_capture(
        self,
        player_id: str,
        objective_id: Optional[str] = None,
        points: Optional[int] = None,
        **metadata: Any,
    ) -> bool:
        """
        Record an objective capture.

        Args:
            player_id: Player who captured
            objective_id: Objective identifier
            points: Points to award (uses config default if None)

        Returns:
            True if objective was recorded
        """
        stats = self._player_stats.get(player_id)
        if not stats:
            return False

        stats.objectives_captured += 1

        # Update team objectives
        if stats.team_id and stats.team_id in self._team_stats:
            self._team_stats[stats.team_id].objectives += 1

        points = points if points is not None else self._config.points_per_objective
        self.add_score(
            player_id,
            points,
            ScoreEventType.OBJECTIVE_CAPTURE,
            objective_id=objective_id,
            **metadata,
        )

        return True

    def record_objective_defend(
        self,
        player_id: str,
        objective_id: Optional[str] = None,
        points: Optional[int] = None,
        **metadata: Any,
    ) -> bool:
        """
        Record an objective defense.

        Args:
            player_id: Player who defended
            objective_id: Objective identifier
            points: Points to award

        Returns:
            True if defense was recorded
        """
        stats = self._player_stats.get(player_id)
        if not stats:
            return False

        stats.objectives_defended += 1

        points = points if points is not None else self._config.points_per_objective // 2
        self.add_score(
            player_id,
            points,
            ScoreEventType.OBJECTIVE_DEFEND,
            objective_id=objective_id,
            **metadata,
        )

        return True

    # =========================================================================
    # DAMAGE TRACKING
    # =========================================================================

    def record_damage(
        self,
        attacker_id: str,
        victim_id: str,
        amount: float,
        max_health: float = 100.0,
    ) -> bool:
        """
        Record damage dealt for assist tracking.

        Args:
            attacker_id: Player dealing damage
            victim_id: Player receiving damage
            amount: Damage amount
            max_health: Victim's max health (for threshold calculation)

        Returns:
            True if damage was recorded
        """
        stats = self._player_stats.get(attacker_id)
        if not stats:
            return False

        stats.record_damage_dealt(victim_id, amount, max_health)

        victim_stats = self._player_stats.get(victim_id)
        if victim_stats:
            victim_stats.damage_taken += amount

        return True

    def record_healing(
        self,
        healer_id: str,
        target_id: str,
        amount: float,
    ) -> bool:
        """
        Record healing done.

        Args:
            healer_id: Player doing healing
            target_id: Player being healed
            amount: Healing amount

        Returns:
            True if healing was recorded
        """
        stats = self._player_stats.get(healer_id)
        if not stats:
            return False

        stats.healing_done += amount
        return True

    # =========================================================================
    # LEADERBOARD
    # =========================================================================

    def get_leaderboard(
        self,
        sort_by: LeaderboardSortKey = LeaderboardSortKey.SCORE,
        limit: Optional[int] = None,
        team_id: Optional[str] = None,
    ) -> List[LeaderboardEntry]:
        """
        Get the current leaderboard.

        Args:
            sort_by: Key to sort by
            limit: Maximum entries to return
            team_id: Filter to specific team

        Returns:
            Sorted list of LeaderboardEntry
        """
        # Gather stats
        entries = []
        for stats in self._player_stats.values():
            if team_id and stats.team_id != team_id:
                continue

            entries.append(LeaderboardEntry(
                rank=0,  # Will be set after sorting
                player_id=stats.player_id,
                score=stats.score,
                kills=stats.kills,
                deaths=stats.deaths,
                assists=stats.assists,
                team_id=stats.team_id,
            ))

        # Sort
        if sort_by == LeaderboardSortKey.SCORE:
            entries.sort(key=lambda e: e.score, reverse=True)
        elif sort_by == LeaderboardSortKey.KILLS:
            entries.sort(key=lambda e: e.kills, reverse=True)
        elif sort_by == LeaderboardSortKey.DEATHS:
            entries.sort(key=lambda e: e.deaths, reverse=False)  # Fewer deaths = better
        elif sort_by == LeaderboardSortKey.ASSISTS:
            entries.sort(key=lambda e: e.assists, reverse=True)
        elif sort_by == LeaderboardSortKey.KD_RATIO:
            entries.sort(key=lambda e: e.kd_ratio, reverse=True)
        elif sort_by == LeaderboardSortKey.KDA_RATIO:
            entries.sort(
                key=lambda e: (e.kills + e.assists) / max(1, e.deaths),
                reverse=True,
            )

        # Set ranks
        for i, entry in enumerate(entries):
            entry.rank = i + 1

        # Apply limit
        if limit:
            entries = entries[:limit]

        return entries

    def get_team_leaderboard(
        self,
        sort_by: LeaderboardSortKey = LeaderboardSortKey.SCORE,
    ) -> List[Tuple[str, int]]:
        """
        Get team leaderboard.

        Args:
            sort_by: Key to sort by

        Returns:
            List of (team_id, score) tuples sorted by score
        """
        teams = []
        for team_id, stats in self._team_stats.items():
            if sort_by == LeaderboardSortKey.SCORE:
                value = stats.score
            elif sort_by == LeaderboardSortKey.KILLS:
                value = stats.kills
            else:
                value = stats.score
            teams.append((team_id, value))

        teams.sort(key=lambda t: t[1], reverse=True)
        return teams

    def get_player_rank(self, player_id: str) -> int:
        """
        Get a player's current rank.

        Args:
            player_id: Player to check

        Returns:
            Rank (1-based), or 0 if not found
        """
        leaderboard = self.get_leaderboard()
        for entry in leaderboard:
            if entry.player_id == player_id:
                return entry.rank
        return 0

    # =========================================================================
    # EVENT HISTORY
    # =========================================================================

    def _record_event(self, event: ScoreEvent) -> None:
        """Record a score event to history."""
        self._event_history.append(event)

        # Trim history if needed
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]

    def get_event_history(
        self,
        player_id: Optional[str] = None,
        event_type: Optional[ScoreEventType] = None,
        limit: int = 100,
    ) -> List[ScoreEvent]:
        """
        Get score event history.

        Args:
            player_id: Filter to specific player
            event_type: Filter to specific event type
            limit: Maximum events to return

        Returns:
            List of ScoreEvent
        """
        events = self._event_history

        if player_id:
            events = [e for e in events if e.player_id == player_id]

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_recent_kills(
        self,
        limit: int = 10,
    ) -> List[ScoreEvent]:
        """Get recent kill events."""
        return self.get_event_history(event_type=ScoreEventType.KILL, limit=limit)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_score_changed(
        self,
        handler: Callable[[str, int, int], None],
    ) -> None:
        """Register handler for score changes (player_id, old, new)."""
        self._on_score_changed.append(handler)

    def on_kill(
        self,
        handler: Callable[[str, str, Dict[str, Any]], None],
    ) -> None:
        """Register handler for kills (killer, victim, details)."""
        self._on_kill.append(handler)

    def on_killstreak(
        self,
        handler: Callable[[str, int], None],
    ) -> None:
        """Register handler for killstreaks (player, streak)."""
        self._on_killstreak.append(handler)

    def on_multi_kill(
        self,
        handler: Callable[[str, int], None],
    ) -> None:
        """Register handler for multi-kills (player, count)."""
        self._on_multi_kill.append(handler)

    def on_first_blood(
        self,
        handler: Callable[[str, str], None],
    ) -> None:
        """Register handler for first blood (killer, victim)."""
        self._on_first_blood.append(handler)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def start_match(self) -> None:
        """Mark the start of a match."""
        self._match_start_time = time.time()
        self._first_blood_awarded = False

    def reset(self) -> None:
        """Reset all scoring data."""
        self._player_stats.clear()
        self._team_stats.clear()
        self._event_history.clear()
        self._first_blood_awarded = False
        self._match_start_time = None

    def get_summary(self) -> Dict[str, Any]:
        """Get scoring summary."""
        return {
            "player_count": len(self._player_stats),
            "team_count": len(self._team_stats),
            "total_kills": sum(s.kills for s in self._player_stats.values()),
            "total_deaths": sum(s.deaths for s in self._player_stats.values()),
            "total_assists": sum(s.assists for s in self._player_stats.values()),
            "first_blood_awarded": self._first_blood_awarded,
            "event_count": len(self._event_history),
            "leaderboard": [
                e.to_dict() if hasattr(e, 'to_dict') else {
                    "rank": e.rank,
                    "player_id": e.player_id,
                    "score": e.score,
                }
                for e in self.get_leaderboard(limit=10)
            ],
        }

    def get_all_player_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all player stats as dictionaries."""
        return {
            player_id: stats.to_dict()
            for player_id, stats in self._player_stats.items()
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ScoreEventType",
    "LeaderboardSortKey",
    # Data classes
    "PlayerStats",
    "TeamStats",
    "ScoreEvent",
    "LeaderboardEntry",
    # System
    "ScoringSystem",
]
