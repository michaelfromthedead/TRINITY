"""
Free-for-all Deathmatch game mode.

Every player for themselves. Win by reaching the kill limit first
or having the highest score when time expires.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import random

from engine.gameplay.combat.game_mode import (
    GameMode,
    GameModeConfig,
    GameModeRules,
    WinCondition,
    WinConditionType,
    ScoringEvent,
    ScoringEventType,
)
from engine.gameplay.combat.spawn_manager import SpawnManager, SpawnRule, SpawnRuleType


@dataclass
class DeathmatchConfig:
    """Configuration specific to Deathmatch mode."""
    kill_limit: int = 25
    time_limit_seconds: float = 600.0  # 10 minutes
    respawn_delay_seconds: float = 3.0
    spawn_protection_seconds: float = 2.0
    kill_points: int = 1
    death_points: int = 0
    suicide_penalty: int = -1
    assist_points: int = 0
    killstreak_bonuses: Dict[int, int] = field(default_factory=dict)
    multi_kill_bonuses: Dict[int, int] = field(default_factory=dict)


class Deathmatch(GameMode):
    """
    Free-for-all Deathmatch game mode.

    Features:
    - Individual scoring (no teams)
    - Kill-based scoring
    - Optional killstreak and multi-kill bonuses
    - First to kill limit or highest score at time limit wins
    """

    def __init__(
        self,
        dm_config: Optional[DeathmatchConfig] = None,
        spawn_manager: Optional[SpawnManager] = None,
    ):
        """
        Initialize Deathmatch mode.

        Args:
            dm_config: Deathmatch-specific configuration
            spawn_manager: Spawn manager instance
        """
        self.dm_config = dm_config or DeathmatchConfig()

        # Build game mode config
        rules = GameModeRules(
            friendly_fire=True,  # FFA - everyone is an enemy
            respawn_enabled=True,
            respawn_delay_seconds=self.dm_config.respawn_delay_seconds,
            max_respawns=None,
            min_players=2,
            max_players=16,
            min_teams=0,
            max_teams=0,
            auto_balance_teams=False,
        )

        win_conditions = [
            WinCondition(
                condition_type=WinConditionType.SCORE_LIMIT,
                target_value=self.dm_config.kill_limit,
                description=f"First to {self.dm_config.kill_limit} kills"
            ),
            WinCondition(
                condition_type=WinConditionType.TIME_LIMIT,
                time_limit_seconds=self.dm_config.time_limit_seconds,
                description="Highest score when time expires"
            ),
        ]

        scoring_values = {
            ScoringEventType.KILL: self.dm_config.kill_points,
            ScoringEventType.DEATH: self.dm_config.death_points,
            ScoringEventType.ASSIST: self.dm_config.assist_points,
            ScoringEventType.PENALTY: self.dm_config.suicide_penalty,
        }

        config = GameModeConfig(
            mode_id="deathmatch",
            mode_name="Deathmatch",
            description="Free-for-all. First to the kill limit wins.",
            rules=rules,
            win_conditions=win_conditions,
            scoring_values=scoring_values,
            time_limit_seconds=self.dm_config.time_limit_seconds,
            score_limit=self.dm_config.kill_limit,
            is_team_based=False,
        )

        super().__init__(config)

        # Spawn manager
        self.spawn_manager = spawn_manager or SpawnManager(
            default_rule=SpawnRule(
                rule_type=SpawnRuleType.DISTANCE_BASED,
                respawn_delay_seconds=self.dm_config.respawn_delay_seconds,
                spawn_protection_seconds=self.dm_config.spawn_protection_seconds,
                min_distance_from_enemies=20.0,
            )
        )

        # Killstreak tracking
        self._killstreaks: Dict[str, int] = {}
        self._last_kill_time: Dict[str, float] = {}
        self._multi_kill_count: Dict[str, int] = {}
        self._multi_kill_window: float = 4.0  # seconds

        # Default bonuses
        if not self.dm_config.killstreak_bonuses:
            self.dm_config.killstreak_bonuses = {
                5: 1,   # +1 for 5 kills
                10: 2,  # +2 for 10 kills
                15: 3,  # +3 for 15 kills
                20: 5,  # +5 for 20 kills
            }

        if not self.dm_config.multi_kill_bonuses:
            self.dm_config.multi_kill_bonuses = {
                2: 1,  # Double kill
                3: 2,  # Triple kill
                4: 3,  # Quad kill
                5: 5,  # Pentakill
            }

    # =========================================================================
    # GameMode Implementation
    # =========================================================================

    def get_spawn_location(self, player_id: str) -> Tuple[float, float, float]:
        """Get spawn location for a player."""
        result = self.spawn_manager.spawn_player(player_id)
        if result:
            return result[0]

        # Fallback to random location if spawn manager fails
        return (
            random.uniform(-50, 50),
            0.0,
            random.uniform(-50, 50)
        )

    def on_player_killed(
        self,
        victim_id: str,
        killer_id: Optional[str] = None,
        weapon: Optional[str] = None,
        assists: Optional[List[str]] = None
    ) -> None:
        """Handle player death event."""
        import time

        # Mark victim as dead
        self.mark_player_dead(victim_id)

        # Reset victim's killstreak
        self._killstreaks[victim_id] = 0
        self._multi_kill_count[victim_id] = 0

        if killer_id and killer_id != victim_id:
            # Valid kill
            self.add_score(
                killer_id,
                ScoringEventType.KILL,
                metadata={"weapon": weapon, "victim": victim_id}
            )

            # Update killstreak
            self._killstreaks[killer_id] = self._killstreaks.get(killer_id, 0) + 1
            streak = self._killstreaks[killer_id]

            # Check for killstreak bonus
            if streak in self.dm_config.killstreak_bonuses:
                bonus = self.dm_config.killstreak_bonuses[streak]
                self.add_score(
                    killer_id,
                    ScoringEventType.BONUS,
                    points=bonus,
                    metadata={"reason": f"killstreak_{streak}"}
                )

            # Check for multi-kill
            current_time = time.time()
            last_kill = self._last_kill_time.get(killer_id, 0)

            if current_time - last_kill <= self._multi_kill_window:
                self._multi_kill_count[killer_id] = self._multi_kill_count.get(killer_id, 0) + 1
                multi = self._multi_kill_count[killer_id]

                if multi in self.dm_config.multi_kill_bonuses:
                    bonus = self.dm_config.multi_kill_bonuses[multi]
                    self.add_score(
                        killer_id,
                        ScoringEventType.BONUS,
                        points=bonus,
                        metadata={"reason": f"multi_kill_{multi}"}
                    )
            else:
                self._multi_kill_count[killer_id] = 1

            self._last_kill_time[killer_id] = current_time

            # Process assists
            if assists:
                for assist_id in assists:
                    if assist_id != killer_id:
                        self.add_score(
                            assist_id,
                            ScoringEventType.ASSIST,
                            metadata={"victim": victim_id}
                        )

        elif killer_id == victim_id:
            # Suicide
            self.add_score(
                victim_id,
                ScoringEventType.PENALTY,
                metadata={"reason": "suicide"}
            )
        else:
            # Environmental death
            self.add_score(
                victim_id,
                ScoringEventType.DEATH,
                metadata={"reason": "environment"}
            )

        # Schedule respawn
        self.spawn_manager.schedule_respawn(victim_id)

    def check_win_condition(self) -> Tuple[bool, Optional[str]]:
        """Check if any win condition is met."""
        for condition in self.config.win_conditions:
            is_met, winner = condition.is_met(self)
            if is_met:
                return (True, winner)
        return (False, None)

    # =========================================================================
    # Deathmatch-Specific Methods
    # =========================================================================

    def get_killstreak(self, player_id: str) -> int:
        """Get a player's current killstreak."""
        return self._killstreaks.get(player_id, 0)

    def get_multi_kill_count(self, player_id: str) -> int:
        """Get a player's current multi-kill count."""
        return self._multi_kill_count.get(player_id, 0)

    def get_rankings(self) -> List[Tuple[str, int, int, int]]:
        """
        Get player rankings.

        Returns:
            List of (player_id, score, kills, deaths) sorted by score
        """
        rankings = []
        for player_id in self.players:
            score = self.player_scores.get(player_id, 0)
            deaths = self.player_deaths.get(player_id, 0)
            # Kills can be approximated from score if kill_points is 1
            kills = score  # Simplified; real implementation would track separately
            rankings.append((player_id, score, kills, deaths))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_player_kd_ratio(self, player_id: str) -> float:
        """Get a player's kill/death ratio."""
        deaths = self.player_deaths.get(player_id, 0)
        score = self.player_scores.get(player_id, 0)
        if deaths == 0:
            return float(score)
        return score / deaths

    # =========================================================================
    # Player Management Overrides
    # =========================================================================

    def add_player(self, player_id: str, team_id: Optional[str] = None) -> bool:
        """Add a player to the game."""
        if super().add_player(player_id, team_id):
            self._killstreaks[player_id] = 0
            self._multi_kill_count[player_id] = 0
            return True
        return False

    def remove_player(self, player_id: str) -> bool:
        """Remove a player from the game."""
        if super().remove_player(player_id):
            self._killstreaks.pop(player_id, None)
            self._multi_kill_count.pop(player_id, None)
            self._last_kill_time.pop(player_id, None)
            self.spawn_manager.remove_player(player_id)
            return True
        return False

    def reset(self) -> None:
        """Reset the game mode."""
        super().reset()
        self._killstreaks.clear()
        self._multi_kill_count.clear()
        self._last_kill_time.clear()
        self.spawn_manager.reset()

    def respawn_player(self, player_id: str) -> bool:
        """Respawn a player."""
        if not self.spawn_manager.is_respawn_ready(player_id):
            return False
        return super().respawn_player(player_id)

    def update(self, delta_time: float) -> None:
        """Update deathmatch state."""
        self.spawn_manager.update()

    def get_stats(self) -> Dict[str, Any]:
        """Get deathmatch statistics."""
        stats = super().get_stats()
        stats.update({
            "kill_limit": self.dm_config.kill_limit,
            "highest_killstreak": max(self._killstreaks.values()) if self._killstreaks else 0,
            "rankings": self.get_rankings(),
        })
        return stats
