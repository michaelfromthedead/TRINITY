"""
Spawn point management and team-based spawning.

Provides:
- SpawnPoint: Individual spawn location with properties
- SpawnRule: Rules for spawn selection and timing
- SpawnManager: Centralized spawn management
- Team spawning configuration
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import random
import time
import math

from .constants import (
    SPAWN_TEAM_BONUS_SCORE,
    SPAWN_TIME_FRESHNESS_MAX_BONUS,
    SPAWN_TIME_FRESHNESS_DIVISOR,
    SPAWN_DISTANCE_MAX_BONUS,
    SPAWN_DISTANCE_DIVISOR,
)


class SpawnRuleType(Enum):
    """Types of spawn selection rules."""
    RANDOM = auto()           # Random spawn point selection
    SEQUENTIAL = auto()       # Round-robin through spawn points
    TEAM_BASED = auto()       # Spawn at team-owned points
    DISTANCE_BASED = auto()   # Spawn far from enemies
    OBJECTIVE_BASED = auto()  # Spawn near objectives
    SAFE_SPAWN = auto()       # Spawn at safest location
    FIXED = auto()            # Always spawn at specific point


class SpawnPointState(Enum):
    """State of a spawn point."""
    AVAILABLE = auto()
    OCCUPIED = auto()
    BLOCKED = auto()
    DISABLED = auto()
    COOLDOWN = auto()


@dataclass
class SpawnPoint:
    """
    Represents a spawn location in the game world.

    Attributes:
        point_id: Unique identifier for this spawn point
        position: World position (x, y, z)
        rotation: Spawn rotation in degrees (yaw, pitch, roll)
        team_id: Team that owns this spawn point (None for neutral)
        spawn_type: Type categorization (e.g., "infantry", "vehicle", "initial")
        priority: Higher priority points are preferred (0-100)
        capacity: How many players can spawn here simultaneously
        tags: Custom tags for filtering
        enabled: Whether this point is currently active
    """
    point_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    team_id: Optional[str] = None
    spawn_type: str = "default"
    priority: int = 50
    capacity: int = 1
    tags: Set[str] = field(default_factory=set)
    enabled: bool = True

    # Runtime state
    _state: SpawnPointState = SpawnPointState.AVAILABLE
    _last_used: float = 0.0
    _current_occupants: int = 0
    _cooldown_until: float = 0.0

    @property
    def is_available(self) -> bool:
        """Check if spawn point is available for use."""
        if not self.enabled:
            return False
        if self._state != SpawnPointState.AVAILABLE:
            if self._state == SpawnPointState.COOLDOWN:
                if time.time() >= self._cooldown_until:
                    self._state = SpawnPointState.AVAILABLE
                else:
                    return False
            else:
                return False
        return self._current_occupants < self.capacity

    def use(self, cooldown_seconds: float = 0.0) -> bool:
        """
        Mark spawn point as used.

        Args:
            cooldown_seconds: Time before point can be used again

        Returns:
            True if spawn point was successfully used
        """
        if not self.is_available:
            return False

        self._current_occupants += 1
        self._last_used = time.time()

        if cooldown_seconds > 0:
            self._cooldown_until = time.time() + cooldown_seconds
            if self._current_occupants >= self.capacity:
                self._state = SpawnPointState.COOLDOWN

        return True

    def release(self) -> None:
        """Release occupancy of spawn point."""
        self._current_occupants = max(0, self._current_occupants - 1)
        if self._state == SpawnPointState.OCCUPIED and self._current_occupants == 0:
            self._state = SpawnPointState.AVAILABLE

    def block(self) -> None:
        """Block spawn point from being used."""
        self._state = SpawnPointState.BLOCKED

    def unblock(self) -> None:
        """Unblock spawn point."""
        if self._state == SpawnPointState.BLOCKED:
            self._state = SpawnPointState.AVAILABLE

    def disable(self) -> None:
        """Disable spawn point."""
        self.enabled = False
        self._state = SpawnPointState.DISABLED

    def enable(self) -> None:
        """Enable spawn point."""
        self.enabled = True
        if self._state == SpawnPointState.DISABLED:
            self._state = SpawnPointState.AVAILABLE


@dataclass
class SpawnRule:
    """
    Defines rules for spawn point selection.

    Attributes:
        rule_type: The type of spawn selection to use
        respawn_delay_seconds: Time before respawn is allowed
        spawn_protection_seconds: Invulnerability time after spawn
        cooldown_seconds: Time before same spawn point can be used again
        min_distance_from_enemies: Minimum distance from enemies for spawn
        max_distance_from_objective: Maximum distance from objective
        prefer_team_spawns: Prefer team-owned spawn points
        avoid_recently_used: Avoid recently used spawn points
        random_offset_radius: Random position offset from spawn point
        require_line_of_sight: Require clear LOS to spawn
        custom_validator: Custom function to validate spawn point
    """
    rule_type: SpawnRuleType = SpawnRuleType.RANDOM
    respawn_delay_seconds: float = 3.0
    spawn_protection_seconds: float = 2.0
    cooldown_seconds: float = 5.0
    min_distance_from_enemies: float = 0.0
    max_distance_from_objective: Optional[float] = None
    prefer_team_spawns: bool = True
    avoid_recently_used: bool = True
    random_offset_radius: float = 0.0
    require_line_of_sight: bool = False
    custom_validator: Optional[Callable[['SpawnPoint', str], bool]] = None


@dataclass
class TeamSpawnConfig:
    """
    Configuration for team-based spawning.

    Attributes:
        team_id: The team this config applies to
        spawn_points: List of spawn point IDs for this team
        spawn_rule: Spawn rule for this team
        initial_spawns: Spawn points to use at match start
        rally_point_enabled: Whether rally points are enabled
        rally_point_cooldown: Cooldown for rally point spawning
    """
    team_id: str
    spawn_points: List[str] = field(default_factory=list)
    spawn_rule: Optional[SpawnRule] = None
    initial_spawns: List[str] = field(default_factory=list)
    rally_point_enabled: bool = False
    rally_point_cooldown: float = 30.0
    rally_point_position: Optional[Tuple[float, float, float]] = None


class SpawnManager:
    """
    Centralized spawn point management.

    Handles spawn point registration, selection, and team-based spawning.
    """

    def __init__(self, default_rule: Optional[SpawnRule] = None):
        """
        Initialize spawn manager.

        Args:
            default_rule: Default spawn rule to use
        """
        self.default_rule = default_rule or SpawnRule()

        # Spawn points
        self._spawn_points: Dict[str, SpawnPoint] = {}
        self._spawn_points_by_team: Dict[str, List[str]] = {}
        self._spawn_points_by_type: Dict[str, List[str]] = {}

        # Team configurations
        self._team_configs: Dict[str, TeamSpawnConfig] = {}

        # Sequential spawn tracking
        self._sequential_index: int = 0
        self._sequential_indices: Dict[str, int] = {}  # Per-team indices

        # Player tracking
        self._player_positions: Dict[str, Tuple[float, float, float]] = {}
        self._player_teams: Dict[str, str] = {}
        self._pending_respawns: Dict[str, float] = {}  # player_id -> respawn_time

        # Callbacks
        self._on_spawn: List[Callable[[str, SpawnPoint], None]] = []
        self._on_respawn_ready: List[Callable[[str], None]] = []

    # =========================================================================
    # Spawn Point Management
    # =========================================================================

    def register_spawn_point(self, spawn_point: SpawnPoint) -> bool:
        """
        Register a spawn point.

        Args:
            spawn_point: The spawn point to register

        Returns:
            True if registration was successful
        """
        if spawn_point.point_id in self._spawn_points:
            return False

        self._spawn_points[spawn_point.point_id] = spawn_point

        # Index by team
        if spawn_point.team_id:
            if spawn_point.team_id not in self._spawn_points_by_team:
                self._spawn_points_by_team[spawn_point.team_id] = []
            self._spawn_points_by_team[spawn_point.team_id].append(spawn_point.point_id)

        # Index by type
        if spawn_point.spawn_type not in self._spawn_points_by_type:
            self._spawn_points_by_type[spawn_point.spawn_type] = []
        self._spawn_points_by_type[spawn_point.spawn_type].append(spawn_point.point_id)

        return True

    def unregister_spawn_point(self, point_id: str) -> bool:
        """Remove a spawn point."""
        if point_id not in self._spawn_points:
            return False

        spawn_point = self._spawn_points.pop(point_id)

        # Remove from indices
        if spawn_point.team_id and spawn_point.team_id in self._spawn_points_by_team:
            self._spawn_points_by_team[spawn_point.team_id].remove(point_id)

        if spawn_point.spawn_type in self._spawn_points_by_type:
            self._spawn_points_by_type[spawn_point.spawn_type].remove(point_id)

        return True

    def get_spawn_point(self, point_id: str) -> Optional[SpawnPoint]:
        """Get a spawn point by ID."""
        return self._spawn_points.get(point_id)

    def get_all_spawn_points(self) -> List[SpawnPoint]:
        """Get all registered spawn points."""
        return list(self._spawn_points.values())

    def get_team_spawn_points(self, team_id: str) -> List[SpawnPoint]:
        """Get all spawn points for a team."""
        point_ids = self._spawn_points_by_team.get(team_id, [])
        return [self._spawn_points[pid] for pid in point_ids if pid in self._spawn_points]

    def get_spawn_points_by_type(self, spawn_type: str) -> List[SpawnPoint]:
        """Get all spawn points of a specific type."""
        point_ids = self._spawn_points_by_type.get(spawn_type, [])
        return [self._spawn_points[pid] for pid in point_ids if pid in self._spawn_points]

    def get_available_spawn_points(
        self,
        team_id: Optional[str] = None,
        spawn_type: Optional[str] = None,
        tags: Optional[Set[str]] = None
    ) -> List[SpawnPoint]:
        """
        Get available spawn points matching criteria.

        Args:
            team_id: Filter by team
            spawn_type: Filter by spawn type
            tags: Required tags

        Returns:
            List of available spawn points
        """
        points = []
        for point in self._spawn_points.values():
            if not point.is_available:
                continue
            if team_id and point.team_id != team_id:
                continue
            if spawn_type and point.spawn_type != spawn_type:
                continue
            if tags and not tags.issubset(point.tags):
                continue
            points.append(point)
        return points

    # =========================================================================
    # Team Configuration
    # =========================================================================

    def configure_team(self, config: TeamSpawnConfig) -> None:
        """Configure spawning for a team."""
        self._team_configs[config.team_id] = config
        self._sequential_indices[config.team_id] = 0

    def get_team_config(self, team_id: str) -> Optional[TeamSpawnConfig]:
        """Get spawn configuration for a team."""
        return self._team_configs.get(team_id)

    def set_rally_point(
        self,
        team_id: str,
        position: Tuple[float, float, float]
    ) -> bool:
        """
        Set rally point for a team.

        Args:
            team_id: The team
            position: Rally point position

        Returns:
            True if rally point was set
        """
        config = self._team_configs.get(team_id)
        if not config or not config.rally_point_enabled:
            return False

        config.rally_point_position = position
        return True

    # =========================================================================
    # Spawn Selection
    # =========================================================================

    def select_spawn_point(
        self,
        player_id: str,
        team_id: Optional[str] = None,
        spawn_type: Optional[str] = None,
        rule: Optional[SpawnRule] = None
    ) -> Optional[SpawnPoint]:
        """
        Select a spawn point for a player.

        Args:
            player_id: The player to spawn
            team_id: The player's team
            spawn_type: Required spawn type
            rule: Spawn rule to use (defaults to team or global rule)

        Returns:
            Selected spawn point or None if none available
        """
        # Determine spawn rule
        if rule is None:
            if team_id and team_id in self._team_configs:
                rule = self._team_configs[team_id].spawn_rule or self.default_rule
            else:
                rule = self.default_rule

        # Get candidate spawn points
        candidates = self._get_candidates(player_id, team_id, spawn_type, rule)
        if not candidates:
            return None

        # Select based on rule type
        selected = None
        if rule.rule_type == SpawnRuleType.RANDOM:
            selected = self._select_random(candidates, rule)
        elif rule.rule_type == SpawnRuleType.SEQUENTIAL:
            selected = self._select_sequential(candidates, team_id)
        elif rule.rule_type == SpawnRuleType.TEAM_BASED:
            selected = self._select_team_based(candidates, team_id, rule)
        elif rule.rule_type == SpawnRuleType.DISTANCE_BASED:
            selected = self._select_distance_based(candidates, player_id, team_id, rule)
        elif rule.rule_type == SpawnRuleType.SAFE_SPAWN:
            selected = self._select_safe_spawn(candidates, team_id, rule)
        elif rule.rule_type == SpawnRuleType.FIXED:
            selected = candidates[0] if candidates else None
        else:
            selected = self._select_random(candidates, rule)

        return selected

    def _get_candidates(
        self,
        player_id: str,
        team_id: Optional[str],
        spawn_type: Optional[str],
        rule: SpawnRule
    ) -> List[SpawnPoint]:
        """Get candidate spawn points based on criteria."""
        candidates = []

        for point in self._spawn_points.values():
            if not point.is_available:
                continue

            # Filter by type
            if spawn_type and point.spawn_type != spawn_type:
                continue

            # Filter by team preference
            if rule.prefer_team_spawns and team_id:
                if point.team_id is not None and point.team_id != team_id:
                    continue

            # Custom validator
            if rule.custom_validator and not rule.custom_validator(point, player_id):
                continue

            # Avoid recently used
            if rule.avoid_recently_used:
                if time.time() - point._last_used < rule.cooldown_seconds:
                    continue

            candidates.append(point)

        return candidates

    def _select_random(
        self,
        candidates: List[SpawnPoint],
        rule: SpawnRule
    ) -> Optional[SpawnPoint]:
        """Select random spawn point, weighted by priority."""
        if not candidates:
            return None

        # Weight by priority
        weights = [p.priority for p in candidates]
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(candidates)

        r = random.uniform(0, total_weight)
        cumulative = 0
        for i, point in enumerate(candidates):
            cumulative += weights[i]
            if r <= cumulative:
                return point

        return candidates[-1]

    def _select_sequential(
        self,
        candidates: List[SpawnPoint],
        team_id: Optional[str]
    ) -> Optional[SpawnPoint]:
        """Select spawn point sequentially (round-robin)."""
        if not candidates:
            return None

        # Sort by priority then ID for consistent ordering
        candidates = sorted(candidates, key=lambda p: (-p.priority, p.point_id))

        if team_id:
            idx = self._sequential_indices.get(team_id, 0)
            self._sequential_indices[team_id] = (idx + 1) % len(candidates)
        else:
            idx = self._sequential_index
            self._sequential_index = (idx + 1) % len(candidates)

        return candidates[idx % len(candidates)]

    def _select_team_based(
        self,
        candidates: List[SpawnPoint],
        team_id: Optional[str],
        rule: SpawnRule
    ) -> Optional[SpawnPoint]:
        """Select spawn point prioritizing team-owned points."""
        if not candidates:
            return None

        # Prefer team spawns
        if team_id:
            team_spawns = [p for p in candidates if p.team_id == team_id]
            if team_spawns:
                return self._select_random(team_spawns, rule)

        # Fall back to neutral spawns
        neutral_spawns = [p for p in candidates if p.team_id is None]
        if neutral_spawns:
            return self._select_random(neutral_spawns, rule)

        return self._select_random(candidates, rule)

    def _select_distance_based(
        self,
        candidates: List[SpawnPoint],
        player_id: str,
        team_id: Optional[str],
        rule: SpawnRule
    ) -> Optional[SpawnPoint]:
        """Select spawn point based on distance from enemies."""
        if not candidates:
            return None

        if rule.min_distance_from_enemies <= 0:
            return self._select_random(candidates, rule)

        # Get enemy positions
        enemy_positions = []
        for pid, pos in self._player_positions.items():
            if pid == player_id:
                continue
            if team_id and self._player_teams.get(pid) == team_id:
                continue
            enemy_positions.append(pos)

        if not enemy_positions:
            return self._select_random(candidates, rule)

        # Score by minimum distance to enemies
        scored = []
        for point in candidates:
            min_dist = float('inf')
            for enemy_pos in enemy_positions:
                dist = self._distance(point.position, enemy_pos)
                min_dist = min(min_dist, dist)

            if min_dist >= rule.min_distance_from_enemies:
                scored.append((point, min_dist))

        if not scored:
            # Fallback: select furthest from any enemy
            best = None
            best_dist = -1
            for point in candidates:
                min_dist = min(
                    self._distance(point.position, ep) for ep in enemy_positions
                )
                if min_dist > best_dist:
                    best = point
                    best_dist = min_dist
            return best

        # Prefer further spawns
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    def _select_safe_spawn(
        self,
        candidates: List[SpawnPoint],
        team_id: Optional[str],
        rule: SpawnRule
    ) -> Optional[SpawnPoint]:
        """Select safest spawn point (combination of factors)."""
        if not candidates:
            return None

        # Score each spawn point
        scored = []
        for point in candidates:
            score = point.priority

            # Bonus for team spawn
            if team_id and point.team_id == team_id:
                score += SPAWN_TEAM_BONUS_SCORE

            # Bonus for less recently used
            time_since_use = time.time() - point._last_used
            score += min(SPAWN_TIME_FRESHNESS_MAX_BONUS, time_since_use / SPAWN_TIME_FRESHNESS_DIVISOR)

            # Distance from enemies
            min_enemy_dist = float('inf')
            for pid, pos in self._player_positions.items():
                if team_id and self._player_teams.get(pid) == team_id:
                    continue
                dist = self._distance(point.position, pos)
                min_enemy_dist = min(min_enemy_dist, dist)

            if min_enemy_dist != float('inf'):
                # Normalize distance score
                score += min(SPAWN_DISTANCE_MAX_BONUS, min_enemy_dist / SPAWN_DISTANCE_DIVISOR)

            scored.append((point, score))

        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    @staticmethod
    def _distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between two points."""
        return math.sqrt(
            (p2[0] - p1[0]) ** 2 +
            (p2[1] - p1[1]) ** 2 +
            (p2[2] - p1[2]) ** 2
        )

    # =========================================================================
    # Spawning
    # =========================================================================

    def spawn_player(
        self,
        player_id: str,
        team_id: Optional[str] = None,
        spawn_type: Optional[str] = None,
        rule: Optional[SpawnRule] = None
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """
        Spawn a player and return spawn location.

        Args:
            player_id: Player to spawn
            team_id: Player's team
            spawn_type: Required spawn type
            rule: Spawn rule to use

        Returns:
            Tuple of (position, rotation) or None if spawning failed
        """
        rule = rule or self.default_rule

        # Check for pending respawn delay
        if player_id in self._pending_respawns:
            if time.time() < self._pending_respawns[player_id]:
                return None
            del self._pending_respawns[player_id]

        # Select spawn point
        spawn_point = self.select_spawn_point(player_id, team_id, spawn_type, rule)
        if not spawn_point:
            return None

        # Use spawn point
        if not spawn_point.use(rule.cooldown_seconds):
            return None

        # Calculate final position with offset
        position = spawn_point.position
        if rule.random_offset_radius > 0:
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, rule.random_offset_radius)
            position = (
                position[0] + radius * math.cos(angle),
                position[1],
                position[2] + radius * math.sin(angle)
            )

        rotation = spawn_point.rotation

        # Track player
        self._player_positions[player_id] = position
        if team_id:
            self._player_teams[player_id] = team_id

        # Emit callback
        for callback in self._on_spawn:
            callback(player_id, spawn_point)

        return (position, rotation)

    def schedule_respawn(
        self,
        player_id: str,
        delay_seconds: Optional[float] = None,
        rule: Optional[SpawnRule] = None
    ) -> float:
        """
        Schedule a player respawn.

        Args:
            player_id: Player to respawn
            delay_seconds: Override respawn delay
            rule: Spawn rule to use for delay

        Returns:
            Time when respawn will be available
        """
        rule = rule or self.default_rule
        delay = delay_seconds if delay_seconds is not None else rule.respawn_delay_seconds
        respawn_time = time.time() + delay
        self._pending_respawns[player_id] = respawn_time

        return respawn_time

    def get_respawn_time(self, player_id: str) -> Optional[float]:
        """Get scheduled respawn time for a player."""
        return self._pending_respawns.get(player_id)

    def is_respawn_ready(self, player_id: str) -> bool:
        """Check if player's respawn is ready."""
        respawn_time = self._pending_respawns.get(player_id)
        if respawn_time is None:
            return True
        return time.time() >= respawn_time

    def cancel_respawn(self, player_id: str) -> bool:
        """Cancel a scheduled respawn."""
        if player_id in self._pending_respawns:
            del self._pending_respawns[player_id]
            return True
        return False

    # =========================================================================
    # Player Tracking
    # =========================================================================

    def update_player_position(
        self,
        player_id: str,
        position: Tuple[float, float, float]
    ) -> None:
        """Update tracked player position."""
        self._player_positions[player_id] = position

    def update_player_team(self, player_id: str, team_id: str) -> None:
        """Update player team assignment."""
        self._player_teams[player_id] = team_id

    def remove_player(self, player_id: str) -> None:
        """Remove player from tracking."""
        self._player_positions.pop(player_id, None)
        self._player_teams.pop(player_id, None)
        self._pending_respawns.pop(player_id, None)

    def get_player_position(self, player_id: str) -> Optional[Tuple[float, float, float]]:
        """Get tracked player position."""
        return self._player_positions.get(player_id)

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_spawn(self, callback: Callable[[str, SpawnPoint], None]) -> None:
        """Register callback for player spawn (player_id, spawn_point)."""
        self._on_spawn.append(callback)

    def on_respawn_ready(self, callback: Callable[[str], None]) -> None:
        """Register callback for respawn ready."""
        self._on_respawn_ready.append(callback)

    # =========================================================================
    # Update
    # =========================================================================

    def update(self) -> None:
        """
        Update spawn manager state.

        Should be called each frame to check for ready respawns.
        """
        current_time = time.time()
        ready_players = []

        for player_id, respawn_time in list(self._pending_respawns.items()):
            if current_time >= respawn_time:
                ready_players.append(player_id)

        for player_id in ready_players:
            for callback in self._on_respawn_ready:
                callback(player_id)

    # =========================================================================
    # Utility
    # =========================================================================

    def reset(self) -> None:
        """Reset spawn manager state."""
        for point in self._spawn_points.values():
            point._state = SpawnPointState.AVAILABLE
            point._current_occupants = 0
            point._last_used = 0.0
            point._cooldown_until = 0.0

        self._sequential_index = 0
        self._sequential_indices = {k: 0 for k in self._sequential_indices}
        self._player_positions.clear()
        self._player_teams.clear()
        self._pending_respawns.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get spawn manager statistics."""
        available = sum(1 for p in self._spawn_points.values() if p.is_available)
        return {
            "total_spawn_points": len(self._spawn_points),
            "available_spawn_points": available,
            "teams_configured": len(self._team_configs),
            "pending_respawns": len(self._pending_respawns),
            "tracked_players": len(self._player_positions),
        }
