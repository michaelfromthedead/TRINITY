"""
Combat System - Teams Module

Provides team/faction management with support for:
- Team membership and allegiance
- IFF (Identify Friend/Foe) system
- Team relationships (hostile, neutral, friendly)
- Friendly fire configuration
- Dynamic team changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import time

from .constants import (
    TeamRelation,
    TeamConfig,
    DEFAULT_TEAM_CONFIG,
    DEFAULT_TEAM_ID,
    NEUTRAL_TEAM_ID,
    MAX_TEAMS,
    FRIENDLY_FIRE_FULL,
    FRIENDLY_FIRE_REDUCED,
    FRIENDLY_FIRE_NONE,
    CombatEventType,
)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class TeamInfo:
    """Information about a team/faction."""

    team_id: int
    name: str = ""
    color: Tuple[int, int, int] = (255, 255, 255)  # RGB
    max_members: int = 0  # 0 = unlimited
    friendly_fire_multiplier: float = FRIENDLY_FIRE_NONE
    spawn_points: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Relationship overrides (default uses global settings)
    _relationship_overrides: Dict[int, TeamRelation] = field(default_factory=dict)

    @property
    def member_count(self) -> int:
        """Count is tracked externally by TeamSystem."""
        return self.metadata.get("_member_count", 0)

    @property
    def is_full(self) -> bool:
        """Check if team is at max capacity."""
        if self.max_members <= 0:
            return False
        return self.member_count >= self.max_members

    def get_relationship_override(self, other_team_id: int) -> Optional[TeamRelation]:
        """Get relationship override for another team."""
        return self._relationship_overrides.get(other_team_id)

    def set_relationship_override(self, other_team_id: int, relation: TeamRelation) -> None:
        """Set relationship override for another team."""
        self._relationship_overrides[other_team_id] = relation

    def clear_relationship_override(self, other_team_id: int) -> None:
        """Clear relationship override for another team."""
        self._relationship_overrides.pop(other_team_id, None)


@dataclass
class TeamMembership:
    """An entity's team membership."""

    entity_id: int
    team_id: int
    joined_at: float = field(default_factory=time.time)
    role: str = "member"  # leader, member, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def membership_duration(self) -> float:
        """Time since joining the team."""
        return time.time() - self.joined_at


@dataclass
class TeamChangeEvent:
    """Event emitted when an entity changes teams."""

    entity_id: int
    old_team_id: Optional[int]
    new_team_id: int
    timestamp: float = field(default_factory=time.time)
    reason: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IFFResult:
    """Result of an IFF (Identify Friend/Foe) check."""

    source_id: int
    target_id: int
    source_team: int
    target_team: int
    relation: TeamRelation
    friendly_fire_multiplier: float
    can_damage: bool
    can_heal: bool
    is_same_team: bool

    @property
    def is_friendly(self) -> bool:
        """Check if relation is friendly."""
        return self.relation == TeamRelation.FRIENDLY

    @property
    def is_hostile(self) -> bool:
        """Check if relation is hostile."""
        return self.relation == TeamRelation.HOSTILE

    @property
    def is_neutral(self) -> bool:
        """Check if relation is neutral."""
        return self.relation == TeamRelation.NEUTRAL


# =============================================================================
# TEAM SYSTEM
# =============================================================================


class TeamSystem:
    """
    System for managing teams and factions.

    Features:
    - Team creation and management
    - Entity team membership
    - IFF (Identify Friend/Foe) checks
    - Configurable team relationships
    - Friendly fire settings per team
    - Team change events
    """

    def __init__(self, config: Optional[TeamConfig] = None) -> None:
        """
        Initialize the team system.

        Args:
            config: Team configuration
        """
        self._config = config or DEFAULT_TEAM_CONFIG

        # Teams registry
        self._teams: Dict[int, TeamInfo] = {}

        # Entity memberships
        self._memberships: Dict[int, TeamMembership] = {}

        # Global relationship matrix
        self._relationships: Dict[Tuple[int, int], TeamRelation] = {}

        # Event handlers
        self._on_team_change: List[Callable[[TeamChangeEvent], None]] = []
        self._on_team_created: List[Callable[[TeamInfo], None]] = []
        self._on_team_removed: List[Callable[[int], None]] = []

        # Create default teams
        self._create_default_team()
        self._create_neutral_team()

    def _create_default_team(self) -> None:
        """Create the default team."""
        self.create_team(
            team_id=DEFAULT_TEAM_ID,
            name="Default",
            friendly_fire_multiplier=self._config.default_friendly_fire,
        )

    def _create_neutral_team(self) -> None:
        """Create the neutral/world team."""
        self.create_team(
            team_id=NEUTRAL_TEAM_ID,
            name="Neutral",
            friendly_fire_multiplier=FRIENDLY_FIRE_NONE,
        )

    @property
    def config(self) -> TeamConfig:
        """Get team configuration."""
        return self._config

    # =========================================================================
    # TEAM MANAGEMENT
    # =========================================================================

    def create_team(
        self,
        team_id: int,
        name: str = "",
        color: Tuple[int, int, int] = (255, 255, 255),
        max_members: int = 0,
        friendly_fire_multiplier: float = FRIENDLY_FIRE_NONE,
        spawn_points: Optional[List[Any]] = None,
        **metadata: Any,
    ) -> TeamInfo:
        """
        Create a new team.

        Args:
            team_id: Unique team ID
            name: Team display name
            color: Team color (RGB)
            max_members: Maximum team size (0 = unlimited)
            friendly_fire_multiplier: Friendly fire damage multiplier
            spawn_points: List of spawn point locations
            **metadata: Additional metadata

        Returns:
            Created TeamInfo

        Raises:
            ValueError: If team ID already exists or max teams exceeded
        """
        if team_id in self._teams and team_id not in (DEFAULT_TEAM_ID, NEUTRAL_TEAM_ID):
            raise ValueError(f"Team ID {team_id} already exists")

        if len(self._teams) >= self._config.max_teams:
            raise ValueError(f"Maximum teams ({self._config.max_teams}) exceeded")

        team = TeamInfo(
            team_id=team_id,
            name=name or f"Team {team_id}",
            color=color,
            max_members=max_members,
            friendly_fire_multiplier=friendly_fire_multiplier,
            spawn_points=spawn_points or [],
            metadata={"_member_count": 0, **metadata},
        )

        self._teams[team_id] = team

        # Set default relationships
        for other_id in self._teams:
            if other_id != team_id:
                self._set_default_relationship(team_id, other_id)

        # Emit event
        for handler in self._on_team_created:
            try:
                handler(team)
            except Exception:
                pass

        return team

    def remove_team(self, team_id: int) -> bool:
        """
        Remove a team.

        Args:
            team_id: Team to remove

        Returns:
            True if team was removed
        """
        if team_id in (DEFAULT_TEAM_ID, NEUTRAL_TEAM_ID):
            return False  # Cannot remove default teams

        if team_id not in self._teams:
            return False

        # Move all members to default team
        for entity_id in list(self._memberships.keys()):
            if self._memberships[entity_id].team_id == team_id:
                self.set_team(entity_id, DEFAULT_TEAM_ID, reason="team_removed")

        # Remove team
        del self._teams[team_id]

        # Clean up relationships
        to_remove = [key for key in self._relationships if team_id in key]
        for key in to_remove:
            del self._relationships[key]

        # Emit event
        for handler in self._on_team_removed:
            try:
                handler(team_id)
            except Exception:
                pass

        return True

    def get_team(self, team_id: int) -> Optional[TeamInfo]:
        """Get team info by ID."""
        return self._teams.get(team_id)

    def get_all_teams(self) -> List[TeamInfo]:
        """Get all teams."""
        return list(self._teams.values())

    def team_exists(self, team_id: int) -> bool:
        """Check if a team exists."""
        return team_id in self._teams

    def get_team_members(self, team_id: int) -> List[int]:
        """Get all entity IDs on a team."""
        return [
            eid for eid, membership in self._memberships.items()
            if membership.team_id == team_id
        ]

    def get_team_member_count(self, team_id: int) -> int:
        """Get number of members on a team."""
        return sum(1 for m in self._memberships.values() if m.team_id == team_id)

    # =========================================================================
    # ENTITY MEMBERSHIP
    # =========================================================================

    def set_team(
        self,
        entity_id: int,
        team_id: int,
        role: str = "member",
        reason: str = "manual",
        **metadata: Any,
    ) -> bool:
        """
        Set an entity's team.

        Args:
            entity_id: Entity to assign
            team_id: Team to join
            role: Role within team
            reason: Reason for team change
            **metadata: Additional metadata

        Returns:
            True if team was set successfully
        """
        if not self._config.allow_team_changes:
            existing = self._memberships.get(entity_id)
            if existing is not None:
                return False

        if team_id not in self._teams:
            return False

        team = self._teams[team_id]
        if team.is_full:
            return False

        # Get old team
        old_membership = self._memberships.get(entity_id)
        old_team_id = old_membership.team_id if old_membership else None

        # Update old team member count
        if old_team_id is not None and old_team_id in self._teams:
            old_count = self._teams[old_team_id].metadata.get("_member_count", 0)
            self._teams[old_team_id].metadata["_member_count"] = max(0, old_count - 1)

        # Create new membership
        membership = TeamMembership(
            entity_id=entity_id,
            team_id=team_id,
            role=role,
            metadata=metadata,
        )
        self._memberships[entity_id] = membership

        # Update new team member count
        new_count = team.metadata.get("_member_count", 0)
        team.metadata["_member_count"] = new_count + 1

        # Emit event
        event = TeamChangeEvent(
            entity_id=entity_id,
            old_team_id=old_team_id,
            new_team_id=team_id,
            reason=reason,
            metadata=metadata,
        )
        self._emit_team_change(event)

        return True

    def get_team_id(self, entity_id: int) -> int:
        """Get an entity's team ID."""
        membership = self._memberships.get(entity_id)
        return membership.team_id if membership else DEFAULT_TEAM_ID

    def get_membership(self, entity_id: int) -> Optional[TeamMembership]:
        """Get an entity's team membership."""
        return self._memberships.get(entity_id)

    def remove_entity(self, entity_id: int) -> bool:
        """
        Remove an entity from the team system.

        Args:
            entity_id: Entity to remove

        Returns:
            True if entity was removed
        """
        membership = self._memberships.pop(entity_id, None)
        if membership:
            team = self._teams.get(membership.team_id)
            if team:
                count = team.metadata.get("_member_count", 0)
                team.metadata["_member_count"] = max(0, count - 1)
            return True
        return False

    def is_on_team(self, entity_id: int, team_id: int) -> bool:
        """Check if entity is on a specific team."""
        return self.get_team_id(entity_id) == team_id

    # =========================================================================
    # RELATIONSHIPS
    # =========================================================================

    def set_relationship(
        self,
        team_a: int,
        team_b: int,
        relation: TeamRelation,
    ) -> None:
        """
        Set relationship between two teams.

        Args:
            team_a: First team ID
            team_b: Second team ID
            relation: Relationship type
        """
        # Store both directions
        self._relationships[(team_a, team_b)] = relation
        self._relationships[(team_b, team_a)] = relation

    def get_relationship(self, team_a: int, team_b: int) -> TeamRelation:
        """
        Get relationship between two teams.

        Args:
            team_a: First team ID
            team_b: Second team ID

        Returns:
            TeamRelation between teams
        """
        # Same team is always friendly
        if team_a == team_b:
            return TeamRelation.FRIENDLY

        # Check for team-specific override
        team_info = self._teams.get(team_a)
        if team_info:
            override = team_info.get_relationship_override(team_b)
            if override is not None:
                return override

        # Check global relationship
        relation = self._relationships.get((team_a, team_b))
        if relation is not None:
            return relation

        # Default to hostile
        return TeamRelation.HOSTILE

    def _set_default_relationship(self, team_a: int, team_b: int) -> None:
        """Set default relationship between teams."""
        # Neutral team is neutral to everyone
        if team_a == NEUTRAL_TEAM_ID or team_b == NEUTRAL_TEAM_ID:
            self.set_relationship(team_a, team_b, TeamRelation.NEUTRAL)
        else:
            # Default to hostile
            self.set_relationship(team_a, team_b, TeamRelation.HOSTILE)

    def set_all_hostile(self) -> None:
        """Set all teams hostile to each other."""
        teams = list(self._teams.keys())
        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
                if team_a != NEUTRAL_TEAM_ID and team_b != NEUTRAL_TEAM_ID:
                    self.set_relationship(team_a, team_b, TeamRelation.HOSTILE)

    def set_all_friendly(self) -> None:
        """Set all teams friendly to each other."""
        teams = list(self._teams.keys())
        for i, team_a in enumerate(teams):
            for team_b in teams[i + 1:]:
                self.set_relationship(team_a, team_b, TeamRelation.FRIENDLY)

    # =========================================================================
    # IFF (IDENTIFY FRIEND/FOE)
    # =========================================================================

    def check_iff(self, source_id: int, target_id: int) -> IFFResult:
        """
        Perform IFF check between two entities.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            IFFResult with relationship information
        """
        source_team = self.get_team_id(source_id)
        target_team = self.get_team_id(target_id)
        relation = self.get_relationship(source_team, target_team)
        is_same_team = source_team == target_team

        # Determine friendly fire multiplier
        if is_same_team:
            team_info = self._teams.get(source_team)
            ff_mult = team_info.friendly_fire_multiplier if team_info else self._config.default_friendly_fire
        else:
            ff_mult = 1.0

        # Can damage?
        can_damage = relation == TeamRelation.HOSTILE
        if is_same_team and self._config.allow_team_damage and ff_mult > 0:
            can_damage = True
        if relation == TeamRelation.NEUTRAL and self._config.allow_team_damage:
            can_damage = True

        # Can heal?
        can_heal = relation in (TeamRelation.FRIENDLY, TeamRelation.NEUTRAL) or is_same_team

        return IFFResult(
            source_id=source_id,
            target_id=target_id,
            source_team=source_team,
            target_team=target_team,
            relation=relation,
            friendly_fire_multiplier=ff_mult,
            can_damage=can_damage,
            can_heal=can_heal,
            is_same_team=is_same_team,
        )

    def can_attack(self, source_id: int, target_id: int) -> bool:
        """Check if source can attack target."""
        return self.check_iff(source_id, target_id).can_damage

    def can_heal(self, source_id: int, target_id: int) -> bool:
        """Check if source can heal target."""
        return self.check_iff(source_id, target_id).can_heal

    def is_friendly(self, entity_a: int, entity_b: int) -> bool:
        """Check if two entities are friendly."""
        return self.check_iff(entity_a, entity_b).is_friendly

    def is_hostile(self, entity_a: int, entity_b: int) -> bool:
        """Check if two entities are hostile."""
        return self.check_iff(entity_a, entity_b).is_hostile

    def get_friendly_fire_multiplier(self, source_id: int, target_id: int) -> float:
        """Get friendly fire damage multiplier between entities."""
        return self.check_iff(source_id, target_id).friendly_fire_multiplier

    # =========================================================================
    # FRIENDLY FIRE
    # =========================================================================

    def set_friendly_fire(self, team_id: int, multiplier: float) -> bool:
        """
        Set friendly fire multiplier for a team.

        Args:
            team_id: Team ID
            multiplier: Damage multiplier (0.0 = no FF, 1.0 = full FF)

        Returns:
            True if team exists and was updated
        """
        team = self._teams.get(team_id)
        if not team:
            return False

        team.friendly_fire_multiplier = max(0.0, min(1.0, multiplier))
        return True

    def enable_friendly_fire(self, team_id: int, multiplier: float = FRIENDLY_FIRE_FULL) -> bool:
        """Enable friendly fire for a team."""
        return self.set_friendly_fire(team_id, multiplier)

    def disable_friendly_fire(self, team_id: int) -> bool:
        """Disable friendly fire for a team."""
        return self.set_friendly_fire(team_id, FRIENDLY_FIRE_NONE)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def on_team_change(self, handler: Callable[[TeamChangeEvent], None]) -> None:
        """Register handler for team change events."""
        self._on_team_change.append(handler)

    def on_team_created(self, handler: Callable[[TeamInfo], None]) -> None:
        """Register handler for team created events."""
        self._on_team_created.append(handler)

    def on_team_removed(self, handler: Callable[[int], None]) -> None:
        """Register handler for team removed events."""
        self._on_team_removed.append(handler)

    def _emit_team_change(self, event: TeamChangeEvent) -> None:
        """Emit team change event."""
        for handler in self._on_team_change:
            try:
                handler(event)
            except Exception:
                pass

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_enemies(self, entity_id: int) -> List[int]:
        """Get all hostile entity IDs for an entity."""
        entity_team = self.get_team_id(entity_id)
        enemies = []

        for other_id, membership in self._memberships.items():
            if other_id == entity_id:
                continue
            if self.get_relationship(entity_team, membership.team_id) == TeamRelation.HOSTILE:
                enemies.append(other_id)

        return enemies

    def get_allies(self, entity_id: int) -> List[int]:
        """Get all friendly entity IDs for an entity."""
        entity_team = self.get_team_id(entity_id)
        allies = []

        for other_id, membership in self._memberships.items():
            if other_id == entity_id:
                continue
            if membership.team_id == entity_team:
                allies.append(other_id)

        return allies

    def get_hostile_teams(self, team_id: int) -> List[int]:
        """Get all teams hostile to a given team."""
        hostile = []
        for other_id in self._teams:
            if other_id != team_id:
                if self.get_relationship(team_id, other_id) == TeamRelation.HOSTILE:
                    hostile.append(other_id)
        return hostile

    def get_allied_teams(self, team_id: int) -> List[int]:
        """Get all teams friendly to a given team."""
        allied = [team_id]  # Include self
        for other_id in self._teams:
            if other_id != team_id:
                if self.get_relationship(team_id, other_id) == TeamRelation.FRIENDLY:
                    allied.append(other_id)
        return allied

    # =========================================================================
    # AUTO-BALANCE
    # =========================================================================

    def get_team_with_fewest_members(self, exclude: Optional[Set[int]] = None) -> int:
        """
        Get the team with fewest members.

        Args:
            exclude: Team IDs to exclude from consideration

        Returns:
            Team ID with fewest members
        """
        exclude = exclude or {NEUTRAL_TEAM_ID}
        min_count = float("inf")
        min_team = DEFAULT_TEAM_ID

        for team_id, team in self._teams.items():
            if team_id in exclude:
                continue
            count = self.get_team_member_count(team_id)
            if count < min_count and not team.is_full:
                min_count = count
                min_team = team_id

        return min_team

    def auto_assign_team(
        self,
        entity_id: int,
        exclude: Optional[Set[int]] = None,
    ) -> int:
        """
        Auto-assign entity to team with fewest members.

        Args:
            entity_id: Entity to assign
            exclude: Team IDs to exclude

        Returns:
            Assigned team ID
        """
        team_id = self.get_team_with_fewest_members(exclude)
        self.set_team(entity_id, team_id, reason="auto_balance")
        return team_id

    # =========================================================================
    # UTILITY
    # =========================================================================

    def clear(self) -> None:
        """Clear all teams and memberships (except defaults)."""
        self._memberships.clear()
        self._relationships.clear()
        self._teams.clear()
        self._create_default_team()
        self._create_neutral_team()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "TeamInfo",
    "TeamMembership",
    "TeamChangeEvent",
    "IFFResult",
    # System
    "TeamSystem",
]
