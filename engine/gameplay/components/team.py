"""
Team Component - Team identification, faction, allegiance, and IFF tagging.

Provides team management for entities including faction membership,
allegiance relationships, and Identification Friend or Foe (IFF) systems.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto, IntFlag
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Set, TYPE_CHECKING

from trinity.descriptors import (
    TrackedDescriptor,
    clear_dirty,
    is_dirty,
)

from engine.gameplay.components.constants import TeamConstants

if TYPE_CHECKING:
    from foundation import to_dict, from_dict


class TeamRelation(Enum):
    """Relationship between teams."""
    SELF = auto()       # Same team
    ALLY = auto()       # Friendly team
    NEUTRAL = auto()    # Neither friend nor foe
    HOSTILE = auto()    # Enemy team


class IFFResponse(IntFlag):
    """IFF (Identification Friend or Foe) response flags."""
    NONE = 0
    FRIEND = 1 << 0      # Identified as friendly
    FOE = 1 << 1         # Identified as hostile
    UNKNOWN = 1 << 2     # Unknown affiliation
    CIVILIAN = 1 << 3    # Non-combatant
    OBJECTIVE = 1 << 4   # Mission objective
    HAZARD = 1 << 5      # Environmental hazard
    PLAYER = 1 << 6      # Player-controlled
    AI = 1 << 7          # AI-controlled


@dataclass
class Faction:
    """
    Represents a faction that entities can belong to.

    Factions define relationships between groups of entities and can
    have multiple teams underneath them.
    """
    id: str
    name: str
    color: tuple[int, int, int] = TeamConstants.DEFAULT_NEUTRAL_COLOR  # RGB color for UI
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Faction):
            return self.id == other.id
        return False


@dataclass
class Team:
    """
    Represents a team within a faction.

    Teams are the basic unit of allegiance for entities.
    """
    id: int
    name: str
    faction: Optional[Faction] = None
    color: Optional[tuple[int, int, int]] = None  # Override faction color
    max_members: int = -1  # -1 = unlimited
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_color(self) -> tuple[int, int, int]:
        """Get the display color (team color or faction color)."""
        if self.color is not None:
            return self.color
        if self.faction is not None:
            return self.faction.color
        return TeamConstants.DEFAULT_NEUTRAL_COLOR

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Team):
            return self.id == other.id
        return False


class TeamRegistry:
    """
    Registry for teams and factions with relationship management.

    Provides centralized team/faction management and relationship queries.
    """

    def __init__(self) -> None:
        self._factions: Dict[str, Faction] = {}
        self._teams: Dict[int, Team] = {}
        self._relations: Dict[tuple[int, int], TeamRelation] = {}
        self._next_team_id = 1

    # =========================================================================
    # FACTION MANAGEMENT
    # =========================================================================

    def register_faction(self, faction: Faction) -> None:
        """Register a faction."""
        self._factions[faction.id] = faction

    def get_faction(self, faction_id: str) -> Optional[Faction]:
        """Get a faction by ID."""
        return self._factions.get(faction_id)

    def get_all_factions(self) -> List[Faction]:
        """Get all registered factions."""
        return list(self._factions.values())

    def remove_faction(self, faction_id: str) -> bool:
        """Remove a faction. Returns True if found and removed."""
        if faction_id in self._factions:
            # Remove faction from all teams
            for team in self._teams.values():
                if team.faction and team.faction.id == faction_id:
                    team.faction = None
            del self._factions[faction_id]
            return True
        return False

    # =========================================================================
    # TEAM MANAGEMENT
    # =========================================================================

    def create_team(
        self,
        name: str,
        faction: Optional[Faction] = None,
        team_id: Optional[int] = None,
        color: Optional[tuple[int, int, int]] = None,
        max_members: int = -1,
    ) -> Team:
        """
        Create a new team.

        Args:
            name: Team name
            faction: Optional faction membership
            team_id: Optional specific ID (auto-generated if not provided)
            color: Optional team color
            max_members: Maximum members (-1 = unlimited)

        Returns:
            The created team
        """
        if team_id is None:
            team_id = self._next_team_id
            self._next_team_id += 1
        elif team_id >= self._next_team_id:
            self._next_team_id = team_id + 1

        team = Team(
            id=team_id,
            name=name,
            faction=faction,
            color=color,
            max_members=max_members,
        )
        self._teams[team_id] = team
        return team

    def get_team(self, team_id: int) -> Optional[Team]:
        """Get a team by ID."""
        return self._teams.get(team_id)

    def get_teams_by_faction(self, faction_id: str) -> List[Team]:
        """Get all teams in a faction."""
        return [
            team for team in self._teams.values()
            if team.faction and team.faction.id == faction_id
        ]

    def get_all_teams(self) -> List[Team]:
        """Get all registered teams."""
        return list(self._teams.values())

    def remove_team(self, team_id: int) -> bool:
        """Remove a team. Returns True if found and removed."""
        if team_id in self._teams:
            # Remove all relations involving this team
            to_remove = [
                key for key in self._relations
                if team_id in key
            ]
            for key in to_remove:
                del self._relations[key]
            del self._teams[team_id]
            return True
        return False

    # =========================================================================
    # RELATIONSHIP MANAGEMENT
    # =========================================================================

    def set_relation(self, team_a_id: int, team_b_id: int, relation: TeamRelation) -> None:
        """
        Set the relationship between two teams (bidirectional).

        Args:
            team_a_id: First team ID
            team_b_id: Second team ID
            relation: Relationship type
        """
        # Store both directions for fast lookup
        self._relations[(team_a_id, team_b_id)] = relation
        self._relations[(team_b_id, team_a_id)] = relation

    def get_relation(self, team_a_id: int, team_b_id: int) -> TeamRelation:
        """
        Get the relationship between two teams.

        Args:
            team_a_id: First team ID
            team_b_id: Second team ID

        Returns:
            Relationship between the teams
        """
        if team_a_id == team_b_id:
            return TeamRelation.SELF

        # Check explicit relation
        relation = self._relations.get((team_a_id, team_b_id))
        if relation is not None:
            return relation

        # Check faction-based relation
        team_a = self._teams.get(team_a_id)
        team_b = self._teams.get(team_b_id)

        if team_a and team_b and team_a.faction and team_b.faction:
            if team_a.faction.id == team_b.faction.id:
                return TeamRelation.ALLY

        # Default to neutral
        return TeamRelation.NEUTRAL

    def clear_relations(self) -> None:
        """Clear all team relations."""
        self._relations.clear()

    def get_allies(self, team_id: int) -> List[int]:
        """Get all allied team IDs."""
        return [
            other_id for other_id in self._teams
            if other_id != team_id and self.get_relation(team_id, other_id) == TeamRelation.ALLY
        ]

    def get_enemies(self, team_id: int) -> List[int]:
        """Get all hostile team IDs."""
        return [
            other_id for other_id in self._teams
            if self.get_relation(team_id, other_id) == TeamRelation.HOSTILE
        ]


# Global team registry singleton
_team_registry: Optional[TeamRegistry] = None


def get_team_registry() -> TeamRegistry:
    """Get the global team registry."""
    global _team_registry
    if _team_registry is None:
        _team_registry = TeamRegistry()
    return _team_registry


def set_team_registry(registry: TeamRegistry) -> None:
    """Set the global team registry."""
    global _team_registry
    _team_registry = registry


class TeamComponent:
    """
    Team component for entity team/faction membership.

    Features:
    - Team and faction assignment
    - IFF (Identification Friend or Foe) tagging
    - Relationship queries
    - Team change callbacks
    - Serialization support

    Attributes:
        team_id: Current team ID
        iff_tags: IFF response flags
    """

    # Tracked descriptors
    team_id = TrackedDescriptor(field_type=int, use_bitmask=True, field_offset=0)

    __slots__ = (
        "__dict__",
        "__weakref__",
        "_iff_tags",
        "_secondary_teams",
        "_custom_relations",
        "_on_team_changed",
        "_entity_id",
    )

    def __init__(
        self,
        team_id: int = 0,
        iff_tags: IFFResponse = IFFResponse.UNKNOWN,
        entity_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the team component.

        Args:
            team_id: Initial team ID
            iff_tags: Initial IFF tags
            entity_id: Optional entity ID for tracking
        """
        self._iff_tags = iff_tags
        self._secondary_teams: Set[int] = set()
        self._custom_relations: Dict[str, TeamRelation] = {}  # entity_id -> relation
        self._on_team_changed: List[Callable[[int, int], None]] = []
        self._entity_id = entity_id

        self.team_id = team_id
        clear_dirty(self)

    # =========================================================================
    # TEAM MEMBERSHIP
    # =========================================================================

    @property
    def team(self) -> Optional[Team]:
        """Get the current team object."""
        return get_team_registry().get_team(self.team_id)

    @property
    def faction(self) -> Optional[Faction]:
        """Get the faction of the current team."""
        team = self.team
        return team.faction if team else None

    @property
    def faction_id(self) -> Optional[str]:
        """Get the faction ID."""
        faction = self.faction
        return faction.id if faction else None

    def set_team(self, team_id: int) -> None:
        """
        Set the team ID.

        Args:
            team_id: New team ID
        """
        old_team = self.team_id
        if old_team != team_id:
            self.team_id = team_id
            for callback in self._on_team_changed:
                callback(old_team, team_id)

    def join_team(self, team: Team) -> None:
        """Join a team by team object."""
        self.set_team(team.id)

    def leave_team(self) -> None:
        """Leave current team (set to team 0 - no team)."""
        self.set_team(0)

    # =========================================================================
    # SECONDARY TEAMS
    # =========================================================================

    @property
    def secondary_teams(self) -> FrozenSet[int]:
        """Get secondary team memberships."""
        return frozenset(self._secondary_teams)

    @property
    def all_teams(self) -> FrozenSet[int]:
        """Get all team memberships (primary + secondary)."""
        teams = {self.team_id}
        teams.update(self._secondary_teams)
        return frozenset(teams)

    def add_secondary_team(self, team_id: int) -> None:
        """Add a secondary team membership."""
        self._secondary_teams.add(team_id)

    def remove_secondary_team(self, team_id: int) -> None:
        """Remove a secondary team membership."""
        self._secondary_teams.discard(team_id)

    def clear_secondary_teams(self) -> None:
        """Clear all secondary team memberships."""
        self._secondary_teams.clear()

    def is_member_of(self, team_id: int) -> bool:
        """Check if member of a specific team (primary or secondary)."""
        return team_id == self.team_id or team_id in self._secondary_teams

    # =========================================================================
    # IFF TAGS
    # =========================================================================

    @property
    def iff_tags(self) -> IFFResponse:
        """Get IFF tags."""
        return self._iff_tags

    @iff_tags.setter
    def iff_tags(self, value: IFFResponse) -> None:
        """Set IFF tags."""
        self._iff_tags = value

    def add_iff_tag(self, tag: IFFResponse) -> None:
        """Add an IFF tag."""
        self._iff_tags |= tag

    def remove_iff_tag(self, tag: IFFResponse) -> None:
        """Remove an IFF tag."""
        self._iff_tags &= ~tag

    def has_iff_tag(self, tag: IFFResponse) -> bool:
        """Check if has a specific IFF tag."""
        return bool(self._iff_tags & tag)

    def is_friendly_iff(self) -> bool:
        """Check if IFF identifies as friendly."""
        return self.has_iff_tag(IFFResponse.FRIEND)

    def is_hostile_iff(self) -> bool:
        """Check if IFF identifies as hostile."""
        return self.has_iff_tag(IFFResponse.FOE)

    def is_player(self) -> bool:
        """Check if this is a player-controlled entity."""
        return self.has_iff_tag(IFFResponse.PLAYER)

    def is_ai(self) -> bool:
        """Check if this is an AI-controlled entity."""
        return self.has_iff_tag(IFFResponse.AI)

    # =========================================================================
    # RELATIONSHIP QUERIES
    # =========================================================================

    def get_relation_to(self, other: TeamComponent) -> TeamRelation:
        """
        Get relationship to another entity.

        Args:
            other: Other entity's team component

        Returns:
            Relationship to the other entity
        """
        # Check for custom relation override
        if other._entity_id and other._entity_id in self._custom_relations:
            return self._custom_relations[other._entity_id]

        # Use team-based relation
        return get_team_registry().get_relation(self.team_id, other.team_id)

    def is_ally(self, other: TeamComponent) -> bool:
        """Check if another entity is an ally."""
        relation = self.get_relation_to(other)
        return relation in (TeamRelation.SELF, TeamRelation.ALLY)

    def is_enemy(self, other: TeamComponent) -> bool:
        """Check if another entity is an enemy."""
        return self.get_relation_to(other) == TeamRelation.HOSTILE

    def is_neutral(self, other: TeamComponent) -> bool:
        """Check if another entity is neutral."""
        return self.get_relation_to(other) == TeamRelation.NEUTRAL

    def is_same_team(self, other: TeamComponent) -> bool:
        """Check if on the same team."""
        return self.team_id == other.team_id

    def is_same_faction(self, other: TeamComponent) -> bool:
        """Check if in the same faction."""
        my_faction = self.faction
        other_faction = other.faction
        if my_faction is None or other_faction is None:
            return False
        return my_faction.id == other_faction.id

    def shares_any_team(self, other: TeamComponent) -> bool:
        """Check if sharing any team (including secondary)."""
        return bool(self.all_teams & other.all_teams)

    # =========================================================================
    # CUSTOM RELATIONS
    # =========================================================================

    def set_custom_relation(self, entity_id: str, relation: TeamRelation) -> None:
        """
        Set a custom relation override for a specific entity.

        Args:
            entity_id: Target entity ID
            relation: Custom relationship
        """
        self._custom_relations[entity_id] = relation

    def clear_custom_relation(self, entity_id: str) -> None:
        """Clear a custom relation override."""
        self._custom_relations.pop(entity_id, None)

    def clear_all_custom_relations(self) -> None:
        """Clear all custom relation overrides."""
        self._custom_relations.clear()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_team_changed(self, callback: Callable[[int, int], None]) -> None:
        """Register callback for team changes (old_team_id, new_team_id)."""
        self._on_team_changed.append(callback)

    def off_team_changed(self, callback: Callable[[int, int], None]) -> None:
        """Unregister team change callback."""
        if callback in self._on_team_changed:
            self._on_team_changed.remove(callback)

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    @property
    def team_name(self) -> str:
        """Get the team name for display."""
        team = self.team
        return team.name if team else "No Team"

    @property
    def faction_name(self) -> str:
        """Get the faction name for display."""
        faction = self.faction
        return faction.name if faction else "No Faction"

    @property
    def team_color(self) -> tuple[int, int, int]:
        """Get the team color for display."""
        team = self.team
        return team.display_color if team else TeamConstants.DEFAULT_NEUTRAL_COLOR

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize team component to dictionary."""
        return {
            "team_id": self.team_id,
            "iff_tags": int(self._iff_tags),
            "secondary_teams": list(self._secondary_teams),
            "custom_relations": {
                k: v.name for k, v in self._custom_relations.items()
            },
            "entity_id": self._entity_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TeamComponent:
        """Deserialize team component from dictionary."""
        component = cls(
            team_id=data.get("team_id", 0),
            iff_tags=IFFResponse(data.get("iff_tags", IFFResponse.UNKNOWN)),
            entity_id=data.get("entity_id"),
        )

        component._secondary_teams = set(data.get("secondary_teams", []))
        component._custom_relations = {
            k: TeamRelation[v]
            for k, v in data.get("custom_relations", {}).items()
        }

        return component

    def __repr__(self) -> str:
        return (
            f"TeamComponent(team={self.team_name}, "
            f"faction={self.faction_name}, iff={self._iff_tags})"
        )


# Descriptor setup
TeamComponent.team_id.__set_name__(TeamComponent, "team_id")


__all__ = [
    "TeamComponent",
    "TeamRelation",
    "IFFResponse",
    "Team",
    "Faction",
    "TeamRegistry",
    "get_team_registry",
    "set_team_registry",
]
