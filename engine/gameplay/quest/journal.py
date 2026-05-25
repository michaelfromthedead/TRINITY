"""
Quest Journal Module.

Provides QuestJournal for quest log, HUD tracker, and world markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable

from .constants import DEFAULT_MAX_HUD_TRACKED_QUESTS, MAX_HUD_OBJECTIVES_DISPLAYED
from .quest import Quest, QuestState, QuestType
from .tracker import QuestEvent, QuestEventType, QuestTracker, TrackedQuest

if TYPE_CHECKING:
    from .objectives import Objective

__all__ = [
    "JournalCategory",
    "JournalEntry",
    "QuestJournal",
    "HUDTracker",
    "WorldMarker",
    "MarkerType",
    "JournalFilter",
]


class JournalCategory(Enum):
    """Quest journal categories."""

    MAIN = auto()
    SIDE = auto()
    DAILY = auto()
    WEEKLY = auto()
    COMPLETED = auto()
    FAILED = auto()
    ALL = auto()


class MarkerType(Enum):
    """Types of world markers."""

    QUEST_GIVER = auto()  # NPC with available quest
    QUEST_TURN_IN = auto()  # NPC to turn in quest
    OBJECTIVE = auto()  # Objective location
    AREA = auto()  # Area to reach
    ENEMY = auto()  # Enemy to kill
    ITEM = auto()  # Item to collect
    INTERACT = auto()  # Object to interact with
    ESCORT_NPC = auto()  # NPC to escort
    WAYPOINT = auto()  # Generic waypoint


@dataclass
class WorldMarker:
    """
    A marker shown in the game world.

    Represents quest objectives, NPCs, and other points of interest.
    """

    id: str
    marker_type: MarkerType
    quest_id: str | None = None
    objective_id: str | None = None
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: float = 0.0
    label: str = ""
    description: str = ""
    icon: str = ""
    color: tuple[int, int, int, int] = (255, 255, 0, 255)  # RGBA
    scale: float = 1.0
    visible: bool = True
    minimap_visible: bool = True
    compass_visible: bool = True
    distance_fade: bool = True
    fade_start: float = 50.0
    fade_end: float = 100.0
    priority: int = 0  # Higher = more important
    tracked: bool = False  # Currently tracked by player

    def set_position(self, x: float, y: float, z: float) -> None:
        """Update marker position."""
        self.position = (x, y, z)

    def show(self) -> None:
        """Show the marker."""
        self.visible = True

    def hide(self) -> None:
        """Hide the marker."""
        self.visible = False


@dataclass
class JournalEntry:
    """
    An entry in the quest journal.

    Contains display information for a quest in the journal UI.
    """

    quest_id: str
    title: str
    description: str
    category: JournalCategory
    quest_type: QuestType
    state: QuestState
    level_requirement: int = 1
    objectives: list[dict[str, Any]] = field(default_factory=list)
    rewards_text: list[str] = field(default_factory=list)
    zone: str = ""
    giver_name: str = ""
    is_tracked: bool = False
    is_pinned: bool = False
    read: bool = False
    accepted_time: float | None = None
    completed_time: float | None = None
    total_progress: float = 0.0
    sort_order: int = 0

    @classmethod
    def from_tracked_quest(
        cls,
        tracked: TrackedQuest,
        is_tracked: bool = False
    ) -> JournalEntry:
        """Create a journal entry from a tracked quest."""
        quest = tracked.quest
        quest_def = quest.definition

        # Determine category
        if quest.state == QuestState.TURNED_IN:
            category = JournalCategory.COMPLETED
        elif quest.state == QuestState.FAILED:
            category = JournalCategory.FAILED
        elif quest_def.quest_type == QuestType.MAIN:
            category = JournalCategory.MAIN
        elif quest_def.quest_type == QuestType.DAILY:
            category = JournalCategory.DAILY
        elif quest_def.quest_type == QuestType.WEEKLY:
            category = JournalCategory.WEEKLY
        else:
            category = JournalCategory.SIDE

        # Build objectives list
        objectives = []
        for obj in tracked.objectives:
            objectives.append({
                "id": obj.id,
                "description": obj.description,
                "progress_text": obj.progress_text,
                "progress": obj.progress,
                "is_complete": obj.is_complete,
                "is_optional": obj.optional,
                "is_hidden": obj.hidden,
            })

        # Calculate total progress
        if tracked.objectives:
            required = [o for o in tracked.objectives if not o.optional]
            total_progress = (
                sum(o.progress for o in required) / len(required)
                if required else 0.0
            )
        else:
            total_progress = 0.0

        # Build rewards text
        rewards_text = []
        for reward in quest_def.rewards:
            if hasattr(reward, "description"):
                rewards_text.append(reward.description)
            else:
                rewards_text.append(str(reward))

        return cls(
            quest_id=quest.id,
            title=quest.name,
            description=quest_def.description,
            category=category,
            quest_type=quest_def.quest_type,
            state=quest.state,
            level_requirement=quest_def.level_requirement,
            objectives=objectives,
            rewards_text=rewards_text,
            zone=quest_def.zone,
            giver_name=quest_def.giver_id,
            is_tracked=is_tracked,
            accepted_time=quest.accepted_at,
            completed_time=quest.completed_at,
            total_progress=total_progress,
        )


@dataclass
class JournalFilter:
    """Filter configuration for journal entries."""

    categories: set[JournalCategory] = field(default_factory=lambda: {JournalCategory.ALL})
    quest_types: set[QuestType] | None = None
    states: set[QuestState] | None = None
    zones: set[str] | None = None
    min_level: int | None = None
    max_level: int | None = None
    search_text: str = ""
    show_hidden: bool = False
    tracked_only: bool = False
    pinned_only: bool = False

    def matches(self, entry: JournalEntry) -> bool:
        """Check if an entry matches this filter."""
        # Category filter
        if JournalCategory.ALL not in self.categories:
            if entry.category not in self.categories:
                return False

        # Quest type filter
        if self.quest_types is not None:
            if entry.quest_type not in self.quest_types:
                return False

        # State filter
        if self.states is not None:
            if entry.state not in self.states:
                return False

        # Zone filter
        if self.zones is not None:
            if entry.zone and entry.zone not in self.zones:
                return False

        # Level filter
        if self.min_level is not None:
            if entry.level_requirement < self.min_level:
                return False
        if self.max_level is not None:
            if entry.level_requirement > self.max_level:
                return False

        # Text search
        if self.search_text:
            search_lower = self.search_text.lower()
            if (
                search_lower not in entry.title.lower()
                and search_lower not in entry.description.lower()
            ):
                return False

        # Tracked/pinned filters
        if self.tracked_only and not entry.is_tracked:
            return False
        if self.pinned_only and not entry.is_pinned:
            return False

        return True


class HUDTracker:
    """
    HUD quest tracker display.

    Shows currently tracked quests and objectives on the game HUD.
    """

    def __init__(self, max_tracked: int = DEFAULT_MAX_HUD_TRACKED_QUESTS) -> None:
        self.max_tracked = max_tracked
        self._tracked_quests: list[str] = []
        self._entries: dict[str, JournalEntry] = {}
        self._on_update: Callable[[], None] | None = None

    @property
    def tracked_quest_ids(self) -> list[str]:
        """Get list of tracked quest IDs."""
        return self._tracked_quests.copy()

    @property
    def tracked_entries(self) -> list[JournalEntry]:
        """Get tracked journal entries in order."""
        return [
            self._entries[qid]
            for qid in self._tracked_quests
            if qid in self._entries
        ]

    def set_update_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when tracker updates."""
        self._on_update = callback

    def _notify_update(self) -> None:
        """Notify that tracker has updated."""
        if self._on_update:
            self._on_update()

    def track_quest(self, quest_id: str, entry: JournalEntry) -> bool:
        """
        Add a quest to the tracker.

        Returns:
            True if quest was added, False if already tracked or at limit
        """
        if quest_id in self._tracked_quests:
            return False
        if len(self._tracked_quests) >= self.max_tracked:
            return False

        self._tracked_quests.append(quest_id)
        self._entries[quest_id] = entry
        entry.is_tracked = True
        self._notify_update()
        return True

    def untrack_quest(self, quest_id: str) -> bool:
        """
        Remove a quest from the tracker.

        Returns:
            True if quest was removed
        """
        if quest_id not in self._tracked_quests:
            return False

        self._tracked_quests.remove(quest_id)
        if quest_id in self._entries:
            self._entries[quest_id].is_tracked = False
            del self._entries[quest_id]
        self._notify_update()
        return True

    def update_entry(self, entry: JournalEntry) -> None:
        """Update a tracked entry."""
        if entry.quest_id in self._entries:
            self._entries[entry.quest_id] = entry
            entry.is_tracked = True
            self._notify_update()

    def reorder(self, quest_ids: list[str]) -> None:
        """Reorder tracked quests."""
        # Only include valid quest IDs
        new_order = [qid for qid in quest_ids if qid in self._tracked_quests]
        # Add any missing ones at the end
        for qid in self._tracked_quests:
            if qid not in new_order:
                new_order.append(qid)
        self._tracked_quests = new_order
        self._notify_update()

    def clear(self) -> None:
        """Clear all tracked quests."""
        for qid in self._tracked_quests:
            if qid in self._entries:
                self._entries[qid].is_tracked = False
        self._tracked_quests.clear()
        self._entries.clear()
        self._notify_update()

    def get_display_data(self) -> list[dict[str, Any]]:
        """Get data formatted for HUD display."""
        data = []
        for entry in self.tracked_entries:
            # Get visible objectives
            visible_objectives = [
                obj for obj in entry.objectives
                if not obj.get("is_hidden", False)
            ]

            data.append({
                "quest_id": entry.quest_id,
                "title": entry.title,
                "progress": entry.total_progress,
                "objectives": visible_objectives[:MAX_HUD_OBJECTIVES_DISPLAYED],  # Limit shown objectives
                "is_complete": entry.state == QuestState.COMPLETE,
            })
        return data


class QuestJournal:
    """
    Quest journal and tracking system.

    Manages:
    - Quest log display
    - Quest tracking for HUD
    - World markers
    - Journal entries
    """

    def __init__(self, tracker: QuestTracker) -> None:
        self._tracker = tracker
        self._entries: dict[str, JournalEntry] = {}
        self._pinned: set[str] = set()
        self._read: set[str] = set()
        self._hud_tracker = HUDTracker()
        self._markers: dict[str, WorldMarker] = {}
        self._on_entry_update: Callable[[JournalEntry], None] | None = None
        self._on_marker_update: Callable[[WorldMarker], None] | None = None

        # Register for quest events
        tracker.add_listener(self._on_quest_event)

    @property
    def hud(self) -> HUDTracker:
        """Get the HUD tracker."""
        return self._hud_tracker

    @property
    def entries(self) -> list[JournalEntry]:
        """Get all journal entries."""
        return list(self._entries.values())

    @property
    def markers(self) -> list[WorldMarker]:
        """Get all world markers."""
        return list(self._markers.values())

    def set_entry_update_callback(
        self,
        callback: Callable[[JournalEntry], None]
    ) -> None:
        """Set callback for entry updates."""
        self._on_entry_update = callback

    def set_marker_update_callback(
        self,
        callback: Callable[[WorldMarker], None]
    ) -> None:
        """Set callback for marker updates."""
        self._on_marker_update = callback

    def _on_quest_event(self, event: QuestEvent) -> None:
        """Handle quest events."""
        quest_id = event.quest_id

        if event.event_type == QuestEventType.QUEST_AVAILABLE:
            self._update_entry(quest_id)

        elif event.event_type == QuestEventType.QUEST_ACCEPTED:
            self._update_entry(quest_id)
            self._create_objective_markers(quest_id)

        elif event.event_type == QuestEventType.QUEST_PROGRESS:
            self._update_entry(quest_id)

        elif event.event_type == QuestEventType.QUEST_OBJECTIVE_COMPLETE:
            self._update_entry(quest_id)
            self._update_markers(quest_id)

        elif event.event_type == QuestEventType.QUEST_COMPLETE:
            self._update_entry(quest_id)
            self._update_markers(quest_id)

        elif event.event_type == QuestEventType.QUEST_TURNED_IN:
            self._update_entry(quest_id)
            self._remove_markers(quest_id)
            self._hud_tracker.untrack_quest(quest_id)

        elif event.event_type == QuestEventType.QUEST_FAILED:
            self._update_entry(quest_id)
            self._remove_markers(quest_id)
            self._hud_tracker.untrack_quest(quest_id)

        elif event.event_type == QuestEventType.QUEST_ABANDONED:
            self._update_entry(quest_id)
            self._remove_markers(quest_id)
            self._hud_tracker.untrack_quest(quest_id)

    def _update_entry(self, quest_id: str) -> None:
        """Update or create a journal entry for a quest."""
        tracked = self._tracker.get_tracked(quest_id)
        if tracked is None:
            return

        is_tracked = quest_id in self._hud_tracker.tracked_quest_ids
        entry = JournalEntry.from_tracked_quest(tracked, is_tracked)
        entry.is_pinned = quest_id in self._pinned
        entry.read = quest_id in self._read

        self._entries[quest_id] = entry

        # Update HUD tracker if tracked
        if is_tracked:
            self._hud_tracker.update_entry(entry)

        if self._on_entry_update:
            self._on_entry_update(entry)

    def _create_objective_markers(self, quest_id: str) -> None:
        """Create world markers for quest objectives."""
        tracked = self._tracker.get_tracked(quest_id)
        if tracked is None:
            return

        quest_def = tracked.quest.definition

        # Create turn-in marker
        if quest_def.turn_in_id or quest_def.giver_id:
            turn_in_id = quest_def.turn_in_id or quest_def.giver_id
            marker = WorldMarker(
                id=f"{quest_id}_turnin",
                marker_type=MarkerType.QUEST_TURN_IN,
                quest_id=quest_id,
                label=f"Turn in: {tracked.quest.name}",
                visible=False,  # Show when complete
            )
            self._markers[marker.id] = marker

        # Create objective markers (positions would be set by game logic)
        for obj in tracked.objectives:
            marker_type = self._get_marker_type_for_objective(obj)
            marker = WorldMarker(
                id=f"{quest_id}_{obj.id}",
                marker_type=marker_type,
                quest_id=quest_id,
                objective_id=obj.id,
                label=obj.description,
                visible=not obj.hidden,
            )
            self._markers[marker.id] = marker

            if self._on_marker_update:
                self._on_marker_update(marker)

    def _get_marker_type_for_objective(self, obj: Objective) -> MarkerType:
        """Determine marker type from objective type."""
        from .objectives import ObjectiveType

        type_map = {
            ObjectiveType.KILL: MarkerType.ENEMY,
            ObjectiveType.COLLECT: MarkerType.ITEM,
            ObjectiveType.TALK: MarkerType.QUEST_GIVER,
            ObjectiveType.REACH: MarkerType.AREA,
            ObjectiveType.ESCORT: MarkerType.ESCORT_NPC,
            ObjectiveType.INTERACT: MarkerType.INTERACT,
        }
        return type_map.get(obj.objective_type, MarkerType.WAYPOINT)

    def _update_markers(self, quest_id: str) -> None:
        """Update markers for a quest."""
        tracked = self._tracker.get_tracked(quest_id)
        if tracked is None:
            return

        # Show turn-in marker when complete
        turn_in_marker_id = f"{quest_id}_turnin"
        if turn_in_marker_id in self._markers:
            marker = self._markers[turn_in_marker_id]
            marker.visible = tracked.quest.state == QuestState.COMPLETE
            if self._on_marker_update:
                self._on_marker_update(marker)

        # Update objective markers
        for obj in tracked.objectives:
            marker_id = f"{quest_id}_{obj.id}"
            if marker_id in self._markers:
                marker = self._markers[marker_id]
                marker.visible = not obj.is_complete and not obj.hidden
                if self._on_marker_update:
                    self._on_marker_update(marker)

    def _remove_markers(self, quest_id: str) -> None:
        """Remove all markers for a quest."""
        to_remove = [
            mid for mid in self._markers
            if self._markers[mid].quest_id == quest_id
        ]
        for marker_id in to_remove:
            del self._markers[marker_id]

    def get_entry(self, quest_id: str) -> JournalEntry | None:
        """Get a journal entry by quest ID."""
        return self._entries.get(quest_id)

    def get_entries(
        self,
        filter: JournalFilter | None = None,
        sort_by: str = "title"
    ) -> list[JournalEntry]:
        """
        Get filtered and sorted journal entries.

        Args:
            filter: Optional filter configuration
            sort_by: Sort field (title, level, progress, accepted_time)

        Returns:
            List of matching journal entries
        """
        entries = list(self._entries.values())

        # Apply filter
        if filter is not None:
            entries = [e for e in entries if filter.matches(e)]

        # Sort
        if sort_by == "title":
            entries.sort(key=lambda e: e.title.lower())
        elif sort_by == "level":
            entries.sort(key=lambda e: e.level_requirement)
        elif sort_by == "progress":
            entries.sort(key=lambda e: e.total_progress, reverse=True)
        elif sort_by == "accepted_time":
            entries.sort(
                key=lambda e: e.accepted_time or 0,
                reverse=True
            )
        elif sort_by == "category":
            entries.sort(key=lambda e: e.category.value)

        # Pinned entries first
        entries.sort(key=lambda e: not e.is_pinned)

        return entries

    def get_entries_by_category(
        self,
        category: JournalCategory
    ) -> list[JournalEntry]:
        """Get entries in a specific category."""
        filter = JournalFilter(categories={category})
        return self.get_entries(filter)

    def pin_quest(self, quest_id: str) -> bool:
        """Pin a quest to the top of the journal."""
        if quest_id not in self._entries:
            return False
        self._pinned.add(quest_id)
        self._entries[quest_id].is_pinned = True
        return True

    def unpin_quest(self, quest_id: str) -> bool:
        """Unpin a quest."""
        if quest_id not in self._pinned:
            return False
        self._pinned.discard(quest_id)
        if quest_id in self._entries:
            self._entries[quest_id].is_pinned = False
        return True

    def mark_read(self, quest_id: str) -> None:
        """Mark a quest as read."""
        self._read.add(quest_id)
        if quest_id in self._entries:
            self._entries[quest_id].read = True

    def track_quest(self, quest_id: str) -> bool:
        """Track a quest on the HUD."""
        entry = self._entries.get(quest_id)
        if entry is None:
            return False
        return self._hud_tracker.track_quest(quest_id, entry)

    def untrack_quest(self, quest_id: str) -> bool:
        """Untrack a quest from the HUD."""
        return self._hud_tracker.untrack_quest(quest_id)

    def get_marker(self, marker_id: str) -> WorldMarker | None:
        """Get a world marker by ID."""
        return self._markers.get(marker_id)

    def get_markers_for_quest(self, quest_id: str) -> list[WorldMarker]:
        """Get all markers for a quest."""
        return [m for m in self._markers.values() if m.quest_id == quest_id]

    def get_visible_markers(self) -> list[WorldMarker]:
        """Get all visible markers."""
        return [m for m in self._markers.values() if m.visible]

    def set_marker_position(
        self,
        marker_id: str,
        x: float,
        y: float,
        z: float
    ) -> bool:
        """Set position for a marker."""
        marker = self._markers.get(marker_id)
        if marker is None:
            return False
        marker.set_position(x, y, z)
        if self._on_marker_update:
            self._on_marker_update(marker)
        return True

    def add_custom_marker(self, marker: WorldMarker) -> None:
        """Add a custom marker."""
        self._markers[marker.id] = marker
        if self._on_marker_update:
            self._on_marker_update(marker)

    def remove_marker(self, marker_id: str) -> bool:
        """Remove a marker."""
        if marker_id not in self._markers:
            return False
        del self._markers[marker_id]
        return True

    def serialize(self) -> dict[str, Any]:
        """Serialize journal state for saving."""
        return {
            "pinned": list(self._pinned),
            "read": list(self._read),
            "tracked": self._hud_tracker.tracked_quest_ids,
        }

    def deserialize(self, data: dict[str, Any]) -> None:
        """Deserialize journal state from saved data."""
        self._pinned = set(data.get("pinned", []))
        self._read = set(data.get("read", []))

        # Restore HUD tracker
        for quest_id in data.get("tracked", []):
            if quest_id in self._entries:
                self._hud_tracker.track_quest(quest_id, self._entries[quest_id])

        # Update entry states
        for entry in self._entries.values():
            entry.is_pinned = entry.quest_id in self._pinned
            entry.read = entry.quest_id in self._read
