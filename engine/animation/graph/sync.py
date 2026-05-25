"""
Animation synchronization system.

Provides functionality for synchronizing animations:
- SyncGroup: Collection of nodes that sync their normalized time
- SyncMarker: Named markers in animations for sync points
- Phase matching: Synchronize locomotion cycles
- Event synchronization: Coordinate events across blended animations

Synchronization is essential for smooth blending between animations
with different durations, especially for locomotion cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from .animation_graph import (
    AnimationNode,
    GraphContext,
    Pose,
)
from .blend_node import AnimationClip, ClipNode
from .config import get_config


# =============================================================================
# SYNC MARKER
# =============================================================================


@dataclass
class SyncMarker:
    """
    A synchronization marker in an animation.

    Markers define specific points in an animation that should align
    with corresponding markers in other animations when blending.

    Common uses:
    - Foot plant markers for locomotion sync
    - Impact markers for combat sync
    - Beat markers for rhythmic animations
    """

    name: str
    normalized_time: float  # 0.0 - 1.0
    bone_index: Optional[int] = None  # Optional bone association
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.normalized_time = max(0.0, min(1.0, self.normalized_time))

    def get_time_for_duration(self, duration: float) -> float:
        """Get the absolute time for a given animation duration."""
        return self.normalized_time * duration

    def distance_to(self, other_normalized_time: float) -> float:
        """Get the shortest distance to another normalized time (considering wrap)."""
        direct = abs(self.normalized_time - other_normalized_time)
        wrapped = 1.0 - direct
        return min(direct, wrapped)


@dataclass
class SyncMarkerTrack:
    """A collection of sync markers for an animation."""

    markers: List[SyncMarker] = field(default_factory=list)

    def add_marker(self, marker: SyncMarker) -> None:
        """Add a marker to the track."""
        self.markers.append(marker)
        self.markers.sort(key=lambda m: m.normalized_time)

    def get_markers_by_name(self, name: str) -> List[SyncMarker]:
        """Get all markers with a specific name."""
        return [m for m in self.markers if m.name == name]

    def get_nearest_marker(self, normalized_time: float,
                           name: Optional[str] = None) -> Optional[SyncMarker]:
        """Get the nearest marker to a normalized time."""
        candidates = self.markers if name is None else self.get_markers_by_name(name)

        if not candidates:
            return None

        nearest = min(candidates, key=lambda m: m.distance_to(normalized_time))
        return nearest

    def get_markers_in_range(self, start: float, end: float) -> List[SyncMarker]:
        """Get markers within a normalized time range."""
        result = []
        for marker in self.markers:
            if start <= end:
                if start <= marker.normalized_time <= end:
                    result.append(marker)
            else:
                # Wrapped range (e.g., 0.9 to 0.1)
                if marker.normalized_time >= start or marker.normalized_time <= end:
                    result.append(marker)
        return result


# =============================================================================
# SYNC ENTRY
# =============================================================================


class SyncMode(Enum):
    """Synchronization modes."""

    NONE = auto()          # No synchronization
    NORMALIZED = auto()    # Match normalized time
    PHASE = auto()         # Match phase (considering markers)
    LEADER_FOLLOWER = auto()  # One animation leads, others follow
    WEIGHTED = auto()      # Weight-based sync


@dataclass
class SyncEntry:
    """An entry in a sync group."""

    node: AnimationNode
    weight: float = 1.0
    is_leader: bool = False
    marker_track: Optional[SyncMarkerTrack] = None
    duration: float = 1.0
    _current_time: float = 0.0
    _normalized_time: float = 0.0

    @property
    def normalized_time(self) -> float:
        """Get the current normalized time."""
        return self._normalized_time

    @normalized_time.setter
    def normalized_time(self, value: float) -> None:
        """Set the normalized time."""
        self._normalized_time = value % 1.0 if value > 1.0 else max(0.0, value)
        self._current_time = self._normalized_time * self.duration

    def advance(self, dt: float, speed: float = 1.0) -> None:
        """Advance the entry's time."""
        if self.duration <= 0:
            return
        self._current_time += dt * speed
        self._normalized_time = (self._current_time / self.duration) % 1.0


# =============================================================================
# SYNC GROUP
# =============================================================================


class SyncGroup:
    """
    A group of animations that synchronize their playback.

    Sync groups ensure that blended animations stay in phase,
    preventing visual artifacts like sliding feet.
    """

    def __init__(self, name: str, mode: SyncMode = SyncMode.NORMALIZED) -> None:
        self.name = name
        self.mode = mode
        self.entries: List[SyncEntry] = []
        self._leader_index: int = 0

    def add_entry(self, node: AnimationNode, weight: float = 1.0,
                  is_leader: bool = False, duration: float = 1.0,
                  marker_track: Optional[SyncMarkerTrack] = None) -> int:
        """Add an entry to the sync group."""
        entry = SyncEntry(
            node=node,
            weight=weight,
            is_leader=is_leader,
            duration=duration,
            marker_track=marker_track,
        )
        self.entries.append(entry)

        if is_leader:
            self._leader_index = len(self.entries) - 1

        return len(self.entries) - 1

    def remove_entry(self, index: int) -> bool:
        """Remove an entry by index."""
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            # Update leader index if needed
            if self._leader_index >= len(self.entries):
                self._leader_index = max(0, len(self.entries) - 1)
            return True
        return False

    def set_leader(self, index: int) -> bool:
        """Set the leader entry."""
        if 0 <= index < len(self.entries):
            for i, entry in enumerate(self.entries):
                entry.is_leader = (i == index)
            self._leader_index = index
            return True
        return False

    def set_weights(self, weights: List[float]) -> None:
        """Set weights for all entries."""
        for i, weight in enumerate(weights):
            if i < len(self.entries):
                self.entries[i].weight = max(0.0, weight)

    def get_leader(self) -> Optional[SyncEntry]:
        """Get the leader entry."""
        if self.entries and self._leader_index < len(self.entries):
            return self.entries[self._leader_index]
        return None

    def update(self, dt: float) -> None:
        """Update the sync group."""
        if not self.entries:
            return

        if self.mode == SyncMode.NONE:
            # No synchronization - each entry advances independently
            for entry in self.entries:
                entry.advance(dt)

        elif self.mode == SyncMode.NORMALIZED:
            # All entries match the weighted average normalized time
            self._sync_normalized(dt)

        elif self.mode == SyncMode.PHASE:
            # Sync based on markers/phase
            self._sync_phase(dt)

        elif self.mode == SyncMode.LEADER_FOLLOWER:
            # Leader advances, followers match
            self._sync_leader_follower(dt)

        elif self.mode == SyncMode.WEIGHTED:
            # Weight-based synchronization
            self._sync_weighted(dt)

    def _sync_normalized(self, dt: float) -> None:
        """Synchronize using normalized time."""
        # Calculate weighted average normalized time advancement
        total_weight = sum(entry.weight for entry in self.entries)
        if total_weight <= 0:
            total_weight = 1.0

        # Calculate average speed (normalized time per second)
        avg_speed = 0.0
        for entry in self.entries:
            if entry.duration > 0:
                speed = entry.weight / entry.duration
                avg_speed += speed
        avg_speed /= total_weight

        # Advance all entries by the same normalized amount
        normalized_advance = avg_speed * dt

        for entry in self.entries:
            entry.normalized_time = entry.normalized_time + normalized_advance

    def _sync_phase(self, dt: float) -> None:
        """Synchronize using phase/markers."""
        leader = self.get_leader()
        if not leader:
            self._sync_normalized(dt)
            return

        # Advance leader
        leader.advance(dt)

        # Find leader's current phase marker
        leader_marker = None
        if leader.marker_track:
            leader_marker = leader.marker_track.get_nearest_marker(
                leader.normalized_time
            )

        # Sync followers to nearest matching marker
        for entry in self.entries:
            if entry.is_leader:
                continue

            if leader_marker and entry.marker_track:
                # Find corresponding marker in follower
                follower_marker = entry.marker_track.get_nearest_marker(
                    entry.normalized_time, leader_marker.name
                )
                if follower_marker:
                    # Calculate target normalized time
                    target_time = follower_marker.normalized_time + \
                                  (leader.normalized_time - leader_marker.normalized_time)
                    target_time = target_time % 1.0
                    entry.normalized_time = target_time
                else:
                    entry.normalized_time = leader.normalized_time
            else:
                entry.normalized_time = leader.normalized_time

    def _sync_leader_follower(self, dt: float) -> None:
        """Synchronize with leader-follower pattern."""
        leader = self.get_leader()
        if not leader:
            self._sync_normalized(dt)
            return

        # Advance leader
        leader.advance(dt)

        # Match all followers to leader's normalized time
        for entry in self.entries:
            if not entry.is_leader:
                entry.normalized_time = leader.normalized_time

    def _sync_weighted(self, dt: float) -> None:
        """Synchronize using weight-based interpolation."""
        # Calculate target normalized time based on weights
        total_weight = sum(entry.weight for entry in self.entries)
        if total_weight <= 0:
            total_weight = 1.0

        # Calculate weighted average current time
        weighted_time = 0.0
        for entry in self.entries:
            weighted_time += entry.normalized_time * (entry.weight / total_weight)

        # Calculate weighted average speed
        weighted_speed = 0.0
        for entry in self.entries:
            if entry.duration > 0:
                speed = 1.0 / entry.duration
                weighted_speed += speed * (entry.weight / total_weight)

        # Advance all entries
        new_time = (weighted_time + weighted_speed * dt) % 1.0
        for entry in self.entries:
            entry.normalized_time = new_time

    def get_synchronized_time(self) -> float:
        """Get the current synchronized normalized time."""
        if not self.entries:
            return 0.0

        if self.mode == SyncMode.LEADER_FOLLOWER:
            leader = self.get_leader()
            return leader.normalized_time if leader else 0.0
        else:
            # Return weighted average
            total_weight = sum(entry.weight for entry in self.entries)
            if total_weight <= 0:
                return 0.0

            return sum(
                entry.normalized_time * (entry.weight / total_weight)
                for entry in self.entries
            )


# =============================================================================
# SYNC GROUP NODE
# =============================================================================


class SyncGroupNode(AnimationNode):
    """
    An animation node that synchronizes multiple child nodes.

    This node wraps a sync group and handles the synchronization
    during evaluation.
    """

    _abstract = False

    def __init__(self, node_id: str, mode: SyncMode = SyncMode.NORMALIZED) -> None:
        super().__init__(node_id)
        self.sync_group = SyncGroup(node_id, mode)
        self._last_dt: float = 0.0

    def add_entry(self, node: AnimationNode, weight: float = 1.0,
                  is_leader: bool = False, duration: float = 1.0,
                  marker_track: Optional[SyncMarkerTrack] = None) -> int:
        """Add an entry to synchronize."""
        return self.sync_group.add_entry(node, weight, is_leader, duration, marker_track)

    def set_weights(self, weights: List[float]) -> None:
        """Set weights for all entries."""
        self.sync_group.set_weights(weights)

    def evaluate(self, context: GraphContext) -> Pose:
        """Evaluate and synchronize all entries."""
        if not self.sync_group.entries:
            return Pose()

        # Update sync group
        self.sync_group.update(context.dt)

        # Evaluate all entries and blend
        total_weight = sum(entry.weight for entry in self.sync_group.entries)
        if total_weight <= 0:
            return Pose()

        result_pose = None

        for entry in self.sync_group.entries:
            if entry.weight <= 0:
                continue

            # Create context with synchronized time
            sync_context = GraphContext(
                parameters=context.parameters,
                dt=context.dt,
                skeleton=context.skeleton,
                bone_masks=context.bone_masks,
                normalized_time=entry.normalized_time,
                sync_group=self.sync_group.name,
                layer_weight=context.layer_weight,
            )

            pose = entry.node.evaluate(sync_context)
            if not pose:
                continue

            normalized_weight = entry.weight / total_weight

            if result_pose is None:
                result_pose = Pose.identity(pose.bone_count())

            # Blend pose with normalized weight
            for i in range(min(result_pose.bone_count(), pose.bone_count())):
                from .animation_graph import Transform
                blended = Transform.identity().lerp(pose.transforms[i], normalized_weight)
                # Accumulate
                if result_pose.transforms[i].position == Transform.identity().position:
                    result_pose.transforms[i] = blended
                else:
                    result_pose.transforms[i] = result_pose.transforms[i] + Transform(
                        position=tuple(
                            c * normalized_weight for c in pose.transforms[i].position
                        ),
                        rotation=blended.rotation,
                        scale=blended.scale,
                    )

        return result_pose or Pose()


# =============================================================================
# SYNC UTILITIES
# =============================================================================


def sync_animations(leader: ClipNode, followers: List[ClipNode],
                    mode: SyncMode = SyncMode.NORMALIZED) -> SyncGroup:
    """
    Create a sync group from a leader and followers.

    Convenience function for synchronizing animation clips.
    """
    group = SyncGroup("auto_sync", mode)

    # Add leader
    group.add_entry(
        node=leader,
        weight=1.0,
        is_leader=True,
        duration=leader.duration,
    )

    # Add followers
    for follower in followers:
        group.add_entry(
            node=follower,
            weight=1.0,
            is_leader=False,
            duration=follower.duration,
        )

    return group


def create_locomotion_markers(
    left_foot_plant: float = 0.0,
    right_foot_plant: float = 0.5,
    left_foot_pass: float = 0.25,
    right_foot_pass: float = 0.75,
) -> SyncMarkerTrack:
    """
    Create standard locomotion sync markers.

    Args:
        left_foot_plant: Normalized time when left foot plants
        right_foot_plant: Normalized time when right foot plants
        left_foot_pass: Normalized time when left foot passes
        right_foot_pass: Normalized time when right foot passes
    """
    track = SyncMarkerTrack()

    track.add_marker(SyncMarker(
        name="left_plant",
        normalized_time=left_foot_plant,
        metadata={"foot": "left", "type": "plant"},
    ))
    track.add_marker(SyncMarker(
        name="right_plant",
        normalized_time=right_foot_plant,
        metadata={"foot": "right", "type": "plant"},
    ))
    track.add_marker(SyncMarker(
        name="left_pass",
        normalized_time=left_foot_pass,
        metadata={"foot": "left", "type": "pass"},
    ))
    track.add_marker(SyncMarker(
        name="right_pass",
        normalized_time=right_foot_pass,
        metadata={"foot": "right", "type": "pass"},
    ))

    return track


def calculate_phase_offset(
    source_markers: SyncMarkerTrack,
    target_markers: SyncMarkerTrack,
    reference_marker_name: str = "left_plant",
) -> float:
    """
    Calculate the phase offset between two marker tracks.

    Returns the normalized time offset to align the reference markers.
    """
    source_marker = None
    target_marker = None

    for marker in source_markers.markers:
        if marker.name == reference_marker_name:
            source_marker = marker
            break

    for marker in target_markers.markers:
        if marker.name == reference_marker_name:
            target_marker = marker
            break

    if source_marker and target_marker:
        return target_marker.normalized_time - source_marker.normalized_time

    return 0.0


# =============================================================================
# EVENT SYNCHRONIZATION
# =============================================================================


@dataclass
class SyncEvent:
    """An event synchronized across animations."""

    name: str
    source_node_id: str
    normalized_time: float
    data: Dict[str, Any] = field(default_factory=dict)


class EventSynchronizer:
    """
    Synchronizes events across multiple animations.

    Ensures that events (like footsteps, impacts) are properly
    coordinated when blending animations.
    """

    def __init__(self) -> None:
        self.event_handlers: Dict[str, List[Callable[[SyncEvent], None]]] = {}
        self._pending_events: List[SyncEvent] = []

    def register_handler(self, event_name: str,
                         handler: Callable[[SyncEvent], None]) -> None:
        """Register a handler for an event type."""
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        self.event_handlers[event_name].append(handler)

    def unregister_handler(self, event_name: str,
                           handler: Callable[[SyncEvent], None]) -> bool:
        """Unregister a handler."""
        if event_name in self.event_handlers:
            try:
                self.event_handlers[event_name].remove(handler)
                return True
            except ValueError:
                pass
        return False

    def queue_event(self, event: SyncEvent) -> None:
        """Queue an event for processing."""
        self._pending_events.append(event)

    def process_events(self, sync_group: Optional[SyncGroup] = None) -> None:
        """Process all pending events."""
        # If we have a sync group, filter events based on sync
        events_to_process = list(self._pending_events)
        self._pending_events.clear()

        # Deduplicate events by name and approximate time
        seen: Set[Tuple[str, float]] = set()
        unique_events = []

        for event in events_to_process:
            # Round normalized time to prevent near-duplicate events
            rounded_time = round(event.normalized_time, 2)
            key = (event.name, rounded_time)

            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        # Dispatch events
        for event in unique_events:
            handlers = self.event_handlers.get(event.name, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception:
                    pass  # Log error in production

    def clear(self) -> None:
        """Clear all pending events."""
        self._pending_events.clear()


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Markers
    "SyncMarker",
    "SyncMarkerTrack",
    # Sync
    "SyncMode",
    "SyncEntry",
    "SyncGroup",
    "SyncGroupNode",
    # Utilities
    "sync_animations",
    "create_locomotion_markers",
    "calculate_phase_offset",
    # Events
    "SyncEvent",
    "EventSynchronizer",
]
