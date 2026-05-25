"""
Ghost System - Ghost replay for racing and speedrun comparisons.

Provides ghost replay functionality for comparing player performance
across multiple attempts or against other players.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, Optional


class GhostRenderMode(Enum):
    """Rendering modes for ghosts."""
    SOLID = auto()  # Fully visible
    TRANSPARENT = auto()  # Semi-transparent
    OUTLINE = auto()  # Outline only
    SILHOUETTE = auto()  # Solid silhouette
    TRAIL = auto()  # Motion trail effect
    HIDDEN = auto()  # Not rendered


@dataclass
class GhostFrame:
    """A single frame of ghost data."""
    frame: int
    timestamp: float
    position: tuple[float, float, float]  # x, y, z
    rotation: tuple[float, float, float, float]  # quaternion
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    animation_state: Optional[str] = None
    custom_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class GhostConfig:
    """Configuration for ghost display.

    Default values are sourced from the centralized config module
    (engine/tooling/replay/config.py) for consistency across the replay system.
    Use GhostConfig.from_defaults() to create with values from config.py.
    """
    # Rendering - defaults from config.py constants
    render_mode: GhostRenderMode = GhostRenderMode.TRANSPARENT
    opacity: float = 0.5  # DEFAULT_GHOST_OPACITY
    color: tuple[int, int, int] = (100, 100, 255)  # DEFAULT_GHOST_COLOR
    outline_color: tuple[int, int, int] = (255, 255, 255)  # DEFAULT_GHOST_OUTLINE_COLOR
    outline_width: float = 2.0  # DEFAULT_GHOST_OUTLINE_WIDTH

    # Trail settings (for TRAIL mode) - from config.py
    trail_length: int = 30  # DEFAULT_GHOST_TRAIL_LENGTH
    trail_fade: bool = True

    # Time offset
    time_offset: float = 0.0  # Seconds (positive = ghost ahead)

    # Interpolation
    interpolate: bool = True
    interpolation_method: str = "linear"  # linear, cubic, hermite

    # Visibility - from config.py
    visible: bool = True
    visible_distance: float = 100.0  # DEFAULT_GHOST_VISIBLE_DISTANCE
    fade_distance: float = 80.0  # DEFAULT_GHOST_FADE_DISTANCE

    # Labels
    show_label: bool = True
    label_text: Optional[str] = None

    # Comparison
    show_time_difference: bool = True
    show_position_marker: bool = False

    @classmethod
    def from_defaults(cls) -> 'GhostConfig':
        """Create GhostConfig with values from centralized config.py.

        This factory method ensures consistency with the replay system's
        centralized configuration constants.

        Returns:
            GhostConfig instance with values from config.py
        """
        from .config import (
            DEFAULT_GHOST_OPACITY,
            DEFAULT_GHOST_COLOR,
            DEFAULT_GHOST_OUTLINE_COLOR,
            DEFAULT_GHOST_OUTLINE_WIDTH,
            DEFAULT_GHOST_TRAIL_LENGTH,
            DEFAULT_GHOST_VISIBLE_DISTANCE,
            DEFAULT_GHOST_FADE_DISTANCE,
        )
        return cls(
            opacity=DEFAULT_GHOST_OPACITY,
            color=DEFAULT_GHOST_COLOR,
            outline_color=DEFAULT_GHOST_OUTLINE_COLOR,
            outline_width=DEFAULT_GHOST_OUTLINE_WIDTH,
            trail_length=DEFAULT_GHOST_TRAIL_LENGTH,
            visible_distance=DEFAULT_GHOST_VISIBLE_DISTANCE,
            fade_distance=DEFAULT_GHOST_FADE_DISTANCE,
        )


@dataclass
class GhostComparison:
    """Comparison data between player and ghost."""
    ghost_id: str
    current_time_difference: float  # Positive = player ahead
    total_time_difference: float
    current_distance: float
    closest_approach: float
    furthest_separation: float
    lead_changes: int
    player_best_segment: Optional[int] = None
    ghost_best_segment: Optional[int] = None


@dataclass
class Ghost:
    """A ghost replay entity."""
    id: str
    name: str
    frames: list[GhostFrame]
    config: GhostConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    # Recording info
    total_time: float = 0.0
    total_frames: int = 0
    checkpoints: list[tuple[int, float]] = field(default_factory=list)  # (frame, time)

    def __post_init__(self):
        """Initialize derived fields."""
        if self.frames:
            self.total_frames = len(self.frames)
            self.total_time = self.frames[-1].timestamp if self.frames else 0.0

    @property
    def duration(self) -> float:
        """Get ghost duration."""
        return self.total_time

    @property
    def frame_count(self) -> int:
        """Get total frame count."""
        return len(self.frames)

    def get_frame(self, frame_num: int) -> Optional[GhostFrame]:
        """Get frame by number.

        Args:
            frame_num: Frame number

        Returns:
            Ghost frame, or None if out of range
        """
        if 0 <= frame_num < len(self.frames):
            return self.frames[frame_num]
        return None

    def get_frame_at_time(self, time: float) -> Optional[GhostFrame]:
        """Get frame at specific time.

        Args:
            time: Timestamp in seconds

        Returns:
            Nearest frame, or None if no frames
        """
        if not self.frames:
            return None

        # Binary search for nearest frame
        left, right = 0, len(self.frames) - 1
        while left < right:
            mid = (left + right) // 2
            if self.frames[mid].timestamp < time:
                left = mid + 1
            else:
                right = mid

        return self.frames[left]

    def get_interpolated_state(
        self,
        time: float,
        method: str = "linear"
    ) -> Optional[dict[str, Any]]:
        """Get interpolated state at time.

        Args:
            time: Timestamp in seconds
            method: Interpolation method

        Returns:
            Interpolated state dictionary
        """
        if not self.frames or len(self.frames) < 2:
            return None

        # Apply time offset
        time += self.config.time_offset

        # Find surrounding frames
        frame_before = None
        frame_after = None

        for i, frame in enumerate(self.frames):
            if frame.timestamp <= time:
                frame_before = frame
            if frame.timestamp > time and frame_after is None:
                frame_after = frame
                break

        if frame_before is None:
            frame_before = self.frames[0]
        if frame_after is None:
            frame_after = self.frames[-1]

        # Calculate interpolation factor
        if frame_before.timestamp == frame_after.timestamp:
            t = 0.0
        else:
            t = (time - frame_before.timestamp) / (frame_after.timestamp - frame_before.timestamp)
            t = max(0.0, min(1.0, t))

        # Interpolate position
        if method == "linear":
            position = self._lerp_tuple(frame_before.position, frame_after.position, t)
        else:
            position = self._lerp_tuple(frame_before.position, frame_after.position, t)

        # Interpolate rotation (slerp for quaternions)
        rotation = self._slerp_quat(frame_before.rotation, frame_after.rotation, t)

        return {
            'position': position,
            'rotation': rotation,
            'velocity': self._lerp_tuple(frame_before.velocity, frame_after.velocity, t),
            'animation_state': frame_before.animation_state,
            'timestamp': time,
        }

    def add_checkpoint(self, frame: int, time: float) -> None:
        """Add a checkpoint.

        Args:
            frame: Frame number
            time: Checkpoint time
        """
        self.checkpoints.append((frame, time))
        self.checkpoints.sort(key=lambda x: x[0])

    @staticmethod
    def _lerp_tuple(a: tuple, b: tuple, t: float) -> tuple:
        """Linear interpolation between tuples."""
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(len(a)))

    @staticmethod
    def _slerp_quat(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float],
        t: float
    ) -> tuple[float, float, float, float]:
        """Spherical linear interpolation for quaternions."""
        # Compute dot product
        dot = sum(a * b for a, b in zip(q1, q2))

        # If negative, negate one quaternion
        if dot < 0:
            q2 = tuple(-x for x in q2)
            dot = -dot

        # Clamp dot
        dot = min(1.0, max(-1.0, dot))

        # Linear interpolation for very similar quaternions
        if dot > 0.9995:
            result = tuple(q1[i] + (q2[i] - q1[i]) * t for i in range(4))
            # Normalize
            mag = sum(x * x for x in result) ** 0.5
            return tuple(x / mag for x in result)

        # Slerp
        import math
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)

        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return tuple(q1[i] * s0 + q2[i] * s1 for i in range(4))


class GhostSystem:
    """Manages ghost replays for racing/speedrun comparisons.

    Provides functionality to record, play back, and compare ghost
    data for performance analysis.
    """
    __slots__ = (
        '_ghosts', '_active_ghosts', '_player_frames',
        '_is_recording', '_recording_start_time', '_current_time',
        '_comparisons', '_callbacks', '_next_ghost_id'
    )

    def __init__(self):
        """Initialize ghost system."""
        self._ghosts: dict[str, Ghost] = {}
        self._active_ghosts: set[str] = set()
        self._player_frames: list[GhostFrame] = []
        self._is_recording = False
        self._recording_start_time = 0.0
        self._current_time = 0.0
        self._comparisons: dict[str, GhostComparison] = {}
        self._callbacks: dict[str, list[Callable]] = {
            'ghost_added': [],
            'ghost_removed': [],
            'lead_change': [],
            'checkpoint_reached': [],
        }
        self._next_ghost_id = 0

    @property
    def ghost_count(self) -> int:
        """Get total number of ghosts."""
        return len(self._ghosts)

    @property
    def active_ghost_count(self) -> int:
        """Get number of active ghosts."""
        return len(self._active_ghosts)

    @property
    def is_recording(self) -> bool:
        """Check if recording player ghost."""
        return self._is_recording

    def add_ghost(
        self,
        frames: list[GhostFrame],
        name: str = "",
        config: Optional[GhostConfig] = None,
        metadata: Optional[dict] = None
    ) -> Ghost:
        """Add a ghost to the system.

        Args:
            frames: List of ghost frames
            name: Ghost name
            config: Ghost configuration
            metadata: Optional metadata

        Returns:
            The created ghost
        """
        ghost_id = self._generate_id()
        ghost = Ghost(
            id=ghost_id,
            name=name or f"Ghost {ghost_id}",
            frames=frames,
            config=config or GhostConfig(),
            metadata=metadata or {}
        )

        self._ghosts[ghost_id] = ghost
        self._notify('ghost_added', ghost)

        return ghost

    def remove_ghost(self, ghost_id: str) -> bool:
        """Remove a ghost from the system.

        Args:
            ghost_id: Ghost ID

        Returns:
            True if ghost was removed
        """
        ghost = self._ghosts.pop(ghost_id, None)
        if ghost:
            self._active_ghosts.discard(ghost_id)
            self._comparisons.pop(ghost_id, None)
            self._notify('ghost_removed', ghost)
            return True
        return False

    def get_ghost(self, ghost_id: str) -> Optional[Ghost]:
        """Get a ghost by ID.

        Args:
            ghost_id: Ghost ID

        Returns:
            Ghost if found
        """
        return self._ghosts.get(ghost_id)

    def activate_ghost(self, ghost_id: str) -> bool:
        """Activate a ghost for display.

        Args:
            ghost_id: Ghost ID

        Returns:
            True if ghost was activated
        """
        if ghost_id in self._ghosts:
            self._active_ghosts.add(ghost_id)
            self._comparisons[ghost_id] = GhostComparison(
                ghost_id=ghost_id,
                current_time_difference=0.0,
                total_time_difference=0.0,
                current_distance=0.0,
                closest_approach=float('inf'),
                furthest_separation=0.0,
                lead_changes=0
            )
            return True
        return False

    def deactivate_ghost(self, ghost_id: str) -> bool:
        """Deactivate a ghost.

        Args:
            ghost_id: Ghost ID

        Returns:
            True if ghost was deactivated
        """
        if ghost_id in self._active_ghosts:
            self._active_ghosts.remove(ghost_id)
            self._comparisons.pop(ghost_id, None)
            return True
        return False

    def start_recording(self) -> None:
        """Start recording player ghost data."""
        self._is_recording = True
        self._player_frames.clear()
        self._recording_start_time = self._current_time

    def stop_recording(self) -> Ghost:
        """Stop recording and create ghost from recorded data.

        Returns:
            The created ghost from recording
        """
        self._is_recording = False

        # Create ghost from recorded frames
        ghost = self.add_ghost(
            frames=list(self._player_frames),
            name="Player Ghost",
            metadata={
                'recorded_at': self._current_time,
                'source': 'player_recording'
            }
        )

        self._player_frames.clear()
        return ghost

    def record_frame(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float, float],
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        animation_state: Optional[str] = None,
        custom_data: Optional[dict] = None
    ) -> None:
        """Record a frame of player data.

        Args:
            position: Player position (x, y, z)
            rotation: Player rotation quaternion
            velocity: Player velocity
            animation_state: Current animation
            custom_data: Additional custom data
        """
        if not self._is_recording:
            return

        frame = GhostFrame(
            frame=len(self._player_frames),
            timestamp=self._current_time - self._recording_start_time,
            position=position,
            rotation=rotation,
            velocity=velocity,
            animation_state=animation_state,
            custom_data=custom_data or {}
        )

        self._player_frames.append(frame)

    def update(
        self,
        delta_time: float,
        player_position: tuple[float, float, float],
        player_time: Optional[float] = None
    ) -> dict[str, dict[str, Any]]:
        """Update ghost system.

        Args:
            delta_time: Time since last update
            player_position: Current player position
            player_time: Optional player race time

        Returns:
            Dictionary of ghost states by ghost ID
        """
        self._current_time += delta_time

        ghost_states = {}

        for ghost_id in self._active_ghosts:
            ghost = self._ghosts.get(ghost_id)
            if not ghost:
                continue

            # Get ghost state at current time
            state = ghost.get_interpolated_state(
                self._current_time,
                ghost.config.interpolation_method
            )

            if state:
                ghost_states[ghost_id] = state

                # Update comparison
                self._update_comparison(ghost_id, ghost, state, player_position, player_time)

        return ghost_states

    def get_ghost_state(self, ghost_id: str, time: Optional[float] = None) -> Optional[dict[str, Any]]:
        """Get ghost state at specific time.

        Args:
            ghost_id: Ghost ID
            time: Time to query (default: current time)

        Returns:
            Ghost state dictionary
        """
        ghost = self._ghosts.get(ghost_id)
        if not ghost:
            return None

        query_time = time if time is not None else self._current_time
        return ghost.get_interpolated_state(query_time, ghost.config.interpolation_method)

    def get_comparison(self, ghost_id: str) -> Optional[GhostComparison]:
        """Get comparison data for a ghost.

        Args:
            ghost_id: Ghost ID

        Returns:
            Comparison data if available
        """
        return self._comparisons.get(ghost_id)

    def get_all_comparisons(self) -> dict[str, GhostComparison]:
        """Get all active comparisons.

        Returns:
            Dictionary of comparisons by ghost ID
        """
        return self._comparisons.copy()

    def set_ghost_time_offset(self, ghost_id: str, offset: float) -> None:
        """Set time offset for a ghost.

        Args:
            ghost_id: Ghost ID
            offset: Time offset in seconds
        """
        ghost = self._ghosts.get(ghost_id)
        if ghost:
            ghost.config.time_offset = offset

    def set_ghost_render_mode(self, ghost_id: str, mode: GhostRenderMode) -> None:
        """Set render mode for a ghost.

        Args:
            ghost_id: Ghost ID
            mode: Render mode
        """
        ghost = self._ghosts.get(ghost_id)
        if ghost:
            ghost.config.render_mode = mode

    def set_ghost_opacity(self, ghost_id: str, opacity: float) -> None:
        """Set opacity for a ghost.

        Args:
            ghost_id: Ghost ID
            opacity: Opacity (0.0 to 1.0)
        """
        ghost = self._ghosts.get(ghost_id)
        if ghost:
            ghost.config.opacity = max(0.0, min(1.0, opacity))

    def get_best_ghost(self) -> Optional[Ghost]:
        """Get the ghost with the best time.

        Returns:
            Best ghost, or None if no ghosts
        """
        if not self._ghosts:
            return None

        return min(self._ghosts.values(), key=lambda g: g.total_time)

    def iter_ghosts(self) -> Iterator[Ghost]:
        """Iterate over all ghosts.

        Yields:
            Ghost objects
        """
        yield from self._ghosts.values()

    def iter_active_ghosts(self) -> Iterator[Ghost]:
        """Iterate over active ghosts.

        Yields:
            Active ghost objects
        """
        for ghost_id in self._active_ghosts:
            ghost = self._ghosts.get(ghost_id)
            if ghost:
                yield ghost

    def clear_ghosts(self) -> None:
        """Remove all ghosts."""
        self._ghosts.clear()
        self._active_ghosts.clear()
        self._comparisons.clear()

    def serialize_ghost(self, ghost_id: str) -> Optional[bytes]:
        """Serialize a ghost to bytes.

        Args:
            ghost_id: Ghost ID

        Returns:
            Serialized bytes, or None if not found
        """
        import json
        import struct

        ghost = self._ghosts.get(ghost_id)
        if not ghost:
            return None

        # Serialize frames
        frame_data = []
        for frame in ghost.frames:
            frame_data.append({
                'frame': frame.frame,
                'timestamp': frame.timestamp,
                'position': frame.position,
                'rotation': frame.rotation,
                'velocity': frame.velocity,
                'animation_state': frame.animation_state,
                'custom_data': frame.custom_data,
            })

        data = {
            'id': ghost.id,
            'name': ghost.name,
            'frames': frame_data,
            'metadata': ghost.metadata,
            'checkpoints': ghost.checkpoints,
        }

        json_bytes = json.dumps(data).encode('utf-8')
        return struct.pack('<I', len(json_bytes)) + json_bytes

    def deserialize_ghost(self, data: bytes, config: Optional[GhostConfig] = None) -> Optional[Ghost]:
        """Deserialize a ghost from bytes.

        Args:
            data: Serialized bytes
            config: Optional ghost configuration

        Returns:
            Deserialized ghost
        """
        import json
        import struct

        length = struct.unpack('<I', data[:4])[0]
        json_bytes = data[4:4 + length]
        ghost_data = json.loads(json_bytes.decode('utf-8'))

        frames = []
        for f in ghost_data['frames']:
            frames.append(GhostFrame(
                frame=f['frame'],
                timestamp=f['timestamp'],
                position=tuple(f['position']),
                rotation=tuple(f['rotation']),
                velocity=tuple(f['velocity']),
                animation_state=f.get('animation_state'),
                custom_data=f.get('custom_data', {}),
            ))

        ghost = Ghost(
            id=ghost_data['id'],
            name=ghost_data['name'],
            frames=frames,
            config=config or GhostConfig(),
            metadata=ghost_data.get('metadata', {}),
            checkpoints=[(c[0], c[1]) for c in ghost_data.get('checkpoints', [])]
        )

        self._ghosts[ghost.id] = ghost
        return ghost

    def on(self, event: str, callback: Callable) -> None:
        """Register event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def off(self, event: str, callback: Callable) -> None:
        """Unregister event callback.

        Args:
            event: Event name
            callback: Callback function
        """
        if event in self._callbacks:
            try:
                self._callbacks[event].remove(callback)
            except ValueError:
                pass

    def _update_comparison(
        self,
        ghost_id: str,
        ghost: Ghost,
        ghost_state: dict[str, Any],
        player_position: tuple[float, float, float],
        player_time: Optional[float]
    ) -> None:
        """Update comparison data for a ghost."""
        comparison = self._comparisons.get(ghost_id)
        if not comparison:
            return

        # Calculate distance
        ghost_pos = ghost_state['position']
        distance = (
            (player_position[0] - ghost_pos[0]) ** 2 +
            (player_position[1] - ghost_pos[1]) ** 2 +
            (player_position[2] - ghost_pos[2]) ** 2
        ) ** 0.5

        comparison.current_distance = distance
        comparison.closest_approach = min(comparison.closest_approach, distance)
        comparison.furthest_separation = max(comparison.furthest_separation, distance)

        # Time difference (if available)
        if player_time is not None:
            ghost_time = ghost_state['timestamp']
            old_diff = comparison.current_time_difference
            comparison.current_time_difference = player_time - ghost_time

            # Check for lead change
            if (old_diff < 0 and comparison.current_time_difference >= 0) or \
               (old_diff >= 0 and comparison.current_time_difference < 0):
                comparison.lead_changes += 1
                self._notify('lead_change', {
                    'ghost_id': ghost_id,
                    'player_ahead': comparison.current_time_difference >= 0
                })

    def _generate_id(self) -> str:
        """Generate unique ghost ID."""
        self._next_ghost_id += 1
        return f"ghost_{self._next_ghost_id}"

    def _notify(self, event: str, data: Any) -> None:
        """Notify event callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception:
                pass
