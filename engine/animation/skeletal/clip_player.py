"""Animation playback controller.

The ClipPlayer handles playback of animation clips including timing,
looping, speed control, events, and pose sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple


# =============================================================================
# Configuration Constants
# =============================================================================

# Minimum blend time to prevent division by zero (seconds)
MIN_BLEND_TIME = 1e-9

# Default playback speed
DEFAULT_PLAYBACK_SPEED = 1.0

# Default blend weight
DEFAULT_BLEND_WEIGHT = 1.0


if TYPE_CHECKING:
    from engine.animation.skeletal.clip import AnimationClip, AnimationEvent
    from engine.animation.skeletal.pose import Pose
    from engine.animation.skeletal.skeleton import Skeleton


class PlaybackMode(Enum):
    """How the animation plays."""

    FORWARD = auto()  # Play forward, stop or loop at end
    REVERSE = auto()  # Play backward, stop or loop at start
    PING_PONG = auto()  # Alternate between forward and reverse


class PlaybackState(Enum):
    """Current state of playback."""

    STOPPED = auto()  # Not playing
    PLAYING = auto()  # Currently playing
    PAUSED = auto()  # Paused at current time


def animation_data(cls):
    """Decorator for animation data classes."""
    cls._animation_data = True
    cls._animation_type = cls.__name__
    return cls


@animation_data
@dataclass
class PlaybackEvent:
    """Event fired during playback.

    Attributes:
        clip_name: Name of the clip.
        event_name: Name of the animation event.
        event_time: Time in clip when event fires.
        data: Optional event data.
    """

    clip_name: str
    event_name: str
    event_time: float
    data: Optional[dict] = None


EventCallback = Callable[[PlaybackEvent], None]


@animation_data
class ClipPlayer:
    """Animation clip playback controller.

    Manages playback timing, looping, speed, and event firing for
    a single animation clip.

    Attributes:
        clip: The animation clip being played.
        time: Current playback time in seconds.
        speed: Playback speed multiplier (1.0 = normal).
        looping: Whether to loop at end/start.
        weight: Blend weight for this player [0, 1].
        mode: Playback mode (forward, reverse, ping-pong).
    """

    def __init__(
        self,
        clip: AnimationClip,
        looping: Optional[bool] = None,
        speed: float = 1.0,
        weight: float = 1.0,
        mode: PlaybackMode = PlaybackMode.FORWARD,
    ) -> None:
        """Initialize clip player.

        Args:
            clip: The animation clip to play.
            looping: Override clip's looping setting (None uses clip default).
            speed: Initial playback speed.
            weight: Initial blend weight.
            mode: Initial playback mode.
        """
        self._clip = clip
        self._time = 0.0
        self._speed = speed
        self._weight = max(0.0, min(1.0, weight))
        self._looping = clip.looping if looping is None else looping
        self._mode = mode
        self._state = PlaybackState.STOPPED
        self._direction = 1.0  # 1.0 = forward, -1.0 = reverse
        self._event_callbacks: List[EventCallback] = []
        self._last_event_time = -1.0  # For tracking which events fired
        self._loop_count = 0

        # Set initial direction based on mode
        if mode == PlaybackMode.REVERSE:
            self._direction = -1.0
            self._time = clip.duration

    @property
    def clip(self) -> AnimationClip:
        """Get the animation clip."""
        return self._clip

    @property
    def time(self) -> float:
        """Get current playback time."""
        return self._time

    @property
    def speed(self) -> float:
        """Get playback speed."""
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        """Set playback speed."""
        self._speed = value

    @property
    def weight(self) -> float:
        """Get blend weight."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        """Set blend weight."""
        self._weight = max(0.0, min(1.0, value))

    @property
    def looping(self) -> bool:
        """Get looping flag."""
        return self._looping

    @looping.setter
    def looping(self, value: bool) -> None:
        """Set looping flag."""
        self._looping = value

    @property
    def mode(self) -> PlaybackMode:
        """Get playback mode."""
        return self._mode

    @mode.setter
    def mode(self, value: PlaybackMode) -> None:
        """Set playback mode."""
        self._mode = value
        if value == PlaybackMode.REVERSE:
            self._direction = -1.0
        elif value == PlaybackMode.FORWARD:
            self._direction = 1.0

    @property
    def state(self) -> PlaybackState:
        """Get current playback state."""
        return self._state

    @property
    def is_playing(self) -> bool:
        """Check if currently playing."""
        return self._state == PlaybackState.PLAYING

    @property
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self._state == PlaybackState.PAUSED

    @property
    def is_stopped(self) -> bool:
        """Check if currently stopped."""
        return self._state == PlaybackState.STOPPED

    @property
    def duration(self) -> float:
        """Get clip duration."""
        return self._clip.duration

    @property
    def normalized_time(self) -> float:
        """Get time as fraction of duration [0, 1]."""
        if self._clip.duration <= 0:
            return 0.0
        return self._time / self._clip.duration

    @property
    def loop_count(self) -> int:
        """Get number of times clip has looped."""
        return self._loop_count

    @property
    def is_at_end(self) -> bool:
        """Check if playback is at the end."""
        return self._time >= self._clip.duration

    @property
    def is_at_start(self) -> bool:
        """Check if playback is at the start."""
        return self._time <= 0.0

    def play(self) -> None:
        """Start or resume playback."""
        if self._state == PlaybackState.STOPPED:
            self._last_event_time = -1.0
            self._loop_count = 0
        self._state = PlaybackState.PLAYING

    def pause(self) -> None:
        """Pause playback at current time."""
        if self._state == PlaybackState.PLAYING:
            self._state = PlaybackState.PAUSED

    def stop(self) -> None:
        """Stop playback and reset to start."""
        self._state = PlaybackState.STOPPED
        self.reset()

    def reset(self) -> None:
        """Reset to initial state based on mode."""
        if self._mode == PlaybackMode.REVERSE:
            self._time = self._clip.duration
            self._direction = -1.0
        else:
            self._time = 0.0
            self._direction = 1.0
        self._last_event_time = -1.0
        self._loop_count = 0

    def set_time(self, time: float, fire_events: bool = False) -> None:
        """Set playback time.

        Args:
            time: New time value.
            fire_events: Whether to fire events crossed during the jump.
        """
        old_time = self._time
        self._time = max(0.0, min(time, self._clip.duration))

        if fire_events and self._state == PlaybackState.PLAYING:
            self._fire_events_in_range(old_time, self._time)

        self._last_event_time = self._time

    def set_normalized_time(self, t: float, fire_events: bool = False) -> None:
        """Set time as fraction of duration [0, 1].

        Args:
            t: Normalized time value.
            fire_events: Whether to fire crossed events.
        """
        self.set_time(t * self._clip.duration, fire_events)

    def update(self, dt: float) -> List[PlaybackEvent]:
        """Update playback by delta time.

        Args:
            dt: Time delta in seconds.

        Returns:
            List of events that fired during this update.
        """
        if self._state != PlaybackState.PLAYING:
            return []

        if self._clip.duration <= 0:
            return []

        fired_events: List[PlaybackEvent] = []

        # Calculate time step
        time_step = dt * self._speed * self._direction

        # Track for event firing
        old_time = self._time
        new_time = self._time + time_step

        # Handle looping/boundaries
        if self._direction > 0:  # Forward
            if new_time >= self._clip.duration:
                if self._looping:
                    # Fire events to end
                    fired_events.extend(self._get_events_in_range(old_time, self._clip.duration))

                    if self._mode == PlaybackMode.PING_PONG:
                        # Reverse direction
                        self._direction = -1.0
                        overflow = new_time - self._clip.duration
                        new_time = self._clip.duration - overflow
                        self._loop_count += 1
                    else:
                        # Loop back to start
                        overflow = new_time - self._clip.duration
                        new_time = overflow % self._clip.duration if self._clip.duration > 0 else 0.0
                        self._loop_count += 1
                        # Fire events from start
                        fired_events.extend(self._get_events_in_range(0.0, new_time))
                else:
                    # Stop at end
                    fired_events.extend(self._get_events_in_range(old_time, self._clip.duration))
                    new_time = self._clip.duration
                    self._state = PlaybackState.STOPPED
            else:
                fired_events.extend(self._get_events_in_range(old_time, new_time))

        else:  # Reverse
            if new_time <= 0:
                if self._looping:
                    # Fire events to start
                    fired_events.extend(self._get_events_in_range(old_time, 0.0))

                    if self._mode == PlaybackMode.PING_PONG:
                        # Reverse direction
                        self._direction = 1.0
                        new_time = -new_time
                        self._loop_count += 1
                    else:
                        # Loop back to end
                        new_time = self._clip.duration + new_time
                        self._loop_count += 1
                        # Fire events from end
                        fired_events.extend(self._get_events_in_range(self._clip.duration, new_time))
                else:
                    # Stop at start
                    fired_events.extend(self._get_events_in_range(old_time, 0.0))
                    new_time = 0.0
                    self._state = PlaybackState.STOPPED
            else:
                fired_events.extend(self._get_events_in_range(old_time, new_time))

        self._time = max(0.0, min(new_time, self._clip.duration))
        self._last_event_time = self._time

        # Call event callbacks
        for event in fired_events:
            for callback in self._event_callbacks:
                callback(event)

        return fired_events

    def _get_events_in_range(
        self, start_time: float, end_time: float
    ) -> List[PlaybackEvent]:
        """Get events that fire in a time range.

        Handles both forward and reverse ranges.
        """
        events = []

        # Determine direction and adjust range
        if start_time <= end_time:
            clip_events = self._clip.get_events_in_range(start_time, end_time)
        else:
            # Reverse: get events from end to start (in reverse)
            clip_events = [
                e for e in reversed(self._clip.events)
                if end_time <= e.time < start_time
            ]

        for clip_event in clip_events:
            events.append(
                PlaybackEvent(
                    clip_name=self._clip.name,
                    event_name=clip_event.name,
                    event_time=clip_event.time,
                    data=clip_event.data,
                )
            )

        return events

    def _fire_events_in_range(
        self, start_time: float, end_time: float
    ) -> None:
        """Fire events in a range (for seeking)."""
        events = self._get_events_in_range(start_time, end_time)
        for event in events:
            for callback in self._event_callbacks:
                callback(event)

    def sample_pose(self, skeleton: Skeleton) -> Pose:
        """Sample the pose at current time.

        Args:
            skeleton: Skeleton to sample for.

        Returns:
            Pose at current playback time.
        """
        return self._clip.sample_pose(skeleton, self._time)

    def sample_pose_at(self, skeleton: Skeleton, time: float) -> Pose:
        """Sample pose at specified time.

        Args:
            skeleton: Skeleton to sample for.
            time: Time to sample at.

        Returns:
            Pose at specified time.
        """
        clamped_time = max(0.0, min(time, self._clip.duration))
        return self._clip.sample_pose(skeleton, clamped_time)

    def add_event_callback(self, callback: EventCallback) -> None:
        """Add an event callback.

        Args:
            callback: Function to call when events fire.
        """
        if callback not in self._event_callbacks:
            self._event_callbacks.append(callback)

    def remove_event_callback(self, callback: EventCallback) -> None:
        """Remove an event callback.

        Args:
            callback: Callback to remove.
        """
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)

    def clear_event_callbacks(self) -> None:
        """Remove all event callbacks."""
        self._event_callbacks.clear()

    def get_root_motion_delta(self, dt: float) -> Tuple:
        """Get root motion delta for this frame.

        Args:
            dt: Time delta.

        Returns:
            Tuple of (translation_delta, rotation_delta).
        """
        if not self._clip.root_motion:
            from engine.core.math import Vec3, Quat
            return (Vec3.zero(), Quat.identity())

        old_time = self._time
        new_time = self._time + dt * self._speed * self._direction
        new_time = max(0.0, min(new_time, self._clip.duration))

        return self._clip.extract_root_motion(old_time, new_time)

    def blend_with(self, other: ClipPlayer, alpha: float, skeleton: Skeleton) -> Pose:
        """Blend this player's pose with another.

        Args:
            other: Other clip player.
            alpha: Blend factor (0 = this, 1 = other).
            skeleton: Skeleton to sample for.

        Returns:
            Blended pose.
        """
        from engine.animation.skeletal.pose import lerp_poses

        pose_a = self.sample_pose(skeleton)
        pose_b = other.sample_pose(skeleton)

        return lerp_poses(pose_a, pose_b, alpha)

    def copy(self) -> ClipPlayer:
        """Create a copy of this player.

        Returns:
            New player with same settings but independent state.
        """
        player = ClipPlayer(
            clip=self._clip,
            looping=self._looping,
            speed=self._speed,
            weight=self._weight,
            mode=self._mode,
        )
        player._time = self._time
        player._state = self._state
        player._direction = self._direction
        player._loop_count = self._loop_count
        player._last_event_time = self._last_event_time
        return player

    def __repr__(self) -> str:
        return (
            f"ClipPlayer('{self._clip.name}', time={self._time:.3f}/{self._clip.duration:.3f}, "
            f"state={self._state.name}, speed={self._speed}, loop={self._looping})"
        )


@animation_data
class ClipQueue:
    """Queue of clips to play in sequence.

    Useful for chaining animations together.
    """

    def __init__(self, skeleton: Skeleton) -> None:
        """Initialize clip queue.

        Args:
            skeleton: Skeleton to use for sampling.
        """
        self._skeleton = skeleton
        self._queue: List[ClipPlayer] = []
        self._current_index = 0
        self._blend_time = 0.0
        self._blend_progress = 0.0
        self._is_blending = False

    @property
    def skeleton(self) -> Skeleton:
        """Get skeleton."""
        return self._skeleton

    @property
    def current_player(self) -> Optional[ClipPlayer]:
        """Get current player."""
        if self._current_index < len(self._queue):
            return self._queue[self._current_index]
        return None

    @property
    def next_player(self) -> Optional[ClipPlayer]:
        """Get next player in queue."""
        next_idx = self._current_index + 1
        if next_idx < len(self._queue):
            return self._queue[next_idx]
        return None

    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    @property
    def is_blending(self) -> bool:
        """Check if currently blending between clips."""
        return self._is_blending

    @property
    def queue_length(self) -> int:
        """Get number of clips in queue."""
        return len(self._queue)

    def enqueue(
        self,
        clip: AnimationClip,
        blend_time: float = 0.0,
        speed: float = 1.0,
    ) -> ClipPlayer:
        """Add a clip to the queue.

        Args:
            clip: Clip to add.
            blend_time: Time to blend into this clip.
            speed: Playback speed.

        Returns:
            The created player.
        """
        player = ClipPlayer(clip, speed=speed)
        self._queue.append(player)

        if len(self._queue) == 1:
            player.play()

        return player

    def clear(self) -> None:
        """Clear all clips from queue."""
        self._queue.clear()
        self._current_index = 0
        self._is_blending = False
        self._blend_progress = 0.0

    def skip(self, blend_time: float = 0.0) -> None:
        """Skip to next clip in queue.

        Args:
            blend_time: Time to blend to next clip.
        """
        if self._current_index + 1 < len(self._queue):
            if blend_time > 0:
                self._blend_time = blend_time
                self._blend_progress = 0.0
                self._is_blending = True
                self._queue[self._current_index + 1].play()
            else:
                self._current_index += 1
                self._queue[self._current_index].play()
                self._is_blending = False

    def update(self, dt: float) -> List[PlaybackEvent]:
        """Update queue playback.

        Args:
            dt: Time delta.

        Returns:
            List of fired events.
        """
        events: List[PlaybackEvent] = []

        if not self._queue:
            return events

        current = self.current_player
        if current is None:
            return events

        # Update current player
        events.extend(current.update(dt))

        # Handle blending
        if self._is_blending:
            next_player = self.next_player
            if next_player:
                events.extend(next_player.update(dt))
                # Prevent division by zero using max with minimum blend time
                self._blend_progress += dt / max(self._blend_time, MIN_BLEND_TIME)
                if self._blend_progress >= 1.0:
                    self._current_index += 1
                    self._is_blending = False
                    self._blend_progress = 0.0

        # Check if current finished
        elif current.is_stopped and not current.looping:
            if self._current_index + 1 < len(self._queue):
                self._current_index += 1
                self._queue[self._current_index].play()

        return events

    def sample_pose(self) -> Optional[Pose]:
        """Sample current pose.

        Returns:
            Current pose, or None if queue is empty.
        """
        if not self._queue:
            return None

        current = self.current_player
        if current is None:
            return None

        if self._is_blending:
            next_player = self.next_player
            if next_player:
                return current.blend_with(next_player, self._blend_progress, self._skeleton)

        return current.sample_pose(self._skeleton)

    def __repr__(self) -> str:
        current = self.current_player
        clip_name = current.clip.name if current else "none"
        return f"ClipQueue(clips={len(self._queue)}, current='{clip_name}', blending={self._is_blending})"


@animation_data
class CrossfadePlayer:
    """Player that handles crossfading between clips.

    Maintains two clip slots and blends between them during transitions.
    """

    def __init__(self, skeleton: Skeleton) -> None:
        """Initialize crossfade player.

        Args:
            skeleton: Skeleton for pose sampling.
        """
        self._skeleton = skeleton
        self._current: Optional[ClipPlayer] = None
        self._next: Optional[ClipPlayer] = None
        self._blend_time = 0.0
        self._blend_elapsed = 0.0
        self._is_crossfading = False

    @property
    def skeleton(self) -> Skeleton:
        """Get skeleton."""
        return self._skeleton

    @property
    def current_clip(self) -> Optional[AnimationClip]:
        """Get current playing clip."""
        return self._current.clip if self._current else None

    @property
    def is_playing(self) -> bool:
        """Check if playing."""
        return self._current is not None and self._current.is_playing

    @property
    def is_crossfading(self) -> bool:
        """Check if currently crossfading."""
        return self._is_crossfading

    @property
    def blend_progress(self) -> float:
        """Get crossfade progress [0, 1]."""
        if not self._is_crossfading or self._blend_time <= MIN_BLEND_TIME:
            return 0.0
        return min(1.0, self._blend_elapsed / self._blend_time)

    def play(
        self,
        clip: AnimationClip,
        blend_time: float = 0.0,
        looping: Optional[bool] = None,
        speed: float = 1.0,
    ) -> ClipPlayer:
        """Play a new clip, optionally crossfading from current.

        Args:
            clip: Clip to play.
            blend_time: Time to crossfade (0 = instant).
            looping: Override clip looping.
            speed: Playback speed.

        Returns:
            The new clip player.
        """
        new_player = ClipPlayer(clip, looping=looping, speed=speed)
        new_player.play()

        if blend_time > 0 and self._current is not None and self._current.is_playing:
            # Start crossfade
            self._next = new_player
            self._blend_time = blend_time
            self._blend_elapsed = 0.0
            self._is_crossfading = True
        else:
            # Instant switch
            self._current = new_player
            self._next = None
            self._is_crossfading = False

        return new_player

    def stop(self, blend_time: float = 0.0) -> None:
        """Stop playback.

        Args:
            blend_time: Time to fade out (currently unused).
        """
        if self._current:
            self._current.stop()
        self._next = None
        self._is_crossfading = False

    def pause(self) -> None:
        """Pause current playback."""
        if self._current:
            self._current.pause()
        if self._next:
            self._next.pause()

    def resume(self) -> None:
        """Resume playback."""
        if self._current:
            self._current.play()
        if self._next:
            self._next.play()

    def update(self, dt: float) -> List[PlaybackEvent]:
        """Update playback.

        Args:
            dt: Time delta.

        Returns:
            List of fired events.
        """
        events: List[PlaybackEvent] = []

        if self._current:
            events.extend(self._current.update(dt))

        if self._is_crossfading and self._next:
            events.extend(self._next.update(dt))
            self._blend_elapsed += dt

            if self._blend_elapsed >= self._blend_time:
                # Crossfade complete
                self._current = self._next
                self._next = None
                self._is_crossfading = False
                self._blend_elapsed = 0.0

        return events

    def sample_pose(self) -> Optional[Pose]:
        """Sample current pose.

        Returns:
            Current pose, or None if nothing playing.
        """
        if not self._current:
            return None

        if self._is_crossfading and self._next:
            alpha = self.blend_progress
            return self._current.blend_with(self._next, alpha, self._skeleton)

        return self._current.sample_pose(self._skeleton)

    def get_current_time(self) -> float:
        """Get current playback time."""
        if self._current:
            return self._current.time
        return 0.0

    def set_current_time(self, time: float) -> None:
        """Set current playback time."""
        if self._current:
            self._current.set_time(time)

    def __repr__(self) -> str:
        clip_name = self._current.clip.name if self._current else "none"
        return (
            f"CrossfadePlayer(clip='{clip_name}', "
            f"crossfading={self._is_crossfading})"
        )
