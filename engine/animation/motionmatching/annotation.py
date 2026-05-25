"""
Motion Matching Annotation - Clip annotation for motion matching.

This module provides annotation tools for motion matching:
- MotionTag: Named tag with frame range
- ContactAnnotation: Per-frame foot contact data
- AnnotatedClip: Clip with tags and contacts
- Auto-detection of contacts from motion data

Usage:
    from engine.animation.motionmatching.annotation import (
        MotionTag, ContactAnnotation, AnnotatedClip, TagType
    )

    # Create annotated clip
    annotated = AnnotatedClip(clip)
    annotated.add_tag(MotionTag("walk", 0, 60))
    annotated.auto_detect_contacts()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import numpy as np


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class TagType(Enum):
    """Standard tag types for motion matching."""
    LOCOMOTION = auto()   # General movement
    IDLE = auto()         # Stationary
    WALK = auto()         # Walking speed
    RUN = auto()          # Running speed
    SPRINT = auto()       # Sprint speed
    JUMP = auto()         # Jump start
    LAND = auto()         # Landing
    FALL = auto()         # Falling/airborne
    TURN_LEFT = auto()    # Turning left
    TURN_RIGHT = auto()   # Turning right
    STRAFE_LEFT = auto()  # Strafing left
    STRAFE_RIGHT = auto() # Strafing right
    CROUCH = auto()       # Crouching movement
    CLIMB = auto()        # Climbing
    SLIDE = auto()        # Sliding
    START = auto()        # Motion start
    STOP = auto()         # Motion stop
    LOOP = auto()         # Loopable section
    CUSTOM = auto()       # Custom user tag


# Import centralized config
from engine.animation.motionmatching.config import (
    DEFAULT_CONTACT_DETECTION,
    DEFAULT_LOCOMOTION_SPEEDS,
    DEFAULT_TURN_DETECTION,
)

# Contact detection thresholds from config
DEFAULT_CONTACT_HEIGHT_THRESHOLD = DEFAULT_CONTACT_DETECTION.height_threshold
DEFAULT_CONTACT_VELOCITY_THRESHOLD = DEFAULT_CONTACT_DETECTION.velocity_threshold


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class MotionTag:
    """A named tag applied to a frame range.

    Attributes:
        name: Tag name (e.g., "walk", "jump")
        start_frame: First frame of tag
        end_frame: Last frame of tag (exclusive)
        tag_type: Optional standard tag type
        metadata: Optional additional metadata
    """
    name: str
    start_frame: int
    end_frame: int
    tag_type: Optional[TagType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.end_frame <= self.start_frame:
            self.end_frame = self.start_frame + 1

        # Auto-detect tag type from name if not provided
        if self.tag_type is None:
            self.tag_type = self._detect_tag_type()

    def _detect_tag_type(self) -> TagType:
        """Detect tag type from name."""
        name_lower = self.name.lower()

        tag_map = {
            'idle': TagType.IDLE,
            'walk': TagType.WALK,
            'run': TagType.RUN,
            'sprint': TagType.SPRINT,
            'jump': TagType.JUMP,
            'land': TagType.LAND,
            'fall': TagType.FALL,
            'turn_left': TagType.TURN_LEFT,
            'turn_right': TagType.TURN_RIGHT,
            'strafe_left': TagType.STRAFE_LEFT,
            'strafe_right': TagType.STRAFE_RIGHT,
            'crouch': TagType.CROUCH,
            'climb': TagType.CLIMB,
            'slide': TagType.SLIDE,
            'start': TagType.START,
            'stop': TagType.STOP,
            'loop': TagType.LOOP,
            'locomotion': TagType.LOCOMOTION,
        }

        for key, tag_type in tag_map.items():
            if key in name_lower:
                return tag_type

        return TagType.CUSTOM

    @property
    def frame_count(self) -> int:
        """Number of frames in tag range."""
        return self.end_frame - self.start_frame

    def contains_frame(self, frame: int) -> bool:
        """Check if frame is within tag range."""
        return self.start_frame <= frame < self.end_frame

    def overlaps(self, other: MotionTag) -> bool:
        """Check if this tag overlaps with another."""
        return not (self.end_frame <= other.start_frame or other.end_frame <= self.start_frame)

    def intersection(self, other: MotionTag) -> Optional[MotionTag]:
        """Get intersection of two tags."""
        if not self.overlaps(other):
            return None

        return MotionTag(
            name=f"{self.name}&{other.name}",
            start_frame=max(self.start_frame, other.start_frame),
            end_frame=min(self.end_frame, other.end_frame),
        )


@dataclass
class ContactAnnotation:
    """Per-frame foot contact annotation.

    Attributes:
        frame_count: Total number of frames
        left_contacts: Per-frame left foot contact (0.0-1.0)
        right_contacts: Per-frame right foot contact (0.0-1.0)
    """
    frame_count: int
    left_contacts: np.ndarray = field(default=None)
    right_contacts: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.left_contacts is None:
            self.left_contacts = np.zeros(self.frame_count, dtype=np.float32)
        if self.right_contacts is None:
            self.right_contacts = np.zeros(self.frame_count, dtype=np.float32)

        self.left_contacts = np.asarray(self.left_contacts, dtype=np.float32)
        self.right_contacts = np.asarray(self.right_contacts, dtype=np.float32)

    def get_contacts(self, frame: int) -> Tuple[float, float]:
        """Get contacts for a specific frame.

        Args:
            frame: Frame number

        Returns:
            Tuple of (left_contact, right_contact)
        """
        if 0 <= frame < self.frame_count:
            return float(self.left_contacts[frame]), float(self.right_contacts[frame])
        return 0.0, 0.0

    def set_contact(
        self,
        frame: int,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """Set contact values for a frame.

        Args:
            frame: Frame number
            left: Left foot contact (0.0-1.0)
            right: Right foot contact (0.0-1.0)
        """
        if 0 <= frame < self.frame_count:
            if left is not None:
                self.left_contacts[frame] = left
            if right is not None:
                self.right_contacts[frame] = right

    def set_contact_range(
        self,
        start_frame: int,
        end_frame: int,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """Set contact values for a frame range.

        Args:
            start_frame: First frame (inclusive)
            end_frame: Last frame (exclusive)
            left: Left foot contact value
            right: Right foot contact value
        """
        start = max(0, start_frame)
        end = min(self.frame_count, end_frame)

        if left is not None:
            self.left_contacts[start:end] = left
        if right is not None:
            self.right_contacts[start:end] = right

    def get_contact_events(
        self,
        foot: str = 'left',
        threshold: float = 0.5,
    ) -> List[Tuple[int, int]]:
        """Get list of contact events (start, end frame pairs).

        Args:
            foot: 'left' or 'right'
            threshold: Contact threshold

        Returns:
            List of (start_frame, end_frame) tuples
        """
        contacts = self.left_contacts if foot == 'left' else self.right_contacts

        events = []
        in_contact = False
        start_frame = 0

        for i, value in enumerate(contacts):
            if value >= threshold and not in_contact:
                in_contact = True
                start_frame = i
            elif value < threshold and in_contact:
                in_contact = False
                events.append((start_frame, i))

        # Handle contact at end
        if in_contact:
            events.append((start_frame, self.frame_count))

        return events


@dataclass
class AnnotatedClip:
    """Animation clip with tags and contact annotations.

    Wraps an animation clip and adds:
    - Motion tags for frame ranges
    - Foot contact annotations
    - Transition markers

    Attributes:
        clip: The underlying animation clip
        tags: List of motion tags
        contacts: Foot contact annotation
        transition_markers: Frames that are valid transition targets
    """
    clip: Any
    tags: List[MotionTag] = field(default_factory=list)
    contacts: Optional[ContactAnnotation] = None
    transition_markers: Optional[Set[int]] = None

    def __post_init__(self):
        # Initialize contacts if not provided
        if self.contacts is None:
            frame_count = getattr(self.clip, 'frame_count', 0)
            if frame_count > 0:
                self.contacts = ContactAnnotation(frame_count)

    # -------------------------------------------------------------------------
    # Clip Properties (Delegated)
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Clip name."""
        return getattr(self.clip, 'name', 'unnamed')

    @property
    def frame_count(self) -> int:
        """Number of frames."""
        return getattr(self.clip, 'frame_count', 0)

    @property
    def frame_rate(self) -> float:
        """Frame rate."""
        return getattr(self.clip, 'frame_rate', 30.0)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return getattr(self.clip, 'duration', 0.0)

    @property
    def is_looping(self) -> bool:
        """Whether clip loops."""
        return getattr(self.clip, 'is_looping', False)

    @property
    def has_root_motion(self) -> bool:
        """Whether clip has root motion."""
        return getattr(self.clip, 'has_root_motion', False)

    # -------------------------------------------------------------------------
    # Tag Management
    # -------------------------------------------------------------------------

    def add_tag(self, tag: MotionTag) -> None:
        """Add a motion tag.

        Args:
            tag: Tag to add
        """
        # Clamp to clip bounds
        tag = MotionTag(
            name=tag.name,
            start_frame=max(0, tag.start_frame),
            end_frame=min(self.frame_count, tag.end_frame),
            tag_type=tag.tag_type,
            metadata=tag.metadata,
        )
        self.tags.append(tag)

    def remove_tag(self, tag_name: str) -> None:
        """Remove all tags with given name.

        Args:
            tag_name: Name of tags to remove
        """
        self.tags = [t for t in self.tags if t.name != tag_name]

    def get_tags_at_frame(self, frame: int) -> List[MotionTag]:
        """Get all tags active at a specific frame.

        Args:
            frame: Frame number

        Returns:
            List of active tags
        """
        return [t for t in self.tags if t.contains_frame(frame)]

    def get_frame_tags(self, frame: int) -> Set[str]:
        """Get tag names for a frame.

        Args:
            frame: Frame number

        Returns:
            Set of tag names
        """
        return {t.name for t in self.get_tags_at_frame(frame)}

    def get_tags_by_type(self, tag_type: TagType) -> List[MotionTag]:
        """Get all tags of a specific type.

        Args:
            tag_type: Tag type to filter by

        Returns:
            List of matching tags
        """
        return [t for t in self.tags if t.tag_type == tag_type]

    def get_tags_by_name(self, name: str) -> List[MotionTag]:
        """Get all tags with a specific name.

        Args:
            name: Tag name to filter by

        Returns:
            List of matching tags
        """
        return [t for t in self.tags if t.name == name]

    @property
    def all_tag_names(self) -> Set[str]:
        """Get set of all unique tag names."""
        return {t.name for t in self.tags}

    # -------------------------------------------------------------------------
    # Contact Access
    # -------------------------------------------------------------------------

    def get_foot_contacts(self, frame: int) -> Tuple[float, float]:
        """Get foot contacts for a frame.

        Args:
            frame: Frame number

        Returns:
            Tuple of (left_contact, right_contact)
        """
        if self.contacts:
            return self.contacts.get_contacts(frame)
        return 0.0, 0.0

    def set_foot_contacts(
        self,
        frame: int,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """Set foot contacts for a frame.

        Args:
            frame: Frame number
            left: Left foot contact
            right: Right foot contact
        """
        if self.contacts:
            self.contacts.set_contact(frame, left, right)

    # -------------------------------------------------------------------------
    # Transition Markers
    # -------------------------------------------------------------------------

    def set_transition_markers(self, frames: Set[int]) -> None:
        """Set valid transition marker frames.

        Args:
            frames: Set of frame indices
        """
        self.transition_markers = {f for f in frames if 0 <= f < self.frame_count}

    def is_transition_frame(self, frame: int) -> bool:
        """Check if frame is a valid transition point.

        Args:
            frame: Frame number

        Returns:
            True if frame is a valid transition target
        """
        if self.transition_markers is None:
            return True  # All frames valid if not specified
        return frame in self.transition_markers

    # -------------------------------------------------------------------------
    # Clip Methods (Delegated)
    # -------------------------------------------------------------------------

    def sample(self, time: float) -> Any:
        """Sample clip at time.

        Args:
            time: Time in seconds

        Returns:
            Pose at time
        """
        if hasattr(self.clip, 'sample'):
            return self.clip.sample(time)
        return None

    def get_frame_pose(self, frame: int) -> Any:
        """Get pose at frame.

        Args:
            frame: Frame number

        Returns:
            Pose at frame
        """
        if hasattr(self.clip, 'get_frame_pose'):
            return self.clip.get_frame_pose(frame)
        return None

    def get_bone_position(self, frame: int, bone_name: str) -> np.ndarray:
        """Get bone position at frame.

        Args:
            frame: Frame number
            bone_name: Bone name

        Returns:
            3D position
        """
        if hasattr(self.clip, 'get_bone_position'):
            return self.clip.get_bone_position(frame, bone_name)
        return np.zeros(3, dtype=np.float32)

    def get_bone_velocity(self, frame: int, bone_name: str) -> np.ndarray:
        """Get bone velocity at frame.

        Args:
            frame: Frame number
            bone_name: Bone name

        Returns:
            3D velocity
        """
        if hasattr(self.clip, 'get_bone_velocity'):
            return self.clip.get_bone_velocity(frame, bone_name)
        return np.zeros(3, dtype=np.float32)

    def get_root_transform(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get root transform at frame.

        Args:
            frame: Frame number

        Returns:
            Tuple of (position, rotation quaternion)
        """
        if hasattr(self.clip, 'get_root_transform'):
            return self.clip.get_root_transform(frame)
        return np.zeros(3, dtype=np.float32), np.array([0, 0, 0, 1], dtype=np.float32)


# =============================================================================
# AUTO-DETECTION
# =============================================================================


def auto_detect_contacts(
    clip: AnnotatedClip,
    left_foot_bone: str = 'left_foot',
    right_foot_bone: str = 'right_foot',
    height_threshold: float = DEFAULT_CONTACT_HEIGHT_THRESHOLD,
    velocity_threshold: float = DEFAULT_CONTACT_VELOCITY_THRESHOLD,
    ground_height: float = 0.0,
) -> ContactAnnotation:
    """Auto-detect foot contacts from clip motion data.

    Detects contacts based on:
    - Foot height above ground
    - Foot velocity

    Args:
        clip: Annotated clip to analyze
        left_foot_bone: Name of left foot bone
        right_foot_bone: Name of right foot bone
        height_threshold: Maximum height for contact
        velocity_threshold: Maximum velocity for contact
        ground_height: Y coordinate of ground

    Returns:
        ContactAnnotation with detected contacts
    """
    frame_count = clip.frame_count
    contacts = ContactAnnotation(frame_count)

    for frame in range(frame_count):
        # Left foot
        left_pos = clip.get_bone_position(frame, left_foot_bone)
        left_vel = clip.get_bone_velocity(frame, left_foot_bone)

        left_height = left_pos[1] - ground_height  # Assuming Y is up
        left_speed = np.linalg.norm(left_vel)

        if left_height < height_threshold and left_speed < velocity_threshold:
            contacts.left_contacts[frame] = 1.0
        else:
            contacts.left_contacts[frame] = 0.0

        # Right foot
        right_pos = clip.get_bone_position(frame, right_foot_bone)
        right_vel = clip.get_bone_velocity(frame, right_foot_bone)

        right_height = right_pos[1] - ground_height
        right_speed = np.linalg.norm(right_vel)

        if right_height < height_threshold and right_speed < velocity_threshold:
            contacts.right_contacts[frame] = 1.0
        else:
            contacts.right_contacts[frame] = 0.0

    # Smooth contacts
    contacts = _smooth_contacts(contacts)

    return contacts


def _smooth_contacts(
    contacts: ContactAnnotation,
    min_contact_frames: int = None,
) -> ContactAnnotation:
    """Smooth contact data to remove noise.

    Removes very short contact/non-contact periods.

    Args:
        contacts: Raw contact annotation
        min_contact_frames: Minimum frames for a contact event (uses config default if None)

    Returns:
        Smoothed contacts
    """
    if min_contact_frames is None:
        min_contact_frames = DEFAULT_CONTACT_DETECTION.min_contact_frames

    def smooth_array(arr: np.ndarray) -> np.ndarray:
        result = arr.copy()
        n = len(arr)

        i = 0
        while i < n:
            # Find run length
            j = i
            while j < n and arr[j] == arr[i]:
                j += 1

            run_length = j - i

            # If run is too short, flip it
            if run_length < min_contact_frames:
                new_value = 0.0 if arr[i] > 0.5 else 1.0
                result[i:j] = new_value

            i = j

        return result

    return ContactAnnotation(
        frame_count=contacts.frame_count,
        left_contacts=smooth_array(contacts.left_contacts),
        right_contacts=smooth_array(contacts.right_contacts),
    )


def auto_detect_locomotion_tags(
    clip: AnnotatedClip,
    speed_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[MotionTag]:
    """Auto-detect locomotion tags from root velocity.

    Args:
        clip: Annotated clip to analyze
        speed_thresholds: Dict mapping tag names to (min_speed, max_speed)

    Returns:
        List of detected tags
    """
    if speed_thresholds is None:
        speed_thresholds = {
            'idle': DEFAULT_LOCOMOTION_SPEEDS.idle_speed,
            'walk': DEFAULT_LOCOMOTION_SPEEDS.walk_speed,
            'run': DEFAULT_LOCOMOTION_SPEEDS.run_speed,
            'sprint': DEFAULT_LOCOMOTION_SPEEDS.sprint_speed,
        }

    # Compute per-frame speeds
    speeds = []
    for frame in range(clip.frame_count):
        root_pos, _ = clip.get_root_transform(frame)
        if frame > 0:
            prev_pos, _ = clip.get_root_transform(frame - 1)
            velocity = (root_pos - prev_pos) * clip.frame_rate
            # Horizontal speed only
            speed = np.sqrt(velocity[0]**2 + velocity[2]**2)
        else:
            speed = 0.0
        speeds.append(speed)

    speeds = np.array(speeds)

    # Find contiguous regions for each tag
    tags = []

    for tag_name, (min_speed, max_speed) in speed_thresholds.items():
        in_range = (speeds >= min_speed) & (speeds < max_speed)

        # Find contiguous regions
        i = 0
        while i < len(in_range):
            if in_range[i]:
                start = i
                while i < len(in_range) and in_range[i]:
                    i += 1
                end = i

                # Only add if region is long enough
                if end - start >= DEFAULT_LOCOMOTION_SPEEDS.min_region_frames:
                    tags.append(MotionTag(
                        name=tag_name,
                        start_frame=start,
                        end_frame=end,
                    ))
            else:
                i += 1

    return tags


def auto_detect_turn_tags(
    clip: AnnotatedClip,
    turn_threshold: float = None,
    min_frames: int = None,
) -> List[MotionTag]:
    """Auto-detect turn tags from root rotation changes.

    Args:
        clip: Annotated clip to analyze
        turn_threshold: Minimum rotation to trigger turn tag (uses config default if None)
        min_frames: Minimum frames for a turn (uses config default if None)

    Returns:
        List of detected turn tags
    """
    if turn_threshold is None:
        turn_threshold = DEFAULT_TURN_DETECTION.turn_threshold
    if min_frames is None:
        min_frames = DEFAULT_TURN_DETECTION.min_turn_frames

    tags = []

    # Compute per-frame rotation changes
    rotations = []
    for frame in range(clip.frame_count):
        _, rot = clip.get_root_transform(frame)
        # Extract yaw from quaternion
        yaw = np.arctan2(
            2.0 * (rot[3] * rot[1] + rot[2] * rot[0]),
            1.0 - 2.0 * (rot[1]**2 + rot[2]**2)
        )
        rotations.append(yaw)

    rotations = np.array(rotations)

    # Compute angular velocity
    angular_vel = np.zeros(len(rotations))
    for i in range(1, len(rotations)):
        diff = rotations[i] - rotations[i-1]
        # Normalize to [-pi, pi]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        angular_vel[i] = diff * clip.frame_rate

    # Find turn regions
    turning_left = angular_vel > turn_threshold
    turning_right = angular_vel < -turn_threshold

    # Process left turns
    i = 0
    while i < len(turning_left):
        if turning_left[i]:
            start = i
            while i < len(turning_left) and turning_left[i]:
                i += 1
            if i - start >= min_frames:
                tags.append(MotionTag(
                    name='turn_left',
                    start_frame=start,
                    end_frame=i,
                    tag_type=TagType.TURN_LEFT,
                ))
        else:
            i += 1

    # Process right turns
    i = 0
    while i < len(turning_right):
        if turning_right[i]:
            start = i
            while i < len(turning_right) and turning_right[i]:
                i += 1
            if i - start >= min_frames:
                tags.append(MotionTag(
                    name='turn_right',
                    start_frame=start,
                    end_frame=i,
                    tag_type=TagType.TURN_RIGHT,
                ))
        else:
            i += 1

    return tags


def auto_detect_all_tags(
    clip: AnnotatedClip,
    detect_contacts: bool = True,
    detect_locomotion: bool = True,
    detect_turns: bool = True,
) -> AnnotatedClip:
    """Auto-detect all tags and contacts for a clip.

    Args:
        clip: Clip to annotate
        detect_contacts: Whether to detect foot contacts
        detect_locomotion: Whether to detect locomotion tags
        detect_turns: Whether to detect turn tags

    Returns:
        Annotated clip with detected data
    """
    # Detect contacts
    if detect_contacts:
        clip.contacts = auto_detect_contacts(clip)

    # Detect locomotion tags
    if detect_locomotion:
        loco_tags = auto_detect_locomotion_tags(clip)
        for tag in loco_tags:
            clip.add_tag(tag)

    # Detect turn tags
    if detect_turns:
        turn_tags = auto_detect_turn_tags(clip)
        for tag in turn_tags:
            clip.add_tag(tag)

    return clip


# =============================================================================
# TAG UTILITIES
# =============================================================================


def merge_overlapping_tags(tags: List[MotionTag]) -> List[MotionTag]:
    """Merge overlapping tags with the same name.

    Args:
        tags: List of tags

    Returns:
        List with overlapping tags merged
    """
    if not tags:
        return []

    # Group by name
    groups: Dict[str, List[MotionTag]] = {}
    for tag in tags:
        if tag.name not in groups:
            groups[tag.name] = []
        groups[tag.name].append(tag)

    # Merge each group
    result = []
    for name, group in groups.items():
        # Sort by start frame
        group.sort(key=lambda t: t.start_frame)

        merged = []
        current = group[0]

        for tag in group[1:]:
            if tag.start_frame <= current.end_frame:
                # Overlapping - extend current
                current = MotionTag(
                    name=current.name,
                    start_frame=current.start_frame,
                    end_frame=max(current.end_frame, tag.end_frame),
                    tag_type=current.tag_type,
                )
            else:
                merged.append(current)
                current = tag

        merged.append(current)
        result.extend(merged)

    return result


def filter_tags_by_duration(
    tags: List[MotionTag],
    min_frames: int = 1,
    max_frames: Optional[int] = None,
) -> List[MotionTag]:
    """Filter tags by duration.

    Args:
        tags: List of tags
        min_frames: Minimum frame count
        max_frames: Optional maximum frame count

    Returns:
        Filtered list
    """
    result = []
    for tag in tags:
        if tag.frame_count >= min_frames:
            if max_frames is None or tag.frame_count <= max_frames:
                result.append(tag)
    return result
