"""Realtime Reflection Probe Capture System.

Implements dynamic cubemap capture for reflection probes with:
- 6-face rendering amortized over multiple frames
- Face-to-render scheduler with round-robin and priority modes
- LOD-biased geometry rendering for performance
- Dynamic object inclusion/exclusion filtering
- Ping-pong buffers for temporal stability
- Priority-based multi-probe management
- Budget limiting for frame time constraints

Reference: RENDERING_CONTEXT.md Section 6.4
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Set

from engine.core.math.geometry import AABB
from engine.core.math.vec import Vec3
from engine.rendering.lighting.baked_probes import (
    BakedProbeAsset,
    BakedProbeConstants,
    BC6HCompressor,
    CaptureConfig,
    CompressionQuality,
    CubemapData,
    CubemapFace,
    CubemapFaceData,
    CubemapMipChain,
    CubemapRenderer,
    CUBEMAP_FACE_DIRECTIONS,
    HDRPixel,
    KTX2Writer,
    MipGenerator,
    MipLevel,
    PrefilteredGenerator,
)

if TYPE_CHECKING:
    from engine.core.ecs.entity import Entity


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

class RealtimeProbeConstants:
    """Constants for realtime probe capture."""
    # Default realtime resolution
    DEFAULT_RESOLUTION: int = 128
    # Maximum realtime resolution
    MAX_RESOLUTION: int = 512
    # Minimum resolution
    MIN_RESOLUTION: int = 32
    # Default update rate (frames between updates)
    DEFAULT_UPDATE_RATE: int = 1
    # Maximum update rate (frames)
    MAX_UPDATE_RATE: int = 60
    # Default LOD bias
    DEFAULT_LOD_BIAS: float = 1.0
    # Maximum LOD bias
    MAX_LOD_BIAS: float = 4.0
    # Default capture budget (face renders per frame)
    DEFAULT_CAPTURE_BUDGET: int = 2
    # Maximum capture budget
    MAX_CAPTURE_BUDGET: int = 12
    # Priority threshold for urgent updates
    URGENT_PRIORITY_THRESHOLD: float = 0.9
    # Decay rate for face dirty priority
    DIRTY_DECAY_RATE: float = 0.95
    # Distance scale factor for priority
    DISTANCE_PRIORITY_FACTOR: float = 0.1
    # Visibility priority boost
    VISIBILITY_PRIORITY_BOOST: float = 0.5


class SchedulerMode(Enum):
    """Face scheduling modes."""
    ROUND_ROBIN = auto()    # Cycle through faces sequentially
    PRIORITY = auto()       # Update faces based on priority
    ADAPTIVE = auto()       # Switch between modes based on scene


class ProbeUpdateReason(Enum):
    """Reasons for probe update."""
    SCHEDULED = auto()      # Regular scheduled update
    DIRTY = auto()          # Scene changed
    CAMERA_MOVED = auto()   # Camera moved significantly
    FORCED = auto()         # Explicit force update
    INITIAL = auto()        # First capture


# -----------------------------------------------------------------------------
# Face Scheduler
# -----------------------------------------------------------------------------

@dataclass
class FaceState:
    """State tracking for a single cubemap face.

    Attributes:
        face: The cubemap face identifier
        dirty: Whether the face needs re-rendering
        last_update_frame: Frame number of last update
        priority: Update priority (0-1, higher = more urgent)
        change_magnitude: Estimated scene change magnitude
    """
    face: CubemapFace
    dirty: bool = True
    last_update_frame: int = -1
    priority: float = 1.0
    change_magnitude: float = 0.0

    def mark_dirty(self, magnitude: float = 1.0) -> None:
        """Mark face as needing update.

        Args:
            magnitude: Change magnitude (0-1)
        """
        self.dirty = True
        self.change_magnitude = max(self.change_magnitude, magnitude)
        self.priority = min(1.0, self.priority + magnitude * 0.5)

    def mark_clean(self, frame: int) -> None:
        """Mark face as updated.

        Args:
            frame: Current frame number
        """
        self.dirty = False
        self.last_update_frame = frame
        self.change_magnitude = 0.0
        self.priority = 0.0

    def decay_priority(self, decay_rate: float) -> None:
        """Decay priority over time.

        Args:
            decay_rate: Decay multiplier (0-1)
        """
        self.priority *= decay_rate


@dataclass
class RealtimeProbeFaceScheduler:
    """Schedules cubemap face rendering across frames.

    Amortizes 6-face cubemap rendering over multiple frames
    using either round-robin or priority-based selection.

    Attributes:
        mode: Scheduling mode
        current_face_index: Current face in round-robin
        faces: State for each face
        frame_count: Total frames processed
    """
    mode: SchedulerMode = SchedulerMode.ROUND_ROBIN
    current_face_index: int = 0
    faces: list[FaceState] = field(default_factory=list)
    frame_count: int = 0
    _decay_rate: float = RealtimeProbeConstants.DIRTY_DECAY_RATE

    def __post_init__(self) -> None:
        """Initialize face states."""
        if not self.faces:
            self.faces = [
                FaceState(face=CubemapFace(i))
                for i in range(BakedProbeConstants.FACE_COUNT)
            ]

    @property
    def decay_rate(self) -> float:
        """Get priority decay rate."""
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, value: float) -> None:
        """Set priority decay rate."""
        self._decay_rate = max(0.0, min(1.0, value))

    def get_next_face(self) -> CubemapFace:
        """Get the next face to render.

        Returns:
            Face to render this frame
        """
        self.frame_count += 1

        if self.mode == SchedulerMode.ROUND_ROBIN:
            return self._get_next_round_robin()
        elif self.mode == SchedulerMode.PRIORITY:
            return self._get_next_priority()
        else:  # ADAPTIVE
            return self._get_next_adaptive()

    def _get_next_round_robin(self) -> CubemapFace:
        """Round-robin face selection."""
        face = CubemapFace(self.current_face_index)
        self.current_face_index = (self.current_face_index + 1) % BakedProbeConstants.FACE_COUNT
        return face

    def _get_next_priority(self) -> CubemapFace:
        """Priority-based face selection."""
        # Find face with highest priority
        max_priority = -1.0
        selected_face = CubemapFace.POSITIVE_X

        for state in self.faces:
            if state.dirty and state.priority > max_priority:
                max_priority = state.priority
                selected_face = state.face

        # If no dirty faces, fall back to round-robin
        if max_priority < 0:
            return self._get_next_round_robin()

        return selected_face

    def _get_next_adaptive(self) -> CubemapFace:
        """Adaptive mode: priority when dirty, round-robin otherwise."""
        # Check if any face is urgently dirty
        urgent_dirty = any(
            s.dirty and s.priority > RealtimeProbeConstants.URGENT_PRIORITY_THRESHOLD
            for s in self.faces
        )

        if urgent_dirty:
            return self._get_next_priority()
        return self._get_next_round_robin()

    def get_next_faces(self, count: int) -> list[CubemapFace]:
        """Get multiple faces to render this frame.

        Args:
            count: Number of faces to get

        Returns:
            List of faces to render
        """
        count = min(count, BakedProbeConstants.FACE_COUNT)
        faces = []

        for _ in range(count):
            face = self.get_next_face()
            if face not in faces:
                faces.append(face)
            else:
                # Find next face not in list
                for i in range(BakedProbeConstants.FACE_COUNT):
                    candidate = CubemapFace((self.current_face_index + i) % BakedProbeConstants.FACE_COUNT)
                    if candidate not in faces:
                        faces.append(candidate)
                        break

        return faces

    def mark_face_dirty(
        self,
        face: CubemapFace,
        magnitude: float = 1.0,
    ) -> None:
        """Mark a specific face as needing update.

        Args:
            face: Face to mark dirty
            magnitude: Change magnitude (0-1)
        """
        self.faces[face.value].mark_dirty(magnitude)

    def mark_all_dirty(self, magnitude: float = 1.0) -> None:
        """Mark all faces as dirty.

        Args:
            magnitude: Change magnitude
        """
        for state in self.faces:
            state.mark_dirty(magnitude)

    def mark_face_clean(self, face: CubemapFace) -> None:
        """Mark a face as updated.

        Args:
            face: Face that was updated
        """
        self.faces[face.value].mark_clean(self.frame_count)

    def get_update_priority(self, face: CubemapFace) -> float:
        """Get update priority for a face.

        Args:
            face: Face to query

        Returns:
            Priority value (0-1)
        """
        return self.faces[face.value].priority

    def is_face_dirty(self, face: CubemapFace) -> bool:
        """Check if a face needs updating.

        Args:
            face: Face to check

        Returns:
            True if face is dirty
        """
        return self.faces[face.value].dirty

    def get_dirty_count(self) -> int:
        """Get number of dirty faces.

        Returns:
            Count of dirty faces
        """
        return sum(1 for s in self.faces if s.dirty)

    def update_priorities(self) -> None:
        """Update all face priorities (decay over time)."""
        for state in self.faces:
            state.decay_priority(self._decay_rate)

    def reset(self) -> None:
        """Reset scheduler state."""
        self.current_face_index = 0
        self.frame_count = 0
        for state in self.faces:
            state.dirty = True
            state.last_update_frame = -1
            state.priority = 1.0
            state.change_magnitude = 0.0


# -----------------------------------------------------------------------------
# Capture Settings
# -----------------------------------------------------------------------------

@dataclass
class RealtimeProbeCaptureSettings:
    """Configuration for realtime probe capture.

    Attributes:
        resolution: Cubemap face resolution (32-512)
        lod_bias: LOD bias for geometry (reduces detail)
        update_rate: Frames between full updates
        include_dynamic_objects: Whether to capture dynamic objects
        distance_scale: Distance-based update frequency scaling
        max_render_distance: Maximum render distance
        temporal_blend_factor: Blend factor for temporal stability (0-1)
        compression_quality: BC6H compression quality
    """
    resolution: int = RealtimeProbeConstants.DEFAULT_RESOLUTION
    lod_bias: float = RealtimeProbeConstants.DEFAULT_LOD_BIAS
    update_rate: int = RealtimeProbeConstants.DEFAULT_UPDATE_RATE
    include_dynamic_objects: bool = True
    distance_scale: float = 1.0
    max_render_distance: float = 500.0
    temporal_blend_factor: float = 0.8
    compression_quality: CompressionQuality = CompressionQuality.FAST

    def __post_init__(self) -> None:
        """Validate settings."""
        self.resolution = max(
            RealtimeProbeConstants.MIN_RESOLUTION,
            min(self.resolution, RealtimeProbeConstants.MAX_RESOLUTION)
        )
        self.lod_bias = max(0.0, min(self.lod_bias, RealtimeProbeConstants.MAX_LOD_BIAS))
        self.update_rate = max(1, min(self.update_rate, RealtimeProbeConstants.MAX_UPDATE_RATE))
        self.distance_scale = max(0.1, min(self.distance_scale, 10.0))
        self.temporal_blend_factor = max(0.0, min(self.temporal_blend_factor, 1.0))

    def get_effective_lod_bias(self, distance: float) -> float:
        """Get LOD bias adjusted for distance.

        Args:
            distance: Distance from camera

        Returns:
            Adjusted LOD bias
        """
        distance_factor = 1.0 + (distance / self.max_render_distance) * self.distance_scale
        return self.lod_bias * distance_factor

    def get_effective_update_rate(self, distance: float) -> int:
        """Get update rate adjusted for distance.

        Args:
            distance: Distance from camera

        Returns:
            Adjusted update rate (frames)
        """
        distance_factor = 1.0 + (distance / self.max_render_distance) * self.distance_scale
        return max(1, int(self.update_rate * distance_factor))


# -----------------------------------------------------------------------------
# Dynamic Object Filter
# -----------------------------------------------------------------------------

@dataclass
class DynamicObjectFilter:
    """Filters objects for inclusion in realtime capture.

    Attributes:
        include_layers: Layer mask for included objects
        exclude_tags: Tags to exclude from capture
        include_moving: Whether to include moving objects
        velocity_threshold: Minimum velocity to consider moving
        excluded_entities: Explicitly excluded entity IDs
        included_entities: Explicitly included entity IDs
    """
    include_layers: int = 0xFFFFFFFF
    exclude_tags: list[str] = field(default_factory=list)
    include_moving: bool = True
    velocity_threshold: float = 0.1
    excluded_entities: Set[int] = field(default_factory=set)
    included_entities: Set[int] = field(default_factory=set)

    def should_include(
        self,
        entity_id: int,
        layer: int,
        tags: Sequence[str],
        velocity: float = 0.0,
        is_static: bool = False,
    ) -> bool:
        """Determine if an entity should be included in capture.

        Args:
            entity_id: Entity identifier
            layer: Entity layer mask
            tags: Entity tags
            velocity: Entity velocity magnitude
            is_static: Whether entity is static

        Returns:
            True if entity should be captured
        """
        # Explicit exclusion
        if entity_id in self.excluded_entities:
            return False

        # Explicit inclusion
        if entity_id in self.included_entities:
            return True

        # Layer check
        if not (layer & self.include_layers):
            return False

        # Tag exclusion
        for tag in tags:
            if tag in self.exclude_tags:
                return False

        # Moving object check
        if not is_static and not self.include_moving:
            if velocity > self.velocity_threshold:
                return False

        return True

    def exclude_entity(self, entity_id: int) -> None:
        """Exclude an entity from capture.

        Args:
            entity_id: Entity to exclude
        """
        self.excluded_entities.add(entity_id)
        self.included_entities.discard(entity_id)

    def include_entity(self, entity_id: int) -> None:
        """Include an entity in capture.

        Args:
            entity_id: Entity to include
        """
        self.included_entities.add(entity_id)
        self.excluded_entities.discard(entity_id)

    def clear_exclusions(self) -> None:
        """Clear all explicit exclusions."""
        self.excluded_entities.clear()
        self.included_entities.clear()


# -----------------------------------------------------------------------------
# Realtime Probe Capture
# -----------------------------------------------------------------------------

class RealtimeProbeCapture(CubemapRenderer):
    """Realtime cubemap capture with temporal stability.

    Extends CubemapRenderer with realtime-specific features:
    - LOD bias for reduced geometry
    - Dynamic object filtering
    - Ping-pong buffers for blending
    - Temporal stability via blending
    """

    def __init__(
        self,
        config: CaptureConfig,
        settings: RealtimeProbeCaptureSettings,
        object_filter: Optional[DynamicObjectFilter] = None,
    ) -> None:
        """Initialize realtime capture.

        Args:
            config: Base capture configuration
            settings: Realtime-specific settings
            object_filter: Object inclusion filter
        """
        super().__init__(config)
        self._settings = settings
        self._object_filter = object_filter or DynamicObjectFilter()

        # Ping-pong buffers
        self._buffer_a: Optional[CubemapData] = None
        self._buffer_b: Optional[CubemapData] = None
        self._current_buffer: int = 0

        # Face completion tracking
        self._completed_faces: Set[CubemapFace] = set()

        # Initialize buffers
        self._init_buffers()

    @property
    def settings(self) -> RealtimeProbeCaptureSettings:
        """Get realtime settings."""
        return self._settings

    @property
    def object_filter(self) -> DynamicObjectFilter:
        """Get object filter."""
        return self._object_filter

    def _init_buffers(self) -> None:
        """Initialize ping-pong buffers."""
        res = self._settings.resolution
        self._buffer_a = CubemapData(resolution=res)
        self._buffer_b = CubemapData(resolution=res)

    def _get_current_buffer(self) -> CubemapData:
        """Get the current write buffer."""
        if self._current_buffer == 0:
            return self._buffer_a
        return self._buffer_b

    def _get_previous_buffer(self) -> CubemapData:
        """Get the previous (read) buffer."""
        if self._current_buffer == 0:
            return self._buffer_b
        return self._buffer_a

    def _swap_buffers(self) -> None:
        """Swap ping-pong buffers."""
        self._current_buffer = 1 - self._current_buffer
        self._completed_faces.clear()

    def capture_face(
        self,
        position: Vec3,
        face: CubemapFace,
        scene_render_func: Optional[Callable[[Vec3, CubemapFace, float], CubemapFaceData]] = None,
    ) -> CubemapFaceData:
        """Capture a single cubemap face.

        Args:
            position: Capture position
            face: Face to capture
            scene_render_func: Optional custom render function

        Returns:
            Captured face data
        """
        lod_bias = self._settings.get_effective_lod_bias(0.0)  # Distance not known here

        if scene_render_func:
            face_data = scene_render_func(position, face, lod_bias)
        else:
            face_data = self._capture_face(position, face)

        # Apply temporal blending with previous frame
        if self._settings.temporal_blend_factor > 0:
            prev_buffer = self._get_previous_buffer()
            prev_face = prev_buffer.get_face(face)
            face_data = self._blend_faces(face_data, prev_face, self._settings.temporal_blend_factor)

        # Store in current buffer
        current_buffer = self._get_current_buffer()
        current_buffer.faces[face.value] = face_data
        self._completed_faces.add(face)

        return face_data

    def _capture_face(self, position: Vec3, face: CubemapFace) -> CubemapFaceData:
        """Internal face capture (to be overridden).

        Args:
            position: Capture position
            face: Face to capture

        Returns:
            Captured face data
        """
        # Default implementation creates empty face
        # Subclasses should override with actual scene rendering
        return CubemapFaceData(face=face, resolution=self._settings.resolution)

    def _blend_faces(
        self,
        current: CubemapFaceData,
        previous: CubemapFaceData,
        blend_factor: float,
    ) -> CubemapFaceData:
        """Blend current face with previous for temporal stability.

        Args:
            current: Current frame face data
            previous: Previous frame face data
            blend_factor: Blend weight for previous (0-1)

        Returns:
            Blended face data
        """
        if current.resolution != previous.resolution:
            return current

        result = CubemapFaceData(face=current.face, resolution=current.resolution)

        for y in range(current.resolution):
            for x in range(current.resolution):
                curr_pixel = current.get_pixel(x, y)
                prev_pixel = previous.get_pixel(x, y)

                blended = HDRPixel(
                    curr_pixel.r * (1 - blend_factor) + prev_pixel.r * blend_factor,
                    curr_pixel.g * (1 - blend_factor) + prev_pixel.g * blend_factor,
                    curr_pixel.b * (1 - blend_factor) + prev_pixel.b * blend_factor,
                )
                result.set_pixel(x, y, blended)

        return result

    def complete_probe(self) -> bool:
        """Check if all faces have been captured.

        Returns:
            True if probe capture is complete
        """
        return len(self._completed_faces) == BakedProbeConstants.FACE_COUNT

    def get_current_cubemap(self) -> CubemapData:
        """Get the current cubemap state.

        Returns:
            Current cubemap (may be partial)
        """
        return self._get_current_buffer()

    def finalize_frame(self) -> Optional[CubemapData]:
        """Finalize the current frame and swap buffers if complete.

        Returns:
            Complete cubemap if all faces done, None otherwise
        """
        if self.complete_probe():
            result = self._get_current_buffer()
            self._swap_buffers()
            return result
        return None

    def reset(self) -> None:
        """Reset capture state."""
        self._completed_faces.clear()
        self._current_buffer = 0
        self._init_buffers()


class FunctionRealtimeProbeCapture(RealtimeProbeCapture):
    """Realtime probe capture using a sample function.

    Useful for testing or procedural environments.
    """

    def __init__(
        self,
        config: CaptureConfig,
        settings: RealtimeProbeCaptureSettings,
        sample_func: Callable[[Vec3, Vec3], Vec3],
        object_filter: Optional[DynamicObjectFilter] = None,
    ) -> None:
        """Initialize with a sample function.

        Args:
            config: Base capture config
            settings: Realtime settings
            sample_func: Function(position, direction) -> color
            object_filter: Object filter
        """
        super().__init__(config, settings, object_filter)
        self._sample_func = sample_func

    def _capture_face(self, position: Vec3, face: CubemapFace) -> CubemapFaceData:
        """Capture using the sample function."""
        res = self._settings.resolution
        face_data = CubemapFaceData(face=face, resolution=res)

        direction, up = CUBEMAP_FACE_DIRECTIONS[face]
        right = direction.cross(up).normalized()

        for y in range(res):
            for x in range(res):
                # Compute direction for this pixel
                u = (x + 0.5) / res * 2.0 - 1.0
                v = (y + 0.5) / res * 2.0 - 1.0

                pixel_dir = (direction + right * u + up * (-v)).normalized()

                # Sample environment
                color = self._sample_func(position, pixel_dir)
                pixel = HDRPixel(color.x, color.y, color.z)
                face_data.set_pixel(x, y, pixel)

        return face_data


# -----------------------------------------------------------------------------
# Realtime Probe
# -----------------------------------------------------------------------------

@dataclass
class RealtimeProbeState:
    """Runtime state for a realtime probe.

    Attributes:
        probe_id: Unique probe identifier
        last_update_frame: Frame of last complete update
        last_partial_frame: Frame of last partial update
        update_count: Total number of complete updates
        is_visible: Whether probe is currently visible
        camera_distance: Distance from camera
        priority: Current update priority (0-1)
    """
    probe_id: int
    last_update_frame: int = -1
    last_partial_frame: int = -1
    update_count: int = 0
    is_visible: bool = True
    camera_distance: float = 0.0
    priority: float = 1.0


@dataclass
class RealtimeReflectionProbe:
    """A realtime reflection probe with amortized update.

    Attributes:
        probe_id: Unique identifier
        name: Human-readable name
        position: World position
        bounds: Influence bounds
        settings: Capture settings
        scheduler: Face scheduler
        capture: Capture handler
        state: Runtime state
    """
    probe_id: int
    name: str
    position: Vec3
    bounds: AABB
    settings: RealtimeProbeCaptureSettings = field(default_factory=RealtimeProbeCaptureSettings)
    scheduler: RealtimeProbeFaceScheduler = field(default_factory=RealtimeProbeFaceScheduler)
    state: RealtimeProbeState = field(default=None)

    # Internal capture handler
    _capture: Optional[RealtimeProbeCapture] = field(default=None, repr=False)
    _current_cubemap: Optional[CubemapData] = field(default=None, repr=False)
    _mip_chain: Optional[CubemapMipChain] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Initialize probe state."""
        if self.state is None:
            self.state = RealtimeProbeState(probe_id=self.probe_id)

    def initialize(
        self,
        capture: RealtimeProbeCapture,
    ) -> None:
        """Initialize probe with capture handler.

        Args:
            capture: Capture handler
        """
        self._capture = capture
        self._current_cubemap = CubemapData(resolution=self.settings.resolution)

    def update_face(
        self,
        frame: int,
        scene_render_func: Optional[Callable[[Vec3, CubemapFace, float], CubemapFaceData]] = None,
    ) -> Optional[CubemapFace]:
        """Update a single face this frame.

        Args:
            frame: Current frame number
            scene_render_func: Optional render function

        Returns:
            Face that was updated, or None
        """
        if self._capture is None:
            return None

        # Get next face to update
        face = self.scheduler.get_next_face()

        # Capture face
        face_data = self._capture.capture_face(
            self.position,
            face,
            scene_render_func,
        )

        # Update current cubemap
        if self._current_cubemap is not None:
            self._current_cubemap.faces[face.value] = face_data

        # Mark face as clean
        self.scheduler.mark_face_clean(face)
        self.state.last_partial_frame = frame

        # Check if complete
        if self._capture.complete_probe():
            self.state.last_update_frame = frame
            self.state.update_count += 1
            self._capture.finalize_frame()

        return face

    def update_faces(
        self,
        frame: int,
        face_count: int,
        scene_render_func: Optional[Callable[[Vec3, CubemapFace, float], CubemapFaceData]] = None,
    ) -> list[CubemapFace]:
        """Update multiple faces this frame.

        Args:
            frame: Current frame number
            face_count: Number of faces to update
            scene_render_func: Optional render function

        Returns:
            List of faces that were updated
        """
        if self._capture is None:
            return []

        faces = self.scheduler.get_next_faces(face_count)

        for face in faces:
            face_data = self._capture.capture_face(
                self.position,
                face,
                scene_render_func,
            )

            if self._current_cubemap is not None:
                self._current_cubemap.faces[face.value] = face_data

            self.scheduler.mark_face_clean(face)

        self.state.last_partial_frame = frame

        if self._capture.complete_probe():
            self.state.last_update_frame = frame
            self.state.update_count += 1
            self._capture.finalize_frame()

        return faces

    def get_cubemap(self) -> Optional[CubemapData]:
        """Get current cubemap.

        Returns:
            Current cubemap or None
        """
        return self._current_cubemap

    def get_mip_chain(self) -> Optional[CubemapMipChain]:
        """Get or generate mip chain.

        Returns:
            Mip chain or None
        """
        if self._current_cubemap is None:
            return None

        # Generate mips on demand
        generator = MipGenerator()
        self._mip_chain = generator.generate_mips(self._current_cubemap)
        return self._mip_chain

    def sample(self, direction: Vec3, roughness: float = 0.0) -> Vec3:
        """Sample the probe in a direction.

        Args:
            direction: Sample direction
            roughness: Surface roughness

        Returns:
            Sampled color
        """
        if self._current_cubemap is None:
            return Vec3(0, 0, 0)

        if roughness > 0.01:
            # Use mip chain for roughness sampling
            mip_chain = self.get_mip_chain()
            if mip_chain:
                pixel = mip_chain.sample_roughness(direction, roughness)
                return pixel.to_vec3()

        # Direct sample
        pixel = self._current_cubemap.sample_direction(direction)
        return pixel.to_vec3()

    def mark_dirty(self, magnitude: float = 1.0) -> None:
        """Mark probe as needing full update.

        Args:
            magnitude: Change magnitude
        """
        self.scheduler.mark_all_dirty(magnitude)

    def mark_face_dirty(self, face: CubemapFace, magnitude: float = 1.0) -> None:
        """Mark a specific face as dirty.

        Args:
            face: Face to mark
            magnitude: Change magnitude
        """
        self.scheduler.mark_face_dirty(face, magnitude)

    def contains(self, point: Vec3) -> bool:
        """Check if point is within probe bounds.

        Args:
            point: Point to check

        Returns:
            True if point is inside bounds
        """
        return self.bounds.contains(point)

    def update_priority(self, camera_position: Vec3, is_visible: bool) -> None:
        """Update probe priority based on camera.

        Args:
            camera_position: Camera world position
            is_visible: Whether probe is currently visible
        """
        self.state.camera_distance = camera_position.distance(self.position)
        self.state.is_visible = is_visible

        # Calculate priority
        distance_factor = 1.0 / (1.0 + self.state.camera_distance * RealtimeProbeConstants.DISTANCE_PRIORITY_FACTOR)
        visibility_boost = RealtimeProbeConstants.VISIBILITY_PRIORITY_BOOST if is_visible else 0.0

        self.state.priority = distance_factor + visibility_boost

    def convert_to_baked(self) -> Optional[BakedProbeAsset]:
        """Convert current state to a baked probe asset.

        Returns:
            Baked probe asset or None
        """
        if self._current_cubemap is None:
            return None

        # Generate prefiltered mip chain
        prefilter = PrefilteredGenerator(
            sample_count=256,
            roughness_levels=6,
        )
        mip_chain = prefilter.generate_prefiltered(self._current_cubemap)

        # Compress
        compressor = BC6HCompressor(self.settings.compression_quality)
        compressed_mips = []
        for mip in mip_chain.mips:
            compressed_faces = compressor.compress_cubemap(mip.cubemap)
            compressed_mips.append(compressed_faces)

        # Write KTX2
        writer = KTX2Writer()
        ktx2_data = writer.write_to_bytes(mip_chain, compressed_mips, supercompress=True)

        return BakedProbeAsset(
            probe_id=self.probe_id,
            name=self.name + "_baked",
            position=self.position,
            bounds=self.bounds,
            resolution=self.settings.resolution,
            mip_count=len(mip_chain.mips),
            is_prefiltered=True,
            ktx2_data=ktx2_data,
            loaded=False,
        )


# -----------------------------------------------------------------------------
# Realtime Probe Manager
# -----------------------------------------------------------------------------

@dataclass
class CaptureBudget:
    """Frame capture budget configuration.

    Attributes:
        max_face_renders: Maximum face renders per frame
        max_time_ms: Maximum time budget in milliseconds
        reserve_urgent: Reserve slots for urgent updates
    """
    max_face_renders: int = RealtimeProbeConstants.DEFAULT_CAPTURE_BUDGET
    max_time_ms: float = 2.0
    reserve_urgent: int = 1

    def __post_init__(self) -> None:
        """Validate budget."""
        self.max_face_renders = max(1, min(self.max_face_renders, RealtimeProbeConstants.MAX_CAPTURE_BUDGET))
        self.max_time_ms = max(0.1, self.max_time_ms)
        self.reserve_urgent = max(0, min(self.reserve_urgent, self.max_face_renders - 1))


class RealtimeProbeManager:
    """Manages multiple realtime reflection probes.

    Handles priority-based scheduling, budget limiting,
    and probe lifecycle management.
    """

    _id_counter: int = 0

    def __init__(
        self,
        budget: Optional[CaptureBudget] = None,
        default_settings: Optional[RealtimeProbeCaptureSettings] = None,
    ) -> None:
        """Initialize probe manager.

        Args:
            budget: Capture budget configuration
            default_settings: Default probe settings
        """
        self._budget = budget or CaptureBudget()
        self._default_settings = default_settings or RealtimeProbeCaptureSettings()
        self._probes: dict[int, RealtimeReflectionProbe] = {}
        self._priority_queue: list[int] = []
        self._frame_count: int = 0
        self._last_update_time: float = 0.0

    @property
    def budget(self) -> CaptureBudget:
        """Get capture budget."""
        return self._budget

    @budget.setter
    def budget(self, value: CaptureBudget) -> None:
        """Set capture budget."""
        self._budget = value

    @property
    def probe_count(self) -> int:
        """Get number of managed probes."""
        return len(self._probes)

    def register_probe(
        self,
        name: str,
        position: Vec3,
        bounds: AABB,
        settings: Optional[RealtimeProbeCaptureSettings] = None,
        capture: Optional[RealtimeProbeCapture] = None,
    ) -> RealtimeReflectionProbe:
        """Register a new realtime probe.

        Args:
            name: Probe name
            position: World position
            bounds: Influence bounds
            settings: Probe settings (uses default if None)
            capture: Capture handler (creates default if None)

        Returns:
            Created probe
        """
        RealtimeProbeManager._id_counter += 1
        probe_id = RealtimeProbeManager._id_counter

        probe_settings = settings or RealtimeProbeCaptureSettings(
            resolution=self._default_settings.resolution,
            lod_bias=self._default_settings.lod_bias,
            update_rate=self._default_settings.update_rate,
            include_dynamic_objects=self._default_settings.include_dynamic_objects,
        )

        probe = RealtimeReflectionProbe(
            probe_id=probe_id,
            name=name,
            position=position,
            bounds=bounds,
            settings=probe_settings,
        )

        if capture is None:
            config = CaptureConfig(resolution=probe_settings.resolution)
            capture = RealtimeProbeCapture(config, probe_settings)

        probe.initialize(capture)

        self._probes[probe_id] = probe
        self._priority_queue.append(probe_id)

        return probe

    def unregister_probe(self, probe_id: int) -> bool:
        """Remove a probe from management.

        Args:
            probe_id: Probe to remove

        Returns:
            True if probe was removed
        """
        if probe_id in self._probes:
            del self._probes[probe_id]
            if probe_id in self._priority_queue:
                self._priority_queue.remove(probe_id)
            return True
        return False

    def get_probe(self, probe_id: int) -> Optional[RealtimeReflectionProbe]:
        """Get a probe by ID.

        Args:
            probe_id: Probe ID

        Returns:
            Probe or None
        """
        return self._probes.get(probe_id)

    def get_all_probes(self) -> list[RealtimeReflectionProbe]:
        """Get all managed probes.

        Returns:
            List of all probes
        """
        return list(self._probes.values())

    def update_probes(
        self,
        camera_position: Vec3,
        visible_probe_ids: Optional[Set[int]] = None,
        scene_render_func: Optional[Callable[[Vec3, CubemapFace, float], CubemapFaceData]] = None,
    ) -> list[tuple[int, list[CubemapFace]]]:
        """Update probes within budget.

        Args:
            camera_position: Camera world position
            visible_probe_ids: Set of visible probe IDs
            scene_render_func: Optional render function

        Returns:
            List of (probe_id, updated_faces) tuples
        """
        self._frame_count += 1
        start_time = time.perf_counter()

        # Update priorities
        for probe in self._probes.values():
            is_visible = visible_probe_ids is None or probe.probe_id in visible_probe_ids
            probe.update_priority(camera_position, is_visible)

        # Sort by priority
        self._priority_queue.sort(
            key=lambda pid: self._probes[pid].state.priority if pid in self._probes else 0,
            reverse=True
        )

        # Allocate budget
        remaining_renders = self._budget.max_face_renders
        urgent_renders = self._budget.reserve_urgent
        updates: list[tuple[int, list[CubemapFace]]] = []

        for probe_id in self._priority_queue:
            if remaining_renders <= 0:
                break

            # Time budget check
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if elapsed_ms >= self._budget.max_time_ms:
                break

            probe = self._probes.get(probe_id)
            if probe is None:
                continue

            # Determine face count based on priority
            if probe.state.priority > RealtimeProbeConstants.URGENT_PRIORITY_THRESHOLD:
                face_count = min(2, remaining_renders, urgent_renders + 1)
            else:
                face_count = 1

            # Update faces
            updated_faces = probe.update_faces(
                self._frame_count,
                face_count,
                scene_render_func,
            )

            if updated_faces:
                updates.append((probe_id, updated_faces))
                remaining_renders -= len(updated_faces)

        self._last_update_time = (time.perf_counter() - start_time) * 1000
        return updates

    def get_capture_budget(self) -> tuple[int, float]:
        """Get current capture budget info.

        Returns:
            Tuple of (max_face_renders, max_time_ms)
        """
        return (self._budget.max_face_renders, self._budget.max_time_ms)

    def set_capture_budget(self, max_faces: int, max_time_ms: float) -> None:
        """Set capture budget.

        Args:
            max_faces: Maximum face renders per frame
            max_time_ms: Maximum time in milliseconds
        """
        self._budget.max_face_renders = max(1, min(max_faces, RealtimeProbeConstants.MAX_CAPTURE_BUDGET))
        self._budget.max_time_ms = max(0.1, max_time_ms)

    def get_last_update_time(self) -> float:
        """Get last update time in milliseconds.

        Returns:
            Update time in ms
        """
        return self._last_update_time

    def find_affecting_probes(
        self,
        point: Vec3,
        max_probes: int = 4,
    ) -> list[tuple[RealtimeReflectionProbe, float]]:
        """Find probes affecting a point.

        Args:
            point: World position
            max_probes: Maximum probes to return

        Returns:
            List of (probe, weight) tuples
        """
        affecting = []

        for probe in self._probes.values():
            if probe.contains(point):
                center = probe.bounds.center
                extent = probe.bounds.max - probe.bounds.min
                max_dist = extent.length() * 0.5

                dist = point.distance(center)
                weight = 1.0 - min(dist / max(max_dist, 0.001), 1.0)

                affecting.append((probe, weight))

        affecting.sort(key=lambda x: x[1], reverse=True)
        return affecting[:max_probes]

    def sample(
        self,
        point: Vec3,
        direction: Vec3,
        roughness: float = 0.0,
    ) -> Vec3:
        """Sample blended probe lighting.

        Args:
            point: World position
            direction: Sample direction
            roughness: Surface roughness

        Returns:
            Blended probe color
        """
        affecting = self.find_affecting_probes(point)

        if not affecting:
            return Vec3(0, 0, 0)

        result = Vec3(0, 0, 0)
        total_weight = 0.0

        for probe, weight in affecting:
            sample = probe.sample(direction, roughness)
            result = result + sample * weight
            total_weight += weight

        if total_weight > 0:
            return result * (1.0 / total_weight)
        return Vec3(0, 0, 0)

    def mark_all_dirty(self, magnitude: float = 1.0) -> None:
        """Mark all probes as needing update.

        Args:
            magnitude: Change magnitude
        """
        for probe in self._probes.values():
            probe.mark_dirty(magnitude)

    def mark_region_dirty(self, bounds: AABB, magnitude: float = 1.0) -> None:
        """Mark probes in a region as dirty.

        Args:
            bounds: Region bounds
            magnitude: Change magnitude
        """
        for probe in self._probes.values():
            if probe.bounds.intersects(bounds):
                probe.mark_dirty(magnitude)

    def clear(self) -> None:
        """Remove all probes."""
        self._probes.clear()
        self._priority_queue.clear()


# -----------------------------------------------------------------------------
# Hybrid Mode Support
# -----------------------------------------------------------------------------

class HybridProbeMode(Enum):
    """Hybrid probe update modes."""
    REALTIME_ONLY = auto()      # Only realtime updates
    BAKED_ONLY = auto()         # Only use baked data
    BAKED_PLUS_DELTA = auto()   # Baked base + realtime delta
    ADAPTIVE = auto()           # Switch based on scene activity


@dataclass
class HybridProbeConfig:
    """Configuration for hybrid baked/realtime probes.

    Attributes:
        mode: Hybrid mode
        baked_asset: Optional baked probe asset
        delta_blend: Blend factor for delta updates
        activity_threshold: Scene activity threshold for adaptive mode
    """
    mode: HybridProbeMode = HybridProbeMode.REALTIME_ONLY
    baked_asset: Optional[BakedProbeAsset] = None
    delta_blend: float = 0.5
    activity_threshold: float = 0.3


class HybridReflectionProbe:
    """Reflection probe supporting baked + realtime hybrid modes.

    Combines a baked base with realtime delta updates for
    dynamic environments with static base lighting.
    """

    def __init__(
        self,
        realtime_probe: RealtimeReflectionProbe,
        config: HybridProbeConfig,
    ) -> None:
        """Initialize hybrid probe.

        Args:
            realtime_probe: Realtime probe component
            config: Hybrid configuration
        """
        self._realtime = realtime_probe
        self._config = config
        self._scene_activity: float = 0.0

    @property
    def realtime(self) -> RealtimeReflectionProbe:
        """Get realtime probe."""
        return self._realtime

    @property
    def config(self) -> HybridProbeConfig:
        """Get hybrid config."""
        return self._config

    @property
    def probe_id(self) -> int:
        """Get probe ID."""
        return self._realtime.probe_id

    def set_baked_asset(self, asset: BakedProbeAsset) -> None:
        """Set baked probe asset.

        Args:
            asset: Baked asset to use
        """
        self._config.baked_asset = asset
        if not asset.loaded:
            asset.load()

    def update_scene_activity(self, activity: float) -> None:
        """Update scene activity level.

        Args:
            activity: Activity level (0-1)
        """
        self._scene_activity = max(0.0, min(1.0, activity))

    def sample(self, direction: Vec3, roughness: float = 0.0) -> Vec3:
        """Sample hybrid probe.

        Args:
            direction: Sample direction
            roughness: Surface roughness

        Returns:
            Sampled color
        """
        mode = self._get_effective_mode()

        if mode == HybridProbeMode.BAKED_ONLY:
            return self._sample_baked(direction, roughness)
        elif mode == HybridProbeMode.REALTIME_ONLY:
            return self._realtime.sample(direction, roughness)
        else:  # BAKED_PLUS_DELTA
            return self._sample_hybrid(direction, roughness)

    def _get_effective_mode(self) -> HybridProbeMode:
        """Get effective mode based on config and activity."""
        if self._config.mode == HybridProbeMode.ADAPTIVE:
            if self._scene_activity > self._config.activity_threshold:
                return HybridProbeMode.BAKED_PLUS_DELTA
            return HybridProbeMode.BAKED_ONLY
        return self._config.mode

    def _sample_baked(self, direction: Vec3, roughness: float) -> Vec3:
        """Sample baked probe only."""
        if self._config.baked_asset is None or not self._config.baked_asset.loaded:
            return Vec3(0, 0, 0)
        return self._config.baked_asset.sample(direction, roughness)

    def _sample_hybrid(self, direction: Vec3, roughness: float) -> Vec3:
        """Sample hybrid baked + realtime."""
        baked = self._sample_baked(direction, roughness)
        realtime = self._realtime.sample(direction, roughness)

        blend = self._config.delta_blend
        return baked * (1 - blend) + realtime * blend
