"""Animation system configuration constants.

This module centralizes magic numbers and configuration values for the
animation crowds and systems modules, making them easy to tune and maintain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnimationTextureConfig:
    """Configuration for animation texture baking."""
    # Maximum texture dimensions
    MAX_TEXTURE_WIDTH: int = 4096
    MAX_TEXTURE_HEIGHT: int = 4096

    # Default texture dimensions (power of 2 for GPU efficiency)
    DEFAULT_TEXTURE_WIDTH: int = 1024
    DEFAULT_TEXTURE_HEIGHT: int = 2048

    # Maximum bones per skeleton for texture baking
    MAX_BONES_PER_TEXTURE: int = 256  # 256 bones * 2 pixels = 512 width

    # Maximum animation frames
    MAX_FRAMES_PER_ANIMATION: int = 4096

    # Pack/unpack value ranges for RGBA8 encoding
    PACK_MIN_VALUE: float = -100.0
    PACK_MAX_VALUE: float = 100.0


@dataclass(frozen=True)
class CrowdRendererConfig:
    """Configuration for crowd rendering."""
    # Instance buffer settings
    MAX_INSTANCES_PER_BATCH: int = 1000
    DEFAULT_BUFFER_CAPACITY: int = 64
    BUFFER_GROWTH_FACTOR: int = 2

    # GPU buffer alignment (bytes, typically 16 for SSE/SIMD)
    BUFFER_ALIGNMENT: int = 16

    # Floats per instance data
    TRANSFORM_FLOATS: int = 16  # 4x4 matrix
    ANIMATION_FLOATS: int = 4   # anim_index, time, speed, lod
    COLOR_FLOATS: int = 4       # RGBA tint


@dataclass(frozen=True)
class CrowdLODConfig:
    """Configuration for crowd LOD system."""
    # Default LOD distance thresholds (meters)
    DEFAULT_LOD_DISTANCES: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)

    # Maximum LOD levels
    MAX_LOD_LEVELS: int = 8

    # Default culling distance (meters)
    DEFAULT_CULL_DISTANCE: float = 300.0

    # LOD transition settings
    DEFAULT_TRANSITION_DURATION: float = 0.2  # seconds
    DEFAULT_HYSTERESIS: float = 1.0  # meters

    # Minimum update rate for distant LODs
    MIN_UPDATE_RATE: float = 0.25

    # Minimum bone count at lowest LOD
    MIN_BONES_AT_LOWEST_LOD: int = 4


@dataclass(frozen=True)
class CrowdBehaviorConfig:
    """Configuration for crowd behavior simulation."""
    # Default agent settings
    DEFAULT_AGENT_SPEED: float = 1.4  # meters/sec (average walking)
    DEFAULT_AGENT_TURN_SPEED: float = 3.14  # radians/sec (~180 deg/sec)
    DEFAULT_AGENT_RADIUS: float = 0.4  # meters

    # Avoidance settings
    DEFAULT_AVOIDANCE_RADIUS: float = 2.0  # meters
    DEFAULT_AVOIDANCE_STRENGTH: float = 1.5
    AVOIDANCE_PRIORITY_MULTIPLIER: float = 1.5

    # Movement settings
    ARRIVAL_THRESHOLD: float = 0.5  # meters
    VELOCITY_SMOOTHING: float = 4.0  # damping factor
    FLEE_ACCELERATION: float = 8.0  # faster response when fleeing

    # Idle behavior
    IDLE_VARIATION_MIN: float = 3.0  # seconds
    IDLE_VARIATION_MAX: float = 8.0  # seconds

    # Fleeing behavior
    FLEE_SPEED_MULTIPLIER: float = 1.5
    FLEE_SAFE_DISTANCE: float = 20.0  # meters

    # Minimum distance to avoid division by zero
    MIN_DISTANCE_EPSILON: float = 0.01


@dataclass(frozen=True)
class AnimationSystemConfig:
    """Configuration for animation ECS systems."""
    # System execution priorities (lower = earlier)
    PRIORITY_ANIMATION_GRAPH: int = 100
    PRIORITY_MOTION_MATCHING: int = 150
    PRIORITY_IK: int = 200
    PRIORITY_PROCEDURAL: int = 300
    PRIORITY_FACIAL: int = 300  # Parallel to procedural (different bones)
    PRIORITY_SKINNING: int = 400
    PRIORITY_CROWD: int = 500

    # Default transition durations
    DEFAULT_GRAPH_TRANSITION: float = 0.2  # seconds
    DEFAULT_MOTION_MATCH_TRANSITION: float = 0.2  # seconds

    # Motion matching settings
    MOTION_MATCH_SEARCH_INTERVAL: int = 10  # frames
    MOTION_MATCH_CONTINUATION_COST: float = 0.5


@dataclass(frozen=True)
class IKConfig:
    """Configuration for IK system."""
    # Solver defaults
    DEFAULT_MAX_ITERATIONS: int = 10
    DEFAULT_POSITION_TOLERANCE: float = 0.001  # meters
    DEFAULT_ROTATION_TOLERANCE: float = 0.01  # radians

    # Solver constraints
    MAX_CHAIN_LENGTH: int = 10

    # Distance thresholds to avoid numerical issues
    MIN_BONE_LENGTH: float = 0.001
    MIN_TARGET_DISTANCE: float = 0.001


@dataclass(frozen=True)
class ProceduralConfig:
    """Configuration for procedural animation."""
    # Spring defaults
    DEFAULT_SPRING_STIFFNESS: float = 10.0
    DEFAULT_SPRING_DAMPING: float = 0.5
    DEFAULT_SPRING_MASS: float = 1.0
    DEFAULT_MAX_STRETCH: float = 0.5

    # Look-at defaults
    DEFAULT_LOOK_SPEED: float = 5.0  # radians/sec
    DEFAULT_HORIZONTAL_LIMIT: float = 1.5708  # pi/2 radians (90 degrees)
    DEFAULT_VERTICAL_LIMIT: float = 1.0472    # pi/3 radians (60 degrees)

    # Sway defaults
    DEFAULT_SWAY_FREQUENCY: float = 1.0  # Hz
    DEFAULT_NOISE_AMOUNT: float = 0.2

    # Breathing defaults
    DEFAULT_BREATH_RATE: float = 0.25  # breaths/sec (15/min)
    DEFAULT_BREATH_DEPTH: float = 0.02


@dataclass(frozen=True)
class SkinningConfig:
    """Configuration for skinning system."""
    # Vertex skinning
    DEFAULT_MAX_INFLUENCES: int = 4

    # Dual quaternion threshold
    DQ_BLEND_THRESHOLD: float = 0.5

    # Numerical stability
    MIN_QUATERNION_LENGTH: float = 0.0001
    MIN_WEIGHT_THRESHOLD: float = 0.0001


@dataclass(frozen=True)
class FacialConfig:
    """Configuration for facial animation."""
    # Lip sync
    DEFAULT_PHONEME_TRANSITION: float = 0.08  # seconds
    SILENCE_VOLUME_THRESHOLD: float = 0.01

    # Eye tracking
    DEFAULT_BLINK_INTERVAL_MIN: float = 2.0  # seconds
    DEFAULT_BLINK_INTERVAL_MAX: float = 6.0  # seconds
    DEFAULT_BLINK_DURATION: float = 0.15  # seconds
    DEFAULT_SACCADE_INTENSITY: float = 0.01


@dataclass(frozen=True)
class CrowdSystemConfig:
    """Configuration for crowd system."""
    # Default update rate
    DEFAULT_UPDATE_RATE: float = 30.0  # updates/sec

    # Instance limits
    DEFAULT_MAX_VISIBLE: int = 10000
    DEFAULT_MAX_AGENTS: int = 100000

    # Default LOD distances for crowd system
    DEFAULT_LOD_DISTANCES: tuple[float, ...] = (20.0, 50.0, 100.0, 200.0)
    DEFAULT_CULL_DISTANCE: float = 300.0


# Global configuration instances
ANIMATION_TEXTURE_CONFIG = AnimationTextureConfig()
CROWD_RENDERER_CONFIG = CrowdRendererConfig()
CROWD_LOD_CONFIG = CrowdLODConfig()
CROWD_BEHAVIOR_CONFIG = CrowdBehaviorConfig()
ANIMATION_SYSTEM_CONFIG = AnimationSystemConfig()
IK_CONFIG = IKConfig()
PROCEDURAL_CONFIG = ProceduralConfig()
SKINNING_CONFIG = SkinningConfig()
FACIAL_CONFIG = FacialConfig()
CROWD_SYSTEM_CONFIG = CrowdSystemConfig()
