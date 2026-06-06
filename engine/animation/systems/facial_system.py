"""ECS system for facial animation (T-AN-9.8).

Comprehensive facial animation processing system that integrates:
- Blend shapes (T-AN-7.1)
- FACS action units (T-AN-7.2)
- Lip sync phonemes (T-AN-7.3)
- Eye animation (T-AN-7.4)

Key Features:
- @system(phase="animation", order=4) annotation for ECS scheduling
- Layered composition with priority handling
- Per-region masking (upper face, lower face, eyes)
- Audio integration for lip sync timing
- Task-parallel per-entity evaluation
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import FACIAL_CONFIG, ANIMATION_SYSTEM_CONFIG

if TYPE_CHECKING:
    from engine.core.tasks.scheduler import TaskScheduler, TaskHandle


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(
    phase: str = "update",
    order: int = 0,
    reads: Optional[Tuple[str, ...]] = None,
    writes: Optional[Tuple[str, ...]] = None,
) -> Callable:
    """Decorator to mark a class as an ECS system with phase scheduling.

    Args:
        phase: Frame phase for execution ("animation", "update", "render", etc.)
        order: Execution order within phase (lower = earlier)
        reads: Component types this system reads from
        writes: Component types this system writes to

    Returns:
        Decorated class with system metadata.
    """
    def decorator(cls: type) -> type:
        cls._system_phase = phase
        cls._system_order = order
        cls._system_reads = reads or ()
        cls._system_writes = writes or ()
        return cls
    return decorator


# =============================================================================
# ENUMERATIONS
# =============================================================================


class EmotionState(Enum):
    """Basic emotion states based on Ekman's universal emotions."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    DISGUSTED = auto()
    FEARFUL = auto()
    CONTEMPT = auto()
    CUSTOM = auto()


class LipSyncPhoneme(Enum):
    """Phoneme categories for lip sync based on Preston Blair chart."""
    SILENCE = auto()  # Closed mouth / rest
    AA = auto()       # "ah" sound - jaw open
    EE = auto()       # "ee" sound - wide smile
    IH = auto()       # "ih" sound - slight smile
    OH = auto()       # "oh" sound - rounded lips
    OO = auto()       # "oo" sound - pursed lips
    EH = auto()       # "eh" sound - relaxed open
    AE = auto()       # "a" as in "cat"
    UH = auto()       # "uh" sound
    CH = auto()       # "ch"/"sh" sounds - funnel
    FV = auto()       # "f"/"v" sounds - teeth on lip
    TH = auto()       # "th" sounds - tongue visible
    MBP = auto()      # "m"/"b"/"p" sounds - closed lips
    LN = auto()       # "l"/"n" sounds - tongue up
    WQ = auto()       # "w"/"q" sounds - rounded


class FacialRegion(Enum):
    """Face regions for masking."""
    UPPER_FACE = auto()   # Forehead, brows
    MID_FACE = auto()     # Eyes, cheeks, nose
    LOWER_FACE = auto()   # Mouth, jaw, chin
    EYES = auto()         # Eyes only (independent)
    BROWS = auto()        # Eyebrows only
    MOUTH = auto()        # Mouth area only
    ALL = auto()          # All regions


class FacialLayerPriority(Enum):
    """Priority levels for facial animation layers.

    Lower values have lower priority and can be overridden.
    """
    BASE = 0           # Base/idle expressions
    BLEND_SHAPE = 10   # Direct blend shape animations
    FACS = 20          # FACS action unit expressions
    EMOTION = 30       # Emotional expressions
    LIP_SYNC = 40      # Lip sync overrides mouth
    EYE = 50           # Eye animation (independent layer)
    OVERRIDE = 100     # Manual overrides


class BlendMode(Enum):
    """How layers combine with lower layers."""
    REPLACE = auto()   # Replace lower layer values
    ADDITIVE = auto()  # Add to lower layer values
    MULTIPLY = auto()  # Multiply with lower layer values
    MAX = auto()       # Take maximum value


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class Expression:
    """Facial expression definition.

    Attributes:
        name: Expression name
        blend_shapes: Blend shape weights (shape_name -> weight)
        bone_offsets: Bone transform offsets (bone_index -> transform)
        intensity: Expression intensity (0-1)
    """
    name: str = ""
    blend_shapes: dict[str, float] = field(default_factory=dict)
    bone_offsets: dict[int, Transform] = field(default_factory=dict)
    intensity: float = 1.0

    def blend_with(self, other: Expression, weight: float) -> Expression:
        """Blend this expression with another.

        Args:
            other: Expression to blend with
            weight: Blend weight (0=this, 1=other)

        Returns:
            New blended expression
        """
        result = Expression(name=f"{self.name}_{other.name}_blend")

        # Blend shapes
        all_shapes = set(self.blend_shapes.keys()) | set(other.blend_shapes.keys())
        for shape in all_shapes:
            val_a = self.blend_shapes.get(shape, 0.0) * self.intensity
            val_b = other.blend_shapes.get(shape, 0.0) * other.intensity
            result.blend_shapes[shape] = val_a * (1 - weight) + val_b * weight

        # Blend bone offsets
        all_bones = set(self.bone_offsets.keys()) | set(other.bone_offsets.keys())
        for bone in all_bones:
            t_a = self.bone_offsets.get(bone, Transform.identity())
            t_b = other.bone_offsets.get(bone, Transform.identity())
            result.bone_offsets[bone] = t_a.lerp(t_b, weight)

        result.intensity = 1.0
        return result

    def scale(self, factor: float) -> Expression:
        """Scale expression intensity.

        Args:
            factor: Scale factor

        Returns:
            New scaled expression
        """
        result = Expression(
            name=self.name,
            intensity=self.intensity * factor,
        )
        result.blend_shapes = {k: v * factor for k, v in self.blend_shapes.items()}
        # Bone offsets would need proper scaling (lerp toward identity)
        result.bone_offsets = self.bone_offsets.copy()
        return result


@dataclass
class FacialLayer:
    """A single facial animation layer.

    Attributes:
        name: Layer identifier
        priority: Layer priority (higher overrides lower)
        weight: Layer blend weight (0-1)
        blend_mode: How this layer combines with others
        region_mask: Which facial regions this layer affects
        blend_shapes: Current blend shape weights
        bone_offsets: Current bone transform offsets
        enabled: Whether layer is active
    """
    name: str
    priority: FacialLayerPriority = FacialLayerPriority.BASE
    weight: float = 1.0
    blend_mode: BlendMode = BlendMode.REPLACE
    region_mask: FacialRegion = FacialRegion.ALL
    blend_shapes: dict[str, float] = field(default_factory=dict)
    bone_offsets: dict[int, Transform] = field(default_factory=dict)
    enabled: bool = True

    def clear(self) -> None:
        """Clear all weights."""
        self.blend_shapes.clear()
        self.bone_offsets.clear()


@dataclass
class FaceRig:
    """Face rig configuration.

    Defines the mapping between abstract facial controls and
    actual blend shapes / bones.

    Attributes:
        blend_shape_names: Available blend shape names
        jaw_bone: Index of jaw bone
        eye_bones: Indices of eye bones (left, right)
        eyelid_bones: Indices of eyelid bones (upper_left, lower_left, upper_right, lower_right)
        eyebrow_bones: Indices of eyebrow bones
        viseme_shapes: Mapping from phonemes to blend shape names
        emotion_shapes: Mapping from emotions to blend shape lists
        region_shapes: Mapping from regions to shape names
    """
    blend_shape_names: list[str] = field(default_factory=list)
    jaw_bone: int = -1
    eye_bones: tuple[int, int] = (-1, -1)
    eyelid_bones: list[int] = field(default_factory=list)
    eyebrow_bones: list[int] = field(default_factory=list)

    # Phoneme to viseme blend shape mapping
    viseme_shapes: dict[LipSyncPhoneme, str] = field(default_factory=dict)
    # Emotion to blend shape mapping
    emotion_shapes: dict[EmotionState, list[str]] = field(default_factory=dict)
    # Region to blend shape names
    region_shapes: dict[FacialRegion, set[str]] = field(default_factory=dict)

    def get_shapes_for_region(self, region: FacialRegion) -> set[str]:
        """Get blend shape names for a facial region."""
        if region == FacialRegion.ALL:
            return set(self.blend_shape_names)
        return self.region_shapes.get(region, set())

    def setup_default_regions(self) -> None:
        """Set up default region mappings based on ARKit naming."""
        self.region_shapes = {
            FacialRegion.UPPER_FACE: {
                "browInnerUp", "browDownLeft", "browDownRight",
                "browOuterUpLeft", "browOuterUpRight",
            },
            FacialRegion.MID_FACE: {
                "eyeBlinkLeft", "eyeBlinkRight",
                "eyeLookUpLeft", "eyeLookUpRight",
                "eyeLookDownLeft", "eyeLookDownRight",
                "eyeLookInLeft", "eyeLookInRight",
                "eyeLookOutLeft", "eyeLookOutRight",
                "eyeWideLeft", "eyeWideRight",
                "eyeSquintLeft", "eyeSquintRight",
                "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
                "noseSneerLeft", "noseSneerRight",
            },
            FacialRegion.LOWER_FACE: {
                "jawOpen", "jawForward", "jawLeft", "jawRight",
                "mouthClose", "mouthFunnel", "mouthPucker",
                "mouthLeft", "mouthRight",
                "mouthSmileLeft", "mouthSmileRight",
                "mouthFrownLeft", "mouthFrownRight",
                "mouthDimpleLeft", "mouthDimpleRight",
                "mouthStretchLeft", "mouthStretchRight",
                "mouthRollLower", "mouthRollUpper",
                "mouthShrugLower", "mouthShrugUpper",
                "mouthPressLeft", "mouthPressRight",
                "mouthLowerDownLeft", "mouthLowerDownRight",
                "mouthUpperUpLeft", "mouthUpperUpRight",
            },
            FacialRegion.EYES: {
                "eyeBlinkLeft", "eyeBlinkRight",
                "eyeLookUpLeft", "eyeLookUpRight",
                "eyeLookDownLeft", "eyeLookDownRight",
                "eyeLookInLeft", "eyeLookInRight",
                "eyeLookOutLeft", "eyeLookOutRight",
                "eyeWideLeft", "eyeWideRight",
                "eyeSquintLeft", "eyeSquintRight",
            },
            FacialRegion.BROWS: {
                "browInnerUp", "browDownLeft", "browDownRight",
                "browOuterUpLeft", "browOuterUpRight",
            },
            FacialRegion.MOUTH: {
                "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
                "mouthLeft", "mouthRight",
                "mouthSmileLeft", "mouthSmileRight",
                "mouthFrownLeft", "mouthFrownRight",
                "mouthStretchLeft", "mouthStretchRight",
                "mouthRollLower", "mouthRollUpper",
                "mouthPressLeft", "mouthPressRight",
                "mouthLowerDownLeft", "mouthLowerDownRight",
                "mouthUpperUpLeft", "mouthUpperUpRight",
            },
        }


@dataclass
class LipSyncState:
    """Current lip sync state.

    Attributes:
        current_phoneme: Active phoneme
        phoneme_weight: Current phoneme weight
        transition_time: Time for phoneme transitions
        audio_intensity: Current audio intensity/volume
        audio_time: Current playback time in audio
        is_speaking: Whether actively speaking
    """
    current_phoneme: LipSyncPhoneme = LipSyncPhoneme.SILENCE
    phoneme_weight: float = 0.0
    transition_time: float = FACIAL_CONFIG.DEFAULT_PHONEME_TRANSITION
    audio_intensity: float = 0.0
    audio_time: float = 0.0
    is_speaking: bool = False

    # Internal transition state
    _previous_phoneme: LipSyncPhoneme = LipSyncPhoneme.SILENCE
    _transition_progress: float = 1.0
    _target_weight: float = 0.0

    def start_speaking(self) -> None:
        """Start lip sync playback."""
        self.is_speaking = True
        self.audio_time = 0.0

    def stop_speaking(self) -> None:
        """Stop lip sync playback."""
        self.is_speaking = False
        self.current_phoneme = LipSyncPhoneme.SILENCE
        self.phoneme_weight = 0.0
        self._transition_progress = 1.0


@dataclass
class EyeState:
    """Eye tracking state.

    Attributes:
        look_target: World position to look at
        look_weight: Look-at blend weight
        blink_timer: Time until next blink
        blink_duration: Duration of blink
        is_blinking: Whether currently blinking
        blink_progress: Current blink progress (0-1)
        pupil_dilation: Pupil dilation factor (0-1)
        saccade_intensity: Intensity of micro eye movements
    """
    look_target: Vec3 = field(default_factory=Vec3.zero)
    look_weight: float = 1.0
    blink_timer: float = field(default_factory=lambda: (
        FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MIN +
        FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MAX
    ) / 2)
    blink_duration: float = FACIAL_CONFIG.DEFAULT_BLINK_DURATION
    is_blinking: bool = False
    blink_progress: float = 0.0
    pupil_dilation: float = 0.5

    # Saccade (micro eye movements)
    saccade_intensity: float = FACIAL_CONFIG.DEFAULT_SACCADE_INTENSITY
    _saccade_offset: Vec3 = field(default_factory=Vec3.zero)
    _saccade_timer: float = 0.0

    # Vergence for near objects
    vergence_angle: float = 0.0

    # Target tracking smoothing
    _smooth_target: Vec3 = field(default_factory=Vec3.zero)
    _target_velocity: Vec3 = field(default_factory=Vec3.zero)


@dataclass
class FACSState:
    """FACS Action Unit state.

    Attributes:
        au_intensities: Current intensity for each AU
        au_left_intensities: Left-side intensities for bilateral AUs
        au_right_intensities: Right-side intensities for bilateral AUs
    """
    au_intensities: dict[str, float] = field(default_factory=dict)
    au_left_intensities: dict[str, float] = field(default_factory=dict)
    au_right_intensities: dict[str, float] = field(default_factory=dict)

    def set_au(
        self,
        au_name: str,
        intensity: float,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """Set AU intensity."""
        self.au_intensities[au_name] = max(0.0, min(1.0, intensity))
        if left is not None:
            self.au_left_intensities[au_name] = max(0.0, min(1.0, left))
        if right is not None:
            self.au_right_intensities[au_name] = max(0.0, min(1.0, right))

    def get_au(self, au_name: str) -> float:
        """Get AU intensity."""
        return self.au_intensities.get(au_name, 0.0)

    def clear(self) -> None:
        """Reset all AUs to zero."""
        self.au_intensities.clear()
        self.au_left_intensities.clear()
        self.au_right_intensities.clear()


@dataclass
class AudioSyncData:
    """Audio synchronization data for lip sync.

    Attributes:
        phoneme_timeline: List of (time, phoneme, weight) tuples
        audio_duration: Total audio duration in seconds
        sample_rate: Audio sample rate
        current_index: Current position in timeline
    """
    phoneme_timeline: list[tuple[float, LipSyncPhoneme, float]] = field(default_factory=list)
    audio_duration: float = 0.0
    sample_rate: int = 44100
    current_index: int = 0

    def get_phoneme_at_time(self, time: float) -> tuple[LipSyncPhoneme, float]:
        """Get phoneme and weight at given time."""
        if not self.phoneme_timeline or time < 0:
            return (LipSyncPhoneme.SILENCE, 0.0)

        # Binary search for efficiency
        left, right = 0, len(self.phoneme_timeline) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self.phoneme_timeline[mid][0] <= time:
                left = mid
            else:
                right = mid - 1

        if left < len(self.phoneme_timeline):
            _, phoneme, weight = self.phoneme_timeline[left]
            return (phoneme, weight)
        return (LipSyncPhoneme.SILENCE, 0.0)

    def add_phoneme(self, time: float, phoneme: LipSyncPhoneme, weight: float = 1.0) -> None:
        """Add phoneme event to timeline."""
        self.phoneme_timeline.append((time, phoneme, weight))
        self.phoneme_timeline.sort(key=lambda x: x[0])


# =============================================================================
# FACIAL COMPONENT
# =============================================================================


@dataclass
class FacialComponent:
    """Component for entities with facial animation.

    This is the main data container for facial animation state.
    It holds all the configuration and runtime state needed for
    blend shapes, FACS, lip sync, and eye animation.

    Attributes:
        face_rig: Face rig configuration
        current_emotion: Current emotion state
        emotion_intensity: Emotion intensity (0-1)
        lip_sync: Lip sync state
        eye_state: Eye tracking state
        facs_state: FACS Action Unit state
        audio_sync: Audio synchronization data
        enabled: Whether facial animation is enabled
    """
    face_rig: FaceRig = field(default_factory=FaceRig)
    current_emotion: EmotionState = EmotionState.NEUTRAL
    emotion_intensity: float = 0.0
    lip_sync: LipSyncState = field(default_factory=LipSyncState)
    eye_state: EyeState = field(default_factory=EyeState)
    facs_state: FACSState = field(default_factory=FACSState)
    audio_sync: AudioSyncData = field(default_factory=AudioSyncData)
    enabled: bool = True

    # Animation layers
    layers: dict[str, FacialLayer] = field(default_factory=dict)

    # Custom expression override
    custom_expression: Optional[Expression] = None
    custom_expression_weight: float = 0.0

    # Output blend shapes (after all layer composition)
    output_blend_shapes: dict[str, float] = field(default_factory=dict)
    output_bone_offsets: dict[int, Transform] = field(default_factory=dict)

    # Expression library
    expressions: dict[str, Expression] = field(default_factory=dict)

    # Dirty flag for optimization
    _dirty: bool = True

    def __post_init__(self) -> None:
        """Initialize default layers."""
        if not self.layers:
            self._setup_default_layers()

    def _setup_default_layers(self) -> None:
        """Set up default animation layers."""
        self.layers = {
            "base": FacialLayer(
                name="base",
                priority=FacialLayerPriority.BASE,
                blend_mode=BlendMode.REPLACE,
            ),
            "blend_shape": FacialLayer(
                name="blend_shape",
                priority=FacialLayerPriority.BLEND_SHAPE,
                blend_mode=BlendMode.ADDITIVE,
            ),
            "facs": FacialLayer(
                name="facs",
                priority=FacialLayerPriority.FACS,
                blend_mode=BlendMode.REPLACE,
            ),
            "emotion": FacialLayer(
                name="emotion",
                priority=FacialLayerPriority.EMOTION,
                blend_mode=BlendMode.REPLACE,
            ),
            "lip_sync": FacialLayer(
                name="lip_sync",
                priority=FacialLayerPriority.LIP_SYNC,
                blend_mode=BlendMode.REPLACE,
                region_mask=FacialRegion.MOUTH,
            ),
            "eye": FacialLayer(
                name="eye",
                priority=FacialLayerPriority.EYE,
                blend_mode=BlendMode.REPLACE,
                region_mask=FacialRegion.EYES,
            ),
            "override": FacialLayer(
                name="override",
                priority=FacialLayerPriority.OVERRIDE,
                blend_mode=BlendMode.REPLACE,
                weight=0.0,
            ),
        }

    def set_emotion(self, emotion: EmotionState, intensity: float = 1.0) -> None:
        """Set current emotion."""
        self.current_emotion = emotion
        self.emotion_intensity = max(0.0, min(1.0, intensity))
        self._dirty = True

    def set_phoneme(self, phoneme: LipSyncPhoneme, weight: float = 1.0) -> None:
        """Set current phoneme for lip sync."""
        if phoneme != self.lip_sync.current_phoneme:
            self.lip_sync._previous_phoneme = self.lip_sync.current_phoneme
            self.lip_sync.current_phoneme = phoneme
            self.lip_sync._transition_progress = 0.0
        self.lip_sync.phoneme_weight = weight
        self._dirty = True

    def set_look_target(self, target: Vec3, weight: float = 1.0) -> None:
        """Set eye look target."""
        self.eye_state.look_target = target
        self.eye_state.look_weight = weight
        self._dirty = True

    def add_expression(self, expression: Expression) -> None:
        """Add expression to library."""
        self.expressions[expression.name] = expression

    def play_expression(self, name: str, weight: float = 1.0) -> bool:
        """Play named expression."""
        if name in self.expressions:
            self.custom_expression = self.expressions[name]
            self.custom_expression_weight = weight
            self._dirty = True
            return True
        return False

    def get_layer(self, name: str) -> Optional[FacialLayer]:
        """Get layer by name."""
        return self.layers.get(name)

    def set_layer_weight(self, name: str, weight: float) -> bool:
        """Set layer weight."""
        if name in self.layers:
            self.layers[name].weight = max(0.0, min(1.0, weight))
            self._dirty = True
            return True
        return False


# =============================================================================
# FACIAL SYSTEM
# =============================================================================


@system(phase="animation", order=ANIMATION_SYSTEM_CONFIG.PRIORITY_FACIAL)
class FacialSystem:
    """ECS system for facial animation.

    Processes facial animation in the following order:
    1. Blend shapes (T-AN-7.1) - base layer
    2. FACS action units (T-AN-7.2) - expression overlay
    3. Lip sync phonemes (T-AN-7.3) - mouth override
    4. Eye animation (T-AN-7.4) - independent layer

    Layer composition respects priorities and region masking.
    """

    def __init__(self) -> None:
        """Initialize facial system."""
        self._emotion_expressions: dict[EmotionState, Expression] = {}
        self._phoneme_blend_shapes: dict[LipSyncPhoneme, dict[str, float]] = {}
        self._au_blend_shapes: dict[str, dict[str, float]] = {}
        self._setup_default_emotions()
        self._setup_default_phonemes()
        self._setup_default_aus()

        # Performance stats
        self._last_update_time_ms: float = 0.0
        self._entities_processed: int = 0

    def _setup_default_emotions(self) -> None:
        """Setup default emotion expressions."""
        self._emotion_expressions[EmotionState.NEUTRAL] = Expression(
            name="neutral",
            blend_shapes={},
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.HAPPY] = Expression(
            name="happy",
            blend_shapes={
                "mouthSmileLeft": 0.8,
                "mouthSmileRight": 0.8,
                "cheekSquintLeft": 0.5,
                "cheekSquintRight": 0.5,
                "eyeSquintLeft": 0.3,
                "eyeSquintRight": 0.3,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.SAD] = Expression(
            name="sad",
            blend_shapes={
                "mouthFrownLeft": 0.6,
                "mouthFrownRight": 0.6,
                "browInnerUp": 0.5,
                "browDownLeft": 0.3,
                "browDownRight": 0.3,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.ANGRY] = Expression(
            name="angry",
            blend_shapes={
                "browDownLeft": 0.7,
                "browDownRight": 0.7,
                "noseSneerLeft": 0.3,
                "noseSneerRight": 0.3,
                "jawForward": 0.2,
                "mouthPressLeft": 0.4,
                "mouthPressRight": 0.4,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.SURPRISED] = Expression(
            name="surprised",
            blend_shapes={
                "browInnerUp": 0.8,
                "browOuterUpLeft": 0.8,
                "browOuterUpRight": 0.8,
                "eyeWideLeft": 0.6,
                "eyeWideRight": 0.6,
                "jawOpen": 0.3,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.DISGUSTED] = Expression(
            name="disgusted",
            blend_shapes={
                "noseSneerLeft": 0.8,
                "noseSneerRight": 0.8,
                "mouthUpperUpLeft": 0.5,
                "mouthUpperUpRight": 0.5,
                "browDownLeft": 0.4,
                "browDownRight": 0.4,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.FEARFUL] = Expression(
            name="fearful",
            blend_shapes={
                "browInnerUp": 0.6,
                "browOuterUpLeft": 0.6,
                "browOuterUpRight": 0.6,
                "eyeWideLeft": 0.5,
                "eyeWideRight": 0.5,
                "mouthStretchLeft": 0.4,
                "mouthStretchRight": 0.4,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.CONTEMPT] = Expression(
            name="contempt",
            blend_shapes={
                "mouthSmileRight": 0.4,
                "mouthDimpleRight": 0.5,
            },
            intensity=1.0,
        )

    def _setup_default_phonemes(self) -> None:
        """Setup default phoneme to blend shape mappings."""
        self._phoneme_blend_shapes = {
            LipSyncPhoneme.SILENCE: {
                "mouthClose": 0.1,
            },
            LipSyncPhoneme.AA: {
                "jawOpen": 0.7,
                "mouthFunnel": 0.1,
            },
            LipSyncPhoneme.EE: {
                "jawOpen": 0.2,
                "mouthSmileLeft": 0.5,
                "mouthSmileRight": 0.5,
                "mouthStretchLeft": 0.3,
                "mouthStretchRight": 0.3,
            },
            LipSyncPhoneme.IH: {
                "jawOpen": 0.15,
                "mouthSmileLeft": 0.3,
                "mouthSmileRight": 0.3,
            },
            LipSyncPhoneme.OH: {
                "jawOpen": 0.4,
                "mouthFunnel": 0.6,
            },
            LipSyncPhoneme.OO: {
                "jawOpen": 0.25,
                "mouthPucker": 0.7,
                "mouthFunnel": 0.4,
            },
            LipSyncPhoneme.EH: {
                "jawOpen": 0.3,
                "mouthStretchLeft": 0.2,
                "mouthStretchRight": 0.2,
            },
            LipSyncPhoneme.AE: {
                "jawOpen": 0.5,
                "mouthStretchLeft": 0.3,
                "mouthStretchRight": 0.3,
            },
            LipSyncPhoneme.UH: {
                "jawOpen": 0.25,
                "mouthFunnel": 0.3,
            },
            LipSyncPhoneme.CH: {
                "jawOpen": 0.15,
                "mouthFunnel": 0.5,
                "mouthPucker": 0.2,
            },
            LipSyncPhoneme.FV: {
                "jawOpen": 0.1,
                "mouthLowerDownLeft": 0.3,
                "mouthLowerDownRight": 0.3,
            },
            LipSyncPhoneme.TH: {
                "jawOpen": 0.15,
            },
            LipSyncPhoneme.MBP: {
                "mouthClose": 0.9,
                "mouthPressLeft": 0.5,
                "mouthPressRight": 0.5,
            },
            LipSyncPhoneme.LN: {
                "jawOpen": 0.2,
            },
            LipSyncPhoneme.WQ: {
                "jawOpen": 0.15,
                "mouthPucker": 0.6,
            },
        }

    def _setup_default_aus(self) -> None:
        """Setup default FACS AU to blend shape mappings."""
        self._au_blend_shapes = {
            "AU1": {"browInnerUp": 1.0},
            "AU2": {"browOuterUpLeft": 1.0, "browOuterUpRight": 1.0},
            "AU4": {"browDownLeft": 1.0, "browDownRight": 1.0},
            "AU5": {"eyeWideLeft": 1.0, "eyeWideRight": 1.0},
            "AU6": {"cheekSquintLeft": 1.0, "cheekSquintRight": 1.0},
            "AU7": {"eyeSquintLeft": 1.0, "eyeSquintRight": 1.0},
            "AU9": {"noseSneerLeft": 1.0, "noseSneerRight": 1.0},
            "AU10": {"mouthUpperUpLeft": 1.0, "mouthUpperUpRight": 1.0},
            "AU12": {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0},
            "AU14": {"mouthDimpleLeft": 1.0, "mouthDimpleRight": 1.0},
            "AU15": {"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0},
            "AU17": {"mouthShrugLower": 1.0},
            "AU20": {"mouthStretchLeft": 1.0, "mouthStretchRight": 1.0},
            "AU23": {"mouthPucker": 0.5},
            "AU24": {"mouthPressLeft": 1.0, "mouthPressRight": 1.0},
            "AU25": {"jawOpen": 0.3},
            "AU26": {"jawOpen": 0.7},
            "AU27": {"jawOpen": 1.0},
            "AU28": {"mouthRollLower": 1.0, "mouthRollUpper": 1.0},
            "AU43": {"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0},
        }

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, FacialComponent]],
    ) -> None:
        """Update all facial components.

        This is the main entry point called by the ECS scheduler.

        Args:
            world: ECS world
            dt: Delta time in seconds
            entity_components: List of (entity, component) tuples
        """
        import time
        start_time = time.perf_counter()

        self._entities_processed = 0
        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_component(component, dt)
            self._entities_processed += 1

        self._last_update_time_ms = (time.perf_counter() - start_time) * 1000

    def _update_component(self, component: FacialComponent, dt: float) -> None:
        """Update single facial component.

        Processes effects in order:
        1. Blend shapes (base layer)
        2. FACS action units
        3. Lip sync phonemes
        4. Eye animation

        Args:
            component: The facial component to update
            dt: Delta time in seconds
        """
        # Clear outputs
        component.output_blend_shapes.clear()
        component.output_bone_offsets.clear()

        # Clear layer outputs
        for layer in component.layers.values():
            layer.clear()

        # 1. Update blend shapes (base layer)
        self._update_blend_shapes(component)

        # 2. Update FACS
        self._update_facs(component)

        # 3. Update emotion
        self._update_emotion(component)

        # 4. Update lip sync
        self._update_lip_sync(component, dt)

        # 5. Update eye tracking
        self._update_eyes(component, dt)

        # 6. Apply custom expression override
        if component.custom_expression and component.custom_expression_weight > 0:
            self._apply_expression_to_layer(
                component.layers.get("override"),
                component.custom_expression,
                component.custom_expression_weight
            )
            component.layers["override"].weight = 1.0

        # 7. Compose all layers
        self._compose_layers(component)

        component._dirty = False

    def _update_blend_shapes(self, component: FacialComponent) -> None:
        """Update base blend shape layer."""
        layer = component.layers.get("blend_shape")
        if not layer or not layer.enabled:
            return

        # Base blend shapes could be driven by external animation clips
        # For now, this layer starts empty and can be populated externally

    def _update_facs(self, component: FacialComponent) -> None:
        """Update FACS action unit layer."""
        layer = component.layers.get("facs")
        if not layer or not layer.enabled:
            return

        facs = component.facs_state
        for au_name, intensity in facs.au_intensities.items():
            if intensity <= 0.001:
                continue

            au_shapes = self._au_blend_shapes.get(au_name, {})
            for shape_name, shape_weight in au_shapes.items():
                # Handle bilateral shapes
                if au_name in facs.au_left_intensities:
                    left_int = facs.au_left_intensities[au_name]
                    if "Left" in shape_name:
                        intensity = left_int
                if au_name in facs.au_right_intensities:
                    right_int = facs.au_right_intensities[au_name]
                    if "Right" in shape_name:
                        intensity = right_int

                current = layer.blend_shapes.get(shape_name, 0.0)
                layer.blend_shapes[shape_name] = current + shape_weight * intensity

    def _update_emotion(self, component: FacialComponent) -> None:
        """Update emotion-based facial expression."""
        layer = component.layers.get("emotion")
        if not layer or not layer.enabled:
            return

        if component.emotion_intensity <= 0:
            return

        emotion_expr = self._emotion_expressions.get(component.current_emotion)
        if emotion_expr:
            self._apply_expression_to_layer(
                layer,
                emotion_expr,
                component.emotion_intensity
            )

    def _update_lip_sync(self, component: FacialComponent, dt: float) -> None:
        """Update lip sync animation.

        Args:
            component: Facial component
            dt: Delta time
        """
        layer = component.layers.get("lip_sync")
        if not layer or not layer.enabled:
            return

        lip_sync = component.lip_sync

        # Update audio time if speaking
        if lip_sync.is_speaking:
            lip_sync.audio_time += dt

            # Get phoneme from audio sync data
            if component.audio_sync.phoneme_timeline:
                phoneme, weight = component.audio_sync.get_phoneme_at_time(
                    lip_sync.audio_time
                )
                if phoneme != lip_sync.current_phoneme:
                    component.set_phoneme(phoneme, weight)

        # Update transition
        if lip_sync._transition_progress < 1.0:
            lip_sync._transition_progress += dt / lip_sync.transition_time
            lip_sync._transition_progress = min(1.0, lip_sync._transition_progress)

        # Apply previous phoneme (blending out)
        if lip_sync._transition_progress < 1.0:
            prev_shapes = self._phoneme_blend_shapes.get(
                lip_sync._previous_phoneme, {}
            )
            prev_weight = (1.0 - lip_sync._transition_progress) * lip_sync.phoneme_weight
            for shape_name, shape_weight in prev_shapes.items():
                current = layer.blend_shapes.get(shape_name, 0.0)
                layer.blend_shapes[shape_name] = max(current, shape_weight * prev_weight)

        # Apply current phoneme (blending in)
        current_shapes = self._phoneme_blend_shapes.get(
            lip_sync.current_phoneme, {}
        )
        current_weight = lip_sync._transition_progress * lip_sync.phoneme_weight
        for shape_name, shape_weight in current_shapes.items():
            current = layer.blend_shapes.get(shape_name, 0.0)
            layer.blend_shapes[shape_name] = max(current, shape_weight * current_weight)

        # Apply jaw bone for open mouth phonemes
        face_rig = component.face_rig
        if face_rig.jaw_bone >= 0:
            jaw_open = 0.0
            if lip_sync.current_phoneme in (
                LipSyncPhoneme.AA, LipSyncPhoneme.OH, LipSyncPhoneme.EH, LipSyncPhoneme.AE
            ):
                jaw_open = 0.5 * lip_sync.phoneme_weight
            elif lip_sync.current_phoneme in (
                LipSyncPhoneme.EE, LipSyncPhoneme.IH, LipSyncPhoneme.OO
            ):
                jaw_open = 0.25 * lip_sync.phoneme_weight

            if jaw_open > 0:
                jaw_rotation = Quat.from_euler(-jaw_open * 0.3, 0, 0)
                layer.bone_offsets[face_rig.jaw_bone] = Transform(
                    rotation=jaw_rotation
                )

    def _update_eyes(self, component: FacialComponent, dt: float) -> None:
        """Update eye tracking and blinking.

        Args:
            component: Facial component
            dt: Delta time
        """
        layer = component.layers.get("eye")
        if not layer or not layer.enabled:
            return

        eye_state = component.eye_state
        face_rig = component.face_rig

        # Update blink timer
        eye_state.blink_timer -= dt
        if eye_state.blink_timer <= 0 and not eye_state.is_blinking:
            eye_state.is_blinking = True
            eye_state.blink_progress = 0.0
            # Random next blink interval
            eye_state.blink_timer = (
                FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MIN +
                random.random() * (
                    FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MAX -
                    FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MIN
                )
            )

        # Process blinking
        if eye_state.is_blinking:
            eye_state.blink_progress += dt / eye_state.blink_duration
            if eye_state.blink_progress >= 1.0:
                eye_state.is_blinking = False
                eye_state.blink_progress = 0.0

            # Blink curve (close then open quickly)
            blink_curve = 1.0 - abs(eye_state.blink_progress * 2.0 - 1.0)
            blink_weight = blink_curve * blink_curve  # Ease in/out

            # Apply to eyelid blend shapes
            layer.blend_shapes["eyeBlinkLeft"] = blink_weight
            layer.blend_shapes["eyeBlinkRight"] = blink_weight

            # Apply to eyelid bones if available
            for eyelid_bone in face_rig.eyelid_bones:
                eyelid_rotation = Quat.from_euler(-blink_weight * 0.5, 0, 0)
                layer.bone_offsets[eyelid_bone] = Transform(rotation=eyelid_rotation)

        # Update saccades (micro eye movements)
        eye_state._saccade_timer -= dt
        if eye_state._saccade_timer <= 0:
            eye_state._saccade_timer = 0.1 + random.random() * 0.15
            eye_state._saccade_offset = Vec3(
                (random.random() * 2 - 1) * eye_state.saccade_intensity,
                (random.random() * 2 - 1) * eye_state.saccade_intensity,
                0,
            )

        # Apply eye look-at with saccade
        if eye_state.look_weight > 0 and face_rig.eye_bones[0] >= 0:
            # Smooth target tracking
            target_diff = eye_state.look_target - eye_state._smooth_target
            smoothing = min(1.0, dt * 10.0)
            eye_state._smooth_target = eye_state._smooth_target + target_diff * smoothing

            # Add saccade offset
            look_offset = eye_state._saccade_offset

            # Calculate eye rotation from look direction
            for i, eye_bone in enumerate(face_rig.eye_bones):
                if eye_bone >= 0:
                    # Apply eye rotation with saccade
                    eye_rotation = Quat.from_euler(
                        look_offset.y * eye_state.look_weight,
                        look_offset.x * eye_state.look_weight,
                        0
                    )
                    existing = layer.bone_offsets.get(eye_bone, Transform.identity())
                    layer.bone_offsets[eye_bone] = Transform(
                        translation=existing.translation,
                        rotation=existing.rotation * eye_rotation,
                        scale=existing.scale,
                    )

            # Apply look direction to blend shapes
            look_up = max(0.0, look_offset.y * eye_state.look_weight)
            look_down = max(0.0, -look_offset.y * eye_state.look_weight)
            look_in = max(0.0, look_offset.x * eye_state.look_weight)
            look_out = max(0.0, -look_offset.x * eye_state.look_weight)

            layer.blend_shapes["eyeLookUpLeft"] = look_up
            layer.blend_shapes["eyeLookUpRight"] = look_up
            layer.blend_shapes["eyeLookDownLeft"] = look_down
            layer.blend_shapes["eyeLookDownRight"] = look_down
            layer.blend_shapes["eyeLookInLeft"] = look_in
            layer.blend_shapes["eyeLookOutRight"] = look_in
            layer.blend_shapes["eyeLookOutLeft"] = look_out
            layer.blend_shapes["eyeLookInRight"] = look_out

    def _apply_expression_to_layer(
        self,
        layer: Optional[FacialLayer],
        expression: Expression,
        weight: float
    ) -> None:
        """Apply expression to a layer.

        Args:
            layer: Target layer
            expression: Expression to apply
            weight: Application weight
        """
        if layer is None:
            return

        effective_weight = weight * expression.intensity

        # Blend shapes
        for shape_name, shape_weight in expression.blend_shapes.items():
            current = layer.blend_shapes.get(shape_name, 0.0)
            layer.blend_shapes[shape_name] = current + shape_weight * effective_weight

        # Bone offsets
        for bone_idx, transform in expression.bone_offsets.items():
            existing = layer.bone_offsets.get(bone_idx, Transform.identity())
            blended = existing.lerp(transform, effective_weight)
            layer.bone_offsets[bone_idx] = blended

    def _compose_layers(self, component: FacialComponent) -> None:
        """Compose all layers into final output.

        Layers are processed in priority order. Each layer's blend mode
        determines how it combines with accumulated values. Region masks
        limit which blend shapes a layer can affect.

        Args:
            component: Facial component
        """
        # Sort layers by priority
        sorted_layers = sorted(
            component.layers.values(),
            key=lambda l: l.priority.value
        )

        # Get region shapes from face rig
        region_shapes = component.face_rig.region_shapes

        for layer in sorted_layers:
            if not layer.enabled or layer.weight <= 0.001:
                continue

            # Get shapes allowed by region mask
            allowed_shapes = component.face_rig.get_shapes_for_region(layer.region_mask)

            for shape_name, shape_weight in layer.blend_shapes.items():
                # Skip if not in allowed region
                if layer.region_mask != FacialRegion.ALL and shape_name not in allowed_shapes:
                    continue

                weighted_value = shape_weight * layer.weight
                current = component.output_blend_shapes.get(shape_name, 0.0)

                if layer.blend_mode == BlendMode.REPLACE:
                    # Blend between current and new based on layer weight
                    component.output_blend_shapes[shape_name] = (
                        current * (1.0 - layer.weight) + weighted_value
                    )
                elif layer.blend_mode == BlendMode.ADDITIVE:
                    component.output_blend_shapes[shape_name] = current + weighted_value
                elif layer.blend_mode == BlendMode.MULTIPLY:
                    component.output_blend_shapes[shape_name] = current * weighted_value
                elif layer.blend_mode == BlendMode.MAX:
                    component.output_blend_shapes[shape_name] = max(current, weighted_value)

            # Compose bone offsets
            for bone_idx, transform in layer.bone_offsets.items():
                existing = component.output_bone_offsets.get(
                    bone_idx, Transform.identity()
                )
                blended = existing.lerp(transform, layer.weight)
                component.output_bone_offsets[bone_idx] = blended

        # Clamp all blend shapes to [0, 1]
        for name in component.output_blend_shapes:
            component.output_blend_shapes[name] = max(
                0.0, min(1.0, component.output_blend_shapes[name])
            )

    def process_audio_for_lip_sync(
        self,
        component: FacialComponent,
        audio_samples: list[float],
        sample_rate: int = 44100
    ) -> None:
        """Process audio samples to determine phonemes.

        This is a simplified implementation for real-time lip sync.
        For production, use pre-authored phoneme data or ML-based
        phoneme recognition.

        Args:
            component: Facial component to update
            audio_samples: Audio sample buffer
            sample_rate: Audio sample rate
        """
        if not audio_samples:
            component.set_phoneme(LipSyncPhoneme.SILENCE)
            return

        # Calculate RMS (volume)
        rms = math.sqrt(sum(s * s for s in audio_samples) / len(audio_samples))
        component.lip_sync.audio_intensity = min(1.0, rms * 10)

        if rms < FACIAL_CONFIG.SILENCE_VOLUME_THRESHOLD:
            component.set_phoneme(LipSyncPhoneme.SILENCE)
            return

        # Simple frequency analysis to guess phoneme
        # Count zero crossings for frequency estimate
        zero_crossings = sum(
            1 for i in range(1, len(audio_samples))
            if audio_samples[i-1] * audio_samples[i] < 0
        )
        frequency_estimate = zero_crossings * sample_rate / (2 * len(audio_samples))

        # Map frequency to rough phoneme category
        if frequency_estimate < 300:
            phoneme = LipSyncPhoneme.OH
        elif frequency_estimate < 600:
            phoneme = LipSyncPhoneme.AA
        elif frequency_estimate < 1000:
            phoneme = LipSyncPhoneme.EH
        elif frequency_estimate < 2000:
            phoneme = LipSyncPhoneme.IH
        else:
            phoneme = LipSyncPhoneme.EE

        component.set_phoneme(phoneme, min(1.0, rms * 5))

    def set_phoneme_timeline(
        self,
        component: FacialComponent,
        timeline: list[tuple[float, str, float]],
    ) -> None:
        """Set pre-authored phoneme timeline for lip sync.

        Args:
            component: Facial component
            timeline: List of (time_seconds, phoneme_name, weight) tuples
        """
        component.audio_sync.phoneme_timeline.clear()

        phoneme_map = {p.name: p for p in LipSyncPhoneme}

        for time, phoneme_name, weight in timeline:
            phoneme = phoneme_map.get(phoneme_name.upper(), LipSyncPhoneme.SILENCE)
            component.audio_sync.add_phoneme(time, phoneme, weight)

        if timeline:
            component.audio_sync.audio_duration = max(t for t, _, _ in timeline)

    def trigger_blink(
        self,
        component: FacialComponent,
        intensity: float = 1.0
    ) -> None:
        """Manually trigger a blink.

        Args:
            component: Facial component
            intensity: Blink intensity (0-1)
        """
        component.eye_state.is_blinking = True
        component.eye_state.blink_progress = 0.0
        # Reset timer to prevent double blink
        component.eye_state.blink_timer = (
            FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MIN +
            random.random() * (
                FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MAX -
                FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MIN
            )
        )

    def get_stats(self) -> dict[str, Any]:
        """Get system performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "last_update_time_ms": self._last_update_time_ms,
            "entities_processed": self._entities_processed,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_face_rig() -> FaceRig:
    """Create a face rig with ARKit-compatible default configuration.

    Returns:
        Configured FaceRig instance
    """
    rig = FaceRig()
    rig.setup_default_regions()

    # Default viseme shapes
    rig.viseme_shapes = {
        LipSyncPhoneme.SILENCE: "mouthClose",
        LipSyncPhoneme.AA: "jawOpen",
        LipSyncPhoneme.EE: "mouthSmileLeft",
        LipSyncPhoneme.OH: "mouthFunnel",
        LipSyncPhoneme.OO: "mouthPucker",
        LipSyncPhoneme.MBP: "mouthClose",
    }

    return rig


def create_facial_component(
    with_default_rig: bool = True,
    with_default_expressions: bool = True,
) -> FacialComponent:
    """Create a facial component with sensible defaults.

    Args:
        with_default_rig: Include default face rig
        with_default_expressions: Include default expression library

    Returns:
        Configured FacialComponent
    """
    component = FacialComponent()

    if with_default_rig:
        component.face_rig = create_default_face_rig()

    if with_default_expressions:
        # Add some preset expressions
        component.add_expression(Expression(
            name="smile",
            blend_shapes={
                "mouthSmileLeft": 0.8,
                "mouthSmileRight": 0.8,
                "cheekSquintLeft": 0.4,
                "cheekSquintRight": 0.4,
            },
        ))
        component.add_expression(Expression(
            name="frown",
            blend_shapes={
                "mouthFrownLeft": 0.7,
                "mouthFrownRight": 0.7,
                "browDownLeft": 0.3,
                "browDownRight": 0.3,
            },
        ))
        component.add_expression(Expression(
            name="surprise",
            blend_shapes={
                "browInnerUp": 0.8,
                "browOuterUpLeft": 0.7,
                "browOuterUpRight": 0.7,
                "eyeWideLeft": 0.5,
                "eyeWideRight": 0.5,
                "jawOpen": 0.3,
            },
        ))

    return component
