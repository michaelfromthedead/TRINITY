"""ECS system for facial animation.

Handles facial expressions, lip sync, and eye tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World
from engine.animation.config import FACIAL_CONFIG


class FacialRegion(Enum):
    """Regions of the face for targeted animation."""
    FULL_FACE = auto()
    UPPER_FACE = auto()  # Brows, eyelids
    LOWER_FACE = auto()  # Mouth, jaw
    LEFT_EYE = auto()
    RIGHT_EYE = auto()
    EYES = auto()  # Both eyes
    BROWS = auto()
    NOSE = auto()
    MOUTH = auto()
    JAW = auto()
    CHEEKS = auto()
    ALL = auto()  # All regions


class FacialLayerPriority(Enum):
    """Priority levels for facial animation layers."""
    BASE = 0  # Base layer (idle expressions)
    EMOTION = 10  # Emotion overlay
    SPEECH = 20  # Lip sync / speech
    PROCEDURAL = 30  # Procedural animation (blinks, etc.)
    OVERRIDE = 100  # Manual override (highest priority)


class BlendMode(Enum):
    """How layers blend with each other."""
    REPLACE = auto()  # Replace lower layers
    ADDITIVE = auto()  # Add to lower layers
    MULTIPLY = auto()  # Multiply with lower layers
    OVERLAY = auto()  # Overlay blend mode


class EmotionState(Enum):
    """Basic emotion states."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    ANGRY = auto()
    SURPRISED = auto()
    DISGUSTED = auto()
    FEARFUL = auto()
    CUSTOM = auto()


class LipSyncPhoneme(Enum):
    """Phoneme categories for lip sync."""
    SILENCE = auto()  # Closed mouth
    AA = auto()  # "ah" sound
    EE = auto()  # "ee" sound
    IH = auto()  # "ih" sound
    OH = auto()  # "oh" sound
    OO = auto()  # "oo" sound
    EH = auto()  # "eh" sound
    AE = auto()  # "a" as in "cat"
    UH = auto()  # "uh" sound
    CH = auto()  # "ch"/"sh" sounds
    FV = auto()  # "f"/"v" sounds
    TH = auto()  # "th" sounds
    MBP = auto()  # "m"/"b"/"p" sounds (closed lips)
    LN = auto()  # "l"/"n" sounds
    WQ = auto()  # "w"/"q" sounds


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
        """Blend this expression with another."""
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

    def scale(self, factor: float) -> 'Expression':
        """Scale the expression intensity and blend shape weights."""
        scaled = Expression(
            name=self.name,
            blend_shapes={k: v * factor for k, v in self.blend_shapes.items()},
            bone_offsets=dict(self.bone_offsets),
            intensity=self.intensity * factor,
        )
        return scaled


@dataclass
class FacialLayer:
    """A layer of facial animation that can be blended.

    Attributes:
        name: Layer identifier
        priority: Layer priority for blending order
        blend_mode: How this layer blends with others
        weight: Layer weight (0-1)
        region_mask: Which facial region this layer affects
        enabled: Whether the layer is active
        blend_shapes: Blend shape weights for this layer
        expression: The expression applied by this layer
    """
    name: str = ""
    priority: FacialLayerPriority = FacialLayerPriority.BASE
    blend_mode: BlendMode = BlendMode.ADDITIVE
    weight: float = 1.0
    region_mask: FacialRegion = FacialRegion.FULL_FACE
    enabled: bool = True
    blend_shapes: dict[str, float] = field(default_factory=dict)
    expression: "Expression | None" = None

    def clear(self) -> None:
        """Clear all blend shapes from this layer."""
        self.blend_shapes.clear()


@dataclass
class FACSState:
    """Facial Action Coding System (FACS) state.

    Represents activation levels of individual Action Units (AUs).

    Attributes:
        action_units: AU activation levels (AU name -> intensity 0-5)
        asymmetry: Left/right asymmetry values
        au_left_intensities: Left-side AU intensities for bilateral AUs
        au_right_intensities: Right-side AU intensities for bilateral AUs
    """
    action_units: dict[str, float] = field(default_factory=dict)
    asymmetry: dict[str, float] = field(default_factory=dict)
    au_left_intensities: dict[str, float] = field(default_factory=dict)
    au_right_intensities: dict[str, float] = field(default_factory=dict)

    def set_au(self, au_name: str, intensity: float, left: float | None = None, right: float | None = None) -> None:
        """Set action unit intensity (0-5 scale) with optional bilateral asymmetry."""
        self.action_units[au_name] = max(0.0, min(5.0, intensity))
        if left is not None:
            self.au_left_intensities[au_name] = left
            self.asymmetry[f"{au_name}_L"] = left
        if right is not None:
            self.au_right_intensities[au_name] = right
            self.asymmetry[f"{au_name}_R"] = right

    def get_au(self, au_name: str) -> float:
        """Get action unit intensity."""
        return self.action_units.get(au_name, 0.0)

    def clear(self) -> None:
        """Clear all action units and asymmetry values."""
        self.action_units.clear()
        self.asymmetry.clear()
        self.au_left_intensities.clear()
        self.au_right_intensities.clear()


@dataclass
class AudioSyncData:
    """Audio synchronization data for lip sync.

    Attributes:
        phonemes: Timed phoneme sequence (time, phoneme, weight)
        intensity_curve: Audio intensity over time
        current_time: Current playback time
        is_playing: Whether audio is currently playing
    """
    phonemes: list[tuple[float, LipSyncPhoneme, float]] = field(default_factory=list)
    intensity_curve: list[tuple[float, float]] = field(default_factory=list)
    current_time: float = 0.0
    is_playing: bool = False

    @property
    def phoneme_timeline(self) -> list[tuple[float, LipSyncPhoneme, float]]:
        """Alias for phonemes list."""
        return self.phonemes

    def add_phoneme(self, time: float, phoneme: LipSyncPhoneme, weight: float = 1.0) -> None:
        """Add a phoneme at a given time with weight."""
        self.phonemes.append((time, phoneme, weight))
        self.phonemes.sort(key=lambda x: x[0])

    def get_phoneme_at_time(self, time: float) -> tuple[LipSyncPhoneme, float]:
        """Get phoneme and weight at given time."""
        if not self.phonemes:
            return (LipSyncPhoneme.SILENCE, 0.0)
        for t, phoneme, weight in reversed(self.phonemes):
            if time >= t:
                return (phoneme, weight)
        return (LipSyncPhoneme.SILENCE, 0.0)

    def get_phoneme_at(self, time: float) -> LipSyncPhoneme:
        """Get phoneme at given time (without weight)."""
        phoneme, _ = self.get_phoneme_at_time(time)
        return phoneme

    def get_intensity_at(self, time: float) -> float:
        """Get audio intensity at given time."""
        if not self.intensity_curve:
            return 0.0
        for i, (t, intensity) in enumerate(self.intensity_curve):
            if time < t:
                if i == 0:
                    return intensity
                prev_t, prev_intensity = self.intensity_curve[i - 1]
                alpha = (time - prev_t) / (t - prev_t)
                return prev_intensity + alpha * (intensity - prev_intensity)
        return self.intensity_curve[-1][1] if self.intensity_curve else 0.0


@dataclass
class FaceRig:
    """Face rig configuration.

    Attributes:
        blend_shape_names: Available blend shape names
        jaw_bone: Index of jaw bone
        eye_bones: Indices of eye bones (left, right)
        eyelid_bones: Indices of eyelid bones
        eyebrow_bones: Indices of eyebrow bones
    """
    blend_shape_names: list[str] = field(default_factory=list)
    jaw_bone: int = -1
    eye_bones: tuple[int, int] = (-1, -1)
    eyelid_bones: list[int] = field(default_factory=list)
    eyebrow_bones: list[int] = field(default_factory=list)

    # Blend shape categories
    viseme_shapes: dict[LipSyncPhoneme, str] = field(default_factory=dict)
    emotion_shapes: dict[EmotionState, list[str]] = field(default_factory=dict)
    region_shapes: dict[FacialRegion, list[str]] = field(default_factory=lambda: {
        FacialRegion.UPPER_FACE: [],
        FacialRegion.LOWER_FACE: [],
        FacialRegion.EYES: [],
        FacialRegion.BROWS: [],
        FacialRegion.NOSE: [],
        FacialRegion.MOUTH: [],
        FacialRegion.CHEEKS: [],
        FacialRegion.FULL_FACE: [],
        FacialRegion.ALL: [],
    })

    def get_shapes_for_region(self, region: FacialRegion) -> list[str]:
        """Get all blend shape names for a region."""
        if region == FacialRegion.ALL:
            return list(self.blend_shape_names)
        return self.region_shapes.get(region, [])


@dataclass
class LipSyncState:
    """Current lip sync state.

    Attributes:
        current_phoneme: Active phoneme
        phoneme_weight: Current phoneme weight
        transition_time: Time for phoneme transitions
        audio_intensity: Current audio intensity
        is_speaking: Whether audio is playing
        audio_time: Current audio playback time
    """
    current_phoneme: LipSyncPhoneme = LipSyncPhoneme.SILENCE
    phoneme_weight: float = 0.0
    transition_time: float = FACIAL_CONFIG.DEFAULT_PHONEME_TRANSITION
    audio_intensity: float = 0.0
    is_speaking: bool = False
    audio_time: float = 0.0

    _previous_phoneme: LipSyncPhoneme = LipSyncPhoneme.SILENCE
    _transition_progress: float = 1.0

    def start_speaking(self) -> None:
        """Start speaking (audio playback begins)."""
        self.is_speaking = True
        self.audio_time = 0.0

    def stop_speaking(self) -> None:
        """Stop speaking and reset to silence."""
        self.is_speaking = False
        self.current_phoneme = LipSyncPhoneme.SILENCE
        self.phoneme_weight = 0.0


@dataclass
class EyeState:
    """Eye tracking state.

    Attributes:
        look_target: World position to look at
        look_weight: Look-at blend weight
        blink_timer: Time until next blink
        blink_duration: Duration of blink
        is_blinking: Whether currently blinking
        blink_progress: Current blink progress
    """
    look_target: Vec3 = field(default_factory=Vec3.zero)
    look_weight: float = 1.0
    blink_timer: float = (FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MIN + FACIAL_CONFIG.DEFAULT_BLINK_INTERVAL_MAX) / 2
    blink_duration: float = FACIAL_CONFIG.DEFAULT_BLINK_DURATION
    is_blinking: bool = False
    blink_progress: float = 0.0

    # Saccade (micro eye movements)
    saccade_intensity: float = FACIAL_CONFIG.DEFAULT_SACCADE_INTENSITY
    _saccade_offset: Vec3 = field(default_factory=Vec3.zero)
    _saccade_timer: float = 0.0


@dataclass
class FacialComponent:
    """Component for entities with facial animation.

    Attributes:
        face_rig: Face rig configuration
        current_emotion: Current emotion state
        emotion_intensity: Emotion intensity (0-1)
        lip_sync: Lip sync state
        eye_state: Eye tracking state
        enabled: Whether facial animation is enabled
    """
    face_rig: FaceRig = field(default_factory=FaceRig)
    current_emotion: EmotionState = EmotionState.NEUTRAL
    emotion_intensity: float = 0.0
    lip_sync: LipSyncState = field(default_factory=LipSyncState)
    eye_state: EyeState = field(default_factory=EyeState)
    enabled: bool = True

    # Custom expression override
    custom_expression: Expression | None = None
    custom_expression_weight: float = 0.0

    # Output
    output_blend_shapes: dict[str, float] = field(default_factory=dict)
    output_bone_offsets: dict[int, Transform] = field(default_factory=dict)

    # State tracking
    _dirty: bool = True

    # Expression library
    expressions: dict[str, Expression] = field(default_factory=dict)

    # FACS state
    facs_state: FACSState = field(default_factory=FACSState)

    # Audio sync data
    audio_sync: AudioSyncData = field(default_factory=AudioSyncData)

    # Animation layers
    layers: dict[str, FacialLayer] = field(default_factory=lambda: {
        "base": FacialLayer(name="base", priority=FacialLayerPriority.BASE),
        "emotion": FacialLayer(name="emotion", priority=FacialLayerPriority.EMOTION),
        "lip_sync": FacialLayer(name="lip_sync", priority=FacialLayerPriority.SPEECH, region_mask=FacialRegion.MOUTH),
        "eye": FacialLayer(name="eye", priority=FacialLayerPriority.PROCEDURAL, region_mask=FacialRegion.EYES),
        "override": FacialLayer(name="override", priority=FacialLayerPriority.PROCEDURAL, weight=1.0),
    })

    def get_layer(self, name: str) -> FacialLayer | None:
        """Get a layer by name."""
        return self.layers.get(name)

    def set_layer_weight(self, name: str, weight: float) -> bool:
        """Set a layer's weight. Returns True if layer exists."""
        if name in self.layers:
            self.layers[name].weight = max(0.0, min(1.0, weight))
            return True
        return False

    def set_emotion(self, emotion: EmotionState, intensity: float = 1.0) -> None:
        """Set current emotion."""
        self.current_emotion = emotion
        self.emotion_intensity = max(0.0, min(1.0, intensity))

    def set_phoneme(self, phoneme: LipSyncPhoneme, weight: float = 1.0) -> None:
        """Set current phoneme for lip sync."""
        if phoneme != self.lip_sync.current_phoneme:
            self.lip_sync._previous_phoneme = self.lip_sync.current_phoneme
            self.lip_sync.current_phoneme = phoneme
            self.lip_sync._transition_progress = 0.0
        self.lip_sync.phoneme_weight = weight

    def set_look_target(self, target: Vec3, weight: float = 1.0) -> None:
        """Set eye look target."""
        self.eye_state.look_target = target
        self.eye_state.look_weight = weight

    def add_expression(self, expression: Expression) -> None:
        """Add expression to library."""
        self.expressions[expression.name] = expression

    def play_expression(self, name: str, weight: float = 1.0) -> bool:
        """Play named expression."""
        if name in self.expressions:
            self.custom_expression = self.expressions[name]
            self.custom_expression_weight = weight
            return True
        return False


class FacialSystem:
    """ECS system for facial animation.

    Updates facial rigs including expressions, lip sync, and eye tracking.
    """
    _system_phase = "animation"

    def __init__(self):
        self._emotion_expressions: dict[EmotionState, Expression] = {}
        self._entities_processed = 0
        self._setup_default_emotions()

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
                "mouthSmile_L": 0.8,
                "mouthSmile_R": 0.8,
                "cheekPuff_L": 0.5,
                "cheekPuff_R": 0.5,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.SAD] = Expression(
            name="sad",
            blend_shapes={
                "frown_left": 0.6,
                "frown_right": 0.6,
                "brow_inner_up": 0.5,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.ANGRY] = Expression(
            name="angry",
            blend_shapes={
                "brow_down_left": 0.7,
                "brow_down_right": 0.7,
                "nose_wrinkle": 0.3,
                "jaw_forward": 0.2,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.SURPRISED] = Expression(
            name="surprised",
            blend_shapes={
                "brow_up_left": 0.8,
                "brow_up_right": 0.8,
                "eye_wide_left": 0.6,
                "eye_wide_right": 0.6,
                "jaw_open": 0.3,
            },
            intensity=1.0,
        )

        self._emotion_expressions[EmotionState.FEARFUL] = Expression(
            name="fearful",
            blend_shapes={
                "brow_up_left": 0.6,
                "brow_up_right": 0.6,
                "brow_inner_up": 0.7,
                "eye_wide_left": 0.5,
                "eye_wide_right": 0.5,
            },
            intensity=1.0,
        )

    def update(
        self,
        world: World,
        dt: float,
        entity_components: list[tuple[Entity, FacialComponent]]
    ) -> None:
        """Update all facial components.

        Args:
            world: ECS world
            dt: Delta time
            entity_components: List of (entity, component) tuples
        """
        count = 0
        for entity, component in entity_components:
            if not component.enabled:
                continue

            self._update_component(component, dt)
            count += 1
        self._entities_processed = count

    def _update_component(self, component: FacialComponent, dt: float) -> None:
        """Update single facial component."""
        # Clear outputs
        component.output_blend_shapes.clear()
        component.output_bone_offsets.clear()

        # Update emotion
        self._update_emotion(component)

        # Update lip sync
        self._update_lip_sync(component, dt)

        # Update eye tracking
        self._update_eyes(component, dt)

        # Apply custom expression
        if component.custom_expression and component.custom_expression_weight > 0:
            self._apply_expression(
                component,
                component.custom_expression,
                component.custom_expression_weight
            )

        # Mark as processed
        component._dirty = False

    def _update_emotion(self, component: FacialComponent) -> None:
        """Update emotion-based facial expression."""
        if component.emotion_intensity <= 0:
            return

        emotion_expr = self._emotion_expressions.get(component.current_emotion)
        if emotion_expr:
            self._apply_expression(component, emotion_expr, component.emotion_intensity)

    def _update_lip_sync(self, component: FacialComponent, dt: float) -> None:
        """Update lip sync animation."""
        lip_sync = component.lip_sync

        # Update audio time when speaking
        if lip_sync.is_speaking:
            lip_sync.audio_time += dt

        # Update transition
        if lip_sync._transition_progress < 1.0:
            lip_sync._transition_progress += dt / lip_sync.transition_time
            lip_sync._transition_progress = min(1.0, lip_sync._transition_progress)

        # Get viseme shapes from face rig
        face_rig = component.face_rig

        # Previous phoneme (blend out)
        if lip_sync._transition_progress < 1.0:
            prev_shape = face_rig.viseme_shapes.get(lip_sync._previous_phoneme)
            if prev_shape:
                prev_weight = (1.0 - lip_sync._transition_progress) * lip_sync.phoneme_weight
                current = component.output_blend_shapes.get(prev_shape, 0.0)
                component.output_blend_shapes[prev_shape] = max(current, prev_weight)

        # Current phoneme (blend in)
        current_shape = face_rig.viseme_shapes.get(lip_sync.current_phoneme)
        if current_shape:
            current_weight = lip_sync._transition_progress * lip_sync.phoneme_weight
            current = component.output_blend_shapes.get(current_shape, 0.0)
            component.output_blend_shapes[current_shape] = max(current, current_weight)

        # Apply jaw bone for open mouth phonemes
        if face_rig.jaw_bone >= 0:
            jaw_open = 0.0
            if lip_sync.current_phoneme in (LipSyncPhoneme.AA, LipSyncPhoneme.OH, LipSyncPhoneme.EH):
                jaw_open = 0.5 * lip_sync.phoneme_weight
            elif lip_sync.current_phoneme in (LipSyncPhoneme.EE, LipSyncPhoneme.IH):
                jaw_open = 0.2 * lip_sync.phoneme_weight

            if jaw_open > 0:
                jaw_rotation = Quat.from_euler(-jaw_open * 0.3, 0, 0)  # Open jaw
                component.output_bone_offsets[face_rig.jaw_bone] = Transform(
                    rotation=jaw_rotation
                )

    def _update_eyes(self, component: FacialComponent, dt: float) -> None:
        """Update eye tracking and blinking."""
        eye_state = component.eye_state
        face_rig = component.face_rig

        # Update blink
        eye_state.blink_timer -= dt
        if eye_state.blink_timer <= 0 and not eye_state.is_blinking:
            eye_state.is_blinking = True
            eye_state.blink_progress = 0.0
            eye_state.blink_timer = 2.0 + math.sin(eye_state.blink_timer * 1.7) * 2.0

        if eye_state.is_blinking:
            eye_state.blink_progress += dt / eye_state.blink_duration
            if eye_state.blink_progress >= 1.0:
                eye_state.is_blinking = False
                eye_state.blink_progress = 0.0

            # Blink curve (close then open)
            blink_curve = 1.0 - abs(eye_state.blink_progress * 2.0 - 1.0)
            blink_weight = blink_curve * blink_curve

            # Apply blink to blend shapes
            component.output_blend_shapes["eyeBlinkLeft"] = blink_weight
            component.output_blend_shapes["eyeBlinkRight"] = blink_weight

            # Apply to eyelids
            for eyelid_bone in face_rig.eyelid_bones:
                eyelid_rotation = Quat.from_euler(-blink_weight * 0.5, 0, 0)
                component.output_bone_offsets[eyelid_bone] = Transform(rotation=eyelid_rotation)

        # Update saccades
        eye_state._saccade_timer -= dt
        if eye_state._saccade_timer <= 0:
            eye_state._saccade_timer = 0.1 + math.sin(dt * 123.456) * 0.1
            eye_state._saccade_offset = Vec3(
                (math.sin(dt * 789.012) * 2 - 1) * eye_state.saccade_intensity,
                (math.cos(dt * 345.678) * 2 - 1) * eye_state.saccade_intensity,
                0,
            )

        # Apply eye look-at
        if eye_state.look_weight > 0 and face_rig.eye_bones[0] >= 0:
            # Simplified look-at: just apply rotation offset
            # Full implementation would compute look direction from target
            look_offset = eye_state._saccade_offset

            for eye_bone in face_rig.eye_bones:
                if eye_bone >= 0:
                    eye_rotation = Quat.from_euler(look_offset.y, look_offset.x, 0)
                    existing = component.output_bone_offsets.get(eye_bone, Transform.identity())
                    component.output_bone_offsets[eye_bone] = Transform(
                        translation=existing.translation,
                        rotation=existing.rotation * eye_rotation,
                        scale=existing.scale,
                    )

    def _apply_expression(
        self,
        component: FacialComponent,
        expression: Expression,
        weight: float
    ) -> None:
        """Apply expression to component outputs."""
        effective_weight = weight * expression.intensity

        # Blend shapes
        for shape_name, shape_weight in expression.blend_shapes.items():
            current = component.output_blend_shapes.get(shape_name, 0.0)
            component.output_blend_shapes[shape_name] = current + shape_weight * effective_weight

        # Bone offsets
        for bone_idx, transform in expression.bone_offsets.items():
            existing = component.output_bone_offsets.get(bone_idx, Transform.identity())
            blended = existing.lerp(transform, effective_weight)
            component.output_bone_offsets[bone_idx] = blended

    def trigger_blink(self, component: FacialComponent) -> None:
        """Trigger a blink on the component."""
        component.eye_state.is_blinking = True
        component.eye_state.blink_progress = 0.0

    def set_phoneme_timeline(
        self,
        component: FacialComponent,
        timeline: list[tuple[float, LipSyncPhoneme, float]]
    ) -> None:
        """Set the phoneme timeline for lip sync."""
        component.audio_sync.phonemes = list(timeline)
        component.audio_sync.phonemes.sort(key=lambda x: x[0])

    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "emotion_expressions": len(self._emotion_expressions),
            "active": True,
            "last_update_time_ms": 0.0,
            "entities_processed": self._entities_processed,
        }

    def process_audio_for_lip_sync(
        self,
        component: FacialComponent,
        audio_samples: list[float],
        sample_rate: int = 44100
    ) -> None:
        """Process audio samples to determine phonemes.

        This is a simplified implementation. Real lip sync would use
        phoneme recognition or pre-authored viseme data.
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
        # This is very basic - real implementation would use FFT and ML
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


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_face_rig() -> FaceRig:
    """Create a default face rig with standard blend shapes.

    Returns:
        A FaceRig configured with common blend shape names and bone indices.
    """
    eye_shapes = ["eyeBlinkLeft", "eyeBlinkRight", "eyeClose_L", "eyeClose_R", "eyeWide_L", "eyeWide_R"]
    brow_shapes = ["browUp_L", "browUp_R", "browDown_L", "browDown_R"]
    mouth_shapes = ["jawOpen", "mouthSmile_L", "mouthSmile_R", "mouthFrown_L", "mouthFrown_R"]
    nose_shapes = ["noseSneer_L", "noseSneer_R"]
    cheek_shapes = ["cheekPuff_L", "cheekPuff_R"]

    all_shapes = eye_shapes + brow_shapes + mouth_shapes + nose_shapes + cheek_shapes

    rig = FaceRig(
        blend_shape_names=all_shapes,
        jaw_bone=0,
        eye_bones=(1, 2),
        eyelid_bones=[3, 4, 5, 6],
        eyebrow_bones=[7, 8],
        region_shapes={
            FacialRegion.EYES: eye_shapes,
            FacialRegion.BROWS: brow_shapes,
            FacialRegion.MOUTH: mouth_shapes,
            FacialRegion.NOSE: nose_shapes,
            FacialRegion.CHEEKS: cheek_shapes,
            FacialRegion.UPPER_FACE: eye_shapes + brow_shapes,
            FacialRegion.LOWER_FACE: mouth_shapes + cheek_shapes,
            FacialRegion.FULL_FACE: all_shapes,
            FacialRegion.ALL: all_shapes,
        }
    )

    # Map visemes to blend shapes
    rig.viseme_shapes = {
        LipSyncPhoneme.SILENCE: "jawOpen",
        LipSyncPhoneme.AA: "jawOpen",
        LipSyncPhoneme.EE: "mouthSmile_L",
        LipSyncPhoneme.OH: "jawOpen",
        LipSyncPhoneme.OO: "jawOpen",
        LipSyncPhoneme.MBP: "jawOpen",
    }

    # Map emotions to blend shape combinations
    rig.emotion_shapes = {
        EmotionState.HAPPY: ["mouthSmile_L", "mouthSmile_R", "cheekPuff_L", "cheekPuff_R"],
        EmotionState.SAD: ["mouthFrown_L", "mouthFrown_R", "browDown_L", "browDown_R"],
        EmotionState.ANGRY: ["browDown_L", "browDown_R", "noseSneer_L", "noseSneer_R"],
        EmotionState.SURPRISED: ["browUp_L", "browUp_R", "eyeWide_L", "eyeWide_R", "jawOpen"],
    }

    return rig


def create_facial_component(
    face_rig: FaceRig | None = None,
    emotion: EmotionState = EmotionState.NEUTRAL,
    with_default_rig: bool = True,
    with_default_expressions: bool = True
) -> FacialComponent:
    """Create a FacialComponent with sensible defaults.

    Args:
        face_rig: Optional custom face rig (uses default if None)
        emotion: Initial emotion state
        with_default_rig: If True and face_rig is None, use default rig
        with_default_expressions: If True, populate default expressions

    Returns:
        Configured FacialComponent
    """
    if face_rig is None and with_default_rig:
        face_rig = create_default_face_rig()
    elif face_rig is None:
        face_rig = FaceRig()

    component = FacialComponent(
        face_rig=face_rig,
        current_emotion=emotion,
        emotion_intensity=0.0 if emotion == EmotionState.NEUTRAL else 0.5
    )

    if with_default_expressions:
        component.expressions = {
            "smile": Expression(name="smile", blend_shapes={"mouthSmile_L": 1.0, "mouthSmile_R": 1.0}),
            "frown": Expression(name="frown", blend_shapes={"mouthFrown_L": 1.0, "mouthFrown_R": 1.0}),
            "surprise": Expression(name="surprise", blend_shapes={"browUp_L": 1.0, "browUp_R": 1.0, "eyeWide_L": 1.0, "eyeWide_R": 1.0}),
        }

    return component


# =============================================================================
# SYSTEM DECORATOR
# =============================================================================


def system(func: Callable) -> Callable:
    """Decorator to mark a function as an ECS system.

    Systems are functions that process components each frame.
    """
    func._is_system = True
    return func
