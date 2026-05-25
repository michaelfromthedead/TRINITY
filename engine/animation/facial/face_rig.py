"""
Complete Facial Rig Controller.

Integrates blend shapes, FACS, lip sync, and eye animation
into a unified facial animation system with priority handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from .blend_shapes import BlendShapeController, BlendShapeSet
from .eye_animation import EyeController, EyeTransform
from .facs import ActionUnit, Expression, FACSController
from .lip_sync import LipSyncController, PhonemeEvent, VisemeEvent


# =============================================================================
# Type Aliases
# =============================================================================

Vector3 = Tuple[float, float, float]


# =============================================================================
# Animation Priority
# =============================================================================


class AnimationPriority(Enum):
    """
    Priority levels for facial animation layers.

    Higher priority animations override lower priority ones.
    """
    IDLE = 0
    EMOTION = 1
    LIP_SYNC = 2
    PROCEDURAL = 3
    OVERRIDE = 4


@dataclass
class AnimationLayer:
    """
    A single animation layer with priority and blending.

    Attributes:
        name: Layer name
        priority: Animation priority
        weight: Layer weight (0-1)
        blend_shapes: Blend shape weights for this layer
        is_additive: If True, add to lower layers; if False, override
    """
    name: str
    priority: AnimationPriority
    weight: float = 1.0
    blend_shapes: dict[str, float] = field(default_factory=dict)
    is_additive: bool = False


# =============================================================================
# Emotion State
# =============================================================================


@dataclass
class EmotionState:
    """
    Represents the current emotional state.

    Attributes:
        expression: Current expression
        intensity: Expression intensity (0-1)
        blend_time: Time to blend to this state
        eyes_follow: Whether eyes should participate in expression
        mouth_override: If set, override mouth with this expression
    """
    expression: Expression = Expression.NEUTRAL
    intensity: float = 1.0
    blend_time: float = 0.3
    eyes_follow: bool = True
    mouth_override: Optional[Expression] = None


# =============================================================================
# Face Rig
# =============================================================================


class FaceRig:
    """
    Complete facial rig controller.

    Combines blend shapes, FACS, lip sync, and eye animation
    into a unified system with priority-based layering.
    """

    def __init__(
        self,
        blend_shape_set: BlendShapeSet,
        facs_controller: Optional[FACSController] = None,
        eye_controller: Optional[EyeController] = None,
        lip_sync_controller: Optional[LipSyncController] = None,
        on_weights_changed: Optional[Callable[[dict[str, float]], None]] = None,
    ) -> None:
        """
        Initialize the face rig.

        Args:
            blend_shape_set: The blend shape set to control
            facs_controller: Optional FACS controller
            eye_controller: Optional eye controller
            lip_sync_controller: Optional lip sync controller
            on_weights_changed: Callback when weights change
        """
        # Core controllers
        self._blend_controller = BlendShapeController(blend_shape_set)
        self._facs_controller = facs_controller or FACSController()
        self._eye_controller = eye_controller or EyeController()
        self._lip_sync_controller = lip_sync_controller or LipSyncController()

        self._on_weights_changed = on_weights_changed

        # Animation layers
        self._layers: dict[str, AnimationLayer] = {}
        self._init_default_layers()

        # Current emotion
        self._emotion = EmotionState()
        self._target_emotion: Optional[EmotionState] = None
        self._emotion_blend_progress: float = 1.0

        # Jaw bone (for procedural jaw movement)
        self._jaw_rotation: float = 0.0
        self._jaw_max_rotation: float = 25.0  # degrees

        # Head tracking (for eye/head coordination)
        self._head_position: Vector3 = (0.0, 0.0, 0.0)
        self._head_forward: Vector3 = (0.0, 0.0, 1.0)

        # Final blended weights
        self._final_weights: dict[str, float] = {}
        self._dirty = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def blend_controller(self) -> BlendShapeController:
        """Get the blend shape controller."""
        return self._blend_controller

    @property
    def facs_controller(self) -> FACSController:
        """Get the FACS controller."""
        return self._facs_controller

    @property
    def eye_controller(self) -> EyeController:
        """Get the eye controller."""
        return self._eye_controller

    @property
    def lip_sync_controller(self) -> LipSyncController:
        """Get the lip sync controller."""
        return self._lip_sync_controller

    @property
    def current_emotion(self) -> EmotionState:
        """Get current emotion state."""
        return self._emotion

    @property
    def jaw_rotation(self) -> float:
        """Get current jaw rotation in degrees."""
        return self._jaw_rotation

    @property
    def dirty(self) -> bool:
        """Check if state has changed."""
        return self._dirty

    # =========================================================================
    # Layer Management
    # =========================================================================

    def _init_default_layers(self) -> None:
        """Initialize default animation layers."""
        self._layers = {
            "idle": AnimationLayer(
                name="idle",
                priority=AnimationPriority.IDLE,
                weight=1.0,
                is_additive=False,
            ),
            "emotion": AnimationLayer(
                name="emotion",
                priority=AnimationPriority.EMOTION,
                weight=1.0,
                is_additive=False,
            ),
            "lip_sync": AnimationLayer(
                name="lip_sync",
                priority=AnimationPriority.LIP_SYNC,
                weight=1.0,
                is_additive=True,  # Lip sync adds to emotion
            ),
            "eyes": AnimationLayer(
                name="eyes",
                priority=AnimationPriority.PROCEDURAL,
                weight=1.0,
                is_additive=True,
            ),
            "override": AnimationLayer(
                name="override",
                priority=AnimationPriority.OVERRIDE,
                weight=0.0,
                is_additive=False,
            ),
        }

    def add_layer(
        self,
        name: str,
        priority: AnimationPriority,
        weight: float = 1.0,
        is_additive: bool = False,
    ) -> AnimationLayer:
        """
        Add a custom animation layer.

        Args:
            name: Layer name
            priority: Animation priority
            weight: Layer weight
            is_additive: Whether layer is additive

        Returns:
            The created layer
        """
        layer = AnimationLayer(
            name=name,
            priority=priority,
            weight=weight,
            is_additive=is_additive,
        )
        self._layers[name] = layer
        return layer

    def get_layer(self, name: str) -> Optional[AnimationLayer]:
        """Get a layer by name."""
        return self._layers.get(name)

    def set_layer_weight(self, name: str, weight: float) -> bool:
        """
        Set layer weight.

        Args:
            name: Layer name
            weight: New weight (0-1)

        Returns:
            True if layer exists
        """
        if name in self._layers:
            self._layers[name].weight = max(0.0, min(1.0, weight))
            return True
        return False

    def set_layer_blend_shapes(
        self,
        name: str,
        blend_shapes: dict[str, float],
    ) -> bool:
        """
        Set blend shapes for a layer.

        Args:
            name: Layer name
            blend_shapes: Blend shape weights

        Returns:
            True if layer exists
        """
        if name in self._layers:
            self._layers[name].blend_shapes = blend_shapes
            return True
        return False

    # =========================================================================
    # Emotion Control
    # =========================================================================

    def set_emotion(
        self,
        emotion: EmotionState,
        blend_time: Optional[float] = None,
    ) -> None:
        """
        Set the emotional state.

        Args:
            emotion: Target emotion state
            blend_time: Override blend time (or use emotion's blend_time)
        """
        if blend_time is not None:
            emotion.blend_time = blend_time

        if emotion.blend_time <= 0:
            # Instant transition
            self._emotion = emotion
            self._target_emotion = None
            self._emotion_blend_progress = 1.0
            self._update_emotion_layer()
        else:
            # Set up blend
            self._target_emotion = emotion
            self._emotion_blend_progress = 0.0

    def set_expression(
        self,
        expression: Expression,
        intensity: float = 1.0,
        blend_time: float = 0.3,
    ) -> None:
        """
        Convenience method to set expression.

        Args:
            expression: The expression
            intensity: Expression intensity
            blend_time: Transition time
        """
        self.set_emotion(EmotionState(
            expression=expression,
            intensity=intensity,
            blend_time=blend_time,
        ))

    def _update_emotion_layer(self) -> None:
        """Update the emotion layer blend shapes."""
        self._facs_controller.set_expression(
            self._emotion.expression,
            self._emotion.intensity,
            blend_time=0,  # We handle blending ourselves
        )
        weights = self._facs_controller.get_blend_shape_weights()

        # Apply mouth override if set
        if self._emotion.mouth_override is not None:
            # Get mouth-specific shapes from override expression
            self._facs_controller.set_expression(self._emotion.mouth_override, 1.0, 0)
            override_weights = self._facs_controller.get_blend_shape_weights()

            # Replace mouth shapes
            mouth_shapes = [
                "jawOpen", "mouthClose", "mouthFunnel", "mouthPucker",
                "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight",
                "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
                "mouthStretchLeft", "mouthStretchRight", "mouthRollLower", "mouthRollUpper",
                "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
                "mouthUpperUpLeft", "mouthUpperUpRight", "mouthShrugLower", "mouthShrugUpper",
            ]
            for shape in mouth_shapes:
                if shape in override_weights:
                    weights[shape] = override_weights[shape]

            # Restore main expression
            self._facs_controller.set_expression(self._emotion.expression, self._emotion.intensity, 0)

        self._layers["emotion"].blend_shapes = weights

    # =========================================================================
    # Lip Sync
    # =========================================================================

    def speak(
        self,
        viseme_sequence: Sequence[VisemeEvent],
        audio_length: float,
    ) -> None:
        """
        Start lip sync animation.

        Args:
            viseme_sequence: Sequence of viseme events
            audio_length: Total audio length in seconds
        """
        self._lip_sync_controller.set_timeline(viseme_sequence)
        self._lip_sync_controller.play()

    def speak_phonemes(
        self,
        phoneme_events: Sequence[PhonemeEvent],
    ) -> None:
        """
        Start lip sync from phoneme events.

        Args:
            phoneme_events: Sequence of phoneme events
        """
        self._lip_sync_controller.set_phoneme_timeline(phoneme_events)
        self._lip_sync_controller.play()

    def stop_speaking(self) -> None:
        """Stop lip sync animation."""
        self._lip_sync_controller.stop()

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._lip_sync_controller.is_playing

    # =========================================================================
    # Eye Control
    # =========================================================================

    def look_at(
        self,
        target: Vector3,
        weight: float = 1.0,
        smooth_speed: float = 10.0,
    ) -> None:
        """
        Make eyes look at a target.

        Args:
            target: Target position in world space
            weight: Look-at weight (0-1)
            smooth_speed: Blend speed
        """
        self._eye_controller.look_at(target, weight, smooth_speed)

    def clear_look_at(self) -> None:
        """Clear look-at target."""
        self._eye_controller.clear_target()

    def blink(self, intensity: float = 1.0) -> None:
        """
        Trigger a blink.

        Args:
            intensity: Blink intensity
        """
        self._eye_controller.blink(intensity)

    def set_head_transform(
        self,
        position: Vector3,
        forward: Vector3,
    ) -> None:
        """
        Set head position and orientation.

        Args:
            position: Head position in world space
            forward: Head forward direction
        """
        self._head_position = position
        self._head_forward = forward
        self._eye_controller.set_head_transform(position, forward)

    # =========================================================================
    # FACS Control
    # =========================================================================

    def set_action_unit(
        self,
        au: ActionUnit,
        intensity: float,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """
        Set a FACS Action Unit intensity.

        Args:
            au: The Action Unit
            intensity: Intensity (0-1)
            left: Left-side intensity for bilateral AUs
            right: Right-side intensity for bilateral AUs
        """
        self._facs_controller.set_au_intensity(au, intensity, left, right)

    # =========================================================================
    # Direct Blend Shape Control
    # =========================================================================

    def set_blend_shape(self, name: str, weight: float) -> bool:
        """
        Set a blend shape weight directly (override layer).

        Args:
            name: Blend shape name
            weight: Weight value

        Returns:
            True if shape exists
        """
        if name in self._layers["override"].blend_shapes or self._blend_controller.shape_set.has_shape(name):
            self._layers["override"].blend_shapes[name] = weight
            self._layers["override"].weight = 1.0
            return True
        return False

    def clear_overrides(self) -> None:
        """Clear all override blend shapes."""
        self._layers["override"].blend_shapes.clear()
        self._layers["override"].weight = 0.0

    # =========================================================================
    # Master Update
    # =========================================================================

    def update(self, dt: float) -> dict[str, float]:
        """
        Master update combining all animation systems.

        Args:
            dt: Delta time in seconds

        Returns:
            Final blended weights
        """
        # Update emotion blending
        if self._target_emotion is not None:
            blend_speed = 1.0 / max(0.001, self._target_emotion.blend_time)
            self._emotion_blend_progress = min(1.0, self._emotion_blend_progress + blend_speed * dt)

            if self._emotion_blend_progress >= 1.0:
                self._emotion = self._target_emotion
                self._target_emotion = None
                self._update_emotion_layer()
            else:
                # Interpolate emotion
                self._update_emotion_blend()

        # Update lip sync
        lip_sync_weights = self._lip_sync_controller.update(dt)
        self._layers["lip_sync"].blend_shapes = lip_sync_weights

        # Calculate jaw from lip sync
        if "jawOpen" in lip_sync_weights:
            self._jaw_rotation = lip_sync_weights["jawOpen"] * self._jaw_max_rotation

        # Update eyes
        self._eye_controller.update(dt)
        eye_weights = self._eye_controller.get_blend_shape_weights()
        self._layers["eyes"].blend_shapes = eye_weights

        # Blend all layers
        self._final_weights = self._blend_layers()

        # Notify change
        self._dirty = True
        if self._on_weights_changed:
            self._on_weights_changed(self._final_weights.copy())

        return self._final_weights

    def _update_emotion_blend(self) -> None:
        """Update emotion blend during transition."""
        if self._target_emotion is None:
            return

        t = self._emotion_blend_progress

        # Get weights for both emotions
        self._facs_controller.set_expression(self._emotion.expression, self._emotion.intensity, 0)
        current_weights = self._facs_controller.get_blend_shape_weights()

        self._facs_controller.set_expression(self._target_emotion.expression, self._target_emotion.intensity, 0)
        target_weights = self._facs_controller.get_blend_shape_weights()

        # Interpolate
        blended = {}
        all_shapes = set(current_weights.keys()) | set(target_weights.keys())
        for shape in all_shapes:
            current = current_weights.get(shape, 0.0)
            target = target_weights.get(shape, 0.0)
            blended[shape] = current * (1.0 - t) + target * t

        self._layers["emotion"].blend_shapes = blended

        # Restore FACS to current emotion
        self._facs_controller.set_expression(self._emotion.expression, self._emotion.intensity, 0)

    def _blend_layers(self) -> dict[str, float]:
        """
        Blend all animation layers respecting priority.

        Returns:
            Final blended weights
        """
        # Sort layers by priority
        sorted_layers = sorted(
            self._layers.values(),
            key=lambda l: l.priority.value,
        )

        result: dict[str, float] = {}

        for layer in sorted_layers:
            if layer.weight <= 0.001:
                continue

            for shape_name, shape_weight in layer.blend_shapes.items():
                weighted_value = shape_weight * layer.weight

                if layer.is_additive:
                    # Add to existing value
                    result[shape_name] = result.get(shape_name, 0.0) + weighted_value
                else:
                    # Blend with existing value based on layer weight
                    if shape_name in result:
                        result[shape_name] = result[shape_name] * (1.0 - layer.weight) + weighted_value
                    else:
                        result[shape_name] = weighted_value

        # Clamp all values
        for name in result:
            result[name] = max(0.0, min(1.0, result[name]))

        return result

    def get_final_weights(self) -> dict[str, float]:
        """Get the final blended weights."""
        return self._final_weights.copy()

    def apply_to_mesh(
        self,
        base_vertices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply current weights to mesh vertices.

        Args:
            base_vertices: Optional override for base vertices

        Returns:
            Morphed vertex positions
        """
        self._blend_controller.set_weights(self._final_weights)
        return self._blend_controller.apply_to_mesh(base_vertices)

    # =========================================================================
    # Eye Transforms
    # =========================================================================

    def get_eye_transforms(self) -> Tuple[EyeTransform, EyeTransform]:
        """
        Get eye transforms for bone animation.

        Returns:
            (left_eye, right_eye) transforms
        """
        return (self._eye_controller.left_eye, self._eye_controller.right_eye)

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset all facial animation to default state."""
        self._emotion = EmotionState()
        self._target_emotion = None
        self._emotion_blend_progress = 1.0

        self._facs_controller.reset_all_aus()
        self._lip_sync_controller.stop()
        self._eye_controller.clear_target()

        for layer in self._layers.values():
            layer.blend_shapes.clear()
            if layer.name == "override":
                layer.weight = 0.0

        self._final_weights.clear()
        self._jaw_rotation = 0.0
        self._dirty = True

    def clear_dirty(self) -> None:
        """Clear the dirty flag."""
        self._dirty = False
        self._facs_controller.clear_dirty()
        self._eye_controller.clear_dirty()
        self._lip_sync_controller.clear_dirty()

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "emotion": {
                "expression": self._emotion.expression.name,
                "intensity": self._emotion.intensity,
            },
            "is_speaking": self.is_speaking,
            "jaw_rotation": self._jaw_rotation,
            "eye_state": self._eye_controller.to_dict(),
            "layer_weights": {name: layer.weight for name, layer in self._layers.items()},
            "final_weights": self._final_weights,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from dictionary."""
        if "emotion" in data:
            try:
                expression = Expression[data["emotion"]["expression"]]
                self._emotion = EmotionState(
                    expression=expression,
                    intensity=data["emotion"].get("intensity", 1.0),
                )
                self._update_emotion_layer()
            except KeyError:
                pass

        if "layer_weights" in data:
            for name, weight in data["layer_weights"].items():
                self.set_layer_weight(name, weight)


# =============================================================================
# Factory Functions
# =============================================================================


def create_face_rig(
    vertex_count: int,
    use_arkit_shapes: bool = True,
) -> FaceRig:
    """
    Create a face rig with default configuration.

    Args:
        vertex_count: Number of vertices in the mesh
        use_arkit_shapes: Whether to use ARKit-compatible shapes

    Returns:
        Configured FaceRig instance
    """
    from .blend_shapes import create_arkit_compatible_set

    if use_arkit_shapes:
        shape_set = create_arkit_compatible_set("face", vertex_count)
    else:
        shape_set = BlendShapeSet(
            name="face",
            base_vertices=np.zeros((vertex_count, 3), dtype=np.float32),
        )

    return FaceRig(blend_shape_set=shape_set)
