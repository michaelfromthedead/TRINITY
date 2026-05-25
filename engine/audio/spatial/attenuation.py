"""Distance Attenuation Models for Spatial Audio.

Implements various attenuation models:
- Linear: volume = 1 - (distance - min) / (max - min)
- Logarithmic: realistic falloff
- Inverse: 1/distance
- Inverse Squared: physically accurate for point sources
- Custom curve: designer-defined falloff

Also includes directional cone attenuation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from engine.audio.spatial.config import (
    CONE_INNER_ANGLE,
    CONE_OUTER_ANGLE,
    CONE_OUTER_GAIN,
    DEFAULT_ROLLOFF,
    MAX_ATTENUATION_DISTANCE,
    MIN_ATTENUATION_DISTANCE,
    AttenuationModel,
    AttenuationShape,
)
from engine.core.math.vec import Vec3


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return 10.0 ** (db / 20.0)


def linear_to_db(linear: float) -> float:
    """Convert linear amplitude to decibels."""
    if linear <= 0.0:
        return -96.0  # Effectively -infinity
    return 20.0 * math.log10(linear)


class AttenuationCurve(ABC):
    """Abstract base class for distance attenuation curves."""

    def __init__(
        self,
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE,
        rolloff: float = DEFAULT_ROLLOFF
    ) -> None:
        self._min_distance = max(0.001, min_distance)
        self._max_distance = max(self._min_distance, max_distance)
        self._rolloff = max(0.0, rolloff)

    @property
    def min_distance(self) -> float:
        """Distance at which attenuation begins."""
        return self._min_distance

    @min_distance.setter
    def min_distance(self, value: float) -> None:
        self._min_distance = max(0.001, value)
        if self._max_distance < self._min_distance:
            self._max_distance = self._min_distance

    @property
    def max_distance(self) -> float:
        """Distance at which sound is culled."""
        return self._max_distance

    @max_distance.setter
    def max_distance(self, value: float) -> None:
        self._max_distance = max(self._min_distance, value)

    @property
    def rolloff(self) -> float:
        """Rolloff factor (1.0 = normal, >1 = faster falloff)."""
        return self._rolloff

    @rolloff.setter
    def rolloff(self, value: float) -> None:
        self._rolloff = max(0.0, value)

    @property
    @abstractmethod
    def model(self) -> AttenuationModel:
        """Get the attenuation model type."""
        pass

    @abstractmethod
    def calculate(self, distance: float) -> float:
        """Calculate attenuation factor for given distance.

        Args:
            distance: Distance from source to listener.

        Returns:
            Attenuation factor (0.0 to 1.0, where 1.0 is full volume).
        """
        pass

    def calculate_db(self, distance: float) -> float:
        """Calculate attenuation in decibels.

        Args:
            distance: Distance from source to listener.

        Returns:
            Attenuation in dB (0 = no attenuation, negative = quieter).
        """
        linear = self.calculate(distance)
        return linear_to_db(linear)


class LinearAttenuation(AttenuationCurve):
    """Linear distance attenuation.

    volume = 1 - rolloff * (distance - min) / (max - min)
    """

    @property
    def model(self) -> AttenuationModel:
        return AttenuationModel.LINEAR

    def calculate(self, distance: float) -> float:
        if distance <= self._min_distance:
            return 1.0
        if distance >= self._max_distance:
            return 0.0

        normalized = (distance - self._min_distance) / (self._max_distance - self._min_distance)
        return max(0.0, 1.0 - self._rolloff * normalized)


class LogarithmicAttenuation(AttenuationCurve):
    """Logarithmic distance attenuation.

    Provides a more realistic falloff that sounds natural to human ears.
    Uses: volume = 1 / (1 + rolloff * log2(distance / min))
    """

    @property
    def model(self) -> AttenuationModel:
        return AttenuationModel.LOGARITHMIC

    def calculate(self, distance: float) -> float:
        if distance <= self._min_distance:
            return 1.0
        if distance >= self._max_distance:
            return 0.0

        # Logarithmic falloff
        ratio = distance / self._min_distance
        attenuation = 1.0 / (1.0 + self._rolloff * math.log2(ratio))

        # Smooth to zero at max distance
        fade_start = self._max_distance * 0.8
        if distance > fade_start:
            fade = 1.0 - (distance - fade_start) / (self._max_distance - fade_start)
            attenuation *= fade

        return max(0.0, attenuation)


class InverseAttenuation(AttenuationCurve):
    """Inverse distance attenuation.

    volume = min_distance / (min_distance + rolloff * (distance - min_distance))

    This is a common model used in many game engines.
    """

    @property
    def model(self) -> AttenuationModel:
        return AttenuationModel.INVERSE

    def calculate(self, distance: float) -> float:
        if distance <= self._min_distance:
            return 1.0
        if distance >= self._max_distance:
            return 0.0

        attenuation = self._min_distance / (
            self._min_distance + self._rolloff * (distance - self._min_distance)
        )

        # Smooth to zero at max distance
        fade_start = self._max_distance * 0.9
        if distance > fade_start:
            fade = 1.0 - (distance - fade_start) / (self._max_distance - fade_start)
            attenuation *= fade

        return max(0.0, attenuation)


class InverseSquaredAttenuation(AttenuationCurve):
    """Inverse squared distance attenuation (physically accurate).

    volume = (min_distance / distance)^2

    This follows the inverse square law for point sources in free field.
    """

    @property
    def model(self) -> AttenuationModel:
        return AttenuationModel.INVERSE_SQUARED

    def calculate(self, distance: float) -> float:
        if distance <= self._min_distance:
            return 1.0
        if distance >= self._max_distance:
            return 0.0

        ratio = self._min_distance / distance
        attenuation = ratio * ratio * self._rolloff

        # Smooth to zero at max distance
        fade_start = self._max_distance * 0.85
        if distance > fade_start:
            fade = 1.0 - (distance - fade_start) / (self._max_distance - fade_start)
            attenuation *= fade

        return max(0.0, min(1.0, attenuation))


class NoAttenuation(AttenuationCurve):
    """No distance attenuation - constant volume."""

    @property
    def model(self) -> AttenuationModel:
        return AttenuationModel.NONE

    def calculate(self, distance: float) -> float:
        if distance >= self._max_distance:
            return 0.0
        return 1.0


@dataclass
class CurvePoint:
    """A point on a custom attenuation curve."""

    distance: float
    value: float

    def __post_init__(self) -> None:
        self.distance = max(0.0, self.distance)
        self.value = max(0.0, min(1.0, self.value))


class CustomCurveAttenuation(AttenuationCurve):
    """Custom curve attenuation using designer-defined points.

    Points are interpolated using linear or smooth (Hermite) interpolation.
    """

    def __init__(
        self,
        points: List[CurvePoint],
        min_distance: float = MIN_ATTENUATION_DISTANCE,
        max_distance: float = MAX_ATTENUATION_DISTANCE,
        smooth: bool = True
    ) -> None:
        super().__init__(min_distance, max_distance, 1.0)
        self._points = sorted(points, key=lambda p: p.distance)
        self._smooth = smooth

        # Ensure we have at least two points
        if len(self._points) < 2:
            self._points = [
                CurvePoint(min_distance, 1.0),
                CurvePoint(max_distance, 0.0)
            ]

    @property
    def model(self) -> AttenuationModel:
        return AttenuationModel.CUSTOM

    @property
    def points(self) -> List[CurvePoint]:
        """Get curve points."""
        return self._points

    def add_point(self, distance: float, value: float) -> None:
        """Add a point to the curve."""
        point = CurvePoint(distance, value)
        self._points.append(point)
        self._points.sort(key=lambda p: p.distance)

    def remove_point(self, index: int) -> bool:
        """Remove a point by index. Cannot remove if only 2 points remain."""
        if len(self._points) <= 2:
            return False
        if 0 <= index < len(self._points):
            self._points.pop(index)
            return True
        return False

    def calculate(self, distance: float) -> float:
        if distance <= self._min_distance:
            return 1.0
        if distance >= self._max_distance:
            return 0.0

        # Find surrounding points
        if distance <= self._points[0].distance:
            return self._points[0].value
        if distance >= self._points[-1].distance:
            return self._points[-1].value

        # Find the two points to interpolate between
        for i in range(len(self._points) - 1):
            p1 = self._points[i]
            p2 = self._points[i + 1]

            if p1.distance <= distance <= p2.distance:
                # Guard against division by zero if consecutive points have same distance
                denominator = p2.distance - p1.distance
                if denominator < 0.0001:
                    # Points are effectively at same distance, use average value
                    return (p1.value + p2.value) * 0.5

                t = (distance - p1.distance) / denominator

                if self._smooth:
                    # Smoothstep interpolation
                    t = t * t * (3.0 - 2.0 * t)

                return p1.value + t * (p2.value - p1.value)

        return 0.0


@dataclass
class ConeAttenuation:
    """Directional cone attenuation for focused sound sources.

    Sound is at full volume within inner_angle, attenuates to outer_gain
    at outer_angle, and stays at outer_gain beyond that.
    """

    inner_angle: float = CONE_INNER_ANGLE
    outer_angle: float = CONE_OUTER_ANGLE
    outer_gain: float = CONE_OUTER_GAIN

    def __post_init__(self) -> None:
        self.inner_angle = max(0.0, min(360.0, self.inner_angle))
        self.outer_angle = max(self.inner_angle, min(360.0, self.outer_angle))
        self.outer_gain = max(0.0, min(1.0, self.outer_gain))

    def calculate(self, source_direction: Vec3, to_listener: Vec3) -> float:
        """Calculate cone attenuation.

        Args:
            source_direction: Normalized direction the source is facing.
            to_listener: Normalized direction from source to listener.

        Returns:
            Attenuation factor (outer_gain to 1.0).
        """
        if self.inner_angle >= 360.0:
            return 1.0

        # Calculate angle between source direction and to-listener direction
        # Note: to_listener points FROM source TO listener
        cos_angle = source_direction.dot(to_listener)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

        half_inner = self.inner_angle / 2.0
        half_outer = self.outer_angle / 2.0

        if angle <= half_inner:
            return 1.0
        elif angle >= half_outer:
            return self.outer_gain
        else:
            # Smooth interpolation between inner and outer
            # Guard against division by zero if inner == outer
            denominator = half_outer - half_inner
            if denominator < 0.0001:
                return self.outer_gain
            t = (angle - half_inner) / denominator
            # Smoothstep for nicer transition
            t = t * t * (3.0 - 2.0 * t)
            return 1.0 - t * (1.0 - self.outer_gain)


@dataclass
class AttenuationVolume:
    """Attenuation shape volume for complex spatial falloff."""

    shape: AttenuationShape = AttenuationShape.SPHERE
    center: Vec3 = field(default_factory=Vec3.zero)
    half_extents: Vec3 = field(default_factory=lambda: Vec3(1.0, 1.0, 1.0))
    direction: Vec3 = field(default_factory=Vec3.forward)
    curve: AttenuationCurve = field(default_factory=LinearAttenuation)
    cone: Optional[ConeAttenuation] = None

    def get_distance(self, listener_pos: Vec3) -> float:
        """Get effective distance based on shape."""
        if self.shape == AttenuationShape.SPHERE:
            return self.center.distance(listener_pos)

        elif self.shape == AttenuationShape.BOX:
            # Distance to closest point on box
            min_corner = self.center - self.half_extents
            max_corner = self.center + self.half_extents

            closest = Vec3(
                max(min_corner.x, min(max_corner.x, listener_pos.x)),
                max(min_corner.y, min(max_corner.y, listener_pos.y)),
                max(min_corner.z, min(max_corner.z, listener_pos.z))
            )
            return closest.distance(listener_pos)

        elif self.shape == AttenuationShape.CAPSULE:
            # Capsule is a line with radius
            # Use line segment from bottom to top
            bottom = self.center - Vec3(0, self.half_extents.y, 0)
            top = self.center + Vec3(0, self.half_extents.y, 0)

            line_vec = top - bottom
            line_len_sq = line_vec.length_squared()

            if line_len_sq < 0.0001:
                return self.center.distance(listener_pos) - self.half_extents.x

            t = max(0.0, min(1.0, (listener_pos - bottom).dot(line_vec) / line_len_sq))
            closest_on_line = bottom + line_vec * t

            return max(0.0, closest_on_line.distance(listener_pos) - self.half_extents.x)

        else:  # CONE and others default to sphere
            return self.center.distance(listener_pos)

    def calculate(self, listener_pos: Vec3) -> float:
        """Calculate total attenuation for listener position."""
        distance = self.get_distance(listener_pos)
        attenuation = self.curve.calculate(distance)

        # Apply cone attenuation if present
        if self.cone is not None:
            to_listener = (listener_pos - self.center)
            length = to_listener.length()
            if length > 0.0001:
                to_listener = to_listener / length
                attenuation *= self.cone.calculate(self.direction, to_listener)

        return attenuation


def create_attenuation(
    model: AttenuationModel,
    min_distance: float = MIN_ATTENUATION_DISTANCE,
    max_distance: float = MAX_ATTENUATION_DISTANCE,
    rolloff: float = DEFAULT_ROLLOFF,
    **kwargs
) -> AttenuationCurve:
    """Factory function to create attenuation curves.

    Args:
        model: Attenuation model type.
        min_distance: Distance where attenuation starts.
        max_distance: Distance where sound is culled.
        rolloff: Rolloff factor.
        **kwargs: Additional arguments for custom curves:
            - points: List[CurvePoint] for CUSTOM model
            - smooth: bool for CUSTOM model

    Returns:
        The created attenuation curve.
    """
    if model == AttenuationModel.LINEAR:
        return LinearAttenuation(min_distance, max_distance, rolloff)

    elif model == AttenuationModel.LOGARITHMIC:
        return LogarithmicAttenuation(min_distance, max_distance, rolloff)

    elif model == AttenuationModel.INVERSE:
        return InverseAttenuation(min_distance, max_distance, rolloff)

    elif model == AttenuationModel.INVERSE_SQUARED:
        return InverseSquaredAttenuation(min_distance, max_distance, rolloff)

    elif model == AttenuationModel.CUSTOM:
        points = kwargs.get("points", [
            CurvePoint(min_distance, 1.0),
            CurvePoint(max_distance, 0.0)
        ])
        smooth = kwargs.get("smooth", True)
        return CustomCurveAttenuation(points, min_distance, max_distance, smooth)

    elif model == AttenuationModel.NONE:
        return NoAttenuation(min_distance, max_distance, rolloff)

    else:
        raise ValueError(f"Unknown attenuation model: {model}")


# Pre-defined attenuation presets
ATTENUATION_PRESETS = {
    "realistic": lambda: InverseSquaredAttenuation(1.0, 100.0, 1.0),
    "linear": lambda: LinearAttenuation(1.0, 50.0, 1.0),
    "ambient": lambda: LogarithmicAttenuation(5.0, 200.0, 0.5),
    "dialog": lambda: InverseAttenuation(2.0, 30.0, 1.5),
    "explosion": lambda: InverseSquaredAttenuation(10.0, 500.0, 0.8),
    "whisper": lambda: InverseAttenuation(0.5, 5.0, 2.0),
    "global": lambda: NoAttenuation(0.0, float("inf"), 0.0),
}


def get_preset(name: str) -> Optional[AttenuationCurve]:
    """Get a preset attenuation curve by name."""
    factory = ATTENUATION_PRESETS.get(name.lower())
    if factory:
        return factory()
    return None
