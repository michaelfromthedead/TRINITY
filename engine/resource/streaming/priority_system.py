"""Streaming priority calculation system."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from engine.resource.constants import (
    CRITICAL_PRIORITY_THRESHOLD,
    DEFAULT_DISTANCE_WEIGHT,
    DEFAULT_FREQUENCY_WEIGHT,
    DEFAULT_SCREEN_SIZE_WEIGHT,
    HIGH_PRIORITY_THRESHOLD,
    LOW_PRIORITY_THRESHOLD,
    NORMAL_PRIORITY_THRESHOLD,
)

__all__ = ["PriorityBucket", "StreamPriorityCalculator"]


class PriorityBucket(enum.IntEnum):
    """Priority buckets from most to least urgent."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# Bucket thresholds (priority value -> bucket).
_BUCKET_THRESHOLDS: tuple[tuple[float, PriorityBucket], ...] = (
    (CRITICAL_PRIORITY_THRESHOLD, PriorityBucket.CRITICAL),
    (HIGH_PRIORITY_THRESHOLD, PriorityBucket.HIGH),
    (NORMAL_PRIORITY_THRESHOLD, PriorityBucket.NORMAL),
    (LOW_PRIORITY_THRESHOLD, PriorityBucket.LOW),
)


@dataclass(slots=True)
class PriorityWeights:
    """Configurable weights for priority calculation."""

    distance: float = DEFAULT_DISTANCE_WEIGHT
    screen_size: float = DEFAULT_SCREEN_SIZE_WEIGHT
    frequency: float = DEFAULT_FREQUENCY_WEIGHT


class StreamPriorityCalculator:
    """Calculates streaming priority from distance, screen coverage, and usage."""

    __slots__ = ("_weights",)

    def __init__(self, weights: PriorityWeights | None = None) -> None:
        self._weights = weights or PriorityWeights()

    @property
    def weights(self) -> PriorityWeights:
        return self._weights

    def calculate_priority(
        self,
        distance: float,
        screen_size: float,
        frequency: float,
        base_priority: float = 0.0,
    ) -> float:
        """Calculate a combined priority score in [0, 1].

        Args:
            distance: Distance to camera (>=0). Closer = higher priority.
            screen_size: Screen-space coverage in [0, 1]. Larger = higher.
            frequency: Usage frequency in [0, 1]. Higher = higher.
            base_priority: Additive base offset in [0, 1].

        Returns:
            Clamped priority value in [0, 1].
        """
        # Distance contribution: inverse fall-off, clamped.
        distance_score = 1.0 / (1.0 + distance)

        w = self._weights
        raw = (
            w.distance * distance_score
            + w.screen_size * screen_size
            + w.frequency * frequency
            + base_priority
        )
        return max(0.0, min(1.0, raw))

    def classify(self, priority: float) -> PriorityBucket:
        """Map a priority value to a bucket."""
        for threshold, bucket in _BUCKET_THRESHOLDS:
            if priority >= threshold:
                return bucket
        return PriorityBucket.BACKGROUND
