"""Tests for StreamPriorityCalculator."""

import pytest

from engine.resource.streaming.priority_system import (
    PriorityBucket,
    PriorityWeights,
    StreamPriorityCalculator,
)


class TestStreamPriorityCalculator:
    def test_close_distance_high_priority(self) -> None:
        calc = StreamPriorityCalculator()
        close = calc.calculate_priority(distance=0.0, screen_size=0.5, frequency=0.5)
        far = calc.calculate_priority(distance=100.0, screen_size=0.5, frequency=0.5)
        assert close > far

    def test_larger_screen_size_higher_priority(self) -> None:
        calc = StreamPriorityCalculator()
        big = calc.calculate_priority(distance=10.0, screen_size=1.0, frequency=0.5)
        small = calc.calculate_priority(distance=10.0, screen_size=0.0, frequency=0.5)
        assert big > small

    def test_output_clamped_to_unit_range(self) -> None:
        calc = StreamPriorityCalculator()
        val = calc.calculate_priority(distance=0.0, screen_size=1.0, frequency=1.0, base_priority=1.0)
        assert val <= 1.0
        val2 = calc.calculate_priority(distance=1e9, screen_size=0.0, frequency=0.0, base_priority=-5.0)
        assert val2 >= 0.0

    def test_classify_critical(self) -> None:
        calc = StreamPriorityCalculator()
        assert calc.classify(0.9) is PriorityBucket.CRITICAL

    def test_classify_background(self) -> None:
        calc = StreamPriorityCalculator()
        assert calc.classify(0.1) is PriorityBucket.BACKGROUND

    def test_custom_weights(self) -> None:
        w = PriorityWeights(distance=1.0, screen_size=0.0, frequency=0.0)
        calc = StreamPriorityCalculator(weights=w)
        # Distance=0 => distance_score=1.0, full weight
        val = calc.calculate_priority(distance=0.0, screen_size=0.0, frequency=0.0)
        assert val == pytest.approx(1.0)
