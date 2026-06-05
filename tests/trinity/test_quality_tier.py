"""Tests for QualityTier enum (T-CC-0.1)."""

import pytest

from trinity.types import QualityTier


class TestQualityTierBasic:
    """Test basic QualityTier enum functionality."""

    def test_tier_ordering(self):
        """Test that tiers are correctly ordered."""
        assert QualityTier.LOW < QualityTier.MEDIUM
        assert QualityTier.MEDIUM < QualityTier.HIGH
        assert QualityTier.HIGH < QualityTier.ULTRA

    def test_tier_values(self):
        """Test tier numeric values."""
        assert QualityTier.LOW.value == 0
        assert QualityTier.MEDIUM.value == 1
        assert QualityTier.HIGH.value == 2
        assert QualityTier.ULTRA.value == 3

    def test_tier_iteration(self):
        """Test iterating over all tiers."""
        tiers = list(QualityTier)
        assert len(tiers) == 4
        assert tiers == [
            QualityTier.LOW,
            QualityTier.MEDIUM,
            QualityTier.HIGH,
            QualityTier.ULTRA,
        ]


class TestQualityTierScore:
    """Test QualityTier score functionality."""

    def test_score_low(self):
        """Test LOW tier score."""
        assert QualityTier.LOW.score == 0.0

    def test_score_medium(self):
        """Test MEDIUM tier score."""
        assert abs(QualityTier.MEDIUM.score - 0.333) < 0.01

    def test_score_high(self):
        """Test HIGH tier score."""
        assert abs(QualityTier.HIGH.score - 0.667) < 0.01

    def test_score_ultra(self):
        """Test ULTRA tier score."""
        assert QualityTier.ULTRA.score == 1.0

    def test_scores_ordered(self):
        """Test that scores are monotonically increasing."""
        prev_score = -1.0
        for tier in QualityTier:
            assert tier.score > prev_score
            prev_score = tier.score


class TestQualityTierFromScore:
    """Test QualityTier.from_score() factory."""

    def test_from_score_low(self):
        """Test score 0.0 maps to LOW."""
        assert QualityTier.from_score(0.0) == QualityTier.LOW

    def test_from_score_low_boundary(self):
        """Test score just below 0.25 maps to LOW."""
        assert QualityTier.from_score(0.24) == QualityTier.LOW

    def test_from_score_medium(self):
        """Test score 0.25 maps to MEDIUM."""
        assert QualityTier.from_score(0.25) == QualityTier.MEDIUM

    def test_from_score_medium_boundary(self):
        """Test score just below 0.5 maps to MEDIUM."""
        assert QualityTier.from_score(0.49) == QualityTier.MEDIUM

    def test_from_score_high(self):
        """Test score 0.5 maps to HIGH."""
        assert QualityTier.from_score(0.5) == QualityTier.HIGH

    def test_from_score_high_boundary(self):
        """Test score just below 0.75 maps to HIGH."""
        assert QualityTier.from_score(0.74) == QualityTier.HIGH

    def test_from_score_ultra(self):
        """Test score 0.75 maps to ULTRA."""
        assert QualityTier.from_score(0.75) == QualityTier.ULTRA

    def test_from_score_max(self):
        """Test score 1.0 maps to ULTRA."""
        assert QualityTier.from_score(1.0) == QualityTier.ULTRA


class TestQualityTierRequirement:
    """Test QualityTier.meets_requirement() method."""

    def test_low_meets_low(self):
        """Test LOW meets LOW requirement."""
        assert QualityTier.LOW.meets_requirement(QualityTier.LOW)

    def test_low_not_meets_medium(self):
        """Test LOW does not meet MEDIUM requirement."""
        assert not QualityTier.LOW.meets_requirement(QualityTier.MEDIUM)

    def test_low_not_meets_high(self):
        """Test LOW does not meet HIGH requirement."""
        assert not QualityTier.LOW.meets_requirement(QualityTier.HIGH)

    def test_low_not_meets_ultra(self):
        """Test LOW does not meet ULTRA requirement."""
        assert not QualityTier.LOW.meets_requirement(QualityTier.ULTRA)

    def test_medium_meets_low(self):
        """Test MEDIUM meets LOW requirement."""
        assert QualityTier.MEDIUM.meets_requirement(QualityTier.LOW)

    def test_medium_meets_medium(self):
        """Test MEDIUM meets MEDIUM requirement."""
        assert QualityTier.MEDIUM.meets_requirement(QualityTier.MEDIUM)

    def test_high_meets_all_lower(self):
        """Test HIGH meets all lower tier requirements."""
        assert QualityTier.HIGH.meets_requirement(QualityTier.LOW)
        assert QualityTier.HIGH.meets_requirement(QualityTier.MEDIUM)
        assert QualityTier.HIGH.meets_requirement(QualityTier.HIGH)

    def test_ultra_meets_all(self):
        """Test ULTRA meets all tier requirements."""
        for tier in QualityTier:
            assert QualityTier.ULTRA.meets_requirement(tier)
