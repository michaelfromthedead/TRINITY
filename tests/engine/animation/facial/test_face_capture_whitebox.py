"""
Whitebox Tests for Face Capture Retargeting (T3.6).

Tests the FaceCaptureRetargeter class with full access to implementation details.
Covers:
- Single source -> single target mapping
- Single source -> multiple targets (one-to-many)
- Multiple sources -> single target (many-to-one accumulation)
- Scale and offset transformations
- Result clamping to [0, 1]
- Missing source shapes (skip silently)
- Zero scale returns offset only
- Edge cases: empty mappings, negative scale, out-of-range inputs
"""

from __future__ import annotations

import pytest

from engine.animation.facial.face_capture import (
    AnimationCurve,
    FaceCaptureClip,
    FaceCaptureRetargeter,
    InterpolationMode,
    RetargetMapping,
)


# =============================================================================
# RetargetMapping Tests
# =============================================================================


class TestRetargetMapping:
    """Tests for the RetargetMapping dataclass."""

    def test_apply_default_scale_and_offset(self) -> None:
        """Test apply() with default scale=1.0 and offset=0.0."""
        mapping = RetargetMapping(source_name="src", target_name="tgt")
        assert mapping.scale == 1.0
        assert mapping.offset == 0.0
        assert mapping.apply(0.5) == 0.5
        assert mapping.apply(0.0) == 0.0
        assert mapping.apply(1.0) == 1.0

    def test_apply_with_scale(self) -> None:
        """Test apply() with custom scale."""
        mapping = RetargetMapping(source_name="src", target_name="tgt", scale=2.0)
        assert mapping.apply(0.5) == 1.0
        assert mapping.apply(0.25) == 0.5

    def test_apply_with_offset(self) -> None:
        """Test apply() with custom offset."""
        mapping = RetargetMapping(source_name="src", target_name="tgt", offset=0.1)
        assert mapping.apply(0.5) == 0.6
        assert mapping.apply(0.0) == 0.1

    def test_apply_with_scale_and_offset(self) -> None:
        """Test apply() with both scale and offset."""
        mapping = RetargetMapping(
            source_name="src", target_name="tgt", scale=0.5, offset=0.2
        )
        # value * 0.5 + 0.2
        assert mapping.apply(1.0) == 0.7
        assert mapping.apply(0.0) == 0.2
        assert mapping.apply(0.4) == pytest.approx(0.4)

    def test_apply_with_negative_scale(self) -> None:
        """Test apply() with negative scale (inverts value)."""
        mapping = RetargetMapping(
            source_name="src", target_name="tgt", scale=-1.0, offset=1.0
        )
        # Inverts: high input -> low output
        assert mapping.apply(0.0) == 1.0
        assert mapping.apply(1.0) == 0.0
        assert mapping.apply(0.5) == 0.5

    def test_apply_zero_scale_returns_offset_only(self) -> None:
        """Test apply() with zero scale returns only the offset."""
        mapping = RetargetMapping(
            source_name="src", target_name="tgt", scale=0.0, offset=0.3
        )
        assert mapping.apply(0.0) == 0.3
        assert mapping.apply(0.5) == 0.3
        assert mapping.apply(1.0) == 0.3
        assert mapping.apply(100.0) == 0.3


# =============================================================================
# FaceCaptureRetargeter: Basic Mapping Tests
# =============================================================================


class TestRetargeterBasicMappings:
    """Tests for basic single source -> single target mappings."""

    def test_single_source_to_single_target(self) -> None:
        """Test simple one-to-one mapping."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("smile_L", "mouthSmile_L")
        retargeter.set_pass_through(False)

        source_weights = {"smile_L": 0.8}
        result = retargeter.retarget(source_weights)

        assert "mouthSmile_L" in result
        assert result["mouthSmile_L"] == 0.8
        assert "smile_L" not in result  # Original not passed through

    def test_multiple_independent_mappings(self) -> None:
        """Test multiple independent source->target mappings."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("smile_L", "mouthSmile_L")
        retargeter.add_mapping("smile_R", "mouthSmile_R")
        retargeter.add_mapping("blink_L", "eyeBlink_L")
        retargeter.set_pass_through(False)

        source_weights = {"smile_L": 0.5, "smile_R": 0.6, "blink_L": 0.9}
        result = retargeter.retarget(source_weights)

        assert result["mouthSmile_L"] == 0.5
        assert result["mouthSmile_R"] == 0.6
        assert result["eyeBlink_L"] == 0.9

    def test_mapping_count_property(self) -> None:
        """Test mapping_count reflects total mappings."""
        retargeter = FaceCaptureRetargeter()
        assert retargeter.mapping_count == 0

        retargeter.add_mapping("a", "b")
        assert retargeter.mapping_count == 1

        retargeter.add_mapping("a", "c")  # One-to-many: same source
        assert retargeter.mapping_count == 2

        retargeter.add_mapping("x", "y")
        assert retargeter.mapping_count == 3


# =============================================================================
# FaceCaptureRetargeter: Scale and Offset Tests
# =============================================================================


class TestRetargeterScaleAndOffset:
    """Tests for scale and offset transformations."""

    def test_scale_multiplies_value(self) -> None:
        """Test that scale factor multiplies the source value."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("jaw_open", "jawOpen", scale=0.5)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"jaw_open": 1.0})
        assert result["jawOpen"] == 0.5

        result = retargeter.retarget({"jaw_open": 0.6})
        assert result["jawOpen"] == pytest.approx(0.3)

    def test_offset_adds_to_value(self) -> None:
        """Test that offset adds to the scaled value."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("brow", "browUp", offset=0.2)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"brow": 0.5})
        assert result["browUp"] == pytest.approx(0.7)

    def test_scale_and_offset_combined(self) -> None:
        """Test combined scale and offset: value * scale + offset."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("lip", "lipPucker", scale=0.5, offset=0.1)
        retargeter.set_pass_through(False)

        # 0.8 * 0.5 + 0.1 = 0.5
        result = retargeter.retarget({"lip": 0.8})
        assert result["lipPucker"] == pytest.approx(0.5)

    def test_zero_scale_returns_offset(self) -> None:
        """Test that zero scale means result equals offset regardless of input."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("any", "target", scale=0.0, offset=0.4)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"any": 1.0})
        assert result["target"] == 0.4

        result = retargeter.retarget({"any": 0.0})
        assert result["target"] == 0.4

        result = retargeter.retarget({"any": 0.5})
        assert result["target"] == 0.4

    def test_negative_scale_inverts_value(self) -> None:
        """Test negative scale inverts the value range."""
        retargeter = FaceCaptureRetargeter()
        # Invert: -1.0 * value + 1.0, so 0->1 and 1->0
        retargeter.add_mapping("open", "closed", scale=-1.0, offset=1.0)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"open": 0.0})
        assert result["closed"] == 1.0

        result = retargeter.retarget({"open": 1.0})
        assert result["closed"] == 0.0

        result = retargeter.retarget({"open": 0.3})
        assert result["closed"] == pytest.approx(0.7)


# =============================================================================
# FaceCaptureRetargeter: One-to-Many Mappings
# =============================================================================


class TestRetargeterOneToMany:
    """Tests for single source -> multiple targets (one-to-many)."""

    def test_single_source_to_multiple_targets(self) -> None:
        """Test that one source can drive multiple targets."""
        retargeter = FaceCaptureRetargeter()
        # Smile drives both mouth corner and cheek
        retargeter.add_mapping("smile", "mouthCorner", scale=1.0)
        retargeter.add_mapping("smile", "cheekPuff", scale=0.3)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"smile": 0.8})

        assert "mouthCorner" in result
        assert "cheekPuff" in result
        assert result["mouthCorner"] == 0.8
        assert result["cheekPuff"] == pytest.approx(0.24)

    def test_one_to_many_with_different_scales(self) -> None:
        """Test one-to-many with varying scale factors."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("brow_raise", "innerBrow", scale=1.0)
        retargeter.add_mapping("brow_raise", "outerBrow", scale=0.7)
        retargeter.add_mapping("brow_raise", "forehead", scale=0.2)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"brow_raise": 1.0})

        assert result["innerBrow"] == 1.0
        assert result["outerBrow"] == pytest.approx(0.7)
        assert result["forehead"] == pytest.approx(0.2)

    def test_one_to_many_with_offsets(self) -> None:
        """Test one-to-many with different offsets."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("frown", "browDown_L", scale=0.5, offset=0.1)
        retargeter.add_mapping("frown", "browDown_R", scale=0.5, offset=0.1)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"frown": 0.6})

        # 0.6 * 0.5 + 0.1 = 0.4
        assert result["browDown_L"] == pytest.approx(0.4)
        assert result["browDown_R"] == pytest.approx(0.4)


# =============================================================================
# FaceCaptureRetargeter: Many-to-One Accumulation
# =============================================================================


class TestRetargeterManyToOne:
    """Tests for multiple sources -> single target (many-to-one accumulation)."""

    def test_multiple_sources_to_single_target_accumulates(self) -> None:
        """Test that multiple sources mapping to same target accumulate."""
        retargeter = FaceCaptureRetargeter()
        # Both smile_L and smile_R contribute to overall smile
        retargeter.add_mapping("smile_L", "smile", scale=0.5)
        retargeter.add_mapping("smile_R", "smile", scale=0.5)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"smile_L": 0.8, "smile_R": 0.6})

        # (0.8 * 0.5) + (0.6 * 0.5) = 0.4 + 0.3 = 0.7
        assert result["smile"] == pytest.approx(0.7)

    def test_many_to_one_with_different_scales(self) -> None:
        """Test many-to-one with different scale factors."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("upper_lip", "mouthPucker", scale=0.6)
        retargeter.add_mapping("lower_lip", "mouthPucker", scale=0.4)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"upper_lip": 1.0, "lower_lip": 1.0})

        # 1.0 * 0.6 + 1.0 * 0.4 = 1.0
        assert result["mouthPucker"] == 1.0

    def test_many_to_one_partial_sources(self) -> None:
        """Test many-to-one when only some sources are present."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("a", "target", scale=0.5)
        retargeter.add_mapping("b", "target", scale=0.3)
        retargeter.add_mapping("c", "target", scale=0.2)
        retargeter.set_pass_through(False)

        # Only 'a' and 'c' present, 'b' missing
        result = retargeter.retarget({"a": 1.0, "c": 1.0})

        # 1.0 * 0.5 + 1.0 * 0.2 = 0.7
        assert result["target"] == pytest.approx(0.7)

    def test_many_to_one_accumulation_clamped(self) -> None:
        """Test that accumulated values are clamped to [0, 1]."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("a", "target", scale=1.0)
        retargeter.add_mapping("b", "target", scale=1.0)
        retargeter.add_mapping("c", "target", scale=1.0)
        retargeter.set_pass_through(False)

        # All at 0.5: sum = 1.5, should clamp to 1.0
        result = retargeter.retarget({"a": 0.5, "b": 0.5, "c": 0.5})

        assert result["target"] == 1.0  # Clamped


# =============================================================================
# FaceCaptureRetargeter: Clamping
# =============================================================================


class TestRetargeterClamping:
    """Tests for result clamping to [0, 1]."""

    def test_clamp_high_value_to_one(self) -> None:
        """Test values above 1.0 are clamped to 1.0."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=2.0)  # Will double
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.8})
        # 0.8 * 2.0 = 1.6 -> clamped to 1.0
        assert result["tgt"] == 1.0

    def test_clamp_negative_value_to_zero(self) -> None:
        """Test values below 0.0 are clamped to 0.0."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=1.0, offset=-0.5)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.3})
        # 0.3 * 1.0 - 0.5 = -0.2 -> clamped to 0.0
        assert result["tgt"] == 0.0

    def test_clamp_with_negative_scale_producing_negative(self) -> None:
        """Test negative scale producing negative values clamps to 0."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=-1.0, offset=0.0)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.5})
        # -0.5 -> clamped to 0.0
        assert result["tgt"] == 0.0

    def test_clamp_with_large_offset(self) -> None:
        """Test large offset clamped correctly."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=0.0, offset=2.0)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.0})
        assert result["tgt"] == 1.0  # Clamped from 2.0

    def test_clamp_with_negative_offset(self) -> None:
        """Test negative offset clamped correctly."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=0.0, offset=-1.0)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.0})
        assert result["tgt"] == 0.0  # Clamped from -1.0


# =============================================================================
# FaceCaptureRetargeter: Missing Source Shapes
# =============================================================================


class TestRetargeterMissingSources:
    """Tests for handling missing source shapes."""

    def test_missing_source_shape_skipped_silently(self) -> None:
        """Test that missing source shapes are skipped without error."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("expected_shape", "target")
        retargeter.set_pass_through(False)

        # Source weights don't include 'expected_shape'
        result = retargeter.retarget({"other_shape": 0.5})

        # Target should not appear since source is missing
        assert "target" not in result

    def test_partial_source_shapes_processed(self) -> None:
        """Test that available sources are processed even if some are missing."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("shape_a", "target_a")
        retargeter.add_mapping("shape_b", "target_b")
        retargeter.add_mapping("shape_c", "target_c")
        retargeter.set_pass_through(False)

        # Only shape_a and shape_c present
        result = retargeter.retarget({"shape_a": 0.5, "shape_c": 0.8})

        assert result["target_a"] == 0.5
        assert result["target_c"] == 0.8
        assert "target_b" not in result

    def test_all_sources_missing(self) -> None:
        """Test when all mapped sources are missing."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("expected", "target")
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"completely_different": 1.0})

        assert result == {}


# =============================================================================
# FaceCaptureRetargeter: Pass-Through Behavior
# =============================================================================


class TestRetargeterPassThrough:
    """Tests for unmapped shape pass-through behavior."""

    def test_pass_through_enabled_by_default(self) -> None:
        """Test that pass-through is enabled by default."""
        retargeter = FaceCaptureRetargeter()
        # No mappings added

        result = retargeter.retarget({"unmapped": 0.7})

        assert "unmapped" in result
        assert result["unmapped"] == 0.7

    def test_pass_through_disabled(self) -> None:
        """Test that disabled pass-through drops unmapped shapes."""
        retargeter = FaceCaptureRetargeter()
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"unmapped": 0.7})

        assert "unmapped" not in result

    def test_pass_through_with_custom_scale(self) -> None:
        """Test pass-through with custom scale factor."""
        retargeter = FaceCaptureRetargeter()
        retargeter.set_pass_through(True, scale=0.5)

        result = retargeter.retarget({"unmapped": 0.8})

        assert result["unmapped"] == pytest.approx(0.4)

    def test_pass_through_mixed_with_mappings(self) -> None:
        """Test pass-through works alongside mapped shapes."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("mapped", "target")
        retargeter.set_pass_through(True, scale=1.0)

        result = retargeter.retarget({"mapped": 0.5, "unmapped": 0.8})

        assert result["target"] == 0.5
        assert result["unmapped"] == 0.8


# =============================================================================
# FaceCaptureRetargeter: Edge Cases
# =============================================================================


class TestRetargeterEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_source_weights(self) -> None:
        """Test retargeting empty source weights."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("a", "b")

        result = retargeter.retarget({})

        assert result == {}

    def test_empty_mappings_pass_through_enabled(self) -> None:
        """Test with no mappings and pass-through enabled."""
        retargeter = FaceCaptureRetargeter()
        # Default pass-through is enabled

        result = retargeter.retarget({"shape1": 0.5, "shape2": 0.7})

        assert result["shape1"] == 0.5
        assert result["shape2"] == 0.7

    def test_empty_mappings_pass_through_disabled(self) -> None:
        """Test with no mappings and pass-through disabled."""
        retargeter = FaceCaptureRetargeter()
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"shape1": 0.5, "shape2": 0.7})

        assert result == {}

    def test_source_value_zero(self) -> None:
        """Test handling of zero source values."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=1.0, offset=0.1)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.0})

        assert result["tgt"] == 0.1

    def test_source_value_one(self) -> None:
        """Test handling of maximum source value (1.0)."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=0.8, offset=0.1)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 1.0})

        assert result["tgt"] == pytest.approx(0.9)

    def test_source_value_above_one(self) -> None:
        """Test handling of source values > 1.0 (out of expected range)."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=0.5)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 1.5})

        # 1.5 * 0.5 = 0.75, within [0,1]
        assert result["tgt"] == pytest.approx(0.75)

    def test_source_value_negative(self) -> None:
        """Test handling of negative source values (out of expected range)."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=1.0, offset=0.5)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": -0.3})

        # -0.3 + 0.5 = 0.2
        assert result["tgt"] == pytest.approx(0.2)

    def test_source_value_very_negative(self) -> None:
        """Test handling of very negative source values (clamped)."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=1.0)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": -2.0})

        # -2.0 -> clamped to 0.0
        assert result["tgt"] == 0.0

    def test_very_small_scale(self) -> None:
        """Test with very small scale factor."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=0.001)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 1.0})

        assert result["tgt"] == pytest.approx(0.001)

    def test_very_large_scale(self) -> None:
        """Test with very large scale factor (result clamped)."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt", scale=100.0)
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"src": 0.1})

        # 0.1 * 100 = 10 -> clamped to 1.0
        assert result["tgt"] == 1.0

    def test_identity_mapping(self) -> None:
        """Test identity mapping (same name, scale=1, offset=0)."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("smile", "smile")
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"smile": 0.6})

        assert result["smile"] == 0.6

    def test_create_identity_mappings(self) -> None:
        """Test bulk creation of identity mappings."""
        retargeter = FaceCaptureRetargeter()
        retargeter.create_identity_mappings(["a", "b", "c"])
        retargeter.set_pass_through(False)

        result = retargeter.retarget({"a": 0.1, "b": 0.2, "c": 0.3})

        assert result["a"] == 0.1
        assert result["b"] == 0.2
        assert result["c"] == 0.3


# =============================================================================
# FaceCaptureRetargeter: Mapping Management
# =============================================================================


class TestRetargeterMappingManagement:
    """Tests for mapping add/remove/clear operations."""

    def test_remove_mapping_by_source_only(self) -> None:
        """Test removing all mappings for a source."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt1")
        retargeter.add_mapping("src", "tgt2")
        assert retargeter.mapping_count == 2

        removed = retargeter.remove_mapping("src")

        assert removed is True
        assert retargeter.mapping_count == 0

    def test_remove_mapping_by_source_and_target(self) -> None:
        """Test removing specific source->target mapping."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "tgt1")
        retargeter.add_mapping("src", "tgt2")

        removed = retargeter.remove_mapping("src", "tgt1")

        assert removed is True
        assert retargeter.mapping_count == 1
        mappings = retargeter.get_mappings("src")
        assert len(mappings) == 1
        assert mappings[0].target_name == "tgt2"

    def test_remove_nonexistent_mapping(self) -> None:
        """Test removing a mapping that doesn't exist."""
        retargeter = FaceCaptureRetargeter()

        removed = retargeter.remove_mapping("nonexistent")

        assert removed is False

    def test_clear_mappings(self) -> None:
        """Test clearing all mappings."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("a", "b")
        retargeter.add_mapping("c", "d")
        assert retargeter.mapping_count == 2

        retargeter.clear_mappings()

        assert retargeter.mapping_count == 0

    def test_get_mapping_returns_first(self) -> None:
        """Test get_mapping returns first mapping for one-to-many."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "first")
        retargeter.add_mapping("src", "second")

        mapping = retargeter.get_mapping("src")

        assert mapping is not None
        assert mapping.target_name == "first"

    def test_get_mappings_returns_all(self) -> None:
        """Test get_mappings returns all mappings for a source."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "first", scale=1.0)
        retargeter.add_mapping("src", "second", scale=0.5)

        mappings = retargeter.get_mappings("src")

        assert len(mappings) == 2
        assert mappings[0].target_name == "first"
        assert mappings[1].target_name == "second"

    def test_get_mappings_nonexistent_source(self) -> None:
        """Test get_mappings for nonexistent source returns empty list."""
        retargeter = FaceCaptureRetargeter()

        mappings = retargeter.get_mappings("nonexistent")

        assert mappings == []


# =============================================================================
# FaceCaptureRetargeter: Constructor with Mappings
# =============================================================================


class TestRetargeterConstructor:
    """Tests for constructor with initial mappings."""

    def test_constructor_with_mappings(self) -> None:
        """Test initializing retargeter with mappings list."""
        mappings = [
            RetargetMapping("a", "b", scale=1.0),
            RetargetMapping("c", "d", scale=0.5),
        ]
        retargeter = FaceCaptureRetargeter(mappings=mappings)

        assert retargeter.mapping_count == 2
        assert retargeter.get_mapping("a").target_name == "b"
        assert retargeter.get_mapping("c").target_name == "d"

    def test_constructor_empty(self) -> None:
        """Test constructor with no mappings."""
        retargeter = FaceCaptureRetargeter()

        assert retargeter.mapping_count == 0

    def test_constructor_with_one_to_many(self) -> None:
        """Test constructor handles one-to-many in initial mappings."""
        mappings = [
            RetargetMapping("src", "tgt1"),
            RetargetMapping("src", "tgt2"),
        ]
        retargeter = FaceCaptureRetargeter(mappings=mappings)

        assert retargeter.mapping_count == 2
        all_mappings = retargeter.get_mappings("src")
        assert len(all_mappings) == 2


# =============================================================================
# FaceCaptureRetargeter: Serialization
# =============================================================================


class TestRetargeterSerialization:
    """Tests for to_dict/from_dict serialization."""

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("a", "b", scale=0.8, offset=0.1)
        retargeter.set_pass_through(False, scale=0.5)

        data = retargeter.to_dict()

        assert "mappings" in data
        assert len(data["mappings"]) == 1
        assert data["mappings"][0]["source_name"] == "a"
        assert data["mappings"][0]["target_name"] == "b"
        assert data["mappings"][0]["scale"] == 0.8
        assert data["mappings"][0]["offset"] == 0.1
        assert data["pass_through_unmapped"] is False
        assert data["unmapped_scale"] == 0.5

    def test_from_dict(self) -> None:
        """Test deserialization from dictionary."""
        data = {
            "mappings": [
                {"source_name": "x", "target_name": "y", "scale": 0.5, "offset": 0.2}
            ],
            "pass_through_unmapped": False,
            "unmapped_scale": 0.7,
        }

        retargeter = FaceCaptureRetargeter.from_dict(data)

        assert retargeter.mapping_count == 1
        mapping = retargeter.get_mapping("x")
        assert mapping.target_name == "y"
        assert mapping.scale == 0.5
        assert mapping.offset == 0.2
        assert retargeter._pass_through_unmapped is False
        assert retargeter._unmapped_scale == 0.7

    def test_round_trip_serialization(self) -> None:
        """Test that to_dict -> from_dict preserves state."""
        original = FaceCaptureRetargeter()
        original.add_mapping("smile_L", "mouthSmile_L", scale=0.9, offset=0.05)
        original.add_mapping("smile_L", "cheek_L", scale=0.3)
        original.add_mapping("blink", "eyeBlink", scale=1.0)
        original.set_pass_through(True, scale=0.8)

        data = original.to_dict()
        restored = FaceCaptureRetargeter.from_dict(data)

        assert restored.mapping_count == original.mapping_count
        assert len(restored.get_mappings("smile_L")) == 2
        assert restored._pass_through_unmapped is True
        assert restored._unmapped_scale == 0.8


# =============================================================================
# FaceCaptureRetargeter: Clip Retargeting
# =============================================================================


class TestRetargeterClipRetargeting:
    """Tests for retargeting entire clips."""

    def test_retarget_clip_basic(self) -> None:
        """Test basic clip retargeting."""
        # Create source clip
        source_clip = FaceCaptureClip(name="source", frame_rate=30.0)
        curve = AnimationCurve(name="smile")
        curve.add_keyframe(0.0, 0.0)
        curve.add_keyframe(1.0, 1.0)
        source_clip.add_curve(curve)

        # Create retargeter
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("smile", "mouthSmile", scale=0.5)
        retargeter.set_pass_through(False)

        # Retarget
        target_clip = retargeter.retarget_clip(source_clip)

        assert target_clip.name == "source_retargeted"
        assert "mouthSmile" in target_clip.curves
        assert "smile" not in target_clip.curves

        # Check keyframe values are scaled and clamped
        target_curve = target_clip.curves["mouthSmile"]
        assert len(target_curve.keyframes) == 2
        assert target_curve.keyframes[0].value == 0.0  # 0 * 0.5 = 0
        assert target_curve.keyframes[1].value == 0.5  # 1 * 0.5 = 0.5

    def test_retarget_clip_custom_name(self) -> None:
        """Test clip retargeting with custom target name."""
        source_clip = FaceCaptureClip(name="source")

        retargeter = FaceCaptureRetargeter()
        target_clip = retargeter.retarget_clip(source_clip, target_name="custom_name")

        assert target_clip.name == "custom_name"

    def test_retarget_clip_preserves_metadata(self) -> None:
        """Test that clip metadata is preserved during retargeting."""
        source_clip = FaceCaptureClip(
            name="source",
            frame_rate=60.0,
            metadata={"actor": "test_actor", "session": 42},
        )

        retargeter = FaceCaptureRetargeter()
        target_clip = retargeter.retarget_clip(source_clip)

        assert target_clip.frame_rate == 60.0
        assert target_clip.metadata["actor"] == "test_actor"
        assert target_clip.metadata["session"] == 42

    def test_retarget_clip_one_to_many(self) -> None:
        """Test clip retargeting with one-to-many mapping."""
        source_clip = FaceCaptureClip(name="source")
        curve = AnimationCurve(name="brow")
        curve.add_keyframe(0.0, 0.5)
        source_clip.add_curve(curve)

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("brow", "browInner", scale=1.0)
        retargeter.add_mapping("brow", "browOuter", scale=0.7)
        retargeter.set_pass_through(False)

        target_clip = retargeter.retarget_clip(source_clip)

        assert "browInner" in target_clip.curves
        assert "browOuter" in target_clip.curves
        assert target_clip.curves["browInner"].keyframes[0].value == 0.5
        assert target_clip.curves["browOuter"].keyframes[0].value == pytest.approx(0.35)


# =============================================================================
# Integration Tests
# =============================================================================


class TestRetargeterIntegration:
    """Integration tests combining multiple features."""

    def test_complex_face_rig_retargeting(self) -> None:
        """Test a realistic face rig retargeting scenario."""
        # Set up retargeter for ARKit to custom rig mapping
        retargeter = FaceCaptureRetargeter()

        # Mouth mappings (one-to-many)
        retargeter.add_mapping("mouthSmile_L", "smile_L", scale=0.9)
        retargeter.add_mapping("mouthSmile_L", "cheek_L", scale=0.2)
        retargeter.add_mapping("mouthSmile_R", "smile_R", scale=0.9)
        retargeter.add_mapping("mouthSmile_R", "cheek_R", scale=0.2)

        # Jaw (scale adjustment)
        retargeter.add_mapping("jawOpen", "jaw", scale=0.7, offset=0.05)

        # Brows (many-to-one for combined expression)
        retargeter.add_mapping("browInnerUp_L", "brow_concern", scale=0.5)
        retargeter.add_mapping("browInnerUp_R", "brow_concern", scale=0.5)

        retargeter.set_pass_through(False)

        # Test weights
        source = {
            "mouthSmile_L": 0.8,
            "mouthSmile_R": 0.7,
            "jawOpen": 0.5,
            "browInnerUp_L": 0.6,
            "browInnerUp_R": 0.4,
            "unmapped_shape": 0.9,  # Should be dropped
        }

        result = retargeter.retarget(source)

        # Verify one-to-many
        assert result["smile_L"] == pytest.approx(0.72)  # 0.8 * 0.9
        assert result["cheek_L"] == pytest.approx(0.16)  # 0.8 * 0.2
        assert result["smile_R"] == pytest.approx(0.63)  # 0.7 * 0.9
        assert result["cheek_R"] == pytest.approx(0.14)  # 0.7 * 0.2

        # Verify scale + offset
        assert result["jaw"] == pytest.approx(0.4)  # 0.5 * 0.7 + 0.05

        # Verify many-to-one accumulation
        # 0.6 * 0.5 + 0.4 * 0.5 = 0.3 + 0.2 = 0.5
        assert result["brow_concern"] == pytest.approx(0.5)

        # Verify unmapped dropped
        assert "unmapped_shape" not in result

    def test_retarget_weights_alias(self) -> None:
        """Test that retarget_weights is an alias for retarget."""
        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("a", "b")
        retargeter.set_pass_through(False)

        weights = {"a": 0.5}
        result1 = retargeter.retarget(weights)
        result2 = retargeter.retarget_weights(weights)

        assert result1 == result2
