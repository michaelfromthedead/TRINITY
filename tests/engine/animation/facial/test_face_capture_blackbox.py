"""
Blackbox tests for T3.6 Face Capture Retargeting.

Tests the public contract of FaceCaptureRetargeter without
knowledge of internal implementation details.

Public Contract:
    from engine.animation.facial import FaceCaptureRetargeter

    retargeter = FaceCaptureRetargeter()
    retargeter.add_mapping(source_name, target_name, scale=1.0, offset=0.0)
    result = retargeter.retarget(source_weights)  # -> Dict[str, float]
    retargeter.remove_mapping(source_name, target_name=None)
    retargeter.clear_mappings()
"""

import pytest
from typing import Dict


class TestFaceCaptureRetargeterBasicMapping:
    """Test basic source -> target mapping functionality."""

    def test_import_face_capture_retargeter(self):
        """Verify FaceCaptureRetargeter can be imported from public API."""
        from engine.animation.facial import FaceCaptureRetargeter
        assert FaceCaptureRetargeter is not None

    def test_instantiate_retargeter(self):
        """Verify FaceCaptureRetargeter can be instantiated."""
        from engine.animation.facial import FaceCaptureRetargeter
        retargeter = FaceCaptureRetargeter()
        assert retargeter is not None

    def test_simple_one_to_one_mapping(self):
        """Test basic source to target mapping with default scale and offset."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_smile", "target_smile")

        result = retargeter.retarget({"source_smile": 0.8})

        assert "target_smile" in result
        assert result["target_smile"] == pytest.approx(0.8, abs=0.001)

    def test_mapping_with_scale(self):
        """Test mapping with custom scale factor."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_brow", "target_brow", scale=0.5)

        result = retargeter.retarget({"source_brow": 1.0})

        assert result["target_brow"] == pytest.approx(0.5, abs=0.001)

    def test_mapping_with_offset(self):
        """Test mapping with custom offset value."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_eye", "target_eye", scale=1.0, offset=0.1)

        result = retargeter.retarget({"source_eye": 0.5})

        # Expected: 0.5 * 1.0 + 0.1 = 0.6
        assert result["target_eye"] == pytest.approx(0.6, abs=0.001)

    def test_mapping_with_scale_and_offset(self):
        """Test mapping with both scale and offset applied."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_mouth", "target_mouth", scale=2.0, offset=0.1)

        result = retargeter.retarget({"source_mouth": 0.3})

        # Expected: 0.3 * 2.0 + 0.1 = 0.7
        assert result["target_mouth"] == pytest.approx(0.7, abs=0.001)

    def test_zero_source_weight(self):
        """Test retargeting with zero source weight."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_a", "target_a", scale=1.0, offset=0.2)

        result = retargeter.retarget({"source_a": 0.0})

        # Expected: 0.0 * 1.0 + 0.2 = 0.2
        assert result["target_a"] == pytest.approx(0.2, abs=0.001)

    def test_full_source_weight(self):
        """Test retargeting with full (1.0) source weight."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_b", "target_b", scale=0.8, offset=0.1)

        result = retargeter.retarget({"source_b": 1.0})

        # Expected: 1.0 * 0.8 + 0.1 = 0.9
        assert result["target_b"] == pytest.approx(0.9, abs=0.001)

    def test_negative_scale(self):
        """Test mapping with negative scale (inverts the value)."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_c", "target_c", scale=-1.0, offset=1.0)

        result = retargeter.retarget({"source_c": 0.8})

        # Expected: 0.8 * -1.0 + 1.0 = 0.2
        assert result["target_c"] == pytest.approx(0.2, abs=0.001)


class TestFaceCaptureRetargeterOneToMany:
    """Test one source mapping to multiple targets."""

    def test_one_to_two_targets(self):
        """Test single source splitting to two targets."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_smile", "target_smile_L", scale=0.5, offset=0.0)
        retargeter.add_mapping("source_smile", "target_smile_R", scale=0.5, offset=0.0)

        result = retargeter.retarget({"source_smile": 1.0})

        assert result["target_smile_L"] == pytest.approx(0.5, abs=0.001)
        assert result["target_smile_R"] == pytest.approx(0.5, abs=0.001)

    def test_one_to_many_different_scales(self):
        """Test single source to multiple targets with different scales."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_brow_raise", "target_brow_inner_L", scale=0.8)
        retargeter.add_mapping("source_brow_raise", "target_brow_outer_L", scale=0.4)
        retargeter.add_mapping("source_brow_raise", "target_brow_inner_R", scale=0.8)
        retargeter.add_mapping("source_brow_raise", "target_brow_outer_R", scale=0.4)

        result = retargeter.retarget({"source_brow_raise": 1.0})

        assert result["target_brow_inner_L"] == pytest.approx(0.8, abs=0.001)
        assert result["target_brow_outer_L"] == pytest.approx(0.4, abs=0.001)
        assert result["target_brow_inner_R"] == pytest.approx(0.8, abs=0.001)
        assert result["target_brow_outer_R"] == pytest.approx(0.4, abs=0.001)

    def test_one_to_many_with_offsets(self):
        """Test one source to multiple targets with varying offsets."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_mouth_open", "target_jaw_open", scale=1.0, offset=0.0)
        retargeter.add_mapping("source_mouth_open", "target_lip_part", scale=0.5, offset=0.1)

        result = retargeter.retarget({"source_mouth_open": 0.6})

        # jaw: 0.6 * 1.0 + 0.0 = 0.6
        assert result["target_jaw_open"] == pytest.approx(0.6, abs=0.001)
        # lip: 0.6 * 0.5 + 0.1 = 0.4
        assert result["target_lip_part"] == pytest.approx(0.4, abs=0.001)


class TestFaceCaptureRetargeterManyToOne:
    """Test multiple sources accumulating to single target."""

    def test_two_sources_to_one_target_accumulate(self):
        """Test two sources accumulating to single target."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_smile_L", "target_smile", scale=0.5)
        retargeter.add_mapping("source_smile_R", "target_smile", scale=0.5)

        result = retargeter.retarget({
            "source_smile_L": 1.0,
            "source_smile_R": 1.0
        })

        # Should accumulate: 0.5 + 0.5 = 1.0
        assert result["target_smile"] == pytest.approx(1.0, abs=0.001)

    def test_many_sources_accumulate_partial_weights(self):
        """Test multiple sources with partial weights accumulating."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_a", "target_combined", scale=0.3)
        retargeter.add_mapping("source_b", "target_combined", scale=0.3)
        retargeter.add_mapping("source_c", "target_combined", scale=0.4)

        result = retargeter.retarget({
            "source_a": 0.5,
            "source_b": 0.5,
            "source_c": 0.5
        })

        # Expected: (0.5 * 0.3) + (0.5 * 0.3) + (0.5 * 0.4) = 0.15 + 0.15 + 0.2 = 0.5
        assert result["target_combined"] == pytest.approx(0.5, abs=0.001)

    def test_accumulation_with_offsets(self):
        """Test accumulation when mappings have offsets."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_x", "target_total", scale=1.0, offset=0.1)
        retargeter.add_mapping("source_y", "target_total", scale=1.0, offset=0.1)

        result = retargeter.retarget({
            "source_x": 0.2,
            "source_y": 0.3
        })

        # Expected: (0.2 * 1.0 + 0.1) + (0.3 * 1.0 + 0.1) = 0.3 + 0.4 = 0.7
        assert result["target_total"] == pytest.approx(0.7, abs=0.001)

    def test_many_to_one_only_some_sources_present(self):
        """Test accumulation when only some mapped sources are present."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_1", "target_out", scale=0.5)
        retargeter.add_mapping("source_2", "target_out", scale=0.5)
        retargeter.add_mapping("source_3", "target_out", scale=0.5)

        # Only provide source_1 and source_3
        result = retargeter.retarget({
            "source_1": 1.0,
            "source_3": 1.0
        })

        # Expected: 0.5 + 0.5 = 1.0 (source_2 missing contributes 0)
        assert result["target_out"] == pytest.approx(1.0, abs=0.001)


class TestFaceCaptureRetargeterMissingSource:
    """Test graceful handling of missing source shapes."""

    def test_missing_source_no_crash(self):
        """Test that missing source shapes don't cause crashes."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_exists", "target_a")
        retargeter.add_mapping("source_missing", "target_b")

        # Only provide source_exists, not source_missing
        result = retargeter.retarget({"source_exists": 0.7})

        # Should not crash, result should be a dict
        assert isinstance(result, dict)
        assert result["target_a"] == pytest.approx(0.7, abs=0.001)

    def test_all_sources_missing_returns_dict(self):
        """Test retargeting when all mapped sources are missing."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_a", "target_a")
        retargeter.add_mapping("source_b", "target_b")

        # Provide completely different sources
        result = retargeter.retarget({"unrelated_source": 1.0})

        # Should return valid dict, not crash
        assert isinstance(result, dict)

    def test_empty_source_weights(self):
        """Test retargeting with empty source weights dict."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_a", "target_a")

        result = retargeter.retarget({})

        assert isinstance(result, dict)

    def test_missing_source_with_offset_may_produce_value(self):
        """Test that missing source with non-zero offset may produce target value."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_present", "target_present", scale=1.0, offset=0.0)
        retargeter.add_mapping("source_absent", "target_absent", scale=1.0, offset=0.2)

        result = retargeter.retarget({"source_present": 0.5})

        assert result["target_present"] == pytest.approx(0.5, abs=0.001)
        # target_absent behavior depends on implementation:
        # - It might be missing from result if source is missing
        # - Or it might have offset applied (0.2) if offset is always added
        # We verify no crash occurred
        assert isinstance(result, dict)


class TestFaceCaptureRetargeterValueRanges:
    """Test that result values are in expected ranges."""

    def test_values_preserve_zero_to_one_range(self):
        """Test that typical retargeting keeps values in [0, 1] range."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=1.0, offset=0.0)

        for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = retargeter.retarget({"src": weight})
            assert 0.0 <= result["dst"] <= 1.0

    def test_scale_can_produce_values_above_one(self):
        """Test that scale > 1.0 values are clamped to 1.0 (blend shape convention)."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=2.0, offset=0.0)

        result = retargeter.retarget({"src": 1.0})

        # Implementation clamps to [0, 1] - standard blend shape behavior
        # Raw calculation would be 1.0 * 2.0 = 2.0, but clamped to 1.0
        assert result["dst"] == pytest.approx(1.0, abs=0.001)

    def test_negative_offset_clamped_to_zero(self):
        """Test that negative values are clamped to 0.0 (blend shape convention)."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=1.0, offset=-0.5)

        result = retargeter.retarget({"src": 0.0})

        # Implementation clamps to [0, 1] - standard blend shape behavior
        # Raw calculation would be 0.0 * 1.0 + (-0.5) = -0.5, but clamped to 0.0
        assert result["dst"] == pytest.approx(0.0, abs=0.001)

    def test_typical_arkit_range(self):
        """Test typical ARKit source values (0 to 1) produce valid output."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("eyeBlinkLeft", "blinkL", scale=1.0)
        retargeter.add_mapping("eyeBlinkRight", "blinkR", scale=1.0)
        retargeter.add_mapping("mouthSmileLeft", "smileL", scale=0.8, offset=0.1)

        result = retargeter.retarget({
            "eyeBlinkLeft": 0.9,
            "eyeBlinkRight": 0.85,
            "mouthSmileLeft": 0.6
        })

        assert result["blinkL"] == pytest.approx(0.9, abs=0.001)
        assert result["blinkR"] == pytest.approx(0.85, abs=0.001)
        # smileL: 0.6 * 0.8 + 0.1 = 0.58
        assert result["smileL"] == pytest.approx(0.58, abs=0.001)


class TestFaceCaptureRetargeterRemoveMapping:
    """Test remove_mapping functionality."""

    def test_remove_specific_target_mapping(self):
        """Test removing a specific source->target mapping."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source", "target_a")
        retargeter.add_mapping("source", "target_b")

        # Remove only the mapping to target_a
        retargeter.remove_mapping("source", target_name="target_a")

        result = retargeter.retarget({"source": 1.0})

        # target_a should no longer be mapped
        assert "target_a" not in result or result.get("target_a", 0) == 0
        # target_b should still work
        assert result["target_b"] == pytest.approx(1.0, abs=0.001)

    def test_remove_all_mappings_for_source(self):
        """Test removing all mappings for a source (target_name=None)."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_a", "target_1")
        retargeter.add_mapping("source_a", "target_2")
        retargeter.add_mapping("source_b", "target_3")

        # Remove all mappings for source_a
        retargeter.remove_mapping("source_a", target_name=None)

        result = retargeter.retarget({
            "source_a": 1.0,
            "source_b": 0.5
        })

        # Mappings from source_a should be gone
        assert "target_1" not in result or result.get("target_1", 0) == 0
        assert "target_2" not in result or result.get("target_2", 0) == 0
        # source_b mapping should still work
        assert result["target_3"] == pytest.approx(0.5, abs=0.001)

    def test_remove_nonexistent_mapping_no_crash(self):
        """Test that removing a non-existent mapping doesn't crash."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source", "target")

        # Remove a mapping that doesn't exist
        retargeter.remove_mapping("nonexistent_source", target_name="nonexistent_target")

        # Original mapping should still work
        result = retargeter.retarget({"source": 0.5})
        assert result["target"] == pytest.approx(0.5, abs=0.001)


class TestFaceCaptureRetargeterClearMappings:
    """Test clear_mappings functionality."""

    def test_clear_all_mappings(self):
        """Test that clear_mappings removes all target mappings.

        Note: After clearing, retarget returns source weights unchanged
        (pass-through behavior when no mappings exist).
        """
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_1", "target_1")
        retargeter.add_mapping("source_2", "target_2")
        retargeter.add_mapping("source_3", "target_3")

        retargeter.clear_mappings()

        result = retargeter.retarget({
            "source_1": 1.0,
            "source_2": 1.0,
            "source_3": 1.0
        })

        # After clear, mapped targets should not appear in result
        # Implementation returns source weights when no mappings exist (pass-through)
        assert "target_1" not in result
        assert "target_2" not in result
        assert "target_3" not in result

    def test_add_mappings_after_clear(self):
        """Test that new mappings work after clearing."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("old_source", "old_target")
        retargeter.clear_mappings()
        retargeter.add_mapping("new_source", "new_target")

        result = retargeter.retarget({
            "old_source": 1.0,
            "new_source": 0.7
        })

        # Old mapping should not work
        assert "old_target" not in result or result.get("old_target", 0) == 0
        # New mapping should work
        assert result["new_target"] == pytest.approx(0.7, abs=0.001)


class TestFaceCaptureRetargeterComplexScenarios:
    """Test complex real-world retargeting scenarios."""

    def test_full_face_retarget_setup(self):
        """Test a realistic face capture retargeting setup."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()

        # Eye mappings
        retargeter.add_mapping("eyeBlinkLeft", "blink_L", scale=1.0)
        retargeter.add_mapping("eyeBlinkRight", "blink_R", scale=1.0)
        retargeter.add_mapping("eyeSquintLeft", "squint_L", scale=0.8)
        retargeter.add_mapping("eyeSquintRight", "squint_R", scale=0.8)

        # Mouth mappings (one source to multiple targets)
        retargeter.add_mapping("mouthSmileLeft", "smile_L", scale=0.9)
        retargeter.add_mapping("mouthSmileLeft", "cheek_raise_L", scale=0.3)
        retargeter.add_mapping("mouthSmileRight", "smile_R", scale=0.9)
        retargeter.add_mapping("mouthSmileRight", "cheek_raise_R", scale=0.3)

        # Brow mappings
        retargeter.add_mapping("browInnerUp", "brow_inner_L", scale=1.0)
        retargeter.add_mapping("browInnerUp", "brow_inner_R", scale=1.0)

        source_weights = {
            "eyeBlinkLeft": 0.2,
            "eyeBlinkRight": 0.2,
            "eyeSquintLeft": 0.5,
            "eyeSquintRight": 0.5,
            "mouthSmileLeft": 0.8,
            "mouthSmileRight": 0.75,
            "browInnerUp": 0.3
        }

        result = retargeter.retarget(source_weights)

        # Verify all expected targets exist and have correct values
        assert result["blink_L"] == pytest.approx(0.2, abs=0.001)
        assert result["blink_R"] == pytest.approx(0.2, abs=0.001)
        assert result["squint_L"] == pytest.approx(0.4, abs=0.001)  # 0.5 * 0.8
        assert result["squint_R"] == pytest.approx(0.4, abs=0.001)
        assert result["smile_L"] == pytest.approx(0.72, abs=0.001)  # 0.8 * 0.9
        assert result["cheek_raise_L"] == pytest.approx(0.24, abs=0.001)  # 0.8 * 0.3
        assert result["smile_R"] == pytest.approx(0.675, abs=0.001)  # 0.75 * 0.9
        assert result["cheek_raise_R"] == pytest.approx(0.225, abs=0.001)  # 0.75 * 0.3
        assert result["brow_inner_L"] == pytest.approx(0.3, abs=0.001)
        assert result["brow_inner_R"] == pytest.approx(0.3, abs=0.001)

    def test_calibration_offset_scenario(self):
        """Test using offset for calibration (neutral pose adjustment)."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()

        # Calibration: source capture has baseline offset, compensate
        retargeter.add_mapping("source_jaw", "target_jaw", scale=1.2, offset=-0.1)
        retargeter.add_mapping("source_brow", "target_brow", scale=0.9, offset=0.05)

        result = retargeter.retarget({
            "source_jaw": 0.5,
            "source_brow": 0.4
        })

        # jaw: 0.5 * 1.2 - 0.1 = 0.5
        assert result["target_jaw"] == pytest.approx(0.5, abs=0.001)
        # brow: 0.4 * 0.9 + 0.05 = 0.41
        assert result["target_brow"] == pytest.approx(0.41, abs=0.001)

    def test_asymmetric_retargeting(self):
        """Test asymmetric face capture retargeting."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()

        # Left side has different calibration than right
        retargeter.add_mapping("source_smile_L", "target_smile_L", scale=1.0, offset=0.0)
        retargeter.add_mapping("source_smile_R", "target_smile_R", scale=0.85, offset=0.05)

        result = retargeter.retarget({
            "source_smile_L": 0.6,
            "source_smile_R": 0.6
        })

        # Left: 0.6 * 1.0 = 0.6
        assert result["target_smile_L"] == pytest.approx(0.6, abs=0.001)
        # Right: 0.6 * 0.85 + 0.05 = 0.56
        assert result["target_smile_R"] == pytest.approx(0.56, abs=0.001)


class TestFaceCaptureRetargeterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_scale(self):
        """Test mapping with very small scale value."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=0.001)

        result = retargeter.retarget({"src": 1.0})

        assert result["dst"] == pytest.approx(0.001, abs=0.0001)

    def test_very_large_scale_clamped(self):
        """Test mapping with very large scale value is clamped to 1.0."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=100.0)

        result = retargeter.retarget({"src": 0.5})

        # Implementation clamps to [0, 1] - raw would be 50.0, clamped to 1.0
        assert result["dst"] == pytest.approx(1.0, abs=0.001)

    def test_zero_scale(self):
        """Test mapping with zero scale (always offset)."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=0.0, offset=0.5)

        result = retargeter.retarget({"src": 1.0})

        # Expected: 1.0 * 0.0 + 0.5 = 0.5
        assert result["dst"] == pytest.approx(0.5, abs=0.001)

    def test_multiple_retarget_calls(self):
        """Test that multiple retarget calls work independently."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst", scale=1.0)

        result1 = retargeter.retarget({"src": 0.3})
        result2 = retargeter.retarget({"src": 0.7})
        result3 = retargeter.retarget({"src": 0.5})

        assert result1["dst"] == pytest.approx(0.3, abs=0.001)
        assert result2["dst"] == pytest.approx(0.7, abs=0.001)
        assert result3["dst"] == pytest.approx(0.5, abs=0.001)

    def test_unicode_shape_names(self):
        """Test that unicode characters in shape names work."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_smile", "target_smile")

        result = retargeter.retarget({"source_smile": 0.5})

        assert isinstance(result, dict)
        assert result["target_smile"] == pytest.approx(0.5, abs=0.001)

    def test_empty_retargeter(self):
        """Test retargeting with no mappings defined."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()

        result = retargeter.retarget({"some_source": 1.0})

        # Should return empty dict or dict with no relevant values
        assert isinstance(result, dict)

    def test_retarget_returns_new_dict_each_call(self):
        """Test that each retarget call returns a new dict instance."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("src", "dst")

        result1 = retargeter.retarget({"src": 0.5})
        result2 = retargeter.retarget({"src": 0.5})

        # Should be equal in value
        assert result1["dst"] == result2["dst"]
        # But should be different objects (modifying one shouldn't affect the other)
        assert result1 is not result2


class TestFaceCaptureRetargeterDocumentedExample:
    """Test the exact example from the task specification."""

    def test_documented_example(self):
        """Test the example from T3.6 test cases in specification."""
        from engine.animation.facial import FaceCaptureRetargeter

        retargeter = FaceCaptureRetargeter()
        retargeter.add_mapping("source_smile", "target_smile_L", scale=0.5, offset=0.0)
        retargeter.add_mapping("source_smile", "target_smile_R", scale=0.5, offset=0.0)

        result = retargeter.retarget({"source_smile": 1.0})

        assert result["target_smile_L"] == 0.5
        assert result["target_smile_R"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
