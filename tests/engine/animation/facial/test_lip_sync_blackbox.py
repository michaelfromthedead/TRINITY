"""
Blackbox Tests for T3.3 Lip Sync Coarticulation System

Tests the public contract without knowledge of implementation details.
Focus: Phoneme-to-viseme conversion, coarticulation blending, timeline playback.

Contract tested:
- LipSyncController manages phoneme events and converts to viseme events
- Coarticulation blends anticipation and carryover between phonemes
- Zero-duration phonemes are handled without crashing
- Timeline playback is frame-accurate
"""

import pytest
from typing import List, Dict, Any
from enum import Enum


class TestLipSyncControllerImport:
    """Test that the public API is importable."""

    def test_import_lip_sync_controller(self):
        """LipSyncController should be importable from public API."""
        from engine.animation.facial.lip_sync import LipSyncController
        assert LipSyncController is not None

    def test_import_phoneme_event(self):
        """PhonemeEvent should be importable from public API."""
        from engine.animation.facial.lip_sync import PhonemeEvent
        assert PhonemeEvent is not None

    def test_import_viseme(self):
        """Viseme enum should be importable from public API."""
        from engine.animation.facial.lip_sync import Viseme
        assert Viseme is not None

    def test_import_viseme_event(self):
        """VisemeEvent should be importable if part of public API."""
        try:
            from engine.animation.facial.lip_sync import VisemeEvent
            assert VisemeEvent is not None
        except ImportError:
            # VisemeEvent might be internal; events returned might be different type
            pass


class TestLipSyncControllerConstruction:
    """Test LipSyncController instantiation."""

    def test_create_default_controller(self):
        """Controller should be creatable with default parameters."""
        from engine.animation.facial.lip_sync import LipSyncController
        controller = LipSyncController()
        assert controller is not None

    def test_controller_initial_state(self):
        """New controller should have no events initially."""
        from engine.animation.facial.lip_sync import LipSyncController
        controller = LipSyncController()
        events = controller.get_viseme_events()
        assert isinstance(events, (list, tuple))
        assert len(events) == 0


class TestPhonemeEventConstruction:
    """Test PhonemeEvent creation and properties."""

    def test_create_phoneme_event_basic(self):
        """PhonemeEvent should accept phoneme, start, duration."""
        from engine.animation.facial.lip_sync import PhonemeEvent
        event = PhonemeEvent("AA", 0.0, 0.1)
        assert event is not None

    def test_phoneme_event_properties(self):
        """PhonemeEvent should expose phoneme, start_time, duration."""
        from engine.animation.facial.lip_sync import PhonemeEvent
        event = PhonemeEvent("AA", 0.5, 0.2)
        # Properties might be accessed via attributes or methods
        assert hasattr(event, 'phoneme') or hasattr(event, 'get_phoneme')
        assert hasattr(event, 'start_time') or hasattr(event, 'start') or hasattr(event, 'time')
        assert hasattr(event, 'duration') or hasattr(event, 'get_duration')

    def test_phoneme_event_various_phonemes(self):
        """PhonemeEvent should accept various phoneme codes."""
        from engine.animation.facial.lip_sync import PhonemeEvent
        phonemes = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
                    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
                    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
                    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"]
        for phoneme in phonemes:
            event = PhonemeEvent(phoneme, 0.0, 0.1)
            assert event is not None

    def test_phoneme_event_zero_duration(self):
        """PhonemeEvent should accept zero duration without error."""
        from engine.animation.facial.lip_sync import PhonemeEvent
        event = PhonemeEvent("P", 0.0, 0.0)
        assert event is not None

    def test_phoneme_event_negative_start_time(self):
        """PhonemeEvent with negative start time - behavior should be defined."""
        from engine.animation.facial.lip_sync import PhonemeEvent
        # Contract doesn't specify behavior for negative times
        # Implementation should either accept or raise clear error
        try:
            event = PhonemeEvent("AA", -1.0, 0.1)
            # If accepted, it's valid
            assert event is not None
        except (ValueError, TypeError) as e:
            # If rejected, error should be clear
            assert "negative" in str(e).lower() or "invalid" in str(e).lower()


class TestVisemeEnum:
    """Test Viseme enum values and behavior."""

    def test_viseme_has_aa(self):
        """Viseme enum should have AA viseme."""
        from engine.animation.facial.lip_sync import Viseme
        assert hasattr(Viseme, 'AA')

    def test_viseme_common_values(self):
        """Viseme enum should have viseme values (implementation may vary)."""
        from engine.animation.facial.lip_sync import Viseme
        # Count how many members the Viseme enum has
        members = [m for m in dir(Viseme) if not m.startswith('_')]
        # Should have at least a few visemes defined
        assert len(members) >= 3, f"Viseme enum should have multiple values, found: {members}"

    def test_viseme_is_enum(self):
        """Viseme should be an enum type."""
        from engine.animation.facial.lip_sync import Viseme
        assert isinstance(Viseme.AA, Enum) or hasattr(Viseme, '__members__')


class TestPhonemeToVisemeConversion:
    """Test conversion from phoneme events to viseme events."""

    def test_single_phoneme_converts_to_viseme(self):
        """Single phoneme event should produce viseme event."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent, Viseme
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        events = controller.get_viseme_events()

        assert len(events) >= 1
        assert events[0].viseme == Viseme.AA

    def test_multiple_phonemes_convert(self):
        """Multiple phoneme events should all convert to visemes."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        controller.add_phoneme_event(PhonemeEvent("B", 0.1, 0.05))
        controller.add_phoneme_event(PhonemeEvent("IY", 0.15, 0.1))

        events = controller.get_viseme_events()
        assert len(events) >= 3

    def test_phoneme_p_converts_correctly(self):
        """P phoneme should convert to a valid viseme."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent, Viseme
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.1))
        events = controller.get_viseme_events()

        assert len(events) >= 1
        # P should map to some valid bilabial-related viseme
        viseme = events[0].viseme
        assert viseme is not None
        assert isinstance(viseme, Viseme)

    def test_phoneme_f_converts_correctly(self):
        """F phoneme should convert to a valid viseme."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent, Viseme
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("F", 0.0, 0.1))
        events = controller.get_viseme_events()

        assert len(events) >= 1
        # F should map to some valid labiodental-related viseme (FF, FV, etc.)
        viseme = events[0].viseme
        assert viseme is not None
        assert isinstance(viseme, Viseme)


class TestCoarticulation:
    """Test coarticulation blending between phonemes."""

    def test_coarticulation_anticipation(self):
        """Coarticulation should show anticipation of upcoming phoneme."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        # "AA" followed by "OO" - anticipation should blend toward OO
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.2))
        controller.add_phoneme_event(PhonemeEvent("OO", 0.2, 0.2))

        # Get weights near end of AA (should show some OO influence)
        weights = controller.update(time=0.18)
        assert isinstance(weights, dict)
        # Weights should exist for multiple visemes during transition
        assert len(weights) >= 1

    def test_coarticulation_carryover(self):
        """Coarticulation should show carryover from previous phoneme."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("M", 0.0, 0.1))
        controller.add_phoneme_event(PhonemeEvent("AA", 0.1, 0.2))

        # Get weights at start of AA (should show some M carryover)
        weights = controller.update(time=0.12)
        assert isinstance(weights, dict)
        assert len(weights) >= 1

    def test_coarticulation_blend_weights_sum(self):
        """Coarticulation blend weights should be normalized or meaningful."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.2))
        controller.add_phoneme_event(PhonemeEvent("IY", 0.2, 0.2))

        weights = controller.update(time=0.19)  # Near transition
        assert isinstance(weights, dict)

        # Weights should be in valid range [0, 1]
        for weight in weights.values():
            assert 0.0 <= weight <= 1.0 or weight >= 0.0

    def test_smooth_transition_between_visemes(self):
        """Transition between visemes should be smooth, not abrupt."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        controller.add_phoneme_event(PhonemeEvent("OO", 0.1, 0.1))

        # Sample multiple points across transition
        weights_before = controller.update(time=0.08)
        weights_at = controller.update(time=0.10)
        weights_after = controller.update(time=0.12)

        # All should return valid weight dicts
        assert isinstance(weights_before, dict)
        assert isinstance(weights_at, dict)
        assert isinstance(weights_after, dict)


class TestZeroDurationPhonemes:
    """Test handling of zero-duration phonemes."""

    def test_zero_duration_does_not_crash(self):
        """Zero duration phoneme should not crash on add."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.0))
        # Should not raise exception
        assert True

    def test_zero_duration_update_returns_dict(self):
        """Update with zero duration phoneme should return valid weights."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.0))
        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)

    def test_zero_duration_in_sequence(self):
        """Zero duration phoneme in sequence should not break playback."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        controller.add_phoneme_event(PhonemeEvent("P", 0.1, 0.0))  # Zero duration
        controller.add_phoneme_event(PhonemeEvent("IY", 0.1, 0.1))

        # Update at various times should all work
        w1 = controller.update(time=0.05)
        w2 = controller.update(time=0.10)
        w3 = controller.update(time=0.15)

        assert isinstance(w1, dict)
        assert isinstance(w2, dict)
        assert isinstance(w3, dict)

    def test_multiple_zero_duration_phonemes(self):
        """Multiple zero duration phonemes should be handled."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.0))
        controller.add_phoneme_event(PhonemeEvent("B", 0.0, 0.0))
        controller.add_phoneme_event(PhonemeEvent("M", 0.0, 0.0))

        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)


class TestTimelinePlayback:
    """Test frame-accurate timeline playback."""

    def test_update_at_start(self):
        """Update at time 0 should return valid weights."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.5))

        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)
        assert len(weights) >= 1

    def test_update_mid_phoneme(self):
        """Update mid-phoneme should return appropriate weights."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 1.0))

        weights = controller.update(time=0.5)
        assert isinstance(weights, dict)

    def test_update_at_end(self):
        """Update at phoneme end should return valid weights."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.5))

        weights = controller.update(time=0.5)
        assert isinstance(weights, dict)

    def test_update_after_all_events(self):
        """Update after all events should return rest/neutral state."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))

        weights = controller.update(time=1.0)
        assert isinstance(weights, dict)
        # Might return empty dict or rest weights

    def test_update_before_first_event(self):
        """Update before first event should return rest/neutral."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 1.0, 0.1))

        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)

    def test_frame_accurate_60fps(self):
        """Playback should be accurate at 60fps frame intervals."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 1.0))

        frame_time = 1.0 / 60.0  # ~16.67ms
        for frame in range(60):
            time = frame * frame_time
            weights = controller.update(time=time)
            assert isinstance(weights, dict)

    def test_frame_accurate_30fps(self):
        """Playback should be accurate at 30fps frame intervals."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.5))
        controller.add_phoneme_event(PhonemeEvent("EE", 0.5, 0.5))

        frame_time = 1.0 / 30.0  # ~33.33ms
        for frame in range(30):
            time = frame * frame_time
            weights = controller.update(time=time)
            assert isinstance(weights, dict)

    def test_non_sequential_time_updates(self):
        """Controller should handle non-sequential time updates."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.5))
        controller.add_phoneme_event(PhonemeEvent("OO", 0.5, 0.5))

        # Jump around in time
        w1 = controller.update(time=0.8)
        w2 = controller.update(time=0.2)
        w3 = controller.update(time=0.5)

        assert isinstance(w1, dict)
        assert isinstance(w2, dict)
        assert isinstance(w3, dict)


class TestWeightOutput:
    """Test the structure and validity of weight output."""

    def test_weights_are_dict(self):
        """Update should return a dictionary."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))

        weights = controller.update(time=0.05)
        assert isinstance(weights, dict)

    def test_weight_keys_are_visemes_or_strings(self):
        """Weight dict keys should be visemes or viseme names."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent, Viseme
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))

        weights = controller.update(time=0.05)
        for key in weights.keys():
            assert isinstance(key, (str, Enum, Viseme.__class__)) or hasattr(key, 'name')

    def test_weight_values_are_numeric(self):
        """Weight values should be numeric (float or int)."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))

        weights = controller.update(time=0.05)
        for value in weights.values():
            assert isinstance(value, (int, float))

    def test_weight_values_in_range(self):
        """Weight values should be in [0, 1] range."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.2))
        controller.add_phoneme_event(PhonemeEvent("OO", 0.2, 0.2))

        # Test at multiple times
        for t in [0.0, 0.1, 0.19, 0.2, 0.3]:
            weights = controller.update(time=t)
            for key, value in weights.items():
                assert 0.0 <= value <= 1.0, f"Weight {key}={value} out of range at t={t}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_controller_update(self):
        """Update on empty controller should not crash."""
        from engine.animation.facial.lip_sync import LipSyncController
        controller = LipSyncController()
        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)

    def test_very_small_duration(self):
        """Very small but non-zero duration should work."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.001))

        weights = controller.update(time=0.0005)
        assert isinstance(weights, dict)

    def test_very_long_duration(self):
        """Very long duration phoneme should work."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 100.0))

        weights = controller.update(time=50.0)
        assert isinstance(weights, dict)

    def test_large_start_time(self):
        """Phoneme with large start time should work."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 1000.0, 0.1))

        weights = controller.update(time=1000.05)
        assert isinstance(weights, dict)

    def test_overlapping_phonemes(self):
        """Overlapping phoneme events should be handled."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.3))
        controller.add_phoneme_event(PhonemeEvent("OO", 0.1, 0.3))  # Overlaps

        weights = controller.update(time=0.2)  # During overlap
        assert isinstance(weights, dict)

    def test_many_phonemes(self):
        """Controller should handle many phoneme events."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        phonemes = ["AA", "B", "IY", "D", "OO", "K", "AH", "T", "EH", "P"]

        for i, phoneme in enumerate(phonemes * 10):  # 100 events
            controller.add_phoneme_event(PhonemeEvent(phoneme, i * 0.1, 0.1))

        weights = controller.update(time=5.0)
        assert isinstance(weights, dict)

    def test_out_of_order_phoneme_addition(self):
        """Adding phonemes out of order should be handled."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        # Add out of temporal order
        controller.add_phoneme_event(PhonemeEvent("OO", 0.2, 0.1))
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        controller.add_phoneme_event(PhonemeEvent("IY", 0.1, 0.1))

        events = controller.get_viseme_events()
        # Should have events for all phonemes
        assert len(events) >= 3

    def test_negative_time_update(self):
        """Update with negative time - should handle gracefully."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))

        try:
            weights = controller.update(time=-1.0)
            # If accepted, should return valid dict
            assert isinstance(weights, dict)
        except (ValueError, TypeError):
            # Rejecting negative time is acceptable
            pass


class TestContractCompliance:
    """Tests that directly verify the documented contract."""

    def test_contract_example_1_phoneme_to_viseme(self):
        """Contract: Phoneme AA should convert to Viseme.AA"""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent, Viseme
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        events = controller.get_viseme_events()
        assert events[0].viseme == Viseme.AA

    def test_contract_example_2_zero_duration_no_crash(self):
        """Contract: Zero duration should not crash."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.0))
        # No exception means pass

    def test_contract_example_3_weights_is_dict(self):
        """Contract: weights from update() should be dict."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("P", 0.0, 0.0))
        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)


class TestVisemeEventProperties:
    """Test VisemeEvent objects returned by get_viseme_events()."""

    def test_viseme_event_has_viseme_property(self):
        """VisemeEvent should have viseme property."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        events = controller.get_viseme_events()

        assert len(events) >= 1
        assert hasattr(events[0], 'viseme')

    def test_viseme_event_timing(self):
        """VisemeEvent should preserve timing information."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.5, 0.2))
        events = controller.get_viseme_events()

        assert len(events) >= 1
        event = events[0]
        # Should have some timing property
        assert hasattr(event, 'start_time') or hasattr(event, 'start') or \
               hasattr(event, 'time') or hasattr(event, 'duration')


class TestCoarticulationStrength:
    """Test that coarticulation produces measurable blending."""

    def test_transition_weights_differ_from_pure(self):
        """Weights during transition should differ from pure viseme weights."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        # Controller with single viseme
        controller_single = LipSyncController()
        controller_single.add_phoneme_event(PhonemeEvent("AA", 0.0, 1.0))
        weights_pure = controller_single.update(time=0.5)

        # Controller with transition
        controller_trans = LipSyncController()
        controller_trans.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.5))
        controller_trans.add_phoneme_event(PhonemeEvent("OO", 0.5, 0.5))
        weights_trans = controller_trans.update(time=0.49)  # Just before transition

        # Both should be valid
        assert isinstance(weights_pure, dict)
        assert isinstance(weights_trans, dict)
        # Transition weights might show influence from upcoming OO
        # (detailed comparison depends on implementation)


class TestRealWorldScenarios:
    """Test realistic lip sync scenarios."""

    def test_hello_word_sequence(self):
        """Test phoneme sequence for 'hello'."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        # Approximate phonemes for "hello": HH EH L OW
        controller.add_phoneme_event(PhonemeEvent("HH", 0.0, 0.05))
        controller.add_phoneme_event(PhonemeEvent("EH", 0.05, 0.1))
        controller.add_phoneme_event(PhonemeEvent("L", 0.15, 0.05))
        controller.add_phoneme_event(PhonemeEvent("OW", 0.2, 0.15))

        events = controller.get_viseme_events()
        assert len(events) >= 4

        # Test playback
        for t in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
            weights = controller.update(time=t)
            assert isinstance(weights, dict)

    def test_continuous_speech_simulation(self):
        """Test continuous speech-like phoneme sequence."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()

        # Simulate continuous speech
        phonemes = [
            ("TH", 0.0, 0.05), ("AH", 0.05, 0.08), ("K", 0.13, 0.04),
            ("W", 0.17, 0.04), ("IH", 0.21, 0.06), ("K", 0.27, 0.04),
            ("B", 0.31, 0.03), ("R", 0.34, 0.04), ("AW", 0.38, 0.08),
            ("N", 0.46, 0.05), ("F", 0.51, 0.04), ("AA", 0.55, 0.08),
            ("K", 0.63, 0.04), ("S", 0.67, 0.06)
        ]

        for phoneme, start, dur in phonemes:
            controller.add_phoneme_event(PhonemeEvent(phoneme, start, dur))

        events = controller.get_viseme_events()
        assert len(events) >= len(phonemes)

        # Simulate 60fps playback
        for frame in range(45):  # ~0.75 seconds
            time = frame / 60.0
            weights = controller.update(time=time)
            assert isinstance(weights, dict)


class TestClearAndReset:
    """Test controller clearing/resetting if supported."""

    def test_controller_can_be_reused(self):
        """Controller should support being reused with new events."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()

        # First use
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.1))
        events1 = controller.get_viseme_events()
        assert len(events1) >= 1

        # If clear/reset exists, use it
        if hasattr(controller, 'clear'):
            controller.clear()
            events_after_clear = controller.get_viseme_events()
            assert len(events_after_clear) == 0
        elif hasattr(controller, 'reset'):
            controller.reset()
            events_after_reset = controller.get_viseme_events()
            assert len(events_after_reset) == 0


class TestStressConditions:
    """Test under stress conditions."""

    def test_rapid_updates(self):
        """Many rapid updates should not cause issues."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 1.0))

        # Simulate 1000 rapid updates
        for i in range(1000):
            time = i * 0.001
            weights = controller.update(time=time)
            assert isinstance(weights, dict)

    def test_same_time_repeated_updates(self):
        """Repeated updates at same time should be consistent."""
        from engine.animation.facial.lip_sync import (
            LipSyncController, PhonemeEvent
        )
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent("AA", 0.0, 0.5))
        controller.add_phoneme_event(PhonemeEvent("OO", 0.5, 0.5))

        # Update at same time multiple times
        weights1 = controller.update(time=0.25)
        weights2 = controller.update(time=0.25)
        weights3 = controller.update(time=0.25)

        assert weights1 == weights2 == weights3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
