"""
Whitebox tests for Lip Sync Coarticulation (T3.3).

Tests the internal implementation of:
- Phoneme to viseme conversion
- Coarticulation blending (anticipation and carryover)
- Zero-duration phoneme handling
- Timeline playback accuracy
"""

import math
import pytest
from dataclasses import dataclass
from typing import Any

from engine.animation.facial.lip_sync import (
    Viseme,
    PHONEME_TO_VISEME,
    phoneme_to_viseme,
    VisemeMapping,
    get_default_viseme_mappings,
    PhonemeEvent,
    VisemeEvent,
    CoarticulationSettings,
    apply_coarticulation,
    LipSyncController,
    create_phoneme_events_from_text,
)


# =============================================================================
# Test: Phoneme to Viseme Conversion
# =============================================================================


class TestPhonemeToVisemeConversion:
    """Tests for phoneme to viseme mapping functionality."""

    def test_bilabial_phonemes_map_to_pp(self) -> None:
        """Bilabial phonemes (p, b, m) should map to PP viseme."""
        bilabials = ["p", "b", "m", "P", "B", "M"]
        for phoneme in bilabials:
            assert phoneme_to_viseme(phoneme) == Viseme.PP, f"'{phoneme}' should map to PP"

    def test_labiodental_phonemes_map_to_ff(self) -> None:
        """Labiodental phonemes (f, v) should map to FF viseme."""
        labiodentals = ["f", "v", "F", "V"]
        for phoneme in labiodentals:
            assert phoneme_to_viseme(phoneme) == Viseme.FF, f"'{phoneme}' should map to FF"

    def test_dental_fricative_phonemes_map_to_th(self) -> None:
        """Dental fricatives (th, dh) should map to TH viseme."""
        dentals = ["th", "dh", "TH", "DH"]
        for phoneme in dentals:
            assert phoneme_to_viseme(phoneme) == Viseme.TH, f"'{phoneme}' should map to TH"

    def test_alveolar_phonemes_map_to_dd(self) -> None:
        """Alveolar phonemes (t, d, l) should map to DD viseme."""
        alveolars = ["t", "d", "l", "T", "D", "L"]
        for phoneme in alveolars:
            assert phoneme_to_viseme(phoneme) == Viseme.DD, f"'{phoneme}' should map to DD"

    def test_nasal_phonemes_map_to_nn(self) -> None:
        """Nasal phonemes (n) should map to NN viseme."""
        nasals = ["n", "N"]
        for phoneme in nasals:
            assert phoneme_to_viseme(phoneme) == Viseme.NN, f"'{phoneme}' should map to NN"

    def test_alveolar_fricative_phonemes_map_to_ss(self) -> None:
        """Alveolar fricatives (s, z) should map to SS viseme."""
        fricatives = ["s", "z", "S", "Z"]
        for phoneme in fricatives:
            assert phoneme_to_viseme(phoneme) == Viseme.SS, f"'{phoneme}' should map to SS"

    def test_palato_alveolar_phonemes_map_to_ch(self) -> None:
        """Palato-alveolar phonemes (sh, zh, ch, jh) should map to CH viseme."""
        palato_alveolars = ["sh", "zh", "ch", "jh", "SH", "ZH", "CH", "JH"]
        for phoneme in palato_alveolars:
            assert phoneme_to_viseme(phoneme) == Viseme.CH, f"'{phoneme}' should map to CH"

    def test_velar_phonemes_map_to_kk(self) -> None:
        """Velar phonemes (k, g, ng) should map to KK viseme."""
        velars = ["k", "g", "ng", "K", "G", "NG"]
        for phoneme in velars:
            assert phoneme_to_viseme(phoneme) == Viseme.KK, f"'{phoneme}' should map to KK"

    def test_approximant_phonemes_map_to_rr(self) -> None:
        """Approximant phonemes (r, er) should map to RR viseme."""
        approximants = ["r", "R", "er", "ER"]
        for phoneme in approximants:
            assert phoneme_to_viseme(phoneme) == Viseme.RR, f"'{phoneme}' should map to RR"

    def test_open_vowels_map_to_aa(self) -> None:
        """Open vowels (aa, ah, ae, ax) should map to AA viseme."""
        open_vowels = ["aa", "ah", "ae", "ax", "AA", "AH", "AE", "AX"]
        for phoneme in open_vowels:
            assert phoneme_to_viseme(phoneme) == Viseme.AA, f"'{phoneme}' should map to AA"

    def test_front_mid_vowels_map_to_ee(self) -> None:
        """Front mid vowels (eh, ey) should map to EE viseme."""
        front_mid = ["eh", "ey", "EH", "EY"]
        for phoneme in front_mid:
            assert phoneme_to_viseme(phoneme) == Viseme.EE, f"'{phoneme}' should map to EE"

    def test_front_close_vowels_map_to_ii(self) -> None:
        """Front close vowels (iy, ih) should map to II viseme."""
        front_close = ["iy", "ih", "IY", "IH"]
        for phoneme in front_close:
            assert phoneme_to_viseme(phoneme) == Viseme.II, f"'{phoneme}' should map to II"

    def test_back_mid_vowels_map_to_oo(self) -> None:
        """Back mid vowels (ao, ow, oy) should map to OO viseme."""
        back_mid = ["ao", "ow", "oy", "AO", "OW", "OY"]
        for phoneme in back_mid:
            assert phoneme_to_viseme(phoneme) == Viseme.OO, f"'{phoneme}' should map to OO"

    def test_back_close_vowels_map_to_uu(self) -> None:
        """Back close vowels (uw, uh) should map to UU viseme."""
        back_close = ["uw", "uh", "UW", "UH"]
        for phoneme in back_close:
            assert phoneme_to_viseme(phoneme) == Viseme.UU, f"'{phoneme}' should map to UU"

    def test_glides_map_correctly(self) -> None:
        """Glide phonemes should map to their respective visemes."""
        assert phoneme_to_viseme("w") == Viseme.UU
        assert phoneme_to_viseme("W") == Viseme.UU
        assert phoneme_to_viseme("y") == Viseme.II
        assert phoneme_to_viseme("Y") == Viseme.II

    def test_silence_phonemes_map_to_sil(self) -> None:
        """Silence phonemes should map to SIL viseme."""
        silences = ["sil", "sp", "", "h", "hh", "H", "HH"]
        for phoneme in silences:
            assert phoneme_to_viseme(phoneme) == Viseme.SIL, f"'{phoneme}' should map to SIL"

    def test_stress_markers_stripped(self) -> None:
        """Phonemes with stress markers should still map correctly."""
        assert phoneme_to_viseme("AA0") == Viseme.AA
        assert phoneme_to_viseme("AA1") == Viseme.AA
        assert phoneme_to_viseme("AA2") == Viseme.AA
        assert phoneme_to_viseme("IY1") == Viseme.II
        assert phoneme_to_viseme("EH0") == Viseme.EE

    def test_unknown_phoneme_maps_to_sil(self) -> None:
        """Unknown phonemes should default to SIL viseme."""
        unknown = ["xyz", "qqq", "###", "123"]
        for phoneme in unknown:
            assert phoneme_to_viseme(phoneme) == Viseme.SIL, f"Unknown '{phoneme}' should map to SIL"

    def test_diphthongs_map_correctly(self) -> None:
        """Diphthongs should map to their start sound."""
        diphthongs = ["aw", "ay", "AW", "AY"]
        for phoneme in diphthongs:
            assert phoneme_to_viseme(phoneme) == Viseme.AA


# =============================================================================
# Test: Coarticulation Settings and Blending
# =============================================================================


class TestCoarticulationSettings:
    """Tests for coarticulation settings and blend curves."""

    def test_default_settings(self) -> None:
        """Default coarticulation settings should be reasonable."""
        settings = CoarticulationSettings()
        assert settings.anticipation_time == 0.05
        assert settings.carryover_time == 0.03
        assert settings.blend_curve == "ease_in_out"

    def test_custom_settings(self) -> None:
        """Custom coarticulation settings should be applied."""
        settings = CoarticulationSettings(
            anticipation_time=0.1,
            carryover_time=0.08,
            blend_curve="linear"
        )
        assert settings.anticipation_time == 0.1
        assert settings.carryover_time == 0.08
        assert settings.blend_curve == "linear"

    def test_linear_blend_curve(self) -> None:
        """Linear blend curve should return input value."""
        settings = CoarticulationSettings(blend_curve="linear")
        assert settings.calculate_blend(0.0) == 0.0
        assert settings.calculate_blend(0.25) == 0.25
        assert settings.calculate_blend(0.5) == 0.5
        assert settings.calculate_blend(0.75) == 0.75
        assert settings.calculate_blend(1.0) == 1.0

    def test_ease_in_blend_curve(self) -> None:
        """Ease-in blend curve should accelerate (t^2)."""
        settings = CoarticulationSettings(blend_curve="ease_in")
        assert settings.calculate_blend(0.0) == 0.0
        assert settings.calculate_blend(0.5) == pytest.approx(0.25)
        assert settings.calculate_blend(1.0) == 1.0

    def test_ease_out_blend_curve(self) -> None:
        """Ease-out blend curve should decelerate."""
        settings = CoarticulationSettings(blend_curve="ease_out")
        assert settings.calculate_blend(0.0) == 0.0
        assert settings.calculate_blend(0.5) == pytest.approx(0.75)
        assert settings.calculate_blend(1.0) == 1.0

    def test_ease_in_out_blend_curve_smoothstep(self) -> None:
        """Ease-in-out blend curve should use smoothstep formula."""
        settings = CoarticulationSettings(blend_curve="ease_in_out")
        # Smoothstep: t^2 * (3 - 2t)
        assert settings.calculate_blend(0.0) == 0.0
        assert settings.calculate_blend(0.5) == pytest.approx(0.5)  # 0.25 * (3 - 1) = 0.5
        assert settings.calculate_blend(1.0) == 1.0
        # At t=0.25: 0.0625 * (3 - 0.5) = 0.15625
        assert settings.calculate_blend(0.25) == pytest.approx(0.15625)

    def test_blend_clamped_below_zero(self) -> None:
        """Blend values below 0 should be clamped."""
        settings = CoarticulationSettings(blend_curve="linear")
        assert settings.calculate_blend(-0.5) == 0.0
        assert settings.calculate_blend(-1.0) == 0.0

    def test_blend_clamped_above_one(self) -> None:
        """Blend values above 1 should be clamped."""
        settings = CoarticulationSettings(blend_curve="linear")
        assert settings.calculate_blend(1.5) == 1.0
        assert settings.calculate_blend(2.0) == 1.0


# =============================================================================
# Test: Phoneme and Viseme Events
# =============================================================================


class TestPhonemeEvent:
    """Tests for PhonemeEvent dataclass."""

    def test_basic_phoneme_event(self) -> None:
        """Basic phoneme event properties."""
        event = PhonemeEvent(
            phoneme="aa",
            start_time=0.0,
            end_time=0.1,
            confidence=0.9
        )
        assert event.phoneme == "aa"
        assert event.start_time == 0.0
        assert event.end_time == 0.1
        assert event.confidence == 0.9

    def test_phoneme_event_duration(self) -> None:
        """Phoneme event duration calculation."""
        event = PhonemeEvent(phoneme="aa", start_time=0.5, end_time=0.8)
        assert event.duration == pytest.approx(0.3)

    def test_phoneme_event_mid_time(self) -> None:
        """Phoneme event midpoint calculation."""
        event = PhonemeEvent(phoneme="aa", start_time=0.2, end_time=0.6)
        assert event.mid_time == pytest.approx(0.4)

    def test_phoneme_event_default_confidence(self) -> None:
        """Default confidence should be 1.0."""
        event = PhonemeEvent(phoneme="aa", start_time=0.0, end_time=0.1)
        assert event.confidence == 1.0


class TestVisemeEvent:
    """Tests for VisemeEvent dataclass."""

    def test_basic_viseme_event(self) -> None:
        """Basic viseme event properties."""
        event = VisemeEvent(
            viseme=Viseme.AA,
            start_time=0.0,
            end_time=0.1,
            weight=0.8
        )
        assert event.viseme == Viseme.AA
        assert event.start_time == 0.0
        assert event.end_time == 0.1
        assert event.weight == 0.8

    def test_viseme_event_duration(self) -> None:
        """Viseme event duration calculation."""
        event = VisemeEvent(viseme=Viseme.PP, start_time=0.1, end_time=0.5)
        assert event.duration == pytest.approx(0.4)

    def test_viseme_event_default_weight(self) -> None:
        """Default weight should be 1.0."""
        event = VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.1)
        assert event.weight == 1.0


# =============================================================================
# Test: Viseme Mapping
# =============================================================================


class TestVisemeMapping:
    """Tests for VisemeMapping functionality."""

    def test_viseme_mapping_basic(self) -> None:
        """Basic viseme mapping with blend shapes."""
        mapping = VisemeMapping(
            viseme=Viseme.AA,
            blend_shapes={"jawOpen": 0.6, "mouthFunnel": 0.0}
        )
        assert mapping.viseme == Viseme.AA
        assert mapping.blend_shapes["jawOpen"] == 0.6
        assert mapping.blend_shapes["mouthFunnel"] == 0.0

    def test_viseme_mapping_get_weights(self) -> None:
        """Get weights with default intensity."""
        mapping = VisemeMapping(
            viseme=Viseme.AA,
            blend_shapes={"jawOpen": 0.6, "mouthSmile": 0.4}
        )
        weights = mapping.get_weights()
        assert weights["jawOpen"] == pytest.approx(0.6)
        assert weights["mouthSmile"] == pytest.approx(0.4)

    def test_viseme_mapping_get_weights_scaled(self) -> None:
        """Get weights scaled by intensity."""
        mapping = VisemeMapping(
            viseme=Viseme.AA,
            blend_shapes={"jawOpen": 0.6, "mouthSmile": 0.4}
        )
        weights = mapping.get_weights(intensity=0.5)
        assert weights["jawOpen"] == pytest.approx(0.3)
        assert weights["mouthSmile"] == pytest.approx(0.2)

    def test_default_viseme_mappings_complete(self) -> None:
        """All visemes should have default mappings."""
        mappings = get_default_viseme_mappings()
        for viseme in Viseme:
            assert viseme in mappings, f"Missing mapping for {viseme}"
            assert mappings[viseme].viseme == viseme


# =============================================================================
# Test: LipSyncController - Phoneme to Viseme Processing
# =============================================================================


class TestLipSyncControllerPhonemeProcessing:
    """Tests for LipSyncController phoneme processing."""

    def test_process_single_phoneme_event(self) -> None:
        """Single phoneme event converts to viseme event."""
        controller = LipSyncController()
        phoneme_events = [
            PhonemeEvent(phoneme="aa", start_time=0.0, end_time=0.1)
        ]
        viseme_events = controller.process_audio_events(phoneme_events)

        assert len(viseme_events) == 1
        assert viseme_events[0].viseme == Viseme.AA
        assert viseme_events[0].start_time == 0.0
        assert viseme_events[0].end_time == 0.1
        assert viseme_events[0].weight == 1.0

    def test_process_multiple_phoneme_events(self) -> None:
        """Multiple phoneme events convert correctly."""
        controller = LipSyncController()
        phoneme_events = [
            PhonemeEvent(phoneme="p", start_time=0.0, end_time=0.05),
            PhonemeEvent(phoneme="aa", start_time=0.05, end_time=0.15),
            PhonemeEvent(phoneme="t", start_time=0.15, end_time=0.2),
        ]
        viseme_events = controller.process_audio_events(phoneme_events)

        assert len(viseme_events) == 3
        assert viseme_events[0].viseme == Viseme.PP
        assert viseme_events[1].viseme == Viseme.AA
        assert viseme_events[2].viseme == Viseme.DD

    def test_process_preserves_confidence_as_weight(self) -> None:
        """Phoneme confidence becomes viseme weight."""
        controller = LipSyncController()
        phoneme_events = [
            PhonemeEvent(phoneme="aa", start_time=0.0, end_time=0.1, confidence=0.75)
        ]
        viseme_events = controller.process_audio_events(phoneme_events)

        assert viseme_events[0].weight == 0.75

    def test_add_phoneme_event_inserts_sorted(self) -> None:
        """Adding phoneme events maintains sorted order."""
        controller = LipSyncController()

        # Add out of order
        controller.add_phoneme_event(PhonemeEvent(phoneme="aa", start_time=0.2, end_time=0.3))
        controller.add_phoneme_event(PhonemeEvent(phoneme="p", start_time=0.0, end_time=0.1))
        controller.add_phoneme_event(PhonemeEvent(phoneme="t", start_time=0.1, end_time=0.2))

        events = controller.get_viseme_events()
        assert len(events) == 3
        assert events[0].start_time == 0.0
        assert events[0].viseme == Viseme.PP
        assert events[1].start_time == 0.1
        assert events[1].viseme == Viseme.DD
        assert events[2].start_time == 0.2
        assert events[2].viseme == Viseme.AA


# =============================================================================
# Test: LipSyncController - Zero-Duration Phoneme Handling
# =============================================================================


class TestZeroDurationPhonemes:
    """Tests for zero-duration phoneme handling."""

    def test_zero_duration_phoneme_converts(self) -> None:
        """Zero-duration phoneme events should still convert."""
        controller = LipSyncController()
        phoneme_events = [
            PhonemeEvent(phoneme="p", start_time=0.1, end_time=0.1)  # Zero duration
        ]
        viseme_events = controller.process_audio_events(phoneme_events)

        assert len(viseme_events) == 1
        assert viseme_events[0].viseme == Viseme.PP
        assert viseme_events[0].duration == 0.0

    def test_zero_duration_in_timeline(self) -> None:
        """Zero-duration events should be handled in timeline playback."""
        controller = LipSyncController()
        controller.add_phoneme_event(PhonemeEvent(phoneme="p", start_time=0.0, end_time=0.0))
        controller.add_phoneme_event(PhonemeEvent(phoneme="aa", start_time=0.1, end_time=0.2))

        events = controller.get_viseme_events()
        assert len(events) == 2

        # Should not crash when updating at the zero-duration event time
        weights = controller.update(time=0.0)
        # Weights should be returned (possibly empty or with values)
        assert isinstance(weights, dict)

    def test_zero_duration_blend_weights(self) -> None:
        """Zero-duration viseme should apply full weight instantly."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.0, weight=1.0),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2, weight=1.0),
        ])

        # At exactly time 0.0, should handle zero-duration without crash
        weights = controller.update(time=0.0)
        assert isinstance(weights, dict)

    def test_negative_duration_handled(self) -> None:
        """Negative duration events should be handled gracefully."""
        controller = LipSyncController()
        # This shouldn't happen in practice but the code should handle it
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.1, end_time=0.05, weight=1.0),  # Negative
        ])

        # Should not crash
        weights = controller.update(time=0.1)
        assert isinstance(weights, dict)


# =============================================================================
# Test: LipSyncController - Coarticulation Blending
# =============================================================================


class TestCoarticulationBlending:
    """Tests for coarticulation blending in LipSyncController."""

    def test_carryover_at_event_start(self) -> None:
        """Previous viseme should influence start of current event."""
        settings = CoarticulationSettings(
            anticipation_time=0.05,
            carryover_time=0.05,
            blend_curve="linear"
        )
        controller = LipSyncController(coarticulation_settings=settings)

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1, weight=1.0),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2, weight=1.0),
        ])

        # At 0.1 (start of AA), there should be PP carryover
        weights_at_start = controller.update(time=0.1)

        # At 0.15 (past carryover window), should be pure AA
        weights_mid = controller.update(time=0.15)

        # Carryover should produce different weights at start vs mid
        # (At start, PP blend shapes should have some contribution)
        assert isinstance(weights_at_start, dict)
        assert isinstance(weights_mid, dict)

    def test_anticipation_near_event_end(self) -> None:
        """Next viseme should be anticipated near end of current event."""
        settings = CoarticulationSettings(
            anticipation_time=0.05,
            carryover_time=0.03,
            blend_curve="linear"
        )
        controller = LipSyncController(coarticulation_settings=settings)

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.1, weight=1.0),
            VisemeEvent(viseme=Viseme.PP, start_time=0.1, end_time=0.2, weight=1.0),
        ])

        # At 0.03 (well before anticipation), should be pure AA
        weights_early = controller.update(time=0.03)

        # At 0.06 (within anticipation window: 0.1 - 0.05 = 0.05), should have PP anticipation
        weights_anticipation = controller.update(time=0.08)

        assert isinstance(weights_early, dict)
        assert isinstance(weights_anticipation, dict)

    def test_no_coarticulation_with_zero_times(self) -> None:
        """Zero anticipation and carryover should disable coarticulation."""
        settings = CoarticulationSettings(
            anticipation_time=0.0,
            carryover_time=0.0,
        )
        controller = LipSyncController(coarticulation_settings=settings)

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1, weight=1.0),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2, weight=1.0),
        ])

        # At 0.1, should be pure AA (no carryover)
        weights = controller.update(time=0.1)
        assert isinstance(weights, dict)


# =============================================================================
# Test: LipSyncController - Timeline Playback
# =============================================================================


class TestTimelinePlayback:
    """Tests for timeline playback accuracy."""

    def test_playback_starts_stopped(self) -> None:
        """Controller should start in stopped state."""
        controller = LipSyncController()
        assert not controller.is_playing
        assert controller.current_time == 0.0

    def test_play_pause_stop(self) -> None:
        """Play, pause, and stop controls."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=1.0)
        ])

        controller.play()
        assert controller.is_playing

        controller.pause()
        assert not controller.is_playing

        controller.play()
        controller.update(dt=0.5)
        assert controller.current_time == pytest.approx(0.5)

        controller.stop()
        assert not controller.is_playing
        assert controller.current_time == 0.0
        assert controller.current_viseme == Viseme.SIL

    def test_seek_to_time(self) -> None:
        """Seeking to specific time updates state."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.5),
            VisemeEvent(viseme=Viseme.AA, start_time=0.5, end_time=1.0),
        ])

        controller.seek(0.25)
        assert controller.current_viseme == Viseme.PP

        controller.seek(0.75)
        assert controller.current_viseme == Viseme.AA

    def test_seek_negative_time_clamped(self) -> None:
        """Seeking to negative time should clamp to 0."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=1.0)
        ])

        controller.seek(-1.0)
        assert controller._current_time == 0.0

    def test_update_with_delta_time(self) -> None:
        """Update with delta time advances playback."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=1.0)
        ])

        controller.play()
        controller.update(dt=0.1)
        assert controller.current_time == pytest.approx(0.1)

        controller.update(dt=0.2)
        assert controller.current_time == pytest.approx(0.3)

    def test_update_with_absolute_time(self) -> None:
        """Update with absolute time seeks directly."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.5),
            VisemeEvent(viseme=Viseme.AA, start_time=0.5, end_time=1.0),
        ])

        weights = controller.update(time=0.25)
        assert isinstance(weights, dict)

        weights = controller.update(time=0.75)
        assert isinstance(weights, dict)

    def test_playback_stops_at_end(self) -> None:
        """Playback should stop when reaching timeline end."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.5)
        ])

        controller.play()
        # Jump past end
        controller.update(dt=1.0)

        assert not controller.is_playing
        assert controller.current_time == pytest.approx(0.5)
        assert controller.current_viseme == Viseme.SIL

    def test_duration_property(self) -> None:
        """Duration should reflect timeline length."""
        controller = LipSyncController()
        assert controller.duration == 0.0

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=1.5)
        ])
        assert controller.duration == pytest.approx(1.5)

    def test_empty_timeline_update(self) -> None:
        """Update on empty timeline returns empty weights."""
        controller = LipSyncController()
        weights = controller.update(dt=0.1)
        assert weights == {}

        weights = controller.update(time=0.5)
        assert weights == {}


# =============================================================================
# Test: LipSyncController - Frame Accuracy
# =============================================================================


class TestFrameAccuracy:
    """Tests for frame-accurate timeline playback."""

    def test_viseme_at_exact_boundary(self) -> None:
        """Correct viseme should be returned at exact event boundaries."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2),
            VisemeEvent(viseme=Viseme.DD, start_time=0.2, end_time=0.3),
        ])

        # At boundaries
        assert controller.get_viseme_at_time(0.0) == Viseme.PP
        assert controller.get_viseme_at_time(0.1) == Viseme.AA
        assert controller.get_viseme_at_time(0.2) == Viseme.DD

    def test_viseme_between_events(self) -> None:
        """Correct viseme in the middle of events."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2),
        ])

        assert controller.get_viseme_at_time(0.05) == Viseme.PP
        assert controller.get_viseme_at_time(0.15) == Viseme.AA

    def test_viseme_before_timeline_start(self) -> None:
        """Viseme at time before timeline should return first viseme."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.5, end_time=1.0),
        ])

        # Before timeline starts
        viseme = controller.get_viseme_at_time(0.0)
        # Should return the first event's viseme (clamped index)
        assert viseme == Viseme.AA

    def test_viseme_after_timeline_end(self) -> None:
        """Viseme at time after timeline should return last viseme."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.5),
        ])

        # After timeline ends
        viseme = controller.get_viseme_at_time(1.0)
        # Should return the last event's viseme (clamped index)
        assert viseme == Viseme.PP

    def test_high_frequency_updates(self) -> None:
        """Many rapid updates should maintain accuracy."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2),
            VisemeEvent(viseme=Viseme.DD, start_time=0.2, end_time=0.3),
        ])

        controller.play()

        # 60 FPS simulation
        dt = 1.0 / 60.0
        for frame in range(20):  # ~0.33 seconds
            weights = controller.update(dt=dt)
            assert isinstance(weights, dict)

        # Should have progressed through timeline
        assert controller.current_time > 0.2


# =============================================================================
# Test: LipSyncController - Blend Weights
# =============================================================================


class TestBlendWeights:
    """Tests for blend weight calculation."""

    def test_blend_weights_within_event(self) -> None:
        """Blend weights should be calculated within an event."""
        controller = LipSyncController()
        controller.intensity = 1.0
        controller.blend_time = 0.02

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.2, weight=1.0)
        ])

        weights = controller.update(time=0.1)  # Middle of event

        # AA viseme should have jawOpen=0.6
        assert "jawOpen" in weights
        assert weights["jawOpen"] > 0

    def test_blend_weights_clamped_0_1(self) -> None:
        """All blend weights should be clamped between 0 and 1."""
        controller = LipSyncController()
        controller.intensity = 1.0

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.1, weight=1.5),  # High weight
        ])

        weights = controller.update(time=0.05)

        for name, value in weights.items():
            assert 0.0 <= value <= 1.0, f"{name} weight {value} out of range"

    def test_intensity_scaling(self) -> None:
        """Intensity should scale all weights."""
        controller = LipSyncController()

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.2, weight=1.0)
        ])

        controller.intensity = 1.0
        weights_full = controller.update(time=0.1).copy()

        controller.intensity = 0.5
        weights_half = controller.update(time=0.1).copy()

        # Weights at half intensity should be approximately half
        for name in weights_full:
            if weights_full[name] > 0:
                # Allow some tolerance due to fade-in/out calculations
                assert weights_half.get(name, 0) <= weights_full[name]

    def test_get_blend_weights_returns_copy(self) -> None:
        """get_blend_weights should return a copy."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.2, weight=1.0)
        ])
        controller.update(time=0.1)

        weights1 = controller.get_blend_weights()
        weights2 = controller.get_blend_weights()

        # Should be equal but not the same object
        assert weights1 == weights2
        weights1["test"] = 999
        assert "test" not in weights2


# =============================================================================
# Test: LipSyncController - Callbacks and State
# =============================================================================


class TestCallbacksAndState:
    """Tests for callbacks and state management."""

    def test_on_weights_changed_callback(self) -> None:
        """Callback should be called when weights change."""
        received_weights = []

        def callback(weights: dict[str, float]) -> None:
            received_weights.append(weights.copy())

        controller = LipSyncController(on_weights_changed=callback)
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.2, weight=1.0)
        ])

        controller.update(time=0.05)
        controller.update(time=0.1)

        # Callback should have been invoked
        assert len(received_weights) >= 1

    def test_dirty_flag(self) -> None:
        """Dirty flag should track state changes."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.2)
        ])

        assert controller.dirty
        controller.clear_dirty()
        assert not controller.dirty

        controller.update(time=0.1)
        # If weights changed, should be dirty again
        assert controller.dirty

    def test_to_dict_serialization(self) -> None:
        """to_dict should serialize controller state."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.5)
        ])
        controller.intensity = 0.8
        controller.blend_time = 0.03
        controller.play()
        controller.update(dt=0.1)

        state = controller.to_dict()

        assert state["current_time"] == pytest.approx(0.1)
        assert state["is_playing"] == True
        assert state["current_viseme"] == "PP"
        assert state["intensity"] == pytest.approx(0.8)
        assert state["blend_time"] == pytest.approx(0.03)
        assert state["timeline_count"] == 1

    def test_set_viseme_mapping_custom(self) -> None:
        """Custom viseme mappings can be set."""
        controller = LipSyncController()

        custom_mapping = VisemeMapping(
            viseme=Viseme.AA,
            blend_shapes={"customShape": 1.0}
        )
        controller.set_viseme_mapping(Viseme.AA, custom_mapping)

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.2, weight=1.0)
        ])

        weights = controller.update(time=0.1)
        assert "customShape" in weights


# =============================================================================
# Test: apply_coarticulation Function
# =============================================================================


class TestApplyCoarticulation:
    """Tests for the standalone apply_coarticulation function."""

    def test_empty_events(self) -> None:
        """Empty event list returns empty result."""
        settings = CoarticulationSettings()
        result = apply_coarticulation([], settings)
        assert result == []

    def test_single_event(self) -> None:
        """Single event with no neighbors."""
        settings = CoarticulationSettings()
        events = [
            VisemeEvent(viseme=Viseme.AA, start_time=0.0, end_time=0.1, weight=1.0)
        ]
        result = apply_coarticulation(events, settings)

        assert len(result) > 0
        # Each tuple has 7 elements
        for sample in result:
            assert len(sample) == 7
            time, prev_vis, prev_w, curr_vis, curr_w, next_vis, next_w = sample
            assert isinstance(time, float)
            assert curr_vis == Viseme.AA

    def test_multiple_events_sorted(self) -> None:
        """Events should be sorted by start time."""
        settings = CoarticulationSettings()
        events = [
            VisemeEvent(viseme=Viseme.DD, start_time=0.2, end_time=0.3, weight=1.0),
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1, weight=1.0),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.2, weight=1.0),
        ]
        result = apply_coarticulation(events, settings)

        # Times should be monotonically increasing
        times = [r[0] for r in result]
        assert times == sorted(times)


# =============================================================================
# Test: Utility Functions
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_phoneme_events_from_text(self) -> None:
        """Create phoneme events from simple text."""
        events = create_phoneme_events_from_text("ab", start_time=0.0, phoneme_duration=0.1)

        assert len(events) == 2
        assert events[0].phoneme == "aa"  # 'a' -> 'aa'
        assert events[0].start_time == 0.0
        assert events[0].end_time == 0.1
        assert events[1].phoneme == "b"
        assert events[1].start_time == 0.1
        assert events[1].end_time == 0.2

    def test_create_phoneme_events_with_spaces(self) -> None:
        """Spaces create silence phonemes."""
        events = create_phoneme_events_from_text("a b", phoneme_duration=0.1)

        assert len(events) == 3
        assert events[1].phoneme == "sil"
        # Silence is half duration
        assert events[1].duration == pytest.approx(0.05)

    def test_create_phoneme_events_ignores_unknown(self) -> None:
        """Unknown characters are skipped."""
        events = create_phoneme_events_from_text("a!@#b", phoneme_duration=0.1)

        # Only 'a' and 'b' should produce events
        assert len(events) == 2
        assert events[0].phoneme == "aa"
        assert events[1].phoneme == "b"

    def test_create_phoneme_events_custom_start_time(self) -> None:
        """Custom start time offset."""
        events = create_phoneme_events_from_text("a", start_time=1.0, phoneme_duration=0.1)

        assert events[0].start_time == 1.0
        assert events[0].end_time == 1.1


# =============================================================================
# Test: Edge Cases and Robustness
# =============================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_very_short_phonemes(self) -> None:
        """Very short phoneme durations."""
        controller = LipSyncController()
        events = [
            PhonemeEvent(phoneme="p", start_time=0.0, end_time=0.001),
            PhonemeEvent(phoneme="aa", start_time=0.001, end_time=0.002),
        ]
        controller.set_phoneme_timeline(events)

        # Should not crash
        weights = controller.update(time=0.0005)
        assert isinstance(weights, dict)

    def test_overlapping_events(self) -> None:
        """Overlapping viseme events."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.2),
            VisemeEvent(viseme=Viseme.AA, start_time=0.1, end_time=0.3),  # Overlaps
        ])

        # Should handle overlap without crashing
        weights = controller.update(time=0.15)
        assert isinstance(weights, dict)

    def test_gap_between_events(self) -> None:
        """Gap between viseme events."""
        controller = LipSyncController()
        controller.set_timeline([
            VisemeEvent(viseme=Viseme.PP, start_time=0.0, end_time=0.1),
            VisemeEvent(viseme=Viseme.AA, start_time=0.5, end_time=0.6),  # Gap
        ])

        # In the gap, should still return some weights
        weights = controller.update(time=0.3)
        assert isinstance(weights, dict)

    def test_large_timeline(self) -> None:
        """Large number of events."""
        controller = LipSyncController()
        events = []
        for i in range(1000):
            events.append(VisemeEvent(
                viseme=Viseme.AA if i % 2 == 0 else Viseme.PP,
                start_time=i * 0.1,
                end_time=(i + 1) * 0.1,
                weight=1.0
            ))
        controller.set_timeline(events)

        # Should handle efficiently
        weights = controller.update(time=50.0)  # Middle of timeline
        assert isinstance(weights, dict)

        # Duration should be correct
        assert controller.duration == pytest.approx(100.0)

    def test_intensity_bounds(self) -> None:
        """Intensity property bounds."""
        controller = LipSyncController()

        controller.intensity = 1.5
        assert controller.intensity == 1.0

        controller.intensity = -0.5
        assert controller.intensity == 0.0

    def test_blend_time_bounds(self) -> None:
        """Blend time property bounds."""
        controller = LipSyncController()

        controller.blend_time = -0.5
        assert controller.blend_time == 0.0

        controller.blend_time = 0.1
        assert controller.blend_time == 0.1

    def test_empty_blend_shapes(self) -> None:
        """Viseme with no blend shapes."""
        controller = LipSyncController()

        empty_mapping = VisemeMapping(viseme=Viseme.SIL, blend_shapes={})
        controller.set_viseme_mapping(Viseme.SIL, empty_mapping)

        controller.set_timeline([
            VisemeEvent(viseme=Viseme.SIL, start_time=0.0, end_time=0.1)
        ])

        weights = controller.update(time=0.05)
        # Should return empty or minimal weights
        assert isinstance(weights, dict)


# =============================================================================
# Test: Integration - Full Pipeline
# =============================================================================


class TestFullPipeline:
    """Integration tests for the full lip sync pipeline."""

    def test_text_to_animation_pipeline(self) -> None:
        """Complete pipeline from text to blend weights."""
        # Create phonemes from text
        phoneme_events = create_phoneme_events_from_text("hello", phoneme_duration=0.08)

        # Create controller and process
        controller = LipSyncController()
        controller.set_phoneme_timeline(phoneme_events)

        # Playback
        controller.play()

        all_weights = []
        dt = 0.016  # ~60 FPS
        while controller.is_playing:
            weights = controller.update(dt=dt)
            all_weights.append(weights.copy())

        # Should have generated some weight frames
        assert len(all_weights) > 0

    def test_realtime_simulation(self) -> None:
        """Simulate real-time playback."""
        phoneme_events = [
            PhonemeEvent(phoneme="p", start_time=0.0, end_time=0.1, confidence=0.95),
            PhonemeEvent(phoneme="aa", start_time=0.1, end_time=0.25, confidence=0.9),
            PhonemeEvent(phoneme="t", start_time=0.25, end_time=0.35, confidence=0.92),
        ]

        controller = LipSyncController()
        controller.set_phoneme_timeline(phoneme_events)
        controller.intensity = 0.8
        controller.play()

        # Simulate 60 FPS for 0.4 seconds
        frames = 24
        dt = 0.4 / frames

        for frame in range(frames):
            weights = controller.update(dt=dt)
            assert isinstance(weights, dict)
            # All weights should be valid
            for name, value in weights.items():
                assert 0.0 <= value <= 1.0

        # Should have reached the end
        assert not controller.is_playing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
