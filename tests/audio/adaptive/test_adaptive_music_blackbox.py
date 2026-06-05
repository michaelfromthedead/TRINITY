"""Blackbox tests for adaptive_music.py -- VerticalRemixer and HorizontalSequencer.

BLACKBOX coverage plan:
  - VerticalRemixer initialization with stem player
  - VerticalRemixer.set_intensity with valid range [0.0, 1.0]
  - VerticalRemixer.set_intensity clamping at boundaries
  - VerticalRemixer.get_intensity returns current intensity
  - VerticalRemixer intensity level activation
  - VerticalRemixer.add_intensity_level adds custom levels
  - VerticalRemixer.update smooths intensity changes
  - HorizontalSequencer initialization
  - HorizontalSequencer.add_section registers new sections
  - HorizontalSequencer section transitions
  - AdaptiveParameters parameter access
  - MusicSection properties
  - IntensityLevel configuration
  - AdaptiveMode enum values

Total: 25+ tests
"""

from __future__ import annotations

import pytest
from typing import List, Optional
from unittest.mock import MagicMock, patch


class TestAdaptiveMode:
    """Tests for AdaptiveMode enumeration."""

    def test_none_mode_exists(self):
        """AdaptiveMode should have NONE mode."""
        from engine.audio.adaptive.adaptive_music import AdaptiveMode

        assert hasattr(AdaptiveMode, 'NONE')

    def test_vertical_mode_exists(self):
        """AdaptiveMode should have VERTICAL mode."""
        from engine.audio.adaptive.adaptive_music import AdaptiveMode

        assert hasattr(AdaptiveMode, 'VERTICAL')

    def test_horizontal_mode_exists(self):
        """AdaptiveMode should have HORIZONTAL mode."""
        from engine.audio.adaptive.adaptive_music import AdaptiveMode

        assert hasattr(AdaptiveMode, 'HORIZONTAL')

    def test_combined_mode_exists(self):
        """AdaptiveMode should have COMBINED mode."""
        from engine.audio.adaptive.adaptive_music import AdaptiveMode

        assert hasattr(AdaptiveMode, 'COMBINED')


class TestBranchType:
    """Tests for BranchType enumeration."""

    def test_sequential_type_exists(self):
        """BranchType should have SEQUENTIAL type."""
        from engine.audio.adaptive.adaptive_music import BranchType

        assert hasattr(BranchType, 'SEQUENTIAL')

    def test_random_type_exists(self):
        """BranchType should have RANDOM type."""
        from engine.audio.adaptive.adaptive_music import BranchType

        assert hasattr(BranchType, 'RANDOM')

    def test_rule_based_type_exists(self):
        """BranchType should have RULE_BASED type."""
        from engine.audio.adaptive.adaptive_music import BranchType

        assert hasattr(BranchType, 'RULE_BASED')

    def test_weighted_type_exists(self):
        """BranchType should have WEIGHTED type."""
        from engine.audio.adaptive.adaptive_music import BranchType

        assert hasattr(BranchType, 'WEIGHTED')


class TestMusicSection:
    """Tests for MusicSection dataclass."""

    def test_create_music_section(self):
        """Should create MusicSection with required fields."""
        from engine.audio.adaptive.adaptive_music import MusicSection

        section = MusicSection(
            section_id="intro",
            name="Introduction",
            start_bar=0,
            end_bar=8
        )

        assert section.section_id == "intro"
        assert section.name == "Introduction"
        assert section.start_bar == 0
        assert section.end_bar == 8

    def test_music_section_length_bars(self):
        """MusicSection should calculate length_bars correctly."""
        from engine.audio.adaptive.adaptive_music import MusicSection

        section = MusicSection(
            section_id="verse",
            name="Verse",
            start_bar=8,
            end_bar=24
        )

        assert section.length_bars == 16

    def test_music_section_defaults(self):
        """MusicSection should have sensible defaults."""
        from engine.audio.adaptive.adaptive_music import MusicSection

        section = MusicSection(
            section_id="test",
            name="Test",
            start_bar=0,
            end_bar=4
        )

        assert section.can_loop is True
        assert section.loop_count == 0
        assert section.next_sections == []
        assert section.intensity_range == (0.0, 1.0)

    def test_music_section_with_next_sections(self):
        """MusicSection should accept next sections list."""
        from engine.audio.adaptive.adaptive_music import MusicSection

        section = MusicSection(
            section_id="verse",
            name="Verse",
            start_bar=8,
            end_bar=24,
            next_sections=["chorus", "bridge"]
        )

        assert "chorus" in section.next_sections
        assert "bridge" in section.next_sections


class TestIntensityLevel:
    """Tests for IntensityLevel dataclass."""

    def test_create_intensity_level(self):
        """Should create IntensityLevel with required fields."""
        from engine.audio.adaptive.adaptive_music import IntensityLevel

        level = IntensityLevel(
            level_id="high",
            threshold=0.7,
            layers={"drums": 1.0, "bass": 0.8}
        )

        assert level.level_id == "high"
        assert level.threshold == 0.7
        assert level.layers["drums"] == 1.0

    def test_intensity_level_defaults(self):
        """IntensityLevel should have sensible defaults."""
        from engine.audio.adaptive.adaptive_music import IntensityLevel

        level = IntensityLevel(
            level_id="test",
            threshold=0.5,
            layers={}
        )

        assert level.sections == []
        assert level.name == ""


class TestAdaptiveParameters:
    """Tests for AdaptiveParameters dataclass."""

    def test_create_parameters(self):
        """Should create AdaptiveParameters with defaults."""
        from engine.audio.adaptive.adaptive_music import AdaptiveParameters

        params = AdaptiveParameters()

        assert params.intensity == 0.5
        assert params.danger == 0.0
        assert params.tension == 0.0
        assert params.energy == 0.5

    def test_get_parameter(self):
        """get should return parameter values."""
        from engine.audio.adaptive.adaptive_music import AdaptiveParameters
        from engine.audio.adaptive.config import PARAM_INTENSITY, PARAM_DANGER

        params = AdaptiveParameters(intensity=0.8, danger=0.5)

        assert params.get(PARAM_INTENSITY) == 0.8
        assert params.get(PARAM_DANGER) == 0.5

    def test_set_parameter(self):
        """set should update parameter values."""
        from engine.audio.adaptive.adaptive_music import AdaptiveParameters
        from engine.audio.adaptive.config import PARAM_INTENSITY

        params = AdaptiveParameters()
        params.set(PARAM_INTENSITY, 0.9)

        assert params.intensity == 0.9

    def test_set_clamps_values(self):
        """set should clamp values to [0, 1]."""
        from engine.audio.adaptive.adaptive_music import AdaptiveParameters
        from engine.audio.adaptive.config import PARAM_INTENSITY

        params = AdaptiveParameters()

        params.set(PARAM_INTENSITY, 1.5)
        assert params.intensity <= 1.0

        params.set(PARAM_INTENSITY, -0.5)
        assert params.intensity >= 0.0

    def test_custom_parameters(self):
        """Should support custom parameters."""
        from engine.audio.adaptive.adaptive_music import AdaptiveParameters

        params = AdaptiveParameters()
        params.set("custom_param", 0.7)

        assert params.get("custom_param") == 0.7

    def test_get_default_for_unknown(self):
        """get should return default for unknown parameters."""
        from engine.audio.adaptive.adaptive_music import AdaptiveParameters

        params = AdaptiveParameters()

        assert params.get("nonexistent") == 0.0
        assert params.get("nonexistent", 0.5) == 0.5


class TestVerticalRemixerInitialization:
    """Tests for VerticalRemixer construction."""

    def test_initialization_with_stem_player(self):
        """VerticalRemixer should initialize with stem player."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        assert remixer is not None
        assert remixer.get_intensity() == 0.5  # Default intensity

    def test_initialization_with_smoothing(self):
        """VerticalRemixer should accept smoothing parameter."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player, smoothing=0.5)

        assert remixer is not None


class TestVerticalRemixerIntensity:
    """Tests for VerticalRemixer intensity control."""

    def test_set_intensity_valid_range(self):
        """set_intensity should accept values in [0.0, 1.0]."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
            remixer.set_intensity(intensity, immediate=True)
            assert abs(remixer.get_intensity() - intensity) < 0.001

    def test_set_intensity_clamps_below_zero(self):
        """set_intensity should clamp negative values."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        remixer.set_intensity(-0.5, immediate=True)
        assert remixer.get_intensity() >= 0.0

    def test_set_intensity_clamps_above_one(self):
        """set_intensity should clamp values above 1."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        remixer.set_intensity(1.5, immediate=True)
        assert remixer.get_intensity() <= 1.0

    def test_immediate_intensity_change(self):
        """immediate=True should skip smoothing."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        remixer.set_intensity(0.8, immediate=True)
        assert abs(remixer.get_intensity() - 0.8) < 0.001


class TestVerticalRemixerIntensityLevels:
    """Tests for VerticalRemixer intensity levels."""

    def test_add_intensity_level(self):
        """add_intensity_level should add custom level."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer, IntensityLevel

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        level = IntensityLevel(
            level_id="custom",
            threshold=0.6,
            layers={"drums": 1.0},
            name="Custom Level"
        )
        remixer.add_intensity_level(level)

        # Verify level was registered by checking it exists via _intensity_levels
        assert "custom" in remixer._intensity_levels
        assert remixer._intensity_levels["custom"].threshold == 0.6

    def test_remove_intensity_level(self):
        """remove_intensity_level should remove level."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer, IntensityLevel

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        level = IntensityLevel(
            level_id="removable",
            threshold=0.5,
            layers={}
        )
        remixer.add_intensity_level(level)
        result = remixer.remove_intensity_level("removable")

        assert result is True

    def test_remove_nonexistent_level(self):
        """remove_intensity_level should return False for missing level."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        result = remixer.remove_intensity_level("nonexistent")
        assert result is False

    def test_default_levels_exist(self):
        """Default intensity levels should be set up."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        # Default levels should exist
        # Test by setting intensity and checking stem_player gets called
        remixer.set_intensity(0.9, immediate=True)
        mock_player.set_blend.assert_called()


class TestVerticalRemixerUpdate:
    """Tests for VerticalRemixer update behavior."""

    def test_update_smooths_intensity(self):
        """update should smooth intensity changes."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        # Set target without immediate
        remixer.set_intensity(1.0)
        initial = remixer.get_intensity()

        # Update should move toward target
        remixer.update(0.1)

        # Should have moved closer to target
        assert remixer.get_intensity() != initial or remixer.get_intensity() == 1.0

    def test_update_applies_level_changes(self):
        """update should apply intensity level changes."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        remixer.set_intensity(0.9, immediate=True)
        remixer.update(0.016)

        # Stem player should have been called
        assert mock_player.set_blend.called


class TestHorizontalSequencerInitialization:
    """Tests for HorizontalSequencer construction."""

    def test_initialization(self):
        """HorizontalSequencer should initialize with clock and callback manager."""
        from engine.audio.adaptive.adaptive_music import HorizontalSequencer
        from engine.audio.adaptive.music_timing import MusicClock
        from engine.audio.adaptive.music_callback import MusicCallbackManager

        clock = MusicClock(bpm=120)
        callback_mgr = MusicCallbackManager(clock=clock)

        sequencer = HorizontalSequencer(clock=clock, callback_manager=callback_mgr)

        assert sequencer is not None


class TestHorizontalSequencerSections:
    """Tests for HorizontalSequencer section management."""

    def test_add_section(self):
        """add_section should register new section."""
        from engine.audio.adaptive.adaptive_music import HorizontalSequencer, MusicSection
        from engine.audio.adaptive.music_timing import MusicClock
        from engine.audio.adaptive.music_callback import MusicCallbackManager

        clock = MusicClock(bpm=120)
        callback_mgr = MusicCallbackManager(clock=clock)
        sequencer = HorizontalSequencer(clock=clock, callback_manager=callback_mgr)

        section = MusicSection(
            section_id="intro",
            name="Introduction",
            start_bar=0,
            end_bar=8
        )
        sequencer.add_section(section)

        # Verify section was registered
        retrieved = sequencer.get_section("intro")
        assert retrieved is not None
        assert retrieved.section_id == "intro"
        assert retrieved.name == "Introduction"

    def test_remove_section(self):
        """remove_section should remove registered section."""
        from engine.audio.adaptive.adaptive_music import HorizontalSequencer, MusicSection
        from engine.audio.adaptive.music_timing import MusicClock
        from engine.audio.adaptive.music_callback import MusicCallbackManager

        clock = MusicClock(bpm=120)
        callback_mgr = MusicCallbackManager(clock=clock)
        sequencer = HorizontalSequencer(clock=clock, callback_manager=callback_mgr)

        section = MusicSection(
            section_id="intro",
            name="Introduction",
            start_bar=0,
            end_bar=8
        )
        sequencer.add_section(section)
        result = sequencer.remove_section("intro")

        assert result is True

    def test_add_multiple_sections(self):
        """Should support multiple sections."""
        from engine.audio.adaptive.adaptive_music import HorizontalSequencer, MusicSection
        from engine.audio.adaptive.music_timing import MusicClock
        from engine.audio.adaptive.music_callback import MusicCallbackManager

        clock = MusicClock(bpm=120)
        callback_mgr = MusicCallbackManager(clock=clock)
        sequencer = HorizontalSequencer(clock=clock, callback_manager=callback_mgr)

        sequencer.add_section(MusicSection("intro", "Intro", 0, 8))
        sequencer.add_section(MusicSection("verse", "Verse", 8, 24))
        sequencer.add_section(MusicSection("chorus", "Chorus", 24, 40))

        # Verify all sections are registered
        assert sequencer.get_section("intro") is not None
        assert sequencer.get_section("verse") is not None
        assert sequencer.get_section("chorus") is not None


class TestEdgeCases:
    """Edge case tests for adaptive music system."""

    def test_rapid_intensity_changes(self):
        """System should handle rapid intensity changes."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        for _ in range(100):
            remixer.set_intensity(0.0, immediate=True)
            remixer.set_intensity(1.0, immediate=True)
            remixer.set_intensity(0.5, immediate=True)

        # Should still be in valid state
        assert 0.0 <= remixer.get_intensity() <= 1.0

    def test_empty_section_list(self):
        """Sequencer should handle no sections gracefully."""
        from engine.audio.adaptive.adaptive_music import HorizontalSequencer
        from engine.audio.adaptive.music_timing import MusicClock
        from engine.audio.adaptive.music_callback import MusicCallbackManager

        clock = MusicClock(bpm=120)
        callback_mgr = MusicCallbackManager(clock=clock)
        sequencer = HorizontalSequencer(clock=clock, callback_manager=callback_mgr)

        # Should not crash with no sections
        assert sequencer is not None

    def test_zero_delta_time_update(self):
        """Update with zero delta should not crash."""
        from engine.audio.adaptive.adaptive_music import VerticalRemixer

        mock_player = MagicMock()
        remixer = VerticalRemixer(stem_player=mock_player)

        remixer.set_intensity(0.8)
        remixer.update(0.0)

        # Should still be valid
        assert 0.0 <= remixer.get_intensity() <= 1.0
