"""
Whitebox tests for adaptive_music.py - Adaptive music system.
"""

import pytest
import time
import threading
from engine.audio.adaptive.adaptive_music import (
    AdaptiveMode,
    BranchType,
    MusicSection,
    IntensityLevel,
    AdaptiveParameters,
    VerticalRemixer,
    HorizontalSequencer,
    AdaptiveMusicSystem,
)
from engine.audio.adaptive.music_timing import MusicClock, TimeSignature
from engine.audio.adaptive.music_stem import LayeredMusicPlayer, StemInfo
from engine.audio.adaptive.music_callback import MusicCallbackManager, CallbackEvent
from engine.audio.adaptive.music_state import MusicStateManager, MusicStateConfig
from engine.audio.adaptive.config import (
    INTENSITY_MIN,
    INTENSITY_MAX,
    VERTICAL_THRESHOLD_LOW,
    VERTICAL_THRESHOLD_MED,
    VERTICAL_THRESHOLD_HIGH,
    LAYER_DRUMS,
    LAYER_BASS,
    LAYER_MELODY,
    LAYER_PADS,
    PARAM_INTENSITY,
    PARAM_DANGER,
    PARAM_TENSION,
    PARAM_ENERGY,
    STATE_EXPLORATION,
    STATE_COMBAT,
    STATE_STEALTH,
    CALLBACK_BAR,
)


class TestAdaptiveMode:
    """Tests for AdaptiveMode enum."""

    def test_adaptive_modes_exist(self):
        """All adaptive modes should exist."""
        assert AdaptiveMode.NONE is not None
        assert AdaptiveMode.VERTICAL is not None
        assert AdaptiveMode.HORIZONTAL is not None
        assert AdaptiveMode.COMBINED is not None


class TestBranchType:
    """Tests for BranchType enum."""

    def test_branch_types_exist(self):
        """All branch types should exist."""
        assert BranchType.SEQUENTIAL is not None
        assert BranchType.RANDOM is not None
        assert BranchType.RULE_BASED is not None
        assert BranchType.WEIGHTED is not None


class TestMusicSection:
    """Tests for MusicSection dataclass."""

    def test_create_music_section(self):
        """Create music section."""
        section = MusicSection(
            section_id="verse_1",
            name="Verse 1",
            start_bar=0,
            end_bar=8,
        )
        assert section.section_id == "verse_1"
        assert section.start_bar == 0
        assert section.end_bar == 8

    def test_section_defaults(self):
        """MusicSection has sensible defaults."""
        section = MusicSection(
            section_id="test",
            name="Test",
            start_bar=0,
            end_bar=4,
        )
        assert section.can_loop is True
        assert section.loop_count == 0
        assert section.next_sections == []
        assert section.intensity_range == (0.0, 1.0)

    def test_section_length_bars(self):
        """Section length in bars calculated correctly."""
        section = MusicSection(
            section_id="test",
            name="Test",
            start_bar=4,
            end_bar=12,
        )
        assert section.length_bars == 8

    def test_section_with_branching(self):
        """Create section with branching."""
        section = MusicSection(
            section_id="verse_1",
            name="Verse 1",
            start_bar=0,
            end_bar=8,
            next_sections=["chorus", "verse_2"],
            weights={"chorus": 0.7, "verse_2": 0.3},
        )
        assert "chorus" in section.next_sections
        assert section.weights["chorus"] == 0.7

    def test_section_with_intensity_range(self):
        """Create section with intensity range."""
        section = MusicSection(
            section_id="combat",
            name="Combat",
            start_bar=0,
            end_bar=8,
            intensity_range=(0.7, 1.0),
        )
        assert section.intensity_range == (0.7, 1.0)


class TestIntensityLevel:
    """Tests for IntensityLevel dataclass."""

    def test_create_intensity_level(self):
        """Create intensity level."""
        level = IntensityLevel(
            level_id="high",
            threshold=0.75,
            layers={"drums": 1.0, "bass": 0.9},
        )
        assert level.level_id == "high"
        assert level.threshold == 0.75
        assert level.layers["drums"] == 1.0

    def test_intensity_level_with_sections(self):
        """Create intensity level with associated sections."""
        level = IntensityLevel(
            level_id="combat",
            threshold=0.8,
            layers={"drums": 1.0},
            sections=["combat_1", "combat_2"],
        )
        assert "combat_1" in level.sections


class TestAdaptiveParameters:
    """Tests for AdaptiveParameters dataclass."""

    def test_create_adaptive_parameters(self):
        """Create adaptive parameters."""
        params = AdaptiveParameters()
        assert params.intensity == 0.5
        assert params.danger == 0.0
        assert params.tension == 0.0
        assert params.energy == 0.5

    def test_get_builtin_parameters(self):
        """Get built-in parameters."""
        params = AdaptiveParameters(intensity=0.8, danger=0.5)
        assert params.get(PARAM_INTENSITY) == 0.8
        assert params.get(PARAM_DANGER) == 0.5
        assert params.get(PARAM_TENSION) == 0.0
        assert params.get(PARAM_ENERGY) == 0.5

    def test_get_custom_parameter(self):
        """Get custom parameter."""
        params = AdaptiveParameters(custom={"custom_param": 0.7})
        assert params.get("custom_param") == 0.7

    def test_get_nonexistent_parameter(self):
        """Get nonexistent parameter returns default."""
        params = AdaptiveParameters()
        assert params.get("nonexistent", default=0.5) == 0.5

    def test_set_builtin_parameters(self):
        """Set built-in parameters."""
        params = AdaptiveParameters()
        params.set(PARAM_INTENSITY, 0.9)
        params.set(PARAM_DANGER, 0.8)
        assert params.intensity == 0.9
        assert params.danger == 0.8

    def test_set_custom_parameter(self):
        """Set custom parameter."""
        params = AdaptiveParameters()
        params.set("custom_param", 0.6)
        assert params.custom["custom_param"] == 0.6

    def test_set_clamps_values(self):
        """Set clamps values to 0-1 range."""
        params = AdaptiveParameters()
        params.set(PARAM_INTENSITY, 1.5)
        assert params.intensity == 1.0
        params.set(PARAM_INTENSITY, -0.5)
        assert params.intensity == 0.0


class TestVerticalRemixer:
    """Tests for VerticalRemixer class."""

    def create_remixer(self):
        """Create vertical remixer with player."""
        player = LayeredMusicPlayer()
        # Add some stems
        player.add_stem(StemInfo("drums", "Drums", LAYER_DRUMS, "/drums.wav"))
        player.add_stem(StemInfo("bass", "Bass", LAYER_BASS, "/bass.wav"))
        player.add_stem(StemInfo("melody", "Melody", LAYER_MELODY, "/melody.wav"))
        player.add_stem(StemInfo("pads", "Pads", LAYER_PADS, "/pads.wav"))
        return VerticalRemixer(player)

    def test_create_vertical_remixer(self):
        """Create vertical remixer."""
        remixer = self.create_remixer()
        assert remixer.get_intensity() == 0.5
        # Should have default intensity levels
        assert len(remixer._intensity_levels) > 0

    def test_add_intensity_level(self):
        """Add custom intensity level."""
        remixer = self.create_remixer()
        level = IntensityLevel(
            level_id="extreme",
            threshold=0.95,
            layers={"drums": 1.0, "bass": 1.0},
        )
        remixer.add_intensity_level(level)
        assert "extreme" in remixer._intensity_levels

    def test_remove_intensity_level(self):
        """Remove intensity level."""
        remixer = self.create_remixer()
        level = IntensityLevel("custom", 0.5, {"drums": 1.0})
        remixer.add_intensity_level(level)
        assert remixer.remove_intensity_level("custom") is True
        assert "custom" not in remixer._intensity_levels

    def test_remove_nonexistent_level(self):
        """Removing nonexistent level returns False."""
        remixer = self.create_remixer()
        assert remixer.remove_intensity_level("nonexistent") is False

    def test_set_intensity(self):
        """Set target intensity."""
        remixer = self.create_remixer()
        remixer.set_intensity(0.8)
        assert remixer._target_intensity == 0.8

    def test_set_intensity_immediate(self):
        """Set intensity immediately."""
        remixer = self.create_remixer()
        remixer.set_intensity(0.9, immediate=True)
        assert remixer._current_intensity == 0.9

    def test_set_intensity_clamps(self):
        """Set intensity clamps to valid range."""
        remixer = self.create_remixer()
        remixer.set_intensity(1.5, immediate=True)
        assert remixer._current_intensity == INTENSITY_MAX
        remixer.set_intensity(-0.5, immediate=True)
        assert remixer._current_intensity == INTENSITY_MIN

    def test_get_intensity(self):
        """Get current intensity."""
        remixer = self.create_remixer()
        remixer.set_intensity(0.7, immediate=True)
        assert remixer.get_intensity() == 0.7

    def test_update_smooths_intensity(self):
        """Update smooths intensity towards target."""
        remixer = self.create_remixer()
        remixer.set_intensity(0.5, immediate=True)
        remixer.set_intensity(1.0)  # Set target without immediate
        remixer.update(delta_time=0.1)
        # Should have moved towards target
        assert remixer._current_intensity > 0.5

    def test_default_levels_defined(self):
        """Default intensity levels are defined."""
        remixer = self.create_remixer()
        assert "low" in remixer._intensity_levels
        assert "medium" in remixer._intensity_levels
        assert "high" in remixer._intensity_levels
        assert "maximum" in remixer._intensity_levels


class TestHorizontalSequencer:
    """Tests for HorizontalSequencer class."""

    def create_sequencer(self):
        """Create horizontal sequencer."""
        clock = MusicClock(bpm=120.0)
        callback_manager = MusicCallbackManager(clock)
        return HorizontalSequencer(clock, callback_manager), clock

    def test_create_horizontal_sequencer(self):
        """Create horizontal sequencer."""
        sequencer, clock = self.create_sequencer()
        assert sequencer.current_section is None

    def test_add_section(self):
        """Add a music section."""
        sequencer, clock = self.create_sequencer()
        section = MusicSection("verse_1", "Verse 1", 0, 8)
        sequencer.add_section(section)
        assert sequencer.get_section("verse_1") is not None

    def test_remove_section(self):
        """Remove a music section."""
        sequencer, clock = self.create_sequencer()
        section = MusicSection("verse_1", "Verse 1", 0, 8)
        sequencer.add_section(section)
        assert sequencer.remove_section("verse_1") is True
        assert sequencer.get_section("verse_1") is None

    def test_remove_nonexistent_section(self):
        """Removing nonexistent section returns False."""
        sequencer, clock = self.create_sequencer()
        assert sequencer.remove_section("nonexistent") is False

    def test_get_section(self):
        """Get section by ID."""
        sequencer, clock = self.create_sequencer()
        section = MusicSection("verse_1", "Verse 1", 0, 8)
        sequencer.add_section(section)
        found = sequencer.get_section("verse_1")
        assert found is not None
        assert found.name == "Verse 1"

    def test_set_branch_type(self):
        """Set branching behavior."""
        sequencer, clock = self.create_sequencer()
        sequencer.set_branch_type(BranchType.RANDOM)
        assert sequencer._branch_type == BranchType.RANDOM

    def test_start_section(self):
        """Start playing a section."""
        sequencer, clock = self.create_sequencer()
        clock.start()
        section = MusicSection("verse_1", "Verse 1", 0, 8)
        sequencer.add_section(section)
        sequencer.start_section("verse_1")
        assert sequencer.current_section is not None
        assert sequencer.current_section.section_id == "verse_1"
        clock.stop()

    def test_queue_next_section(self):
        """Queue a section to play next."""
        sequencer, clock = self.create_sequencer()
        sequencer.add_section(MusicSection("verse_1", "Verse 1", 0, 8))
        sequencer.add_section(MusicSection("chorus", "Chorus", 8, 16))
        sequencer.queue_next_section("chorus")
        assert sequencer._next_section is not None
        assert sequencer._next_section.section_id == "chorus"

    def test_section_change_callback(self):
        """Section change callback is invoked."""
        sequencer, clock = self.create_sequencer()
        clock.start()
        changes = []

        def on_change(old_section, new_section):
            changes.append((old_section, new_section))

        sequencer.set_on_section_change(on_change)
        sequencer.add_section(MusicSection("verse_1", "Verse 1", 0, 8))
        sequencer.add_section(MusicSection("verse_2", "Verse 2", 8, 16))
        sequencer.start_section("verse_1")
        sequencer.start_section("verse_2")
        assert len(changes) == 2
        clock.stop()

    def test_clear_sections(self):
        """Clear all sections."""
        sequencer, clock = self.create_sequencer()
        sequencer.add_section(MusicSection("s1", "S1", 0, 8))
        sequencer.add_section(MusicSection("s2", "S2", 8, 16))
        sequencer.clear()
        assert sequencer.get_section("s1") is None
        assert sequencer.current_section is None


class TestAdaptiveMusicSystem:
    """Tests for AdaptiveMusicSystem class."""

    def create_system(self):
        """Create adaptive music system."""
        clock = MusicClock(bpm=120.0)
        player = LayeredMusicPlayer()
        player.add_stem(StemInfo("drums", "Drums", LAYER_DRUMS, "/drums.wav"))
        player.add_stem(StemInfo("bass", "Bass", LAYER_BASS, "/bass.wav"))
        callback_manager = MusicCallbackManager(clock)
        return AdaptiveMusicSystem(clock, player, callback_manager), clock

    def test_create_adaptive_system(self):
        """Create adaptive music system."""
        system, clock = self.create_system()
        assert system.mode == AdaptiveMode.COMBINED
        assert system.vertical_remixer is not None
        assert system.horizontal_sequencer is not None

    def test_mode_property(self):
        """Get and set adaptive mode."""
        system, clock = self.create_system()
        system.mode = AdaptiveMode.VERTICAL
        assert system.mode == AdaptiveMode.VERTICAL

    def test_set_parameter(self):
        """Set music parameter."""
        system, clock = self.create_system()
        system.set_parameter(PARAM_INTENSITY, 0.8)
        assert system.parameters.intensity == 0.8

    def test_set_intensity_updates_remixer(self):
        """Setting intensity updates vertical remixer."""
        system, clock = self.create_system()
        system.set_parameter(PARAM_INTENSITY, 0.9, immediate=True)
        assert system.vertical_remixer._current_intensity == 0.9

    def test_get_parameter(self):
        """Get parameter value."""
        system, clock = self.create_system()
        system.set_parameter(PARAM_DANGER, 0.6)
        assert system.get_parameter(PARAM_DANGER) == 0.6

    def test_add_parameter_rule(self):
        """Add parameter processing rule."""
        system, clock = self.create_system()
        rule_calls = []

        def test_rule(params):
            rule_calls.append(params.intensity)

        system.add_parameter_rule(test_rule)
        system.set_parameter(PARAM_INTENSITY, 0.7)
        assert len(rule_calls) == 1
        assert rule_calls[0] == 0.7

    def test_trigger_combat(self):
        """Trigger combat music."""
        system, clock = self.create_system()
        system.trigger_combat()
        assert system.parameters.intensity == 0.9
        assert system.parameters.danger == 0.8

    def test_trigger_exploration(self):
        """Trigger exploration music."""
        system, clock = self.create_system()
        system.trigger_exploration()
        assert system.parameters.intensity == 0.3
        assert system.parameters.danger == 0.1

    def test_trigger_stealth(self):
        """Trigger stealth music."""
        system, clock = self.create_system()
        system.trigger_stealth()
        assert system.parameters.intensity == 0.4
        assert system.parameters.tension == 0.7

    def test_increase_intensity(self):
        """Increase intensity."""
        system, clock = self.create_system()
        system.set_parameter(PARAM_INTENSITY, 0.5)
        system.increase_intensity(0.2)
        assert system.parameters.intensity == 0.7

    def test_decrease_intensity(self):
        """Decrease intensity."""
        system, clock = self.create_system()
        system.set_parameter(PARAM_INTENSITY, 0.5)
        system.decrease_intensity(0.2)
        assert system.parameters.intensity == 0.3

    def test_update_with_vertical_mode(self):
        """Update with vertical mode active."""
        system, clock = self.create_system()
        system.mode = AdaptiveMode.VERTICAL
        system.set_parameter(PARAM_INTENSITY, 0.8)
        # Should not raise
        system.update(delta_time=0.016)

    def test_update_with_combined_mode(self):
        """Update with combined mode active."""
        system, clock = self.create_system()
        system.mode = AdaptiveMode.COMBINED
        # Should not raise
        system.update(delta_time=0.016)

    def test_with_state_manager(self):
        """System works with state manager."""
        clock = MusicClock(bpm=120.0)
        player = LayeredMusicPlayer()
        callback_manager = MusicCallbackManager(clock)
        state_manager = MusicStateManager(clock)
        state_manager.register_state(MusicStateConfig(state_id=STATE_COMBAT))
        state_manager.register_state(MusicStateConfig(state_id=STATE_EXPLORATION))
        state_manager.register_state(MusicStateConfig(state_id=STATE_STEALTH))

        system = AdaptiveMusicSystem(
            clock, player, callback_manager, state_manager
        )
        system.trigger_combat()
        assert state_manager.current_state_id == STATE_COMBAT
