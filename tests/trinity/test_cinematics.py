"""
Tests for Trinity Pattern - Tier 35: CINEMATICS Decorators
"""

import pytest

from trinity.decorators.cinematics import camera_track, cutscene
from trinity.decorators.registry import Tier, registry


class TestCutsceneDecorator:
    """Test @cutscene decorator."""

    def test_basic_application(self):
        """Test basic @cutscene application."""

        @cutscene(id="intro_cutscene")
        class IntroCutscene:
            pass

        assert hasattr(IntroCutscene, "_cutscene")
        assert IntroCutscene._cutscene is True
        assert IntroCutscene._cutscene_id == "intro_cutscene"
        assert IntroCutscene._cutscene_skippable is True
        assert IntroCutscene._cutscene_pause_gameplay is True
        assert "cutscene" in IntroCutscene._applied_decorators

    def test_with_custom_params(self):
        """Test @cutscene with custom parameters."""

        @cutscene(id="boss_death", skippable=False, pause_gameplay=False)
        class BossDeathCutscene:
            pass

        assert BossDeathCutscene._cutscene is True
        assert BossDeathCutscene._cutscene_id == "boss_death"
        assert BossDeathCutscene._cutscene_skippable is False
        assert BossDeathCutscene._cutscene_pause_gameplay is False

    def test_registry_registration(self):
        """Test that @cutscene is registered in the registry."""
        spec = registry.get("cutscene")
        assert spec is not None
        assert spec.name == "cutscene"
        assert spec.tier == Tier.CINEMATICS
        assert "class" in spec.target_types

    def test_tags_created(self):
        """Test that @cutscene creates proper tags."""

        @cutscene(id="test", skippable=True, pause_gameplay=True)
        class TestCutscene:
            pass

        assert hasattr(TestCutscene, "_tags")
        assert TestCutscene._tags.get("cutscene") is True
        assert TestCutscene._tags.get("cutscene_id") == "test"
        assert TestCutscene._tags.get("cutscene_skippable") is True
        assert TestCutscene._tags.get("cutscene_pause_gameplay") is True

    def test_validation_empty_id(self):
        """Test @cutscene validation rejects empty id."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @cutscene(id="")
            class BadCutscene:
                pass

    def test_validation_no_id(self):
        """Test @cutscene validation rejects missing id."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @cutscene()
            class BadCutscene:
                pass

    def test_composition_with_other_decorators(self):
        """Test @cutscene can be composed with other decorators."""
        from trinity.decorators.ecs_core import component

        @cutscene(id="test")
        @component(name="CutsceneComponent")
        class CutsceneComponent:
            pass

        assert CutsceneComponent._cutscene is True
        assert CutsceneComponent._component is True
        assert "cutscene" in CutsceneComponent._applied_decorators
        assert "component" in CutsceneComponent._applied_decorators


class TestCameraTrackDecorator:
    """Test @camera_track decorator."""

    def test_basic_application(self):
        """Test basic @camera_track application."""

        @camera_track()
        class BasicCamera:
            pass

        assert hasattr(BasicCamera, "_camera_track")
        assert BasicCamera._camera_track is True
        assert BasicCamera._camera_track_blend_in == 0.5
        assert BasicCamera._camera_track_blend_out == 0.5
        assert "camera_track" in BasicCamera._applied_decorators

    def test_with_custom_blend_times(self):
        """Test @camera_track with custom blend times."""

        @camera_track(blend_in=1.0, blend_out=2.0)
        class CustomBlendCamera:
            pass

        assert CustomBlendCamera._camera_track is True
        assert CustomBlendCamera._camera_track_blend_in == 1.0
        assert CustomBlendCamera._camera_track_blend_out == 2.0

    def test_with_zero_blend_times(self):
        """Test @camera_track with zero blend times."""

        @camera_track(blend_in=0.0, blend_out=0.0)
        class NoBlendCamera:
            pass

        assert NoBlendCamera._camera_track is True
        assert NoBlendCamera._camera_track_blend_in == 0.0
        assert NoBlendCamera._camera_track_blend_out == 0.0

    def test_registry_registration(self):
        """Test that @camera_track is registered in the registry."""
        spec = registry.get("camera_track")
        assert spec is not None
        assert spec.name == "camera_track"
        assert spec.tier == Tier.CINEMATICS
        assert "class" in spec.target_types

    def test_tags_created(self):
        """Test that @camera_track creates proper tags."""

        @camera_track(blend_in=1.5, blend_out=2.5)
        class TestCamera:
            pass

        assert hasattr(TestCamera, "_tags")
        assert TestCamera._tags.get("camera_track") is True
        assert TestCamera._tags.get("camera_track_blend_in") == 1.5
        assert TestCamera._tags.get("camera_track_blend_out") == 2.5

    def test_validation_negative_blend_in(self):
        """Test @camera_track validation rejects negative blend_in."""
        with pytest.raises(ValueError, match="blend_in must be >= 0"):

            @camera_track(blend_in=-1.0)
            class BadCamera:
                pass

    def test_validation_negative_blend_out(self):
        """Test @camera_track validation rejects negative blend_out."""
        with pytest.raises(ValueError, match="blend_out must be >= 0"):

            @camera_track(blend_out=-0.5)
            class BadCamera:
                pass

    def test_validation_invalid_type_blend_in(self):
        """Test @camera_track validation rejects invalid blend_in type."""
        with pytest.raises(ValueError, match="blend_in must be >= 0"):

            @camera_track(blend_in="invalid")
            class BadCamera:
                pass

    def test_composition(self):
        """Test @camera_track composition with @cutscene."""

        @camera_track(blend_in=1.0, blend_out=1.0)
        @cutscene(id="test")
        class CutsceneWithCamera:
            pass

        assert CutsceneWithCamera._camera_track is True
        assert CutsceneWithCamera._cutscene is True
        assert "camera_track" in CutsceneWithCamera._applied_decorators
        assert "cutscene" in CutsceneWithCamera._applied_decorators


class TestCinematicsIntegration:
    """Test integration between cinematics decorators."""

    def test_full_cutscene_with_camera(self):
        """Test complete cutscene with camera track."""

        @camera_track(blend_in=0.75, blend_out=0.75)
        @cutscene(id="epic_intro", skippable=False, pause_gameplay=True)
        class EpicIntroCutscene:
            pass

        assert EpicIntroCutscene._cutscene is True
        assert EpicIntroCutscene._cutscene_id == "epic_intro"
        assert EpicIntroCutscene._cutscene_skippable is False
        assert EpicIntroCutscene._camera_track is True
        assert EpicIntroCutscene._camera_track_blend_in == 0.75
        assert EpicIntroCutscene._camera_track_blend_out == 0.75

    def test_multiple_camera_tracks(self):
        """Test that multiple camera tracks can be created."""

        @camera_track(blend_in=0.5, blend_out=0.5)
        class Camera1:
            pass

        @camera_track(blend_in=1.0, blend_out=1.0)
        class Camera2:
            pass

        @camera_track(blend_in=0.0, blend_out=2.0)
        class Camera3:
            pass

        assert Camera1._camera_track_blend_in == 0.5
        assert Camera2._camera_track_blend_in == 1.0
        assert Camera3._camera_track_blend_out == 2.0

    def test_registries(self):
        """Test that all cinematics decorators use cinematics registry."""

        @cutscene(id="c1")
        class C1:
            pass

        @camera_track()
        class C2:
            pass

        assert "cinematics" in C1._registries
        assert "cinematics" in C2._registries

    def test_tier_ordering(self):
        """Test that cinematics decorators are in correct tier."""
        specs = registry.by_tier(Tier.CINEMATICS)
        names = {spec.name for spec in specs}
        assert "cutscene" in names
        assert "camera_track" in names

    def test_composition_with_narrative(self):
        """Test cinematics can compose with narrative decorators."""
        from trinity.decorators.narrative import dialogue, voice_over

        @camera_track()
        @cutscene(id="dialogue_cutscene")
        @voice_over(audio_asset="cutscene_vo.wav")
        @dialogue(id="cutscene_dialogue", speaker="hero")
        class NarrativeCutscene:
            pass

        assert NarrativeCutscene._cutscene is True
        assert NarrativeCutscene._camera_track is True
        assert NarrativeCutscene._voice_over is True
        assert NarrativeCutscene._dialogue is True


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_integer_blend_times(self):
        """Test that integer blend times work."""

        @camera_track(blend_in=1, blend_out=2)
        class IntegerBlendCamera:
            pass

        assert IntegerBlendCamera._camera_track_blend_in == 1
        assert IntegerBlendCamera._camera_track_blend_out == 2

    def test_large_blend_times(self):
        """Test very large blend times."""

        @camera_track(blend_in=1000.0, blend_out=9999.9)
        class LargeBlendCamera:
            pass

        assert LargeBlendCamera._camera_track_blend_in == 1000.0
        assert LargeBlendCamera._camera_track_blend_out == 9999.9

    def test_cutscene_with_special_id(self):
        """Test cutscene with special characters in id."""

        @cutscene(id="level-1_intro_v2")
        class SpecialIdCutscene:
            pass

        assert SpecialIdCutscene._cutscene_id == "level-1_intro_v2"

    def test_applied_steps_recorded(self):
        """Test that applied steps are recorded."""

        @cutscene(id="test")
        class TestClass:
            pass

        assert hasattr(TestClass, "_applied_steps")
        assert len(TestClass._applied_steps) > 0

    def test_describe_op_creates_schema(self):
        """Test that DESCRIBE op creates schema."""

        @cutscene(id="test")
        class TestClass:
            pass

        assert hasattr(TestClass, "_described")
        assert TestClass._described is True
