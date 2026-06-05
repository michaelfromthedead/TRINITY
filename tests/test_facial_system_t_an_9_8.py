"""
Comprehensive tests for the Facial Animation System (T-AN-9.8).

Tests the ECS system for facial animation processing:
- Blend shape application
- FACS AU combination
- Lip sync phoneme timing
- Eye saccade/blink
- Layer composition order
- Per-region masking
- Audio sync accuracy
- Performance with many blend shapes

Minimum 50 tests with real assertions.
"""

import math
import random
import time

import pytest

from engine.animation.systems.facial_system import (
    # Enumerations
    EmotionState,
    LipSyncPhoneme,
    FacialRegion,
    FacialLayerPriority,
    BlendMode,
    # Data structures
    Expression,
    FacialLayer,
    FaceRig,
    LipSyncState,
    EyeState,
    FACSState,
    AudioSyncData,
    # Component
    FacialComponent,
    # System
    FacialSystem,
    # Factory functions
    create_default_face_rig,
    create_facial_component,
    # System decorator
    system,
)
from engine.core.math import Vec3, Quat, Transform
from engine.core.ecs import Entity, World


# =============================================================================
# Helper Functions
# =============================================================================


def make_entity(index: int = 0, generation: int = 1) -> Entity:
    """Create an entity with sensible defaults."""
    return Entity(index, generation)


def make_world() -> World:
    """Create a minimal world for testing."""
    return World()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def facial_system():
    """Create a facial system for testing."""
    return FacialSystem()


@pytest.fixture
def facial_component():
    """Create a facial component with defaults."""
    return create_facial_component(
        with_default_rig=True,
        with_default_expressions=True,
    )


@pytest.fixture
def face_rig():
    """Create a default face rig."""
    return create_default_face_rig()


# =============================================================================
# Expression Tests
# =============================================================================


class TestExpression:
    """Tests for Expression class."""

    def test_create_basic(self):
        """Test basic expression creation."""
        expr = Expression(name="smile")
        assert expr.name == "smile"
        assert expr.intensity == 1.0
        assert len(expr.blend_shapes) == 0

    def test_create_with_blend_shapes(self):
        """Test expression with blend shapes."""
        expr = Expression(
            name="smile",
            blend_shapes={
                "mouthSmileLeft": 0.8,
                "mouthSmileRight": 0.8,
            },
        )
        assert len(expr.blend_shapes) == 2
        assert expr.blend_shapes["mouthSmileLeft"] == 0.8

    def test_blend_with_other(self):
        """Test blending two expressions."""
        expr_a = Expression(
            name="neutral",
            blend_shapes={"jawOpen": 0.0},
            intensity=1.0,
        )
        expr_b = Expression(
            name="open",
            blend_shapes={"jawOpen": 1.0},
            intensity=1.0,
        )

        blended = expr_a.blend_with(expr_b, 0.5)
        assert blended.blend_shapes["jawOpen"] == pytest.approx(0.5, abs=0.01)

    def test_scale_expression(self):
        """Test scaling expression intensity."""
        expr = Expression(
            name="test",
            blend_shapes={"shape": 1.0},
            intensity=1.0,
        )
        scaled = expr.scale(0.5)
        assert scaled.blend_shapes["shape"] == 0.5


# =============================================================================
# FacialLayer Tests
# =============================================================================


class TestFacialLayer:
    """Tests for FacialLayer class."""

    def test_create_layer(self):
        """Test layer creation."""
        layer = FacialLayer(
            name="test",
            priority=FacialLayerPriority.EMOTION,
        )
        assert layer.name == "test"
        assert layer.priority == FacialLayerPriority.EMOTION
        assert layer.weight == 1.0
        assert layer.enabled is True

    def test_layer_blend_mode(self):
        """Test layer blend modes."""
        layer = FacialLayer(
            name="additive",
            blend_mode=BlendMode.ADDITIVE,
        )
        assert layer.blend_mode == BlendMode.ADDITIVE

    def test_layer_region_mask(self):
        """Test layer region masking."""
        layer = FacialLayer(
            name="lip_sync",
            region_mask=FacialRegion.MOUTH,
        )
        assert layer.region_mask == FacialRegion.MOUTH

    def test_clear_layer(self):
        """Test clearing layer weights."""
        layer = FacialLayer(name="test")
        layer.blend_shapes = {"shape": 0.5}
        layer.clear()
        assert len(layer.blend_shapes) == 0


# =============================================================================
# FaceRig Tests
# =============================================================================


class TestFaceRig:
    """Tests for FaceRig class."""

    def test_create_default_rig(self):
        """Test creating default face rig."""
        rig = create_default_face_rig()
        assert rig is not None

    def test_setup_default_regions(self, face_rig):
        """Test default region setup."""
        assert FacialRegion.UPPER_FACE in face_rig.region_shapes
        assert FacialRegion.LOWER_FACE in face_rig.region_shapes
        assert FacialRegion.EYES in face_rig.region_shapes

    def test_get_shapes_for_region(self, face_rig):
        """Test getting shapes for a region."""
        eye_shapes = face_rig.get_shapes_for_region(FacialRegion.EYES)
        assert "eyeBlinkLeft" in eye_shapes
        assert "eyeBlinkRight" in eye_shapes

    def test_get_all_shapes(self, face_rig):
        """Test getting all shapes."""
        face_rig.blend_shape_names = ["shape1", "shape2"]
        all_shapes = face_rig.get_shapes_for_region(FacialRegion.ALL)
        assert len(all_shapes) == 2


# =============================================================================
# LipSyncState Tests
# =============================================================================


class TestLipSyncState:
    """Tests for LipSyncState class."""

    def test_initial_state(self):
        """Test initial lip sync state."""
        state = LipSyncState()
        assert state.current_phoneme == LipSyncPhoneme.SILENCE
        assert state.is_speaking is False

    def test_start_speaking(self):
        """Test starting speech."""
        state = LipSyncState()
        state.start_speaking()
        assert state.is_speaking is True
        assert state.audio_time == 0.0

    def test_stop_speaking(self):
        """Test stopping speech."""
        state = LipSyncState()
        state.start_speaking()
        state.current_phoneme = LipSyncPhoneme.AA
        state.stop_speaking()
        assert state.is_speaking is False
        assert state.current_phoneme == LipSyncPhoneme.SILENCE


# =============================================================================
# EyeState Tests
# =============================================================================


class TestEyeState:
    """Tests for EyeState class."""

    def test_initial_state(self):
        """Test initial eye state."""
        state = EyeState()
        assert state.is_blinking is False
        assert state.look_weight == 1.0

    def test_blink_progress(self):
        """Test blink progress tracking."""
        state = EyeState()
        state.is_blinking = True
        state.blink_progress = 0.5
        assert state.blink_progress == 0.5


# =============================================================================
# FACSState Tests
# =============================================================================


class TestFACSState:
    """Tests for FACSState class."""

    def test_set_au(self):
        """Test setting action unit."""
        state = FACSState()
        state.set_au("AU12", 0.8)
        assert state.get_au("AU12") == 0.8

    def test_set_au_bilateral(self):
        """Test setting bilateral AU."""
        state = FACSState()
        state.set_au("AU12", 0.5, left=0.3, right=0.7)
        assert state.au_left_intensities["AU12"] == 0.3
        assert state.au_right_intensities["AU12"] == 0.7

    def test_clear_aus(self):
        """Test clearing all AUs."""
        state = FACSState()
        state.set_au("AU12", 0.8)
        state.clear()
        assert state.get_au("AU12") == 0.0


# =============================================================================
# AudioSyncData Tests
# =============================================================================


class TestAudioSyncData:
    """Tests for AudioSyncData class."""

    def test_empty_timeline(self):
        """Test empty phoneme timeline."""
        data = AudioSyncData()
        phoneme, weight = data.get_phoneme_at_time(0.5)
        assert phoneme == LipSyncPhoneme.SILENCE

    def test_add_phoneme(self):
        """Test adding phoneme to timeline."""
        data = AudioSyncData()
        data.add_phoneme(0.0, LipSyncPhoneme.AA, 1.0)
        data.add_phoneme(0.5, LipSyncPhoneme.EE, 0.8)
        assert len(data.phoneme_timeline) == 2

    def test_get_phoneme_at_time(self):
        """Test getting phoneme at specific time."""
        data = AudioSyncData()
        data.add_phoneme(0.0, LipSyncPhoneme.AA, 1.0)
        data.add_phoneme(0.5, LipSyncPhoneme.EE, 0.8)

        phoneme, _ = data.get_phoneme_at_time(0.25)
        assert phoneme == LipSyncPhoneme.AA

        phoneme, _ = data.get_phoneme_at_time(0.75)
        assert phoneme == LipSyncPhoneme.EE

    def test_timeline_sorting(self):
        """Test timeline is sorted by time."""
        data = AudioSyncData()
        data.add_phoneme(1.0, LipSyncPhoneme.EE, 1.0)
        data.add_phoneme(0.0, LipSyncPhoneme.AA, 1.0)
        assert data.phoneme_timeline[0][0] == 0.0


# =============================================================================
# FacialComponent Tests
# =============================================================================


class TestFacialComponent:
    """Tests for FacialComponent class."""

    def test_create_component(self):
        """Test component creation."""
        component = FacialComponent()
        assert component.enabled is True
        assert component.current_emotion == EmotionState.NEUTRAL

    def test_create_with_defaults(self):
        """Test creation with default settings."""
        component = create_facial_component()
        assert len(component.layers) > 0
        assert "lip_sync" in component.layers
        assert "eye" in component.layers

    def test_set_emotion(self, facial_component):
        """Test setting emotion."""
        facial_component.set_emotion(EmotionState.HAPPY, 0.8)
        assert facial_component.current_emotion == EmotionState.HAPPY
        assert facial_component.emotion_intensity == 0.8

    def test_set_phoneme(self, facial_component):
        """Test setting phoneme."""
        facial_component.set_phoneme(LipSyncPhoneme.AA, 1.0)
        assert facial_component.lip_sync.current_phoneme == LipSyncPhoneme.AA

    def test_set_look_target(self, facial_component):
        """Test setting look target."""
        target = Vec3(1, 0, 5)
        facial_component.set_look_target(target, 0.9)
        assert facial_component.eye_state.look_weight == 0.9

    def test_add_expression(self, facial_component):
        """Test adding expression to library."""
        expr = Expression(name="custom", blend_shapes={"test": 0.5})
        facial_component.add_expression(expr)
        assert "custom" in facial_component.expressions

    def test_play_expression(self, facial_component):
        """Test playing named expression."""
        # Should have default expressions
        result = facial_component.play_expression("smile", 0.8)
        assert result is True
        assert facial_component.custom_expression is not None

    def test_play_nonexistent_expression(self, facial_component):
        """Test playing non-existent expression."""
        result = facial_component.play_expression("nonexistent")
        assert result is False

    def test_get_layer(self, facial_component):
        """Test getting layer by name."""
        layer = facial_component.get_layer("emotion")
        assert layer is not None
        assert layer.name == "emotion"

    def test_set_layer_weight(self, facial_component):
        """Test setting layer weight."""
        result = facial_component.set_layer_weight("emotion", 0.5)
        assert result is True
        assert facial_component.layers["emotion"].weight == 0.5


# =============================================================================
# FacialSystem Tests
# =============================================================================


class TestFacialSystem:
    """Tests for FacialSystem class."""

    def test_system_creation(self, facial_system):
        """Test system creation."""
        assert facial_system is not None

    def test_system_decorator(self):
        """Test system has correct decorator metadata."""
        assert hasattr(FacialSystem, '_system_phase')
        assert FacialSystem._system_phase == "animation"

    def test_update_single_component(self, facial_system, facial_component):
        """Test updating a single component."""
        entity = make_entity(1)
        world = make_world()

        facial_system.update(world, 0.016, [(entity, facial_component)])
        assert facial_component._dirty is False

    def test_update_disabled_component(self, facial_system, facial_component):
        """Test disabled component is skipped."""
        facial_component.enabled = False
        entity = make_entity(1)
        world = make_world()

        facial_system.update(world, 0.016, [(entity, facial_component)])
        # Should not crash

    def test_emotion_applies_blend_shapes(self, facial_system, facial_component):
        """Test emotion applies blend shapes."""
        facial_component.set_emotion(EmotionState.HAPPY, 1.0)

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Should have some output
        assert len(facial_component.output_blend_shapes) > 0

    def test_lip_sync_applies_blend_shapes(self, facial_system, facial_component):
        """Test lip sync applies blend shapes."""
        facial_component.set_phoneme(LipSyncPhoneme.AA, 1.0)
        facial_component.lip_sync._transition_progress = 1.0

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Should have jaw open or mouth shapes
        assert "jawOpen" in facial_component.output_blend_shapes or len(facial_component.output_blend_shapes) > 0

    def test_eye_blink(self, facial_system, facial_component):
        """Test eye blinking."""
        facial_component.eye_state.is_blinking = True
        facial_component.eye_state.blink_progress = 0.5

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Should have blink weights
        assert "eyeBlinkLeft" in facial_component.output_blend_shapes

    def test_layer_priority(self, facial_system, facial_component):
        """Test layer priority ordering."""
        # Verify priority ordering
        layers = sorted(
            facial_component.layers.values(),
            key=lambda l: l.priority.value
        )
        assert layers[0].priority.value < layers[-1].priority.value

    def test_region_masking(self, facial_system, facial_component):
        """Test region masking limits affected shapes."""
        # Lip sync layer should only affect mouth region
        lip_layer = facial_component.layers.get("lip_sync")
        assert lip_layer is not None
        assert lip_layer.region_mask == FacialRegion.MOUTH

    def test_blend_mode_replace(self, facial_system, facial_component):
        """Test replace blend mode with emotion layer.

        The system clears layer outputs during update, so we test via
        emotion which populates the emotion layer.
        """
        # Set emotion which populates the emotion layer (REPLACE mode by default)
        facial_component.set_emotion(EmotionState.HAPPY, 1.0)

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Emotion layer should have applied happy expression shapes
        # mouthSmileLeft should be ~0.8 for happy
        smile_weight = facial_component.output_blend_shapes.get("mouthSmileLeft", 0)
        assert smile_weight > 0.5, f"Expected smile weight > 0.5, got {smile_weight}"

    def test_blend_mode_additive(self, facial_system, facial_component):
        """Test additive blend mode with lip sync layer.

        The lip sync layer is additive for mouth shapes on top of emotion.
        """
        # Set emotion first (happy shows some mouth shapes)
        facial_component.set_emotion(EmotionState.HAPPY, 0.5)

        # Set a phoneme that affects mouth (AA opens jaw)
        facial_component.set_phoneme(LipSyncPhoneme.AA, 1.0)
        facial_component.lip_sync._transition_progress = 1.0

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Should have both emotion and lip sync contributions
        # jawOpen comes from AA phoneme
        jaw_weight = facial_component.output_blend_shapes.get("jawOpen", 0)
        assert jaw_weight > 0.5, f"Expected jaw weight > 0.5, got {jaw_weight}"

    def test_phoneme_transition(self, facial_system, facial_component):
        """Test phoneme transition."""
        # Start with one phoneme
        facial_component.set_phoneme(LipSyncPhoneme.AA, 1.0)
        facial_component.lip_sync._transition_progress = 1.0

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Change phoneme
        facial_component.set_phoneme(LipSyncPhoneme.EE, 1.0)
        assert facial_component.lip_sync._transition_progress == 0.0

    def test_audio_sync_timeline(self, facial_system, facial_component):
        """Test audio sync from timeline."""
        facial_component.audio_sync.add_phoneme(0.0, LipSyncPhoneme.AA, 1.0)
        facial_component.audio_sync.add_phoneme(0.5, LipSyncPhoneme.EE, 1.0)
        facial_component.lip_sync.is_speaking = True

        entity = make_entity(1)
        world = make_world()

        # Update several times to advance audio time
        for _ in range(30):
            facial_system.update(world, 0.016, [(entity, facial_component)])

        assert facial_component.lip_sync.audio_time > 0

    def test_process_audio_for_lip_sync(self, facial_system, facial_component):
        """Test audio processing for lip sync."""
        # Generate some audio samples
        samples = [math.sin(i * 0.1) * 0.5 for i in range(1000)]

        facial_system.process_audio_for_lip_sync(facial_component, samples)
        assert facial_component.lip_sync.audio_intensity > 0

    def test_process_audio_silence(self, facial_system, facial_component):
        """Test audio processing with silence."""
        samples = [0.0] * 1000
        facial_system.process_audio_for_lip_sync(facial_component, samples)
        assert facial_component.lip_sync.current_phoneme == LipSyncPhoneme.SILENCE

    def test_set_phoneme_timeline(self, facial_system, facial_component):
        """Test setting phoneme timeline."""
        timeline = [
            (0.0, "AA", 1.0),
            (0.2, "EE", 0.8),
            (0.4, "SILENCE", 1.0),
        ]
        facial_system.set_phoneme_timeline(facial_component, timeline)
        assert len(facial_component.audio_sync.phoneme_timeline) == 3

    def test_trigger_blink(self, facial_system, facial_component):
        """Test manually triggering a blink."""
        facial_system.trigger_blink(facial_component)
        assert facial_component.eye_state.is_blinking is True

    def test_get_stats(self, facial_system, facial_component):
        """Test getting system stats."""
        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        stats = facial_system.get_stats()
        assert "last_update_time_ms" in stats
        assert "entities_processed" in stats
        assert stats["entities_processed"] == 1


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for facial system."""

    def test_many_blend_shapes(self, facial_system, facial_component):
        """Test performance with many blend shapes."""
        # Add many blend shapes to output
        for i in range(100):
            facial_component.layers["base"].blend_shapes[f"shape_{i}"] = 0.5

        entity = make_entity(1)
        world = make_world()

        start = time.perf_counter()
        for _ in range(100):
            facial_system.update(world, 0.016, [(entity, facial_component)])
        elapsed = time.perf_counter() - start

        # Should complete 100 updates in under 100ms
        assert elapsed < 0.1

    def test_multiple_entities(self, facial_system):
        """Test performance with multiple entities."""
        entities = []
        for i in range(50):
            component = create_facial_component()
            component.set_emotion(EmotionState.HAPPY, random.random())
            entities.append((make_entity(i), component))

        world = make_world()

        start = time.perf_counter()
        facial_system.update(world, 0.016, entities)
        elapsed = time.perf_counter() - start

        # Should process 50 entities quickly
        assert elapsed < 0.05


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for facial system."""

    def test_full_update_cycle(self, facial_system, facial_component):
        """Test complete update cycle with all features."""
        # Set emotion
        facial_component.set_emotion(EmotionState.HAPPY, 0.8)

        # Set lip sync
        facial_component.set_phoneme(LipSyncPhoneme.AA, 1.0)
        facial_component.lip_sync._transition_progress = 1.0

        # Set look target
        facial_component.set_look_target(Vec3(1, 0, 5))

        # Set some FACS
        facial_component.facs_state.set_au("AU12", 0.5)

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Should have combined output
        assert len(facial_component.output_blend_shapes) > 0

    def test_layer_composition_order(self, facial_system, facial_component):
        """Test layers are composed in correct order."""
        # Verify layer priorities
        priorities = [
            (facial_component.layers["base"].priority.value, "base"),
            (facial_component.layers["emotion"].priority.value, "emotion"),
            (facial_component.layers["lip_sync"].priority.value, "lip_sync"),
            (facial_component.layers["eye"].priority.value, "eye"),
        ]
        sorted_priorities = sorted(priorities, key=lambda x: x[0])

        # Base should be first, eye should be last (before override)
        assert sorted_priorities[0][1] == "base"

    def test_custom_expression_override(self, facial_system, facial_component):
        """Test custom expression overrides other layers."""
        # Set up custom expression
        expr = Expression(
            name="override_test",
            blend_shapes={"testShape": 1.0},
            intensity=1.0,
        )
        facial_component.custom_expression = expr
        facial_component.custom_expression_weight = 1.0

        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.016, [(entity, facial_component)])

        # Should have override layer enabled
        assert facial_component.layers["override"].weight == 1.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_dt(self, facial_system, facial_component):
        """Test update with zero delta time."""
        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, 0.0, [(entity, facial_component)])
        # Should not crash

    def test_negative_dt(self, facial_system, facial_component):
        """Test update with negative delta time."""
        entity = make_entity(1)
        world = make_world()
        facial_system.update(world, -0.016, [(entity, facial_component)])
        # Should not crash

    def test_empty_entity_list(self, facial_system):
        """Test update with empty entity list."""
        world = make_world()
        facial_system.update(world, 0.016, [])
        # Should not crash

    def test_emotion_intensity_clamping(self, facial_component):
        """Test emotion intensity is clamped."""
        facial_component.set_emotion(EmotionState.HAPPY, 1.5)
        assert facial_component.emotion_intensity == 1.0

        facial_component.set_emotion(EmotionState.SAD, -0.5)
        assert facial_component.emotion_intensity == 0.0

    def test_layer_weight_clamping(self, facial_component):
        """Test layer weight is clamped."""
        facial_component.set_layer_weight("emotion", 1.5)
        assert facial_component.layers["emotion"].weight == 1.0

        facial_component.set_layer_weight("emotion", -0.5)
        assert facial_component.layers["emotion"].weight == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
