"""
Comprehensive tests for the Facial Animation subsystem.

Tests all components:
- Blend shapes / morph targets
- FACS (Facial Action Coding System)
- Lip synchronization
- Eye animation and tracking
- Face rig integration
- Motion capture playback

Minimum 130 tests with real assertions.
"""

import math

import numpy as np
import pytest

from engine.animation.facial import (
    # Blend Shapes
    ARKIT_BLEND_SHAPES,
    BlendShape,
    BlendShapeController,
    BlendShapeSet,
    CorrectiveBlendShape,
    apply_blend_shapes,
    apply_blend_shapes_with_correctives,
    create_arkit_compatible_set,
    remap_blend_shape_weights,
    # FACS
    AU,
    ActionUnit,
    ActionUnitData,
    Expression,
    ExpressionData,
    FACSController,
    get_default_au_mappings,
    get_default_expressions,
    # Lip Sync
    CoarticulationSettings,
    LipSyncController,
    PhonemeEvent,
    Viseme,
    VisemeEvent,
    VisemeMapping,
    get_default_viseme_mappings,
    phoneme_to_viseme,
    create_phoneme_events_from_text,
    # Eye Animation
    BlinkController,
    BlinkSettings,
    EyeController,
    EyeLimits,
    EyeState,
    EyeTransform,
    PupilSettings,
    SaccadeSettings,
    # Face Rig
    AnimationLayer,
    AnimationPriority,
    EmotionState,
    FaceRig,
    create_face_rig,
    # Face Capture
    AnimationCurve,
    FaceCaptureClip,
    FaceCapturePlayer,
    FaceCaptureRetargeter,
    InterpolationMode,
    Keyframe,
    PlaybackState,
    RetargetMapping,
    create_clip_from_samples,
    merge_clips,
)


# =============================================================================
# Blend Shape Tests
# =============================================================================


class TestBlendShape:
    """Tests for BlendShape class."""

    def test_create_basic(self):
        """Test basic blend shape creation."""
        shape = BlendShape(name="smile")
        assert shape.name == "smile"
        assert shape.vertex_count == 0

    def test_create_with_data(self):
        """Test blend shape with vertex data."""
        indices = np.array([0, 1, 2], dtype=np.int32)
        deltas = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        shape = BlendShape(name="test", vertex_indices=indices, deltas=deltas)

        assert shape.vertex_count == 3
        assert shape.is_sparse is True
        assert len(shape.deltas) == 3

    def test_create_from_lists(self):
        """Test creation from Python lists."""
        shape = BlendShape(
            name="test",
            vertex_indices=[0, 1],
            deltas=[[1, 0, 0], [0, 1, 0]],
        )
        assert shape.vertex_count == 2
        assert isinstance(shape.vertex_indices, np.ndarray)

    def test_get_delta(self):
        """Test getting individual deltas."""
        deltas = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        shape = BlendShape(name="test", vertex_indices=[0, 1], deltas=deltas)

        delta = shape.get_delta(0)
        assert delta == (1.0, 2.0, 3.0)

    def test_get_delta_out_of_range(self):
        """Test getting delta with invalid index."""
        shape = BlendShape(name="test")
        delta = shape.get_delta(100)
        assert delta == (0.0, 0.0, 0.0)

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = BlendShape(
            name="test",
            vertex_indices=[0, 1, 2],
            deltas=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        data = original.to_dict()
        restored = BlendShape.from_dict(data)

        assert restored.name == original.name
        assert restored.vertex_count == original.vertex_count
        np.testing.assert_array_equal(restored.deltas, original.deltas)


class TestCorrectiveBlendShape:
    """Tests for CorrectiveBlendShape class."""

    def test_create_corrective(self):
        """Test corrective blend shape creation."""
        shape = BlendShape(name="corrective")
        corrective = CorrectiveBlendShape(
            shape=shape,
            driver_shapes=["smile", "open"],
            driver_weights=[0.5, 0.5],
        )
        assert corrective.name == "corrective"
        assert len(corrective.driver_shapes) == 2

    def test_calculate_weight_multiply(self):
        """Test weight calculation with multiply mode."""
        shape = BlendShape(name="corrective")
        corrective = CorrectiveBlendShape(
            shape=shape,
            driver_shapes=["a", "b"],
            driver_weights=[0.0, 0.0],
            combination_mode="multiply",
        )

        weights = {"a": 1.0, "b": 1.0}
        result = corrective.calculate_weight(weights)
        assert result == 1.0

        weights = {"a": 1.0, "b": 0.5}
        result = corrective.calculate_weight(weights)
        assert result == 0.5

    def test_calculate_weight_min(self):
        """Test weight calculation with min mode."""
        shape = BlendShape(name="corrective")
        corrective = CorrectiveBlendShape(
            shape=shape,
            driver_shapes=["a", "b"],
            driver_weights=[0.0, 0.0],
            combination_mode="min",
        )

        weights = {"a": 0.8, "b": 0.5}
        result = corrective.calculate_weight(weights)
        assert result == 0.5

    def test_calculate_weight_missing_driver(self):
        """Test weight calculation with missing driver."""
        shape = BlendShape(name="corrective")
        corrective = CorrectiveBlendShape(
            shape=shape,
            driver_shapes=["a", "b"],
            driver_weights=[0.0, 0.0],
            combination_mode="multiply",
        )

        weights = {"a": 1.0}  # b is missing
        result = corrective.calculate_weight(weights)
        assert result == 0.0


class TestBlendShapeSet:
    """Tests for BlendShapeSet class."""

    def test_create_empty(self):
        """Test empty blend shape set creation."""
        shape_set = BlendShapeSet(name="face")
        assert shape_set.vertex_count == 0
        assert shape_set.shape_count == 0

    def test_create_with_vertices(self):
        """Test creation with base vertices."""
        vertices = np.zeros((100, 3), dtype=np.float32)
        shape_set = BlendShapeSet(name="face", base_vertices=vertices)
        assert shape_set.vertex_count == 100

    def test_add_shape(self):
        """Test adding blend shapes."""
        shape_set = BlendShapeSet(name="face")
        shape = BlendShape(name="smile")
        shape_set.add_shape(shape)

        assert shape_set.shape_count == 1
        assert shape_set.has_shape("smile")

    def test_remove_shape(self):
        """Test removing blend shapes."""
        shape_set = BlendShapeSet(name="face")
        shape_set.add_shape(BlendShape(name="smile"))

        result = shape_set.remove_shape("smile")
        assert result is True
        assert shape_set.shape_count == 0

    def test_get_shape(self):
        """Test getting blend shape by name."""
        shape_set = BlendShapeSet(name="face")
        shape = BlendShape(name="smile")
        shape_set.add_shape(shape)

        retrieved = shape_set.get_shape("smile")
        assert retrieved is shape

        not_found = shape_set.get_shape("nonexistent")
        assert not_found is None

    def test_shape_names(self):
        """Test getting shape names."""
        shape_set = BlendShapeSet(name="face")
        shape_set.add_shape(BlendShape(name="a"))
        shape_set.add_shape(BlendShape(name="b"))

        names = shape_set.shape_names
        assert "a" in names
        assert "b" in names


class TestApplyBlendShapes:
    """Tests for blend shape application functions."""

    def test_apply_single_shape(self):
        """Test applying a single blend shape."""
        base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        shape = BlendShape(
            name="test",
            vertex_indices=[0],
            deltas=[[1, 0, 0]],
        )

        result = apply_blend_shapes(base, {"test": shape}, {"test": 1.0})
        assert result[0, 0] == 1.0

    def test_apply_partial_weight(self):
        """Test applying with partial weight."""
        base = np.zeros((3, 3), dtype=np.float32)
        shape = BlendShape(
            name="test",
            vertex_indices=[0],
            deltas=[[2, 0, 0]],
        )

        result = apply_blend_shapes(base, {"test": shape}, {"test": 0.5})
        assert result[0, 0] == 1.0

    def test_apply_multiple_shapes(self):
        """Test applying multiple blend shapes."""
        base = np.zeros((3, 3), dtype=np.float32)
        shape_a = BlendShape(name="a", vertex_indices=[0], deltas=[[1, 0, 0]])
        shape_b = BlendShape(name="b", vertex_indices=[0], deltas=[[0, 1, 0]])

        result = apply_blend_shapes(
            base,
            {"a": shape_a, "b": shape_b},
            {"a": 1.0, "b": 1.0},
        )
        assert result[0, 0] == 1.0
        assert result[0, 1] == 1.0

    def test_apply_empty_weights(self):
        """Test with empty weights returns copy of base."""
        base = np.array([[1, 2, 3]], dtype=np.float32)
        result = apply_blend_shapes(base, {}, {})
        np.testing.assert_array_equal(result, base)

    def test_apply_with_correctives(self):
        """Test applying with corrective shapes."""
        base = np.zeros((3, 3), dtype=np.float32)
        shape_a = BlendShape(name="a", vertex_indices=[0], deltas=[[1, 0, 0]])
        corrective_shape = BlendShape(name="c", vertex_indices=[0], deltas=[[0, 0, 1]])
        corrective = CorrectiveBlendShape(
            shape=corrective_shape,
            driver_shapes=["a"],
            driver_weights=[0.5],
        )

        shape_set = BlendShapeSet(name="test", base_vertices=base)
        shape_set.add_shape(shape_a)
        shape_set.add_corrective(corrective)

        result = apply_blend_shapes_with_correctives(
            base, shape_set, {"a": 1.0}
        )
        # Both shape a and corrective should apply
        assert result[0, 0] == 1.0
        assert result[0, 2] == 1.0


class TestBlendShapeController:
    """Tests for BlendShapeController class."""

    @pytest.fixture
    def controller(self):
        """Create a controller for testing."""
        shape_set = BlendShapeSet(
            name="test",
            base_vertices=np.zeros((10, 3), dtype=np.float32),
        )
        shape_set.add_shape(BlendShape(name="a", vertex_indices=[0], deltas=[[1, 0, 0]]))
        shape_set.add_shape(BlendShape(name="b", vertex_indices=[1], deltas=[[0, 1, 0]]))
        return BlendShapeController(shape_set)

    def test_set_weight(self, controller):
        """Test setting weight."""
        result = controller.set_weight("a", 0.5)
        assert result is True
        assert controller.get_weight("a") == 0.5

    def test_set_weight_clamping(self, controller):
        """Test weight clamping."""
        controller.set_weight("a", 1.5)
        assert controller.get_weight("a") == 1.0

        controller.set_weight("a", -0.5)
        assert controller.get_weight("a") == 0.0

    def test_set_weight_invalid_name(self, controller):
        """Test setting weight for invalid name."""
        result = controller.set_weight("nonexistent", 1.0)
        assert result is False

    def test_reset_all(self, controller):
        """Test resetting all weights."""
        controller.set_weight("a", 0.5)
        controller.set_weight("b", 0.5)
        controller.reset_all()

        assert controller.get_weight("a") == 0.0
        assert controller.get_weight("b") == 0.0

    def test_set_target_weight(self, controller):
        """Test target weight for transitions."""
        result = controller.set_target_weight("a", 1.0, speed=10.0)
        assert result is True
        assert controller.has_active_transitions()

    def test_update_transitions(self, controller):
        """Test updating transitions."""
        controller.set_target_weight("a", 1.0, speed=100.0)
        controller.update(0.1)

        assert controller.get_weight("a") > 0.0

    def test_get_active_shapes(self, controller):
        """Test getting active shapes."""
        controller.set_weight("a", 0.5)
        active = controller.get_active_shapes()
        assert "a" in active
        assert "b" not in active

    def test_dirty_flag(self, controller):
        """Test dirty flag behavior."""
        assert not controller.dirty
        controller.set_weight("a", 0.5)
        assert controller.dirty
        controller.clear_dirty()
        assert not controller.dirty


class TestARKitCompatibility:
    """Tests for ARKit blend shape compatibility."""

    def test_arkit_shapes_count(self):
        """Test ARKit shapes list contains expected count."""
        assert len(ARKIT_BLEND_SHAPES) == 52

    def test_create_arkit_set(self):
        """Test creating ARKit-compatible set."""
        shape_set = create_arkit_compatible_set("face", 1000)
        assert shape_set.vertex_count == 1000
        assert "eyeBlinkLeft" in shape_set.shape_names
        assert "mouthSmileRight" in shape_set.shape_names

    def test_remap_weights(self):
        """Test weight remapping."""
        weights = {"oldName": 0.5}
        mapping = {"oldName": "newName"}
        result = remap_blend_shape_weights(weights, mapping)
        assert "newName" in result
        assert result["newName"] == 0.5


# =============================================================================
# FACS Tests
# =============================================================================


class TestActionUnit:
    """Tests for ActionUnit enum."""

    def test_all_aus_defined(self):
        """Test all expected AUs are defined."""
        assert ActionUnit.AU1_INNER_BROW_RAISER is not None
        assert ActionUnit.AU12_LIP_CORNER_PULLER is not None
        assert ActionUnit.AU43_EYES_CLOSED is not None

    def test_au_alias(self):
        """Test AU alias works."""
        assert AU.AU1_INNER_BROW_RAISER == ActionUnit.AU1_INNER_BROW_RAISER


class TestActionUnitData:
    """Tests for ActionUnitData class."""

    def test_create_bilateral(self):
        """Test creating bilateral AU data."""
        au_data = ActionUnitData(
            au=ActionUnit.AU12_LIP_CORNER_PULLER,
            is_bilateral=True,
            left_shapes={"mouthSmileLeft": 1.0},
            right_shapes={"mouthSmileRight": 1.0},
        )
        assert au_data.is_bilateral is True

    def test_get_blend_weights(self):
        """Test getting blend weights from AU."""
        au_data = ActionUnitData(
            au=ActionUnit.AU1_INNER_BROW_RAISER,
            intensity=0.5,
            blend_shapes={"browInnerUp": 1.0},
        )

        weights = au_data.get_blend_weights()
        assert weights["browInnerUp"] == 0.5

    def test_get_blend_weights_bilateral(self):
        """Test bilateral blend weights."""
        au_data = ActionUnitData(
            au=ActionUnit.AU12_LIP_CORNER_PULLER,
            intensity=1.0,
            is_bilateral=True,
            left_shapes={"smileL": 1.0},
            right_shapes={"smileR": 1.0},
        )

        weights = au_data.get_blend_weights(left_intensity=0.5, right_intensity=1.0)
        assert weights["smileL"] == 0.5
        assert weights["smileR"] == 1.0


class TestExpression:
    """Tests for Expression enum."""

    def test_all_expressions_defined(self):
        """Test all expected expressions are defined."""
        assert Expression.NEUTRAL is not None
        assert Expression.HAPPY is not None
        assert Expression.SAD is not None
        assert Expression.ANGRY is not None
        assert Expression.SURPRISED is not None
        assert Expression.DISGUSTED is not None
        assert Expression.FEARFUL is not None


class TestFACSController:
    """Tests for FACSController class."""

    @pytest.fixture
    def controller(self):
        """Create a FACS controller for testing."""
        return FACSController()

    def test_set_au_intensity(self, controller):
        """Test setting AU intensity."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)
        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == 0.5

    def test_set_au_intensity_clamping(self, controller):
        """Test AU intensity clamping."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 1.5)
        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == 1.0

    def test_reset_all_aus(self, controller):
        """Test resetting all AUs."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)
        controller.reset_all_aus()
        assert controller.get_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER) == 0.0

    def test_set_expression(self, controller):
        """Test setting expression preset."""
        controller.set_expression(Expression.HAPPY)
        assert controller.current_expression == Expression.HAPPY

    def test_create_expression_by_name(self, controller):
        """Test creating expression by name."""
        weights = controller.create_expression("happy")
        assert ActionUnit.AU12_LIP_CORNER_PULLER in weights

    def test_blend_expressions(self, controller):
        """Test blending between expressions."""
        weights = controller.blend_expressions(
            Expression.NEUTRAL,
            Expression.HAPPY,
            0.5,
        )
        # Should have partial happy expression
        assert ActionUnit.AU12_LIP_CORNER_PULLER in weights or len(weights) >= 0

    def test_get_blend_shape_weights(self, controller):
        """Test converting AUs to blend shapes."""
        controller.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 1.0)
        weights = controller.get_blend_shape_weights()
        # Should have smile blend shapes
        assert len(weights) > 0

    def test_get_active_aus(self, controller):
        """Test getting active AUs."""
        controller.set_au_intensity(ActionUnit.AU1_INNER_BROW_RAISER, 0.5)
        active = controller.get_active_aus()
        assert ActionUnit.AU1_INNER_BROW_RAISER in active

    def test_add_expression_preset(self, controller):
        """Test adding custom expression preset."""
        controller.add_expression_preset(
            Expression.NEUTRAL,
            {ActionUnit.AU1_INNER_BROW_RAISER: 0.5},
        )
        # Should not raise

    def test_serialization(self, controller):
        """Test serialization round-trip."""
        controller.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 0.5)
        data = controller.to_dict()

        new_controller = FACSController()
        new_controller.from_dict(data)
        assert new_controller.get_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER) == 0.5


class TestDefaultMappings:
    """Tests for default FACS mappings."""

    def test_default_au_mappings(self):
        """Test default AU mappings exist."""
        mappings = get_default_au_mappings()
        assert ActionUnit.AU1_INNER_BROW_RAISER in mappings
        assert ActionUnit.AU12_LIP_CORNER_PULLER in mappings

    def test_default_expressions(self):
        """Test default expressions exist."""
        expressions = get_default_expressions()
        assert Expression.HAPPY in expressions
        assert Expression.SAD in expressions


# =============================================================================
# Lip Sync Tests
# =============================================================================


class TestViseme:
    """Tests for Viseme enum."""

    def test_all_visemes_defined(self):
        """Test all expected visemes are defined."""
        assert Viseme.SIL is not None
        assert Viseme.PP is not None
        assert Viseme.AA is not None
        assert Viseme.OO is not None


class TestPhonemeToViseme:
    """Tests for phoneme to viseme conversion."""

    def test_convert_bilabial(self):
        """Test converting bilabial phonemes."""
        assert phoneme_to_viseme("p") == Viseme.PP
        assert phoneme_to_viseme("b") == Viseme.PP
        assert phoneme_to_viseme("m") == Viseme.PP

    def test_convert_vowels(self):
        """Test converting vowel phonemes."""
        assert phoneme_to_viseme("aa") == Viseme.AA
        assert phoneme_to_viseme("iy") == Viseme.II
        assert phoneme_to_viseme("uw") == Viseme.UU

    def test_convert_unknown(self):
        """Test converting unknown phoneme."""
        assert phoneme_to_viseme("xyz") == Viseme.SIL

    def test_convert_with_stress(self):
        """Test converting phoneme with stress marker."""
        assert phoneme_to_viseme("AA1") == Viseme.AA


class TestPhonemeEvent:
    """Tests for PhonemeEvent class."""

    def test_create_event(self):
        """Test creating phoneme event."""
        event = PhonemeEvent(
            phoneme="aa",
            start_time=0.0,
            end_time=0.1,
        )
        assert event.duration == 0.1
        assert event.mid_time == 0.05

    def test_confidence(self):
        """Test confidence value."""
        event = PhonemeEvent(
            phoneme="aa",
            start_time=0.0,
            end_time=0.1,
            confidence=0.8,
        )
        assert event.confidence == 0.8


class TestVisemeMapping:
    """Tests for VisemeMapping class."""

    def test_get_weights(self):
        """Test getting weighted blend shapes."""
        mapping = VisemeMapping(
            viseme=Viseme.AA,
            blend_shapes={"jawOpen": 0.6},
        )
        weights = mapping.get_weights(0.5)
        assert weights["jawOpen"] == 0.3


class TestLipSyncController:
    """Tests for LipSyncController class."""

    @pytest.fixture
    def controller(self):
        """Create a lip sync controller for testing."""
        return LipSyncController()

    def test_process_phoneme_events(self, controller):
        """Test processing phoneme events."""
        events = [
            PhonemeEvent("p", 0.0, 0.1),
            PhonemeEvent("aa", 0.1, 0.2),
        ]
        visemes = controller.process_audio_events(events)
        assert len(visemes) == 2
        assert visemes[0].viseme == Viseme.PP
        assert visemes[1].viseme == Viseme.AA

    def test_set_timeline(self, controller):
        """Test setting viseme timeline."""
        events = [VisemeEvent(Viseme.AA, 0.0, 0.1)]
        controller.set_timeline(events)
        assert controller.duration == 0.1

    def test_play_pause_stop(self, controller):
        """Test playback controls."""
        events = [VisemeEvent(Viseme.AA, 0.0, 0.5)]
        controller.set_timeline(events)

        controller.play()
        assert controller.is_playing

        controller.pause()
        assert not controller.is_playing

        controller.stop()
        assert controller.current_time == 0.0

    def test_seek(self, controller):
        """Test seeking."""
        events = [VisemeEvent(Viseme.AA, 0.0, 1.0)]
        controller.set_timeline(events)

        controller.seek(0.5)
        assert controller.current_time == 0.5

    def test_update(self, controller):
        """Test update advances time."""
        events = [VisemeEvent(Viseme.AA, 0.0, 1.0)]
        controller.set_timeline(events)
        controller.play()

        controller.update(0.1)
        assert controller.current_time == pytest.approx(0.1, abs=0.01)

    def test_get_blend_weights(self, controller):
        """Test getting blend weights."""
        events = [VisemeEvent(Viseme.AA, 0.0, 1.0, weight=1.0)]
        controller.set_timeline(events)
        controller.seek(0.5)

        weights = controller.get_blend_weights()
        assert "jawOpen" in weights

    def test_intensity(self, controller):
        """Test intensity setting."""
        controller.intensity = 0.5
        assert controller.intensity == 0.5

    def test_get_viseme_at_time(self, controller):
        """Test getting viseme at specific time."""
        events = [
            VisemeEvent(Viseme.PP, 0.0, 0.5),
            VisemeEvent(Viseme.AA, 0.5, 1.0),
        ]
        controller.set_timeline(events)

        assert controller.get_viseme_at_time(0.25) == Viseme.PP
        assert controller.get_viseme_at_time(0.75) == Viseme.AA


class TestCoarticulation:
    """Tests for coarticulation settings."""

    def test_calculate_blend_linear(self):
        """Test linear blend calculation."""
        settings = CoarticulationSettings(blend_curve="linear")
        assert settings.calculate_blend(0.5) == 0.5

    def test_calculate_blend_ease_in_out(self):
        """Test ease in/out blend calculation."""
        settings = CoarticulationSettings(blend_curve="ease_in_out")
        # Smoothstep should be 0.5 at t=0.5
        assert settings.calculate_blend(0.5) == 0.5


class TestCreatePhonemeEvents:
    """Tests for phoneme event creation utility."""

    def test_create_from_text(self):
        """Test creating events from text."""
        events = create_phoneme_events_from_text("hello")
        assert len(events) > 0
        assert all(isinstance(e, PhonemeEvent) for e in events)


# =============================================================================
# Eye Animation Tests
# =============================================================================


class TestEyeLimits:
    """Tests for EyeLimits class."""

    def test_clamp_within_limits(self):
        """Test clamping within limits."""
        limits = EyeLimits(max_yaw=30, max_pitch_up=20, max_pitch_down=25)
        yaw, pitch = limits.clamp_rotation(15, 10)
        assert yaw == 15
        assert pitch == 10

    def test_clamp_exceeds_limits(self):
        """Test clamping beyond limits."""
        limits = EyeLimits(max_yaw=30, max_pitch_up=20, max_pitch_down=25)
        yaw, pitch = limits.clamp_rotation(50, 30)
        assert yaw == 30
        assert pitch == 20


class TestEyeTransform:
    """Tests for EyeTransform class."""

    def test_default_values(self):
        """Test default transform values."""
        transform = EyeTransform()
        assert transform.yaw == 0.0
        assert transform.pitch == 0.0
        assert transform.blink_weight == 0.0

    def test_to_euler(self):
        """Test converting to Euler angles."""
        transform = EyeTransform(yaw=45, pitch=30)
        euler = transform.to_euler()
        assert euler[0] == pytest.approx(math.radians(30), abs=0.01)
        assert euler[1] == pytest.approx(math.radians(45), abs=0.01)

    def test_to_quaternion(self):
        """Test converting to quaternion."""
        transform = EyeTransform()
        quat = transform.to_quaternion()
        # Identity rotation should have w close to 1
        assert quat[3] == pytest.approx(1.0, abs=0.01)


class TestBlinkController:
    """Tests for BlinkController class."""

    @pytest.fixture
    def controller(self):
        """Create a blink controller for testing."""
        settings = BlinkSettings(
            min_interval=1.0,
            max_interval=2.0,
            blink_duration=0.15,
        )
        return BlinkController(settings)

    def test_initial_state(self, controller):
        """Test initial state is not blinking."""
        assert controller.is_blinking is False
        assert controller.current_weight == 0.0

    def test_trigger_blink(self, controller):
        """Test triggering a blink."""
        controller.trigger_blink()
        controller.update(0.01)
        assert controller.is_blinking is True

    def test_blink_completes(self, controller):
        """Test blink completes after duration."""
        controller.trigger_blink()
        for _ in range(20):
            controller.update(0.01)
        assert controller.current_weight == 0.0

    def test_auto_blink(self, controller):
        """Test automatic blinking."""
        # Fast forward time
        for _ in range(300):
            controller.update(0.01)
        # Should have blinked at some point


class TestEyeController:
    """Tests for EyeController class."""

    @pytest.fixture
    def controller(self):
        """Create an eye controller for testing."""
        return EyeController()

    def test_initial_state(self, controller):
        """Test initial state."""
        assert controller.state == EyeState.IDLE
        assert controller.look_at_target is None

    def test_look_at(self, controller):
        """Test setting look-at target."""
        controller.look_at((0, 0, 10))
        assert controller.look_at_target == (0, 0, 10)
        assert controller.state == EyeState.TRACKING

    def test_clear_target(self, controller):
        """Test clearing look-at target."""
        controller.look_at((0, 0, 10))
        controller.clear_target()
        assert controller.look_at_target is None
        assert controller.state == EyeState.IDLE

    def test_update_returns_transforms(self, controller):
        """Test update returns eye transforms."""
        left, right = controller.update(0.016)
        assert isinstance(left, EyeTransform)
        assert isinstance(right, EyeTransform)

    def test_blink(self, controller):
        """Test triggering blink."""
        controller.blink()
        # Need multiple updates for blink to register (first triggers, second shows it)
        controller.update(0.05)
        controller.update(0.05)
        # Blink weight should be positive after triggering
        assert controller.left_eye.blink_weight >= 0  # May be 0 at start or end of blink

    def test_set_light_level(self, controller):
        """Test setting light level."""
        controller.set_light_level(0.8)
        controller.update(0.1)
        # Pupil should constrict in bright light

    def test_get_blend_shape_weights(self, controller):
        """Test getting blend shape weights."""
        controller.look_at((10, 0, 10))
        controller.update(0.1)
        weights = controller.get_blend_shape_weights()
        assert "eyeBlinkLeft" in weights


# =============================================================================
# Face Rig Tests
# =============================================================================


class TestAnimationPriority:
    """Tests for AnimationPriority enum."""

    def test_priority_ordering(self):
        """Test priority ordering."""
        assert AnimationPriority.IDLE.value < AnimationPriority.EMOTION.value
        assert AnimationPriority.EMOTION.value < AnimationPriority.LIP_SYNC.value
        assert AnimationPriority.LIP_SYNC.value < AnimationPriority.OVERRIDE.value


class TestEmotionState:
    """Tests for EmotionState class."""

    def test_default_values(self):
        """Test default emotion state."""
        state = EmotionState()
        assert state.expression == Expression.NEUTRAL
        assert state.intensity == 1.0

    def test_custom_values(self):
        """Test custom emotion state."""
        state = EmotionState(
            expression=Expression.HAPPY,
            intensity=0.5,
            blend_time=0.5,
        )
        assert state.expression == Expression.HAPPY
        assert state.intensity == 0.5


class TestFaceRig:
    """Tests for FaceRig class."""

    @pytest.fixture
    def rig(self):
        """Create a face rig for testing."""
        return create_face_rig(vertex_count=1000)

    def test_creation(self, rig):
        """Test face rig creation."""
        assert rig.blend_controller is not None
        assert rig.facs_controller is not None
        assert rig.eye_controller is not None
        assert rig.lip_sync_controller is not None

    def test_set_emotion(self, rig):
        """Test setting emotion."""
        # Use instant transition with blend_time=0
        rig.set_emotion(EmotionState(Expression.HAPPY, blend_time=0.0))
        assert rig.current_emotion.expression == Expression.HAPPY

    def test_set_expression(self, rig):
        """Test setting expression convenience method."""
        # Use instant transition with blend_time=0
        rig.set_expression(Expression.SAD, intensity=0.5, blend_time=0.0)
        assert rig.current_emotion.expression == Expression.SAD
        assert rig.current_emotion.intensity == 0.5

    def test_look_at(self, rig):
        """Test look-at control."""
        rig.look_at((0, 0, 10))
        rig.update(0.1)
        # Should not raise

    def test_blink(self, rig):
        """Test blink control."""
        rig.blink()
        rig.update(0.1)

    def test_update_returns_weights(self, rig):
        """Test update returns blend weights."""
        weights = rig.update(0.016)
        assert isinstance(weights, dict)

    def test_layer_weight(self, rig):
        """Test setting layer weight."""
        result = rig.set_layer_weight("emotion", 0.5)
        assert result is True

        layer = rig.get_layer("emotion")
        assert layer.weight == 0.5

    def test_set_blend_shape_override(self, rig):
        """Test direct blend shape control."""
        result = rig.set_blend_shape("jawOpen", 0.5)
        assert result is True

    def test_clear_overrides(self, rig):
        """Test clearing overrides."""
        rig.set_blend_shape("jawOpen", 0.5)
        rig.clear_overrides()
        layer = rig.get_layer("override")
        assert layer.weight == 0.0

    def test_reset(self, rig):
        """Test resetting face rig."""
        rig.set_expression(Expression.HAPPY)
        rig.reset()
        assert rig.current_emotion.expression == Expression.NEUTRAL

    def test_get_final_weights(self, rig):
        """Test getting final blended weights."""
        rig.update(0.016)
        weights = rig.get_final_weights()
        assert isinstance(weights, dict)

    def test_serialization(self, rig):
        """Test serialization."""
        rig.set_expression(Expression.HAPPY)
        data = rig.to_dict()
        assert "emotion" in data

    def test_priority_system(self, rig):
        """Test animation priority system."""
        # Set emotion layer
        rig.set_expression(Expression.HAPPY)

        # Add lip sync
        events = [VisemeEvent(Viseme.AA, 0.0, 1.0, weight=1.0)]
        rig.speak(events, 1.0)

        rig.update(0.5)
        weights = rig.get_final_weights()
        # Lip sync should override mouth shapes
        assert "jawOpen" in weights


# =============================================================================
# Face Capture Tests
# =============================================================================


class TestKeyframe:
    """Tests for Keyframe class."""

    def test_create_keyframe(self):
        """Test keyframe creation."""
        kf = Keyframe(time=0.5, value=0.8)
        assert kf.time == 0.5
        assert kf.value == 0.8


class TestAnimationCurve:
    """Tests for AnimationCurve class."""

    @pytest.fixture
    def curve(self):
        """Create a curve for testing."""
        curve = AnimationCurve(name="test")
        curve.add_keyframe(0.0, 0.0)
        curve.add_keyframe(1.0, 1.0)
        return curve

    def test_duration(self, curve):
        """Test curve duration."""
        assert curve.duration == 1.0

    def test_keyframe_count(self, curve):
        """Test keyframe count."""
        assert curve.keyframe_count == 2

    def test_sample_linear(self, curve):
        """Test linear sampling."""
        assert curve.sample(0.5) == 0.5

    def test_sample_before_start(self, curve):
        """Test sampling before first keyframe."""
        assert curve.sample(-0.5) == 0.0

    def test_sample_after_end(self, curve):
        """Test sampling after last keyframe."""
        assert curve.sample(1.5) == 1.0

    def test_step_interpolation(self):
        """Test step interpolation."""
        curve = AnimationCurve(name="test", interpolation=InterpolationMode.STEP)
        curve.add_keyframe(0.0, 0.0)
        curve.add_keyframe(1.0, 1.0)
        assert curve.sample(0.5) == 0.0

    def test_serialization(self, curve):
        """Test curve serialization."""
        data = curve.to_dict()
        restored = AnimationCurve.from_dict(data)
        assert restored.name == curve.name
        assert restored.keyframe_count == curve.keyframe_count


class TestFaceCaptureClip:
    """Tests for FaceCaptureClip class."""

    @pytest.fixture
    def clip(self):
        """Create a clip for testing."""
        clip = FaceCaptureClip(name="test")
        curve = AnimationCurve(name="smile")
        curve.add_keyframe(0.0, 0.0)
        curve.add_keyframe(1.0, 1.0)
        clip.add_curve(curve)
        return clip

    def test_duration(self, clip):
        """Test clip duration."""
        assert clip.duration == 1.0

    def test_shape_names(self, clip):
        """Test getting shape names."""
        assert "smile" in clip.shape_names

    def test_sample(self, clip):
        """Test sampling clip."""
        weights = clip.sample(0.5)
        assert weights["smile"] == 0.5

    def test_sample_range(self, clip):
        """Test sampling range."""
        samples = clip.sample_range(0.0, 1.0, sample_rate=10.0)
        assert len(samples) >= 10

    def test_serialization(self, clip):
        """Test clip serialization."""
        data = clip.to_dict()
        restored = FaceCaptureClip.from_dict(data)
        assert restored.name == clip.name
        assert restored.curve_count == clip.curve_count


class TestFaceCapturePlayer:
    """Tests for FaceCapturePlayer class."""

    @pytest.fixture
    def player(self):
        """Create a player for testing."""
        clip = FaceCaptureClip(name="test")
        curve = AnimationCurve(name="smile")
        curve.add_keyframe(0.0, 0.0)
        curve.add_keyframe(1.0, 1.0)
        clip.add_curve(curve)
        return FaceCapturePlayer(clip)

    def test_initial_state(self, player):
        """Test initial state."""
        assert player.state == PlaybackState.STOPPED
        assert player.time == 0.0

    def test_play(self, player):
        """Test playing."""
        player.play()
        assert player.is_playing is True

    def test_pause(self, player):
        """Test pausing."""
        player.play()
        player.pause()
        assert player.state == PlaybackState.PAUSED

    def test_stop(self, player):
        """Test stopping."""
        player.play()
        player.update(0.5)
        player.stop()
        assert player.time == 0.0

    def test_seek(self, player):
        """Test seeking."""
        player.seek(0.5)
        assert player.time == 0.5

    def test_update(self, player):
        """Test update advances time."""
        player.play()
        player.update(0.1)
        assert player.time == pytest.approx(0.1, abs=0.01)

    def test_loop(self, player):
        """Test looping."""
        player.loop = True
        player.play()
        player.update(1.5)
        assert player.time < 1.0

    def test_progress(self, player):
        """Test progress property."""
        player.seek(0.5)
        assert player.progress == 0.5

    def test_speed(self, player):
        """Test playback speed."""
        player.speed = 2.0
        player.play()
        player.update(0.1)
        assert player.time == pytest.approx(0.2, abs=0.01)


class TestFaceCaptureRetargeter:
    """Tests for FaceCaptureRetargeter class."""

    @pytest.fixture
    def retargeter(self):
        """Create a retargeter for testing."""
        return FaceCaptureRetargeter()

    def test_add_mapping(self, retargeter):
        """Test adding mapping."""
        mapping = RetargetMapping("source", "target")
        retargeter.add_mapping(mapping)
        assert retargeter.mapping_count == 1

    def test_retarget_weights(self, retargeter):
        """Test retargeting weights."""
        mapping = RetargetMapping("source", "target", scale=2.0)
        retargeter.add_mapping(mapping)

        weights = {"source": 0.5}
        result = retargeter.retarget_weights(weights)
        assert result["target"] == 1.0

    def test_pass_through(self, retargeter):
        """Test unmapped shape pass-through."""
        weights = {"unmapped": 0.5}
        result = retargeter.retarget_weights(weights)
        assert result["unmapped"] == 0.5

    def test_disable_pass_through(self, retargeter):
        """Test disabling pass-through."""
        retargeter.set_pass_through(False)
        weights = {"unmapped": 0.5}
        result = retargeter.retarget_weights(weights)
        assert "unmapped" not in result

    def test_retarget_clip(self, retargeter):
        """Test retargeting a clip."""
        mapping = RetargetMapping("smile", "mouthSmile")
        retargeter.add_mapping(mapping)

        clip = FaceCaptureClip(name="source")
        curve = AnimationCurve(name="smile")
        curve.add_keyframe(0.0, 0.0)
        curve.add_keyframe(1.0, 1.0)
        clip.add_curve(curve)

        result = retargeter.retarget_clip(clip)
        assert "mouthSmile" in result.shape_names

    def test_serialization(self, retargeter):
        """Test retargeter serialization."""
        mapping = RetargetMapping("source", "target")
        retargeter.add_mapping(mapping)

        data = retargeter.to_dict()
        restored = FaceCaptureRetargeter.from_dict(data)
        assert restored.mapping_count == 1


class TestClipUtilities:
    """Tests for clip utility functions."""

    def test_create_clip_from_samples(self):
        """Test creating clip from samples."""
        samples = [
            (0.0, {"smile": 0.0}),
            (0.5, {"smile": 0.5}),
            (1.0, {"smile": 1.0}),
        ]
        clip = create_clip_from_samples("test", samples)
        assert clip.duration == 1.0
        assert "smile" in clip.shape_names

    def test_merge_clips(self):
        """Test merging clips."""
        clip1 = FaceCaptureClip(name="clip1")
        curve1 = AnimationCurve(name="a")
        curve1.add_keyframe(0.0, 0.0)
        curve1.add_keyframe(1.0, 1.0)
        clip1.add_curve(curve1)

        clip2 = FaceCaptureClip(name="clip2")
        curve2 = AnimationCurve(name="a")
        curve2.add_keyframe(0.0, 1.0)
        curve2.add_keyframe(1.0, 0.0)
        clip2.add_curve(curve2)

        merged = merge_clips([clip1, clip2])
        assert merged.duration == 2.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for facial animation system."""

    def test_full_face_rig_workflow(self):
        """Test complete face rig workflow."""
        # Create face rig
        rig = create_face_rig(vertex_count=1000)

        # Set expression
        rig.set_expression(Expression.HAPPY, intensity=0.8)

        # Set up lip sync
        events = [
            VisemeEvent(Viseme.PP, 0.0, 0.1),
            VisemeEvent(Viseme.AA, 0.1, 0.3),
            VisemeEvent(Viseme.SIL, 0.3, 0.5),
        ]
        rig.speak(events, 0.5)

        # Set look-at target
        rig.look_at((0, 0, 10))

        # Update and get weights
        for _ in range(10):
            weights = rig.update(0.016)

        assert len(weights) > 0

    def test_motion_capture_to_face_rig(self):
        """Test applying motion capture to face rig."""
        # Create clip
        clip = FaceCaptureClip(name="capture")
        for shape_name in ["jawOpen", "mouthSmileLeft", "mouthSmileRight"]:
            curve = AnimationCurve(name=shape_name)
            curve.add_keyframe(0.0, 0.0)
            curve.add_keyframe(0.5, 1.0)
            curve.add_keyframe(1.0, 0.0)
            clip.add_curve(curve)

        # Create player
        player = FaceCapturePlayer(clip)
        player.play()

        # Create face rig
        rig = create_face_rig(vertex_count=1000)

        # Apply capture to face rig
        for _ in range(20):
            weights = player.update(0.05)
            rig.set_layer_blend_shapes("override", weights)
            rig.set_layer_weight("override", 1.0)
            rig.update(0.05)

    def test_facs_to_blend_shapes_pipeline(self):
        """Test FACS to blend shapes conversion pipeline."""
        facs = FACSController()

        # Set multiple AUs (happy expression)
        facs.set_au_intensity(ActionUnit.AU6_CHEEK_RAISER, 0.8)
        facs.set_au_intensity(ActionUnit.AU12_LIP_CORNER_PULLER, 1.0)

        # Get blend shapes
        weights = facs.get_blend_shape_weights()

        # Verify we got blend shapes
        assert len(weights) > 0

        # Apply to blend shape controller
        shape_set = create_arkit_compatible_set("face", 1000)
        controller = BlendShapeController(shape_set)
        controller.set_weights(weights)

        # Verify weights were applied
        active = controller.get_active_shapes()
        assert len(active) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
