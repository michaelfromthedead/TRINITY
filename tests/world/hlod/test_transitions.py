"""
Tests for HLOD transition handling.

Tests LOD transitions, blend calculations, screen space error, and visibility.
"""

import math
import pytest

from engine.world.hlod.transitions import (
    TransitionConstants,
    TransitionMode,
    TransitionState,
    TransitionSettings,
    LODTransition,
    TransitionCalculator,
    ScreenSpaceError,
    HLODTransitionManager,
    VisibilityResult,
    HLODVisibilitySystem,
)
from engine.world.hlod.generator import AABB, Vec3


# =============================================================================
# TRANSITION SETTINGS TESTS
# =============================================================================


class TestTransitionSettings:
    """Tests for TransitionSettings."""

    def test_default_creation(self) -> None:
        """Test default settings creation."""
        settings = TransitionSettings()
        assert settings.mode == TransitionMode.DITHERED
        assert settings.transition_range == TransitionConstants.DEFAULT_TRANSITION_RANGE
        assert settings.dither_scale == TransitionConstants.DEFAULT_DITHER_SCALE

    def test_custom_creation(self) -> None:
        """Test custom settings creation."""
        settings = TransitionSettings(
            mode=TransitionMode.CROSSFADE,
            transition_range=100.0,
            dither_scale=2.0,
            morph_speed=10.0,
        )
        assert settings.mode == TransitionMode.CROSSFADE
        assert settings.transition_range == 100.0
        assert settings.dither_scale == 2.0
        assert settings.morph_speed == 10.0

    def test_invalid_transition_range(self) -> None:
        """Test invalid transition range."""
        with pytest.raises(ValueError):
            TransitionSettings(transition_range=-10.0)

    def test_invalid_dither_scale(self) -> None:
        """Test invalid dither scale."""
        with pytest.raises(ValueError):
            TransitionSettings(dither_scale=0.0)

        with pytest.raises(ValueError):
            TransitionSettings(dither_scale=-1.0)

    def test_invalid_morph_speed(self) -> None:
        """Test invalid morph speed."""
        with pytest.raises(ValueError):
            TransitionSettings(morph_speed=0.0)

    def test_invalid_hysteresis_factor(self) -> None:
        """Test invalid hysteresis factor."""
        with pytest.raises(ValueError):
            TransitionSettings(hysteresis_factor=-0.1)

        with pytest.raises(ValueError):
            TransitionSettings(hysteresis_factor=1.5)


# =============================================================================
# LOD TRANSITION TESTS
# =============================================================================


class TestLODTransition:
    """Tests for LODTransition."""

    def test_default_creation(self) -> None:
        """Test default transition creation."""
        transition = LODTransition()
        assert transition.from_lod == 0
        assert transition.to_lod == 0
        assert transition.blend_factor == 0.0
        assert transition.state == TransitionState.STABLE

    def test_start_transition(self) -> None:
        """Test starting a transition."""
        transition = LODTransition()
        transition.start(
            from_lod=0,
            to_lod=1,
            start_distance=100.0,
            end_distance=150.0,
        )

        assert transition.from_lod == 0
        assert transition.to_lod == 1
        assert transition.blend_factor == 0.0
        assert transition.state == TransitionState.TRANSITIONING
        assert transition.is_active

    def test_update_transition(self) -> None:
        """Test updating transition progress."""
        transition = LODTransition()
        transition.start(
            from_lod=0,
            to_lod=1,
            start_distance=100.0,
            end_distance=200.0,
        )

        # At midpoint
        blend = transition.update(150.0, [100.0, 200.0])
        assert blend == pytest.approx(0.5)
        assert transition.is_active

    def test_update_transition_complete(self) -> None:
        """Test transition completion."""
        transition = LODTransition()
        transition.start(
            from_lod=0,
            to_lod=1,
            start_distance=100.0,
            end_distance=200.0,
        )

        # Past end distance
        blend = transition.update(250.0, [100.0, 200.0])
        assert blend == pytest.approx(1.0)
        assert transition.is_complete

    def test_complete_transition(self) -> None:
        """Test manually completing transition."""
        transition = LODTransition()
        transition.start(0, 1, 100.0, 200.0)

        transition.complete()

        assert transition.blend_factor == 1.0
        assert transition.state == TransitionState.COMPLETE

    def test_cancel_transition(self) -> None:
        """Test canceling transition."""
        transition = LODTransition()
        transition.start(0, 1, 100.0, 200.0)
        transition.update(150.0, [100.0, 200.0])

        transition.cancel()

        assert transition.blend_factor == 0.0
        assert transition.state == TransitionState.STABLE

    def test_reset_transition(self) -> None:
        """Test resetting transition."""
        transition = LODTransition()
        transition.start(0, 1, 100.0, 200.0)
        transition.complete()

        transition.reset()

        assert transition.from_lod == 1  # Now at to_lod
        assert transition.state == TransitionState.STABLE

    def test_current_lod(self) -> None:
        """Test current LOD calculation."""
        transition = LODTransition()
        transition.start(0, 1, 100.0, 200.0)

        # Blend < 0.5, should be from_lod
        transition.update(125.0, [100.0, 200.0])
        assert transition.current_lod == 0

        # Blend > 0.5, should be to_lod
        transition.update(175.0, [100.0, 200.0])
        assert transition.current_lod == 1


# =============================================================================
# TRANSITION CALCULATOR TESTS
# =============================================================================


class TestTransitionCalculator:
    """Tests for TransitionCalculator."""

    @pytest.fixture
    def calculator(self) -> TransitionCalculator:
        """Create a transition calculator."""
        return TransitionCalculator(TransitionSettings(
            mode=TransitionMode.DITHERED,
            transition_range=50.0,
        ))

    def test_calculate_blend_instant(self) -> None:
        """Test blend calculation with instant mode."""
        calculator = TransitionCalculator(TransitionSettings(
            mode=TransitionMode.INSTANT,
        ))

        # Before midpoint (midpoint = (50+100)/2 = 75)
        blend = calculator.calculate_blend(70.0, 50.0, 100.0)
        assert blend == 0.0

        # At or after midpoint
        blend = calculator.calculate_blend(75.0, 50.0, 100.0)
        assert blend == 1.0

        # After midpoint
        blend = calculator.calculate_blend(80.0, 50.0, 100.0)
        assert blend == 1.0

    def test_calculate_blend_smooth(self, calculator: TransitionCalculator) -> None:
        """Test blend calculation with smooth transition."""
        # Before transition zone
        blend = calculator.calculate_blend(50.0, 50.0, 100.0)
        assert blend == 0.0

        # In transition zone
        blend = calculator.calculate_blend(75.0, 50.0, 100.0)
        assert 0.0 < blend < 1.0

        # After transition zone
        blend = calculator.calculate_blend(150.0, 50.0, 100.0)
        assert blend == 1.0

    def test_calculate_blend_range(self, calculator: TransitionCalculator) -> None:
        """Test blend is always in [0, 1] range."""
        for dist in range(0, 200, 10):
            blend = calculator.calculate_blend(float(dist), 50.0, 100.0)
            assert 0.0 <= blend <= 1.0

    def test_get_dither_pattern(self, calculator: TransitionCalculator) -> None:
        """Test dither pattern generation."""
        # At 0% blend, all pixels should be low LOD
        for x in range(4):
            for y in range(4):
                result = calculator.get_dither_pattern(0.0, x, y)
                assert not result

        # At 100% blend, all pixels should be high LOD
        for x in range(4):
            for y in range(4):
                result = calculator.get_dither_pattern(1.0, x, y)
                assert result

    def test_dither_pattern_gradual(self, calculator: TransitionCalculator) -> None:
        """Test that dither pattern gradually transitions."""
        # Count pixels showing high LOD at different blend levels
        def count_high_lod(blend: float) -> int:
            count = 0
            for x in range(4):
                for y in range(4):
                    if calculator.get_dither_pattern(blend, x, y):
                        count += 1
            return count

        # More pixels should show high LOD as blend increases
        count_25 = count_high_lod(0.25)
        count_50 = count_high_lod(0.50)
        count_75 = count_high_lod(0.75)

        assert count_25 <= count_50 <= count_75

    def test_get_morph_factor(self) -> None:
        """Test morph factor calculation."""
        calculator = TransitionCalculator(TransitionSettings(
            mode=TransitionMode.MORPHING,
        ))

        # Should apply smooth step
        morph = calculator.get_morph_factor(0.5, 0)
        assert 0.0 < morph < 1.0

        morph = calculator.get_morph_factor(0.0, 0)
        assert morph == pytest.approx(0.0)

        morph = calculator.get_morph_factor(1.0, 0)
        assert morph == pytest.approx(1.0)


# =============================================================================
# SCREEN SPACE ERROR TESTS
# =============================================================================


class TestScreenSpaceError:
    """Tests for ScreenSpaceError."""

    @pytest.fixture
    def error_calc(self) -> ScreenSpaceError:
        """Create a screen space error calculator."""
        return ScreenSpaceError(error_threshold=2.0)

    def test_calculate_error(self, error_calc: ScreenSpaceError) -> None:
        """Test error calculation."""
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        camera_pos = Vec3(0, 0, 10)  # 10 units away

        error = error_calc.calculate_error(
            bounds,
            camera_pos,
            fov=math.radians(60),
            screen_height=1080,
        )

        assert error > 0

    def test_error_increases_with_closer_distance(
        self,
        error_calc: ScreenSpaceError,
    ) -> None:
        """Test that error increases as object gets closer."""
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        fov = math.radians(60)

        error_far = error_calc.calculate_error(
            bounds,
            Vec3(0, 0, 100),
            fov,
            1080,
        )
        error_close = error_calc.calculate_error(
            bounds,
            Vec3(0, 0, 10),
            fov,
            1080,
        )

        assert error_close > error_far

    def test_error_at_zero_distance(self, error_calc: ScreenSpaceError) -> None:
        """Test error at zero distance returns infinity."""
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))

        error = error_calc.calculate_error(
            bounds,
            Vec3(0, 0, 0),  # At center
            math.radians(60),
            1080,
        )

        assert error == float("inf")

    def test_calculate_error_from_radius(
        self,
        error_calc: ScreenSpaceError,
    ) -> None:
        """Test error calculation from bounding radius."""
        error = error_calc.calculate_error_from_radius(
            radius=1.0,
            distance=10.0,
            fov=math.radians(60),
            screen_height=1080,
        )

        assert error > 0

    def test_get_lod_for_error(self, error_calc: ScreenSpaceError) -> None:
        """Test LOD selection based on error."""
        thresholds = [100.0, 50.0, 25.0, 10.0]

        # Large error = close = low LOD index
        lod = error_calc.get_lod_for_error(200.0, thresholds)
        assert lod == 0

        # Small error = far = high LOD index
        lod = error_calc.get_lod_for_error(5.0, thresholds)
        assert lod == 4  # Beyond all thresholds

    def test_compute_optimal_distance(
        self,
        error_calc: ScreenSpaceError,
    ) -> None:
        """Test computing optimal distance for target error."""
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))

        distance = error_calc.compute_optimal_distance(
            bounds,
            target_error=100.0,
            fov=math.radians(60),
            screen_height=1080,
        )

        assert distance > 0

    def test_optimal_distance_zero_error(
        self,
        error_calc: ScreenSpaceError,
    ) -> None:
        """Test optimal distance with zero target error."""
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))

        distance = error_calc.compute_optimal_distance(
            bounds,
            target_error=0.0,
            fov=math.radians(60),
            screen_height=1080,
        )

        assert distance == float("inf")

    def test_error_threshold_property(self, error_calc: ScreenSpaceError) -> None:
        """Test error threshold property."""
        assert error_calc.error_threshold == 2.0

        error_calc.error_threshold = 5.0
        assert error_calc.error_threshold == 5.0

    def test_invalid_error_threshold(self, error_calc: ScreenSpaceError) -> None:
        """Test invalid error threshold."""
        with pytest.raises(ValueError):
            error_calc.error_threshold = 0.0


# =============================================================================
# HLOD TRANSITION MANAGER TESTS
# =============================================================================


class TestHLODTransitionManager:
    """Tests for HLODTransitionManager."""

    @pytest.fixture
    def manager(self) -> HLODTransitionManager:
        """Create a transition manager."""
        return HLODTransitionManager(TransitionSettings(
            mode=TransitionMode.DITHERED,
            transition_range=50.0,
        ))

    @pytest.fixture
    def cell_bounds(self) -> dict:
        """Create cell bounds for testing."""
        return {
            (0, 0): AABB(Vec3(0, 0, 0), Vec3(100, 100, 100)),
            (1, 0): AABB(Vec3(200, 0, 0), Vec3(300, 100, 100)),
        }

    @pytest.fixture
    def lod_thresholds(self) -> list:
        """Create LOD thresholds."""
        return [100.0, 200.0, 400.0, 800.0]

    def test_creation(self, manager: HLODTransitionManager) -> None:
        """Test manager creation."""
        assert manager.settings.mode == TransitionMode.DITHERED
        assert len(manager.active_transitions) == 0

    def test_update_triggers_transition(
        self,
        manager: HLODTransitionManager,
        cell_bounds: dict,
        lod_thresholds: list,
    ) -> None:
        """Test that update triggers transitions."""
        # Camera far from cell (0,0) center
        camera_pos = Vec3(50, 50, 500)

        manager.update(camera_pos, cell_bounds, lod_thresholds)

        # Should have created transitions for cells
        assert len(manager.active_transitions) == 2

    def test_get_transition_state(
        self,
        manager: HLODTransitionManager,
    ) -> None:
        """Test getting transition state."""
        manager.start_transition(
            (0, 0),
            from_lod=0,
            to_lod=1,
            start_distance=100.0,
            end_distance=200.0,
        )

        from_lod, to_lod, blend = manager.get_transition_state((0, 0))

        assert from_lod == 0
        assert to_lod == 1
        assert blend == 0.0

    def test_get_transition_state_no_transition(
        self,
        manager: HLODTransitionManager,
    ) -> None:
        """Test getting state for cell with no transition."""
        from_lod, to_lod, blend = manager.get_transition_state((0, 0))

        assert from_lod == 0
        assert to_lod == 0
        assert blend == 0.0

    def test_start_transition(self, manager: HLODTransitionManager) -> None:
        """Test manually starting transition."""
        manager.start_transition(
            (0, 0),
            from_lod=0,
            to_lod=2,
            start_distance=50.0,
            end_distance=150.0,
        )

        assert manager.is_transitioning((0, 0))

    def test_complete_transition(self, manager: HLODTransitionManager) -> None:
        """Test force completing transition."""
        manager.start_transition((0, 0), 0, 1, 100.0, 200.0)

        manager.complete_transition((0, 0))

        assert not manager.is_transitioning((0, 0))
        assert manager.get_current_lod((0, 0)) == 1

    def test_cancel_transition(self, manager: HLODTransitionManager) -> None:
        """Test canceling transition."""
        manager.start_transition((0, 0), 0, 1, 100.0, 200.0)

        manager.cancel_transition((0, 0))

        assert not manager.is_transitioning((0, 0))

    def test_is_transitioning(self, manager: HLODTransitionManager) -> None:
        """Test transition state check."""
        assert not manager.is_transitioning((0, 0))

        manager.start_transition((0, 0), 0, 1, 100.0, 200.0)
        assert manager.is_transitioning((0, 0))

    def test_get_current_lod(self, manager: HLODTransitionManager) -> None:
        """Test getting current LOD."""
        assert manager.get_current_lod((0, 0)) == 0

        manager.start_transition((0, 0), 0, 2, 100.0, 200.0)
        manager.complete_transition((0, 0))

        assert manager.get_current_lod((0, 0)) == 2

    def test_clear_transitions(self, manager: HLODTransitionManager) -> None:
        """Test clearing all transitions."""
        manager.start_transition((0, 0), 0, 1, 100.0, 200.0)
        manager.start_transition((1, 0), 0, 1, 100.0, 200.0)

        manager.clear_transitions()

        assert len(manager.active_transitions) == 0

    def test_remove_cell(self, manager: HLODTransitionManager) -> None:
        """Test removing cell from tracking."""
        manager.start_transition((0, 0), 0, 1, 100.0, 200.0)
        manager.complete_transition((0, 0))

        manager.remove_cell((0, 0))

        assert (0, 0) not in manager.active_transitions
        assert manager.get_current_lod((0, 0)) == 0  # Returns default


# =============================================================================
# VISIBILITY RESULT TESTS
# =============================================================================


class TestVisibilityResult:
    """Tests for VisibilityResult."""

    def test_creation(self) -> None:
        """Test visibility result creation."""
        result = VisibilityResult(
            cell_id=(0, 0),
            is_visible=True,
            lod_index=1,
            blend_factor=0.5,
            screen_error=100.0,
            distance=200.0,
        )

        assert result.cell_id == (0, 0)
        assert result.is_visible
        assert result.lod_index == 1
        assert result.blend_factor == 0.5
        assert result.screen_error == 100.0
        assert result.distance == 200.0


# =============================================================================
# HLOD VISIBILITY SYSTEM TESTS
# =============================================================================


class TestHLODVisibilitySystem:
    """Tests for HLODVisibilitySystem."""

    @pytest.fixture
    def system(self) -> HLODVisibilitySystem:
        """Create a visibility system."""
        system = HLODVisibilitySystem(error_threshold=2.0)
        system.lod_thresholds = [100.0, 200.0, 400.0, 800.0]
        system.max_distance = 1000.0
        return system

    def test_creation(self, system: HLODVisibilitySystem) -> None:
        """Test system creation."""
        assert system.error_threshold == 2.0

    def test_add_cell(self, system: HLODVisibilitySystem) -> None:
        """Test adding cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(100, 100, 100))
        system.add_cell((0, 0), bounds)

        # Cell should be trackable
        result = system.get_cell_visibility((0, 0))
        assert result is not None

    def test_remove_cell(self, system: HLODVisibilitySystem) -> None:
        """Test removing cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(100, 100, 100))
        system.add_cell((0, 0), bounds)

        system.remove_cell((0, 0))

        result = system.get_cell_visibility((0, 0))
        assert result is None

    def test_update(self, system: HLODVisibilitySystem) -> None:
        """Test updating camera position."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(100, 100, 100))
        system.add_cell((0, 0), bounds)

        system.update(
            camera_position=Vec3(50, 50, 200),
            camera_forward=Vec3(0, 0, -1),
            fov=math.radians(60),
            screen_height=1080,
        )

        # Should update internal state
        result = system.get_cell_visibility((0, 0))
        assert result is not None

    def test_get_visible_cells(self, system: HLODVisibilitySystem) -> None:
        """Test getting visible cells."""
        # Add cells at different positions
        system.add_cell((0, 0), AABB(Vec3(0, 0, 0), Vec3(100, 100, 100)))
        system.add_cell((1, 0), AABB(Vec3(500, 0, 0), Vec3(600, 100, 100)))

        system.update(
            camera_position=Vec3(50, 50, 50),
            camera_forward=Vec3(0, 0, -1),
        )

        visible = system.get_visible_cells()

        # Should return visibility results
        assert isinstance(visible, list)
        for result in visible:
            assert isinstance(result, VisibilityResult)

    def test_get_lod_assignments(self, system: HLODVisibilitySystem) -> None:
        """Test getting LOD assignments."""
        system.add_cell((0, 0), AABB(Vec3(0, 0, 0), Vec3(100, 100, 100)))
        system.add_cell((1, 0), AABB(Vec3(200, 0, 0), Vec3(300, 100, 100)))

        system.update(camera_position=Vec3(50, 50, 50))

        assignments = system.get_lod_assignments()

        # Should return dict mapping cell_id to LOD
        assert isinstance(assignments, dict)

    def test_distance_culling(self, system: HLODVisibilitySystem) -> None:
        """Test distance-based culling."""
        system.max_distance = 500.0

        # Near cell
        system.add_cell((0, 0), AABB(Vec3(0, 0, 0), Vec3(100, 100, 100)))
        # Far cell
        system.add_cell((1, 0), AABB(Vec3(1000, 0, 0), Vec3(1100, 100, 100)))

        system.update(camera_position=Vec3(50, 50, 50))

        visible = system.get_visible_cells()
        cell_ids = [r.cell_id for r in visible]

        # Only near cell should be visible
        assert (0, 0) in cell_ids
        assert (1, 0) not in cell_ids

    def test_frustum_culling_behind_camera(
        self,
        system: HLODVisibilitySystem,
    ) -> None:
        """Test frustum culling for cells behind camera."""
        # Cell in front of camera
        system.add_cell((0, 0), AABB(Vec3(0, 0, -200), Vec3(100, 100, -100)))
        # Cell behind camera
        system.add_cell((1, 0), AABB(Vec3(0, 0, 100), Vec3(100, 100, 200)))

        system.update(
            camera_position=Vec3(50, 50, 0),
            camera_forward=Vec3(0, 0, -1),
        )

        visible = system.get_visible_cells()
        cell_ids = [r.cell_id for r in visible]

        # Cell behind camera should not be visible
        assert (1, 0) not in cell_ids

    def test_get_cell_visibility(self, system: HLODVisibilitySystem) -> None:
        """Test getting visibility for specific cell."""
        bounds = AABB(Vec3(0, 0, 0), Vec3(100, 100, 100))
        system.add_cell((0, 0), bounds)

        system.update(camera_position=Vec3(50, 50, 200))

        result = system.get_cell_visibility((0, 0))

        assert result is not None
        assert result.cell_id == (0, 0)
        assert result.distance > 0

    def test_get_cell_visibility_nonexistent(
        self,
        system: HLODVisibilitySystem,
    ) -> None:
        """Test getting visibility for nonexistent cell."""
        result = system.get_cell_visibility((99, 99))
        assert result is None

    def test_configure_transitions(self, system: HLODVisibilitySystem) -> None:
        """Test configuring transition settings."""
        settings = TransitionSettings(
            mode=TransitionMode.CROSSFADE,
            transition_range=100.0,
        )

        system.configure_transitions(settings)

        # Verify settings were applied via internal transition manager
        applied_settings = system._transition_manager.settings
        assert applied_settings.mode == TransitionMode.CROSSFADE
        assert applied_settings.transition_range == 100.0

    def test_lod_thresholds_property(self, system: HLODVisibilitySystem) -> None:
        """Test LOD thresholds property."""
        assert system.lod_thresholds == [100.0, 200.0, 400.0, 800.0]

        system.lod_thresholds = [50.0, 100.0, 200.0]
        assert system.lod_thresholds == [50.0, 100.0, 200.0]

    def test_max_distance_property(self, system: HLODVisibilitySystem) -> None:
        """Test max distance property."""
        assert system.max_distance == 1000.0

        system.max_distance = 500.0
        assert system.max_distance == 500.0

    def test_error_threshold_property(self, system: HLODVisibilitySystem) -> None:
        """Test error threshold property."""
        assert system.error_threshold == 2.0

        system.error_threshold = 5.0
        assert system.error_threshold == 5.0


# =============================================================================
# TRANSITION MODE TESTS
# =============================================================================


class TestTransitionMode:
    """Tests for TransitionMode enum."""

    def test_all_modes_exist(self) -> None:
        """Test that all transition modes exist."""
        assert TransitionMode.INSTANT is not None
        assert TransitionMode.DITHERED is not None
        assert TransitionMode.CROSSFADE is not None
        assert TransitionMode.MORPHING is not None

    def test_modes_are_unique(self) -> None:
        """Test that all modes have unique values."""
        modes = [
            TransitionMode.INSTANT,
            TransitionMode.DITHERED,
            TransitionMode.CROSSFADE,
            TransitionMode.MORPHING,
        ]
        assert len(set(modes)) == len(modes)


# =============================================================================
# TRANSITION STATE TESTS
# =============================================================================


class TestTransitionState:
    """Tests for TransitionState enum."""

    def test_all_states_exist(self) -> None:
        """Test that all transition states exist."""
        assert TransitionState.STABLE is not None
        assert TransitionState.TRANSITIONING is not None
        assert TransitionState.COMPLETE is not None

    def test_states_are_unique(self) -> None:
        """Test that all states have unique values."""
        states = [
            TransitionState.STABLE,
            TransitionState.TRANSITIONING,
            TransitionState.COMPLETE,
        ]
        assert len(set(states)) == len(states)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_transition_with_same_lod(self) -> None:
        """Test transition when from_lod equals to_lod."""
        transition = LODTransition()
        transition.start(1, 1, 100.0, 200.0)

        # Should still work correctly
        blend = transition.update(150.0, [100.0, 200.0])
        assert 0.0 <= blend <= 1.0

    def test_transition_with_zero_range(self) -> None:
        """Test transition with zero distance range."""
        transition = LODTransition()
        transition.start(0, 1, 100.0, 100.0)  # Same start/end distance

        # Should complete immediately
        blend = transition.update(100.0, [100.0])
        assert blend == 1.0

    def test_visibility_with_very_small_bounds(self) -> None:
        """Test visibility with very small bounds."""
        system = HLODVisibilitySystem()
        tiny_bounds = AABB(Vec3(0, 0, 0), Vec3(0.001, 0.001, 0.001))
        system.add_cell((0, 0), tiny_bounds)

        system.update(camera_position=Vec3(0, 0, 1))

        result = system.get_cell_visibility((0, 0))
        assert result is not None

    def test_visibility_with_camera_at_center(self) -> None:
        """Test visibility with camera at cell center."""
        system = HLODVisibilitySystem()
        bounds = AABB(Vec3(-50, -50, -50), Vec3(50, 50, 50))
        system.add_cell((0, 0), bounds)

        # Camera at center of bounds
        system.update(camera_position=Vec3(0, 0, 0))

        result = system.get_cell_visibility((0, 0))
        assert result is not None
        # Screen error should be very large (infinite)
        assert result.screen_error == float("inf")

    def test_dither_pattern_wrap(self) -> None:
        """Test that dither pattern wraps correctly."""
        calculator = TransitionCalculator()

        # Test that pattern wraps at boundary
        result_0 = calculator.get_dither_pattern(0.5, 0, 0)
        result_4 = calculator.get_dither_pattern(0.5, 4, 0)  # Should wrap to same as (0, 0)

        assert result_0 == result_4

    def test_screen_error_with_narrow_fov(self) -> None:
        """Test screen error with very narrow FOV."""
        error_calc = ScreenSpaceError()
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))

        # Very narrow FOV (like a sniper scope)
        error_narrow = error_calc.calculate_error(
            bounds,
            Vec3(0, 0, 10),
            math.radians(10),
            1080,
        )

        # Normal FOV
        error_normal = error_calc.calculate_error(
            bounds,
            Vec3(0, 0, 10),
            math.radians(60),
            1080,
        )

        # Narrow FOV should result in larger screen error (more zoomed in)
        assert error_narrow > error_normal


# =============================================================================
# TRANSITION SMOOTHNESS TESTS
# =============================================================================


class TestTransitionSmoothness:
    """Tests verifying smooth transitions without visual popping."""

    def test_blend_factor_continuity(self) -> None:
        """Test that blend factor changes continuously (no jumps)."""
        calculator = TransitionCalculator(TransitionSettings(
            mode=TransitionMode.DITHERED,
            transition_range=50.0,
        ))

        # Sample blend factors at small distance increments
        previous_blend = None
        max_jump = 0.0

        for dist in range(0, 200, 1):
            blend = calculator.calculate_blend(float(dist), 50.0, 150.0)

            if previous_blend is not None:
                jump = abs(blend - previous_blend)
                max_jump = max(max_jump, jump)

            previous_blend = blend

        # Maximum jump should be small (no popping)
        # With 1 unit steps over 200 units, max jump should be <= 0.05
        assert max_jump < 0.1, f"Blend factor jump too large: {max_jump}"

    def test_blend_factor_monotonic_increasing(self) -> None:
        """Test that blend factor increases monotonically with distance."""
        calculator = TransitionCalculator(TransitionSettings(
            mode=TransitionMode.CROSSFADE,
            transition_range=50.0,
        ))

        previous_blend = 0.0

        for dist in range(0, 200, 5):
            blend = calculator.calculate_blend(float(dist), 50.0, 150.0)
            assert blend >= previous_blend, (
                f"Blend factor decreased: {previous_blend} -> {blend} at dist={dist}"
            )
            previous_blend = blend

    def test_smooth_step_produces_smooth_curve(self) -> None:
        """Test that smooth step function produces smooth transitions."""
        calculator = TransitionCalculator(TransitionSettings(
            mode=TransitionMode.MORPHING,
        ))

        # Get morph factors at fine intervals
        values = []
        for i in range(101):
            t = i / 100.0
            morph = calculator.get_morph_factor(t, 0)
            values.append(morph)

        # Check for smoothness: second derivative should not have sharp changes
        # (simplified check: no sudden jumps in first derivative)
        max_accel = 0.0
        for i in range(2, len(values)):
            v0, v1, v2 = values[i-2], values[i-1], values[i]
            accel = abs((v2 - v1) - (v1 - v0))
            max_accel = max(max_accel, accel)

        # Smooth step should have smooth acceleration
        assert max_accel < 0.02, f"Transition not smooth enough, max acceleration: {max_accel}"

    def test_dither_pattern_gradual_coverage(self) -> None:
        """Test that dither pattern provides gradual LOD coverage."""
        calculator = TransitionCalculator(TransitionSettings(
            mode=TransitionMode.DITHERED,
            dither_scale=1.0,
        ))

        # At various blend levels, count pixels showing high LOD
        coverage_at_blend = []

        for blend_percent in range(0, 101, 5):
            blend = blend_percent / 100.0
            high_lod_count = 0

            for x in range(4):
                for y in range(4):
                    if calculator.get_dither_pattern(blend, x, y):
                        high_lod_count += 1

            coverage_at_blend.append((blend, high_lod_count))

        # Coverage should generally increase with blend
        # Allow some non-monotonicity due to dither pattern discretization
        increasing_count = 0
        for i in range(1, len(coverage_at_blend)):
            if coverage_at_blend[i][1] >= coverage_at_blend[i-1][1]:
                increasing_count += 1

        # At least 80% of transitions should be non-decreasing
        assert increasing_count >= len(coverage_at_blend) * 0.8, (
            "Dither pattern coverage not increasing smoothly"
        )

    def test_no_popping_at_threshold_boundaries(self) -> None:
        """Test that transitions are smooth at LOD threshold boundaries."""
        settings = TransitionSettings(
            mode=TransitionMode.DITHERED,
            transition_range=25.0,  # Transition starts 25 units before threshold
        )
        calculator = TransitionCalculator(settings)

        # Test around threshold boundary (near_threshold=50, far_threshold=100)
        # Transition should start at 75 (100-25) and end at 100

        # Just before transition zone
        blend_before = calculator.calculate_blend(74.0, 50.0, 100.0)
        assert blend_before == 0.0, "Should be fully at near LOD before transition zone"

        # Just inside transition zone
        blend_start = calculator.calculate_blend(76.0, 50.0, 100.0)
        assert 0.0 < blend_start < 0.2, "Should have started transitioning"

        # Just after transition zone
        blend_after = calculator.calculate_blend(101.0, 50.0, 100.0)
        assert blend_after == 1.0, "Should be fully at far LOD after transition zone"


# =============================================================================
# SCREEN SPACE ERROR ACCURACY TESTS
# =============================================================================


class TestScreenSpaceErrorAccuracy:
    """Tests for screen space error calculation accuracy."""

    def test_error_proportional_to_inverse_distance(self) -> None:
        """Test that error is roughly proportional to 1/distance."""
        error_calc = ScreenSpaceError()
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        fov = math.radians(60)

        error_10 = error_calc.calculate_error(bounds, Vec3(0, 0, 10), fov, 1080)
        error_20 = error_calc.calculate_error(bounds, Vec3(0, 0, 20), fov, 1080)

        # At double distance, error should be roughly half
        ratio = error_10 / error_20
        assert 1.8 <= ratio <= 2.2, f"Expected ratio ~2, got {ratio}"

    def test_error_proportional_to_object_size(self) -> None:
        """Test that error is proportional to object size."""
        error_calc = ScreenSpaceError()
        fov = math.radians(60)
        camera_pos = Vec3(0, 0, 50)

        small_bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))  # Size 2
        large_bounds = AABB(Vec3(-2, -2, -2), Vec3(2, 2, 2))  # Size 4

        error_small = error_calc.calculate_error(small_bounds, camera_pos, fov, 1080)
        error_large = error_calc.calculate_error(large_bounds, camera_pos, fov, 1080)

        # Larger object should have larger error
        ratio = error_large / error_small
        assert 1.8 <= ratio <= 2.2, f"Expected ratio ~2, got {ratio}"

    def test_error_at_known_configuration(self) -> None:
        """Test error calculation against known expected values."""
        error_calc = ScreenSpaceError()

        # Unit sphere at distance 10, 60 degree FOV, 1080p
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        camera_pos = Vec3(0, 0, 10)
        fov = math.radians(60)

        error = error_calc.calculate_error(bounds, camera_pos, fov, 1080)

        # Expected: world_size=2, projected_size = (2/10) / tan(30) = 0.346
        # screen_size = 0.346 * 540 = ~187 pixels
        # Allow some variance due to implementation details
        assert 100 < error < 300, f"Error {error} outside expected range"

    def test_lod_selection_consistency(self) -> None:
        """Test that LOD selection is consistent across similar errors."""
        error_calc = ScreenSpaceError()
        thresholds = [100.0, 50.0, 25.0, 10.0]

        # Same error should always return same LOD
        for _ in range(10):
            lod_200 = error_calc.get_lod_for_error(200.0, thresholds)
            lod_75 = error_calc.get_lod_for_error(75.0, thresholds)
            lod_30 = error_calc.get_lod_for_error(30.0, thresholds)

            assert lod_200 == 0  # Largest error = closest = LOD 0
            assert lod_75 == 1   # Between 100 and 50
            assert lod_30 == 2   # Between 50 and 25

    def test_extreme_fov_handling(self) -> None:
        """Test error calculation with extreme FOV values."""
        error_calc = ScreenSpaceError()
        bounds = AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))
        camera_pos = Vec3(0, 0, 10)

        # Very narrow FOV (telescope view)
        error_narrow = error_calc.calculate_error(
            bounds, camera_pos, math.radians(5), 1080
        )
        assert error_narrow > 0 and error_narrow != float("inf")

        # Very wide FOV (fisheye)
        error_wide = error_calc.calculate_error(
            bounds, camera_pos, math.radians(120), 1080
        )
        assert error_wide > 0 and error_wide != float("inf")

        # Narrow should be much larger than wide
        assert error_narrow > error_wide * 5
