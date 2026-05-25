"""
Tests for Trinity Pattern Tier 36: GAME_AI decorators.
"""

import pytest

from trinity.decorators.game_ai import (
    VALID_SENSES,
    ai_debug,
    behavior_tree,
    blackboard,
    perception,
    utility_ai,
)
from trinity.decorators.registry import Tier, registry


class TestBehaviorTree:
    """Tests for @behavior_tree decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @behavior_tree(id="patrol", debug_name="Patrol AI")
        class PatrolBehavior:
            pass

        assert hasattr(PatrolBehavior, "_behavior_tree")
        assert PatrolBehavior._behavior_tree is True
        assert PatrolBehavior._bt_id == "patrol"
        assert PatrolBehavior._bt_debug_name == "Patrol AI"

    def test_without_debug_name(self):
        """Test decorator without optional debug_name."""

        @behavior_tree(id="chase")
        class ChaseBehavior:
            pass

        assert ChaseBehavior._behavior_tree is True
        assert ChaseBehavior._bt_id == "chase"
        assert ChaseBehavior._bt_debug_name is None

    def test_validation_empty_id(self):
        """Test validation fails when id is empty."""
        with pytest.raises(ValueError, match="id must be non-empty"):

            @behavior_tree(id="")
            class InvalidBehavior:
                pass

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @behavior_tree(id="attack")
        class AttackBehavior:
            pass

        assert hasattr(AttackBehavior, "_tags")
        assert AttackBehavior._tags["behavior_tree"] is True
        assert AttackBehavior._tags["bt_id"] == "attack"

    def test_registry_registration(self):
        """Test that decorator is registered in the registry."""

        @behavior_tree(id="test")
        class TestBehavior:
            pass

        assert hasattr(TestBehavior, "_registries")
        assert "game_ai" in TestBehavior._registries

    def test_schema_described(self):
        """Test that schema is described."""

        @behavior_tree(id="test")
        class AnnotatedBehavior:
            state: str
            priority: int

        assert hasattr(AnnotatedBehavior, "_described")
        assert AnnotatedBehavior._described is True


class TestUtilityAI:
    """Tests for @utility_ai decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @utility_ai(id="decision", update_rate=0.1)
        class DecisionAI:
            pass

        assert hasattr(DecisionAI, "_utility_ai")
        assert DecisionAI._utility_ai is True
        assert DecisionAI._utility_id == "decision"
        assert DecisionAI._utility_update_rate == 0.1

    def test_default_update_rate(self):
        """Test default update_rate is used."""

        @utility_ai(id="default")
        class DefaultAI:
            pass

        assert DefaultAI._utility_update_rate == 0.5

    def test_validation_empty_id(self):
        """Test validation fails when id is empty."""
        with pytest.raises(ValueError, match="id must be non-empty"):

            @utility_ai(id="")
            class InvalidAI:
                pass

    def test_validation_negative_update_rate(self):
        """Test validation fails when update_rate <= 0."""
        with pytest.raises(ValueError, match="update_rate must be > 0"):

            @utility_ai(id="test", update_rate=-1)
            class InvalidAI:
                pass

    def test_validation_zero_update_rate(self):
        """Test validation fails when update_rate is zero."""
        with pytest.raises(ValueError, match="update_rate must be > 0"):

            @utility_ai(id="test", update_rate=0)
            class InvalidAI:
                pass

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @utility_ai(id="test", update_rate=1.0)
        class TestAI:
            pass

        assert hasattr(TestAI, "_tags")
        assert TestAI._tags["utility_ai"] is True
        assert TestAI._tags["utility_id"] == "test"
        assert TestAI._tags["utility_update_rate"] == 1.0


class TestBlackboard:
    """Tests for @blackboard decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @blackboard
        class SharedMemory:
            pass

        assert hasattr(SharedMemory, "_blackboard")
        assert SharedMemory._blackboard is True

    def test_with_parens(self):
        """Test decorator works with parentheses."""

        @blackboard()
        class SharedMemory:
            pass

        assert SharedMemory._blackboard is True

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @blackboard
        class TestMemory:
            pass

        assert hasattr(TestMemory, "_tags")
        assert TestMemory._tags["blackboard"] is True

    def test_registry_registration(self):
        """Test that decorator is registered in the registry."""

        @blackboard
        class TestMemory:
            pass

        assert hasattr(TestMemory, "_registries")
        assert "game_ai" in TestMemory._registries


class TestAIDebug:
    """Tests for @ai_debug decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @ai_debug
        class DebugAI:
            pass

        assert hasattr(DebugAI, "_ai_debug")
        assert DebugAI._ai_debug is True

    def test_with_parens(self):
        """Test decorator works with parentheses."""

        @ai_debug()
        class DebugAI:
            pass

        assert DebugAI._ai_debug is True

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @ai_debug
        class TestDebug:
            pass

        assert hasattr(TestDebug, "_tags")
        assert TestDebug._tags["ai_debug"] is True


class TestPerception:
    """Tests for @perception decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @perception(sense="sight", range=100.0, fov=90.0)
        class VisionSensor:
            pass

        assert hasattr(VisionSensor, "_perception")
        assert VisionSensor._perception is True
        assert VisionSensor._perception_sense == "sight"
        assert VisionSensor._perception_range == 100.0
        assert VisionSensor._perception_fov == 90.0

    def test_without_fov(self):
        """Test decorator without optional fov."""

        @perception(sense="hearing", range=50.0)
        class HearingSensor:
            pass

        assert HearingSensor._perception is True
        assert HearingSensor._perception_sense == "hearing"
        assert HearingSensor._perception_range == 50.0
        assert HearingSensor._perception_fov is None

    def test_all_valid_senses(self):
        """Test all valid sense types."""
        for sense in VALID_SENSES:

            @perception(sense=sense, range=10.0)
            class Sensor:
                pass

            assert Sensor._perception_sense == sense

    def test_validation_invalid_sense(self):
        """Test validation fails with invalid sense."""
        with pytest.raises(ValueError, match="Invalid sense"):

            @perception(sense="invalid", range=10.0)
            class InvalidSensor:
                pass

    def test_validation_negative_range(self):
        """Test validation fails when range <= 0."""
        with pytest.raises(ValueError, match="range must be > 0"):

            @perception(sense="sight", range=-10.0)
            class InvalidSensor:
                pass

    def test_validation_zero_range(self):
        """Test validation fails when range is zero."""
        with pytest.raises(ValueError, match="range must be > 0"):

            @perception(sense="sight", range=0)
            class InvalidSensor:
                pass

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @perception(sense="damage", range=5.0)
        class DamageSensor:
            pass

        assert hasattr(DamageSensor, "_tags")
        assert DamageSensor._tags["perception"] is True
        assert DamageSensor._tags["perception_sense"] == "damage"
        assert DamageSensor._tags["perception_range"] == 5.0


class TestDecoratorComposition:
    """Tests for combining GAME_AI decorators."""

    def test_behavior_tree_with_blackboard(self):
        """Test combining behavior_tree and blackboard."""

        @blackboard
        @behavior_tree(id="complex")
        class ComplexAI:
            pass

        assert ComplexAI._behavior_tree is True
        assert ComplexAI._blackboard is True

    def test_utility_ai_with_perception(self):
        """Test combining utility_ai and perception."""

        @perception(sense="sight", range=100.0)
        @utility_ai(id="aware", update_rate=0.2)
        class AwareAI:
            pass

        assert AwareAI._utility_ai is True
        assert AwareAI._perception is True

    def test_multiple_perceptions(self):
        """Test applying multiple perception decorators."""

        @perception(sense="hearing", range=50.0)
        @perception(sense="sight", range=100.0)
        class MultiSenseAI:
            pass

        assert MultiSenseAI._perception is True
        # Last applied wins for attributes
        assert MultiSenseAI._perception_sense == "hearing"

    def test_ai_debug_with_behavior_tree(self):
        """Test combining ai_debug with behavior_tree."""

        @ai_debug
        @behavior_tree(id="debugged")
        class DebuggableAI:
            pass

        assert DebuggableAI._behavior_tree is True
        assert DebuggableAI._ai_debug is True


class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_all_decorators_registered(self):
        """Test that all GAME_AI decorators are registered."""
        decorators = ["behavior_tree", "utility_ai", "blackboard", "ai_debug", "perception"]

        for dec_name in decorators:
            spec = registry.get(dec_name)
            assert spec is not None, f"Decorator '{dec_name}' not registered"
            assert spec.tier == Tier.GAME_AI
            assert spec.name == dec_name

    def test_decorator_metadata(self):
        """Test decorator metadata in registry."""
        spec = registry.get("behavior_tree")
        assert spec is not None
        assert spec.unique is True
        assert spec.foundation is False
        assert "class" in spec.target_types

    def test_perception_not_unique(self):
        """Test that perception can be applied multiple times."""
        spec = registry.get("perception")
        assert spec is not None
        assert spec.unique is False

    def test_tier_listing(self):
        """Test that all decorators appear in tier listing."""
        tier_decorators = registry.by_tier(Tier.GAME_AI)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "behavior_tree" in decorator_names
        assert "utility_ai" in decorator_names
        assert "blackboard" in decorator_names
        assert "ai_debug" in decorator_names
        assert "perception" in decorator_names


class TestAppliedSteps:
    """Tests for applied steps tracking."""

    def test_steps_tracked(self):
        """Test that applied steps are tracked."""

        @behavior_tree(id="test")
        class TestBehavior:
            pass

        assert hasattr(TestBehavior, "_applied_steps")
        assert len(TestBehavior._applied_steps) > 0

    def test_decorator_name_tracked(self):
        """Test that decorator name is tracked."""

        @utility_ai(id="test")
        class TestAI:
            pass

        assert hasattr(TestAI, "_applied_decorators")
        assert "utility_ai" in TestAI._applied_decorators


class TestValidConstants:
    """Tests for valid constants."""

    def test_valid_senses_content(self):
        """Test VALID_SENSES contains expected values."""
        assert "sight" in VALID_SENSES
        assert "hearing" in VALID_SENSES
        assert "damage" in VALID_SENSES
        assert "squad" in VALID_SENSES
        assert len(VALID_SENSES) == 4

    def test_valid_senses_immutable(self):
        """Test VALID_SENSES is immutable."""
        assert isinstance(VALID_SENSES, frozenset)
