"""
Tests for Trinity Pattern Tier 37: PROCEDURAL decorators.
"""

import pytest

from trinity.decorators.procedural import (
    VALID_SEED_SOURCES,
    constraint,
    procedural,
    seeded,
)
from trinity.decorators.registry import Tier, registry


class TestSeeded:
    """Tests for @seeded decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @seeded(seed_source="world")
        class WorldGenerator:
            pass

        assert hasattr(WorldGenerator, "_seeded")
        assert WorldGenerator._seeded is True
        assert WorldGenerator._seed_source == "world"

    def test_default_seed_source(self):
        """Test default seed_source is 'world'."""

        @seeded()
        class DefaultGenerator:
            pass

        assert DefaultGenerator._seed_source == "world"

    def test_all_valid_seed_sources(self):
        """Test all valid seed sources."""
        for source in VALID_SEED_SOURCES:

            @seeded(seed_source=source)
            class Generator:
                pass

            assert Generator._seed_source == source

    def test_validation_invalid_seed_source(self):
        """Test validation fails with invalid seed_source."""
        with pytest.raises(ValueError, match="Invalid seed_source"):

            @seeded(seed_source="invalid")
            class InvalidGenerator:
                pass

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @seeded(seed_source="chunk")
        class ChunkGenerator:
            pass

        assert hasattr(ChunkGenerator, "_tags")
        assert ChunkGenerator._tags["seeded"] is True
        assert ChunkGenerator._tags["seed_source"] == "chunk"

    def test_registry_registration(self):
        """Test that decorator is registered in the registry."""

        @seeded(seed_source="entity")
        class EntityGenerator:
            pass

        assert hasattr(EntityGenerator, "_registries")
        assert "procedural" in EntityGenerator._registries

    def test_schema_described(self):
        """Test that schema is described."""

        @seeded(seed_source="world")
        class AnnotatedGenerator:
            seed: int
            noise_scale: float

        assert hasattr(AnnotatedGenerator, "_described")
        assert AnnotatedGenerator._described is True


class TestProcedural:
    """Tests for @procedural decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @procedural(cache=True)
        class CachedContent:
            pass

        assert hasattr(CachedContent, "_procedural")
        assert CachedContent._procedural is True
        assert CachedContent._procedural_cache is True
        assert CachedContent._procedural_validate is None

    def test_default_cache(self):
        """Test default cache is True."""

        @procedural()
        class DefaultCached:
            pass

        assert DefaultCached._procedural_cache is True

    def test_cache_disabled(self):
        """Test cache can be disabled."""

        @procedural(cache=False)
        class Uncached:
            pass

        assert Uncached._procedural_cache is False

    def test_with_validator(self):
        """Test decorator with custom validator."""

        def custom_validator(x):
            return x > 0

        @procedural(validate=custom_validator)
        class ValidatedContent:
            pass

        assert ValidatedContent._procedural_validate is custom_validator

    def test_no_params(self):
        """Test decorator works without parameters."""

        @procedural
        class SimpleContent:
            pass

        assert SimpleContent._procedural is True
        assert SimpleContent._procedural_cache is True

    def test_with_parens(self):
        """Test decorator works with empty parentheses."""

        @procedural()
        class SimpleContent:
            pass

        assert SimpleContent._procedural is True

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        @procedural(cache=False)
        class TestContent:
            pass

        assert hasattr(TestContent, "_tags")
        assert TestContent._tags["procedural"] is True
        assert TestContent._tags["procedural_cache"] is False


class TestConstraint:
    """Tests for @constraint decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        def rule1(x):
            return x > 0

        def rule2(x):
            return x < 100

        @constraint(rules=[rule1, rule2])
        class ConstrainedContent:
            pass

        assert hasattr(ConstrainedContent, "_constraint")
        assert ConstrainedContent._constraint is True
        assert len(ConstrainedContent._constraint_rules) == 2
        assert ConstrainedContent._constraint_rules[0] is rule1
        assert ConstrainedContent._constraint_rules[1] is rule2

    def test_single_rule(self):
        """Test decorator with single rule."""

        def single_rule(x):
            return True

        @constraint(rules=[single_rule])
        class SingleConstraint:
            pass

        assert len(SingleConstraint._constraint_rules) == 1
        assert SingleConstraint._constraint_rules[0] is single_rule

    def test_validation_empty_rules(self):
        """Test validation fails when rules is empty."""
        with pytest.raises(ValueError, match="rules must be a non-empty list"):

            @constraint(rules=[])
            class InvalidConstraint:
                pass

    def test_validation_not_list(self):
        """Test validation fails when rules is not a list."""
        with pytest.raises(TypeError, match="rules must be a list"):

            @constraint(rules="not_a_list")
            class InvalidConstraint:
                pass

    def test_tags_applied(self):
        """Test that tags are properly applied."""

        def test_rule(x):
            return True

        @constraint(rules=[test_rule])
        class TestConstraint:
            pass

        assert hasattr(TestConstraint, "_tags")
        assert TestConstraint._tags["constraint"] is True

    def test_rules_are_copied(self):
        """Test that rules list is copied, not referenced."""

        def rule1(x):
            return True

        original_rules = [rule1]

        @constraint(rules=original_rules)
        class TestConstraint:
            pass

        # Modify original list
        original_rules.append(lambda x: False)

        # Decorator should have only one rule
        assert len(TestConstraint._constraint_rules) == 1


class TestDecoratorComposition:
    """Tests for combining PROCEDURAL decorators."""

    def test_seeded_with_procedural(self):
        """Test combining seeded and procedural."""

        @procedural(cache=True)
        @seeded(seed_source="chunk")
        class SeededProcedural:
            pass

        assert SeededProcedural._seeded is True
        assert SeededProcedural._procedural is True

    def test_seeded_with_constraint(self):
        """Test combining seeded and constraint."""

        def rule(x):
            return x >= 0

        @constraint(rules=[rule])
        @seeded(seed_source="world")
        class ConstrainedSeeded:
            pass

        assert ConstrainedSeeded._seeded is True
        assert ConstrainedSeeded._constraint is True

    def test_all_three_decorators(self):
        """Test combining all three procedural decorators."""

        def rule(x):
            return True

        @constraint(rules=[rule])
        @procedural(cache=True)
        @seeded(seed_source="entity")
        class FullyDecorated:
            pass

        assert FullyDecorated._seeded is True
        assert FullyDecorated._procedural is True
        assert FullyDecorated._constraint is True

    def test_multiple_constraints(self):
        """Test applying multiple constraint decorators."""

        def rule1(x):
            return x > 0

        def rule2(x):
            return x < 100

        @constraint(rules=[rule2])
        @constraint(rules=[rule1])
        class MultiConstrained:
            pass

        assert MultiConstrained._constraint is True
        # Last applied wins for attributes
        assert len(MultiConstrained._constraint_rules) == 1


class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_all_decorators_registered(self):
        """Test that all PROCEDURAL decorators are registered."""
        decorators = ["seeded", "procedural", "constraint"]

        for dec_name in decorators:
            spec = registry.get(dec_name)
            assert spec is not None, f"Decorator '{dec_name}' not registered"
            assert spec.tier == Tier.PROCEDURAL
            assert spec.name == dec_name

    def test_decorator_metadata(self):
        """Test decorator metadata in registry."""
        spec = registry.get("seeded")
        assert spec is not None
        assert spec.unique is True
        assert spec.foundation is False
        assert "class" in spec.target_types

    def test_constraint_not_unique(self):
        """Test that constraint can be applied multiple times."""
        spec = registry.get("constraint")
        assert spec is not None
        assert spec.unique is False

    def test_procedural_unique(self):
        """Test that procedural is unique."""
        spec = registry.get("procedural")
        assert spec is not None
        assert spec.unique is True

    def test_tier_listing(self):
        """Test that all decorators appear in tier listing."""
        tier_decorators = registry.by_tier(Tier.PROCEDURAL)
        decorator_names = {spec.name for spec in tier_decorators}

        assert "seeded" in decorator_names
        assert "procedural" in decorator_names
        assert "constraint" in decorator_names


class TestAppliedSteps:
    """Tests for applied steps tracking."""

    def test_steps_tracked(self):
        """Test that applied steps are tracked."""

        @seeded(seed_source="world")
        class TestGenerator:
            pass

        assert hasattr(TestGenerator, "_applied_steps")
        assert len(TestGenerator._applied_steps) > 0

    def test_decorator_name_tracked(self):
        """Test that decorator name is tracked."""

        @procedural(cache=True)
        class TestContent:
            pass

        assert hasattr(TestContent, "_applied_decorators")
        assert "procedural" in TestContent._applied_decorators


class TestValidConstants:
    """Tests for valid constants."""

    def test_valid_seed_sources_content(self):
        """Test VALID_SEED_SOURCES contains expected values."""
        assert "world" in VALID_SEED_SOURCES
        assert "chunk" in VALID_SEED_SOURCES
        assert "entity" in VALID_SEED_SOURCES
        assert "explicit" in VALID_SEED_SOURCES
        assert len(VALID_SEED_SOURCES) == 4

    def test_valid_seed_sources_immutable(self):
        """Test VALID_SEED_SOURCES is immutable."""
        assert isinstance(VALID_SEED_SOURCES, frozenset)


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_seeded_explicit_source(self):
        """Test explicit seed source."""

        @seeded(seed_source="explicit")
        class ExplicitGenerator:
            pass

        assert ExplicitGenerator._seed_source == "explicit"

    def test_procedural_with_none_validator(self):
        """Test procedural with None validator."""

        @procedural(validate=None)
        class NoValidator:
            pass

        assert NoValidator._procedural_validate is None

    def test_constraint_with_lambda_rules(self):
        """Test constraint with lambda functions as rules."""

        @constraint(rules=[lambda x: x > 0, lambda x: x < 100])
        class LambdaConstrained:
            pass

        assert len(LambdaConstrained._constraint_rules) == 2
        # Test that lambdas work
        assert LambdaConstrained._constraint_rules[0](5) is True
        assert LambdaConstrained._constraint_rules[0](-1) is False
