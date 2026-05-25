"""
Tests for Trinity Pattern - Tier 43: DESTRUCTION Decorators
"""

import pytest

from trinity.decorators.destruction import (
    VALID_FRACTURE_PATTERN,
    VALID_JOINT_TYPE,
    DamageResistanceConfig,
    DamageTypeConfig,
    DestructibleConfig,
    FractureConfig,
    JointConfig,
    PhysicsMaterialConfig,
    damage_resistance,
    damage_type,
    destructible,
    fracture,
    joint,
    physics_material,
)
from trinity.decorators.registry import registry


class TestDestructible:
    """Test @destructible decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @destructible()
        class TestObject:
            pass

        assert hasattr(TestObject, "_destructible")
        assert TestObject._destructible is True
        assert TestObject._destructible_health == 100.0
        assert TestObject._destructible_fracture_depth == 2
        assert TestObject._destructible_debris_lifetime == 10.0
        assert isinstance(TestObject._destructible_config, DestructibleConfig)

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @destructible(health=500.0, fracture_depth=4, debris_lifetime=30.0)
        class StrongObject:
            pass

        assert StrongObject._destructible_health == 500.0
        assert StrongObject._destructible_fracture_depth == 4
        assert StrongObject._destructible_debris_lifetime == 30.0

    def test_invalid_health(self):
        """Test validation of health parameter."""
        with pytest.raises(ValueError, match="health must be > 0"):

            @destructible(health=0)
            class NoHealth:
                pass

        with pytest.raises(ValueError, match="health must be > 0"):

            @destructible(health=-10.0)
            class NegHealth:
                pass

    def test_invalid_fracture_depth(self):
        """Test validation of fracture_depth parameter."""
        with pytest.raises(ValueError, match="fracture_depth must be >= 0"):

            @destructible(fracture_depth=-1)
            class BadDepth:
                pass

    def test_zero_fracture_depth_allowed(self):
        """Test that fracture_depth=0 is allowed."""

        @destructible(fracture_depth=0)
        class NoFracture:
            pass

        assert NoFracture._destructible_fracture_depth == 0

    def test_invalid_debris_lifetime(self):
        """Test validation of debris_lifetime parameter."""
        with pytest.raises(ValueError, match="debris_lifetime must be >= 0"):

            @destructible(debris_lifetime=-1.0)
            class BadLifetime:
                pass

    def test_registry_registration(self):
        """Test that decorator is registered properly."""
        spec = registry.get("destructible")
        assert spec is not None
        assert spec.name == "destructible"
        assert spec.tier.name == "DESTRUCTION"


class TestDamageType:
    """Test @damage_type decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @damage_type(id="fire", base_multiplier=1.5)
        class FireDamage:
            pass

        assert hasattr(FireDamage, "_damage_type")
        assert FireDamage._damage_type is True
        assert FireDamage._damage_type_id == "fire"
        assert FireDamage._damage_type_multiplier == 1.5

    def test_custom_params(self):
        """Test decorator with various multipliers."""

        @damage_type(id="explosive", base_multiplier=2.0)
        class ExplosiveDamage:
            pass

        assert ExplosiveDamage._damage_type_id == "explosive"
        assert ExplosiveDamage._damage_type_multiplier == 2.0

    def test_empty_id_validation(self):
        """Test validation of empty id."""
        with pytest.raises(ValueError, match="id must be a non-empty string"):

            @damage_type(id="")
            class NoID:
                pass

    def test_invalid_multiplier(self):
        """Test validation of base_multiplier parameter."""
        with pytest.raises(ValueError, match="base_multiplier must be > 0"):

            @damage_type(id="ice", base_multiplier=0)
            class ZeroMult:
                pass

        with pytest.raises(ValueError, match="base_multiplier must be > 0"):

            @damage_type(id="ice", base_multiplier=-1.0)
            class NegMult:
                pass


class TestDamageResistance:
    """Test @damage_resistance decorator."""

    def test_basic_application(self):
        """Test basic decorator application."""

        @damage_resistance(resistances={"fire": 0.5, "ice": 0.8})
        class ResistantObject:
            pass

        assert hasattr(ResistantObject, "_damage_resistance")
        assert ResistantObject._damage_resistance is True
        assert ResistantObject._damage_resistance_values == {"fire": 0.5, "ice": 0.8}

    def test_custom_params(self):
        """Test decorator with various resistances."""

        @damage_resistance(resistances={"physical": 0.9, "magic": 0.3})
        class MagicWeak:
            pass

        assert "physical" in MagicWeak._damage_resistance_values
        assert "magic" in MagicWeak._damage_resistance_values
        assert MagicWeak._damage_resistance_values["physical"] == 0.9

    def test_empty_resistances_validation(self):
        """Test validation of empty resistances dict."""
        with pytest.raises(ValueError, match="resistances must be a non-empty dict"):

            @damage_resistance(resistances={})
            class NoResist:
                pass

    def test_non_dict_resistances_validation(self):
        """Test validation of non-dict resistances."""
        with pytest.raises(TypeError, match="resistances must be a dict"):

            @damage_resistance(resistances="invalid")
            class BadType:
                pass


class TestFracture:
    """Test @fracture decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @fracture()
        class TestObject:
            pass

        assert hasattr(TestObject, "_fracture")
        assert TestObject._fracture is True
        assert TestObject._fracture_pattern == "voronoi"
        assert TestObject._fracture_min_size == 0.1
        assert TestObject._fracture_interior_material is None

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @fracture(pattern="radial", min_size=0.5, interior_material="concrete")
        class RadialFracture:
            pass

        assert RadialFracture._fracture_pattern == "radial"
        assert RadialFracture._fracture_min_size == 0.5
        assert RadialFracture._fracture_interior_material == "concrete"

    def test_all_valid_patterns(self):
        """Test all valid fracture patterns."""
        for pattern in VALID_FRACTURE_PATTERN:

            @fracture(pattern=pattern)
            class TestFrac:
                pass

            assert TestFrac._fracture_pattern == pattern

    def test_invalid_pattern(self):
        """Test validation of pattern parameter."""
        with pytest.raises(ValueError, match="Invalid pattern"):

            @fracture(pattern="invalid")
            class BadPattern:
                pass

    def test_invalid_min_size(self):
        """Test validation of min_size parameter."""
        with pytest.raises(ValueError, match="min_size must be > 0"):

            @fracture(min_size=0)
            class ZeroSize:
                pass

        with pytest.raises(ValueError, match="min_size must be > 0"):

            @fracture(min_size=-1.0)
            class NegSize:
                pass


class TestPhysicsMaterial:
    """Test @physics_material decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @physics_material()
        class TestObject:
            pass

        assert hasattr(TestObject, "_physics_material")
        assert TestObject._physics_material is True
        assert TestObject._physics_friction == 0.5
        assert TestObject._physics_restitution == 0.3
        assert TestObject._physics_density == 1.0

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @physics_material(friction=0.8, restitution=0.9, density=2.5)
        class BouncyObject:
            pass

        assert BouncyObject._physics_friction == 0.8
        assert BouncyObject._physics_restitution == 0.9
        assert BouncyObject._physics_density == 2.5

    def test_zero_values_allowed(self):
        """Test that zero values are allowed."""

        @physics_material(friction=0.0, restitution=0.0, density=0.0)
        class ZeroPhys:
            pass

        assert ZeroPhys._physics_friction == 0.0
        assert ZeroPhys._physics_restitution == 0.0
        assert ZeroPhys._physics_density == 0.0

    def test_invalid_friction(self):
        """Test validation of negative friction."""
        with pytest.raises(ValueError, match="friction must be >= 0"):

            @physics_material(friction=-0.5)
            class BadFriction:
                pass

    def test_invalid_restitution(self):
        """Test validation of negative restitution."""
        with pytest.raises(ValueError, match="restitution must be >= 0"):

            @physics_material(restitution=-0.5)
            class BadRestitution:
                pass

    def test_invalid_density(self):
        """Test validation of negative density."""
        with pytest.raises(ValueError, match="density must be >= 0"):

            @physics_material(density=-0.5)
            class BadDensity:
                pass


class TestJoint:
    """Test @joint decorator."""

    def test_basic_application(self):
        """Test basic decorator application with defaults."""

        @joint(type="fixed")
        class FixedJoint:
            pass

        assert hasattr(FixedJoint, "_joint")
        assert FixedJoint._joint is True
        assert FixedJoint._joint_type == "fixed"
        assert FixedJoint._joint_break_force is None
        assert FixedJoint._joint_break_torque is None

    def test_custom_params(self):
        """Test decorator with custom parameters."""

        @joint(type="hinge", break_force=1000.0, break_torque=500.0)
        class HingeJoint:
            pass

        assert HingeJoint._joint_type == "hinge"
        assert HingeJoint._joint_break_force == 1000.0
        assert HingeJoint._joint_break_torque == 500.0

    def test_all_valid_types(self):
        """Test all valid joint types."""
        for jtype in VALID_JOINT_TYPE:

            @joint(type=jtype)
            class TestJoint:
                pass

            assert TestJoint._joint_type == jtype

    def test_invalid_type(self):
        """Test validation of type parameter."""
        with pytest.raises(ValueError, match="Invalid type"):

            @joint(type="invalid")
            class BadType:
                pass

    def test_optional_break_values(self):
        """Test that break force and torque are optional."""

        @joint(type="ball")
        class UnbreakableJoint:
            pass

        assert UnbreakableJoint._joint_break_force is None
        assert UnbreakableJoint._joint_break_torque is None


class TestDecoratorComposition:
    """Test combining multiple destruction decorators."""

    def test_multiple_decorators(self):
        """Test applying multiple decorators to same class."""

        @fracture(pattern="voronoi")
        @physics_material(friction=0.7)
        @destructible(health=200.0)
        class ComplexObject:
            pass

        assert ComplexObject._destructible is True
        assert ComplexObject._physics_material is True
        assert ComplexObject._fracture is True

    def test_damage_system_composition(self):
        """Test damage type and resistance together."""

        @damage_resistance(resistances={"fire": 0.5})
        @damage_type(id="fire", base_multiplier=1.5)
        class FireResistant:
            pass

        assert FireResistant._damage_type is True
        assert FireResistant._damage_resistance is True

    def test_full_destruction_stack(self):
        """Test applying all destruction decorators."""

        @joint(type="spring", break_force=100.0)
        @physics_material(friction=0.6, restitution=0.4, density=1.5)
        @fracture(pattern="radial", min_size=0.2)
        @damage_resistance(resistances={"physical": 0.8})
        @damage_type(id="impact", base_multiplier=1.0)
        @destructible(health=300.0, fracture_depth=3)
        class FullyDestructible:
            pass

        # Verify all decorators applied
        assert hasattr(FullyDestructible, "_applied_decorators")
        applied = FullyDestructible._applied_decorators
        assert "destructible" in applied
        assert "damage_type" in applied
        assert "damage_resistance" in applied
        assert "fracture" in applied
        assert "physics_material" in applied
        assert "joint" in applied


class TestRegistryIntegration:
    """Test integration with decorator registry."""

    def test_all_decorators_registered(self):
        """Test that all destruction decorators are registered."""
        decorators = [
            "destructible",
            "damage_type",
            "damage_resistance",
            "fracture",
            "physics_material",
            "joint",
        ]

        for name in decorators:
            spec = registry.get(name)
            assert spec is not None, f"Decorator {name} not registered"
            assert spec.tier.name == "DESTRUCTION"

    def test_destruction_tier_contains_decorators(self):
        """Test that DESTRUCTION tier contains our decorators."""
        from trinity.decorators.registry import Tier

        destruction_specs = registry.by_tier(Tier.DESTRUCTION)
        names = [spec.name for spec in destruction_specs]

        assert "destructible" in names
        assert "damage_type" in names
        assert "damage_resistance" in names
        assert "fracture" in names
        assert "physics_material" in names
        assert "joint" in names
