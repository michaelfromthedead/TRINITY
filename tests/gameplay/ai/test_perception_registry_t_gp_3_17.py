"""
Comprehensive unit tests for Perception Registry integration with Foundation Registry.

Task: T-GP-3.17 - Wire @perception decorator

Tests cover:
- @perception decorator registration with all sense types
- @sense decorator for individual sense configuration
- Range and FOV metadata storage
- Registry query by sense type
- Multiple perception configs coexistence
- PerceptionConfig.from_registry() factory method
- Decay time configuration
- Performance benchmarks
"""

import gc
import time
import pytest
import sys

sys.path.insert(0, '/home/user/dev/USER/PROJECTS_VOID/TRINITY')

from foundation import registry, Registry
from engine.gameplay.constants import (
    PerceptionSense,
    PERCEPTION_DEFAULT_SIGHT_RANGE,
    PERCEPTION_DEFAULT_HEARING_RANGE,
    PERCEPTION_DEFAULT_FOV,
)
from engine.gameplay.ai.perception_registry import (
    perception,
    sense,
    get_all_perception_configs,
    get_perception_configs_by_sense,
    get_all_sense_configs,
    get_sense_configs_by_type,
    get_perception_by_name,
    get_sense_by_name,
    PerceptionConfig,
    SenseConfig,
    create_perception_config_from_registry,
    create_sense_config_from_registry,
    TAG_PERCEPTION,
    TAG_SENSE,
    VALID_SENSE_TYPES,
    SENSE_TYPE_MAP,
    DEFAULT_RANGES,
    DEFAULT_DECAY_TIMES,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Store original state and restore after each test."""
    original_types = dict(registry._types)
    original_names = dict(registry._names)
    original_metadata = dict(registry._metadata)
    original_instances = dict(registry._instances)
    yield
    registry._types.clear()
    registry._types.update(original_types)
    registry._names.clear()
    registry._names.update(original_names)
    registry._metadata.clear()
    registry._metadata.update(original_metadata)
    registry._instances.clear()
    registry._instances.update(original_instances)


# =============================================================================
# Perception Decorator Registration Tests
# =============================================================================


class TestPerceptionDecorator:
    """Tests for @perception decorator."""

    def test_perception_registers_class_with_registry(self):
        """Verify @perception registers the class with Foundation Registry."""
        @perception(sense="sight", range=50.0, fov=90.0)
        class TestSightPerception:
            pass

        assert registry.is_registered(TestSightPerception)

    def test_perception_adds_perception_tag(self):
        """Verify @perception adds the perception tag."""
        @perception(sense="hearing", range=30.0)
        class TestHearingPerception:
            pass

        assert registry.has_tag(TestHearingPerception, TAG_PERCEPTION)

    def test_perception_adds_sense_type_tag(self):
        """Verify @perception adds sense-specific tag."""
        @perception(sense="damage")
        class TestDamagePerception:
            pass

        assert registry.has_tag(TestDamagePerception, "sense_damage")

    def test_perception_stores_sense_metadata(self):
        """Verify @perception stores sense type in metadata."""
        @perception(sense="squad")
        class TestSquadPerception:
            pass

        assert registry.get_metadata(TestSquadPerception, "sense") == "squad"

    def test_perception_stores_range_metadata(self):
        """Verify @perception stores range in metadata."""
        @perception(sense="sight", range=100.0)
        class TestLongRangePerception:
            pass

        assert registry.get_metadata(TestLongRangePerception, "range") == 100.0

    def test_perception_stores_fov_metadata(self):
        """Verify @perception stores FOV in metadata."""
        @perception(sense="sight", fov=120.0)
        class TestWideFOVPerception:
            pass

        assert registry.get_metadata(TestWideFOVPerception, "fov") == 120.0

    def test_perception_stores_decay_time_metadata(self):
        """Verify @perception stores decay_time in metadata."""
        @perception(sense="hearing", decay_time=5.0)
        class TestSlowDecayPerception:
            pass

        assert registry.get_metadata(TestSlowDecayPerception, "decay_time") == 5.0

    def test_perception_stores_sense_enum(self):
        """Verify @perception stores sense enum in metadata."""
        @perception(sense="sight")
        class TestSightPerception:
            pass

        assert registry.get_metadata(TestSightPerception, "sense_enum") == PerceptionSense.SIGHT

    def test_perception_custom_name(self):
        """Verify @perception uses custom name when provided."""
        @perception(sense="sight", name="custom.sniper.perception")
        class SniperPerception:
            pass

        assert registry.get("custom.sniper.perception") is SniperPerception

    def test_perception_description(self):
        """Verify @perception stores description in metadata."""
        @perception(sense="hearing", description="Enhanced hearing for stealth detection")
        class EnhancedHearing:
            pass

        assert registry.get_metadata(EnhancedHearing, "description") == "Enhanced hearing for stealth detection"

    def test_perception_instance_tracking(self):
        """Verify @perception enables instance tracking when requested."""
        @perception(sense="sight", track_instances=True)
        class TrackedSightPerception:
            pass

        obj1 = TrackedSightPerception()
        obj2 = TrackedSightPerception()
        assert registry.instance_count(TrackedSightPerception) == 2

    def test_perception_class_attributes(self):
        """Verify @perception sets class attributes for introspection."""
        @perception(sense="damage", range=10.0, fov=360.0, decay_time=4.0)
        class DamagePerception:
            pass

        assert DamagePerception._perception is True
        assert DamagePerception._perception_sense == "damage"
        assert DamagePerception._perception_sense_enum == PerceptionSense.DAMAGE
        assert DamagePerception._perception_range == 10.0
        assert DamagePerception._perception_fov == 360.0
        assert DamagePerception._perception_decay_time == 4.0

    def test_perception_invalid_sense_raises_error(self):
        """Verify @perception raises ValueError for invalid sense types."""
        with pytest.raises(ValueError, match="Invalid sense type"):
            @perception(sense="telepathy")
            class InvalidPerception:
                pass

    def test_perception_sight_registers(self):
        """Verify sight sense type registers correctly."""
        @perception(sense="sight")
        class SightPerception:
            pass

        assert registry.has_tag(SightPerception, TAG_PERCEPTION)
        assert registry.get_metadata(SightPerception, "sense") == "sight"

    def test_perception_hearing_registers(self):
        """Verify hearing sense type registers correctly."""
        @perception(sense="hearing")
        class HearingPerception:
            pass

        assert registry.has_tag(HearingPerception, TAG_PERCEPTION)
        assert registry.get_metadata(HearingPerception, "sense") == "hearing"

    def test_perception_damage_registers(self):
        """Verify damage sense type registers correctly."""
        @perception(sense="damage")
        class DamagePerception:
            pass

        assert registry.has_tag(DamagePerception, TAG_PERCEPTION)
        assert registry.get_metadata(DamagePerception, "sense") == "damage"

    def test_perception_squad_registers(self):
        """Verify squad sense type registers correctly."""
        @perception(sense="squad")
        class SquadPerception:
            pass

        assert registry.has_tag(SquadPerception, TAG_PERCEPTION)
        assert registry.get_metadata(SquadPerception, "sense") == "squad"

    def test_perception_default_range_sight(self):
        """Verify default range for sight sense."""
        @perception(sense="sight")
        class DefaultSight:
            pass

        assert registry.get_metadata(DefaultSight, "range") == PERCEPTION_DEFAULT_SIGHT_RANGE

    def test_perception_default_range_hearing(self):
        """Verify default range for hearing sense."""
        @perception(sense="hearing")
        class DefaultHearing:
            pass

        assert registry.get_metadata(DefaultHearing, "range") == PERCEPTION_DEFAULT_HEARING_RANGE

    def test_perception_default_fov(self):
        """Verify default FOV is applied."""
        @perception(sense="sight")
        class DefaultFOV:
            pass

        assert registry.get_metadata(DefaultFOV, "fov") == PERCEPTION_DEFAULT_FOV


# =============================================================================
# Sense Decorator Registration Tests
# =============================================================================


class TestSenseDecorator:
    """Tests for @sense decorator."""

    def test_sense_registers_class_with_registry(self):
        """Verify @sense registers the class with Foundation Registry."""
        @sense(type="sight", range=50.0)
        class TestSightSense:
            pass

        assert registry.is_registered(TestSightSense)

    def test_sense_adds_sense_tag(self):
        """Verify @sense adds the sense tag."""
        @sense(type="hearing")
        class TestHearingSense:
            pass

        assert registry.has_tag(TestHearingSense, TAG_SENSE)

    def test_sense_adds_type_specific_tag(self):
        """Verify @sense adds type-specific tag."""
        @sense(type="damage")
        class TestDamageSense:
            pass

        assert registry.has_tag(TestDamageSense, "sense_damage")

    def test_sense_stores_type_metadata(self):
        """Verify @sense stores sense_type in metadata."""
        @sense(type="squad")
        class TestSquadSense:
            pass

        assert registry.get_metadata(TestSquadSense, "sense_type") == "squad"

    def test_sense_stores_range_metadata(self):
        """Verify @sense stores range in metadata."""
        @sense(type="sight", range=75.0)
        class TestRangeSense:
            pass

        assert registry.get_metadata(TestRangeSense, "range") == 75.0

    def test_sense_stores_fov_metadata(self):
        """Verify @sense stores FOV in metadata."""
        @sense(type="sight", fov=180.0)
        class TestFOVSense:
            pass

        assert registry.get_metadata(TestFOVSense, "fov") == 180.0

    def test_sense_stores_decay_time_metadata(self):
        """Verify @sense stores decay_time in metadata."""
        @sense(type="hearing", decay_time=10.0)
        class TestDecaySense:
            pass

        assert registry.get_metadata(TestDecaySense, "decay_time") == 10.0

    def test_sense_class_attributes(self):
        """Verify @sense sets class attributes for introspection."""
        @sense(type="sight", range=100.0, fov=90.0, decay_time=5.0)
        class SightSense:
            pass

        assert SightSense._sense is True
        assert SightSense._sense_type == "sight"
        assert SightSense._sense_type_enum == PerceptionSense.SIGHT
        assert SightSense._sense_range == 100.0
        assert SightSense._sense_fov == 90.0
        assert SightSense._sense_decay_time == 5.0

    def test_sense_invalid_type_raises_error(self):
        """Verify @sense raises ValueError for invalid sense types."""
        with pytest.raises(ValueError, match="Invalid sense type"):
            @sense(type="smell")
            class InvalidSense:
                pass

    def test_sense_all_valid_types(self):
        """Verify all valid sense types can be registered."""
        for sense_type in VALID_SENSE_TYPES:
            @sense(type=sense_type, name=f"test.sense.{sense_type}")
            class TestSense:
                pass

            assert registry.get_metadata(TestSense, "sense_type") == sense_type


# =============================================================================
# Registry Query Tests
# =============================================================================


class TestPerceptionQueries:
    """Tests for perception registry queries."""

    def test_get_all_perception_configs(self):
        """Verify get_all_perception_configs returns all registered perceptions."""
        @perception(sense="sight", name="query.sight")
        class SightPerception:
            pass

        @perception(sense="hearing", name="query.hearing")
        class HearingPerception:
            pass

        configs = get_all_perception_configs()
        assert SightPerception in configs
        assert HearingPerception in configs

    def test_get_perception_configs_by_sense_sight(self):
        """Verify filtering perceptions by sight sense."""
        @perception(sense="sight", name="filter.sight")
        class SightPerception:
            pass

        @perception(sense="hearing", name="filter.hearing")
        class HearingPerception:
            pass

        configs = get_perception_configs_by_sense("sight")
        assert SightPerception in configs
        assert HearingPerception not in configs

    def test_get_perception_configs_by_sense_hearing(self):
        """Verify filtering perceptions by hearing sense."""
        @perception(sense="sight", name="filter2.sight")
        class SightPerception:
            pass

        @perception(sense="hearing", name="filter2.hearing")
        class HearingPerception:
            pass

        configs = get_perception_configs_by_sense("hearing")
        assert HearingPerception in configs
        assert SightPerception not in configs

    def test_get_perception_configs_by_sense_damage(self):
        """Verify filtering perceptions by damage sense."""
        @perception(sense="damage", name="filter3.damage")
        class DamagePerception:
            pass

        configs = get_perception_configs_by_sense("damage")
        assert DamagePerception in configs

    def test_get_perception_configs_by_sense_squad(self):
        """Verify filtering perceptions by squad sense."""
        @perception(sense="squad", name="filter4.squad")
        class SquadPerception:
            pass

        configs = get_perception_configs_by_sense("squad")
        assert SquadPerception in configs

    def test_get_all_sense_configs(self):
        """Verify get_all_sense_configs returns all registered senses."""
        @sense(type="sight", name="sense.query.sight")
        class SightSense:
            pass

        @sense(type="hearing", name="sense.query.hearing")
        class HearingSense:
            pass

        configs = get_all_sense_configs()
        assert SightSense in configs
        assert HearingSense in configs

    def test_get_sense_configs_by_type(self):
        """Verify filtering senses by type."""
        @sense(type="sight", name="sense.filter.sight")
        class SightSense:
            pass

        @sense(type="damage", name="sense.filter.damage")
        class DamageSense:
            pass

        sight_configs = get_sense_configs_by_type("sight")
        assert SightSense in sight_configs
        assert DamageSense not in sight_configs

    def test_get_perception_by_name(self):
        """Verify getting perception by name."""
        @perception(sense="sight", name="named.eagle.sight")
        class EagleSight:
            pass

        result = get_perception_by_name("named.eagle.sight")
        assert result is EagleSight

    def test_get_sense_by_name(self):
        """Verify getting sense by name."""
        @sense(type="hearing", name="named.bat.hearing")
        class BatHearing:
            pass

        result = get_sense_by_name("named.bat.hearing")
        assert result is BatHearing

    def test_query_invalid_sense_type_returns_empty(self):
        """Verify querying invalid sense type returns empty list."""
        configs = get_perception_configs_by_sense("invalid")
        assert configs == []

    def test_get_sense_configs_invalid_type_returns_empty(self):
        """Verify querying invalid sense type returns empty list for senses."""
        configs = get_sense_configs_by_type("invalid")
        assert configs == []


# =============================================================================
# Multiple Perception Configs Tests
# =============================================================================


class TestMultiplePerceptionConfigs:
    """Tests for multiple perception configs coexisting."""

    def test_multiple_sight_perceptions_coexist(self):
        """Verify multiple sight perceptions can coexist."""
        @perception(sense="sight", range=50.0, name="multi.sniper")
        class SniperSight:
            pass

        @perception(sense="sight", range=100.0, name="multi.eagle")
        class EagleSight:
            pass

        @perception(sense="sight", range=25.0, name="multi.close")
        class CloseRangeSight:
            pass

        configs = get_perception_configs_by_sense("sight")
        assert len([c for c in configs if c in [SniperSight, EagleSight, CloseRangeSight]]) == 3

    def test_mixed_sense_perceptions_coexist(self):
        """Verify perceptions with different senses coexist."""
        @perception(sense="sight", name="mixed.sight")
        class SightPerception:
            pass

        @perception(sense="hearing", name="mixed.hearing")
        class HearingPerception:
            pass

        @perception(sense="damage", name="mixed.damage")
        class DamagePerception:
            pass

        @perception(sense="squad", name="mixed.squad")
        class SquadPerception:
            pass

        all_configs = get_all_perception_configs()
        assert SightPerception in all_configs
        assert HearingPerception in all_configs
        assert DamagePerception in all_configs
        assert SquadPerception in all_configs

    def test_perception_and_sense_coexist(self):
        """Verify @perception and @sense decorated classes coexist."""
        @perception(sense="sight", name="coexist.perception")
        class SightPerception:
            pass

        @sense(type="sight", name="coexist.sense")
        class SightSense:
            pass

        perceptions = get_all_perception_configs()
        senses = get_all_sense_configs()

        assert SightPerception in perceptions
        assert SightSense in senses
        assert SightPerception not in senses
        assert SightSense not in perceptions


# =============================================================================
# PerceptionConfig Factory Tests
# =============================================================================


class TestPerceptionConfigFactory:
    """Tests for PerceptionConfig.from_registry() factory method."""

    def test_perception_config_from_registry(self):
        """Verify PerceptionConfig.from_registry creates config from registry."""
        @perception(sense="sight", range=75.0, fov=110.0, decay_time=4.0, name="factory.eagle")
        class EaglePerception:
            pass

        config = PerceptionConfig.from_registry("factory.eagle")

        assert config.name == "factory.eagle"
        assert config.sense == "sight"
        assert config.sense_enum == PerceptionSense.SIGHT
        assert config.range == 75.0
        assert config.fov == 110.0
        assert config.decay_time == 4.0
        assert config.source_class is EaglePerception

    def test_perception_config_from_registry_not_found(self):
        """Verify PerceptionConfig.from_registry raises for unknown name."""
        with pytest.raises(ValueError, match="not found in registry"):
            PerceptionConfig.from_registry("nonexistent.perception")

    def test_perception_config_from_registry_not_perception(self):
        """Verify PerceptionConfig.from_registry raises for non-perception class."""
        @sense(type="sight", name="notperception.sense")
        class NotPerception:
            pass

        with pytest.raises(ValueError, match="not a registered perception config"):
            PerceptionConfig.from_registry("notperception.sense")

    def test_perception_config_from_class(self):
        """Verify PerceptionConfig.from_class creates config from decorated class."""
        @perception(sense="hearing", range=40.0, fov=360.0, decay_time=2.0)
        class HearingPerception:
            pass

        config = PerceptionConfig.from_class(HearingPerception)

        assert config.sense == "hearing"
        assert config.sense_enum == PerceptionSense.HEARING
        assert config.range == 40.0
        assert config.fov == 360.0
        assert config.decay_time == 2.0
        assert config.source_class is HearingPerception

    def test_perception_config_from_class_not_perception(self):
        """Verify PerceptionConfig.from_class raises for non-perception class."""
        class NotPerception:
            pass

        with pytest.raises(ValueError, match="is not a perception config"):
            PerceptionConfig.from_class(NotPerception)

    def test_perception_config_to_dict(self):
        """Verify PerceptionConfig.to_dict produces correct output."""
        @perception(sense="damage", range=0.0, fov=360.0, decay_time=5.0, name="dict.damage")
        class DamagePerception:
            pass

        config = PerceptionConfig.from_registry("dict.damage")
        result = config.to_dict()

        assert result["name"] == "dict.damage"
        assert result["sense"] == "damage"
        assert result["sense_enum"] == PerceptionSense.DAMAGE.value
        assert result["range"] == 0.0
        assert result["fov"] == 360.0
        assert result["decay_time"] == 5.0

    def test_create_perception_config_from_registry(self):
        """Verify create_perception_config_from_registry helper function."""
        @perception(sense="squad", range=100.0, name="helper.squad")
        class SquadPerception:
            pass

        config = create_perception_config_from_registry("helper.squad")
        assert config.sense == "squad"
        assert config.range == 100.0

    def test_create_perception_config_with_overrides(self):
        """Verify create_perception_config_from_registry with overrides."""
        @perception(sense="sight", range=50.0, fov=90.0, name="override.sight")
        class SightPerception:
            pass

        config = create_perception_config_from_registry(
            "override.sight",
            range=100.0,
            fov=180.0
        )
        assert config.range == 100.0
        assert config.fov == 180.0


# =============================================================================
# SenseConfig Factory Tests
# =============================================================================


class TestSenseConfigFactory:
    """Tests for SenseConfig.from_registry() factory method."""

    def test_sense_config_from_registry(self):
        """Verify SenseConfig.from_registry creates config from registry."""
        @sense(type="sight", range=60.0, fov=100.0, decay_time=3.0, name="sense.factory.eagle")
        class EagleSense:
            pass

        config = SenseConfig.from_registry("sense.factory.eagle")

        assert config.name == "sense.factory.eagle"
        assert config.sense_type == "sight"
        assert config.sense_type_enum == PerceptionSense.SIGHT
        assert config.range == 60.0
        assert config.fov == 100.0
        assert config.decay_time == 3.0
        assert config.source_class is EagleSense

    def test_sense_config_from_registry_not_found(self):
        """Verify SenseConfig.from_registry raises for unknown name."""
        with pytest.raises(ValueError, match="not found in registry"):
            SenseConfig.from_registry("nonexistent.sense")

    def test_sense_config_from_registry_not_sense(self):
        """Verify SenseConfig.from_registry raises for non-sense class."""
        @perception(sense="sight", name="notsense.perception")
        class NotSense:
            pass

        with pytest.raises(ValueError, match="not a registered sense config"):
            SenseConfig.from_registry("notsense.perception")

    def test_sense_config_from_class(self):
        """Verify SenseConfig.from_class creates config from decorated class."""
        @sense(type="hearing", range=45.0, fov=360.0, decay_time=2.5)
        class HearingSense:
            pass

        config = SenseConfig.from_class(HearingSense)

        assert config.sense_type == "hearing"
        assert config.sense_type_enum == PerceptionSense.HEARING
        assert config.range == 45.0
        assert config.fov == 360.0
        assert config.decay_time == 2.5
        assert config.source_class is HearingSense

    def test_sense_config_from_class_not_sense(self):
        """Verify SenseConfig.from_class raises for non-sense class."""
        class NotSense:
            pass

        with pytest.raises(ValueError, match="is not a sense config"):
            SenseConfig.from_class(NotSense)

    def test_sense_config_to_dict(self):
        """Verify SenseConfig.to_dict produces correct output."""
        @sense(type="squad", range=150.0, fov=360.0, decay_time=15.0, name="sense.dict.squad")
        class SquadSense:
            pass

        config = SenseConfig.from_registry("sense.dict.squad")
        result = config.to_dict()

        assert result["name"] == "sense.dict.squad"
        assert result["sense_type"] == "squad"
        assert result["sense_type_enum"] == PerceptionSense.SQUAD.value
        assert result["range"] == 150.0
        assert result["fov"] == 360.0
        assert result["decay_time"] == 15.0

    def test_create_sense_config_from_registry(self):
        """Verify create_sense_config_from_registry helper function."""
        @sense(type="damage", range=0.0, name="sense.helper.damage")
        class DamageSense:
            pass

        config = create_sense_config_from_registry("sense.helper.damage")
        assert config.sense_type == "damage"
        assert config.range == 0.0

    def test_create_sense_config_with_overrides(self):
        """Verify create_sense_config_from_registry with overrides."""
        @sense(type="hearing", range=30.0, fov=360.0, name="sense.override.hearing")
        class HearingSense:
            pass

        config = create_sense_config_from_registry(
            "sense.override.hearing",
            range=60.0,
            decay_time=10.0
        )
        assert config.range == 60.0
        assert config.decay_time == 10.0


# =============================================================================
# Decay Time Configuration Tests
# =============================================================================


class TestDecayTimeConfiguration:
    """Tests for decay time configuration."""

    def test_default_decay_time_sight(self):
        """Verify default decay time for sight."""
        @perception(sense="sight", name="decay.sight")
        class SightPerception:
            pass

        assert registry.get_metadata(SightPerception, "decay_time") == DEFAULT_DECAY_TIMES["sight"]

    def test_default_decay_time_hearing(self):
        """Verify default decay time for hearing."""
        @perception(sense="hearing", name="decay.hearing")
        class HearingPerception:
            pass

        assert registry.get_metadata(HearingPerception, "decay_time") == DEFAULT_DECAY_TIMES["hearing"]

    def test_default_decay_time_damage(self):
        """Verify default decay time for damage."""
        @perception(sense="damage", name="decay.damage")
        class DamagePerception:
            pass

        assert registry.get_metadata(DamagePerception, "decay_time") == DEFAULT_DECAY_TIMES["damage"]

    def test_default_decay_time_squad(self):
        """Verify default decay time for squad."""
        @perception(sense="squad", name="decay.squad")
        class SquadPerception:
            pass

        assert registry.get_metadata(SquadPerception, "decay_time") == DEFAULT_DECAY_TIMES["squad"]

    def test_custom_decay_time_overrides_default(self):
        """Verify custom decay time overrides default."""
        @perception(sense="sight", decay_time=10.0, name="decay.custom")
        class CustomDecay:
            pass

        assert registry.get_metadata(CustomDecay, "decay_time") == 10.0

    def test_zero_decay_time(self):
        """Verify zero decay time can be set."""
        @perception(sense="sight", decay_time=0.0, name="decay.zero")
        class ZeroDecay:
            pass

        assert registry.get_metadata(ZeroDecay, "decay_time") == 0.0


# =============================================================================
# Constants and Mappings Tests
# =============================================================================


class TestConstantsAndMappings:
    """Tests for module constants and mappings."""

    def test_valid_sense_types_complete(self):
        """Verify VALID_SENSE_TYPES contains all expected types."""
        expected = {"sight", "hearing", "damage", "squad"}
        assert VALID_SENSE_TYPES == expected

    def test_sense_type_map_complete(self):
        """Verify SENSE_TYPE_MAP maps all valid types to enums."""
        assert SENSE_TYPE_MAP["sight"] == PerceptionSense.SIGHT
        assert SENSE_TYPE_MAP["hearing"] == PerceptionSense.HEARING
        assert SENSE_TYPE_MAP["damage"] == PerceptionSense.DAMAGE
        assert SENSE_TYPE_MAP["squad"] == PerceptionSense.SQUAD

    def test_default_ranges_defined(self):
        """Verify DEFAULT_RANGES has values for all sense types."""
        for sense_type in VALID_SENSE_TYPES:
            assert sense_type in DEFAULT_RANGES

    def test_default_decay_times_defined(self):
        """Verify DEFAULT_DECAY_TIMES has values for all sense types."""
        for sense_type in VALID_SENSE_TYPES:
            assert sense_type in DEFAULT_DECAY_TIMES

    def test_tag_perception_constant(self):
        """Verify TAG_PERCEPTION constant value."""
        assert TAG_PERCEPTION == "perception"

    def test_tag_sense_constant(self):
        """Verify TAG_SENSE constant value."""
        assert TAG_SENSE == "sense"


# =============================================================================
# Performance Benchmark Tests
# =============================================================================


class TestPerformanceBenchmarks:
    """Performance benchmark tests for perception registry."""

    def test_100_queries_under_50ms(self):
        """Verify 100 registry queries complete under 50ms."""
        # Register several perception configs
        for i in range(10):
            @perception(sense="sight", range=float(i * 10), name=f"perf.sight.{i}")
            class PerfSightPerception:
                pass

            @perception(sense="hearing", range=float(i * 5), name=f"perf.hearing.{i}")
            class PerfHearingPerception:
                pass

        # Perform 100 queries
        start_time = time.perf_counter()
        for _ in range(100):
            get_all_perception_configs()
        elapsed = (time.perf_counter() - start_time) * 1000

        assert elapsed < 50, f"100 queries took {elapsed:.2f}ms, expected < 50ms"

    def test_sense_type_filter_performance(self):
        """Verify sense type filtering is fast."""
        # Register perceptions
        for i in range(20):
            @perception(sense="sight", name=f"filter.perf.sight.{i}")
            class FilterSightPerception:
                pass

            @perception(sense="hearing", name=f"filter.perf.hearing.{i}")
            class FilterHearingPerception:
                pass

        # Perform filtered queries
        start_time = time.perf_counter()
        for _ in range(50):
            get_perception_configs_by_sense("sight")
            get_perception_configs_by_sense("hearing")
        elapsed = (time.perf_counter() - start_time) * 1000

        assert elapsed < 50, f"100 filtered queries took {elapsed:.2f}ms, expected < 50ms"

    def test_factory_instantiation_performance(self):
        """Verify factory instantiation is fast."""
        @perception(sense="sight", range=50.0, fov=90.0, name="factory.perf")
        class FactoryPerfPerception:
            pass

        # Perform 100 factory instantiations
        start_time = time.perf_counter()
        for _ in range(100):
            PerceptionConfig.from_registry("factory.perf")
        elapsed = (time.perf_counter() - start_time) * 1000

        assert elapsed < 50, f"100 factory calls took {elapsed:.2f}ms, expected < 50ms"

    def test_registration_performance(self):
        """Verify registration is fast."""
        start_time = time.perf_counter()
        for i in range(50):
            @perception(sense="sight", name=f"reg.perf.{i}")
            class RegPerfPerception:
                pass
        elapsed = (time.perf_counter() - start_time) * 1000

        assert elapsed < 100, f"50 registrations took {elapsed:.2f}ms, expected < 100ms"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_range(self):
        """Verify zero range is valid."""
        @perception(sense="damage", range=0.0, name="edge.zero.range")
        class ZeroRangePerception:
            pass

        assert registry.get_metadata(ZeroRangePerception, "range") == 0.0

    def test_negative_range_allowed(self):
        """Verify negative range is allowed (may be used for special cases)."""
        @perception(sense="sight", range=-10.0, name="edge.negative.range")
        class NegativeRangePerception:
            pass

        assert registry.get_metadata(NegativeRangePerception, "range") == -10.0

    def test_large_range(self):
        """Verify large range values work."""
        @perception(sense="squad", range=10000.0, name="edge.large.range")
        class LargeRangePerception:
            pass

        assert registry.get_metadata(LargeRangePerception, "range") == 10000.0

    def test_zero_fov(self):
        """Verify zero FOV is valid."""
        @perception(sense="sight", fov=0.0, name="edge.zero.fov")
        class ZeroFOVPerception:
            pass

        assert registry.get_metadata(ZeroFOVPerception, "fov") == 0.0

    def test_full_fov(self):
        """Verify 360 degree FOV is valid."""
        @perception(sense="hearing", fov=360.0, name="edge.full.fov")
        class FullFOVPerception:
            pass

        assert registry.get_metadata(FullFOVPerception, "fov") == 360.0

    def test_empty_description(self):
        """Verify empty description is valid."""
        @perception(sense="sight", description="", name="edge.empty.desc")
        class EmptyDescPerception:
            pass

        # Empty string should not be stored as description
        assert registry.get_metadata(EmptyDescPerception, "description") is None

    def test_unicode_description(self):
        """Verify unicode characters in description work."""
        @perception(sense="sight", description="Perception avec des caracteres speciaux", name="edge.unicode")
        class UnicodePerception:
            pass

        desc = registry.get_metadata(UnicodePerception, "description")
        assert "caracteres" in desc

    def test_reregistration_same_name(self):
        """Verify re-registration with same name doesn't raise."""
        @perception(sense="sight", name="edge.reregister")
        class FirstPerception:
            pass

        # This should not raise (already registered case)
        @perception(sense="hearing", name="edge.reregister.2")
        class SecondPerception:
            pass

        # Verify both are registered
        assert registry.is_registered(FirstPerception)
        assert registry.is_registered(SecondPerception)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_perception_workflow(self):
        """Test complete workflow: register, query, instantiate."""
        # 1. Register perception
        @perception(
            sense="sight",
            range=100.0,
            fov=120.0,
            decay_time=5.0,
            description="Sniper vision",
            name="integration.sniper"
        )
        class SniperVision:
            pass

        # 2. Query and verify
        configs = get_perception_configs_by_sense("sight")
        assert SniperVision in configs

        # 3. Get by name
        found = get_perception_by_name("integration.sniper")
        assert found is SniperVision

        # 4. Create config from registry
        config = PerceptionConfig.from_registry("integration.sniper")
        assert config.sense == "sight"
        assert config.range == 100.0
        assert config.fov == 120.0
        assert config.decay_time == 5.0

        # 5. Create config from class
        config2 = PerceptionConfig.from_class(SniperVision)
        assert config2.sense == config.sense
        assert config2.range == config.range

        # 6. Convert to dict
        data = config.to_dict()
        assert data["sense"] == "sight"
        assert data["range"] == 100.0

    def test_multiple_sense_types_workflow(self):
        """Test workflow with all sense types."""
        # Register all sense types
        @perception(sense="sight", name="workflow.sight")
        class SightPerception:
            pass

        @perception(sense="hearing", name="workflow.hearing")
        class HearingPerception:
            pass

        @perception(sense="damage", name="workflow.damage")
        class DamagePerception:
            pass

        @perception(sense="squad", name="workflow.squad")
        class SquadPerception:
            pass

        # Query each type
        sight_configs = get_perception_configs_by_sense("sight")
        hearing_configs = get_perception_configs_by_sense("hearing")
        damage_configs = get_perception_configs_by_sense("damage")
        squad_configs = get_perception_configs_by_sense("squad")

        assert SightPerception in sight_configs
        assert HearingPerception in hearing_configs
        assert DamagePerception in damage_configs
        assert SquadPerception in squad_configs

        # Query all
        all_configs = get_all_perception_configs()
        assert len([c for c in all_configs if c in [
            SightPerception, HearingPerception, DamagePerception, SquadPerception
        ]]) == 4

    def test_perception_and_sense_combination(self):
        """Test using both @perception and @sense decorators together."""
        # High-level perception config
        @perception(sense="sight", range=50.0, name="combo.perception")
        class BasicSight:
            pass

        # Individual sense components
        @sense(type="sight", range=100.0, fov=120.0, name="combo.eagle.sight")
        class EagleSight:
            pass

        @sense(type="hearing", range=80.0, name="combo.bat.hearing")
        class BatHearing:
            pass

        # Query separately
        perceptions = get_all_perception_configs()
        senses = get_all_sense_configs()

        assert BasicSight in perceptions
        assert BasicSight not in senses

        assert EagleSight in senses
        assert BatHearing in senses
        assert EagleSight not in perceptions
