"""Tests for Trinity Pattern Tier 48: WORLD_BUILDING decorators."""

import pytest

from trinity.decorators.world_building import (
    VALID_NAVMESH_MODIFIERS,
    VALID_TRIGGER_EVENTS,
    VALID_WATER_TYPES,
    foliage_type,
    level_instance,
    navmesh_modifier,
    procedural_placement,
    trigger_volume,
    water_body,
)
from trinity.decorators.registry import Tier, registry


# ============================================================================
# foliage_type tests
# ============================================================================


def test_foliage_type_basic():
    """Test basic @foliage_type application."""

    @foliage_type(density=2.0, cull_distance=5000.0, collision=True, wind_response=0.5)
    class GrassCluster:
        pass

    assert hasattr(GrassCluster, "_foliage_type")
    assert GrassCluster._foliage_type is True
    assert GrassCluster._foliage_density == 2.0
    assert GrassCluster._foliage_cull_distance == 5000.0
    assert GrassCluster._foliage_collision is True
    assert GrassCluster._foliage_wind_response == 0.5
    assert "foliage_type" in GrassCluster._applied_decorators


def test_foliage_type_defaults():
    """Test @foliage_type with default parameters."""

    @foliage_type()
    class TreeCluster:
        pass

    assert TreeCluster._foliage_density == 1.0
    assert TreeCluster._foliage_cull_distance == 10000.0
    assert TreeCluster._foliage_collision is False
    assert TreeCluster._foliage_wind_response == 1.0


def test_foliage_type_invalid_density():
    """Test @foliage_type with invalid density."""
    with pytest.raises(ValueError, match="density must be > 0"):

        @foliage_type(density=0)
        class InvalidFoliage:
            pass


def test_foliage_type_invalid_cull_distance():
    """Test @foliage_type with invalid cull_distance."""
    with pytest.raises(ValueError, match="cull_distance must be > 0"):

        @foliage_type(cull_distance=-100)
        class InvalidFoliage:
            pass


def test_foliage_type_invalid_wind_response():
    """Test @foliage_type with invalid wind_response."""
    with pytest.raises(ValueError, match="wind_response must be >= 0"):

        @foliage_type(wind_response=-0.5)
        class InvalidFoliage:
            pass


def test_foliage_type_tags():
    """Test @foliage_type tags."""

    @foliage_type(density=1.5)
    class FoliageTest:
        pass

    assert hasattr(FoliageTest, "_tags")
    assert FoliageTest._tags["foliage_type"] is True
    assert FoliageTest._tags["foliage_density"] == 1.5


# ============================================================================
# procedural_placement tests
# ============================================================================


def test_procedural_placement_basic():
    """Test basic @procedural_placement application."""

    @procedural_placement(
        density=0.5, noise="simplex", slope_range=(10, 45), height_range=(0, 100)
    )
    class RockScatter:
        pass

    assert hasattr(RockScatter, "_procedural_placement")
    assert RockScatter._procedural_placement is True
    assert RockScatter._placement_density == 0.5
    assert RockScatter._placement_noise == "simplex"
    assert RockScatter._placement_slope_range == (10, 45)
    assert RockScatter._placement_height_range == (0, 100)


def test_procedural_placement_defaults():
    """Test @procedural_placement with default parameters."""

    @procedural_placement(density=1.0)
    class DefaultPlacement:
        pass

    assert DefaultPlacement._placement_noise == "perlin"
    assert DefaultPlacement._placement_slope_range == (0, 90)
    assert DefaultPlacement._placement_height_range is None


def test_procedural_placement_no_density():
    """Test @procedural_placement without required density parameter."""
    with pytest.raises(ValueError, match="density is required"):

        @procedural_placement()
        class InvalidPlacement:
            pass


def test_procedural_placement_invalid_density():
    """Test @procedural_placement with invalid density."""
    with pytest.raises(ValueError, match="density must be > 0"):

        @procedural_placement(density=-1)
        class InvalidPlacement:
            pass


def test_procedural_placement_invalid_slope_range():
    """Test @procedural_placement with invalid slope_range."""
    with pytest.raises(ValueError, match="slope_range\\[0\\] must be <= slope_range\\[1\\]"):

        @procedural_placement(density=1.0, slope_range=(90, 0))
        class InvalidPlacement:
            pass


# ============================================================================
# level_instance tests
# ============================================================================


def test_level_instance_basic():
    """Test basic @level_instance application."""

    @level_instance(always_loaded=True, load_on_proximity=False, proximity_radius=2000.0)
    class SubLevel:
        pass

    assert hasattr(SubLevel, "_level_instance")
    assert SubLevel._level_instance is True
    assert SubLevel._level_always_loaded is True
    assert SubLevel._level_load_on_proximity is False
    assert SubLevel._level_proximity_radius == 2000.0


def test_level_instance_defaults():
    """Test @level_instance with default parameters."""

    @level_instance()
    class DefaultLevel:
        pass

    assert DefaultLevel._level_always_loaded is False
    assert DefaultLevel._level_load_on_proximity is True
    assert DefaultLevel._level_proximity_radius == 10000.0


def test_level_instance_invalid_proximity_radius():
    """Test @level_instance with invalid proximity_radius."""
    with pytest.raises(ValueError, match="proximity_radius must be > 0"):

        @level_instance(proximity_radius=0)
        class InvalidLevel:
            pass


# ============================================================================
# water_body tests
# ============================================================================


def test_water_body_basic():
    """Test basic @water_body application."""

    @water_body(type="ocean", wave_source=True)
    class OceanVolume:
        pass

    assert hasattr(OceanVolume, "_water_body")
    assert OceanVolume._water_body is True
    assert OceanVolume._water_type == "ocean"
    assert OceanVolume._water_wave_source is True


def test_water_body_all_types():
    """Test all valid water types."""
    for water_type in VALID_WATER_TYPES:

        @water_body(type=water_type)
        class WaterTest:
            pass

        assert WaterTest._water_type == water_type


def test_water_body_defaults():
    """Test @water_body with default parameters (should fail without type)."""
    with pytest.raises(ValueError, match="Invalid type"):

        @water_body()
        class DefaultWater:
            pass


def test_water_body_invalid_type():
    """Test @water_body with invalid type."""
    with pytest.raises(ValueError, match="Invalid type"):

        @water_body(type="pond")
        class InvalidWater:
            pass


# ============================================================================
# navmesh_modifier tests
# ============================================================================


def test_navmesh_modifier_basic():
    """Test basic @navmesh_modifier application."""

    @navmesh_modifier(area_class="walkable", modifier="include")
    class WalkableArea:
        pass

    assert hasattr(WalkableArea, "_navmesh_modifier")
    assert WalkableArea._navmesh_modifier is True
    assert WalkableArea._navmesh_area_class == "walkable"
    assert WalkableArea._navmesh_modifier_type == "include"


def test_navmesh_modifier_all_modifiers():
    """Test all valid modifier types."""
    for mod_type in VALID_NAVMESH_MODIFIERS:

        @navmesh_modifier(modifier=mod_type)
        class NavTest:
            pass

        assert NavTest._navmesh_modifier_type == mod_type


def test_navmesh_modifier_defaults():
    """Test @navmesh_modifier with default parameters."""

    @navmesh_modifier()
    class DefaultNav:
        pass

    assert DefaultNav._navmesh_area_class == "default"
    assert DefaultNav._navmesh_modifier_type == "replace"


def test_navmesh_modifier_invalid_modifier():
    """Test @navmesh_modifier with invalid modifier."""
    with pytest.raises(ValueError, match="Invalid modifier"):

        @navmesh_modifier(modifier="invalid")
        class InvalidNav:
            pass


# ============================================================================
# trigger_volume tests
# ============================================================================


def test_trigger_volume_basic():
    """Test basic @trigger_volume application."""

    @trigger_volume(events=["on_enter", "on_exit"], filter_tags={"player", "enemy"})
    class TriggerZone:
        pass

    assert hasattr(TriggerZone, "_trigger_volume")
    assert TriggerZone._trigger_volume is True
    assert "on_enter" in TriggerZone._trigger_events
    assert "on_exit" in TriggerZone._trigger_events
    assert "player" in TriggerZone._trigger_filter_tags
    assert "enemy" in TriggerZone._trigger_filter_tags


def test_trigger_volume_all_events():
    """Test all valid trigger events."""

    @trigger_volume(events=list(VALID_TRIGGER_EVENTS))
    class AllEventsTest:
        pass

    for event in VALID_TRIGGER_EVENTS:
        assert event in AllEventsTest._trigger_events


def test_trigger_volume_defaults():
    """Test @trigger_volume with default filter_tags."""

    @trigger_volume(events=["on_enter"])
    class DefaultTrigger:
        pass

    assert isinstance(DefaultTrigger._trigger_filter_tags, set)
    assert len(DefaultTrigger._trigger_filter_tags) == 0


def test_trigger_volume_empty_events():
    """Test @trigger_volume with empty events list."""
    with pytest.raises(ValueError, match="events must be a non-empty list"):

        @trigger_volume(events=[])
        class InvalidTrigger:
            pass


def test_trigger_volume_invalid_event():
    """Test @trigger_volume with invalid event."""
    with pytest.raises(ValueError, match="Invalid event"):

        @trigger_volume(events=["on_invalid"])
        class InvalidTrigger:
            pass


# ============================================================================
# Registry tests
# ============================================================================


def test_world_building_registry():
    """Test that all WORLD_BUILDING decorators are registered."""
    decorators = registry.by_tier(Tier.WORLD_BUILDING)
    names = {d.name for d in decorators}

    assert "foliage_type" in names
    assert "procedural_placement" in names
    assert "level_instance" in names
    assert "water_body" in names
    assert "navmesh_modifier" in names
    assert "trigger_volume" in names


def test_world_building_tier():
    """Test that all decorators have the correct tier."""
    for name in [
        "foliage_type",
        "procedural_placement",
        "level_instance",
        "water_body",
        "navmesh_modifier",
        "trigger_volume",
    ]:
        spec = registry.get(name)
        assert spec is not None
        assert spec.tier == Tier.WORLD_BUILDING


# ============================================================================
# Composition tests
# ============================================================================


def test_composition_foliage_and_procedural():
    """Test composing @foliage_type and @procedural_placement."""

    @procedural_placement(density=1.5, slope_range=(0, 30))
    @foliage_type(density=2.0, wind_response=0.8)
    class ProceduralFoliage:
        pass

    assert ProceduralFoliage._foliage_type is True
    assert ProceduralFoliage._procedural_placement is True
    assert ProceduralFoliage._foliage_density == 2.0
    assert ProceduralFoliage._placement_density == 1.5


def test_composition_level_and_trigger():
    """Test composing @level_instance and @trigger_volume."""

    @trigger_volume(events=["on_enter"])
    @level_instance(load_on_proximity=True)
    class StreamedLevel:
        pass

    assert StreamedLevel._level_instance is True
    assert StreamedLevel._trigger_volume is True


def test_composition_water_and_navmesh():
    """Test composing @water_body and @navmesh_modifier."""

    @navmesh_modifier(modifier="exclude")
    @water_body(type="river")
    class RiverWithNav:
        pass

    assert RiverWithNav._water_body is True
    assert RiverWithNav._navmesh_modifier is True
    assert RiverWithNav._water_type == "river"
    assert RiverWithNav._navmesh_modifier_type == "exclude"


# ============================================================================
# Steps introspection tests
# ============================================================================


def test_foliage_type_steps():
    """Test @foliage_type generates correct steps."""

    @foliage_type(density=1.5)
    class StepsTest:
        pass

    assert hasattr(StepsTest, "_applied_steps")
    steps = StepsTest._applied_steps
    assert len(steps) > 0

    # Check that TAG and REGISTER steps are present
    from trinity.decorators.ops import Op

    ops = [s.op for s in steps]
    assert Op.TAG in ops
    assert Op.REGISTER in ops
