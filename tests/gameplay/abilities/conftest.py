"""
Shared fixtures for abilities system tests.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, List, Optional
from uuid import uuid4

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeSet,
    create_standard_attributes,
)
from engine.gameplay.abilities.constants import (
    AbilityPhase,
    AbilityEndReason,
    EffectType,
    ModifierOperation,
    StackingMode,
    TargetingMode,
    AreaShape,
)
from engine.gameplay.abilities.effects import (
    EffectContext,
    EffectModifier,
    InstantEffect,
    DurationEffect,
    InfiniteEffect,
    PeriodicEffect,
    EffectContainer,
)
from engine.gameplay.abilities.targeting import (
    Vector3,
    TargetData,
    TargetFilter,
    SelfTargeting,
    ActorTargeting,
    PointTargeting,
    AreaTargeting,
    ConfirmationTargeting,
)
from engine.gameplay.abilities.tags import (
    GameplayTag,
    GameplayTagContainer,
    GameplayTagQuery,
    GameplayTagRegistry,
)


# =============================================================================
# MOCK ENTITIES
# =============================================================================


@dataclass
class MockEntity:
    """Mock entity for testing."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "TestEntity"
    position: Vector3 = field(default_factory=Vector3.zero)
    tags: GameplayTagContainer = field(default_factory=GameplayTagContainer)
    attributes: AttributeSet = field(default_factory=AttributeSet)
    is_valid: bool = True
    is_alive: bool = True
    team: int = 0

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MockEntity):
            return self.id == other.id
        return False


@dataclass
class MockAbilitySpec:
    """Mock ability specification for testing."""

    name: str = "TestAbility"
    cooldown: float = 1.0
    mana_cost: float = 10.0
    stamina_cost: float = 0.0
    global_cooldown: float = 1.0
    can_be_cancelled: bool = True
    can_be_interrupted: bool = True
    is_passive: bool = False
    is_toggled: bool = False
    is_channeled: bool = False
    channel_duration: float = 0.0
    granted_tags: List[str] = field(default_factory=list)
    blocked_by_tags: List[str] = field(default_factory=list)
    cancel_tags: List[str] = field(default_factory=list)
    effects: List[Any] = field(default_factory=list)


@dataclass
class MockAbilityInstance:
    """Mock ability instance for testing."""

    spec: MockAbilitySpec
    owner: Optional[MockEntity] = None
    phase: AbilityPhase = AbilityPhase.NONE
    is_active: bool = False
    remaining_cooldown: float = 0.0
    channel_time: float = 0.0
    is_toggled_on: bool = False


@dataclass
class MockBuff:
    """Mock buff for testing."""

    name: str = "TestBuff"
    duration: float = 10.0
    stacking_mode: StackingMode = StackingMode.NONE
    max_stacks: int = 1
    current_stacks: int = 1
    remaining_duration: float = 10.0
    is_debuff: bool = False
    category: str = "buff"
    icon: str = ""
    effects: List[Any] = field(default_factory=list)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def empty_attribute():
    """Create an empty attribute."""
    return Attribute(name="test", base_value=0.0)


@pytest.fixture
def health_attribute():
    """Create a health attribute with standard values."""
    return Attribute(name="health", base_value=100.0, min_value=0.0, max_value=1000.0)


@pytest.fixture
def standard_attributes():
    """Create a standard attribute set."""
    return create_standard_attributes()


@pytest.fixture
def empty_tag_container():
    """Create an empty tag container."""
    return GameplayTagContainer()


@pytest.fixture
def populated_tag_container():
    """Create a tag container with common tags."""
    container = GameplayTagContainer()
    container.add_many([
        "ability.offensive.fire",
        "ability.defensive.shield",
        "status.buff.strength",
        "status.debuff.slow",
    ])
    return container


@pytest.fixture
def mock_entity():
    """Create a basic mock entity."""
    entity = MockEntity()
    entity.attributes = create_standard_attributes()
    return entity


@pytest.fixture
def mock_source_entity():
    """Create a source entity (caster)."""
    entity = MockEntity(name="Source", team=1)
    entity.position = Vector3(0, 0, 0)
    entity.attributes = create_standard_attributes()
    return entity


@pytest.fixture
def mock_target_entity():
    """Create a target entity."""
    entity = MockEntity(name="Target", team=2)
    entity.position = Vector3(5, 0, 0)
    entity.attributes = create_standard_attributes()
    return entity


@pytest.fixture
def mock_friendly_entity():
    """Create a friendly entity."""
    entity = MockEntity(name="Friendly", team=1)
    entity.position = Vector3(3, 0, 0)
    entity.attributes = create_standard_attributes()
    return entity


@pytest.fixture
def mock_candidates(mock_source_entity, mock_target_entity, mock_friendly_entity):
    """Create a list of candidate targets."""
    return [mock_source_entity, mock_target_entity, mock_friendly_entity]


@pytest.fixture
def instant_damage_effect():
    """Create an instant damage effect."""
    return InstantEffect(
        name="instant_damage",
        modifiers=[
            EffectModifier(
                attribute="health",
                operation=ModifierOperation.ADD,
                base_magnitude=-25.0,
            )
        ],
    )


@pytest.fixture
def duration_buff_effect():
    """Create a duration buff effect."""
    return DurationEffect(
        name="strength_buff",
        duration=10.0,
        modifiers=[
            EffectModifier(
                attribute="damage",
                operation=ModifierOperation.MULTIPLY,
                base_magnitude=0.5,
            )
        ],
    )


@pytest.fixture
def periodic_dot_effect():
    """Create a periodic damage over time effect."""
    return PeriodicEffect(
        name="poison",
        duration=5.0,
        tick_rate=1.0,
        modifiers=[
            EffectModifier(
                attribute="health",
                operation=ModifierOperation.ADD,
                base_magnitude=-10.0,
            )
        ],
    )


@pytest.fixture
def effect_container(standard_attributes, empty_tag_container):
    """Create an effect container."""
    return EffectContainer(standard_attributes, empty_tag_container)


@pytest.fixture
def basic_target_filter():
    """Create a basic target filter."""
    return TargetFilter(
        allow_self=False,
        allow_dead=False,
        allow_friendly=True,
        allow_hostile=True,
    )


@pytest.fixture
def hostile_only_filter():
    """Create a filter for hostile targets only."""
    return TargetFilter(
        allow_self=False,
        allow_dead=False,
        allow_friendly=False,
        allow_hostile=True,
    )


@pytest.fixture
def self_targeting():
    """Create a self-targeting system."""
    return SelfTargeting()


@pytest.fixture
def actor_targeting():
    """Create an actor targeting system."""
    return ActorTargeting(max_range=30.0)


@pytest.fixture
def point_targeting():
    """Create a point targeting system."""
    return PointTargeting(max_range=50.0)


@pytest.fixture
def circle_aoe_targeting():
    """Create a circular AOE targeting system."""
    return AreaTargeting(
        shape=AreaShape.CIRCLE,
        radius=5.0,
        max_range=30.0,
    )


@pytest.fixture
def cone_aoe_targeting():
    """Create a cone AOE targeting system."""
    return AreaTargeting(
        shape=AreaShape.CONE,
        radius=10.0,
        cone_angle=60.0,
        max_range=0.0,
    )


@pytest.fixture
def mock_ability_spec():
    """Create a mock ability spec."""
    return MockAbilitySpec()


@pytest.fixture
def mock_buff():
    """Create a mock buff."""
    return MockBuff()


@pytest.fixture
def effect_context(mock_source_entity, mock_target_entity):
    """Create an effect context."""
    return EffectContext(
        source=mock_source_entity,
        target=mock_target_entity,
        level=1,
        magnitude_multiplier=1.0,
        duration_multiplier=1.0,
    )


@pytest.fixture(autouse=True)
def clear_tag_registry():
    """Clear the tag registry before each test."""
    GameplayTagRegistry.clear()
    yield
    GameplayTagRegistry.clear()
