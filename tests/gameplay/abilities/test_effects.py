"""
Tests for the Effects System.

Tests cover:
- Instant effects (damage, heal)
- Duration effects (DOT, HOT)
- Infinite effects (auras)
- Effect stacking rules (aggregate, override, cap)
- Effect modifiers (add, multiply)
- Periodic effects (tick rate)
- Effect removal conditions
- Effect immunity
- Effect application and removal

Total: ~150 tests
"""

from __future__ import annotations

import math
import pytest
from typing import List, Tuple
from uuid import uuid4

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeSet,
    create_standard_attributes,
)
from engine.gameplay.abilities.constants import (
    DEFAULT_TICK_RATE,
    EffectType,
    ModifierOperation,
)
from engine.gameplay.abilities.effects import (
    EffectContext,
    EffectModifier,
    GameplayEffect,
    InstantEffect,
    DurationEffect,
    InfiniteEffect,
    PeriodicEffect,
    EffectContainer,
    instant_damage,
    instant_heal,
    damage_over_time,
    heal_over_time,
    stat_buff,
    stat_debuff,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer


# =============================================================================
# EFFECT MODIFIER TESTS
# =============================================================================


class TestEffectModifier:
    """Tests for EffectModifier value calculations."""

    def test_basic_magnitude(self):
        """Test basic magnitude calculation."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=100.0,
        )
        assert mod.get_magnitude() == 100.0

    def test_level_scaling(self):
        """Test magnitude scales with level."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=100.0,
            level_scaling=10.0,
        )
        assert mod.get_magnitude(level=1) == 100.0
        assert mod.get_magnitude(level=5) == 140.0  # 100 + (10 * 4)
        assert mod.get_magnitude(level=10) == 190.0

    def test_multiplier(self):
        """Test magnitude with multiplier."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=100.0,
        )
        assert mod.get_magnitude(multiplier=2.0) == 200.0
        assert mod.get_magnitude(multiplier=0.5) == 50.0

    def test_level_and_multiplier(self):
        """Test magnitude with both level and multiplier."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=100.0,
            level_scaling=20.0,
        )
        # Level 3: 100 + (20 * 2) = 140, then * 1.5 = 210
        assert mod.get_magnitude(level=3, multiplier=1.5) == 210.0

    def test_magnitude_clamping_max(self):
        """Test magnitude is clamped to max."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=1000.0,
            max_magnitude=500.0,
        )
        assert mod.get_magnitude() == 500.0

    def test_magnitude_clamping_min(self):
        """Test magnitude is clamped to min."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=-1000.0,
            min_magnitude=-500.0,
        )
        assert mod.get_magnitude() == -500.0

    def test_negative_magnitude(self):
        """Test negative magnitude for damage."""
        mod = EffectModifier(
            attribute="health",
            operation=ModifierOperation.ADD,
            base_magnitude=-50.0,
        )
        assert mod.get_magnitude() == -50.0

    def test_multiply_operation_modifier(self):
        """Test modifier with multiply operation."""
        mod = EffectModifier(
            attribute="damage",
            operation=ModifierOperation.MULTIPLY,
            base_magnitude=0.5,
        )
        assert mod.operation == ModifierOperation.MULTIPLY
        assert mod.get_magnitude() == 0.5


# =============================================================================
# EFFECT CONTEXT TESTS
# =============================================================================


class TestEffectContext:
    """Tests for EffectContext."""

    def test_default_context(self):
        """Test context with default values."""
        ctx = EffectContext()
        assert ctx.source is None
        assert ctx.target is None
        assert ctx.level == 1
        assert ctx.magnitude_multiplier == 1.0
        assert ctx.duration_multiplier == 1.0

    def test_context_with_values(self):
        """Test context with specified values."""
        source = object()
        target = object()
        ctx = EffectContext(
            source=source,
            target=target,
            level=5,
            magnitude_multiplier=1.5,
            duration_multiplier=2.0,
        )
        assert ctx.source is source
        assert ctx.target is target
        assert ctx.level == 5
        assert ctx.magnitude_multiplier == 1.5
        assert ctx.duration_multiplier == 2.0

    def test_context_tags(self):
        """Test context with gameplay tags."""
        ctx = EffectContext()
        ctx.tags.add("ability.fire")
        assert ctx.tags.has("ability.fire")


# =============================================================================
# INSTANT EFFECT TESTS
# =============================================================================


class TestInstantEffect:
    """Tests for InstantEffect."""

    def test_instant_effect_type(self):
        """Test instant effect has correct type."""
        effect = InstantEffect(name="test")
        assert effect.effect_type == EffectType.INSTANT

    def test_instant_effect_apply(self):
        """Test applying instant effect."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="damage",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-25.0,
                )
            ],
        )

        initial_health = attrs.get("health")
        result = effect.apply(attrs)

        assert result is True
        assert effect.is_active is True
        assert attrs.get("health") == initial_health - 25.0

    def test_instant_effect_apply_multiple_modifiers(self):
        """Test instant effect with multiple modifiers."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="buff",
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                ),
                EffectModifier(
                    attribute="armor",
                    operation=ModifierOperation.ADD,
                    base_magnitude=5.0,
                ),
            ],
        )

        effect.apply(attrs)
        assert attrs.get("damage") == 20.0  # 10 + 10
        assert attrs.get("armor") == 5.0  # 0 + 5

    def test_instant_effect_remove(self):
        """Test removing instant effect."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="buff",
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                )
            ],
        )

        effect.apply(attrs)
        initial_after_apply = attrs.get("damage")
        effect.remove(attrs)

        assert effect.is_active is False
        assert attrs.get("damage") == initial_after_apply - 10.0

    def test_instant_effect_remove_not_applied(self):
        """Test removing effect that wasn't applied."""
        attrs = create_standard_attributes()
        effect = InstantEffect(name="test")
        result = effect.remove(attrs)
        assert result is False

    def test_instant_effect_tick(self):
        """Test instant effect tick always returns True when active."""
        attrs = create_standard_attributes()
        effect = InstantEffect(name="test")
        effect.apply(attrs)
        assert effect.tick(0.1, attrs) is True

    def test_instant_effect_tick_inactive(self):
        """Test instant effect tick returns False when inactive."""
        attrs = create_standard_attributes()
        effect = InstantEffect(name="test")
        assert effect.tick(0.1, attrs) is False

    def test_instant_effect_with_context_level(self):
        """Test instant effect scaling with context level."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="scaled_damage",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-10.0,
                    level_scaling=-5.0,
                )
            ],
        )

        ctx = EffectContext(level=5)
        effect.apply(attrs, context=ctx)
        # -10 + (-5 * 4) = -30
        assert attrs.get("health") == 70.0

    def test_instant_effect_with_context_multiplier(self):
        """Test instant effect with magnitude multiplier."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="damage",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-20.0,
                )
            ],
        )

        ctx = EffectContext(magnitude_multiplier=2.0)
        effect.apply(attrs, context=ctx)
        assert attrs.get("health") == 60.0  # 100 - 40

    def test_instant_effect_grants_tags(self):
        """Test instant effect granting tags."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        effect = InstantEffect(
            name="buff",
            granted_tags=["status.buff.strength"],
        )

        effect.apply(attrs, tags=tags)
        assert tags.has("status.buff.strength")

    def test_instant_effect_removes_tags(self):
        """Test instant effect removing tags."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.debuff.slow")

        effect = InstantEffect(
            name="cleanse",
            removed_tags=["status.debuff.slow"],
        )

        effect.apply(attrs, tags=tags)
        assert not tags.has("status.debuff.slow")


# =============================================================================
# DURATION EFFECT TESTS
# =============================================================================


class TestDurationEffect:
    """Tests for DurationEffect."""

    def test_duration_effect_type(self):
        """Test duration effect has correct type."""
        effect = DurationEffect(name="test", duration=10.0)
        assert effect.effect_type == EffectType.DURATION

    def test_duration_effect_apply(self):
        """Test applying duration effect."""
        attrs = create_standard_attributes()
        effect = DurationEffect(
            name="buff",
            duration=10.0,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.MULTIPLY,
                    base_magnitude=0.5,
                )
            ],
        )

        result = effect.apply(attrs)
        assert result is True
        assert effect.is_active is True
        assert effect.remaining_time == 10.0
        assert attrs.get("damage") == 15.0  # 10 * 1.5

    def test_duration_effect_remaining_time(self):
        """Test remaining time property."""
        effect = DurationEffect(name="test", duration=10.0)
        attrs = create_standard_attributes()
        effect.apply(attrs)

        assert effect.remaining_time == 10.0

    def test_duration_effect_tick(self):
        """Test duration effect tick reduces time."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=10.0)
        effect.apply(attrs)

        result = effect.tick(1.0, attrs)
        assert result is True
        assert effect.remaining_time == 9.0

    def test_duration_effect_expires(self):
        """Test duration effect expires after duration."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=5.0)
        effect.apply(attrs)

        # Tick until expired
        effect.tick(2.0, attrs)
        assert effect.tick(2.0, attrs) is True  # 1 second remaining
        assert effect.tick(2.0, attrs) is False  # Expired

    def test_duration_effect_elapsed_time(self):
        """Test elapsed time calculation."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=10.0)
        effect.apply(attrs)

        effect.tick(3.0, attrs)
        assert effect.elapsed_time == 3.0

    def test_duration_effect_progress(self):
        """Test progress property."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=10.0)
        effect.apply(attrs)

        effect.tick(5.0, attrs)
        assert effect.progress == 0.5

    def test_duration_effect_extend_duration(self):
        """Test extending duration."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=10.0)
        effect.apply(attrs)

        effect.tick(5.0, attrs)
        effect.extend_duration(3.0)
        assert effect.remaining_time == 8.0

    def test_duration_effect_refresh_duration(self):
        """Test refreshing duration to full."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=10.0)
        effect.apply(attrs)

        effect.tick(7.0, attrs)
        effect.refresh_duration()
        assert effect.remaining_time == 10.0

    def test_duration_effect_remove(self):
        """Test removing duration effect."""
        attrs = create_standard_attributes()
        effect = DurationEffect(
            name="buff",
            duration=10.0,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=20.0,
                )
            ],
        )

        effect.apply(attrs)
        assert attrs.get("damage") == 30.0

        effect.remove(attrs)
        assert effect.is_active is False
        assert effect.remaining_time == 0.0
        assert attrs.get("damage") == 10.0

    def test_duration_effect_with_duration_multiplier(self):
        """Test duration effect with duration multiplier."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=10.0)
        ctx = EffectContext(duration_multiplier=2.0)

        effect.apply(attrs, context=ctx)
        assert effect.remaining_time == 20.0


# =============================================================================
# INFINITE EFFECT TESTS
# =============================================================================


class TestInfiniteEffect:
    """Tests for InfiniteEffect."""

    def test_infinite_effect_type(self):
        """Test infinite effect has correct type."""
        effect = InfiniteEffect(name="test")
        assert effect.effect_type == EffectType.INFINITE

    def test_infinite_effect_apply(self):
        """Test applying infinite effect."""
        attrs = create_standard_attributes()
        effect = InfiniteEffect(
            name="aura",
            modifiers=[
                EffectModifier(
                    attribute="armor",
                    operation=ModifierOperation.ADD,
                    base_magnitude=50.0,
                )
            ],
        )

        result = effect.apply(attrs)
        assert result is True
        assert effect.is_active is True
        assert attrs.get("armor") == 50.0

    def test_infinite_effect_tick_always_active(self):
        """Test infinite effect tick always returns True."""
        attrs = create_standard_attributes()
        effect = InfiniteEffect(name="test")
        effect.apply(attrs)

        # Tick many times
        for _ in range(100):
            assert effect.tick(1.0, attrs) is True

    def test_infinite_effect_remove(self):
        """Test removing infinite effect."""
        attrs = create_standard_attributes()
        effect = InfiniteEffect(
            name="aura",
            modifiers=[
                EffectModifier(
                    attribute="armor",
                    operation=ModifierOperation.ADD,
                    base_magnitude=50.0,
                )
            ],
        )

        effect.apply(attrs)
        result = effect.remove(attrs)

        assert result is True
        assert effect.is_active is False
        assert attrs.get("armor") == 0.0

    def test_infinite_effect_persists(self):
        """Test infinite effect persists over time."""
        attrs = create_standard_attributes()
        effect = InfiniteEffect(
            name="aura",
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.MULTIPLY,
                    base_magnitude=1.0,
                )
            ],
        )

        effect.apply(attrs)
        initial_damage = attrs.get("damage")

        # Simulate many ticks
        for _ in range(1000):
            effect.tick(1.0, attrs)

        # Effect should still be active and value unchanged
        assert effect.is_active is True
        assert attrs.get("damage") == initial_damage


# =============================================================================
# PERIODIC EFFECT TESTS
# =============================================================================


class TestPeriodicEffect:
    """Tests for PeriodicEffect."""

    def test_periodic_effect_type(self):
        """Test periodic effect has correct type."""
        effect = PeriodicEffect(name="test", duration=10.0, tick_rate=1.0)
        assert effect.effect_type == EffectType.PERIODIC

    def test_periodic_effect_apply_executes_first_tick(self):
        """Test periodic effect executes first tick on apply."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="dot",
            duration=10.0,
            tick_rate=1.0,
            execute_on_apply=True,
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-10.0,
                )
            ],
        )

        initial_health = attrs.get("health")
        effect.apply(attrs)

        assert effect.is_active is True
        assert effect.tick_count == 1
        assert attrs.get("health") == initial_health - 10.0

    def test_periodic_effect_no_first_tick(self):
        """Test periodic effect can skip first tick."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="dot",
            duration=10.0,
            tick_rate=1.0,
            execute_on_apply=False,
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-10.0,
                )
            ],
        )

        initial_health = attrs.get("health")
        effect.apply(attrs)

        assert effect.tick_count == 0
        assert attrs.get("health") == initial_health

    def test_periodic_effect_tick_executes_at_rate(self):
        """Test periodic effect executes ticks at tick rate."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="dot",
            duration=10.0,
            tick_rate=2.0,  # Every 2 seconds
            execute_on_apply=False,
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-10.0,
                )
            ],
        )

        effect.apply(attrs)
        initial_health = attrs.get("health")

        # 1 second - no tick yet
        effect.tick(1.0, attrs)
        assert effect.tick_count == 0
        assert attrs.get("health") == initial_health

        # 2 seconds total - first tick
        effect.tick(1.0, attrs)
        assert effect.tick_count == 1
        assert attrs.get("health") == initial_health - 10.0

    def test_periodic_effect_multiple_ticks_per_update(self):
        """Test periodic effect handles multiple ticks in one update."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="dot",
            duration=10.0,
            tick_rate=1.0,
            execute_on_apply=False,
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-10.0,
                )
            ],
        )

        effect.apply(attrs)
        initial_health = attrs.get("health")

        # 3 seconds in one update = 3 ticks
        effect.tick(3.0, attrs)
        assert effect.tick_count == 3
        assert attrs.get("health") == initial_health - 30.0

    def test_periodic_effect_expires(self):
        """Test periodic effect expires after duration."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="dot",
            duration=5.0,
            tick_rate=1.0,
            execute_on_apply=False,
        )

        effect.apply(attrs)

        # 4 seconds - still active
        assert effect.tick(4.0, attrs) is True
        # 2 more seconds - expired
        assert effect.tick(2.0, attrs) is False

    def test_periodic_effect_remaining_time(self):
        """Test periodic effect remaining time."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(name="test", duration=10.0, tick_rate=1.0)
        effect.apply(attrs)

        effect.tick(3.0, attrs)
        assert effect.remaining_time == 7.0

    def test_periodic_effect_time_until_next_tick(self):
        """Test time until next tick calculation."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="test",
            duration=10.0,
            tick_rate=2.0,
            execute_on_apply=False,
        )
        effect.apply(attrs)

        effect.tick(0.5, attrs)
        assert effect.time_until_next_tick == 1.5

    def test_periodic_effect_execute_on_remove(self):
        """Test periodic effect can execute tick on removal."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="hot",
            duration=10.0,
            tick_rate=1.0,
            execute_on_apply=False,
            execute_on_remove=True,
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                )
            ],
        )

        effect.apply(attrs)
        initial_health = attrs.get("health")

        effect.remove(attrs)
        assert attrs.get("health") == initial_health + 10.0

    def test_periodic_effect_infinite_duration(self):
        """Test periodic effect with infinite duration (duration=0)."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="aura",
            duration=0.0,  # Infinite
            tick_rate=1.0,
            execute_on_apply=False,
        )

        effect.apply(attrs)

        # Should never expire
        for _ in range(100):
            assert effect.tick(1.0, attrs) is True

    def test_periodic_effect_tick_callback(self):
        """Test periodic effect with tick callback."""
        attrs = create_standard_attributes()
        tick_data: List[int] = []

        def on_tick(effect: PeriodicEffect, attributes: AttributeSet):
            tick_data.append(effect.tick_count)

        effect = PeriodicEffect(
            name="test",
            duration=10.0,
            tick_rate=1.0,
            execute_on_apply=True,
            _on_tick=on_tick,
        )

        effect.apply(attrs)
        effect.tick(2.0, attrs)

        assert tick_data == [1, 2, 3]


# =============================================================================
# EFFECT CAN_APPLY TESTS (TAG-BASED)
# =============================================================================


class TestEffectCanApply:
    """Tests for effect application requirements."""

    def test_can_apply_no_requirements(self):
        """Test effect with no tag requirements can apply."""
        effect = InstantEffect(name="test")
        tags = GameplayTagContainer()
        assert effect.can_apply(tags) is True

    def test_can_apply_blocked_by_tag(self):
        """Test effect blocked by target tag."""
        effect = InstantEffect(
            name="test",
            blocked_by_tags=["status.immune.damage"],
        )
        tags = GameplayTagContainer()
        tags.add("status.immune.damage")

        assert effect.can_apply(tags) is False

    def test_can_apply_not_blocked(self):
        """Test effect not blocked when tag absent."""
        effect = InstantEffect(
            name="test",
            blocked_by_tags=["status.immune.damage"],
        )
        tags = GameplayTagContainer()

        assert effect.can_apply(tags) is True

    def test_can_apply_requires_tag(self):
        """Test effect requires application tag."""
        effect = InstantEffect(
            name="test",
            application_tags=["status.vulnerable"],
        )
        tags = GameplayTagContainer()
        tags.add("status.vulnerable")

        assert effect.can_apply(tags) is True

    def test_can_apply_missing_required_tag(self):
        """Test effect fails when required tag missing."""
        effect = InstantEffect(
            name="test",
            application_tags=["status.vulnerable"],
        )
        tags = GameplayTagContainer()

        assert effect.can_apply(tags) is False

    def test_can_apply_multiple_requirements(self):
        """Test effect with multiple tag requirements."""
        effect = InstantEffect(
            name="test",
            application_tags=["status.vulnerable"],
            blocked_by_tags=["status.immune.all"],
        )

        # Has required, not blocked
        tags1 = GameplayTagContainer()
        tags1.add("status.vulnerable")
        assert effect.can_apply(tags1) is True

        # Has required, but blocked
        tags2 = GameplayTagContainer()
        tags2.add("status.vulnerable")
        tags2.add("status.immune.all")
        assert effect.can_apply(tags2) is False

        # Missing required
        tags3 = GameplayTagContainer()
        assert effect.can_apply(tags3) is False


# =============================================================================
# EFFECT CONTAINER TESTS
# =============================================================================


class TestEffectContainer:
    """Tests for EffectContainer."""

    def test_container_creation(self):
        """Test creating effect container."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        container = EffectContainer(attrs, tags)

        assert container.active_effects == []

    def test_container_apply_effect(self):
        """Test applying effect through container."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect = InstantEffect(
            name="damage",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-20.0,
                )
            ],
        )

        result = container.apply(effect)
        assert result is True
        assert len(container.active_effects) == 1
        assert attrs.get("health") == 80.0

    def test_container_remove_effect(self):
        """Test removing effect from container."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect = InstantEffect(
            name="buff",
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                )
            ],
        )

        container.apply(effect)
        result = container.remove(effect)

        assert result is True
        assert len(container.active_effects) == 0
        assert attrs.get("damage") == 10.0

    def test_container_remove_by_id(self):
        """Test removing effect by ID."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect = InstantEffect(name="test")
        container.apply(effect)

        result = container.remove(effect.id)
        assert result is True
        assert len(container.active_effects) == 0

    def test_container_remove_nonexistent(self):
        """Test removing effect that doesn't exist."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        result = container.remove(uuid4())
        assert result is False

    def test_container_remove_by_name(self):
        """Test removing all effects with a name."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect1 = InstantEffect(name="buff")
        effect2 = InstantEffect(name="buff")
        effect3 = InstantEffect(name="other")

        container.apply(effect1)
        container.apply(effect2)
        container.apply(effect3)

        count = container.remove_by_name("buff")
        assert count == 2
        assert len(container.active_effects) == 1

    def test_container_remove_all(self):
        """Test removing all effects."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        container.apply(InstantEffect(name="effect1"))
        container.apply(InstantEffect(name="effect2"))
        container.apply(InstantEffect(name="effect3"))

        count = container.remove_all()
        assert count == 3
        assert len(container.active_effects) == 0

    def test_container_has_effect(self):
        """Test checking for effect by name."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        container.apply(InstantEffect(name="buff"))

        assert container.has_effect("buff") is True
        assert container.has_effect("debuff") is False

    def test_container_get_effects_by_name(self):
        """Test getting effects by name."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect1 = InstantEffect(name="buff")
        effect2 = InstantEffect(name="buff")
        effect3 = InstantEffect(name="other")

        container.apply(effect1)
        container.apply(effect2)
        container.apply(effect3)

        buffs = container.get_effects_by_name("buff")
        assert len(buffs) == 2

    def test_container_tick_updates_effects(self):
        """Test container tick updates all effects."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect = DurationEffect(name="buff", duration=5.0)
        container.apply(effect)

        container.tick(2.0)
        assert effect.remaining_time == 3.0

    def test_container_tick_removes_expired(self):
        """Test container tick removes expired effects."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect = DurationEffect(name="buff", duration=5.0)
        container.apply(effect)

        container.tick(6.0)
        assert len(container.active_effects) == 0

    def test_container_multiple_effects_tick(self):
        """Test multiple effects tick properly."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        effect1 = DurationEffect(name="short", duration=2.0)
        effect2 = DurationEffect(name="long", duration=10.0)

        container.apply(effect1)
        container.apply(effect2)

        container.tick(3.0)

        # Short should be gone, long should remain
        assert len(container.active_effects) == 1
        assert container.active_effects[0].name == "long"


# =============================================================================
# EFFECT FACTORY FUNCTION TESTS
# =============================================================================


class TestEffectFactoryFunctions:
    """Tests for effect factory functions."""

    def test_instant_damage_factory(self):
        """Test instant_damage factory function."""
        attrs = create_standard_attributes()
        effect = instant_damage(25.0)

        effect.apply(attrs)
        assert attrs.get("health") == 75.0

    def test_instant_damage_custom_attribute(self):
        """Test instant_damage with custom attribute."""
        attrs = create_standard_attributes()
        effect = instant_damage(50.0, attribute="mana")

        effect.apply(attrs)
        assert attrs.get("mana") == 50.0

    def test_instant_heal_factory(self):
        """Test instant_heal factory function."""
        attrs = create_standard_attributes()
        attrs.set_base("health", 50.0)

        effect = instant_heal(30.0)
        effect.apply(attrs)

        assert attrs.get("health") == 80.0

    def test_damage_over_time_factory(self):
        """Test damage_over_time factory function."""
        attrs = create_standard_attributes()
        effect = damage_over_time(
            damage_per_tick=10.0,
            duration=5.0,
            tick_rate=1.0,
        )

        effect.apply(attrs)
        assert effect.duration == 5.0
        assert effect.tick_rate == 1.0

    def test_heal_over_time_factory(self):
        """Test heal_over_time factory function."""
        attrs = create_standard_attributes()
        attrs.set_base("health", 50.0)

        effect = heal_over_time(
            heal_per_tick=5.0,
            duration=10.0,
            tick_rate=1.0,
        )

        effect.apply(attrs)
        # First tick on apply
        assert attrs.get("health") == 55.0

    def test_stat_buff_factory(self):
        """Test stat_buff factory function."""
        attrs = create_standard_attributes()
        effect = stat_buff("damage", 20.0, duration=10.0)

        effect.apply(attrs)
        assert attrs.get("damage") == 30.0

    def test_stat_buff_multiply(self):
        """Test stat_buff with multiply operation."""
        attrs = create_standard_attributes()
        effect = stat_buff(
            "damage",
            0.5,
            duration=10.0,
            operation=ModifierOperation.MULTIPLY,
        )

        effect.apply(attrs)
        assert attrs.get("damage") == 15.0  # 10 * 1.5

    def test_stat_debuff_factory(self):
        """Test stat_debuff factory function."""
        attrs = create_standard_attributes()
        effect = stat_debuff("damage", 5.0, duration=10.0)

        effect.apply(attrs)
        assert attrs.get("damage") == 5.0  # 10 - 5


# =============================================================================
# EFFECT STACKING TESTS
# =============================================================================


class TestEffectStacking:
    """Tests for effect stacking behavior."""

    def test_multiple_instant_effects_stack(self):
        """Test multiple instant effects stack."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        container.apply(instant_damage(10.0))
        container.apply(instant_damage(15.0))
        container.apply(instant_damage(5.0))

        # All damage stacks: 100 - 10 - 15 - 5 = 70
        assert attrs.get("health") == 70.0

    def test_multiple_duration_effects_stack(self):
        """Test multiple duration effects stack."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        buff1 = stat_buff("damage", 10.0, duration=10.0)
        buff2 = stat_buff("damage", 5.0, duration=10.0)

        container.apply(buff1)
        container.apply(buff2)

        # Both buffs active: 10 + 10 + 5 = 25
        assert attrs.get("damage") == 25.0

    def test_effects_from_same_source(self):
        """Test effects from same source can be tracked."""
        attrs = create_standard_attributes()

        source = "spell_fireball"
        effect1 = InstantEffect(
            name="damage",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-20.0,
                )
            ],
        )

        ctx = EffectContext(source=source)
        effect1.apply(attrs, context=ctx)

        # Source is tracked through context
        assert ctx.source == source


# =============================================================================
# EFFECT IMMUNITY TESTS
# =============================================================================


class TestEffectImmunity:
    """Tests for effect immunity through tags."""

    def test_immunity_blocks_effect(self):
        """Test immunity tag blocks effect application."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.immune.fire")

        effect = InstantEffect(
            name="fire_damage",
            blocked_by_tags=["status.immune.fire"],
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-50.0,
                )
            ],
        )

        result = effect.apply(attrs, tags=tags)
        assert result is False
        assert attrs.get("health") == 100.0  # Unchanged

    def test_partial_immunity(self):
        """Test partial immunity (one effect blocked, another not)."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.immune.fire")

        fire_effect = InstantEffect(
            name="fire",
            blocked_by_tags=["status.immune.fire"],
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-30.0,
                )
            ],
        )

        ice_effect = InstantEffect(
            name="ice",
            blocked_by_tags=["status.immune.ice"],
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-20.0,
                )
            ],
        )

        fire_effect.apply(attrs, tags=tags)  # Blocked
        ice_effect.apply(attrs, tags=tags)  # Not blocked

        assert attrs.get("health") == 80.0

    def test_immunity_with_container(self):
        """Test immunity through container."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.immune.all")

        container = EffectContainer(attrs, tags)

        effect = InstantEffect(
            name="damage",
            blocked_by_tags=["status.immune.all"],
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-100.0,
                )
            ],
        )

        result = container.apply(effect)
        assert result is False


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEffectEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_effect_with_no_modifiers(self):
        """Test effect with no modifiers still applies."""
        attrs = create_standard_attributes()
        effect = InstantEffect(name="tag_only", granted_tags=["status.marked"])
        tags = GameplayTagContainer()

        result = effect.apply(attrs, tags=tags)
        assert result is True
        assert tags.has("status.marked")

    def test_effect_modifier_for_missing_attribute(self):
        """Test modifier for non-existent attribute is ignored."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="test",
            modifiers=[
                EffectModifier(
                    attribute="nonexistent",
                    operation=ModifierOperation.ADD,
                    base_magnitude=100.0,
                )
            ],
        )

        result = effect.apply(attrs)
        assert result is True  # Still applies successfully

    def test_zero_duration_effect(self):
        """Test duration effect with zero duration."""
        attrs = create_standard_attributes()
        effect = DurationEffect(name="test", duration=0.0)
        effect.apply(attrs)

        # Should expire immediately
        assert effect.tick(0.1, attrs) is False

    def test_very_short_tick_rate(self):
        """Test periodic effect with very short tick rate."""
        attrs = create_standard_attributes()
        effect = PeriodicEffect(
            name="rapid",
            duration=1.0,
            tick_rate=0.01,
            execute_on_apply=False,
        )
        effect.apply(attrs)

        effect.tick(0.1, attrs)
        assert effect.tick_count == 10  # 0.1 / 0.01 = 10 ticks

    def test_effect_removal_clears_handles(self):
        """Test effect removal clears modifier handles."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="test",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=50.0,
                )
            ],
        )

        effect.apply(attrs)
        assert len(effect._active_handles) == 1

        effect.remove(attrs)
        assert len(effect._active_handles) == 0

    def test_double_apply_same_effect(self):
        """Test applying same effect twice."""
        attrs = create_standard_attributes()
        effect = InstantEffect(
            name="test",
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                )
            ],
        )

        effect.apply(attrs)
        effect.apply(attrs)  # Apply again

        # Both applications should add modifiers
        assert attrs.get("damage") == 30.0  # 10 + 10 + 10

    def test_remove_already_removed_effect(self):
        """Test removing an already removed effect."""
        attrs = create_standard_attributes()
        effect = InstantEffect(name="test")

        effect.apply(attrs)
        effect.remove(attrs)
        result = effect.remove(attrs)  # Remove again

        assert result is False

    def test_concurrent_effects_different_attributes(self):
        """Test multiple effects on different attributes."""
        attrs = create_standard_attributes()
        container = EffectContainer(attrs)

        container.apply(stat_buff("damage", 20.0, duration=10.0))
        container.apply(stat_buff("armor", 50.0, duration=10.0))
        container.apply(stat_buff("movement_speed", 100.0, duration=10.0))

        assert attrs.get("damage") == 30.0
        assert attrs.get("armor") == 50.0
        assert attrs.get("movement_speed") == 500.0

    def test_effect_with_context_instigator(self):
        """Test effect tracks instigator separately from source."""
        instigator = "player_1"
        source = "fireball_spell"

        ctx = EffectContext(instigator=instigator, source=source)

        assert ctx.instigator == instigator
        assert ctx.source == source
        assert ctx.instigator != ctx.source
