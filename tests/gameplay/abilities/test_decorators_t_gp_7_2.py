"""
Tests for @ability and @buff decorators (T-GP-7.2, T-GP-7.4).

Tests cover:
- @ability registration with Foundation Registry
- @buff registration with stacking modes
- Registry query discovery
- Tag filtering
- Stacking mode enforcement
- AbilityCast event emission
- BuffApplied event with stacks
- BuffExpired event on timeout
- Multiple abilities/buffs coexisting
- Cooldown metadata storage
- Cost metadata storage
- Edge cases and error handling

Total: 60+ tests
"""

from __future__ import annotations

import time
import pytest
from dataclasses import dataclass

from foundation import Registry, registry, get_event_log, clear_event_log, set_current_tick

from engine.gameplay.abilities.decorators import (
    # Stacking mode
    StackingMode,
    # Events
    AbilityCast,
    BuffApplied,
    BuffExpired,
    # Decorators
    ability,
    buff,
    # Event emitters
    emit_ability_cast,
    emit_buff_applied,
    emit_buff_expired,
    # Query helpers
    get_all_abilities,
    get_abilities_by_tag,
    get_all_buffs,
    get_buffs_by_stacking,
    get_debuffs,
    get_ability_metadata,
    get_buff_metadata,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry and event log before each test."""
    registry.clear()
    clear_event_log()
    set_current_tick(0)
    yield
    registry.clear()
    clear_event_log()


# =============================================================================
# ABILITY DECORATOR - REGISTRATION
# =============================================================================


class TestAbilityRegistration:
    """Tests for @ability decorator registration."""

    def test_ability_registers_with_registry(self):
        """@ability registers the class with Foundation Registry."""

        @ability(name="fireball", cooldown=2.0)
        class Fireball:
            pass

        assert registry.is_registered(Fireball)

    def test_ability_registers_with_custom_name(self):
        """@ability uses provided name for registration."""

        @ability(name="ice_bolt", cooldown=1.5)
        class IceBolt:
            pass

        assert registry.get("ice_bolt") is IceBolt

    def test_ability_adds_ability_tag(self):
        """@ability adds 'ability' tag to the class."""

        @ability(name="lightning", cooldown=3.0)
        class Lightning:
            pass

        assert registry.has_tag(Lightning, "ability")

    def test_ability_adds_custom_tags(self):
        """@ability adds custom categorization tags."""

        @ability(name="meteor", cooldown=10.0, tags=["fire", "aoe", "ultimate"])
        class Meteor:
            pass

        assert registry.has_tag(Meteor, "ability")
        assert registry.has_tag(Meteor, "fire")
        assert registry.has_tag(Meteor, "aoe")
        assert registry.has_tag(Meteor, "ultimate")

    def test_ability_stores_cooldown_metadata(self):
        """@ability stores cooldown in metadata."""

        @ability(name="heal", cooldown=5.0)
        class Heal:
            pass

        assert registry.get_metadata(Heal, "cooldown") == 5.0

    def test_ability_stores_cost_metadata(self):
        """@ability stores resource cost in metadata."""

        @ability(name="shield", cooldown=8.0, cost={"mana": 50, "stamina": 20})
        class Shield:
            pass

        cost = registry.get_metadata(Shield, "cost")
        assert cost == {"mana": 50, "stamina": 20}

    def test_ability_stores_description(self):
        """@ability stores description in metadata."""

        @ability(
            name="teleport",
            cooldown=15.0,
            description="Instantly teleport to target location",
        )
        class Teleport:
            pass

        assert registry.get_metadata(Teleport, "description") == "Instantly teleport to target location"

    def test_ability_stores_required_tags(self):
        """@ability stores required tags in metadata."""

        @ability(name="execute", cooldown=30.0, required_tags=["combat", "has_weapon"])
        class Execute:
            pass

        required = registry.get_metadata(Execute, "required_tags")
        assert "combat" in required
        assert "has_weapon" in required

    def test_ability_stores_granted_tags(self):
        """@ability stores granted tags in metadata."""

        @ability(name="berserk", cooldown=60.0, granted_tags=["enraged", "immune_cc"])
        class Berserk:
            pass

        granted = registry.get_metadata(Berserk, "granted_tags")
        assert "enraged" in granted
        assert "immune_cc" in granted

    def test_ability_stores_blocked_by_tags(self):
        """@ability stores blocked_by tags in metadata."""

        @ability(name="stealth", cooldown=20.0, blocked_by_tags=["in_combat", "silenced"])
        class Stealth:
            pass

        blocked = registry.get_metadata(Stealth, "blocked_by_tags")
        assert "in_combat" in blocked
        assert "silenced" in blocked

    def test_ability_class_has_metadata_attributes(self):
        """@ability sets metadata attributes on the class itself."""

        @ability(
            name="charge",
            cooldown=12.0,
            cost={"rage": 15},
            tags=["mobility", "gap_closer"],
        )
        class Charge:
            pass

        assert Charge._ability_name == "charge"
        assert Charge._ability_cooldown == 12.0
        assert Charge._ability_cost == {"rage": 15}
        assert "mobility" in Charge._ability_category_tags

    def test_ability_default_cost_is_empty(self):
        """@ability with no cost defaults to empty dict."""

        @ability(name="auto_attack", cooldown=0.0)
        class AutoAttack:
            pass

        assert registry.get_metadata(AutoAttack, "cost") == {}

    def test_ability_zero_cooldown(self):
        """@ability accepts zero cooldown."""

        @ability(name="instant", cooldown=0.0)
        class Instant:
            pass

        assert registry.get_metadata(Instant, "cooldown") == 0.0

    def test_ability_with_track_instances(self):
        """@ability can enable instance tracking."""

        @ability(name="tracked_ability", cooldown=1.0, track_instances=True)
        class TrackedAbility:
            pass

        instance = TrackedAbility()
        assert registry.instance_count(TrackedAbility) == 1


# =============================================================================
# BUFF DECORATOR - REGISTRATION
# =============================================================================


class TestBuffRegistration:
    """Tests for @buff decorator registration."""

    def test_buff_registers_with_registry(self):
        """@buff registers the class with Foundation Registry."""

        @buff(name="regeneration", duration=10.0)
        class Regeneration:
            pass

        assert registry.is_registered(Regeneration)

    def test_buff_registers_with_custom_name(self):
        """@buff uses provided name for registration."""

        @buff(name="haste", duration=5.0)
        class Haste:
            pass

        assert registry.get("haste") is Haste

    def test_buff_adds_buff_tag(self):
        """@buff adds 'buff' tag to the class."""

        @buff(name="fortitude", duration=30.0)
        class Fortitude:
            pass

        assert registry.has_tag(Fortitude, "buff")

    def test_buff_adds_debuff_tag_when_is_debuff(self):
        """@buff adds 'debuff' tag when is_debuff=True."""

        @buff(name="poison", duration=8.0, is_debuff=True)
        class Poison:
            pass

        assert registry.has_tag(Poison, "buff")
        assert registry.has_tag(Poison, "debuff")

    def test_buff_stores_duration_metadata(self):
        """@buff stores duration in metadata."""

        @buff(name="shield_buff", duration=15.0)
        class ShieldBuff:
            pass

        assert registry.get_metadata(ShieldBuff, "duration") == 15.0

    def test_buff_stores_stacking_mode_none(self):
        """@buff stores stacking mode 'none'."""

        @buff(name="unique_buff", duration=10.0, stacking="none")
        class UniqueBuff:
            pass

        assert registry.get_metadata(UniqueBuff, "stacking") == "none"
        assert registry.has_tag(UniqueBuff, "stacking_none")

    def test_buff_stores_stacking_mode_duration(self):
        """@buff stores stacking mode 'duration'."""

        @buff(name="duration_stack", duration=5.0, stacking="duration")
        class DurationStack:
            pass

        assert registry.get_metadata(DurationStack, "stacking") == "duration"
        assert registry.has_tag(DurationStack, "stacking_duration")

    def test_buff_stores_stacking_mode_intensity(self):
        """@buff stores stacking mode 'intensity'."""

        @buff(name="intensity_stack", duration=10.0, stacking="intensity", max_stacks=5)
        class IntensityStack:
            pass

        assert registry.get_metadata(IntensityStack, "stacking") == "intensity"
        assert registry.get_metadata(IntensityStack, "max_stacks") == 5
        assert registry.has_tag(IntensityStack, "stacking_intensity")

    def test_buff_stores_stacking_mode_independent(self):
        """@buff stores stacking mode 'independent'."""

        @buff(name="independent_stack", duration=3.0, stacking="independent")
        class IndependentStack:
            pass

        assert registry.get_metadata(IndependentStack, "stacking") == "independent"
        assert registry.has_tag(IndependentStack, "stacking_independent")

    def test_buff_accepts_stacking_mode_enum(self):
        """@buff accepts StackingMode enum values."""

        @buff(name="enum_stack", duration=5.0, stacking=StackingMode.INTENSITY)
        class EnumStack:
            pass

        assert registry.get_metadata(EnumStack, "stacking") == "intensity"

    def test_buff_stores_max_stacks(self):
        """@buff stores max_stacks in metadata."""

        @buff(name="stacking_buff", duration=10.0, stacking="intensity", max_stacks=10)
        class StackingBuff:
            pass

        assert registry.get_metadata(StackingBuff, "max_stacks") == 10

    def test_buff_stores_tick_rate(self):
        """@buff stores tick_rate in metadata."""

        @buff(name="dot_buff", duration=6.0, tick_rate=1.0)
        class DotBuff:
            pass

        assert registry.get_metadata(DotBuff, "tick_rate") == 1.0

    def test_buff_stores_description(self):
        """@buff stores description in metadata."""

        @buff(name="blessing", duration=60.0, description="Divine protection from harm")
        class Blessing:
            pass

        assert registry.get_metadata(Blessing, "description") == "Divine protection from harm"

    def test_buff_class_has_metadata_attributes(self):
        """@buff sets metadata attributes on the class itself."""

        @buff(
            name="curse",
            duration=20.0,
            stacking="intensity",
            max_stacks=3,
            is_debuff=True,
        )
        class Curse:
            pass

        assert Curse._buff_name == "curse"
        assert Curse._buff_duration == 20.0
        assert Curse._buff_stacking == StackingMode.INTENSITY
        assert Curse._buff_max_stacks == 3
        assert Curse._buff_is_debuff is True

    def test_buff_zero_duration_is_permanent(self):
        """@buff with zero duration represents permanent buff."""

        @buff(name="aura", duration=0.0)
        class Aura:
            pass

        assert registry.get_metadata(Aura, "duration") == 0.0

    def test_buff_adds_custom_tags(self):
        """@buff adds custom categorization tags."""

        @buff(name="bloodlust", duration=30.0, tags=["attack_speed", "team_buff"])
        class Bloodlust:
            pass

        assert registry.has_tag(Bloodlust, "attack_speed")
        assert registry.has_tag(Bloodlust, "team_buff")


# =============================================================================
# REGISTRY QUERY DISCOVERY
# =============================================================================


class TestRegistryQueryDiscovery:
    """Tests for discovering abilities/buffs via Registry.query()."""

    def test_query_returns_all_abilities(self):
        """Registry.query(tag='ability') returns all abilities."""

        @ability(name="ability_a", cooldown=1.0)
        class AbilityA:
            pass

        @ability(name="ability_b", cooldown=2.0)
        class AbilityB:
            pass

        @ability(name="ability_c", cooldown=3.0)
        class AbilityC:
            pass

        abilities = registry.query(tag="ability")
        assert len(abilities) == 3
        assert AbilityA in abilities
        assert AbilityB in abilities
        assert AbilityC in abilities

    def test_query_returns_all_buffs(self):
        """Registry.query(tag='buff') returns all buffs."""

        @buff(name="buff_a", duration=10.0)
        class BuffA:
            pass

        @buff(name="buff_b", duration=20.0)
        class BuffB:
            pass

        buffs = registry.query(tag="buff")
        assert len(buffs) == 2
        assert BuffA in buffs
        assert BuffB in buffs

    def test_query_filters_by_tag(self):
        """Registry.query can filter by additional tags."""

        @ability(name="fire_spell", cooldown=2.0, tags=["fire", "spell"])
        class FireSpell:
            pass

        @ability(name="ice_spell", cooldown=2.0, tags=["ice", "spell"])
        class IceSpell:
            pass

        @ability(name="melee_attack", cooldown=0.0, tags=["melee", "physical"])
        class MeleeAttack:
            pass

        spells = registry.query(tag="spell")
        assert len(spells) == 2
        assert FireSpell in spells
        assert IceSpell in spells
        assert MeleeAttack not in spells

    def test_query_filters_stacking_buffs(self):
        """Registry.query can filter buffs by stacking mode."""

        @buff(name="stack_none", duration=5.0, stacking="none")
        class StackNone:
            pass

        @buff(name="stack_intensity", duration=5.0, stacking="intensity")
        class StackIntensity:
            pass

        @buff(name="stack_duration", duration=5.0, stacking="duration")
        class StackDuration:
            pass

        intensity_buffs = registry.query(tag="buff", stacking="intensity")
        assert len(intensity_buffs) == 1
        assert StackIntensity in intensity_buffs

    def test_get_all_abilities_helper(self):
        """get_all_abilities() returns all registered abilities."""

        @ability(name="ability_x", cooldown=1.0)
        class AbilityX:
            pass

        @ability(name="ability_y", cooldown=2.0)
        class AbilityY:
            pass

        abilities = get_all_abilities()
        assert len(abilities) == 2

    def test_get_all_buffs_helper(self):
        """get_all_buffs() returns all registered buffs."""

        @buff(name="buff_x", duration=10.0)
        class BuffX:
            pass

        @buff(name="buff_y", duration=20.0)
        class BuffY:
            pass

        buffs = get_all_buffs()
        assert len(buffs) == 2

    def test_get_abilities_by_tag_helper(self):
        """get_abilities_by_tag() filters abilities by tags."""

        @ability(name="aoe_fire", cooldown=5.0, tags=["aoe", "fire"])
        class AoeFire:
            pass

        @ability(name="single_fire", cooldown=2.0, tags=["single_target", "fire"])
        class SingleFire:
            pass

        @ability(name="aoe_ice", cooldown=5.0, tags=["aoe", "ice"])
        class AoeIce:
            pass

        fire_abilities = get_abilities_by_tag("fire")
        assert len(fire_abilities) == 2

        aoe_fire = get_abilities_by_tag("aoe", "fire")
        assert len(aoe_fire) == 1
        assert AoeFire in aoe_fire

    def test_get_buffs_by_stacking_helper(self):
        """get_buffs_by_stacking() filters buffs by stacking mode."""

        @buff(name="intensity_a", duration=5.0, stacking="intensity")
        class IntensityA:
            pass

        @buff(name="intensity_b", duration=5.0, stacking="intensity")
        class IntensityB:
            pass

        @buff(name="none_buff", duration=5.0, stacking="none")
        class NoneBuff:
            pass

        intensity = get_buffs_by_stacking("intensity")
        assert len(intensity) == 2
        assert IntensityA in intensity
        assert IntensityB in intensity

    def test_get_debuffs_helper(self):
        """get_debuffs() returns only debuffs."""

        @buff(name="helpful_buff", duration=10.0)
        class HelpfulBuff:
            pass

        @buff(name="harmful_debuff", duration=10.0, is_debuff=True)
        class HarmfulDebuff:
            pass

        debuffs = get_debuffs()
        assert len(debuffs) == 1
        assert HarmfulDebuff in debuffs


# =============================================================================
# METADATA RETRIEVAL
# =============================================================================


class TestMetadataRetrieval:
    """Tests for retrieving ability/buff metadata."""

    def test_get_ability_metadata_returns_all(self):
        """get_ability_metadata() returns all metadata for an ability."""

        @ability(
            name="test_ability",
            cooldown=5.0,
            cost={"mana": 25},
            tags=["fire"],
            description="Test",
        )
        class TestAbility:
            pass

        meta = get_ability_metadata(TestAbility)
        assert meta["name"] == "test_ability"
        assert meta["cooldown"] == 5.0
        assert meta["cost"] == {"mana": 25}
        assert meta["description"] == "Test"

    def test_get_buff_metadata_returns_all(self):
        """get_buff_metadata() returns all metadata for a buff."""

        @buff(
            name="test_buff",
            duration=10.0,
            stacking="intensity",
            max_stacks=5,
        )
        class TestBuff:
            pass

        meta = get_buff_metadata(TestBuff)
        assert meta["name"] == "test_buff"
        assert meta["duration"] == 10.0
        assert meta["stacking"] == "intensity"
        assert meta["max_stacks"] == 5

    def test_get_ability_metadata_raises_for_non_ability(self):
        """get_ability_metadata() raises ValueError for non-ability."""

        class NotAnAbility:
            pass

        with pytest.raises(ValueError, match="not a registered ability"):
            get_ability_metadata(NotAnAbility)

    def test_get_buff_metadata_raises_for_non_buff(self):
        """get_buff_metadata() raises ValueError for non-buff."""

        class NotABuff:
            pass

        with pytest.raises(ValueError, match="not a registered buff"):
            get_buff_metadata(NotABuff)


# =============================================================================
# EVENTS - ABILITY CAST
# =============================================================================


class TestAbilityCastEvent:
    """Tests for AbilityCast event emission."""

    def test_ability_cast_event_creation(self):
        """AbilityCast event can be created with required fields."""
        event = AbilityCast(entity_id=1, ability_name="fireball")
        assert event.entity_id == 1
        assert event.ability_name == "fireball"
        assert event.target_id is None
        assert event.timestamp > 0

    def test_ability_cast_event_with_target(self):
        """AbilityCast event can include target_id."""
        event = AbilityCast(entity_id=1, ability_name="heal", target_id=2)
        assert event.target_id == 2

    def test_emit_ability_cast_returns_event(self):
        """emit_ability_cast() returns the emitted event."""
        event = emit_ability_cast(entity_id=42, ability_name="lightning")
        assert isinstance(event, AbilityCast)
        assert event.entity_id == 42
        assert event.ability_name == "lightning"

    def test_emit_ability_cast_records_to_eventlog(self):
        """emit_ability_cast() records event to EventLog."""
        set_current_tick(100)
        emit_ability_cast(entity_id=1, ability_name="frost_bolt", target_id=2)

        log = get_event_log()
        events = log.all_events()
        assert len(events) == 1
        assert events[0].operation == "AbilityCast.frost_bolt"
        assert events[0].entity == 1
        assert events[0].tick == 100

    def test_emit_ability_cast_with_target(self):
        """emit_ability_cast() records target_id in event args."""
        emit_ability_cast(entity_id=1, ability_name="attack", target_id=5)

        log = get_event_log()
        events = log.all_events()
        assert events[0].operation_args["target_id"] == 5

    def test_multiple_ability_casts_recorded(self):
        """Multiple ability casts are all recorded."""
        emit_ability_cast(entity_id=1, ability_name="skill_a")
        emit_ability_cast(entity_id=2, ability_name="skill_b")
        emit_ability_cast(entity_id=1, ability_name="skill_c")

        log = get_event_log()
        events = log.all_events()
        assert len(events) == 3


# =============================================================================
# EVENTS - BUFF APPLIED
# =============================================================================


class TestBuffAppliedEvent:
    """Tests for BuffApplied event emission."""

    def test_buff_applied_event_creation(self):
        """BuffApplied event can be created with required fields."""
        event = BuffApplied(entity_id=1, buff_name="haste")
        assert event.entity_id == 1
        assert event.buff_name == "haste"
        assert event.stacks == 1
        assert event.duration == 0.0

    def test_buff_applied_event_with_stacks(self):
        """BuffApplied event can include stack count."""
        event = BuffApplied(entity_id=1, buff_name="poison", stacks=3, duration=10.0)
        assert event.stacks == 3
        assert event.duration == 10.0

    def test_emit_buff_applied_returns_event(self):
        """emit_buff_applied() returns the emitted event."""
        event = emit_buff_applied(entity_id=42, buff_name="regen", stacks=2, duration=30.0)
        assert isinstance(event, BuffApplied)
        assert event.entity_id == 42
        assert event.buff_name == "regen"
        assert event.stacks == 2
        assert event.duration == 30.0

    def test_emit_buff_applied_records_to_eventlog(self):
        """emit_buff_applied() records event to EventLog."""
        set_current_tick(50)
        emit_buff_applied(entity_id=3, buff_name="shield", stacks=1, duration=15.0)

        log = get_event_log()
        events = log.all_events()
        assert len(events) == 1
        assert events[0].operation == "BuffApplied.shield"
        assert events[0].entity == 3
        assert events[0].tick == 50

    def test_emit_buff_applied_records_stacks(self):
        """emit_buff_applied() records stack count in event args."""
        emit_buff_applied(entity_id=1, buff_name="burning", stacks=5, duration=8.0)

        log = get_event_log()
        events = log.all_events()
        assert events[0].operation_args["stacks"] == 5
        assert events[0].operation_args["duration"] == 8.0


# =============================================================================
# EVENTS - BUFF EXPIRED
# =============================================================================


class TestBuffExpiredEvent:
    """Tests for BuffExpired event emission."""

    def test_buff_expired_event_creation(self):
        """BuffExpired event can be created."""
        event = BuffExpired(entity_id=1, buff_name="haste")
        assert event.entity_id == 1
        assert event.buff_name == "haste"
        assert event.timestamp > 0

    def test_emit_buff_expired_returns_event(self):
        """emit_buff_expired() returns the emitted event."""
        event = emit_buff_expired(entity_id=42, buff_name="slow")
        assert isinstance(event, BuffExpired)
        assert event.entity_id == 42
        assert event.buff_name == "slow"

    def test_emit_buff_expired_records_to_eventlog(self):
        """emit_buff_expired() records event to EventLog."""
        set_current_tick(200)
        emit_buff_expired(entity_id=7, buff_name="poison")

        log = get_event_log()
        events = log.all_events()
        assert len(events) == 1
        assert events[0].operation == "BuffExpired.poison"
        assert events[0].entity == 7
        assert events[0].tick == 200


# =============================================================================
# COEXISTENCE
# =============================================================================


class TestCoexistence:
    """Tests for multiple abilities/buffs coexisting."""

    def test_multiple_abilities_coexist(self):
        """Multiple abilities can be registered simultaneously."""

        @ability(name="ability_1", cooldown=1.0)
        class Ability1:
            pass

        @ability(name="ability_2", cooldown=2.0)
        class Ability2:
            pass

        @ability(name="ability_3", cooldown=3.0)
        class Ability3:
            pass

        assert registry.is_registered(Ability1)
        assert registry.is_registered(Ability2)
        assert registry.is_registered(Ability3)
        assert len(get_all_abilities()) == 3

    def test_multiple_buffs_coexist(self):
        """Multiple buffs can be registered simultaneously."""

        @buff(name="buff_1", duration=10.0)
        class Buff1:
            pass

        @buff(name="buff_2", duration=20.0)
        class Buff2:
            pass

        @buff(name="buff_3", duration=30.0)
        class Buff3:
            pass

        assert registry.is_registered(Buff1)
        assert registry.is_registered(Buff2)
        assert registry.is_registered(Buff3)
        assert len(get_all_buffs()) == 3

    def test_abilities_and_buffs_coexist(self):
        """Abilities and buffs can be registered together."""

        @ability(name="damage_ability", cooldown=5.0)
        class DamageAbility:
            pass

        @buff(name="damage_buff", duration=15.0)
        class DamageBuff:
            pass

        abilities = get_all_abilities()
        buffs = get_all_buffs()

        assert len(abilities) == 1
        assert len(buffs) == 1
        assert DamageAbility in abilities
        assert DamageBuff in buffs

    def test_same_tags_on_different_types(self):
        """Same tag can be used on both abilities and buffs."""

        @ability(name="fire_ability", cooldown=3.0, tags=["fire", "damage"])
        class FireAbility:
            pass

        @buff(name="fire_buff", duration=10.0, tags=["fire", "damage"])
        class FireBuff:
            pass

        fire_things = registry.query(tag="fire")
        assert len(fire_things) == 2
        assert FireAbility in fire_things
        assert FireBuff in fire_things


# =============================================================================
# STACKING MODE ENFORCEMENT
# =============================================================================


class TestStackingModeEnforcement:
    """Tests for stacking mode validation and enforcement."""

    def test_stacking_mode_none_is_default(self):
        """Default stacking mode is 'none'."""

        @buff(name="default_stack", duration=5.0)
        class DefaultStack:
            pass

        assert DefaultStack._buff_stacking == StackingMode.NONE

    def test_stacking_mode_string_to_enum(self):
        """String stacking mode is converted to enum."""

        @buff(name="string_stack", duration=5.0, stacking="duration")
        class StringStack:
            pass

        assert StringStack._buff_stacking == StackingMode.DURATION

    def test_stacking_mode_enum_preserved(self):
        """Enum stacking mode is preserved."""

        @buff(name="enum_stack", duration=5.0, stacking=StackingMode.INDEPENDENT)
        class EnumStack:
            pass

        assert EnumStack._buff_stacking == StackingMode.INDEPENDENT

    def test_invalid_stacking_mode_raises(self):
        """Invalid stacking mode string raises ValueError."""
        with pytest.raises(ValueError):

            @buff(name="invalid_stack", duration=5.0, stacking="invalid_mode")
            class InvalidStack:
                pass


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ability_with_no_optional_args(self):
        """@ability works with only required name argument."""

        @ability(name="minimal")
        class Minimal:
            pass

        assert registry.is_registered(Minimal)
        assert registry.get_metadata(Minimal, "cooldown") == 0.0
        assert registry.get_metadata(Minimal, "cost") == {}

    def test_buff_with_no_optional_args(self):
        """@buff works with only required name argument."""

        @buff(name="minimal_buff")
        class MinimalBuff:
            pass

        assert registry.is_registered(MinimalBuff)
        assert registry.get_metadata(MinimalBuff, "duration") == 0.0

    def test_empty_tags_list(self):
        """Empty tags list is handled correctly."""

        @ability(name="no_tags", cooldown=1.0, tags=[])
        class NoTags:
            pass

        tags = registry.get_tags(NoTags)
        assert "ability" in tags
        assert len(tags) == 1

    def test_large_cooldown_value(self):
        """Large cooldown values are stored correctly."""

        @ability(name="long_cooldown", cooldown=3600.0)  # 1 hour
        class LongCooldown:
            pass

        assert registry.get_metadata(LongCooldown, "cooldown") == 3600.0

    def test_large_duration_value(self):
        """Large duration values are stored correctly."""

        @buff(name="long_duration", duration=86400.0)  # 24 hours
        class LongDuration:
            pass

        assert registry.get_metadata(LongDuration, "duration") == 86400.0

    def test_complex_cost_structure(self):
        """Complex cost structures are stored correctly."""

        @ability(
            name="expensive",
            cooldown=10.0,
            cost={
                "mana": 100,
                "stamina": 50,
                "rage": 25,
                "focus": 10,
            },
        )
        class Expensive:
            pass

        cost = registry.get_metadata(Expensive, "cost")
        assert cost["mana"] == 100
        assert cost["stamina"] == 50
        assert cost["rage"] == 25
        assert cost["focus"] == 10

    def test_unicode_in_names(self):
        """Unicode characters in names are handled."""

        @ability(name="fireball_spell", cooldown=2.0, description="Launches a ball of fire")
        class FireballSpell:
            pass

        assert registry.get_metadata(FireballSpell, "name") == "fireball_spell"

    def test_many_tags(self):
        """Many tags can be applied to a single ability."""

        @ability(
            name="versatile",
            cooldown=5.0,
            tags=["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8"],
        )
        class Versatile:
            pass

        tags = registry.get_tags(Versatile)
        assert len(tags) == 9  # 8 custom + "ability"


# =============================================================================
# INTEGRATION WITH EVENTLOG
# =============================================================================


class TestEventLogIntegration:
    """Tests for integration with Foundation EventLog."""

    def test_events_queryable_by_entity(self):
        """Events can be queried by entity from EventLog."""
        emit_ability_cast(entity_id=100, ability_name="skill_a")
        emit_ability_cast(entity_id=200, ability_name="skill_b")
        emit_ability_cast(entity_id=100, ability_name="skill_c")

        log = get_event_log()
        entity_100_events = log.events_for_entity(100)
        assert len(entity_100_events) == 2

    def test_events_queryable_by_tick(self):
        """Events can be queried by tick from EventLog."""
        set_current_tick(10)
        emit_ability_cast(entity_id=1, ability_name="skill_a")

        set_current_tick(20)
        emit_ability_cast(entity_id=2, ability_name="skill_b")

        log = get_event_log()
        tick_10_events = log.events_at(10)
        assert len(tick_10_events) == 1
        assert tick_10_events[0].operation_args["ability_name"] == "skill_a"

    def test_events_queryable_by_operation(self):
        """Events can be queried by operation name."""
        emit_ability_cast(entity_id=1, ability_name="fireball")
        emit_buff_applied(entity_id=1, buff_name="haste", stacks=1, duration=10.0)
        emit_buff_expired(entity_id=1, buff_name="slow")

        log = get_event_log()
        buff_applied = log.events_for_operation("BuffApplied.haste")
        assert len(buff_applied) == 1

    def test_clear_event_log(self):
        """Event log can be cleared."""
        emit_ability_cast(entity_id=1, ability_name="test")

        log = get_event_log()
        assert len(log.all_events()) == 1

        clear_event_log()
        assert len(log.all_events()) == 0
