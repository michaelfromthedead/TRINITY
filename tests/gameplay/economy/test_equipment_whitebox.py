"""
WHITEBOX Tests for Equipment System (T-ECON-1.4)

Tests:
- Equipment slots and restrictions
- Stat modifiers and bonuses
- Resistance modifiers
- Special effects
- Equipment sets and set bonuses
- Equip/unequip operations
- Durability system
- Serialization
"""
import pytest
from uuid import UUID, uuid4
from typing import Dict, Set, Optional

from engine.gameplay.economy.equipment import (
    StatModifier,
    ResistanceModifier,
    SpecialEffect,
    EquipmentStats,
    EquipmentDefinition,
    EquipmentInstance,
    SetBonus,
    EquipmentSet,
    EquipmentContainer,
    EquipmentRegistry,
)
from engine.gameplay.economy.inventory import ItemDefinition, ECONOMY_SCHEMA_VERSION
from engine.gameplay.economy.constants import (
    AttributeType,
    ResistanceType,
    EquipmentSlot,
    ItemType,
    Rarity,
    EXCLUSIVE_SLOTS,
    MAX_RESISTANCE_PERCENT,
    UPGRADE_BONUS_PER_LEVEL,
    DEFAULT_MAX_DURABILITY,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset equipment registry before each test."""
    EquipmentRegistry.reset()
    yield


@pytest.fixture
def basic_sword_def():
    """Basic iron sword definition."""
    return EquipmentDefinition(
        id="iron_sword",
        name="Iron Sword",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.COMMON,
        weight=5.0,
        base_value=100,
        slot=EquipmentSlot.MAIN_HAND,
        stats=EquipmentStats(
            damage=10.0,
            attack_speed=1.0,
        ),
    )


@pytest.fixture
def basic_shield_def():
    """Basic shield definition."""
    return EquipmentDefinition(
        id="iron_shield",
        name="Iron Shield",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.COMMON,
        weight=8.0,
        base_value=80,
        slot=EquipmentSlot.OFF_HAND,
        stats=EquipmentStats(
            armor=15.0,
            block_chance=0.15,
        ),
    )


@pytest.fixture
def basic_helmet_def():
    """Basic helmet definition."""
    return EquipmentDefinition(
        id="iron_helmet",
        name="Iron Helmet",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.COMMON,
        weight=4.0,
        base_value=75,
        slot=EquipmentSlot.HEAD,
        stats=EquipmentStats(
            armor=10.0,
        ),
    )


@pytest.fixture
def two_hand_weapon_def():
    """Two-handed weapon definition."""
    return EquipmentDefinition(
        id="greatsword",
        name="Iron Greatsword",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.UNCOMMON,
        weight=12.0,
        base_value=200,
        slot=EquipmentSlot.TWO_HAND,
        stats=EquipmentStats(
            damage=25.0,
            attack_speed=0.7,
        ),
    )


@pytest.fixture
def ring_def():
    """Ring definition."""
    return EquipmentDefinition(
        id="gold_ring",
        name="Gold Ring",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.RARE,
        weight=0.1,
        base_value=500,
        slot=EquipmentSlot.RING_1,
        stats=EquipmentStats(
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.LUCK, flat_bonus=5),
            ),
        ),
    )


@pytest.fixture
def equipment_container():
    """Empty equipment container."""
    return EquipmentContainer(owner_id="player1")


@pytest.fixture
def sword_instance(basic_sword_def):
    """Instance of basic sword."""
    return EquipmentInstance(definition=basic_sword_def, quantity=1)


@pytest.fixture
def shield_instance(basic_shield_def):
    """Instance of basic shield."""
    return EquipmentInstance(definition=basic_shield_def, quantity=1)


@pytest.fixture
def helmet_instance(basic_helmet_def):
    """Instance of basic helmet."""
    return EquipmentInstance(definition=basic_helmet_def, quantity=1)


# =============================================================================
# STAT MODIFIER TESTS
# =============================================================================


class TestStatModifier:
    """Whitebox tests for StatModifier."""

    def test_basic_creation(self):
        """Test basic stat modifier creation."""
        mod = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
        )
        assert mod.stat_type == AttributeType.STRENGTH
        assert mod.flat_bonus == 10.0
        assert mod.percent_bonus == 0.0
        assert mod.multiplier == 1.0

    def test_apply_flat_bonus(self):
        """Flat bonus should add to base."""
        mod = StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=10.0)
        result = mod.apply(50.0)
        assert result == 60.0

    def test_apply_percent_bonus(self):
        """Percent bonus should multiply base + flat."""
        mod = StatModifier(stat_type=AttributeType.STRENGTH, percent_bonus=0.2)
        result = mod.apply(50.0)
        assert result == pytest.approx(60.0)  # 50 * 1.2

    def test_apply_multiplier(self):
        """Multiplier should apply last."""
        mod = StatModifier(stat_type=AttributeType.STRENGTH, multiplier=1.5)
        result = mod.apply(50.0)
        assert result == pytest.approx(75.0)  # 50 * 1.5

    def test_apply_combined(self):
        """All modifiers should apply in order."""
        mod = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
            percent_bonus=0.2,
            multiplier=1.5,
        )
        # (50 + 10) * 1.2 * 1.5 = 108
        result = mod.apply(50.0)
        assert result == pytest.approx(108.0)

    def test_combine_same_stat(self):
        """Combining modifiers for same stat should work."""
        mod1 = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=10.0,
            percent_bonus=0.1,
            multiplier=1.2,
        )
        mod2 = StatModifier(
            stat_type=AttributeType.STRENGTH,
            flat_bonus=5.0,
            percent_bonus=0.05,
            multiplier=1.1,
        )
        combined = mod1.combine(mod2)
        assert combined.flat_bonus == 15.0
        assert combined.percent_bonus == pytest.approx(0.15)
        assert combined.multiplier == pytest.approx(1.32)  # 1.2 * 1.1

    def test_combine_different_stat_raises(self):
        """Combining modifiers for different stats should raise."""
        mod1 = StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=10.0)
        mod2 = StatModifier(stat_type=AttributeType.DEXTERITY, flat_bonus=10.0)
        with pytest.raises(ValueError, match="different stats"):
            mod1.combine(mod2)

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        mod = StatModifier(
            stat_type=AttributeType.INTELLIGENCE,
            flat_bonus=15.0,
            percent_bonus=0.25,
            multiplier=1.3,
        )
        data = mod.to_dict()
        restored = StatModifier.from_dict(data)
        assert restored.stat_type == mod.stat_type
        assert restored.flat_bonus == mod.flat_bonus
        assert restored.percent_bonus == mod.percent_bonus
        assert restored.multiplier == mod.multiplier


# =============================================================================
# RESISTANCE MODIFIER TESTS
# =============================================================================


class TestResistanceModifier:
    """Whitebox tests for ResistanceModifier."""

    def test_basic_creation(self):
        """Test basic resistance modifier creation."""
        mod = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=10.0,
        )
        assert mod.resistance_type == ResistanceType.FIRE
        assert mod.flat_bonus == 10.0
        assert mod.percent_bonus == 0.0

    def test_apply_flat_bonus(self):
        """Flat bonus should add to base."""
        mod = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=0.2,
        )
        result = mod.apply(0.3)
        assert result == 0.5

    def test_apply_clamped_to_max(self):
        """Result should be clamped to max resistance."""
        mod = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=0.5,
        )
        result = mod.apply(0.5)
        assert result == MAX_RESISTANCE_PERCENT

    def test_combine_same_type(self):
        """Combining same resistance types should work."""
        mod1 = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=10.0,
            percent_bonus=5.0,
        )
        mod2 = ResistanceModifier(
            resistance_type=ResistanceType.FIRE,
            flat_bonus=5.0,
            percent_bonus=3.0,
        )
        combined = mod1.combine(mod2)
        assert combined.flat_bonus == 15.0
        assert combined.percent_bonus == 8.0

    def test_combine_different_type_raises(self):
        """Combining different resistance types should raise."""
        mod1 = ResistanceModifier(resistance_type=ResistanceType.FIRE, flat_bonus=10.0)
        mod2 = ResistanceModifier(resistance_type=ResistanceType.ICE, flat_bonus=10.0)
        with pytest.raises(ValueError, match="different resistances"):
            mod1.combine(mod2)

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        mod = ResistanceModifier(
            resistance_type=ResistanceType.LIGHTNING,
            flat_bonus=15.0,
            percent_bonus=10.0,
        )
        data = mod.to_dict()
        restored = ResistanceModifier.from_dict(data)
        assert restored.resistance_type == mod.resistance_type
        assert restored.flat_bonus == mod.flat_bonus
        assert restored.percent_bonus == mod.percent_bonus


# =============================================================================
# SPECIAL EFFECT TESTS
# =============================================================================


class TestSpecialEffect:
    """Whitebox tests for SpecialEffect."""

    def test_basic_creation(self):
        """Test basic special effect creation."""
        effect = SpecialEffect(
            effect_id="fire_damage",
            name="Fire Damage",
            description="Deals additional fire damage",
            parameters={"damage": 5, "duration": 3.0},
        )
        assert effect.effect_id == "fire_damage"
        assert effect.name == "Fire Damage"
        assert effect.parameters["damage"] == 5

    def test_hash_based_on_id_and_name(self):
        """Hash should be based on id and name."""
        effect1 = SpecialEffect(effect_id="fire", name="Fire")
        effect2 = SpecialEffect(effect_id="fire", name="Fire")
        assert hash(effect1) == hash(effect2)

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        effect = SpecialEffect(
            effect_id="lifesteal",
            name="Lifesteal",
            description="Heals on hit",
            parameters={"percent": 0.1},
        )
        data = effect.to_dict()
        restored = SpecialEffect.from_dict(data)
        assert restored.effect_id == effect.effect_id
        assert restored.name == effect.name
        assert restored.description == effect.description
        assert restored.parameters == effect.parameters


# =============================================================================
# EQUIPMENT STATS TESTS
# =============================================================================


class TestEquipmentStats:
    """Whitebox tests for EquipmentStats."""

    def test_basic_creation(self):
        """Test basic equipment stats creation."""
        stats = EquipmentStats(
            armor=20.0,
            damage=15.0,
            attack_speed=1.2,
            block_chance=0.1,
        )
        assert stats.armor == 20.0
        assert stats.damage == 15.0
        assert stats.attack_speed == 1.2
        assert stats.block_chance == 0.1

    def test_with_attribute_modifiers(self):
        """Stats with attribute modifiers should work."""
        stats = EquipmentStats(
            armor=10.0,
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=5),
                StatModifier(stat_type=AttributeType.DEXTERITY, flat_bonus=3),
            ),
        )
        assert len(stats.attribute_modifiers) == 2

    def test_with_resistance_modifiers(self):
        """Stats with resistance modifiers should work."""
        stats = EquipmentStats(
            armor=10.0,
            resistance_modifiers=(
                ResistanceModifier(resistance_type=ResistanceType.FIRE, flat_bonus=10),
            ),
        )
        assert len(stats.resistance_modifiers) == 1

    def test_with_special_effects(self):
        """Stats with special effects should work."""
        stats = EquipmentStats(
            damage=20.0,
            special_effects=(
                SpecialEffect(effect_id="crit", name="Critical Strike"),
            ),
        )
        assert len(stats.special_effects) == 1

    def test_combine_basic_stats(self):
        """Combining stats should add basic values."""
        stats1 = EquipmentStats(armor=10.0, damage=5.0)
        stats2 = EquipmentStats(armor=15.0, damage=10.0)
        combined = stats1.combine(stats2)
        assert combined.armor == 25.0
        assert combined.damage == 15.0

    def test_combine_attribute_modifiers(self):
        """Combining should merge attribute modifiers for same stat."""
        stats1 = EquipmentStats(
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=5),
            ),
        )
        stats2 = EquipmentStats(
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=3),
                StatModifier(stat_type=AttributeType.DEXTERITY, flat_bonus=2),
            ),
        )
        combined = stats1.combine(stats2)
        assert len(combined.attribute_modifiers) == 2
        # Find strength modifier
        str_mod = next(m for m in combined.attribute_modifiers if m.stat_type == AttributeType.STRENGTH)
        assert str_mod.flat_bonus == 8

    def test_combine_special_effects_no_duplicates(self):
        """Combining should not duplicate special effects."""
        effect = SpecialEffect(effect_id="fire", name="Fire")
        stats1 = EquipmentStats(special_effects=(effect,))
        stats2 = EquipmentStats(special_effects=(effect,))
        combined = stats1.combine(stats2)
        assert len(combined.special_effects) == 1

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        stats = EquipmentStats(
            armor=20.0,
            damage=15.0,
            attack_speed=1.1,
            block_chance=0.2,
            attribute_modifiers=(
                StatModifier(stat_type=AttributeType.STRENGTH, flat_bonus=10),
            ),
            resistance_modifiers=(
                ResistanceModifier(resistance_type=ResistanceType.FIRE, flat_bonus=15),
            ),
            special_effects=(
                SpecialEffect(effect_id="test", name="Test Effect"),
            ),
        )
        data = stats.to_dict()
        restored = EquipmentStats.from_dict(data)
        assert restored.armor == stats.armor
        assert restored.damage == stats.damage
        assert len(restored.attribute_modifiers) == 1
        assert len(restored.resistance_modifiers) == 1
        assert len(restored.special_effects) == 1


# =============================================================================
# EQUIPMENT DEFINITION TESTS
# =============================================================================


class TestEquipmentDefinition:
    """Whitebox tests for EquipmentDefinition."""

    def test_basic_creation(self, basic_sword_def):
        """Test basic equipment definition creation."""
        assert basic_sword_def.id == "iron_sword"
        assert basic_sword_def.slot == EquipmentSlot.MAIN_HAND
        assert basic_sword_def.item_type == ItemType.EQUIPMENT
        assert basic_sword_def.max_stack == 1

    def test_always_non_stackable(self):
        """Equipment should always have max_stack = 1."""
        equip = EquipmentDefinition(
            id="test",
            name="Test",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            max_stack=99,  # Should be overridden
        )
        assert equip.max_stack == 1

    def test_with_required_attributes(self):
        """Equipment can have attribute requirements."""
        equip = EquipmentDefinition(
            id="heavy_armor",
            name="Heavy Armor",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            required_attributes={
                AttributeType.STRENGTH: 20,
                AttributeType.CONSTITUTION: 15,
            },
        )
        assert equip.required_attributes[AttributeType.STRENGTH] == 20

    def test_with_sockets(self):
        """Equipment can have sockets."""
        equip = EquipmentDefinition(
            id="socketed_sword",
            name="Socketed Sword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            socket_count=3,
        )
        assert equip.socket_count == 3

    def test_with_set_id(self):
        """Equipment can belong to a set."""
        equip = EquipmentDefinition(
            id="dragon_helm",
            name="Dragon Helm",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.HEAD,
            set_id="dragon_set",
        )
        assert equip.set_id == "dragon_set"

    def test_serialization_round_trip(self, basic_sword_def):
        """Serialization should preserve data."""
        data = basic_sword_def.to_dict()
        restored = EquipmentDefinition.from_dict(data)
        assert restored.id == basic_sword_def.id
        assert restored.name == basic_sword_def.name
        assert restored.slot == basic_sword_def.slot
        assert restored.stats.damage == basic_sword_def.stats.damage


# =============================================================================
# EQUIPMENT INSTANCE TESTS
# =============================================================================


class TestEquipmentInstance:
    """Whitebox tests for EquipmentInstance."""

    def test_basic_creation(self, sword_instance, basic_sword_def):
        """Test basic equipment instance creation."""
        assert sword_instance.definition == basic_sword_def
        assert sword_instance.slot == EquipmentSlot.MAIN_HAND
        assert isinstance(sword_instance.instance_id, UUID)

    def test_equipment_def_property(self, sword_instance, basic_sword_def):
        """equipment_def should return typed definition."""
        assert sword_instance.equipment_def == basic_sword_def

    def test_effective_stats_no_upgrade(self, sword_instance):
        """Effective stats should equal base with no upgrades."""
        base = sword_instance.equipment_def.stats
        effective = sword_instance.effective_stats
        assert effective.damage == base.damage

    def test_effective_stats_with_upgrade(self, basic_sword_def):
        """Effective stats should scale with upgrade level."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            upgrade_level=5,
        )
        base_damage = basic_sword_def.stats.damage
        expected = base_damage * (1 + 5 * UPGRADE_BONUS_PER_LEVEL)
        assert instance.effective_stats.damage == pytest.approx(expected)

    def test_enchantments(self, basic_sword_def):
        """Equipment can have enchantments."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            enchantments=["fire_enchant", "lifesteal"],
        )
        assert len(instance.enchantments) == 2

    def test_socketed_gems(self, basic_sword_def):
        """Equipment can have socketed gems."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            socketed_gems=["ruby", "sapphire"],
        )
        assert len(instance.socketed_gems) == 2

    def test_serialization_round_trip(self, basic_sword_def):
        """Serialization should preserve data."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            durability=75.0,
            enchantments=["fire"],
            socketed_gems=["ruby"],
            upgrade_level=3,
        )
        data = instance.to_dict()
        restored = EquipmentInstance.from_dict(data)
        assert restored.durability == instance.durability
        assert restored.enchantments == instance.enchantments
        assert restored.socketed_gems == instance.socketed_gems
        assert restored.upgrade_level == instance.upgrade_level


# =============================================================================
# EQUIPMENT SET TESTS
# =============================================================================


class TestSetBonus:
    """Whitebox tests for SetBonus."""

    def test_basic_creation(self):
        """Test basic set bonus creation."""
        bonus = SetBonus(
            pieces_required=2,
            stats=EquipmentStats(armor=10.0),
            description="2-piece bonus",
        )
        assert bonus.pieces_required == 2
        assert bonus.stats.armor == 10.0

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        bonus = SetBonus(
            pieces_required=4,
            stats=EquipmentStats(damage=25.0),
            description="4-piece bonus",
        )
        data = bonus.to_dict()
        restored = SetBonus.from_dict(data)
        assert restored.pieces_required == bonus.pieces_required
        assert restored.stats.damage == bonus.stats.damage


class TestEquipmentSet:
    """Whitebox tests for EquipmentSet."""

    def test_basic_creation(self):
        """Test basic equipment set creation."""
        eq_set = EquipmentSet(
            set_id="dragon",
            name="Dragon Set",
            piece_ids=frozenset({"dragon_helm", "dragon_chest", "dragon_gloves"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=20)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=50)),
            ),
        )
        assert eq_set.set_id == "dragon"
        assert len(eq_set.piece_ids) == 3
        assert len(eq_set.bonuses) == 2

    def test_get_active_bonuses_none(self):
        """No bonuses active with 0-1 pieces."""
        eq_set = EquipmentSet(
            set_id="dragon",
            name="Dragon Set",
            piece_ids=frozenset({"dragon_helm", "dragon_chest", "dragon_gloves"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=20)),
            ),
        )
        active = eq_set.get_active_bonuses({"dragon_helm"})
        assert len(active) == 0

    def test_get_active_bonuses_partial(self):
        """2-piece bonus active with 2 pieces."""
        eq_set = EquipmentSet(
            set_id="dragon",
            name="Dragon Set",
            piece_ids=frozenset({"dragon_helm", "dragon_chest", "dragon_gloves"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=20)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=50)),
            ),
        )
        active = eq_set.get_active_bonuses({"dragon_helm", "dragon_chest"})
        assert len(active) == 1
        assert active[0].pieces_required == 2

    def test_get_active_bonuses_full(self):
        """All bonuses active with full set."""
        eq_set = EquipmentSet(
            set_id="dragon",
            name="Dragon Set",
            piece_ids=frozenset({"dragon_helm", "dragon_chest", "dragon_gloves"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=20)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=50)),
            ),
        )
        active = eq_set.get_active_bonuses({"dragon_helm", "dragon_chest", "dragon_gloves"})
        assert len(active) == 2

    def test_serialization_round_trip(self):
        """Serialization should preserve data."""
        eq_set = EquipmentSet(
            set_id="dragon",
            name="Dragon Set",
            piece_ids=frozenset({"dragon_helm", "dragon_chest"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=20)),
            ),
        )
        data = eq_set.to_dict()
        restored = EquipmentSet.from_dict(data)
        assert restored.set_id == eq_set.set_id
        assert restored.name == eq_set.name
        assert restored.piece_ids == eq_set.piece_ids


# =============================================================================
# EQUIPMENT CONTAINER TESTS
# =============================================================================


class TestEquipmentContainer:
    """Whitebox tests for EquipmentContainer."""

    def test_basic_creation(self, equipment_container):
        """Test basic container creation."""
        assert equipment_container.owner_id == "player1"
        assert isinstance(equipment_container.id, UUID)

    def test_limited_slots(self):
        """Container with limited slots should work."""
        container = EquipmentContainer(
            owner_id="player1",
            allowed_slots={EquipmentSlot.MAIN_HAND, EquipmentSlot.OFF_HAND},
        )
        assert len(container._allowed_slots) == 2

    def test_can_equip_valid(self, equipment_container, sword_instance):
        """can_equip should return True for valid equip."""
        can, reason = equipment_container.can_equip(sword_instance)
        assert can is True
        assert reason == ""

    def test_can_equip_slot_not_allowed(self, sword_instance):
        """can_equip should fail for disallowed slot."""
        container = EquipmentContainer(
            owner_id="player1",
            allowed_slots={EquipmentSlot.HEAD},  # No main hand
        )
        can, reason = container.can_equip(sword_instance)
        assert can is False
        assert "not available" in reason

    def test_can_equip_level_requirement(self, equipment_container):
        """can_equip should check level requirement."""
        equip_def = EquipmentDefinition(
            id="high_level",
            name="High Level",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            level_requirement=50,
        )
        instance = EquipmentInstance(definition=equip_def, quantity=1)
        can, reason = equipment_container.can_equip(
            instance,
            character_stats={AttributeType.WISDOM: 10},  # Level too low
        )
        assert can is False
        assert "level" in reason.lower()

    def test_can_equip_attribute_requirement(self, equipment_container):
        """can_equip should check attribute requirements."""
        equip_def = EquipmentDefinition(
            id="heavy_armor",
            name="Heavy Armor",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            required_attributes={AttributeType.STRENGTH: 20},
        )
        instance = EquipmentInstance(definition=equip_def, quantity=1)
        can, reason = equipment_container.can_equip(
            instance,
            character_stats={AttributeType.STRENGTH: 10},  # Not enough
        )
        assert can is False
        assert "STRENGTH" in reason

    def test_equip_basic(self, equipment_container, sword_instance):
        """Basic equip should work."""
        success, old = equipment_container.equip(sword_instance)
        assert success is True
        assert old is None
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == sword_instance

    def test_equip_replace(self, equipment_container, basic_sword_def):
        """Equipping to occupied slot should replace."""
        sword1 = EquipmentInstance(definition=basic_sword_def, quantity=1)
        sword2 = EquipmentInstance(definition=basic_sword_def, quantity=1)

        equipment_container.equip(sword1)
        success, old = equipment_container.equip(sword2)

        assert success is True
        assert old == sword1
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == sword2

    def test_equip_two_hand_removes_one_hand(self, equipment_container, sword_instance, shield_instance, two_hand_weapon_def):
        """Two-hand weapon should unequip main and off hand."""
        two_hand = EquipmentInstance(definition=two_hand_weapon_def, quantity=1)

        equipment_container.equip(sword_instance)
        equipment_container.equip(shield_instance)
        success, old = equipment_container.equip(two_hand)

        assert success is True
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None
        assert equipment_container.get(EquipmentSlot.OFF_HAND) is None
        assert equipment_container.get(EquipmentSlot.TWO_HAND) == two_hand

    def test_equip_ring_in_either_slot(self, equipment_container, ring_def):
        """Ring can go in either ring slot."""
        ring1 = EquipmentInstance(definition=ring_def, quantity=1)
        ring2 = EquipmentInstance(definition=ring_def, quantity=1)

        equipment_container.equip(ring1, slot=EquipmentSlot.RING_1)
        equipment_container.equip(ring2, slot=EquipmentSlot.RING_2)

        assert equipment_container.get(EquipmentSlot.RING_1) == ring1
        assert equipment_container.get(EquipmentSlot.RING_2) == ring2

    def test_equip_force_bypasses_checks(self, equipment_container):
        """Force equip should bypass requirement checks."""
        equip_def = EquipmentDefinition(
            id="impossible",
            name="Impossible",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            required_attributes={AttributeType.STRENGTH: 999},
        )
        instance = EquipmentInstance(definition=equip_def, quantity=1)
        success, old = equipment_container.equip(instance, force=True)
        assert success is True

    def test_unequip(self, equipment_container, sword_instance):
        """Unequip should remove item."""
        equipment_container.equip(sword_instance)
        removed = equipment_container.unequip(EquipmentSlot.MAIN_HAND)
        assert removed == sword_instance
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None

    def test_unequip_empty_slot(self, equipment_container):
        """Unequip empty slot should return None."""
        removed = equipment_container.unequip(EquipmentSlot.HEAD)
        assert removed is None

    def test_unequip_all(self, equipment_container, sword_instance, shield_instance, helmet_instance):
        """Unequip all should remove everything."""
        equipment_container.equip(sword_instance)
        equipment_container.equip(shield_instance)
        equipment_container.equip(helmet_instance)

        removed = equipment_container.unequip_all()
        assert len(removed) == 3
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None
        assert equipment_container.get(EquipmentSlot.OFF_HAND) is None
        assert equipment_container.get(EquipmentSlot.HEAD) is None

    def test_swap(self, equipment_container, basic_sword_def):
        """Swap should exchange items."""
        sword1 = EquipmentInstance(definition=basic_sword_def, quantity=1)
        sword2 = EquipmentInstance(definition=basic_sword_def, quantity=1)

        equipment_container.equip(sword1)
        old = equipment_container.swap(EquipmentSlot.MAIN_HAND, sword2)

        assert old == sword1
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == sword2

    def test_is_slot_empty(self, equipment_container, sword_instance):
        """is_slot_empty should check occupation."""
        assert equipment_container.is_slot_empty(EquipmentSlot.MAIN_HAND) is True
        equipment_container.equip(sword_instance)
        assert equipment_container.is_slot_empty(EquipmentSlot.MAIN_HAND) is False

    def test_get_all_equipped(self, equipment_container, sword_instance, shield_instance):
        """get_all_equipped should return all equipped items."""
        equipment_container.equip(sword_instance)
        equipment_container.equip(shield_instance)

        equipped = equipment_container.get_all_equipped()
        assert len(equipped) == 2
        slots = {slot for slot, item in equipped}
        assert EquipmentSlot.MAIN_HAND in slots
        assert EquipmentSlot.OFF_HAND in slots

    def test_get_equipped_ids(self, equipment_container, sword_instance, shield_instance):
        """get_equipped_ids should return item IDs."""
        equipment_container.equip(sword_instance)
        equipment_container.equip(shield_instance)

        ids = equipment_container.get_equipped_ids()
        assert "iron_sword" in ids
        assert "iron_shield" in ids

    def test_find_by_id(self, equipment_container, sword_instance):
        """find_by_id should find equipped item."""
        equipment_container.equip(sword_instance)
        result = equipment_container.find_by_id("iron_sword")
        assert result is not None
        slot, item = result
        assert slot == EquipmentSlot.MAIN_HAND
        assert item == sword_instance


class TestEquipmentContainerStats:
    """Tests for equipment container stat calculation."""

    def test_combined_stats_empty(self, equipment_container):
        """Empty container should have zero stats."""
        stats = equipment_container.combined_stats
        assert stats.armor == 0.0
        assert stats.damage == 0.0

    def test_combined_stats_single_item(self, equipment_container, sword_instance):
        """Single item stats should be reflected."""
        equipment_container.equip(sword_instance)
        stats = equipment_container.combined_stats
        assert stats.damage == sword_instance.effective_stats.damage

    def test_combined_stats_multiple_items(self, equipment_container, sword_instance, shield_instance, helmet_instance):
        """Multiple item stats should combine."""
        equipment_container.equip(sword_instance)
        equipment_container.equip(shield_instance)
        equipment_container.equip(helmet_instance)

        stats = equipment_container.combined_stats
        expected_armor = (
            shield_instance.effective_stats.armor + helmet_instance.effective_stats.armor
        )
        assert stats.armor == expected_armor
        assert stats.damage == sword_instance.effective_stats.damage

    def test_combined_stats_with_set_bonus(self, equipment_container):
        """Set bonuses should be included in combined stats."""
        # Create set pieces
        helm_def = EquipmentDefinition(
            id="set_helm",
            name="Set Helm",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.HEAD,
            stats=EquipmentStats(armor=10),
            set_id="test_set",
        )
        chest_def = EquipmentDefinition(
            id="set_chest",
            name="Set Chest",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.CHEST,
            stats=EquipmentStats(armor=15),
            set_id="test_set",
        )

        # Create set with bonus
        test_set = EquipmentSet(
            set_id="test_set",
            name="Test Set",
            piece_ids=frozenset({"set_helm", "set_chest"}),
            bonuses=(SetBonus(pieces_required=2, stats=EquipmentStats(armor=25)),),
        )
        equipment_container.register_set(test_set)

        # Equip pieces
        equipment_container.equip(EquipmentInstance(definition=helm_def, quantity=1))
        equipment_container.equip(EquipmentInstance(definition=chest_def, quantity=1))

        stats = equipment_container.combined_stats
        # 10 + 15 + 25 bonus = 50
        assert stats.armor == 50.0

    def test_get_attribute_modifier(self, equipment_container, ring_def):
        """get_attribute_modifier should find specific modifier."""
        ring = EquipmentInstance(definition=ring_def, quantity=1)
        equipment_container.equip(ring, slot=EquipmentSlot.RING_1)

        mod = equipment_container.get_attribute_modifier(AttributeType.LUCK)
        assert mod is not None
        assert mod.flat_bonus == 5

    def test_get_total_armor(self, equipment_container, helmet_instance, shield_instance):
        """get_total_armor should return sum."""
        equipment_container.equip(helmet_instance)
        equipment_container.equip(shield_instance)

        total = equipment_container.get_total_armor()
        expected = helmet_instance.effective_stats.armor + shield_instance.effective_stats.armor
        assert total == expected

    def test_has_effect(self, equipment_container):
        """has_effect should check for special effects."""
        equip_def = EquipmentDefinition(
            id="fire_sword",
            name="Fire Sword",
            item_type=ItemType.EQUIPMENT,
            slot=EquipmentSlot.MAIN_HAND,
            stats=EquipmentStats(
                damage=20,
                special_effects=(SpecialEffect(effect_id="fire_damage", name="Fire"),),
            ),
        )
        instance = EquipmentInstance(definition=equip_def, quantity=1)
        equipment_container.equip(instance)

        assert equipment_container.has_effect("fire_damage") is True
        assert equipment_container.has_effect("ice_damage") is False


class TestEquipmentContainerDurability:
    """Tests for equipment durability system."""

    def test_reduce_durability(self, equipment_container, basic_sword_def):
        """reduce_durability should decrease durability."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            durability=100.0,
        )
        equipment_container.equip(sword)

        broke = equipment_container.reduce_durability(EquipmentSlot.MAIN_HAND, 25.0)
        assert broke is False
        assert equipment_container.get(EquipmentSlot.MAIN_HAND).durability == 75.0

    def test_reduce_durability_breaks_item(self, equipment_container, basic_sword_def):
        """Item should break when durability reaches 0."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            durability=10.0,
        )
        equipment_container.equip(sword)

        broke = equipment_container.reduce_durability(EquipmentSlot.MAIN_HAND, 15.0)
        assert broke is True
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None

    def test_reduce_all_durability(self, equipment_container, basic_sword_def, basic_shield_def):
        """reduce_all_durability should affect all items."""
        sword = EquipmentInstance(definition=basic_sword_def, quantity=1, durability=20.0)
        shield = EquipmentInstance(definition=basic_shield_def, quantity=1, durability=30.0)
        equipment_container.equip(sword)
        equipment_container.equip(shield)

        broken = equipment_container.reduce_all_durability(25.0)
        assert len(broken) == 1  # Only sword broke
        assert EquipmentSlot.MAIN_HAND in broken
        assert equipment_container.get(EquipmentSlot.OFF_HAND).durability == 5.0

    def test_repair(self, equipment_container, basic_sword_def):
        """repair should restore durability."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            durability=50.0,
        )
        equipment_container.equip(sword)

        repaired = equipment_container.repair(EquipmentSlot.MAIN_HAND, 30.0)
        assert repaired == 30.0
        assert equipment_container.get(EquipmentSlot.MAIN_HAND).durability == 80.0

    def test_repair_full(self, equipment_container, basic_sword_def):
        """repair with None should fully repair."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            quantity=1,
            durability=50.0,
        )
        equipment_container.equip(sword)

        repaired = equipment_container.repair(EquipmentSlot.MAIN_HAND)
        assert repaired == 50.0
        assert equipment_container.get(EquipmentSlot.MAIN_HAND).durability == DEFAULT_MAX_DURABILITY


class TestEquipmentContainerEvents:
    """Tests for equipment change callbacks."""

    def test_add_change_listener(self, equipment_container, sword_instance):
        """Change listener should be called on equip."""
        events = []
        equipment_container.add_change_listener(
            lambda slot, old, new: events.append((slot, old, new))
        )

        equipment_container.equip(sword_instance)
        assert len(events) == 1
        slot, old, new = events[0]
        assert slot == EquipmentSlot.MAIN_HAND
        assert old is None
        assert new == sword_instance

    def test_remove_change_listener(self, equipment_container, sword_instance):
        """Removed listener should not be called."""
        events = []
        listener = lambda slot, old, new: events.append((slot, old, new))
        equipment_container.add_change_listener(listener)
        equipment_container.remove_change_listener(listener)

        equipment_container.equip(sword_instance)
        assert len(events) == 0


class TestEquipmentContainerSerialization:
    """Tests for equipment container serialization."""

    def test_serialization_round_trip(self, equipment_container, sword_instance, shield_instance):
        """Serialization should preserve state."""
        equipment_container.equip(sword_instance)
        equipment_container.equip(shield_instance)

        data = equipment_container.to_dict(embed_definitions=True)
        restored = EquipmentContainer.from_dict(data)

        assert restored.owner_id == equipment_container.owner_id
        assert restored.get(EquipmentSlot.MAIN_HAND) is not None
        assert restored.get(EquipmentSlot.OFF_HAND) is not None

    def test_serialization_with_registry(self, equipment_container, basic_sword_def, basic_shield_def):
        """Serialization with registry should work."""
        registry = EquipmentRegistry.instance()
        registry.register_equipment(basic_sword_def)
        registry.register_equipment(basic_shield_def)

        sword = EquipmentInstance(definition=basic_sword_def, quantity=1)
        shield = EquipmentInstance(definition=basic_shield_def, quantity=1)
        equipment_container.equip(sword)
        equipment_container.equip(shield)

        data = equipment_container.to_dict(embed_definitions=False)
        restored = EquipmentContainer.from_dict(data, {
            "iron_sword": basic_sword_def,
            "iron_shield": basic_shield_def,
        })

        assert restored.get(EquipmentSlot.MAIN_HAND).definition == basic_sword_def


# =============================================================================
# EQUIPMENT REGISTRY TESTS
# =============================================================================


class TestEquipmentRegistry:
    """Whitebox tests for EquipmentRegistry."""

    def test_singleton(self):
        """Registry should be singleton."""
        EquipmentRegistry.reset()
        reg1 = EquipmentRegistry.instance()
        reg2 = EquipmentRegistry.instance()
        assert reg1 is reg2

    def test_register_equipment(self, basic_sword_def):
        """register_equipment should add definition."""
        registry = EquipmentRegistry.instance()
        registry.register_equipment(basic_sword_def)
        assert registry.get_equipment("iron_sword") == basic_sword_def

    def test_register_duplicate_raises(self, basic_sword_def):
        """Duplicate registration should raise."""
        registry = EquipmentRegistry.instance()
        registry.register_equipment(basic_sword_def)
        with pytest.raises(ValueError, match="already registered"):
            registry.register_equipment(basic_sword_def)

    def test_register_set(self):
        """register_set should add set definition."""
        registry = EquipmentRegistry.instance()
        eq_set = EquipmentSet(
            set_id="test",
            name="Test Set",
            piece_ids=frozenset(),
            bonuses=(),
        )
        registry.register_set(eq_set)
        assert registry.get_set("test") == eq_set

    def test_get_by_slot(self, basic_sword_def, basic_shield_def, basic_helmet_def):
        """get_by_slot should filter by slot."""
        registry = EquipmentRegistry.instance()
        registry.register_equipment(basic_sword_def)
        registry.register_equipment(basic_shield_def)
        registry.register_equipment(basic_helmet_def)

        main_hand = registry.get_by_slot(EquipmentSlot.MAIN_HAND)
        assert len(main_hand) == 1
        assert main_hand[0].id == "iron_sword"

    def test_clear(self, basic_sword_def):
        """clear should remove all registrations."""
        registry = EquipmentRegistry.instance()
        registry.register_equipment(basic_sword_def)
        registry.clear()
        assert registry.get_equipment("iron_sword") is None
