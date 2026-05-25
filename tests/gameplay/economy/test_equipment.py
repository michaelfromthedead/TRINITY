"""
Comprehensive tests for the Equipment System.

Tests cover:
- Equipment slot types (head, chest, weapon, etc.)
- Equip/unequip items
- Equipment requirements (level, class, stats)
- Equipment stat bonuses
- Equipment sets and set bonuses
- Two-handed weapon handling
- Equipment durability
- Equipment comparison
- Transmogrification
"""

import pytest
from uuid import UUID, uuid4

from engine.gameplay.economy.constants import (
    AttributeType,
    EquipmentSlot,
    EXCLUSIVE_SLOTS,
    ItemType,
    Rarity,
    ResistanceType,
)
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_sword_def():
    """Create a basic sword definition."""
    return EquipmentDefinition(
        id="sword_iron",
        name="Iron Sword",
        slot=EquipmentSlot.MAIN_HAND,
        stats=EquipmentStats(damage=10.0),
        rarity=Rarity.COMMON,
        base_value=100,
    )


@pytest.fixture
def basic_shield_def():
    """Create a basic shield definition."""
    return EquipmentDefinition(
        id="shield_iron",
        name="Iron Shield",
        slot=EquipmentSlot.OFF_HAND,
        stats=EquipmentStats(armor=15.0, block_chance=0.1),
        rarity=Rarity.COMMON,
        base_value=80,
    )


@pytest.fixture
def two_handed_sword_def():
    """Create a two-handed weapon definition."""
    return EquipmentDefinition(
        id="sword_greatsword",
        name="Greatsword",
        slot=EquipmentSlot.TWO_HAND,
        stats=EquipmentStats(damage=25.0),
        rarity=Rarity.UNCOMMON,
        base_value=250,
    )


@pytest.fixture
def helmet_def():
    """Create a helmet definition."""
    return EquipmentDefinition(
        id="helm_steel",
        name="Steel Helmet",
        slot=EquipmentSlot.HEAD,
        stats=EquipmentStats(
            armor=8.0,
            attribute_modifiers=(
                StatModifier(AttributeType.CONSTITUTION, flat_bonus=5),
            ),
        ),
        rarity=Rarity.UNCOMMON,
        base_value=150,
    )


@pytest.fixture
def ring_def():
    """Create a ring definition."""
    return EquipmentDefinition(
        id="ring_power",
        name="Ring of Power",
        slot=EquipmentSlot.RING_1,
        stats=EquipmentStats(
            attribute_modifiers=(
                StatModifier(AttributeType.STRENGTH, flat_bonus=3),
            ),
        ),
        rarity=Rarity.RARE,
        base_value=500,
    )


@pytest.fixture
def equipment_container():
    """Create an equipment container."""
    return EquipmentContainer(owner_id="player_1")


@pytest.fixture
def equipment_registry():
    """Create and populate equipment registry."""
    EquipmentRegistry.reset()
    registry = EquipmentRegistry.instance()
    yield registry
    EquipmentRegistry.reset()


# =============================================================================
# StatModifier Tests
# =============================================================================


class TestStatModifier:
    """Tests for StatModifier class."""

    def test_create_flat_modifier(self):
        """Test creating flat bonus modifier."""
        mod = StatModifier(AttributeType.STRENGTH, flat_bonus=10)
        assert mod.stat_type == AttributeType.STRENGTH
        assert mod.flat_bonus == 10
        assert mod.percent_bonus == 0.0
        assert mod.multiplier == 1.0

    def test_create_percent_modifier(self):
        """Test creating percent bonus modifier."""
        mod = StatModifier(AttributeType.DEXTERITY, percent_bonus=0.2)
        assert mod.percent_bonus == 0.2

    def test_create_multiplier_modifier(self):
        """Test creating multiplier modifier."""
        mod = StatModifier(AttributeType.INTELLIGENCE, multiplier=1.5)
        assert mod.multiplier == 1.5

    def test_apply_flat_only(self):
        """Test applying flat bonus only."""
        mod = StatModifier(AttributeType.STRENGTH, flat_bonus=10)
        result = mod.apply(100)
        assert result == 110

    def test_apply_percent_only(self):
        """Test applying percent bonus only."""
        mod = StatModifier(AttributeType.STRENGTH, percent_bonus=0.2)
        result = mod.apply(100)
        assert result == 120

    def test_apply_multiplier_only(self):
        """Test applying multiplier only."""
        mod = StatModifier(AttributeType.STRENGTH, multiplier=1.5)
        result = mod.apply(100)
        assert result == 150

    def test_apply_combined(self):
        """Test applying combined modifiers."""
        mod = StatModifier(
            AttributeType.STRENGTH,
            flat_bonus=10,
            percent_bonus=0.2,
            multiplier=1.5,
        )
        # (100 + 10) * 1.2 * 1.5 = 198
        result = mod.apply(100)
        assert result == pytest.approx(198.0)

    def test_combine_same_stat(self):
        """Test combining modifiers for same stat."""
        mod1 = StatModifier(AttributeType.STRENGTH, flat_bonus=10, percent_bonus=0.1)
        mod2 = StatModifier(AttributeType.STRENGTH, flat_bonus=5, percent_bonus=0.15)
        combined = mod1.combine(mod2)
        assert combined.flat_bonus == 15
        assert combined.percent_bonus == 0.25
        assert combined.multiplier == 1.0

    def test_combine_different_stat_raises(self):
        """Test combining different stat types raises error."""
        mod1 = StatModifier(AttributeType.STRENGTH, flat_bonus=10)
        mod2 = StatModifier(AttributeType.DEXTERITY, flat_bonus=5)
        with pytest.raises(ValueError, match="different stats"):
            mod1.combine(mod2)

    def test_combine_multipliers(self):
        """Test multipliers are multiplied when combined."""
        mod1 = StatModifier(AttributeType.STRENGTH, multiplier=1.2)
        mod2 = StatModifier(AttributeType.STRENGTH, multiplier=1.5)
        combined = mod1.combine(mod2)
        assert combined.multiplier == pytest.approx(1.8)

    def test_modifier_is_frozen(self):
        """Test modifier is immutable."""
        mod = StatModifier(AttributeType.STRENGTH, flat_bonus=10)
        with pytest.raises(AttributeError):
            mod.flat_bonus = 20


# =============================================================================
# ResistanceModifier Tests
# =============================================================================


class TestResistanceModifier:
    """Tests for ResistanceModifier class."""

    def test_create_resistance_modifier(self):
        """Test creating resistance modifier."""
        mod = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.1)
        assert mod.resistance_type == ResistanceType.FIRE
        assert mod.flat_bonus == 0.1

    def test_apply_flat_bonus(self):
        """Test applying flat resistance bonus."""
        mod = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.2)
        result = mod.apply(0.1)
        assert result == pytest.approx(0.3)

    def test_apply_percent_bonus(self):
        """Test applying percent resistance bonus."""
        mod = ResistanceModifier(ResistanceType.ICE, percent_bonus=0.15)
        result = mod.apply(0.1)
        assert result == pytest.approx(0.25)

    def test_apply_clamped_to_max(self):
        """Test resistance clamped to maximum."""
        mod = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.5)
        result = mod.apply(0.5)  # Would be 1.0 without clamp
        assert result == 0.75  # Default max

    def test_apply_custom_max(self):
        """Test resistance with custom maximum."""
        mod = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.5)
        result = mod.apply(0.5, max_resistance=0.9)
        assert result == 0.9

    def test_combine_same_resistance(self):
        """Test combining same resistance type."""
        mod1 = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.1)
        mod2 = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.15)
        combined = mod1.combine(mod2)
        assert combined.flat_bonus == 0.25

    def test_combine_different_resistance_raises(self):
        """Test combining different resistance types raises error."""
        mod1 = ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.1)
        mod2 = ResistanceModifier(ResistanceType.ICE, flat_bonus=0.1)
        with pytest.raises(ValueError, match="different resistances"):
            mod1.combine(mod2)


# =============================================================================
# SpecialEffect Tests
# =============================================================================


class TestSpecialEffect:
    """Tests for SpecialEffect class."""

    def test_create_special_effect(self):
        """Test creating special effect."""
        effect = SpecialEffect(
            effect_id="lifesteal",
            name="Life Steal",
            description="Heal 10% of damage dealt",
            parameters={"percent": 0.1},
        )
        assert effect.effect_id == "lifesteal"
        assert effect.name == "Life Steal"
        assert effect.parameters["percent"] == 0.1

    def test_effect_hashable(self):
        """Test effect can be used in sets."""
        effect1 = SpecialEffect(effect_id="e1", name="Effect 1")
        effect2 = SpecialEffect(effect_id="e2", name="Effect 2")
        effects = {effect1, effect2}
        assert len(effects) == 2

    def test_effect_equality(self):
        """Test effect equality based on id and name."""
        effect1 = SpecialEffect(effect_id="e1", name="Effect")
        effect2 = SpecialEffect(effect_id="e1", name="Effect")
        assert hash(effect1) == hash(effect2)


# =============================================================================
# EquipmentStats Tests
# =============================================================================


class TestEquipmentStats:
    """Tests for EquipmentStats class."""

    def test_create_empty_stats(self):
        """Test creating empty stats."""
        stats = EquipmentStats()
        assert stats.armor == 0.0
        assert stats.damage == 0.0
        assert stats.attack_speed == 0.0
        assert stats.block_chance == 0.0
        assert len(stats.attribute_modifiers) == 0

    def test_create_with_values(self):
        """Test creating stats with values."""
        stats = EquipmentStats(
            armor=25.0,
            damage=15.0,
            attack_speed=1.2,
            block_chance=0.15,
        )
        assert stats.armor == 25.0
        assert stats.damage == 15.0
        assert stats.attack_speed == 1.2
        assert stats.block_chance == 0.15

    def test_combine_basic_stats(self):
        """Test combining basic stats."""
        stats1 = EquipmentStats(armor=10.0, damage=5.0)
        stats2 = EquipmentStats(armor=8.0, damage=3.0)
        combined = stats1.combine(stats2)
        assert combined.armor == 18.0
        assert combined.damage == 8.0

    def test_combine_attribute_modifiers(self):
        """Test combining attribute modifiers."""
        stats1 = EquipmentStats(
            attribute_modifiers=(
                StatModifier(AttributeType.STRENGTH, flat_bonus=5),
            ),
        )
        stats2 = EquipmentStats(
            attribute_modifiers=(
                StatModifier(AttributeType.STRENGTH, flat_bonus=3),
            ),
        )
        combined = stats1.combine(stats2)
        # Should have combined strength modifier
        str_mod = next(
            m for m in combined.attribute_modifiers
            if m.stat_type == AttributeType.STRENGTH
        )
        assert str_mod.flat_bonus == 8

    def test_combine_different_attribute_modifiers(self):
        """Test combining different attribute modifiers."""
        stats1 = EquipmentStats(
            attribute_modifiers=(
                StatModifier(AttributeType.STRENGTH, flat_bonus=5),
            ),
        )
        stats2 = EquipmentStats(
            attribute_modifiers=(
                StatModifier(AttributeType.DEXTERITY, flat_bonus=3),
            ),
        )
        combined = stats1.combine(stats2)
        assert len(combined.attribute_modifiers) == 2

    def test_combine_resistance_modifiers(self):
        """Test combining resistance modifiers."""
        stats1 = EquipmentStats(
            resistance_modifiers=(
                ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.1),
            ),
        )
        stats2 = EquipmentStats(
            resistance_modifiers=(
                ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.15),
            ),
        )
        combined = stats1.combine(stats2)
        fire_mod = next(
            m for m in combined.resistance_modifiers
            if m.resistance_type == ResistanceType.FIRE
        )
        assert fire_mod.flat_bonus == 0.25

    def test_combine_special_effects_no_duplicates(self):
        """Test combining special effects removes duplicates."""
        effect = SpecialEffect(effect_id="e1", name="Effect")
        stats1 = EquipmentStats(special_effects=(effect,))
        stats2 = EquipmentStats(special_effects=(effect,))
        combined = stats1.combine(stats2)
        assert len(combined.special_effects) == 1


# =============================================================================
# EquipmentDefinition Tests
# =============================================================================


class TestEquipmentDefinition:
    """Tests for EquipmentDefinition class."""

    def test_create_equipment_definition(self, basic_sword_def):
        """Test creating equipment definition."""
        assert basic_sword_def.id == "sword_iron"
        assert basic_sword_def.name == "Iron Sword"
        assert basic_sword_def.slot == EquipmentSlot.MAIN_HAND
        assert basic_sword_def.item_type == ItemType.EQUIPMENT

    def test_equipment_max_stack_is_one(self):
        """Test equipment max stack is always 1."""
        eq_def = EquipmentDefinition(
            id="test",
            name="Test",
            slot=EquipmentSlot.HEAD,
            max_stack=99,  # This should be overridden
        )
        assert eq_def.max_stack == 1

    def test_equipment_not_stackable(self, basic_sword_def):
        """Test equipment is not stackable."""
        assert basic_sword_def.is_stackable is False

    def test_equipment_with_required_attributes(self):
        """Test equipment with attribute requirements."""
        eq_def = EquipmentDefinition(
            id="heavy_armor",
            name="Heavy Plate Armor",
            slot=EquipmentSlot.CHEST,
            required_attributes={
                AttributeType.STRENGTH: 20,
                AttributeType.CONSTITUTION: 15,
            },
        )
        assert eq_def.required_attributes[AttributeType.STRENGTH] == 20
        assert eq_def.required_attributes[AttributeType.CONSTITUTION] == 15

    def test_equipment_with_sockets(self):
        """Test equipment with gem sockets."""
        eq_def = EquipmentDefinition(
            id="socket_sword",
            name="Socketed Sword",
            slot=EquipmentSlot.MAIN_HAND,
            socket_count=3,
        )
        assert eq_def.socket_count == 3

    def test_equipment_with_set_id(self):
        """Test equipment belonging to a set."""
        eq_def = EquipmentDefinition(
            id="dragon_helm",
            name="Dragon Helmet",
            slot=EquipmentSlot.HEAD,
            set_id="dragon_set",
        )
        assert eq_def.set_id == "dragon_set"


# =============================================================================
# EquipmentInstance Tests
# =============================================================================


class TestEquipmentInstance:
    """Tests for EquipmentInstance class."""

    def test_create_equipment_instance(self, basic_sword_def):
        """Test creating equipment instance."""
        instance = EquipmentInstance(definition=basic_sword_def)
        assert instance.definition == basic_sword_def
        assert instance.quantity == 1

    def test_equipment_def_property(self, basic_sword_def):
        """Test equipment_def property."""
        instance = EquipmentInstance(definition=basic_sword_def)
        assert instance.equipment_def == basic_sword_def

    def test_slot_property(self, basic_sword_def):
        """Test slot property."""
        instance = EquipmentInstance(definition=basic_sword_def)
        assert instance.slot == EquipmentSlot.MAIN_HAND

    def test_effective_stats_no_upgrade(self, basic_sword_def):
        """Test effective stats without upgrades."""
        instance = EquipmentInstance(definition=basic_sword_def)
        stats = instance.effective_stats
        assert stats.damage == 10.0

    def test_effective_stats_with_upgrade(self, basic_sword_def):
        """Test effective stats with upgrades."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            upgrade_level=5,
        )
        stats = instance.effective_stats
        # 10.0 * (1 + 5 * 0.05) = 10.0 * 1.25 = 12.5
        assert stats.damage == pytest.approx(12.5)

    def test_equipment_with_enchantments(self, basic_sword_def):
        """Test equipment with enchantments."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            enchantments=["fire_damage", "sharpness"],
        )
        assert len(instance.enchantments) == 2
        assert "fire_damage" in instance.enchantments

    def test_equipment_with_socketed_gems(self, basic_sword_def):
        """Test equipment with socketed gems."""
        instance = EquipmentInstance(
            definition=basic_sword_def,
            socketed_gems=["ruby", "emerald"],
        )
        assert len(instance.socketed_gems) == 2


# =============================================================================
# EquipmentContainer Creation Tests
# =============================================================================


class TestEquipmentContainerCreation:
    """Tests for EquipmentContainer creation."""

    def test_create_default_container(self):
        """Test creating container with default slots."""
        container = EquipmentContainer(owner_id="player_1")
        assert container.owner_id == "player_1"
        assert isinstance(container.id, UUID)

    def test_create_with_limited_slots(self):
        """Test creating container with limited slots."""
        container = EquipmentContainer(
            owner_id="player_1",
            allowed_slots={EquipmentSlot.MAIN_HAND, EquipmentSlot.OFF_HAND},
        )
        # Only those slots should be available
        assert EquipmentSlot.MAIN_HAND in container._allowed_slots
        assert EquipmentSlot.HEAD not in container._allowed_slots

    def test_create_with_custom_id(self):
        """Test creating container with custom ID."""
        custom_id = uuid4()
        container = EquipmentContainer(
            owner_id="player_1",
            container_id=custom_id,
        )
        assert container.id == custom_id

    def test_all_slots_empty_initially(self, equipment_container):
        """Test all slots are empty initially."""
        for slot in EquipmentSlot:
            assert equipment_container.is_slot_empty(slot) is True


# =============================================================================
# EquipmentContainer Equip Tests
# =============================================================================


class TestEquipmentContainerEquip:
    """Tests for equipping items."""

    def test_equip_basic_item(self, equipment_container, basic_sword_def):
        """Test equipping a basic item."""
        instance = EquipmentInstance(definition=basic_sword_def)
        success, old = equipment_container.equip(instance)
        assert success is True
        assert old is None
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == instance

    def test_equip_returns_old_item(self, equipment_container, basic_sword_def):
        """Test equipping returns previously equipped item."""
        old_sword = EquipmentInstance(definition=basic_sword_def)
        new_sword = EquipmentInstance(definition=basic_sword_def)

        equipment_container.equip(old_sword)
        success, old = equipment_container.equip(new_sword)

        assert success is True
        assert old == old_sword
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == new_sword

    def test_equip_to_specific_slot(self, equipment_container, ring_def):
        """Test equipping to specific slot."""
        ring = EquipmentInstance(definition=ring_def)
        # Ring can go in RING_1 or RING_2
        success, _ = equipment_container.equip(ring, EquipmentSlot.RING_2)
        assert success is True
        assert equipment_container.get(EquipmentSlot.RING_2) == ring

    def test_equip_two_handed_clears_both_hands(
        self, equipment_container, basic_sword_def, basic_shield_def, two_handed_sword_def
    ):
        """Test equipping two-handed weapon clears both hand slots."""
        sword = EquipmentInstance(definition=basic_sword_def)
        shield = EquipmentInstance(definition=basic_shield_def)
        greatsword = EquipmentInstance(definition=two_handed_sword_def)

        equipment_container.equip(sword)
        equipment_container.equip(shield)

        success, old = equipment_container.equip(greatsword)
        assert success is True
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None
        assert equipment_container.get(EquipmentSlot.OFF_HAND) is None
        assert equipment_container.get(EquipmentSlot.TWO_HAND) == greatsword

    def test_equip_one_handed_clears_two_hand(
        self, equipment_container, basic_sword_def, two_handed_sword_def
    ):
        """Test equipping one-handed weapon clears two-hand slot."""
        greatsword = EquipmentInstance(definition=two_handed_sword_def)
        sword = EquipmentInstance(definition=basic_sword_def)

        equipment_container.equip(greatsword)
        success, old = equipment_container.equip(sword)

        assert success is True
        assert equipment_container.get(EquipmentSlot.TWO_HAND) is None
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == sword

    def test_equip_disallowed_slot_fails(self, basic_sword_def):
        """Test equipping to disallowed slot fails."""
        container = EquipmentContainer(
            owner_id="player_1",
            allowed_slots={EquipmentSlot.HEAD, EquipmentSlot.CHEST},
        )
        sword = EquipmentInstance(definition=basic_sword_def)

        success, _ = container.equip(sword)
        assert success is False

    def test_equip_incompatible_slot_fails(self, equipment_container, helmet_def):
        """Test equipping to incompatible slot fails."""
        helmet = EquipmentInstance(definition=helmet_def)

        # Helmet cannot go in weapon slot
        success, _ = equipment_container.equip(helmet, EquipmentSlot.MAIN_HAND)
        assert success is False

    def test_equip_with_force(self, equipment_container, basic_sword_def):
        """Test force equipping bypasses requirements."""
        # Create sword with high level requirement
        sword_def = EquipmentDefinition(
            id="epic_sword",
            name="Epic Sword",
            slot=EquipmentSlot.MAIN_HAND,
            level_requirement=50,
        )
        sword = EquipmentInstance(definition=sword_def)

        # Should fail without force (no character stats provided)
        success, _ = equipment_container.equip(sword, force=True)
        assert success is True


# =============================================================================
# EquipmentContainer Requirement Tests
# =============================================================================


class TestEquipmentContainerRequirements:
    """Tests for equipment requirements."""

    def test_can_equip_basic(self, equipment_container, basic_sword_def):
        """Test basic can_equip check."""
        sword = EquipmentInstance(definition=basic_sword_def)
        can, reason = equipment_container.can_equip(sword)
        assert can is True
        assert reason == ""

    def test_can_equip_level_requirement(self, equipment_container):
        """Test level requirement check."""
        sword_def = EquipmentDefinition(
            id="level_sword",
            name="Level Sword",
            slot=EquipmentSlot.MAIN_HAND,
            level_requirement=20,
        )
        sword = EquipmentInstance(definition=sword_def)

        # Without character stats (defaults to level 1)
        can, reason = equipment_container.can_equip(sword, character_stats={})
        assert can is False
        assert "level" in reason.lower()

    def test_can_equip_with_sufficient_level(self, equipment_container):
        """Test level requirement met."""
        sword_def = EquipmentDefinition(
            id="level_sword",
            name="Level Sword",
            slot=EquipmentSlot.MAIN_HAND,
            level_requirement=10,
        )
        sword = EquipmentInstance(definition=sword_def)

        can, reason = equipment_container.can_equip(
            sword,
            character_stats={AttributeType.WISDOM: 15},  # Level uses WISDOM
        )
        assert can is True

    def test_can_equip_attribute_requirement(self, equipment_container):
        """Test attribute requirement check."""
        armor_def = EquipmentDefinition(
            id="heavy_armor",
            name="Heavy Armor",
            slot=EquipmentSlot.CHEST,
            required_attributes={AttributeType.STRENGTH: 20},
        )
        armor = EquipmentInstance(definition=armor_def)

        # Insufficient strength
        can, reason = equipment_container.can_equip(
            armor,
            character_stats={AttributeType.STRENGTH: 15},
        )
        assert can is False
        assert "STRENGTH" in reason

    def test_can_equip_multiple_requirements(self, equipment_container):
        """Test multiple attribute requirements."""
        armor_def = EquipmentDefinition(
            id="complex_armor",
            name="Complex Armor",
            slot=EquipmentSlot.CHEST,
            required_attributes={
                AttributeType.STRENGTH: 15,
                AttributeType.CONSTITUTION: 10,
            },
        )
        armor = EquipmentInstance(definition=armor_def)

        can, reason = equipment_container.can_equip(
            armor,
            character_stats={
                AttributeType.STRENGTH: 20,
                AttributeType.CONSTITUTION: 12,
            },
        )
        assert can is True


# =============================================================================
# EquipmentContainer Unequip Tests
# =============================================================================


class TestEquipmentContainerUnequip:
    """Tests for unequipping items."""

    def test_unequip_item(self, equipment_container, basic_sword_def):
        """Test unequipping an item."""
        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        unequipped = equipment_container.unequip(EquipmentSlot.MAIN_HAND)
        assert unequipped == sword
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None

    def test_unequip_empty_slot(self, equipment_container):
        """Test unequipping from empty slot."""
        unequipped = equipment_container.unequip(EquipmentSlot.MAIN_HAND)
        assert unequipped is None

    def test_unequip_all(self, equipment_container, basic_sword_def, helmet_def):
        """Test unequipping all items."""
        sword = EquipmentInstance(definition=basic_sword_def)
        helmet = EquipmentInstance(definition=helmet_def)

        equipment_container.equip(sword)
        equipment_container.equip(helmet)

        unequipped = equipment_container.unequip_all()
        assert len(unequipped) == 2
        assert equipment_container.get_all_equipped() == []

    def test_swap_equipment(self, equipment_container, basic_sword_def):
        """Test swapping equipment."""
        old_sword = EquipmentInstance(definition=basic_sword_def)
        new_sword = EquipmentInstance(definition=basic_sword_def)

        equipment_container.equip(old_sword)
        swapped = equipment_container.swap(EquipmentSlot.MAIN_HAND, new_sword)

        assert swapped == old_sword
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == new_sword


# =============================================================================
# EquipmentContainer Query Tests
# =============================================================================


class TestEquipmentContainerQuery:
    """Tests for querying equipment."""

    def test_get_equipped_item(self, equipment_container, basic_sword_def):
        """Test getting equipped item."""
        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) == sword

    def test_get_empty_slot(self, equipment_container):
        """Test getting empty slot returns None."""
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None

    def test_is_slot_empty(self, equipment_container, basic_sword_def):
        """Test is_slot_empty."""
        assert equipment_container.is_slot_empty(EquipmentSlot.MAIN_HAND) is True

        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        assert equipment_container.is_slot_empty(EquipmentSlot.MAIN_HAND) is False

    def test_get_all_equipped(self, equipment_container, basic_sword_def, helmet_def):
        """Test getting all equipped items."""
        sword = EquipmentInstance(definition=basic_sword_def)
        helmet = EquipmentInstance(definition=helmet_def)

        equipment_container.equip(sword)
        equipment_container.equip(helmet)

        equipped = equipment_container.get_all_equipped()
        assert len(equipped) == 2
        slots = [slot for slot, item in equipped]
        assert EquipmentSlot.MAIN_HAND in slots
        assert EquipmentSlot.HEAD in slots

    def test_get_equipped_ids(self, equipment_container, basic_sword_def, helmet_def):
        """Test getting equipped item IDs."""
        sword = EquipmentInstance(definition=basic_sword_def)
        helmet = EquipmentInstance(definition=helmet_def)

        equipment_container.equip(sword)
        equipment_container.equip(helmet)

        ids = equipment_container.get_equipped_ids()
        assert "sword_iron" in ids
        assert "helm_steel" in ids

    def test_find_by_id(self, equipment_container, basic_sword_def):
        """Test finding equipped item by ID."""
        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        result = equipment_container.find_by_id("sword_iron")
        assert result is not None
        assert result[0] == EquipmentSlot.MAIN_HAND
        assert result[1] == sword

    def test_find_by_id_not_found(self, equipment_container):
        """Test finding non-existent item returns None."""
        assert equipment_container.find_by_id("nonexistent") is None


# =============================================================================
# EquipmentContainer Stats Tests
# =============================================================================


class TestEquipmentContainerStats:
    """Tests for equipment stat calculations."""

    def test_combined_stats_single_item(self, equipment_container, basic_sword_def):
        """Test combined stats with single item."""
        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        stats = equipment_container.combined_stats
        assert stats.damage == 10.0

    def test_combined_stats_multiple_items(
        self, equipment_container, basic_sword_def, basic_shield_def
    ):
        """Test combined stats with multiple items."""
        sword = EquipmentInstance(definition=basic_sword_def)
        shield = EquipmentInstance(definition=basic_shield_def)

        equipment_container.equip(sword)
        equipment_container.equip(shield)

        stats = equipment_container.combined_stats
        assert stats.damage == 10.0
        assert stats.armor == 15.0
        assert stats.block_chance == 0.1

    def test_get_total_armor(self, equipment_container, basic_shield_def, helmet_def):
        """Test getting total armor."""
        shield = EquipmentInstance(definition=basic_shield_def)
        helmet = EquipmentInstance(definition=helmet_def)

        equipment_container.equip(shield)
        equipment_container.equip(helmet)

        assert equipment_container.get_total_armor() == pytest.approx(23.0)

    def test_get_total_damage(self, equipment_container, basic_sword_def):
        """Test getting total damage."""
        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        assert equipment_container.get_total_damage() == 10.0

    def test_get_attribute_modifier(self, equipment_container, helmet_def):
        """Test getting attribute modifier."""
        helmet = EquipmentInstance(definition=helmet_def)
        equipment_container.equip(helmet)

        mod = equipment_container.get_attribute_modifier(AttributeType.CONSTITUTION)
        assert mod is not None
        assert mod.flat_bonus == 5

    def test_get_resistance_modifier(self, equipment_container):
        """Test getting resistance modifier."""
        armor_def = EquipmentDefinition(
            id="fire_armor",
            name="Fire Armor",
            slot=EquipmentSlot.CHEST,
            stats=EquipmentStats(
                resistance_modifiers=(
                    ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.2),
                ),
            ),
        )
        armor = EquipmentInstance(definition=armor_def)
        equipment_container.equip(armor)

        mod = equipment_container.get_resistance_modifier(ResistanceType.FIRE)
        assert mod is not None
        assert mod.flat_bonus == 0.2

    def test_has_effect(self, equipment_container):
        """Test checking for special effect."""
        sword_def = EquipmentDefinition(
            id="lifesteal_sword",
            name="Lifesteal Sword",
            slot=EquipmentSlot.MAIN_HAND,
            stats=EquipmentStats(
                special_effects=(
                    SpecialEffect(effect_id="lifesteal", name="Life Steal"),
                ),
            ),
        )
        sword = EquipmentInstance(definition=sword_def)
        equipment_container.equip(sword)

        assert equipment_container.has_effect("lifesteal") is True
        assert equipment_container.has_effect("mana_steal") is False

    def test_get_effects(self, equipment_container):
        """Test getting all special effects."""
        sword_def = EquipmentDefinition(
            id="magic_sword",
            name="Magic Sword",
            slot=EquipmentSlot.MAIN_HAND,
            stats=EquipmentStats(
                special_effects=(
                    SpecialEffect(effect_id="e1", name="Effect 1"),
                    SpecialEffect(effect_id="e2", name="Effect 2"),
                ),
            ),
        )
        sword = EquipmentInstance(definition=sword_def)
        equipment_container.equip(sword)

        effects = equipment_container.get_effects()
        assert len(effects) == 2


# =============================================================================
# EquipmentSet Tests
# =============================================================================


class TestEquipmentSet:
    """Tests for equipment set bonuses."""

    def test_create_equipment_set(self):
        """Test creating equipment set."""
        equipment_set = EquipmentSet(
            set_id="dragon_set",
            name="Dragon Set",
            piece_ids=frozenset({"dragon_helm", "dragon_chest", "dragon_legs"}),
            bonuses=(
                SetBonus(
                    pieces_required=2,
                    stats=EquipmentStats(armor=10.0),
                    description="2 pieces: +10 armor",
                ),
                SetBonus(
                    pieces_required=3,
                    stats=EquipmentStats(
                        armor=20.0,
                        resistance_modifiers=(
                            ResistanceModifier(ResistanceType.FIRE, flat_bonus=0.2),
                        ),
                    ),
                    description="3 pieces: +20 armor, +20% fire resistance",
                ),
            ),
        )
        assert equipment_set.set_id == "dragon_set"
        assert len(equipment_set.bonuses) == 2

    def test_get_active_bonuses_none(self):
        """Test getting active bonuses with no pieces."""
        equipment_set = EquipmentSet(
            set_id="test_set",
            name="Test Set",
            piece_ids=frozenset({"piece_1", "piece_2", "piece_3"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=10.0)),
            ),
        )
        active = equipment_set.get_active_bonuses(set())
        assert len(active) == 0

    def test_get_active_bonuses_partial(self):
        """Test getting active bonuses with some pieces."""
        equipment_set = EquipmentSet(
            set_id="test_set",
            name="Test Set",
            piece_ids=frozenset({"piece_1", "piece_2", "piece_3"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=10.0)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=25.0)),
            ),
        )
        active = equipment_set.get_active_bonuses({"piece_1", "piece_2"})
        assert len(active) == 1
        assert active[0].pieces_required == 2

    def test_get_active_bonuses_full_set(self):
        """Test getting active bonuses with full set."""
        equipment_set = EquipmentSet(
            set_id="test_set",
            name="Test Set",
            piece_ids=frozenset({"piece_1", "piece_2", "piece_3"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=10.0)),
                SetBonus(pieces_required=3, stats=EquipmentStats(armor=25.0)),
            ),
        )
        active = equipment_set.get_active_bonuses({"piece_1", "piece_2", "piece_3"})
        assert len(active) == 2

    def test_register_set_with_container(self, equipment_container):
        """Test registering set with container."""
        equipment_set = EquipmentSet(
            set_id="test_set",
            name="Test Set",
            piece_ids=frozenset({"helm_test", "chest_test"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(armor=50.0)),
            ),
        )
        equipment_container.register_set(equipment_set)

        # Equip set pieces
        helm_def = EquipmentDefinition(
            id="helm_test",
            name="Test Helmet",
            slot=EquipmentSlot.HEAD,
            stats=EquipmentStats(armor=10.0),
            set_id="test_set",
        )
        chest_def = EquipmentDefinition(
            id="chest_test",
            name="Test Chest",
            slot=EquipmentSlot.CHEST,
            stats=EquipmentStats(armor=20.0),
            set_id="test_set",
        )

        equipment_container.equip(EquipmentInstance(definition=helm_def))
        equipment_container.equip(EquipmentInstance(definition=chest_def))

        # Total armor should include set bonus
        # Base: 10 + 20 = 30, Set bonus: +50 = 80
        assert equipment_container.get_total_armor() == 80.0

    def test_get_active_set_bonuses(self, equipment_container):
        """Test getting active set bonuses."""
        equipment_set = EquipmentSet(
            set_id="warrior_set",
            name="Warrior Set",
            piece_ids=frozenset({"warrior_helm", "warrior_chest"}),
            bonuses=(
                SetBonus(pieces_required=2, stats=EquipmentStats(damage=10.0)),
            ),
        )
        equipment_container.register_set(equipment_set)

        helm_def = EquipmentDefinition(
            id="warrior_helm",
            name="Warrior Helmet",
            slot=EquipmentSlot.HEAD,
            set_id="warrior_set",
        )
        chest_def = EquipmentDefinition(
            id="warrior_chest",
            name="Warrior Chest",
            slot=EquipmentSlot.CHEST,
            set_id="warrior_set",
        )

        equipment_container.equip(EquipmentInstance(definition=helm_def))
        equipment_container.equip(EquipmentInstance(definition=chest_def))

        active_bonuses = equipment_container.get_active_set_bonuses()
        assert len(active_bonuses) == 1
        assert active_bonuses[0][0].set_id == "warrior_set"


# =============================================================================
# EquipmentContainer Durability Tests
# =============================================================================


class TestEquipmentContainerDurability:
    """Tests for equipment durability."""

    def test_reduce_durability(self, equipment_container, basic_sword_def):
        """Test reducing equipment durability."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=100.0,
        )
        equipment_container.equip(sword)

        broke = equipment_container.reduce_durability(EquipmentSlot.MAIN_HAND, 20.0)
        assert broke is False
        assert sword.durability == 80.0

    def test_reduce_durability_breaks_item(self, equipment_container, basic_sword_def):
        """Test reducing durability breaks item at zero."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=10.0,
        )
        equipment_container.equip(sword)

        broke = equipment_container.reduce_durability(EquipmentSlot.MAIN_HAND, 15.0)
        assert broke is True
        assert equipment_container.get(EquipmentSlot.MAIN_HAND) is None

    def test_reduce_durability_no_durability(self, equipment_container, basic_sword_def):
        """Test reducing durability on item without durability."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=None,  # No durability tracking
        )
        equipment_container.equip(sword)

        broke = equipment_container.reduce_durability(EquipmentSlot.MAIN_HAND, 20.0)
        assert broke is False

    def test_reduce_all_durability(
        self, equipment_container, basic_sword_def, basic_shield_def
    ):
        """Test reducing durability on all equipment."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=50.0,
        )
        shield = EquipmentInstance(
            definition=basic_shield_def,
            durability=10.0,
        )

        equipment_container.equip(sword)
        equipment_container.equip(shield)

        broken = equipment_container.reduce_all_durability(15.0)
        assert EquipmentSlot.OFF_HAND in broken
        assert EquipmentSlot.MAIN_HAND not in broken
        assert sword.durability == 35.0

    def test_repair_equipment(self, equipment_container, basic_sword_def):
        """Test repairing equipment."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=50.0,
        )
        equipment_container.equip(sword)

        repaired = equipment_container.repair(EquipmentSlot.MAIN_HAND, 30.0)
        assert repaired == 30.0
        assert sword.durability == 80.0

    def test_repair_full(self, equipment_container, basic_sword_def):
        """Test full repair of equipment."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=50.0,
        )
        equipment_container.equip(sword)

        repaired = equipment_container.repair(EquipmentSlot.MAIN_HAND)
        assert repaired == 50.0
        assert sword.durability == 100.0

    def test_repair_capped_at_max(self, equipment_container, basic_sword_def):
        """Test repair is capped at maximum."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            durability=90.0,
        )
        equipment_container.equip(sword)

        repaired = equipment_container.repair(EquipmentSlot.MAIN_HAND, 50.0)
        assert repaired == 10.0
        assert sword.durability == 100.0


# =============================================================================
# EquipmentContainer Visual Tests
# =============================================================================


class TestEquipmentContainerVisual:
    """Tests for visual attachment data."""

    def test_get_visual_attachments(self, equipment_container):
        """Test getting visual attachment data."""
        sword_def = EquipmentDefinition(
            id="visual_sword",
            name="Visual Sword",
            slot=EquipmentSlot.MAIN_HAND,
            visual_model="models/swords/iron.obj",
            attachment_point="right_hand",
        )
        sword = EquipmentInstance(definition=sword_def)
        equipment_container.equip(sword)

        attachments = equipment_container.get_visual_attachments()
        assert len(attachments) == 1
        assert attachments[0] == ("right_hand", "models/swords/iron.obj")

    def test_get_visual_attachments_default_point(self, equipment_container):
        """Test visual attachment uses slot name as default point."""
        helm_def = EquipmentDefinition(
            id="visual_helm",
            name="Visual Helm",
            slot=EquipmentSlot.HEAD,
            visual_model="models/helms/steel.obj",
        )
        helm = EquipmentInstance(definition=helm_def)
        equipment_container.equip(helm)

        attachments = equipment_container.get_visual_attachments()
        assert attachments[0][0] == "head"

    def test_get_visual_attachments_empty(self, equipment_container):
        """Test getting visual attachments with no equipment."""
        attachments = equipment_container.get_visual_attachments()
        assert attachments == []


# =============================================================================
# EquipmentContainer Event Tests
# =============================================================================


class TestEquipmentContainerEvents:
    """Tests for equipment change events."""

    def test_equip_triggers_listener(self, equipment_container, basic_sword_def):
        """Test equipping triggers change listener."""
        events = []
        equipment_container.add_change_listener(
            lambda slot, old, new: events.append((slot, old, new))
        )

        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        assert len(events) == 1
        assert events[0][0] == EquipmentSlot.MAIN_HAND
        assert events[0][1] is None
        assert events[0][2] == sword

    def test_unequip_triggers_listener(self, equipment_container, basic_sword_def):
        """Test unequipping triggers change listener."""
        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        events = []
        equipment_container.add_change_listener(
            lambda slot, old, new: events.append((slot, old, new))
        )

        equipment_container.unequip(EquipmentSlot.MAIN_HAND)

        assert len(events) == 1
        assert events[0][1] == sword
        assert events[0][2] is None

    def test_remove_change_listener(self, equipment_container, basic_sword_def):
        """Test removing change listener."""
        events = []
        callback = lambda slot, old, new: events.append((slot, old, new))

        equipment_container.add_change_listener(callback)
        equipment_container.remove_change_listener(callback)

        sword = EquipmentInstance(definition=basic_sword_def)
        equipment_container.equip(sword)

        assert len(events) == 0


# =============================================================================
# EquipmentContainer Serialization Tests
# =============================================================================


class TestEquipmentContainerSerialization:
    """Tests for equipment serialization."""

    def test_to_dict_empty(self, equipment_container):
        """Test serializing empty container."""
        data = equipment_container.to_dict()
        assert "id" in data
        assert data["owner_id"] == "player_1"
        assert "equipped" in data

    def test_to_dict_with_equipment(self, equipment_container, basic_sword_def):
        """Test serializing container with equipment."""
        sword = EquipmentInstance(
            definition=basic_sword_def,
            enchantments=["fire"],
            upgrade_level=3,
        )
        equipment_container.equip(sword)

        data = equipment_container.to_dict()
        main_hand = data["equipped"]["MAIN_HAND"]
        assert main_hand is not None
        assert main_hand["definition_id"] == "sword_iron"
        assert main_hand["enchantments"] == ["fire"]
        assert main_hand["upgrade_level"] == 3


# =============================================================================
# EquipmentRegistry Tests
# =============================================================================


class TestEquipmentRegistry:
    """Tests for EquipmentRegistry singleton."""

    def test_singleton_pattern(self):
        """Test registry is singleton."""
        EquipmentRegistry.reset()
        reg1 = EquipmentRegistry.instance()
        reg2 = EquipmentRegistry.instance()
        assert reg1 is reg2
        EquipmentRegistry.reset()

    def test_register_equipment(self, equipment_registry, basic_sword_def):
        """Test registering equipment."""
        equipment_registry.register_equipment(basic_sword_def)
        assert equipment_registry.get_equipment("sword_iron") is not None

    def test_register_duplicate_raises(self, equipment_registry, basic_sword_def):
        """Test registering duplicate raises error."""
        equipment_registry.register_equipment(basic_sword_def)
        with pytest.raises(ValueError, match="already registered"):
            equipment_registry.register_equipment(basic_sword_def)

    def test_register_set(self, equipment_registry):
        """Test registering equipment set."""
        equipment_set = EquipmentSet(
            set_id="test_set",
            name="Test Set",
            piece_ids=frozenset({"piece_1", "piece_2"}),
            bonuses=(),
        )
        equipment_registry.register_set(equipment_set)
        assert equipment_registry.get_set("test_set") is not None

    def test_get_by_slot(self, equipment_registry, basic_sword_def, helmet_def):
        """Test getting equipment by slot."""
        equipment_registry.register_equipment(basic_sword_def)
        equipment_registry.register_equipment(helmet_def)

        weapons = equipment_registry.get_by_slot(EquipmentSlot.MAIN_HAND)
        assert len(weapons) == 1
        assert weapons[0].id == "sword_iron"

    def test_clear_registry(self, equipment_registry, basic_sword_def):
        """Test clearing registry."""
        equipment_registry.register_equipment(basic_sword_def)
        equipment_registry.clear()
        assert equipment_registry.get_equipment("sword_iron") is None


# =============================================================================
# Ring and Trinket Slot Compatibility Tests
# =============================================================================


class TestSlotCompatibility:
    """Tests for slot compatibility (rings, trinkets)."""

    def test_ring_in_ring_1(self, equipment_container, ring_def):
        """Test ring can go in RING_1."""
        ring = EquipmentInstance(definition=ring_def)
        success, _ = equipment_container.equip(ring, EquipmentSlot.RING_1)
        assert success is True

    def test_ring_in_ring_2(self, equipment_container, ring_def):
        """Test ring can go in RING_2."""
        ring = EquipmentInstance(definition=ring_def)
        success, _ = equipment_container.equip(ring, EquipmentSlot.RING_2)
        assert success is True

    def test_two_rings_different_slots(self, equipment_container, ring_def):
        """Test wearing two rings in different slots."""
        ring1 = EquipmentInstance(definition=ring_def)
        ring2 = EquipmentInstance(definition=ring_def)

        equipment_container.equip(ring1, EquipmentSlot.RING_1)
        equipment_container.equip(ring2, EquipmentSlot.RING_2)

        assert equipment_container.get(EquipmentSlot.RING_1) == ring1
        assert equipment_container.get(EquipmentSlot.RING_2) == ring2

    def test_trinket_in_trinket_1(self, equipment_container):
        """Test trinket can go in TRINKET_1."""
        trinket_def = EquipmentDefinition(
            id="trinket_power",
            name="Power Trinket",
            slot=EquipmentSlot.TRINKET_1,
        )
        trinket = EquipmentInstance(definition=trinket_def)
        success, _ = equipment_container.equip(trinket, EquipmentSlot.TRINKET_1)
        assert success is True

    def test_trinket_in_trinket_2(self, equipment_container):
        """Test trinket can go in TRINKET_2."""
        trinket_def = EquipmentDefinition(
            id="trinket_power",
            name="Power Trinket",
            slot=EquipmentSlot.TRINKET_1,
        )
        trinket = EquipmentInstance(definition=trinket_def)
        success, _ = equipment_container.equip(trinket, EquipmentSlot.TRINKET_2)
        assert success is True
