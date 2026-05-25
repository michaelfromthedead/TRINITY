"""
Comprehensive tests for the Inventory Container System.

Tests cover:
- Inventory creation with capacity
- Add/remove items
- Stack management (stackable items)
- Inventory queries (find, count, has)
- Inventory sorting
- Inventory filters
- Item transfer between inventories
- Overflow handling
- Weight/capacity limits
- Inventory persistence
"""

import pytest
from uuid import UUID, uuid4

from engine.gameplay.economy.constants import (
    ContainerType,
    DEFAULT_CONTAINER_SLOTS,
    DEFAULT_STACK_LIMITS,
    DEFAULT_WEIGHT_LIMITS,
    EconomyEvent,
    ItemType,
    MAX_STACK_SIZE,
    Rarity,
    STACKABLE_TYPES,
)
from engine.gameplay.economy.inventory import (
    ItemDefinition,
    ItemInstance,
    InventorySlot,
    InventoryContainer,
    InventoryEvent,
    ItemRegistry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_item_def():
    """Create a basic non-stackable item definition."""
    return ItemDefinition(
        id="sword_iron",
        name="Iron Sword",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.COMMON,
        weight=2.5,
        base_value=100,
    )


@pytest.fixture
def stackable_item_def():
    """Create a stackable item definition."""
    return ItemDefinition(
        id="potion_health",
        name="Health Potion",
        item_type=ItemType.CONSUMABLE,
        rarity=Rarity.COMMON,
        max_stack=99,
        weight=0.1,
        base_value=25,
    )


@pytest.fixture
def material_item_def():
    """Create a material item definition."""
    return ItemDefinition(
        id="ore_iron",
        name="Iron Ore",
        item_type=ItemType.MATERIAL,
        rarity=Rarity.COMMON,
        max_stack=999,
        weight=0.5,
        base_value=5,
    )


@pytest.fixture
def player_inventory():
    """Create a player inventory container."""
    return InventoryContainer(
        container_type=ContainerType.PLAYER_INVENTORY,
        owner_id="player_1",
    )


@pytest.fixture
def limited_inventory():
    """Create a small inventory with 5 slots."""
    return InventoryContainer(
        container_type=ContainerType.PLAYER_INVENTORY,
        slot_count=5,
        weight_limit=50.0,
        owner_id="player_1",
    )


@pytest.fixture
def item_registry():
    """Create and populate an item registry."""
    ItemRegistry.reset()
    registry = ItemRegistry.instance()
    registry.register(ItemDefinition(
        id="sword_iron",
        name="Iron Sword",
        item_type=ItemType.EQUIPMENT,
        weight=2.5,
        base_value=100,
    ))
    registry.register(ItemDefinition(
        id="potion_health",
        name="Health Potion",
        item_type=ItemType.CONSUMABLE,
        max_stack=99,
        weight=0.1,
        base_value=25,
    ))
    registry.register(ItemDefinition(
        id="ore_iron",
        name="Iron Ore",
        item_type=ItemType.MATERIAL,
        max_stack=999,
        weight=0.5,
        base_value=5,
    ))
    yield registry
    ItemRegistry.reset()


# =============================================================================
# ItemDefinition Tests
# =============================================================================


class TestItemDefinition:
    """Tests for ItemDefinition class."""

    def test_create_basic_definition(self):
        """Test creating a basic item definition."""
        item_def = ItemDefinition(
            id="test_item",
            name="Test Item",
            item_type=ItemType.EQUIPMENT,
        )
        assert item_def.id == "test_item"
        assert item_def.name == "Test Item"
        assert item_def.item_type == ItemType.EQUIPMENT

    def test_definition_default_values(self):
        """Test default values are set correctly."""
        item_def = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.EQUIPMENT,
        )
        assert item_def.rarity == Rarity.COMMON
        assert item_def.max_stack == 1
        assert item_def.weight == 0.0
        assert item_def.base_value == 0
        assert item_def.level_requirement == 1

    def test_definition_empty_id_raises(self):
        """Test that empty id raises ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            ItemDefinition(id="", name="Test", item_type=ItemType.EQUIPMENT)

    def test_definition_empty_name_raises(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ItemDefinition(id="test", name="", item_type=ItemType.EQUIPMENT)

    def test_stackable_item_type(self):
        """Test stackable item types."""
        consumable = ItemDefinition(
            id="pot",
            name="Potion",
            item_type=ItemType.CONSUMABLE,
            max_stack=99,
        )
        assert consumable.is_stackable is True

    def test_non_stackable_item_type(self):
        """Test non-stackable item types."""
        equipment = ItemDefinition(
            id="sword",
            name="Sword",
            item_type=ItemType.EQUIPMENT,
        )
        assert equipment.is_stackable is False

    def test_auto_stack_limit_consumable(self):
        """Test auto stack limit for consumable."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.CONSUMABLE,
            max_stack=-1,  # Will be auto-set
        )
        assert item.max_stack == DEFAULT_STACK_LIMITS[ItemType.CONSUMABLE]

    def test_auto_stack_limit_material(self):
        """Test auto stack limit for material."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            max_stack=0,  # Will be auto-set
        )
        assert item.max_stack == DEFAULT_STACK_LIMITS[ItemType.MATERIAL]

    def test_stack_limit_clamped_to_max(self):
        """Test stack limit clamped to maximum."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            max_stack=MAX_STACK_SIZE + 1000,
        )
        assert item.max_stack == MAX_STACK_SIZE

    def test_negative_weight_normalized(self):
        """Test negative weight normalized to zero."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.JUNK,
            weight=-5.0,
        )
        assert item.weight == 0.0

    def test_negative_value_normalized(self):
        """Test negative value normalized to zero."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.JUNK,
            base_value=-100,
        )
        assert item.base_value == 0

    def test_level_requirement_minimum(self):
        """Test level requirement minimum is 1."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.EQUIPMENT,
            level_requirement=0,
        )
        assert item.level_requirement == 1

    def test_definition_equality(self):
        """Test definition equality by id."""
        def1 = ItemDefinition(id="test", name="Test 1", item_type=ItemType.JUNK)
        def2 = ItemDefinition(id="test", name="Test 2", item_type=ItemType.JUNK)
        assert def1 == def2

    def test_definition_hash(self):
        """Test definition hash based on id."""
        def1 = ItemDefinition(id="test", name="Test 1", item_type=ItemType.JUNK)
        def2 = ItemDefinition(id="test", name="Test 2", item_type=ItemType.JUNK)
        assert hash(def1) == hash(def2)

    def test_definition_with_metadata(self):
        """Test definition with custom metadata."""
        item = ItemDefinition(
            id="magic_sword",
            name="Magic Sword",
            item_type=ItemType.EQUIPMENT,
            metadata={"enchantment": "fire", "damage_bonus": 10},
        )
        assert item.metadata["enchantment"] == "fire"
        assert item.metadata["damage_bonus"] == 10

    def test_definition_with_flags(self):
        """Test definition with flags."""
        item = ItemDefinition(
            id="quest_item",
            name="Quest Item",
            item_type=ItemType.QUEST,
            flags=frozenset({"no_drop", "no_trade", "unique"}),
        )
        assert "no_drop" in item.flags
        assert "unique" in item.flags


# =============================================================================
# ItemInstance Tests
# =============================================================================


class TestItemInstance:
    """Tests for ItemInstance class."""

    def test_create_instance(self, basic_item_def):
        """Test creating an item instance."""
        instance = ItemInstance(definition=basic_item_def, quantity=1)
        assert instance.definition == basic_item_def
        assert instance.quantity == 1
        assert isinstance(instance.instance_id, UUID)

    def test_instance_unique_id(self, basic_item_def):
        """Test each instance has unique ID."""
        inst1 = ItemInstance(definition=basic_item_def)
        inst2 = ItemInstance(definition=basic_item_def)
        assert inst1.instance_id != inst2.instance_id

    def test_instance_zero_quantity_raises(self, basic_item_def):
        """Test zero quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            ItemInstance(definition=basic_item_def, quantity=0)

    def test_instance_negative_quantity_raises(self, basic_item_def):
        """Test negative quantity raises error."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            ItemInstance(definition=basic_item_def, quantity=-1)

    def test_instance_exceeds_max_stack_raises(self, stackable_item_def):
        """Test exceeding max stack raises error."""
        with pytest.raises(ValueError, match="exceeds max stack"):
            ItemInstance(definition=stackable_item_def, quantity=100)

    def test_item_id_property(self, basic_item_def):
        """Test item_id property returns definition id."""
        instance = ItemInstance(definition=basic_item_def)
        assert instance.item_id == "sword_iron"

    def test_total_weight_single(self, basic_item_def):
        """Test total weight for single item."""
        instance = ItemInstance(definition=basic_item_def, quantity=1)
        assert instance.total_weight == 2.5

    def test_total_weight_stack(self, stackable_item_def):
        """Test total weight for stacked items."""
        instance = ItemInstance(definition=stackable_item_def, quantity=10)
        assert instance.total_weight == pytest.approx(1.0)  # 10 * 0.1

    def test_total_value_single(self, basic_item_def):
        """Test total value for single item."""
        instance = ItemInstance(definition=basic_item_def, quantity=1)
        assert instance.total_value == 100

    def test_total_value_stack(self, stackable_item_def):
        """Test total value for stacked items."""
        instance = ItemInstance(definition=stackable_item_def, quantity=10)
        assert instance.total_value == 250  # 10 * 25

    def test_can_add_more(self, stackable_item_def):
        """Test can_add_more property."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        assert instance.can_add_more is True

    def test_cannot_add_more_at_max(self, stackable_item_def):
        """Test can_add_more is False when at max stack."""
        instance = ItemInstance(definition=stackable_item_def, quantity=99)
        assert instance.can_add_more is False

    def test_space_remaining(self, stackable_item_def):
        """Test space_remaining property."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        assert instance.space_remaining == 49

    def test_can_stack_with_same_item(self, stackable_item_def):
        """Test can_stack_with for identical items."""
        inst1 = ItemInstance(definition=stackable_item_def, quantity=10)
        inst2 = ItemInstance(definition=stackable_item_def, quantity=5)
        assert inst1.can_stack_with(inst2) is True

    def test_cannot_stack_different_items(self, stackable_item_def):
        """Test cannot stack different item types."""
        other_def = ItemDefinition(
            id="potion_mana",
            name="Mana Potion",
            item_type=ItemType.CONSUMABLE,
            max_stack=99,
        )
        inst1 = ItemInstance(definition=stackable_item_def, quantity=10)
        inst2 = ItemInstance(definition=other_def, quantity=5)
        assert inst1.can_stack_with(inst2) is False

    def test_cannot_stack_non_stackable(self, basic_item_def):
        """Test non-stackable items cannot stack."""
        inst1 = ItemInstance(definition=basic_item_def)
        inst2 = ItemInstance(definition=basic_item_def)
        assert inst1.can_stack_with(inst2) is False

    def test_cannot_stack_different_binding(self, stackable_item_def):
        """Test cannot stack items bound to different entities."""
        inst1 = ItemInstance(definition=stackable_item_def, quantity=10, bound_to="player1")
        inst2 = ItemInstance(definition=stackable_item_def, quantity=5, bound_to="player2")
        assert inst1.can_stack_with(inst2) is False

    def test_can_stack_same_binding(self, stackable_item_def):
        """Test can stack items with same binding."""
        inst1 = ItemInstance(definition=stackable_item_def, quantity=10, bound_to="player1")
        inst2 = ItemInstance(definition=stackable_item_def, quantity=5, bound_to="player1")
        assert inst1.can_stack_with(inst2) is True

    def test_cannot_stack_different_custom_data(self, stackable_item_def):
        """Test cannot stack items with different custom data."""
        inst1 = ItemInstance(
            definition=stackable_item_def,
            quantity=10,
            custom_data={"enchant": "fire"},
        )
        inst2 = ItemInstance(
            definition=stackable_item_def,
            quantity=5,
            custom_data={"enchant": "ice"},
        )
        assert inst1.can_stack_with(inst2) is False

    def test_can_stack_same_custom_data(self, stackable_item_def):
        """Test can stack items with same custom data."""
        inst1 = ItemInstance(
            definition=stackable_item_def,
            quantity=10,
            custom_data={"enchant": "fire"},
        )
        inst2 = ItemInstance(
            definition=stackable_item_def,
            quantity=5,
            custom_data={"enchant": "fire"},
        )
        assert inst1.can_stack_with(inst2) is True

    def test_split_stack(self, stackable_item_def):
        """Test splitting a stack."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        split = instance.split(20)
        assert instance.quantity == 30
        assert split.quantity == 20
        assert split.definition == instance.definition

    def test_split_zero_raises(self, stackable_item_def):
        """Test splitting zero raises error."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        with pytest.raises(ValueError, match="must be positive"):
            instance.split(0)

    def test_split_negative_raises(self, stackable_item_def):
        """Test splitting negative raises error."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        with pytest.raises(ValueError, match="must be positive"):
            instance.split(-5)

    def test_split_all_raises(self, stackable_item_def):
        """Test splitting entire stack raises error."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        with pytest.raises(ValueError, match="less than total"):
            instance.split(50)

    def test_split_more_than_total_raises(self, stackable_item_def):
        """Test splitting more than total raises error."""
        instance = ItemInstance(definition=stackable_item_def, quantity=50)
        with pytest.raises(ValueError, match="less than total"):
            instance.split(60)

    def test_split_preserves_binding(self, stackable_item_def):
        """Test split preserves bound_to."""
        instance = ItemInstance(
            definition=stackable_item_def,
            quantity=50,
            bound_to="player1",
        )
        split = instance.split(20)
        assert split.bound_to == "player1"

    def test_split_copies_custom_data(self, stackable_item_def):
        """Test split copies custom data."""
        instance = ItemInstance(
            definition=stackable_item_def,
            quantity=50,
            custom_data={"quality": "high"},
        )
        split = instance.split(20)
        assert split.custom_data == {"quality": "high"}
        # Ensure it's a copy
        split.custom_data["quality"] = "low"
        assert instance.custom_data["quality"] == "high"

    def test_merge_from_success(self, stackable_item_def):
        """Test successful merge."""
        inst1 = ItemInstance(definition=stackable_item_def, quantity=50)
        inst2 = ItemInstance(definition=stackable_item_def, quantity=30)
        merged = inst1.merge_from(inst2)
        assert merged == 30
        assert inst1.quantity == 80
        assert inst2.quantity == 0

    def test_merge_from_partial(self, stackable_item_def):
        """Test partial merge when target stack fills."""
        inst1 = ItemInstance(definition=stackable_item_def, quantity=90)
        inst2 = ItemInstance(definition=stackable_item_def, quantity=20)
        merged = inst1.merge_from(inst2)
        assert merged == 9  # Only 9 can fit (90 + 9 = 99)
        assert inst1.quantity == 99
        assert inst2.quantity == 11

    def test_merge_from_incompatible_raises(self, stackable_item_def):
        """Test merge from incompatible item raises error."""
        other_def = ItemDefinition(
            id="potion_mana",
            name="Mana Potion",
            item_type=ItemType.CONSUMABLE,
            max_stack=99,
        )
        inst1 = ItemInstance(definition=stackable_item_def, quantity=50)
        inst2 = ItemInstance(definition=other_def, quantity=20)
        with pytest.raises(ValueError, match="cannot be stacked"):
            inst1.merge_from(inst2)

    def test_clone_instance(self, stackable_item_def):
        """Test cloning an instance."""
        instance = ItemInstance(
            definition=stackable_item_def,
            quantity=25,
            bound_to="player1",
            durability=80.0,
            custom_data={"enchant": "fire"},
        )
        clone = instance.clone()
        assert clone.quantity == 25
        assert clone.bound_to == "player1"
        assert clone.durability == 80.0
        assert clone.custom_data == {"enchant": "fire"}
        assert clone.instance_id != instance.instance_id

    def test_clone_independent_custom_data(self, stackable_item_def):
        """Test cloned custom data is independent."""
        instance = ItemInstance(
            definition=stackable_item_def,
            quantity=10,
            custom_data={"value": 100},
        )
        clone = instance.clone()
        clone.custom_data["value"] = 200
        assert instance.custom_data["value"] == 100


# =============================================================================
# InventorySlot Tests
# =============================================================================


class TestInventorySlot:
    """Tests for InventorySlot class."""

    def test_create_empty_slot(self):
        """Test creating an empty slot."""
        slot = InventorySlot(index=0)
        assert slot.index == 0
        assert slot.item is None
        assert slot.locked is False
        assert slot.filter_type is None

    def test_is_empty_true(self):
        """Test is_empty for empty slot."""
        slot = InventorySlot(index=0)
        assert slot.is_empty is True

    def test_is_empty_false(self, basic_item_def):
        """Test is_empty for occupied slot."""
        item = ItemInstance(definition=basic_item_def)
        slot = InventorySlot(index=0, item=item)
        assert slot.is_empty is False

    def test_is_available_empty_unlocked(self):
        """Test is_available for empty unlocked slot."""
        slot = InventorySlot(index=0)
        assert slot.is_available is True

    def test_is_available_locked(self):
        """Test is_available for locked slot."""
        slot = InventorySlot(index=0, locked=True)
        assert slot.is_available is False

    def test_is_available_occupied(self, basic_item_def):
        """Test is_available for occupied slot."""
        item = ItemInstance(definition=basic_item_def)
        slot = InventorySlot(index=0, item=item)
        assert slot.is_available is False

    def test_accepts_no_filter(self, basic_item_def):
        """Test accepts when no filter set."""
        item = ItemInstance(definition=basic_item_def)
        slot = InventorySlot(index=0)
        assert slot.accepts(item) is True

    def test_accepts_matching_filter(self, basic_item_def):
        """Test accepts when filter matches."""
        item = ItemInstance(definition=basic_item_def)
        slot = InventorySlot(index=0, filter_type=ItemType.EQUIPMENT)
        assert slot.accepts(item) is True

    def test_rejects_non_matching_filter(self, basic_item_def):
        """Test rejects when filter doesn't match."""
        item = ItemInstance(definition=basic_item_def)
        slot = InventorySlot(index=0, filter_type=ItemType.CONSUMABLE)
        assert slot.accepts(item) is False

    def test_rejects_locked(self, basic_item_def):
        """Test rejects when slot is locked."""
        item = ItemInstance(definition=basic_item_def)
        slot = InventorySlot(index=0, locked=True)
        assert slot.accepts(item) is False


# =============================================================================
# InventoryContainer Creation Tests
# =============================================================================


class TestInventoryContainerCreation:
    """Tests for InventoryContainer creation and initialization."""

    def test_create_default_player_inventory(self):
        """Test creating player inventory with defaults."""
        inv = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert inv.container_type == ContainerType.PLAYER_INVENTORY
        assert inv.slot_count == DEFAULT_CONTAINER_SLOTS[ContainerType.PLAYER_INVENTORY]
        assert inv.weight_limit == DEFAULT_WEIGHT_LIMITS[ContainerType.PLAYER_INVENTORY]

    def test_create_with_custom_slots(self):
        """Test creating inventory with custom slot count."""
        inv = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=20,
        )
        assert inv.slot_count == 20

    def test_create_with_custom_weight_limit(self):
        """Test creating inventory with custom weight limit."""
        inv = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            weight_limit=200.0,
        )
        assert inv.weight_limit == 200.0

    def test_create_with_owner(self):
        """Test creating inventory with owner ID."""
        inv = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            owner_id="player_123",
        )
        assert inv.owner_id == "player_123"

    def test_create_with_custom_id(self):
        """Test creating inventory with custom container ID."""
        custom_id = uuid4()
        inv = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            container_id=custom_id,
        )
        assert inv.id == custom_id

    def test_unique_id_generation(self):
        """Test unique ID generation."""
        inv1 = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        inv2 = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert inv1.id != inv2.id

    def test_initial_weight_zero(self):
        """Test initial weight is zero."""
        inv = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert inv.current_weight == 0.0

    def test_is_empty_initially(self):
        """Test inventory is empty initially."""
        inv = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert inv.is_empty is True

    def test_is_not_full_initially(self):
        """Test inventory is not full initially."""
        inv = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert inv.is_full is False

    def test_empty_slot_count_equals_total(self):
        """Test empty slot count equals total initially."""
        inv = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            slot_count=30,
        )
        assert inv.empty_slot_count == 30
        assert inv.used_slot_count == 0


# =============================================================================
# InventoryContainer Add Tests
# =============================================================================


class TestInventoryContainerAdd:
    """Tests for adding items to inventory."""

    def test_add_single_item(self, player_inventory, basic_item_def):
        """Test adding a single item."""
        item = ItemInstance(definition=basic_item_def)
        success, qty = player_inventory.add(item)
        assert success is True
        assert qty == 1
        assert player_inventory.used_slot_count == 1

    def test_add_stackable_item(self, player_inventory, stackable_item_def):
        """Test adding stackable items."""
        item = ItemInstance(definition=stackable_item_def, quantity=25)
        success, qty = player_inventory.add(item)
        assert success is True
        assert qty == 25
        assert player_inventory.count_item("potion_health") == 25

    def test_add_to_existing_stack(self, player_inventory, stackable_item_def):
        """Test adding to existing stack."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=25)
        item2 = ItemInstance(definition=stackable_item_def, quantity=30)
        player_inventory.add(item1)
        success, qty = player_inventory.add(item2)
        assert success is True
        assert qty == 30
        assert player_inventory.count_item("potion_health") == 55
        assert player_inventory.used_slot_count == 1  # Merged into one slot

    def test_add_to_specific_slot(self, player_inventory, basic_item_def):
        """Test adding to a specific slot."""
        item = ItemInstance(definition=basic_item_def)
        success, qty = player_inventory.add(item, target_slot=5)
        assert success is True
        assert player_inventory.get_item(5) == item

    def test_add_to_occupied_slot_fails(self, player_inventory, basic_item_def):
        """Test adding to occupied slot fails for non-stackable."""
        item1 = ItemInstance(definition=basic_item_def)
        item2 = ItemInstance(definition=basic_item_def)
        player_inventory.add(item1, target_slot=0)
        success, qty = player_inventory.add(item2, target_slot=0)
        assert success is False
        assert qty == 0

    def test_add_stack_to_occupied_slot(self, player_inventory, stackable_item_def):
        """Test adding stackable to occupied slot with same item."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=30)
        item2 = ItemInstance(definition=stackable_item_def, quantity=20)
        player_inventory.add(item1, target_slot=0)
        success, qty = player_inventory.add(item2, target_slot=0)
        assert success is True
        assert qty == 20
        assert player_inventory.get_item(0).quantity == 50

    def test_add_updates_weight(self, player_inventory, basic_item_def):
        """Test adding item updates current weight."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item)
        assert player_inventory.current_weight == 2.5

    def test_add_weight_accumulates(self, player_inventory, basic_item_def):
        """Test weight accumulates with multiple items."""
        item1 = ItemInstance(definition=basic_item_def)
        item2 = ItemInstance(definition=basic_item_def)
        player_inventory.add(item1)
        player_inventory.add(item2)
        assert player_inventory.current_weight == 5.0

    def test_add_zero_quantity_fails(self, player_inventory, stackable_item_def):
        """Test adding zero quantity fails."""
        item = ItemInstance(definition=stackable_item_def, quantity=10)
        item.quantity = 0  # Force zero
        success, qty = player_inventory.add(item)
        assert success is False
        assert qty == 0

    def test_add_without_auto_stack(self, player_inventory, stackable_item_def):
        """Test adding without auto-stacking."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=25)
        item2 = ItemInstance(definition=stackable_item_def, quantity=30)
        player_inventory.add(item1)
        success, qty = player_inventory.add(item2, auto_stack=False)
        assert success is True
        assert player_inventory.used_slot_count == 2  # Two separate stacks

    def test_add_overflow_to_multiple_slots(self, player_inventory, stackable_item_def):
        """Test adding more than max stack creates multiple slots."""
        # First fill one stack
        item1 = ItemInstance(definition=stackable_item_def, quantity=90)
        player_inventory.add(item1)
        # Add more than remaining stack space
        item2 = ItemInstance(definition=stackable_item_def, quantity=30)
        success, qty = player_inventory.add(item2)
        assert success is True
        assert qty == 30
        # Should have created two stacks (99 + 21)
        assert player_inventory.count_item("potion_health") == 120

    def test_add_to_full_inventory_fails(self, limited_inventory, basic_item_def):
        """Test adding to full inventory fails."""
        # Fill all 5 slots
        for _ in range(5):
            limited_inventory.add(ItemInstance(definition=basic_item_def))

        new_item = ItemInstance(definition=basic_item_def)
        success, qty = limited_inventory.add(new_item)
        assert success is False
        assert qty == 0

    def test_add_exceeds_weight_limit_fails(self, limited_inventory, basic_item_def):
        """Test adding item that exceeds weight limit fails."""
        # Limited inventory has 50.0 weight limit
        # Item weighs 2.5, so we can fit 20
        for _ in range(20):
            limited_inventory.add(ItemInstance(definition=basic_item_def))

        # Next item should fail due to weight
        item = ItemInstance(definition=basic_item_def)
        success, qty = limited_inventory.add(item)
        assert success is False
        assert qty == 0

    def test_can_add_check_weight(self, limited_inventory, basic_item_def):
        """Test can_add checks weight limit."""
        # Fill to near weight limit
        for _ in range(19):
            limited_inventory.add(ItemInstance(definition=basic_item_def))

        # One more should fit
        item = ItemInstance(definition=basic_item_def)
        assert limited_inventory.can_add(item) is True
        limited_inventory.add(item)

        # Now should not fit
        assert limited_inventory.can_add(ItemInstance(definition=basic_item_def)) is False

    def test_add_definition_creates_instance(self, player_inventory, stackable_item_def):
        """Test add_definition creates and adds item."""
        success, qty = player_inventory.add_definition(stackable_item_def, quantity=50)
        assert success is True
        assert qty == 50
        assert player_inventory.count_item("potion_health") == 50


# =============================================================================
# InventoryContainer Remove Tests
# =============================================================================


class TestInventoryContainerRemove:
    """Tests for removing items from inventory."""

    def test_remove_at_entire_stack(self, player_inventory, basic_item_def):
        """Test removing entire item from slot."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item)
        removed = player_inventory.remove_at(0)
        assert removed == item
        assert player_inventory.is_empty is True

    def test_remove_at_partial_stack(self, player_inventory, stackable_item_def):
        """Test removing partial stack."""
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item)
        removed = player_inventory.remove_at(0, quantity=20)
        assert removed.quantity == 20
        assert player_inventory.get_item(0).quantity == 30

    def test_remove_at_empty_slot(self, player_inventory):
        """Test removing from empty slot returns None."""
        removed = player_inventory.remove_at(0)
        assert removed is None

    def test_remove_at_locked_slot(self, player_inventory, basic_item_def):
        """Test removing from locked slot returns None."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=0)
        player_inventory.lock_slot(0)
        removed = player_inventory.remove_at(0)
        assert removed is None

    def test_remove_at_zero_quantity(self, player_inventory, stackable_item_def):
        """Test removing zero quantity returns None."""
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item)
        removed = player_inventory.remove_at(0, quantity=0)
        assert removed is None

    def test_remove_updates_weight(self, player_inventory, basic_item_def):
        """Test removing updates current weight."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item)
        player_inventory.remove_at(0)
        assert player_inventory.current_weight == 0.0

    def test_remove_item_by_id(self, player_inventory, stackable_item_def):
        """Test removing item by definition ID."""
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item)
        removed = player_inventory.remove_item("potion_health", 30)
        assert removed == 30
        assert player_inventory.count_item("potion_health") == 20

    def test_remove_item_from_multiple_stacks(self, player_inventory, stackable_item_def):
        """Test removing from multiple stacks."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=99)
        item2 = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item1)
        player_inventory.add(item2, auto_stack=False)

        removed = player_inventory.remove_item("potion_health", 120)
        assert removed == 120
        assert player_inventory.count_item("potion_health") == 29

    def test_remove_item_not_found(self, player_inventory):
        """Test removing non-existent item returns 0."""
        removed = player_inventory.remove_item("nonexistent", 10)
        assert removed == 0

    def test_remove_item_less_than_requested(self, player_inventory, stackable_item_def):
        """Test removing more than available returns actual removed."""
        item = ItemInstance(definition=stackable_item_def, quantity=30)
        player_inventory.add(item)
        removed = player_inventory.remove_item("potion_health", 50)
        assert removed == 30
        assert player_inventory.count_item("potion_health") == 0

    def test_clear_removes_all(self, player_inventory, basic_item_def, stackable_item_def):
        """Test clear removes all items."""
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.add(ItemInstance(definition=stackable_item_def, quantity=50))

        removed = player_inventory.clear()
        assert len(removed) == 2
        assert player_inventory.is_empty is True
        assert player_inventory.current_weight == 0.0


# =============================================================================
# InventoryContainer Query Tests
# =============================================================================


class TestInventoryContainerQuery:
    """Tests for querying inventory contents."""

    def test_get_slot_valid_index(self, player_inventory):
        """Test getting slot by valid index."""
        slot = player_inventory.get_slot(0)
        assert slot.index == 0

    def test_get_slot_invalid_index(self, player_inventory):
        """Test getting slot by invalid index raises."""
        with pytest.raises(IndexError):
            player_inventory.get_slot(1000)

    def test_get_slot_negative_index(self, player_inventory):
        """Test getting slot by negative index raises."""
        with pytest.raises(IndexError):
            player_inventory.get_slot(-1)

    def test_get_item_valid(self, player_inventory, basic_item_def):
        """Test getting item from valid slot."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=5)
        assert player_inventory.get_item(5) == item

    def test_get_item_empty_slot(self, player_inventory):
        """Test getting item from empty slot returns None."""
        assert player_inventory.get_item(0) is None

    def test_find_item_exists(self, player_inventory, basic_item_def):
        """Test finding existing item."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=5)
        result = player_inventory.find_item("sword_iron")
        assert result is not None
        assert result[0] == 5
        assert result[1] == item

    def test_find_item_not_exists(self, player_inventory):
        """Test finding non-existent item returns None."""
        assert player_inventory.find_item("nonexistent") is None

    def test_find_all_items(self, player_inventory, stackable_item_def):
        """Test finding all stacks of an item."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=50)
        item2 = ItemInstance(definition=stackable_item_def, quantity=30)
        player_inventory.add(item1)
        player_inventory.add(item2, auto_stack=False)

        results = player_inventory.find_all_items("potion_health")
        assert len(results) == 2

    def test_count_item_single_stack(self, player_inventory, stackable_item_def):
        """Test counting items in single stack."""
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item)
        assert player_inventory.count_item("potion_health") == 50

    def test_count_item_multiple_stacks(self, player_inventory, stackable_item_def):
        """Test counting items across multiple stacks."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=99)
        item2 = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item1)
        player_inventory.add(item2, auto_stack=False)
        assert player_inventory.count_item("potion_health") == 149

    def test_count_item_not_found(self, player_inventory):
        """Test counting non-existent item returns 0."""
        assert player_inventory.count_item("nonexistent") == 0

    def test_find_empty_slot(self, player_inventory):
        """Test finding empty slot."""
        slot_idx = player_inventory.find_empty_slot()
        assert slot_idx == 0  # First slot should be empty

    def test_find_empty_slot_none_available(self, limited_inventory, basic_item_def):
        """Test finding empty slot when none available."""
        for _ in range(5):
            limited_inventory.add(ItemInstance(definition=basic_item_def))
        assert limited_inventory.find_empty_slot() is None

    def test_find_stackable_slot(self, player_inventory, stackable_item_def):
        """Test finding stackable slot."""
        existing = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(existing, target_slot=3)

        new_item = ItemInstance(definition=stackable_item_def, quantity=10)
        slot_idx = player_inventory.find_stackable_slot(new_item)
        assert slot_idx == 3

    def test_find_stackable_slot_none_available(self, player_inventory, stackable_item_def):
        """Test finding stackable slot when stack is full."""
        existing = ItemInstance(definition=stackable_item_def, quantity=99)
        player_inventory.add(existing)

        new_item = ItemInstance(definition=stackable_item_def, quantity=10)
        slot_idx = player_inventory.find_stackable_slot(new_item)
        assert slot_idx is None

    def test_find_stackable_slot_non_stackable(self, player_inventory, basic_item_def):
        """Test finding stackable slot for non-stackable item."""
        new_item = ItemInstance(definition=basic_item_def)
        slot_idx = player_inventory.find_stackable_slot(new_item)
        assert slot_idx is None


# =============================================================================
# InventoryContainer Move Tests
# =============================================================================


class TestInventoryContainerMove:
    """Tests for moving items within inventory."""

    def test_move_to_empty_slot(self, player_inventory, basic_item_def):
        """Test moving item to empty slot."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=0)
        success = player_inventory.move(0, 5)
        assert success is True
        assert player_inventory.get_item(0) is None
        assert player_inventory.get_item(5) == item

    def test_move_same_slot_fails(self, player_inventory, basic_item_def):
        """Test moving to same slot fails."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=0)
        success = player_inventory.move(0, 0)
        assert success is False

    def test_move_from_empty_slot_fails(self, player_inventory):
        """Test moving from empty slot fails."""
        success = player_inventory.move(0, 5)
        assert success is False

    def test_move_from_locked_slot_fails(self, player_inventory, basic_item_def):
        """Test moving from locked slot fails."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=0)
        player_inventory.lock_slot(0)
        success = player_inventory.move(0, 5)
        assert success is False

    def test_move_to_locked_slot_fails(self, player_inventory, basic_item_def):
        """Test moving to locked slot fails."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=0)
        player_inventory.lock_slot(5)
        success = player_inventory.move(0, 5)
        assert success is False

    def test_move_swap_items(self, player_inventory, basic_item_def):
        """Test swapping two items."""
        item1 = ItemInstance(definition=basic_item_def)
        other_def = ItemDefinition(
            id="axe_iron",
            name="Iron Axe",
            item_type=ItemType.EQUIPMENT,
        )
        item2 = ItemInstance(definition=other_def)

        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=5)

        success = player_inventory.move(0, 5)
        assert success is True
        assert player_inventory.get_item(0) == item2
        assert player_inventory.get_item(5) == item1

    def test_move_merge_stacks(self, player_inventory, stackable_item_def):
        """Test moving stackable items merges."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=30)
        item2 = ItemInstance(definition=stackable_item_def, quantity=40)
        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=5, auto_stack=False)

        success = player_inventory.move(0, 5)
        assert success is True
        assert player_inventory.get_item(0) is None
        assert player_inventory.get_item(5).quantity == 70

    def test_move_partial_merge(self, player_inventory, stackable_item_def):
        """Test partial merge when moving to nearly full stack."""
        item1 = ItemInstance(definition=stackable_item_def, quantity=60)
        item2 = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=5, auto_stack=False)

        success = player_inventory.move(0, 5)
        assert success is True
        # item2 should be full (99), item1 has remainder (11)
        assert player_inventory.get_item(5).quantity == 99
        assert player_inventory.get_item(0).quantity == 11

    def test_split_stack(self, player_inventory, stackable_item_def):
        """Test splitting a stack."""
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item, target_slot=0)

        new_slot = player_inventory.split(0, 20)
        assert new_slot is not None
        assert player_inventory.get_item(0).quantity == 30
        assert player_inventory.get_item(new_slot).quantity == 20

    def test_split_invalid_quantity(self, player_inventory, stackable_item_def):
        """Test splitting with invalid quantity returns None."""
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item, target_slot=0)

        assert player_inventory.split(0, 0) is None
        assert player_inventory.split(0, 50) is None
        assert player_inventory.split(0, 60) is None

    def test_split_from_empty_slot(self, player_inventory):
        """Test splitting from empty slot returns None."""
        assert player_inventory.split(0, 10) is None

    def test_split_no_empty_slot(self, limited_inventory, stackable_item_def):
        """Test splitting when no empty slot available."""
        # Fill all slots
        for i in range(5):
            limited_inventory.add(
                ItemInstance(definition=stackable_item_def, quantity=20),
                target_slot=i,
            )

        assert limited_inventory.split(0, 10) is None


# =============================================================================
# InventoryContainer Transfer Tests
# =============================================================================


class TestInventoryContainerTransfer:
    """Tests for transferring items between inventories."""

    def test_transfer_to_success(self, player_inventory, basic_item_def):
        """Test successful transfer to another inventory."""
        target = InventoryContainer(container_type=ContainerType.CHEST)
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item)

        success, qty = player_inventory.transfer_to(target, 0)
        assert success is True
        assert qty == 1
        assert player_inventory.is_empty is True
        assert target.count_item("sword_iron") == 1

    def test_transfer_to_partial_quantity(self, player_inventory, stackable_item_def):
        """Test transferring partial quantity."""
        target = InventoryContainer(container_type=ContainerType.CHEST)
        item = ItemInstance(definition=stackable_item_def, quantity=50)
        player_inventory.add(item)

        success, qty = player_inventory.transfer_to(target, 0, quantity=30)
        assert success is True
        assert qty == 30
        assert player_inventory.count_item("potion_health") == 20
        assert target.count_item("potion_health") == 30

    def test_transfer_to_empty_slot(self, player_inventory):
        """Test transferring from empty slot fails."""
        target = InventoryContainer(container_type=ContainerType.CHEST)
        success, qty = player_inventory.transfer_to(target, 0)
        assert success is False
        assert qty == 0

    def test_transfer_to_full_target(self, player_inventory, basic_item_def):
        """Test transferring to full target fails."""
        target = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=1,
        )
        item1 = ItemInstance(definition=basic_item_def)
        item2 = ItemInstance(definition=basic_item_def)

        target.add(item1)
        player_inventory.add(item2)

        success, qty = player_inventory.transfer_to(target, 0)
        assert success is False
        assert qty == 0
        assert player_inventory.count_item("sword_iron") == 1

    def test_transfer_all_to(self, player_inventory, basic_item_def, stackable_item_def):
        """Test transferring all items."""
        target = InventoryContainer(container_type=ContainerType.CHEST)
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.add(ItemInstance(definition=stackable_item_def, quantity=50))

        total = player_inventory.transfer_all_to(target)
        assert total == 51
        assert player_inventory.is_empty is True

    def test_transfer_all_partial(self, player_inventory, basic_item_def):
        """Test transfer all with partial success."""
        target = InventoryContainer(
            container_type=ContainerType.CHEST,
            slot_count=2,
        )
        for _ in range(5):
            player_inventory.add(ItemInstance(definition=basic_item_def))

        total = player_inventory.transfer_all_to(target)
        assert total == 2  # Only 2 slots available
        assert player_inventory.count_item("sword_iron") == 3


# =============================================================================
# InventoryContainer Sort Tests
# =============================================================================


class TestInventoryContainerSort:
    """Tests for sorting inventory."""

    def test_sort_by_default_key(self, player_inventory):
        """Test sorting with default key."""
        defs = [
            ItemDefinition(id="z_item", name="Zebra", item_type=ItemType.JUNK),
            ItemDefinition(id="a_item", name="Apple", item_type=ItemType.CONSUMABLE),
            ItemDefinition(id="m_item", name="Mango", item_type=ItemType.MATERIAL),
        ]
        for d in defs:
            player_inventory.add(ItemInstance(definition=d))

        player_inventory.sort()

        # Default sort: by type, then rarity (desc), then name
        items = [(slot.item.item_id if slot.item else None) for slot in player_inventory]
        non_empty = [i for i in items if i is not None]
        # Should be sorted by type first
        assert len(non_empty) == 3

    def test_sort_with_custom_key(self, player_inventory):
        """Test sorting with custom key."""
        defs = [
            ItemDefinition(id="c", name="Charlie", item_type=ItemType.JUNK, base_value=10),
            ItemDefinition(id="a", name="Alpha", item_type=ItemType.JUNK, base_value=30),
            ItemDefinition(id="b", name="Bravo", item_type=ItemType.JUNK, base_value=20),
        ]
        for d in defs:
            player_inventory.add(ItemInstance(definition=d))

        player_inventory.sort(key=lambda x: x.definition.base_value, reverse=True)

        items = [slot.item for slot in player_inventory if slot.item]
        assert items[0].item_id == "a"  # Value 30
        assert items[1].item_id == "b"  # Value 20
        assert items[2].item_id == "c"  # Value 10

    def test_sort_preserves_locked_slots(self, player_inventory, basic_item_def):
        """Test sorting preserves locked slots."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=5)
        player_inventory.lock_slot(5)

        # Add more items
        for i in range(3):
            player_inventory.add(ItemInstance(definition=ItemDefinition(
                id=f"item_{i}",
                name=f"Item {i}",
                item_type=ItemType.JUNK,
            )))

        player_inventory.sort()

        # Locked slot should still have the sword
        assert player_inventory.get_item(5) == item

    def test_compact_merges_stacks(self, player_inventory, stackable_item_def):
        """Test compact merges stackable items."""
        # Add scattered partial stacks
        player_inventory.add(
            ItemInstance(definition=stackable_item_def, quantity=30),
            target_slot=0,
        )
        player_inventory.add(
            ItemInstance(definition=stackable_item_def, quantity=25),
            target_slot=5,
            auto_stack=False,
        )
        player_inventory.add(
            ItemInstance(definition=stackable_item_def, quantity=20),
            target_slot=10,
            auto_stack=False,
        )

        assert player_inventory.used_slot_count == 3
        freed = player_inventory.compact()

        assert player_inventory.count_item("potion_health") == 75
        assert player_inventory.used_slot_count <= 1  # All merged
        assert freed >= 2

    def test_compact_moves_to_front(self, player_inventory, basic_item_def):
        """Test compact moves items to front slots."""
        # Add items in scattered positions
        player_inventory.add(ItemInstance(definition=basic_item_def), target_slot=10)
        player_inventory.add(ItemInstance(definition=basic_item_def), target_slot=20)

        player_inventory.compact()

        # Items should be at front
        assert player_inventory.get_item(0) is not None or player_inventory.get_item(1) is not None


# =============================================================================
# InventoryContainer Filter Tests
# =============================================================================


class TestInventoryContainerFilter:
    """Tests for slot filters."""

    def test_set_slot_filter(self, player_inventory):
        """Test setting slot filter."""
        player_inventory.set_slot_filter(0, ItemType.EQUIPMENT)
        assert player_inventory.get_slot(0).filter_type == ItemType.EQUIPMENT

    def test_clear_slot_filter(self, player_inventory):
        """Test clearing slot filter."""
        player_inventory.set_slot_filter(0, ItemType.EQUIPMENT)
        player_inventory.set_slot_filter(0, None)
        assert player_inventory.get_slot(0).filter_type is None

    def test_add_respects_filter(self, player_inventory, basic_item_def, stackable_item_def):
        """Test adding item respects slot filter."""
        player_inventory.set_slot_filter(0, ItemType.CONSUMABLE)

        # Equipment should not go in slot 0
        equipment = ItemInstance(definition=basic_item_def)
        player_inventory.add(equipment, target_slot=0)
        assert player_inventory.get_item(0) is None

        # Consumable should go in slot 0
        consumable = ItemInstance(definition=stackable_item_def, quantity=10)
        player_inventory.add(consumable, target_slot=0)
        assert player_inventory.get_item(0) is not None

    def test_lock_slot(self, player_inventory):
        """Test locking a slot."""
        player_inventory.lock_slot(0)
        assert player_inventory.get_slot(0).locked is True

    def test_unlock_slot(self, player_inventory):
        """Test unlocking a slot."""
        player_inventory.lock_slot(0)
        player_inventory.unlock_slot(0)
        assert player_inventory.get_slot(0).locked is False


# =============================================================================
# InventoryContainer Resize Tests
# =============================================================================


class TestInventoryContainerResize:
    """Tests for resizing inventory."""

    def test_resize_expand(self, player_inventory):
        """Test expanding inventory."""
        original = player_inventory.slot_count
        success = player_inventory.resize(original + 10)
        assert success is True
        assert player_inventory.slot_count == original + 10

    def test_resize_shrink_empty(self, player_inventory):
        """Test shrinking empty inventory."""
        success = player_inventory.resize(10)
        assert success is True
        assert player_inventory.slot_count == 10

    def test_resize_shrink_with_items_fails(self, player_inventory, basic_item_def):
        """Test shrinking with items in truncated area fails."""
        player_inventory.add(
            ItemInstance(definition=basic_item_def),
            target_slot=player_inventory.slot_count - 1,
        )
        success = player_inventory.resize(10)
        assert success is False

    def test_resize_to_zero_fails(self, player_inventory):
        """Test resizing to zero fails."""
        success = player_inventory.resize(0)
        assert success is False


# =============================================================================
# InventoryContainer Weight Tests
# =============================================================================


class TestInventoryContainerWeight:
    """Tests for weight management."""

    def test_weight_available_unlimited(self, player_inventory):
        """Test weight available for unlimited container."""
        inv = InventoryContainer(
            container_type=ContainerType.SHOP,  # Unlimited weight
        )
        assert inv.weight_available == float('inf')

    def test_weight_available_limited(self, limited_inventory):
        """Test weight available for limited container."""
        assert limited_inventory.weight_available == 50.0

    def test_weight_available_after_add(self, limited_inventory, basic_item_def):
        """Test weight available decreases after add."""
        item = ItemInstance(definition=basic_item_def)  # 2.5 weight
        limited_inventory.add(item)
        assert limited_inventory.weight_available == pytest.approx(47.5)

    def test_is_over_weight(self, limited_inventory, basic_item_def):
        """Test over weight detection."""
        # Force items in without weight check
        for i in range(25):  # 25 * 2.5 = 62.5 > 50
            slot = limited_inventory.get_slot(i % 5)
            if slot.item:
                slot.item.quantity += 1  # This doesn't work for equipment
            else:
                slot.item = ItemInstance(definition=basic_item_def)
                limited_inventory._current_weight += 2.5

        assert limited_inventory.is_over_weight is True

    def test_unlimited_weight_never_over(self, player_inventory, basic_item_def):
        """Test unlimited weight container never reports over weight."""
        inv = InventoryContainer(
            container_type=ContainerType.SHOP,
            slot_count=100,
        )
        for _ in range(50):
            inv.add(ItemInstance(definition=basic_item_def))
        assert inv.is_over_weight is False


# =============================================================================
# InventoryContainer Event Tests
# =============================================================================


class TestInventoryContainerEvents:
    """Tests for inventory events."""

    def test_event_on_add(self, player_inventory, basic_item_def):
        """Test event emitted on add."""
        events = []
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.add(ItemInstance(definition=basic_item_def))

        assert len(events) == 1
        assert events[0].event_type == EconomyEvent.ITEM_ADDED

    def test_event_on_remove(self, player_inventory, basic_item_def):
        """Test event emitted on remove."""
        events = []
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.remove_at(0)

        assert len(events) == 1
        assert events[0].event_type == EconomyEvent.ITEM_REMOVED

    def test_event_on_move(self, player_inventory, basic_item_def):
        """Test event emitted on move."""
        events = []
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.move(0, 5)

        assert len(events) >= 1
        assert any(e.event_type == EconomyEvent.ITEM_MOVED for e in events)

    def test_event_on_merge(self, player_inventory, stackable_item_def):
        """Test event emitted on merge."""
        events = []
        player_inventory.add(ItemInstance(definition=stackable_item_def, quantity=30))
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.add(ItemInstance(definition=stackable_item_def, quantity=20))

        assert any(e.event_type == EconomyEvent.ITEM_MERGED for e in events)

    def test_event_on_split(self, player_inventory, stackable_item_def):
        """Test event emitted on split."""
        events = []
        player_inventory.add(ItemInstance(definition=stackable_item_def, quantity=50))
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.split(0, 20)

        assert any(e.event_type == EconomyEvent.ITEM_SPLIT for e in events)

    def test_remove_listener(self, player_inventory, basic_item_def):
        """Test removing event listener."""
        events = []
        callback = lambda e: events.append(e)
        player_inventory.add_listener(callback)
        player_inventory.remove_listener(callback)

        player_inventory.add(ItemInstance(definition=basic_item_def))

        assert len(events) == 0


# =============================================================================
# InventoryContainer Transaction Tests
# =============================================================================


class TestInventoryContainerTransactions:
    """Tests for transaction management."""

    def test_transaction_batches_events(self, player_inventory, basic_item_def):
        """Test transaction batches events."""
        events = []
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.begin_transaction()
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.add(ItemInstance(definition=basic_item_def))

        assert len(events) == 0  # Events held

        player_inventory.commit_transaction()

        assert len(events) == 2  # Events released

    def test_transaction_rollback_discards_events(self, player_inventory, basic_item_def):
        """Test rollback discards events."""
        events = []
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.begin_transaction()
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.rollback_transaction()

        assert len(events) == 0  # Events discarded

    def test_multiple_transactions(self, player_inventory, basic_item_def):
        """Test multiple sequential transactions."""
        events = []
        player_inventory.add_listener(lambda e: events.append(e))

        player_inventory.begin_transaction()
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.commit_transaction()

        player_inventory.begin_transaction()
        player_inventory.add(ItemInstance(definition=basic_item_def))
        player_inventory.commit_transaction()

        assert len(events) == 2


# =============================================================================
# InventoryContainer Iteration Tests
# =============================================================================


class TestInventoryContainerIteration:
    """Tests for iterating over inventory."""

    def test_iterate_all_slots(self, player_inventory):
        """Test iterating over all slots."""
        count = 0
        for slot in player_inventory:
            count += 1
        assert count == player_inventory.slot_count

    def test_iterate_items_only(self, player_inventory, basic_item_def):
        """Test iterating over items only."""
        player_inventory.add(ItemInstance(definition=basic_item_def), target_slot=5)
        player_inventory.add(ItemInstance(definition=basic_item_def), target_slot=10)

        items = list(player_inventory.items())
        assert len(items) == 2
        assert items[0][0] == 5
        assert items[1][0] == 10

    def test_len_returns_slot_count(self, player_inventory):
        """Test len returns slot count."""
        assert len(player_inventory) == player_inventory.slot_count

    def test_getitem_access(self, player_inventory, basic_item_def):
        """Test bracket notation access."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=3)
        assert player_inventory[3] == item
        assert player_inventory[0] is None


# =============================================================================
# InventoryContainer Serialization Tests
# =============================================================================


class TestInventoryContainerSerialization:
    """Tests for inventory serialization."""

    def test_to_dict_empty(self, player_inventory):
        """Test serializing empty inventory."""
        data = player_inventory.to_dict()
        assert "id" in data
        assert data["type"] == "PLAYER_INVENTORY"
        assert len(data["slots"]) == player_inventory.slot_count

    def test_to_dict_with_items(self, player_inventory, basic_item_def):
        """Test serializing inventory with items."""
        item = ItemInstance(definition=basic_item_def)
        player_inventory.add(item, target_slot=5)

        data = player_inventory.to_dict()
        assert data["slots"][5]["item"] is not None
        assert data["slots"][5]["item"]["definition_id"] == "sword_iron"

    def test_to_dict_preserves_slot_state(self, player_inventory):
        """Test serialization preserves slot state."""
        player_inventory.lock_slot(3)
        player_inventory.set_slot_filter(5, ItemType.EQUIPMENT)

        data = player_inventory.to_dict()
        assert data["slots"][3]["locked"] is True
        assert data["slots"][5]["filter_type"] == "EQUIPMENT"


# =============================================================================
# ItemRegistry Tests
# =============================================================================


class TestItemRegistry:
    """Tests for ItemRegistry singleton."""

    def test_singleton_pattern(self):
        """Test registry is singleton."""
        ItemRegistry.reset()
        reg1 = ItemRegistry.instance()
        reg2 = ItemRegistry.instance()
        assert reg1 is reg2
        ItemRegistry.reset()

    def test_register_item(self, item_registry):
        """Test registering item definition."""
        assert item_registry.exists("sword_iron") is True

    def test_register_duplicate_raises(self, item_registry):
        """Test registering duplicate raises error."""
        with pytest.raises(ValueError, match="already registered"):
            item_registry.register(ItemDefinition(
                id="sword_iron",
                name="Another Sword",
                item_type=ItemType.EQUIPMENT,
            ))

    def test_get_registered(self, item_registry):
        """Test getting registered item."""
        item = item_registry.get("sword_iron")
        assert item is not None
        assert item.name == "Iron Sword"

    def test_get_unregistered(self, item_registry):
        """Test getting unregistered item returns None."""
        assert item_registry.get("nonexistent") is None

    def test_get_or_raise_success(self, item_registry):
        """Test get_or_raise with existing item."""
        item = item_registry.get_or_raise("sword_iron")
        assert item.name == "Iron Sword"

    def test_get_or_raise_fails(self, item_registry):
        """Test get_or_raise with missing item raises."""
        with pytest.raises(KeyError, match="Unknown item"):
            item_registry.get_or_raise("nonexistent")

    def test_all_items(self, item_registry):
        """Test getting all items."""
        all_items = item_registry.all()
        assert len(all_items) == 3

    def test_by_type(self, item_registry):
        """Test filtering by type."""
        equipment = item_registry.by_type(ItemType.EQUIPMENT)
        assert len(equipment) == 1
        assert equipment[0].id == "sword_iron"

    def test_by_rarity(self, item_registry):
        """Test filtering by rarity."""
        common = item_registry.by_rarity(Rarity.COMMON)
        assert len(common) == 3

    def test_clear_registry(self, item_registry):
        """Test clearing registry."""
        item_registry.clear()
        assert len(item_registry.all()) == 0

    def test_reset_creates_new_instance(self):
        """Test reset creates new instance."""
        ItemRegistry.reset()
        reg1 = ItemRegistry.instance()
        reg1.register(ItemDefinition(id="test", name="Test", item_type=ItemType.JUNK))

        ItemRegistry.reset()
        reg2 = ItemRegistry.instance()

        assert reg2.get("test") is None
        ItemRegistry.reset()
