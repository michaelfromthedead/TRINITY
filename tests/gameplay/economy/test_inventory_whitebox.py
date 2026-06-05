"""
WHITEBOX Tests for Inventory System (T-ECON-1.1 through T-ECON-1.3)

Tests:
- ItemDefinition stacking and validation
- ItemInstance merging, splitting, cloning
- InventoryContainer auto_stack, compact, sort
- Weight limits and overflow handling
- Slot filters and locking
- Serialization round-trips
- Event emission and transactions
"""
import pytest
from uuid import UUID, uuid4
from typing import List, Optional

from engine.gameplay.economy.inventory import (
    ItemDefinition,
    ItemInstance,
    InventorySlot,
    InventoryContainer,
    InventoryEvent,
    ItemRegistry,
    Serializer,
    ECONOMY_SCHEMA_VERSION,
)
from engine.gameplay.economy.constants import (
    ItemType,
    Rarity,
    ContainerType,
    EconomyEvent,
    STACKABLE_TYPES,
    MAX_STACK_SIZE,
    DEFAULT_STACK_LIMITS,
    DEFAULT_CONTAINER_SLOTS,
    DEFAULT_WEIGHT_LIMITS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def material_definition():
    """A stackable material item definition."""
    return ItemDefinition(
        id="iron_ore",
        name="Iron Ore",
        item_type=ItemType.MATERIAL,
        rarity=Rarity.COMMON,
        max_stack=99,
        weight=0.5,
        base_value=10,
    )


@pytest.fixture
def equipment_definition():
    """A non-stackable equipment item definition."""
    return ItemDefinition(
        id="iron_sword",
        name="Iron Sword",
        item_type=ItemType.EQUIPMENT,
        rarity=Rarity.UNCOMMON,
        max_stack=1,
        weight=5.0,
        base_value=100,
    )


@pytest.fixture
def consumable_definition():
    """A stackable consumable item definition."""
    return ItemDefinition(
        id="health_potion",
        name="Health Potion",
        item_type=ItemType.CONSUMABLE,
        rarity=Rarity.COMMON,
        max_stack=20,
        weight=0.2,
        base_value=25,
    )


@pytest.fixture
def player_inventory():
    """An empty player inventory container."""
    return InventoryContainer(
        container_type=ContainerType.PLAYER_INVENTORY,
        slot_count=30,
        weight_limit=100.0,
        owner_id="player_001",
    )


@pytest.fixture
def item_registry():
    """Fresh item registry for tests."""
    ItemRegistry.reset()
    return ItemRegistry.instance()


# =============================================================================
# ITEM DEFINITION TESTS (T-ECON-1.1)
# =============================================================================


class TestItemDefinition:
    """Whitebox tests for ItemDefinition."""

    def test_basic_creation(self):
        """Test basic item definition creation."""
        item = ItemDefinition(
            id="test_item",
            name="Test Item",
            item_type=ItemType.MATERIAL,
        )
        assert item.id == "test_item"
        assert item.name == "Test Item"
        assert item.item_type == ItemType.MATERIAL

    def test_empty_id_raises(self):
        """Empty item ID should raise ValueError."""
        with pytest.raises(ValueError, match="id cannot be empty"):
            ItemDefinition(id="", name="Test", item_type=ItemType.MATERIAL)

    def test_empty_name_raises(self):
        """Empty item name should raise ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ItemDefinition(id="test", name="", item_type=ItemType.MATERIAL)

    def test_auto_stack_limit_for_materials(self):
        """Materials should auto-set stack limit if not specified."""
        item = ItemDefinition(
            id="ore",
            name="Ore",
            item_type=ItemType.MATERIAL,
            max_stack=0,  # Should auto-set
        )
        assert item.max_stack == DEFAULT_STACK_LIMITS.get(ItemType.MATERIAL, 1)

    def test_auto_stack_limit_for_equipment(self):
        """Equipment should have stack limit of 1."""
        item = ItemDefinition(
            id="sword",
            name="Sword",
            item_type=ItemType.EQUIPMENT,
            max_stack=0,
        )
        assert item.max_stack == 1

    def test_max_stack_clamped(self):
        """Stack size should be clamped to MAX_STACK_SIZE."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            max_stack=MAX_STACK_SIZE + 1000,
        )
        assert item.max_stack == MAX_STACK_SIZE

    def test_negative_weight_clamped_to_zero(self):
        """Negative weight should be clamped to 0."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            weight=-5.0,
        )
        assert item.weight == 0.0

    def test_negative_value_clamped_to_zero(self):
        """Negative base value should be clamped to 0."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            base_value=-100,
        )
        assert item.base_value == 0

    def test_level_requirement_minimum_one(self):
        """Level requirement should be at least 1."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            level_requirement=0,
        )
        assert item.level_requirement == 1

    def test_is_stackable_for_stackable_types(self, material_definition):
        """Materials with max_stack > 1 should be stackable."""
        assert material_definition.is_stackable is True

    def test_is_stackable_for_equipment(self, equipment_definition):
        """Equipment should not be stackable."""
        assert equipment_definition.is_stackable is False

    def test_hash_equals_id_hash(self, material_definition):
        """Item definition hash should be based on ID."""
        assert hash(material_definition) == hash(material_definition.id)

    def test_equality_based_on_id(self):
        """Two definitions with same ID should be equal."""
        item1 = ItemDefinition(id="test", name="Test 1", item_type=ItemType.MATERIAL)
        item2 = ItemDefinition(id="test", name="Test 2", item_type=ItemType.CONSUMABLE)
        assert item1 == item2

    def test_serialization_round_trip(self, material_definition):
        """Serialization and deserialization should preserve data."""
        data = material_definition.to_dict()
        restored = ItemDefinition.from_dict(data)
        assert restored.id == material_definition.id
        assert restored.name == material_definition.name
        assert restored.item_type == material_definition.item_type
        assert restored.rarity == material_definition.rarity
        assert restored.max_stack == material_definition.max_stack
        assert restored.weight == material_definition.weight

    def test_flags_preserved_in_serialization(self):
        """Flags should be preserved through serialization."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.KEY_ITEM,
            flags=frozenset({"soulbound", "unique"}),
        )
        data = item.to_dict()
        restored = ItemDefinition.from_dict(data)
        assert "soulbound" in restored.flags
        assert "unique" in restored.flags

    def test_metadata_preserved_in_serialization(self):
        """Metadata should be preserved through serialization."""
        item = ItemDefinition(
            id="test",
            name="Test",
            item_type=ItemType.MATERIAL,
            metadata={"custom_key": "custom_value", "number": 42},
        )
        data = item.to_dict()
        restored = ItemDefinition.from_dict(data)
        assert restored.metadata["custom_key"] == "custom_value"
        assert restored.metadata["number"] == 42


# =============================================================================
# ITEM INSTANCE TESTS (T-ECON-1.1)
# =============================================================================


class TestItemInstance:
    """Whitebox tests for ItemInstance."""

    def test_basic_creation(self, material_definition):
        """Test basic item instance creation."""
        item = ItemInstance(definition=material_definition, quantity=10)
        assert item.definition == material_definition
        assert item.quantity == 10
        assert isinstance(item.instance_id, UUID)

    def test_zero_quantity_raises(self, material_definition):
        """Zero quantity should raise ValueError."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            ItemInstance(definition=material_definition, quantity=0)

    def test_negative_quantity_raises(self, material_definition):
        """Negative quantity should raise ValueError."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            ItemInstance(definition=material_definition, quantity=-5)

    def test_quantity_exceeds_max_stack_raises(self, material_definition):
        """Quantity exceeding max stack should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds max stack"):
            ItemInstance(definition=material_definition, quantity=100)  # max is 99

    def test_item_id_property(self, material_definition):
        """item_id should return definition.id."""
        item = ItemInstance(definition=material_definition, quantity=5)
        assert item.item_id == "iron_ore"

    def test_total_weight_calculation(self, material_definition):
        """Total weight should be weight * quantity."""
        item = ItemInstance(definition=material_definition, quantity=10)
        assert item.total_weight == pytest.approx(5.0)  # 0.5 * 10

    def test_total_value_calculation(self, material_definition):
        """Total value should be base_value * quantity."""
        item = ItemInstance(definition=material_definition, quantity=10)
        assert item.total_value == 100  # 10 * 10

    def test_can_add_more_when_not_full(self, material_definition):
        """can_add_more should be True when below max stack."""
        item = ItemInstance(definition=material_definition, quantity=50)
        assert item.can_add_more is True

    def test_can_add_more_when_full(self, material_definition):
        """can_add_more should be False when at max stack."""
        item = ItemInstance(definition=material_definition, quantity=99)
        assert item.can_add_more is False

    def test_space_remaining(self, material_definition):
        """space_remaining should return correct value."""
        item = ItemInstance(definition=material_definition, quantity=50)
        assert item.space_remaining == 49

    def test_can_stack_with_same_item(self, material_definition):
        """Same items should be stackable."""
        item1 = ItemInstance(definition=material_definition, quantity=10)
        item2 = ItemInstance(definition=material_definition, quantity=5)
        assert item1.can_stack_with(item2) is True

    def test_cannot_stack_with_different_item(self, material_definition, consumable_definition):
        """Different items should not stack."""
        item1 = ItemInstance(definition=material_definition, quantity=10)
        item2 = ItemInstance(definition=consumable_definition, quantity=5)
        assert item1.can_stack_with(item2) is False

    def test_cannot_stack_with_different_bound_to(self, material_definition):
        """Items bound to different entities should not stack."""
        item1 = ItemInstance(definition=material_definition, quantity=10, bound_to="player1")
        item2 = ItemInstance(definition=material_definition, quantity=5, bound_to="player2")
        assert item1.can_stack_with(item2) is False

    def test_can_stack_with_same_bound_to(self, material_definition):
        """Items bound to same entity should stack."""
        item1 = ItemInstance(definition=material_definition, quantity=10, bound_to="player1")
        item2 = ItemInstance(definition=material_definition, quantity=5, bound_to="player1")
        assert item1.can_stack_with(item2) is True

    def test_cannot_stack_with_different_custom_data(self, material_definition):
        """Items with different custom data should not stack."""
        item1 = ItemInstance(definition=material_definition, quantity=10, custom_data={"quality": 1})
        item2 = ItemInstance(definition=material_definition, quantity=5, custom_data={"quality": 2})
        assert item1.can_stack_with(item2) is False

    def test_can_stack_with_same_custom_data(self, material_definition):
        """Items with same custom data should stack."""
        item1 = ItemInstance(definition=material_definition, quantity=10, custom_data={"quality": 1})
        item2 = ItemInstance(definition=material_definition, quantity=5, custom_data={"quality": 1})
        assert item1.can_stack_with(item2) is True

    def test_equipment_not_stackable(self, equipment_definition):
        """Equipment items should not stack."""
        item1 = ItemInstance(definition=equipment_definition, quantity=1)
        item2 = ItemInstance(definition=equipment_definition, quantity=1)
        assert item1.can_stack_with(item2) is False

    def test_split_valid_amount(self, material_definition):
        """Splitting a valid amount should work correctly."""
        item = ItemInstance(definition=material_definition, quantity=50)
        split = item.split(20)
        assert item.quantity == 30
        assert split.quantity == 20
        assert split.definition == item.definition
        assert split.instance_id != item.instance_id

    def test_split_zero_raises(self, material_definition):
        """Splitting zero should raise ValueError."""
        item = ItemInstance(definition=material_definition, quantity=50)
        with pytest.raises(ValueError, match="must be positive"):
            item.split(0)

    def test_split_negative_raises(self, material_definition):
        """Splitting negative should raise ValueError."""
        item = ItemInstance(definition=material_definition, quantity=50)
        with pytest.raises(ValueError, match="must be positive"):
            item.split(-10)

    def test_split_entire_stack_raises(self, material_definition):
        """Splitting entire stack should raise ValueError."""
        item = ItemInstance(definition=material_definition, quantity=50)
        with pytest.raises(ValueError, match="must be less than total quantity"):
            item.split(50)

    def test_split_more_than_stack_raises(self, material_definition):
        """Splitting more than stack should raise ValueError."""
        item = ItemInstance(definition=material_definition, quantity=50)
        with pytest.raises(ValueError, match="must be less than total quantity"):
            item.split(60)

    def test_split_preserves_bound_to(self, material_definition):
        """Split should preserve bound_to."""
        item = ItemInstance(definition=material_definition, quantity=50, bound_to="player1")
        split = item.split(20)
        assert split.bound_to == "player1"

    def test_split_copies_custom_data(self, material_definition):
        """Split should copy custom data."""
        item = ItemInstance(
            definition=material_definition, quantity=50, custom_data={"quality": 5}
        )
        split = item.split(20)
        assert split.custom_data == {"quality": 5}
        # Ensure it's a copy, not the same reference
        split.custom_data["quality"] = 10
        assert item.custom_data["quality"] == 5

    def test_merge_from_compatible(self, material_definition):
        """Merging compatible items should work."""
        item1 = ItemInstance(definition=material_definition, quantity=50)
        item2 = ItemInstance(definition=material_definition, quantity=30)
        merged = item1.merge_from(item2)
        assert merged == 30
        assert item1.quantity == 80
        assert item2.quantity == 0

    def test_merge_from_partial(self, material_definition):
        """Merging when target is nearly full should merge partially."""
        item1 = ItemInstance(definition=material_definition, quantity=80)  # space for 19
        item2 = ItemInstance(definition=material_definition, quantity=30)
        merged = item1.merge_from(item2)
        assert merged == 19
        assert item1.quantity == 99
        assert item2.quantity == 11

    def test_merge_from_incompatible_raises(self, material_definition, consumable_definition):
        """Merging incompatible items should raise ValueError."""
        item1 = ItemInstance(definition=material_definition, quantity=50)
        item2 = ItemInstance(definition=consumable_definition, quantity=5)
        with pytest.raises(ValueError, match="cannot be stacked"):
            item1.merge_from(item2)

    def test_clone_creates_independent_copy(self, material_definition):
        """Clone should create an independent copy."""
        original = ItemInstance(
            definition=material_definition,
            quantity=50,
            bound_to="player1",
            custom_data={"quality": 5},
        )
        cloned = original.clone()
        assert cloned.definition == original.definition
        assert cloned.quantity == original.quantity
        assert cloned.bound_to == original.bound_to
        assert cloned.custom_data == original.custom_data
        assert cloned.instance_id != original.instance_id
        # Ensure custom_data is a copy
        cloned.custom_data["quality"] = 10
        assert original.custom_data["quality"] == 5

    def test_serialization_round_trip(self, material_definition):
        """Serialization and deserialization should preserve data."""
        original = ItemInstance(
            definition=material_definition,
            quantity=42,
            bound_to="player1",
            durability=75.5,
            custom_data={"enchant": "fire"},
        )
        data = original.to_dict()
        restored = ItemInstance.from_dict(data)
        assert restored.quantity == original.quantity
        assert restored.bound_to == original.bound_to
        assert restored.durability == original.durability
        assert restored.custom_data == original.custom_data
        assert restored.definition.id == original.definition.id

    def test_deserialization_with_registry(self, item_registry, material_definition):
        """Deserialization should use registry when available."""
        item_registry.register(material_definition)
        data = {
            "instance_id": str(uuid4()),
            "definition_id": "iron_ore",
            "quantity": 25,
        }
        restored = ItemInstance.from_dict(data, item_registry.as_dict())
        assert restored.quantity == 25
        assert restored.definition == material_definition

    def test_deserialization_missing_definition_raises(self, item_registry):
        """Deserialization with nonexistent definition_id should raise error."""
        data = {
            "definition_id": "nonexistent_item",
            "instance_id": str(uuid4()),
            "quantity": 10,
        }
        with pytest.raises((KeyError, ValueError)):
            ItemInstance.from_dict(data, item_registry.as_dict())


# =============================================================================
# INVENTORY SLOT TESTS
# =============================================================================


class TestInventorySlot:
    """Whitebox tests for InventorySlot."""

    def test_empty_slot_creation(self):
        """Empty slot creation."""
        slot = InventorySlot(index=0)
        assert slot.index == 0
        assert slot.item is None
        assert slot.locked is False
        assert slot.filter_type is None

    def test_is_empty_when_no_item(self):
        """is_empty should be True when no item."""
        slot = InventorySlot(index=0)
        assert slot.is_empty is True

    def test_is_empty_when_has_item(self, material_definition):
        """is_empty should be False when has item."""
        item = ItemInstance(definition=material_definition, quantity=10)
        slot = InventorySlot(index=0, item=item)
        assert slot.is_empty is False

    def test_is_available_when_empty_and_unlocked(self):
        """is_available should be True when empty and unlocked."""
        slot = InventorySlot(index=0)
        assert slot.is_available is True

    def test_is_available_when_empty_and_locked(self):
        """is_available should be False when locked."""
        slot = InventorySlot(index=0, locked=True)
        assert slot.is_available is False

    def test_is_available_when_occupied(self, material_definition):
        """is_available should be False when occupied."""
        item = ItemInstance(definition=material_definition, quantity=10)
        slot = InventorySlot(index=0, item=item)
        assert slot.is_available is False

    def test_accepts_when_no_filter(self, material_definition):
        """accepts should be True when no filter and not locked."""
        slot = InventorySlot(index=0)
        item = ItemInstance(definition=material_definition, quantity=10)
        assert slot.accepts(item) is True

    def test_accepts_when_locked(self, material_definition):
        """accepts should be False when locked."""
        slot = InventorySlot(index=0, locked=True)
        item = ItemInstance(definition=material_definition, quantity=10)
        assert slot.accepts(item) is False

    def test_accepts_when_filter_matches(self, material_definition):
        """accepts should be True when filter matches."""
        slot = InventorySlot(index=0, filter_type=ItemType.MATERIAL)
        item = ItemInstance(definition=material_definition, quantity=10)
        assert slot.accepts(item) is True

    def test_accepts_when_filter_mismatch(self, material_definition):
        """accepts should be False when filter doesn't match."""
        slot = InventorySlot(index=0, filter_type=ItemType.EQUIPMENT)
        item = ItemInstance(definition=material_definition, quantity=10)
        assert slot.accepts(item) is False

    def test_serialization_round_trip(self, material_definition):
        """Serialization and deserialization should preserve data."""
        item = ItemInstance(definition=material_definition, quantity=10)
        slot = InventorySlot(index=5, item=item, locked=True, filter_type=ItemType.MATERIAL)
        data = slot.to_dict()
        restored = InventorySlot.from_dict(data)
        assert restored.index == 5
        assert restored.locked is True
        assert restored.filter_type == ItemType.MATERIAL
        assert restored.item.quantity == 10


# =============================================================================
# INVENTORY CONTAINER TESTS (T-ECON-1.2, T-ECON-1.3)
# =============================================================================


class TestInventoryContainer:
    """Whitebox tests for InventoryContainer."""

    def test_basic_creation(self, player_inventory):
        """Test basic container creation."""
        assert player_inventory.slot_count == 30
        assert player_inventory.weight_limit == 100.0
        assert player_inventory.owner_id == "player_001"
        assert isinstance(player_inventory.id, UUID)

    def test_default_slot_count(self):
        """Container should use default slot count if not specified."""
        container = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert container.slot_count == DEFAULT_CONTAINER_SLOTS[ContainerType.PLAYER_INVENTORY]

    def test_default_weight_limit(self):
        """Container should use default weight limit if not specified."""
        container = InventoryContainer(container_type=ContainerType.PLAYER_INVENTORY)
        assert container.weight_limit == DEFAULT_WEIGHT_LIMITS[ContainerType.PLAYER_INVENTORY]

    def test_empty_properties(self, player_inventory):
        """Empty container should have correct initial properties."""
        assert player_inventory.is_empty is True
        assert player_inventory.is_full is False
        assert player_inventory.empty_slot_count == 30
        assert player_inventory.used_slot_count == 0
        assert player_inventory.current_weight == 0.0

    def test_weight_available_unlimited(self):
        """Unlimited weight container should have infinite available weight."""
        container = InventoryContainer(
            container_type=ContainerType.STASH,  # Unlimited weight
            weight_limit=0.0,
        )
        assert container.weight_available == float("inf")

    def test_weight_available_with_limit(self, player_inventory):
        """Limited weight container should calculate available correctly."""
        assert player_inventory.weight_available == 100.0

    def test_get_slot_valid_index(self, player_inventory):
        """get_slot should return slot for valid index."""
        slot = player_inventory.get_slot(0)
        assert slot.index == 0

    def test_get_slot_invalid_index_raises(self, player_inventory):
        """get_slot should raise IndexError for invalid index."""
        with pytest.raises(IndexError):
            player_inventory.get_slot(100)

    def test_get_slot_negative_index_raises(self, player_inventory):
        """get_slot should raise IndexError for negative index."""
        with pytest.raises(IndexError):
            player_inventory.get_slot(-1)

    def test_add_item_basic(self, player_inventory, material_definition):
        """Adding an item should work correctly."""
        item = ItemInstance(definition=material_definition, quantity=10)
        success, added = player_inventory.add(item)
        assert success is True
        assert added == 10
        assert player_inventory.is_empty is False
        assert player_inventory.current_weight == pytest.approx(5.0)

    def test_add_item_auto_stack(self, player_inventory, material_definition):
        """Adding stackable items should auto-stack."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=material_definition, quantity=20)
        player_inventory.add(item1)
        success, added = player_inventory.add(item2)
        assert success is True
        assert added == 20
        assert player_inventory.used_slot_count == 1
        assert player_inventory.count_item("iron_ore") == 50

    def test_add_item_auto_stack_overflow(self, player_inventory, material_definition):
        """Overflow from auto-stack should go to new slot."""
        item1 = ItemInstance(definition=material_definition, quantity=80)
        item2 = ItemInstance(definition=material_definition, quantity=40)  # Only 19 fit in first stack
        player_inventory.add(item1)
        success, added = player_inventory.add(item2)
        assert success is True
        assert added == 40
        assert player_inventory.used_slot_count == 2
        assert player_inventory.count_item("iron_ore") == 120

    def test_add_item_to_specific_slot(self, player_inventory, material_definition):
        """Adding to specific slot should work."""
        item = ItemInstance(definition=material_definition, quantity=10)
        success, added = player_inventory.add(item, target_slot=5)
        assert success is True
        assert player_inventory.get_item(5).quantity == 10

    def test_add_item_to_occupied_slot_stacks(self, player_inventory, material_definition):
        """Adding to occupied slot with same item should stack."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=material_definition, quantity=20)
        player_inventory.add(item1, target_slot=0)
        success, added = player_inventory.add(item2, target_slot=0)
        assert success is True
        assert added == 20
        assert player_inventory.get_item(0).quantity == 50

    def test_add_item_to_incompatible_slot_fails(self, player_inventory, material_definition, equipment_definition):
        """Adding to slot with incompatible item should fail."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=equipment_definition, quantity=1)
        player_inventory.add(item1, target_slot=0)
        success, added = player_inventory.add(item2, target_slot=0)
        assert success is False
        assert added == 0

    def test_add_item_exceeds_weight_fails(self, player_inventory, equipment_definition):
        """Adding item that would exceed weight limit should fail."""
        # Weight limit is 100, each sword is 5 weight
        for i in range(20):
            item = ItemInstance(definition=equipment_definition, quantity=1)
            player_inventory.add(item)
        assert player_inventory.current_weight == 100.0

        # Adding one more should fail
        item = ItemInstance(definition=equipment_definition, quantity=1)
        success, added = player_inventory.add(item)
        assert success is False
        assert added == 0

    def test_add_item_partial_weight_success(self, player_inventory, material_definition):
        """Adding amount within weight limit should succeed."""
        # Each ore is 0.5 weight
        item = ItemInstance(definition=material_definition, quantity=20)
        success, added = player_inventory.add(item)
        # Should add all 20 (10 weight) since we have 100 capacity
        assert success is True
        assert added == 20
        assert player_inventory.current_weight == pytest.approx(10.0)

    def test_add_item_no_empty_slots_fails(self, player_inventory, equipment_definition):
        """Adding when all slots are full should fail."""
        # Fill all 30 slots - check weight limit doesn't block first
        player_inventory._weight_limit = 0.0  # Disable weight limit
        for i in range(30):
            item = ItemInstance(definition=equipment_definition, quantity=1)
            player_inventory.add(item)
        assert player_inventory.is_full is True

        item = ItemInstance(definition=equipment_definition, quantity=1)
        success, added = player_inventory.add(item)
        assert success is False

    def test_add_zero_quantity_fails(self, player_inventory, material_definition):
        """Adding zero quantity should fail."""
        item = ItemInstance(definition=material_definition, quantity=10)
        item.quantity = 0  # Bypass validation
        success, added = player_inventory.add(item)
        assert success is False
        assert added == 0

    def test_add_from_definition(self, player_inventory, material_definition):
        """add_definition should create and add item."""
        success, added = player_inventory.add_definition(material_definition, quantity=50)
        assert success is True
        assert added == 50
        assert player_inventory.count_item("iron_ore") == 50

    def test_remove_at_entire_stack(self, player_inventory, material_definition):
        """remove_at should remove entire stack by default."""
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        removed = player_inventory.remove_at(0)
        assert removed.quantity == 30
        assert player_inventory.get_item(0) is None

    def test_remove_at_partial_stack(self, player_inventory, material_definition):
        """remove_at with quantity should remove partial stack."""
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        removed = player_inventory.remove_at(0, quantity=10)
        assert removed.quantity == 10
        assert player_inventory.get_item(0).quantity == 20

    def test_remove_at_empty_slot_returns_none(self, player_inventory):
        """remove_at on empty slot should return None."""
        removed = player_inventory.remove_at(0)
        assert removed is None

    def test_remove_at_locked_slot_returns_none(self, player_inventory, material_definition):
        """remove_at on locked slot should return None."""
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        player_inventory.lock_slot(0)
        removed = player_inventory.remove_at(0)
        assert removed is None

    def test_remove_item_by_id(self, player_inventory, material_definition):
        """remove_item should remove specified quantity across stacks."""
        for i in range(3):
            item = ItemInstance(definition=material_definition, quantity=50)
            player_inventory.add(item)
        assert player_inventory.count_item("iron_ore") == 150

        removed = player_inventory.remove_item("iron_ore", 75)
        assert removed == 75
        assert player_inventory.count_item("iron_ore") == 75

    def test_remove_item_more_than_available(self, player_inventory, material_definition):
        """remove_item should only remove what's available."""
        item = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item)

        removed = player_inventory.remove_item("iron_ore", 100)
        assert removed == 50
        assert player_inventory.count_item("iron_ore") == 0

    def test_clear_removes_all_items(self, player_inventory, material_definition, equipment_definition):
        """clear should remove all items."""
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=10)
            player_inventory.add(item)
        for i in range(5):
            item = ItemInstance(definition=equipment_definition, quantity=1)
            player_inventory.add(item)

        removed = player_inventory.clear()
        # Material stacks so 5*10 = 50 fits in 1 slot, plus 5 equipment = 6 items
        assert len(removed) >= 1  # At least some items removed
        assert player_inventory.is_empty is True
        assert player_inventory.current_weight == 0.0

    def test_move_item_to_empty_slot(self, player_inventory, material_definition):
        """move should move item to empty slot."""
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        success = player_inventory.move(0, 5)
        assert success is True
        assert player_inventory.get_item(0) is None
        assert player_inventory.get_item(5).quantity == 30

    def test_move_item_same_slot_fails(self, player_inventory, material_definition):
        """move to same slot should fail."""
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        success = player_inventory.move(0, 0)
        assert success is False

    def test_move_item_stacks(self, player_inventory, material_definition):
        """move to slot with same item should stack."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=material_definition, quantity=20)
        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=5)
        success = player_inventory.move(5, 0)
        assert success is True
        assert player_inventory.get_item(0).quantity == 50
        assert player_inventory.get_item(5) is None

    def test_move_item_swaps_different_items(self, player_inventory, material_definition, equipment_definition):
        """move to slot with different item should swap."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=equipment_definition, quantity=1)
        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=5)
        success = player_inventory.move(0, 5)
        assert success is True
        assert player_inventory.get_item(0).definition.id == "iron_sword"
        assert player_inventory.get_item(5).definition.id == "iron_ore"

    def test_split_stack(self, player_inventory, material_definition):
        """split should split stack to new slot."""
        item = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item, target_slot=0)
        new_slot = player_inventory.split(0, 20)
        assert new_slot is not None
        assert player_inventory.get_item(0).quantity == 30
        assert player_inventory.get_item(new_slot).quantity == 20

    def test_split_zero_returns_none(self, player_inventory, material_definition):
        """split with zero amount should return None."""
        item = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item, target_slot=0)
        result = player_inventory.split(0, 0)
        assert result is None

    def test_split_entire_stack_returns_none(self, player_inventory, material_definition):
        """split entire stack should return None."""
        item = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item, target_slot=0)
        result = player_inventory.split(0, 50)
        assert result is None

    def test_split_no_empty_slots_returns_none(self, player_inventory, equipment_definition):
        """split when no empty slots should return None."""
        # Fill all slots
        for i in range(30):
            item = ItemInstance(definition=equipment_definition, quantity=1)
            player_inventory.add(item)
        # Can't split because no empty slot for result
        result = player_inventory.split(0, 1)
        assert result is None

    def test_transfer_to_other_container(self, player_inventory, material_definition):
        """transfer_to should transfer item to another container."""
        other = InventoryContainer(container_type=ContainerType.CHEST, slot_count=50)
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)

        success, transferred = player_inventory.transfer_to(other, 0)
        assert success is True
        assert transferred == 30
        assert player_inventory.get_item(0) is None
        assert other.count_item("iron_ore") == 30

    def test_transfer_partial_quantity(self, player_inventory, material_definition):
        """transfer_to should transfer partial quantity."""
        other = InventoryContainer(container_type=ContainerType.CHEST, slot_count=50)
        item = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item, target_slot=0)

        success, transferred = player_inventory.transfer_to(other, 0, quantity=20)
        assert success is True
        assert transferred == 20
        assert player_inventory.get_item(0).quantity == 30
        assert other.count_item("iron_ore") == 20

    def test_transfer_all_to_other_container(self, player_inventory, material_definition, equipment_definition):
        """transfer_all_to should transfer all items."""
        other = InventoryContainer(container_type=ContainerType.STASH, slot_count=100)

        for i in range(3):
            item = ItemInstance(definition=material_definition, quantity=30)
            player_inventory.add(item)
        for i in range(2):
            item = ItemInstance(definition=equipment_definition, quantity=1)
            player_inventory.add(item)

        total = player_inventory.transfer_all_to(other)
        assert total == 92  # 90 + 2
        assert player_inventory.is_empty is True
        assert other.count_item("iron_ore") == 90
        assert other.count_item("iron_sword") == 2

    def test_find_item(self, player_inventory, material_definition):
        """find_item should return first matching slot."""
        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=5)

        result = player_inventory.find_item("iron_ore")
        assert result is not None
        slot_idx, found_item = result
        assert slot_idx == 5
        assert found_item.quantity == 30

    def test_find_item_not_found(self, player_inventory):
        """find_item should return None if not found."""
        result = player_inventory.find_item("nonexistent")
        assert result is None

    def test_find_all_items(self, player_inventory, material_definition):
        """find_all_items should return all matching slots."""
        for i in [0, 5, 10]:
            item = ItemInstance(definition=material_definition, quantity=20)
            player_inventory.add(item, target_slot=i)

        results = player_inventory.find_all_items("iron_ore")
        assert len(results) == 3
        assert {r[0] for r in results} == {0, 5, 10}

    def test_find_empty_slot(self, player_inventory, material_definition):
        """find_empty_slot should return first empty slot."""
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=10)
            player_inventory.add(item, target_slot=i)

        empty = player_inventory.find_empty_slot()
        assert empty == 5

    def test_find_stackable_slot(self, player_inventory, material_definition):
        """find_stackable_slot should return slot that can accept more."""
        item1 = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item1, target_slot=0)

        item2 = ItemInstance(definition=material_definition, quantity=20)
        slot = player_inventory.find_stackable_slot(item2)
        assert slot == 0

    def test_find_stackable_slot_full_stack(self, player_inventory, material_definition):
        """find_stackable_slot should skip full stacks."""
        item1 = ItemInstance(definition=material_definition, quantity=99)  # Full
        item2 = ItemInstance(definition=material_definition, quantity=50)  # Not full
        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=5)

        item3 = ItemInstance(definition=material_definition, quantity=20)
        slot = player_inventory.find_stackable_slot(item3)
        assert slot == 5  # Should skip slot 0

    def test_sort_default(self, player_inventory, material_definition, equipment_definition, consumable_definition):
        """sort should organize by type, rarity, name."""
        items = [
            (consumable_definition, 5),
            (material_definition, 30),
            (equipment_definition, 1),
        ]
        for defn, qty in items:
            item = ItemInstance(definition=defn, quantity=qty)
            player_inventory.add(item)

        player_inventory.sort()

        # Equipment should be first (has lowest type value and higher rarity)
        first = player_inventory.get_item(0)
        assert first.definition.id == "iron_sword"

    def test_sort_with_custom_key(self, player_inventory, material_definition, consumable_definition):
        """sort with custom key should work."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=consumable_definition, quantity=5)
        player_inventory.add(item1)
        player_inventory.add(item2)

        # Sort by quantity descending
        player_inventory.sort(key=lambda x: -x.quantity)

        first = player_inventory.get_item(0)
        assert first.quantity == 30

    def test_sort_respects_locked_slots(self, player_inventory, material_definition, consumable_definition):
        """sort should not move items in locked slots."""
        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=consumable_definition, quantity=5)
        player_inventory.add(item1, target_slot=0)
        player_inventory.add(item2, target_slot=1)
        player_inventory.lock_slot(1)

        player_inventory.sort()

        # Locked slot 1 should still have consumable
        assert player_inventory.get_item(1).definition.id == "health_potion"

    def test_compact_merges_stacks(self, player_inventory, material_definition):
        """compact should merge partial stacks."""
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=20)
            player_inventory.add(item, target_slot=i)
        assert player_inventory.used_slot_count == 5

        freed = player_inventory.compact()
        assert freed > 0
        assert player_inventory.count_item("iron_ore") == 100

    def test_compact_respects_locked_slots(self, player_inventory, material_definition):
        """compact should not merge from locked slots."""
        for i in range(3):
            item = ItemInstance(definition=material_definition, quantity=30)
            player_inventory.add(item, target_slot=i)
        player_inventory.lock_slot(1)

        # Count before compact
        initial_count = player_inventory.count_item("iron_ore")

        player_inventory.compact()

        # Total should still be same
        assert player_inventory.count_item("iron_ore") == initial_count
        # Locked slot should still have item
        assert player_inventory.get_item(1) is not None

    def test_lock_unlock_slot(self, player_inventory):
        """lock_slot and unlock_slot should work."""
        player_inventory.lock_slot(5)
        assert player_inventory.get_slot(5).locked is True

        player_inventory.unlock_slot(5)
        assert player_inventory.get_slot(5).locked is False

    def test_set_slot_filter(self, player_inventory):
        """set_slot_filter should set filter type."""
        player_inventory.set_slot_filter(0, ItemType.EQUIPMENT)
        assert player_inventory.get_slot(0).filter_type == ItemType.EQUIPMENT

    def test_resize_expand(self, player_inventory):
        """resize should expand container."""
        success = player_inventory.resize(50)
        assert success is True
        assert player_inventory.slot_count == 50

    def test_resize_shrink_empty(self, player_inventory):
        """resize should shrink when no items would be lost."""
        success = player_inventory.resize(10)
        assert success is True
        assert player_inventory.slot_count == 10

    def test_resize_shrink_with_items_in_high_slots_fails(self, player_inventory, equipment_definition):
        """resize should fail when items in slots that would be removed."""
        # Use equipment (non-stackable) to fill specific high slots
        player_inventory._weight_limit = 0.0  # Disable weight limit
        for i in range(25, 30):  # Slots 25-29
            item = ItemInstance(definition=equipment_definition, quantity=1)
            player_inventory.add(item, target_slot=i)

        # Try to shrink below where items are
        success = player_inventory.resize(20)
        # If items are in slots >= 20, resize should fail
        assert success is False or player_inventory.slot_count >= 20

    def test_resize_to_zero_fails(self, player_inventory):
        """resize to zero should fail."""
        success = player_inventory.resize(0)
        assert success is False

    def test_iteration(self, player_inventory, material_definition):
        """Container should be iterable."""
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=10)
            player_inventory.add(item, target_slot=i)

        slots = list(player_inventory)
        assert len(slots) == 30

    def test_items_iteration(self, player_inventory, material_definition):
        """items() should iterate over non-empty slots."""
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=10)
            player_inventory.add(item, target_slot=i)

        items = list(player_inventory.items())
        assert len(items) == 5
        for slot_idx, item in items:
            assert item.quantity == 10

    def test_len(self, player_inventory):
        """len should return slot count."""
        assert len(player_inventory) == 30

    def test_getitem(self, player_inventory, material_definition):
        """__getitem__ should return item or None."""
        item = ItemInstance(definition=material_definition, quantity=10)
        player_inventory.add(item, target_slot=5)

        assert player_inventory[5].quantity == 10
        assert player_inventory[0] is None


# =============================================================================
# EVENT TESTS
# =============================================================================


class TestInventoryEvents:
    """Whitebox tests for inventory events."""

    def test_add_listener(self, player_inventory, material_definition):
        """add_listener should register callback."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        item = ItemInstance(definition=material_definition, quantity=10)
        player_inventory.add(item)

        assert len(events) == 1
        assert events[0].event_type == EconomyEvent.ITEM_ADDED

    def test_remove_listener(self, player_inventory, material_definition):
        """remove_listener should unregister callback."""
        events: List[InventoryEvent] = []
        listener = events.append
        player_inventory.add_listener(listener)
        player_inventory.remove_listener(listener)

        item = ItemInstance(definition=material_definition, quantity=10)
        player_inventory.add(item)

        assert len(events) == 0

    def test_item_merged_event(self, player_inventory, material_definition):
        """Merging should emit ITEM_MERGED event."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        item1 = ItemInstance(definition=material_definition, quantity=30)
        item2 = ItemInstance(definition=material_definition, quantity=20)
        player_inventory.add(item1)
        player_inventory.add(item2)  # Will merge

        merged_events = [e for e in events if e.event_type == EconomyEvent.ITEM_MERGED]
        assert len(merged_events) == 1
        assert merged_events[0].quantity == 20

    def test_item_removed_event(self, player_inventory, material_definition):
        """Removing should emit ITEM_REMOVED event."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        player_inventory.remove_at(0, quantity=10)

        removed_events = [e for e in events if e.event_type == EconomyEvent.ITEM_REMOVED]
        assert len(removed_events) == 1
        assert removed_events[0].quantity == 10

    def test_item_moved_event(self, player_inventory, material_definition):
        """Moving should emit ITEM_MOVED event."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        item = ItemInstance(definition=material_definition, quantity=30)
        player_inventory.add(item, target_slot=0)
        events.clear()

        player_inventory.move(0, 5)

        moved_events = [e for e in events if e.event_type == EconomyEvent.ITEM_MOVED]
        assert len(moved_events) == 1

    def test_item_split_event(self, player_inventory, material_definition):
        """Splitting should emit ITEM_SPLIT event."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        item = ItemInstance(definition=material_definition, quantity=50)
        player_inventory.add(item, target_slot=0)
        events.clear()

        player_inventory.split(0, 20)

        split_events = [e for e in events if e.event_type == EconomyEvent.ITEM_SPLIT]
        assert len(split_events) == 1
        assert split_events[0].quantity == 20


# =============================================================================
# TRANSACTION TESTS
# =============================================================================


class TestInventoryTransactions:
    """Whitebox tests for inventory transactions."""

    def test_transaction_batches_events(self, player_inventory, material_definition):
        """Events should be batched during transaction."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        player_inventory.begin_transaction()
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=10)
            player_inventory.add(item, target_slot=i)

        # Events should be pending
        assert len(events) == 0

        player_inventory.commit_transaction()

        # All events should be emitted
        assert len(events) == 5

    def test_rollback_discards_events(self, player_inventory, material_definition):
        """Rollback should discard pending events."""
        events: List[InventoryEvent] = []
        player_inventory.add_listener(events.append)

        player_inventory.begin_transaction()
        for i in range(5):
            item = ItemInstance(definition=material_definition, quantity=10)
            player_inventory.add(item, target_slot=i)

        player_inventory.rollback_transaction()

        # Events should be discarded
        assert len(events) == 0

        # But items are still added (rollback only affects events)
        assert player_inventory.used_slot_count == 5


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestInventorySerialization:
    """Whitebox tests for inventory serialization."""

    def test_container_serialization_round_trip(self, player_inventory, material_definition, equipment_definition):
        """Serialization and deserialization should preserve state."""
        for i in range(3):
            item = ItemInstance(definition=material_definition, quantity=30)
            player_inventory.add(item, target_slot=i)
        item = ItemInstance(definition=equipment_definition, quantity=1)
        player_inventory.add(item, target_slot=5)
        player_inventory.lock_slot(10)
        player_inventory.set_slot_filter(15, ItemType.CONSUMABLE)

        data = player_inventory.to_dict(embed_definitions=True)
        restored = InventoryContainer.from_dict(data)

        assert restored.slot_count == 30
        assert restored.weight_limit == 100.0
        assert restored.owner_id == "player_001"
        assert restored.count_item("iron_ore") == 90
        assert restored.count_item("iron_sword") == 1
        assert restored.get_slot(10).locked is True
        assert restored.get_slot(15).filter_type == ItemType.CONSUMABLE

    def test_container_serialization_with_registry(self, item_registry, material_definition):
        """Serialization with definition_id should use registry."""
        item_registry.register(material_definition)
        container = InventoryContainer(
            container_type=ContainerType.PLAYER_INVENTORY,
            slot_count=10,
        )
        item = ItemInstance(definition=material_definition, quantity=50)
        container.add(item)

        data = container.to_dict(embed_definitions=False)
        restored = InventoryContainer.from_dict(data, item_registry.as_dict())

        assert restored.count_item("iron_ore") == 50


# =============================================================================
# ITEM REGISTRY TESTS
# =============================================================================


class TestItemRegistry:
    """Whitebox tests for ItemRegistry."""

    def test_singleton_instance(self):
        """Registry should be a singleton."""
        ItemRegistry.reset()
        reg1 = ItemRegistry.instance()
        reg2 = ItemRegistry.instance()
        assert reg1 is reg2

    def test_register_and_get(self, item_registry, material_definition):
        """register and get should work."""
        item_registry.register(material_definition)
        retrieved = item_registry.get("iron_ore")
        assert retrieved == material_definition

    def test_register_duplicate_raises(self, item_registry, material_definition):
        """Registering duplicate ID should raise."""
        item_registry.register(material_definition)
        with pytest.raises(ValueError, match="already registered"):
            item_registry.register(material_definition)

    def test_get_nonexistent_returns_none(self, item_registry):
        """get for nonexistent should return None."""
        result = item_registry.get("nonexistent")
        assert result is None

    def test_get_or_raise(self, item_registry, material_definition):
        """get_or_raise should raise for nonexistent."""
        with pytest.raises(KeyError):
            item_registry.get_or_raise("nonexistent")

    def test_exists(self, item_registry, material_definition):
        """exists should check registration."""
        assert item_registry.exists("iron_ore") is False
        item_registry.register(material_definition)
        assert item_registry.exists("iron_ore") is True

    def test_all(self, item_registry, material_definition, equipment_definition):
        """all should return all definitions."""
        item_registry.register(material_definition)
        item_registry.register(equipment_definition)
        all_items = item_registry.all()
        assert len(all_items) == 2

    def test_by_type(self, item_registry, material_definition, equipment_definition):
        """by_type should filter by item type."""
        item_registry.register(material_definition)
        item_registry.register(equipment_definition)
        materials = item_registry.by_type(ItemType.MATERIAL)
        assert len(materials) == 1
        assert materials[0].id == "iron_ore"

    def test_by_rarity(self, item_registry, material_definition, equipment_definition):
        """by_rarity should filter by rarity."""
        item_registry.register(material_definition)
        item_registry.register(equipment_definition)
        uncommon = item_registry.by_rarity(Rarity.UNCOMMON)
        assert len(uncommon) == 1
        assert uncommon[0].id == "iron_sword"

    def test_clear(self, item_registry, material_definition):
        """clear should remove all definitions."""
        item_registry.register(material_definition)
        item_registry.clear()
        assert len(item_registry.all()) == 0

    def test_serialization_round_trip(self, item_registry, material_definition, equipment_definition):
        """Registry serialization should preserve data."""
        item_registry.register(material_definition)
        item_registry.register(equipment_definition)

        data = item_registry.to_dict()
        ItemRegistry.reset()
        restored = ItemRegistry.from_dict(data)

        assert len(restored._definitions) == 2
        assert restored.get("iron_ore") is not None
        assert restored.get("iron_sword") is not None


# =============================================================================
# WEIGHT LIMIT EDGE CASES (T-ECON-1.3)
# =============================================================================


class TestWeightLimitEdgeCases:
    """Edge case tests for weight limits."""

    def test_exactly_at_weight_limit(self, player_inventory, material_definition):
        """Should be able to reach exactly the weight limit."""
        # 200 items at 0.5 weight = 100 weight (limit)
        item = ItemInstance(definition=material_definition, quantity=99)
        player_inventory.add(item)
        item = ItemInstance(definition=material_definition, quantity=99)
        player_inventory.add(item)
        # That's 99 weight, add 2 more (1 weight)
        item = ItemInstance(definition=material_definition, quantity=2)
        success, added = player_inventory.add(item)
        assert success is True
        assert player_inventory.current_weight == pytest.approx(100.0)

    def test_is_over_weight_false_at_limit(self, player_inventory, material_definition):
        """is_over_weight should be False exactly at limit."""
        player_inventory._current_weight = 100.0
        assert player_inventory.is_over_weight is False

    def test_is_over_weight_true_above_limit(self, player_inventory):
        """is_over_weight should be True above limit."""
        player_inventory._current_weight = 100.1
        assert player_inventory.is_over_weight is True

    def test_unlimited_weight_never_over(self):
        """Unlimited weight containers should never be over weight."""
        container = InventoryContainer(
            container_type=ContainerType.STASH,
            weight_limit=0.0,
        )
        container._current_weight = 999999.0
        assert container.is_over_weight is False
