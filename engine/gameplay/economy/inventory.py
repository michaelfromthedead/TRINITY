"""
Inventory Container System.

Provides inventory containers with slots, stacking, and item management operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from uuid import UUID, uuid4

from foundation import (
    register_type,
    to_dict as foundation_to_dict,
    from_dict as foundation_from_dict,
    schema_hash,
)

from .constants import (
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


# =============================================================================
# Serialization Decorators
# =============================================================================


# Schema version for economy components
ECONOMY_SCHEMA_VERSION = 1


def serializable(
    name: Optional[str] = None,
    version: int = ECONOMY_SCHEMA_VERSION,
    exclude_fields: Optional[Set[str]] = None,
) -> Callable[[Type], Type]:
    """Decorator to mark a class as serializable for session persistence.

    Registers the class with Foundation's Serializer and adds metadata
    for version tracking and field exclusion.

    Usage:
        @serializable(version=1)
        class Inventory:
            ...

    Args:
        name: Optional custom type name for registration.
        version: Schema version for migration support.
        exclude_fields: Fields to exclude from serialization.

    Returns:
        Decorated class registered with Foundation Serializer.
    """
    def decorator(cls: Type) -> Type:
        # Register with Foundation
        type_name = name or f"{cls.__module__}.{cls.__name__}"
        register_type(cls, type_name)

        # Add serialization metadata
        cls._serializable = True
        cls._serializable_version = version
        cls._serializable_exclude = exclude_fields or set()

        return cls

    return decorator


def transient(field_name: str) -> str:
    """Mark a field as transient (not persisted).

    Usage:
        @dataclass
        class MyClass:
            cached_lookup: Dict = field(default_factory=dict, metadata={"transient": True})

    This is a helper to document transient fields. The actual exclusion
    is handled by checking metadata in to_dict/from_dict methods.
    """
    return field_name


class Serializer:
    """Serializer utility for economy components.

    Provides static methods for serialization that wrap Foundation's
    serializer with economy-specific handling.
    """

    @staticmethod
    def to_dict(obj: Any, include_schema: bool = True) -> Dict[str, Any]:
        """Serialize an object to a dictionary.

        Args:
            obj: Object to serialize.
            include_schema: Whether to include schema hash.

        Returns:
            Dictionary representation.
        """
        return foundation_to_dict(obj, include_schema_hash=include_schema)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Any:
        """Deserialize an object from a dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Deserialized object.
        """
        return foundation_from_dict(data)


# =============================================================================
# Item Definition
# =============================================================================


@serializable(version=1)
@dataclass
class ItemDefinition:
    """
    Template definition for an item type.

    This is the static data that defines what an item is.
    """
    id: str
    name: str
    item_type: ItemType
    rarity: Rarity = Rarity.COMMON
    max_stack: int = 1
    weight: float = 0.0
    base_value: int = 0
    level_requirement: int = 1
    description: str = ""
    icon: str = ""
    model: str = ""
    flags: frozenset = field(default_factory=frozenset)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize definition."""
        if not self.id:
            raise ValueError("Item id cannot be empty")
        if not self.name:
            raise ValueError("Item name cannot be empty")

        # Auto-set stack limits based on type if not specified
        if self.max_stack <= 0:
            self.max_stack = DEFAULT_STACK_LIMITS.get(self.item_type, 1)

        # Clamp to maximum
        self.max_stack = min(self.max_stack, MAX_STACK_SIZE)

        # Ensure non-negative values
        self.weight = max(0.0, self.weight)
        self.base_value = max(0, self.base_value)
        self.level_requirement = max(1, self.level_requirement)

    @property
    def is_stackable(self) -> bool:
        """Check if this item type can stack."""
        return self.item_type in STACKABLE_TYPES and self.max_stack > 1

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemDefinition):
            return False
        return self.id == other.id

    def to_dict(self) -> Dict[str, Any]:
        """Serialize item definition to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "item_type": self.item_type.name,
            "rarity": self.rarity.name,
            "max_stack": self.max_stack,
            "weight": self.weight,
            "base_value": self.base_value,
            "level_requirement": self.level_requirement,
            "description": self.description,
            "icon": self.icon,
            "model": self.model,
            "flags": list(self.flags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemDefinition":
        """Deserialize item definition from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            item_type=ItemType[data["item_type"]],
            rarity=Rarity[data.get("rarity", "COMMON")],
            max_stack=data.get("max_stack", 1),
            weight=data.get("weight", 0.0),
            base_value=data.get("base_value", 0),
            level_requirement=data.get("level_requirement", 1),
            description=data.get("description", ""),
            icon=data.get("icon", ""),
            model=data.get("model", ""),
            flags=frozenset(data.get("flags", [])),
            metadata=data.get("metadata", {}),
        )


@serializable(version=1)
@dataclass
class ItemInstance:
    """
    An instance of an item in the game world.

    This represents a specific stack of items owned by an entity.
    """
    definition: ItemDefinition
    quantity: int = 1
    instance_id: UUID = field(default_factory=uuid4)
    bound_to: Optional[str] = None
    durability: Optional[float] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate instance."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.quantity > self.definition.max_stack:
            raise ValueError(
                f"Quantity {self.quantity} exceeds max stack {self.definition.max_stack}"
            )

    @property
    def item_id(self) -> str:
        """Get the item definition id."""
        return self.definition.id

    @property
    def total_weight(self) -> float:
        """Get total weight of this stack."""
        return self.definition.weight * self.quantity

    @property
    def total_value(self) -> int:
        """Get total value of this stack."""
        return self.definition.base_value * self.quantity

    @property
    def can_add_more(self) -> bool:
        """Check if more items can be added to this stack."""
        return self.quantity < self.definition.max_stack

    @property
    def space_remaining(self) -> int:
        """Get remaining stack space."""
        return self.definition.max_stack - self.quantity

    def can_stack_with(self, other: ItemInstance) -> bool:
        """Check if another instance can stack with this one."""
        if not self.definition.is_stackable:
            return False
        if self.definition.id != other.definition.id:
            return False
        if self.bound_to != other.bound_to:
            return False
        # Items with custom data may not stack
        if self.custom_data or other.custom_data:
            return self.custom_data == other.custom_data
        return True

    def split(self, amount: int) -> ItemInstance:
        """
        Split this stack and return a new instance with the split amount.

        Args:
            amount: Number of items to split off

        Returns:
            New ItemInstance with the split amount

        Raises:
            ValueError: If amount is invalid
        """
        if amount <= 0:
            raise ValueError("Split amount must be positive")
        if amount >= self.quantity:
            raise ValueError("Split amount must be less than total quantity")

        self.quantity -= amount
        return ItemInstance(
            definition=self.definition,
            quantity=amount,
            bound_to=self.bound_to,
            custom_data=dict(self.custom_data),
        )

    def merge_from(self, other: ItemInstance) -> int:
        """
        Merge another stack into this one.

        Args:
            other: Stack to merge from

        Returns:
            Amount that was merged (may be less than other.quantity if stack full)

        Raises:
            ValueError: If items cannot stack together
        """
        if not self.can_stack_with(other):
            raise ValueError("Items cannot be stacked together")

        space = self.space_remaining
        merge_amount = min(space, other.quantity)

        if merge_amount > 0:
            self.quantity += merge_amount
            other.quantity -= merge_amount

        return merge_amount

    def clone(self) -> ItemInstance:
        """Create a copy of this item instance with a new UUID."""
        return ItemInstance(
            definition=self.definition,
            quantity=self.quantity,
            bound_to=self.bound_to,
            durability=self.durability,
            custom_data=dict(self.custom_data),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize item instance to dictionary."""
        return {
            "instance_id": str(self.instance_id),
            "definition": self.definition.to_dict(),
            "quantity": self.quantity,
            "bound_to": self.bound_to,
            "durability": self.durability,
            "custom_data": self.custom_data,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        definition_registry: Optional[Dict[str, "ItemDefinition"]] = None,
    ) -> "ItemInstance":
        """Deserialize item instance from dictionary.

        Args:
            data: Dictionary representation.
            definition_registry: Optional registry to look up definitions by ID.
                If provided and data contains definition_id, uses registry.
                Otherwise, deserializes embedded definition.

        Returns:
            ItemInstance object.
        """
        # Support both embedded definition and ID reference
        if definition_registry and "definition_id" in data:
            definition = definition_registry.get(data["definition_id"])
            if definition is None:
                raise KeyError(f"Unknown item definition: {data['definition_id']}")
        elif "definition" in data:
            definition = ItemDefinition.from_dict(data["definition"])
        else:
            raise ValueError("Item instance data must contain 'definition' or 'definition_id'")

        return cls(
            definition=definition,
            quantity=data.get("quantity", 1),
            instance_id=UUID(data["instance_id"]) if "instance_id" in data else uuid4(),
            bound_to=data.get("bound_to"),
            durability=data.get("durability"),
            custom_data=data.get("custom_data", {}),
        )


# =============================================================================
# Inventory Slot
# =============================================================================


@serializable(version=1)
@dataclass
class InventorySlot:
    """A single slot in an inventory container."""
    index: int
    item: Optional[ItemInstance] = None
    locked: bool = False
    filter_type: Optional[ItemType] = None

    @property
    def is_empty(self) -> bool:
        """Check if slot is empty."""
        return self.item is None

    @property
    def is_available(self) -> bool:
        """Check if slot can accept items."""
        return not self.locked and self.is_empty

    def accepts(self, item: ItemInstance) -> bool:
        """Check if slot can accept the given item."""
        if self.locked:
            return False
        if self.filter_type is not None:
            if item.definition.item_type != self.filter_type:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize slot to dictionary."""
        return {
            "index": self.index,
            "item": self.item.to_dict() if self.item else None,
            "locked": self.locked,
            "filter_type": self.filter_type.name if self.filter_type else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        definition_registry: Optional[Dict[str, "ItemDefinition"]] = None,
    ) -> "InventorySlot":
        """Deserialize slot from dictionary."""
        item = None
        if data.get("item"):
            item = ItemInstance.from_dict(data["item"], definition_registry)

        filter_type = None
        if data.get("filter_type"):
            filter_type = ItemType[data["filter_type"]]

        return cls(
            index=data["index"],
            item=item,
            locked=data.get("locked", False),
            filter_type=filter_type,
        )


# =============================================================================
# Event System
# =============================================================================


@dataclass
class InventoryEvent:
    """Event emitted by inventory operations."""
    event_type: EconomyEvent
    container_id: UUID
    item: Optional[ItemInstance] = None
    slot_index: Optional[int] = None
    quantity: int = 0
    source_slot: Optional[int] = None
    target_slot: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[InventoryEvent], None]


# =============================================================================
# Inventory Container
# =============================================================================


@serializable(version=1, exclude_fields={"_listeners", "_pending_events", "_transaction_active"})
class InventoryContainer:
    """
    A container that holds items in slots.

    Supports adding, removing, moving, splitting, and merging items.
    """

    def __init__(
        self,
        container_type: ContainerType,
        slot_count: Optional[int] = None,
        weight_limit: Optional[float] = None,
        owner_id: Optional[str] = None,
        container_id: Optional[UUID] = None,
    ) -> None:
        """
        Initialize inventory container.

        Args:
            container_type: Type of container
            slot_count: Number of slots (uses default if None)
            weight_limit: Maximum weight (uses default if None, 0 = unlimited)
            owner_id: ID of entity that owns this container
            container_id: Unique ID for this container instance
        """
        self._id = container_id or uuid4()
        self._type = container_type
        self._owner_id = owner_id

        # Initialize slots
        slot_count = slot_count or DEFAULT_CONTAINER_SLOTS.get(container_type, 20)
        self._slots: List[InventorySlot] = [
            InventorySlot(index=i) for i in range(slot_count)
        ]

        # Weight management
        self._weight_limit = weight_limit if weight_limit is not None else \
            DEFAULT_WEIGHT_LIMITS.get(container_type, 0.0)
        self._current_weight: float = 0.0

        # Event listeners
        self._listeners: List[EventCallback] = []

        # Transaction tracking
        self._transaction_active: bool = False
        self._pending_events: List[InventoryEvent] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def id(self) -> UUID:
        """Get container unique ID."""
        return self._id

    @property
    def container_type(self) -> ContainerType:
        """Get container type."""
        return self._type

    @property
    def owner_id(self) -> Optional[str]:
        """Get owner entity ID."""
        return self._owner_id

    @property
    def slot_count(self) -> int:
        """Get total number of slots."""
        return len(self._slots)

    @property
    def weight_limit(self) -> float:
        """Get weight limit (0 = unlimited)."""
        return self._weight_limit

    @property
    def current_weight(self) -> float:
        """Get current total weight."""
        return self._current_weight

    @property
    def weight_available(self) -> float:
        """Get remaining weight capacity."""
        if self._weight_limit <= 0:
            return float('inf')
        return max(0.0, self._weight_limit - self._current_weight)

    @property
    def is_over_weight(self) -> bool:
        """Check if container is over weight limit."""
        if self._weight_limit <= 0:
            return False
        return self._current_weight > self._weight_limit

    @property
    def empty_slot_count(self) -> int:
        """Get number of empty slots."""
        return sum(1 for slot in self._slots if slot.is_empty)

    @property
    def used_slot_count(self) -> int:
        """Get number of used slots."""
        return self.slot_count - self.empty_slot_count

    @property
    def is_full(self) -> bool:
        """Check if all slots are occupied."""
        return self.empty_slot_count == 0

    @property
    def is_empty(self) -> bool:
        """Check if container has no items."""
        return self.used_slot_count == 0

    # -------------------------------------------------------------------------
    # Slot Access
    # -------------------------------------------------------------------------

    def get_slot(self, index: int) -> InventorySlot:
        """
        Get slot by index.

        Args:
            index: Slot index

        Returns:
            InventorySlot at index

        Raises:
            IndexError: If index out of range
        """
        if index < 0 or index >= len(self._slots):
            raise IndexError(f"Slot index {index} out of range")
        return self._slots[index]

    def get_item(self, index: int) -> Optional[ItemInstance]:
        """
        Get item in slot.

        Args:
            index: Slot index

        Returns:
            ItemInstance or None if empty
        """
        return self.get_slot(index).item

    def find_item(self, item_id: str) -> Optional[Tuple[int, ItemInstance]]:
        """
        Find first slot containing item type.

        Args:
            item_id: Item definition ID

        Returns:
            Tuple of (slot_index, item) or None if not found
        """
        for slot in self._slots:
            if slot.item and slot.item.item_id == item_id:
                return (slot.index, slot.item)
        return None

    def find_all_items(self, item_id: str) -> List[Tuple[int, ItemInstance]]:
        """
        Find all slots containing item type.

        Args:
            item_id: Item definition ID

        Returns:
            List of (slot_index, item) tuples
        """
        results = []
        for slot in self._slots:
            if slot.item and slot.item.item_id == item_id:
                results.append((slot.index, slot.item))
        return results

    def count_item(self, item_id: str) -> int:
        """
        Count total quantity of an item type.

        Args:
            item_id: Item definition ID

        Returns:
            Total quantity across all slots
        """
        total = 0
        for slot in self._slots:
            if slot.item and slot.item.item_id == item_id:
                total += slot.item.quantity
        return total

    def find_empty_slot(self) -> Optional[int]:
        """
        Find first empty available slot.

        Returns:
            Slot index or None if no empty slots
        """
        for slot in self._slots:
            if slot.is_available:
                return slot.index
        return None

    def find_stackable_slot(self, item: ItemInstance) -> Optional[int]:
        """
        Find a slot where item can stack.

        Args:
            item: Item to find stack for

        Returns:
            Slot index or None if no stackable slot
        """
        if not item.definition.is_stackable:
            return None

        for slot in self._slots:
            if slot.item and slot.item.can_stack_with(item) and slot.item.can_add_more:
                return slot.index
        return None

    # -------------------------------------------------------------------------
    # Add Operations
    # -------------------------------------------------------------------------

    def can_add(self, item: ItemInstance) -> bool:
        """
        Check if item can be added to container.

        Args:
            item: Item to check

        Returns:
            True if item can be added
        """
        # Check weight
        if self._weight_limit > 0:
            if item.total_weight > self.weight_available:
                return False

        # Check for stackable slot
        if self.find_stackable_slot(item) is not None:
            return True

        # Check for empty slot
        return self.find_empty_slot() is not None

    def add(
        self,
        item: ItemInstance,
        target_slot: Optional[int] = None,
        auto_stack: bool = True,
    ) -> Tuple[bool, int]:
        """
        Add item to container.

        Args:
            item: Item instance to add
            target_slot: Specific slot to add to (optional)
            auto_stack: Whether to auto-stack with existing items

        Returns:
            Tuple of (success, quantity_added)
        """
        if item.quantity <= 0:
            return (False, 0)

        # Check weight
        if self._weight_limit > 0 and item.total_weight > self.weight_available:
            return (False, 0)

        quantity_to_add = item.quantity
        quantity_added = 0

        # Target specific slot
        if target_slot is not None:
            slot = self.get_slot(target_slot)
            if slot.is_empty and slot.accepts(item):
                slot.item = item
                self._current_weight += item.total_weight
                self._emit_event(EconomyEvent.ITEM_ADDED, item, target_slot, item.quantity)
                return (True, item.quantity)
            elif slot.item and slot.item.can_stack_with(item):
                merged = slot.item.merge_from(item)
                if merged > 0:
                    self._current_weight += item.definition.weight * merged
                    self._emit_event(
                        EconomyEvent.ITEM_MERGED, slot.item, target_slot, merged
                    )
                    quantity_added += merged
                if item.quantity > 0:
                    return (False, quantity_added)
                return (True, quantity_added)
            return (False, 0)

        # Auto-stack first
        if auto_stack and item.definition.is_stackable:
            while item.quantity > 0:
                stack_slot_idx = self.find_stackable_slot(item)
                if stack_slot_idx is None:
                    break
                stack_slot = self._slots[stack_slot_idx]
                merged = stack_slot.item.merge_from(item)
                if merged > 0:
                    self._current_weight += item.definition.weight * merged
                    self._emit_event(
                        EconomyEvent.ITEM_MERGED, stack_slot.item, stack_slot_idx, merged
                    )
                    quantity_added += merged

        # Add remaining to empty slots
        while item.quantity > 0:
            empty_idx = self.find_empty_slot()
            if empty_idx is None:
                break

            slot = self._slots[empty_idx]
            if not slot.accepts(item):
                continue

            # Split if needed for weight limit
            add_amount = item.quantity
            if self._weight_limit > 0:
                weight_per = item.definition.weight
                if weight_per > 0:
                    max_by_weight = int(self.weight_available / weight_per)
                    add_amount = min(add_amount, max_by_weight)

            if add_amount <= 0:
                break

            if add_amount < item.quantity:
                new_item = item.split(item.quantity - add_amount)
                slot.item = new_item
                added = slot.item.quantity
                self._current_weight += slot.item.total_weight
                self._emit_event(EconomyEvent.ITEM_ADDED, slot.item, empty_idx, added)
                quantity_added += added
            else:
                # Clone the item so the slot owns its own reference
                cloned = item.clone()
                slot.item = cloned
                added = add_amount
                self._current_weight += cloned.total_weight
                self._emit_event(EconomyEvent.ITEM_ADDED, cloned, empty_idx, added)
                quantity_added += added
                item.quantity = 0

        return (quantity_added == quantity_to_add, quantity_added)

    def add_definition(
        self,
        definition: ItemDefinition,
        quantity: int = 1,
    ) -> Tuple[bool, int]:
        """
        Create and add item from definition.

        Args:
            definition: Item definition
            quantity: Quantity to add

        Returns:
            Tuple of (success, quantity_added)
        """
        item = ItemInstance(definition=definition, quantity=quantity)
        return self.add(item)

    # -------------------------------------------------------------------------
    # Remove Operations
    # -------------------------------------------------------------------------

    def remove_at(self, slot_index: int, quantity: Optional[int] = None) -> Optional[ItemInstance]:
        """
        Remove item from slot.

        Args:
            slot_index: Slot to remove from
            quantity: Amount to remove (None = all)

        Returns:
            Removed ItemInstance or None if slot empty
        """
        slot = self.get_slot(slot_index)
        if slot.is_empty or slot.locked:
            return None

        item = slot.item

        if quantity is None or quantity >= item.quantity:
            # Remove entire stack
            slot.item = None
            self._current_weight -= item.total_weight
            self._emit_event(EconomyEvent.ITEM_REMOVED, item, slot_index, item.quantity)
            return item
        else:
            # Split stack
            if quantity <= 0:
                return None
            removed = item.split(quantity)
            self._current_weight -= removed.total_weight
            self._emit_event(EconomyEvent.ITEM_REMOVED, removed, slot_index, quantity)
            return removed

    def remove_item(self, item_id: str, quantity: int = 1) -> int:
        """
        Remove quantity of item type from container.

        Args:
            item_id: Item definition ID
            quantity: Amount to remove

        Returns:
            Actual quantity removed
        """
        removed = 0
        remaining = quantity

        # Find all stacks of this item
        stacks = self.find_all_items(item_id)

        for slot_idx, item in stacks:
            if remaining <= 0:
                break

            take = min(remaining, item.quantity)
            self.remove_at(slot_idx, take)
            removed += take
            remaining -= take

        return removed

    def clear(self) -> List[ItemInstance]:
        """
        Remove all items from container.

        Returns:
            List of removed items
        """
        removed = []
        for slot in self._slots:
            if slot.item:
                removed.append(slot.item)
                self._emit_event(
                    EconomyEvent.ITEM_REMOVED, slot.item, slot.index, slot.item.quantity
                )
                slot.item = None

        self._current_weight = 0.0
        return removed

    # -------------------------------------------------------------------------
    # Move Operations
    # -------------------------------------------------------------------------

    def move(self, from_slot: int, to_slot: int) -> bool:
        """
        Move item between slots.

        Args:
            from_slot: Source slot index
            to_slot: Target slot index

        Returns:
            True if move successful
        """
        if from_slot == to_slot:
            return False

        source = self.get_slot(from_slot)
        target = self.get_slot(to_slot)

        if source.is_empty or source.locked or target.locked:
            return False

        if target.is_empty:
            if not target.accepts(source.item):
                return False
            target.item = source.item
            source.item = None
            self._emit_event(
                EconomyEvent.ITEM_MOVED,
                target.item,
                to_slot,
                target.item.quantity,
                metadata={"source_slot": from_slot}
            )
            return True

        # Try stacking
        if source.item.can_stack_with(target.item):
            merged = target.item.merge_from(source.item)
            if merged > 0:
                self._emit_event(
                    EconomyEvent.ITEM_MERGED, target.item, to_slot, merged
                )
            if source.item.quantity == 0:
                source.item = None
            return True

        # Swap items
        if source.accepts(target.item) and target.accepts(source.item):
            source.item, target.item = target.item, source.item
            self._emit_event(
                EconomyEvent.ITEM_MOVED,
                source.item,
                from_slot,
                source.item.quantity,
                metadata={"swapped_with": to_slot}
            )
            self._emit_event(
                EconomyEvent.ITEM_MOVED,
                target.item,
                to_slot,
                target.item.quantity,
                metadata={"swapped_with": from_slot}
            )
            return True

        return False

    def split(self, slot_index: int, quantity: int) -> Optional[int]:
        """
        Split stack into a new slot.

        Args:
            slot_index: Source slot
            quantity: Amount to split

        Returns:
            New slot index or None if failed
        """
        slot = self.get_slot(slot_index)
        if slot.is_empty or slot.locked:
            return None

        if quantity <= 0 or quantity >= slot.item.quantity:
            return None

        empty_idx = self.find_empty_slot()
        if empty_idx is None:
            return None

        new_item = slot.item.split(quantity)
        self._slots[empty_idx].item = new_item

        self._emit_event(EconomyEvent.ITEM_SPLIT, new_item, empty_idx, quantity)
        return empty_idx

    # -------------------------------------------------------------------------
    # Transfer Operations
    # -------------------------------------------------------------------------

    def transfer_to(
        self,
        target: InventoryContainer,
        slot_index: int,
        quantity: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """
        Transfer item to another container.

        Args:
            target: Target container
            slot_index: Source slot
            quantity: Amount to transfer (None = all)

        Returns:
            Tuple of (success, quantity_transferred)
        """
        slot = self.get_slot(slot_index)
        if slot.is_empty:
            return (False, 0)

        item = slot.item
        transfer_qty = quantity if quantity is not None else item.quantity
        transfer_qty = min(transfer_qty, item.quantity)

        if transfer_qty <= 0:
            return (False, 0)

        # Create copy for transfer
        if transfer_qty == item.quantity:
            transfer_item = item
        else:
            transfer_item = ItemInstance(
                definition=item.definition,
                quantity=transfer_qty,
                bound_to=item.bound_to,
                custom_data=dict(item.custom_data),
            )

        # Try to add to target
        success, added = target.add(transfer_item)

        if added > 0:
            # Remove from source
            if added >= item.quantity:
                slot.item = None
            else:
                item.quantity -= added
            self._current_weight -= item.definition.weight * added
            self._emit_event(EconomyEvent.ITEM_REMOVED, item, slot_index, added)

        return (added == transfer_qty, added)

    def transfer_all_to(self, target: InventoryContainer) -> int:
        """
        Transfer all items to another container.

        Args:
            target: Target container

        Returns:
            Total quantity transferred
        """
        total = 0
        for slot in self._slots:
            if slot.item:
                _, transferred = self.transfer_to(target, slot.index)
                total += transferred
        return total

    # -------------------------------------------------------------------------
    # Sorting
    # -------------------------------------------------------------------------

    def sort(
        self,
        key: Optional[Callable[[ItemInstance], Any]] = None,
        reverse: bool = False,
    ) -> None:
        """
        Sort items in container.

        Args:
            key: Sort key function (default: by type, rarity, name)
            reverse: Reverse sort order
        """
        # Collect all items
        items = []
        for slot in self._slots:
            if slot.item and not slot.locked:
                items.append(slot.item)
                slot.item = None

        # Default sort key
        if key is None:
            def default_key(item: ItemInstance):
                return (
                    item.definition.item_type.value,
                    -item.definition.rarity.value,
                    item.definition.name,
                )
            key = default_key

        # Sort items
        items.sort(key=key, reverse=reverse)

        # Place back
        item_idx = 0
        for slot in self._slots:
            if slot.locked:
                continue
            if item_idx < len(items):
                slot.item = items[item_idx]
                item_idx += 1

    def compact(self) -> int:
        """
        Merge all stackable items and compact to front.

        Returns:
            Number of slots freed
        """
        initial_used = self.used_slot_count

        # Group items by definition
        item_groups: Dict[str, List[ItemInstance]] = {}
        locked_items: Dict[int, ItemInstance] = {}

        for slot in self._slots:
            if slot.item:
                if slot.locked:
                    locked_items[slot.index] = slot.item
                else:
                    key = slot.item.item_id
                    if key not in item_groups:
                        item_groups[key] = []
                    item_groups[key].append(slot.item)
                slot.item = None

        # Merge stacks
        merged_items = []
        for item_id, items in item_groups.items():
            if not items:
                continue

            current = items[0]
            for other in items[1:]:
                if current.can_stack_with(other):
                    merged = current.merge_from(other)
                    if other.quantity > 0:
                        merged_items.append(current)
                        current = other
                else:
                    merged_items.append(current)
                    current = other
            merged_items.append(current)

        # Sort merged items
        merged_items.sort(key=lambda x: (
            x.definition.item_type.value,
            -x.definition.rarity.value,
            x.definition.name,
        ))

        # Place back
        item_idx = 0
        for slot in self._slots:
            if slot.index in locked_items:
                slot.item = locked_items[slot.index]
            elif item_idx < len(merged_items):
                slot.item = merged_items[item_idx]
                item_idx += 1

        return initial_used - self.used_slot_count

    # -------------------------------------------------------------------------
    # Slot Management
    # -------------------------------------------------------------------------

    def lock_slot(self, slot_index: int) -> None:
        """Lock a slot to prevent modifications."""
        self.get_slot(slot_index).locked = True

    def unlock_slot(self, slot_index: int) -> None:
        """Unlock a slot."""
        self.get_slot(slot_index).locked = False

    def set_slot_filter(
        self,
        slot_index: int,
        item_type: Optional[ItemType],
    ) -> None:
        """Set item type filter for slot."""
        self.get_slot(slot_index).filter_type = item_type

    def resize(self, new_size: int) -> bool:
        """
        Resize container.

        Args:
            new_size: New slot count

        Returns:
            True if resize successful (may fail if items would be lost)
        """
        if new_size < 1:
            return False

        if new_size < len(self._slots):
            # Check if items would be lost
            for i in range(new_size, len(self._slots)):
                if self._slots[i].item:
                    return False
            self._slots = self._slots[:new_size]
        else:
            # Expand
            for i in range(len(self._slots), new_size):
                self._slots.append(InventorySlot(index=i))

        return True

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __iter__(self) -> Iterator[InventorySlot]:
        """Iterate over all slots."""
        return iter(self._slots)

    def items(self) -> Iterator[Tuple[int, ItemInstance]]:
        """Iterate over non-empty slots."""
        for slot in self._slots:
            if slot.item:
                yield (slot.index, slot.item)

    def __len__(self) -> int:
        """Get slot count."""
        return len(self._slots)

    def __getitem__(self, index: int) -> Optional[ItemInstance]:
        """Get item by slot index."""
        return self.get_item(index)

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def add_listener(self, callback: EventCallback) -> None:
        """Add event listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: EventCallback) -> None:
        """Remove event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _emit_event(
        self,
        event_type: EconomyEvent,
        item: Optional[ItemInstance],
        slot_index: Optional[int],
        quantity: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit an inventory event."""
        event = InventoryEvent(
            event_type=event_type,
            container_id=self._id,
            item=item,
            slot_index=slot_index,
            quantity=quantity,
            metadata=metadata or {},
        )

        if self._transaction_active:
            self._pending_events.append(event)
        else:
            for listener in self._listeners:
                listener(event)

    # -------------------------------------------------------------------------
    # Transactions
    # -------------------------------------------------------------------------

    def begin_transaction(self) -> None:
        """Begin a transaction (batch events)."""
        self._transaction_active = True
        self._pending_events = []

    def commit_transaction(self) -> None:
        """Commit transaction and emit all events."""
        self._transaction_active = False
        for event in self._pending_events:
            for listener in self._listeners:
                listener(event)
        self._pending_events = []

    def rollback_transaction(self) -> None:
        """Rollback transaction (discard pending events).

        NOTE: State changes already applied during the transaction are NOT
        reverted. Only un-emitted pending events are discarded. Use with
        caution — if the caller needs a full state rollback, a deeper
        snapshot/restore mechanism is required.
        """
        self._transaction_active = False
        self._pending_events = []

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self, embed_definitions: bool = False) -> Dict[str, Any]:
        """Serialize container to dictionary.

        Args:
            embed_definitions: If True, embed full item definitions.
                If False, only store definition IDs (requires registry for restore).

        Returns:
            Dictionary representation.
        """
        return {
            "__version__": ECONOMY_SCHEMA_VERSION,
            "id": str(self._id),
            "type": self._type.name,
            "owner_id": self._owner_id,
            "weight_limit": self._weight_limit,
            "current_weight": self._current_weight,
            "slots": [slot.to_dict() if embed_definitions else {
                "index": slot.index,
                "locked": slot.locked,
                "filter_type": slot.filter_type.name if slot.filter_type else None,
                "item": self._item_to_dict(slot.item) if slot.item else None,
            } for slot in self._slots],
        }

    def _item_to_dict(self, item: ItemInstance) -> Dict[str, Any]:
        """Serialize item instance (reference mode)."""
        return {
            "instance_id": str(item.instance_id),
            "definition_id": item.definition.id,
            "quantity": item.quantity,
            "bound_to": item.bound_to,
            "durability": item.durability,
            "custom_data": item.custom_data,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        definition_registry: Optional[Dict[str, "ItemDefinition"]] = None,
    ) -> "InventoryContainer":
        """Deserialize container from dictionary.

        Args:
            data: Dictionary representation.
            definition_registry: Registry of item definitions for lookup.
                Required if items were saved with definition_id references.

        Returns:
            InventoryContainer instance.
        """
        container = cls(
            container_type=ContainerType[data["type"]],
            slot_count=len(data.get("slots", [])),
            weight_limit=data.get("weight_limit"),
            owner_id=data.get("owner_id"),
            container_id=UUID(data["id"]) if "id" in data else None,
        )

        # Restore slots
        for slot_data in data.get("slots", []):
            idx = slot_data["index"]
            if idx < len(container._slots):
                slot = container._slots[idx]
                slot.locked = slot_data.get("locked", False)

                filter_type_name = slot_data.get("filter_type")
                slot.filter_type = ItemType[filter_type_name] if filter_type_name else None

                item_data = slot_data.get("item")
                if item_data:
                    # Check if full item instance or just reference
                    if "definition" in item_data:
                        slot.item = ItemInstance.from_dict(item_data)
                    elif "definition_id" in item_data and definition_registry:
                        definition = definition_registry.get(item_data["definition_id"])
                        if definition:
                            slot.item = ItemInstance(
                                definition=definition,
                                quantity=item_data.get("quantity", 1),
                                instance_id=UUID(item_data["instance_id"]) if "instance_id" in item_data else uuid4(),
                                bound_to=item_data.get("bound_to"),
                                durability=item_data.get("durability"),
                                custom_data=item_data.get("custom_data", {}),
                            )
                        else:
                            # Skip items with unknown definitions
                            pass

        # Restore weight (recalculate to ensure accuracy)
        container._current_weight = sum(
            slot.item.total_weight for slot in container._slots if slot.item
        )

        return container


# =============================================================================
# Item Registry
# =============================================================================


class ItemRegistry:
    """Global registry of item definitions."""

    _instance: Optional[ItemRegistry] = None

    def __init__(self) -> None:
        self._definitions: Dict[str, ItemDefinition] = {}

    @classmethod
    def instance(cls) -> ItemRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ItemRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset registry (for testing)."""
        cls._instance = None

    def register(self, definition: ItemDefinition) -> None:
        """Register an item definition."""
        if definition.id in self._definitions:
            raise ValueError(f"Item '{definition.id}' already registered")
        self._definitions[definition.id] = definition

    def get(self, item_id: str) -> Optional[ItemDefinition]:
        """Get item definition by ID."""
        return self._definitions.get(item_id)

    def get_or_raise(self, item_id: str) -> ItemDefinition:
        """Get item definition or raise error."""
        definition = self.get(item_id)
        if definition is None:
            raise KeyError(f"Unknown item: {item_id}")
        return definition

    def exists(self, item_id: str) -> bool:
        """Check if item is registered."""
        return item_id in self._definitions

    def all(self) -> List[ItemDefinition]:
        """Get all registered definitions."""
        return list(self._definitions.values())

    def by_type(self, item_type: ItemType) -> List[ItemDefinition]:
        """Get definitions by item type."""
        return [d for d in self._definitions.values() if d.item_type == item_type]

    def by_rarity(self, rarity: Rarity) -> List[ItemDefinition]:
        """Get definitions by rarity."""
        return [d for d in self._definitions.values() if d.rarity == rarity]

    def clear(self) -> None:
        """Clear all definitions."""
        self._definitions.clear()

    def as_dict(self) -> Dict[str, ItemDefinition]:
        """Get definitions as dictionary for use with from_dict methods."""
        return dict(self._definitions)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize registry to dictionary."""
        return {
            "__version__": ECONOMY_SCHEMA_VERSION,
            "definitions": {
                item_id: definition.to_dict()
                for item_id, definition in self._definitions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ItemRegistry":
        """Deserialize registry from dictionary."""
        registry = cls()
        for item_id, def_data in data.get("definitions", {}).items():
            registry._definitions[item_id] = ItemDefinition.from_dict(def_data)
        return registry
