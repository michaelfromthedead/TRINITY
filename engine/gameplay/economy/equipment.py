"""
Equipment System.

Handles equippable items, equipment slots, stat bonuses, and visual attachments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from .constants import (
    AttributeType,
    DEFAULT_MAX_DURABILITY,
    EconomyEvent,
    EquipmentSlot,
    EXCLUSIVE_SLOTS,
    ItemType,
    MAX_RESISTANCE_PERCENT,
    Rarity,
    ResistanceType,
    UPGRADE_BONUS_PER_LEVEL,
)
from .inventory import ItemDefinition, ItemInstance


# =============================================================================
# Stat Modifiers
# =============================================================================


@dataclass(frozen=True)
class StatModifier:
    """A modifier to a stat value."""
    stat_type: AttributeType
    flat_bonus: float = 0.0
    percent_bonus: float = 0.0  # 0.1 = +10%
    multiplier: float = 1.0     # Final multiplier

    def apply(self, base_value: float) -> float:
        """Apply this modifier to a base value."""
        return (base_value + self.flat_bonus) * (1.0 + self.percent_bonus) * self.multiplier

    def combine(self, other: StatModifier) -> StatModifier:
        """Combine with another modifier for the same stat."""
        if self.stat_type != other.stat_type:
            raise ValueError("Cannot combine modifiers for different stats")
        return StatModifier(
            stat_type=self.stat_type,
            flat_bonus=self.flat_bonus + other.flat_bonus,
            percent_bonus=self.percent_bonus + other.percent_bonus,
            multiplier=self.multiplier * other.multiplier,
        )


@dataclass(frozen=True)
class ResistanceModifier:
    """A modifier to a resistance value."""
    resistance_type: ResistanceType
    flat_bonus: float = 0.0     # Direct addition
    percent_bonus: float = 0.0  # Percentage increase

    def apply(self, base_value: float, max_resistance: float = MAX_RESISTANCE_PERCENT) -> float:
        """Apply this modifier and clamp to maximum."""
        result = base_value + self.flat_bonus + self.percent_bonus
        return min(result, max_resistance)

    def combine(self, other: ResistanceModifier) -> ResistanceModifier:
        """Combine with another modifier."""
        if self.resistance_type != other.resistance_type:
            raise ValueError("Cannot combine modifiers for different resistances")
        return ResistanceModifier(
            resistance_type=self.resistance_type,
            flat_bonus=self.flat_bonus + other.flat_bonus,
            percent_bonus=self.percent_bonus + other.percent_bonus,
        )


# =============================================================================
# Special Effects
# =============================================================================


@dataclass(frozen=True)
class SpecialEffect:
    """A special effect that equipment can provide."""
    effect_id: str
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.effect_id, self.name))


# =============================================================================
# Equipment Stats
# =============================================================================


@dataclass
class EquipmentStats:
    """Stats provided by a piece of equipment."""
    armor: float = 0.0
    damage: float = 0.0
    attack_speed: float = 0.0
    block_chance: float = 0.0
    attribute_modifiers: Tuple[StatModifier, ...] = ()
    resistance_modifiers: Tuple[ResistanceModifier, ...] = ()
    special_effects: Tuple[SpecialEffect, ...] = ()

    def combine(self, other: EquipmentStats) -> EquipmentStats:
        """Combine with another equipment stats."""
        # Combine attribute modifiers
        attr_mods: Dict[AttributeType, StatModifier] = {}
        for mod in self.attribute_modifiers:
            attr_mods[mod.stat_type] = mod
        for mod in other.attribute_modifiers:
            if mod.stat_type in attr_mods:
                attr_mods[mod.stat_type] = attr_mods[mod.stat_type].combine(mod)
            else:
                attr_mods[mod.stat_type] = mod

        # Combine resistance modifiers
        res_mods: Dict[ResistanceType, ResistanceModifier] = {}
        for mod in self.resistance_modifiers:
            res_mods[mod.resistance_type] = mod
        for mod in other.resistance_modifiers:
            if mod.resistance_type in res_mods:
                res_mods[mod.resistance_type] = res_mods[mod.resistance_type].combine(mod)
            else:
                res_mods[mod.resistance_type] = mod

        # Combine special effects (no duplicates)
        effects: Set[SpecialEffect] = set(self.special_effects)
        effects.update(other.special_effects)

        return EquipmentStats(
            armor=self.armor + other.armor,
            damage=self.damage + other.damage,
            attack_speed=self.attack_speed + other.attack_speed,
            block_chance=self.block_chance + other.block_chance,
            attribute_modifiers=tuple(attr_mods.values()),
            resistance_modifiers=tuple(res_mods.values()),
            special_effects=tuple(effects),
        )


# =============================================================================
# Equipment Definition
# =============================================================================


@dataclass
class EquipmentDefinition(ItemDefinition):
    """Extended item definition for equipment."""
    slot: EquipmentSlot = EquipmentSlot.MAIN_HAND
    stats: EquipmentStats = field(default_factory=EquipmentStats)
    required_attributes: Dict[AttributeType, int] = field(default_factory=dict)
    visual_model: str = ""
    attachment_point: str = ""
    socket_count: int = 0
    set_id: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and set item type to equipment."""
        self.item_type = ItemType.EQUIPMENT
        self.max_stack = 1  # Equipment never stacks
        super().__post_init__()


@dataclass
class EquipmentInstance(ItemInstance):
    """Instance of an equipped item with dynamic stats."""
    enchantments: List[str] = field(default_factory=list)
    socketed_gems: List[str] = field(default_factory=list)
    upgrade_level: int = 0

    @property
    def equipment_def(self) -> EquipmentDefinition:
        """Get typed equipment definition."""
        return self.definition  # type: ignore

    @property
    def slot(self) -> EquipmentSlot:
        """Get equipment slot."""
        return self.equipment_def.slot

    @property
    def effective_stats(self) -> EquipmentStats:
        """Calculate effective stats including upgrades."""
        base = self.equipment_def.stats
        # Apply upgrade multiplier
        upgrade_mult = 1.0 + (self.upgrade_level * UPGRADE_BONUS_PER_LEVEL)
        return EquipmentStats(
            armor=base.armor * upgrade_mult,
            damage=base.damage * upgrade_mult,
            attack_speed=base.attack_speed,
            block_chance=base.block_chance,
            attribute_modifiers=base.attribute_modifiers,
            resistance_modifiers=base.resistance_modifiers,
            special_effects=base.special_effects,
        )


# =============================================================================
# Equipment Set Bonuses
# =============================================================================


@dataclass
class SetBonus:
    """Bonus granted for wearing multiple pieces of a set."""
    pieces_required: int
    stats: EquipmentStats
    description: str = ""


@dataclass
class EquipmentSet:
    """A set of equipment that provides bonuses when worn together."""
    set_id: str
    name: str
    piece_ids: FrozenSet[str]
    bonuses: Tuple[SetBonus, ...]

    def get_active_bonuses(self, equipped_piece_ids: Set[str]) -> List[SetBonus]:
        """Get bonuses active for the given equipped pieces."""
        equipped_count = len(self.piece_ids.intersection(equipped_piece_ids))
        return [b for b in self.bonuses if equipped_count >= b.pieces_required]


# =============================================================================
# Equipment Container
# =============================================================================


EquipChangeCallback = Callable[[EquipmentSlot, Optional[EquipmentInstance], Optional[EquipmentInstance]], None]


class EquipmentContainer:
    """
    Container for equipped items.

    Manages equipment slots, stat calculation, and visual attachments.
    """

    def __init__(
        self,
        owner_id: str,
        allowed_slots: Optional[Set[EquipmentSlot]] = None,
        container_id: Optional[UUID] = None,
    ) -> None:
        """
        Initialize equipment container.

        Args:
            owner_id: ID of the entity wearing equipment
            allowed_slots: Which slots are available (None = all)
            container_id: Unique container ID
        """
        self._id = container_id or uuid4()
        self._owner_id = owner_id
        self._allowed_slots = allowed_slots or set(EquipmentSlot)

        # Equipment by slot
        self._equipped: Dict[EquipmentSlot, Optional[EquipmentInstance]] = {
            slot: None for slot in self._allowed_slots
        }

        # Cached combined stats
        self._combined_stats: Optional[EquipmentStats] = None
        self._stats_dirty: bool = True

        # Set registry
        self._set_registry: Dict[str, EquipmentSet] = {}

        # Event listeners
        self._change_listeners: List[EquipChangeCallback] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def id(self) -> UUID:
        """Get container ID."""
        return self._id

    @property
    def owner_id(self) -> str:
        """Get owner ID."""
        return self._owner_id

    @property
    def combined_stats(self) -> EquipmentStats:
        """Get combined stats from all equipment."""
        if self._stats_dirty or self._combined_stats is None:
            self._recalculate_stats()
        return self._combined_stats  # type: ignore

    # -------------------------------------------------------------------------
    # Equip/Unequip Operations
    # -------------------------------------------------------------------------

    def can_equip(
        self,
        item: EquipmentInstance,
        slot: Optional[EquipmentSlot] = None,
        character_stats: Optional[Dict[AttributeType, int]] = None,
    ) -> Tuple[bool, str]:
        """
        Check if an item can be equipped.

        Args:
            item: Equipment to check
            slot: Target slot (uses item's default if None)
            character_stats: Character's attribute values for requirement check

        Returns:
            Tuple of (can_equip, reason_if_not)
        """
        target_slot = slot or item.slot

        # Check slot allowed
        if target_slot not in self._allowed_slots:
            return (False, f"Slot {target_slot.name} not available")

        # Check slot compatibility
        if slot and item.slot != slot:
            # Some items can go in multiple slots (e.g., ring in ring_1 or ring_2)
            compatible = self._is_slot_compatible(item.slot, slot)
            if not compatible:
                return (False, f"Item cannot be equipped in {slot.name}")

        # Check level requirement
        level = character_stats.get(AttributeType.WISDOM, 1) if character_stats else 1
        if item.definition.level_requirement > level:
            return (False, f"Requires level {item.definition.level_requirement}")

        # Check attribute requirements
        if character_stats:
            for attr, required in item.equipment_def.required_attributes.items():
                actual = character_stats.get(attr, 0)
                if actual < required:
                    return (False, f"Requires {required} {attr.name}")

        return (True, "")

    def _is_slot_compatible(self, item_slot: EquipmentSlot, target_slot: EquipmentSlot) -> bool:
        """Check if item slot is compatible with target slot."""
        # Same slot is always compatible
        if item_slot == target_slot:
            return True

        # Ring items can go in either ring slot
        ring_slots = {EquipmentSlot.RING_1, EquipmentSlot.RING_2}
        if item_slot in ring_slots and target_slot in ring_slots:
            return True

        # Trinket items can go in either trinket slot
        trinket_slots = {EquipmentSlot.TRINKET_1, EquipmentSlot.TRINKET_2}
        if item_slot in trinket_slots and target_slot in trinket_slots:
            return True

        return False

    def equip(
        self,
        item: EquipmentInstance,
        slot: Optional[EquipmentSlot] = None,
        force: bool = False,
    ) -> Tuple[bool, Optional[EquipmentInstance]]:
        """
        Equip an item.

        Args:
            item: Equipment to equip
            slot: Target slot (uses item's default if None)
            force: Skip requirement checks

        Returns:
            Tuple of (success, unequipped_item)
        """
        target_slot = slot or item.slot

        if not force:
            can, reason = self.can_equip(item, target_slot)
            if not can:
                return (False, None)

        # Handle exclusive slots (e.g., two-hand weapons)
        unequipped: List[EquipmentInstance] = []
        if target_slot in EXCLUSIVE_SLOTS:
            for exclusive in EXCLUSIVE_SLOTS[target_slot]:
                if exclusive in self._equipped and self._equipped[exclusive]:
                    unequipped.append(self._equipped[exclusive])
                    self._equipped[exclusive] = None

        # Also check reverse - if equipping main/off when two-hand equipped
        for slot_key, exclusive_set in EXCLUSIVE_SLOTS.items():
            if target_slot in exclusive_set:
                if slot_key in self._equipped and self._equipped[slot_key]:
                    unequipped.append(self._equipped[slot_key])
                    self._equipped[slot_key] = None

        # Unequip existing item in target slot
        old_item = self._equipped.get(target_slot)
        if old_item:
            unequipped.insert(0, old_item)

        # Equip new item
        self._equipped[target_slot] = item
        self._stats_dirty = True

        # Notify listeners
        for listener in self._change_listeners:
            listener(target_slot, old_item, item)

        return (True, unequipped[0] if unequipped else None)

    def unequip(self, slot: EquipmentSlot) -> Optional[EquipmentInstance]:
        """
        Unequip item from slot.

        Args:
            slot: Slot to unequip

        Returns:
            Unequipped item or None if slot was empty
        """
        if slot not in self._equipped:
            return None

        item = self._equipped[slot]
        if item:
            self._equipped[slot] = None
            self._stats_dirty = True

            # Notify listeners
            for listener in self._change_listeners:
                listener(slot, item, None)

        return item

    def unequip_all(self) -> List[EquipmentInstance]:
        """
        Unequip all items.

        Returns:
            List of unequipped items
        """
        items = []
        for slot in list(self._equipped.keys()):
            item = self.unequip(slot)
            if item:
                items.append(item)
        return items

    def swap(self, slot: EquipmentSlot, new_item: EquipmentInstance) -> Optional[EquipmentInstance]:
        """
        Swap equipment in slot.

        Args:
            slot: Target slot
            new_item: New equipment

        Returns:
            Old equipment or None
        """
        old = self.unequip(slot)
        success, _ = self.equip(new_item, slot)
        if not success and old:
            self.equip(old, slot, force=True)
            return None
        return old

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    def get(self, slot: EquipmentSlot) -> Optional[EquipmentInstance]:
        """Get equipment in slot."""
        return self._equipped.get(slot)

    def is_slot_empty(self, slot: EquipmentSlot) -> bool:
        """Check if slot is empty."""
        return self._equipped.get(slot) is None

    def get_all_equipped(self) -> List[Tuple[EquipmentSlot, EquipmentInstance]]:
        """Get all equipped items."""
        return [
            (slot, item)
            for slot, item in self._equipped.items()
            if item is not None
        ]

    def get_equipped_ids(self) -> Set[str]:
        """Get set of equipped item IDs."""
        return {
            item.item_id
            for item in self._equipped.values()
            if item is not None
        }

    def find_by_id(self, item_id: str) -> Optional[Tuple[EquipmentSlot, EquipmentInstance]]:
        """Find equipped item by definition ID."""
        for slot, item in self._equipped.items():
            if item and item.item_id == item_id:
                return (slot, item)
        return None

    # -------------------------------------------------------------------------
    # Stat Calculation
    # -------------------------------------------------------------------------

    def _recalculate_stats(self) -> None:
        """Recalculate combined stats from all equipment."""
        base = EquipmentStats()

        for slot, item in self._equipped.items():
            if item:
                base = base.combine(item.effective_stats)

        # Add set bonuses
        equipped_ids = self.get_equipped_ids()
        for equipment_set in self._set_registry.values():
            for bonus in equipment_set.get_active_bonuses(equipped_ids):
                base = base.combine(bonus.stats)

        self._combined_stats = base
        self._stats_dirty = False

    def get_attribute_modifier(self, attr: AttributeType) -> Optional[StatModifier]:
        """Get combined modifier for an attribute."""
        for mod in self.combined_stats.attribute_modifiers:
            if mod.stat_type == attr:
                return mod
        return None

    def get_resistance_modifier(self, res: ResistanceType) -> Optional[ResistanceModifier]:
        """Get combined modifier for a resistance."""
        for mod in self.combined_stats.resistance_modifiers:
            if mod.resistance_type == res:
                return mod
        return None

    def get_total_armor(self) -> float:
        """Get total armor value."""
        return self.combined_stats.armor

    def get_total_damage(self) -> float:
        """Get total damage bonus."""
        return self.combined_stats.damage

    def has_effect(self, effect_id: str) -> bool:
        """Check if any equipment provides an effect."""
        return any(
            e.effect_id == effect_id
            for e in self.combined_stats.special_effects
        )

    def get_effects(self) -> Tuple[SpecialEffect, ...]:
        """Get all active special effects."""
        return self.combined_stats.special_effects

    # -------------------------------------------------------------------------
    # Set Bonuses
    # -------------------------------------------------------------------------

    def register_set(self, equipment_set: EquipmentSet) -> None:
        """Register an equipment set."""
        self._set_registry[equipment_set.set_id] = equipment_set
        self._stats_dirty = True

    def get_active_set_bonuses(self) -> List[Tuple[EquipmentSet, SetBonus]]:
        """Get all active set bonuses."""
        equipped_ids = self.get_equipped_ids()
        result = []
        for equipment_set in self._set_registry.values():
            for bonus in equipment_set.get_active_bonuses(equipped_ids):
                result.append((equipment_set, bonus))
        return result

    # -------------------------------------------------------------------------
    # Visual Attachments
    # -------------------------------------------------------------------------

    def get_visual_attachments(self) -> List[Tuple[str, str]]:
        """
        Get visual attachment data for rendering.

        Returns:
            List of (attachment_point, model_path) tuples
        """
        attachments = []
        for slot, item in self._equipped.items():
            if item:
                eq_def = item.equipment_def
                if eq_def.visual_model:
                    point = eq_def.attachment_point or slot.name.lower()
                    attachments.append((point, eq_def.visual_model))
        return attachments

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def add_change_listener(self, callback: EquipChangeCallback) -> None:
        """Add equipment change listener."""
        self._change_listeners.append(callback)

    def remove_change_listener(self, callback: EquipChangeCallback) -> None:
        """Remove equipment change listener."""
        if callback in self._change_listeners:
            self._change_listeners.remove(callback)

    # -------------------------------------------------------------------------
    # Durability
    # -------------------------------------------------------------------------

    def reduce_durability(self, slot: EquipmentSlot, amount: float) -> bool:
        """
        Reduce durability of equipment in slot.

        Args:
            slot: Slot to reduce durability
            amount: Durability to reduce

        Returns:
            True if item broke (durability reached 0)
        """
        item = self._equipped.get(slot)
        if not item or item.durability is None:
            return False

        item.durability = max(0.0, item.durability - amount)
        if item.durability <= 0:
            self.unequip(slot)
            return True
        return False

    def reduce_all_durability(self, amount: float) -> List[EquipmentSlot]:
        """
        Reduce durability of all equipment.

        Args:
            amount: Durability to reduce

        Returns:
            List of slots where equipment broke
        """
        broken = []
        for slot in list(self._equipped.keys()):
            if self.reduce_durability(slot, amount):
                broken.append(slot)
        return broken

    def repair(self, slot: EquipmentSlot, amount: Optional[float] = None) -> float:
        """
        Repair equipment in slot.

        Args:
            slot: Slot to repair
            amount: Amount to repair (None = full repair)

        Returns:
            Amount actually repaired
        """
        item = self._equipped.get(slot)
        if not item or item.durability is None:
            return 0.0

        max_durability = DEFAULT_MAX_DURABILITY
        if amount is None:
            repaired = max_durability - item.durability
            item.durability = max_durability
        else:
            old = item.durability
            item.durability = min(max_durability, item.durability + amount)
            repaired = item.durability - old

        return repaired

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": str(self._id),
            "owner_id": self._owner_id,
            "equipped": {
                slot.name: self._item_to_dict(item) if item else None
                for slot, item in self._equipped.items()
            },
        }

    def _item_to_dict(self, item: EquipmentInstance) -> Dict[str, Any]:
        """Serialize equipment instance."""
        return {
            "instance_id": str(item.instance_id),
            "definition_id": item.definition.id,
            "durability": item.durability,
            "enchantments": item.enchantments,
            "socketed_gems": item.socketed_gems,
            "upgrade_level": item.upgrade_level,
            "custom_data": item.custom_data,
        }


# =============================================================================
# Equipment Registry
# =============================================================================


class EquipmentRegistry:
    """Registry for equipment definitions and sets."""

    _instance: Optional[EquipmentRegistry] = None

    def __init__(self) -> None:
        self._definitions: Dict[str, EquipmentDefinition] = {}
        self._sets: Dict[str, EquipmentSet] = {}

    @classmethod
    def instance(cls) -> EquipmentRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = EquipmentRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset registry (for testing)."""
        cls._instance = None

    def register_equipment(self, definition: EquipmentDefinition) -> None:
        """Register equipment definition."""
        if definition.id in self._definitions:
            raise ValueError(f"Equipment '{definition.id}' already registered")
        self._definitions[definition.id] = definition

    def register_set(self, equipment_set: EquipmentSet) -> None:
        """Register equipment set."""
        if equipment_set.set_id in self._sets:
            raise ValueError(f"Set '{equipment_set.set_id}' already registered")
        self._sets[equipment_set.set_id] = equipment_set

    def get_equipment(self, item_id: str) -> Optional[EquipmentDefinition]:
        """Get equipment definition."""
        return self._definitions.get(item_id)

    def get_set(self, set_id: str) -> Optional[EquipmentSet]:
        """Get equipment set."""
        return self._sets.get(set_id)

    def get_by_slot(self, slot: EquipmentSlot) -> List[EquipmentDefinition]:
        """Get all equipment for a slot."""
        return [d for d in self._definitions.values() if d.slot == slot]

    def clear(self) -> None:
        """Clear all registrations."""
        self._definitions.clear()
        self._sets.clear()
