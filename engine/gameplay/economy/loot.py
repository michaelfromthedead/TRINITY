"""
Loot System.

Provides loot tables with weighted entries, conditions, nested tables,
loot rolling with RNG, pity system, and luck bonuses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)
from uuid import UUID, uuid4

from .constants import (
    DEFAULT_MAX_DROPS,
    DEFAULT_MAX_LEVEL,
    DEFAULT_MAX_VALUE,
    DEFAULT_MIN_LEVEL,
    LUCK_BONUS_PER_POINT,
    MAX_LUCK_BONUS,
    PITY_INCREMENT,
    PITY_RESET_ON_SUCCESS,
    PITY_WEIGHT_BOOST,
    RARITY_DROP_WEIGHTS,
    RARITY_PITY_THRESHOLDS,
    Rarity,
)
from .inventory import ItemDefinition, ItemInstance


# =============================================================================
# Random Source Protocol
# =============================================================================


class RandomSource(Protocol):
    """Protocol for random number generation (for determinism)."""

    def random(self) -> float:
        """Return random float in [0.0, 1.0)."""
        ...

    def randint(self, a: int, b: int) -> int:
        """Return random integer in [a, b] inclusive."""
        ...

    def choice(self, seq: List[Any]) -> Any:
        """Return random element from sequence."""
        ...


class DefaultRandomSource:
    """Default random source using Python's random module."""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: List[Any]) -> Any:
        return self._rng.choice(seq)


class SeededRandomSource:
    """Seeded random source for deterministic loot."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def random(self) -> float:
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def choice(self, seq: List[Any]) -> Any:
        return self._rng.choice(seq)


# =============================================================================
# Loot Conditions
# =============================================================================


@dataclass(frozen=True)
class LootCondition:
    """
    Base condition for loot drops.

    This is an abstract base class. Subclasses must implement evaluate().
    Subclasses use __post_init__ to set the condition_type field.
    """
    condition_type: str = field(default="base", init=False)

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition against context.

        Subclasses must override this method.

        Args:
            context: Dictionary containing evaluation context

        Returns:
            True if condition is met, False otherwise

        Raises:
            NotImplementedError: If called on the base class directly
        """
        raise NotImplementedError(
            f"LootCondition.evaluate() must be implemented by subclass. "
            f"Got condition_type='{self.condition_type}'"
        )


@dataclass(frozen=True)
class LevelCondition(LootCondition):
    """Condition based on player/enemy level."""
    min_level: int = DEFAULT_MIN_LEVEL
    max_level: int = DEFAULT_MAX_LEVEL

    def __post_init__(self):
        object.__setattr__(self, 'condition_type', 'level')

    def evaluate(self, context: Dict[str, Any]) -> bool:
        level = context.get('level', 1)
        return self.min_level <= level <= self.max_level


@dataclass(frozen=True)
class QuestCondition(LootCondition):
    """Condition based on quest state."""
    quest_id: str
    required_state: str = 'completed'

    def __post_init__(self):
        object.__setattr__(self, 'condition_type', 'quest')

    def evaluate(self, context: Dict[str, Any]) -> bool:
        quests = context.get('quests', {})
        return quests.get(self.quest_id) == self.required_state


@dataclass(frozen=True)
class FlagCondition(LootCondition):
    """Condition based on a boolean flag."""
    flag_name: str
    expected_value: bool = True

    def __post_init__(self):
        object.__setattr__(self, 'condition_type', 'flag')

    def evaluate(self, context: Dict[str, Any]) -> bool:
        flags = context.get('flags', {})
        return flags.get(self.flag_name, False) == self.expected_value


@dataclass(frozen=True)
class AttributeCondition(LootCondition):
    """Condition based on character attribute."""
    attribute: str
    min_value: int = 0
    max_value: int = DEFAULT_MAX_VALUE

    def __post_init__(self):
        object.__setattr__(self, 'condition_type', 'attribute')

    def evaluate(self, context: Dict[str, Any]) -> bool:
        attributes = context.get('attributes', {})
        value = attributes.get(self.attribute, 0)
        return self.min_value <= value <= self.max_value


@dataclass(frozen=True)
class RandomChanceCondition(LootCondition):
    """Condition with random chance."""
    chance: float  # 0.0 to 1.0

    def __post_init__(self):
        object.__setattr__(self, 'condition_type', 'random_chance')

    def evaluate(self, context: Dict[str, Any]) -> bool:
        rng = context.get('rng', random)
        return rng.random() < self.chance


# =============================================================================
# Loot Entries
# =============================================================================


@dataclass
class LootEntry:
    """A single entry in a loot table."""
    item_id: str
    weight: float = 1.0
    min_quantity: int = 1
    max_quantity: int = 1
    conditions: Tuple[LootCondition, ...] = ()
    guaranteed: bool = False
    unique: bool = False  # Can only drop once per roll session

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError("Weight cannot be negative")
        if self.min_quantity < 1:
            raise ValueError("min_quantity must be at least 1")
        if self.max_quantity < self.min_quantity:
            raise ValueError("max_quantity must be >= min_quantity")

    def check_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if all conditions are met."""
        return all(cond.evaluate(context) for cond in self.conditions)

    def roll_quantity(self, rng: RandomSource) -> int:
        """Roll quantity within range."""
        return rng.randint(self.min_quantity, self.max_quantity)


@dataclass
class NestedTableEntry:
    """Entry that references another loot table."""
    table_id: str
    weight: float = 1.0
    conditions: Tuple[LootCondition, ...] = ()
    rolls_override: Optional[int] = None

    def check_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if all conditions are met."""
        return all(cond.evaluate(context) for cond in self.conditions)


@dataclass
class CurrencyEntry:
    """Entry for currency drops."""
    currency_type: str
    min_amount: int
    max_amount: int
    weight: float = 1.0
    conditions: Tuple[LootCondition, ...] = ()

    def check_conditions(self, context: Dict[str, Any]) -> bool:
        return all(cond.evaluate(context) for cond in self.conditions)

    def roll_amount(self, rng: RandomSource) -> int:
        return rng.randint(self.min_amount, self.max_amount)


LootTableEntry = Union[LootEntry, NestedTableEntry, CurrencyEntry]


# =============================================================================
# Loot Result
# =============================================================================


@dataclass
class LootDrop:
    """A single loot drop result."""
    item_id: str
    quantity: int
    rarity: Optional[Rarity] = None
    source_table: Optional[str] = None
    was_pity: bool = False


@dataclass
class CurrencyDrop:
    """A currency drop result."""
    currency_type: str
    amount: int
    source_table: Optional[str] = None


@dataclass
class LootResult:
    """Result of rolling a loot table."""
    items: List[LootDrop] = field(default_factory=list)
    currencies: List[CurrencyDrop] = field(default_factory=list)
    rolls_performed: int = 0
    pity_triggered: bool = False


# =============================================================================
# Pity System
# =============================================================================


@dataclass
class PityTracker:
    """
    Tracks pity counters for guaranteed rare drops.

    The pity system ensures players eventually get rare items
    after a certain number of unsuccessful attempts.
    """
    counters: Dict[Rarity, int] = field(default_factory=dict)

    def increment(self, target_rarity: Rarity) -> None:
        """Increment pity counter for failed roll."""
        for rarity in Rarity:
            if rarity >= target_rarity:
                self.counters[rarity] = self.counters.get(rarity, 0) + PITY_INCREMENT

    def check_pity(self, rarity: Rarity) -> bool:
        """Check if pity should trigger for rarity."""
        threshold = RARITY_PITY_THRESHOLDS.get(rarity, 0)
        if threshold == 0:
            return False
        return self.counters.get(rarity, 0) >= threshold

    def reset(self, rarity: Rarity) -> None:
        """Reset pity counter for rarity (and lower rarities)."""
        if PITY_RESET_ON_SUCCESS:
            for r in Rarity:
                if r <= rarity:
                    self.counters[r] = 0

    def get_progress(self, rarity: Rarity) -> Tuple[int, int]:
        """Get pity progress (current, threshold)."""
        threshold = RARITY_PITY_THRESHOLDS.get(rarity, 0)
        current = self.counters.get(rarity, 0)
        return (current, threshold)


# =============================================================================
# Loot Table
# =============================================================================


@dataclass
class LootTable:
    """
    A table of possible loot drops with weights and conditions.

    Supports:
    - Weighted random selection
    - Conditional entries
    - Nested tables
    - Guaranteed drops
    - Pity system integration
    - Luck bonuses
    """
    table_id: str
    entries: List[LootTableEntry] = field(default_factory=list)
    rolls: int = 1
    guaranteed_entries: List[LootEntry] = field(default_factory=list)
    empty_weight: float = 0.0  # Weight for "nothing drops"
    min_drops: int = 0
    max_drops: int = DEFAULT_MAX_DROPS
    unique_drops: bool = True  # Each item can only drop once per roll

    def add_entry(self, entry: LootTableEntry) -> None:
        """Add an entry to the table."""
        self.entries.append(entry)

    def add_guaranteed(self, entry: LootEntry) -> None:
        """Add a guaranteed drop."""
        self.guaranteed_entries.append(entry)


# =============================================================================
# Loot Roller
# =============================================================================


class LootRoller:
    """
    Rolls loot tables to generate drops.

    Features:
    - Weighted random selection
    - Luck bonus scaling
    - Pity system for guaranteed rares
    - Nested table resolution
    - Drop deduplication
    """

    def __init__(
        self,
        rng: Optional[RandomSource] = None,
        item_registry: Optional[Dict[str, ItemDefinition]] = None,
    ):
        """
        Initialize loot roller.

        Args:
            rng: Random source (uses default if None)
            item_registry: Item definition registry for validation
        """
        self._rng = rng or DefaultRandomSource()
        self._item_registry = item_registry or {}
        self._table_registry: Dict[str, LootTable] = {}
        self._pity_trackers: Dict[str, PityTracker] = {}

    # -------------------------------------------------------------------------
    # Table Management
    # -------------------------------------------------------------------------

    def register_table(self, table: LootTable) -> None:
        """Register a loot table."""
        self._table_registry[table.table_id] = table

    def get_table(self, table_id: str) -> Optional[LootTable]:
        """Get registered loot table."""
        return self._table_registry.get(table_id)

    def get_or_create_pity(self, entity_id: str) -> PityTracker:
        """Get or create pity tracker for an entity."""
        if entity_id not in self._pity_trackers:
            self._pity_trackers[entity_id] = PityTracker()
        return self._pity_trackers[entity_id]

    # -------------------------------------------------------------------------
    # Roll Operations
    # -------------------------------------------------------------------------

    def roll(
        self,
        table: Union[str, LootTable],
        context: Optional[Dict[str, Any]] = None,
        luck: float = 0.0,
        entity_id: Optional[str] = None,
        rolls_override: Optional[int] = None,
    ) -> LootResult:
        """
        Roll a loot table.

        Args:
            table: Table ID or LootTable instance
            context: Context for condition evaluation
            luck: Luck value for bonus calculations
            entity_id: Entity ID for pity tracking
            rolls_override: Override number of rolls

        Returns:
            LootResult with all drops
        """
        # Resolve table
        if isinstance(table, str):
            resolved = self._table_registry.get(table)
            if not resolved:
                raise ValueError(f"Unknown loot table: {table}")
            table = resolved

        context = context or {}
        context['rng'] = self._rng

        # Get pity tracker
        pity = self.get_or_create_pity(entity_id) if entity_id else PityTracker()

        # Calculate luck bonus
        luck_bonus = min(luck * LUCK_BONUS_PER_POINT, MAX_LUCK_BONUS)

        result = LootResult()
        dropped_items: set = set()
        num_rolls = rolls_override if rolls_override is not None else table.rolls

        # Process guaranteed drops first
        for entry in table.guaranteed_entries:
            if entry.check_conditions(context):
                drop = self._create_drop(entry, table.table_id)
                result.items.append(drop)
                dropped_items.add(entry.item_id)

        # Perform random rolls
        for _ in range(num_rolls):
            drop = self._roll_once(
                table, context, luck_bonus, pity, dropped_items
            )
            if drop:
                if isinstance(drop, LootDrop):
                    result.items.append(drop)
                    if table.unique_drops:
                        dropped_items.add(drop.item_id)
                elif isinstance(drop, CurrencyDrop):
                    result.currencies.append(drop)
                elif isinstance(drop, LootResult):
                    # Nested table result
                    result.items.extend(drop.items)
                    result.currencies.extend(drop.currencies)
                    result.pity_triggered = result.pity_triggered or drop.pity_triggered

            result.rolls_performed += 1

        # Enforce min/max drops
        if len(result.items) < table.min_drops:
            # Roll more times to meet minimum
            attempts = 0
            while len(result.items) < table.min_drops and attempts < 100:
                drop = self._roll_once(
                    table, context, luck_bonus, pity, dropped_items
                )
                if drop and isinstance(drop, LootDrop):
                    result.items.append(drop)
                    if table.unique_drops:
                        dropped_items.add(drop.item_id)
                attempts += 1

        if len(result.items) > table.max_drops:
            result.items = result.items[:table.max_drops]

        return result

    def _roll_once(
        self,
        table: LootTable,
        context: Dict[str, Any],
        luck_bonus: float,
        pity: PityTracker,
        dropped: set,
    ) -> Optional[Union[LootDrop, CurrencyDrop, LootResult]]:
        """Roll once on a loot table."""
        # Filter eligible entries
        eligible: List[Tuple[LootTableEntry, float]] = []
        total_weight = table.empty_weight

        for entry in table.entries:
            # Skip already dropped unique items
            if isinstance(entry, LootEntry) and entry.unique and entry.item_id in dropped:
                continue

            # Check conditions
            if not entry.check_conditions(context):
                continue

            # Apply luck bonus to weight
            weight = entry.weight * (1.0 + luck_bonus)

            # Check for pity override
            if isinstance(entry, LootEntry):
                item_def = self._item_registry.get(entry.item_id)
                if item_def:
                    if pity.check_pity(item_def.rarity):
                        weight *= PITY_WEIGHT_BOOST  # Massively boost pity items

            eligible.append((entry, weight))
            total_weight += weight

        if not eligible or total_weight <= 0:
            return None

        # Roll weighted random
        roll = self._rng.random() * total_weight

        # Check for empty roll
        if roll < table.empty_weight:
            return None

        roll -= table.empty_weight
        cumulative = 0.0

        for entry, weight in eligible:
            cumulative += weight
            if roll < cumulative:
                return self._resolve_entry(entry, table.table_id, context, pity)

        return None

    def _resolve_entry(
        self,
        entry: LootTableEntry,
        source_table: str,
        context: Dict[str, Any],
        pity: PityTracker,
    ) -> Optional[Union[LootDrop, CurrencyDrop, LootResult]]:
        """Resolve an entry into a drop."""
        if isinstance(entry, LootEntry):
            drop = self._create_drop(entry, source_table)

            # Update pity
            item_def = self._item_registry.get(entry.item_id)
            if item_def:
                pity.reset(item_def.rarity)

            return drop

        elif isinstance(entry, NestedTableEntry):
            # Recursively roll nested table
            nested = self._table_registry.get(entry.table_id)
            if nested:
                return self.roll(
                    nested,
                    context,
                    rolls_override=entry.rolls_override,
                )
            return None

        elif isinstance(entry, CurrencyEntry):
            return CurrencyDrop(
                currency_type=entry.currency_type,
                amount=entry.roll_amount(self._rng),
                source_table=source_table,
            )

        return None

    def _create_drop(self, entry: LootEntry, source_table: str) -> LootDrop:
        """Create a loot drop from an entry."""
        quantity = entry.roll_quantity(self._rng)
        item_def = self._item_registry.get(entry.item_id)
        rarity = item_def.rarity if item_def else None

        return LootDrop(
            item_id=entry.item_id,
            quantity=quantity,
            rarity=rarity,
            source_table=source_table,
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def preview(
        self,
        table: Union[str, LootTable],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Preview drop chances for a table.

        Args:
            table: Table ID or LootTable instance
            context: Context for condition evaluation

        Returns:
            List of (item_id, probability) tuples
        """
        if isinstance(table, str):
            resolved = self._table_registry.get(table)
            if not resolved:
                return []
            table = resolved

        context = context or {}

        # Calculate total weight
        total_weight = table.empty_weight
        eligible: List[Tuple[str, float]] = []

        for entry in table.entries:
            if not entry.check_conditions(context):
                continue

            if isinstance(entry, LootEntry):
                eligible.append((entry.item_id, entry.weight))
                total_weight += entry.weight
            elif isinstance(entry, NestedTableEntry):
                eligible.append((f"[Table: {entry.table_id}]", entry.weight))
                total_weight += entry.weight
            elif isinstance(entry, CurrencyEntry):
                eligible.append((f"[Currency: {entry.currency_type}]", entry.weight))
                total_weight += entry.weight

        if total_weight <= 0:
            return []

        # Calculate probabilities
        result = []
        for item_id, weight in eligible:
            probability = weight / total_weight
            result.append((item_id, probability))

        # Add empty probability
        if table.empty_weight > 0:
            result.append(("Nothing", table.empty_weight / total_weight))

        return sorted(result, key=lambda x: -x[1])

    def simulate(
        self,
        table: Union[str, LootTable],
        iterations: int = 1000,
        context: Optional[Dict[str, Any]] = None,
        luck: float = 0.0,
    ) -> Dict[str, int]:
        """
        Simulate many rolls to get drop distribution.

        Args:
            table: Table to simulate
            iterations: Number of simulations
            context: Roll context
            luck: Luck value

        Returns:
            Dict of item_id to drop count
        """
        counts: Dict[str, int] = {}

        for i in range(iterations):
            result = self.roll(table, context, luck, entity_id=f"sim_{i}")
            for drop in result.items:
                counts[drop.item_id] = counts.get(drop.item_id, 0) + drop.quantity

        return counts


# =============================================================================
# Loot Table Registry
# =============================================================================


class LootTableRegistry:
    """Global registry for loot tables."""

    _instance: Optional[LootTableRegistry] = None

    def __init__(self):
        self._tables: Dict[str, LootTable] = {}

    @classmethod
    def instance(cls) -> LootTableRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = LootTableRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset registry."""
        cls._instance = None

    def register(self, table: LootTable) -> None:
        """Register a loot table."""
        if table.table_id in self._tables:
            raise ValueError(f"Table '{table.table_id}' already registered")
        self._tables[table.table_id] = table

    def get(self, table_id: str) -> Optional[LootTable]:
        """Get loot table."""
        return self._tables.get(table_id)

    def all(self) -> List[LootTable]:
        """Get all tables."""
        return list(self._tables.values())

    def clear(self) -> None:
        """Clear all tables."""
        self._tables.clear()


# =============================================================================
# Loot Builder (Fluent API)
# =============================================================================


class LootTableBuilder:
    """Fluent builder for loot tables."""

    def __init__(self, table_id: str):
        self._table_id = table_id
        self._entries: List[LootTableEntry] = []
        self._guaranteed: List[LootEntry] = []
        self._rolls = 1
        self._empty_weight = 0.0
        self._min_drops = 0
        self._max_drops = DEFAULT_MAX_DROPS
        self._unique_drops = True

    def rolls(self, count: int) -> LootTableBuilder:
        """Set number of rolls."""
        self._rolls = count
        return self

    def empty_weight(self, weight: float) -> LootTableBuilder:
        """Set empty drop weight."""
        self._empty_weight = weight
        return self

    def min_drops(self, count: int) -> LootTableBuilder:
        """Set minimum drops."""
        self._min_drops = count
        return self

    def max_drops(self, count: int) -> LootTableBuilder:
        """Set maximum drops."""
        self._max_drops = count
        return self

    def unique_drops(self, unique: bool) -> LootTableBuilder:
        """Set unique drops mode."""
        self._unique_drops = unique
        return self

    def add_item(
        self,
        item_id: str,
        weight: float = 1.0,
        min_qty: int = 1,
        max_qty: int = 1,
        conditions: Optional[Tuple[LootCondition, ...]] = None,
        unique: bool = False,
    ) -> LootTableBuilder:
        """Add an item entry."""
        self._entries.append(LootEntry(
            item_id=item_id,
            weight=weight,
            min_quantity=min_qty,
            max_quantity=max_qty,
            conditions=conditions or (),
            unique=unique,
        ))
        return self

    def add_guaranteed(
        self,
        item_id: str,
        min_qty: int = 1,
        max_qty: int = 1,
        conditions: Optional[Tuple[LootCondition, ...]] = None,
    ) -> LootTableBuilder:
        """Add a guaranteed drop."""
        self._guaranteed.append(LootEntry(
            item_id=item_id,
            weight=1.0,
            min_quantity=min_qty,
            max_quantity=max_qty,
            conditions=conditions or (),
            guaranteed=True,
        ))
        return self

    def add_nested(
        self,
        table_id: str,
        weight: float = 1.0,
        conditions: Optional[Tuple[LootCondition, ...]] = None,
        rolls_override: Optional[int] = None,
    ) -> LootTableBuilder:
        """Add a nested table reference."""
        self._entries.append(NestedTableEntry(
            table_id=table_id,
            weight=weight,
            conditions=conditions or (),
            rolls_override=rolls_override,
        ))
        return self

    def add_currency(
        self,
        currency_type: str,
        min_amount: int,
        max_amount: int,
        weight: float = 1.0,
        conditions: Optional[Tuple[LootCondition, ...]] = None,
    ) -> LootTableBuilder:
        """Add a currency entry."""
        self._entries.append(CurrencyEntry(
            currency_type=currency_type,
            min_amount=min_amount,
            max_amount=max_amount,
            weight=weight,
            conditions=conditions or (),
        ))
        return self

    def build(self) -> LootTable:
        """Build the loot table."""
        return LootTable(
            table_id=self._table_id,
            entries=self._entries,
            rolls=self._rolls,
            guaranteed_entries=self._guaranteed,
            empty_weight=self._empty_weight,
            min_drops=self._min_drops,
            max_drops=self._max_drops,
            unique_drops=self._unique_drops,
        )
