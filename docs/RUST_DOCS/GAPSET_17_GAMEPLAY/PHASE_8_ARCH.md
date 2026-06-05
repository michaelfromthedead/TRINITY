# Phase 8: Inventory, Equipment, Loot, Crafting & Economy — Architecture

## Overview

Complete item economy system: inventory management, equipment with socket attachment, loot tables with RNG/pity, crafting recipes/stations/quality, and multi-currency trading.

## Component Breakdown

### Inventory (`economy/inventory.py`)

```
InventoryContainer
├── Slots (configurable count)
├── Item stacking (max stack per type)
├── Weight capacity tracking
├── Add/remove/transfer operations
├── Transaction support (atomic operations)
├── Sort/compact functions
└── Weight limit enforcement

ItemDefinition / ItemInstance
├── id, name, stack_count, max_stack, weight
├── Rarity: common/uncommon/rare/epic/legendary
├── Item types: Equipment, Consumable, Material, KeyItem, Currency
└── ItemRegistry (singleton)
```

### Equipment (`economy/equipment.py`)

```
EquipmentComponent
├── Slots: head, chest, hands, legs, feet, weapon, off-hand, ring, necklace
├── Slot type validation on equip
├── Preview support (swap without committing)
├── Socket attachment system
├── Skin override (material/texture replacement)
├── Show/hide slot toggles
└── Modifier application on equip / removal on unequip
    ├── Flat add
    ├── Multiply
    └── Override
```

### Loot (`economy/loot.py`)

```
LootTable
├── Entries (item_ref + weight)
├── Conditions: player level, quest state, game progress
├── Nested tables for groups
└── Serialization support

Loot Rolling
├── Weighted RNG selection
├── Pity system (increasing probability after consecutive failures)
├── Luck stat modifier on weights
└── Returns item list
```

### Crafting (`economy/crafting.py`)

```
Recipe
├── Ingredient list (item ID + count)
├── Output item + quantity
├── Station requirement
└── Recipe discovery system

Stations: workbench, forge, cooking fire, etc.

Crafting Process
├── Validate requirements (ingredients, station, skill)
├── Consume ingredients from inventory
├── Create output item(s)
├── Quality variance (based on skill)
└── @crafting / @recipe / @ingredient / @crafting_station decorators
```

### Economy

```
Currency
├── Types: gold, silver, copper, tokens, etc.
├── Add/remove/transfer/exchange
└── Exchange rates between types

Trading
├── TradeOffer: items + currencies per party
├── Offer lifecycle: accept / reject / timeout
├── Atomic execution (both sides simultaneously)
└── @economy decorator
```

### Serialization

```
All inventory/economy components @serializable
├── Save/load player inventory
├── ContentStore integration for structural sharing
└── Cross-session persistence
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `economy/inventory.py` | — | InventoryContainer, ItemDefinition, trading |
| `economy/items.py` | — | Item types, ItemRegistry |
| `economy/equipment.py` | 767 | Equipment slots, sockets, modifiers |
| `economy/loot.py` | 884 | Loot tables, RNG, pity system |
| `economy/crafting.py` | 947 | Recipes, stations, quality system |

## Dependencies

- Phase 1 entity framework (Actor, ComponentStore)
- Phase 7 (Attribute system for equipment modifiers)
- Foundation: @serializable, ContentStore
