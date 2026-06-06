"""
Gameplay Abilities System.

Provides systems for ability definitions, effects, attributes, targeting,
and buff/debuff management with Foundation Registry integration.

Foundation Integration (T-GP-7.11):
- TrackedDescriptor wiring for ability attributes
- Automatic dirty flag tracking on attribute changes
- Change subscriptions via tracker.on_change()
- Batch updates for atomic changes
"""

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

from engine.gameplay.abilities.tags import (
    GameplayTag,
    GameplayTagContainer,
    GameplayTagQuery,
    GameplayTagRegistry,
    gameplay_tag,
    ability_with_tags,
)

from engine.gameplay.abilities.attributes import (
    # Core attribute classes
    Attribute,
    AttributeModifier,
    AttributeModifierHandle,
    AttributeSet,
    DerivedAttribute,
    create_standard_attributes,
    # Foundation Tracker integration (T-GP-7.11)
    AttributeTracker,
    attribute_tracker,
    TrackedAttributeDescriptor,
    tracked_attribute,
    AttributeChangeCallback,
    # Tracked ability attributes
    TrackedAbilityAttribute,
    TrackedVitalAttribute,
    TrackedCooldownAttribute,
    # Tracked attribute set
    TrackedAttributeSet,
    create_tracked_standard_attributes,
)

__all__ = [
    # Stacking mode
    "StackingMode",
    # Events
    "AbilityCast",
    "BuffApplied",
    "BuffExpired",
    # Decorators
    "ability",
    "buff",
    # Event emitters
    "emit_ability_cast",
    "emit_buff_applied",
    "emit_buff_expired",
    # Query helpers
    "get_all_abilities",
    "get_abilities_by_tag",
    "get_all_buffs",
    "get_buffs_by_stacking",
    "get_debuffs",
    "get_ability_metadata",
    "get_buff_metadata",
    # Tags
    "GameplayTag",
    "GameplayTagContainer",
    "GameplayTagQuery",
    "GameplayTagRegistry",
    "gameplay_tag",
    "ability_with_tags",
    # Attributes (T-GP-7.11)
    "Attribute",
    "AttributeModifier",
    "AttributeModifierHandle",
    "AttributeSet",
    "DerivedAttribute",
    "create_standard_attributes",
    # Foundation Tracker integration (T-GP-7.11)
    "AttributeTracker",
    "attribute_tracker",
    "TrackedAttributeDescriptor",
    "tracked_attribute",
    "AttributeChangeCallback",
    # Tracked ability attributes
    "TrackedAbilityAttribute",
    "TrackedVitalAttribute",
    "TrackedCooldownAttribute",
    # Tracked attribute set
    "TrackedAttributeSet",
    "create_tracked_standard_attributes",
]
