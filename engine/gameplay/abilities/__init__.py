"""
Gameplay Abilities System.

Provides a comprehensive ability system including:
- Attributes with modifiers and derived values
- Gameplay tags for categorization and filtering
- Ability and buff decorators with registry integration
- Event system for ability casts and buff applications
- Tracked attributes with change notifications
"""

from __future__ import annotations

# Stacking mode
from .decorators import StackingMode

# Events
from .decorators import AbilityCast, BuffApplied, BuffExpired

# Decorators
from .decorators import ability, buff

# Event emitters
from .decorators import emit_ability_cast, emit_buff_applied, emit_buff_expired

# Query helpers
from .decorators import (
    get_all_abilities,
    get_abilities_by_tag,
    get_all_buffs,
    get_buffs_by_stacking,
    get_debuffs,
    get_ability_metadata,
    get_buff_metadata,
)

# Tags
from .tags import (
    GameplayTag,
    GameplayTagContainer,
    GameplayTagQuery,
    GameplayTagRegistry,
    ability_with_tags,
    gameplay_tag,
)

# Attributes
from .attributes import (
    Attribute,
    AttributeChangeCallback,
    AttributeModifier,
    AttributeModifierHandle,
    AttributeSet,
    AttributeTracker,
    DerivedAttribute,
    TrackedAbilityAttribute,
    TrackedAttributeDescriptor,
    TrackedAttributeSet,
    TrackedCooldownAttribute,
    TrackedVitalAttribute,
    attribute_tracker,
    create_standard_attributes,
    create_tracked_standard_attributes,
    tracked_attribute,
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
    "ability_with_tags",
    "gameplay_tag",
    # Attributes
    "Attribute",
    "AttributeChangeCallback",
    "AttributeModifier",
    "AttributeModifierHandle",
    "AttributeSet",
    "AttributeTracker",
    "DerivedAttribute",
    "TrackedAbilityAttribute",
    "TrackedAttributeDescriptor",
    "TrackedAttributeSet",
    "TrackedCooldownAttribute",
    "TrackedVitalAttribute",
    "attribute_tracker",
    "create_standard_attributes",
    "create_tracked_standard_attributes",
    "tracked_attribute",
]
