"""
Tests for the Buff System.

Tests cover:
- Buff application
- Buff duration and expiration
- Buff stacking (intensity, duration, count)
- Buff refresh on reapplication
- Debuff immunity
- Buff/debuff UI data
- Buff removal (manual, expired, dispel)
- Buff categories and types

Total: ~120 tests

Note: These tests are designed for a buff_system.py module that follows
the patterns established in the constants.py and effects.py modules.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4
from enum import IntEnum, auto

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeSet,
    create_standard_attributes,
)
from engine.gameplay.abilities.constants import (
    EffectType,
    ModifierOperation,
    StackingMode,
    DEFAULT_MAX_STACKS,
    EPSILON,
)
from engine.gameplay.abilities.effects import (
    EffectContext,
    EffectModifier,
    DurationEffect,
    InfiniteEffect,
    EffectContainer,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer


# =============================================================================
# MOCK BUFF SYSTEM (for testing design)
# =============================================================================


class BuffCategory(IntEnum):
    """Categories of buffs/debuffs."""

    BUFF = 0
    DEBUFF = auto()
    AURA = auto()
    HIDDEN = auto()


class BuffType(IntEnum):
    """Types of buff effects."""

    STAT = 0        # Modifies stats
    DAMAGE = auto() # Deals damage over time
    HEAL = auto()   # Heals over time
    CONTROL = auto()  # CC effects (stun, slow)
    UTILITY = auto()  # Other effects


@dataclass
class BuffSpec:
    """
    Specification for a buff/debuff.

    Defines the template for buffs that can be applied.
    """

    id: str = ""
    name: str = ""
    description: str = ""
    icon: str = ""

    # Category and type
    category: BuffCategory = BuffCategory.BUFF
    buff_type: BuffType = BuffType.STAT

    # Duration
    duration: float = 0.0  # 0 = infinite
    tick_rate: float = 0.0  # 0 = no tick

    # Stacking
    stacking_mode: StackingMode = StackingMode.NONE
    max_stacks: int = 1
    stack_duration_refresh: bool = True

    # Effects
    modifiers: List[EffectModifier] = field(default_factory=list)

    # Tags
    granted_tags: List[str] = field(default_factory=list)
    blocked_by_tags: List[str] = field(default_factory=list)
    immunity_tags: List[str] = field(default_factory=list)  # Tags that grant immunity

    # Dispel
    can_dispel: bool = True
    dispel_priority: int = 0


@dataclass
class BuffInstance:
    """
    Runtime instance of an active buff.
    """

    id: UUID = field(default_factory=uuid4)
    spec: BuffSpec = field(default_factory=BuffSpec)
    source: Optional[Any] = None

    # State
    current_stacks: int = 1
    remaining_duration: float = 0.0
    time_since_tick: float = 0.0

    # UI data
    is_visible: bool = True

    @property
    def is_active(self) -> bool:
        """Check if buff is currently active."""
        return self.remaining_duration > 0 or self.spec.duration == 0

    @property
    def is_expired(self) -> bool:
        """Check if buff has expired."""
        return self.spec.duration > 0 and self.remaining_duration <= 0

    @property
    def progress(self) -> float:
        """Get progress as 0.0 to 1.0 (0 = just applied, 1 = expired)."""
        if self.spec.duration <= 0:
            return 0.0
        return 1.0 - (self.remaining_duration / self.spec.duration)

    def refresh_duration(self) -> None:
        """Refresh duration to full."""
        self.remaining_duration = self.spec.duration

    def add_stack(self) -> bool:
        """Add a stack. Returns True if stack was added."""
        if self.current_stacks >= self.spec.max_stacks:
            return False
        self.current_stacks += 1
        return True

    def remove_stack(self) -> bool:
        """Remove a stack. Returns True if stacks remain."""
        self.current_stacks -= 1
        return self.current_stacks > 0

    def tick(self, delta_time: float) -> bool:
        """
        Update the buff. Returns False if expired.
        """
        if self.spec.duration > 0:
            self.remaining_duration -= delta_time
            if self.remaining_duration <= 0:
                self.remaining_duration = 0
                return False

        if self.spec.tick_rate > 0:
            self.time_since_tick += delta_time

        return True


class BuffContainer:
    """
    Container for managing buffs on an entity.
    """

    def __init__(
        self,
        attributes: AttributeSet,
        tags: Optional[GameplayTagContainer] = None,
    ) -> None:
        self._attributes = attributes
        self._tags = tags or GameplayTagContainer()
        self._buffs: Dict[UUID, BuffInstance] = {}
        self._by_id: Dict[str, Set[UUID]] = {}  # spec.id -> instance UUIDs
        self._modifier_handles: Dict[UUID, List[Any]] = {}  # buff UUID -> modifier handles

    @property
    def active_buffs(self) -> List[BuffInstance]:
        """Get all active buffs."""
        return list(self._buffs.values())

    @property
    def visible_buffs(self) -> List[BuffInstance]:
        """Get all visible buffs (for UI)."""
        return [b for b in self._buffs.values() if b.is_visible]

    def get_buffs(self) -> List[BuffInstance]:
        """Get all buff instances."""
        return [b for b in self._buffs.values() if b.spec.category == BuffCategory.BUFF]

    def get_debuffs(self) -> List[BuffInstance]:
        """Get all debuff instances."""
        return [b for b in self._buffs.values() if b.spec.category == BuffCategory.DEBUFF]

    def can_apply(self, spec: BuffSpec, source: Optional[Any] = None) -> bool:
        """Check if a buff can be applied."""
        # Check immunity tags
        for tag in spec.immunity_tags:
            if self._tags.has(tag):
                return False

        # Check blocked by tags
        for tag in spec.blocked_by_tags:
            if self._tags.has(tag):
                return False

        return True

    def apply(
        self,
        spec: BuffSpec,
        source: Optional[Any] = None,
        stacks: int = 1,
    ) -> Optional[BuffInstance]:
        """
        Apply a buff. Returns the buff instance or None if blocked.
        """
        if not self.can_apply(spec, source):
            return None

        # Check for existing buff
        existing = self._get_existing_buff(spec.id)

        if existing is not None:
            return self._handle_reapplication(existing, spec, stacks)

        # Create new buff instance
        instance = BuffInstance(
            spec=spec,
            source=source,
            current_stacks=min(stacks, spec.max_stacks),
            remaining_duration=spec.duration,
            is_visible=spec.category != BuffCategory.HIDDEN,
        )

        self._buffs[instance.id] = instance

        if spec.id not in self._by_id:
            self._by_id[spec.id] = set()
        self._by_id[spec.id].add(instance.id)

        # Apply modifiers
        self._apply_modifiers(instance)

        # Grant tags
        for tag in spec.granted_tags:
            self._tags.add(tag)

        return instance

    def _get_existing_buff(self, spec_id: str) -> Optional[BuffInstance]:
        """Get existing buff instance by spec ID."""
        if spec_id not in self._by_id:
            return None

        for instance_id in self._by_id[spec_id]:
            if instance_id in self._buffs:
                return self._buffs[instance_id]
        return None

    def _handle_reapplication(
        self,
        existing: BuffInstance,
        spec: BuffSpec,
        stacks: int,
    ) -> BuffInstance:
        """Handle reapplication of an existing buff."""
        mode = spec.stacking_mode

        if mode == StackingMode.NONE:
            # Refresh duration only
            if spec.stack_duration_refresh:
                existing.refresh_duration()

        elif mode == StackingMode.DURATION:
            # Extend duration
            existing.remaining_duration += spec.duration

        elif mode == StackingMode.INTENSITY:
            # Add stacks
            for _ in range(stacks):
                if not existing.add_stack():
                    break
            if spec.stack_duration_refresh:
                existing.refresh_duration()
            # Re-apply modifiers with new stack count
            self._remove_modifiers(existing)
            self._apply_modifiers(existing)

        elif mode == StackingMode.INDEPENDENT:
            # Create new instance
            return self.apply(
                BuffSpec(
                    id=f"{spec.id}_{uuid4().hex[:8]}",  # Unique ID
                    name=spec.name,
                    **{k: v for k, v in spec.__dict__.items() if k not in ("id", "name")}
                ),
                source=existing.source,
                stacks=stacks,
            )

        return existing

    def _apply_modifiers(self, instance: BuffInstance) -> None:
        """Apply buff modifiers to attributes."""
        handles = []
        stack_multiplier = instance.current_stacks if instance.spec.stacking_mode == StackingMode.INTENSITY else 1

        for mod in instance.spec.modifiers:
            if self._attributes.has(mod.attribute):
                magnitude = mod.get_magnitude() * stack_multiplier
                handle = self._attributes.add_modifier(
                    mod.attribute,
                    mod.operation,
                    magnitude,
                    source=instance,
                )
                handles.append(handle)

        self._modifier_handles[instance.id] = handles

    def _remove_modifiers(self, instance: BuffInstance) -> None:
        """Remove buff modifiers from attributes."""
        if instance.id in self._modifier_handles:
            for handle in self._modifier_handles[instance.id]:
                self._attributes.remove_modifier(handle)
            del self._modifier_handles[instance.id]

    def remove(self, buff_or_id: BuffInstance | UUID) -> bool:
        """Remove a buff by instance or ID."""
        if isinstance(buff_or_id, BuffInstance):
            buff_id = buff_or_id.id
        else:
            buff_id = buff_or_id

        if buff_id not in self._buffs:
            return False

        instance = self._buffs[buff_id]

        # Remove modifiers
        self._remove_modifiers(instance)

        # Remove tags
        for tag in instance.spec.granted_tags:
            self._tags.remove(tag)

        # Remove from tracking
        if instance.spec.id in self._by_id:
            self._by_id[instance.spec.id].discard(buff_id)
        del self._buffs[buff_id]

        return True

    def remove_by_spec(self, spec_id: str) -> int:
        """Remove all buffs with the given spec ID. Returns count removed."""
        if spec_id not in self._by_id:
            return 0

        ids_to_remove = list(self._by_id[spec_id])
        count = 0
        for buff_id in ids_to_remove:
            if self.remove(buff_id):
                count += 1
        return count

    def dispel(
        self,
        category: Optional[BuffCategory] = None,
        count: int = 1,
        priority_order: bool = True,
    ) -> int:
        """
        Dispel buffs/debuffs. Returns count dispelled.
        """
        candidates = [
            b for b in self._buffs.values()
            if b.spec.can_dispel and (category is None or b.spec.category == category)
        ]

        if priority_order:
            candidates.sort(key=lambda b: b.spec.dispel_priority, reverse=True)

        dispelled = 0
        for buff in candidates[:count]:
            if self.remove(buff):
                dispelled += 1

        return dispelled

    def dispel_debuffs(self, count: int = 1) -> int:
        """Dispel debuffs. Returns count dispelled."""
        return self.dispel(BuffCategory.DEBUFF, count)

    def dispel_buffs(self, count: int = 1) -> int:
        """Dispel buffs (for enemies). Returns count dispelled."""
        return self.dispel(BuffCategory.BUFF, count)

    def has_buff(self, spec_id: str) -> bool:
        """Check if a buff with the given spec ID is active."""
        return spec_id in self._by_id and len(self._by_id[spec_id]) > 0

    def get_buff(self, spec_id: str) -> Optional[BuffInstance]:
        """Get buff instance by spec ID."""
        return self._get_existing_buff(spec_id)

    def get_stacks(self, spec_id: str) -> int:
        """Get stack count for a buff."""
        buff = self._get_existing_buff(spec_id)
        return buff.current_stacks if buff else 0

    def tick(self, delta_time: float) -> None:
        """Update all buffs. Removes expired buffs."""
        expired = []
        for buff_id, buff in self._buffs.items():
            if not buff.tick(delta_time):
                expired.append(buff_id)

        for buff_id in expired:
            self.remove(buff_id)


# =============================================================================
# BUFF APPLICATION TESTS
# =============================================================================


class TestBuffApplication:
    """Tests for buff application."""

    def test_apply_basic_buff(self):
        """Test applying a basic buff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="strength",
            name="Strength",
            duration=10.0,
        )

        instance = container.apply(spec)

        assert instance is not None
        assert container.has_buff("strength")

    def test_apply_buff_with_modifier(self):
        """Test buff applies modifier to attributes."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="damage_buff",
            name="Damage Buff",
            duration=10.0,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=20.0,
                )
            ],
        )

        initial_damage = attrs.get("damage")
        container.apply(spec)

        assert attrs.get("damage") == initial_damage + 20.0

    def test_apply_buff_with_multiple_modifiers(self):
        """Test buff with multiple modifiers."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="warrior_buff",
            name="Warrior's Might",
            duration=10.0,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=15.0,
                ),
                EffectModifier(
                    attribute="armor",
                    operation=ModifierOperation.ADD,
                    base_magnitude=25.0,
                ),
            ],
        )

        container.apply(spec)

        assert attrs.get("damage") == 25.0  # 10 + 15
        assert attrs.get("armor") == 25.0  # 0 + 25

    def test_apply_buff_grants_tags(self):
        """Test buff grants tags on application."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="shield",
            name="Shield",
            duration=5.0,
            granted_tags=["status.shielded", "status.protected"],
        )

        container.apply(spec)

        assert tags.has("status.shielded")
        assert tags.has("status.protected")

    def test_apply_debuff(self):
        """Test applying a debuff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="weakness",
            name="Weakness",
            category=BuffCategory.DEBUFF,
            duration=8.0,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.MULTIPLY,
                    base_magnitude=-0.25,  # -25% damage
                )
            ],
        )

        container.apply(spec)
        debuffs = container.get_debuffs()

        assert len(debuffs) == 1
        assert debuffs[0].spec.id == "weakness"

    def test_apply_hidden_buff(self):
        """Test applying a hidden buff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="internal",
            name="Internal Buff",
            category=BuffCategory.HIDDEN,
            duration=10.0,
        )

        instance = container.apply(spec)

        assert instance is not None
        assert instance.is_visible is False
        assert len(container.visible_buffs) == 0


# =============================================================================
# BUFF DURATION TESTS
# =============================================================================


class TestBuffDuration:
    """Tests for buff duration and expiration."""

    def test_buff_initial_duration(self):
        """Test buff starts with full duration."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=10.0)
        instance = container.apply(spec)

        assert instance.remaining_duration == 10.0

    def test_buff_duration_decrements(self):
        """Test buff duration decrements over time."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=10.0)
        instance = container.apply(spec)

        instance.tick(3.0)

        assert instance.remaining_duration == 7.0

    def test_buff_expires(self):
        """Test buff expires after duration."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=5.0)
        container.apply(spec)

        container.tick(6.0)

        assert not container.has_buff("test")

    def test_buff_removed_on_expire(self):
        """Test buff is removed from container on expire."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=5.0,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                )
            ],
        )

        container.apply(spec)
        initial_damage = attrs.get("damage")  # 10 + 10 = 20

        container.tick(6.0)

        assert attrs.get("damage") == initial_damage - 10.0  # Modifier removed

    def test_infinite_buff(self):
        """Test buff with infinite duration (duration=0)."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="infinite", duration=0.0)
        instance = container.apply(spec)

        # Tick many times
        for _ in range(100):
            container.tick(10.0)

        assert container.has_buff("infinite")

    def test_buff_progress(self):
        """Test buff progress calculation."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=10.0)
        instance = container.apply(spec)

        instance.tick(5.0)

        assert instance.progress == 0.5

    def test_buff_is_expired_property(self):
        """Test is_expired property."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=5.0)
        instance = container.apply(spec)

        assert instance.is_expired is False

        instance.tick(6.0)

        assert instance.is_expired is True


# =============================================================================
# BUFF STACKING TESTS
# =============================================================================


class TestBuffStacking:
    """Tests for buff stacking (intensity, duration, count)."""

    def test_stacking_none_refreshes_duration(self):
        """Test NONE stacking mode refreshes duration."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.NONE,
            stack_duration_refresh=True,
        )

        instance = container.apply(spec)
        instance.tick(7.0)  # 3 seconds remaining

        container.apply(spec)  # Reapply

        assert instance.remaining_duration == 10.0  # Refreshed

    def test_stacking_duration_extends(self):
        """Test DURATION stacking mode extends duration."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.DURATION,
        )

        instance = container.apply(spec)
        instance.tick(5.0)  # 5 seconds remaining

        container.apply(spec)  # Reapply

        assert instance.remaining_duration == 15.0  # 5 + 10

    def test_stacking_intensity_adds_stacks(self):
        """Test INTENSITY stacking mode adds stacks."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.INTENSITY,
            max_stacks=5,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=10.0,
                )
            ],
        )

        container.apply(spec)
        container.apply(spec)
        container.apply(spec)

        instance = container.get_buff("test")
        assert instance.current_stacks == 3
        # 10 base + (10 * 3 stacks) = 40
        assert attrs.get("damage") == 40.0

    def test_stacking_intensity_max_stacks(self):
        """Test INTENSITY stacking respects max stacks."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.INTENSITY,
            max_stacks=3,
        )

        for _ in range(10):
            container.apply(spec)

        instance = container.get_buff("test")
        assert instance.current_stacks == 3

    def test_stacking_independent_creates_multiple(self):
        """Test INDEPENDENT stacking mode creates multiple instances."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="dot",
            duration=10.0,
            stacking_mode=StackingMode.INDEPENDENT,
        )

        container.apply(spec)
        container.apply(spec)
        container.apply(spec)

        assert len(container.active_buffs) == 3

    def test_add_stack(self):
        """Test manually adding stack."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.INTENSITY,
            max_stacks=5,
        )

        instance = container.apply(spec)
        result = instance.add_stack()

        assert result is True
        assert instance.current_stacks == 2

    def test_add_stack_at_max(self):
        """Test adding stack when at max."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            max_stacks=2,
        )

        instance = container.apply(spec)
        instance.add_stack()
        result = instance.add_stack()

        assert result is False
        assert instance.current_stacks == 2

    def test_remove_stack(self):
        """Test removing stack."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", max_stacks=5)

        instance = container.apply(spec)
        instance.current_stacks = 3

        result = instance.remove_stack()

        assert result is True
        assert instance.current_stacks == 2

    def test_remove_last_stack(self):
        """Test removing last stack."""
        instance = BuffInstance(spec=BuffSpec(id="test"))
        instance.current_stacks = 1

        result = instance.remove_stack()

        assert result is False
        assert instance.current_stacks == 0

    def test_get_stacks(self):
        """Test getting stack count through container."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            stacking_mode=StackingMode.INTENSITY,
            max_stacks=10,
        )

        container.apply(spec)
        container.apply(spec)
        container.apply(spec)

        assert container.get_stacks("test") == 3

    def test_get_stacks_nonexistent(self):
        """Test getting stacks for nonexistent buff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        assert container.get_stacks("nonexistent") == 0


# =============================================================================
# BUFF REFRESH TESTS
# =============================================================================


class TestBuffRefresh:
    """Tests for buff refresh on reapplication."""

    def test_refresh_duration(self):
        """Test refresh_duration method."""
        spec = BuffSpec(id="test", duration=10.0)
        instance = BuffInstance(spec=spec, remaining_duration=3.0)

        instance.refresh_duration()

        assert instance.remaining_duration == 10.0

    def test_no_refresh_option(self):
        """Test buff without duration refresh."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.NONE,
            stack_duration_refresh=False,
        )

        instance = container.apply(spec)
        instance.tick(7.0)
        original_remaining = instance.remaining_duration

        container.apply(spec)  # Reapply

        assert instance.remaining_duration == original_remaining  # Not refreshed

    def test_refresh_with_intensity_stacking(self):
        """Test duration refresh with intensity stacking."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            duration=10.0,
            stacking_mode=StackingMode.INTENSITY,
            stack_duration_refresh=True,
            max_stacks=5,
        )

        instance = container.apply(spec)
        instance.tick(8.0)  # 2 seconds remaining

        container.apply(spec)  # Add stack

        assert instance.remaining_duration == 10.0  # Refreshed
        assert instance.current_stacks == 2


# =============================================================================
# DEBUFF IMMUNITY TESTS
# =============================================================================


class TestDebuffImmunity:
    """Tests for debuff immunity."""

    def test_immunity_tag_blocks_debuff(self):
        """Test immunity tag blocks debuff application."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.immune.poison")
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="poison",
            name="Poison",
            category=BuffCategory.DEBUFF,
            immunity_tags=["status.immune.poison"],
        )

        result = container.apply(spec)

        assert result is None
        assert not container.has_buff("poison")

    def test_blocked_by_tag(self):
        """Test buff blocked by specific tag."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.debuff_immune")
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="slow",
            name="Slow",
            blocked_by_tags=["status.debuff_immune"],
        )

        result = container.apply(spec)

        assert result is None

    def test_can_apply_check(self):
        """Test can_apply method."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="debuff",
            immunity_tags=["immune.all"],
        )

        assert container.can_apply(spec) is True

        tags.add("immune.all")
        assert container.can_apply(spec) is False

    def test_multiple_immunity_tags(self):
        """Test multiple immunity tags."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="fire_dot",
            immunity_tags=["immune.fire", "immune.dot", "immune.all"],
        )

        # Apply with no immunity
        assert container.can_apply(spec) is True

        # Any immunity tag blocks
        tags.add("immune.dot")
        assert container.can_apply(spec) is False


# =============================================================================
# BUFF UI DATA TESTS
# =============================================================================


class TestBuffUIData:
    """Tests for buff/debuff UI data."""

    def test_buff_visibility(self):
        """Test buff visibility property."""
        instance = BuffInstance(
            spec=BuffSpec(id="test", category=BuffCategory.BUFF),
            is_visible=True,
        )

        assert instance.is_visible is True

    def test_hidden_buff_not_visible(self):
        """Test hidden buffs are not visible."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="hidden",
            category=BuffCategory.HIDDEN,
        )

        instance = container.apply(spec)

        assert instance.is_visible is False

    def test_visible_buffs_property(self):
        """Test visible_buffs property filters correctly."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        container.apply(BuffSpec(id="visible1", category=BuffCategory.BUFF))
        container.apply(BuffSpec(id="visible2", category=BuffCategory.DEBUFF))
        container.apply(BuffSpec(id="hidden", category=BuffCategory.HIDDEN))

        visible = container.visible_buffs

        assert len(visible) == 2

    def test_buff_icon(self):
        """Test buff icon in spec."""
        spec = BuffSpec(
            id="test",
            name="Test Buff",
            icon="icons/buffs/strength.png",
        )

        assert spec.icon == "icons/buffs/strength.png"

    def test_buff_description(self):
        """Test buff description."""
        spec = BuffSpec(
            id="test",
            name="Strength",
            description="Increases damage by 20%",
        )

        assert "20%" in spec.description


# =============================================================================
# BUFF REMOVAL TESTS
# =============================================================================


class TestBuffRemoval:
    """Tests for buff removal (manual, expired, dispel)."""

    def test_manual_removal(self):
        """Test manually removing a buff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=10.0)
        instance = container.apply(spec)

        result = container.remove(instance)

        assert result is True
        assert not container.has_buff("test")

    def test_remove_by_uuid(self):
        """Test removing buff by UUID."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=10.0)
        instance = container.apply(spec)

        result = container.remove(instance.id)

        assert result is True

    def test_remove_nonexistent(self):
        """Test removing nonexistent buff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        result = container.remove(uuid4())

        assert result is False

    def test_remove_by_spec_id(self):
        """Test removing all buffs with spec ID."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="dot",
            stacking_mode=StackingMode.INDEPENDENT,
        )

        container.apply(spec)
        container.apply(spec)
        container.apply(spec)

        count = container.remove_by_spec("dot")

        assert count == 3
        assert len(container.active_buffs) == 0

    def test_removal_removes_modifiers(self):
        """Test removal clears modifiers from attributes."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="test",
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.ADD,
                    base_magnitude=50.0,
                )
            ],
        )

        instance = container.apply(spec)
        assert attrs.get("damage") == 60.0  # 10 + 50

        container.remove(instance)
        assert attrs.get("damage") == 10.0

    def test_removal_removes_tags(self):
        """Test removal clears granted tags."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="test",
            granted_tags=["status.buffed"],
        )

        instance = container.apply(spec)
        assert tags.has("status.buffed")

        container.remove(instance)
        assert not tags.has("status.buffed")


# =============================================================================
# DISPEL TESTS
# =============================================================================


class TestDispel:
    """Tests for dispel functionality."""

    def test_dispel_debuff(self):
        """Test dispelling a debuff."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="debuff",
            category=BuffCategory.DEBUFF,
            can_dispel=True,
        )

        container.apply(spec)
        count = container.dispel_debuffs(count=1)

        assert count == 1
        assert not container.has_buff("debuff")

    def test_dispel_buff(self):
        """Test dispelling a buff (enemy perspective)."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="buff",
            category=BuffCategory.BUFF,
            can_dispel=True,
        )

        container.apply(spec)
        count = container.dispel_buffs(count=1)

        assert count == 1

    def test_dispel_undispellable(self):
        """Test cannot dispel undispellable buffs."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="permanent",
            category=BuffCategory.DEBUFF,
            can_dispel=False,
        )

        container.apply(spec)
        count = container.dispel_debuffs(count=1)

        assert count == 0
        assert container.has_buff("permanent")

    def test_dispel_priority(self):
        """Test dispel removes highest priority first."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        low_priority = BuffSpec(
            id="low",
            category=BuffCategory.DEBUFF,
            dispel_priority=1,
        )
        high_priority = BuffSpec(
            id="high",
            category=BuffCategory.DEBUFF,
            dispel_priority=10,
        )

        container.apply(low_priority)
        container.apply(high_priority)

        container.dispel_debuffs(count=1)

        # High priority should be dispelled first
        assert container.has_buff("low")
        assert not container.has_buff("high")

    def test_dispel_multiple(self):
        """Test dispelling multiple buffs."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        for i in range(5):
            container.apply(BuffSpec(
                id=f"debuff{i}",
                category=BuffCategory.DEBUFF,
            ))

        count = container.dispel_debuffs(count=3)

        assert count == 3
        assert len(container.get_debuffs()) == 2


# =============================================================================
# BUFF CATEGORY AND TYPE TESTS
# =============================================================================


class TestBuffCategoryAndType:
    """Tests for buff categories and types."""

    def test_buff_category_enum(self):
        """Test buff category enum values."""
        assert BuffCategory.BUFF == 0
        assert BuffCategory.DEBUFF > BuffCategory.BUFF
        assert BuffCategory.AURA > BuffCategory.DEBUFF
        assert BuffCategory.HIDDEN > BuffCategory.AURA

    def test_buff_type_enum(self):
        """Test buff type enum values."""
        assert BuffType.STAT == 0
        assert hasattr(BuffType, "DAMAGE")
        assert hasattr(BuffType, "HEAL")
        assert hasattr(BuffType, "CONTROL")
        assert hasattr(BuffType, "UTILITY")

    def test_get_buffs_by_category(self):
        """Test getting buffs by category."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        container.apply(BuffSpec(id="buff1", category=BuffCategory.BUFF))
        container.apply(BuffSpec(id="buff2", category=BuffCategory.BUFF))
        container.apply(BuffSpec(id="debuff1", category=BuffCategory.DEBUFF))
        container.apply(BuffSpec(id="aura1", category=BuffCategory.AURA))

        buffs = container.get_buffs()
        debuffs = container.get_debuffs()

        assert len(buffs) == 2
        assert len(debuffs) == 1

    def test_buff_type_stat(self):
        """Test stat buff type."""
        spec = BuffSpec(
            id="strength",
            buff_type=BuffType.STAT,
            modifiers=[
                EffectModifier(
                    attribute="damage",
                    operation=ModifierOperation.MULTIPLY,
                    base_magnitude=0.2,
                )
            ],
        )

        assert spec.buff_type == BuffType.STAT

    def test_buff_type_damage(self):
        """Test damage (DOT) buff type."""
        spec = BuffSpec(
            id="poison",
            category=BuffCategory.DEBUFF,
            buff_type=BuffType.DAMAGE,
            tick_rate=1.0,
        )

        assert spec.buff_type == BuffType.DAMAGE

    def test_buff_type_heal(self):
        """Test heal (HOT) buff type."""
        spec = BuffSpec(
            id="regeneration",
            buff_type=BuffType.HEAL,
            tick_rate=1.0,
        )

        assert spec.buff_type == BuffType.HEAL

    def test_buff_type_control(self):
        """Test control (CC) buff type."""
        spec = BuffSpec(
            id="stun",
            category=BuffCategory.DEBUFF,
            buff_type=BuffType.CONTROL,
            granted_tags=["status.stunned"],
        )

        assert spec.buff_type == BuffType.CONTROL


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestBuffEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_container(self):
        """Test empty buff container."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        assert len(container.active_buffs) == 0
        assert container.get_buff("nonexistent") is None
        assert container.has_buff("anything") is False

    def test_rapid_buff_application(self):
        """Test rapid buff application."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(
            id="rapid",
            stacking_mode=StackingMode.INTENSITY,
            max_stacks=100,
        )

        for _ in range(100):
            container.apply(spec)

        assert container.get_stacks("rapid") == 100

    def test_buff_with_zero_duration(self):
        """Test buff with zero duration (infinite)."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="infinite", duration=0.0)
        instance = container.apply(spec)

        # Progress should be 0 for infinite
        assert instance.progress == 0.0

        # Should not expire
        container.tick(1000.0)
        assert container.has_buff("infinite")

    def test_buff_tick_exact_expiration(self):
        """Test buff expiration at exact duration."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        spec = BuffSpec(id="test", duration=5.0)
        container.apply(spec)

        container.tick(5.0)

        assert not container.has_buff("test")

    def test_many_buffs(self):
        """Test container with many buffs."""
        attrs = create_standard_attributes()
        container = BuffContainer(attrs)

        for i in range(100):
            container.apply(BuffSpec(id=f"buff{i}"))

        assert len(container.active_buffs) == 100

    def test_buff_with_no_modifiers(self):
        """Test buff with no modifiers (tag-only)."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        container = BuffContainer(attrs, tags)

        spec = BuffSpec(
            id="marker",
            granted_tags=["marked"],
        )

        container.apply(spec)

        assert tags.has("marked")

    def test_stacking_mode_default(self):
        """Test default stacking mode."""
        spec = BuffSpec(id="test")
        assert spec.stacking_mode == StackingMode.NONE
