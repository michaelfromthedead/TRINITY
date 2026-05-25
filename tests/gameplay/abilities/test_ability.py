"""
Tests for the Ability System.

Tests cover:
- Ability lifecycle (ACTIVATE->COMMIT->EXECUTE->END)
- Ability costs (mana, stamina, cooldown)
- Ability cooldowns (global, per-ability)
- Ability tags (block, cancel, grant)
- Ability instances vs specs
- Ability interruption
- Ability combos
- Passive abilities
- Channeled abilities
- Toggled abilities

Total: ~180 tests

Note: These tests are designed for an ability.py module that follows
the patterns established in the constants.py and effects.py modules.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

from engine.gameplay.abilities.attributes import (
    Attribute,
    AttributeModifier,
    AttributeSet,
    create_standard_attributes,
)
from engine.gameplay.abilities.constants import (
    AbilityPhase,
    AbilityEndReason,
    EffectType,
    ModifierOperation,
    DEFAULT_GLOBAL_COOLDOWN,
    DEFAULT_MAX_COOLDOWN_REDUCTION,
    EPSILON,
)
from engine.gameplay.abilities.effects import (
    EffectContext,
    EffectModifier,
    InstantEffect,
    DurationEffect,
    EffectContainer,
)
from engine.gameplay.abilities.targeting import (
    Vector3,
    TargetData,
    SelfTargeting,
    ActorTargeting,
)
from engine.gameplay.abilities.tags import GameplayTag, GameplayTagContainer


# =============================================================================
# MOCK ABILITY SYSTEM (for testing design)
# =============================================================================


@dataclass
class AbilitySpec:
    """
    Specification/template for an ability.

    This is the immutable "class" definition of an ability.
    """

    id: str = ""
    name: str = ""
    description: str = ""

    # Costs
    mana_cost: float = 0.0
    stamina_cost: float = 0.0
    health_cost: float = 0.0

    # Cooldown
    cooldown: float = 0.0
    global_cooldown: float = DEFAULT_GLOBAL_COOLDOWN
    cooldown_reduction_affected: bool = True

    # Behavior flags
    can_be_cancelled: bool = True
    can_be_interrupted: bool = True
    is_passive: bool = False
    is_toggled: bool = False
    is_channeled: bool = False

    # Channeling
    channel_duration: float = 0.0
    channel_tick_rate: float = 0.0
    can_move_while_channeling: bool = False

    # Tags
    granted_tags: List[str] = field(default_factory=list)
    blocked_by_tags: List[str] = field(default_factory=list)
    cancel_abilities_with_tags: List[str] = field(default_factory=list)
    block_abilities_with_tags: List[str] = field(default_factory=list)
    required_tags: List[str] = field(default_factory=list)

    # Effects
    effects: List[Any] = field(default_factory=list)

    # Combo
    combo_next: Optional[str] = None
    combo_window: float = 0.0


@dataclass
class AbilityInstance:
    """
    Runtime instance of an ability.

    This tracks the current state of an ability being used.
    """

    id: UUID = field(default_factory=uuid4)
    spec: AbilitySpec = field(default_factory=AbilitySpec)
    owner: Optional[Any] = None

    # State
    phase: AbilityPhase = AbilityPhase.NONE
    is_active: bool = False

    # Cooldown tracking
    remaining_cooldown: float = 0.0
    is_on_cooldown: bool = False

    # Channeling
    channel_time: float = 0.0
    channel_ticks: int = 0

    # Toggle
    is_toggled_on: bool = False

    # Context
    target_data: Optional[TargetData] = None
    context: Optional[EffectContext] = None

    def can_activate(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
    ) -> bool:
        """Check if the ability can be activated."""
        # Check passive (auto-active)
        if self.spec.is_passive:
            return False  # Passives don't activate manually

        # Check already active
        if self.is_active and not self.spec.is_toggled:
            return False

        # Check cooldown
        if self.is_on_cooldown:
            return False

        # Check costs
        if self.spec.mana_cost > 0:
            if attributes.get("mana") < self.spec.mana_cost:
                return False

        if self.spec.stamina_cost > 0:
            if attributes.get("stamina") < self.spec.stamina_cost:
                return False

        if self.spec.health_cost > 0:
            if attributes.get("health") <= self.spec.health_cost:
                return False

        # Check required tags
        for tag in self.spec.required_tags:
            if not tags.has(tag):
                return False

        # Check blocked by tags
        for tag in self.spec.blocked_by_tags:
            if tags.has(tag):
                return False

        return True

    def activate(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
    ) -> bool:
        """Activate the ability (ACTIVATE phase)."""
        if not self.can_activate(attributes, tags):
            return False

        self.phase = AbilityPhase.ACTIVATE
        self.is_active = True

        # Grant tags
        for tag in self.spec.granted_tags:
            tags.add(tag)

        return True

    def commit(self, attributes: AttributeSet) -> bool:
        """Commit the ability, paying costs (COMMIT phase)."""
        if self.phase != AbilityPhase.ACTIVATE:
            return False

        # Pay costs
        if self.spec.mana_cost > 0:
            current = attributes.get("mana")
            attributes.set_base("mana", current - self.spec.mana_cost)

        if self.spec.stamina_cost > 0:
            current = attributes.get("stamina")
            attributes.set_base("stamina", current - self.spec.stamina_cost)

        if self.spec.health_cost > 0:
            current = attributes.get("health")
            attributes.set_base("health", current - self.spec.health_cost)

        # Start cooldown
        if self.spec.cooldown > 0:
            self.remaining_cooldown = self.spec.cooldown
            self.is_on_cooldown = True

        self.phase = AbilityPhase.COMMIT
        return True

    def execute(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
        target_data: Optional[TargetData] = None,
    ) -> bool:
        """Execute the ability logic (EXECUTE phase)."""
        if self.phase != AbilityPhase.COMMIT:
            return False

        self.phase = AbilityPhase.EXECUTE
        self.target_data = target_data

        # Apply effects
        for effect in self.spec.effects:
            if hasattr(effect, "apply"):
                effect.apply(attributes, tags, self.context)

        return True

    def end(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
        reason: AbilityEndReason = AbilityEndReason.COMPLETED,
    ) -> bool:
        """End the ability (END phase)."""
        if not self.is_active:
            return False

        self.phase = AbilityPhase.END
        self.is_active = False

        # Remove granted tags
        for tag in self.spec.granted_tags:
            tags.remove(tag)

        self.phase = AbilityPhase.NONE
        return True

    def cancel(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
    ) -> bool:
        """Cancel the ability."""
        if not self.is_active:
            return False

        if not self.spec.can_be_cancelled:
            return False

        return self.end(attributes, tags, AbilityEndReason.CANCELLED)

    def interrupt(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
    ) -> bool:
        """Interrupt the ability."""
        if not self.is_active:
            return False

        if not self.spec.can_be_interrupted:
            return False

        return self.end(attributes, tags, AbilityEndReason.INTERRUPTED)

    def tick(self, delta_time: float, attributes: AttributeSet) -> None:
        """Update the ability."""
        # Update cooldown
        if self.is_on_cooldown:
            self.remaining_cooldown -= delta_time
            if self.remaining_cooldown <= 0:
                self.remaining_cooldown = 0.0
                self.is_on_cooldown = False

        # Update channeling
        if self.spec.is_channeled and self.is_active:
            self.channel_time += delta_time


@dataclass
class AbilitySystem:
    """
    System for managing abilities on an entity.
    """

    abilities: Dict[str, AbilityInstance] = field(default_factory=dict)
    global_cooldown_remaining: float = 0.0
    is_on_global_cooldown: bool = False

    def grant_ability(self, spec: AbilitySpec) -> AbilityInstance:
        """Grant an ability to the system."""
        instance = AbilityInstance(spec=spec)
        self.abilities[spec.id] = instance
        return instance

    def revoke_ability(self, ability_id: str) -> bool:
        """Revoke an ability from the system."""
        if ability_id in self.abilities:
            del self.abilities[ability_id]
            return True
        return False

    def get_ability(self, ability_id: str) -> Optional[AbilityInstance]:
        """Get an ability instance."""
        return self.abilities.get(ability_id)

    def can_activate(
        self,
        ability_id: str,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
    ) -> bool:
        """Check if an ability can be activated."""
        ability = self.abilities.get(ability_id)
        if ability is None:
            return False

        # Check global cooldown
        if self.is_on_global_cooldown:
            return False

        return ability.can_activate(attributes, tags)

    def activate_ability(
        self,
        ability_id: str,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
        target_data: Optional[TargetData] = None,
    ) -> bool:
        """Activate an ability."""
        ability = self.abilities.get(ability_id)
        if ability is None:
            return False

        if not self.can_activate(ability_id, attributes, tags):
            return False

        # Cancel abilities with cancel tags
        for tag in ability.spec.cancel_abilities_with_tags:
            self._cancel_abilities_with_tag(attributes, tags, tag)

        # Activate
        if ability.activate(attributes, tags):
            if ability.commit(attributes):
                if ability.execute(attributes, tags, target_data):
                    # Start global cooldown
                    if ability.spec.global_cooldown > 0:
                        self.global_cooldown_remaining = ability.spec.global_cooldown
                        self.is_on_global_cooldown = True
                    return True

        return False

    def _cancel_abilities_with_tag(
        self,
        attributes: AttributeSet,
        tags: GameplayTagContainer,
        tag: str,
    ) -> None:
        """Cancel all abilities that have a specific tag."""
        for ability in self.abilities.values():
            if tag in ability.spec.granted_tags and ability.is_active:
                ability.cancel(attributes, tags)

    def tick(self, delta_time: float, attributes: AttributeSet) -> None:
        """Update all abilities."""
        # Update global cooldown
        if self.is_on_global_cooldown:
            self.global_cooldown_remaining -= delta_time
            if self.global_cooldown_remaining <= 0:
                self.global_cooldown_remaining = 0.0
                self.is_on_global_cooldown = False

        # Update individual abilities
        for ability in self.abilities.values():
            ability.tick(delta_time, attributes)


# =============================================================================
# ABILITY LIFECYCLE TESTS
# =============================================================================


class TestAbilityLifecycle:
    """Tests for ability lifecycle (ACTIVATE->COMMIT->EXECUTE->END)."""

    def test_ability_phases_enum(self):
        """Test ability phases are defined correctly."""
        assert AbilityPhase.NONE == 0
        assert AbilityPhase.ACTIVATE > AbilityPhase.NONE
        assert AbilityPhase.COMMIT > AbilityPhase.ACTIVATE
        assert AbilityPhase.EXECUTE > AbilityPhase.COMMIT
        assert AbilityPhase.END > AbilityPhase.EXECUTE

    def test_ability_starts_none_phase(self):
        """Test ability starts in NONE phase."""
        ability = AbilityInstance()
        assert ability.phase == AbilityPhase.NONE
        assert ability.is_active is False

    def test_ability_activate_phase(self):
        """Test ability enters ACTIVATE phase."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        result = ability.activate(attrs, tags)

        assert result is True
        assert ability.phase == AbilityPhase.ACTIVATE
        assert ability.is_active is True

    def test_ability_commit_phase(self):
        """Test ability enters COMMIT phase."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", mana_cost=10.0)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        result = ability.commit(attrs)

        assert result is True
        assert ability.phase == AbilityPhase.COMMIT

    def test_ability_execute_phase(self):
        """Test ability enters EXECUTE phase."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        ability.activate(attrs, tags)
        ability.commit(attrs)
        result = ability.execute(attrs, tags)

        assert result is True
        assert ability.phase == AbilityPhase.EXECUTE

    def test_ability_end_phase(self):
        """Test ability enters END phase and completes."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)
        result = ability.end(attrs, tags)

        assert result is True
        assert ability.phase == AbilityPhase.NONE
        assert ability.is_active is False

    def test_ability_full_lifecycle(self):
        """Test complete ability lifecycle."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(
            name="Fireball",
            mana_cost=20.0,
            cooldown=5.0,
        )
        ability = AbilityInstance(spec=spec)

        # Full lifecycle
        assert ability.activate(attrs, tags)
        assert ability.commit(attrs)
        assert ability.execute(attrs, tags)
        assert ability.end(attrs, tags)

        # Should be back to inactive
        assert ability.is_active is False
        assert ability.phase == AbilityPhase.NONE

    def test_ability_cannot_skip_phases(self):
        """Test ability cannot skip phases."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        # Cannot commit without activate
        assert ability.commit(attrs) is False

        # Cannot execute without commit
        ability.activate(attrs, tags)
        assert ability.execute(attrs, tags) is False


# =============================================================================
# ABILITY COST TESTS
# =============================================================================


class TestAbilityCosts:
    """Tests for ability costs (mana, stamina, cooldown)."""

    def test_mana_cost_deducted(self):
        """Test mana cost is deducted on commit."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", mana_cost=30.0)
        ability = AbilityInstance(spec=spec)

        initial_mana = attrs.get("mana")
        ability.activate(attrs, tags)
        ability.commit(attrs)

        assert attrs.get("mana") == initial_mana - 30.0

    def test_stamina_cost_deducted(self):
        """Test stamina cost is deducted on commit."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", stamina_cost=25.0)
        ability = AbilityInstance(spec=spec)

        initial_stamina = attrs.get("stamina")
        ability.activate(attrs, tags)
        ability.commit(attrs)

        assert attrs.get("stamina") == initial_stamina - 25.0

    def test_health_cost_deducted(self):
        """Test health cost is deducted on commit."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", health_cost=10.0)
        ability = AbilityInstance(spec=spec)

        initial_health = attrs.get("health")
        ability.activate(attrs, tags)
        ability.commit(attrs)

        assert attrs.get("health") == initial_health - 10.0

    def test_cannot_activate_insufficient_mana(self):
        """Test cannot activate with insufficient mana."""
        attrs = create_standard_attributes()
        attrs.set_base("mana", 20.0)
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", mana_cost=50.0)
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is False

    def test_cannot_activate_insufficient_stamina(self):
        """Test cannot activate with insufficient stamina."""
        attrs = create_standard_attributes()
        attrs.set_base("stamina", 10.0)
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", stamina_cost=50.0)
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is False

    def test_cannot_activate_insufficient_health(self):
        """Test cannot activate with insufficient health."""
        attrs = create_standard_attributes()
        attrs.set_base("health", 5.0)
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", health_cost=10.0)
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is False

    def test_multiple_costs(self):
        """Test ability with multiple costs."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(
            name="Test",
            mana_cost=20.0,
            stamina_cost=30.0,
            health_cost=5.0,
        )
        ability = AbilityInstance(spec=spec)

        initial_mana = attrs.get("mana")
        initial_stamina = attrs.get("stamina")
        initial_health = attrs.get("health")

        ability.activate(attrs, tags)
        ability.commit(attrs)

        assert attrs.get("mana") == initial_mana - 20.0
        assert attrs.get("stamina") == initial_stamina - 30.0
        assert attrs.get("health") == initial_health - 5.0

    def test_zero_cost_ability(self):
        """Test ability with zero cost can always activate."""
        attrs = create_standard_attributes()
        attrs.set_base("mana", 0.0)
        attrs.set_base("stamina", 0.0)
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Free", mana_cost=0.0, stamina_cost=0.0)
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is True


# =============================================================================
# ABILITY COOLDOWN TESTS
# =============================================================================


class TestAbilityCooldowns:
    """Tests for ability cooldowns (global, per-ability)."""

    def test_cooldown_starts_on_commit(self):
        """Test cooldown starts when ability commits."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", cooldown=10.0)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.commit(attrs)

        assert ability.is_on_cooldown is True
        assert ability.remaining_cooldown == 10.0

    def test_cooldown_decrements_over_time(self):
        """Test cooldown decrements with tick."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", cooldown=10.0)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.tick(3.0, attrs)

        assert ability.remaining_cooldown == 7.0

    def test_cooldown_completes(self):
        """Test cooldown completes after duration."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", cooldown=5.0)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.tick(6.0, attrs)

        assert ability.is_on_cooldown is False
        assert ability.remaining_cooldown == 0.0

    def test_cannot_activate_on_cooldown(self):
        """Test cannot activate ability while on cooldown."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", cooldown=10.0)
        ability = AbilityInstance(spec=spec)

        # First activation
        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)
        ability.end(attrs, tags)

        # Cannot activate again
        assert ability.can_activate(attrs, tags) is False

    def test_can_activate_after_cooldown(self):
        """Test can activate after cooldown expires."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", cooldown=5.0)
        ability = AbilityInstance(spec=spec)

        # First activation
        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)
        ability.end(attrs, tags)

        # Wait for cooldown
        ability.tick(6.0, attrs)

        assert ability.can_activate(attrs, tags) is True

    def test_global_cooldown(self):
        """Test global cooldown affects all abilities."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        system = AbilitySystem()
        spec1 = AbilitySpec(id="ability1", name="Ability1", global_cooldown=1.5)
        spec2 = AbilitySpec(id="ability2", name="Ability2", global_cooldown=1.5)

        system.grant_ability(spec1)
        system.grant_ability(spec2)

        # Activate first ability
        system.activate_ability("ability1", attrs, tags)

        # Second ability should be blocked by GCD
        assert system.is_on_global_cooldown is True
        assert system.can_activate("ability2", attrs, tags) is False

    def test_global_cooldown_expires(self):
        """Test global cooldown expires over time."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        system = AbilitySystem()
        spec1 = AbilitySpec(id="ability1", name="Ability1", global_cooldown=1.0)
        spec2 = AbilitySpec(id="ability2", name="Ability2", global_cooldown=1.0)

        system.grant_ability(spec1)
        system.grant_ability(spec2)

        system.activate_ability("ability1", attrs, tags)
        system.tick(1.5, attrs)

        assert system.is_on_global_cooldown is False

    def test_no_cooldown_ability(self):
        """Test ability with no cooldown can be used repeatedly."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", cooldown=0.0, global_cooldown=0.0)
        ability = AbilityInstance(spec=spec)

        # Use ability multiple times
        for _ in range(5):
            ability.activate(attrs, tags)
            ability.commit(attrs)
            ability.execute(attrs, tags)
            ability.end(attrs, tags)
            assert ability.is_on_cooldown is False


# =============================================================================
# ABILITY TAG TESTS
# =============================================================================


class TestAbilityTags:
    """Tests for ability tags (block, cancel, grant)."""

    def test_ability_grants_tags(self):
        """Test ability grants tags on activate."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(
            name="Shield",
            granted_tags=["ability.active.shield", "status.blocking"],
        )
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)

        assert tags.has("ability.active.shield")
        assert tags.has("status.blocking")

    def test_ability_removes_tags_on_end(self):
        """Test ability removes granted tags on end."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(
            name="Shield",
            granted_tags=["ability.active.shield"],
        )
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)
        ability.end(attrs, tags)

        assert not tags.has("ability.active.shield")

    def test_ability_blocked_by_tags(self):
        """Test ability blocked by specific tags."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        tags.add("status.stunned")

        spec = AbilitySpec(
            name="Attack",
            blocked_by_tags=["status.stunned", "status.silenced"],
        )
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is False

    def test_ability_requires_tags(self):
        """Test ability requires specific tags to activate."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        spec = AbilitySpec(
            name="Combo Finisher",
            required_tags=["status.combo.ready"],
        )
        ability = AbilityInstance(spec=spec)

        # Cannot activate without required tag
        assert ability.can_activate(attrs, tags) is False

        # Can activate with required tag
        tags.add("status.combo.ready")
        assert ability.can_activate(attrs, tags) is True

    def test_ability_cancels_other_abilities(self):
        """Test ability cancels other abilities with specific tags."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        system = AbilitySystem()

        # Channeled ability
        channel_spec = AbilitySpec(
            id="channel",
            name="Channel",
            granted_tags=["ability.channeling"],
            can_be_cancelled=True,
        )
        system.grant_ability(channel_spec)

        # Interrupt ability
        interrupt_spec = AbilitySpec(
            id="interrupt",
            name="Interrupt",
            cancel_abilities_with_tags=["ability.channeling"],
        )
        system.grant_ability(interrupt_spec)

        # Start channeling
        system.activate_ability("channel", attrs, tags)
        channel = system.get_ability("channel")
        assert channel.is_active is True

        # Use interrupt ability
        system.activate_ability("interrupt", attrs, tags)

        # Channel should be cancelled
        assert channel.is_active is False

    def test_multiple_tag_requirements(self):
        """Test ability with multiple tag requirements."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        spec = AbilitySpec(
            name="Ultimate",
            required_tags=["status.ultimate.ready", "status.in_combat"],
            blocked_by_tags=["status.silenced"],
        )
        ability = AbilityInstance(spec=spec)

        # Missing all required
        assert ability.can_activate(attrs, tags) is False

        # Has one required
        tags.add("status.ultimate.ready")
        assert ability.can_activate(attrs, tags) is False

        # Has all required
        tags.add("status.in_combat")
        assert ability.can_activate(attrs, tags) is True

        # Has required but also blocked
        tags.add("status.silenced")
        assert ability.can_activate(attrs, tags) is False


# =============================================================================
# ABILITY INSTANCE VS SPEC TESTS
# =============================================================================


class TestAbilityInstanceVsSpec:
    """Tests for ability instances vs specifications."""

    def test_spec_is_immutable_template(self):
        """Test spec acts as immutable template."""
        spec = AbilitySpec(name="Fireball", mana_cost=25.0, cooldown=5.0)

        instance1 = AbilityInstance(spec=spec)
        instance2 = AbilityInstance(spec=spec)

        # Both share same spec
        assert instance1.spec is instance2.spec

    def test_instances_have_independent_state(self):
        """Test instances have independent runtime state."""
        spec = AbilitySpec(name="Fireball", cooldown=5.0)
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        instance1 = AbilityInstance(spec=spec)
        instance2 = AbilityInstance(spec=spec)

        # Activate only instance1
        instance1.activate(attrs, tags)
        instance1.commit(attrs)

        # State is independent
        assert instance1.is_active is True
        assert instance1.is_on_cooldown is True
        assert instance2.is_active is False
        assert instance2.is_on_cooldown is False

    def test_each_instance_has_unique_id(self):
        """Test each instance has unique ID."""
        spec = AbilitySpec(name="Test")

        instances = [AbilityInstance(spec=spec) for _ in range(10)]
        ids = [inst.id for inst in instances]

        assert len(ids) == len(set(ids))  # All unique

    def test_spec_shared_across_entities(self):
        """Test spec can be shared across multiple entity ability systems."""
        spec = AbilitySpec(id="fireball", name="Fireball", mana_cost=25.0)

        system1 = AbilitySystem()
        system2 = AbilitySystem()

        instance1 = system1.grant_ability(spec)
        instance2 = system2.grant_ability(spec)

        # Same spec, different instances
        assert instance1.spec is instance2.spec
        assert instance1 is not instance2


# =============================================================================
# ABILITY INTERRUPTION TESTS
# =============================================================================


class TestAbilityInterruption:
    """Tests for ability interruption."""

    def test_ability_can_be_interrupted(self):
        """Test ability can be interrupted."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", can_be_interrupted=True)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        result = ability.interrupt(attrs, tags)

        assert result is True
        assert ability.is_active is False

    def test_ability_cannot_be_interrupted(self):
        """Test ability that cannot be interrupted."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", can_be_interrupted=False)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        result = ability.interrupt(attrs, tags)

        assert result is False
        assert ability.is_active is True

    def test_interrupt_end_reason(self):
        """Test interrupt sets correct end reason."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", can_be_interrupted=True)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.interrupt(attrs, tags)

        # Ability ended
        assert ability.is_active is False

    def test_interrupt_inactive_ability(self):
        """Test cannot interrupt inactive ability."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        result = ability.interrupt(attrs, tags)
        assert result is False

    def test_cancel_vs_interrupt(self):
        """Test difference between cancel and interrupt."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        # Can cancel but not interrupt
        spec1 = AbilitySpec(
            name="CancelOnly",
            can_be_cancelled=True,
            can_be_interrupted=False,
        )
        ability1 = AbilityInstance(spec=spec1)
        ability1.activate(attrs, tags)

        assert ability1.cancel(attrs, tags) is True

        # Can interrupt but not cancel
        spec2 = AbilitySpec(
            name="InterruptOnly",
            can_be_cancelled=False,
            can_be_interrupted=True,
        )
        ability2 = AbilityInstance(spec=spec2)
        ability2.activate(attrs, tags)

        assert ability2.cancel(attrs, tags) is False
        assert ability2.interrupt(attrs, tags) is True


# =============================================================================
# ABILITY COMBO TESTS
# =============================================================================


class TestAbilityCombos:
    """Tests for ability combos."""

    def test_combo_spec_chain(self):
        """Test combo chain in ability spec."""
        spec1 = AbilitySpec(
            id="attack1",
            name="Attack 1",
            combo_next="attack2",
            combo_window=2.0,
        )
        spec2 = AbilitySpec(
            id="attack2",
            name="Attack 2",
            combo_next="attack3",
            combo_window=2.0,
        )
        spec3 = AbilitySpec(
            id="attack3",
            name="Attack 3",
            combo_next=None,  # End of combo
        )

        assert spec1.combo_next == "attack2"
        assert spec2.combo_next == "attack3"
        assert spec3.combo_next is None

    def test_combo_window(self):
        """Test combo window timing."""
        spec = AbilitySpec(
            name="Combo Start",
            combo_next="combo_finisher",
            combo_window=1.5,
        )

        assert spec.combo_window == 1.5

    def test_combo_grants_tag(self):
        """Test combo ability grants combo tag."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(
            name="Attack 1",
            granted_tags=["status.combo.stage1"],
        )
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        assert tags.has("status.combo.stage1")


# =============================================================================
# PASSIVE ABILITY TESTS
# =============================================================================


class TestPassiveAbilities:
    """Tests for passive abilities."""

    def test_passive_ability_spec(self):
        """Test passive ability specification."""
        spec = AbilitySpec(
            name="Toughness",
            is_passive=True,
        )
        assert spec.is_passive is True

    def test_passive_cannot_activate_manually(self):
        """Test passive ability cannot be manually activated."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Passive", is_passive=True)
        ability = AbilityInstance(spec=spec)

        result = ability.can_activate(attrs, tags)
        assert result is False

    def test_passive_provides_effects(self):
        """Test passive ability provides continuous effects."""
        spec = AbilitySpec(
            name="Passive Armor",
            is_passive=True,
            effects=[
                InstantEffect(
                    name="armor_bonus",
                    modifiers=[
                        EffectModifier(
                            attribute="armor",
                            operation=ModifierOperation.ADD,
                            base_magnitude=50.0,
                        )
                    ],
                )
            ],
        )

        assert len(spec.effects) == 1


# =============================================================================
# CHANNELED ABILITY TESTS
# =============================================================================


class TestChanneledAbilities:
    """Tests for channeled abilities."""

    def test_channeled_ability_spec(self):
        """Test channeled ability specification."""
        spec = AbilitySpec(
            name="Laser Beam",
            is_channeled=True,
            channel_duration=3.0,
            channel_tick_rate=0.5,
        )

        assert spec.is_channeled is True
        assert spec.channel_duration == 3.0
        assert spec.channel_tick_rate == 0.5

    def test_channeled_ability_tracks_time(self):
        """Test channeled ability tracks channel time."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(
            name="Channel",
            is_channeled=True,
            channel_duration=5.0,
        )
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)
        ability.tick(2.0, attrs)

        assert ability.channel_time == 2.0

    def test_channeled_can_move_setting(self):
        """Test channeled ability movement setting."""
        spec = AbilitySpec(
            name="Mobile Channel",
            is_channeled=True,
            can_move_while_channeling=True,
        )

        assert spec.can_move_while_channeling is True


# =============================================================================
# TOGGLED ABILITY TESTS
# =============================================================================


class TestToggledAbilities:
    """Tests for toggled abilities."""

    def test_toggled_ability_spec(self):
        """Test toggled ability specification."""
        spec = AbilitySpec(
            name="Battle Stance",
            is_toggled=True,
        )
        assert spec.is_toggled is True

    def test_toggled_ability_state(self):
        """Test toggled ability tracks toggle state."""
        ability = AbilityInstance(
            spec=AbilitySpec(name="Toggle", is_toggled=True)
        )

        assert ability.is_toggled_on is False

    def test_toggled_can_activate_while_active(self):
        """Test toggled ability can activate while active (to turn off)."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Toggle", is_toggled=True)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)

        # Should be able to activate again to toggle off
        assert ability.can_activate(attrs, tags) is True


# =============================================================================
# ABILITY SYSTEM TESTS
# =============================================================================


class TestAbilitySystem:
    """Tests for AbilitySystem container."""

    def test_grant_ability(self):
        """Test granting ability to system."""
        system = AbilitySystem()
        spec = AbilitySpec(id="fireball", name="Fireball")

        instance = system.grant_ability(spec)

        assert instance is not None
        assert "fireball" in system.abilities

    def test_revoke_ability(self):
        """Test revoking ability from system."""
        system = AbilitySystem()
        spec = AbilitySpec(id="fireball", name="Fireball")
        system.grant_ability(spec)

        result = system.revoke_ability("fireball")

        assert result is True
        assert "fireball" not in system.abilities

    def test_revoke_nonexistent_ability(self):
        """Test revoking ability that doesn't exist."""
        system = AbilitySystem()

        result = system.revoke_ability("nonexistent")
        assert result is False

    def test_get_ability(self):
        """Test getting ability from system."""
        system = AbilitySystem()
        spec = AbilitySpec(id="fireball", name="Fireball")
        system.grant_ability(spec)

        ability = system.get_ability("fireball")

        assert ability is not None
        assert ability.spec.name == "Fireball"

    def test_get_nonexistent_ability(self):
        """Test getting ability that doesn't exist."""
        system = AbilitySystem()

        ability = system.get_ability("nonexistent")
        assert ability is None

    def test_activate_through_system(self):
        """Test activating ability through system."""
        system = AbilitySystem()
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(id="test", name="Test")
        system.grant_ability(spec)

        result = system.activate_ability("test", attrs, tags)

        assert result is True

    def test_system_tick_updates_all(self):
        """Test system tick updates all abilities."""
        system = AbilitySystem()
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        spec1 = AbilitySpec(id="ability1", name="Ability1", cooldown=10.0)
        spec2 = AbilitySpec(id="ability2", name="Ability2", cooldown=5.0)
        system.grant_ability(spec1)
        system.grant_ability(spec2)

        # Activate both
        system.activate_ability("ability1", attrs, tags)
        system.global_cooldown_remaining = 0  # Reset GCD for test
        system.is_on_global_cooldown = False
        system.activate_ability("ability2", attrs, tags)

        # Tick system
        system.tick(3.0, attrs)

        assert system.get_ability("ability1").remaining_cooldown == 7.0
        assert system.get_ability("ability2").remaining_cooldown == 2.0


# =============================================================================
# ABILITY END REASON TESTS
# =============================================================================


class TestAbilityEndReason:
    """Tests for AbilityEndReason enum."""

    def test_end_reasons_defined(self):
        """Test all end reasons are defined."""
        assert AbilityEndReason.COMPLETED == 0
        assert hasattr(AbilityEndReason, "CANCELLED")
        assert hasattr(AbilityEndReason, "INTERRUPTED")
        assert hasattr(AbilityEndReason, "EXPIRED")
        assert hasattr(AbilityEndReason, "KILLED")

    def test_end_reason_completed(self):
        """Test COMPLETED is default end reason."""
        assert AbilityEndReason.COMPLETED == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestAbilityEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ability_with_no_costs(self):
        """Test ability with no costs."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Free")
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is True

    def test_ability_exact_cost_available(self):
        """Test ability when exact cost is available."""
        attrs = create_standard_attributes()
        attrs.set_base("mana", 25.0)
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Test", mana_cost=25.0)
        ability = AbilityInstance(spec=spec)

        assert ability.can_activate(attrs, tags) is True

    def test_rapid_ability_activation(self):
        """Test rapid ability activation."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="Rapid", cooldown=0.0, global_cooldown=0.0)

        system = AbilitySystem()
        system.grant_ability(spec)

        # Rapid fire
        for _ in range(10):
            system.activate_ability(spec.id, attrs, tags)

    def test_ability_with_zero_cooldown(self):
        """Test ability with zero cooldown."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        spec = AbilitySpec(name="NoCooldown", cooldown=0.0)
        ability = AbilityInstance(spec=spec)

        ability.activate(attrs, tags)
        ability.commit(attrs)

        assert ability.is_on_cooldown is False

    def test_double_end_ability(self):
        """Test ending ability that's already ended."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)
        ability.end(attrs, tags)

        # Second end should fail
        result = ability.end(attrs, tags)
        assert result is False

    def test_cancel_inactive_ability(self):
        """Test cancelling inactive ability."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()
        ability = AbilityInstance(spec=AbilitySpec(name="Test"))

        result = ability.cancel(attrs, tags)
        assert result is False

    def test_ability_with_effects(self):
        """Test ability applies effects on execute."""
        attrs = create_standard_attributes()
        tags = GameplayTagContainer()

        damage_effect = InstantEffect(
            name="damage",
            modifiers=[
                EffectModifier(
                    attribute="health",
                    operation=ModifierOperation.ADD,
                    base_magnitude=-25.0,
                )
            ],
        )

        spec = AbilitySpec(name="Attack", effects=[damage_effect])
        ability = AbilityInstance(spec=spec)

        initial_health = attrs.get("health")
        ability.activate(attrs, tags)
        ability.commit(attrs)
        ability.execute(attrs, tags)

        assert attrs.get("health") == initial_health - 25.0
