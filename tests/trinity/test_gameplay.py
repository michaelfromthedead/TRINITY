"""
Tests for Tier 47: GAMEPLAY decorators.
"""

import pytest

from trinity.decorators.gameplay import (
    VALID_STACKING,
    ability,
    buff,
    gameplay_tag,
    interactable,
    quest,
    spawner,
)
from trinity.decorators.registry import Tier, registry


# =============================================================================
# ability tests
# =============================================================================


def test_ability_defaults():
    """Test @ability with default parameters."""

    @ability()
    class BasicAbility:
        pass

    assert hasattr(BasicAbility, "_ability")
    assert BasicAbility._ability is True
    assert BasicAbility._ability_cost == {}
    assert BasicAbility._ability_cooldown == 0.0
    assert BasicAbility._ability_tags == set()
    assert BasicAbility._ability_blocked_by == set()


def test_ability_with_cost():
    """Test @ability with resource cost."""

    @ability(cost={"mana": 50.0, "stamina": 20.0}, cooldown=5.0)
    class CostlyAbility:
        pass

    assert CostlyAbility._ability_cost == {"mana": 50.0, "stamina": 20.0}
    assert CostlyAbility._ability_cooldown == 5.0


def test_ability_with_tags():
    """Test @ability with tags and blocked_by."""

    @ability(tags={"offensive", "fire"}, blocked_by={"silenced", "stunned"})
    class TaggedAbility:
        pass

    assert "offensive" in TaggedAbility._ability_tags
    assert "fire" in TaggedAbility._ability_tags
    assert "silenced" in TaggedAbility._ability_blocked_by
    assert "stunned" in TaggedAbility._ability_blocked_by


def test_ability_invalid_cooldown():
    """Test @ability with invalid cooldown."""
    with pytest.raises(ValueError, match="cooldown must be >= 0"):

        @ability(cooldown=-1.0)
        class BadAbility:
            pass


def test_ability_registry():
    """Test that @ability is registered correctly."""
    spec = registry._decorators.get("ability")
    assert spec is not None
    assert spec.tier == Tier.GAMEPLAY
    assert spec.name == "ability"


# =============================================================================
# buff tests
# =============================================================================


def test_buff_defaults():
    """Test @buff with default parameters."""

    @buff()
    class BasicBuff:
        pass

    assert hasattr(BasicBuff, "_buff")
    assert BasicBuff._buff is True
    assert BasicBuff._buff_duration is None
    assert BasicBuff._buff_stacking == "none"
    assert BasicBuff._buff_max_stacks == 1
    assert BasicBuff._buff_tick_rate == 0.0


def test_buff_with_duration():
    """Test @buff with duration."""

    @buff(duration=10.0)
    class TimedBuff:
        pass

    assert TimedBuff._buff_duration == 10.0


def test_buff_all_stacking_modes():
    """Test @buff with all valid stacking modes."""
    for stacking_mode in VALID_STACKING:

        @buff(stacking=stacking_mode, max_stacks=5)
        class StackableBuff:
            pass

        assert StackableBuff._buff_stacking == stacking_mode
        assert StackableBuff._buff_max_stacks == 5


def test_buff_with_tick_rate():
    """Test @buff with tick rate (DoT/HoT)."""

    @buff(duration=20.0, tick_rate=2.0)
    class DamageOverTime:
        pass

    assert DamageOverTime._buff_duration == 20.0
    assert DamageOverTime._buff_tick_rate == 2.0


def test_buff_invalid_stacking():
    """Test @buff with invalid stacking mode."""
    with pytest.raises(ValueError, match="invalid stacking"):

        @buff(stacking="invalid_mode")
        class BadBuff:
            pass


def test_buff_invalid_max_stacks():
    """Test @buff with invalid max_stacks."""
    with pytest.raises(ValueError, match="max_stacks must be > 0"):

        @buff(max_stacks=0)
        class BadBuff:
            pass


def test_buff_invalid_tick_rate():
    """Test @buff with invalid tick_rate."""
    with pytest.raises(ValueError, match="tick_rate must be >= 0"):

        @buff(tick_rate=-1.0)
        class BadBuff:
            pass


# =============================================================================
# gameplay_tag tests
# =============================================================================


def test_gameplay_tag_basic():
    """Test @gameplay_tag with hierarchy."""

    @gameplay_tag(hierarchy="Status.Buff.Strength")
    class StrengthTag:
        pass

    assert hasattr(StrengthTag, "_gameplay_tag")
    assert StrengthTag._gameplay_tag is True
    assert StrengthTag._tag_hierarchy == "Status.Buff.Strength"


def test_gameplay_tag_complex_hierarchy():
    """Test @gameplay_tag with complex hierarchy."""

    @gameplay_tag(hierarchy="Character.State.Combat.Attacking.Heavy")
    class HeavyAttackTag:
        pass

    assert HeavyAttackTag._tag_hierarchy == "Character.State.Combat.Attacking.Heavy"


def test_gameplay_tag_empty_hierarchy():
    """Test @gameplay_tag with empty hierarchy."""
    with pytest.raises(ValueError, match="hierarchy must be non-empty"):

        @gameplay_tag(hierarchy="")
        class BadTag:
            pass


# =============================================================================
# spawner tests
# =============================================================================


def test_spawner_defaults():
    """Test @spawner with default parameters."""

    @spawner(prefab="Enemy")
    class EnemySpawner:
        pass

    assert hasattr(EnemySpawner, "_spawner")
    assert EnemySpawner._spawner is True
    assert EnemySpawner._spawner_prefab == "Enemy"
    assert EnemySpawner._spawner_pool_size == 10
    assert EnemySpawner._spawner_spawn_rate == 1.0
    assert EnemySpawner._spawner_max_alive is None


def test_spawner_custom():
    """Test @spawner with custom parameters."""

    @spawner(prefab="Boss", pool_size=5, spawn_rate=0.5, max_alive=3)
    class BossSpawner:
        pass

    assert BossSpawner._spawner_prefab == "Boss"
    assert BossSpawner._spawner_pool_size == 5
    assert BossSpawner._spawner_spawn_rate == 0.5
    assert BossSpawner._spawner_max_alive == 3


def test_spawner_empty_prefab():
    """Test @spawner with empty prefab."""
    with pytest.raises(ValueError, match="prefab must be non-empty"):

        @spawner(prefab="")
        class BadSpawner:
            pass


def test_spawner_invalid_pool_size():
    """Test @spawner with invalid pool_size."""
    with pytest.raises(ValueError, match="pool_size must be > 0"):

        @spawner(prefab="Entity", pool_size=0)
        class BadSpawner:
            pass


def test_spawner_invalid_spawn_rate():
    """Test @spawner with invalid spawn_rate."""
    with pytest.raises(ValueError, match="spawn_rate must be > 0"):

        @spawner(prefab="Entity", spawn_rate=0.0)
        class BadSpawner:
            pass


# =============================================================================
# interactable tests
# =============================================================================


def test_interactable_defaults():
    """Test @interactable with default parameters."""

    @interactable(prompt="Press E to interact")
    class Door:
        pass

    assert hasattr(Door, "_interactable")
    assert Door._interactable is True
    assert Door._interactable_prompt == "Press E to interact"
    assert Door._interactable_range == 2.0
    assert Door._interactable_hold_time == 0.0


def test_interactable_custom():
    """Test @interactable with custom parameters."""

    @interactable(prompt="Hold F to open", range=3.5, hold_time=1.5)
    class Chest:
        pass

    assert Chest._interactable_prompt == "Hold F to open"
    assert Chest._interactable_range == 3.5
    assert Chest._interactable_hold_time == 1.5


def test_interactable_empty_prompt():
    """Test @interactable with empty prompt."""
    with pytest.raises(ValueError, match="prompt must be non-empty"):

        @interactable(prompt="")
        class BadObject:
            pass


def test_interactable_invalid_range():
    """Test @interactable with invalid range."""
    with pytest.raises(ValueError, match="range must be > 0"):

        @interactable(prompt="Test", range=0.0)
        class BadObject:
            pass


def test_interactable_invalid_hold_time():
    """Test @interactable with invalid hold_time."""
    with pytest.raises(ValueError, match="hold_time must be >= 0"):

        @interactable(prompt="Test", hold_time=-1.0)
        class BadObject:
            pass


# =============================================================================
# quest tests
# =============================================================================


def test_quest_basic():
    """Test @quest with id only."""

    @quest(id="main_quest_1")
    class MainQuest:
        pass

    assert hasattr(MainQuest, "_quest")
    assert MainQuest._quest is True
    assert MainQuest._quest_id == "main_quest_1"
    assert MainQuest._quest_prerequisites == []
    assert MainQuest._quest_rewards == []


def test_quest_with_prerequisites():
    """Test @quest with prerequisites."""

    @quest(id="quest_2", prerequisites=["quest_1", "tutorial"])
    class SecondQuest:
        pass

    assert SecondQuest._quest_prerequisites == ["quest_1", "tutorial"]


def test_quest_with_rewards():
    """Test @quest with rewards."""

    @quest(id="quest_3", rewards=[("gold", 100), ("xp", 500)])
    class RewardQuest:
        pass

    assert RewardQuest._quest_rewards == [("gold", 100), ("xp", 500)]


def test_quest_full():
    """Test @quest with all parameters."""

    @quest(
        id="epic_quest",
        prerequisites=["prologue", "chapter_1"],
        rewards=[("gold", 500), ("xp", 2000), ("item_sword", 1)],
    )
    class EpicQuest:
        pass

    assert EpicQuest._quest_id == "epic_quest"
    assert len(EpicQuest._quest_prerequisites) == 2
    assert len(EpicQuest._quest_rewards) == 3


def test_quest_empty_id():
    """Test @quest with empty id."""
    with pytest.raises(ValueError, match="id must be non-empty"):

        @quest(id="")
        class BadQuest:
            pass


# =============================================================================
# Composition tests
# =============================================================================


def test_ability_with_buff():
    """Test composing @ability with @buff."""

    @ability(cost={"mana": 30}, cooldown=10.0, tags={"offensive"})
    @buff(duration=5.0, stacking="intensity", max_stacks=3)
    class DamageBuffAbility:
        pass

    # Check ability attributes
    assert DamageBuffAbility._ability is True
    assert DamageBuffAbility._ability_cost == {"mana": 30}
    assert "offensive" in DamageBuffAbility._ability_tags

    # Check buff attributes
    assert DamageBuffAbility._buff is True
    assert DamageBuffAbility._buff_duration == 5.0
    assert DamageBuffAbility._buff_stacking == "intensity"


def test_spawner_with_interactable():
    """Test spawner that can be interacted with."""

    @spawner(prefab="TreasureChest", pool_size=20, spawn_rate=0.1)
    @interactable(prompt="Open chest", range=2.5)
    class InteractableSpawner:
        pass

    assert InteractableSpawner._spawner is True
    assert InteractableSpawner._interactable is True


def test_quest_with_tag():
    """Test quest with gameplay tag."""

    @quest(id="tagged_quest", rewards=[("xp", 100)])
    @gameplay_tag(hierarchy="Quest.Main.Chapter1")
    class TaggedQuest:
        pass

    assert TaggedQuest._quest is True
    assert TaggedQuest._gameplay_tag is True
    assert TaggedQuest._tag_hierarchy == "Quest.Main.Chapter1"


# =============================================================================
# Tags and registry tests
# =============================================================================


def test_gameplay_tags():
    """Test that decorators add correct tags."""

    @ability(cooldown=5.0)
    @buff(duration=10.0)
    class AbilityWithBuff:
        pass

    tags = AbilityWithBuff._tags
    assert "ability" in tags
    assert "buff" in tags


def test_gameplay_registries():
    """Test that all decorators register to gameplay."""

    @ability()
    @buff()
    @gameplay_tag(hierarchy="Test")
    @spawner(prefab="Test")
    @interactable(prompt="Test")
    @quest(id="test")
    class GameplayEntity:
        pass

    assert "gameplay" in GameplayEntity._registries


def test_all_decorators_registered():
    """Test that all GAMEPLAY decorators are in registry."""
    expected = [
        "ability",
        "buff",
        "gameplay_tag",
        "spawner",
        "interactable",
        "quest",
    ]

    for name in expected:
        spec = registry._decorators.get(name)
        assert spec is not None, f"Decorator {name} not registered"
        assert spec.tier == Tier.GAMEPLAY


# =============================================================================
# Complex gameplay scenarios
# =============================================================================


def test_rpg_ability_system():
    """Test complex RPG ability with multiple decorators."""

    @ability(
        cost={"mana": 75, "stamina": 25},
        cooldown=15.0,
        tags={"offensive", "magic", "fire"},
        blocked_by={"silenced", "stunned", "disarmed"},
    )
    @buff(
        duration=8.0, stacking="duration", max_stacks=1, tick_rate=1.0
    )  # Burning DoT
    @gameplay_tag(hierarchy="Ability.Magic.Elemental.Fire.Fireball")
    class Fireball:
        pass

    assert Fireball._ability_cost == {"mana": 75, "stamina": 25}
    assert Fireball._ability_cooldown == 15.0
    assert "fire" in Fireball._ability_tags
    assert "silenced" in Fireball._ability_blocked_by
    assert Fireball._buff_duration == 8.0
    assert Fireball._buff_tick_rate == 1.0
    assert "Fireball" in Fireball._tag_hierarchy


def test_spawn_system():
    """Test entity spawner system."""

    @spawner(prefab="ZombieEnemy", pool_size=50, spawn_rate=2.0, max_alive=20)
    @gameplay_tag(hierarchy="Spawner.Enemy.Zombie")
    class ZombieSpawner:
        pass

    assert ZombieSpawner._spawner_prefab == "ZombieEnemy"
    assert ZombieSpawner._spawner_pool_size == 50
    assert ZombieSpawner._spawner_max_alive == 20


def test_interactive_quest_object():
    """Test interactive object that starts a quest."""

    @interactable(prompt="Read ancient tablet", range=1.5, hold_time=2.0)
    @quest(
        id="ancient_knowledge",
        prerequisites=["find_temple"],
        rewards=[("xp", 1000), ("ancient_rune", 1)],
    )
    @gameplay_tag(hierarchy="Quest.Trigger.Tablet")
    class AncientTablet:
        pass

    assert AncientTablet._interactable is True
    assert AncientTablet._interactable_hold_time == 2.0
    assert AncientTablet._quest is True
    assert AncientTablet._quest_id == "ancient_knowledge"
