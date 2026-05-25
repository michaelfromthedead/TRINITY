"""
Comprehensive tests for Quest Rewards.

Tests cover:
- Experience rewards
- Currency rewards
- Item rewards
- Reputation rewards
- Ability/skill rewards
- Choice-based rewards
- Bonus rewards (completion time)
- Reward scaling
"""

import pytest
from dataclasses import dataclass
from typing import Any, List, Dict
from unittest.mock import Mock, MagicMock

from engine.gameplay.quest.quest import QuestDefinition, QuestType


# =============================================================================
# Reward Data Classes (if not in source, define for testing)
# =============================================================================

@dataclass
class Reward:
    """Base class for quest rewards."""
    id: str
    description: str = ""

    def grant(self, context: Any) -> bool:
        """Grant the reward to the player."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Serialize reward to dictionary."""
        return {"id": self.id, "description": self.description}


@dataclass
class ExperienceReward(Reward):
    """Experience point reward."""
    amount: int = 0
    skill_type: str = ""  # Empty for general XP

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("amount must be >= 0")

    def grant(self, context: Any) -> bool:
        if hasattr(context, 'add_experience'):
            return context.add_experience(self.amount, self.skill_type)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "experience",
            "id": self.id,
            "amount": self.amount,
            "skill_type": self.skill_type,
        }


@dataclass
class CurrencyReward(Reward):
    """Currency/money reward."""
    amount: int = 0
    currency_type: str = "gold"

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("amount must be >= 0")

    def grant(self, context: Any) -> bool:
        if hasattr(context, 'add_currency'):
            return context.add_currency(self.amount, self.currency_type)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "currency",
            "id": self.id,
            "amount": self.amount,
            "currency_type": self.currency_type,
        }


@dataclass
class ItemReward(Reward):
    """Item reward."""
    item_id: str = ""
    quantity: int = 1
    quality: str = "normal"

    def __post_init__(self):
        if not self.item_id:
            raise ValueError("item_id cannot be empty")
        if self.quantity < 1:
            raise ValueError("quantity must be >= 1")

    def grant(self, context: Any) -> bool:
        if hasattr(context, 'add_item'):
            return context.add_item(self.item_id, self.quantity, self.quality)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "item",
            "id": self.id,
            "item_id": self.item_id,
            "quantity": self.quantity,
            "quality": self.quality,
        }


@dataclass
class ReputationReward(Reward):
    """Reputation/faction standing reward."""
    faction_id: str = ""
    amount: int = 0

    def __post_init__(self):
        if not self.faction_id:
            raise ValueError("faction_id cannot be empty")

    def grant(self, context: Any) -> bool:
        if hasattr(context, 'change_reputation'):
            return context.change_reputation(self.faction_id, self.amount)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "reputation",
            "id": self.id,
            "faction_id": self.faction_id,
            "amount": self.amount,
        }


@dataclass
class AbilityReward(Reward):
    """Ability/skill unlock reward."""
    ability_id: str = ""
    ability_level: int = 1

    def __post_init__(self):
        if not self.ability_id:
            raise ValueError("ability_id cannot be empty")
        if self.ability_level < 1:
            raise ValueError("ability_level must be >= 1")

    def grant(self, context: Any) -> bool:
        if hasattr(context, 'unlock_ability'):
            return context.unlock_ability(self.ability_id, self.ability_level)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ability",
            "id": self.id,
            "ability_id": self.ability_id,
            "ability_level": self.ability_level,
        }


@dataclass
class ChoiceReward(Reward):
    """Reward with player choice between options."""
    options: List[Reward] = None
    max_choices: int = 1

    def __post_init__(self):
        if self.options is None:
            self.options = []
        if self.max_choices < 1:
            raise ValueError("max_choices must be >= 1")

    def grant(self, context: Any, chosen_indices: List[int] = None) -> bool:
        if chosen_indices is None:
            return False
        if len(chosen_indices) > self.max_choices:
            return False

        success = True
        for idx in chosen_indices:
            if 0 <= idx < len(self.options):
                if not self.options[idx].grant(context):
                    success = False
        return success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "choice",
            "id": self.id,
            "options": [o.to_dict() for o in self.options],
            "max_choices": self.max_choices,
        }


@dataclass
class BonusReward(Reward):
    """Bonus reward based on conditions."""
    base_reward: Reward = None
    condition_type: str = "time"  # "time", "stealth", "no_damage", etc.
    threshold: float = 0.0
    multiplier: float = 1.0

    def check_condition(self, context: Any) -> bool:
        """Check if bonus condition is met."""
        if self.condition_type == "time":
            completion_time = getattr(context, 'completion_time', float('inf'))
            return completion_time <= self.threshold
        elif self.condition_type == "no_damage":
            damage_taken = getattr(context, 'damage_taken', 0)
            return damage_taken == 0
        elif self.condition_type == "stealth":
            detected = getattr(context, 'times_detected', 0)
            return detected == 0
        return False

    def grant(self, context: Any) -> bool:
        if self.base_reward and self.check_condition(context):
            # Apply multiplier if applicable
            if hasattr(self.base_reward, 'amount'):
                original = self.base_reward.amount
                self.base_reward.amount = int(original * self.multiplier)
                result = self.base_reward.grant(context)
                self.base_reward.amount = original
                return result
            return self.base_reward.grant(context)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "bonus",
            "id": self.id,
            "base_reward": self.base_reward.to_dict() if self.base_reward else None,
            "condition_type": self.condition_type,
            "threshold": self.threshold,
            "multiplier": self.multiplier,
        }


@dataclass
class ScaledReward(Reward):
    """Reward that scales with player level or other factors."""
    base_reward: Reward = None
    scale_factor: str = "level"  # "level", "difficulty", "completion_percent"
    min_scale: float = 1.0
    max_scale: float = 10.0
    scale_formula: str = "linear"  # "linear", "exponential", "logarithmic"

    def calculate_scale(self, context: Any) -> float:
        """Calculate the scaling multiplier."""
        if self.scale_factor == "level":
            level = getattr(context, 'player_level', 1)
            base_level = 1
            if self.scale_formula == "linear":
                scale = 1.0 + (level - base_level) * 0.1
            elif self.scale_formula == "exponential":
                scale = 1.0 * (1.1 ** (level - base_level))
            else:
                scale = 1.0
        elif self.scale_factor == "difficulty":
            difficulty = getattr(context, 'difficulty', 1)
            scale = 1.0 + (difficulty - 1) * 0.25
        else:
            scale = 1.0

        return max(self.min_scale, min(self.max_scale, scale))

    def grant(self, context: Any) -> bool:
        if self.base_reward is None:
            return False

        scale = self.calculate_scale(context)

        if hasattr(self.base_reward, 'amount'):
            original = self.base_reward.amount
            self.base_reward.amount = int(original * scale)
            result = self.base_reward.grant(context)
            self.base_reward.amount = original
            return result

        return self.base_reward.grant(context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "scaled",
            "id": self.id,
            "base_reward": self.base_reward.to_dict() if self.base_reward else None,
            "scale_factor": self.scale_factor,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "scale_formula": self.scale_formula,
        }


class RewardFactory:
    """Factory for creating rewards from data."""

    _types = {
        "experience": ExperienceReward,
        "currency": CurrencyReward,
        "item": ItemReward,
        "reputation": ReputationReward,
        "ability": AbilityReward,
        "choice": ChoiceReward,
        "bonus": BonusReward,
        "scaled": ScaledReward,
    }

    @classmethod
    def create(cls, reward_type: str, **kwargs) -> Reward:
        """Create a reward of the given type."""
        if reward_type not in cls._types:
            raise ValueError(f"Unknown reward type: {reward_type}")
        return cls._types[reward_type](**kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Reward:
        """Create a reward from dictionary data."""
        reward_type = data.pop("type")
        return cls.create(reward_type, **data)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_context():
    """Create a mock reward context."""
    context = Mock()
    context.add_experience = Mock(return_value=True)
    context.add_currency = Mock(return_value=True)
    context.add_item = Mock(return_value=True)
    context.change_reputation = Mock(return_value=True)
    context.unlock_ability = Mock(return_value=True)
    context.player_level = 10
    context.difficulty = 2
    context.completion_time = 100.0
    context.damage_taken = 0
    context.times_detected = 0
    return context


@pytest.fixture
def xp_reward():
    """Create an experience reward."""
    return ExperienceReward(id="xp_1", amount=100)


@pytest.fixture
def gold_reward():
    """Create a currency reward."""
    return CurrencyReward(id="gold_1", amount=500, currency_type="gold")


@pytest.fixture
def item_reward():
    """Create an item reward."""
    return ItemReward(id="item_1", item_id="sword_epic", quantity=1, quality="epic")


# =============================================================================
# Experience Reward Tests
# =============================================================================

class TestExperienceReward:
    """Tests for experience reward functionality."""

    def test_experience_reward_creation(self):
        """Test creating an experience reward."""
        reward = ExperienceReward(id="xp_basic", amount=100)
        assert reward.amount == 100
        assert reward.skill_type == ""

    def test_experience_reward_with_skill(self):
        """Test experience reward for specific skill."""
        reward = ExperienceReward(id="xp_combat", amount=50, skill_type="combat")
        assert reward.skill_type == "combat"

    def test_experience_reward_grant(self, mock_context, xp_reward):
        """Test granting experience reward."""
        result = xp_reward.grant(mock_context)

        assert result is True
        mock_context.add_experience.assert_called_once_with(100, "")

    def test_experience_reward_grant_with_skill(self, mock_context):
        """Test granting skill-specific experience."""
        reward = ExperienceReward(id="xp_magic", amount=75, skill_type="magic")
        result = reward.grant(mock_context)

        assert result is True
        mock_context.add_experience.assert_called_once_with(75, "magic")

    def test_experience_reward_negative_raises(self):
        """Test that negative experience raises error."""
        with pytest.raises(ValueError, match="amount must be >= 0"):
            ExperienceReward(id="invalid", amount=-50)

    def test_experience_reward_zero_valid(self):
        """Test that zero experience is valid."""
        reward = ExperienceReward(id="zero_xp", amount=0)
        assert reward.amount == 0

    def test_experience_reward_serialization(self, xp_reward):
        """Test experience reward serialization."""
        data = xp_reward.to_dict()

        assert data["type"] == "experience"
        assert data["amount"] == 100


# =============================================================================
# Currency Reward Tests
# =============================================================================

class TestCurrencyReward:
    """Tests for currency reward functionality."""

    def test_currency_reward_creation(self):
        """Test creating a currency reward."""
        reward = CurrencyReward(id="gold_basic", amount=1000)
        assert reward.amount == 1000
        assert reward.currency_type == "gold"

    def test_currency_reward_different_type(self):
        """Test currency reward with different currency type."""
        reward = CurrencyReward(id="gems", amount=50, currency_type="gems")
        assert reward.currency_type == "gems"

    def test_currency_reward_grant(self, mock_context, gold_reward):
        """Test granting currency reward."""
        result = gold_reward.grant(mock_context)

        assert result is True
        mock_context.add_currency.assert_called_once_with(500, "gold")

    def test_currency_reward_negative_raises(self):
        """Test that negative currency raises error."""
        with pytest.raises(ValueError, match="amount must be >= 0"):
            CurrencyReward(id="invalid", amount=-100)

    def test_currency_reward_multiple_types(self, mock_context):
        """Test multiple currency type rewards."""
        gold = CurrencyReward(id="gold", amount=100, currency_type="gold")
        silver = CurrencyReward(id="silver", amount=50, currency_type="silver")
        tokens = CurrencyReward(id="tokens", amount=10, currency_type="event_token")

        gold.grant(mock_context)
        silver.grant(mock_context)
        tokens.grant(mock_context)

        assert mock_context.add_currency.call_count == 3

    def test_currency_reward_serialization(self, gold_reward):
        """Test currency reward serialization."""
        data = gold_reward.to_dict()

        assert data["type"] == "currency"
        assert data["amount"] == 500
        assert data["currency_type"] == "gold"


# =============================================================================
# Item Reward Tests
# =============================================================================

class TestItemReward:
    """Tests for item reward functionality."""

    def test_item_reward_creation(self):
        """Test creating an item reward."""
        reward = ItemReward(id="sword", item_id="iron_sword", quantity=1)
        assert reward.item_id == "iron_sword"
        assert reward.quantity == 1
        assert reward.quality == "normal"

    def test_item_reward_with_quality(self):
        """Test item reward with quality."""
        reward = ItemReward(id="epic_sword", item_id="sword", quantity=1, quality="epic")
        assert reward.quality == "epic"

    def test_item_reward_multiple_quantity(self):
        """Test item reward with multiple quantity."""
        reward = ItemReward(id="potions", item_id="health_potion", quantity=10)
        assert reward.quantity == 10

    def test_item_reward_grant(self, mock_context, item_reward):
        """Test granting item reward."""
        result = item_reward.grant(mock_context)

        assert result is True
        mock_context.add_item.assert_called_once_with("sword_epic", 1, "epic")

    def test_item_reward_empty_id_raises(self):
        """Test that empty item_id raises error."""
        with pytest.raises(ValueError, match="item_id cannot be empty"):
            ItemReward(id="invalid", item_id="")

    def test_item_reward_zero_quantity_raises(self):
        """Test that zero quantity raises error."""
        with pytest.raises(ValueError, match="quantity must be >= 1"):
            ItemReward(id="invalid", item_id="item", quantity=0)

    def test_item_reward_serialization(self, item_reward):
        """Test item reward serialization."""
        data = item_reward.to_dict()

        assert data["type"] == "item"
        assert data["item_id"] == "sword_epic"
        assert data["quantity"] == 1
        assert data["quality"] == "epic"

    def test_item_reward_grant_fails_full_inventory(self, mock_context):
        """Test item reward when inventory is full."""
        mock_context.add_item.return_value = False

        reward = ItemReward(id="item", item_id="large_item", quantity=1)
        result = reward.grant(mock_context)

        assert result is False


# =============================================================================
# Reputation Reward Tests
# =============================================================================

class TestReputationReward:
    """Tests for reputation reward functionality."""

    def test_reputation_reward_creation(self):
        """Test creating a reputation reward."""
        reward = ReputationReward(id="rep_guild", faction_id="mages_guild", amount=100)
        assert reward.faction_id == "mages_guild"
        assert reward.amount == 100

    def test_reputation_reward_negative(self):
        """Test reputation reward with negative amount."""
        reward = ReputationReward(id="rep_enemy", faction_id="bandits", amount=-50)
        assert reward.amount == -50

    def test_reputation_reward_grant(self, mock_context):
        """Test granting reputation reward."""
        reward = ReputationReward(id="rep", faction_id="faction_a", amount=25)
        result = reward.grant(mock_context)

        assert result is True
        mock_context.change_reputation.assert_called_once_with("faction_a", 25)

    def test_reputation_reward_empty_faction_raises(self):
        """Test that empty faction_id raises error."""
        with pytest.raises(ValueError, match="faction_id cannot be empty"):
            ReputationReward(id="invalid", faction_id="", amount=100)

    def test_reputation_reward_multiple_factions(self, mock_context):
        """Test rewards affecting multiple factions."""
        reward1 = ReputationReward(id="r1", faction_id="faction_a", amount=50)
        reward2 = ReputationReward(id="r2", faction_id="faction_b", amount=-25)

        reward1.grant(mock_context)
        reward2.grant(mock_context)

        calls = mock_context.change_reputation.call_args_list
        assert len(calls) == 2

    def test_reputation_reward_serialization(self):
        """Test reputation reward serialization."""
        reward = ReputationReward(id="rep", faction_id="guild", amount=100)
        data = reward.to_dict()

        assert data["type"] == "reputation"
        assert data["faction_id"] == "guild"
        assert data["amount"] == 100


# =============================================================================
# Ability Reward Tests
# =============================================================================

class TestAbilityReward:
    """Tests for ability/skill unlock reward functionality."""

    def test_ability_reward_creation(self):
        """Test creating an ability reward."""
        reward = AbilityReward(id="skill_unlock", ability_id="fireball")
        assert reward.ability_id == "fireball"
        assert reward.ability_level == 1

    def test_ability_reward_with_level(self):
        """Test ability reward with specific level."""
        reward = AbilityReward(id="skill", ability_id="heal", ability_level=3)
        assert reward.ability_level == 3

    def test_ability_reward_grant(self, mock_context):
        """Test granting ability reward."""
        reward = AbilityReward(id="skill", ability_id="stealth", ability_level=2)
        result = reward.grant(mock_context)

        assert result is True
        mock_context.unlock_ability.assert_called_once_with("stealth", 2)

    def test_ability_reward_empty_id_raises(self):
        """Test that empty ability_id raises error."""
        with pytest.raises(ValueError, match="ability_id cannot be empty"):
            AbilityReward(id="invalid", ability_id="")

    def test_ability_reward_zero_level_raises(self):
        """Test that zero ability_level raises error."""
        with pytest.raises(ValueError, match="ability_level must be >= 1"):
            AbilityReward(id="invalid", ability_id="skill", ability_level=0)

    def test_ability_reward_serialization(self):
        """Test ability reward serialization."""
        reward = AbilityReward(id="skill", ability_id="teleport", ability_level=1)
        data = reward.to_dict()

        assert data["type"] == "ability"
        assert data["ability_id"] == "teleport"


# =============================================================================
# Choice Reward Tests
# =============================================================================

class TestChoiceReward:
    """Tests for choice-based reward functionality."""

    def test_choice_reward_creation(self):
        """Test creating a choice reward."""
        options = [
            ItemReward(id="opt1", item_id="sword", quantity=1),
            ItemReward(id="opt2", item_id="shield", quantity=1),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=1)

        assert len(reward.options) == 2
        assert reward.max_choices == 1

    def test_choice_reward_multiple_choices(self):
        """Test choice reward allowing multiple selections."""
        options = [
            ItemReward(id="opt1", item_id="item1", quantity=1),
            ItemReward(id="opt2", item_id="item2", quantity=1),
            ItemReward(id="opt3", item_id="item3", quantity=1),
        ]
        reward = ChoiceReward(id="multi_choice", options=options, max_choices=2)

        assert reward.max_choices == 2

    def test_choice_reward_grant_single(self, mock_context):
        """Test granting choice reward with single selection."""
        options = [
            CurrencyReward(id="gold", amount=100, currency_type="gold"),
            ItemReward(id="item", item_id="rare_gem", quantity=1),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=1)

        result = reward.grant(mock_context, chosen_indices=[0])

        assert result is True
        mock_context.add_currency.assert_called_once()
        mock_context.add_item.assert_not_called()

    def test_choice_reward_grant_multiple(self, mock_context):
        """Test granting choice reward with multiple selections."""
        options = [
            ExperienceReward(id="xp", amount=50),
            CurrencyReward(id="gold", amount=100, currency_type="gold"),
            ItemReward(id="item", item_id="potion", quantity=1),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=2)

        result = reward.grant(mock_context, chosen_indices=[0, 2])

        assert result is True
        mock_context.add_experience.assert_called_once()
        mock_context.add_item.assert_called_once()

    def test_choice_reward_exceed_max_choices(self, mock_context):
        """Test that exceeding max_choices fails."""
        options = [
            ItemReward(id="opt1", item_id="item1", quantity=1),
            ItemReward(id="opt2", item_id="item2", quantity=1),
            ItemReward(id="opt3", item_id="item3", quantity=1),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=1)

        result = reward.grant(mock_context, chosen_indices=[0, 1])

        assert result is False

    def test_choice_reward_invalid_index(self, mock_context):
        """Test choice reward with invalid index."""
        options = [
            ItemReward(id="opt1", item_id="item1", quantity=1),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=1)

        result = reward.grant(mock_context, chosen_indices=[5])

        assert result is True  # Returns True but grants nothing

    def test_choice_reward_no_selection(self, mock_context):
        """Test choice reward with no selection."""
        options = [
            ItemReward(id="opt1", item_id="item1", quantity=1),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=1)

        result = reward.grant(mock_context, chosen_indices=None)

        assert result is False

    def test_choice_reward_max_choices_zero_raises(self):
        """Test that max_choices=0 raises error."""
        with pytest.raises(ValueError, match="max_choices must be >= 1"):
            ChoiceReward(id="invalid", options=[], max_choices=0)

    def test_choice_reward_serialization(self):
        """Test choice reward serialization."""
        options = [
            ItemReward(id="opt1", item_id="item1", quantity=1),
            CurrencyReward(id="opt2", amount=100, currency_type="gold"),
        ]
        reward = ChoiceReward(id="choice", options=options, max_choices=1)

        data = reward.to_dict()

        assert data["type"] == "choice"
        assert len(data["options"]) == 2
        assert data["max_choices"] == 1


# =============================================================================
# Bonus Reward Tests
# =============================================================================

class TestBonusReward:
    """Tests for bonus reward functionality."""

    def test_bonus_reward_time_condition(self, mock_context):
        """Test bonus reward with time condition."""
        base = ExperienceReward(id="base_xp", amount=100)
        bonus = BonusReward(
            id="time_bonus",
            base_reward=base,
            condition_type="time",
            threshold=120.0,  # 2 minutes
            multiplier=1.5,
        )

        mock_context.completion_time = 100.0  # Under threshold
        result = bonus.grant(mock_context)

        assert result is True
        mock_context.add_experience.assert_called_with(150, "")  # 100 * 1.5

    def test_bonus_reward_time_condition_fails(self, mock_context):
        """Test bonus reward when time condition fails."""
        base = ExperienceReward(id="base_xp", amount=100)
        bonus = BonusReward(
            id="time_bonus",
            base_reward=base,
            condition_type="time",
            threshold=60.0,
            multiplier=1.5,
        )

        mock_context.completion_time = 120.0  # Over threshold
        result = bonus.grant(mock_context)

        assert result is False

    def test_bonus_reward_no_damage_condition(self, mock_context):
        """Test bonus reward with no-damage condition."""
        base = CurrencyReward(id="base_gold", amount=500, currency_type="gold")
        bonus = BonusReward(
            id="no_damage_bonus",
            base_reward=base,
            condition_type="no_damage",
            multiplier=2.0,
        )

        mock_context.damage_taken = 0
        result = bonus.grant(mock_context)

        assert result is True
        mock_context.add_currency.assert_called_with(1000, "gold")  # 500 * 2.0

    def test_bonus_reward_stealth_condition(self, mock_context):
        """Test bonus reward with stealth condition."""
        base = ExperienceReward(id="base_xp", amount=200)
        bonus = BonusReward(
            id="stealth_bonus",
            base_reward=base,
            condition_type="stealth",
            multiplier=1.25,
        )

        mock_context.times_detected = 0
        result = bonus.grant(mock_context)

        assert result is True
        mock_context.add_experience.assert_called_with(250, "")  # 200 * 1.25

    def test_bonus_reward_stealth_fails(self, mock_context):
        """Test bonus reward when detected."""
        base = ExperienceReward(id="base_xp", amount=200)
        bonus = BonusReward(
            id="stealth_bonus",
            base_reward=base,
            condition_type="stealth",
            multiplier=1.25,
        )

        mock_context.times_detected = 3
        result = bonus.grant(mock_context)

        assert result is False

    def test_bonus_reward_serialization(self):
        """Test bonus reward serialization."""
        base = ExperienceReward(id="base", amount=100)
        bonus = BonusReward(
            id="bonus",
            base_reward=base,
            condition_type="time",
            threshold=60.0,
            multiplier=1.5,
        )

        data = bonus.to_dict()

        assert data["type"] == "bonus"
        assert data["condition_type"] == "time"
        assert data["threshold"] == 60.0
        assert data["multiplier"] == 1.5


# =============================================================================
# Scaled Reward Tests
# =============================================================================

class TestScaledReward:
    """Tests for scaled reward functionality."""

    def test_scaled_reward_by_level(self, mock_context):
        """Test reward scaling by player level."""
        base = ExperienceReward(id="base_xp", amount=100)
        scaled = ScaledReward(
            id="scaled_xp",
            base_reward=base,
            scale_factor="level",
            scale_formula="linear",
        )

        mock_context.player_level = 10
        result = scaled.grant(mock_context)

        assert result is True
        # Level 10: 1.0 + (10-1) * 0.1 = 1.9
        expected = int(100 * 1.9)
        mock_context.add_experience.assert_called_with(expected, "")

    def test_scaled_reward_by_difficulty(self, mock_context):
        """Test reward scaling by difficulty."""
        base = CurrencyReward(id="base_gold", amount=1000, currency_type="gold")
        scaled = ScaledReward(
            id="scaled_gold",
            base_reward=base,
            scale_factor="difficulty",
        )

        mock_context.difficulty = 3
        result = scaled.grant(mock_context)

        assert result is True
        # Difficulty 3: 1.0 + (3-1) * 0.25 = 1.5
        expected = int(1000 * 1.5)
        mock_context.add_currency.assert_called_with(expected, "gold")

    def test_scaled_reward_min_cap(self, mock_context):
        """Test scaled reward respects minimum cap."""
        base = ExperienceReward(id="base", amount=100)
        scaled = ScaledReward(
            id="scaled",
            base_reward=base,
            scale_factor="level",
            min_scale=0.5,
            max_scale=5.0,
        )

        mock_context.player_level = 1
        result = scaled.grant(mock_context)

        assert result is True
        # Level 1 scale = 1.0, but min is 0.5, so 1.0 is used
        # Actually level 1: 1.0 + (1-1)*0.1 = 1.0

    def test_scaled_reward_max_cap(self, mock_context):
        """Test scaled reward respects maximum cap."""
        base = ExperienceReward(id="base", amount=100)
        scaled = ScaledReward(
            id="scaled",
            base_reward=base,
            scale_factor="level",
            min_scale=1.0,
            max_scale=2.0,
        )

        mock_context.player_level = 100
        result = scaled.grant(mock_context)

        assert result is True
        # Level 100 would give huge scale, but capped at 2.0
        mock_context.add_experience.assert_called_with(200, "")  # 100 * 2.0

    def test_scaled_reward_exponential(self, mock_context):
        """Test exponential scaling formula."""
        base = ExperienceReward(id="base", amount=100)
        scaled = ScaledReward(
            id="scaled",
            base_reward=base,
            scale_factor="level",
            scale_formula="exponential",
            max_scale=10.0,
        )

        mock_context.player_level = 5
        result = scaled.grant(mock_context)

        assert result is True
        # Level 5: 1.0 * (1.1 ** 4) = 1.4641

    def test_scaled_reward_serialization(self):
        """Test scaled reward serialization."""
        base = ExperienceReward(id="base", amount=100)
        scaled = ScaledReward(
            id="scaled",
            base_reward=base,
            scale_factor="level",
            min_scale=0.5,
            max_scale=3.0,
            scale_formula="linear",
        )

        data = scaled.to_dict()

        assert data["type"] == "scaled"
        assert data["scale_factor"] == "level"
        assert data["min_scale"] == 0.5
        assert data["max_scale"] == 3.0


# =============================================================================
# Reward Factory Tests
# =============================================================================

class TestRewardFactory:
    """Tests for reward factory functionality."""

    def test_factory_create_experience(self):
        """Test factory creates experience reward."""
        reward = RewardFactory.create("experience", id="xp", amount=100)
        assert isinstance(reward, ExperienceReward)
        assert reward.amount == 100

    def test_factory_create_currency(self):
        """Test factory creates currency reward."""
        reward = RewardFactory.create("currency", id="gold", amount=500, currency_type="gold")
        assert isinstance(reward, CurrencyReward)

    def test_factory_create_item(self):
        """Test factory creates item reward."""
        reward = RewardFactory.create("item", id="sword", item_id="iron_sword", quantity=1)
        assert isinstance(reward, ItemReward)

    def test_factory_create_reputation(self):
        """Test factory creates reputation reward."""
        reward = RewardFactory.create("reputation", id="rep", faction_id="guild", amount=50)
        assert isinstance(reward, ReputationReward)

    def test_factory_create_ability(self):
        """Test factory creates ability reward."""
        reward = RewardFactory.create("ability", id="skill", ability_id="fireball")
        assert isinstance(reward, AbilityReward)

    def test_factory_unknown_type_raises(self):
        """Test factory raises error for unknown type."""
        with pytest.raises(ValueError, match="Unknown reward type"):
            RewardFactory.create("unknown", id="test")

    def test_factory_from_dict(self):
        """Test factory creates from dictionary."""
        data = {
            "type": "experience",
            "id": "xp",
            "amount": 100,
            "skill_type": "combat",
        }
        reward = RewardFactory.from_dict(data.copy())

        assert isinstance(reward, ExperienceReward)
        assert reward.amount == 100


# =============================================================================
# Quest Definition Rewards Tests
# =============================================================================

class TestQuestDefinitionRewards:
    """Tests for rewards in quest definitions."""

    def test_quest_with_rewards(self):
        """Test quest definition with rewards."""
        rewards = [
            ExperienceReward(id="xp", amount=1000),
            CurrencyReward(id="gold", amount=500, currency_type="gold"),
        ]
        quest_def = QuestDefinition(
            id="quest_with_rewards",
            name="Rewarding Quest",
            description="A quest with rewards",
            rewards=rewards,
        )

        assert len(quest_def.rewards) == 2

    def test_quest_with_multiple_reward_types(self):
        """Test quest with multiple reward types."""
        rewards = [
            ExperienceReward(id="xp", amount=500),
            CurrencyReward(id="gold", amount=1000, currency_type="gold"),
            ItemReward(id="item", item_id="rare_weapon", quantity=1, quality="rare"),
            ReputationReward(id="rep", faction_id="adventurers", amount=100),
        ]
        quest_def = QuestDefinition(
            id="multi_reward_quest",
            name="Multi-Reward Quest",
            description="",
            rewards=rewards,
        )

        assert len(quest_def.rewards) == 4

    def test_quest_rewards_empty_default(self):
        """Test that quest rewards default to empty list."""
        quest_def = QuestDefinition(
            id="no_rewards",
            name="No Rewards Quest",
            description="",
        )

        assert quest_def.rewards == []


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestRewardEdgeCases:
    """Tests for reward edge cases and boundary conditions."""

    def test_zero_amount_rewards(self, mock_context):
        """Test rewards with zero amounts."""
        xp = ExperienceReward(id="zero_xp", amount=0)
        gold = CurrencyReward(id="zero_gold", amount=0, currency_type="gold")

        xp.grant(mock_context)
        gold.grant(mock_context)

        mock_context.add_experience.assert_called_with(0, "")
        mock_context.add_currency.assert_called_with(0, "gold")

    def test_very_large_amounts(self, mock_context):
        """Test rewards with very large amounts."""
        xp = ExperienceReward(id="big_xp", amount=1000000000)
        result = xp.grant(mock_context)

        assert result is True
        mock_context.add_experience.assert_called_with(1000000000, "")

    def test_nested_choice_rewards(self, mock_context):
        """Test nested choice rewards."""
        inner_options = [
            ItemReward(id="inner1", item_id="item1", quantity=1),
            ItemReward(id="inner2", item_id="item2", quantity=1),
        ]
        inner_choice = ChoiceReward(id="inner", options=inner_options, max_choices=1)

        outer_options = [
            ExperienceReward(id="xp", amount=100),
            inner_choice,
        ]
        outer_choice = ChoiceReward(id="outer", options=outer_options, max_choices=1)

        # Select inner choice
        outer_choice.grant(mock_context, chosen_indices=[1])

        # Inner choice needs its own selection
        inner_choice.grant(mock_context, chosen_indices=[0])
        mock_context.add_item.assert_called()

    def test_bonus_with_no_base_reward(self, mock_context):
        """Test bonus reward with no base reward."""
        bonus = BonusReward(
            id="empty_bonus",
            base_reward=None,
            condition_type="time",
            threshold=60.0,
        )

        result = bonus.grant(mock_context)
        assert result is False

    def test_scaled_with_no_base_reward(self, mock_context):
        """Test scaled reward with no base reward."""
        scaled = ScaledReward(
            id="empty_scaled",
            base_reward=None,
            scale_factor="level",
        )

        result = scaled.grant(mock_context)
        assert result is False

    def test_item_reward_special_quality(self):
        """Test item reward with special quality."""
        reward = ItemReward(
            id="legendary",
            item_id="excalibur",
            quantity=1,
            quality="legendary",
        )
        assert reward.quality == "legendary"

    def test_reputation_reward_large_negative(self, mock_context):
        """Test large negative reputation reward."""
        reward = ReputationReward(id="enemy", faction_id="bandits", amount=-1000)
        result = reward.grant(mock_context)

        assert result is True
        mock_context.change_reputation.assert_called_with("bandits", -1000)
