"""
Trinity Pattern - Tier 47: GAMEPLAY Decorators

Gameplay mechanics decorators for abilities, buffs/debuffs, gameplay tags,
entity spawners, player interactions, and quest systems.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar

from trinity.decorators.ops import Op, Step, make_decorator
from trinity.decorators.registry import DecoratorSpec, Tier, registry

T = TypeVar("T")

# =============================================================================
# VALID VALUES
# =============================================================================

VALID_STACKING = frozenset({"none", "duration", "intensity", "independent"})

# =============================================================================
# VALIDATORS
# =============================================================================


def _validate_ability(
    cost: dict[str, float] = {},
    cooldown: float = 0.0,
    tags: set[str] = set(),
    blocked_by: set[str] = set(),
    **_: Any
) -> None:
    """Validate @ability parameters."""
    if cooldown < 0:
        raise ValueError(f"@ability: cooldown must be >= 0, got {cooldown}")


def _validate_buff(
    duration: float | None = None,
    stacking: str = "none",
    max_stacks: int = 1,
    tick_rate: float = 0.0,
    **_: Any
) -> None:
    """Validate @buff parameters."""
    if stacking not in VALID_STACKING:
        raise ValueError(
            f"@buff: invalid stacking '{stacking}'. "
            f"Valid stacking modes: {sorted(VALID_STACKING)}"
        )
    if max_stacks <= 0:
        raise ValueError(f"@buff: max_stacks must be > 0, got {max_stacks}")
    if tick_rate < 0:
        raise ValueError(f"@buff: tick_rate must be >= 0, got {tick_rate}")


def _validate_gameplay_tag(hierarchy: str = "", **_: Any) -> None:
    """Validate @gameplay_tag parameters."""
    if not hierarchy:
        raise ValueError("@gameplay_tag: hierarchy must be non-empty")


def _validate_spawner(
    prefab: str = "",
    pool_size: int = 10,
    spawn_rate: float = 1.0,
    max_alive: int | None = None,
    **_: Any
) -> None:
    """Validate @spawner parameters."""
    if not prefab:
        raise ValueError("@spawner: prefab must be non-empty")
    if pool_size <= 0:
        raise ValueError(f"@spawner: pool_size must be > 0, got {pool_size}")
    if spawn_rate <= 0:
        raise ValueError(f"@spawner: spawn_rate must be > 0, got {spawn_rate}")


def _validate_interactable(
    prompt: str = "", range: float = 2.0, hold_time: float = 0.0, **_: Any
) -> None:
    """Validate @interactable parameters."""
    if not prompt:
        raise ValueError("@interactable: prompt must be non-empty")
    if range <= 0:
        raise ValueError(f"@interactable: range must be > 0, got {range}")
    if hold_time < 0:
        raise ValueError(f"@interactable: hold_time must be >= 0, got {hold_time}")


def _validate_quest(
    id: str = "", prerequisites: list[str] = [], rewards: list[tuple[str, int]] = [], **_: Any
) -> None:
    """Validate @quest parameters."""
    if not id:
        raise ValueError("@quest: id must be non-empty")


# =============================================================================
# STEP BUILDERS
# =============================================================================


def _ability_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @ability decorator."""
    cost = params.get("cost", {})
    cooldown = params.get("cooldown", 0.0)
    tags = params.get("tags", set())
    blocked_by = params.get("blocked_by", set())

    return [
        Step(Op.TAG, {"key": "ability", "value": True}),
        Step(Op.TAG, {"key": "ability_cost", "value": dict(cost)}),
        Step(Op.TAG, {"key": "ability_cooldown", "value": cooldown}),
        Step(Op.TAG, {"key": "ability_tags", "value": set(tags)}),
        Step(Op.TAG, {"key": "ability_blocked_by", "value": set(blocked_by)}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


def _buff_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @buff decorator."""
    duration = params.get("duration")
    stacking = params.get("stacking", "none")
    max_stacks = params.get("max_stacks", 1)
    tick_rate = params.get("tick_rate", 0.0)

    return [
        Step(Op.TAG, {"key": "buff", "value": True}),
        Step(Op.TAG, {"key": "buff_duration", "value": duration}),
        Step(Op.TAG, {"key": "buff_stacking", "value": stacking}),
        Step(Op.TAG, {"key": "buff_max_stacks", "value": max_stacks}),
        Step(Op.TAG, {"key": "buff_tick_rate", "value": tick_rate}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


def _gameplay_tag_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @gameplay_tag decorator."""
    hierarchy = params.get("hierarchy", "")

    return [
        Step(Op.TAG, {"key": "gameplay_tag", "value": True}),
        Step(Op.TAG, {"key": "tag_hierarchy", "value": hierarchy}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


def _spawner_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @spawner decorator."""
    prefab = params.get("prefab", "")
    pool_size = params.get("pool_size", 10)
    spawn_rate = params.get("spawn_rate", 1.0)
    max_alive = params.get("max_alive")

    return [
        Step(Op.TAG, {"key": "spawner", "value": True}),
        Step(Op.TAG, {"key": "spawner_prefab", "value": prefab}),
        Step(Op.TAG, {"key": "spawner_pool_size", "value": pool_size}),
        Step(Op.TAG, {"key": "spawner_spawn_rate", "value": spawn_rate}),
        Step(Op.TAG, {"key": "spawner_max_alive", "value": max_alive}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


def _interactable_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @interactable decorator."""
    prompt = params.get("prompt", "")
    range_val = params.get("range", 2.0)
    hold_time = params.get("hold_time", 0.0)

    return [
        Step(Op.TAG, {"key": "interactable", "value": True}),
        Step(Op.TAG, {"key": "interactable_prompt", "value": prompt}),
        Step(Op.TAG, {"key": "interactable_range", "value": range_val}),
        Step(Op.TAG, {"key": "interactable_hold_time", "value": hold_time}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


def _quest_steps(params: dict[str, Any]) -> list[Step]:
    """Build steps for @quest decorator."""
    quest_id = params.get("id", "")
    prerequisites = params.get("prerequisites", [])
    rewards = params.get("rewards", [])

    return [
        Step(Op.TAG, {"key": "quest", "value": True}),
        Step(Op.TAG, {"key": "quest_id", "value": quest_id}),
        Step(Op.TAG, {"key": "quest_prerequisites", "value": list(prerequisites)}),
        Step(Op.TAG, {"key": "quest_rewards", "value": list(rewards)}),
        Step(Op.REGISTER, {"registry": "gameplay"}),
    ]


# =============================================================================
# AFTER-STEPS
# =============================================================================


def _after_ability(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @ability is applied."""
    target._ability = True
    target._ability_cost = dict(params.get("cost", {}))
    target._ability_cooldown = params.get("cooldown", 0.0)
    target._ability_tags = set(params.get("tags", set()))
    target._ability_blocked_by = set(params.get("blocked_by", set()))
    return None


def _after_buff(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @buff is applied."""
    target._buff = True
    target._buff_duration = params.get("duration")
    target._buff_stacking = params.get("stacking", "none")
    target._buff_max_stacks = params.get("max_stacks", 1)
    target._buff_tick_rate = params.get("tick_rate", 0.0)
    return None


def _after_gameplay_tag(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @gameplay_tag is applied."""
    target._gameplay_tag = True
    target._tag_hierarchy = params.get("hierarchy", "")
    return None


def _after_spawner(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @spawner is applied."""
    target._spawner = True
    target._spawner_prefab = params.get("prefab", "")
    target._spawner_pool_size = params.get("pool_size", 10)
    target._spawner_spawn_rate = params.get("spawn_rate", 1.0)
    target._spawner_max_alive = params.get("max_alive")
    return None


def _after_interactable(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @interactable is applied."""
    target._interactable = True
    target._interactable_prompt = params.get("prompt", "")
    target._interactable_range = params.get("range", 2.0)
    target._interactable_hold_time = params.get("hold_time", 0.0)
    return None


def _after_quest(target: Any, params: dict[str, Any]) -> Any:
    """Set attributes after @quest is applied."""
    target._quest = True
    target._quest_id = params.get("id", "")
    target._quest_prerequisites = list(params.get("prerequisites", []))
    target._quest_rewards = list(params.get("rewards", []))
    return None


# =============================================================================
# DECORATOR DEFINITIONS
# =============================================================================

ability = make_decorator(
    name="ability",
    steps=_ability_steps,
    doc="Declare gameplay ability with cost, cooldown, and tags.",
    validate=_validate_ability,
    after_steps=_after_ability,
)

buff = make_decorator(
    name="buff",
    steps=_buff_steps,
    doc="Declare buff/debuff with duration, stacking mode, and tick rate.",
    validate=_validate_buff,
    after_steps=_after_buff,
)

gameplay_tag = make_decorator(
    name="gameplay_tag",
    steps=_gameplay_tag_steps,
    doc="Declare hierarchical gameplay tag for gameplay systems.",
    validate=_validate_gameplay_tag,
    after_steps=_after_gameplay_tag,
)

spawner = make_decorator(
    name="spawner",
    steps=_spawner_steps,
    doc="Declare entity spawner with pooling and spawn rate.",
    validate=_validate_spawner,
    after_steps=_after_spawner,
)

interactable = make_decorator(
    name="interactable",
    steps=_interactable_steps,
    doc="Declare player-interactable object with prompt and range.",
    validate=_validate_interactable,
    after_steps=_after_interactable,
)

quest = make_decorator(
    name="quest",
    steps=_quest_steps,
    doc="Declare quest with prerequisites and rewards.",
    validate=_validate_quest,
    after_steps=_after_quest,
)

# =============================================================================
# REGISTRY REGISTRATION
# =============================================================================

_REGISTRY_ENTRIES: list[tuple[str, Any, tuple[str, ...]]] = [
    ("ability", ability, ("class",)),
    ("buff", buff, ("class",)),
    ("gameplay_tag", gameplay_tag, ("class",)),
    ("spawner", spawner, ("class",)),
    ("interactable", interactable, ("class",)),
    ("quest", quest, ("class",)),
]

for _name, _func, _targets in _REGISTRY_ENTRIES:
    if _name not in registry._decorators:
        _spec = DecoratorSpec(
            name=_name,
            tier=Tier.GAMEPLAY,
            func=_func,
            unique=True,
            foundation=False,
            doc=getattr(_func, "__doc__", ""),
            target_types=_targets,
        )
        registry._decorators[_name] = _spec
        registry._by_tier[Tier.GAMEPLAY].append(_spec)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ability",
    "buff",
    "gameplay_tag",
    "spawner",
    "interactable",
    "quest",
    "VALID_STACKING",
]
