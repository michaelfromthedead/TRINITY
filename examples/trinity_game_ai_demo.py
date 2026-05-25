#!/usr/bin/env python3
"""
Demo of Trinity Pattern Tier 36: GAME_AI decorators.

Demonstrates behavior trees, utility AI, blackboards, perception, and debugging.
"""

from trinity.decorators.game_ai import (
    ai_debug,
    behavior_tree,
    blackboard,
    perception,
    utility_ai,
)
from trinity.decorators.registry import inspect_decorated, registry


# Example 1: Behavior Tree for enemy AI
@ai_debug
@behavior_tree(id="enemy_patrol", debug_name="Enemy Patrol Behavior")
class EnemyPatrolAI:
    """AI that patrols an area and attacks on sight."""

    state: str
    patrol_points: list


# Example 2: Utility AI for NPC decision making
@utility_ai(id="npc_decision", update_rate=0.25)
class NPCDecisionAI:
    """NPC that makes decisions based on utility scores."""

    hunger: float
    fatigue: float
    social_need: float


# Example 3: Shared memory blackboard
@blackboard
class AIBlackboard:
    """Shared memory for AI agents to communicate."""

    enemy_positions: dict
    threats: list
    objectives: list


# Example 4: Multi-sense perception system
@perception(sense="sight", range=100.0, fov=90.0)
class VisionSensor:
    """Visual perception component."""

    visible_entities: list


@perception(sense="hearing", range=50.0)
class HearingSensor:
    """Audio perception component."""

    audible_sounds: list


@perception(sense="damage", range=5.0)
class DamageSensor:
    """Damage detection sensor."""

    damage_sources: list


# Example 5: Complex AI combining multiple decorators
@ai_debug
@perception(sense="squad", range=200.0)
@utility_ai(id="squad_leader", update_rate=0.5)
class SquadLeaderAI:
    """Squad leader AI with perception and utility-based decision making."""

    squad_members: list
    tactical_state: str


def main():
    """Demonstrate the GAME_AI decorators."""
    print("=" * 70)
    print("Trinity Pattern - Tier 36: GAME_AI Decorators Demo")
    print("=" * 70)

    # Show behavior tree
    print("\n1. Behavior Tree AI:")
    print(f"   ID: {EnemyPatrolAI._bt_id}")
    print(f"   Debug Name: {EnemyPatrolAI._bt_debug_name}")
    print(f"   AI Debug Enabled: {EnemyPatrolAI._ai_debug}")
    info = inspect_decorated(EnemyPatrolAI)
    print(f"   Applied Decorators: {', '.join(info.decorators)}")

    # Show utility AI
    print("\n2. Utility AI:")
    print(f"   ID: {NPCDecisionAI._utility_id}")
    print(f"   Update Rate: {NPCDecisionAI._utility_update_rate}s")
    print(f"   Tier: {info.tier}")

    # Show blackboard
    print("\n3. AI Blackboard:")
    print(f"   Enabled: {AIBlackboard._blackboard}")
    print(f"   Registry: {AIBlackboard._registries}")

    # Show perception sensors
    print("\n4. Perception Sensors:")
    sensors = [
        ("Vision", VisionSensor),
        ("Hearing", HearingSensor),
        ("Damage", DamageSensor),
    ]
    for name, sensor in sensors:
        print(f"   {name}:")
        print(f"     Sense: {sensor._perception_sense}")
        print(f"     Range: {sensor._perception_range}")
        if hasattr(sensor, "_perception_fov") and sensor._perception_fov:
            print(f"     FOV: {sensor._perception_fov}°")

    # Show complex AI
    print("\n5. Squad Leader AI (Complex):")
    print(f"   Utility ID: {SquadLeaderAI._utility_id}")
    print(f"   Update Rate: {SquadLeaderAI._utility_update_rate}s")
    print(f"   Perception: {SquadLeaderAI._perception_sense}")
    print(f"   Perception Range: {SquadLeaderAI._perception_range}")
    print(f"   Debug Mode: {SquadLeaderAI._ai_debug}")

    # Show registry stats
    print("\n6. Registry Statistics:")
    game_ai_specs = registry.by_tier(36)  # Tier.GAME_AI = 36
    print(f"   GAME_AI decorators registered: {len(game_ai_specs)}")
    print(f"   Decorator names: {', '.join(s.name for s in game_ai_specs)}")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
