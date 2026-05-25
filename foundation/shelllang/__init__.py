"""
ShellLang - A minimal semantic core with ergonomic sugar.

ShellLang provides a dual-interface shell for both humans and AI:
- Humans get fluent, chainable syntax with immediate feedback
- AI gets structured JSON commands with validation and dry-run

The 5 Semantic Primitives:
    ENTITY      uint64 identifier
    COMPONENT   typed data attached to entity
    QUERY       entity predicate -> [entity]
    MUTATE      (entity, field, value) -> tracked change
    SNAPSHOT    frozen world state

Architecture:
    core.py     World, Entity, Snapshot, Change (~100 lines)
    sugar.py    EntityProxy, QueryResult, TypeQuery, TimeManager (~300 lines)
    ai.py       AIInterface with execute/validate/dry_run (~200 lines)
    repl.py     Shell, Feedback (~100 lines)
"""

from foundation.shelllang.core import (
    Change,
    Component,
    Entity,
    Snapshot,
    World,
)
from foundation.shelllang.sugar import (
    ComponentProxy,
    EntityProxy,
    QueryResult,
    TimeManager,
    TypeQuery,
)
from foundation.shelllang.ai import AIInterface
from foundation.shelllang.repl import (
    Feedback,
    Shell,
    echo,
)

__all__ = [
    # Core primitives
    "World",
    "Entity",
    "Component",
    "Snapshot",
    "Change",
    # Sugar
    "EntityProxy",
    "ComponentProxy",
    "QueryResult",
    "TypeQuery",
    "TimeManager",
    # AI
    "AIInterface",
    # REPL
    "Shell",
    "Feedback",
    "echo",
]
