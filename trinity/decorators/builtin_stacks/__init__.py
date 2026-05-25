"""Built-in decorator stacks for common game development patterns."""

from trinity.decorators.builtin_stacks.core import (
    production_component,
    safe_system,
    saveable_data,
)
from trinity.decorators.builtin_stacks.network import (
    bandwidth_efficient,
    networked_entity,
    predicted_entity,
    secure_multiplayer,
)
from trinity.decorators.builtin_stacks.persistence import (
    deterministic_data,
    replay_ready,
    versioned_saveable,
)
from trinity.decorators.builtin_stacks.streaming import (
    lod_scalable,
    streaming_chunk,
)
from trinity.decorators.builtin_stacks.ai import (
    complete_ai,
)
from trinity.decorators.builtin_stacks.development import (
    profiled_dev,
)
from trinity.decorators.builtin_stacks.composite import (
    competitive_entity,
    mmo_entity,
    moddable_content,
    multiplayer_character,
    open_world_entity,
)

__all__ = [
    # Core
    "production_component",
    "safe_system",
    "saveable_data",
    # Network
    "networked_entity",
    "bandwidth_efficient",
    "predicted_entity",
    "secure_multiplayer",
    # Persistence
    "versioned_saveable",
    "replay_ready",
    "deterministic_data",
    # Streaming
    "streaming_chunk",
    "lod_scalable",
    # AI
    "complete_ai",
    # Development
    "profiled_dev",
    # Composite
    "multiplayer_character",
    "competitive_entity",
    "open_world_entity",
    "mmo_entity",
    "moddable_content",
]
