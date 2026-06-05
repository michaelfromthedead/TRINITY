"""
World partition streaming system for TRINITY.

Provides async cell loading, priority-based streaming, and cell state management
for open-world streaming.

Modules:
    world_partition: Core decorators and chunk types
    cell_state_machine: Cell lifecycle state management
    async_loader: Async loading pipeline
    priority_system: Streaming priority computation
"""

from engine.streaming.world_partition import (
    # Decorators
    chunk,
    streamable,
    loading_priority,
    unloadable,
    # Core types
    WorldChunk,
    ChunkConfig,
    StreamableConfig,
    LoadingPriorityConfig,
    UnloadableConfig,
    # Enums
    StreamPriority,
)
from engine.streaming.cell_state_machine import (
    CellState,
    CellStateMachine,
    CellStateError,
    StateTransitionCallback,
)
from engine.streaming.async_loader import (
    # Pipeline stages
    LoadStage,
    LoadRequest,
    LoadResult,
    LoadError,
    # Loaders
    TerrainLoader,
    HeightDataLoader,
    GPUUploader,
    # Pipeline
    AsyncLoadPipeline,
    LoadPipelineConfig,
)
from engine.streaming.priority_system import (
    # Priority computation
    PriorityComputer,
    PriorityConfig,
    PriorityFactors,
    CellPriority,
    # Activation
    CellActivationTrigger,
    ActivationEvent,
    ActivationType,
)

__all__ = [
    # Decorators
    "chunk",
    "streamable",
    "loading_priority",
    "unloadable",
    # Core types
    "WorldChunk",
    "ChunkConfig",
    "StreamableConfig",
    "LoadingPriorityConfig",
    "UnloadableConfig",
    # Enums
    "StreamPriority",
    # State machine
    "CellState",
    "CellStateMachine",
    "CellStateError",
    "StateTransitionCallback",
    # Async loader
    "LoadStage",
    "LoadRequest",
    "LoadResult",
    "LoadError",
    "TerrainLoader",
    "HeightDataLoader",
    "GPUUploader",
    "AsyncLoadPipeline",
    "LoadPipelineConfig",
    # Priority system
    "PriorityComputer",
    "PriorityConfig",
    "PriorityFactors",
    "CellPriority",
    "CellActivationTrigger",
    "ActivationEvent",
    "ActivationType",
]
