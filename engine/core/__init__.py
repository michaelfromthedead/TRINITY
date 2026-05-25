"""engine.core -- Runtime systems: Engine loop, frame timing, constants."""

from engine.core.constants import (
    DEFAULT_TARGET_FPS,
    DEFAULT_FIXED_TIMESTEP,
    MAX_DELTA_TIME,
    DEFAULT_TIME_SCALE,
    DEFAULT_FRAME_ALLOCATOR_SIZE,
    ENTITY_INDEX_BITS,
    ENTITY_GENERATION_BITS,
    MAX_ENTITIES,
    DEFAULT_ARCHETYPE_CHUNK_SIZE,
    MAX_COMPONENTS,
    DEFAULT_WORKER_COUNT,
    TASK_PRIORITY_CRITICAL,
    TASK_PRIORITY_HIGH,
    TASK_PRIORITY_NORMAL,
    TASK_PRIORITY_LOW,
    TASK_PRIORITY_IDLE,
    DEFAULT_TASK_STACK_SIZE,
    SESSION_VERSION,
    MAX_CHECKPOINTS,
)
from engine.core.frame import (
    FramePhase,
    FrameTimer,
    FrameAllocator,
    FrameContext,
    FixedTimestepAccumulator,
)
from engine.core.engine import Engine

__all__ = [
    # Engine
    "Engine",
    # Frame
    "FramePhase",
    "FrameTimer",
    "FrameAllocator",
    "FrameContext",
    "FixedTimestepAccumulator",
    # Constants
    "DEFAULT_TARGET_FPS",
    "DEFAULT_FIXED_TIMESTEP",
    "MAX_DELTA_TIME",
    "DEFAULT_TIME_SCALE",
    "DEFAULT_FRAME_ALLOCATOR_SIZE",
    "ENTITY_INDEX_BITS",
    "ENTITY_GENERATION_BITS",
    "MAX_ENTITIES",
    "DEFAULT_ARCHETYPE_CHUNK_SIZE",
    "MAX_COMPONENTS",
    "DEFAULT_WORKER_COUNT",
    "TASK_PRIORITY_CRITICAL",
    "TASK_PRIORITY_HIGH",
    "TASK_PRIORITY_NORMAL",
    "TASK_PRIORITY_LOW",
    "TASK_PRIORITY_IDLE",
    "DEFAULT_TASK_STACK_SIZE",
    "SESSION_VERSION",
    "MAX_CHECKPOINTS",
]
