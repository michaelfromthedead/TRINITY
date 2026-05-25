"""
Post-Processing Stack and Effect System

Provides the core infrastructure for post-processing effects including:
- PostProcessEffect base class for all effects
- PostProcessStack for ordered effect chain management
- PostProcessStackExecutor for frame graph integration
- PostProcessVolume for spatial blending of post-process settings
- Quality presets (Low/Medium/High/Ultra) for conditional execution
- Intermediate target management for effect chaining
- Execution flags for per-effect conditional execution
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

from engine.rendering.framegraph.pass_node import PassFlags

if TYPE_CHECKING:
    from engine.rendering.framegraph.frame_graph import FrameGraph
    from engine.rendering.framegraph.pass_node import PassNode


class BlendMode(Enum):
    """Blend modes for effect parameter interpolation."""

    LERP = auto()  # Linear interpolation
    OVERRIDE = auto()  # Complete override
    ADDITIVE = auto()  # Additive blend
    MULTIPLY = auto()  # Multiplicative blend


class EffectPriority(Enum):
    """Priority levels for effect execution order."""

    EXPOSURE = 0
    BLOOM = 100
    DEPTH_OF_FIELD = 200
    MOTION_BLUR = 300
    AMBIENT_OCCLUSION = 400
    TONEMAPPING = 500
    COLOR_GRADING = 600
    ANTIALIASING = 700
    UPSCALING = 800
    CUSTOM = 1000


class EffectQuality(Enum):
    """Quality preset levels for the post-process stack."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    ULTRA = 3


class ExecutionFlags(IntEnum):
    """Bitmask flags controlling per-effect conditional execution."""

    NONE = 0
    SKIP_IF_DISABLED = 1 << 0
    """Skip execution when the effect is disabled."""
    SKIP_ON_FIRST_FRAME = 1 << 1
    """Skip execution on the first frame (e.g., temporal effects needing history)."""
    SKIP_IF_NO_INPUT = 1 << 2
    """Skip execution if required input resources are missing."""
    FORCE_ASYNC = 1 << 3
    """Force this effect onto the async compute queue."""
    ALWAYS = 1 << 4
    """Always execute even if quality preset would filter out (e.g., final output)."""


class EffectExecutionPath(Enum):
    """Execution mode for post-process effects."""

    FRAME_GRAPH_PASS = auto()
    """Each effect becomes a separate pass in the S1 Frame Graph (production)."""
    DIRECT_CALL = auto()
    """Execute effects directly without frame graph (testing/debugging)."""
    MERGED_COMPUTE = auto()
    """Future: merge compatible effects into a single compute dispatch."""


@dataclass
class EffectSettings:
    """Base class for effect-specific settings."""

    enabled: bool = True
    weight: float = 1.0
    priority: int = EffectPriority.CUSTOM.value

    def lerp(self, other: "EffectSettings", t: float) -> "EffectSettings":
        """Interpolate between two settings instances.

        Args:
            other: Target settings to interpolate towards.
            t: Interpolation factor [0, 1].

        Returns:
            Interpolated settings.
        """
        raise NotImplementedError("Subclasses must implement lerp")


T = TypeVar("T", bound=EffectSettings)


# ==============================================================================
# Quality Preset System
# ==============================================================================


@dataclass
class QualityPreset:
    """Defines which effects are active at a given quality level.

    Quality presets control the active effect set and provide per-effect
    configuration hints (e.g., sample counts, buffer resolutions).
    """

    name: str
    """Human-readable preset name (e.g., 'Low', 'Medium', 'High', 'Ultra')."""

    quality: EffectQuality
    """The quality level enum value."""

    active_effects: set[str] = field(default_factory=set)
    """Set of effect names that are active at this quality level."""

    description: str = ""
    """Human-readable description of this quality level."""

    effect_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Per-effect configuration overrides (e.g., sample counts, resolutions)."""

    def is_effect_active(self, effect_name: str) -> bool:
        """Check if an effect is active at this quality level.

        Args:
            effect_name: Name of the effect to check.

        Returns:
            True if the effect should run at this quality level.
        """
        return effect_name in self.active_effects

    def get_effect_config(
        self,
        effect_name: str,
        default: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """Get per-effect configuration overrides.

        Args:
            effect_name: Name of the effect.
            default: Default value if no config exists.

        Returns:
            Effect configuration dict or default.
        """
        return self.effect_configs.get(effect_name, default)


QUALITY_PRESET_LOW = QualityPreset(
    name="Low",
    quality=EffectQuality.LOW,
    active_effects={
        "Exposure",
        "Tonemapping",
        "FXAA",
    },
    description="Minimum visual quality. Only exposure, tonemapping, and FXAA.",
    effect_configs={
        "Exposure": {"bin_count": 64, "cull_percentile": 5.0},
        "Tonemapping": {"operator": "Reinhard"},
        "FXAA": {"quality": "low"},
    },
)

QUALITY_PRESET_MEDIUM = QualityPreset(
    name="Medium",
    quality=EffectQuality.MEDIUM,
    active_effects={
        "Exposure",
        "Bloom",
        "Tonemapping",
        "ColorGrading",
        "SMAA",
    },
    description="Balanced quality. Adds bloom and color grading.",
    effect_configs={
        "Exposure": {"bin_count": 128, "cull_percentile": 2.0},
        "Bloom": {"quality": "medium", "mip_levels": 3},
        "Tonemapping": {"operator": "ACES"},
        "ColorGrading": {"lut_size": 16},
        "SMAA": {"quality": "medium"},
    },
)

QUALITY_PRESET_HIGH = QualityPreset(
    name="High",
    quality=EffectQuality.HIGH,
    active_effects={
        "Exposure",
        "Bloom",
        "DepthOfField",
        "MotionBlur",
        "AmbientOcclusion",
        "Tonemapping",
        "ColorGrading",
        "TAA",
    },
    description="High quality with cinematic effects. Adds DOF, motion blur, AO, TAA.",
    effect_configs={
        "Exposure": {"bin_count": 256, "cull_percentile": 1.0},
        "Bloom": {"quality": "high", "mip_levels": 5},
        "DepthOfField": {"quality": "high", "max_coc_radius": 16},
        "MotionBlur": {"quality": "high", "sample_count": 12},
        "AmbientOcclusion": {"method": "HBAO", "quality": "high"},
        "Tonemapping": {"operator": "ACES"},
        "ColorGrading": {"lut_size": 32},
        "TAA": {"quality": "high"},
    },
)

QUALITY_PRESET_ULTRA = QualityPreset(
    name="Ultra",
    quality=EffectQuality.ULTRA,
    active_effects={
        "Exposure",
        "Bloom",
        "DepthOfField",
        "MotionBlur",
        "AmbientOcclusion",
        "Tonemapping",
        "ColorGrading",
        "TAA",
        "Upscaling",
    },
    description="Maximum quality with upscaling. All effects enabled.",
    effect_configs={
        "Exposure": {"bin_count": 512, "cull_percentile": 0.5},
        "Bloom": {"quality": "ultra", "mip_levels": 6},
        "DepthOfField": {"quality": "ultra", "max_coc_radius": 32},
        "MotionBlur": {"quality": "ultra", "sample_count": 16},
        "AmbientOcclusion": {"method": "GTAO", "quality": "ultra"},
        "Tonemapping": {"operator": "AgX"},
        "ColorGrading": {"lut_size": 64},
        "TAA": {"quality": "ultra", "variance_clip": True},
        "Upscaling": {"quality": "ultra", "method": "auto"},
    },
)

QUALITY_PRESETS: dict[EffectQuality, QualityPreset] = {
    EffectQuality.LOW: QUALITY_PRESET_LOW,
    EffectQuality.MEDIUM: QUALITY_PRESET_MEDIUM,
    EffectQuality.HIGH: QUALITY_PRESET_HIGH,
    EffectQuality.ULTRA: QUALITY_PRESET_ULTRA,
}


def get_quality_preset(
    quality: Union[EffectQuality, str],
) -> QualityPreset:
    """Resolve a quality preset by enum value or name.

    Args:
        quality: An EffectQuality enum value or a string name
            (case-insensitive, e.g., 'high', 'High', 'HIGH').

    Returns:
        The matching QualityPreset.

    Raises:
        ValueError: If the quality level is unknown.
    """
    if isinstance(quality, EffectQuality):
        if quality in QUALITY_PRESETS:
            return QUALITY_PRESETS[quality]
        raise ValueError(f"Unknown quality level: {quality}")

    if isinstance(quality, str):
        name_lower = quality.lower()
        for preset in QUALITY_PRESETS.values():
            if preset.name.lower() == name_lower:
                return preset
        raise ValueError(f"Unknown quality preset name: '{quality}'")

    raise ValueError(f"Invalid quality type: {type(quality).__name__}")


# ==============================================================================
# PostProcessContext
# ==============================================================================


@dataclass
class PostProcessContext:
    """Execution context for post-processing effects.

    Carries per-frame state including timing, quality preset, RHI resources,
    camera position, and history buffers. Passed through the effect chain.
    """

    rhi_command_list: Optional[Any] = None
    """RHI command list for recording GPU commands (None = CPU-only/testing)."""

    rhi_device: Optional[Any] = None
    """RHI device reference for resource operations."""

    frame_index: int = 0
    """Current frame counter. Starts at 0 and increments per execute()."""

    quality: EffectQuality = EffectQuality.HIGH
    """Current quality preset level."""

    delta_time: float = 0.016
    """Time since the last frame in seconds (default ~60 FPS)."""

    camera_position: Optional[Tuple[float, float, float]] = None
    """World-space camera position for volume blending."""

    history_buffers: Dict[str, Any] = field(default_factory=dict)
    """Persistent history buffers for temporal effects (TAA, motion blur)."""

    @property
    def is_first_frame(self) -> bool:
        """Check if this is the first frame of rendering.

        Temporal effects use this to initialize history buffers.
        """
        return self.frame_index <= 1


# ==============================================================================
# Intermediate Target Management
# ==============================================================================


@dataclass
class IntermediateTarget:
    """Represents an intermediate render target for effect chaining."""

    handle: Any
    """Frame graph resource handle for this target."""

    name: str
    """Name of this intermediate target."""

    width: int
    """Width in pixels."""

    height: int
    """Height in pixels."""

    format: str
    """Texture format string."""


class IntermediateTargetManager:
    """Manages a pool of intermediate render targets for ping-pong rendering.

    Effects are chained by passing output from one effect as input to the next.
    The pool provides alternating read/write targets to avoid
    read-after-write hazards without allocating per-effect resources.
    """

    def __init__(self, pool_size: int = 2, format: str = "R11G11B10_FLOAT") -> None:
        """Initialize the intermediate target manager.

        Args:
            pool_size: Number of intermediate targets in the pool.
                Default 2 for ping-pong. Must be >= 1.
            format: Default format for intermediate targets.

        Raises:
            ValueError: If pool_size < 1.
        """
        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")
        self._pool_size: int = pool_size
        self._format: str = format
        self._targets: List[IntermediateTarget] = []
        self._ready: bool = False
        self._width: int = 0
        self._height: int = 0

    @property
    def pool_size(self) -> int:
        """Number of intermediate targets in the pool."""
        return self._pool_size

    @property
    def format(self) -> str:
        """Format of intermediate targets."""
        return self._format

    @format.setter
    def format(self, value: str) -> None:
        self._format = value

    def resize(self, width: int, height: int) -> None:
        """Mark targets for reallocation at new resolution.

        Args:
            width: New width.
            height: New height.
        """
        self._width = width
        self._height = height
        self._ready = False

    def allocate(
        self,
        frame_graph: "FrameGraph",
        width: int,
        height: int,
    ) -> List[Any]:
        """Allocate intermediate targets in the frame graph.

        Args:
            frame_graph: The frame graph to create resources in.
            width: Target width.
            height: Target height.

        Returns:
            List of resource handles for the allocated targets.
        """
        from engine.rendering.framegraph.frame_graph import FrameGraph
        from engine.rendering.framegraph.resource_manager import ResourceFormat

        self._width = width
        self._height = height
        self._targets.clear()

        format_map: dict[str, ResourceFormat] = {
            "R8G8B8A8_UNORM": ResourceFormat.R8G8B8A8_UNORM,
            "R8G8B8A8_SRGB": ResourceFormat.R8G8B8A8_SRGB,
            "R11G11B10_FLOAT": ResourceFormat.R16G16B16A16_FLOAT,
            "R16G16B16A16_FLOAT": ResourceFormat.R16G16B16A16_FLOAT,
            "R32G32B32A32_FLOAT": ResourceFormat.R32G32B32A32_FLOAT,
        }
        resolved_format = format_map.get(self._format, ResourceFormat.R16G16B16A16_FLOAT)

        handles: List[Any] = []
        for i in range(self._pool_size):
            name = f"PostProcess_Intermediate_{i}"
            handle = frame_graph.create_texture(
                name=name,
                format=resolved_format,
                width=width,
                height=height,
            )
            target = IntermediateTarget(
                handle=handle,
                name=name,
                width=width,
                height=height,
                format=self._format,
            )
            self._targets.append(target)
            handles.append(handle)

        self._ready = True
        return handles

    def get_target(self, index: int) -> Optional[Any]:
        """Get a specific intermediate target by index.

        Args:
            index: Target index (0-based).

        Returns:
            Resource handle or None if not allocated.
        """
        if not self._ready or index >= len(self._targets):
            return None
        return self._targets[index].handle

    def get_ping_pong(self, effect_index: int) -> Tuple[Optional[Any], Optional[Any]]:
        """Get read/write targets for ping-pong rendering.

        Alternates which target is read and which is written based on
        the effect index to avoid hazards.

        Args:
            effect_index: Index of the effect in the chain.

        Returns:
            Tuple of (read_handle, write_handle). Either may be None
            if not allocated.
        """
        if not self._ready or len(self._targets) < 2:
            return None, None

        read_idx = effect_index % self._pool_size
        write_idx = (effect_index + 1) % self._pool_size
        return self._targets[read_idx].handle, self._targets[write_idx].handle

    def get_read_target(self, effect_index: int) -> Optional[Any]:
        """Get the read target for a given effect index.

        Args:
            effect_index: Index of the effect.

        Returns:
            Resource handle for reading, or None.
        """
        if not self._ready:
            return None
        idx = effect_index % self._pool_size
        return self._targets[idx].handle if idx < len(self._targets) else None

    def get_write_target(self, effect_index: int) -> Optional[Any]:
        """Get the write target for a given effect index.

        For pool_size=1, returns None (no separate write target).

        Args:
            effect_index: Index of the effect.

        Returns:
            Resource handle for writing, or None.
        """
        if not self._ready or self._pool_size < 2:
            return None
        idx = (effect_index + 1) % self._pool_size
        return self._targets[idx].handle if idx < len(self._targets) else None

    def reset(self) -> None:
        """Clear all allocated targets."""
        self._targets.clear()
        self._ready = False


# ==============================================================================
# PostProcessEffect
# ==============================================================================


class PostProcessEffect(ABC, Generic[T]):
    """Abstract base class for all post-processing effects.

    Each effect defines how to render itself and manages its settings.
    Effects are composable and can be enabled/disabled independently.
    """

    def __init__(
        self,
        name: str,
        settings: Optional[T] = None,
        priority: int = EffectPriority.CUSTOM.value,
    ) -> None:
        """Initialize the post-process effect.

        Args:
            name: Unique identifier for the effect.
            settings: Effect-specific settings instance.
            priority: Execution order priority (lower = earlier).
        """
        self._id: str = str(uuid.uuid4())
        self._name: str = name
        self._settings: Optional[T] = settings
        self._priority: int = priority
        self._enabled: bool = True
        self._dirty: bool = True
        self._execution_flags: int = (
            ExecutionFlags.SKIP_IF_DISABLED.value
            | ExecutionFlags.SKIP_IF_NO_INPUT.value
        )

    @property
    def id(self) -> str:
        """Unique identifier for this effect instance."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable name of the effect."""
        return self._name

    @property
    def priority(self) -> int:
        """Execution order priority."""
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value
        self._dirty = True

    @property
    def enabled(self) -> bool:
        """Whether the effect is currently active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        self._dirty = True

    @property
    def settings(self) -> Optional[T]:
        """Effect-specific settings."""
        return self._settings

    @settings.setter
    def settings(self, value: T) -> None:
        self._settings = value
        self._dirty = True

    @property
    def dirty(self) -> bool:
        """Whether the effect needs to update GPU resources."""
        return self._dirty

    def mark_clean(self) -> None:
        """Mark the effect as up-to-date."""
        self._dirty = False

    def mark_dirty(self) -> None:
        """Mark the effect as needing update."""
        self._dirty = True

    @property
    def execution_flags(self) -> int:
        """Current execution flags bitmask."""
        return self._execution_flags

    def set_execution_flags(self, flags: int) -> None:
        """Set execution flags for this effect.

        Args:
            flags: Bitmask of ExecutionFlags values.
        """
        self._execution_flags = flags
        self._dirty = True

    def has_execution_flag(self, flag: ExecutionFlags) -> bool:
        """Check if a specific execution flag is set.

        Args:
            flag: The flag to check.

        Returns:
            True if the flag is set.
        """
        return bool(self._execution_flags & flag.value)

    def should_execute(
        self,
        context: PostProcessContext,
        quality_preset: Optional[QualityPreset] = None,
    ) -> bool:
        """Determine if this effect should execute in the current context.

        Checks:
        1. ALWAYS flag bypasses all checks.
        2. Effect must be enabled (SKIP_IF_DISABLED).
        3. Effect must be active in the current quality preset.
        4. Must not be first frame (SKIP_ON_FIRST_FRAME).

        Args:
            context: Current execution context.
            quality_preset: Quality preset to check against.

        Returns:
            True if the effect should execute.
        """
        if self.has_execution_flag(ExecutionFlags.ALWAYS):
            return True

        if self.has_execution_flag(ExecutionFlags.SKIP_IF_DISABLED) and not self._enabled:
            return False

        if quality_preset is not None:
            active = quality_preset.active_effects
            if len(active) > 0 and self._name not in active:
                has_preset_entry = any(
                    self._name in p.active_effects
                    for p in QUALITY_PRESETS.values()
                )
                if has_preset_entry:
                    return False

        if self.has_execution_flag(ExecutionFlags.SKIP_ON_FIRST_FRAME):
            if context.is_first_frame:
                return False

        return True

    def execute_with_context(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        context: PostProcessContext,
    ) -> None:
        """Execute the effect with a full PostProcessContext.

        Default implementation extracts delta_time and delegates to execute().
        Subclasses may override for context-aware execution.

        Args:
            inputs: Dictionary of input resources by name.
            outputs: Dictionary of output resources by name.
            context: PostProcessContext with per-frame state.
        """
        self.execute(inputs, outputs, context.delta_time)

    def execute_on_rhi(
        self,
        command_list: Any,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        context: PostProcessContext,
    ) -> None:
        """Execute the effect using RHI command list.

        Default implementation delegates to execute_with_context.
        Subclasses should override this to record actual GPU commands.

        Args:
            command_list: RHI command list for recording GPU commands.
            inputs: Dictionary of input resources by name.
            outputs: Dictionary of output resources by name.
            context: PostProcessContext with per-frame state.
        """
        self.execute_with_context(inputs, outputs, context)

    @abstractmethod
    def get_required_inputs(self) -> List[str]:
        """Get list of required input resources.

        Returns:
            List of resource names this effect reads from.
        """
        pass

    @abstractmethod
    def get_outputs(self) -> List[str]:
        """Get list of output resources.

        Returns:
            List of resource names this effect writes to.
        """
        pass

    @abstractmethod
    def setup(self, width: int, height: int) -> None:
        """Initialize or resize effect resources.

        Args:
            width: Target render width in pixels.
            height: Target render height in pixels.
        """
        pass

    @abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        delta_time: float,
    ) -> None:
        """Execute the effect.

        Args:
            inputs: Dictionary of input resources by name.
            outputs: Dictionary of output resources by name.
            delta_time: Time since last frame in seconds.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Release any GPU resources held by the effect."""
        pass

    def add_to_frame_graph(self, frame_graph: "FrameGraph") -> "PassNode":
        """Add this effect as a pass in the frame graph.

        Creates a pass node with appropriate type (compute or graphics)
        and declares read/write resource dependencies.

        Args:
            frame_graph: The frame graph to add to.

        Returns:
            The created render pass node.
        """
        pass_node = frame_graph.add_pass(
            name=f"PostProcess_{self._name}",
            pass_type="compute" if self.is_compute_effect() else "graphics",
        )

        for input_name in self.get_required_inputs():
            input_resource = frame_graph.get_resource(input_name)
            if input_resource:
                pass_node.read(input_resource)

        for output_name in self.get_outputs():
            output_resource = frame_graph.get_resource(output_name)
            if output_resource:
                pass_node.write(output_resource)

        return pass_node

    def is_compute_effect(self) -> bool:
        """Whether this effect uses compute shaders.

        Returns:
            True if compute-based, False for graphics pipeline.
        """
        return False


# ==============================================================================
# PostProcessStack
# ==============================================================================


@dataclass
class PostProcessStackConfig:
    """Configuration for the post-process stack."""

    hdr_enabled: bool = True
    hdr_format: str = "R16G16B16A16_FLOAT"
    intermediate_format: str = "R11G11B10_FLOAT"
    output_format: str = "R8G8B8A8_UNORM"
    auto_exposure_enabled: bool = True
    history_buffer_count: int = 2


class PostProcessStack:
    """Manages an ordered chain of post-processing effects.

    The stack maintains effects in priority order and handles:
    - Effect insertion and removal
    - Automatic resource management
    - Frame graph integration
    - Quality preset filtering
    - Effect blending from volumes
    - Frame index tracking for temporal effects
    """

    def __init__(
        self,
        config: Optional[PostProcessStackConfig] = None,
        quality: EffectQuality = EffectQuality.HIGH,
    ) -> None:
        """Initialize the post-process stack.

        Args:
            config: Stack configuration options.
            quality: Initial quality preset level.
        """
        self._config: PostProcessStackConfig = config or PostProcessStackConfig()
        self._effects: List[PostProcessEffect] = []
        self._effect_map: Dict[str, PostProcessEffect] = {}
        self._width: int = 0
        self._height: int = 0
        self._dirty: bool = True
        self._current_history_index: int = 0
        self._volumes: List["PostProcessVolume"] = []
        self._quality: EffectQuality = quality
        self._quality_preset: QualityPreset = QUALITY_PRESETS[quality]
        self._frame_index: int = 0

    @property
    def config(self) -> PostProcessStackConfig:
        """Stack configuration."""
        return self._config

    @property
    def effects(self) -> List[PostProcessEffect]:
        """Ordered list of effects."""
        return self._effects.copy()

    @property
    def width(self) -> int:
        """Current render width."""
        return self._width

    @property
    def height(self) -> int:
        """Current render height."""
        return self._height

    @property
    def quality(self) -> EffectQuality:
        """Current quality preset level."""
        return self._quality

    @property
    def quality_preset(self) -> QualityPreset:
        """Current quality preset."""
        return self._quality_preset

    @property
    def frame_index(self) -> int:
        """Current frame counter."""
        return self._frame_index

    def set_quality(self, quality: EffectQuality) -> None:
        """Switch to a different quality preset.

        Changes which effects are active based on the quality level.

        Args:
            quality: Target quality level.
        """
        if quality == self._quality:
            return
        self._quality = quality
        self._quality_preset = QUALITY_PRESETS[quality]
        self._dirty = True

    def advance_frame(self) -> None:
        """Advance the frame counter.

        Called after each stack execution to track frame progress
        for temporal effects.
        """
        self._frame_index += 1

    def get_active_effects(self) -> List[PostProcessEffect]:
        """Get effects that are active at the current quality preset.

        Filters effects by:
        1. Individual enabled/disabled state
        2. Quality preset active set (only for effects known to presets)

        Returns:
            List of active effects in priority order.
        """
        active: List[PostProcessEffect] = []
        known_effect_names: set[str] = set()
        for p in QUALITY_PRESETS.values():
            known_effect_names.update(p.active_effects)

        for effect in self._effects:
            if not effect.enabled:
                continue
            if effect.name in known_effect_names:
                if not self._quality_preset.is_effect_active(effect.name):
                    # ALWAYS flag bypasses quality preset filtering
                    if not effect.has_execution_flag(ExecutionFlags.ALWAYS):
                        continue
            active.append(effect)
        return active

    def add_effect(self, effect: PostProcessEffect) -> None:
        """Add an effect to the stack.

        The effect is inserted in priority order.

        Args:
            effect: The effect to add.

        Raises:
            ValueError: If an effect with the same name already exists.
        """
        if effect.name in self._effect_map:
            raise ValueError(f"Effect '{effect.name}' already exists in stack")

        self._effects.append(effect)
        self._effects.sort(key=lambda e: e.priority)
        self._effect_map[effect.name] = effect
        self._dirty = True

        if self._width > 0 and self._height > 0:
            effect.setup(self._width, self._height)

    def remove_effect(self, name: str) -> Optional[PostProcessEffect]:
        """Remove an effect from the stack.

        Args:
            name: Name of the effect to remove.

        Returns:
            The removed effect, or None if not found.
        """
        effect = self._effect_map.pop(name, None)
        if effect:
            self._effects.remove(effect)
            effect.cleanup()
            self._dirty = True
        return effect

    def get_effect(self, name: str) -> Optional[PostProcessEffect]:
        """Get an effect by name.

        Args:
            name: Name of the effect.

        Returns:
            The effect, or None if not found.
        """
        return self._effect_map.get(name)

    def enable_effect(self, name: str, enabled: bool = True) -> None:
        """Enable or disable an effect.

        Args:
            name: Name of the effect.
            enabled: Whether to enable the effect.
        """
        effect = self._effect_map.get(name)
        if effect:
            effect.enabled = enabled
            self._dirty = True

    def resize(self, width: int, height: int) -> None:
        """Resize all effect resources.

        Args:
            width: New render width.
            height: New render height.
        """
        if width == self._width and height == self._height:
            return

        self._width = width
        self._height = height

        for effect in self._effects:
            effect.setup(width, height)

        self._dirty = True

    def build_frame_graph(self, frame_graph: "FrameGraph") -> None:
        """Add all enabled effects to the frame graph.

        Uses the quality preset to filter which effects get pass nodes.
        Each enabled effect declares its resource dependencies.

        Args:
            frame_graph: The frame graph to populate.
        """
        for effect in self.get_active_effects():
            effect.add_to_frame_graph(frame_graph)

    def execute(
        self,
        hdr_input: Any,
        output: Any,
        delta_time: float,
        camera_position: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Execute the full post-process chain.

        Args:
            hdr_input: HDR scene color input resource.
            output: Final output resource (typically backbuffer).
            delta_time: Time since last frame.
            camera_position: Camera position for volume blending.
        """
        if camera_position and self._volumes:
            self._apply_volume_blending(camera_position)

        context = PostProcessContext(
            quality=self._quality,
            delta_time=delta_time,
            frame_index=self._frame_index,
            camera_position=camera_position,
        )
        self.execute_with_context(hdr_input, output, context)

    def execute_with_context(
        self,
        hdr_input: Any,
        output: Any,
        context: PostProcessContext,
    ) -> None:
        """Execute the post-process chain with a full context.

        Chains effects in priority order, passing output from one
        effect as input to the next. Skips effects filtered by
        quality preset or individual state.

        Args:
            hdr_input: HDR scene color input.
            output: Final output resource.
            context: PostProcessContext with per-frame state.
        """
        if context.camera_position and self._volumes:
            self._apply_volume_blending(context.camera_position)

        current_input = hdr_input
        candidate_effects = self.get_active_effects()

        for i, effect in enumerate(candidate_effects):
            if not effect.should_execute(context, self._quality_preset):
                continue

            is_last = i == len(candidate_effects) - 1
            current_output = output if is_last else self._get_intermediate_target(i)

            inputs = {"color": current_input}
            outputs = {"color": current_output}

            if context.rhi_command_list is not None:
                effect.execute_on_rhi(
                    context.rhi_command_list,
                    inputs,
                    outputs,
                    context,
                )
            else:
                effect.execute_with_context(inputs, outputs, context)

            current_input = current_output

        self._frame_index += 1

        self._current_history_index = (
            self._current_history_index + 1
        ) % self._config.history_buffer_count

    def _get_intermediate_target(self, index: int) -> Any:
        """Get an intermediate render target for ping-pong rendering.

        Args:
            index: Effect index for target selection.

        Returns:
            Render target resource.
        """
        return None

    def _apply_volume_blending(
        self, camera_position: Tuple[float, float, float]
    ) -> None:
        """Blend effect settings from active volumes.

        Args:
            camera_position: Current camera position.
        """
        for volume in self._volumes:
            if volume.contains_point(camera_position):
                weight = volume.get_blend_weight(camera_position)
                volume.apply_to_stack(self, weight)

    def add_volume(self, volume: "PostProcessVolume") -> None:
        """Register a post-process volume.

        Args:
            volume: The volume to add.
        """
        self._volumes.append(volume)
        self._volumes.sort(key=lambda v: v.priority, reverse=True)

    def remove_volume(self, volume: "PostProcessVolume") -> None:
        """Unregister a post-process volume.

        Args:
            volume: The volume to remove.
        """
        if volume in self._volumes:
            self._volumes.remove(volume)

    def cleanup(self) -> None:
        """Release all resources."""
        for effect in self._effects:
            effect.cleanup()
        self._effects.clear()
        self._effect_map.clear()
        self._volumes.clear()


# ==============================================================================
# PostProcessStackExecutor
# ==============================================================================


class PostProcessStackExecutor:
    """Orchestrates post-process effect execution via the S1 Frame Graph.

    The executor bridges the PostProcessStack with the Frame Graph by:
    - Creating pass nodes for each active effect
    - Setting up resource dependencies (read/write)
    - Attaching execution callbacks to pass nodes
    - Managing intermediate targets for effect chaining
    - Supporting direct execution for testing
    - Tagging async compute effects
    """

    def __init__(
        self,
        stack: PostProcessStack,
        frame_graph: Optional["FrameGraph"] = None,
        rhi_device: Optional[Any] = None,
        execution_path: EffectExecutionPath = EffectExecutionPath.FRAME_GRAPH_PASS,
    ) -> None:
        """Initialize the post-process stack executor.

        Args:
            stack: The PostProcessStack to execute.
            frame_graph: Optional S1 Frame Graph for pass creation.
            rhi_device: Optional RHI device for resource operations.
            execution_path: Execution mode (frame graph, direct, merged).
        """
        self._stack: PostProcessStack = stack
        self._frame_graph: Optional["FrameGraph"] = frame_graph
        self._rhi_device: Optional[Any] = rhi_device
        self._execution_path: EffectExecutionPath = execution_path
        self._is_built: bool = False
        self._context: PostProcessContext = PostProcessContext()
        self._intermediate_mgr: IntermediateTargetManager = IntermediateTargetManager(
            format=stack.config.intermediate_format,
        )
        self._hdr_handle: Optional[Any] = None
        self._output_handle: Optional[Any] = None
        self._has_resources: bool = False

    @property
    def stack(self) -> PostProcessStack:
        """The managed post-process stack."""
        return self._stack

    @property
    def frame_graph(self) -> Optional["FrameGraph"]:
        """The frame graph, if bound."""
        return self._frame_graph

    @frame_graph.setter
    def frame_graph(self, fg: Optional["FrameGraph"]) -> None:
        """Bind a frame graph to this executor.

        Setting a new frame graph invalidates the current build state.

        Args:
            fg: The frame graph to bind.
        """
        self._frame_graph = fg
        self._is_built = False

    @property
    def execution_path(self) -> EffectExecutionPath:
        """Current execution path."""
        return self._execution_path

    @execution_path.setter
    def execution_path(self, path: EffectExecutionPath) -> None:
        """Change the execution path."""
        self._execution_path = path

    @property
    def is_built(self) -> bool:
        """Whether the executor has built its frame graph passes."""
        return self._is_built

    def set_context(self, context: PostProcessContext) -> None:
        """Set the execution context.

        Args:
            context: PostProcessContext with per-frame state.
        """
        self._context = context

    def update_context(
        self,
        rhi_command_list: Optional[Any] = None,
        rhi_device: Optional[Any] = None,
        frame_index: Optional[int] = None,
        quality: Optional[EffectQuality] = None,
        delta_time: Optional[float] = None,
        camera_position: Optional[Tuple[float, float, float]] = None,
        history_buffers: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Partially update the execution context fields.

        Only provided fields are updated; others retain their current values.

        Args:
            rhi_command_list: RHI command list.
            rhi_device: RHI device.
            frame_index: Frame counter.
            quality: Quality preset level.
            delta_time: Time since last frame.
            camera_position: Camera position.
            history_buffers: History buffer dictionary.
        """
        if rhi_command_list is not None:
            self._context.rhi_command_list = rhi_command_list
        if rhi_device is not None:
            self._context.rhi_device = rhi_device
        if frame_index is not None:
            self._context.frame_index = frame_index
        if quality is not None:
            self._context.quality = quality
        if delta_time is not None:
            self._context.delta_time = delta_time
        if camera_position is not None:
            self._context.camera_position = camera_position
        if history_buffers is not None:
            self._context.history_buffers.update(history_buffers)

    def prepare_resources(
        self,
        width: int,
        height: int,
        hdr_format: Optional[str] = None,
        intermediate_format: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> None:
        """Prepare render targets for the post-process chain.

        Creates HDR input, output, and intermediate targets in the frame
        graph. If no frame graph is bound, only sets up sizes and calls
        effect setup().

        Args:
            width: Render width.
            height: Render height.
            hdr_format: Override HDR format (default from config).
            intermediate_format: Override intermediate format.
            output_format: Override output format.
        """
        self._stack.resize(width, height)

        if hdr_format:
            self._stack.config.hdr_format = hdr_format
        if intermediate_format:
            self._stack.config.intermediate_format = intermediate_format
            self._intermediate_mgr.format = intermediate_format
        if output_format:
            self._stack.config.output_format = output_format

        if self._frame_graph is None:
            return

        from engine.rendering.framegraph.resource_manager import ResourceFormat

        hdr_fmt_str = self._stack.config.hdr_format
        hdr_fmt_map: dict[str, ResourceFormat] = {
            "R16G16B16A16_FLOAT": ResourceFormat.R16G16B16A16_FLOAT,
            "R32G32B32A32_FLOAT": ResourceFormat.R32G32B32A32_FLOAT,
            "R11G11B10_FLOAT": ResourceFormat.R16G16B16A16_FLOAT,
        }
        hdr_fmt = hdr_fmt_map.get(hdr_fmt_str, ResourceFormat.R16G16B16A16_FLOAT)

        self._hdr_handle = self._frame_graph.create_texture(
            name="PostProcess_HDRInput",
            format=hdr_fmt,
            width=width,
            height=height,
        )

        out_fmt_str = self._stack.config.output_format
        out_fmt_map: dict[str, ResourceFormat] = {
            "R8G8B8A8_UNORM": ResourceFormat.R8G8B8A8_UNORM,
            "R8G8B8A8_SRGB": ResourceFormat.R8G8B8A8_SRGB,
            "R16G16B16A16_FLOAT": ResourceFormat.R16G16B16A16_FLOAT,
            "R11G11B10_FLOAT": ResourceFormat.R16G16B16A16_FLOAT,
        }
        out_fmt = out_fmt_map.get(out_fmt_str, ResourceFormat.R8G8B8A8_UNORM)

        self._output_handle = self._frame_graph.create_texture(
            name="PostProcess_Output",
            format=out_fmt,
            width=width,
            height=height,
        )

        self._intermediate_mgr.allocate(self._frame_graph, width, height)
        self._has_resources = True

    def build_passes(
        self,
        hdr_input: Optional[Any] = None,
        output: Optional[Any] = None,
        context: Optional[PostProcessContext] = None,
    ) -> None:
        """Build frame graph pass nodes for all active effects.

        Creates a pass node for each enabled effect that is active in the
        current quality preset. Each pass node declares its resource
        dependencies and attaches an execution callback.

        Args:
            hdr_input: Override HDR input resource handle.
            output: Override output resource handle.
            context: Execution context for callback creation.

        Raises:
            RuntimeError: If no frame graph is bound.
            RuntimeError: If HDR input and output handles are not available.
        """
        if self._frame_graph is None:
            raise RuntimeError(
                "Frame graph is required to build passes. "
                "Use execute_direct() for direct execution."
            )

        hdr_handle = hdr_input or self._hdr_handle
        output_handle = output or self._output_handle

        if hdr_handle is None or output_handle is None:
            raise RuntimeError(
                "HDR input and output must be provided. "
                "Call prepare_resources() first or pass handles."
            )

        ctx = context or self._context

        active_effects = self._stack.get_active_effects()

        current_input: Any = hdr_handle

        for i, effect in enumerate(active_effects):
            is_last = i == len(active_effects) - 1

            if is_last:
                effect_output: Any = output_handle
            else:
                _, write_tgt = self._intermediate_mgr.get_ping_pong(i)
                effect_output = write_tgt

            pass_node = effect.add_to_frame_graph(self._frame_graph)

            if current_input is not None and current_input != hdr_handle:
                pass_node.read(current_input)

            if not is_last and effect_output is not None:
                pass_node.write(effect_output)

            if effect_output is not None:
                callback = self._make_rhi_effect_callback(
                    effect,
                    current_input,
                    effect_output,
                    ctx,
                )
                pass_node.set_execute(callback)

            if effect.has_execution_flag(ExecutionFlags.FORCE_ASYNC):
                pass_node.set_flag(PassFlags.ASYNC_COMPUTE)

            current_input = effect_output

        self._is_built = True

    def execute_direct(
        self,
        hdr_input: Any,
        output: Any,
        context: Optional[PostProcessContext] = None,
    ) -> None:
        """Execute all effects directly, bypassing the frame graph.

        Chains effects in priority order, filtering by quality preset
        and individual state. Useful for testing and debugging.

        Args:
            hdr_input: HDR input resource.
            output: Final output resource.
            context: Optional execution context (uses internal if not provided).
        """
        ctx = context or self._context
        self._stack.execute_with_context(hdr_input, output, ctx)

    def rebuild_if_needed(self) -> bool:
        """Rebuild frame graph passes if the stack is dirty.

        Checks if the stack needs a rebuild and triggers it if so.
        This handles cases where effects are added/removed or
        quality presets change between frames.

        Returns:
            True if a rebuild occurred, False if already clean.
        """
        if not self._stack._dirty:
            return False

        if self._frame_graph is not None and self._hdr_handle is not None:
            self.build_passes()

        self._stack._dirty = False
        return True

    def reset(self) -> None:
        """Reset the executor state.

        Clears all allocated resources and invalidates the build.
        """
        self._intermediate_mgr.reset()
        self._hdr_handle = None
        self._output_handle = None
        self._has_resources = False
        self._is_built = False

    def _make_rhi_effect_callback(
        self,
        effect: PostProcessEffect,
        effect_input: Any,
        effect_output: Any,
        context: PostProcessContext,
    ) -> Callable:
        """Create an execution callback for a frame graph pass.

        The callback bridges the frame graph's pass execution to the
        effect's execute_on_rhi or execute_with_context method.

        Args:
            effect: The effect to execute.
            effect_input: Input resource handle.
            effect_output: Output resource handle.
            context: PostProcessContext for frame state.

        Returns:
            A callable compatible with PassNode.set_execute().
        """

        def callback(frame_graph_context: Any) -> None:
            """Execution callback invoked by the frame graph.

            Args:
                frame_graph_context: Platform-specific rendering context
                    from the frame graph.
            """
            inputs: Dict[str, Any] = {"color": effect_input}
            outputs: Dict[str, Any] = {"color": effect_output}

            rhi_cmd_list = getattr(frame_graph_context, "command_list", None)
            if rhi_cmd_list is None:
                rhi_cmd_list = context.rhi_command_list

            if rhi_cmd_list is not None:
                effect.execute_on_rhi(rhi_cmd_list, inputs, outputs, context)
            else:
                effect.execute_with_context(inputs, outputs, context)

        return callback


# ==============================================================================
# Volume Shapes
# ==============================================================================


@dataclass
class VolumeShape:
    """Base class for volume shapes."""

    pass


@dataclass
class BoxVolumeShape(VolumeShape):
    """Axis-aligned box volume shape."""

    min_bounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_bounds: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def contains(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is inside the box."""
        return all(
            self.min_bounds[i] <= point[i] <= self.max_bounds[i] for i in range(3)
        )

    def distance_to_boundary(self, point: Tuple[float, float, float]) -> float:
        """Calculate distance from point to nearest boundary."""
        distances = []
        for i in range(3):
            dist_to_min = point[i] - self.min_bounds[i]
            dist_to_max = self.max_bounds[i] - point[i]
            distances.append(min(dist_to_min, dist_to_max))
        return min(distances)


@dataclass
class SphereVolumeShape(VolumeShape):
    """Spherical volume shape."""

    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def contains(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is inside the sphere."""
        dist_sq = sum((point[i] - self.center[i]) ** 2 for i in range(3))
        return dist_sq <= self.radius**2

    def distance_to_boundary(self, point: Tuple[float, float, float]) -> float:
        """Calculate distance from point to sphere boundary."""
        dist = sum((point[i] - self.center[i]) ** 2 for i in range(3)) ** 0.5
        return self.radius - dist


@dataclass
class PostProcessVolumeSettings:
    """Settings container for a post-process volume."""

    effect_overrides: Dict[str, EffectSettings] = field(default_factory=dict)


class PostProcessVolume:
    """Spatial region that influences post-process effect settings.

    Volumes allow localized post-processing effects that blend based on
    camera position within the volume.
    """

    def __init__(
        self,
        shape: VolumeShape,
        settings: PostProcessVolumeSettings,
        priority: int = 0,
        blend_distance: float = 0.0,
        global_volume: bool = False,
    ) -> None:
        """Initialize the post-process volume.

        Args:
            shape: The spatial shape of the volume.
            settings: Effect settings overrides.
            priority: Blending priority (higher = more important).
            blend_distance: Distance over which to blend at edges.
            global_volume: If True, affects entire scene regardless of shape.
        """
        self._id: str = str(uuid.uuid4())
        self._shape: VolumeShape = shape
        self._settings: PostProcessVolumeSettings = settings
        self._priority: int = priority
        self._blend_distance: float = blend_distance
        self._global: bool = global_volume
        self._enabled: bool = True

    @property
    def id(self) -> str:
        """Unique identifier."""
        return self._id

    @property
    def priority(self) -> int:
        """Blending priority."""
        return self._priority

    @priority.setter
    def priority(self, value: int) -> None:
        self._priority = value

    @property
    def enabled(self) -> bool:
        """Whether the volume is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def settings(self) -> PostProcessVolumeSettings:
        """Effect settings overrides."""
        return self._settings

    @settings.setter
    def settings(self, value: PostProcessVolumeSettings) -> None:
        self._settings = value

    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is affected by this volume.

        Args:
            point: World-space position to check.

        Returns:
            True if the point is within the volume's influence.
        """
        if not self._enabled:
            return False

        if self._global:
            return True

        if isinstance(self._shape, (BoxVolumeShape, SphereVolumeShape)):
            if self._shape.contains(point):
                return True
            if self._blend_distance > 0:
                boundary_dist = abs(self._shape.distance_to_boundary(point))
                return boundary_dist <= self._blend_distance

        return False

    def get_blend_weight(self, point: Tuple[float, float, float]) -> float:
        """Calculate the blend weight for a point.

        Args:
            point: World-space position.

        Returns:
            Blend weight in [0, 1].
        """
        if not self._enabled:
            return 0.0

        if self._global:
            return 1.0

        if isinstance(self._shape, (BoxVolumeShape, SphereVolumeShape)):
            if self._shape.contains(point):
                if self._blend_distance <= 0:
                    return 1.0
                boundary_dist = self._shape.distance_to_boundary(point)
                return min(1.0, boundary_dist / self._blend_distance)
            else:
                if self._blend_distance <= 0:
                    return 0.0
                boundary_dist = abs(self._shape.distance_to_boundary(point))
                if boundary_dist > self._blend_distance:
                    return 0.0
                return 1.0 - (boundary_dist / self._blend_distance)

        return 0.0

    def apply_to_stack(self, stack: PostProcessStack, weight: float) -> None:
        """Apply this volume's settings to a stack.

        Args:
            stack: The post-process stack to modify.
            weight: Blend weight [0, 1].
        """
        if weight <= 0:
            return

        for effect_name, override_settings in self._settings.effect_overrides.items():
            effect = stack.get_effect(effect_name)
            if effect and effect.settings:
                blended = effect.settings.lerp(override_settings, weight)
                effect.settings = blended


__all__ = [
    "BlendMode",
    "EffectPriority",
    "EffectQuality",
    "ExecutionFlags",
    "EffectExecutionPath",
    "EffectSettings",
    "QualityPreset",
    "get_quality_preset",
    "QUALITY_PRESETS",
    "QUALITY_PRESET_LOW",
    "QUALITY_PRESET_MEDIUM",
    "QUALITY_PRESET_HIGH",
    "QUALITY_PRESET_ULTRA",
    "PostProcessContext",
    "IntermediateTarget",
    "IntermediateTargetManager",
    "PostProcessEffect",
    "PostProcessStackConfig",
    "PostProcessStack",
    "PostProcessStackExecutor",
    "VolumeShape",
    "BoxVolumeShape",
    "SphereVolumeShape",
    "PostProcessVolumeSettings",
    "PostProcessVolume",
]
