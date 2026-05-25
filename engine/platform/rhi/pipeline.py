"""RHI Pipeline state objects."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Union
import threading

from ..constants import SHADER_HANDLE_START, PIPELINE_HANDLE_START


class ShaderStage(Enum):
    """Shader pipeline stage."""
    VERTEX = auto()
    PIXEL = auto()
    COMPUTE = auto()
    HULL = auto()
    DOMAIN = auto()
    GEOMETRY = auto()
    MESH = auto()
    TASK = auto()
    RAY_GENERATION = auto()
    MISS = auto()
    CLOSEST_HIT = auto()
    ANY_HIT = auto()
    INTERSECTION = auto()


@dataclass
class ShaderDesc:
    """Shader descriptor."""
    stage: ShaderStage
    source: Union[bytes, str]
    entry_point: str = "main"


class Shader(ABC):
    """Abstract shader object."""

    @property
    @abstractmethod
    def desc(self) -> ShaderDesc:
        """Get shader descriptor."""
        pass

    @property
    @abstractmethod
    def handle(self) -> int:
        """Get native handle."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if shader is valid."""
        pass


class PrimitiveTopology(Enum):
    """Primitive topology."""
    TRIANGLE_LIST = auto()
    TRIANGLE_STRIP = auto()
    LINE_LIST = auto()
    LINE_STRIP = auto()
    POINT_LIST = auto()


class FillMode(Enum):
    """Polygon fill mode."""
    SOLID = auto()
    WIREFRAME = auto()


class CullMode(Enum):
    """Polygon culling mode."""
    NONE = auto()
    FRONT = auto()
    BACK = auto()


@dataclass
class RasterizerState:
    """Rasterizer state."""
    fill_mode: FillMode = FillMode.SOLID
    cull_mode: CullMode = CullMode.BACK
    front_ccw: bool = False
    depth_bias: int = 0
    depth_clip: bool = True


@dataclass
class DepthStencilState:
    """Depth-stencil state."""
    depth_test: bool = True
    depth_write: bool = True
    depth_func: 'CompareOp' = None  # Will default to LESS

    def __post_init__(self):
        if self.depth_func is None:
            from .resources import CompareOp
            self.depth_func = CompareOp.LESS


class BlendFactor(Enum):
    """Blend factor."""
    ZERO = auto()
    ONE = auto()
    SRC_COLOR = auto()
    INV_SRC_COLOR = auto()
    SRC_ALPHA = auto()
    INV_SRC_ALPHA = auto()
    DST_COLOR = auto()
    INV_DST_COLOR = auto()
    DST_ALPHA = auto()
    INV_DST_ALPHA = auto()


class BlendOp(Enum):
    """Blend operation."""
    ADD = auto()
    SUBTRACT = auto()
    REV_SUBTRACT = auto()
    MIN = auto()
    MAX = auto()


@dataclass
class BlendState:
    """Blend state."""
    enabled: bool = False
    src_color: BlendFactor = BlendFactor.ONE
    dst_color: BlendFactor = BlendFactor.ZERO
    color_op: BlendOp = BlendOp.ADD
    src_alpha: BlendFactor = BlendFactor.ONE
    dst_alpha: BlendFactor = BlendFactor.ZERO
    alpha_op: BlendOp = BlendOp.ADD


@dataclass
class GraphicsPipelineDesc:
    """Graphics pipeline descriptor."""
    vertex_shader: Optional[ShaderDesc] = None
    pixel_shader: Optional[ShaderDesc] = None
    geometry_shader: Optional[ShaderDesc] = None
    hull_shader: Optional[ShaderDesc] = None
    domain_shader: Optional[ShaderDesc] = None
    topology: PrimitiveTopology = PrimitiveTopology.TRIANGLE_LIST
    rasterizer: RasterizerState = field(default_factory=RasterizerState)
    depth_stencil: DepthStencilState = field(default_factory=DepthStencilState)
    blend: BlendState = field(default_factory=BlendState)
    render_target_formats: List['Format'] = field(default_factory=list)
    depth_format: Optional['Format'] = None


@dataclass
class ComputePipelineDesc:
    """Compute pipeline descriptor."""
    compute_shader: ShaderDesc = None


@dataclass
class RaytracingPipelineDesc:
    """Raytracing pipeline descriptor."""
    ray_gen_shader: Optional[ShaderDesc] = None
    miss_shaders: List[ShaderDesc] = field(default_factory=list)
    hit_groups: List[dict] = field(default_factory=list)
    max_recursion_depth: int = 1


class PipelineType(Enum):
    """Pipeline type."""
    GRAPHICS = auto()
    COMPUTE = auto()
    RAYTRACING = auto()


class PipelineState(ABC):
    """Abstract pipeline state object."""

    @property
    @abstractmethod
    def desc(self) -> Union[GraphicsPipelineDesc, ComputePipelineDesc, RaytracingPipelineDesc]:
        """Get pipeline descriptor."""
        pass

    @property
    @abstractmethod
    def handle(self) -> int:
        """Get native handle."""
        pass

    @property
    @abstractmethod
    def pipeline_type(self) -> PipelineType:
        """Get pipeline type."""
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if pipeline is valid."""
        pass


class NullShader(Shader):
    """Null implementation of Shader."""

    _next_handle = SHADER_HANDLE_START
    _lock = threading.Lock()

    def __init__(self, desc: ShaderDesc):
        self._desc = desc
        with NullShader._lock:
            self._handle = NullShader._next_handle
            NullShader._next_handle += 1
        self._valid = True

    @property
    def desc(self) -> ShaderDesc:
        """Get shader descriptor."""
        return self._desc

    @property
    def handle(self) -> int:
        """Get native handle."""
        return self._handle

    def is_valid(self) -> bool:
        """Check if shader is valid."""
        return self._valid


class NullPipelineState(PipelineState):
    """Null implementation of PipelineState."""

    _next_handle = PIPELINE_HANDLE_START
    _lock = threading.Lock()

    def __init__(self, desc: Union[GraphicsPipelineDesc, ComputePipelineDesc, RaytracingPipelineDesc], pipeline_type: PipelineType):
        self._desc = desc
        self._pipeline_type = pipeline_type
        with NullPipelineState._lock:
            self._handle = NullPipelineState._next_handle
            NullPipelineState._next_handle += 1
        self._valid = True

    @property
    def desc(self) -> Union[GraphicsPipelineDesc, ComputePipelineDesc, RaytracingPipelineDesc]:
        """Get pipeline descriptor."""
        return self._desc

    @property
    def handle(self) -> int:
        """Get native handle."""
        return self._handle

    @property
    def pipeline_type(self) -> PipelineType:
        """Get pipeline type."""
        return self._pipeline_type

    def is_valid(self) -> bool:
        """Check if pipeline is valid."""
        return self._valid


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .resources import Format, CompareOp
