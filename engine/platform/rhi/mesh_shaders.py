"""RHI Mesh shader support (stub implementation)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from ..constants import DEFAULT_MESH_MAX_VERTICES, DEFAULT_MESH_MAX_PRIMITIVES


@dataclass
class MeshPipelineDesc:
    """Mesh shader pipeline descriptor."""
    task_shader: Optional['ShaderDesc'] = None
    mesh_shader: Optional['ShaderDesc'] = None
    pixel_shader: Optional['ShaderDesc'] = None
    max_vertices: int = DEFAULT_MESH_MAX_VERTICES
    max_primitives: int = DEFAULT_MESH_MAX_PRIMITIVES
    topology: 'PrimitiveTopology' = None

    def __post_init__(self):
        if self.topology is None:
            from .pipeline import PrimitiveTopology
            self.topology = PrimitiveTopology.TRIANGLE_LIST


# Note: dispatch_mesh is already handled in CommandList
# This module provides the mesh-specific pipeline descriptor


# Forward declarations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .pipeline import ShaderDesc, PrimitiveTopology
