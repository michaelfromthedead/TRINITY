"""FlowForge Backend - Python sidecar for visual programming environment.

This package provides:
- IPC protocol for communication with TypeScript frontend
- AST parsing for Trinity ECS definitions
- Trinity game engine adapter with runtime introspection
- Code generation utilities (future)
"""

__version__ = "0.1.0"

# Re-export key modules for convenience
from . import trinity_introspection

__all__ = [
    "__version__",
    "trinity_introspection",
]
