"""
Networking module for AI Game Engine.

Provides serialization, transport, and network quality management.
"""

from .config import NetworkConfig, DEFAULT_CONFIG, get_config

__all__ = [
    'NetworkConfig',
    'DEFAULT_CONFIG',
    'get_config',
]
