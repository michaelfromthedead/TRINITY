"""
Audio backend registry.

Provides a central registry for audio backend implementations.
Backends can be registered and retrieved by name.
"""

from typing import Dict, Type, Optional
from ..audio_device import AudioBackend


class BackendRegistry:
    """Registry for audio backend implementations."""

    def __init__(self):
        """Initialize backend registry."""
        self._backends: Dict[str, Type[AudioBackend]] = {}
        self._default_backend: Optional[str] = None

    def register(
        self,
        name: str,
        backend_class: Type[AudioBackend],
        set_default: bool = False
    ) -> None:
        """Register an audio backend.

        Args:
            name: Backend name
            backend_class: Backend class to register
            set_default: Whether to set as default backend
        """
        self._backends[name] = backend_class

        if set_default or self._default_backend is None:
            self._default_backend = name

    def get(self, name: str) -> Optional[Type[AudioBackend]]:
        """Get a backend by name.

        Args:
            name: Backend name

        Returns:
            Backend class or None if not found
        """
        return self._backends.get(name)

    def get_default(self) -> Optional[Type[AudioBackend]]:
        """Get the default backend.

        Returns:
            Default backend class or None
        """
        if self._default_backend is None:
            return None
        return self._backends.get(self._default_backend)

    def list_backends(self) -> list[str]:
        """List all registered backend names.

        Returns:
            List of backend names
        """
        return list(self._backends.keys())

    def create_backend(self, name: Optional[str] = None) -> Optional[AudioBackend]:
        """Create a backend instance.

        Args:
            name: Backend name (uses default if None)

        Returns:
            Backend instance or None if not found
        """
        if name is None:
            backend_class = self.get_default()
        else:
            backend_class = self.get(name)

        if backend_class is None:
            return None

        return backend_class()


# Global backend registry
_registry = BackendRegistry()


def register_backend(
    name: str,
    backend_class: Type[AudioBackend],
    set_default: bool = False
) -> None:
    """Register an audio backend.

    Args:
        name: Backend name
        backend_class: Backend class to register
        set_default: Whether to set as default backend
    """
    _registry.register(name, backend_class, set_default)


def get_backend(name: str) -> Optional[Type[AudioBackend]]:
    """Get a backend by name.

    Args:
        name: Backend name

    Returns:
        Backend class or None if not found
    """
    return _registry.get(name)


def get_default_backend() -> Optional[Type[AudioBackend]]:
    """Get the default backend.

    Returns:
        Default backend class or None
    """
    return _registry.get_default()


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        List of backend names
    """
    return _registry.list_backends()


def create_backend(name: Optional[str] = None) -> Optional[AudioBackend]:
    """Create a backend instance.

    Args:
        name: Backend name (uses default if None)

    Returns:
        Backend instance or None if not found
    """
    return _registry.create_backend(name)


# Auto-register null backend
from .null_backend import NullAudioBackend
register_backend("null", NullAudioBackend, set_default=True)
