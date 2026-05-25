"""
Generic backend registry for platform subsystems.

Provides a thread-safe, type-parameterized registry for pluggable backends
across all platform subsystems (audio, window, input, services, etc.).

Usage::

    from engine.platform.registry import BackendRegistry

    class AudioBackend:
        def play(self) -> None: ...

    class NullAudioBackend(AudioBackend):
        def play(self) -> None: ...

    registry: BackendRegistry[AudioBackend] = BackendRegistry()
    registry.register("null", NullAudioBackend, set_default=True)

    backend = registry.create()          # -> NullAudioBackend instance
    backend = registry.create("null")    # explicit name

Subsystems follow the environment-variable convention for runtime backend
selection (e.g. ``TRINITY_AUDIO_BACKEND``), falling back to the registered
default, then to ``"null"`` if no default was set.

See ``docs/INVESTIGATION_PHASE_X_OUTPUT_CORRECTED/engine_platform/PHASE_1_ARCH.md``
for the full architectural rationale (ADR-P1-001).
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class BackendRegistry(Generic[T]):
    """Thread-safe registry that maps backend names to implementation classes.

    Type parameter ``T`` is the backend interface (protocol or abstract base
    class) that all registered implementations must satisfy.

    Typical lifecycle:

    1. Subsystem backends register themselves at import time via ``register()``.
    2. At startup the subsystem queries the registry for a named backend
       (from env var, config, or platform default).
    3. ``create()`` instantiates the chosen backend, passing through any
       constructor arguments.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._backends: dict[str, type[T]] = {}
        self._default: str | None = None

    def register(
        self, name: str, backend_cls: type[T], set_default: bool = False
    ) -> None:
        """Register a backend implementation class under *name*.

        Parameters
        ----------
        name:
            Unique key for this backend (e.g. ``"null"``, ``"alsa"``,
            ``"wasapi"``).
        backend_cls:
            The class implementing interface ``T``.
        set_default:
            If ``True``, make this backend the default for ``create()`` calls
            that do not specify a name.
        """
        with self._lock:
            self._backends[name] = backend_cls
            if set_default:
                self._default = name

    def get(self, name: str) -> type[T] | None:
        """Look up a registered backend class by *name*.

        Returns ``None`` when *name* has not been registered.
        """
        with self._lock:
            return self._backends.get(name)

    def create(self, name: str | None = None, *args: object, **kwargs: object) -> T:
        """Create an instance of the backend identified by *name*.

        Parameters
        ----------
        name:
            Backend to instantiate.  When ``None`` the default backend is used.
        *args, **kwargs:
            Forwarded to the backend class constructor.

        Returns
        -------
        An instance of the registered class.

        Raises
        ------
        ValueError
            If *name* is not registered, or if no default has been set and
            *name* was not given.
        """
        with self._lock:
            resolved = name if name is not None else self._default
            if resolved is None:
                raise ValueError(
                    "No default backend set. "
                    "Either register a default or pass an explicit name."
                )
            cls = self._backends.get(resolved)
            if cls is None:
                raise ValueError(
                    f"Unknown backend: {resolved!r}. "
                    f"Registered backends: {list(self._backends)}"
                )
            return cls(*args, **kwargs)

    def list(self) -> list[str]:
        """Return a sorted snapshot of all registered backend names."""
        with self._lock:
            return sorted(self._backends)

    def default(self) -> str | None:
        """Return the name of the current default backend, or ``None``."""
        with self._lock:
            return self._default
