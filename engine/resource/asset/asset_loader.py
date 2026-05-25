"""Asset loading abstractions: sync and async loaders."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from engine.resource.constants import DEFAULT_LOADER_WORKERS, DEFAULT_LOAD_PRIORITY

__all__ = ["AssetLoader", "SyncLoader", "AsyncLoader", "LoadRequest", "LoadResult"]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoadResult:
    """Result of a load operation."""
    success: bool
    data: Any = None
    error: str | None = None

    @staticmethod
    def ok(data: Any) -> LoadResult:
        return LoadResult(success=True, data=data)

    @staticmethod
    def fail(error: str) -> LoadResult:
        return LoadResult(success=False, error=error)


@dataclass(slots=True)
class LoadRequest:
    """Describes a pending load."""
    path: str
    asset_type: type
    priority: int = DEFAULT_LOAD_PRIORITY
    callback: Callable[[LoadResult], None] | None = None


class AssetLoader(ABC):
    """Interface for asset loading backends."""
    __slots__ = ()

    @abstractmethod
    def load(self, path: str, asset_type: type) -> LoadResult:
        """Load an asset from *path*."""
        ...

    @abstractmethod
    def unload(self, data: Any) -> None:
        """Release loaded data."""
        ...


class SyncLoader(AssetLoader):
    """Loads assets synchronously by reading the file from disk."""
    __slots__ = ()

    def load(self, path: str, asset_type: type) -> LoadResult:
        try:
            with open(path, "rb") as f:
                data = f.read()
            return LoadResult.ok(data)
        except Exception as exc:
            return LoadResult.fail(str(exc))

    def unload(self, data: Any) -> None:
        pass  # nothing to release for raw bytes


class AsyncLoader(AssetLoader):
    """Loads assets via a thread pool."""
    __slots__ = ("_executor", "_inner")

    def __init__(
        self,
        inner: AssetLoader | None = None,
        max_workers: int = DEFAULT_LOADER_WORKERS,
    ) -> None:
        self._inner: AssetLoader = inner or SyncLoader()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def load(self, path: str, asset_type: type) -> LoadResult:
        return self._inner.load(path, asset_type)

    def load_async(self, request: LoadRequest) -> Future[LoadResult]:
        """Submit a load request to the thread pool."""
        future = self._executor.submit(self._inner.load, request.path, request.asset_type)
        if request.callback:
            def _on_complete(f: Future[LoadResult], cb: Callable[[LoadResult], None] = request.callback) -> None:
                exc = f.exception()
                if exc:
                    logger.error("Async load failed: %s", exc)
                else:
                    cb(f.result())
            future.add_done_callback(_on_complete)
        return future

    def unload(self, data: Any) -> None:
        self._inner.unload(data)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
