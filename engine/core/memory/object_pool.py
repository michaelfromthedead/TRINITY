"""Generic object pool — reuse objects instead of repeated construction."""

from __future__ import annotations

import logging
from typing import Callable, Generic, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ObjectPool(Generic[T]):
    """Pool that recycles objects created by *factory*.

    Call :meth:`acquire` to get an object (recycled or new) and
    :meth:`release` to return it for future reuse.
    """

    def __init__(
        self,
        factory: Callable[[], T],
        initial_size: int = 0,
        max_size: int | None = None,
        reset_func: Callable[[T], None] | None = None,
    ) -> None:
        self._factory = factory
        self._pool: List[T] = [factory() for _ in range(initial_size)]
        self._total_created = initial_size
        self._max_size = max_size
        self._reset_func = reset_func

    def acquire(self) -> T:
        if self._pool:
            logger.debug("object_pool reuse (available=%d)", len(self._pool) - 1)
            return self._pool.pop()
        self._total_created += 1
        logger.debug("object_pool create new (total=%d)", self._total_created)
        return self._factory()

    def release(self, obj: T) -> None:
        if self._reset_func is not None:
            self._reset_func(obj)
        if self._max_size is not None and len(self._pool) >= self._max_size:
            logger.debug("object_pool release discarded (at max_size=%d)", self._max_size)
            return
        self._pool.append(obj)
        logger.debug("object_pool release (available=%d)", len(self._pool))

    @property
    def available(self) -> int:
        return len(self._pool)

    @property
    def total_created(self) -> int:
        return self._total_created
