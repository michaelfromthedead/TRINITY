"""Central streaming coordinator with priority queue."""

from __future__ import annotations

import enum
import heapq
import itertools
from dataclasses import dataclass, field

from engine.resource.constants import MAX_CONCURRENT_STREAMS

__all__ = [
    "StreamType",
    "StreamState",
    "StreamPriority",
    "StreamRequest",
    "StreamManager",
    "MAX_CONCURRENT_STREAMS",
]


class StreamType(enum.Enum):
    """Asset stream categories."""

    TEXTURE_MIP = "texture_mip"
    MESH_LOD = "mesh_lod"
    AUDIO_CHUNK = "audio_chunk"
    WORLD_CHUNK = "world_chunk"


class StreamState(enum.Enum):
    """Lifecycle states of a stream request."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StreamPriority(enum.IntEnum):
    """Priority levels (lower value = higher priority)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


_id_counter = itertools.count(1)


@dataclass(slots=True)
class StreamRequest:
    """A single streaming request."""

    request_id: int = field(default_factory=lambda: next(_id_counter), init=False)
    asset_id: str = ""
    priority: StreamPriority = StreamPriority.NORMAL
    stream_type: StreamType = StreamType.TEXTURE_MIP
    state: StreamState = field(default=StreamState.PENDING, init=False)
    bytes_loaded: int = field(default=0, init=False)
    bytes_total: int = field(default=0, init=False)

    def __lt__(self, other: StreamRequest) -> bool:  # type: ignore[override]
        return (self.priority, self.request_id) < (other.priority, other.request_id)


class StreamManager:
    """Central streaming coordinator managing a priority queue of requests."""

    __slots__ = ("_pending", "_active", "_completed", "_all")

    def __init__(self) -> None:
        self._pending: list[StreamRequest] = []  # heapq
        self._active: dict[int, StreamRequest] = {}
        self._completed: dict[int, StreamRequest] = {}
        self._all: dict[int, StreamRequest] = {}

    def request_stream(
        self,
        asset_id: str,
        priority: StreamPriority = StreamPriority.NORMAL,
        stream_type: StreamType = StreamType.TEXTURE_MIP,
    ) -> StreamRequest:
        """Create and enqueue a new stream request."""
        req = StreamRequest(asset_id=asset_id, priority=priority, stream_type=stream_type)
        heapq.heappush(self._pending, req)
        self._all[req.request_id] = req
        return req

    def cancel(self, request_id: int) -> bool:
        """Cancel a pending or active request. Returns True if found."""
        req = self._all.get(request_id)
        if req is None:
            return False
        if req.state in (StreamState.COMPLETE, StreamState.CANCELLED, StreamState.FAILED):
            return False
        req.state = StreamState.CANCELLED
        self._active.pop(request_id, None)
        self._completed[request_id] = req
        return True

    def update(self) -> None:
        """Process pending requests up to MAX_CONCURRENT_STREAMS active."""
        # Promote pending to active.
        while len(self._active) < MAX_CONCURRENT_STREAMS and self._pending:
            req = heapq.heappop(self._pending)
            if req.state == StreamState.CANCELLED:
                continue
            req.state = StreamState.ACTIVE
            self._active[req.request_id] = req

        # Simulate progress: advance active requests.
        finished: list[int] = []
        for rid, req in self._active.items():
            if req.state == StreamState.CANCELLED:
                finished.append(rid)
                continue
            if req.bytes_total > 0:
                req.bytes_loaded = min(req.bytes_loaded + req.bytes_total, req.bytes_total)
                if req.bytes_loaded >= req.bytes_total:
                    req.state = StreamState.COMPLETE
                    finished.append(rid)
            else:
                # Zero-size requests complete immediately.
                req.state = StreamState.COMPLETE
                finished.append(rid)

        for rid in finished:
            req = self._active.pop(rid)
            self._completed[rid] = req

    def get_active_streams(self) -> list[StreamRequest]:
        """Return currently active stream requests."""
        return list(self._active.values())

    def get_pending_count(self) -> int:
        """Return number of pending requests (excludes cancelled)."""
        return sum(1 for r in self._pending if r.state == StreamState.PENDING)
