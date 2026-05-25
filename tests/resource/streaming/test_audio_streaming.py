"""Tests for AudioStreamManager."""

from engine.resource.streaming.audio_streaming import (
    AUDIO_CHUNK_SIZE,
    AudioChunk,
    AudioStreamManager,
)


class TestAudioChunk:
    def test_defaults(self) -> None:
        chunk = AudioChunk()
        assert chunk.chunk_index == 0
        assert chunk.sample_count == AUDIO_CHUNK_SIZE
        assert chunk.data is None

    def test_sample_offset(self) -> None:
        chunk = AudioChunk(chunk_index=5, sample_offset=5 * AUDIO_CHUNK_SIZE)
        assert chunk.sample_offset == 5 * AUDIO_CHUNK_SIZE


class TestAudioStreamManager:
    def test_empty_buffered_range(self) -> None:
        mgr = AudioStreamManager()
        assert mgr.get_buffered_range("unknown") == (0, 0)

    def test_request_and_update(self) -> None:
        mgr = AudioStreamManager()
        mgr.request_chunks("audio_01", start_chunk=0, count=3)
        mgr.update()
        start, end = mgr.get_buffered_range("audio_01")
        assert start == 0
        assert end == 3

    def test_non_contiguous_chunks(self) -> None:
        mgr = AudioStreamManager()
        mgr.request_chunks("a", start_chunk=5, count=2)
        mgr.update()
        start, end = mgr.get_buffered_range("a")
        assert start == 5
        assert end == 7

    def test_chunk_data_populated(self) -> None:
        mgr = AudioStreamManager()
        mgr.request_chunks("a", start_chunk=0, count=1)
        mgr.update()
        buf = mgr._buffers["a"]
        assert buf[0].data is not None
        assert len(buf[0].data) == AUDIO_CHUNK_SIZE

    def test_multiple_requests_extend_buffer(self) -> None:
        mgr = AudioStreamManager()
        mgr.request_chunks("a", start_chunk=0, count=2)
        mgr.update()
        mgr.request_chunks("a", start_chunk=2, count=3)
        mgr.update()
        start, end = mgr.get_buffered_range("a")
        assert start == 0
        assert end == 5
