"""
Tests for high-resolution timing.
"""
import time
import pytest
from engine.platform.os.timing import Timer, Stopwatch


def test_timer_ticks():
    """Test timer tick counting."""
    timer = Timer()

    ticks1 = timer.ticks()
    time.sleep(0.01)
    ticks2 = timer.ticks()

    assert ticks2 > ticks1


def test_timer_seconds():
    """Test timer seconds."""
    timer = Timer()

    sec1 = timer.seconds()
    time.sleep(0.01)
    sec2 = timer.seconds()

    assert sec2 > sec1


def test_timer_ticks_per_second():
    """Test ticks per second constant."""
    timer = Timer()
    assert timer.ticks_per_second() == 1_000_000_000


def test_timer_update():
    """Test timer update and delta."""
    timer = Timer()

    time.sleep(0.01)
    delta = timer.update()

    assert delta > 0
    assert timer.delta() == delta


def test_timer_delta_seconds():
    """Test delta in seconds."""
    timer = Timer()

    time.sleep(0.01)
    timer.update()

    delta_sec = timer.delta_seconds()
    assert delta_sec >= 0.01
    assert delta_sec < 0.1


def test_timer_delta_milliseconds():
    """Test delta in milliseconds."""
    timer = Timer()

    time.sleep(0.01)
    timer.update()

    delta_ms = timer.delta_milliseconds()
    assert delta_ms >= 10
    assert delta_ms < 100


def test_timer_elapsed():
    """Test elapsed time."""
    timer = Timer()

    time.sleep(0.02)

    elapsed = timer.elapsed()
    assert elapsed > 0

    elapsed_sec = timer.elapsed_seconds()
    assert elapsed_sec >= 0.02
    assert elapsed_sec < 0.1


def test_timer_reset():
    """Test timer reset."""
    timer = Timer()

    time.sleep(0.01)
    timer.update()

    timer.reset()

    elapsed = timer.elapsed_seconds()
    assert elapsed < 0.01


def test_timer_pause_resume():
    """Test timer pause and resume."""
    timer = Timer()

    timer.update()
    assert not timer.is_paused()

    timer.pause()
    assert timer.is_paused()

    time.sleep(0.02)
    delta = timer.update()
    assert delta == 0  # No delta while paused

    timer.resume()
    assert not timer.is_paused()

    time.sleep(0.01)
    delta = timer.update()
    assert delta > 0


def test_stopwatch_basic():
    """Test basic stopwatch operations."""
    sw = Stopwatch()

    assert not sw.is_running()

    sw.start()
    assert sw.is_running()

    time.sleep(0.01)

    elapsed = sw.stop()
    assert not sw.is_running()
    assert elapsed > 0


def test_stopwatch_elapsed():
    """Test stopwatch elapsed time."""
    sw = Stopwatch()

    sw.start()
    time.sleep(0.01)

    elapsed_ns = sw.elapsed()
    elapsed_sec = sw.elapsed_seconds()
    elapsed_ms = sw.elapsed_milliseconds()
    elapsed_us = sw.elapsed_microseconds()

    assert elapsed_ns > 0
    assert elapsed_sec >= 0.01
    assert elapsed_ms >= 10
    assert elapsed_us >= 10000

    sw.stop()


def test_stopwatch_reset():
    """Test stopwatch reset."""
    sw = Stopwatch()

    sw.start()
    time.sleep(0.01)
    sw.stop()

    elapsed1 = sw.elapsed()
    assert elapsed1 > 0

    sw.reset()
    assert sw.elapsed() == 0
    assert not sw.is_running()


def test_stopwatch_restart():
    """Test stopwatch restart."""
    sw = Stopwatch()

    sw.start()
    time.sleep(0.01)

    sw.restart()
    assert sw.is_running()

    elapsed = sw.elapsed_seconds()
    assert elapsed < 0.01


def test_stopwatch_context_manager():
    """Test stopwatch as context manager."""
    sw = Stopwatch()

    with sw:
        time.sleep(0.01)
        assert sw.is_running()

    assert not sw.is_running()
    assert sw.elapsed_seconds() >= 0.01


def test_stopwatch_multiple_start_stop():
    """Test multiple start/stop cycles."""
    sw = Stopwatch()

    sw.start()
    time.sleep(0.01)
    elapsed1 = sw.stop()

    sw.start()
    time.sleep(0.01)
    elapsed2 = sw.stop()

    # Total elapsed should be cumulative
    total = sw.elapsed()
    assert total >= elapsed1
    assert total >= elapsed2


def test_timer_multiple_updates():
    """Test multiple timer updates."""
    timer = Timer()

    deltas = []
    for _ in range(5):
        time.sleep(0.01)
        delta = timer.update()
        deltas.append(delta)

    # All deltas should be positive
    for delta in deltas:
        assert delta > 0


def test_stopwatch_elapsed_while_running():
    """Test getting elapsed time while stopwatch is running."""
    sw = Stopwatch()

    sw.start()
    time.sleep(0.01)

    # Should be able to get elapsed time while running
    elapsed1 = sw.elapsed()
    assert elapsed1 > 0

    time.sleep(0.01)

    # Elapsed should increase
    elapsed2 = sw.elapsed()
    assert elapsed2 > elapsed1

    sw.stop()


def test_timer_precision():
    """Test timer precision."""
    timer = Timer()

    # Measure a very short sleep
    time.sleep(0.001)
    timer.update()

    delta_sec = timer.delta_seconds()
    # Should be able to measure millisecond precision
    assert delta_sec >= 0.001
    assert delta_sec < 0.01


def test_stopwatch_precision():
    """Test stopwatch precision."""
    sw = Stopwatch()

    sw.start()
    time.sleep(0.001)
    sw.stop()

    elapsed_ms = sw.elapsed_milliseconds()
    # Should be able to measure millisecond precision
    assert elapsed_ms >= 1
    assert elapsed_ms < 10
