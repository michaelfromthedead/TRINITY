"""
Tests for replay_browser.py - Browse, search, and filter replays.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from engine.tooling.replay.replay_browser import (
    ReplayBrowser,
    ReplayEntry,
    ReplayFilter,
    ReplaySortOrder,
    ReplaySearchResult,
)
from engine.tooling.replay.replay_file import ReplayFile, ReplayMetadata
from engine.tooling.replay.input_recorder import RecordedInput, InputType
from engine.tooling.replay.state_recorder import StateSnapshot


def create_test_replay(
    tmpdir: Path,
    name: str,
    player: str = "Player1",
    map_name: str = "TestMap",
    duration: float = 60.0,
    tags: list[str] = None,
    result: str = None,
    recorded_at: datetime = None
) -> Path:
    """Create a test replay file."""
    path = tmpdir / f"{name}.replay"

    replay = ReplayFile()
    inputs = [
        RecordedInput(
            input_type=InputType.KEYBOARD,
            timestamp=i * 0.016,
            frame=i,
            device_id=0,
            data={'key': 'A'}
        )
        for i in range(int(duration * 60))
    ]
    snapshots = [
        StateSnapshot(
            frame=0,
            timestamp=0.0,
            state_data={'value': 0},
            checksum='',
            size_bytes=0
        )
    ]

    meta = ReplayMetadata(
        game_name="TestGame",
        player_name=player,
        map_name=map_name,
        duration=duration,
        tags=tags or [],
        result=result,
        recorded_at=recorded_at or datetime.now()
    )
    replay.set_data(inputs, snapshots, metadata=meta)
    replay.save(path)

    return path


class TestReplayFilter:
    """Tests for ReplayFilter dataclass."""

    def test_default_filter(self):
        """Test default filter matches everything."""
        filter = ReplayFilter()
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(player_name="Test"),
            file_size=1000,
            modified_time=datetime.now()
        )
        assert filter.matches(entry)

    def test_text_search_player(self):
        """Test text search in player name."""
        filter = ReplayFilter(search_text="alice", search_in_player=True)
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(player_name="Alice123"),
            file_size=1000,
            modified_time=datetime.now()
        )
        assert filter.matches(entry)

    def test_text_search_map(self):
        """Test text search in map name."""
        filter = ReplayFilter(search_text="desert", search_in_map=True)
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(map_name="Desert Storm"),
            file_size=1000,
            modified_time=datetime.now()
        )
        assert filter.matches(entry)

    def test_text_search_tags(self):
        """Test text search in tags."""
        filter = ReplayFilter(search_text="ranked", search_in_tags=True)
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(tags=["ranked", "competitive"]),
            file_size=1000,
            modified_time=datetime.now()
        )
        assert filter.matches(entry)

    def test_date_after_filter(self):
        """Test filtering by date after."""
        cutoff = datetime.now() - timedelta(days=7)
        filter = ReplayFilter(recorded_after=cutoff)

        recent_entry = ReplayEntry(
            path=Path("/recent.replay"),
            metadata=ReplayMetadata(recorded_at=datetime.now()),
            file_size=1000,
            modified_time=datetime.now()
        )
        old_entry = ReplayEntry(
            path=Path("/old.replay"),
            metadata=ReplayMetadata(recorded_at=datetime.now() - timedelta(days=30)),
            file_size=1000,
            modified_time=datetime.now()
        )

        assert filter.matches(recent_entry)
        assert not filter.matches(old_entry)

    def test_duration_filter(self):
        """Test filtering by duration."""
        filter = ReplayFilter(min_duration=30.0, max_duration=120.0)

        short_entry = ReplayEntry(
            path=Path("/short.replay"),
            metadata=ReplayMetadata(duration=15.0),
            file_size=1000,
            modified_time=datetime.now()
        )
        medium_entry = ReplayEntry(
            path=Path("/medium.replay"),
            metadata=ReplayMetadata(duration=60.0),
            file_size=1000,
            modified_time=datetime.now()
        )
        long_entry = ReplayEntry(
            path=Path("/long.replay"),
            metadata=ReplayMetadata(duration=180.0),
            file_size=1000,
            modified_time=datetime.now()
        )

        assert not filter.matches(short_entry)
        assert filter.matches(medium_entry)
        assert not filter.matches(long_entry)

    def test_player_filter(self):
        """Test filtering by player name."""
        filter = ReplayFilter(player_name="Alice")

        alice_entry = ReplayEntry(
            path=Path("/alice.replay"),
            metadata=ReplayMetadata(player_name="Alice"),
            file_size=1000,
            modified_time=datetime.now()
        )
        bob_entry = ReplayEntry(
            path=Path("/bob.replay"),
            metadata=ReplayMetadata(player_name="Bob"),
            file_size=1000,
            modified_time=datetime.now()
        )

        assert filter.matches(alice_entry)
        assert not filter.matches(bob_entry)

    def test_result_filter(self):
        """Test filtering by result."""
        filter = ReplayFilter(result="win")

        win_entry = ReplayEntry(
            path=Path("/win.replay"),
            metadata=ReplayMetadata(result="win"),
            file_size=1000,
            modified_time=datetime.now()
        )
        loss_entry = ReplayEntry(
            path=Path("/loss.replay"),
            metadata=ReplayMetadata(result="loss"),
            file_size=1000,
            modified_time=datetime.now()
        )

        assert filter.matches(win_entry)
        assert not filter.matches(loss_entry)

    def test_required_tags_filter(self):
        """Test filtering by required tags."""
        filter = ReplayFilter(required_tags=["ranked", "solo"])

        matching_entry = ReplayEntry(
            path=Path("/match.replay"),
            metadata=ReplayMetadata(tags=["ranked", "solo", "extra"]),
            file_size=1000,
            modified_time=datetime.now()
        )
        partial_entry = ReplayEntry(
            path=Path("/partial.replay"),
            metadata=ReplayMetadata(tags=["ranked"]),  # Missing "solo"
            file_size=1000,
            modified_time=datetime.now()
        )

        assert filter.matches(matching_entry)
        assert not filter.matches(partial_entry)

    def test_excluded_tags_filter(self):
        """Test filtering by excluded tags."""
        filter = ReplayFilter(excluded_tags=["test", "debug"])

        normal_entry = ReplayEntry(
            path=Path("/normal.replay"),
            metadata=ReplayMetadata(tags=["ranked"]),
            file_size=1000,
            modified_time=datetime.now()
        )
        test_entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(tags=["test"]),
            file_size=1000,
            modified_time=datetime.now()
        )

        assert filter.matches(normal_entry)
        assert not filter.matches(test_entry)

    def test_custom_filter(self):
        """Test custom filter function."""
        def my_filter(entry):
            return entry.file_size > 5000

        filter = ReplayFilter(custom_filter=my_filter)

        small_entry = ReplayEntry(
            path=Path("/small.replay"),
            metadata=ReplayMetadata(),
            file_size=1000,
            modified_time=datetime.now()
        )
        large_entry = ReplayEntry(
            path=Path("/large.replay"),
            metadata=ReplayMetadata(),
            file_size=10000,
            modified_time=datetime.now()
        )

        assert not filter.matches(small_entry)
        assert filter.matches(large_entry)


class TestReplayEntry:
    """Tests for ReplayEntry dataclass."""

    def test_create_entry(self):
        """Test creating an entry."""
        entry = ReplayEntry(
            path=Path("/replays/test.replay"),
            metadata=ReplayMetadata(player_name="Player1", duration=120.5),
            file_size=50000,
            modified_time=datetime.now()
        )
        assert entry.filename == "test.replay"
        assert entry.metadata.player_name == "Player1"

    def test_duration_formatted(self):
        """Test formatted duration string."""
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(duration=3665.0),  # 1h 1m 5s
            file_size=1000,
            modified_time=datetime.now()
        )
        assert "1:01:05" in entry.duration_formatted

    def test_size_formatted(self):
        """Test formatted size string."""
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(),
            file_size=1536000,  # ~1.5 MB
            modified_time=datetime.now()
        )
        assert "MB" in entry.size_formatted

    def test_to_dict(self):
        """Test converting to dictionary."""
        entry = ReplayEntry(
            path=Path("/test.replay"),
            metadata=ReplayMetadata(player_name="Test"),
            file_size=1000,
            modified_time=datetime.now()
        )
        data = entry.to_dict()

        assert 'path' in data
        assert 'filename' in data
        assert 'metadata' in data
        assert 'file_size' in data


class TestReplaySearchResult:
    """Tests for ReplaySearchResult dataclass."""

    def test_create_result(self):
        """Test creating a search result."""
        result = ReplaySearchResult(
            entries=[],
            total_count=100,
            filter_used=None,
            sort_order=ReplaySortOrder.DATE_NEWEST,
            search_time=0.05,
            page=1,
            page_size=50
        )
        assert result.total_count == 100
        assert result.total_pages == 2

    def test_pagination_properties(self):
        """Test pagination properties."""
        result = ReplaySearchResult(
            entries=[],
            total_count=120,
            filter_used=None,
            sort_order=ReplaySortOrder.DATE_NEWEST,
            search_time=0.0,
            page=2,
            page_size=50
        )
        assert result.total_pages == 3
        assert result.has_previous_page
        assert result.has_next_page


class TestReplayBrowser:
    """Tests for ReplayBrowser class."""

    def test_create_browser(self):
        """Test creating a browser."""
        browser = ReplayBrowser()
        assert browser.cache_size == 0

    def test_add_root_path(self):
        """Test adding root path."""
        browser = ReplayBrowser()
        browser.add_root_path("/replays")

        assert Path("/replays") in browser.root_paths

    def test_remove_root_path(self):
        """Test removing root path."""
        browser = ReplayBrowser(["/replays"])
        removed = browser.remove_root_path("/replays")

        assert removed
        assert Path("/replays") not in browser.root_paths

    def test_scan_directory(self):
        """Test scanning directory for replays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test replays
            create_test_replay(tmpdir, "replay1")
            create_test_replay(tmpdir, "replay2")

            browser = ReplayBrowser([tmpdir])
            count = browser.scan()

            assert count == 2
            assert browser.cache_size == 2

    def test_search_all(self):
        """Test searching all replays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "replay1", player="Alice")
            create_test_replay(tmpdir, "replay2", player="Bob")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            result = browser.search()
            assert result.total_count == 2

    def test_search_with_filter(self):
        """Test searching with filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "replay1", player="Alice")
            create_test_replay(tmpdir, "replay2", player="Bob")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            filter = ReplayFilter(player_name="Alice")
            result = browser.search(filter)

            assert result.total_count == 1
            assert result.entries[0].metadata.player_name == "Alice"

    def test_search_sort_order(self):
        """Test search with sort order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "short", duration=30.0)
            create_test_replay(tmpdir, "long", duration=120.0)

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            result = browser.search(sort=ReplaySortOrder.DURATION_LONGEST)
            # The longer replay should be first
            assert result.entries[0].metadata.duration > result.entries[1].metadata.duration

    def test_search_pagination(self):
        """Test search pagination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i in range(15):
                create_test_replay(tmpdir, f"replay{i}")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            result = browser.search(page=1, page_size=10)
            assert len(result.entries) == 10
            assert result.total_count == 15
            assert result.has_next_page

    def test_get_entry(self):
        """Test getting specific entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            path = create_test_replay(tmpdir, "test")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            entry = browser.get_entry(path)
            assert entry is not None

    def test_get_unique_values(self):
        """Test getting unique field values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "replay1", map_name="Map1")
            create_test_replay(tmpdir, "replay2", map_name="Map2")
            create_test_replay(tmpdir, "replay3", map_name="Map1")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            maps = browser.get_maps()
            assert len(maps) == 2
            assert "Map1" in maps
            assert "Map2" in maps

    def test_get_statistics(self):
        """Test getting collection statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "replay1", duration=60.0, result="win")
            create_test_replay(tmpdir, "replay2", duration=120.0, result="loss")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            stats = browser.get_statistics()
            assert stats['total_replays'] == 2
            assert stats['wins'] == 1
            assert stats['losses'] == 1

    def test_find_recent(self):
        """Test finding recent replays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(
                tmpdir, "recent",
                recorded_at=datetime.now()
            )
            create_test_replay(
                tmpdir, "old",
                recorded_at=datetime.now() - timedelta(days=30)
            )

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            recent = browser.find_recent(days=7)
            assert len(recent) == 1

    def test_find_by_player(self):
        """Test finding replays by player."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "replay1", player="Alice")
            create_test_replay(tmpdir, "replay2", player="Bob")
            create_test_replay(tmpdir, "replay3", player="Alice")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            alice_replays = browser.find_by_player("Alice")
            assert len(alice_replays) == 2

    def test_delete_replay(self):
        """Test deleting a replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            path = create_test_replay(tmpdir, "to_delete")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            assert browser.cache_size == 1
            deleted = browser.delete_replay(path)
            assert deleted
            assert not path.exists()
            assert browser.cache_size == 0

    def test_clear_cache(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "test")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            assert browser.cache_size == 1
            browser.clear_cache()
            assert browser.cache_size == 0

    def test_progress_callback(self):
        """Test scan progress callback."""
        progress_updates = []

        def on_progress(current, total):
            progress_updates.append((current, total))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            for i in range(5):
                create_test_replay(tmpdir, f"replay{i}")

            browser = ReplayBrowser([tmpdir])
            browser.set_progress_callback(on_progress)
            browser.scan()

            assert len(progress_updates) == 5
            assert progress_updates[-1][0] == 5

    def test_iter_entries(self):
        """Test iterating over entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            create_test_replay(tmpdir, "replay1")
            create_test_replay(tmpdir, "replay2")

            browser = ReplayBrowser([tmpdir])
            browser.scan()

            entries = list(browser.iter_entries())
            assert len(entries) == 2
