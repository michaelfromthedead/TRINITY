"""
Tests for dirty tracking functionality.
"""
import pytest
import time

from engine.tooling.undo.dirty_tracking import (
    DirtyTracker,
    DirtyState,
    DirtyInfo,
    DocumentDirtyTracker,
    SavePromptResult,
)


class TestDirtyState:
    """Tests for DirtyState enum."""

    def test_all_states_exist(self):
        """Test all expected states exist."""
        assert hasattr(DirtyState, "CLEAN")
        assert hasattr(DirtyState, "DIRTY")
        assert hasattr(DirtyState, "SAVING")
        assert hasattr(DirtyState, "ERROR")


class TestSavePromptResult:
    """Tests for SavePromptResult enum."""

    def test_all_results_exist(self):
        """Test all expected results exist."""
        assert hasattr(SavePromptResult, "SAVE")
        assert hasattr(SavePromptResult, "DONT_SAVE")
        assert hasattr(SavePromptResult, "CANCEL")


class TestDirtyInfo:
    """Tests for DirtyInfo."""

    def test_info_creation(self):
        """Test creating dirty info."""
        info = DirtyInfo(
            document_id="doc1",
            document_name="My Document",
            state=DirtyState.CLEAN,
        )

        assert info.document_id == "doc1"
        assert info.document_name == "My Document"
        assert info.state == DirtyState.CLEAN
        assert info.is_dirty is False

    def test_is_dirty_property(self):
        """Test is_dirty property."""
        info = DirtyInfo(
            document_id="doc1",
            document_name="Test",
            state=DirtyState.CLEAN,
        )
        assert info.is_dirty is False

        info.state = DirtyState.DIRTY
        assert info.is_dirty is True

    def test_unsaved_duration(self):
        """Test unsaved duration calculation."""
        info = DirtyInfo(
            document_id="doc1",
            document_name="Test",
            state=DirtyState.DIRTY,
            dirty_since=time.time() - 10,
        )

        assert info.unsaved_duration >= 10

    def test_time_since_save(self):
        """Test time since save calculation."""
        info = DirtyInfo(
            document_id="doc1",
            document_name="Test",
            state=DirtyState.CLEAN,
            last_saved=time.time() - 60,
        )

        assert info.time_since_save >= 60

        # No last_saved
        info2 = DirtyInfo(
            document_id="doc2",
            document_name="Test",
            state=DirtyState.CLEAN,
        )
        assert info2.time_since_save is None


class TestDirtyTracker:
    """Tests for DirtyTracker."""

    def setup_method(self):
        """Create fresh tracker for each test."""
        self.tracker = DirtyTracker(auto_subscribe=False)

    def test_tracker_initialization(self):
        """Test DirtyTracker initializes correctly."""
        assert self.tracker is not None

    def test_track_object(self):
        """Test tracking an object."""
        class TestDoc:
            pass

        doc = TestDoc()
        info = self.tracker.track(doc, "doc1", "My Document")

        assert info is not None
        assert info.document_id == "doc1"
        assert info.state == DirtyState.CLEAN

    def test_untrack_object(self):
        """Test untracking an object."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.track(doc, "doc1")

        result = self.tracker.untrack(doc)

        assert result is True
        assert self.tracker.get_info(doc) is None

    def test_untrack_not_tracked(self):
        """Test untracking object that wasn't tracked."""
        class TestDoc:
            pass

        doc = TestDoc()
        result = self.tracker.untrack(doc)

        assert result is False

    def test_mark_dirty(self):
        """Test marking object as dirty."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.track(doc, "doc1")

        self.tracker.mark_dirty(doc, "field1")

        info = self.tracker.get_info(doc)
        assert info.is_dirty is True
        assert "field1" in info.dirty_fields

    def test_mark_clean(self):
        """Test marking object as clean."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.track(doc, "doc1")
        self.tracker.mark_dirty(doc)

        self.tracker.mark_clean(doc)

        info = self.tracker.get_info(doc)
        assert info.is_dirty is False
        assert info.last_saved is not None

    def test_is_dirty(self):
        """Test is_dirty check."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.track(doc, "doc1")

        assert self.tracker.is_dirty(doc) is False

        self.tracker.mark_dirty(doc)
        assert self.tracker.is_dirty(doc) is True

    def test_get_all_dirty(self):
        """Test getting all dirty objects."""
        class TestDoc:
            pass

        doc1 = TestDoc()
        doc2 = TestDoc()
        doc3 = TestDoc()

        self.tracker.track(doc1, "doc1")
        self.tracker.track(doc2, "doc2")
        self.tracker.track(doc3, "doc3")

        self.tracker.mark_dirty(doc1)
        self.tracker.mark_dirty(doc3)

        dirty = self.tracker.get_all_dirty()

        assert len(dirty) == 2
        docs = [d[0] for d in dirty]
        assert doc1 in docs
        assert doc3 in docs
        assert doc2 not in docs

    def test_any_dirty(self):
        """Test any_dirty check."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.track(doc, "doc1")

        assert self.tracker.any_dirty() is False

        self.tracker.mark_dirty(doc)
        assert self.tracker.any_dirty() is True

    def test_dirty_callback(self):
        """Test on_dirty callback."""
        class TestDoc:
            pass

        doc = TestDoc()
        dirty_events = []

        self.tracker.on_dirty(lambda o, i: dirty_events.append((o, i)))
        self.tracker.track(doc, "doc1")

        self.tracker.mark_dirty(doc)

        assert len(dirty_events) == 1
        assert dirty_events[0][0] is doc

    def test_clean_callback(self):
        """Test on_clean callback."""
        class TestDoc:
            pass

        doc = TestDoc()
        clean_events = []

        self.tracker.on_clean(lambda o, i: clean_events.append((o, i)))
        self.tracker.track(doc, "doc1")
        self.tracker.mark_dirty(doc)

        self.tracker.mark_clean(doc)

        assert len(clean_events) == 1
        assert clean_events[0][0] is doc

    def test_change_count(self):
        """Test change count tracking."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.track(doc, "doc1")

        self.tracker.mark_dirty(doc, "field1")
        self.tracker.mark_dirty(doc, "field2")
        self.tracker.mark_dirty(doc, "field3")

        info = self.tracker.get_info(doc)
        assert info.change_count == 3


class TestDocumentDirtyTracker:
    """Tests for DocumentDirtyTracker."""

    def setup_method(self):
        """Create fresh tracker for each test."""
        self.prompt_results = []
        self.tracker = DocumentDirtyTracker(
            prompt_callback=lambda name: self._prompt(name),
        )

    def _prompt(self, name):
        """Mock prompt callback."""
        if self.prompt_results:
            return self.prompt_results.pop(0)
        return SavePromptResult.DONT_SAVE

    def test_tracker_initialization(self):
        """Test DocumentDirtyTracker initializes correctly."""
        assert self.tracker.dirty_count == 0

    def test_register_document(self):
        """Test registering a document."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.register_document(doc, "doc1", "My Document")

        assert self.tracker.is_dirty("doc1") is False

    def test_unregister_document(self):
        """Test unregistering a document."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.register_document(doc, "doc1")
        self.tracker.unregister_document("doc1")

        assert self.tracker.is_dirty("doc1") is False

    def test_mark_saved(self):
        """Test marking document as saved."""
        class TestDoc:
            x = 10

        doc = TestDoc()
        self.tracker.register_document(doc, "doc1")

        # Make it dirty first
        doc.x = 20  # Simulated change

        self.tracker.mark_saved("doc1")
        assert self.tracker.is_dirty("doc1") is False

    def test_get_dirty_documents(self):
        """Test getting dirty document IDs."""
        class TestDoc:
            pass

        doc1 = TestDoc()
        doc2 = TestDoc()

        self.tracker.register_document(doc1, "doc1")
        self.tracker.register_document(doc2, "doc2")

        # Make doc1 dirty through internal tracker
        self.tracker._tracker.mark_dirty(doc1, "field")

        dirty = self.tracker.get_dirty_documents()

        assert "doc1" in dirty
        assert "doc2" not in dirty

    def test_can_close_clean_document(self):
        """Test can_close for clean document."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.register_document(doc, "doc1")

        can_close, result = self.tracker.can_close("doc1")

        assert can_close is True
        assert result is None

    def test_can_close_dirty_save(self):
        """Test can_close with save prompt."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.register_document(doc, "doc1", "My Document")
        self.tracker._tracker.mark_dirty(doc, "field")

        self.prompt_results = [SavePromptResult.SAVE]

        can_close, result = self.tracker.can_close("doc1")

        assert can_close is True
        assert result == SavePromptResult.SAVE

    def test_can_close_dirty_cancel(self):
        """Test can_close with cancel."""
        class TestDoc:
            pass

        doc = TestDoc()
        self.tracker.register_document(doc, "doc1", "My Document")
        self.tracker._tracker.mark_dirty(doc, "field")

        self.prompt_results = [SavePromptResult.CANCEL]

        can_close, result = self.tracker.can_close("doc1")

        assert can_close is False
        assert result == SavePromptResult.CANCEL

    def test_can_close_all(self):
        """Test can_close_all for multiple documents."""
        class TestDoc:
            pass

        doc1 = TestDoc()
        doc2 = TestDoc()

        self.tracker.register_document(doc1, "doc1", "Doc 1")
        self.tracker.register_document(doc2, "doc2", "Doc 2")

        self.tracker._tracker.mark_dirty(doc1, "field")
        self.tracker._tracker.mark_dirty(doc2, "field")

        self.prompt_results = [
            SavePromptResult.SAVE,
            SavePromptResult.DONT_SAVE,
        ]

        can_close, results = self.tracker.can_close_all()

        assert can_close is True
        assert len(results) == 2

    def test_can_close_all_cancelled(self):
        """Test can_close_all when cancelled."""
        class TestDoc:
            pass

        doc1 = TestDoc()
        doc2 = TestDoc()

        self.tracker.register_document(doc1, "doc1", "Doc 1")
        self.tracker.register_document(doc2, "doc2", "Doc 2")

        self.tracker._tracker.mark_dirty(doc1, "field")
        self.tracker._tracker.mark_dirty(doc2, "field")

        self.prompt_results = [
            SavePromptResult.CANCEL,
        ]

        can_close, results = self.tracker.can_close_all()

        assert can_close is False


class TestDirtyTrackerAutoSubscribe:
    """Tests for automatic Foundation Tracker subscription."""

    def test_auto_subscribe(self):
        """Test automatic subscription to Foundation Tracker."""
        # Create tracker with auto_subscribe=True
        # This would subscribe to Foundation's tracker
        tracker = DirtyTracker(auto_subscribe=True)

        # Just verify it doesn't error
        assert tracker is not None
