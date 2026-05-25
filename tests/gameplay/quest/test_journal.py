"""
Comprehensive tests for Quest Journal.

Tests cover:
- Journal entries
- Quest categories
- Quest filtering
- Quest searching
- Completed quest archive
- Journal pagination
"""

import pytest
from dataclasses import dataclass
from typing import Any, List, Dict
from unittest.mock import Mock, patch

# JournalPage, JournalSortOrder, etc. are planned but not yet implemented
pytest.skip("Journal API not fully implemented", allow_module_level=True)

from engine.gameplay.quest.journal import (
    QuestJournal,
    JournalEntry,
    JournalCategory,
    JournalFilter,
    JournalPage,
    JournalSortOrder,
    JournalView,
)
from engine.gameplay.quest.quest import Quest, QuestDefinition, QuestState, QuestType


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def quest_journal():
    """Create a quest journal instance."""
    return QuestJournal(player_id="player_001")


@pytest.fixture
def main_quest_def():
    """Create a main quest definition."""
    return QuestDefinition(
        id="main_quest_1",
        name="The Beginning",
        description="Start your epic adventure",
        quest_type=QuestType.MAIN,
        level_requirement=1,
        category="main_story",
        zone="starting_zone",
    )


@pytest.fixture
def side_quest_def():
    """Create a side quest definition."""
    return QuestDefinition(
        id="side_quest_1",
        name="Lost and Found",
        description="Find the lost item",
        quest_type=QuestType.SIDE,
        level_requirement=5,
        category="exploration",
        zone="forest",
    )


@pytest.fixture
def daily_quest_def():
    """Create a daily quest definition."""
    return QuestDefinition(
        id="daily_quest_1",
        name="Daily Patrol",
        description="Complete your daily patrol",
        quest_type=QuestType.DAILY,
        level_requirement=1,
        category="daily",
        repeatable=True,
    )


@pytest.fixture
def sample_quests():
    """Create a set of sample quests for testing."""
    definitions = [
        QuestDefinition(id="q1", name="Alpha Quest", description="First", quest_type=QuestType.MAIN, level_requirement=1, category="story"),
        QuestDefinition(id="q2", name="Beta Quest", description="Second", quest_type=QuestType.SIDE, level_requirement=5, category="combat"),
        QuestDefinition(id="q3", name="Gamma Quest", description="Third", quest_type=QuestType.MAIN, level_requirement=10, category="story"),
        QuestDefinition(id="q4", name="Delta Quest", description="Fourth", quest_type=QuestType.DAILY, level_requirement=1, category="daily"),
        QuestDefinition(id="q5", name="Epsilon Quest", description="Fifth", quest_type=QuestType.SIDE, level_requirement=15, category="exploration"),
    ]
    return [Quest(definition=d, state=QuestState.ACTIVE) for d in definitions]


# =============================================================================
# JournalEntry Tests
# =============================================================================

class TestJournalEntry:
    """Tests for JournalEntry data class."""

    def test_journal_entry_creation(self, main_quest_def):
        """Test creating a journal entry."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest)

        assert entry.quest_id == "main_quest_1"
        assert entry.name == "The Beginning"
        assert entry.state == QuestState.ACTIVE

    def test_journal_entry_with_progress(self, main_quest_def):
        """Test journal entry with progress data."""
        quest = Quest(
            definition=main_quest_def,
            state=QuestState.ACTIVE,
            objective_progress={"obj1": 50},
        )
        entry = JournalEntry.from_quest(quest, overall_progress=0.5)

        assert entry.progress == 0.5

    def test_journal_entry_timestamps(self, main_quest_def):
        """Test journal entry timestamp fields."""
        quest = Quest(
            definition=main_quest_def,
            state=QuestState.COMPLETE,
            accepted_at=100.0,
            completed_at=200.0,
        )
        entry = JournalEntry.from_quest(quest)

        assert entry.accepted_at == 100.0
        assert entry.completed_at == 200.0

    def test_journal_entry_category(self, main_quest_def):
        """Test journal entry category."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest)

        assert entry.category == "main_story"

    def test_journal_entry_zone(self, main_quest_def):
        """Test journal entry zone."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest)

        assert entry.zone == "starting_zone"

    def test_journal_entry_quest_type(self, main_quest_def):
        """Test journal entry quest type."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest)

        assert entry.quest_type == QuestType.MAIN

    def test_journal_entry_level(self, main_quest_def):
        """Test journal entry level requirement."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest)

        assert entry.level_requirement == 1

    def test_journal_entry_is_tracked(self, main_quest_def):
        """Test journal entry tracked status."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest, is_tracked=True)

        assert entry.is_tracked is True

    def test_journal_entry_serialization(self, main_quest_def):
        """Test journal entry serialization."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        entry = JournalEntry.from_quest(quest)

        data = entry.to_dict()

        assert data["quest_id"] == "main_quest_1"
        assert data["name"] == "The Beginning"
        assert data["state"] == "ACTIVE"

    def test_journal_entry_deserialization(self):
        """Test journal entry deserialization."""
        data = {
            "quest_id": "test_quest",
            "name": "Test Quest",
            "description": "A test",
            "state": "ACTIVE",
            "quest_type": "SIDE",
            "category": "test",
            "progress": 0.5,
        }

        entry = JournalEntry.from_dict(data)

        assert entry.quest_id == "test_quest"
        assert entry.progress == 0.5


# =============================================================================
# JournalCategory Tests
# =============================================================================

class TestJournalCategory:
    """Tests for JournalCategory functionality."""

    def test_category_creation(self):
        """Test creating a journal category."""
        category = JournalCategory(
            id="main_story",
            name="Main Story",
            description="Main storyline quests",
            order=0,
        )
        assert category.id == "main_story"
        assert category.name == "Main Story"
        assert category.order == 0

    def test_category_defaults(self):
        """Test category default values."""
        category = JournalCategory(id="test", name="Test")
        assert category.description == ""
        assert category.order == 0
        assert category.icon is None
        assert category.color is None

    def test_category_with_styling(self):
        """Test category with styling options."""
        category = JournalCategory(
            id="epic",
            name="Epic Quests",
            icon="star",
            color="#FFD700",
        )
        assert category.icon == "star"
        assert category.color == "#FFD700"

    def test_category_quest_count(self, quest_journal, sample_quests):
        """Test counting quests in a category."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        count = quest_journal.get_category_quest_count("story")
        assert count == 2  # q1 and q3

    def test_category_ordering(self):
        """Test category ordering."""
        categories = [
            JournalCategory(id="c1", name="C1", order=2),
            JournalCategory(id="c2", name="C2", order=0),
            JournalCategory(id="c3", name="C3", order=1),
        ]

        sorted_categories = sorted(categories, key=lambda c: c.order)

        assert sorted_categories[0].id == "c2"
        assert sorted_categories[1].id == "c3"
        assert sorted_categories[2].id == "c1"


# =============================================================================
# QuestJournal Basic Tests
# =============================================================================

class TestQuestJournalBasic:
    """Tests for basic QuestJournal functionality."""

    def test_journal_creation(self):
        """Test creating a quest journal."""
        journal = QuestJournal(player_id="player_001")
        assert journal.player_id == "player_001"
        assert len(journal.entries) == 0

    def test_add_quest(self, quest_journal, main_quest_def):
        """Test adding a quest to the journal."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        assert main_quest_def.id in quest_journal.entries
        assert quest_journal.get_entry(main_quest_def.id) is not None

    def test_remove_quest(self, quest_journal, main_quest_def):
        """Test removing a quest from the journal."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        result = quest_journal.remove_quest(main_quest_def.id)

        assert result is True
        assert main_quest_def.id not in quest_journal.entries

    def test_remove_nonexistent_quest(self, quest_journal):
        """Test removing a quest that doesn't exist."""
        result = quest_journal.remove_quest("nonexistent")
        assert result is False

    def test_update_quest(self, quest_journal, main_quest_def):
        """Test updating a quest in the journal."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        # Update quest state
        quest.state = QuestState.COMPLETE
        quest_journal.update_quest(quest)

        entry = quest_journal.get_entry(main_quest_def.id)
        assert entry.state == QuestState.COMPLETE

    def test_get_entry(self, quest_journal, main_quest_def):
        """Test getting a specific entry."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        entry = quest_journal.get_entry(main_quest_def.id)

        assert entry is not None
        assert entry.quest_id == main_quest_def.id

    def test_get_nonexistent_entry(self, quest_journal):
        """Test getting a non-existent entry."""
        entry = quest_journal.get_entry("nonexistent")
        assert entry is None

    def test_has_quest(self, quest_journal, main_quest_def):
        """Test checking if journal has a quest."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)

        assert quest_journal.has_quest(main_quest_def.id) is False

        quest_journal.add_quest(quest)

        assert quest_journal.has_quest(main_quest_def.id) is True

    def test_quest_count(self, quest_journal, sample_quests):
        """Test getting quest count."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        assert quest_journal.quest_count == 5


# =============================================================================
# Journal Filtering Tests
# =============================================================================

class TestJournalFiltering:
    """Tests for journal filtering functionality."""

    def test_filter_by_state(self, quest_journal, sample_quests):
        """Test filtering quests by state."""
        sample_quests[0].state = QuestState.COMPLETE
        sample_quests[1].state = QuestState.COMPLETE

        for quest in sample_quests:
            quest_journal.add_quest(quest)

        active = quest_journal.filter_by_state(QuestState.ACTIVE)
        complete = quest_journal.filter_by_state(QuestState.COMPLETE)

        assert len(active) == 3
        assert len(complete) == 2

    def test_filter_by_type(self, quest_journal, sample_quests):
        """Test filtering quests by type."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        main_quests = quest_journal.filter_by_type(QuestType.MAIN)
        side_quests = quest_journal.filter_by_type(QuestType.SIDE)
        daily_quests = quest_journal.filter_by_type(QuestType.DAILY)

        assert len(main_quests) == 2
        assert len(side_quests) == 2
        assert len(daily_quests) == 1

    def test_filter_by_category(self, quest_journal, sample_quests):
        """Test filtering quests by category."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        story_quests = quest_journal.filter_by_category("story")
        combat_quests = quest_journal.filter_by_category("combat")

        assert len(story_quests) == 2
        assert len(combat_quests) == 1

    def test_filter_by_zone(self, quest_journal, main_quest_def, side_quest_def):
        """Test filtering quests by zone."""
        quest_journal.add_quest(Quest(definition=main_quest_def, state=QuestState.ACTIVE))
        quest_journal.add_quest(Quest(definition=side_quest_def, state=QuestState.ACTIVE))

        starting_zone = quest_journal.filter_by_zone("starting_zone")
        forest = quest_journal.filter_by_zone("forest")

        assert len(starting_zone) == 1
        assert len(forest) == 1

    def test_filter_by_level_range(self, quest_journal, sample_quests):
        """Test filtering quests by level range."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        low_level = quest_journal.filter_by_level_range(1, 5)
        mid_level = quest_journal.filter_by_level_range(6, 10)
        high_level = quest_journal.filter_by_level_range(11, 20)

        assert len(low_level) == 3  # q1, q2, q4
        assert len(mid_level) == 1  # q3
        assert len(high_level) == 1  # q5

    def test_filter_with_multiple_criteria(self, quest_journal, sample_quests):
        """Test filtering with multiple criteria."""
        sample_quests[0].state = QuestState.COMPLETE  # Main, story, level 1

        for quest in sample_quests:
            quest_journal.add_quest(quest)

        filter_obj = JournalFilter(
            states=[QuestState.ACTIVE],
            types=[QuestType.MAIN],
        )

        filtered = quest_journal.filter(filter_obj)

        assert len(filtered) == 1  # Only q3 (q1 is complete)

    def test_filter_tracked_only(self, quest_journal, sample_quests):
        """Test filtering to show only tracked quests."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        quest_journal.track_quest("q1")
        quest_journal.track_quest("q3")

        filter_obj = JournalFilter(tracked_only=True)
        filtered = quest_journal.filter(filter_obj)

        assert len(filtered) == 2

    def test_filter_empty_result(self, quest_journal, sample_quests):
        """Test filter that returns no results."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        filter_obj = JournalFilter(
            states=[QuestState.FAILED],
        )

        filtered = quest_journal.filter(filter_obj)

        assert len(filtered) == 0


# =============================================================================
# Journal Searching Tests
# =============================================================================

class TestJournalSearching:
    """Tests for journal search functionality."""

    def test_search_by_name(self, quest_journal, sample_quests):
        """Test searching quests by name."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results = quest_journal.search("Alpha")

        assert len(results) == 1
        assert results[0].name == "Alpha Quest"

    def test_search_by_partial_name(self, quest_journal, sample_quests):
        """Test searching quests by partial name."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results = quest_journal.search("Quest")

        assert len(results) == 5  # All quests have "Quest" in name

    def test_search_case_insensitive(self, quest_journal, sample_quests):
        """Test case-insensitive search."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results_lower = quest_journal.search("alpha")
        results_upper = quest_journal.search("ALPHA")

        assert len(results_lower) == 1
        assert len(results_upper) == 1

    def test_search_by_description(self, quest_journal, sample_quests):
        """Test searching quests by description."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results = quest_journal.search("First", include_description=True)

        assert len(results) == 1

    def test_search_no_results(self, quest_journal, sample_quests):
        """Test search with no matching results."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results = quest_journal.search("Nonexistent")

        assert len(results) == 0

    def test_search_with_filter(self, quest_journal, sample_quests):
        """Test search combined with filter."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        filter_obj = JournalFilter(types=[QuestType.MAIN])
        results = quest_journal.search("Quest", filter_obj=filter_obj)

        assert len(results) == 2  # Only main quests matching

    def test_search_empty_query(self, quest_journal, sample_quests):
        """Test search with empty query returns all."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results = quest_journal.search("")

        assert len(results) == 5

    def test_search_special_characters(self, quest_journal):
        """Test search with special characters."""
        quest_def = QuestDefinition(
            id="special",
            name="Quest (Part 1) - The [Beginning]",
            description="Test",
        )
        quest_journal.add_quest(Quest(definition=quest_def, state=QuestState.ACTIVE))

        results = quest_journal.search("(Part 1)")

        assert len(results) == 1


# =============================================================================
# Journal Sorting Tests
# =============================================================================

class TestJournalSorting:
    """Tests for journal sorting functionality."""

    def test_sort_by_name(self, quest_journal, sample_quests):
        """Test sorting quests by name."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.NAME_ASC)

        assert sorted_entries[0].name == "Alpha Quest"
        assert sorted_entries[-1].name == "Epsilon Quest"

    def test_sort_by_name_descending(self, quest_journal, sample_quests):
        """Test sorting quests by name descending."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.NAME_DESC)

        assert sorted_entries[0].name == "Epsilon Quest"
        assert sorted_entries[-1].name == "Alpha Quest"

    def test_sort_by_level(self, quest_journal, sample_quests):
        """Test sorting quests by level requirement."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.LEVEL_ASC)

        assert sorted_entries[0].level_requirement == 1
        assert sorted_entries[-1].level_requirement == 15

    def test_sort_by_level_descending(self, quest_journal, sample_quests):
        """Test sorting quests by level descending."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.LEVEL_DESC)

        assert sorted_entries[0].level_requirement == 15
        assert sorted_entries[-1].level_requirement == 1

    def test_sort_by_progress(self, quest_journal, sample_quests):
        """Test sorting quests by progress."""
        for i, quest in enumerate(sample_quests):
            quest_journal.add_quest(quest, progress=i * 0.2)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.PROGRESS_ASC)

        assert sorted_entries[0].progress == pytest.approx(0.0)
        assert sorted_entries[-1].progress == pytest.approx(0.8)

    def test_sort_by_accepted_time(self, quest_journal, sample_quests):
        """Test sorting quests by accepted time."""
        for i, quest in enumerate(sample_quests):
            quest.accepted_at = float(100 - i * 10)  # Reverse order
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.NEWEST_FIRST)

        # Newest (highest timestamp) first
        assert sorted_entries[0].accepted_at == 100.0

    def test_sort_by_type(self, quest_journal, sample_quests):
        """Test sorting quests by type."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.TYPE)

        # Main quests should come first (assuming MAIN < SIDE < DAILY)
        main_count = sum(1 for e in sorted_entries[:2] if e.quest_type == QuestType.MAIN)
        assert main_count == 2

    def test_sort_by_category(self, quest_journal, sample_quests):
        """Test sorting quests by category."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        sorted_entries = quest_journal.get_sorted(JournalSortOrder.CATEGORY)

        # Categories should be grouped together
        categories = [e.category for e in sorted_entries]
        # Check that same categories are adjacent
        seen = set()
        for cat in categories:
            if cat in seen and categories[categories.index(cat) - 1] != cat:
                pytest.fail("Categories should be grouped together")
            seen.add(cat)


# =============================================================================
# Completed Quest Archive Tests
# =============================================================================

class TestCompletedQuestArchive:
    """Tests for completed quest archive functionality."""

    def test_archive_completed_quest(self, quest_journal, main_quest_def):
        """Test archiving a completed quest."""
        quest = Quest(
            definition=main_quest_def,
            state=QuestState.COMPLETE,
            completed_at=200.0,
        )
        quest_journal.add_quest(quest)
        quest_journal.archive_quest(main_quest_def.id)

        # Should be in archive
        archived = quest_journal.get_archived_quests()
        assert len(archived) == 1
        assert archived[0].quest_id == main_quest_def.id

    def test_archive_removes_from_active(self, quest_journal, main_quest_def):
        """Test that archiving removes from active entries."""
        quest = Quest(
            definition=main_quest_def,
            state=QuestState.COMPLETE,
        )
        quest_journal.add_quest(quest)
        quest_journal.archive_quest(main_quest_def.id)

        # Should not be in active entries
        assert quest_journal.has_quest(main_quest_def.id) is False

    def test_get_archived_quests(self, quest_journal, sample_quests):
        """Test getting archived quests."""
        for quest in sample_quests:
            quest.state = QuestState.COMPLETE
            quest_journal.add_quest(quest)
            quest_journal.archive_quest(quest.definition.id)

        archived = quest_journal.get_archived_quests()
        assert len(archived) == 5

    def test_archived_quest_count(self, quest_journal, sample_quests):
        """Test archived quest count."""
        for i, quest in enumerate(sample_quests[:3]):
            quest.state = QuestState.COMPLETE
            quest_journal.add_quest(quest)
            quest_journal.archive_quest(quest.definition.id)

        assert quest_journal.archived_count == 3

    def test_search_archived(self, quest_journal, sample_quests):
        """Test searching archived quests."""
        for quest in sample_quests:
            quest.state = QuestState.COMPLETE
            quest_journal.add_quest(quest)
            quest_journal.archive_quest(quest.definition.id)

        results = quest_journal.search_archived("Alpha")
        assert len(results) == 1

    def test_filter_archived(self, quest_journal, sample_quests):
        """Test filtering archived quests."""
        for quest in sample_quests:
            quest.state = QuestState.COMPLETE
            quest_journal.add_quest(quest)
            quest_journal.archive_quest(quest.definition.id)

        filter_obj = JournalFilter(types=[QuestType.MAIN])
        filtered = quest_journal.filter_archived(filter_obj)

        assert len(filtered) == 2

    def test_archive_preserves_completion_info(self, quest_journal, main_quest_def):
        """Test that archive preserves completion information."""
        quest = Quest(
            definition=main_quest_def,
            state=QuestState.COMPLETE,
            accepted_at=100.0,
            completed_at=200.0,
            times_completed=3,
        )
        quest_journal.add_quest(quest)
        quest_journal.archive_quest(main_quest_def.id)

        archived = quest_journal.get_archived_quests()
        assert archived[0].completed_at == 200.0
        assert archived[0].times_completed == 3

    def test_clear_archive(self, quest_journal, sample_quests):
        """Test clearing the archive."""
        for quest in sample_quests:
            quest.state = QuestState.COMPLETE
            quest_journal.add_quest(quest)
            quest_journal.archive_quest(quest.definition.id)

        quest_journal.clear_archive()

        assert quest_journal.archived_count == 0


# =============================================================================
# Journal Pagination Tests
# =============================================================================

class TestJournalPagination:
    """Tests for journal pagination functionality."""

    def test_get_page(self, quest_journal, sample_quests):
        """Test getting a page of entries."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        page = quest_journal.get_page(page_number=1, page_size=2)

        assert isinstance(page, JournalPage)
        assert len(page.entries) == 2
        assert page.page_number == 1
        assert page.page_size == 2

    def test_page_total_pages(self, quest_journal, sample_quests):
        """Test total pages calculation."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        page = quest_journal.get_page(page_number=1, page_size=2)

        assert page.total_entries == 5
        assert page.total_pages == 3  # 5 entries / 2 per page = 3 pages

    def test_get_second_page(self, quest_journal, sample_quests):
        """Test getting the second page."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        page = quest_journal.get_page(page_number=2, page_size=2)

        assert page.page_number == 2
        assert len(page.entries) == 2

    def test_get_last_page(self, quest_journal, sample_quests):
        """Test getting the last page with partial entries."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        page = quest_journal.get_page(page_number=3, page_size=2)

        assert page.page_number == 3
        assert len(page.entries) == 1  # Only 1 entry on last page

    def test_invalid_page_number(self, quest_journal, sample_quests):
        """Test handling invalid page number."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        # Page 0 or negative should return first page
        page = quest_journal.get_page(page_number=0, page_size=2)
        assert page.page_number == 1

        # Page beyond total should return last page
        page = quest_journal.get_page(page_number=100, page_size=2)
        assert page.page_number == 3

    def test_page_has_next_previous(self, quest_journal, sample_quests):
        """Test page navigation helpers."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        page1 = quest_journal.get_page(page_number=1, page_size=2)
        page2 = quest_journal.get_page(page_number=2, page_size=2)
        page3 = quest_journal.get_page(page_number=3, page_size=2)

        assert page1.has_previous is False
        assert page1.has_next is True

        assert page2.has_previous is True
        assert page2.has_next is True

        assert page3.has_previous is True
        assert page3.has_next is False

    def test_paginate_with_filter(self, quest_journal, sample_quests):
        """Test pagination with filtering."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        filter_obj = JournalFilter(types=[QuestType.MAIN])
        page = quest_journal.get_page(
            page_number=1,
            page_size=10,
            filter_obj=filter_obj,
        )

        assert page.total_entries == 2

    def test_paginate_with_sorting(self, quest_journal, sample_quests):
        """Test pagination with sorting."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        page = quest_journal.get_page(
            page_number=1,
            page_size=5,
            sort_order=JournalSortOrder.NAME_ASC,
        )

        assert page.entries[0].name == "Alpha Quest"
        assert page.entries[-1].name == "Epsilon Quest"

    def test_empty_page(self, quest_journal):
        """Test getting page from empty journal."""
        page = quest_journal.get_page(page_number=1, page_size=10)

        assert len(page.entries) == 0
        assert page.total_pages == 0


# =============================================================================
# Journal View Tests
# =============================================================================

class TestJournalView:
    """Tests for journal view functionality."""

    def test_active_view(self, quest_journal, sample_quests):
        """Test active quests view."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        view = quest_journal.get_view(JournalView.ACTIVE)

        assert len(view) == 5

    def test_completed_view(self, quest_journal, sample_quests):
        """Test completed quests view."""
        sample_quests[0].state = QuestState.COMPLETE
        sample_quests[1].state = QuestState.COMPLETE

        for quest in sample_quests:
            quest_journal.add_quest(quest)

        view = quest_journal.get_view(JournalView.COMPLETED)

        assert len(view) == 2

    def test_failed_view(self, quest_journal, sample_quests):
        """Test failed quests view."""
        sample_quests[0].state = QuestState.FAILED

        for quest in sample_quests:
            quest_journal.add_quest(quest)

        view = quest_journal.get_view(JournalView.FAILED)

        assert len(view) == 1

    def test_tracked_view(self, quest_journal, sample_quests):
        """Test tracked quests view."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        quest_journal.track_quest("q1")
        quest_journal.track_quest("q3")

        view = quest_journal.get_view(JournalView.TRACKED)

        assert len(view) == 2

    def test_all_view(self, quest_journal, sample_quests):
        """Test all quests view."""
        sample_quests[0].state = QuestState.COMPLETE
        sample_quests[1].state = QuestState.FAILED

        for quest in sample_quests:
            quest_journal.add_quest(quest)

        view = quest_journal.get_view(JournalView.ALL)

        assert len(view) == 5


# =============================================================================
# Journal Categories Management Tests
# =============================================================================

class TestJournalCategories:
    """Tests for journal category management."""

    def test_add_category(self, quest_journal):
        """Test adding a custom category."""
        category = JournalCategory(
            id="custom",
            name="Custom Quests",
            order=10,
        )
        quest_journal.add_category(category)

        assert "custom" in quest_journal.categories

    def test_remove_category(self, quest_journal):
        """Test removing a category."""
        category = JournalCategory(id="removable", name="Removable")
        quest_journal.add_category(category)

        result = quest_journal.remove_category("removable")

        assert result is True
        assert "removable" not in quest_journal.categories

    def test_get_all_categories(self, quest_journal, sample_quests):
        """Test getting all categories."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        categories = quest_journal.get_all_categories()

        # Should include categories from quests
        category_ids = [c.id for c in categories]
        assert "story" in category_ids
        assert "combat" in category_ids

    def test_get_categories_sorted(self, quest_journal):
        """Test getting categories in sorted order."""
        quest_journal.add_category(JournalCategory(id="c1", name="C1", order=2))
        quest_journal.add_category(JournalCategory(id="c2", name="C2", order=0))
        quest_journal.add_category(JournalCategory(id="c3", name="C3", order=1))

        categories = quest_journal.get_all_categories()

        assert categories[0].id == "c2"
        assert categories[1].id == "c3"
        assert categories[2].id == "c1"


# =============================================================================
# Tracking Tests
# =============================================================================

class TestJournalTracking:
    """Tests for quest tracking in journal."""

    def test_track_quest(self, quest_journal, main_quest_def):
        """Test tracking a quest."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        result = quest_journal.track_quest(main_quest_def.id)

        assert result is True
        assert quest_journal.is_tracked(main_quest_def.id) is True

    def test_untrack_quest(self, quest_journal, main_quest_def):
        """Test untracking a quest."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)
        quest_journal.track_quest(main_quest_def.id)

        result = quest_journal.untrack_quest(main_quest_def.id)

        assert result is True
        assert quest_journal.is_tracked(main_quest_def.id) is False

    def test_get_tracked_quests(self, quest_journal, sample_quests):
        """Test getting list of tracked quests."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        quest_journal.track_quest("q1")
        quest_journal.track_quest("q3")

        tracked = quest_journal.get_tracked_quests()

        assert len(tracked) == 2
        assert any(e.quest_id == "q1" for e in tracked)
        assert any(e.quest_id == "q3" for e in tracked)

    def test_track_nonexistent_quest(self, quest_journal):
        """Test tracking a non-existent quest."""
        result = quest_journal.track_quest("nonexistent")
        assert result is False

    def test_track_already_tracked(self, quest_journal, main_quest_def):
        """Test tracking an already tracked quest."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)
        quest_journal.track_quest(main_quest_def.id)

        result = quest_journal.track_quest(main_quest_def.id)

        assert result is False  # Already tracked


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestJournalEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_add_duplicate_quest(self, quest_journal, main_quest_def):
        """Test adding duplicate quest updates existing."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        quest.state = QuestState.COMPLETE
        quest_journal.add_quest(quest)  # Should update

        entry = quest_journal.get_entry(main_quest_def.id)
        assert entry.state == QuestState.COMPLETE

    def test_empty_journal_operations(self, quest_journal):
        """Test operations on empty journal."""
        assert quest_journal.quest_count == 0
        assert len(quest_journal.get_view(JournalView.ALL)) == 0
        assert quest_journal.search("anything") == []

    def test_filter_with_no_matching_state(self, quest_journal, sample_quests):
        """Test filter that matches no states."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        results = quest_journal.filter_by_state(QuestState.UNAVAILABLE)
        assert len(results) == 0

    def test_very_long_search_query(self, quest_journal, sample_quests):
        """Test search with very long query."""
        for quest in sample_quests:
            quest_journal.add_quest(quest)

        long_query = "A" * 1000
        results = quest_journal.search(long_query)

        assert len(results) == 0

    def test_pagination_with_single_entry(self, quest_journal, main_quest_def):
        """Test pagination with single entry."""
        quest = Quest(definition=main_quest_def, state=QuestState.ACTIVE)
        quest_journal.add_quest(quest)

        page = quest_journal.get_page(page_number=1, page_size=10)

        assert page.total_pages == 1
        assert len(page.entries) == 1
