"""
AssetSearch - Advanced search with filters and saved searches.

Provides comprehensive asset search capabilities:
- Full-text search
- Field-specific search
- Filter-based search
- Query parsing
- Saved searches
- Search history
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol, Union

from trinity.decorators.dev import editor


class SearchOperator(Enum):
    """Search query operators."""

    EQUALS = auto()
    NOT_EQUALS = auto()
    CONTAINS = auto()
    NOT_CONTAINS = auto()
    STARTS_WITH = auto()
    ENDS_WITH = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    GREATER_EQUAL = auto()
    LESS_EQUAL = auto()
    IN = auto()
    NOT_IN = auto()
    MATCHES = auto()  # Regex
    EXISTS = auto()
    NOT_EXISTS = auto()


class SearchFieldType(Enum):
    """Types of searchable fields."""

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    DATE = auto()
    PATH = auto()
    TAG = auto()
    SIZE = auto()


@dataclass
class SearchFilter:
    """A search filter condition.

    Attributes:
        field: Field to filter on
        operator: Filter operator
        value: Filter value
        field_type: Type of the field
        case_sensitive: Whether string comparison is case-sensitive
    """

    field: str
    operator: SearchOperator
    value: Any
    field_type: SearchFieldType = SearchFieldType.STRING
    case_sensitive: bool = False

    def matches(self, item: dict[str, Any]) -> bool:
        """Check if an item matches this filter.

        Args:
            item: Item to check (dict with field values)

        Returns:
            True if matches
        """
        field_value = self._get_nested_value(item, self.field)

        # Handle EXISTS/NOT_EXISTS
        if self.operator == SearchOperator.EXISTS:
            return field_value is not None
        if self.operator == SearchOperator.NOT_EXISTS:
            return field_value is None

        if field_value is None:
            return False

        # Normalize strings for case-insensitive comparison
        compare_value = self.value
        if isinstance(field_value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            if isinstance(compare_value, str):
                compare_value = compare_value.lower()

        # Apply operator
        if self.operator == SearchOperator.EQUALS:
            return field_value == compare_value
        elif self.operator == SearchOperator.NOT_EQUALS:
            return field_value != compare_value
        elif self.operator == SearchOperator.CONTAINS:
            if isinstance(field_value, str):
                return compare_value in field_value
            elif isinstance(field_value, (list, set)):
                return compare_value in field_value
            return False
        elif self.operator == SearchOperator.NOT_CONTAINS:
            if isinstance(field_value, str):
                return compare_value not in field_value
            elif isinstance(field_value, (list, set)):
                return compare_value not in field_value
            return True
        elif self.operator == SearchOperator.STARTS_WITH:
            return isinstance(field_value, str) and field_value.startswith(compare_value)
        elif self.operator == SearchOperator.ENDS_WITH:
            return isinstance(field_value, str) and field_value.endswith(compare_value)
        elif self.operator == SearchOperator.GREATER_THAN:
            return field_value > compare_value
        elif self.operator == SearchOperator.LESS_THAN:
            return field_value < compare_value
        elif self.operator == SearchOperator.GREATER_EQUAL:
            return field_value >= compare_value
        elif self.operator == SearchOperator.LESS_EQUAL:
            return field_value <= compare_value
        elif self.operator == SearchOperator.IN:
            return field_value in compare_value
        elif self.operator == SearchOperator.NOT_IN:
            return field_value not in compare_value
        elif self.operator == SearchOperator.MATCHES:
            if isinstance(field_value, str):
                flags = 0 if self.case_sensitive else re.IGNORECASE
                return bool(re.search(self.value, field_value, flags))
            return False

        return False

    def _get_nested_value(self, item: dict[str, Any], field_path: str) -> Any:
        """Get a nested field value using dot notation."""
        parts = field_path.split(".")
        value = item

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

        return value

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field": self.field,
            "operator": self.operator.name,
            "value": self.value,
            "field_type": self.field_type.name,
            "case_sensitive": self.case_sensitive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchFilter":
        """Create from dictionary."""
        return cls(
            field=data["field"],
            operator=SearchOperator[data.get("operator", "EQUALS")],
            value=data.get("value"),
            field_type=SearchFieldType[data.get("field_type", "STRING")],
            case_sensitive=data.get("case_sensitive", False),
        )


@dataclass
class SearchQuery:
    """A complete search query.

    Attributes:
        text: Full-text search query
        filters: List of filter conditions
        sort_field: Field to sort by
        sort_ascending: Sort direction
        limit: Maximum results
        offset: Results offset for pagination
        include_paths: Only search in these paths
        exclude_paths: Exclude these paths from search
    """

    text: str = ""
    filters: list[SearchFilter] = field(default_factory=list)
    sort_field: str = "name"
    sort_ascending: bool = True
    limit: Optional[int] = None
    offset: int = 0
    include_paths: list[Path] = field(default_factory=list)
    exclude_paths: list[Path] = field(default_factory=list)

    def add_filter(
        self,
        field: str,
        operator: SearchOperator,
        value: Any,
        field_type: SearchFieldType = SearchFieldType.STRING,
    ) -> "SearchQuery":
        """Add a filter to the query.

        Args:
            field: Field to filter on
            operator: Filter operator
            value: Filter value
            field_type: Type of the field

        Returns:
            Self for chaining
        """
        self.filters.append(SearchFilter(
            field=field,
            operator=operator,
            value=value,
            field_type=field_type,
        ))
        return self

    def matches(self, item: dict[str, Any]) -> bool:
        """Check if an item matches the query.

        Args:
            item: Item to check

        Returns:
            True if matches
        """
        # Check text search
        if self.text:
            text_lower = self.text.lower()
            # Search in name, description, and tags
            name = str(item.get("name", "")).lower()
            description = str(item.get("description", "")).lower()
            tags = [str(t).lower() for t in item.get("tags", [])]

            if not (
                text_lower in name
                or text_lower in description
                or any(text_lower in tag for tag in tags)
            ):
                return False

        # Check path filters
        item_path = Path(item.get("path", ""))
        if self.include_paths:
            if not any(self._is_under_path(item_path, p) for p in self.include_paths):
                return False
        if self.exclude_paths:
            if any(self._is_under_path(item_path, p) for p in self.exclude_paths):
                return False

        # Check all filters
        for filter in self.filters:
            if not filter.matches(item):
                return False

        return True

    def _is_under_path(self, path: Path, parent: Path) -> bool:
        """Check if path is under parent."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "filters": [f.to_dict() for f in self.filters],
            "sort_field": self.sort_field,
            "sort_ascending": self.sort_ascending,
            "limit": self.limit,
            "offset": self.offset,
            "include_paths": [str(p) for p in self.include_paths],
            "exclude_paths": [str(p) for p in self.exclude_paths],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchQuery":
        """Create from dictionary."""
        return cls(
            text=data.get("text", ""),
            filters=[SearchFilter.from_dict(f) for f in data.get("filters", [])],
            sort_field=data.get("sort_field", "name"),
            sort_ascending=data.get("sort_ascending", True),
            limit=data.get("limit"),
            offset=data.get("offset", 0),
            include_paths=[Path(p) for p in data.get("include_paths", [])],
            exclude_paths=[Path(p) for p in data.get("exclude_paths", [])],
        )

    @classmethod
    def parse(cls, query_string: str) -> "SearchQuery":
        """Parse a search query string.

        Supports syntax like:
        - Simple text search: "texture"
        - Field search: name:hero
        - Operators: size>1000 size<=5000
        - Tags: tag:character
        - Type: type:mesh
        - Extension: ext:fbx
        - Quoted strings: name:"my asset"

        Args:
            query_string: Query string to parse

        Returns:
            Parsed SearchQuery
        """
        query = cls()

        # Tokenize
        tokens = cls._tokenize(query_string)

        for token in tokens:
            if ":" in token or any(op in token for op in [">=", "<=", ">", "<", "="]):
                # Field filter
                filter = cls._parse_filter_token(token)
                if filter:
                    query.filters.append(filter)
            else:
                # Text search
                if query.text:
                    query.text += " " + token
                else:
                    query.text = token

        return query

    @classmethod
    def _tokenize(cls, query_string: str) -> list[str]:
        """Tokenize a query string."""
        tokens = []
        current = ""
        in_quotes = False
        quote_char = None

        for char in query_string:
            if char in "\"'" and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif char == " " and not in_quotes:
                if current:
                    tokens.append(current)
                    current = ""
            else:
                current += char

        if current:
            tokens.append(current)

        return tokens

    @classmethod
    def _parse_filter_token(cls, token: str) -> Optional[SearchFilter]:
        """Parse a filter token."""
        # Try different patterns

        # Comparison operators
        for op_str, op in [
            (">=", SearchOperator.GREATER_EQUAL),
            ("<=", SearchOperator.LESS_EQUAL),
            (">", SearchOperator.GREATER_THAN),
            ("<", SearchOperator.LESS_THAN),
            ("=", SearchOperator.EQUALS),
        ]:
            if op_str in token:
                parts = token.split(op_str, 1)
                if len(parts) == 2:
                    field, value = parts

                    # Determine field type
                    field_type = SearchFieldType.STRING
                    if field in ("size", "width", "height"):
                        field_type = SearchFieldType.INTEGER
                        try:
                            value = int(value)
                        except ValueError:
                            pass

                    return SearchFilter(
                        field=field,
                        operator=op,
                        value=value,
                        field_type=field_type,
                    )

        # Field:value pattern
        if ":" in token:
            parts = token.split(":", 1)
            if len(parts) == 2:
                field, value = parts

                # Special fields
                if field == "tag":
                    return SearchFilter(
                        field="tags",
                        operator=SearchOperator.CONTAINS,
                        value=value,
                        field_type=SearchFieldType.TAG,
                    )
                elif field == "type":
                    return SearchFilter(
                        field="asset_type",
                        operator=SearchOperator.EQUALS,
                        value=value.upper(),
                    )
                elif field == "ext":
                    return SearchFilter(
                        field="extension",
                        operator=SearchOperator.EQUALS,
                        value=value.lower(),
                    )
                else:
                    return SearchFilter(
                        field=field,
                        operator=SearchOperator.CONTAINS,
                        value=value,
                    )

        return None


@dataclass
class SearchResult:
    """Result of a search operation.

    Attributes:
        query: The query that produced this result
        items: Matching items
        total_count: Total matches (before pagination)
        search_time_ms: Time to execute search
        suggestions: Search suggestions
    """

    query: SearchQuery
    items: list[dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    search_time_ms: float = 0.0
    suggestions: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        """Number of pages at current limit."""
        if not self.query.limit:
            return 1
        return (self.total_count + self.query.limit - 1) // self.query.limit

    @property
    def current_page(self) -> int:
        """Current page number (1-indexed)."""
        if not self.query.limit:
            return 1
        return (self.query.offset // self.query.limit) + 1

    @property
    def has_more(self) -> bool:
        """Whether there are more results."""
        return self.query.offset + len(self.items) < self.total_count


@dataclass
class SavedSearch:
    """A saved search query.

    Attributes:
        id: Unique identifier
        name: Search name
        description: Search description
        query: The saved query
        created_at: Creation timestamp
        last_used: Last use timestamp
        use_count: Number of times used
        is_favorite: Whether this is a favorite
    """

    id: str
    name: str
    description: str = ""
    query: SearchQuery = field(default_factory=SearchQuery)
    created_at: float = field(default_factory=time.time)
    last_used: Optional[float] = None
    use_count: int = 0
    is_favorite: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "query": self.query.to_dict(),
            "created_at": self.created_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "is_favorite": self.is_favorite,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SavedSearch":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            query=SearchQuery.from_dict(data.get("query", {})),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used"),
            use_count=data.get("use_count", 0),
            is_favorite=data.get("is_favorite", False),
        )


class AssetDataProvider(Protocol):
    """Protocol for providing asset data to search."""

    def get_all_assets(self) -> list[dict[str, Any]]:
        """Get all assets as searchable dictionaries."""
        ...


@editor(category="Assets")
class AssetSearch:
    """Advanced asset search engine.

    Provides:
    - Full-text search
    - Filtered search
    - Query parsing
    - Saved searches
    - Search history
    - Index-based optimization

    Attributes:
        data_provider: Provider for asset data
        storage_path: Path for saved searches
        _saved_searches: Saved search queries
        _search_history: Recent search queries
        _index: Search index (simple inverted index)
    """

    def __init__(
        self,
        data_provider: Optional[AssetDataProvider] = None,
        storage_path: Optional[Union[str, Path]] = None,
        max_history: int = 100,
    ) -> None:
        """Initialize the search engine.

        Args:
            data_provider: Provider for asset data
            storage_path: Path for persisting saved searches
            max_history: Maximum history entries
        """
        self.data_provider = data_provider
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_history = max_history

        self._saved_searches: dict[str, SavedSearch] = {}
        self._search_history: list[SearchQuery] = []
        self._index: dict[str, set[str]] = {}  # term -> set of asset paths
        self._indexed_assets: dict[str, dict[str, Any]] = {}

        # Load saved searches
        self._load_saved_searches()

    def search(
        self,
        query: Union[str, SearchQuery],
        use_index: bool = True,
    ) -> SearchResult:
        """Execute a search.

        Args:
            query: Search query (string or SearchQuery)
            use_index: Use index for optimization

        Returns:
            SearchResult with matches
        """
        start_time = time.perf_counter()

        # Parse query if string
        if isinstance(query, str):
            query = SearchQuery.parse(query)

        # Get all assets
        if self.data_provider:
            all_assets = self.data_provider.get_all_assets()
        else:
            all_assets = list(self._indexed_assets.values())

        # Filter using index if available
        if use_index and query.text and self._index:
            candidate_paths = self._get_candidates_from_index(query.text)
            all_assets = [a for a in all_assets if a.get("path") in candidate_paths]

        # Apply query
        matching = [asset for asset in all_assets if query.matches(asset)]

        # Sort
        if query.sort_field:
            matching.sort(
                key=lambda x: x.get(query.sort_field, ""),
                reverse=not query.sort_ascending,
            )

        total_count = len(matching)

        # Paginate
        if query.offset:
            matching = matching[query.offset:]
        if query.limit:
            matching = matching[:query.limit]

        # Add to history
        self._add_to_history(query)

        # Calculate search time
        search_time = (time.perf_counter() - start_time) * 1000

        # Generate suggestions
        suggestions = self._generate_suggestions(query, all_assets)

        return SearchResult(
            query=query,
            items=matching,
            total_count=total_count,
            search_time_ms=search_time,
            suggestions=suggestions,
        )

    def quick_search(self, text: str, limit: int = 10) -> list[dict[str, Any]]:
        """Quick text search for autocomplete.

        Args:
            text: Search text
            limit: Maximum results

        Returns:
            Matching items
        """
        query = SearchQuery(text=text, limit=limit)
        result = self.search(query)
        return result.items

    def index_asset(self, asset: dict[str, Any]) -> None:
        """Add an asset to the search index.

        Args:
            asset: Asset data dictionary
        """
        path = asset.get("path", "")
        if not path:
            return

        self._indexed_assets[path] = asset

        # Index text fields
        text_fields = ["name", "description"]
        for field in text_fields:
            value = asset.get(field, "")
            if value:
                terms = self._tokenize_for_index(str(value))
                for term in terms:
                    if term not in self._index:
                        self._index[term] = set()
                    self._index[term].add(path)

        # Index tags
        for tag in asset.get("tags", []):
            term = str(tag).lower()
            if term not in self._index:
                self._index[term] = set()
            self._index[term].add(path)

    def remove_from_index(self, path: Union[str, Path]) -> None:
        """Remove an asset from the index.

        Args:
            path: Asset path
        """
        path_str = str(path)
        self._indexed_assets.pop(path_str, None)

        # Remove from term index
        for term, paths in list(self._index.items()):
            paths.discard(path_str)
            if not paths:
                del self._index[term]

    def clear_index(self) -> None:
        """Clear the search index."""
        self._index.clear()
        self._indexed_assets.clear()

    def rebuild_index(self) -> int:
        """Rebuild the entire search index.

        Returns:
            Number of assets indexed
        """
        self.clear_index()

        if not self.data_provider:
            return 0

        assets = self.data_provider.get_all_assets()
        for asset in assets:
            self.index_asset(asset)

        return len(assets)

    def save_search(
        self,
        name: str,
        query: SearchQuery,
        description: str = "",
    ) -> SavedSearch:
        """Save a search query.

        Args:
            name: Search name
            query: Query to save
            description: Search description

        Returns:
            Saved search object
        """
        import uuid
        search_id = str(uuid.uuid4())[:8]

        saved = SavedSearch(
            id=search_id,
            name=name,
            description=description,
            query=query,
        )

        self._saved_searches[search_id] = saved
        self._persist_saved_searches()

        return saved

    def delete_saved_search(self, search_id: str) -> bool:
        """Delete a saved search.

        Args:
            search_id: Search ID to delete

        Returns:
            True if deleted
        """
        if search_id in self._saved_searches:
            del self._saved_searches[search_id]
            self._persist_saved_searches()
            return True
        return False

    def get_saved_search(self, search_id: str) -> Optional[SavedSearch]:
        """Get a saved search by ID."""
        return self._saved_searches.get(search_id)

    def get_saved_searches(
        self,
        favorites_only: bool = False,
    ) -> list[SavedSearch]:
        """Get all saved searches.

        Args:
            favorites_only: Only return favorites

        Returns:
            List of saved searches
        """
        searches = list(self._saved_searches.values())
        if favorites_only:
            searches = [s for s in searches if s.is_favorite]
        return sorted(searches, key=lambda s: s.use_count, reverse=True)

    def run_saved_search(self, search_id: str) -> Optional[SearchResult]:
        """Run a saved search.

        Args:
            search_id: Search ID to run

        Returns:
            SearchResult or None if not found
        """
        saved = self._saved_searches.get(search_id)
        if not saved:
            return None

        # Update usage stats
        saved.last_used = time.time()
        saved.use_count += 1
        self._persist_saved_searches()

        return self.search(saved.query)

    def get_search_history(self, limit: int = 20) -> list[SearchQuery]:
        """Get recent search history.

        Args:
            limit: Maximum entries

        Returns:
            Recent queries
        """
        return self._search_history[-limit:]

    def clear_history(self) -> None:
        """Clear search history."""
        self._search_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get search statistics."""
        return {
            "indexed_assets": len(self._indexed_assets),
            "index_terms": len(self._index),
            "saved_searches": len(self._saved_searches),
            "history_entries": len(self._search_history),
        }

    def _tokenize_for_index(self, text: str) -> list[str]:
        """Tokenize text for indexing."""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        terms = re.split(r"[^a-z0-9]+", text)
        return [t for t in terms if t and len(t) > 1]

    def _get_candidates_from_index(self, text: str) -> set[str]:
        """Get candidate paths from index for a query."""
        terms = self._tokenize_for_index(text)
        if not terms:
            return set(self._indexed_assets.keys())

        # Intersect results for all terms
        candidates = None
        for term in terms:
            term_matches = set()
            # Also match partial terms
            for indexed_term, paths in self._index.items():
                if term in indexed_term or indexed_term.startswith(term):
                    term_matches.update(paths)

            if candidates is None:
                candidates = term_matches
            else:
                candidates &= term_matches

        return candidates or set()

    def _add_to_history(self, query: SearchQuery) -> None:
        """Add a query to history."""
        # Don't add empty queries
        if not query.text and not query.filters:
            return

        self._search_history.append(query)

        # Trim to max size
        while len(self._search_history) > self.max_history:
            self._search_history.pop(0)

    def _generate_suggestions(
        self,
        query: SearchQuery,
        all_assets: list[dict[str, Any]],
    ) -> list[str]:
        """Generate search suggestions."""
        suggestions = []

        if query.text:
            text_lower = query.text.lower()

            # Collect unique terms that start with query
            seen = set()
            for asset in all_assets[:100]:  # Limit for performance
                name = asset.get("name", "")
                if name and text_lower in name.lower():
                    if name not in seen:
                        suggestions.append(name)
                        seen.add(name)

                for tag in asset.get("tags", []):
                    tag_str = str(tag)
                    if text_lower in tag_str.lower() and tag_str not in seen:
                        suggestions.append(f"tag:{tag_str}")
                        seen.add(tag_str)

        return suggestions[:10]

    def _load_saved_searches(self) -> None:
        """Load saved searches from disk."""
        if not self.storage_path:
            return

        searches_file = self.storage_path / "saved_searches.json"
        if not searches_file.exists():
            return

        try:
            with open(searches_file, "r") as f:
                data = json.load(f)

            for search_data in data.get("searches", []):
                saved = SavedSearch.from_dict(search_data)
                self._saved_searches[saved.id] = saved

        except Exception:
            pass

    def _persist_saved_searches(self) -> None:
        """Persist saved searches to disk."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        searches_file = self.storage_path / "saved_searches.json"

        data = {
            "searches": [s.to_dict() for s in self._saved_searches.values()]
        }

        with open(searches_file, "w") as f:
            json.dump(data, f, indent=2)


__all__ = [
    "SearchOperator",
    "SearchFieldType",
    "SearchFilter",
    "SearchQuery",
    "SearchResult",
    "SavedSearch",
    "AssetSearch",
]
