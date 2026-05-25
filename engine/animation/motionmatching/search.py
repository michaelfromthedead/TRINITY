"""
Motion Matching Search - Database search algorithms.

This module provides search algorithms for motion matching:
- SearchConfig: Configuration for search behavior
- MotionSearch: Main search class with acceleration structures
- Brute force search (baseline)
- KD-tree acceleration for large databases
- Locality-sensitive hashing for approximate search
- Cost function with weighted feature distances

Usage:
    from engine.animation.motionmatching.search import (
        MotionSearch, SearchConfig, SearchResult
    )

    # Create searcher with database
    search = MotionSearch(database)

    # Search for best matches
    results = search.search(query_features, config)
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)
import numpy as np

from engine.animation.motionmatching.database import (
    DatabaseEntry,
    MotionDatabase,
)
from engine.animation.motionmatching.features import FeatureSet


# =============================================================================
# CONSTANTS AND ENUMS
# =============================================================================


class SearchMethod(Enum):
    """Available search methods."""
    BRUTE_FORCE = auto()      # Linear scan (guaranteed optimal)
    KD_TREE = auto()          # KD-tree for exact nearest neighbors
    LSH = auto()              # Locality-sensitive hashing (approximate)
    HIERARCHICAL = auto()     # Hierarchical clustering


# Import centralized config
from engine.animation.motionmatching.config import (
    DEFAULT_SEARCH_PARAMS,
    DEFAULT_FEATURE_WEIGHTS,
)

# Default search parameters
DEFAULT_MAX_RESULTS = DEFAULT_SEARCH_PARAMS.max_results
DEFAULT_COST_THRESHOLD = DEFAULT_SEARCH_PARAMS.default_cost_threshold
COST_EPSILON = DEFAULT_SEARCH_PARAMS.cost_epsilon


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SearchConfig:
    """Configuration for motion matching search.

    Attributes:
        feature_weights: Per-feature dimension weights
        max_results: Maximum number of results to return
        cost_threshold: Maximum cost for valid matches
        method: Search method to use
        required_tags: Tags that must be present in results
        excluded_tags: Tags that must not be present
        only_transition_candidates: Only return entries marked as transition candidates
        exclude_current_clip: Clip index to exclude from results
        exclude_frames_range: (clip_idx, start_frame, end_frame) to exclude
        min_frame_distance: Minimum frame distance from current position
    """
    feature_weights: Optional[np.ndarray] = None
    max_results: int = DEFAULT_MAX_RESULTS
    cost_threshold: float = DEFAULT_COST_THRESHOLD
    method: SearchMethod = SearchMethod.BRUTE_FORCE

    required_tags: Optional[Set[str]] = None
    excluded_tags: Optional[Set[str]] = None
    only_transition_candidates: bool = True

    exclude_current_clip: Optional[int] = None
    exclude_frames_range: Optional[Tuple[int, int, int]] = None
    min_frame_distance: int = 0

    # KD-tree specific
    kd_tree_leaf_size: int = 40

    # LSH specific
    lsh_num_tables: int = 10
    lsh_num_hashes: int = 8
    lsh_bucket_width: float = 1.0


@dataclass
class SearchResult:
    """Result from motion matching search.

    Attributes:
        entry: The matched database entry
        entry_index: Index of entry in database
        cost: Total cost (lower is better)
        feature_costs: Per-feature category costs
    """
    entry: DatabaseEntry
    entry_index: int
    cost: float
    feature_costs: Dict[str, float] = field(default_factory=dict)

    def __lt__(self, other: SearchResult) -> bool:
        """Compare by cost for heap operations."""
        return self.cost < other.cost

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchResult):
            return NotImplemented
        return self.entry_index == other.entry_index


# =============================================================================
# COST FUNCTIONS
# =============================================================================


def compute_cost(
    query: np.ndarray,
    candidate: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_cost: float = None,
) -> float:
    """Compute weighted squared distance cost.

    Cost = sum of (weight_i * (query_i - candidate_i)^2)

    Uses float64 internally for numerical stability with large vectors.

    Args:
        query: Query feature vector
        candidate: Candidate feature vector
        weights: Optional per-feature weights
        max_cost: Optional early termination threshold

    Returns:
        Total cost (clamped to prevent overflow)
    """
    # Use float64 for numerical stability
    diff = (query - candidate).astype(np.float64)
    squared_diff = diff * diff

    # Clamp individual squared differences to prevent overflow
    max_diff = 1e10
    squared_diff = np.clip(squared_diff, 0, max_diff)

    if weights is not None:
        weighted = squared_diff * weights.astype(np.float64)
    else:
        weighted = squared_diff

    total = np.sum(weighted)

    # Clamp final result and convert back to float
    return float(min(total, 1e15))


def compute_cost_vectorized(
    query: np.ndarray,
    candidates: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute costs for multiple candidates at once.

    Uses float64 internally for numerical stability with large vectors.

    Args:
        query: Query feature vector (1D)
        candidates: Candidate feature matrix (num_candidates x feature_dim)
        weights: Optional per-feature weights (1D)

    Returns:
        Array of costs (num_candidates,)
    """
    # Handle empty candidates
    if candidates.size == 0:
        return np.array([], dtype=np.float32)

    # Use float64 for numerical stability
    diff = (candidates.astype(np.float64) - query.astype(np.float64))
    squared_diff = diff * diff

    # Clamp to prevent overflow
    max_diff = 1e10
    squared_diff = np.clip(squared_diff, 0, max_diff)

    if weights is not None:
        weighted = squared_diff * weights.astype(np.float64)
    else:
        weighted = squared_diff

    costs = np.sum(weighted, axis=1)

    # Clamp final results
    return np.clip(costs, 0, 1e15).astype(np.float32)


def compute_partial_cost(
    query: np.ndarray,
    candidate: np.ndarray,
    weights: Optional[np.ndarray] = None,
    feature_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute cost with per-feature breakdown.

    Args:
        query: Query feature vector
        candidate: Candidate feature vector
        weights: Optional per-feature weights
        feature_ranges: Mapping of feature names to (start, end) indices

    Returns:
        Tuple of (total_cost, dict of per-feature costs)
    """
    diff = query - candidate
    squared_diff = diff * diff

    if weights is not None:
        weighted = squared_diff * weights
    else:
        weighted = squared_diff

    total_cost = float(np.sum(weighted))

    feature_costs: Dict[str, float] = {}
    if feature_ranges:
        for name, (start, end) in feature_ranges.items():
            feature_costs[name] = float(np.sum(weighted[start:end]))

    return total_cost, feature_costs


# =============================================================================
# KD-TREE IMPLEMENTATION
# =============================================================================


class KDTreeNode:
    """Node in a KD-tree.

    Attributes:
        split_dim: Dimension used for splitting
        split_value: Value used for splitting
        left: Left child (values <= split_value)
        right: Right child (values > split_value)
        indices: Leaf node indices (only for leaves)
    """
    __slots__ = ['split_dim', 'split_value', 'left', 'right', 'indices']

    def __init__(
        self,
        split_dim: int = -1,
        split_value: float = 0.0,
        left: Optional[KDTreeNode] = None,
        right: Optional[KDTreeNode] = None,
        indices: Optional[np.ndarray] = None,
    ):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        self.indices = indices

    @property
    def is_leaf(self) -> bool:
        return self.indices is not None


class KDTree:
    """KD-tree for fast nearest neighbor search.

    Builds a spatial partition of the feature space for
    efficient searching in O(log n) average case.
    """

    def __init__(
        self,
        data: np.ndarray,
        leaf_size: int = None,
    ):
        """Build KD-tree from data.

        Args:
            data: Feature matrix (num_samples x feature_dim)
            leaf_size: Maximum points in leaf nodes (uses config default if None)

        Raises:
            ValueError: If data is empty
        """
        if leaf_size is None:
            leaf_size = DEFAULT_SEARCH_PARAMS.kd_tree_leaf_size

        self.data = data
        self.leaf_size = leaf_size
        self._is_empty = len(data) == 0

        # Handle empty data
        if self._is_empty:
            self.dimension = 1
            self.root = KDTreeNode(indices=np.array([], dtype=np.int32))
            return

        self.dimension = data.shape[1] if len(data.shape) > 1 else 1

        indices = np.arange(len(data))
        self.root = self._build_tree(indices, 0)

    def _build_tree(
        self,
        indices: np.ndarray,
        depth: int,
    ) -> KDTreeNode:
        """Recursively build tree.

        Args:
            indices: Indices of points in this subtree
            depth: Current depth in tree

        Returns:
            KDTreeNode for this subtree
        """
        if len(indices) <= self.leaf_size:
            return KDTreeNode(indices=indices)

        # Choose split dimension (cycle through dimensions)
        split_dim = depth % self.dimension

        # Find median for split value
        values = self.data[indices, split_dim]
        median_idx = len(values) // 2
        sorted_indices = np.argsort(values)
        split_value = values[sorted_indices[median_idx]]

        # Partition indices
        left_mask = values <= split_value
        right_mask = ~left_mask

        # Handle edge case where all points have same value
        if not np.any(right_mask):
            return KDTreeNode(indices=indices)

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        return KDTreeNode(
            split_dim=split_dim,
            split_value=split_value,
            left=self._build_tree(left_indices, depth + 1),
            right=self._build_tree(right_indices, depth + 1),
        )

    def query(
        self,
        point: np.ndarray,
        k: int = 1,
        weights: Optional[np.ndarray] = None,
        max_cost: float = float('inf'),
        filter_fn: Optional[Callable[[int], bool]] = None,
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors.

        Args:
            point: Query point
            k: Number of neighbors to find
            weights: Optional per-dimension weights
            max_cost: Maximum cost threshold
            filter_fn: Optional function to filter candidates by index

        Returns:
            List of (index, cost) tuples, sorted by cost
        """
        # Handle empty tree
        if self._is_empty:
            return []

        # Use max-heap (negate costs) to track k best
        best: List[Tuple[float, int]] = []  # (-cost, index)

        def search(node: KDTreeNode, best_cost: float) -> float:
            if node.is_leaf:
                # Check all points in leaf
                for idx in node.indices:
                    if filter_fn and not filter_fn(idx):
                        continue

                    cost = compute_cost(point, self.data[idx], weights)
                    if cost < best_cost and cost < max_cost:
                        if len(best) < k:
                            heapq.heappush(best, (-cost, idx))
                        elif cost < -best[0][0]:
                            heapq.heapreplace(best, (-cost, idx))
                        best_cost = -best[0][0] if len(best) == k else max_cost
                return best_cost

            # Check split dimension
            dim = node.split_dim
            val = point[dim]
            if weights is not None:
                dim_weight = weights[dim]
            else:
                dim_weight = 1.0

            # Determine which child to search first
            if val <= node.split_value:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            # Search closer child first
            best_cost = search(first, best_cost)

            # Check if we need to search other child
            dist_to_split = (val - node.split_value) ** 2 * dim_weight
            if dist_to_split < best_cost:
                best_cost = search(second, best_cost)

            return best_cost

        search(self.root, float('inf'))

        # Convert to results
        results = [(-cost, idx) for cost, idx in best]
        results.sort()  # Sort by cost ascending
        return [(idx, cost) for cost, idx in results]


# =============================================================================
# LOCALITY-SENSITIVE HASHING
# =============================================================================


class LSHTable:
    """Single LSH table with random projections.

    Uses random hyperplanes to hash similar vectors to same buckets.
    """

    def __init__(
        self,
        dimension: int,
        num_hashes: int,
        bucket_width: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Initialize LSH table.

        Args:
            dimension: Feature dimension
            num_hashes: Number of hash functions
            bucket_width: Width of hash buckets
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        # Random projection vectors
        self.projections = rng.standard_normal((num_hashes, dimension))
        self.projections = self.projections.astype(np.float32)

        # Random offsets
        self.offsets = rng.uniform(0, bucket_width, num_hashes)
        self.offsets = self.offsets.astype(np.float32)

        self.bucket_width = bucket_width
        self.buckets: Dict[Tuple[int, ...], List[int]] = {}

    def _hash(self, point: np.ndarray) -> Tuple[int, ...]:
        """Compute hash for a point.

        Args:
            point: Feature vector

        Returns:
            Hash as tuple of integers
        """
        projected = np.dot(self.projections, point)
        hashed = np.floor((projected + self.offsets) / self.bucket_width)
        return tuple(hashed.astype(np.int32))

    def insert(self, index: int, point: np.ndarray) -> None:
        """Insert point into hash table.

        Args:
            index: Index of point
            point: Feature vector
        """
        h = self._hash(point)
        if h not in self.buckets:
            self.buckets[h] = []
        self.buckets[h].append(index)

    def query(self, point: np.ndarray) -> List[int]:
        """Query for candidate neighbors.

        Args:
            point: Query feature vector

        Returns:
            List of candidate indices
        """
        h = self._hash(point)
        return self.buckets.get(h, [])


class LSHIndex:
    """Locality-sensitive hashing index for approximate nearest neighbors.

    Uses multiple hash tables to increase recall while maintaining
    sub-linear query time.
    """

    def __init__(
        self,
        dimension: int,
        num_tables: int = 10,
        num_hashes: int = 8,
        bucket_width: float = 1.0,
    ):
        """Initialize LSH index.

        Args:
            dimension: Feature dimension
            num_tables: Number of hash tables (more = higher recall)
            num_hashes: Number of hash functions per table (more = fewer false positives)
            bucket_width: Width of hash buckets
        """
        self.dimension = dimension
        self.num_tables = num_tables

        self.tables = [
            LSHTable(dimension, num_hashes, bucket_width, seed=i)
            for i in range(num_tables)
        ]

    def build(self, data: np.ndarray) -> None:
        """Build index from data.

        Args:
            data: Feature matrix (num_samples x feature_dim)
        """
        for i, point in enumerate(data):
            for table in self.tables:
                table.insert(i, point)

    def query(self, point: np.ndarray) -> Set[int]:
        """Query for candidate neighbors.

        Args:
            point: Query feature vector

        Returns:
            Set of candidate indices from all tables
        """
        candidates: Set[int] = set()
        for table in self.tables:
            candidates.update(table.query(point))
        return candidates


# =============================================================================
# MOTION SEARCH
# =============================================================================


class MotionSearch:
    """Main search class for motion matching.

    Provides multiple search algorithms with acceleration structures
    for efficient nearest neighbor search in motion databases.
    """

    def __init__(
        self,
        database: MotionDatabase,
        method: SearchMethod = SearchMethod.BRUTE_FORCE,
        config: Optional[SearchConfig] = None,
    ):
        """Initialize motion search.

        Args:
            database: Motion database to search
            method: Default search method
            config: Default search configuration
        """
        self.database = database
        self.default_method = method
        self.default_config = config or SearchConfig()

        # Acceleration structures (built lazily)
        self._kd_tree: Optional[KDTree] = None
        self._lsh_index: Optional[LSHIndex] = None

        # Build default structures if needed
        if method == SearchMethod.KD_TREE:
            self._build_kd_tree(self.default_config)
        elif method == SearchMethod.LSH:
            self._build_lsh_index(self.default_config)

    def _build_kd_tree(self, config: SearchConfig) -> None:
        """Build KD-tree for search."""
        if self.database.feature_matrix is None:
            self.database.finalize()

        features = self.database.get_features(normalize=True)
        self._kd_tree = KDTree(features, leaf_size=config.kd_tree_leaf_size)

    def _build_lsh_index(self, config: SearchConfig) -> None:
        """Build LSH index for search."""
        if self.database.feature_matrix is None:
            self.database.finalize()

        features = self.database.get_features(normalize=True)
        self._lsh_index = LSHIndex(
            dimension=self.database.feature_dimension,
            num_tables=config.lsh_num_tables,
            num_hashes=config.lsh_num_hashes,
            bucket_width=config.lsh_bucket_width,
        )
        self._lsh_index.build(features)

    def search(
        self,
        query: Union[FeatureSet, np.ndarray],
        config: Optional[SearchConfig] = None,
    ) -> List[SearchResult]:
        """Search for best matching frames.

        Args:
            query: Query features (FeatureSet or numpy array)
            config: Search configuration (uses default if None)

        Returns:
            List of SearchResult sorted by cost (ascending)
        """
        config = config or self.default_config

        # Extract query vector
        if isinstance(query, FeatureSet):
            query_vector = query.values
            feature_weights = config.feature_weights
            if feature_weights is None and query.weights is not None:
                feature_weights = query.weights
            feature_ranges = query.feature_ranges
        else:
            query_vector = query
            feature_weights = config.feature_weights
            feature_ranges = None

        # Normalize query if database has normalization
        if self.database.normalization is not None:
            query_vector = self.database.normalization.normalize(query_vector)

        # Build filter function
        filter_fn = self._build_filter(config)

        # Dispatch to appropriate search method
        method = config.method or self.default_method

        if method == SearchMethod.BRUTE_FORCE:
            results = self._search_brute_force(
                query_vector, config, feature_weights, filter_fn, feature_ranges
            )
        elif method == SearchMethod.KD_TREE:
            if self._kd_tree is None:
                self._build_kd_tree(config)
            results = self._search_kd_tree(
                query_vector, config, feature_weights, filter_fn, feature_ranges
            )
        elif method == SearchMethod.LSH:
            if self._lsh_index is None:
                self._build_lsh_index(config)
            results = self._search_lsh(
                query_vector, config, feature_weights, filter_fn, feature_ranges
            )
        else:
            results = self._search_brute_force(
                query_vector, config, feature_weights, filter_fn, feature_ranges
            )

        return results

    def _build_filter(
        self, config: SearchConfig
    ) -> Callable[[int], bool]:
        """Build filter function from config.

        Args:
            config: Search configuration

        Returns:
            Function that returns True for valid entries
        """
        def filter_fn(index: int) -> bool:
            entry = self.database.get_entry(index)
            if entry is None:
                return False

            # Check transition candidate
            if config.only_transition_candidates and not entry.is_transition_candidate:
                return False

            # Check required tags
            if config.required_tags:
                if not config.required_tags.issubset(entry.tags):
                    return False

            # Check excluded tags
            if config.excluded_tags:
                if config.excluded_tags.intersection(entry.tags):
                    return False

            # Check clip exclusion
            if config.exclude_current_clip is not None:
                if entry.clip_index == config.exclude_current_clip:
                    return False

            # Check frame range exclusion
            if config.exclude_frames_range:
                clip_idx, start, end = config.exclude_frames_range
                if entry.clip_index == clip_idx and start <= entry.frame < end:
                    return False

            return True

        return filter_fn

    def _search_brute_force(
        self,
        query: np.ndarray,
        config: SearchConfig,
        weights: Optional[np.ndarray],
        filter_fn: Callable[[int], bool],
        feature_ranges: Optional[Dict[str, Tuple[int, int]]],
    ) -> List[SearchResult]:
        """Brute force search (linear scan).

        Args:
            query: Query feature vector
            config: Search configuration
            weights: Feature weights
            filter_fn: Filter function
            feature_ranges: Feature ranges for partial cost breakdown

        Returns:
            List of SearchResult
        """
        features = self.database.get_features(normalize=True)

        # Handle empty database
        if features.size == 0 or len(features) == 0:
            return []

        # Compute all costs at once (vectorized)
        costs = compute_cost_vectorized(query, features, weights)

        # Add cost modifiers
        for i, entry in enumerate(self.database.entries):
            costs[i] += entry.cost_modifier

        # Get indices sorted by cost
        sorted_indices = np.argsort(costs)

        results: List[SearchResult] = []
        for idx in sorted_indices:
            idx = int(idx)

            if not filter_fn(idx):
                continue

            cost = float(costs[idx])
            if cost > config.cost_threshold:
                break

            entry = self.database.get_entry(idx)

            # Compute partial costs if requested
            if feature_ranges:
                _, partial_costs = compute_partial_cost(
                    query, features[idx], weights, feature_ranges
                )
            else:
                partial_costs = {}

            results.append(SearchResult(
                entry=entry,
                entry_index=idx,
                cost=cost,
                feature_costs=partial_costs,
            ))

            if len(results) >= config.max_results:
                break

        return results

    def _search_kd_tree(
        self,
        query: np.ndarray,
        config: SearchConfig,
        weights: Optional[np.ndarray],
        filter_fn: Callable[[int], bool],
        feature_ranges: Optional[Dict[str, Tuple[int, int]]],
    ) -> List[SearchResult]:
        """KD-tree accelerated search.

        Args:
            query: Query feature vector
            config: Search configuration
            weights: Feature weights
            filter_fn: Filter function
            feature_ranges: Feature ranges for partial cost breakdown

        Returns:
            List of SearchResult
        """
        # Query KD-tree
        candidates = self._kd_tree.query(
            query,
            k=config.max_results * 3,  # Get extra to account for filtering
            weights=weights,
            max_cost=config.cost_threshold,
            filter_fn=filter_fn,
        )

        features = self.database.get_features(normalize=True)
        results: List[SearchResult] = []

        for idx, cost in candidates:
            entry = self.database.get_entry(idx)

            # Add cost modifier
            cost += entry.cost_modifier

            if cost > config.cost_threshold:
                continue

            # Compute partial costs if requested
            if feature_ranges:
                _, partial_costs = compute_partial_cost(
                    query, features[idx], weights, feature_ranges
                )
            else:
                partial_costs = {}

            results.append(SearchResult(
                entry=entry,
                entry_index=idx,
                cost=cost,
                feature_costs=partial_costs,
            ))

            if len(results) >= config.max_results:
                break

        return results

    def _search_lsh(
        self,
        query: np.ndarray,
        config: SearchConfig,
        weights: Optional[np.ndarray],
        filter_fn: Callable[[int], bool],
        feature_ranges: Optional[Dict[str, Tuple[int, int]]],
    ) -> List[SearchResult]:
        """LSH accelerated approximate search.

        Args:
            query: Query feature vector
            config: Search configuration
            weights: Feature weights
            filter_fn: Filter function
            feature_ranges: Feature ranges for partial cost breakdown

        Returns:
            List of SearchResult
        """
        # Get candidates from LSH
        candidate_indices = self._lsh_index.query(query)

        if not candidate_indices:
            # Fallback to brute force if no candidates
            return self._search_brute_force(
                query, config, weights, filter_fn, feature_ranges
            )

        features = self.database.get_features(normalize=True)

        # Compute costs for candidates
        results: List[SearchResult] = []

        for idx in candidate_indices:
            if not filter_fn(idx):
                continue

            cost = compute_cost(query, features[idx], weights)
            entry = self.database.get_entry(idx)
            cost += entry.cost_modifier

            if cost > config.cost_threshold:
                continue

            # Compute partial costs if requested
            if feature_ranges:
                _, partial_costs = compute_partial_cost(
                    query, features[idx], weights, feature_ranges
                )
            else:
                partial_costs = {}

            results.append(SearchResult(
                entry=entry,
                entry_index=idx,
                cost=cost,
                feature_costs=partial_costs,
            ))

        # Sort by cost and limit
        results.sort(key=lambda r: r.cost)
        return results[:config.max_results]

    def find_best_match(
        self,
        query: Union[FeatureSet, np.ndarray],
        config: Optional[SearchConfig] = None,
    ) -> Optional[SearchResult]:
        """Find single best matching frame.

        Args:
            query: Query features
            config: Search configuration

        Returns:
            Best SearchResult or None if no match found
        """
        if config is None:
            config = SearchConfig(max_results=1)
        else:
            config = SearchConfig(
                feature_weights=config.feature_weights,
                max_results=1,
                cost_threshold=config.cost_threshold,
                method=config.method,
                required_tags=config.required_tags,
                excluded_tags=config.excluded_tags,
                only_transition_candidates=config.only_transition_candidates,
                exclude_current_clip=config.exclude_current_clip,
                exclude_frames_range=config.exclude_frames_range,
            )

        results = self.search(query, config)
        return results[0] if results else None

    def rebuild_index(
        self,
        method: Optional[SearchMethod] = None,
        config: Optional[SearchConfig] = None,
    ) -> None:
        """Rebuild acceleration structures.

        Args:
            method: Method to rebuild for (or current default)
            config: Configuration for building
        """
        method = method or self.default_method
        config = config or self.default_config

        if method == SearchMethod.KD_TREE:
            self._kd_tree = None
            self._build_kd_tree(config)
        elif method == SearchMethod.LSH:
            self._lsh_index = None
            self._build_lsh_index(config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_trajectory_cost(
    query_trajectory: np.ndarray,
    candidate_trajectory: np.ndarray,
    position_weight: float = 1.0,
    facing_weight: float = 1.0,
) -> float:
    """Compute cost between two trajectories.

    Args:
        query_trajectory: Query trajectory features
        candidate_trajectory: Candidate trajectory features
        position_weight: Weight for position component
        facing_weight: Weight for facing component

    Returns:
        Total trajectory cost
    """
    # Assuming trajectory is stored as [pos_x, pos_y, pos_z, face_x, face_y] per point
    num_points = len(query_trajectory) // 5

    total_cost = 0.0
    for i in range(num_points):
        offset = i * 5

        # Position cost
        pos_diff = query_trajectory[offset:offset+3] - candidate_trajectory[offset:offset+3]
        total_cost += position_weight * np.sum(pos_diff ** 2)

        # Facing cost
        face_diff = query_trajectory[offset+3:offset+5] - candidate_trajectory[offset+3:offset+5]
        total_cost += facing_weight * np.sum(face_diff ** 2)

    return total_cost


def compute_pose_cost(
    query_pose: np.ndarray,
    candidate_pose: np.ndarray,
    position_weight: float = 1.0,
    velocity_weight: float = 0.5,
) -> float:
    """Compute cost between two poses.

    Args:
        query_pose: Query pose features (positions + velocities)
        candidate_pose: Candidate pose features
        position_weight: Weight for position features
        velocity_weight: Weight for velocity features

    Returns:
        Total pose cost
    """
    # Assuming pose is [pos_bone1, pos_bone2, ..., vel_bone1, vel_bone2, ...]
    half = len(query_pose) // 2

    # Position cost
    pos_diff = query_pose[:half] - candidate_pose[:half]
    pos_cost = position_weight * np.sum(pos_diff ** 2)

    # Velocity cost
    vel_diff = query_pose[half:] - candidate_pose[half:]
    vel_cost = velocity_weight * np.sum(vel_diff ** 2)

    return pos_cost + vel_cost
