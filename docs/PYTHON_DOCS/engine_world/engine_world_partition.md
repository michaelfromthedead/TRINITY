# Investigation: engine/world/partition

## Summary
The world partition module provides a complete, functional implementation of spatial streaming with 2D grid-based cell management, multi-source streaming control, and layered data organization. This is production-quality code with proper state machines, priority queues, memory budgeting, and comprehensive data layer support.

## Files
| File | Lines | Status | Notes |
|------|-------|--------|-------|
| `__init__.py` | 65 | Complete | Exports 21 public symbols |
| `cell.py` | 553 | Complete | Full StreamingCell with state machine, callbacks, serialization |
| `streaming.py` | 784 | Complete | WorldStreaming manager, sources, volumes, budget tracking |
| `grid.py` | 437 | Complete | WorldGrid with spatial queries, coordinate conversion |
| `data_layer.py` | 549 | Complete | DataLayer/DataLayerManager with 8 layer types |
| `constants.py` | 181 | Complete | Well-organized configuration constants |

## Partition Components
- **StreamingCell**: Core unit with 5 states (UNLOADED, LOADING, LOADED, ACTIVATED, UNLOADING)
- **CellCoord**: Immutable 2D coordinates with distance calculations
- **CellActor**: Actor references within cells with persistence flags
- **WorldGrid**: 2D grid with configurable dimensions and cell size
- **WorldStreaming**: Main orchestrator with queues, budget, callbacks
- **StreamingSource**: Abstract base with Player, Camera, Custom implementations
- **StreamingVolume**: Trigger zones for preload/blocking/unload behavior
- **StreamingConfig/Budget**: Resource constraints and configuration
- **DataLayer**: Content organization (Runtime, Gameplay, Foliage, Audio, etc.)
- **DataLayerManager**: Multi-layer coordination with priority ordering

## Implementation
- Real world partitioning? **YES** - Full 2D grid with world-to-cell coordinate conversion, bounds checking, neighbor queries
- Real streaming? **YES** - Priority-based load/unload queues, concurrent operation limits, memory budget tracking, hysteresis to prevent thrashing
- Real level cells? **YES** - Complete state machine (5 states), actor/foliage content, HLOD proxy support, serialization/deserialization

## Verdict
**REAL IMPLEMENTATION**

This is a fully functional world partition system comparable to Unreal Engine's World Partition. All core subsystems are complete with proper:
- State machines and lifecycle management
- Priority-based streaming decisions
- Memory and IO budgeting
- Multiple streaming source types
- Data layer organization (8 layer types)
- Callbacks and event hooks
- Serialization support
- Spatial query methods

## Evidence

### Cell State Machine (cell.py:25-32)
```python
class CellState(Enum):
    """Loading state of a streaming cell."""
    UNLOADED = auto()   # No data in memory
    LOADING = auto()    # Async load in progress
    LOADED = auto()     # Data loaded but not active
    ACTIVATED = auto()  # Fully active and ticking
    UNLOADING = auto()  # Async unload in progress
```

### Streaming Priority System (streaming.py:551-576)
```python
def _queue_load_requests(self, cells: List[StreamingCell]) -> None:
    """Queue cells for loading with priority."""
    for cell in cells:
        priority = 0
        min_distance = float('inf')
        for source in self.sources:
            if source.is_active:
                dist = cell.distance_to_point(source.position)
                if dist < min_distance:
                    min_distance = dist
                    priority = source.priority
        # Closer cells get higher priority
        priority = int(priority + (PRIORITY_DISTANCE_BASE - min_distance / PRIORITY_DISTANCE_DIVISOR))
```

### Spatial Query Implementation (grid.py:202-253)
```python
def get_cells_in_radius(self, center: Vec3, radius: float, include_partial: bool = True) -> List[StreamingCell]:
    """Get all cells within a radius of a world position."""
    effective_cell_size = max(self.cell_size, MIN_CELL_SIZE)
    cell_radius = int(radius / effective_cell_size) + 1
    center_coord = self.world_to_cell_coord(center)
    # ... proper AABB-sphere intersection check
```

### Data Layer Organization (data_layer.py:33-43)
```python
class DataLayerType(Enum):
    RUNTIME = auto()     # Always loaded with the cell
    DEFAULT = auto()     # Standard content layer
    GAMEPLAY = auto()    # Interactive objects, triggers
    LANDSCAPE = auto()   # Terrain data
    FOLIAGE = auto()     # Vegetation, grass
    AUDIO = auto()       # Sound actors, ambient audio
    NAVIGATION = auto()  # NavMesh data
    LIGHTING = auto()    # Light sources, probes
    VFX = auto()         # Particles, decals
```

### Memory Budget Tracking (streaming.py:287-335)
```python
@dataclass
class StreamingBudget:
    memory_mb: float = DEFAULT_MEMORY_BUDGET_MB
    io_mbps: float = DEFAULT_IO_BANDWIDTH_MBPS
    frame_ms: float = DEFAULT_FRAME_TIME_BUDGET_MS
    # Current usage tracking with reserve/release methods
```
