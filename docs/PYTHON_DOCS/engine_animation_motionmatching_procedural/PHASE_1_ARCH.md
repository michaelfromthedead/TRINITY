# PHASE 1 ARCH: Core Animation Infrastructure

**RDC Workflow Output**
**Generated:** 2026-05-23
**Phase:** 1 of 3

---

## Phase Overview

Phase 1 establishes the foundational data structures and algorithms that both motion matching and procedural animation depend on. This phase focuses on the "what" and "how" of animation data representation.

---

## 1. Motion Database Architecture

### 1.1 Data Structures

```
MotionDatabase
├── entries: List[DatabaseEntry]
├── clips: List[ClipMetadata]
├── normalization_stats: NormalizationStats
├── tag_indices: Dict[str, Set[int]]
├── clip_ranges: Dict[int, Tuple[int, int]]
└── quantization: QuantizationLevel

DatabaseEntry
├── clip_index: int
├── frame: int
├── features: np.ndarray
├── tags: FrozenSet[str]
├── cost_modifier: float
└── is_transition_candidate: bool

ClipMetadata
├── name: str
├── frame_count: int
├── frame_rate: float
├── duration: float
├── is_looping: bool
└── root_motion_per_frame: Optional[np.ndarray]
```

### 1.2 Serialization Format (MMDB)

Binary format with gzip compression:
- Magic number validation
- Header with version, dimensions, quantization level
- Normalization statistics block
- Clip metadata array
- Feature data (raw or quantized)
- Tag index mapping

### 1.3 Quantization Strategy

| Level | Storage | Precision | Memory Ratio |
|-------|---------|-----------|--------------|
| NONE | float32 | Full | 1.0x |
| FLOAT16 | float16 | ~3 decimal places | 0.5x |
| INT16 | int16 + scale | ~4 decimal places | 0.5x |
| INT8 | int8 + scale | ~2 decimal places | 0.25x |

---

## 2. Feature Extraction Pipeline

### 2.1 Feature Types

| Feature | Dimensions | Description |
|---------|------------|-------------|
| Bone Position | 3 per bone | Local-space XYZ |
| Bone Velocity | 3 per bone | Frame-to-frame delta |
| Trajectory Position | 3 per time point | Future world positions |
| Trajectory Facing | 2 per time point | 2D direction vector |
| Foot Contact | 1 per foot | 0.0-1.0 contact state |

### 2.2 Standard Bone Set

```
KEY_BONES = [
    "hips",        # Root reference
    "left_foot",   # Contact detection
    "right_foot",  # Contact detection
    "left_hand",   # Arm position
    "right_hand",  # Arm position
    "head",        # Facing reference
]
```

### 2.3 Normalization

- **z-score**: `(value - mean) / std`
- **min-max**: `(value - min) / (max - min)`
- Statistics computed during database build, stored in MMDB

---

## 3. Search Acceleration Structures

### 3.1 KD-Tree

```
KDTreeNode
├── split_dim: int
├── split_value: float
├── left: Optional[KDTreeNode]
├── right: Optional[KDTreeNode]
└── indices: Optional[np.ndarray]  # Leaf only
```

Construction:
1. Select split dimension (cycling through feature dimensions)
2. Find median value in that dimension
3. Partition entries by median
4. Recurse until leaf_size threshold

Search:
1. Traverse to containing leaf
2. Check all entries in leaf
3. Backtrack checking sibling subtrees if distance to split plane < best distance

### 3.2 Locality-Sensitive Hashing (LSH)

```
LSHIndex
├── num_tables: int
├── num_hashes_per_table: int
├── projection_vectors: List[np.ndarray]
└── hash_tables: List[Dict[int, List[int]]]
```

Construction:
1. Generate random projection vectors per table
2. For each entry: compute hash via dot product signs
3. Insert entry index into bucket

Search:
1. Hash query using same projections
2. Retrieve candidate set from matching buckets
3. Compute exact distance on candidates only

---

## 4. Quaternion and Vector Utilities

### 4.1 Quaternion Operations

| Operation | Formula |
|-----------|---------|
| Multiply | Hamilton product |
| Inverse | Conjugate / magnitude^2 |
| SLERP | sin-weighted interpolation |
| Axis-Angle | Extract rotation axis and angle |
| Normalize | q / magnitude(q) |

### 4.2 Vector Operations

| Operation | Formula |
|-----------|---------|
| Add | v1 + v2 |
| Subtract | v1 - v2 |
| Scale | v * scalar |
| Dot | sum(v1 * v2) |
| Length | sqrt(dot(v, v)) |
| Normalize | v / length(v) |
| Cross | 3D cross product |

---

## 5. Configuration Architecture

### 5.1 Pattern

Each module has a `config.py` with frozen dataclasses:

```python
@dataclass(frozen=True)
class FeatureWeightConfig:
    bone_position_weight: float = 1.0
    bone_velocity_weight: float = 0.5
    trajectory_position_weight: float = 1.2
    trajectory_facing_weight: float = 1.0
    foot_contact_weight: float = 0.3
```

### 5.2 Benefits

- Immutable prevents runtime modification
- Type-safe configuration
- Documentation via docstrings
- Default values for quick start
- `__post_init__` validation

---

## 6. Protocol Interfaces

### 6.1 Pose Protocol

```python
class Pose(Protocol):
    def get_bone_position(self, bone_index: int) -> Vec3: ...
    def get_bone_rotation(self, bone_index: int) -> Quaternion: ...
    def set_bone_position(self, bone_index: int, position: Vec3) -> None: ...
    def set_bone_rotation(self, bone_index: int, rotation: Quaternion) -> None: ...
```

### 6.2 Skeleton Protocol

```python
class Skeleton(Protocol):
    def get_bone_index(self, name: str) -> int: ...
    def get_bone_name(self, index: int) -> str: ...
    def get_bone_parent(self, index: int) -> Optional[int]: ...
    def get_bone_count(self) -> int: ...
```

### 6.3 PhysicsWorld Protocol (for ragdoll)

```python
class PhysicsWorld(Protocol):
    def create_rigid_body(self, config: RigidBodyConfig) -> RigidBodyHandle: ...
    def create_joint(self, config: JointConfig) -> JointHandle: ...
    def get_transform(self, body: RigidBodyHandle) -> Transform: ...
    def set_kinematic_target(self, body: RigidBodyHandle, target: Transform) -> None: ...
```

---

## 7. Architectural Decisions

### 7.1 Why NumPy Arrays for Features?

- Vectorized cost computation (100x faster than loops)
- Memory-efficient storage with dtype control
- Interoperability with C-based physics engines

### 7.2 Why Protocol Interfaces?

- Decoupling from concrete implementations
- Enable mock injection for testing
- Support multiple physics backends

### 7.3 Why Frozen Dataclasses for Config?

- Prevent accidental mutation
- Thread-safe configuration sharing
- Clear documentation of defaults
