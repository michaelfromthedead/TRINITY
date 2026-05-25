# PHASE 3 ARCH: Physics-Based Animation and Secondary Effects

**RDC Workflow Output**
**Generated:** 2026-05-23
**Phase:** 3 of 3

---

## Phase Overview

Phase 3 implements physics-based animation systems and secondary motion effects. These systems add believability and physical responsiveness to characters.

---

## 1. Spring Bone System

### 1.1 Architecture

```
SpringChain
├── bones: List[SpringBone]
├── root_bone_index: int
├── collision_shapes: List[CollisionShape]
├── wind_config: WindForceConfig
└── physics_config: SpringPhysicsConfig

SpringBone
├── bone_index: int
├── position: Vec3
├── previous_position: Vec3
├── rest_position: Vec3
├── stiffness: float
├── damping: float
└── mass: float
```

### 1.2 Verlet Integration

Update equation:
```
acceleration = (gravity + wind + spring_force - damping * velocity) / mass
new_position = 2 * position - previous_position + acceleration * dt^2
previous_position = position
position = new_position
```

Stability considerations:
- Timestep clamping: dt_clamped = min(dt, max_timestep)
- Substeps for large dt: if dt > threshold, divide into multiple steps
- Position clamping to prevent explosion

### 1.3 Distance Constraints

Position-based dynamics constraint solving:
```
delta = position_b - position_a
distance = length(delta)
error = distance - rest_distance
correction = delta * (error / distance) * 0.5  # 0.5 for equal mass
position_a += correction
position_b -= correction
```

Iteration count: 4-8 iterations typical for stable chains.

### 1.4 Collision Detection

**Sphere Collision:**
```
to_point = bone_position - sphere_center
distance = length(to_point)
if distance < sphere_radius:
    normal = to_point / distance
    bone_position = sphere_center + normal * sphere_radius
```

**Capsule Collision:**
1. Project bone position onto capsule axis
2. Clamp to capsule segment
3. Compute distance to projected point
4. If penetrating, push out along normal

---

## 2. Ragdoll System

### 2.1 Architecture

```
RagdollController
├── bodies: Dict[int, RagdollBody]
├── joints: List[RagdollJoint]
├── state: RagdollState
├── blend_weight: float
├── blend_duration: float
└── physics_world: PhysicsWorld

RagdollBody
├── bone_index: int
├── rigid_body_handle: RigidBodyHandle
├── mass: float
├── collision_group: int
└── is_active: bool

RagdollJoint
├── parent_body: int
├── child_body: int
├── joint_handle: JointHandle
├── limits: JointLimits
└── motor: Optional[JointMotor]
```

### 2.2 State Machine

```
RagdollState
├── KINEMATIC    # Animation drives physics bodies
├── DYNAMIC      # Physics simulation drives bones
└── BLENDING     # Interpolating between animation and physics
```

Transitions:
- KINEMATIC → DYNAMIC: Death, stun, ragdoll trigger
- DYNAMIC → BLENDING: Recovery initiation
- BLENDING → KINEMATIC: Recovery complete (blend_weight = 0)

### 2.3 Kinematic-Dynamic Transition

On transition to DYNAMIC:
1. Set all bodies to dynamic mode
2. Apply current bone velocities to bodies
3. Optionally apply impulse (hit direction)
4. Disable animation control

On transition to BLENDING:
1. Capture current physics pose
2. Begin blend_weight interpolation (1.0 → 0.0)
3. Each frame: pose = lerp(animation_pose, physics_pose, blend_weight)

### 2.4 Joint Limits

```
JointLimits
├── twist_lower: float    # Min twist rotation (radians)
├── twist_upper: float    # Max twist rotation
├── swing1_limit: float   # Cone angle limit (axis 1)
├── swing2_limit: float   # Cone angle limit (axis 2)
└── contact_distance: float  # Soft limit margin
```

Typical values:
- Neck twist: +/- 45°
- Elbow swing: 0° to 150° (no hyperextension)
- Hip swing: +/- 45° each axis

### 2.5 Active Ragdoll (Motorized Joints)

```
JointMotor
├── target_rotation: Quaternion
├── strength: float       # Motor torque
├── damping: float        # Angular damping
└── max_force: float      # Force clamp
```

Use cases:
- Stunned characters fighting to stay upright
- Puppeted animation blending with physics
- PD-controlled active ragdoll

---

## 3. Twist Bone Distribution

### 3.1 Swing-Twist Decomposition

Given a rotation quaternion Q and twist axis T:
1. Project Q's rotation axis onto T
2. Extract the component rotating around T (twist)
3. Remainder is swing rotation

```
rotation_axis, rotation_angle = quat_to_axis_angle(Q)
twist_component = dot(rotation_axis, twist_axis)
twist_angle = rotation_angle * twist_component
twist_quat = quat_from_axis_angle(twist_axis, twist_angle)
swing_quat = quat_multiply(quat_inverse(twist_quat), Q)
```

### 3.2 Distribution Modes

**Single Twist Bone:**
- One helper bone receives full twist
- Used for forearm twist helper

**Multi-Bone Distribution:**
- N bones receive twist_angle / N each
- Weighted distribution: bone_i receives weight_i * twist_angle

### 3.3 Typical Configuration

| Limb | Twist Axis | Helper Bones |
|------|------------|--------------|
| Upper Arm | Local X | 1-2 |
| Forearm | Local X | 2-3 |
| Upper Leg | Local Y | 1-2 |
| Lower Leg | Local Y | 1 |

---

## 4. Secondary Motion System

### 4.1 DelayedMotion

Time-buffered motion lag for jiggle physics effect:
```
DelayedMotion
├── buffer: RingBuffer[Transform]
├── delay_time: float
├── sample_rate: float
└── blend_factor: float
```

Update:
1. Push current transform to buffer
2. Read transform from delay_time ago
3. Output: lerp(current, delayed, blend_factor)

### 4.2 OscillatingMotion

Sine-wave oscillation overlay:
```
OscillatingMotion
├── frequency: float      # Hz
├── amplitude: Vec3       # Per-axis amplitude
├── phase_offset: float   # Phase shift
└── decay: float          # Amplitude decay over time
```

Update:
```
offset = amplitude * sin(2π * frequency * time + phase_offset)
```

### 4.3 NoiseMotion

Perlin noise FBM for organic subtle motion:
```
NoiseMotion
├── noise: PerlinNoise
├── frequency: float
├── amplitude: Vec3
├── octaves: int
└── persistence: float
```

Update:
```
for axis in [x, y, z]:
    offset[axis] = amplitude[axis] * noise.fbm(time * frequency, octaves, persistence)
```

### 4.4 ImpulseResponse

Damped spring response to sudden movements:
```
ImpulseResponse
├── current_offset: Vec3
├── velocity: Vec3
├── stiffness: float
├── damping: float
└── acceleration_threshold: float
```

Update:
1. Compute acceleration = (current_velocity - previous_velocity) / dt
2. If |acceleration| > threshold: add impulse to velocity
3. Apply spring force: F = -stiffness * offset - damping * velocity
4. Integrate position and velocity

### 4.5 MotionComposer

Stackable effect combination:
```
MotionComposer
├── effects: List[SecondaryMotionEffect]
└── blend_mode: ADD | MULTIPLY | OVERRIDE
```

Update:
1. Compute each effect's offset
2. Combine according to blend mode
3. Apply to bone transform

---

## 5. Wind Forces

### 5.1 Configuration

```
WindForceConfig
├── direction: Vec3       # World-space wind direction
├── strength: float       # Base force magnitude
├── turbulence: float     # Noise amplitude
├── frequency: float      # Turbulence variation speed
└── gust_probability: float
```

### 5.2 Wind Application

Applied to spring bones:
```
wind_force = direction * strength
turbulence_offset = perlin_noise(time * frequency) * turbulence
gust = random_gust() if random() < gust_probability else 0
total_wind = wind_force + turbulence_offset + gust
acceleration += total_wind / bone_mass
```

---

## 6. Physics World Protocol

### 6.1 Required Interface

```python
class PhysicsWorld(Protocol):
    # Rigid body management
    def create_rigid_body(self, config: RigidBodyConfig) -> RigidBodyHandle: ...
    def destroy_rigid_body(self, handle: RigidBodyHandle) -> None: ...
    def get_transform(self, handle: RigidBodyHandle) -> Transform: ...
    def set_kinematic_target(self, handle: RigidBodyHandle, target: Transform) -> None: ...
    def set_velocity(self, handle: RigidBodyHandle, linear: Vec3, angular: Vec3) -> None: ...
    def apply_impulse(self, handle: RigidBodyHandle, impulse: Vec3, point: Vec3) -> None: ...
    
    # Joint management
    def create_joint(self, config: JointConfig) -> JointHandle: ...
    def destroy_joint(self, handle: JointHandle) -> None: ...
    def set_motor_target(self, handle: JointHandle, target: Quaternion) -> None: ...
```

### 6.2 Expected Backends

- NVIDIA PhysX
- Bullet Physics
- Box2D (2D games)
- Custom physics (protocol adapter)

---

## 7. Integration Order

When combining multiple systems on one character:

1. **Motion Matching / Animation** (base pose)
2. **Ragdoll Blending** (if in BLENDING state)
3. **Twist Bone Distribution** (helper bones)
4. **Look-At Controller** (head/eyes)
5. **Breathing** (chest/spine additive)
6. **Secondary Motion** (additive effects)
7. **Spring Bones** (simulated geometry)
8. **IK Corrections** (foot placement, hand placement)

Each layer operates on the output of the previous layer.
