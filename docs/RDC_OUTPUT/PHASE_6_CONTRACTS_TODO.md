# PHASE 6: Contract Annotation — TODO

**Duration:** Ongoing (incremental)

---

## Tasks

### T-CONT-6.1: Create trinity_contracts crate ✓
- [x] Create `crates/trinity-contracts/Cargo.toml`
- [x] Add dependencies: proc-macro2, syn, quote, proptest
- [x] Create macro crate: `crates/trinity-contracts-macros/`

### T-CONT-6.2: Parse #[contract] attribute ✓
- [x] Implement proc macro entry point
- [x] Parse function signature
- [x] Extract inner attributes (#![requires], #![ensures], etc.)

### T-CONT-6.3: Runtime check generation
- [ ] Generate debug_assert! for requires
- [ ] Generate debug_assert! for ensures
- [ ] Preserve original function body

### T-CONT-6.4: Property test generation
- [ ] Generate test module
- [ ] Convert requires to proptest strategies
- [ ] Convert ensures to prop_assert!

### T-CONT-6.5: synth schema extraction
- [ ] Parse requires constraints
- [ ] Convert to synth schema JSON
- [ ] Store in contracts table

### T-CONT-6.6: Layout contracts
- [ ] Parse #[layout(size = N, align = M)]
- [ ] Generate compile-time size/align checks
- [ ] Link to WGSL struct mirrors

### T-CONT-6.7: Algebraic properties
- [ ] Parse #![property(commutative)]
- [ ] Generate specialized tests
- [ ] Integrate with synth for input generation

### T-CONT-6.8: Incremental rollout
- [ ] Start with 10 high-value functions
- [ ] Validate proc macro correctness
- [ ] Expand to remaining codebase

---

## Priority Order for Annotation

| Priority | Target | Reason |
|----------|--------|--------|
| P1 | gpu_driven/*.rs | Caught alignment bug |
| P1 | frame_graph/*.rs | Complex state machine |
| P2 | buffer_*.rs | Memory safety critical |
| P2 | IK modules | Numeric precision |
| P3 | All remaining | Coverage |

---

## Estimates

| Task | Optimistic | Realistic | Pessimistic |
|------|------------|-----------|-------------|
| T-CONT-6.1 | 2h | 4h | 8h |
| T-CONT-6.2 | 4h | 8h | 16h |
| T-CONT-6.3 | 4h | 8h | 16h |
| T-CONT-6.4 | 8h | 16h | 32h |
| T-CONT-6.5 | 4h | 8h | 16h |
| T-CONT-6.6 | 4h | 8h | 16h |
| T-CONT-6.7 | 4h | 8h | 16h |
| T-CONT-6.8 | Ongoing | Ongoing | Ongoing |
| **Total (core)** | **30h** | **60h** | **120h** |
