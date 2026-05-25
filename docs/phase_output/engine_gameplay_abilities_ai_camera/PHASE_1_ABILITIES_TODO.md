# PHASE 1 TODO: Abilities Subsystem

**Phase**: 1 of 3
**Subsystem**: engine/gameplay/abilities
**Status**: Investigation Complete

---

## 1. Verification Tasks

### 1.1 Attribute System
- [ ] **T-ABIL-1.1**: Verify `_recalculate()` applies modifiers in correct order
- [ ] **T-ABIL-1.2**: Test dirty flag invalidation on modifier add/remove
- [ ] **T-ABIL-1.3**: Test derived attribute dependency tracking
- [ ] **T-ABIL-1.4**: Test AttributeSet iteration and bulk operations
- [ ] **T-ABIL-1.5**: Test clamping to min/max bounds

### 1.2 Effect System
- [ ] **T-ABIL-2.1**: Verify InstantEffect applies and completes in one frame
- [ ] **T-ABIL-2.2**: Test DurationEffect expiration after duration
- [ ] **T-ABIL-2.3**: Test InfiniteEffect persists until explicit removal
- [ ] **T-ABIL-2.4**: Test PeriodicEffect tick accumulation
- [ ] **T-ABIL-2.5**: Test EffectContainer lifecycle (add/tick/remove)
- [ ] **T-ABIL-2.6**: Test effect stacking rules

### 1.3 Targeting System
- [ ] **T-ABIL-3.1**: Test SelfTargeting returns only self
- [ ] **T-ABIL-3.2**: Test ActorTargeting respects range and filter
- [ ] **T-ABIL-3.3**: Test PointTargeting with valid/invalid positions
- [ ] **T-ABIL-3.4**: Test AreaTargeting CIRCLE shape geometry
- [ ] **T-ABIL-3.5**: Test AreaTargeting CONE shape geometry
- [ ] **T-ABIL-3.6**: Test AreaTargeting RECTANGLE shape geometry
- [ ] **T-ABIL-3.7**: Test AreaTargeting LINE shape geometry
- [ ] **T-ABIL-3.8**: Test AreaTargeting CAPSULE shape geometry
- [ ] **T-ABIL-3.9**: Test ConfirmationTargeting confirm/cancel flow

### 1.4 Gameplay Tags
- [ ] **T-ABIL-4.1**: Test exact tag matching
- [ ] **T-ABIL-4.2**: Test single wildcard matching (`ability.*.fire`)
- [ ] **T-ABIL-4.3**: Test trailing wildcard matching (`ability.*`)
- [ ] **T-ABIL-4.4**: Test GameplayTagContainer has_tag/has_any/has_all
- [ ] **T-ABIL-4.5**: Test GameplayTagQuery all_of/any_of/none_of
- [ ] **T-ABIL-4.6**: Test registry LRU cache behavior

---

## 2. Integration Tasks

### 2.1 Trinity Pattern Integration
- [ ] **T-ABIL-5.1**: Register Attribute with ComponentMeta
- [ ] **T-ABIL-5.2**: Install TrackedDescriptor on Attribute.current_value
- [ ] **T-ABIL-5.3**: Register Effect types with ComponentMeta
- [ ] **T-ABIL-5.4**: Create EffectAppliedEvent with EventMeta
- [ ] **T-ABIL-5.5**: Create EffectRemovedEvent with EventMeta

### 2.2 Foundation Integration
- [ ] **T-ABIL-6.1**: Connect attribute changes to Tracker
- [ ] **T-ABIL-6.2**: Connect effect events to EventLog
- [ ] **T-ABIL-6.3**: Register ability types with Registry

---

## 3. Future Enhancements (Out of Scope)

### 3.1 Cooldown Manager
- Cooldown tracking per ability
- Global cooldown support
- Cooldown reduction calculations

### 3.2 Network Replication
- Attribute sync protocol
- Effect prediction and reconciliation
- Lag compensation

### 3.3 Editor Tooling
- Visual effect designer
- Attribute curve editor
- Tag hierarchy browser

---

## 4. Acceptance Criteria

| Task Group | Criteria |
|------------|----------|
| Attribute | All modifier orders tested, caching verified |
| Effect | All 4 types lifecycle tested |
| Targeting | All 5 shapes geometry verified |
| Tags | Wildcard matching 100% correct |
| Integration | Trinity metaclasses/descriptors wired |
