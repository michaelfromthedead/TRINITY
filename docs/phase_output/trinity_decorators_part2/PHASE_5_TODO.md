# PHASE 5 TODO: Social, Economy & Analytics Systems

## Overview

Validate social, economy, achievements, analytics, and narrative decorators.

---

## T5.1: Validate Social Decorators

**File**: `trinity/decorators/social.py`

**Tasks**:
- [ ] Verify `@social` marks social integration
- [ ] Verify `@leaderboard` accepts name and sort order
- [ ] Verify `@leaderboard` validates scope (global, friends, regional)
- [ ] Verify `@shareable` marks content for sharing
- [ ] Verify `@presence` enables online presence

**Leaderboard Fields**:
- [ ] name - leaderboard identifier
- [ ] sort - "asc" or "desc" (default: desc)
- [ ] scope - "global", "friends", "regional"

**Acceptance Criteria**:
- All 4 decorators produce correct steps
- All register in "social" registry
- Full ops chain implemented

---

## T5.2: Validate Economy Decorators

**File**: `trinity/decorators/economy.py`

**Tasks**:
- [ ] Verify `@currency` validates type against VALID_CURRENCY_TYPES
- [ ] Verify `@transaction` logs economic transactions
- [ ] Verify `@mtx` marks microtransaction points
- [ ] Verify `@daily_reward` configures reward schedule

**Currency Types**:
- [ ] "soft" - earnable in-game
- [ ] "hard" - purchasable
- [ ] "premium" - exclusive currency
- [ ] "seasonal" - time-limited

**Acceptance Criteria**:
- All 4 decorators follow 6-part pattern
- Currency type validation produces actionable error
- Transaction and mtx mark sensitive operations
- All register in "economy" registry

---

## T5.3: Validate Achievements Decorators

**File**: `trinity/decorators/achievements.py`

**Tasks**:
- [ ] Verify `@achievement` accepts name and points
- [ ] Verify `@achievement` supports hidden flag
- [ ] Verify `@progress` tracks completion percentage
- [ ] Verify `@stat` accumulates numeric values

**Achievement Fields**:
- [ ] name - achievement identifier
- [ ] points - gamerscore/trophy points
- [ ] hidden - show/hide until unlocked

**Acceptance Criteria**:
- All 3 decorators produce correct steps
- Progress uses Op.TRACK for updates
- All register in "achievements" registry

---

## T5.4: Validate Analytics Decorators

**File**: `trinity/decorators/analytics.py`

**Tasks**:
- [ ] Verify `@telemetry` validates consent_level
- [ ] Verify consent levels align with GDPR requirements
- [ ] Verify `@funnel` tracks conversion steps
- [ ] Verify `@heatmap` collects spatial data

**Consent Levels**:
- [ ] "essential" - required for operation
- [ ] "functional" - improves experience
- [ ] "analytics" - usage tracking
- [ ] "marketing" - advertising

**GDPR Compliance Check**:
- [ ] Consent level validation enforced at decoration time
- [ ] Invalid consent level produces actionable error
- [ ] Consent level stored in metadata

**Acceptance Criteria**:
- All 3 decorators follow 6-part pattern
- Consent validation is GDPR-aware
- All register in "analytics" registry

---

## T5.5: Validate Narrative Decorators

**File**: `trinity/decorators/narrative.py`

**Tasks**:
- [ ] Verify `@dialogue` accepts id and speaker
- [ ] Verify `@dialogue` supports emotion parameter
- [ ] Verify `@conversation` tracks conversation state
- [ ] Verify `@voice_over` references audio assets

**Dialogue Fields**:
- [ ] id - dialogue node identifier
- [ ] speaker - character speaking
- [ ] emotion - emotional context (default: neutral)

**Acceptance Criteria**:
- All 3 decorators produce correct steps
- Dialogue tree configuration complete
- All register in "narrative" registry

---

## Summary

| Task | File | Decorators | Lines |
|------|------|------------|-------|
| T5.1 | social.py | 4 | 233 |
| T5.2 | economy.py | 4 | 228 |
| T5.3 | achievements.py | 3 | 197 |
| T5.4 | analytics.py | 3 | 195 |
| T5.5 | narrative.py | 3 | 192 |

**Total**: 17 decorators, 1,045 lines
