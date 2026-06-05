# PHASE 5 ARCHITECTURE: Social, Economy & Analytics Systems

## Scope

Player engagement and monetization decorators:
- `social.py` (233 lines)
- `economy.py` (228 lines)
- `achievements.py` (197 lines)
- `analytics.py` (195 lines)
- `narrative.py` (192 lines)

## Architecture Pattern: Full Ops Chain

These files implement complete ops chains for player-facing systems with careful validation of sensitive parameters (currencies, transactions, consent).

## Component: Social Decorators

**File**: `trinity/decorators/social.py` (233 lines)

**Decorators**: `@social`, `@leaderboard`, `@shareable`, `@presence`

**Architecture**:
- Social integration markers
- Leaderboard configuration
- Content sharing metadata
- Online presence tracking

**Pattern**:
```python
def _leaderboard_steps(params):
    return [
        Step(Op.TAG, {"key": "leaderboard", "value": params["name"]}),
        Step(Op.TAG, {"key": "leaderboard_sort", "value": params.get("sort", "desc")}),
        Step(Op.TAG, {"key": "leaderboard_scope", "value": params.get("scope", "global")}),
        Step(Op.REGISTER, {"registry": "social"}),
    ]
```

## Component: Economy Decorators

**File**: `trinity/decorators/economy.py` (228 lines)

**Decorators**: `@currency`, `@transaction`, `@mtx`, `@daily_reward`

**Architecture**:
- Currency type validation
- Transaction logging
- Microtransaction markers (mtx)
- Daily reward scheduling

**Validation**:
```python
VALID_CURRENCY_TYPES = frozenset({"soft", "hard", "premium", "seasonal"})

def _validate_currency(currency_type=None, **_):
    if currency_type not in VALID_CURRENCY_TYPES:
        raise ValueError(
            f"@currency: invalid type '{currency_type}'. "
            f"Valid types: {sorted(VALID_CURRENCY_TYPES)}"
        )
```

## Component: Achievements Decorators

**File**: `trinity/decorators/achievements.py` (197 lines)

**Decorators**: `@achievement`, `@progress`, `@stat`

**Architecture**:
- Achievement definition
- Progress tracking
- Stat accumulation

**Pattern**:
```python
def _achievement_steps(params):
    return [
        Step(Op.TAG, {"key": "achievement", "value": params["name"]}),
        Step(Op.TAG, {"key": "achievement_points", "value": params.get("points", 0)}),
        Step(Op.TAG, {"key": "achievement_hidden", "value": params.get("hidden", False)}),
        Step(Op.REGISTER, {"registry": "achievements"}),
    ]
```

## Component: Analytics Decorators

**File**: `trinity/decorators/analytics.py` (195 lines)

**Decorators**: `@telemetry`, `@funnel`, `@heatmap`

**Architecture**:
- Consent level validation (GDPR compliance)
- Funnel tracking for conversion
- Heatmap data collection

**Consent Levels**:
```python
VALID_CONSENT_LEVELS = frozenset({"essential", "functional", "analytics", "marketing"})

def _validate_telemetry(consent_level=None, **_):
    if consent_level not in VALID_CONSENT_LEVELS:
        raise ValueError(
            f"@telemetry: invalid consent_level '{consent_level}'. "
            f"Valid levels: {sorted(VALID_CONSENT_LEVELS)}"
        )
```

## Component: Narrative Decorators

**File**: `trinity/decorators/narrative.py` (192 lines)

**Decorators**: `@dialogue`, `@conversation`, `@voice_over`

**Architecture**:
- Dialogue tree configuration
- Conversation state tracking
- Voice-over asset references

**Pattern**:
```python
def _dialogue_steps(params):
    return [
        Step(Op.TAG, {"key": "dialogue", "value": params["id"]}),
        Step(Op.TAG, {"key": "dialogue_speaker", "value": params.get("speaker")}),
        Step(Op.TAG, {"key": "dialogue_emotion", "value": params.get("emotion", "neutral")}),
        Step(Op.REGISTER, {"registry": "narrative"}),
    ]
```

## Op Types Used

| Op | Purpose | Files |
|----|---------|-------|
| `Op.TAG` | Store social/economy metadata | All |
| `Op.REGISTER` | Register in "social", "economy", "analytics" | All |
| `Op.TRACK` | Track achievement progress | achievements.py |
| `Op.VALIDATE` | Validate consent levels | analytics.py |

## Key Decisions

1. **Currency type validation**: Prevents invalid currency types in economy system
2. **Consent level validation**: GDPR compliance for analytics
3. **Registry separation**: social, economy, achievements, analytics, narrative registries
4. **Progress tracking**: Op.TRACK for achievement progress updates
5. **Sensitive data handling**: Transaction and mtx decorators mark sensitive operations
