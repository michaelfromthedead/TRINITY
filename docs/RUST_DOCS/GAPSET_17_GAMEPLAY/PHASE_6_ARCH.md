# Phase 6: AI Tier 3 — MCTS Solver — Architecture

## Status: NOT IMPLEMENTED

**All 8 tasks are absent.** No MCTS module exists in the codebase. This document describes the planned architecture.

## Planned Architecture

### Core Tree

```
MCTSNode
├── State reference
├── Visit count
├── Total score / Q-value
├── Children (indexed by action)
├── Parent reference
└── Unexpanded actions
```

### Four Phases (Standard MCTS)

```
Selection
├── UCB1 formula: Q + C * sqrt(ln(N) / n)
├── C = exploration constant (configurable)
└── Traverse until leaf node

Expansion
├── On first visit: expand all legal actions
├── One child per action
└── Expansion only once per node

Simulation (Playout)
├── Random action selection (default)
├── Pluggable heuristic policy (domain-specific)
└── Terminal state reached

Backpropagation
├── Result propagated to root
├── Each node: visits += 1, score += result
└── Q = score / visits
```

### Iteration Budget

```
MCTSSolver
├── Configurable: time-based (ms) or count-based
├── After budget: most-visited child = best action
└── NOT highest-scored — most-visited
```

### Integration

```
Domain Adapter (Combat)
├── GameState interface: legal actions, terminal conditions, reward
├── Actions: movement, attack, defend, use ability
├── Terminal: all enemies dead, time elapsed
└── Reward function

Fallback Chain (Auto-Tier Selection)
├── Try MCTS first (if budget sufficient)
├── Fallback to HTN (if applicable)
├── Fallback to GOAP
├── Fallback to Utility AI
├── Fallback to BT (reliable baseline)
└── Configurable per agent
```

## Key Files (Planned)

| File | Purpose |
|------|---------|
| `ai/mcts/mcts.py` | MCTSNode, UCB1, four phases |
| `ai/mcts/solver.py` | Iteration budget, best-action selection |
| `ai/mcts/combat_adapter.py` | Domain adapter for combat |
| `ai/mcts/fallback.py` | AI tier fallback chain |

## Dependencies

- Phase 4 (GOAP) for fallback chain
- Phase 3 (Combat AI) for domain adapter
- Phase 5 (Navigation) for movement actions
