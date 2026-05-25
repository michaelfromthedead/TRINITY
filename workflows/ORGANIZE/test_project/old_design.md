# Old Design — v0.1 Architecture (SUPERSEDED)

SUPERSEDED BY docs/overview.md — kept for reference only.

## Original Plan (January 2026)

We originally planned to use a Monte Carlo approach rather than LLG integration.
This was abandoned after profiling showed the Metropolis step dominated wall time.

## Abandoned Classes

- `MCEngine` — Metropolis Monte Carlo sweeper
- `SpinCluster` — Wolff cluster update (never finished)
- `EnergyAccumulator` — incremental energy tracker

All of these were deleted in commit a3f9e12. This document records why.

## Lesson

For continuous spin models, LLG with RK4 outperforms MC at moderate temperatures.
See Leliaert et al. (2018) for a comparison benchmark.
