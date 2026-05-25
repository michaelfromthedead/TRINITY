# Spin Physics Research — Project Overview

## Research Context

This project investigates spin relaxation times in two-dimensional Heisenberg
antiferromagnets using stochastic Landau–Lifshitz–Gilbert (LLG) simulations.

## Simulation Approach

We numerically integrate the stochastic LLG equation on a square lattice
with periodic boundary conditions, using a Runge–Kutta scheme of order 4.
Thermal fluctuations are modelled as Gaussian white noise.

## Output Structure

Each simulation run produces:
- A trajectory file in HDF5 format under `data/runs/`
- A summary CSV appended to `data/summary.csv`
- Console logs captured by the `spin_physics` logger

## Reproducing Results

1. Copy `config/default.toml` to `config/run_YYYYMMDD.toml`
2. Edit lattice size and temperature
3. Run `python src/main.py --config config/run_YYYYMMDD.toml`
4. Analyse outputs with the notebook in `notebooks/`
