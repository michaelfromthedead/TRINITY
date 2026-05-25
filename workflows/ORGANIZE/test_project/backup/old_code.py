# backup — old MC engine (archived 2026-02-03)
# This file is kept for historical reference only. Do not import.

class MCEngine:
    """Metropolis Monte Carlo sweeper — abandoned in favour of LLG."""
    def __init__(self, lattice_n, temperature):
        self.n = lattice_n
        self.T = temperature
