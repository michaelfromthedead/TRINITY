# rough notes — preface framing

**Date:** September 2024
**Nature:** Rough notes on what this book is trying to do. Not a preface draft — framing notes for my own reference as I start writing.

---

## what this book is and isn't

This isn't a textbook. I'm not trying to teach spin the way Sakurai or Griffiths teaches it.
The goal is different: I want the reader to understand why spin is strange, where the strangeness
comes from, and why the field-theoretic treatment is not just a formalism upgrade but actually
dissolves the explanatory problem that QM I leaves unresolved.

The non-locality problem. That's the entry point. The EPR correlations are real — Aspect's
experiment confirms them. QM I describes the correlations correctly. But it doesn't tell us why
spin correlations are non-local in a way that respects special relativity. The field-theoretic
treatment does. That's the arc.

## the reader

Graduate student who has done QM I. They know: wave functions, superposition, the measurement
postulate, spin-1/2 as an abstract two-state system (they've done the Pauli matrices), and
they've heard of Feynman diagrams. They probably don't know: why SU(2) and not SO(3), what
a spinor actually is (not just a two-component thing), or what the Lorentz group has to do
with spin.

I am NOT assuming they've done QFT. I'm teaching the minimal QFT they need.

## the pedagogical contract

Show before tell. Always. Never state a theorem before showing the phenomenon that motivates it.
The Larmor precession chapter must start with a physical observation — the spinning top precessing
under gravity — before we touch an equation.

The ordering is deliberate: spin before angular momentum. Standard textbooks do it the other way
(angular momentum is more general, so introduce it first, then specialize to spin). That's pedagogically
backwards. The reader needs the concrete case (spin) to make the abstract case (angular momentum)
feel like a generalization rather than an imposition.

## feynman diagrams

I want to use them as intuition tools only. Not as a computational framework. The reader coming
from QM I may have learned Feynman diagrams as "things you draw to compute amplitudes." I need
to establish early that in this book they're doing something different — they're showing what
happens physically, not what you compute.

This distinction matters for chapter 2 or 3 — wherever I introduce the field-theoretic framing.

## notes on notation

I'll use SI units throughout. Reduced Planck constant will be hbar throughout (never h/2pi written
out except on first introduction). Pauli matrices will be sigma_i. The spin operator S = (hbar/2) sigma.
Angular momentum will be L, total angular momentum J = L + S.

---

## rough idea for book arc (as of September 2024)

1. Why non-locality? (motivation, EPR, field-theoretic resolution thesis)
2. Reference frames and symmetry (why SU(2) vs SO(3) — the double cover story)
3. Larmor precession (concrete spin phenomenon — physical before formal)
4. Angular momentum as general case (fulfilling the spin-first promise)
5. Synthesis: spin as Lorentz representation, graviton as spin-2, open questions

This ordering may change. The key constraint: spin (ch3) must come before angular momentum (ch4).
Everything else is flexible.

---

## open questions as of September 2024

- Do I need a chapter on the Dirac equation, or can I get away with an extended section in ch4?
  Current leaning: extended section. A full Dirac chapter would require more QFT prerequisite
  than I want to assume.

- How much differential geometry does the spin-2 field discussion require? I don't want to
  teach GR. I want the reader to understand linearized gravity well enough to see where the
  spin-2 coupling comes from. This needs to be carefully calibrated.

- The synthesis chapter (ch5) is the hardest to write. By definition it needs everything else
  to be done first. I'm leaving it as an outline until the other chapters are mature.
