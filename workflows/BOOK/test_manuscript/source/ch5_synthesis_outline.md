# Chapter 5 — Synthesis: Spin, Field, and Gravity (outline)

**Date:** April 2025
**Nature:** Outline only. This is the last chapter and I haven't written it yet.
  The physics is clear to me but I haven't found the right opening.
**Status:** OUTLINE_ONLY — structure + 1-line descriptions per section, no prose.

---

## Chapter goal

Deliver the book's thesis. Connect spin formalism (CH_01–CH_04) to quantum field theory.
Show that spin is a Lorentz group label, not just a quantum number. Argue that the graviton
must be spin-2. Return to the EPR non-locality puzzle from CH_01 and show how the field-theoretic
picture addresses it. End with open research questions.

---

## Section outline

### 5.1 From quantum number to field representation
Introduce quantum field theory perspective: particles are not objects, they are quanta of fields.
Each field transforms under a specific representation of the Lorentz group. The representation
labels the spin.
  - Scalar field (spin 0): Klein-Gordon equation. Higgs boson.
  - Vector field (spin 1): Maxwell equations (photon). Yang-Mills fields (W, Z, gluons).
  - Spinor field (spin 1/2): Dirac equation. Electrons, quarks.
  - Tensor field (spin 2): linearized Einstein equations. Graviton (hypothetical).

This reframes everything: spin is not a property that a particle has; it is the label of the
Lorentz representation that the particle's field transforms under.

### 5.2 Why the graviton must be spin 2
The argument from force-mediation:
  - Massless fields mediate long-range forces.
  - Spin-0 (scalar) mediates a force with no polarization dependence — attractive for like charges.
  - Spin-1 (vector) mediates an electromagnetic-like force — attracts opposite charges, repels like.
  - Gravity attracts all mass-energy, regardless of sign. This eliminates spin-0 and spin-1.
  - Spin-2 (tensor) mediates a force that is universally attractive. This matches gravity.
The graviton, if it exists, is the quantum of a spin-2 field.

Note: this argument is heuristic, not a derivation. The actual argument requires Weinberg's
soft-limit analysis or the Lorentz invariance constraints on the S-matrix. Keep this at the
intuitive level.

### 5.3 The spin-2 field and linearized gravity
Linearized general relativity: decompose the spacetime metric as g_mu_nu = eta_mu_nu + h_mu_nu
where h_mu_nu is a small perturbation. The field equations for h_mu_nu (the linearized Einstein
equations) describe the propagation of the graviton.

The graviton has two polarization states (helicity +2 and -2). This is the spin-2 analog of
the two polarization states of the photon (helicity +1 and -1).

The coupling of the graviton field to matter is through the stress-energy tensor: the interaction
term is h_mu_nu T^mu_nu. This is why gravity couples to all forms of energy — the stress-energy
tensor includes all energy-momentum contributions.

### 5.4 Return to non-locality: what QFT tells us
Come back to the EPR puzzle from CH_01. The field-theoretic answer:
  - Spin is a representation label of the Lorentz group. The entangled state (EPR singlet) is
    a state of the composite system with total J = 0.
  - In QFT, the commutator of field operators at space-like separations vanishes (for bosons) or
    anti-commutes (for fermions). This is required by Lorentz invariance.
  - The commutation structure means that measurements on space-like-separated particles do not
    interfere with each other — the ordering of measurements is irrelevant to the statistics.
  - The EPR correlations are real and non-local in the sense that you cannot explain them
    by pre-existing hidden variables. But they are also non-signaling: you cannot use the
    correlations to send information faster than light. This is guaranteed by the Lorentz
    structure of the QFT.

So: QFT does not eliminate the non-locality; it situates it in a framework where the non-locality
cannot be exploited for signaling. The "mystery" of EPR is that it seems to imply non-local
correlations that exist without a non-local mechanism — and QFT shows why this is not a
contradiction: the Lorentz group structure of field theory is exactly the structure that
enforces causality while permitting non-local correlations.

### 5.5 Open research questions
End with genuine open questions:
  - Quantum gravity: we have a spin-2 graviton in linearized theory, but a full quantum theory
    of gravity does not exist. What replaces linearized gravity at strong fields?
  - The interpretation of the EPR correlations: is the field-theoretic account of non-locality
    truly satisfying, or does it just push the mystery to the structure of the vacuum state?
  - The spin-statistics theorem: we have argued that spin-1/2 particles are fermions from
    analogy and from Dirac's equation, but a rigorous proof of the spin-statistics theorem
    requires the full apparatus of axiomatic QFT (Wightman axioms). Why does the universe
    obey spin-statistics?
  - Beyond linearized gravity: do spin effects survive into strong-field regime? What is the
    spin of the Kerr black hole (the rotating solution of GR) in a quantum sense?

---

## Opening I want to find

The chapter needs an opening that connects back to CH_01. Something like: "We began with a puzzle.
Two particles, separated across a laboratory, with correlated spins. We have now built the
machinery — spin, symmetry, the representation theory of the rotation group, angular momentum —
that lets us understand the structure behind the puzzle."

Then: set up the field-theoretic reframing.

I need a good concrete opening — NOT a declaration. Something to show first. Maybe: the graviton
polarizations? The spinning black hole? The two slit experiment with entangled particles? Need
to decide.

---

## References I need to include
- Weinberg, S. (1964). Photons and gravitons in S-matrix theory. Physical Review 135, B1049.
  (The soft-limit argument for why graviton must be spin-2)
- Haag, R. (1992). Local Quantum Physics. Springer. (Axiomatic QFT, spin-statistics)
- Bell, J. S. (1987). Speakable and Unspeakable in Quantum Mechanics. Cambridge.
  (Bell's own reflections on non-locality and what QFT says about it)
- Wald, R. M. (1984). General Relativity. University of Chicago Press.
  (Linearized gravity, spin-2 field)
