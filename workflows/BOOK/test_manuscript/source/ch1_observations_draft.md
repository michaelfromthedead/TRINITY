# Chapter 1 — Why Non-Locality (draft)

**Date:** October 2024 (revised January 2025)
**Nature:** Partial draft — first three sections complete, fourth section outline only.
**Status:** DRAFT — prose is rough but structurally present.

---

## Section 1: The EPR Observation

Imagine two particles produced together in a quantum process — say, an electron and a positron
created in a pair-production event — that are then separated across a large distance. The particles
are prepared in an entangled spin state: their total spin is zero, so whatever spin the electron
has, the positron has the opposite.

Now measure the electron's spin along some axis. You find it is, say, spin-up. According to
quantum mechanics, the positron's spin along the same axis is now determined — it is spin-down.
Not because some signal traveled from the electron to the positron. Not because the positron
somehow "checked" with the electron. The correlation is there, immediately, regardless of
the separation between the particles.

This is the EPR phenomenon. Einstein, Podolsky, and Rosen described it in 1935 as a challenge
to quantum mechanics (Einstein, Podolsky, and Rosen 1935). They argued that if quantum mechanics
is complete, then this non-local correlation implies that information travels faster than light.
But information doesn't travel faster than light. Therefore, they concluded, quantum mechanics
must be incomplete — there must be "hidden variables" that determine the spin outcomes in advance.

Bell (1964) showed that any hidden-variable theory that reproduces the quantum predictions
must exhibit correlations that violate a specific inequality — Bell's inequality. And Aspect,
Grangier, and Roger (1982) demonstrated experimentally that nature violates Bell's inequality.
The EPR correlations are real, and they are not explained by hidden variables.

So where does this leave us? Quantum mechanics describes the phenomenon correctly. The formalism
works. But the question of why spin correlations are non-local — in what sense "non-local,"
what physical mechanism produces the correlation — is not answered by the formalism alone.

This is the puzzle this book is organized around.

---

## Section 2: What Quantum Mechanics I Tells Us (and Doesn't)

Standard quantum mechanics (what I'll call QM I — the wave-function formalism, the measurement
postulate, the Hilbert space framework) describes spin as an internal degree of freedom of
a particle. A spin-1/2 particle has two spin states: spin-up and spin-down along any chosen axis.
The state space is a two-dimensional complex vector space. The spin operators are represented
by the Pauli matrices.

This is completely correct as far as it goes. And it goes far — far enough to compute atomic
energy levels to impressive precision, to predict the Zeeman effect, to describe the behavior
of spin in magnetic fields.

But QM I is a framework for describing quantum states and their evolution. It is not a framework
for describing how physical fields propagate, how interactions are mediated, or why correlations
between separated systems have the structure they do. For that, we need quantum field theory.

The explanatory gap is this: QM I tells us that the two particles are entangled, and that
measuring one determines the other. It does not tell us why the spin correlation structure
is what it is — why the correlations are non-local in a way that nonetheless cannot be used
for faster-than-light communication. The field-theoretic treatment of spin, through the
structure of the Lorentz group and the representation theory of spin, provides the framework
in which this question can be properly asked and partially answered.

---

## Section 3: The Field-Theoretic Resolution Thesis

The thesis of this book is the following: a field-theoretic treatment of spin — treating spin
as a label on irreducible representations of the Lorentz group, rather than as an abstract
internal quantum number — places spin correlations in their proper physical context and dissolves
several of the apparent paradoxes that QM I leaves unresolved.

This is not a claim that quantum field theory is "more true" than quantum mechanics. It is a
claim that quantum field theory is the right framework for the questions that the EPR phenomenon
raises. The non-locality of spin correlations is not a mystery within QFT — it is a consequence
of the structure of quantum fields, of the fact that space-like-separated field operators commute
(or anti-commute for fermions), and of the entangled preparation of the particles.

We will build toward this thesis gradually. The book begins with the concrete case — spin as
a physical phenomenon, observable in Larmor precession and in the Stern-Gerlach experiment —
and moves toward the abstract: spin as a Lorentz representation, the graviton as a spin-2 field,
the coupling of spin to spacetime geometry.

---

## Section 4: The Pedagogical Contract and Chapter Ordering [OUTLINE ONLY]

[Note: this section needs to be written. Key points to make:

- State explicitly: intuition before formalism. Every chapter begins with a physical phenomenon
  or observation, not with a definition or theorem.
  
- State explicitly: spin before angular momentum. Standard textbooks reverse this (angular
  momentum is more general; teach it first, then specialize to spin). This book reverses the
  standard ordering because the concrete case (spin: two discrete states, Larmor precession)
  makes the abstract case (angular momentum: a continuous group with infinitely many representations)
  feel like a generalization of something the reader already holds, rather than an imposition
  of unfamiliar structure.
  
- Feynman diagrams will appear in this book as intuition-pump tools, not computational devices.
  A reader who has learned Feynman diagrams as perturbation-theory shorthand should be aware
  of this distinction. Chapter 3 will address it explicitly.
  
- Brief chapter-by-chapter overview: non-locality puzzle → reference frames and symmetry →
  Larmor precession and spin → angular momentum as the general case → synthesis and open questions.
  
- End with a genuine question for the reader: "Why does the structure of spin correlations
  respect the speed of light even though the correlations themselves appear instantaneous?"
  This is what the book will work toward.]

---

## References (notes — not formatted)

- Einstein, A., Podolsky, B., & Rosen, N. (1935). Can quantum-mechanical description of physical
  reality be considered complete? Physical Review 47, 777–780.
  
- Bell, J. S. (1964). On the Einstein-Podolsky-Rosen paradox. Physics 1(3), 195–200.

- Aspect, A., Grangier, P., & Roger, G. (1982). Experimental realization of EPR-Bohm experiment.
  Physical Review Letters 49, 91.
  
- I should also include the original EPR response papers and Bell's subsequent papers. Will
  compile proper bibliography when draft is more complete.
