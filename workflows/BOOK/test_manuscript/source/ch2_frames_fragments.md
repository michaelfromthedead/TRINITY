# Chapter 2 — Reference Frames and Symmetry (fragments)

**Date:** November 2024
**Nature:** Rough fragments — collected notes and partial prose on reference frame invariance,
symmetry groups, and the SU(2) vs SO(3) distinction. Not a draft — more like organized notes.
**Status:** DRAFT (rough — fragments, not flowing prose)

---

## Fragment A: Why reference frames matter here

[written Nov 14]

The question of reference frames feels abstract until you connect it to something physical.
Here is the connection: spin is supposed to be a property of a particle that is independent
of how you describe the particle — independent of your coordinate system. But if you rotate
your coordinate system, what happens to the spin?

For orbital angular momentum, the answer is clear from classical mechanics. The angular
momentum vector rotates with the coordinate system. Rotate by 120 degrees, the angular
momentum vector rotates by 120 degrees.

For spin-1/2, the answer is stranger. Rotate the coordinate system by 360 degrees — which
should return you to the identical description — and the spin state picks up a minus sign.
It takes a 720-degree rotation to return the spin state to its original value.

This is the fact that SO(3) (the group of rotations in 3D space) cannot capture. SO(3) has
representations where a 360-degree rotation returns you to the starting point. Spin-1/2 does
not have this property. You need SU(2) — the group of 2x2 unitary matrices with determinant 1 —
and specifically you need the fundamental (spin-1/2) representation of SU(2).

SU(2) is a double cover of SO(3). For every element of SO(3), there are two elements of SU(2)
that project to it. This is the mathematical fact that underlies the physical fact: two different
SU(2) operations (differing by a sign) produce the same rotation of space, but they produce
different transformations of a spin-1/2 state.

---

## Fragment B: Historical note on the discovery

[written Nov 17]

The historical story here is interesting and worth including. The double-cover structure
of SU(2) was known to mathematicians long before spin was discovered. Cartan classified
all Lie groups and their representations in the early 20th century. The spin-1/2 representation
of SU(2) was a mathematical object before Goudsmit and Uhlenbeck (1925) proposed that electrons
have an intrinsic angular momentum corresponding to it.

This is a case where mathematical structure preceded physical application — and not by a little.
Cartan's work was roughly contemporaneous with the early quantum theory, but the mathematical
community and the physics community were not communicating well about this. When Pauli introduced
the Pauli matrices to describe spin, he was effectively constructing the spin-1/2 representation
of SU(2) from the physics side, without full awareness of the pre-existing mathematical framework.

The connection between Pauli's matrices and Cartan's representation theory was made explicit
by Weyl (1928) in his famous book on group theory and quantum mechanics. This is why the
Socratic approach to chapter 2 should ask: which came first, the mathematics or the physics?
The answer is genuinely interesting.

---

## Fragment C: The Lorentz group and why it matters for non-locality

[written December 2024]

For the purposes of this book's thesis — that a field-theoretic treatment of spin dissolves
the non-locality puzzle — we need more than just the rotation group. We need the full Lorentz
group: rotations and boosts.

The Lorentz group has two types of representations relevant to us. Scalar representations
(spin 0): these transform trivially under rotations and in a specific way under boosts. Vector
representations (spin 1): these are the familiar four-vectors of special relativity.
Spinor representations (spin 1/2): these are the two-component objects that transform under
the (1/2, 0) or (0, 1/2) representations of the Lorentz algebra.

The connection to non-locality is through the spin-statistics theorem. The fact that spin-1/2
particles are fermions — that their field operators anti-commute at space-like separations —
is a consequence of the Lorentz group representation structure of the spinor field. This
anti-commutativity is what ensures that the EPR correlations cannot be used for faster-than-light
communication: any measurement on the electron side commutes with any measurement on the
positron side (in the sense that the order of measurements doesn't affect the observable outcomes).

[NOTE: I need to be careful here. The argument is subtle and I want to make sure I have the
causal structure right before writing this as prose. The key claim is that space-like commutation
of observables follows from the Lorentz structure of the field theory. This needs careful exposition.]

---

## Fragment D: What SU(2) is, concretely

[written November 2024 — this should probably go earlier in the chapter]

SU(2) is the group of 2x2 complex matrices U such that U†U = I (unitary) and det(U) = 1
(special). Explicitly, every element of SU(2) can be written as:

U = [[a, -b*], [b, a*]]

where |a|^2 + |b|^2 = 1. The group is compact (a closed, bounded subset of the space of
2x2 matrices) and simply connected (unlike SO(3), which has a non-trivial fundamental group).

The generators of SU(2) (the elements of its Lie algebra su(2)) are:

T_i = sigma_i / 2

where sigma_i are the Pauli matrices:

sigma_1 = [[0, 1], [1, 0]]
sigma_2 = [[0, -i], [i, 0]]
sigma_3 = [[1, 0], [0, -1]]

These satisfy [T_i, T_j] = i epsilon_ijk T_k, which is the su(2) commutation relation (same
as the angular momentum algebra).

This is the algebra that will appear throughout the book. It is worth spending time on it here
so the reader has it as a reference point.

---

## Fragment E: The pedagogical ordering question

[written November 30, 2024]

A question I keep asking myself: should this chapter come before or after the Larmor precession
chapter?

The argument for before: the SU(2) structure motivates why spin is discrete. Once you know
that spin is a representation of SU(2), the two-state structure (up/down) follows. This is
the "right" logical order.

The argument for after: the Larmor precession is the concrete phenomenon. A reader who has
seen a spinning top precess, and then seen the quantum version of it (the Stern-Gerlach
experiment), has physical intuition for what spin is doing. The SU(2) story then explains
why the concrete thing behaves as it does.

Current decision (November 30): keep this chapter (reference frames and symmetry) before
Larmor precession. The reason: reference frames are needed to understand what it means for
spin to be a property that is independent of how you describe the system. Larmor precession
is a concrete phenomenon that the reader can observe (in principle) — but to understand what
it reveals about spin's mathematical nature, they need the frame-invariance idea first.

This decision may be revisited in the spring draft.

---

## References to add

Goudsmit, S. and Uhlenbeck, G. E. (1925) — spin of the electron
Weyl, H. (1928) — The Theory of Groups and Quantum Mechanics
Pauli, W. — original Pauli matrix paper (need to find exact citation)
Cartan, E. — his classification work (the specific paper on double cover)
