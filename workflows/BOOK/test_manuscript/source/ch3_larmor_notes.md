# Chapter 3 — Larmor Precession and Spin (technical notes)

**Date:** January–February 2025
**Nature:** Dense technical notes. Not prose. Facts, derivations, key claims.
**Status:** NOTES_ONLY — these are the notes I will write from, not the chapter itself.

---

## Key physical phenomenon: the spinning top

Starting point for the chapter: a gyroscope (or spinning top) in the Earth's gravitational field.
It doesn't fall. It precesses. The axis traces a cone around the vertical. Rate of precession
depends on the angular momentum L and the applied torque tau: omega_precession = tau / L_sin_theta.

This is the CLASSICAL phenomenon. The quantum analog is the spin precession in a magnetic field.
An electron with spin S in a magnetic field B experiences a torque: tau = mu × B where
mu = -g_s (e/2m_e) S is the magnetic moment. The precession rate is the Larmor frequency:

omega_L = g_s (eB) / (2m_e)

For the electron, g_s ≈ 2 (the anomalous g-factor). This will be noted but not derived —
deriving g_s requires QED which is beyond scope.

---

## Key claim: spin is NOT classical rotation

I want to be emphatic about this in the chapter. Spin is not a little ball spinning on its axis.
The key evidence:

1. The size argument: if the electron were a classical spinning ball with angular momentum hbar/2,
   and the electron has a known upper bound on its size (~10^-18 m from scattering experiments),
   then the surface velocity would exceed c. The classical rotating-ball picture fails numerically.

2. The topological argument: under a 360-degree rotation of the coordinate system, a spin-1/2
   state picks up a phase of -1 (not +1 as a classical object would). This is the SU(2) double-
   cover fact. A 720-degree rotation is needed to return to the original state. No classical object
   does this.

3. The Stern-Gerlach argument: a classical spinning ball in an inhomogeneous magnetic field would
   be deflected by a continuous range of angles (depending on the orientation of its spin axis).
   The Stern-Gerlach experiment shows only TWO spots — quantized deflections. This is not what
   a classical object does.

---

## Derivation: Larmor frequency from the equation of motion

Start with the torque equation for a magnetic dipole in a field:
dL/dt = tau = mu × B

For an electron: mu = -g_s (e / 2m_e) S = -gamma S
where gamma = g_s (e / 2m_e) is the gyromagnetic ratio.

So: dS/dt = -gamma S × B = gamma B × S

This is a precession equation. For B = B_0 hat_z:
dS_x/dt = gamma B_0 S_y
dS_y/dt = -gamma B_0 S_x
dS_z/dt = 0

Solution: S_x(t) = S_perp cos(omega_L t + phi)
          S_y(t) = S_perp sin(omega_L t + phi)
          S_z(t) = S_z(0) = constant

where omega_L = gamma B_0 = g_s (e B_0) / (2 m_e)

This is the Larmor precession. The spin vector precesses around the magnetic field direction at
the Larmor frequency.

---

## Quantum spin: the two-state system

Quantum mechanically, spin-1/2 is described by a two-dimensional complex Hilbert space.
The basis states are |up> and |down> along some chosen axis (conventionally z).

The spin operators along each axis are:
S_x = (hbar/2) sigma_x
S_y = (hbar/2) sigma_y
S_z = (hbar/2) sigma_z

Eigenvalues of S_z: +hbar/2 (spin-up) and -hbar/2 (spin-down).

The Stern-Gerlach experiment: a beam of electrons passes through an inhomogeneous magnetic
field gradient dB_z/dz. The force on each electron is F_z = (mu_z)(dB_z/dz). Because mu_z
is quantized (two values), only two deflections appear on the screen: two spots.

---

## The 360-degree sign flip

The key mathematical fact about spin-1/2: the rotation operator for spin-1/2 is

R(theta, hat_n) = exp(-i theta hat_n · S / hbar) = exp(-i theta hat_n · sigma / 2)

For a 360-degree rotation about any axis:
R(2pi, hat_n) = exp(-i pi hat_n · sigma) = -I

That minus sign. The identity matrix times -1. A 360-degree rotation returns the POSITION
of the electron to its original location, but multiplies the spin state by -1. (The overall
phase -1 is not physically observable for a single spin, but it becomes observable in
interference experiments.)

For a 720-degree rotation:
R(4pi, hat_n) = exp(-2 pi i hat_n · sigma) = +I

720 degrees to return to the original state. This is the signature of a spin-1/2 (i.e., SU(2))
representation. Integer-spin representations (SO(3) representations) return to the original
state after 360 degrees.

---

## Connection back to the Lorentz group (preview)

[Note: this section will preview the synthesis chapter. Larmor precession is the concrete
phenomenon; the Lorentz group is the abstract structure that explains it. Mention:

- The boost generators and rotation generators together generate the Lorentz group
- The spin-1/2 spinor is the (1/2, 0) representation of the Lorentz algebra
- The precession rate (Larmor frequency) has a relativistic correction (Thomas precession)
  which requires the full Lorentz group to understand

Keep this brief. The synthesis chapter will develop it.]

---

## Notes on what the Stern-Gerlach experiment teaches us

The Stern-Gerlach experiment was designed (Gerlach and Stern 1922) to test whether atomic
magnetic moments are quantized. If they are quantized, the beam splits into discrete spots.
If they are continuous (classical), the beam smears into a band.

The original experiment used silver atoms. Silver has a single unpaired electron in the outer
shell with spin-1/2. The result: two spots. Quantization confirmed.

The key lesson for this book: the quantization of the spin component along ANY axis is the
physical manifestation of the mathematical structure of SU(2). The algebra forces the eigenvalues.
The concrete experimental fact is a consequence of the abstract mathematical structure — but
we should approach it the other way for pedagogical purposes: start with the experiment, extract
the structure.

---

## References

Gerlach, W. and Stern, O. (1922). Der experimentelle Nachweis der Richtungsquantelung im
Magnetfeld. Zeitschrift für Physik 9, 349–352.

Larmor, J. (1897). On the theory of the magnetic influence on spectra. Philosophical Magazine
44, 503–512.

Sakurai, J. J. and Napolitano, J. (2017). Modern Quantum Mechanics. 3rd ed. Cambridge.
[Chapter 1 on spin — good comparison for level and notation]

---

## Things I still need to figure out

- How much of the Thomas precession to include? It's relevant but requires more SR background
  than I want to assume. Probably a footnote or brief appendix note.

- The g-factor: g_s ≈ 2 is the Dirac equation prediction; g_s ≈ 2.002 is the QED correction.
  I should mention both but not derive either from scratch.

- The spin-echo and NMR applications: I could use these as concrete applications of Larmor
  precession in the modern world. Would make the chapter more practical. Tentatively yes.
