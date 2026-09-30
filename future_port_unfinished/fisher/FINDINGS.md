# Detailed findings (2026-09-27)

Every number below was measured in this session with the scripts in
`tests/` (lsst_y1: 5 source bins, 15 pairs, NLA; halofit; CAMB settings of
the notebook wrappers unless stated). They are recorded for whoever
implements or revisits this idea; `README.md` has the summary.

## 1. The math (Python prototype, `doc/fisher_proto.py`)

Two Gaussian source bins, NLA, Takahashi halofit, inputs = central
differences of CAMB tables on fixed grids, truth = central difference of
the full C_ell with the same step:

| X | worst \|analytic/FD - 1\|, ell = 30-3000, 3 pairs |
|---|---|
| ln A_s | 5.7e-5 |
| Omega_m | 7.8e-5 |
| w0 | 1.1e-4 (no parameter-specific code) |

These are the FD truncation level. Table-only response models (no input
P response) fail: A_s via the amplitude-redshift degeneracy
(d_z lnP_NL / d_z lnP_L) is off by -1.0%, +2.8%, +13.8%, +34.6% at
ell = 30, 300, 1000, 3000 (halofit depends on Omega_m(z), not only on
sigma8(z)); Omega_m via Gamma-shape shift + growth ODE is off by -12%
to +27%. So dlnP_NL/dX(k, z) must be an input.

## 2. The C draft reproduces the likelihood

`dC_ss_dX_tomo_limber` C vs the likelihood's C_ss: max rel diff 1.1e-15;
`dxi_pm_dX_tomo` xi vs `xi_pm_tomo`: 5.0e-14 (xi+), 1.6e-15 (xi-). Same
Gauss-Legendre rule, limits, kernels and interpolants.

## 3. Chain rule with CAMB noise removed (`validate_chain_rule_synthetic.py`)

Perturbed cosmologies are synthetic: every set_cosmology table moved
exactly along the loaded response (lnP_NL +- sR, chi +- s chi_X,
lnG +- s lnG_X, Omega_m(1 +- s eps)). Max \|analytic/FD - 1\| over 15 pairs:

| X | s | ell=30 | 100 | 300 | 1000 | 3000 | xi+ | xi- |
|---|---|---|---|---|---|---|---|---|
| ln A_s | 1e-3 | 4.4e-10 | 5.9e-10 | 4.7e-10 | 4.4e-10 | 3.4e-9 | 3.3e-7 | 3.1e-7 |
| ln A_s | 1e-4 | 1.3e-11 | 1.8e-11 | 9.5e-12 | 1.1e-11 | 3.9e-11 | 3.3e-9 | 3.2e-9 |
| Omega_m | 1e-3 | 1.8e-5 | 5.1e-5 | 3.1e-5 | 3.5e-6 | 5.3e-6 | 1.8e-5 | 1.5e-5 |
| Omega_m | 1e-4 | 1.7e-5 | 5.4e-5 | 3.2e-5 | 2.3e-6 | 7.3e-6 | 1.4e-6 | 3.2e-6 |
| w0 | 1e-3 | 8.1e-5 | 3.8e-4 | 2.1e-4 | 1.6e-5 | 4.8e-5 | 1.6e-5 | 3.0e-5 |
| w0 | 1e-4 | 1.3e-4 | 3.8e-4 | 2.1e-4 | 1.6e-5 | 4.8e-5 | 1.6e-5 | 3.0e-5 |

The P-response path (A_s) is exact to rounding. The step-independent
floors for Omega_m (5e-5) and w0 (4e-4) are consistency, not noise: the
analytic side uses spline derivatives (cubic chi_X', bicubic
dlnP/dlnk) while cosmolike's C is built on piecewise-linear interpolants
(linear chi with interpolated finite-difference slopes, bilinear lnP,
linear g_tomo coarse table). Differentiating cosmolike's own discrete
interpolants instead would remove the floor.

## 4. Does cosmolike add noise to finite differences? (`cosmolike_fd_noise.py`)

Along the same smooth synthetic path, finite differences of cosmolike's
C_ss and xi+ at Fisher-sized steps (3-point and the notebook's 5-point
stencil), against the analytic derivative. Max over ell x pairs:

| X | s | C_ss 3pt | C_ss 5pt | xi+ 3pt | xi+ 5pt |
|---|---|---|---|---|---|
| ln A_s | 1e-3 | 3.6e-7 | 6.7e-13 | 3.0e-7 | 5.8e-12 |
| ln A_s | 1e-2 | 3.6e-5 | 1.6e-9 | 3.0e-5 | 1.2e-9 |
| ln A_s | 6e-2 | 1.3e-3 | 2.1e-6 | 1.1e-3 | 1.5e-6 |
| Omega_m | 1e-3 | 4.2e-5 | 5.4e-5 | 1.3e-5 | 3.3e-7 |
| Omega_m | 1e-2 | 2.0e-3 | 5.9e-5 | 1.3e-3 | 1.2e-6 |
| Omega_m | 6e-2 | 7.4e-2 | 4.9e-3 | 4.7e-2 | 2.2e-3 |
| w0 | 1e-3 | 3.8e-4 | 3.9e-4 | 3.8e-6 | 4.7e-6 |
| w0 | 1e-2 | 3.4e-4 | 3.3e-4 | 4.7e-6 | 2.3e-6 |
| w0 | 6e-2 | 3.6e-4 | 7.5e-5 | 2.6e-4 | 3.7e-6 |

(Omega_m and w0 steps are absolute: s = 6e-2 in Omega_m is a 20% step.)
Errors scale as s^2 (3-point) and s^4 (5-point) down to the section-3
consistency floor: textbook truncation, no noise. **Cosmolike does not
add noise to finite-difference derivatives.**

## 5. Against real CAMB (`validate_fisher_draft.py`, `converge_step_size.py`)

With the same step for the response and the finite difference, analytic
and FD agree to <= 6e-4 at ell <= 1000 (ln A_s, Omega_m, w0, C_ss and
xi+). But both move with the step. dlnC/dlnA_s at ell = 1000, pair (4,4):

| step | 0.0025 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|
| analytic | 1.2357 | 1.4968 | 1.3974 | 1.4353 |
| FD | 1.2357 | 1.4966 | 1.3974 | 1.4352 |

One-sided derivatives disagree by up to ~20% (pair (2,2): forward 1.30
vs backward 1.49 at ell = 1000, 1.05 vs 1.27 at ell = 3000). Pair (0,0) at ell = 3000 shows a
step-independent ~1.2% analytic-FD offset: the analytic weights use the
fiducial CAMB table, the central FD effectively the mean of the +- tables,
and those differ by the CAMB jitter below. It disappears on the synthetic
path (4e-11), and switching IA off does not remove it (0.9%).

## 6. The noise is CAMB's halofit (`camb_jitter.py`, `camb_jitter_slices.py`)

jitter = max \|lnP(X0) - (lnP(X0+s) + lnP(X0-s))/2\| (s = 0.01 in ln A_s,
0.01 < k < 10 h/Mpc, z < 1.5):

| CAMB setting | P_lin | P_NL |
|---|---|---|
| AccuracyBoost 1 (default) | 1.1e-14 | 5.0e-3 |
| AccuracyBoost 2 | 8.9e-15 | 5.0e-3 |
| AccuracyBoost 3 | 9.8e-15 | 5.0e-3 |
| CAMBAccuracyBoost 2 | 7.1e-15 | 5.0e-3 |
| k_per_logint 40 | 1.1e-14 | 5.0e-3 |
| **halofit tolerance 1e-7** (patch) | — | **2.9e-5** (ln A_s), 3.4e-4 (Omega_m, s = 0.003) |

- The linear spectrum is clean; the nonlinear one is not, and no accuracy
  knob helps.
- The jitter is coherent in whole redshift slices (z = 0.34, 0.51, 0.86,
  1.03: ~2.5e-3 across every nonlinear k; z = 0, 0.17, 0.69, 1.20: ~1e-5).
- Mechanism: `fortran/halofit.f90:322` stops the bisection for the
  nonlinear scale at \|sigma(R) - 1\| <= 1e-3; the stopping point jumps with
  the cosmology and shifts the whole one-halo regime of that slice.
- Fix: `cocoa_installation_libraries/camb_changes/camb/halofit.patch`
  (1e-3 -> 1e-7), applied by `setup_camb.sh` when
  `PATCH_CAMB_HALOFIT_TOLERANCE=1`; the remaining 2.9e-5
  is the genuine curvature (s^2/2) d^2lnP/dlnAs^2. Cost: 4.3 -> 4.4 s per
  CAMB run. Verified in a scratch CAMB build (link with
  RECOMBINATION_FILES="recfast cosmorec", as compile_camb.sh does; the
  plain `setup.py build` silently falls back to the old camblib.so when
  the Fortran link fails, so check the library timestamp).
- CAMB is deterministic at identical inputs (difference 0.0), so caching
  CAMB outputs by argument is valid.

## 7. The notebook Fisher, reproduced (`notebook_fisher_noise.py`)

lsst_y1 EXAMPLE_EVALUATE1 "Fisher based on 5-stencil rule": 17
parameters, relative step h, Gaussian priors, FoM(A_s, Omega_m):

| case | h = 0.01 | 0.02 | 0.03 | 0.05 | 0.08 |
|---|---|---|---|---|---|
| (a) the notebook: stock CAMB, AccuracyBoost 1 | 205.8 | 159.0 | 179.4 | 140.3 | 153.1 |
| (b) stock CAMB at CAMB AccuracyBoost 3, cosmolike at 1 | 205.0 | 158.5 | 180.7 | 140.4 | 151.3 |
| (c) cosmolike only (synthetic path) | 152.71 | 152.71 | 152.71 | 152.72 | 152.80 |
| (d) patched CAMB (halofit tolerance 1e-7), AccuracyBoost 1 | 146.0 | 145.8 | 145.6 | 145.6 | 147.1 |

- sigma(A_s) swings 0.38-0.56 in (a) and stays 0.538-0.544 in (d);
  sigma(Omega_m) 0.01267-0.01354 in (a), 0.012885-0.012889 in (d).
- (e) With the patch, AccuracyBoost stops mattering: FoM at h = 0.03 is
  145.6, 145.2, 145.1 at AccuracyBoost 1, 2, 3. The notebook's "we need
  high accuracy boost" was CAMB jitter, not accuracy.
- (b) shows raising CAMB's accuracy does not help (and costs ~4 min per
  Fisher instead of ~15 s).
- (c) moves the five cosmological parameters along synthetic tables
  (fiducial + t x a stock-CAMB response): flat to 6e-4 over an 8x range
  of steps, so cosmolike contributes no step dependence. Its level
  (152.7) differs from (d) because its derivatives are linearized once
  from a stock-CAMB response, i.e. one realization of the jitter.
- Timing per Fisher (17 parameters, CAMB cached at identical inputs):
  (a) 10-18 s, (b) 118-245 s, (d) 12-22 s; (e) 48 s at AccuracyBoost 2,
  92 s at 3.
- A helper bug found by (c): for X = H0 the response re-gridding clamped
  the k edges (np.interp), corrupting the edge slope that cosmolike's
  linear extrapolation multiplies at the high-k tail xi+ reads; the
  synthetic path then exploded at large steps (FoM 236 and 593 at
  h = 0.05, 0.08). Fixed in `fisher_camb_response.py` (linear
  extrapolation); the numbers above are after the fix.

## 8. Implementation notes for a future port

- **Inputs on the fiducial grids.** The C++ setter checks every node. For
  X = h, `get_camb_cosmology` shifts the k grid to h/Mpc after evaluating,
  so perturbed tables must be re-interpolated before differencing
  (`fisher_camb_response.py` does it), extrapolating linearly past the
  edges — clamping corrupts the response where cosmolike extrapolates
  (section 7).
- **eps_X = dlnOmega_m/dX** carries the explicit Omega_m of the lensing
  prefactor and of the NLA amplitude; in Cocoa's basis it is 1/Omega_m for
  Omega_m and 0 for everything else (h included: h units; m_nu included:
  Omega_m contains the neutrinos). A basis sampling omega_m gives
  eps_h = -2/h.
- **Growth.** cosmolike's D = a G(z)/G(0) is normalized at z = 0, so
  dlnD/dX(z) = dlnG/dX(z) - dlnG/dX(0).
- **z splines** stop at z = 10 (+4 guard nodes): the chi table jumps from
  z ~ 50 to the recombination block at z ~ 1070 and a global spline
  through that gap is meaningless.
- **Beyond the k table** p_nonlin extrapolates lnP linearly from the last
  bracket; the response lookup must extrapolate the same way (derivative
  of the extrapolation = extrapolation of the derivative).
- **Thread safety.** GSL splines are evaluated with NULL accelerators
  (binary search), which makes the read-only parallel evaluation safe.
- **Lensing efficiency.** dg/dX = -chi_X Q + chi R, integrated on g_tomo's
  own fine grid and stored on its coarse grid, so it is the derivative of
  the g the likelihood interpolates.
- **Scope.** NLA only: the TATT one-loop kernels would need FAST-PT FFTs of
  the response inside cosmolike. Cosmic shear only (gs/gg would add galaxy
  bias, magnification and the lens kernels).
- **Cost.** C + 3 derivatives ~40 ms, xi + 3 derivatives ~90 ms, dominated
  by the per-call bicubic spline of lnP_NL and the lensing-efficiency
  response; cache them on cosmology.random if it matters.

## 9. Unrelated bug found on the way

`cosmo2D_scuts.c`, `dlnxi_dlnk_pm_tomo_nointerp`: the normalization loop
calls `xi_pm_tomo(p, ...)` with p = 0 for the xi+ row, but in
`xi_pm_tomo` pm = 1 selects xi+ — so dlnxi+/dlnk is divided by xi- and
vice versa. RF_xi is unaffected (a constant normalization cancels) and
the notebooks plot it normalized; the raw values are off by xi+/xi-. Fix:
pass `1 - p`.
