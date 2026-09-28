 # Analytic Fisher derivatives, and the cosmolike noise test (experimental)

Status: **experimental draft, not compiled by any project.** Everything
here was built and run on 2026-09-27 against lsst_y1 (cosmic shear, NLA).

## Why this code exists

The Fisher forecasts of the EXAMPLE_EVALUATE notebooks differentiate the
data vector by finite differences (5-point stencil, CAMB + cosmolike
re-run at every step), and the figure of merit moved with the step size
and with AccuracyBoost ("we need high accuracy boost to do Fisher"). The
question was: **does the cosmolike engine itself — its interpolation
tables, quadratures and log-ell grids — add too much noise to
finite-difference derivatives?**

To answer it, this folder builds a derivative that never differences
cosmolike: the Limber C_ss and xi_pm are differentiated analytically
under the integral (chain rule), fed only by the response of CAMB's
tables to the parameter. Comparing the two isolates cosmolike's
contribution from CAMB's.

## Answer

**No. Cosmolike adds no measurable noise; the noise is CAMB's halofit.**

- Along perfectly smooth inputs, cosmolike's finite differences converge
  as s^2 (3-point) and s^4 (5-point stencil) to the analytic derivative,
  e.g. d lnC/d lnA_s: 6.7e-13 at s = 1e-3 and 2.1e-6 at s = 6e-2 with the
  5-point stencil. Pure truncation, no noise.
- CAMB's nonlinear P(k) jitters by 5e-3 in ln P between neighbouring
  cosmologies, in whole redshift slices, at every CAMB accuracy setting
  (AccuracyBoost 1-3, CAMBAccuracyBoost 2, k_per_logint 40). The linear
  P(k) is clean to 1e-14. Cause: `fortran/halofit.f90:322` stops the
  bisection for the nonlinear scale at |sigma(R) - 1| <= 1e-3.
  The patch `cocoa_installation_libraries/camb_changes/camb/halofit.patch`
  (1e-3 -> 1e-7) removes it (5e-3 -> 3e-5) at no measurable cost; Cocoa
  applies it when `PATCH_CAMB_HALOFIT_TOLERANCE=1` is set in
  `set_installation_options.sh` (off by default).
- The notebook Fisher reproduced (17 parameters, 5-point stencil,
  AccuracyBoost 1, FoM(A_s, Omega_m) at h = 0.01, 0.02, 0.03, 0.05, 0.08):

  | case | FoM |
  |---|---|
  | stock CAMB (the notebook) | 205.8, 159.0, 179.4, 140.3, 153.1 |
  | stock CAMB, CAMB AccuracyBoost 3 | 205.0, 158.5, 180.7, 140.4, 151.3 |
  | cosmolike only (smooth synthetic inputs) | 152.71, 152.71, 152.71, 152.72, 152.80 |
  | CAMB with halofit tolerance 1e-7 | 146.0, 145.8, 145.6, 145.6, 147.1 |

  With the patch the finite-difference Fisher is stable at
  AccuracyBoost 1, and AccuracyBoost 2 and 3 give the same answer
  (145.2, 145.1 at h = 0.03): the notebook's "we need high accuracy
  boost" was CAMB's jitter. Every measurement is in
  [FINDINGS.md](FINDINGS.md).

So the practical fix for the notebooks is the CAMB patch (the
`PATCH_CAMB_HALOFIT_TOLERANCE` key; the Fisher cells of the lsst_y1 and
roman_real notebooks carry the warning), not the analytic machinery. The analytic draft stays here because it is the
natural adapter for differentiable Boltzmann codes (next section).

## Adapting to differentiable (JAX) Boltzmann codes

People are writing CAMB alternatives in JAX with automatic
differentiation (for example jax-cosmo, DISCO-EB, and neural emulators
such as CosmoPower-JAX). Such a code gives the four inputs this draft needs
**exactly**, in one forward-mode pass (`jax.jacfwd` over the parameter
vector), on the set_cosmology grids:

| input | meaning | from a JAX code |
|---|---|---|
| `dlnPNL_dX` | dlnP_NL/dX at fixed (k, z) | Jacobian of ln P_NL(k, z) |
| `dchi_dX` | dchi/dX at fixed z (Mpc/h) | Jacobian of chi(z) |
| `dlnG_dX` | dlnG/dX at fixed z | Jacobian of ln D(z) |
| `dlnOm_dX` | explicit dlnOmega_m/dX | 1/Omega_m for X = Omega_m, else 0 (Cocoa's basis) |

cosmolike then remains the fast C projection engine and returns exact
dC_ss/dX and dxi_pm/dX with no finite difference anywhere; the same chain
rule would give likelihood gradients for gradient-based samplers. One
caution: a JAX halofit that finds the nonlinear scale by bisection with a
fixed tolerance reproduces CAMB's jitter unless the root is tight or
differentiated implicitly.

## Contents

| file | role |
|---|---|
| `cosmo2d_fisher.c/.h` | C core: response storage; C_ss + dC_ss/dX (`_work` design, one pass for all parameters); xi_pm + dxi_pm/dX |
| `cosmo2d_fisher_wrapper.cpp/.hpp` | C++: grid-checked response setter, pybind wrappers (`generic_interface.cpp` needs no change) |
| `interface_bindings.cpp` | the `m.def` blocks a project adds to its `interface.cpp` |
| `fisher_camb_response.py` | the four inputs from CAMB by central differences |
| `doc/` | derivation and verification PDF, its LaTeX, the Python prototype |
| `FINDINGS.md` | every measurement, mechanism, and implementation note |
| `tests/validate_fisher_draft.py` | draft vs CAMB finite differences (C_ss, xi_pm) |
| `tests/validate_chain_rule_synthetic.py` | chain rule with CAMB noise removed |
| `tests/converge_step_size.py` | analytic and FD versus the step |
| `tests/cosmolike_fd_noise.py` | cosmolike's own FD noise at Fisher-sized steps |
| `tests/camb_jitter.py`, `tests/camb_jitter_slices.py` | CAMB P_NL jitter vs accuracy; where it lives |
| `tests/notebook_fisher_noise.py` | the notebook Fisher, split into CAMB and cosmolike contributions |

## Trying the draft

In a copy of a cosmic-shear project's `interface/` (lsst_y1 was used):

1. `MakefileCosmolike`: add
   `${ROOTDIR}/external_modules/code/cosmolike_core/future_port_unfinished/fisher/cosmo2d_fisher.c`
   to `CSOURCES` and `./cosmo2d_fisher.o` to `OBJECTC`; add
   `.../fisher/cosmo2d_fisher_wrapper.cpp` to `CPPSOURCES` and
   `./cosmo2d_fisher_wrapper.o` to `OBJECTCPP`.
2. `interface.cpp`: add the include and the blocks from
   `interface_bindings.cpp`.
3. Build with `make -f MakefileCosmolike all` and put that directory
   first on `sys.path` (the tests read it from `FISHER_BUILD_DIR`):

```python
from fisher_camb_response import get_camb_response
ci.reset_fisher_response()
ci.set_fisher_response(ip=0, **get_camb_response("As_1e9", 0.02, point, log=True, **camb_kwargs))
ci.set_fisher_response(ip=1, **get_camb_response("omegam", 0.01, point, **camb_kwargs))
C, dC = ci.dC_ss_dX_tomo_limber(l=ell)            # dC[ip] has the shape of C
xip, xim, dxip, dxim = ci.dxi_pm_dX_tomo()
```

The CAMB patch is installed through Cocoa: uncomment
`PATCH_CAMB_HALOFIT_TOLERANCE=1` in `set_installation_options.sh`, then
reinstall CAMB (`OVERWRITE_EXISTING_CAMB_CODE=1`, `setup_camb.sh`,
`compile_camb.sh`).

## Scope and open items

- NLA only (TATT excluded by design); cosmic shear only (no gs/gg).
- Omega_m/w0 derivatives carry a <= 4e-4 consistency floor (spline
  derivatives vs cosmolike's piecewise-linear interpolants); see
  FINDINGS.md section 3.
- Per-call setup dominates the ~40 ms (C) / ~90 ms (xi) cost.
