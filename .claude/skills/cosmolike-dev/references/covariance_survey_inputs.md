# Covariance radial inputs and all-pairs Limber spectra

**Accuracy policy update (2026-10-03):** historical mentions below of a
1e-6 production refinement gate are superseded by
[the FoM-based protocol](covariance_accuracy.md). The measurements are
retained; small-entry relative errors are diagnostics, not a universal
production blocker.

The first survey integration ticket adds `spectra_cov.c/.h` and the small
`generic_interface_cov.cpp/.hpp` binding in `cosmolike/covariances/`.
Only LSST links the binding at this stage. No existing core C file changed.
This is a measured intermediate implementation, not a frozen production
covariance contract or a completed non-Limber survey generator.

## Inputs and physics

`covariance_limber_spectra` returns every lens/source field pair, including
overlapping lenses and pairs excluded from the measured data vector.
All pairs use common positive radial quadrature weights. The spectrum is
a sum of field outer products, which preserves positive semidefiniteness
without rejecting signed magnification or NLA terms. Lens bias is linear;
IA is NLA. Source and magnification efficiencies currently require flat
geometry. The multipole factors follow the core harmonic convention;
observed-shear noise and the real-space spin conversion remain separate.

Radial quadrature panels and cumulative-efficiency samples are explicit
covariance inputs; no data-vector precision knob is changed. The source
and lens efficiencies use their own cumulative trapezoids, factorized as
`g = A - chi B`. The core's foreground cutoff (`a > 1-dac`) is deliberately
absent. Reusing that cutoff initially produced a quadrature step with a
4.39e-5 source-spectrum difference when doubling 256 to 512 nodes.
The new efficiency agrees with an independent direct source integral to
3e-6 absolute in the four-catalog test. Its 4097-node default is an input
sampling default, not a certified survey accuracy setting.

RSD, when requested, belongs to each lens field in every pair, including
galaxy-shear. A pair-dependent RSD choice would lose the common-field
positive-semidefinite construction. This entry is **Limber only** at every
multipole. It does not silently substitute for non-Limber spectra.

## Independent checks and numerical limits

The external `covariance_reference/` holds the pinned massless, flat
two-lens/two-source configuration, its CAMB archive, an independent NumPy
power interpolation and matrix contraction, and direct lensing integrals.
The source and lens bins overlap; magnification has both signs; number
densities and per-component shape dispersions are explicitly recorded.
The raw cap mask and preliminary SSC/cNG node inputs are documented in the
[projection diagnostic](covariance_projection_measurements.md).

Six LSST checks pass in the optimized build. They cover linear/nonlinear
power and RSD choices, all ten field pairs, units, positive semidefiniteness,
signed foregrounds, quadrature refinement, lensing efficiencies, shape
validation and bitwise repeated/1/4/8-thread agreement. The NumPy
contraction uses the supplied radial windows: it validates their use, not
an independent IA or redshift-distribution model. The separate direct
lensing test checks the efficiency construction itself.

Doubling 256 to 512 radial nodes per panel changes a spectrum by as much
as 1.5014e-5 relative to its corresponding auto-spectrum scale; 512 to
1024 gives 4.64e-6 in the exploratory run. This does **not** meet the
study's full 1e-6 refinement gate. Do not freeze settings or infer a
delta-chi-squared bound from these preliminary checks.

The reproducible `compare_core_spectra.py` compares six multipoles from
2 to 50000, three source pairs, four lens-source pairs and two lens autos.
It matches the core's RSD choices for this diagnostic. Maximum relative
differences are 6.3182e-4 (ss), 2.5452e-4 (gs), 1.4701e-5 (gg). They are
not an optimization drift claim: covariance owns different efficiency
and integration grids and retains the foreground interval.

## Measured implementation

Power reads are batched at fixed scale factor, so the redshift lookup
is shared by all multipoles. Full field windows are precomputed before
the pair loop. Two SIMDe lanes accumulate two independent pair spectra
in the same radial order; OpenMP owns complete outputs. There is no
cross-thread reduction, threaded BLAS or persistent covariance cache.

Apple M2 Pro, eight OpenMP threads, strict optimized build: two warm-ups
and 21 timed calls, CAMB excluded; means and sample standard deviations.
These times include radial setup, power/window reads, allocations and
Python output copies. They are not full covariance times.

| Multipoles | 1 thread (ms) | 4 threads (ms) | 8 threads (ms) |
|---:|---:|---:|---:|
| 64 | 13.806 ± 0.314 | 4.031 ± 0.085 | 2.746 ± 0.146 |
| 512 | 101.515 ± 0.568 | 27.141 ± 0.158 | 14.949 ± 0.939 |

An external scalar comparison replaces only the pair reduction with
ordered scalar `fma`; no production switch was added. The 512-multipole
SIMDe/scalar means are 100.962/102.236 ms (one thread), 27.024/27.164 ms
(four), 14.327/14.432 ms (eight). Outputs agree bitwise. The gain is small
because power and RSD preparation dominate. Generated native assembly
contains two-lane `fmul`/`fmla` in the pair loop and cumulative integrals.
Linux hardware performance counters were unavailable on this host.

Raw artifacts: `spectra_timing.json`, `spectra_loop_comparison.json`,
`spectra_core_comparison.json`, and `spectra_simd.s` under the external
`covariance_reference/results/` directory.

## Separate didactic red-eye pass

After the six focused tests passed, reread the full C/header and binding
as one ticket. This is a manual self-review; Fable was unavailable.
The comments now derive the cumulative geometry, distinguish density
from lensing/IA, explain the positive radial measure and field-matrix
construction, identify every SIMD lane and worker's output, and state
which arrays own memory. All C/header lines fit within 80 columns and
guards have one predicate per line. The endpoint branch and RSD distance
guard stop unsupported inputs with an explanation. Remaining accuracy
and physics limits are stated rather than hidden by recovery logic.

This pass was complete before starting the next SSC ticket. The isolated
debug LSST library passes all six new checks plus ten existing example and
non-Limber regressions (16 total), with UBSan/float-divide-by-zero enabled.
All seven project suites also pass: 405 passes with 21 external-library
skips. Enabling all covariance libraries passes 34 focused tests together,
including those 21, for 426 distinct tests. No references were refrozen.
