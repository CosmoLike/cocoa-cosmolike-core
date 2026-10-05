# Covariance angular-bin and Fourier-band operators

`operators_cov.c/.h` adds geometry-only operators for the existing supplied-
matrix contractions. The real-space output rows are xi+, xi-, gamma_t and w,
with angular bin as the second index and integer ell as the final index.
The Fourier output averages inclusive integer bands with exact mode counts.
These functions do not select a cosmology, mask, spectrum or survey default.

## Spin convention and bin measure

For unit-normalized observed fields, the four angular kernels are the small
rotation-matrix elements d22, d2,-2, d20 and d00, multiplied by
(2 ell+1)/(4 pi). Their Jacobi-polynomial representation is checked
independently against the finite factorial rotation sum at 60 digits.
The tangential sign agrees with the positive associated-Legendre convention
in [Friedrich et al., Appendices A/B](https://arxiv.org/html/2012.08568v3).

The input spectrum must use the same field convention as its noise. Matching
the existing core real-space transformation requires multiplying each source
leg of its supplied spectrum by sqrt[(ell-1)(ell+2)/(ell(ell+1))]. That is
an explicit conversion when matching those transforms, not a factor to apply
again to a spectrum already defined for observed shear. White shape noise
remains sigma_component^2/n. The survey-level spectrum/estimator convention
still needs its separate validation; these operators cannot establish it.
Xi+ uses EE+BB and xi- uses EE-BB; the caller supplies the appropriate signs.

The angular measure is sin(theta) dtheta divided by the bin's cosine width.
Half-angle sine/cosine factors preserve small spin kernels without subtracting
two nearly equal antiderivatives. Tabulated Gauss-Legendre quadrature integrates
the bin; this is **numerical averaging**, not an exact endpoint formula.
The node count is explicit and must be refined for the requested widest bin
and ell cutoff. Mask-dependent pair-separation weighting is not included.

Spin rows vanish at ell=0,1. The scalar row retains both modes so that any
estimator's low-mode removal remains explicit. Its signal and analytic noise
must use consistent mode treatment; a driver must not silently drop only one.

The Fourier operator uses (2 ell+1)/N_band, where
N_band=(ell_last-ell_first+1)(ell_last+ell_first+1). It permits overlapping
bands, whose covariance is correspondingly nonzero. A constant Gaussian
Wick numerator projects to that numerator divided by fsky*N_band. A constant
cNG matrix stays constant: no second mode-count division is allowed.
This follows [Krause & Eifler, Appendix A](https://arxiv.org/html/1601.05779v1).

## Numerical checks

Six focused checks pass with the isolated optimized and debug libraries:

- Low-degree bin averages versus independent 60-digit factorial rotation
  matrices, including a tiny xi- with no absolute-tolerance escape.
- Bin edges at zero and pi and a 1e-14-radian-wide bin near theta=1e-8.
- Independent SciPy polynomial evaluations through ell=50000 in all twenty
  logarithmic 2.5–250 arcmin bins: maximum absolute error divided by harmonic
  mode density is 3.262e-12.
- Angular refinement from 256 to 512 nodes per bin, at every integer ell
  through 50000: maximum on that same scale is 2.839e-11. This does not bound
  relative error at kernel zeros or certify a complete covariance ell cutoff.
- Exact discrete Fourier mode counts, overlapping bands, the Gaussian
  mode-count limit and constant connected covariance.
- Repeated 1/4/8-thread calls and a changed/restored geometry agree bitwise;
  sentinel columns check padded output rows.

The one-ulp difference between reciprocal multiplication and direct division
in the Fourier weights is allowed explicitly (5e-16 relative tolerance).
There is no scalar production switch. C uses SIMDe for node recurrence and
averaging, with each complete output assigned to one OpenMP worker. The
Jacobi coefficients follow [DLMF 18.9.1–2](https://dlmf.nist.gov/18.9).

External reproduction files are `operators_reference.py`, `build_operators.sh`
and `benchmark_operators.py` in `test/covariance_reference/`. LSST's opt-in
`test_covariance_operators.py` needs `COSMOLIKE_COVARIANCE_REFERENCE` and
`COSMOLIKE_OPERATORS_LIBRARY`. Logs are `results/operators*_tests.log`.
The operator library is isolated; no project interface is relinked by this
ticket. The seven-project regression run preceding it remains unchanged.

## Geometry-build timing

Apple M2 Pro, strict IEEE Clang 19.1.7, with three warm-ups and eleven
measured calls per thread count. All four probes, twenty 2.5–250 arcmin
bins, ell=0..50000, and 512 angular nodes per bin:

| Threads | Mean ± sample standard deviation (ms) |
|---:|---:|
| 1 | 1123.931 ± 6.752 |
| 4 | 484.894 ± 7.471 |
| 8 | 253.610 ± 5.040 |

C coefficient construction, internal allocations and angular averages are
included; Python output allocation, cosmology and covariance projection are
excluded. The resulting geometry can be retained across cosmologies. Desktop
applications remained active, but no project test jobs ran concurrently.
Raw results are `results/operators_baseline_timing.json`. Native disassembly
in `results/operators_simd.disassembly.txt` confirms two-lane multiply and
fused arithmetic in the recurrence. These are not full covariance timings.

## Separate didactic red-eye pass

After optimized/debug tests passed, reread the complete C/header and the
reference/test entry points as one ticket. The function documentation now
distinguishes observed shear from the core transform convention, derives the
area measure and integer mode weights, explains the Jacobi recurrence and
its two rolling rows, and identifies every SIMD lane and worker-owned row.
Small-angle tests retain a zero absolute tolerance wherever a nonzero signal
must survive. Every C/header line fits 80 columns and each predicate occupies
its own line. This is a manual self-review; Fable was unavailable.

The full rewrite remains open. These tested operators do not replace the
remaining non-Limber, mask, survey-accuracy, layout and output gates.
