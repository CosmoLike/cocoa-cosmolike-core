# Covariance non-Limber source study

Status: source study, not an implementation or validation claim.
The covariance implementation must remain inside `cosmolike/covariances/`.

## Physics boundary

[Fang, Krause, Eifler & MacCrann (2019), Sections 2--3](https://arxiv.org/html/1911.11947)
separate a coherently growing linear field from a nonlinear correction
projected with Limber. Their Eqs. 10--12 rearrange the unequal-distance
linear projection into two single-Bessel transforms and one k integral.
The same construction gives every internal field pair required by Wick's
theorem; exclusion from the measured data vector is no reason to omit a
cross-bin spectrum. [DES Y6, Appendix F](https://arxiv.org/html/2503.13631v1)
explicitly identifies non-Limber covariance as a cross-tomographic improvement.

For a dimensionless density contrast projected with a window W of inverse
distance, the density transfer is integral dlnchi [chi W D] j_l(k chi).
A shear transfer instead has the spin-2 angular derivative factor
sqrt((l-1)l(l+1)(l+2))/k^2 and radial integrand W D/chi. Lensing
magnification uses l(l+1)/k^2 with its own radial window. These factors
must agree with the existing harmonic convention before any additional
real-space source-leg conversion is applied.

Retaining non-Limber spectra in the Gaussian contractions is distinct from
removing the long-mode Limber approximation in SSC or the projected
equal-time approximation in cNG. Do not claim those latter extensions
merely because the Gaussian spectra change. The existing joint selected
cluster one-halo spectrum remains part of its nonlinear correction.

## Actual data-vector implementation inspected

The current `cosmo2D.c` pipeline is `cfftlog_ells_p1`,
`cfftlog_ells_p2`, then `C_cl_tomo_core`/`C_gs_tomo_core`.
The inspected source provides these reusable optimization ideas:

- Transform each active radial component forward once. That transform
  does not depend on l and is reused by all multipoles and field pairs.
- Construct FFTW plans serially. Reuse them when transform dimensions are
  unchanged; execute with each worker's own new-array buffers.
- Use even transform lengths factorizable into 2, 3, 5 and 7, with zero
  guards. Keep the guards' logarithmic extent fixed under grid refinement.
- Process inverse transforms in blocks of 16 multipoles, keeping scratch
  bounded. The covariance adaptation should distribute fields AND block
  multipoles together, rather than launch one team for each field.
- Compute the gamma ratio explicitly for two seed multipoles, then use
  Gamma(z+1)=z Gamma(z) separately for even and odd multipoles.
- Share the reciprocal k grid and k^3 P(k) among all field pairs. The
  current galaxy--shear code already moves these reads outside pair sums.
- Replace repeated phase trigonometry with a complex phase recurrence,
  periodically recomputed from the exact phase to bound accumulated error.
- Precompute powers of the reversed distance grid. SIMDe normalizes
  several adjacent inverse-transform samples without repeated pow calls.
- Keep complete per-output sums on one worker. A covariance adaptation
  can use SIMD lanes for distinct field pairs, retaining deterministic
  summation order when the OpenMP worker count changes.

## Covariance-specific decisions still requiring tests

The data-vector code uses a lens-bin pivot spectrum and freezes individual
pairs when their correction is small relative to that pair's Limber value.
A covariance contains sign-changing and nearly zero cross spectra, so that
relative stopping criterion must not be copied without testing. Shared
field transfers and a common positive k measure also make the linear
field matrix a Gram matrix. Preserve that consistency when choosing the
growth anchor, subtraction and transition to Limber.

The forward and inverse linear terms must use the same separable growth
convention. Subtracting a differently evolved linear spectrum would leave
an artificial high-l residual. A common growth anchor needs comparison
against direct radial integration and the supplied linear-power tables;
massless neutrinos alone do not prove perfectly scale-independent growth.

Refine uniform log-distance grids by adding intervals without moving old
nodes. Test padding, radial limits, multipole cutoff and interpolated power
separately. Analytic Gaussian-Bessel integrals provide normalization and
phase checks; direct oscillatory quadrature provides a separate check for
catalog windows, including narrow/discontinuous cluster selections.
Check all field-pair spectra, matrix positivity, thread determinism and
the complete projected covariance. No eigenvalue clipping or diagonal
jitter may hide a failed physical or numerical construction.
