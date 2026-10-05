# Raw-mask pair geometry for pure-noise covariance

**Accuracy policy update (2026-10-03):** historical mentions below of a
1e-6 production refinement gate are superseded by
[the FoM-based protocol](covariance_accuracy.md). The measurements are
retained; small-entry relative errors are diagnostics, not a universal
production blocker.

`mask_cov.c/.h` computes ordered-pair angular areas from the same raw mask
power spectrum used by SSC. The scalar-bin operator retains every mask mode,
including its monopole and dipole. These are footprint modes, irrespective
of low-mode removal in a cosmological estimator.

For the scalar operator K_L averaged in a bin with cosine width Delta_x,

    A_pair = 8 pi^2 Delta_x sum_L C_L^mask K_L.

This follows the spherical-harmonic addition theorem and
[Friedrich et al., Appendix C, Eqs. 104–108](https://arxiv.org/html/2012.08568v3).
It is an ordered-pair area in sr^2; multiplying by two number densities in
sr^-1 gives the expected unclustered pair count. It can be passed directly
to `gaussian_noise_pair_cov`, which supplies the catalog and shear-component
factors. An additional unordered-pair factor would double count them.

The normalization guard requires C_0=Omega^2/(4 pi). No division by C_0,
Omega or f_sky is performed. A full-sky footprint returns 8 pi^2 Delta_x.
The production noise contract assumes a common binary mask and uniform
noise within it. Different catalog footprints and general object weights
need their appropriate pair counts; this routine does not infer them.
Nonpositive reconstructed pair areas fail explicitly rather than being clipped.

## Independent geometry and resolution checks

Three tests pass in isolated optimized and debug builds:

- Full-sky pair areas and the auto-clustering Gaussian factor against
  closed forms, with zero/pi endpoints and narrow bins.
- A 12300 deg^2 cap against a 60-digit direct spherical-intersection
  calculation. The reference uses two spherical sectors minus two triangles,
  then integrates the overlap over separation; it uses no mask harmonics.
- Bitwise 1/4/8-thread outputs, a changed/restored cap, odd bin counts and
  sentinel-protected output arrays.

For all twenty 2.5–250 arcmin bins, maximum relative errors against direct
cap geometry are:

| Mask L_max | Maximum pair-area error |
|---:|---:|
| 1024 | 1.5770e-4 |
| 4096 | 1.0496e-5 |
| 16384 | 6.1211e-8 |
| 32768 | 1.0569e-8 |

On the pinned survey's 832 shells with the external smooth linear-power
interpolant, maximum SSC variance changes relative to L_max=32768 are
1.0521e-4, 1.2935e-6 and 9.8745e-9 for L_max=1024, 4096 and 16384.
The preliminary projection's L_max=4096 therefore does not establish the
full 1e-6 accuracy gate, even though it is much better for SSC than for
noise pair counts. L_max is a mask-resolution input to refine, not a hard-
coded universal setting; these results apply to this cap and these bins.

## Cost and didactic review

Each SIMDe lane sums one whole bin in fixed multipole order; OpenMP assigns
complete pairs of bins. The function has no allocation, global cache or
BLAS call. For twenty bins and 32769 mask modes on the Apple M2 Pro, after
three warm-ups and 21 measured calls, mean ± sample standard deviation is
0.410 ± 0.009, 0.175 ± 0.006 and 0.144 ± 0.005 ms at 1/4/8 threads.
The scalar operators and mask are already constructed. This excludes
geometry construction, cosmology and all other covariance terms. Desktop
applications remained active; no test jobs ran concurrently. Native
disassembly confirms two-lane fused multiply-add in the mask contraction.

After both three-test runs passed, a separate manual red-eye review read
the complete C/header and reference/test entry points. It checked the
ordered-pair definition, raw-mask versus SSC normalization, footprint versus
signal multipoles, the full-sky limit, mask truncation, ownership and SIMD
lanes. C/header lines fit 80 columns, with one predicate per line. This is
a self-review; Fable was unavailable. No project interface or existing
covariance file is changed by this isolated ticket.

External files: `mask_reference.py`, `build_mask.sh`,
`check_mask_resolution.py`, and `results/mask*_tests.log`,
`results/mask_resolution.{json,npz}`, `results/mask_simd.disassembly.txt`.
LSST's opt-in `test_covariance_mask.py` uses the reference directory and
`COSMOLIKE_MASK_LIBRARY`/`COSMOLIKE_OPERATORS_LIBRARY`.
