# cNG perturbation-theory angular averages

`perturbation_cov.c/.h` implements the three planar tree-level averages
needed by the halo trispectrum. It consumes supplied linear-power samples
and a shared angular quadrature; it does not yet construct halo moments,
assemble the halo trispectrum, or project a full survey covariance.

## Physics and independent reference

The density/velocity recursion was checked directly against
[Bernardeau et al., Eqs. 43–45](https://arxiv.org/pdf/astro-ph/0112551),
including the n=3 denominator 18 and symmetrization over six permutations.
The independent NumPy reference explicitly enumerates the four cubic-leg
and twelve two-quadratic-leg Wick diagrams. It does not call C or use the
reduced planar trispectrum formula for this comparison. Multiplicities
follow the tree trispectrum in
[Takada & Hu, Section III](https://arxiv.org/html/1302.6994v3).

The optimized formula combines the two F2 terms sharing an internal
momentum before evaluation. In the nearly opposite-wavevector corner,
the two individual terms can be large while their sum is finite. The
implementation uses `c = 1+cos(theta)` supplied as
`2*sin((pi-theta)/2)^2`, and `s^2 = (K-Q)^2 + 2*K*Q*c`, avoiding a
subtraction between nearly equal squared wavenumbers. The exact-diagonal
bracket tends to `13*P/14`. Zero-internal-momentum SSC channels are excluded
analytically; there is no NaN-to-zero recovery or eigenvalue clipping.

The F3 angular contribution has a closed form, derived in study report
10, Section 2.4. Its two branches meet at `-4/63` for K=Q and are checked
against the independently symmetrized recursion. Averaging is in the
transverse plane, with dtheta/pi, not the 3D solid-angle measure.

## Checks and remaining accuracy work

Five tests pass in both optimized and debug/UBSan builds:

- Explicit Wick diagrams at five moderate (K,Q) points agree with the
  reduced C averages to 2e-12 relative.
- The closed F3 average agrees with a direct angular integral of the
  six-permutation recursion to 1e-12 for Q/K from 0.01 through 100.
- Exact and near-diagonal cases, including K=1000 for a spectrum peaking
  near unity, agree with 60-digit direct-F2 integration to 2e-11. Doubling
  64 to 128 GL nodes on 20 graded panels agrees to 1e-12 for these cases.
- Distance-unit rescaling gives the correct length^3, length^6 and
  length^9 outputs to 1e-12; exchanging K,Q leaves the averages unchanged.
- Odd pair counts, padded rows and repeated 1/4/8-thread calls agree
  bitwise. Two SIMD lanes hold different pairs, preserving angular order.

These tests use a smooth analytic power spectrum. A uniform 128-node
rule differed by 3.68e-7 in the near-diagonal tree trispectrum at
K=1,Q=1.0001; the graded rule resolves that feature. This is evidence for
angular refinement, not a production setting for the supplied CAMB grid.
The subsequent [projection diagnostic](covariance_projection_measurements.md)
measures angular refinement with real CAMB tables, combines halo moments,
and tests projected matrices. The CosmoCov convention and full-survey
accuracy gates remain open.

External artifacts are `cng_reference.py`, `build_perturbation.sh` and
`results/perturbation{,_debug}_tests.log`. LSST discovers the five tests
when `COSMOLIKE_COVARIANCE_REFERENCE` and
`COSMOLIKE_PERTURBATION_LIBRARY` point to that reference and isolated library.

## Separate didactic red-eye pass

After both five-test runs passed, manually reread the entire C/header
and comparison code before starting halo integration. The comments explain
the physical meaning of F2/F3, the planar measure, Wick multiplicities,
the corner rearrangement, each stage of the SIMD expression, input units,
ownership, thread ownership and the unused odd lane. All C/header lines
fit 80 columns; guards have one predicate per line. The code keeps
unsupported zero angular denominators out of the calculation explicitly.
This was a self-review; no Fable review was run. The remaining halo and
survey work is explicit, so passing these tests cannot be mistaken for a
validated full cNG covariance.
