# SSC mask and shell-response foundation

`cosmolike/covariances/ssc_cov.c/.h` supplies two convention-explicit
building blocks. They operate on supplied power/response tables; the halo
response model and survey generator are not implemented by these functions.
No existing core C file was modified. The independent reference remains
outside git in `test/covariance_reference/ssc_reference.py`.

## Physical contract

`ssc_mask_variance_cov` accepts the **raw** angular mask spectrum,
`C_L = sum_M |w_LM|^2/(2L+1)`, and its area integral in steradians. The
monopole must satisfy `C_0 = area^2/(4*pi)`; a C_0-normalized file fails
with an explanatory error instead of silently changing the covariance.
The output is the long-mode **Limber** background strength

`sigma_b^2(chi) = sum_L (2L+1) C_L P_lin((L+1/2)/f_K,a)/(area*f_K)^2`.

It has units of length, because it multiplies a radial Dirac delta in
the background covariance. The projected SSC integral is
`integral dchi sigma_b^2 Phi_i Phi_j`. A finite shell's dimensionless
variance is not interchangeable with this quantity.

This projection follows [Takada & Hu, Appendix A, Eq. 54](https://arxiv.org/html/1302.6994v3).
The spherical mask normalization follows the harmonic expansion in
[Barreira, Krause & Schmidt, Eqs. 60–62](https://arxiv.org/html/1711.07467v3).
Using the discrete mask spectrum does not remove the long-mode Limber
approximation. Low mask multipoles and large footprints require a
separate beyond-Limber density/tidal calculation before accuracy claims.

`ssc_shell_response_cov` constructs

`Phi_AB = W_A W_B (dP/d delta_b)/f_K^2 - (U_A+U_B) C_AB`.

Here U differentiates the estimator's catalog mean; it is not assumed
to equal a short-mode, ell-dependent field window. The supplied C_AB
must be the same model used in the Gaussian calculation. Differentiating
the complete projected estimator derives the radial mean subtraction;
a narrow constant galaxy slice recovers one bias times P per galaxy leg.
Choosing the nonlinear response, galaxy-mean/RSD/magnification model and
spin normalization remains the survey driver's explicit responsibility.

## Projection and the general radial kernel

Reuse the existing SIMDe `gaussian_project_cov` contraction rather than
duplicate matrix loops. For local Limber SSC its two operators are Phi
and its common weight is `dchi*sigma_b^2`. The function's operation is a
weighted dot product; it does not depend on an angular interpretation of
the integration axis.

For a general positive-semidefinite shell covariance K, supply a factor
`B[mode][radial]` satisfying `K = B^T B`. First project Phi against B with
weight dchi; then project those mode responses against themselves with
unit weights. This computes `S K S^T` without threaded BLAS, eigenvalue
clipping, or a hard-coded diagonal-only interface. Constructing B with
the correct non-Limber density and tidal physics is a separate task.

## Validation and didactic review

Six focused tests pass in both optimized strict and debug/UBSan builds:

- Spherical-cap mask coefficients against 60-digit direct angular
  integrals, including the monopole and modes through L=128.
- A band-limited mask with constant P against its exact sky integral;
  invariance under mask rescaling and distance-unit conversion.
- Full projected-estimator finite differences, signed responses and
  covariance-owned mean normalization.
- The analytic narrow-slice galaxy-mean subtraction.
- Limber and correlated-kernel projections against independent NumPy,
  with correlation eigenvalues >= -1e-12 and negative cross terms retained.
- Odd-size SIMD tails, padded-row canaries, repeated and 1/4/8-thread
  calls, with bitwise thread agreement.

`build_ssc.sh` produces isolated optimized/debug libraries; no running
project library is overwritten. Set `COSMOLIKE_COVARIANCE_REFERENCE` and
`COSMOLIKE_SSC_LIBRARY` to run `tests/test_covariance_ssc.py` in LSST.
Logs are `covariance_reference/results/ssc{,_debug}_tests.log`.
Native assembly in `ssc_simd.s` confirms vector fused multiply-add,
multiply, divide and fused subtract.

After the project test jobs finished, component timings on the Apple M2
Pro used three warm-up calls and 21 measured calls per case. Values below
are mean ± sample standard deviation in milliseconds, with strict IEEE
Clang 19.1.7. Desktop applications remained active; the machine was not
reserved exclusively for this measurement.

| Component and dimensions | 1 thread | 4 threads | 8 threads |
|---|---:|---:|---:|
| Mask variance, 832 shells × 4097 modes | 2.124 ± 0.089 | 0.601 ± 0.023 | 0.317 ± 0.025 |
| Shell response, 240 rows × 832 shells | 0.091 ± 0.003 | 0.045 ± 0.001 | 0.057 ± 0.006 |
| Projection, 240 × 240 outputs × 832 shells | 3.699 ± 0.062 | 1.022 ± 0.028 | 0.571 ± 0.031 |

These measurements exclude physical input construction, CAMB and Python
allocation. They are not full covariance timings. The small response
workload is faster with four threads than eight; no automatic team-size
heuristic was added from this one case. The external benchmark is
`benchmark_covariance_components.py`, with raw measurements in
`results/component_timings.json`.

A separate manual didactic red-eye pass followed both six-test runs,
before beginning the next component. It checked the entire C/header:
mask versus matter multipoles, the radial delta and its units, the origin
of the subtraction sign, full-spectrum versus shell quantities, ownership,
SIMD-lane meaning, and odd tails are explained at their point of use.
All C/header lines fit 80 columns; conditions have one comparison per
line. Unsupported mask normalization stops explicitly. This was a
self-review, not a Fable review. The code makes no claim to a calibrated
halo response, full SSC survey prediction, or frozen precision settings.
