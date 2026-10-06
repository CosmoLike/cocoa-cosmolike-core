# Lower halo mass limit: 2026-10-06

The explicitly authorized change lowers limits.halo_m[RANGE_MIN] from
1e6 to 1e4 Msun/h. This shared table-domain setting lives in structs.c;
its one-line change is part of this range-extension ticket, not permission
to move covariance algorithms into the data-vector C files.

Covariance examples use halo_mass_edges() from the shared Python package.
It adds two one-decade mass panels below 1e6, retaining all eight original
panels through 1e17 exactly. All seven project adapters call that helper.
Global accuracy and integration accuracy leave the physical edges fixed;
integration accuracy still selects the GSL nodes inside each panel.

No halo bias, multiplicity normalization, concentration relation, cb
convention, or additive completion prescription changes. In the covariance
I11 calculation the missing response multiplies the profile at the minimum
mass: the matter specialization of Mead et al. (2020), Appendix A Eq. 52.
I12/I13 retain their existing integrals without a corresponding completion.
The separate cluster-selection mass interval is unchanged.

## Evidence before adoption

OneCov-benchmark- commit 935e0cf records the complete comparison and scripts.
Both scientific installations were untouched during that study: an isolated
build extended the table domain to 1e2 for controlled 1e6/1e4/1e2 integration
cutoffs. The full LSST default-accuracy covariance uses 1560 entries,
Gaussian non-Limber spectra and separate G/SSC/cNG; no matrix entries were
masked or repaired. Common CAMB inputs have SHA256
26ed1a29b47f82e2aa3de13de93e5ccd3bf88bc7ceb802125157ca12211fff70.

Largest total generalized variance-mode changes, fractionally:

- Common wide table, 1e6 versus 1e2: 4.5581e-8.
- Common wide table, 1e4 versus 1e2: 1.7958e-9.
- Original versus wide table at the same 1e6 cutoff: 5.9639e-6.
- Original table/cutoff versus wide table/1e2 cutoff: 5.9578e-6.

All total matrices are positive definite; G is bitwise unchanged.
The table-domain effect must not be conflated with the integration cutoff.
It also motivates checking the actual production 1e4 table after adoption.

At z=1 the missing I11 weight falls from 31.97% to 28.38% at 1e4, and 25.51%
at 1e2. This is a reduction in bias-weighted completion weight, not a
covariance error. The existing 1e6 covariance already has negligible
measured sensitivity. Lower masses depend more heavily on the input
power's high-k continuation; numerical agreement does not calibrate the
extrapolated power or halo fits, nor establish general Fisher convergence.

Three sequential full runs/cutoff, M2 Pro, 8 OpenMP threads, BLAS 1:
49.57 +/- 2.12, 50.88 +/- 0.83, 50.18 +/- 1.69 s for 1e6/1e4/1e2.
First-use tables are included; CAMB setup and output writing are excluded.
All physical arrays repeat bitwise within each cutoff. Scatter prevents
a precise slowdown claim. The earlier 64% cost increase applied only to
isolated halo moments, not the complete covariance.

## Validation of the production setting

The actual production 1e4-domain LSST matrix took 50.628 s at eight
threads. It is positive definite (smallest variance-normalized eigenvalue
2.26985e-4). G is bitwise equal to the archived controls. Maximum total
variance-mode changes are 3.13474e-6 versus the original production 1e6
matrix, 3.08926e-6 versus the wide-domain 1e4 control, and 3.08907e-6
versus wide-domain 1e2. Component checks show no large cancellation:
versus original production the maximum SSC/total mode contribution is
4.13231e-7 and cNG/total is 3.12557e-6. Each full 1560x1560 matrix entry
is included. Comparison records and figures are in OneCov-benchmark- at
results/cutoff_production4_vs_{native6,wide4,wide2}_20261006.json.

Roman's independent sigma integral needed a wider lower x=kR boundary.
The old x_min=1e-4 omitted about 1e-4 of the variance at M=1e4. Using
x_min=1e-6 reduces the FFTLog difference there to 2.5854e-7. Across nine
masses, the largest difference is 6.98894e-6, within the unchanged 2e-5
tolerance. Lowering x_min again to 1e-8 and doubling 2 million samples
to 4 million changes the reference by at most 2.58105e-11. This fixes
the test's truncated integration interval; the production FFTLog is intact.

Before accepting the halo snapshot refresh, all 25 independent halo
invariant/cache/thread tests passed, including the slow spectrum tier.
The generator --halo rewrote only halo_reference.json and its manifest
hash; all input grids, likelihood vectors and likelihood chi2 references
are unchanged. The fitted fnu, hb1nu and NFW profile probes stay bitwise
equal. Concentration, slopes, ngal, bgal and p_gm move by <9e-7 relatively.
bias_norm is deliberately a finite-range integral: at a=0.05 it rises
from 0.0179257 to 0.0930927 when the lower bound moves, without a fit change.

The default p_gg mass quadrature also regrids over the enlarged range.
Its maximum snapshot change is 1.8917% at k=142.81 h/Mpc, but only
4.0210e-5 fractionally for k<=10. At integration level 2 the cutoff
change for k<=10 falls to 7.9788e-7. Do not claim precise default HOD
point spectra at arbitrarily high k: oscillatory high-k mass integrals
still need refinement. The complete 2115-entry Roman HOD data vector
with its production mask gives delta chi2(default, integration 2) =
0.00247093, compared with 0.00243134 before the range extension.
The old/new default-vector difference is 0.000123087 in delta chi2;
the integration-2 difference is 3.44238e-7. Both remain well below 0.2.

The cluster covariance suite exposed two reference-comparison issues.
Grouping the independent exponential as gamma*(nu*nu), instead of the C
reader's (gamma*nu)*nu, changes an exponent near -240.94 by 5.68434e-14.
The affected weights are between 1e-245 and 1e-113; their relative
discrepancy is 5.66e-14. A 60-digit exponential check confirms ordinary
rounding in both orders. The formula comparison now allows 1e-13, while
the bitwise 1/2/4/8-thread requirement is unchanged. The separate moment
sum now compares with the supplied weights, retaining its 3e-15 tolerance.
This separates integrator precision from the formula's rounding allowance.
All 50 cluster covariance tests then pass. No C calculation was changed.

## Project regression results

All seven project interfaces were rebuilt in the optimized strict mode.
Data-vector and covariance sectors ran separately and sequentially, with
COCOA_HALO_SLOW=1 for Roman's optional halo tier. The environment requested
eight OpenMP threads and one BLAS thread; individual test modules retain
their own thread settings, including explicit thread-repeatability sweeps.

| Project | Data-vector sector | Covariance sector |
| --- | --- | --- |
| LSST Y1 | 57 passed | 120 passed |
| Roman Fourier | 45 passed | 1 passed |
| Roman KL | 49 passed | 1 passed |
| DES Y3 | 62 passed, 1 inverse-guard failure; nonlinear module rerun: 2 passed | 1 passed |
| DES x Planck | 45 passed | 1 passed |
| DES cluster | 29 passed | 50 passed after the rounding corrections above |
| Roman real | Combined run aborted at the inverse guard; completed modules passed, remaining modules passed separately | 1 passed |

Roman real collected 104 tests. The combined run completed its halo,
halo-cache and IA-accuracy modules before aborting while initializing
the supplied covariance for the IA-cache module. Separate runs covered
the halo checks again and all remaining modules: 56 passed across ten
files. These include IA cache/thread repeatability, HOD spectra, both
non-Limber paths, nonlinear bias, notebook API, photo-z and scale cuts.
Do not describe the aborted aggregate run as a clean all-green result.

The focused debug build passed 11 covariance halo checks without sanitizer
warnings. A separate data-vector-only build reports has_covariance=False
and reproduces the frozen LSST NLA/TATT chi2 values exactly:
0.023705928404497645 and 0.011953501626625753. Rebuilding the original
1e6-domain interface and recomputing its complete LSST covariance gives
bitwise-identical G, SSC, cNG and total arrays to the archived control.

## Pre-existing supplied-covariance inversion instability

An intermittent inverse residual failure interrupted the combined Roman
suite and one DES Y3 nonlinear worker. The same guard fails with the
preserved original DES Y3 binary, before any halo evaluation: covariance
load 11 of 40 gave a residual of 0.86458. This is not introduced by the
mass-cutoff change. The failed/remaining modules pass when run separately.

An isolated diagnostic build preserved a second failure. Its input is
finite, exactly symmetric, positive definite, and unchanged by inversion.
The saved inverse has max |R R^-1 - I| = 0.007572, whereas a separate
inversion of that same input gives 2.30e-14. OpenBLAS reports one thread
both before and after. The exact lower-level cause remains unresolved;
do not attribute it to a particular library or memory race without further
evidence. The existing guard correctly stops initialization. No production
inverse code or likelihood reference was changed in this cutoff patch.

Logs, original interfaces, the captured matrices and follow-up results are
preserved under test/low_mass_cutoff_validation/20261006. In particular,
des_original_loading.log and inverse_diagnostic_interface/analysis.json
record the reproduction and matrix checks. Keep this regression issue
separate from the demonstrated low-mass and full-covariance convergence.
