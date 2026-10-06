# Study plan: an effective unresolved halo population

Status: source and literature study, 2026-10-06. No numerical runs or
production changes accompany this document. The production baseline is
core d95867f: FFTLog bias 0.8, covariance panels down to 1e-40 Msun/h,
guarded Wynn extrapolation of I11, and residual zero-wavenumber completion.
Its broader regression validation is tracked separately. This study does
not make that validation complete.

The useful first experiment is small: replace the extrapolated population
below a chosen mass by a population with specified total mass and linear
response. Keep every resolved abundance, bias and profile unchanged.
Do not describe this as a complete implementation of Schmidt's EHM, or
promise a more accurate covariance without independent calibration.

## Physics anchors

- [Schmidt (2016), Appendix A, Eqs. 55--63](https://arxiv.org/html/1511.02231v3#A1)
  truncates the mass integrals, adds a population at the cutoff, and sets
  its abundance and bias from mass and response constraints. Equation 60
  gives cutoff dependence suppressed by `(k R_s)^2` for the deterministic
  term when `k R_s << 1`. Equations 61--63 additionally constrain the
  unresolved stochastic field. Section II.3 requires mass-weighted
  nonlinear biases to vanish; stochastic matter power must scale as
  `k^4` at small k. Section III.1, Eqs. 33--35, illustrates a stochastic
  covariance with a compensating term. Appendix A's argument cannot be
  transferred to an unchanged Poisson halo covariance by fixing two
  scalar integrals alone.
- [Tinker et al. (2010), Section III.1, Eqs. 6--7](https://arxiv.org/html/1001.3162)
  gives the fitted linear bias and its bias-weighted consistency condition.
  Its discussion distinguishes the evolving abundance from the weakly
  evolving bias. CoCoA's actual normalization is described below; do not
  infer that it enforces both constraints from a paper citation alone.
- [Mead et al. (2020), Appendix A, Eqs. 49--52](https://arxiv.org/html/2005.00009v2#A1)
  gives additive two-halo completions, including a cutoff profile. It
  explicitly applies the alteration only to the two-halo term. This
  HMx paper is the source of CoCoA's existing completion; it is not a
  demonstration that CoCoA implements every HMcode or EHM ingredient.
- [Takada & Hu (2013), Eqs. 25--29 and corrected Eq. 44](https://arxiv.org/html/1302.6994v3)
  defines the halo moments, their connected partitions and the isotropic
  response. Its linear-halo-bias approximation is the relevant comparison
  for CoCoA's existing assembly, not Schmidt's full bias expansion.
- [Li, Hu & Takada (2016)](https://arxiv.org/abs/1511.01454)
  tests linear halo bias as the abundance response in separate universes.
  This supplies a route to physical calibration beyond integral identities.

These sources have distinct roles. The experiment below is an application
to the current code, with its extra assumptions stated explicitly.

## What the current code actually does

`cosmolike/halo.c::fnu` keeps the Tinker shape and determines its amplitude
from the full bias-weighted integral. `hb1nu` retains the fitted bias.
The ordinary multiplicity integral is not independently normalized.

`cosmolike/covariances/halo_cov.c::halo_moments_cov` builds cb abundance
and volume weights. For ordinary finite panels it returns

$$
I_{11}(k)=I_{11}^{>}(k)+[1-B_s]u_s(k),
$$

where `B_s` is the finite integral of the biased mass weight and `u_s`
is the normalized NFW profile at the lower mass. For the production
eleven-panel tail it first extrapolates I11 and B with Wynn, then adds
only the residual zero-k completion. The five higher-moment outputs are
direct integrals over the supplied mass interval.

`non_gaussian_cov.c::halo_response_cov` forms
`P_halo = I11^2 P_lin + I02`, combines growth, dilation and I12, then
optionally transfers that fractional response to the supplied nonlinear
power. `halo_trispectrum_cov` assembles 1h, 2h(1+3), 2h(2+2), 3h and 4h
from I11/I12/I13/I04 and tree averages. Neither routine contains a
mass-dependent stochastic covariance or nonlinear halo-bias operators.

`cosmolike_notebook_utils/covariance/survey.py::_matter_covariance_tables`
uses those moments at central and shifted wavenumbers, recomputes the
two-halo logarithmic slope, then projects SSC and cNG. An altered I11
must also reach both shifted evaluations; otherwise the response would
mix two models. Gaussian spectra and the supplied nonlinear power stay
fixed in this study.

Restrict the first experiment to the supported massless-neutrino matter
covariance. Then rho_cb equals rho_m. A massive-neutrino extension needs
separate treatment of the total-matter response; it is not authorized by
this exercise. The data-vector HOD floor, halo fits and sigma tables stay
unchanged throughout.

## Mass and response are different weights

Define the following from the current fitted functions above a cutoff Ms:

$$
F_s=\int_{M_s}^{M_{\max}}d\ln M\,
       {dn\over d\ln M}{M\over\bar\rho},\qquad
B_s=\int_{M_s}^{M_{\max}}d\ln M\,
       {dn\over d\ln M}{M\over\bar\rho}b_1(M).
$$

F counts mass; B counts its response to a long density fluctuation.
With `V_s = Ms/rho`, the effective abundance and response are

$$
\bar n_s={1-F_s\over V_s},\qquad
b_s={1-B_s\over1-F_s}.
$$

Thus its mass weight is `1-F_s`, while its biased mass weight is `1-B_s`.
These are two independent requirements. The first is absent from the
current completion API; the second is already its exact I11 coefficient.
Merely computing and naming `n_s` and `b_s` therefore leaves ordinary
finite-panel I11 unchanged.

Existing saved diagnostics at Ms=1e4 give the following orientation.
They predate the final FFTLog stability fix and must be remeasured before
using them as acceptance results.

| z | F_s | B_s | b_s |
| ---: | ---: | ---: | ---: |
| 0.1 | 0.71190 | 0.81145 | 0.65445 |
| 0.5 | 0.66521 | 0.76894 | 0.69015 |
| 1.0 | 0.60535 | 0.71615 | 0.71923 |

Source: OneCov-benchmark-'s archived skill reference
`historical_normalization.md`, backed by its normalization study. These
are code diagnostics, not measured abundances of an unresolved population.

### Why this must replace a tail

The retained full fitted multiplicity has ordinary mass integral
1.04832034 at z=1, while its bias-weighted integral is one. The archived
finite mass integral through 1e-40 is already 1.001026 at that redshift.
Those are respectively the fitted limit and a finite integral; production
Wynn extrapolates only I11, not an ordinary-mass output.

Adding positive mass to either excess cannot produce unit total mass.
Adding the effective population for Ms=1e4 on top of the current deep
tail would count that tail twice. In the illustrative z=1 numbers,
the fitted tail below Ms contains `1.04832 - 0.60535`, whereas the
replacement is assigned `1 - 0.60535`. They differ by the full 0.04832
normalization excess. This is a change in the low-mass model, not a
more accurate evaluation of the original fitted integral.

Before accepting any cutoff, measure F and B with a quadrature error.
Require `1-F > 0` by a margin larger than that error. If F exceeds one,
reject that cutoff for a positive-population interpretation; do not clip
the excess. If F is indistinguishable from one while `1-B` is not,
the proposed single population is ill-conditioned. Negative `1-B` would
require negative effective linear bias: mathematically possible for an
anticorrelated component, but outside the intended positive low-mass halo
interpretation and a reason to stop for physical review.

Do not change the normalization of retained halos to hide a failed sign
test. Also check the high-mass tail before identifying every deficit as
low-mass material: the numerical upper bound is 1e17, not infinity.

## Two different levels of implementation

### A. Bounded moment-sensitivity experiment

Treat the effective population as one additional mass species inside
CoCoA's existing moment algebra. This is an explicit Poisson-like closure
for the experiment, not the full stochastic EHM. Here Poisson-like means
using the same independent halo-count moments as the existing assembly.
Stochastic correlations describe the residual fluctuations of those
counts after their response to the large-scale density has been removed.
Direct substitution
into the current moment definitions gives

$$
\Delta I_{0\mu}=(1-F_s)V_s^{\mu-1}\prod_{j=1}^{\mu}u_s(k_j),
\qquad
\Delta I_{1\mu}=(1-B_s)V_s^{\mu-1}\prod_{j=1}^{\mu}u_s(k_j).
$$

Here `I0mu` means no bias and mu profiles; `I1mu` means one linear-bias
factor and mu profiles. For the five existing outputs the additions are:

| Output | Additional weight before its profile product |
| --- | --- |
| I02 | `(1-F_s) V_s` |
| I12 | `(1-B_s) V_s` |
| I13, either ordering | `(1-B_s) V_s^2` |
| I04 | `(1-F_s) V_s^3` |

I11 already includes `(1-B_s) u_s`; do not add it again. Every higher
moment starts from the truncated integral above Ms. Keeping its deep
tail while adding a replacement would also double-count those moments.

This algebra suggests little effect on higher moments at a low cutoff:
every extra density leg contributes another small volume V_s. Existing
direct-integral tests from 1e4 to 1e-20 found maximum fractional changes
2.31261e-7 for I02, 1.21305e-7 for I12, 1.73195e-14 for either I13,
and zero at saved precision for I04. Those numbers concern extending a
fitted tail, not this different single-species replacement.

The existing low-level interface already accepts finite mass panels and
returns completed I11 plus finite higher moments. F and B can be measured
from the public abundance/bias readers, with profile samples at Ms.
First implement the sensitivity calculation in an external study using
these inputs and existing assembly functions; no production C change or
new binding is needed merely to decide whether the idea matters.

### B. Full EHM covariance

The current Poisson-like one-halo term has positive `I02(0)`. Appending
another positive one-halo contribution cannot make it vanish as k^4.
A normalized background and I11 therefore do not repair that issue.
An actual EHM extension would need halo stochastic cross-correlations,
their response and higher-point statistics, plus the appropriate bias
operators and profile responses. A two-point stochastic covariance alone
does not specify the four-point function used by cNG.

This requires a separate derivation and calibration. It is not a small
patch to `completion[row]`, and the present experiment does not authorize
it. In particular, a positive total survey covariance would not prove
all halo-level consistency conditions have been satisfied.

## Experiment sequence and acceptance gates

1. **Freeze inputs.** Use the finally validated Wynn commit, its common
   CAMB tables, LSST Y1 survey and exact Gaussian settings. Record file
   hashes, masses, density convention, profile, cosmologies and compiler.
   Do not retune linear bias, HOD, target nonlinear power or halo fits.

2. **Check eligibility before computing covariance.** Initially try
   Ms=1e4 and 1e6 Msun/h; add 1e8 only to expose cutoff sensitivity.
   Include z=0.01, 0.1, 0.5, 1 and 3, then all survey shell redshifts
   for the surviving cutoff. Repeat for the two changed cosmologies in
   the Wynn regression. Measure F, B, `1-F`, `1-B`, b_s and their
   integration changes using 96, 128 and 256 nodes, retaining panel
   edges. Use 512 only if that identifies an unresolved numerical error.
   None of these trial cutoffs is declared simulation-calibrated.

3. **Separate three arms.** W is current deep/Wynn production. F uses
   ordinary finite panels at Ms with current I11 completion and no
   higher-moment additions. U starts from F and adds the effective
   species to all five higher moments. Comparing F with U isolates the
   ordinary-mass completion; comparing W with F isolates the changed
   low-mass integration and I11 profile placement. Preserve common
   retained quadrature panels so regridding is not mistaken for physics.

4. **Inspect physical ingredients.** Include k=0, logarithmic samples
   from 0.001 to 300 h/Mpc, unequal k pairs and the actual shell k range.
   Measure I11, I02, I12, both I13, I04, each trispectrum partition,
   response and its slope. Verify the algebraic mass/response sums and
   finite-panel I11 equivalence before projecting. Check both the core
   NFW radius and Lagrangian radius; near-observer shells can reach much
   larger k than the typical survey mode. `k R_s << 1` must be tested
   over the contributing domain, not assumed from the name "low mass".

5. **Compare complete matrices.** For the best eligible cutoff generate
   all 1560 LSST entries in real space, retaining G, SSC, cNG and total
   separately. First require G to match bitwise with common inputs.
   Check every component for finite entries, symmetry and correct sum.
   Test total Cholesky positivity both before and after production cuts;
   report the correlation-normalized minimum eigenvalue. Do not repair
   negative eigenvalues. Report component spectra, without requiring cNG
   alone to be positive definite or dividing by its tiny entries.

6. **Use informative difference measures.** Solve
   `C_U v = lambda C_W v` on the full unmasked matrix. For each separate
   component X inspect eigenvalues of `L^-1 (X_U-X_W) L^-T`, where
   `C_W = L L^T`. Plot four matrices normalized by
   `sqrt(C_W,ii C_W,jj)`. This prevents tiny SSC/cNG denominators and
   component cancellation from hiding a problem. A 1e-3 mode scale is
   an initial numerical-convergence goal, not evidence of better physics.
   Do not substitute the data-vector delta-chi2 criterion. A physical
   gain would require a matched simulation/response reference; Fisher
   tests also require specified derivatives and parameter choices.

7. **Time only on a quiet machine.** Record shared halo construction,
   projection and full production-CLI time separately at eight threads.
   Compare at least three sequential runs of W/F/U with identical timing
   boundaries. Concurrent validation or benchmark runs invalidate a
   speedup claim. The theoretical reduction in mass nodes is only a
   hypothesis about runtime; power reads and other stages may dominate.

Stop the candidate if positivity/sign checks fail, the required cutoff
is outside the resolved-scale regime, or numerical refinement changes
the result enough to obscure the model comparison. If moment differences
and total modes are negligible, keep production unchanged unless a
specific measured benefit justifies a new option. Agreement with OneCov
alone is not that benefit.

## Smallest later production patch, only if justified

For the bounded U prescription, `halo_cov.c` already has abundance,
bias and profile weights. One additional ordinary mass sum can supply F;
the existing biased sum supplies B. Explicitly selecting a finite cutoff
would bypass Wynn and replace the existing completion with the single
population's weights. Add each higher-moment weight once in the same
deterministic SIMD sums; preserve output shapes and shared C kernels.
Avoid dividing by `1-F` in the hot path: the moment weights need F and B,
while b_s is a diagnostic.

An explicit covariance-owned setting is needed if both prescriptions
remain available; do not silently infer a new physical model from array
length. Keep the data-vector C files, fits, HOD floor, wrapper array
conventions and optimized production interface unchanged. A production
proposal must explain the chosen stochastic approximation and include
all-project regressions plus debug, disabled-covariance and thread checks.
This is a proposed scope, not permission to implement it.

The immediate possible gain is transparent mass accounting with a
controlled finite cutoff and less reliance on extreme extrapolation.
Neither a smaller additive correction nor exact scalar constraints prove
a more accurate SSC or cNG covariance. The finite-panel I11 identity
and the missing stochastic physics are the central limits of the study.
