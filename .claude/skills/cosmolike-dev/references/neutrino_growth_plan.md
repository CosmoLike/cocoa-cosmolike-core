# Growth and neutrino-aware halo quantities: physical and numerical design

Massive neutrinos make linear growth scale dependent. Halo statistics and
lensing also depend on different density fields. A single present-day
variance rescaled by one total-matter growth factor therefore cannot describe
all halo masses and redshifts consistently.

The physical evidence is in `fable_review_growth_factor.md`,
`fable_review_neutrino_halos.md` and `fable_review_concentration_growth.md`.
`growth_factor_measurements.md` records the CAMB sampling-scale tests;
`sigma_fftlog_implementation.md` describes the implementation and checks.

## Growth-table convention

Likelihood and notebook growth tables use `growth_k = 0.05/Mpc`. At the
horizon-scale alternative k = 5e-4/Mpc, dark-energy perturbations change
CAMB growth by 0.5–0.9% at z = 0.5–2 for w != -1 relative to the scales
modeled by the data vector. The sub-horizon choice avoids importing that
large-scale behavior into scale-independent NLA/TATT growth factors.

The independent DES cluster reference retains its external non-halo growth
convention. Its halo reference integrates the evolving cb spectrum directly;
it does not inherit either scale-independent growth approximation.

## Evolving variance and its mass derivative

`cosmo3D.c` tabulates both fields on (ln M, a): sigma²_m from P_lin with
rho_m in the mass-radius map, and sigma²_cb from P_cb with rho_cb. The mass
slope d ln sigma/d ln M is evaluated at each scale factor too. Likelihoods
supply P_cb consistently through `set_linear_power_spectrum_cb`, including
for direct halo queries after model evaluation.

FFTLog uses the closed-form Mellin kernel of the squared top-hat window;
a second inverse transform supplies its logarithmic derivative. The
2026-10-02 study reached 1.9e-9 in sigma² and 1.2e-8 in the mass slope over
1e6–1e17 Msun/h, versus 5.5e-5 and 2.7e-3 for the tested quadrature.
For 256 scale-factor nodes, the Python prototype took 19 ms serial or
13.6 ms with four workers, versus 1010 ms for quadrature. These values
belong to that configuration, not the extended covariance-tail domain.

Serially created FFTW plans and per-thread work arrays are reused. Whole
(field, a) rows remain with one worker: this preserves deterministic sums
and outperformed the tested split-transform layout. The implementation
reference owns the production continuation, bias and table-domain settings.

## Physical field for each consumer

| consumer | field | evidence |
|---|---|---|
| cluster counts and cluster bias (`halo_cluster.c:1591-1642`) | cb: sigma_cb(M, a), rho_cb, no D rescaling | strong (Castro et al. 2023: < 1% to 0.32 eV) |
| Tinker f(nu), halo bias, bias_norm (`halo.c:641`, 275, 756) | cb, the same nu in all three | strong |
| HOD tables, p_gg and p_gm halo statistics (`halo.c:3408`, 5089, 5624, 7524) | cb | strong |
| concentration, Bhattacharya 2013 (`halo.c:677-700`) | nu from sigma_cb(M, a) | medium |
| halo-model IA (`ia_tables`, `halo.c:7166`) | cb halo statistics | inference only |
| halo-model p_mm | Outside the compiled data-vector engine; a total-matter halo model requires its own neutrino prescription. | separate model |


The halo mass function, bias, bias normalization, concentration, HOD and
cluster statistics use nu = delta_c/sigma_cb(M,z), with rho_cb in the
statistical mass-radius map. The M200m overdensity radius and lensing mass
weight retain total matter. The total-matter variance remains available
as a diagnostic; there is no halo-field switch.

Bhattacharya's concentration fit has no massive-neutrino calibration that
selects a unique growth factor. Its implemented continuation uses
D_cb(M,a) = sigma_cb(M,a)/sigma_cb(M,1) in the D^1.15 prefactor and the
corresponding evolving variance in nu. The measured concentration change is
0.14–1.3% for mnu = 0.06–0.6 eV and z = 0.3–1, smaller than the fit's
approximately ±20% cosmology dependence. This is a consistency prescription,
not an independently calibrated neutrino concentration relation.

The halo-model p_mm, p_my and p_yy sources remain in
`future_port_unfinished/`, outside the compiled engine. Restoring thermal-SZ
spectra requires cb halo statistics and the cold sigma_8(z) appropriate to
the HMcode-2020 damping coefficient. Covariance-owned halo spectra have a
separate physical contract.

## Implementation and validation requirements

- Piecewise-uniform power tables use direct indexing. Setters always prepare
  their segment metadata; a diagnostic build must exercise the same reader.
- Explicit OpenMP loops own parallelism. Keep OpenBLAS at one thread and
  check the actual linked backend when validating that constraint.
- Use optimized strict-IEEE and debug builds. Fast-math has produced failed
  covariance-inverse checks and is not a supported numerical mode.
- Compare both NLA and TATT, full vectors and chi-squared, with one/eight
  threads. Change frozen references only for an explained physical change,
  with independent validation and preserved provenance.
- In the massless-neutrino limit, the two variance fields should agree to
  the residual scale dependence of the supplied growth (below 1e-4 in the
  recorded tests). At nonzero mass, test each consumer against an independent
  calculation rather than assuming a shared cancellation.

## Separate physical extensions

Non-halo galaxy spectra using P_cb and P_cb,m require nonlinear cb and
cross spectra; DES Y3's total-matter approximation has a different precision
scope from a Euclid-like forecast. Scale-dependent RSD growth, redshift-
dependent one-loop calculations and alternative mass-function/concentration
fits likewise require separate validation. None follows automatically from
using evolving cb variance in the halo statistics.
