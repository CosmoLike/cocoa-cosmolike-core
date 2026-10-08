# Halo-model option menu, 2026-10-08

Owner request: add concentration, mass-function and halo-bias options
beyond the defaults, following what OneCov and TJPCov expose, without
changing any default. Implemented in `cosmolike/halo.c` and
`cosmolike/halo.h` on `bugfix`; the selectors stay
`like.halo_model[0..2]` with defaults 0 (set in `structs.c`).

## The menu

| Selector | 0 (default, unchanged) | 1 (new) |
|---|---|---|
| halo_model[0] (HMF) | HMF_TINKER_2010 | HMF_TINKER_2008 |
| halo_model[1] (bias) | HALO_BIAS_TINKER_2010 | HALO_BIAS_SHETH_MO_TORMEN_2001 |
| halo_model[2] (conc) | CONCENTRATION_BHATTACHARYA_2013 | CONCENTRATION_DUFFY_2008 |

All three new fits are written at this file's Delta = 200 rho_mean
halo definition:

- Tinker et al. 2008 (0803.2706, Eqs. 3 and 5-8, Table 2 at
  Delta = 200m): f_T08(sigma) = A[(sigma/b)^-q + 1] exp(-c/sigma^2)
  with A = 0.186 a^0.14, q = 1.47 a^0.06, b = 2.57 a^alpha,
  alpha = 0.0106756286522959060767 (log10 alpha =
  -(0.75/log10(200/75))^1.2 = -1.97160654139105438701), c = 1.19.
  Convention bridge: halo.c's f(nu) = f_T08(delta_c/nu)/nu, since
  dln(1/sigma) = dln(nu). The paper's own amplitude and raw-a scaling
  are used (CCL/TJPCov convention, no z <= 3 clamp and no tinker_alpha
  normalization).
- Sheth, Mo & Tormen 2001 (astro-ph/9907024, Eq. 8; a = 0.707,
  b = 0.5, c = 0.6), with precomputed literals sqrt(a) =
  0.840832920383116303209, sqrt(a) b = 0.420416460191558151604,
  b(1-c)(1-c/2) = 0.14, 1/(sqrt(a) delta_c) = 0.705395561738249015697
  at the house delta_c = 1.686. Calibrated on virial halos; at 200m it
  is a cross-code approximation (pyccl exposes it there only with
  mass_def_strict=False, which is how TJPCov uses it).
- Duffy et al. 2008 (0804.2486, Table 1, full sample, mean-200 row):
  c = 10.14 (M/2e12 Msun/h)^-0.081 a^1.01. A pure (M, z) fit: no
  growth factor, so no cb extension with massive neutrinos (unlike the
  Bhattacharya default's D_cb^1.15 factor).

Mechanics: `fnu_params`/`hb1nu_params` gained an `int model` tag plus
the new fits' fields; `fnu_core`/`hb1nu_core` branch on the tag;
`conc()` gained a switch case. `fnu_shape` pins HMF_TINKER_2010, so the
cluster fixed-alpha path keeps the Tinker 2010 shape by construction,
and both cluster bindings (cluster_interface_cov.cpp:515,
cluster_wrapper_cov.cpp:479) require halo_model[0] == HMF_TINKER_2010,
which protects the 0.368/alpha(a) identity from the T08 option. No
Python setter for halo_model exists yet; selecting an option currently
needs interface-level code.

Caveat 1 (data-vector pass, ea613ca): halo_cluster.c checks
halo_model[0] and [3] only. Nothing guards [1] (bias) or [2]
(concentration), so SMT01 or Duffy 2008 would flow silently into the
cluster b_nl and one-halo P1h tables, while the commentary and
arXiv 2503.13631 assume Tinker 2010 bias and Bhattacharya 2013 (with
its D_cb^1.15 neutrino treatment; Duffy has no growth factor at all).
Owner decision needed before exposing the menu to cluster runs: guard
the cluster path or document the combination.

Caveat 2 (halo batch, 0a761b5; Fable-verified): selecting
HALO_BIAS_SHETH_MO_TORMEN_2001 also RESCALES the default Tinker 2010
mass function, because tinker_alpha normalizes int b f dnu = 1 with the
selected bias: alpha = 0.3207 at z = 0 instead of 0.3684 (-13 percent;
-14.5 percent at z = 1; independently reproduced as 0.3208/0.3686).
This changes every mass-function consumer and will not match pyccl's
Tinker10 + Sheth01 (pyccl does not re-normalize). Owner options: pin
the Tinker bias inside tinker_alpha, or guard the combination.

Caveat 3 (halo batch, 0a761b5): bias_norm, hod_tables, p_gm, p_gg and
ia_tables are not keyed on like.halo_model[0..2] (only tinker_alpha
is): changing the menu mid-process returns stale tables. Harmless
today only because no Python setter exists. Under HMF_TINKER_2008,
int f dnu diverges logarithmically at small nu, so 1 - bias_norm loses
its "missing share" meaning.

## Validation (ccl conda env, scratchpad/halo_menu_pyccl_check.py)

Formula-level comparison of the exact committed expressions against
pyccl at Delta = 200m, z in {0, 0.25, 1, 2}, M in [1e12, 1e15] Msun/h:

- Duffy08 vs ConcentrationDuffy08: max relative difference = 0 (exact).
- Tinker08 f(sigma) vs MassFuncTinker08._get_fsigma: 1.8e-11 (the
  double-rounded alpha literal).
- SMT01 vs HaloBiasSheth01 (mass_def_strict=False): 1.1e-7 at pyccl's
  delta_c = 1.68647; 2.7e-4 with halo.c's delta_c = 1.686. The 2.7e-4
  is the documented delta_c convention difference, not an error.
- All four 21-digit literals reproduced by mpmath at 25 digits.

Defaults regression: full rebuild, then lsst_y1 covariance suite
(128 passed) and the des_cluster data-vector suite rerun on the new
halo.c; the default branches' arithmetic is untouched (the new code is
a tag test before the existing expressions).
