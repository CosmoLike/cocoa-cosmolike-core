# Literature review: P_cb vs P_mm in the halo model with massive neutrinos

Scope: verification for the single a=1 sigma^2(M) table (cosmo3D.c:1983, header
cosmo3D.c:1740-1975) and its consumers in
`/Users/vivianmiranda/data/COCOA/september2026/test/cocoa/Cocoa/external_modules/code/cosmolike_core/cosmolike/halo.c`.
Every claim is tagged VERIFIED (source read today) or NOT VERIFIED (memory/inference).
Notation: "cb" = CDM+baryons; $f_\nu=\Omega_\nu/\Omega_m$; $\nu=\delta_c/\sigma$.

## 1. Halo mass function: universal in sigma_cb, not sigma_m

- Ichiki & Takada 2012 (1108.4688), spherical collapse with Boltzmann neutrinos:
  neutrinos stay quasi-linear ($\delta_\nu\lesssim$ few) during collapse; their
  contribution to halo mass is ignored; the collapse is "monitored" by the linear
  cb overdensity. Massive neutrinos change $\delta_c$ by less than ~0.5% for
  $m_\nu\lesssim0.5$ eV (<0.1% for 0.05-0.1 eV). Their prescription (their Eq. for
  $dn/d\ln M$): $n(M)=\frac{\bar\rho_{cb}}{M}f(\nu_{cb})\frac{d\nu}{d\ln M}$ with
  $\sigma^2_{cb}$ from $P^L_{cb}$ and $M=\frac{4\pi}{3}\bar\rho_{cb,0}R^3$.
  Abundance drop up to factor 2 at $\sim5\times10^{15}\,h^{-1}M_\odot$ for 0.1 eV,
  larger at higher z; a $\sigma_8$-matched massless model reproduces the MDM mass
  function only to ~30%. VERIFIED (1108.4688 abstract; Results Sec. 3.1-3.2,
  Eq. dndlnM, Figs. 4-9).
- Castorina et al. 2014 (1311.1212): $\nu f(\nu)$ is near-universal only with
  $\nu=\delta_c/\sigma_{cc}$; with $\sigma_{mm}$ "large departures from
  universality" grow with $m_\nu$ (their Fig. 2). With $\sigma_{cc}$ the residual
  universality violation is $\lesssim$10% at z=1, same as in massless LCDM;
  residual $m_\nu$ dependence at fixed $\sigma_{cc}$ is a few % at 0.6 eV. Mass
  and density: halos found in the CDM component; $M=\frac{4\pi}{3}\rho_{cdm}R^3$
  and $\rho\to\rho_{cdm}$ in $n(M)$ (their Sec. 2.2, Eq. M-R). VERIFIED
  (1311.1212 Secs. 2.2 and 4, Figs. 1-3, 5).
- Costanzi et al. 2013 (1311.1514): Tinker 2008 (SO, $\Delta=200\bar\rho$) plus
  the "cold dark matter prescription" ($\sigma$ from $P_{cc}$, $\rho=\rho_{cdm}$)
  reproduces the simulated SO HMF; the older "matter prescription" ($\sigma$ from
  $P_{mm}$, $\rho_{cdm}$; Brandbyge et al.) fails increasingly with $m_\nu$ and
  mass. Neutrino contribution to SO halo masses: 0.01% (1e13) to 0.5% (1e15
  $h^{-1}M_\odot$) at 0.6 eV. VERIFIED (1311.1514 Sec. 3, Figs. 1-2).
- Later confirmation, Castro et al. 2023 Euclid HMF (2208.02174): they adopt
  exactly this model ("replace $P_m$ by $P_{cb}$ in the mass variance and ignore
  the neutrino contribution to the Lagrangian patch mass") and validate the HMF
  *response* $\mathcal{R}=n^{(\Sigma m_\nu)}/n^{(0)}$ against DEMNUni (0.16,
  0.32 eV) and the Euclid neutrino code-comparison runs (0.15-0.6 eV): better
  than 1% for $\Sigma m_\nu\le0.32$ eV over the calibrated mass range
  ($\gtrsim 3-4\times10^{13}M_\odot$); a few % at 0.6 eV; below
  $\sim3\times10^{13}M_\odot$ the prescription underestimates the neutrino
  suppression. VERIFIED (2208.02174 Secs. 2.2 and 5.2, Figs. 10-11).
- Error of the total-matter prescription: no paper quotes one clean number; the
  verified quantifications are (i) the b_m vs b_c 5% shift at 0.6 eV (Sec. 2
  below), (ii) Castorina Fig. 2 / Costanzi Fig. 1 showing the $\sigma_m$ curves
  departing from the measured HMF where the $\sigma_{cb}$ curves track it, and
  (iii) the exponential-tail sensitivity (IT12 factor ~2 at 5e15, 0.1 eV), which
  is the same lever arm as the group's own CAMB check (0.1-0.9% growth error ->
  ~20% in Tinker counts at 1e15, z=1). The direction is unambiguous: using
  total-matter sigma *and* total-matter D(a) underpredicts counts and breaks
  universality. VERIFIED for (i)-(iii) as cited; the 20% number is the group's
  own test, not from the literature.

## 2. Halo bias: scale-independent and universal only relative to cb

- Castorina et al. 2014 (1311.1212 Sec. 5): define
  $b_c^{(hh)}=\sqrt{P_{hh}/P_{cc}}$, $b_c^{(hc)}=P_{hc}/P_{cc}$ and the analogous
  $b_m$'s. With $\delta_h=b_c\,\delta_c$ one has $P_{hc}=b_c P_{cc}$ and
  $P_{hm}=b_c P_{cm}$; hence $b_m(k)$ inherits the scale dependence of
  $P_{cm}/P_{mm}$ ($\simeq 1/(1-f_\nu)$ at $k\gg k_{nr}$, 1 at $k\ll k_{nr}$).
  Measured: $b_c(k)$ flat for $k\lesssim0.07\,h/$Mpc, $b_m(k)$ scale dependent
  even on linear scales; at $k=0.07\,h/$Mpc $b_m$ is 5% larger than $b_c$ for
  0.6 eV ($\approx 1/(1-f_\nu)$, $f_\nu\simeq0.046$). Bias vs
  $\nu=\delta_c/\sigma_{cc}$ is universal across z, $m_\nu$ and $\sigma_8$-matched
  models (their Fig. 9); vs $\sigma_{mm}$ it is not. VERIFIED (1311.1212 Sec. 5,
  Eqs. biasc-biasxm, Figs. 6-9). Caution: the printed asymptotic factor in their
  Eqs. reads $b_c(1-f_\nu)$, but their own measurement ("5% larger") and the
  spectra imply $b_m \to b_c/(1-f_\nu)$; treat the displayed limit as a typo.
- Villaescusa-Navarro et al. 2014 (1311.0866 Sec. 3): same conclusion from SO
  halos, in P(k) and xi(r); Tinker 2010 bias + cb prescription matches the
  large-scale bias, the matter prescription overshoots; the scale dependence of
  $b_{hm}=P_{hm}/P_{mm}$ at $k<0.1\,h/$Mpc disappears when computed as
  $P_{hc}/P_{cc}$. VERIFIED (1311.0866 Sec. 3, Figs. 4-8).
- Castorina et al. 2015 DEMNUni (1505.07148): higher-resolution confirmation;
  "the proper definition of the halo bias should be made with respect to the cold
  rather than the total matter distribution"; improper choice biases growth-rate
  measurements at the 1-2% level. VERIFIED at abstract level (arxiv.org/abs/
  1505.07148, isolated summary).
- Consequences for surveys: Raccanelli, Verde & Villaescusa-Navarro 2018
  (1704.07837): neglecting the induced scale dependence is safe for current
  surveys, non-negligible for future ones; simple recipe advocated. Vagnozzi et
  al. 2018 (1807.04672): "galaxies trace cb"; defining bias wrt total matter
  makes it $M_\nu$- and scale-dependent on large scales; on Euclid-like mocks the
  uncorrected bias shifts the inferred $M_\nu$ (and correlated parameters);
  correction = define bias wrt cb, implemented in CLASS. VERIFIED (both
  abstracts).

## 3. Which spectrum for which observable

Theory (from Secs. 1-2 evidence): with a single linear bias $b$ defined against cb,
$$P_{gg}=b^2 P_{cb},\qquad P_{gm}=b\,P_{cb,m},\qquad P_{mm}\ \text{(total) for shear},$$
where $P_{cb,m}=(1-f_\nu)P_{cb}+f_\nu P_{cb,\nu}$ is the cb x total-matter cross
spectrum — this is what galaxy-galaxy lensing probes, since lensing weighs total
matter while the galaxies sit in cb halos ($P_{hm}=b_c P_{cm}$, Castorina 2014
Sec. 5). VERIFIED as algebra + simulation support above; no single paper states
the $\gamma_t$ equation in exactly this form in what I read (the pieces are:
$P_{hh}=b^2P_{cc}$ and $P_{hm}=b_cP_{cm}$, both VERIFIED in 1311.1212; HOD
galaxies on CDM halos in 1311.0866 Sec. 4).

Practice in current analyses/codes:
- DES Y3 (Krause et al. 2021, 2105.13548 Sec. 4.1): $P_{mm}$ = Halofit
  (Takahashi 2012) + Bird 2012 neutrino prescription, total matter; galaxy bias
  linear wrt the *nonlinear total* matter: $P_{\delta_g A}=b_1 P_{mA}$. They
  explicitly state that "neglecting scale dependence of galaxy bias due to
  massive neutrinos" was shown not to bias DES-Y3 (their stress tests). So DES Y3
  = total-matter everywhere, validated as adequate at DES Y3 precision. VERIFIED
  (2105.13548 Sec. 4.1).
- CLASS-PT (Chudaykin et al. 2020, 2004.10607): "we use the linear power spectrum
  for the 'cold dark matter+baryons' ('cb') fluid as an input in all loop
  calculations" for biased tracers; cb is the default (flag cb=No reverts).
  VERIFIED (ar5iv html of 2004.10607, quoted).
- FOLPSnu (Noriega et al. 2022, 2208.02791) handles massive neutrinos in the
  1-loop galaxy power spectrum (used by DESI full shape alongside velocileptors);
  VERIFIED that it models neutrinos in PT (abstract); its cb-tracer statement and
  the exact DESI/velocileptors cb convention: NOT VERIFIED (attempted, not found
  in fetched text).
- Euclid cluster counts: cb prescription adopted (Castro et al. 2023, Sec. 1
  above) — VERIFIED. Euclid spectroscopic GC recipe (EP VII, 1910.09273) using
  $P_{cb}$: NOT VERIFIED (fetch truncated; recalled as true).
- CCL (Chisari et al. 2019, 1812.05995): the paper's Table 1 caveat: the halo
  model "should not be used for massive neutrino models because the current
  version does not distinguish between the cold matter, relevant for clustering,
  and all matter." VERIFIED (ar5iv of 1812.05995, quoted). Behavior of current
  pyccl releases: NOT VERIFIED.
- KiDS-1000 (HMcode-2016 inside): NOT VERIFIED (not checked).

## 4. Halo model of the total-matter spectrum

- Massara, Villaescusa-Navarro & Viel 2014 (1410.6813 Sec. 4): decomposition
  (their Eq. 19)
  $$P_{mm}=(1-f_\nu)^2P_{cc}+2f_\nu(1-f_\nu)P_{c\nu}+f_\nu^2P_{\nu\nu},$$
  with $P_{cc}$ from a halo model built entirely on the cb prescription (their
  Eqs. 30-33, 37-38): $\nu_c=\delta_c/\sigma_c$ with $\sigma_c^2$ from
  $P^L_{c}$, $M_c=\frac{4}{3}\pi\bar\rho_c R^3$, 2-halo term
  $[\int f b\,u]^2 P^L_c$, ST $f$ and $b$, and the *standard LCDM concentration
  formula* ("well described by the standard formula... as our N-body simulations
  showed"). Neutrino and cross terms: linear theory suffices for $P_{mm}$ —
  replacing them by the full nonlinear versions changes $P_{mm}$ by <1%.
  Residuals of the cb halo model vs sims: 15-20% at $k\sim0.2$-$2\,h/$Mpc (z=0),
  up to 30% at z=1 (the usual 1-/2-halo transition problem, not a neutrino
  issue). VERIFIED (1410.6813 Sec. 4, Eqs. 18-51, Figs. 2, 5).
- HMcode-2020 (Mead et al. 2021, 2009.01858): peak height
  $\nu=\delta_c(z)/\sigma_{cc}(M,z)$ (their Eq. peak_height, explicitly citing
  Massara 2014 and Castorina 2014); 2-halo term uses the *total* linear
  $P^{lin}_{mm}$; neutrinos added back by lowering the matter window:
  $W(M,k\to0)=(1-f_\nu)M/\bar\rho$, i.e. 1-halo term $\propto(1-f_\nu)^2$, "hot
  neutrinos cannot cluster in haloes"; $M$ is the mass in an equivalent
  neutrino-free universe, and the $M$-$R$ map uses the total $\bar\rho$ (unlike
  Castorina's $\rho_{cdm}$). Concentration: Bullock-type formation redshift
  solved with $\sigma_{cc}(\gamma M,z)$; Dolag correction computed in an
  LCDM-equivalent with the neutrino mass converted to CDM. All fitted-parameter
  cosmology dependence is through $\sigma_{8,cc}(z)$ and $n^{eff}_{cc}$: "all
  the cosmology dependence of parameters in our model depend on the *cold*
  spectrum". RMS accuracy 2.5% on Mira-Titan (massive-nu) nodes. VERIFIED
  (2009.01858 Secs. 2, 4, 5, Table 2).
- The damping used at halo.c:4238/4267, $k_s=0.05618\,\sigma_8^{-1.013}(z)$, is
  HMcode-2020's one-halo damping $k_*$, Table 2 of 2009.01858, where the fitted
  variable is $\sigma_{8,\mathrm{cc}}(z)$ — the **cold** sigma8, not the total.
  halo.c itself cites "2009.01858 Table 2" (line 4039). VERIFIED (2009.01858
  Table 2; halo.c:4027-4039, 4266).
- HMx (Mead et al. 2020, 2005.00009) has **no** sigma8-dependent damping of its
  own: its formalism is written for total matter ($\sigma$ from the linear
  matter field, $M=\frac{4}{3}\pi R^3\bar\rho$, bias wrt linear matter, Sec. 2),
  and its fitted parameters (Table 2) are functions of z and $T_{AGN}$ only; HMx
  is a *response* multiplied by HMcode. So the owner's question "which sigma8
  does the HMx k_s fit use" resolves to: that formula is HMcode-2020's, and it
  uses $\sigma_{8,cc}(z)$. VERIFIED (2005.00009 Secs. 2 and 5, Table 2;
  2009.01858 Table 2). Side observation from the code: halo.c approximates
  $\sigma_8(z)$ as (table $\sigma_8$) x a; HMcode-2020 wants $\sigma_{8,cc}$
  *evolved with the growth factor*; under HALO_FIELD_CB the table gives the
  right field, and replacing a by D(a) would match the paper more closely
  (code reading, halo.c:4266; the a-vs-D gap is a separate small approximation,
  NOT a literature claim).

## 5. halo.c parameterizations vs the state of the art

Implemented options (read from halo.c): HMF: `HMF_TINKER_2010` only
(fnu_params_at, halo.c:619-635; Tinker 2010 Eqs. 8-12, evolution frozen at
z=3); bias: `HALO_BIAS_TINKER_2010` only (halo.c:275-291); concentration:
`CONCENTRATION_BHATTACHARYA_2013` only (halo.c:677-700,
$c=9.0\,\nu^{-0.29}D^{1.15}$, $\Delta=200\bar\rho$ full sample); plus
`bias_norm` (halo.c:756+) enforcing $\int b f\,d\nu=1$. VERIFIED (file read).

(a) Mass function. Tinker 2008/2010 with $\sigma_{cb}$ is the standard and is
*accurate as a neutrino response*: <1% in $\mathcal{R}$ for
$\Sigma m_\nu\le0.32$ eV at cluster masses (Castro 2023, Sec. 1). The weak link
is the *baseline* (massless) Tinker calibration, not the neutrino mapping:
Castro 2023 find Tinker08 differs from their sub-percent calibration by up to
3% below $10^{15}M_\odot$ and >5% above $2\times10^{15}M_\odot$ at z=0
(2208.02174 Sec. 5.3, VERIFIED); Bocquet et al. 2020 Mira-Titan emulator
(2003.12116): universal-form fits can be biased by up to 30% at
$10^{14}M_\odot/h$, z=0, for $M_{200c}$ (abstract, VERIFIED); Aemulus-nu HMF
emulator (Shen et al. 2024, 2410.00913): Tinker08 + cb prescription is
"significantly" less accurate/precise than the emulator and its cosmology
response depends on the fiducial (their Fig., Sec. "Design", VERIFIED).
Recommendation: keep Tinker 2010 + $\sigma_{cb}$ (and cb growth in
$\nu(M,a)$) as the cb-aware baseline; for cluster-count *calibration* accuracy,
the Castro 2023 fit or an emulator (Mira-Titan, Aemulus-nu) is the current
state of the art — but that is an orthogonal (massless-baseline) upgrade.

(b) Concentration. Bhattacharya et al. 2013 (1112.5479): gravity-only massless
LCDM/wCDM sims, 2e12-2e15 $h^{-1}M_\odot$, z=0-2; the c-nu relation has a
near-constant slope but is NOT universal across cosmology (±20% over the wCDM
space) and the scatter is ~0.33 (abstract, VERIFIED). No neutrinos in the
calibration. With neutrinos, the literature evaluates c(M) on the cold field:
Massara 2014 found the standard LCDM formula adequate for cb halos (VERIFIED,
Sec. 4 above); HMcode-2020 computes the formation redshift from
$\sigma_{cc}$ and converts $m_\nu$ to CDM in the Dolag term (VERIFIED).
Diemer & Joyce 2019 (1809.07326): c(nu, n_eff, alpha_eff) model, 5% in
LCDM/scale-free (abstract, VERIFIED); its neutrino usage (Colossus): NOT
VERIFIED. Recommendation: Bhattacharya 2013 is adequate at the current accuracy
*provided* nu is built from $\sigma_{cb}$ and a cb growth factor — i.e. the
HALO_FIELD_CB table plus a cb D(a); the residual error from c(M) itself (tens
of %) dwarfs the neutrino subtlety in the 1-halo term. For a future upgrade,
Diemer & Joyce 2019 with cb-based nu is the better-motivated choice
(the cb-based-nu extension itself: inference, NOT VERIFIED).

## 6. Consumer-by-consumer recommendation

| Consumer (halo.c / halo_cluster.c) | Field for sigma(M), rho, nu | Evidence |
|---|---|---|
| Cluster counts + cluster bias (halo_cluster.c:1591-1642) | cb: $\sigma_{cb}$, $\rho_{cb}$, cb growth in $\nu=\delta_c/(\sigma D_{cb})$ | Strong, VERIFIED: 1108.4688; 1311.1212; 1311.1514; 2208.02174 (<1% to 0.32 eV) |
| Tinker fnu + halo bias + bias_norm (halo.c:641, 275, 756) | cb throughout, consistently (same nu in f, b and the $\int bf=1$ norm) | Strong, VERIFIED: 1311.1212 Figs. 2, 9; 1311.0866 Sec. 3 |
| HOD tables, p_gg 2-halo (halo.c:3408, 5624) | halo statistics from cb; 2-halo $b^2 P_{cb}$ (not $b^2P_{mm}$) | Strong for the principle, VERIFIED: 1311.0866 Sec. 4 (HOD on CDM halos); 1807.04672; 2004.10607. Current code multiplies total-matter nonlinear Pdelta (halo.c:5352, 5803); DES Y3 validated that this is harmless at Y3 precision (2105.13548) |
| p_gm / gamma_t (halo.c:5089) | mixed: galaxies cb, lensing total: $P_{gm}=b\,P_{cb,m}$; 1-halo with the total-matter halo profile | Medium: assembled from VERIFIED pieces ($P_{hm}=b_cP_{cm}$, 1311.1212); no single reference states the gamma_t form |
| Cosmic shear / p_mm, p_my, p_yy (halo.c:3749, 4092, 4498) | observable is total matter; best halo-model internals: cold mass function/bias/concentration, 1-halo $\times(1-f_\nu)^2$, 2-halo $=P^{lin}_{mm}$ (total) | Strong, VERIFIED: 1410.6813; 2009.01858 Sec. 2. The current refusal of HALO_FIELD_CB ("I11^2 P_lin has no cb form") is over-cautious: HMcode-2020 gives exactly the cb-internals + total-P_lin recipe |
| HMx damping k_s (halo.c:4238, 4267) | $\sigma_{8,cc}(z)$ (cold) per the source of the fit, HMcode-2020 Table 2 | Strong, VERIFIED: 2009.01858 Table 2 |
| conc() Bhattacharya (halo.c:677-700) | nu from $\sigma_{cb}$ + cb growth | Medium: 1410.6813 (LCDM c(M) fine for cb halos), 2009.01858 ($\sigma_{cc}$ in z_f); no dedicated nu-c(M) calibration read |
| Halo-model IA (ia_tables, halo.c:7166) | halo statistics (f, b, c) from cb; tidal field is total matter | Inference only, NOT VERIFIED: no paper on cb vs m in halo-model IA was read |

Bottom line: every consumer that counts or weights *halos* (counts, bias, HOD,
concentration, IA occupation) belongs on the cb field with a cb growth factor;
only the *linear spectrum in 2-halo terms of total-matter observables* (shear,
y-cross) should stay total-matter, which is precisely HMcode-2020's split. The
single a=1 sigma^2 table rescaled by the total-matter D(a) at
$k_0=5\times10^{-4}$/Mpc reproduces neither convention exactly in HALO_FIELD_CB
mode: the table field is cb but the growth is total-matter and scale-fixed,
which is the 0.1-0.9% (growth) -> up to ~20% (counts at 1e15, z=1) effect the
group measured; the literature-standard cure is $\sigma_{cb}(M,z)$ grown with
the cb growth at halo scales (Castro 2023 operational definition, VERIFIED).
