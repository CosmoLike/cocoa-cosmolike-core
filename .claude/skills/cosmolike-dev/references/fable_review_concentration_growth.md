# Which D(z) in Bhattacharya 2013's c = 9.0 nu^-0.29 D^1.15 with massive neutrinos?

Follow-up to fable_review_F_neutrino_halos.md. Code reference: conc(),
`/Users/vivianmiranda/data/COCOA/september2026/test/cocoa/Cocoa/external_modules/code/cosmolike_core/cosmolike/halo.c:677-700`.
Tags: VERIFIED (source read, with arXiv id + section/eq/fig) or NOT VERIFIED.

## 1. What D(z) is in Bhattacharya et al. 2013 (1112.5479)

- The fit (their Table "c(nu)-nu fitting formulae", Sec. 4.2): for
  $\Delta=200\rho_b$ (the halo.c definition), full sample,
  $c = D(z)^{1.15}\,9.0\,\nu^{-0.29}$; relaxed: $D^{1.2}\,10.1\,\nu^{-0.34}$.
  $\nu=\delta_c(z)/\sigma(M,z)$ with $\delta_c=1.673$ for the reference
  cosmology, "varying only mildly"; their $\nu$-$M$ fit is
  $\nu(M,z)\approx\frac{1}{D(z)}[1.12(M/5\times10^{13}\mathrm{M}_\odot/h)^{0.3}+0.53]$,
  i.e. in their system $\sigma(M,z)=\sigma(M,0)D(z)$ with one and the same
  scale-independent linear growth $D$, normalized $D(0)=1$ (required for the
  z=0 amplitude 9.0 and the 1/D form of the $\nu$-$M$ fit to be mutually
  consistent). VERIFIED (1112.5479 Sec. 4.2, Table tab:fit).
- Cosmologies and sigma: the calibration suite is the massless WMAP7-like
  reference LCDM (GS/HACC runs) plus 18 massless Coyote wCDM models
  ($(1.3\,$Gpc$)^3$, Gadget-2; wCDM concentrations rescaled by 1.05 for force
  resolution). All runs are neutrino-free, so "total matter" = cb and
  $\sigma(M,z)$ is the total-matter variance of massless cosmologies; the
  question "total or cb sigma" does not arise inside the paper. VERIFIED
  (1112.5479 Sec. 5, Table table_wcdm; abstract).
- Physical role of $D^{1.15}$: it is the redshift evolution of the amplitude of
  the c-nu relation at fixed $\nu$ (the c-M relation flattens with z; in
  c-nu form the shape is constant and only the amplitude drops). Measured
  evolutions: $c_{200}(\nu)\sim D^{0.54}$, $c_{vir}\sim D^{0.9}$,
  $c_{200\rho_b}\sim D^{1.15}$ (the $\Delta$ definition absorbs different
  amounts of background evolution). VERIFIED (1112.5479 Sec. 4.2 and
  Fig. c-Mnu). The c-nu relation is NOT universal across cosmology: +-20%
  over the wCDM space while D itself varies by <5% there — they explicitly
  note that a Dolag-style growth-ratio multiplier cannot explain the
  cosmology dependence. VERIFIED (1112.5479 Sec. 5, Fig. meancwcdm).
- How D was computed numerically in their analysis (ODE vs fit): not stated in
  the sections read. NOT VERIFIED.

So: in B13 there is a single growth function, entering twice — through
$\nu\propto 1/D$ and through the $D^{1.15}$ prefactor. Net sensitivity of the
$200\rho_b$ full-sample fit to D at fixed $\sigma(M,0)$:
$$c\;\propto\;\nu^{-0.29}D^{1.15}\;\propto\;D^{0.29}\,D^{1.15}=D^{1.44}.$$
(Derivation from the VERIFIED fit; the 1.44 combination itself is my algebra.)

## 2. c(M,z) in simulations WITH massive neutrinos

- Mummery et al. 2017 (1702.02064, BAHAMAS/cosmo-OWLS): neutrino
  free-streaming lowers the amplitude of the c-M relation with "no significant
  effect" on profile shapes inside $r_{200}$; the amplitude drop is driven by
  the halo-mass reduction above $\log M_{200c}\simeq14.5$ and by a slight
  scale-radius increase below; neutrino and baryon effects multiply
  independently to few-% accuracy. Their Duffy-form fits
  ($c=A(M/10^{14}\mathrm{M}_\odot)^B(1+z)^C$, DM-only runs, WMAP9 block):
  $A = 4.553,\,4.498,\,4.411,\,4.329,\,4.055$ for the five neutrino runs in
  increasing $\Sigma m_\nu$ (labels nua-nue; the text identifies nua as the
  massless baseline; BAHAMAS masses 0.06-0.48 eV for the rest — label-to-mass
  map inferred, NOT VERIFIED beyond that). That is a monotonic ~11% drop in
  amplitude from 0 to the heaviest run, i.e. roughly 20-25% per eV, with B and
  C nearly unchanged. VERIFIED (1702.02064 abstract; Sec. 4 "Halo structure";
  Appendix Table tab:DuffyABC), except the stated label-mass mapping.
- Brandbyge et al. 2010 (1004.4105 Sec. 3.2): at matched total mass
  ($10^{13}\mathrm{M}_\odot$), massive neutrinos lower the inner density
  ($r\lesssim100\,h^{-1}$kpc) and raise it outside: lower concentration,
  explained by later formation ($c\propto1/a_c$) because the linear transfer
  function is suppressed. Their $\delta_\nu=0$ experiment (0.6 eV): removing
  the neutrino *perturbations* lowers the halo density a further 3-4%, i.e.
  neutrino clustering partially counteracts the background effect — a
  Boltzmann-code cb growth (which includes the neutrino source term in the cb
  equations) captures this; a pure background rescaling does not. VERIFIED
  (1004.4105 Sec. 3.2, Fig. relative_profiles).
- Massara et al. 2014 (1410.6813 Sec. 4.2): for cb halos in their neutrino
  sims the concentration "is well described by the standard formula" of the
  LCDM case — i.e. an unmodified massless c(M) applied to cb halos works at
  their accuracy. VERIFIED (1410.6813 Sec. 4.2, text at Eq. 37).
- Ichiki & Takada 2012 (1108.4688 Sec. 3.2): the neutrino effect on collapse is
  captured by the cb linear growth; nonlinear neutrino clustering is unlikely
  to change the internal mass distribution of halos. VERIFIED (1108.4688
  Sec. 3.2, first paragraph).
- HMcode-2020 (2009.01858, Sec. "HMcode 2020"): the only widely used model
  with an explicit, documented neutrino treatment of concentration. Formation
  redshift from $\frac{g(z_f)}{g(z)}\sigma_{\mathrm{cc}}(\gamma M,z)=\delta_c(z)$ —
  the *cold* variance; for scale-dependent growth they use the linear growth
  "in the large-scale limit with neutrinos clustered along with CDM" as the
  time variable $g(z)$; the Dolag correction's LCDM-equivalent converts the
  neutrino mass into CDM. VERIFIED (2009.01858 Sec. 4, Eqs.
  concentration_mass, formation_redshift, Dolag_correction).
- DEMNUni, Quijote, Mira-Titan, Euclid neutrino code comparison, Bayer et al.,
  Ishiyama (Uchuu), Hagstotz et al.: I found no dedicated c(M, m_nu)
  calibration from these suites in the searches performed. NOT VERIFIED
  (absence claim; searched today). HMcode-2016 (1602.02154) neutrino details:
  not read. NOT VERIFIED.

None of the above expresses c(M, m_nu) through a *total-matter* growth; where
a field is specified it is cb/cold (Massara, HMcode-2020), and where the
mechanism is analyzed it is the cb formation history (Brandbyge, Ichiki &
Takada).

## 3. Which D for c = A nu^a D^b with neutrinos?

- Direct literature answer: **no paper addresses which growth factor to insert
  in the Bhattacharya (or any $A\nu^aD^b$) fit in a massive-neutrino
  cosmology.** The fit was never calibrated with neutrinos; the choice is an
  extrapolation whichever D is used. (Plain statement; absence NOT VERIFIED
  beyond the searches done.)
- Best-supported choice by consistency: in B13 the D in the prefactor and the
  D in $\nu$ are the same function, the one that carries $\sigma(M,z)$ from
  z=0 to z. Under the cb prescription that function is the cb growth at halo
  scales, $D_{cb}(z)\equiv\sigma_{cb}(M,z)/\sigma_{cb}(M,0)$ (mildly
  mass-dependent; a single $D_{cb}$ at a representative halo-scale k is the
  practical version). This is also HMcode-2020's logic: cold variance carries
  the physics, a single growth function is only a time-translation device.
  VERIFIED precedents as cited in Sec. 2; the application to B13 specifically
  is inference, NOT VERIFIED.
- Size of the choice: with the group's CAMB numbers ($D_{cb}$ at halo scales
  vs total-matter D at $k_0=5\times10^{-4}$/Mpc differing by 0.1-0.9% for
  $m_\nu=0.06$-0.6 eV at z=0.3-1), the full sensitivity $c\propto D^{1.44}$
  (prefactor $D^{1.15}$ plus $\nu^{-0.29}\propto D^{+0.29}$) gives
  **0.14-1.3% in c** between the two growth choices (the task's quoted
  1.15x route alone gives 0.1-1.0%). For comparison, all VERIFIED error
  scales of the same quantity are larger: +-20% cosmology non-universality of
  c-nu (1112.5479 Sec. 5), $\sigma_c=0.33c$ per-halo scatter (1112.5479
  Sec. 4.3), 10-20% systematic spread between simulation analyses and a 1.05
  force-resolution rescale (1112.5479 Secs. 4.1, 5), ~11% physical neutrino
  suppression of the c(M) amplitude at 0.48 eV (1702.02064 Table
  tab:DuffyABC), 3-4% neutrino-perturbation effect at 0.6-1.2 eV (1004.4105).
- The scale-independent Eisenstein-Hu-type $D_1$ (or cosmolike's D from
  $P_{lin}$ at $k_0=5\times10^{-4}$/Mpc) is the *least* motivated option with
  neutrinos: at $k_0<k_{nr}$ it tracks the total-matter large-scale growth,
  which misses the free-streaming suppression at halo scales entirely — this
  is exactly the 0.1-0.9% growth error the group measured, amplified by the
  exponential HMF tail elsewhere. (Code behavior per the task context and
  cosmo3D.c:1740-1975, read; characterization VERIFIED from code.)

## 4. Recommendation for cosmolike

1. Keep Bhattacharya 2013 as the c(M) model for now: its own +-20%
   non-universality and the 0.33c scatter dominate every neutrino-related
   subtlety, and Massara 2014 verified that an unmodified massless c(M) on cb
   halos is adequate at current halo-model accuracy.
2. In HALO_FIELD_CB mode, build *one* cb growth factor $D_{cb}(a)$ at halo
   scales (e.g. from $\sqrt{P_{cb}(k_h,a)/P_{cb}(k_h,1)}$ at
   $k_h\sim0.1$-0.5/Mpc, or $\sigma_{8,cb}(z)$ ratios) and use it everywhere
   the a=1 sigma table is rescaled — fnu, halo bias, bias_norm *and* both D's
   inside conc() (the $\nu$ argument and the $D^{1.15}$ prefactor). Splitting
   the two D's in conc() has no calibration basis and would break the
   internal consistency of the B13 fit (and the group's
   historical-consistency rule). The conc()-specific gain from the better D is
   only 0.14-1.3%, but it comes for free once the shared growfac is fixed for
   the mass function, where the same 0.1-0.9% matters at the tens-of-percent
   level in cluster counts.
3. B13's $D(0)=1$ normalization matches cosmolike's growfac convention
   (D(a=1)=1, halo.c:677-700 comment); no renormalization needed. VERIFIED
   (code + 1112.5479 Table tab:fit).
4. If an explicit neutrino treatment is ever wanted: HMcode-2020's
   formation-redshift model is the documented precedent (cold variance +
   large-scale growth with neutrinos clustered along CDM, VERIFIED); Diemer &
   Joyce 2019 with cb-based peak height is the better-motivated modern c(M)
   (5% in massless LCDM/scale-free, VERIFIED abstract of 1809.07326), but its
   massive-neutrino usage is NOT VERIFIED anywhere I read. Note from Mummery
   2017 that the neutrino effect on c(M) is mostly an amplitude shift tracking
   the suppressed growth — a $\nu$-based model fed with $\sigma_{cb}$ already
   captures most of it automatically.
