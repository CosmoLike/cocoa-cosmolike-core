# Plan: growth factor and neutrino-aware halo quantities in cosmolike

Status: proposal for the owner's approval (2026-10-02). Nothing is
implemented. Evidence, saved in cosmolike_core
`.claude/skills/cosmolike-dev/references/`:

| file | content |
|---|---|
| `fable_review_growth_factor.md` | origin of the D(z) in NLA/TATT; every growfac consumer (file:line) |
| `growth_factor_measurements.md` | CAMB: dark-energy perturbations on/off, k scan, neutrino mass vs Eisenstein & Hu 1999 |
| `fable_review_neutrino_halos.md` | literature (VERIFIED with arXiv sections): P_cb vs P_mm per consumer; state of the halo.c fits |

Rules for every phase: correctness and consistency over speed; one change
per commit on `bugfix`; stored (frozen) chi2 values are re-recorded only
where a change moves them by design, in a separate commit that states why;
determinism sweep (1 and 8 threads); NLA and TATT; at most two 4-thread
jobs; a Fable didactic review at the end.

## Phase 1: growth factor at a sub-horizon k (Python only)

The likelihoods build cosmolike's growth table from CAMB's linear spectrum
at k0 = 5e-4/Mpc, a horizon scale. For w != -1, CAMB's dark-energy
perturbations change the growth there by 0.5-0.9% (z = 0.5-2) relative to
every scale cosmolike models.

1. Replace `0.0005` by one attribute `growth_k` (default 0.05/Mpc) in the two
   G_growth lines of each project's `likelihood/_cosmolike_prototype_base.py`:
   lsst_y1, roman_real, roman_fourier, roman_kl, des_y3, desy1xplanck,
   des_cluster.
2. Same change in the test copies (`roman_real/tests/test_notebook_interface.py`).
3. Check w = -1, mnu = 0: the chi2 should not move (< 1e-3).
4. Run all tests of the seven projects; re-record the moved stored chi2 values.
5. Rerun the CoCoA vs DESC-CCL comparison with the new table (no `growth_sub`
   workaround); update its README.

## Phase 2: sigma^2(M, a) from both fields (cosmo3D.c)

Today one sigma^2(M) table is built at a = 1 from the field a global switch
picks (`like.halo_model[4]`), and every consumer rescales it with the
total-matter D(a) at k0.

1. Build two tables on the existing (ln M, a) grid:
   - sigma^2_m(M, a) from P_lin(k, a), R(M) with rho_m;
   - sigma^2_cb(M, a) from P_cb(k, a), R(M) with rho_cb.
2. The algorithm follows the study in `sigma_fftlog/` (owner request,
   2026-10-02): (a) an FFTLog sigma^2(R) without ringing, to the accuracy of
   the existing lobe-by-lobe quadrature in x = kR; (b) which is faster for
   N_a redshifts: the quadrature per a, a precomputed quadrature-weight
   matrix (one matrix product for all a), or FFTLog threaded over a as the
   non-Limber FFTLog is in cosmo2D.c.
   Study result (2026-10-02, `sigma_fftlog/README.md`): FFTLog reaches
   1.9e-9 in sigma^2 and 1.2e-8 in d ln sigma/d ln M over 1e6-1e17 Msun/h
   (b = 1.5, input continued with p_lin's edge power laws to 1e5 h/Mpc,
   closed-form Mellin kernel of W^2, slope from a second inverse FFT);
   the quad has 5.5e-5 and 2.7e-3. In Python, for 256 a nodes: FFTLog
   19 ms (13.6 ms threaded per a node at 4 threads) against 1010 ms for the
   quad, bit-identical across thread counts; the BLAS weight matrices are
   not deterministic. Decided (owner, 2026-10-02): FFTLog over the a nodes, threaded as
   cosmo2D.c threads the non-Limber FFTLog (whole a rows per thread, plans
   built once), with the study's settings.
3. Provide d ln sigma/d ln M per a as well.
4. Request P_cb from CAMB whenever the run has halo consumers (HOD, cluster
   counts, halo-model IA, the halo-model p_mm); today only with
   `halo_matter_field: 1`. Every run has massive neutrinos (the minimum
   mnu is 0.06 eV), so the condition is the halo consumers alone. The C++
   hand-off (`set_linear_power_spectrum_cb`) already exists.
5. Check: with mnu = 0 both tables agree with D^2(a) sigma^2(M, 1) to the
   growth's own scale dependence (< 1e-4); the tests stay inside their band.

## Phase 3: each consumer on its field (halo.c, halo_cluster.c)

| consumer | field | evidence (Fable F) |
|---|---|---|
| cluster counts and cluster bias (`halo_cluster.c:1591-1642`) | cb: sigma_cb(M, a), rho_cb, no D rescaling | strong (Castro et al. 2023: < 1% to 0.32 eV) |
| Tinker f(nu), halo bias, bias_norm (`halo.c:641`, 275, 756) | cb, the same nu in all three | strong |
| HOD tables, p_gg and p_gm halo statistics (`halo.c:3408`, 5089, 5624, 7524) | cb | strong |
| concentration, Bhattacharya 2013 (`halo.c:677-700`) | nu from sigma_cb(M, a) | medium |
| halo-model IA (`ia_tables`, `halo.c:7166`) | cb halo statistics | inference only |
| p_mm (`halo.c`) | open question: p_mm feeds no data vector (only its Python wrapper and tests); HMcode-2020 is the published neutrino recipe for a halo-model P_mm | not decided |

p_my and p_yy (the thermal-SZ spectra) are not compiled: they are kept in
cosmolike_core `future_port_unfinished/halo_tsz.c` (owner, 2026-10-02).
Restoring them needs the same cb treatment and the cold sigma_8(z) in
their HMcode-2020 damping k_s.

Check: with mnu = 0 every consumer reproduces Phase 2 bit for bit or within
1e-4 (the two fields coincide); with mnu > 0 quantify each consumer
(des_cluster counts first).

## Outside this plan (separate decisions)

1. Galaxy clustering and galaxy-galaxy lensing outside the halo model:
   P_gg = b^2 P_cb and P_gm = b P_cb,m. Needs nonlinear cb and cb x m
   spectra; DES Y3 validated total matter as adequate at its precision,
   Vagnozzi et al. 2018 show it matters for Euclid-like data.
2. Scale-dependent growth rate f_cb(k, z) for RSD (Eisenstein & Hu 1999).
3. One-loop terms on P_lin(k, z) at a few z nodes (heavy neutrinos).
4. Mass function and concentration baselines: Castro et al. 2023 fit or an
   emulator (Mira-Titan, Aemulus-nu); Diemer & Joyce 2019 concentration.

## Decisions taken

1. The halo mass function and every halo statistic built on it (halo bias,
   bias_norm, concentration, HOD, cluster counts and bias) use
   nu = delta_c/sigma_cb(M, z), with sigma_cb from P_cb(k, z) and the
   mass-radius map through rho_cb: the prescription under which Tinker-type
   fits stay universal with massive neutrinos (owner, 2026-10-02; evidence:
   fable_review_neutrino_halos.md, Secs. 1-2).
2. Mass-radius map: rho_cb for the halo statistics (Castorina et al. 2014,
   Castro et al. 2023) (owner, 2026-10-02). The halo-model p_mm is an open
   question (below).
3. Phase 1 approved: growth factor at k = 0.05/Mpc in the seven
   likelihoods (owner, 2026-10-02).
4. p_my and p_yy are not compiled: kept in cosmolike_core
   future_port_unfinished/ (owner, 2026-10-02).
5. Open, literature searched (Fable, fable_review_concentration_growth.md):
   no paper says which growth factor the D^1.15 of the Bhattacharya 2013
   concentration takes with massive neutrinos (the fit has no neutrino
   calibration). In the fit the D of the prefactor and the D inside nu are
   one function, the one carrying sigma(M, z) from z = 0 to z. Recommended:
   D_cb(M, a) = sigma_cb(M, a)/sigma_cb(M, 1) in both places (it follows
   from the Phase 2 table). Size: c moves by 0.14-1.3% (mnu = 0.06-0.6 eV,
   z = 0.3-1) against the fit's own +-20% cosmology dependence. Awaiting the
   owner.
6. p_mm (halo-model matter spectrum, no likelihood uses it) moves to
   future_port_unfinished/ (owner, 2026-10-02); the covariance rewrite
   builds its own P_hm next to D_hm.
7. sigma^2(M, a) by FFTLog (owner, 2026-10-02; Phase 2, step 2).

## Decisions for the owner (open)

1. des_cluster reference scripts (`tests/reference/ref_cosmology.py` and
   three others mirror the DES reference code at k0 = 5e-4/Mpc): follow
   Phase 1, or keep the external convention?
2. Phases 2 and 3 as scoped (with p_mm, p_my, p_yy out of the compiled
   code, Phase 3 covers the halo statistics and the HOD spectra only)?
3. The D^1.15 factor of the Bhattacharya concentration: the recommended
   cb growth at halo scales (decision taken item 5)?
