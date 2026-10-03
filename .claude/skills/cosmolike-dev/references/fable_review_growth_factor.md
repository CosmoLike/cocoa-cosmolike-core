# Review E: which D(z) do NLA/TATT (and the rest of cosmolike) actually want?

Question under review: "In NLA, the 1/D(z) factor comes from k_NL, which depends on
D(z) — so the definition of that D(z) comes from the time evolution of k in which
sigma(z) = 1 for every z, right?" — and the proposal to compute k_NL(z) and use the
growth at k_NL(z) as the IA growth factor.

Short answer: no. The 1/D(z) in the (N)LA amplitude is pure linear theory — it never
references k_NL or sigma(R,z) = 1. "Nonlinear" in NLA means only the substitution
P_lin -> P_NL in the spectra; the amplitude is carried over from the linear derivation
unchanged. The D in every IA formula is the linear growth of the density/tidal field at
the sub-horizon k of the P(k) it multiplies. The k_NL(z) proposal is therefore not
physically required, and numerically it is the growth at a fixed sub-horizon k to
within ~0.03% (see Section 3) — while breaking the f = dlnD/dlna property of the table.

## 1. Physics: where the 1/D(z) comes from

**Linear alignment (Catelan, Kamionkowski & Blandford 2001; Hirata & Seljak 2004).**
The intrinsic shear is set by the linear tidal field at the epoch of galaxy formation
(HS04 eq. 12, their S[.] is a smoothing):

$$\gamma^I = -\frac{C_1}{4\pi G}\left(\partial_x^2-\partial_y^2,\ 2\partial_x\partial_y\right)\Psi_P ,$$

with $\Psi_P$ the linear potential at formation, during matter domination. Poisson in
comoving Fourier space:

$$\Psi(k,z) = -\frac{3}{2}\,\Omega_m H_0^2\,\frac{(1+z)\,\delta_{\rm lin}(k,z)}{k^2},
\qquad \delta_{\rm lin}(k,z) = D(z)\,\delta_0(k),\ D(0)=1 .$$

During matter domination $(1+z)D(z) \to$ const, so $\Psi$ freezes. Writing the frozen
$\Psi_P$ in terms of the linear density field at the *observed* redshift,

$$\Psi_P(k) = -\frac{3}{2}\,\Omega_m \frac{H_0^2}{k^2}\,[(1+z)D(z)]_{\rm MD}\;
\frac{\delta_{\rm lin}(k,z)}{D(z)} ,$$

the constant $[(1+z)D]_{\rm MD}$ is absorbed into $C_1$, and the matter–intrinsic
cross-spectrum becomes

$$P_{\delta,\gamma^I}(k,z) = -A_1\,\frac{C_1\rho_{\rm crit}\Omega_m}{D(z)}\,P_{\rm lin}(k,z) .$$

The 2010 erratum to HS04 fixes the redshift scaling: the prefactor is the comoving
$\bar\rho_{m,0}=\rho_{\rm crit}\Omega_m$, not $\bar\rho_m(z)$, so the amplitude goes as
$1/D(z)$ with no extra $(1+z)$ factors. So the $1/D$ is bookkeeping: the alignment
field is *frozen at formation* (it does not grow), while the density it is correlated
against grows as $D(z)$. The condition $\sigma(R,z)=1$ (i.e. $k_{\rm NL}$) appears
nowhere. **NLA (Bridle & King 2007)** is this formula with $P_{\rm lin}\to P_{\rm NL}$
(halofit) — an ad hoc swap of the spectrum only; the "nonlinear" refers to the P(k),
never to the amplitude. $k_{\rm NL}$ enters NLA only implicitly, through where halofit
bends $P_{\rm NL}$ away from $P_{\rm lin}$.

**TATT (Blazek et al. 2019, PRD 100, 103506).** Expansion of $\gamma^I$ in the
*formation-epoch* tidal field $s_{ij}$:
$\gamma^I_{ij} = c_1 s_{ij} + c_2 (s_{ik}s_{kj} - \tfrac{1}{3}\delta_{ij}s^2) + b_{\rm TA} c_1\,\delta\, s_{ij}+\dots$, with

$$c_1(z) = -A_1\,\frac{\bar C_1\rho_{\rm crit}\Omega_m}{D(z)},\qquad
  c_2(z) = 5A_2\,\frac{\bar C_1\rho_{\rm crit}\Omega_m}{D(z)^2} .$$

One frozen tidal field per power: the TT term is quadratic, hence $1/D^2$. The one-loop
spectra are FAST-PT convolutions of $P_{\rm lin}(k,z{=}0)$ with EdS kernels; each term
quadratic in $P_{\rm lin}$ evolves as $D(z)^4$ under separable growth, so the code
multiplies the $a=1$ tables by $D^4$. Those convolution integrals are supported at
quasi-linear $k \sim 0.05\text{--}1\,h/$Mpc.

**Which scale's growth.** In all of these, D converts between the formation-epoch field
and the late-time *linear* field at the same $k$ as the $P(k)$ in the formula. With
scale-dependent growth the exact generalization is $D(k,z)$ matched in $k$ inside the
integrand; any single-k table is an approximation, and the best single scale is in the
quasi-linear window the observables weight ($\sim 0.05$–$0.5$/Mpc) — not the horizon
scale $k_0 = 5\times10^{-4}$/Mpc ($\approx 2.2\,H_0/c$), and not a $\sigma=1$ condition.

## 2. Code audit (all paths relative to Cocoa/external_modules/code/cosmolike_core/cosmolike/)

Table source: projects/lsst_y1/likelihood/_cosmolike_prototype_base.py:437 (roman_real:440),
`G = sqrt(PKL.P(z, 5e-4)/PKL.P(0, 5e-4))*(1+z)`, normalized at z_2D[-1]; fed through
generic_interface.cpp:2470 `set_growth` into `cosmology.G`. Every C-side D and f below
inherits this single table — that is the consistency mechanism to preserve.

**Readers (cosmo3D.c, the 5 sites).** growfac (522), norm_growfac (568/616),
f_growth (705/741, f = dlnD/dlna = slope of the table), norm_growfac_all (807/867),
growfac_all (947). Verified: D = G·a/G(0); f is a finite-difference slope of the table.

**IA amplitudes** (defined IA.c, built in cosmo2D.c cores; none in radial_weights.c):
- IA.c:293 `x = Omega_m*c1rhocrit_ia/growfac_a` — the NLA/TA $1/D$. IA.c:358 — the TT
  $1/D^2$. IA.c:372–417 BTA (no D). Physical k: quasi-linear (the k of P_delta in the
  same integrand). With k0 at w = -0.9: A1 biased +0.8% (z=1), A2 +1.6%. Flag (soft:
  largely degenerate with the A_IA, eta nuisances, but a parameter-inference bias).
- Callers: cosmo2D.c:3176–3178 (dC_ss_dlnk_tomo_limber_work), 4217–4219
  (C_gs_tomo_limber_work), 7401 (dC_ks_dlnk_tomo_limber_work); node fill
  create_cosmo_nodes 2183 / create_cosmo_nodes_lens 2346; cluster-source leg
  cosmo2D_cluster.c:445 (+1265 warmup read), used with IA_A1_Z1.

**TATT / one-loop $D^4$.** pt_cfastpt.c contains no growfac: its tables are built from
p_lin(k, 1.0) (pt_cfastpt.c:213, 280, 415, 1056). The $D^4$ lives in the Limber cores:
cosmo2D.c:3168 (g4, ss), 4435 and 4452 (gs: one-loop bias + TATT), 5664 (gg), 6337
(gk). Physical k: FAST-PT support, quasi-linear. With k0 at w = -0.9:
$(1.0081)^4 - 1 \approx +3.3\%$ on the one-loop pieces at z = 1 (the pieces are
themselves corrections, so the C_l impact is a fraction of that). Flag.

**Halo model / mass function** — $\nu = \delta_c/(\sigma(M)\,D(a))$ with $\sigma(M)$
tabulated at a = 1 (cosmo3D.c:1747 sigma2 header; 1962: P_cb table at a = 1, consumers
rescale by total-matter D "as the DES reference code does"):
halo.c:689–690 conc (Bhattacharya: $\nu$ and $D^{1.15}$), 837 bias_norm, 3418+3437
hod_tables, 3684 halo_warmup, 3874 p_mm, 4254 p_my, 4650 p_yy, 5109 p_gm, 5660 p_gg,
7188 ia_tables (halo-model IA), 7500 hod_bgal_direct;
halo_cluster.c:1626 (AR_GROWTH fill) consumed at 1642 by the Tinker dn/dlnM and bias of
the **cluster counts**. Physical k: $\sigma(M)$'s integrand peaks at
$k \sim 1/R(M) \sim 0.1$–$3\,h$/Mpc — sub-horizon by construction. **Strongest flag:**
at w = -0.9 the k0 table overstates D by ~0.8% (z=1) to ~0.9% (z=2); with the Tinker
sensitivity $d\ln n/d\ln\sigma \simeq 2$–4 at cluster masses this is a coherent
~2–3% bias in predicted cluster counts — far above the 0.5% bar.

**Growth rate f (RSD).** radial_weights.c:184 f_rsd -> f_growth; cosmo2D.c:9109–9111
(C_cl_tomo_core, FFTLog RSD row $-\chi\,n\,D\,(H/H_0)\,f$ at 9122) and 9718–9720
(C_gs_tomo_core); W_RSD in the Limber cores via radial_weights.c:190+. With k0 the
horizon boost grows from +0.81% (z=1) to +0.93% (z=2), so
$\Delta f \approx \Delta\ln D/\Delta\ln a \approx -0.3\%$ at w = -0.9. Below 0.5%; noted.

**Bias evolution.** bias.c:41–42 (B1_PER_BIN_PASS_EVOLV, D-ratio across a bin), 50
(B1_GROWTH_SCALING, 1/D). Ratios over modest $\Delta z$ or absorbed by the per-bin
nuisance: < 0.1% effect. No flag.

**Non-Limber FKEM separable term.** cosmo2D.c:8881–8890 (design note), 9174/9825 and
4173/5434 (per-bin pivot $1/D(a_{\rm piv})^2$), 4260/5500 (comments: the
scale-dependent part of the growth deliberately lives in the Limber P_delta term). Only
$D(z)/D(z_{\rm piv})$ across one bin's support enters — matches the measured fact that
the choice barely matters here. No flag. But both legs must keep sharing one table or
the FFTLog/Limber cancellation at high l degrades.

**Warmups (no physics).** cosmo2D.c:3041, 7279, 9053; cosmo2D_cluster.c:409/419 (enum
and note); halo_cluster.c:1265.

## 3. Options for the D the table should carry

Reference numbers (given; CAMB 1.6.7, $D(k,z{=}1)/D(k_0,z{=}1)-1$ in %):

| k [1/Mpc]            | 1e-3 | 3e-3 | 0.01 | 0.05 | 0.2  |
|----------------------|------|------|------|------|------|
| w=-0.9, DE pert on   | 0.48 | 0.77 | 0.81 | 0.81 | 0.81 |
| w=-0.9, DE pert off  | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| w=-0.9, mnu=0.06     | 0.48 | 0.78 | 0.85 | 0.92 | 0.95 |
| w=-1                 | 0.02 | 0.02 | 0.02 | 0.02 | 0.02 |
| w=-1, mnu=0.06       | 0.02 | 0.03 | 0.06 | 0.13 | 0.16 |

(a) **Keep k0 = 5e-4.** A horizon scale: for w != -1 it carries CAMB's fluid-DE
perturbation response that no sub-horizon consumer should see (+0.8–0.9% in D), and it
misses the neutrino suppression (0.1–0.2%). It already produced a real artifact: the
$\Delta\chi^2 = 8.76$ CoCoA-vs-CCL difference removed by handing CCL the k = 0.05
growth. Section 2 shows every consumer is sub-horizon physics. Keeping it is only
defensible as a frozen convention at w = -1.

(b) **Fixed sub-horizon k.** D(k) is flat for $k \gtrsim 0.01$ at mnu = 0 and has only
the mild neutrino slope at mnu = 0.06 (0.85 -> 0.95 over 0.01 -> 0.2). k = 0.05/Mpc
sits inside the $\sigma(M)$ / one-loop / shear window, is the scale already validated
in the CCL fix, and differs from k = 0.2 by 0.03%. One-line change per likelihood; C
code untouched; f stays a clean fixed-scale slope.

(c) **Owner's growth at $k_{\rm NL}(z)$.** $k_{\rm NL}(z)$ (from $\sigma(1/k,z)=1$) runs
over ~0.15–1/Mpc for z = 0–2 — entirely inside the flat/saturated part of D(k) (the
0.06 eV free-streaming scale is well below 0.2/Mpc, so the suppression has saturated).
Numerically it equals option (b) to ~0.03%, i.e. unmeasurable against the 0.2 chi2
tolerance. Physically it answers a question the derivation never asks (Section 1). And
it actively breaks the table's second job: with $\tilde D(z) = D(k_{\rm NL}(z), z)$,

$$\frac{d\ln\tilde D}{d\ln a} = f(k_{\rm NL}) + \frac{\partial\ln D}{\partial\ln k}\,
\frac{d\ln k_{\rm NL}}{d\ln a},$$

so f_growth (cosmo3D.c:705) inherits a spurious term — an artifact fed straight into
RSD (radial_weights.c:184) and the FFTLog RSD row (cosmo2D.c:9111). Plus a new
root-find per z. Recommend against; recorded here once.

(d) **Scale-independent ODE growth, smooth DE (CCL-style).** Removes the horizon
contamination exactly as (b) does (the DE-perturbation effect is confined near the
horizon: the "pert off" row is flat at +0.02%), but discards the neutrino slope
entirely, needs a new integrator, and abandons the "growth is a CAMB ratio"
convention. Strictly less physics than (b) for more code.

**Different D per formula?** Exact physics would use D at each formula's own k (IA at
the integrand's k, $\sigma(M)$ at 1/R(M), f at the Limber node's k) — but these all
live in the same window where D(k) varies by <= 0.03%, and splitting tables would break
the shared-table consistency the non-Limber cancellation and the frozen tests rely on.
One table at one sub-horizon k serves everything to < 0.1%.

## 4. Recommendation and plan

Adopt (b): move the sampling wavenumber from 5e-4 to 0.05/Mpc in the Python
likelihoods. Do not add a $k_{\rm NL}$ routine. Keep the $\sqrt{P(z,k)/P(0,k)}\,(1+z)$
construction and the normalization at z_2D[-1] (historical-consistency rule). No C-side
change; `cosmology.G` stays the single source for D and f.

Steps:
1. projects/lsst_y1/likelihood/_cosmolike_prototype_base.py:437–440: replace 0.0005 by
   a class attribute (e.g. `growth_k0 = 0.05`, overridable in yaml) used in both the
   G_growth and z_norm lines; same edit in projects/roman_real (line 440). Comment why:
   5e-4/Mpc ~ 2.2 H0/c; CAMB's fluid DE perturbations shift D there by ~0.8% (w=-0.9,
   z=1) relative to every sub-horizon scale the code actually models.
2. Validation (per the frozen-chi2 protocol, chi2 to 4 decimals, pass band 0.2):
   - w = -1, mnu = 0: the table shifts by a ~k-independent +0.02% which the z_norm
     division should cancel; expect chi2 unchanged to < 1e-3. Verify, do not assume.
   - w = -1, mnu = 0.06: D shifts z-dependently by <= 0.16%; re-record frozen chi2.
   - w = -0.9 (DE perturbations on): quantify dD(z) (~ -0.8%), df (~ +0.3%), the shift
     in cluster counts (expect ~2–3% at Tinker cluster masses) and in the IA term;
     re-record frozen chi2 in a new commit on bugfix stating the change is by design.
   - Determinism sweep across thread counts; run all IA models (NLA, TATT branches) per
     the consistency-over-speed rule; re-run the CCL comparison with CCL back on its
     own default growth — the Delta chi2 = 8.76 workaround should no longer be needed.
3. Save this report's conclusions into the cosmolike_core skill references
   (save-Fable-insights rule) before any re-ask.

**Verified vs inferred.** Verified by reading code: every file:line above (IA 1/D and
1/D^2 factors; pt_cfastpt.c building all tables from p_lin(k, 1.0) with no growfac; the
$D^4$ sites; $\nu = \delta_c/(\sigma(M{,}a{=}1)\,D)$ sites; f as the table slope; the
FKEM pivot ratios; the Python table construction and its z_2D[-1] normalization).
Inferred, not re-run: all percentages come from the CAMB numbers supplied with the
task; the Tinker sensitivity $d\ln n/d\ln\sigma \sim 2$–4 is literature; the claim that
the w = -1 +0.02% offset cancels in the normalized table assumes it is nearly
z-independent — the step-2 w = -1 run is the check. Equations follow CKB 2001, HS04 +
2010 erratum, BK07 and Blazek et al. 2019 from memory; no papers were fetched.
