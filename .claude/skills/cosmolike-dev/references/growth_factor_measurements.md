# Growth factor: measurements behind the G_growth sampling-k decision (2026-10-02)

Companion to `fable_review_growth_factor.md`. CAMB 1.6.7, H0 = 67.32,
Omega_m = 0.3, Omega_b = 0.04, As = 2.1e-9, ns = 0.96605, one massive
neutrino. D(k,z) = sqrt(P_lin(k,z)/P_lin(k,0)) (total matter), k0 = 5e-4/Mpc
(the likelihoods' current sampling k).

## 1. Dark-energy perturbations (w = -0.9, z = 1), D(k,z)/D(k0,z) - 1 [%]

| k [1/Mpc] | 1e-3 | 3e-3 | 0.01 | 0.05 | 0.2 |
|---|---|---|---|---|---|
| w=-0.9, mnu=0, DE perturbations on | +0.48 | +0.77 | +0.81 | +0.81 | +0.81 |
| w=-0.9, mnu=0, DE perturbations off | +0.02 | +0.02 | +0.02 | +0.02 | +0.02 |
| w=-1, mnu=0 | +0.02 | +0.02 | +0.02 | +0.02 | +0.02 |

CAMB field to switch them off: `setattr(p.DarkEnergy, "___no_perturbations", True)`
("no_perturbations" silently does nothing). Script: CCL-benchmark
`cocoa_comparison/scripts/diag_de_perturbations.py`.

## 2. Which k: DESC-CCL FKEM given D at k (LSST-Y1, w = -0.9, 3x2pt Delta chi2 vs CoCoA)

| k [1/Mpc] | 5e-4 | 1e-3 | 5e-3 | 0.01 | 0.05 | 0.2 |
|---|---|---|---|---|---|---|
| Delta chi2 | 8.76 | 2.38 | 0.341 | 0.267 | 0.168 | 0.153 |

Both codes in Limber: 0.16 (the floor). 1e-3 is still inside the
dark-energy step.

## 3. Neutrinos (w = -1): D(k,z)/D(k0,z) - 1 [%], CAMB vs Eisenstein & Hu 1999

| mnu, z | source | 1e-3 | 3e-3 | 0.01 | 0.03 | 0.05 | 0.1 | 0.2 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| 0.06, 1 | CAMB | +0.02 | +0.03 | +0.06 | +0.11 | +0.13 | +0.15 | +0.16 | +0.16 |
| 0.06, 1 | EH99 D_cbnu | +0.00 | +0.01 | +0.05 | +0.10 | +0.12 | +0.13 | +0.13 | +0.14 |
| 0.30, 1 | CAMB | +0.01 | +0.02 | +0.05 | +0.14 | +0.23 | +0.39 | +0.54 | +0.71 |
| 0.30, 1 | EH99 D_cbnu | +0.00 | +0.01 | +0.03 | +0.14 | +0.23 | +0.40 | +0.54 | +0.68 |
| 0.60, 2 | CAMB | +0.04 | +0.05 | +0.09 | +0.25 | +0.44 | +0.87 | +1.45 | +2.40 |
| 0.60, 2 | EH99 D_cbnu | +0.00 | +0.00 | +0.05 | +0.22 | +0.42 | +0.87 | +1.46 | +2.33 |

EH99 eqs. 4, 5, 11-14 (`eh99_growth.py` here) reproduce CAMB's neutrino
scale dependence to <= 0.07% in D. CAMB's extra +0.02% (z=1) / +0.04%
(z=2) at k = 1e-3 is k-independent and present at mnu = 0 (horizon-scale,
not neutrinos); EH99's D_1 has no such term.

Consequence: at mnu = 0.06 eV, D is flat to 0.03% above k = 0.05/Mpc. At
mnu = 0.3-0.6 eV there is no flat region: D changes by 0.5-1.1% (z = 1)
and 1-2% (z = 2) between k = 0.05 and 1/Mpc. With mnu sampled to high
values, any single-k D is off by ~1% for some consumer (sigma(M) and the
one-loop terms live at k ~ 0.1-1; RSD and the non-Limber term at
k ~ 3e-3-0.1); the scale-dependent fix there is to use P_cb(k,z) directly
(sigma(M,z) per z, one-loop at a few z nodes) instead of D scaling.
