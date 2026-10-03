# Halo response and trispectrum assembly

`non_gaussian_cov.c/.h` assembles a dimensional halo power response and
the five separated halo trispectrum terms from supplied moments and tree
averages. It adds no survey factors, shot-noise trispectrum or geometry.

The response takes its growth coefficient, dilation coefficient, slope
and absolute/fractional normalization explicitly. The published isotropic
halo form uses 47/21, 1/3 and the logarithmic slope of **I11² P_lin**:
[Takada & Hu v3, corrected Eq. 44](https://arxiv.org/html/1302.6994v3).
The projected squeezed-tree construction uses 17/7, 1/2 and the slope of
P_lin; its linear limit is R1+RK/6 as in
[Barreira, Krause & Schmidt](https://arxiv.org/html/1711.07467v3).
These are distinguished deliberately. The latter is not a calibrated
nonlinear tidal response and does not supersede the published correction.
Multiplying D_halo/P_halo by a nonlinear target power is an explicit model
option. The survey driver must measure these physics choices before a
default is frozen.

The trispectrum retains separate 1h, 2h(1+3), 2h(2+2), 3h and 4h arrays.
The 1+3 term has four isolated-leg choices, giving coefficient two on
each of its two displayed magnitude combinations. The independent
reference enumerates labelled halo partitions and explicit tree diagrams;
it does not just transcribe the reduced assembly equation. Negative
contributions are retained. No eigenvalue clipping or NaN repair is used.

Five checks pass in optimized and debug/UBSan builds: labelled halo
partitions; response against differentiation of a specified power model
with scale-dependent I11; the projected linear tree limit; length^9
dimensions and retained negative tree terms; and bitwise repeated
1/4/8-thread outputs with odd sizes and padded rows.

Native disassembly in the external
`results/non_gaussian_simd.disassembly.txt` confirms vector multiply,
divide, fused multiply-add and fused multiply-subtract instructions.

The references are external `non_gaussian_reference.py` and
`cng_reference.py`. `build_non_gaussian.sh` produces isolated libraries;
LSST tests use `COSMOLIKE_NON_GAUSSIAN_LIBRARY` plus the external-reference
path. Logs are `results/non_gaussian{,_debug}_tests.log`.

After both five-test runs, a separate manual didactic red-eye pass read
the full C/header and tests. It checked the two response slopes, units,
fractional versus dimensional response, halo-partition counting, array
roles, SIMD-lane ownership and odd tails. C/header lines fit 80 columns;
guards have one comparison per line. This was a self-review, not Fable.
The numerical survey projection, physics-delta and PSD gates remain open.
