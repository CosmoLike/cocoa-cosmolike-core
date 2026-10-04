# Cluster all-pairs Limber projection

## Scope and model

`spectra_cluster_cov.c/h` projects supplied radial windows, selected
cluster biases, nonlinear matter power and one-halo cluster mass profiles.
It reads no global state and does not choose a selection or mass function.
The existing cluster covariance C++ file exposes
`covariance_cluster_spectra`, linked only by DES cluster.

Output is every cluster-base and cluster-cluster spectrum, including
different redshift and richness categories. The mean model follows
To et al. (2021), arXiv:2008.10757, Secs. 4.1.2--4.1.3: cc and cg use
biased nonlinear power; cluster lensing additionally has the own-halo
profile without an extra halo-bias factor. Base galaxy windows already
contain galaxy bias. Base source windows are lensing only. The current
boundary excludes IA, magnification, RSD and non-Limber corrections.

Only a source partner receives the core harmonic spin factor. Angular
real-space conversion and catalog noise remain separate, as in the
galaxy/shear covariance. No unsupported component is silently inserted
into a full matrix. This is not a finished Gaussian survey generator or
the cluster SSC/cNG/count-spectrum calculation.

## Numerical structure

Cluster windows and their biased versions share radial scratch. Pair maps
are explicit. Two SIMDe lanes own different pair integrals, with identical
radial addition order for any thread count. OpenMP collapses multipole and
pair-group indices so neither few multipoles nor few cluster categories
limits the available independent work. Local restrict pointers describe
the input rows of the integration loop. No scalar fallback, MPI call or
data-vector C change was introduced.

This is an implementation layout, not evidence for a speedup. Quiet
1/2/4/8-thread timing and x86 measurements remain to be done. Correctness
checks below overlapped the sequential project regression runner; their
elapsed times must not be used for optimization claims.

## Validation (2026-10-04)

The optimized DES cluster interface rebuilt successfully. Its covariance
suite passes 13 checks: six existing count cases, one survey adapter case
and six cluster-spectrum cases. No likelihood reference was changed.

The new project tests check:

- Closed shell integrals for one-halo-only power, proving it contributes
  only to cluster lensing and is invariant under a change of halo bias.
- Closed biased-power integrals with the correct normalization for both
  one and two cluster windows.
- Independent NumPy contractions on general supplied power/profile tables,
  including signed profiles, one/even/odd node counts and an odd pair count.
- A change of length units, output ownership, malformed input rejection
  and bitwise 1/2/4/8-thread repeatability.

All comparisons use a 2e-14 relative tolerance. An isolated production-C
debug build (-O0, undefined-behavior and float-divide-by-zero sanitizers)
passes 16 further cases: 1/2/9/257 nodes at 1/2/4/8 threads. It agrees with
the optimized implementation within 2e-14 and is itself bitwise across
thread counts. The installed project library remains optimized.

The DES physical pilot uses the same fixed-selection inputs as
`covariance_cluster_counts.md`. Its independent Python projection retains
22 fields: six galaxies, four sources and 12 cluster populations. C and
NumPy differ by at most 1.999e-15 relative across 49 multipoles from 2 to
100000, and C repeats bitwise at 1/2/4/8 threads. The noise-inclusive field
matrix has no negative modes on that grid; its smallest correlation
eigenvalue is 0.071763 at ell about 9.69. Cross-redshift cluster spectra
are explicitly nonzero. This necessary field check does not establish
positivity of a future full G+SSC+cNG+counts matrix or survey convergence.

External developer evidence: `test/covariance_reference/`
`cluster_angular_consistency.py` and its JSON report in `results/`.
Debug evidence is `/tmp/check_cluster_spectra_debug.py` and
`/tmp/cluster-spectra-debug-test.log` in this session. The analytic tests
are tracked in `des_cluster/tests/covariance/test_spectra_cluster.py`.

## Didactic review

The manual review followed the tests before starting another major ticket.
It checked the normalized versus biased cluster window, different cg/cs
mean models, core versus observed shear conventions, units, complete pair
coverage, output ownership and unsupported physics boundaries. Every
SIMDe call states lane meanings and its mathematical role. Loop overviews
explain the physical calculation and why work is independent. C/header
and modified C++ lines fit 80 columns, with one comparison per line and
separate paragraphs. No claim of a separate-agent review is made.
