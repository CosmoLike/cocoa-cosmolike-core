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

## Complete two-point Gaussian pilot

The external `cluster_gaussian_pilot.py` assembles all 140 measured
two-point rows (2800 angular entries) from 22 internal fields, including
all crossed spectra. This omits the 12 counts, SSC, cNG and Y transform;
it is not a completed 2812-entry joint forecast. Settings are ell_max
10000, mask_ell_max 4096, 64 radial nodes per panel, 128 angular nodes and
nwindow 4097. No controlled timing is claimed.

The Gaussian correlation matrix has minimum eigenvalue 0.00742072.
Cholesky succeeds both before and after diagonal rescaling, with maximum
correlation-normalized reconstruction residuals 1.70e-15 and 1.78e-15.
The diagonal ranges from 3.79e-15 to 48.3863. At that dynamic range the
NumPy raw eigensolver reports -3.83e-15, while SciPy's evr and evd drivers
give positive minima 2.32e-15 and 2.27e-15. This is a conditioning issue
in the raw eigenproblem, not evidence that the underlying pilot is
indefinite. No matrix entry or eigenvalue was repaired.

Consequently the shared `covariance_modes` positivity flag now uses the
correlation eigenproblem, an invertible diagonal congruence preserving
inertia. Raw eigenvalues remain available as diagnostics. Its regression
tests preserve known positive and negative modes across extreme unit
changes, reject zero/negative diagonals, and check input immutability.
All nine notebook-tools checks pass. A separate manual review checked
the distinction between rescaling and regularization, explained the
quadratic-form argument, and retained the limits of this Gaussian pilot.

## Didactic review

The manual review followed the tests before starting another major ticket.
It checked the normalized versus biased cluster window, different cg/cs
mean models, core versus observed shear conventions, units, complete pair
coverage, output ownership and unsupported physics boundaries. Every
SIMDe call states lane meanings and its mathematical role. Loop overviews
explain the physical calculation and why work is independent. C/header
and modified C++ lines fit 80 columns, with one comparison per line and
separate paragraphs. No claim of a separate-agent review is made.

## Shared notebook preparation (2026-10-04)

`survey_cluster.py` contains observable_layout, selected_windows and
all_pairs_spectra. Counts follow the actual ss,gs,gg,cg,N,cc,cs ordering.
The 12 count positions are 1240..1251; cluster-lensing rows begin at 1852.
Fields are galaxies, clusters, sources. The normalized q and absolute
selected density are computed on identical shells, retaining all crossed
spectra irrespective of measured row exclusions.

Spectra are streamed in blocks of 1024 multipoles to bound temporary
power/profile memory at high boosts. This is an allocation choice, not
an accuracy control: it changes neither nodes nor any radial sum order.
All numerical radial contractions reuse the existing C components.

Six project checks pass for full/smaller layouts, measured exclusions,
analytic normalization, 1/2/4/8-thread repeatability, every field pair,
source factors, invalid inputs and a 1027-multipole case crossing the
block boundary. The independent NumPy sums agree within 3e-14 relative;
results repeat bitwise at one/eight threads. A physical 22-field,
49-multipole comparison with the separate DES pilot has maximum relative
difference 2.665e-15. The first test attempt exposed an indexing-shape
mistake in the independent test's matrix assignment; explicit np.ix_
assignment fixed the oracle, without changing production arithmetic.

The subsequent manual didactic review checked count versus contrast
normalization, exact source/category IDs, every measured family, full
internal cross coverage, noise versus signal, source factors, block
memory lifetime, input ownership and failure messages. Public README
text explains the physics and source responsibilities directly.
No performance gain is claimed from these contended correctness runs.
External physical check: covariance_reference/check_cluster_survey_preparation.py.
