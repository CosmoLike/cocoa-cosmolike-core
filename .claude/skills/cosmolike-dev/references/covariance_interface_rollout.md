# Covariance interfaces, survey assembly and cluster boundaries

## Numerical ownership

`assembly_cov.c` owns whole Gaussian matrices and connected catalog
projections. The production `ci.covariance` submodule borrows contiguous
NumPy inputs and returns owned C-order arrays. Notebook wrappers validate,
allocate and copy Armadillo arrays around the same C kernels. Neither C++
interface is a second numerical implementation.

Covariance C files belong in `cosmolike/covariances/` and end in `_cov.c`;
cluster extensions end in `_cluster_cov.c`. C does not call MPI. Shared
Python orchestration owns survey setup and dispatch; project adapters
supply survey choices. OpenMP handles independent numerical outputs and
BLAS remains single threaded.

The global interpolation boost and independent quadrature selector resolve
into explicit grids and rules. Save those resolved values with every
output. A project README summarizes the workflow and links its covariance
guide; it does not duplicate that guide's commands or timing tables.

## Armadillo notebook boundary

`components_wrapper_cov.cpp` supplies components and radial spectra;
`covariance_wrapper_cov.cpp` supplies complete Gaussian matrices and
connected projection; `cluster_wrapper_cov.cpp` supplies cluster
quantities. Their typed interfaces use Armadillo columns, matrices and
cubes. Short C workspaces are copied explicitly, preserving physical axis
order and the kernels' summation order.

The CARMA 0.7 borrowing caster can rearrange an input and rejects some
views. Its small-cube move path caused an invalid free in a 16–64-element
cube check. `notebook_bindings_cov.hpp` therefore copies Python buffers
into owning Armadillo containers; CARMA exports the results. Columns are
reshaped to one dimension only at the Python boundary.

NumPy's `OWNDATA` flag is not the ownership test: CARMA retains allocations
in a capsule. Check independence from input mutation, views and later
calls. Boundary tests cover C/Fortran order, sliced and read-only arrays,
rectangular axes, small cubes and vector versus singleton-matrix ranks.

Cluster moments expose `density` and `biased_density` matrices and `J01`,
`J11`, `J02`, `J03_KKQ`, `J03_KQQ` cubes. Shared consumers use these names
rather than the retired packed four-dimensional outputs. Their selection
normalization and length units are part of the interface contract.

The boundary validation covered 151 optimized covariance checks, 100
frozen data-vector checks and 65 sanitizer checks across the common and
cluster interfaces. The count identifies this validation record, not the
current size of the test suite. No frozen reference changed. Earlier
wrapper timings do not measure the later owning-copy boundary.

## Whole-matrix Gaussian assembly

Validate field spectra and the observable map once, then reuse field-pair
rows and block scratch. Fourier assembly retains harmonic pure noise;
real-space assembly uses analytic pair noise. Tests compare every entry
with independent NumPy contractions and existing component calls,
including overlapping Fourier bands and one/eight-thread execution.

A supplied-spectrum benchmark on 2026-10-03 used an M2 Pro, eight OpenMP
threads, BLAS one, 60 observables × 26 bins and ell=2..100000. After one
warmup, three Python-block evaluations took 24.8933, 25.0347 and 25.0106 s;
whole-wrapper evaluations took 4.9314, 4.8475 and 4.8370 s. Outputs were
bitwise equal. The timing includes wrapper allocation but excludes physical
spectra, CAMB and SSC/cNG. It is not a full-survey benchmark.

## Shared real-space and Fourier workflows

`survey.fourier_covariance` shares matter, response and radial preparation
with real-space assembly. Integer (2ell+1)-weighted bands replace angular
kernels; Fourier rows omit xi-. Gaussian terms include finite-band pure
noise, and SSC/cNG retain all crossed bins. Empty probe groups are skipped
before C contractions.

Fourier averages core C_ell directly and uses
sqrt[(ell-1)ell(ell+1)(ell+2)]/(ell+1/2)^2 per non-Gaussian source leg.
Real space retains the conversion matching Cocoa's angular transform.
These estimator conventions need explicit comparison before interpreting
cross-code differences as numerical errors.

Band-merging tests compare direct wider bands with H C H^T for every
component and with the corresponding merged mean signal. Complete arrays
repeat bitwise at one and eight threads. A full 1560-row LSST parity check
also matches saved G/SSC/cNG/total and mean arrays bitwise.

`survey._matter_covariance_tables` owns catalog-independent halo work. It
receives geometry, compressed measurement operators, mask multipoles and
resolved numerical settings, and returns projected matter trispectra,
responses and long-mode power. Catalog windows and survey-mean corrections
remain in the survey assembler. Two-lens/two-source real/Fourier checks
verify bitwise component, geometry and signal equality at one/eight
threads when sharing this preparation.

## Survey adapters and physical scope

`forecast.py` owns initialization, unit conversion, real/Fourier dispatch
and archives with resolved settings. Project adapters supply actual
redshift files, bin counts, noise and explicit fiducial assumptions.
DESxPlanck and DES cluster require unit lens-photo-z stretch factors in
initialization. Small adapter tests check row counts, saved metadata and
thread/backend parity; their coarse grids are not accuracy prescriptions.

Roman KL's shape dispersion 0.035 is divided by sqrt(2) for the
per-component convention; its lens density is an explicit forecast input.
Roman real/Fourier examples use 2415 deg2 and density 41.3 as stated
choices, not a reproduction of the ten-bin arXiv:2004.05271 survey.
Galaxy/shear adapters do not acquire CMB or selected-cluster observables
merely by using those projects' catalogs. The joint angular DES model and
its limits are in `covariance_cluster_joint.md`; Gaussian non-Limber and
IA support is in `covariance_nonlimber_ia_plan.md`.

The early full LSST teaching example gave maximum boost-1/2 variance-ratio
changes of about 0.77 in real space and 0.25 in Fourier space, with positive
totals. Positive definiteness and a successful notebook execution are
therefore insufficient evidence of inference accuracy. These measurements
refer to that example's saved settings, not current default convergence.

## Cluster references and conventions

The original checkout `test/cosmolike_core`, branch `cluster_chto`, contains
`theory/covariances_cluster.c`: counts, counts–two-point crosses, cluster
lensing and selected halo moments. The larger DES implementation is
`test/lighthouse/cpp/cov_clusters_fullsky.c`. Inspect other historical
branches explicitly; the cluster branch does not contain every later
LSS×CMB development.

The external inventories `cosmocov_port_study/06_cosmolike_family_covariance_inventory.md`,
`08_cluster_files_design_reference.md` and `11_fable_counts_x_2pt_ssc.md`
identify source conventions and potential defects. The last derives a
missing f_K^-2 in a legacy counts–two-point SSC expression using dN/dchi.
An inferred inconsistency is not a published erratum. Independent physics
references include Takada–Spergel (1307.4399), Schaan–Takada–Spergel
(1406.3330), To/Krause et al. (2008.10757) and the DES joint analysis
(2503.13631).

The DES layout has three cluster-redshift bins, four richness bins, six
lens bins, four source bins and twenty angular bins. Cluster lensing uses
Sigma=Y gamma_t; both covariance and cross blocks require that transform.
Count normalization and overlapping-catalog noise require independent
checks. The count-shell component is documented in
`covariance_cluster_counts.md`, with the complete joint assembly in
`covariance_cluster_joint.md`.

## YAML production entry point and validation

`command_line.py` reads `theory`, `params`, `sampler: evaluate` and `output`
through Cobaya's YAML reader and parameterization. Prior parameters require
explicit evaluate overrides. Numerical controls inherit the project
covariance defaults. Thin project runners select the shared production
backend; DES cluster selects its joint angular adapter. This runner
computes one fixed-cosmology matrix and disables Cobaya MPI initialization.

The production-boundary validation on 2026-10-04 passed 162 covariance
checks: 108 LSST, 49 DES cluster and one per remaining project. Adapter
checks compare real/Fourier components and signals exactly between both
interfaces at one/eight threads. Direct-input checks include read-only
arrays, rejection of noncontiguous layouts and result lifetime. Sixteen
Gaussian, array-boundary and small-survey checks also passed sanitizers.
Disabled builds omit every covariance source and binding.

Full LSST and Roman real CLI results matched notebook archives exactly
for G, SSC, cNG, total and signal. Single construction times were 67.55
and 72.94 s on the M2 Pro at eight threads; these were workflow checks,
not repeated performance estimates or present-day timing claims. Use
`covariance_cli_timing.md` for the controlled timing scope.

The associated seven-project data-vector validation passed 392 checks,
including Roman's opt-in halo checks, without refreezing. Launch-path
failures must remain distinguishable from numerical failures and from
successful unchanged retries. Test-suite elapsed times are not benchmarks.
