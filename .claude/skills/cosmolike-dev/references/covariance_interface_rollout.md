# Covariance interface, notebooks and cluster extension

## Armadillo notebook boundary

All covariance numerical C++ wrappers take Armadillo columns, matrices and
cubes. `components_wrapper_cov.cpp` supplies components and radial spectra;
`covariance_wrapper_cov.cpp` supplies whole Gaussian matrices and connected
projection; `cluster_wrapper_cov.cpp` supplies cluster quantities. Callable
headers declare their typed APIs. No `py::array_t` or `std::vector` row/plane
maps remain in these wrappers. Copy short C workspaces explicitly, following
`cosmo2D_wrapper.cpp`; preserve the existing C kernels and summation order.

Python conversion and registration are separate interface files. The local
CARMA 0.7 borrowing caster can rearrange an input and rejects certain views;
its small-cube move path also caused an invalid free in the 16--64-element
cube check. `notebook_bindings_cov.hpp` therefore copies Python buffers into
owning Armadillo containers, retaining physical axis order. CARMA exports
results. Columns are reshaped to 1D only at the Python boundary. NumPy's
OWNDATA flag is not the ownership test: CARMA retains the allocation in a
capsule. Test independence from inputs, views and later calls instead.

Cluster moments expose `density` and `biased_density` matrices, and `J01`,
`J11`, `J02`, `J03_KKQ`, `J03_KQQ` cubes. Shared workflows and tests consume
these names; the former packed four-dimensional `single`/`pair` outputs are
retired. Numerical physics and units are unchanged.

## Requested order

1. Add notebook-facing C++ wrappers, following the data-vector wrapper
   conventions, while retaining the low-level component calls for testing.
2. Expand the LSST covariance notebook to real-space and Fourier-space
   matrices, with Gaussian, SSC, connected and total outputs, accuracy
   comparisons, positivity diagnostics and shared plotting.
3. Replicate thin project adapters and notebooks across DES, Roman and LSST,
   using each project's actual bins and explicit physical inputs. Keep
   algorithms in cosmolike_notebook_utils. Each main project README needs
   a self-contained covariance running section in Cocoa's house style.
4. Implement cluster 6x2pt + counts covariance for des_cluster, including
   cross correlations with counts, with independent physical references.

Each major piece requires tests, a separate didactic review, and a small
local commit. Never push. Keep all covariance C work in covariances/;
new C filenames end _cov.c, cluster extensions end _cluster_cov.c.
No C MPI calls. Preserve one BLAS thread and explicit 8--10-worker OpenMP
parallelism. Numerical settings resolve from one user accuracy boost;
retain the resolved settings alongside outputs.

The first cluster component and its independent checks are recorded in
`covariance_cluster_counts.md`: count-shell responses, Poisson counts and
SSC from supplied selected abundances. The full cluster generator remains
to be implemented; do not describe the galaxy/shear notebook as 6x2pt+N.

## First C++ wrapper ticket

The current component bindings expose spectra, geometry, halo moments,
responses and weighted contractions. A high-level Gaussian wrapper should
receive all field spectra and the observable map, validate once, reuse
scratch across blocks, and return the owned whole matrix. Supply real-space
and Fourier-space entry points; Fourier retains harmonic pure noise, real
space uses analytic pair noise. Compare against the existing per-block
calls and independent NumPy contractions. Then expose shared SSC/cNG
assembly without duplicating physics. Python owns survey setup, inspection,
plotting and future MPI scheduling.

The measured full runs are in covariance_full_survey_timing.md. Their
Gaussian Python block loop takes 25.30 s (LSST) / 75.02 s (Roman); avoiding
repeated boundary validation and array copies is useful but must be
measured, not assumed to improve the total. No benchmark may overlap a
build, test, CAMB job or notebook execution.

## Cluster sources located

The separate original checkout is test/cosmolike_core, remote
CosmoLike/cosmolike_core, checked-out cluster_chto. Its
`theory/covariances_cluster.c` contains counts-counts, counts-two-point,
cluster-lensing auto/cross terms and selected cluster halo moments.
The larger DES joint implementation is
`test/lighthouse/cpp/cov_clusters_fullsky.c`.
Read other original branches using git show: the checked-out old cluster
branch does not contain later master/LSSxCMB covariance developments.

The existing external study identifies these sources and potential defects:
`cosmocov_port_study/06_cosmolike_family_covariance_inventory.md`,
`08_cluster_files_design_reference.md`, and
`11_fable_counts_x_2pt_ssc.md`.
The latter derives a missing f_K^-2 in legacy counts-two-point SSC when
using dN/dchi. Independently verify against Takada--Spergel 1307.4399 and
Schaan--Takada--Spergel 1406.3330 before implementing. Do not call inferred
paper inconsistencies a published erratum. Read To/Krause et al.
2008.10757 and the DES joint-analysis paper 2503.13631.

The present des_cluster Gaussian Python reference is
`projects/des_cluster/tests/reference/ref_covariance_full.py`.
The actual layout has three cluster redshift bins, four richness bins,
six galaxy lens bins, four source bins and twenty angular bins. The
cluster-lensing observable is Sigma=Y*gamma_t, so its covariance must
include that convention. Count normalization and overlapping catalog noise
need explicit tests. Current galaxy/shear full forecasts use massless
neutrinos, Limber, linear bias and zero IA/magnification/RSD; do not imply
that those choices automatically reproduce any project's supplied matrix.

## Documentation contract

Read cocoa/.claude/skills/cocoa-maintenance/SKILL.md and imitate the main
Cocoa README: numbered contents/anchors, assumptions, Step flows, separate
covariance versus data-vector testing commands, and Appendix FAQs. Render
with markdown-it, check local links and anchors, and inspect notebook
figures. Public documentation links papers and tracked user files, never
bot skills or local external harness paths.

## Gaussian wrapper validation (2026-10-03)

The real/Fourier Gaussian wrappers validate supplied arrays once and reuse
field-pair rows and block scratch. The real wrapper preserves the C Wick,
projection and analytic pair-noise operation order. Tests compare all
entries bitwise at 1/2/4/8 threads; Fourier is independently checked with
NumPy including overlapping bands. Returned arrays retain ownership across
calls; malformed shapes, fields, area, noise and nonfinite values fail
before C calls. All 60 LSST covariance checks pass. The complete 1560-row
LSST smoke forecast matches its saved G/SSC/cNG/total and mean arrays bitwise.

Isolated supplied-spectrum timing: Apple M2 Pro, eight OpenMP workers,
BLAS one, 60 observables x 26 bins, ell=2..100000, one untimed call per
implementation then three measured calls. Both methods process identical
synthetic positive field spectra and physical spin-bin operators. Python
blocks: 24.8933, 25.0347, 25.0106 s; C++ wrapper: 4.9314, 4.8475, 4.8370 s.
Every measured output is bitwise equal. No concurrent numerical job/build
ran; CPU inspection showed only ordinary desktop processes beside the
benchmark. This measures Gaussian assembly including wrapper allocation,
not physical spectra, CAMB, SSC/cNG or full-survey timing.

Manual didactic review checked field crossings, real/Fourier noise,
owned outputs, loop overviews, scratch lifetime, one comparison per line,
and the 80-column C++ limit. No new SIMD intrinsic or C parallel loop was
introduced: the wrapper uses the existing documented production C calls.

## Shared Fourier forecast assembly

`survey.fourier_covariance` shares the real-space matter/response/radial
pipeline. Integer (2ell+1)-weighted bands replace angular kernels; Fourier
rows omit xi-. Gaussian includes finite-band pure noise. SSC/cNG retain
all crossed bins. Empty probe groups are skipped before C contractions.
The source transfer is explicit: Fourier averages core C_ell directly and
uses sqrt[(ell-1)ell(ell+1)(ell+2)]/(ell+1/2)^2 per NG source leg. Real space
retains the existing conversion needed to match Cocoa's angular transform;
no data-vector C convention was changed. Estimator/convention validation
beyond these defined predictions remains part of the physical survey gate.

Tests merge adjacent bands and compare the directly computed wider band
against H C H^T for every G/SSC/cNG/total entry. Mean signals also match
direct core-spectrum averaging. Complete arrays repeat bitwise at one and
eight threads. All 61 LSST covariance checks pass. Manual didactic review
checked mode-count weights, two-sided transformation, fixed scientific
band endpoints during refinement, the single boost's NG grid, and the
normalization distinction. No new C or SIMD arithmetic was introduced.

## LSST notebook and reusable forecast boundary

`forecast.py` owns initialization through ordinary galaxy/source setters,
unit conversion, the real/Fourier dispatcher and output archives with
fully resolved settings. Project adapters supply only survey choices.
The LSST notebook executes complete 1560-row angular and 675-row bandpower
G/SSC/cNG matrices at boosts 1 and 2, prints total positivity/generalized
mode diagnostics, renders six figures and saves arrays/settings/CAMB inputs.
All four totals are positive definite. Boost 1 versus 2 maximum variance
ratios change by about 0.77 (real) and 0.25 (Fourier), and maximum individual
error changes are about 4.5%/5.7%. These are teaching resolutions, not
accepted inference defaults. Notebook stage timings are not controlled
benchmark claims. Preserve that limitation in every project port.

Validation: 62 LSST covariance checks passed, including archive roundtrip
without pickle and Fourier-axis labeling. The committed LSST cells ran
headlessly through nbconvert; figures were extracted and visually inspected.
Manual didactic review checked dimensions, input/output ownership, unit
conversions, band ordering, component labels, refinement interpretation and
cold-path readability. README rendering with Markdown-it, tables and all
local links/anchors passed. No quoted installation block was changed.
Main README has a separate anchored covariance workflow with setup,
compilation, notebook execution, accuracy control and saved-output paths.

## Project adapter rollout

All seven interfaces bind the galaxy/shear covariance components. Thin
adapters use the actual project redshift files and bin counts. DESxPlanck
and DES cluster need explicit unit lens-photo-z stretch factors in their
setters; initialize_forecast handles this through one optional setting.
These initial notebooks cover galaxy/shear only, not CMB or cluster
observables. The later DES joint angular example and its model limits are
recorded in `covariance_cluster_joint.md`.
Roman KL's 0.035 quadrature shape dispersion is converted to a per-component
value by division by sqrt(2); its added lens density is an explicit forecast
assumption. Roman real/Fourier use 2415 deg2 and density 41.3 as explicit
example choices, not as a reproduction of the 2004.05271 survey.

The shared cocoa_covariance_testing.py runs each adapter through real and
Fourier projections with real catalog files, a small measured subset and
one/eight threads. All six added project checks passed bitwise component
repeatability, positive subset totals, full row-count contracts and archive
metadata roundtrips. Small test grids are not an accuracy prescription.
Each project's data_vector/ and covariance/ tests are separate; the moved
likelihood test function bodies and stored snapshots remain unchanged.

Manual didactic review checked units and noise conventions, measured versus
internal pair cuts, accuracy versus scientific band edges, model scope,
function side effects, explanatory paragraph breaks and readable settings.
There are no new C arithmetic loops or SIMD intrinsics in this ticket.
README pages use numbered contents, explicit setup/run steps, file guides,
physics FAQs and direct paper links. Markdown-it rendering and local link
and anchor validation pass. Notebook execution and full data-vector
regressions are recorded with the per-project commits.

## Shared matter tables for cluster assembly (2026-10-04)

The catalog-independent halo loop is extracted as
`survey._matter_covariance_tables`. Its inputs are the existing radial
geometry, compressed measurement operators, mask multipole count and
resolved integration settings. It returns angularly projected matter
trispectra, responses and long-mode power. Catalog windows and observed
mean corrections remain with the survey assembler. The extraction changes
no numerical expression, array layout or order of operations.

The original file is retained outside git for a direct comparison. Both
versions compute a physical two-lens/two-source forecast in real and Fourier
space at one and eight threads. G, SSC, cNG, total, mean signals, geometry,
pair areas and coarse multipoles are bitwise identical in all four cases.
The external check is `/tmp/check_covariance_matter_extraction.py`, with
output in `/tmp/check-covariance-matter-extraction.log`. This is a correctness
check during ongoing regression tests, not a performance measurement.

The subsequent manual didactic review checked the physical reason for
sharing matter tables, the transform/source-factor contract, every returned
shape and unit, and the distinction between a common matter response and
catalog normalization. The halo loop remains explained in physical stages.
No C, SIMD operation, quadrature or interpolation choice was changed.

## Final project regressions (2026-10-04)

All seven data-vector test collections pass after the interface/notebook
rollout and shared matter extraction. No frozen likelihood reference was
changed. Roman real's ordinary run skipped its 24 opt-in halo checks; a
separate run enabled and passed all 24, so no planned check remains skipped.

| Project | Data-vector checks passed | Covariance checks passed |
|---|---:|---:|
| LSST Y1 | 57 | 63 |
| Roman real | 104 | 1 |
| Roman Fourier | 45 | 1 |
| Roman KL | 49 | 1 |
| DES Y3 | 63 | 1 |
| DESxPlanck | 45 | 1 |
| DES cluster | 29 | 46 |
| Total | 392 | 114 |

The five single covariance adapter tests each exercise real and Fourier
forecasts, component repeatability, catalog layout and archive metadata.
LSST Y1 covers the common numerical components; DES cluster also covers
counts, selected halos, all-pairs spectra and the joint angular forecast.
An initial Roman KL collection failed because its interface directory was
missing from PYTHONPATH; the corrected environment passed all 49 tests.
The source and references were unchanged to resolve that import error.

External logs are `/tmp/covariance-port-<project>-data-vector.log`,
`/tmp/covariance-port-roman_real-slow-data-vector.log`,
`/tmp/covariance-port-lsst_y1-components.log`,
`/tmp/des-cluster-covariance-complete.log` and
`/tmp/covariance-final-<project>.log`. Test elapsed times are not benchmarks.
The DES 1/2/4/8-thread measurement was run only after every numerical test,
notebook and build had finished; its scope and results are recorded in
`covariance_cluster_joint.md`.

The later high-boost memory audit and bitwise-preserving spectrum batching
are recorded in `covariance_limber_batches.md`. That change passes the
updated 116 covariance checks; it does not change the data-vector C code.
