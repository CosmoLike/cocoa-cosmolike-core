# Planned covariance non-Limber spectra and intrinsic alignment

Status: **Gaussian kernels and survey workflows implemented; validation below**.
The active scope is non-Limber gg/gs and NLA/TATT in Gaussian covariance.
SSC/cNG calculations are explicitly excluded. The stopped LSST integration
validation stays stopped. Implement and commit in small tested blocks.

### Completed transform component

`fftlog_cov.c/.h` owns shared FFTW plans and per-worker buffers. Forward
transforms are retained across 16-multipole blocks. Gamma and phase
recurrences reduce special-function calls; density and lensing use the
same radial bias with different Mellin kernels. The lensing denominator
stays inside the kernel, following N5K Eq. 24.

The tracked LSST `test_covariance_fftlog.py` runs against the normal
project library. Four optimized tests pass: analytic Gaussian-Bessel
integrals for ell 2, 3, 17, 32, 33, 47, 96, 97, 111 and both kernel
choices; an observer-endpoint lensing integral; reuse after a different
multipole block; and bitwise one/eight-thread agreement. Peak-scaled
errors are below 1e-8. Doubling radial resolution from 2049 to 4097 did
not remove the roughly 3e-9 floor in the most demanding high-ell Gaussian
test over its extended reciprocal range. This is a component check, not
a selected survey setting or a speed benchmark. The survey validation below extends these component checks. Didactic review checked scalar SIMD
equivalents, buffer ownership, phase/padding explanation and 80 columns.

### All-pairs gg/gs component

`nonlimber_cov.c` projects density, magnification and signed NLA with a
common z=0 power anchor and the supplied growth table. Its matched Limber
subtraction uses exactly D(a)^2 P(k,1). The notebook and direct production
`covariance_spectra` entries add this correction to all galaxy-containing
pairs through an explicit cutoff. Shear-shear stays Limber. RSD and
massive-neutrino requests are rejected. No SSC/cNG function was changed.

Nine optimized LSST component tests pass, including the existing radial
spectrum tests. A two-lens/two-source survey with nonzero signed
magnification and NLA agrees with direct spherical-Bessel quadrature at
ell 2/8/30 to 1.2e-6/9.5e-7/3.0e-6 in variance-normalized spectrum units.
The independent k sum was refined from 2049 to 4097 samples. Doubling the
FFT radial grid 4097 -> 8193 changes spectra by 1.7e-7. Both backends and
one/eight threads agree bitwise; sampled hybrid matrices are positive
definite and ell=300 corrections are below 1% of variance normalization.
These checks do not establish each project's cutoff or covariance/Fisher
accuracy. The TATT and survey sections below record subsequent work.

### Gaussian TATT component

`ia_cov.c` adds all-pairs TATT E corrections and BB, retaining core FAST-PT
normalization, growth and finite-k support. NLA is recovered bitwise when
A2=bTA=0. `assembly_cov.c` adds BB signal/mixed-noise contractions with the
xi+/xi- sign product; analytic pure noise is included once. Both C++ paths
share these routines; production accepts contiguous arrays and notebook
wrappers retain Armadillo cubes. The result's optional `b_spectra` is None
outside TATT. TATT+RSD is rejected; non-Limber corrects only the linear
alignment part of gg/gs, with the loop terms staying Limber.

Nineteen optimized tests pass (IA, production and notebook-array suites).
The new tests compare with the independent data-vector projection: TATT
E corrections agree at 3.4e-5 and BB at 9.7e-5, normalized by each pair's
peak over ell=10..5000. The reference uses core integration level 2;
covariance uses 256 nodes per panel. A direct Wick sum verifies BB signs,
mixed noise and zero-B recovery. Production/notebook arrays and one/eight
threads agree bitwise. These checks validate model plumbing and component
projection, not survey-level accuracy or a complete nonlinear IA covariance.
Didactic review covered TATT signs, kernel meanings, E/B lane separation,
core interpolation support and the distinct Gaussian/SSC/cNG scope.

### Shared workflows and tested non-Limber baselines (2026-10-05)

The shared configuration resolves `gaussian.nonlimber`, `ia`, `A1`, `A2`
and `B_TA`; amplitudes are per-bin constants or scalar broadcasts. The
core supplies growth evolution, not an inferred redshift power law.
CLI threads come only from `OMP_NUM_THREADS`. Both C++ paths call the
same C kernels. Saving includes the Gaussian mean and the separate
`ssc_normalization_signal`; legacy no-IA results remain saveable.

A small full G+SSC+cNG test changes zero-IA Limber to non-Limber NLA and
TATT. SSC, cNG and their normalization signal remain bitwise unchanged.
Gaussian changes and total positivity are checked. All 119 LSST covariance
tests passed in the optimized build after workflow wiring.

Actual survey tests use full real/Fourier Gaussian matrices, all internal
pairs, the supplied forecast noise and ell_max=100000 for real space.
The table reports max |lambda-1| for C_fine v = lambda C_base v.
The finer run doubles both the correction cutoff and logarithmic radial
interval count. Other integration/interpolation settings remain fixed.

| Galaxy/shear project | Base cutoff | Base radial samples | Real modes | Fourier modes |
| --- | ---: | ---: | ---: | ---: |
| lsst_y1 | 1000 | 4097 | 0.01160% | 0.01047% |
| roman_real | 1000 | 8193 | 0.01260% | 0.01363% |
| roman_fourier | 1000 | 8193 | 0.01730% | 0.01851% |
| roman_kl | 4000 | 8193 | 0.00270% | 0.00272% |
| des_y3 | 1000 | 4097 | 0.00298% | 0.00290% |
| desy1xplanck | 1000 | 4097 | 0.00413% | 0.00362% |
| des_cluster | 1000 | 4097 | 0.00517% | 0.00494% |

All 28 base/refined matrices are positive definite without likelihood
cuts or eigenvalue repair. DES cluster here means its galaxy/shear adapter,
not joint 6x2pt+N. The joint adapter rejects non-Limber or IA until its
cluster transfers are implemented. DESxPlanck still excludes CMB fields.
These values test the new non-Limber controls at zero IA, not convergence
of all quadratures, angular transforms, interpolation, cosmologies or
Fisher derivatives. No full G+SSC+cNG integration run was restarted.

A separate all-project test checks finite NLA/TATT spectra and bitwise
production/notebook agreement with actual catalog inputs. The broad-bin
surveys use cutoff 1000; narrow Roman KL requires 4000. Roman examples
use 8193 radial samples; the other examples use 4097. Global boost refines
both settings, while integration_accuracy remains independent.

The partial hybrid (non-Limber gg/gs, Limber ss) is not automatically a
positive signal matrix: Roman real/Fourier have negative low-ell noiseless
field modes, including ell=2. Their catalog shot/shape noise restores
positive observed-field matrices, and the complete Gaussian matrices
above are positive. Do not generalize this to arbitrary lower-noise
surveys or repair modes. Extending ss consistently is separate physics
work. Keep these diagnostics when changing the model or source densities.

The external reproduction scripts/results are under
`test/covariance_reference/check_gaussian_{projects,matrices}.py` and
`results/gaussian_project_checks/`. Their paths belong in this development
record, not in public human README instructions. Tracked component and
project-boundary tests are the portable regression coverage.

## Original starting point (historical)


| Layer | Available behavior | Missing behavior |
| --- | --- | --- |
| Data-vector `cosmo2D.c` | FFTLog non-Limber galaxy autos and galaxy–shear spectra, including the linear/NLA source kernel. | Its selected pair maps and autos-only clustering do not supply the full covariance field matrix. |
| Covariance `spectra_cov.c` | All-pairs Limber spectra; separate density, lensing/magnification and signed NLA windows; optional Limber RSD. | Covariance-owned non-Limber transforms and matched linear subtraction. |
| Covariance `forecast.py` | Initializes a massless-neutrino, linear-bias forecast. | It initializes IA off and explicitly sets all IA amplitudes to zero. |
| Covariance `survey.py` | Builds the full G/SSC/cNG matrix from shared arrays. | It requests `include_ia=False`, `include_rsd=False`; its connected weights select lensing alone for sources. |
| Cluster forecast | All-pairs supplied cluster Limber spectra and joint assembly. | Cluster non-Limber transfers and a validated joint nonzero-IA model. |

Zero IA is a restriction of the current complete forecast, not an absence
of NLA throughout CosmoLike, nor a requirement of covariance physics.
Turning on one low-level flag would update Gaussian spectra without
automatically updating every SSC/cNG and cluster source leg. Complete the
model consistently before advertising nonzero-IA forecast support.

Current low-level NLA checks validate signed windows and contractions;
some references consume the same supplied windows. They do not independently
validate the full IA model or establish survey-level NLA convergence.

## Physics sources and limits of adoption claims

- [Fang, Krause, Eifler & MacCrann](https://arxiv.org/html/1911.11947),
  Sections 2–4: linear non-Limber plus nonlinear Limber correction, FFTLog
  transforms, galaxy density/RSD/magnification and lensing kernels.
- [Leonard et al., N5K](https://arxiv.org/html/2212.04291): independent
  non-Limber comparison and the lensing kernel with `j_l(x)/x^2`.
- [To et al., DES Y6](https://arxiv.org/html/2503.13631v1), Appendix F:
  non-Limber covariance is an explicit improvement for cross-tomographic
  correlations. This does not establish non-Limber treatment of every
  shear term or of the connected four-point function.
- [Krause & Eifler](https://arxiv.org/html/1601.05779v1), covariance
  appendix and IA discussion: multiprobe contractions and model context.
- [Hirata & Seljak](https://arxiv.org/abs/astro-ph/0406275) and
  [Bridle & King](https://arxiv.org/abs/0705.0166): tidal alignment,
  GI/II correlations and the nonlinear-alignment prescription.
- [Takada & Hu](https://arxiv.org/html/1302.6994v3), corrected Eq. 44 and
  Appendix A: matter responses, survey-mean normalization and projected
  SSC. These equations alone do not specify an IA response model.
- [Friedrich et al., DES Y3](https://arxiv.org/html/2012.08568v3): estimator,
  noise, angular-bin and covariance validation conventions.

Use the papers for physics. Compare legacy routines only after matching
their conventions and active call paths. The existing
[DES audit](covariance_des_physics_audit.md) identifies non-Limber cluster
density spectra in Lighthouse and distinguishes active from unused code.
The [FFTLog study](covariance_nonlimber_study.md) records the data-vector
optimization patterns and unresolved growth/kernel choices.

## Shared scope and ownership

Covariances are generated before inference and are never recomputed inside
MCMC. Prepare shared inputs once per generation, retain them across matrix
blocks, and release temporary workspaces afterwards. An explicit workspace
may reuse FFTW plans within this calculation; a chain-oriented persistent
cosmology cache is not a prerequisite.

Every new C implementation stays in `cosmolike/covariances/` and ends in
`_cov.c`. Do not edit data-vector C to obtain additional covariance pairs.
Read public cosmology/catalog APIs, but own covariance grids and scratch.
Warm lazy shared readers serially before parallel use.

Heavy algorithms belong in shared C. Production `_interface` bindings use
direct NumPy storage; notebook `_wrapper` adapters use Armadillo vectors,
matrices and cubes. Both call the same numerical routines. Shared Python
orchestration belongs in `cosmolike_notebook_utils/covariance/`; project
folders contain only survey choices, examples and tests.

Retain flat geometry, massless neutrinos and linear galaxy bias for the
first validation. Keep magnification and RSD as explicit follow-on field
components, never silently ignored requested options. Massive-neutrino
unequal-time growth, TATT, stochastic IA, nonlinear/tidal IA responses,
and non-Limber SSC/cNG need separate physical extensions.

## Original non-Limber ticket and remaining extensions

### N1. Freeze the field and normalization contract

Define one density or observed E-mode field per catalog, with all internal
cross pairs. A measured-row mask must not remove a spectrum from a Wick
contraction. For example, the covariance of `(g_i,s_j)` and `(g_k,s_l)`
needs `(g_i,g_k)`, `(s_j,s_l)`, `(g_i,s_l)` and `(s_j,g_k)`.

Write down the transfer normalization, distance units, power units, spin
factors and low multipoles before coding. Ordinary density uses a radial
integral of `W_A(chi) D(chi) j_l(k chi)`. Lensing has the spin-2 derivative
factor and `j_l(k chi)/(k chi)^2`. Derive the IA transfer from the tidal
field; do not treat it as another scalar density merely because its
radial window is local. Keep observed white noise outside signal factors.

For one common separable linear field, the target is

    C_exact,lin[A,B] = (2/pi) integral dlnk k^3 P0(k) F_A(l,k) F_B(l,k)
    C_model[A,B] = C_exact,lin[A,B] + C_Limber,nonlin[A,B]
                   - C_Limber,matched-lin[A,B].

The subtracted term must use exactly the same growth, anchor spectrum and
windows as the transformed term. Test their high-ell cancellation. Choose
the growth convention from supplied-table comparisons, including w != -1;
massless neutrinos alone do not prove scale-independent growth at every k.
Do not copy independent per-pair pivots without proving a consistent
joint field matrix. A common positive k measure makes the linear term a
Gram matrix, but does not guarantee positivity of the complete hybrid
nonlinear result after subtraction. Check that result separately.

Deliverable: equations, axis/units contract, supported-model guards and
an independent small reference. No public default changes at this stage.

### N2. Validate a covariance-owned transform component

Proposed files: `nonlimber_cov.c/.h`. Keep helpers private unless a second
consumer needs them. Use a caller-owned grouped workspace and serial FFTW
planning; reuse plans while their dimensions and execution requirements
match. Own and release all real/complex buffers explicitly.

Follow the actual `cfftlog_ells_p1/p2` structure in `cosmo2D.c`:

1. Sample each active radial field component once on a uniform log grid.
2. Reuse its forward FFT for all multipoles and pair contractions.
3. Process inverse transforms in bounded multipole blocks.
4. Share k-grid powers and kernel factors across all field pairs.
5. Contract complete pairs in a fixed order on individual workers.

Compare the current external k^-2 lensing formulation with the N5K
`j_l(x)/x^2` Mellin formulation. Resolve observer-endpoint and phase-origin
behavior using analytic tests before choosing one. Carry over gamma and
phase recurrences only with error checks; do not multiply sine/cosine or
gamma evaluations unnecessarily inside every pair.

Keep refinements nested: add log-grid intervals without moving old nodes,
retain the guards' physical logarithmic width, and round the base FFT
length once to an even 2/3/5/7-factor size. Dyadic length refinement must
keep the Fourier period fixed. Test input support, padding and reciprocal
grid offsets separately, including narrow cluster selections.

Initial tests: analytic Gaussian–Bessel transforms; direct spherical-Bessel
quadrature for selected density and spin-2 windows; all-pair symmetry and
linear positive semidefiniteness; signed/near-zero crosses; units; repeated
calls with changed shapes; one/eight-thread agreement; debug sanitizers.
Direct references must refine oscillatory integrals rather than assume an
arbitrary quadrature count resolves them.

### N3. Assemble all-pairs hybrid spectra and expose both APIs

Build gg, gs and ss from shared field transfers, retaining cross-bin and
excluded gs pairs. Keep the existing Limber entry for independent tests.
Use one field-consistent multipole transition; initially calculate every
pair through a supplied cutoff. Do not copy the data-vector per-pair
relative early exit, which is unreliable for a cross spectrum near zero.
Test cutoff extension and continuity before optimizing that transition.

Expose transfers and the exact-linear, matched-Limber and final spectra
as separately inspectable notebook quantities. Production returns the
owned final arrays needed by the shared survey driver without Armadillo
conversion. Validate shape, units and lifetimes in both boundaries; do not
copy physics into wrappers. Proposed binding edits belong in existing
covariance `production_interface_cov.cpp`, component wrappers/bindings and
their headers, plus the optional project build lists as needed.

Integrate the spectra into `survey.py` without changing its G/SSC/cNG
separation. Real-space and integer-band Fourier assembly must consume the
same field convention. Compare the two backends and check that zero
non-Limber correction recovers the existing Limber result.

### N4. Extend to selected cluster density fields

Supply normalized selected-cluster windows and bias from the existing
cluster preparation. Transform cluster–cluster, cluster–galaxy and
cluster–shear pairs in the same shared field system. Preserve the current
selected one-halo spectrum as part of the nonlinear Limber contribution;
do not subtract it as though it were linear power.

Test narrow redshift bins and every crossed family before projecting all
blocks through the existing cluster-lensing Y transformation. Counts,
count Poisson noise and count SSC are unchanged by this spectra ticket.
It does not add the separately missing non-SSC count–spectrum covariance.

### N5. Select accuracy controls and measure costs

Add covariance-owned internal controls for log-distance resolution,
padding/support, non-Limber multipole extent and any coarse/dense tables.
Names and base values become public only after tests. The global
`accuracy_boost` multiplies each internal refinement; it must not replace
the survey's tuned baseline. Refine node counts by intervals and preserve
the old sample positions. `integration_accuracy` remains the independent
quadrature selector, with the supported precomputed GSL rules and a
64-node minimum. FFT lengths are not GSL quadrature counts.

Try exact coarse samples followed by cubic-spline construction of a dense
table and linear hot lookups only where measured transfer smoothness
permits. Oscillatory transfers require off-grid tests; the successful
fixed-k halo interpolation is not evidence that they interpolate equally
well. Start with direct samples and measure both error and saved work.

Profile radial preparation, forward FFTs, inverse FFTs, contraction and
projection separately. Test OpenMP layouts spanning fields and multipole
blocks, with enough work for 8–10 cores; avoid teams limited by a few bins.
Measure 1/2/4/8 threads on a quiet laptop, one numerical job at a time,
and retain unconditional SIMDe. Report M2 results as M2 results; x86
scaling needs its own measurement. Threads come from `OMP_NUM_THREADS`;
BLAS stays at one and C does not call MPI.

### N6. Validate each survey before enabling its default

Begin with a small LSST configuration, then full LSST Y1 and Roman real;
use the actual bins, redshift files, angular ranges and likelihood cuts.
Continue with Roman Fourier/KL, DES Y3, DESxPlanck's supported galaxy/shear
sector and the DES joint cluster forecast. Do not present missing CMB
fields as a completed DESxPlanck 6x2pt covariance.

Compare refinement sequences and Limber/non-Limber physical differences
separately. Keep G, SSC, cNG and total diagnostics. Check per-ell field
positivity and full/selected total positive definiteness without clipping
eigenvalues or adding jitter. cNG alone need not be positive semidefinite.
Use generalized covariance variance ratios and Fisher parameter errors/FoM
with fixed, independently converged data-vector derivatives and priors.
The proposed 1e-3 mode scale is a starting diagnostic, not an automatically
accepted universal threshold. The data-vector delta-chi2 target is not a
covariance accuracy certificate.

Complete optimized/debug checks, optional-covariance-disabled builds and
data-vector regression tests. Finish a separate didactic review of each
major component before starting the next: physics rationale, scalar SIMD
equivalent, explanation of every intrinsic, ownership, paragraph breaks,
one declaration/comparison per line and 80-column C/header lines. Commit
each validated ticket locally; never push.

## Original IA ticket; connected IA remains outside the current scope

### I1. Make the forecast model explicit

First audit `IA.c`, `radial_inputs_cov`, `forecast.py`, `survey.py`,
`forecast_cluster.py` and `survey_cluster.py`. Record the NLA normalization,
growth factor, sign, redshift evolution and source-bin mapping. Compare
the existing low-level window against an independently calculated NLA
amplitude, rather than reusing that window in both sides of the test.

Expose `none` and `NLA` plus named amplitudes/evolution in the shared
configuration. Match each project's established nuisance convention;
do not invent one common nonzero amplitude for all surveys. Remove the
unconditional zeroing only when this configuration is consumed throughout
the workflow. Reject unsupported TATT requests clearly. Save the resolved
IA model and parameters beside each generated matrix.

### I2. Complete Gaussian NLA and connect it to non-Limber

For each observed source E field use lensing plus the signed IA field.
Expand ss into GG, GI, IG and II, and gs into gG and gI. Retain both GI
orderings for unequal source bins, all crossed pairs and unchanged white
shape noise. Preserve exactly one spin conversion per source leg.

With the linear-NLA prescription, combine its local radial amplitude with
the spin-2 transfer from N1–N3. Use the same split and matched linear
subtraction as lensing; nonlinear NLA power remains in the Limber residual.
Do not turn off IA whenever non-Limber is selected.

Test zero amplitude recovery, a sign reversal (odd GI/gI and even II),
bin permutations, an independent GG/GI/IG/II expansion and high-ell matching.
Compare production/notebook calls and one/eight-thread outputs in real
and Fourier space. Include cluster–I when adding the cluster source legs.

### I3. Specify and validate the SSC/cNG NLA approximation

A two-point NLA prescription does not uniquely determine every connected
four-point or background-response term. Derive and document the proposed
deterministic, fixed-amplitude projected-NLA approximation before enabling
it: hold the IA amplitude fixed under the background perturbation and use
the signed source window on every relevant leg of the matter response
and trispectrum. This is a model choice to validate, not a consequence of
Takada & Hu alone or a claim of a complete intrinsic-shape halo model.

Replace the source-lensing-only selection in `survey.py` consistently.
Keep survey-mean subtraction for density catalogs distinct from source
shape terms. Expand products explicitly in an independent reference,
including one through four IA legs and their signs. Propagate the same
source-field choice into cluster–shear and count–source SSC crosses before
the Y transformation. Counts themselves do not acquire an IA weight.

Measure the effect of this approximation on full covariance modes and
Fisher results over the intended NLA range. If its response/trispectrum
assumptions cannot be justified, expose Gaussian-NLA as a limited component
and retain an explicit guard for a full NLA forecast. Do not silently mix
Gaussian NLA with zero-IA SSC/cNG and label it complete NLA support.

### I4. Release survey examples after validation

Update shared settings/metadata and each project's thin YAML/notebook
examples; show zero-IA versus a stated NLA configuration and separate
G/SSC/cNG effects. Explain assumptions in human READMEs with direct paper
references. Test the same cosmology/nuisance configuration through CLI
and notebook paths, array ownership, repeated model changes and disabled
covariance builds. Run the project data-vector checks without refreezing
their supplied likelihood covariances.

## Completion boundaries

N1–N6 complete non-Limber **spectra used by Gaussian covariance** under the
declared hybrid approximation. They do not remove long-mode Limber from
SSC or unequal-time approximations from cNG. Archive these model choices
separately; never change a single blanket label to “fully non-Limber.”

I1–I4 complete the explicitly validated NLA forecast approximation only.
Gaussian TATT/B modes are now implemented as recorded above. Stochastic
IA and IA-specific nonlinear/tidal responses remain separate extensions. A converged numerical calculation is not
proof that any of these physical approximations is adequate for a survey.

The Gaussian-only implementation authorization supersedes the original
planning-only status. Never test integration above
level 4; Roman on this laptop stops at level 3. Preserve completed LSST
levels 0–3 and both interrupted level-4 folders. Never restart the paused
overnight validation automatically.
