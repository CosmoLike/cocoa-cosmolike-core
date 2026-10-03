# Covariance-owned halo mass moments

`halo_cov.c/.h` builds I11 and the five pair moments I02(K,Q), I12(K,Q),
I13(K,Q,Q), I13(K,K,Q), I04(K,K,Q,Q). It calls the public halo readers
allowed by study Section 5.0, with covariance-owned mass panels and GL
node counts. It does not modify existing core C or Ntable settings.

The abundance, peak height, bias, concentration growth and moment density
use cb. These are **cb-field moments**, normalized by M/rho_cb. The
public NFW profile retains the core's M200m radius definition. A full
massive-neutrino total-matter covariance requires the corresponding cold
fractions and cross terms; these moments alone do not supply that model.
The pinned integration test is massless, where cb and total matter agree.

I11 receives `[1 - integral dn b M/rho_cb] u(k|M_min)` on the same mass
rule used for its resolved integral. Higher moments receive no completion
and no bias-normalization divisor. This keeps I11(0)=1 while following
the core convention for the higher moments. The physical definition is
[Takada & Hu, Eq. 25](https://arxiv.org/html/1302.6994v3); the low-mass
completion follows [Mead et al., Appendix A](https://arxiv.org/abs/2005.00009).
The cb mass-function prescription is reviewed in the skill's neutrino
halo reference and [Castorina et al.](https://arxiv.org/abs/1311.1212).

Each abundance, bias and concentration is computed once per (a,M).
Each NFW profile is computed once per (a,k,M), shared by all pairs.
Mass weights are grouped by physical role. I11 and the five pair sums
use two SIMDe lanes for independent outputs, preserving mass-node order;
OpenMP owns entire outputs. Core lazy tables are warmed serially. There
is no covariance cache or BLAS call.

## Numerical checks

Five tests pass with the isolated optimized and debug/UBSan builder:

- NumPy-generated GL nodes and independent moment contractions of shared
  physical samples agree within 2e-11; the measured maximum pair difference
  was 8.62e-13. This validates integration and array roles, not a separate
  calibration of the core halo fits or NFW table.
- I11(0)=1 for lower mass cutoffs 1e6, 1e9 and 1e11 M_sun/h; the two I13
  roles coincide exactly on the diagonal.
- At a=0.35,0.7,0.95, k=0 through 300 h/Mpc and eight logarithmic mass
  panels covering 1e6–1e17 M_sun/h, mass refinement is measured below.
- The independent reference has the expected length^3,^6,^9 dimensions.
- Complete repeated builds at 1/4/8 threads agree bitwise; changing and
  restoring scale factors gives changed and restored outputs respectively.

| Nodes per panel, before → after | Maximum relative pair-moment change |
|---|---:|
| 64 → 128 | 3.22135e-2 |
| 128 → 256 | 4.15854e-3 |
| 256 → 512 | 1.82173e-5 |
| 512 → 1024 | 9.51201e-8 |

The last I11 change is 8.48e-9. High-k profile oscillations make low-node
rules inadequate even when I11 already appears converged. This finite
scan does not establish the production rule for every radial/multipole
node. Accuracy of the public sigma and NFW tables remains part of the
input-state scan. A native macOS sampling profile on eight scale factors,
64 wavenumbers and 4096 mass nodes assigns 2398 of 3877 main-thread samples
to profile construction and 1220 to pair contractions. Repeated radius,
logarithm and trigonometric work in the public NFW reader is a measured
optimization candidate. No private kernel copy was added before the
survey-level accuracy checks; these samples are not a timing benchmark.
The raw profile is `results/halo_native_profile.txt` outside git.

After the project tests finished, that same 8 × 64 × 4096 workload took
142.330 ± 7.466, 37.539 ± 0.498 and 20.599 ± 1.109 ms with 1, 4 and 8
threads respectively (mean ± sample standard deviation, 21 calls after
three warm-ups). The Apple M2 Pro used strict IEEE Clang 19.1.7; desktop
applications remained active. Core physics readers were warm. Internal
workspace allocation and moment construction are included; CAMB, initial
sigma-table construction and Python allocation are excluded. This is not
a complete survey covariance timing. See the external
`benchmark_covariance_components.py` and `results/component_timings.json`.
Native disassembly in `results/halo_simd.disassembly.txt` confirms vector
multiply and fused multiply-add instructions in the contractions.

External files: `halo_inputs.py` reads the shared physics, while
`halo_reference.py` independently contracts numeric arrays and imports no
project. `build_halo.sh` links the isolated builder against the existing
LSST interface. Logs and sampled inputs are in `covariance_reference/`.
Set `COSMOLIKE_HALO_COVARIANCE_LIBRARY` and the usual external-reference
path to run LSST's `test_covariance_halo.py`.

## Didactic red-eye pass

After the five focused tests passed, reread the complete C/header before
assembling responses or trispectra. Comments distinguish peak-height
physics, number density, volume factors, cb versus total observables,
profile mass definition, unresolved mass, the five moment roles and each
precomputation stage. Allocation ownership and deterministic output lanes
are explicit. Every C/header line fits 80 columns, with one predicate per
line. A simple density guard rejects uninitialized input. This is a manual
self-review, not a Fable review. Full covariance/model accuracy is still
an open gate.
