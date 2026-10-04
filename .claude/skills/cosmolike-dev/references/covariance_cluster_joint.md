# Joint angular cluster forecast

## Model boundary

`forecast_cluster.compute_forecast` assembles the DES 2812-entry layout:
ss, gs, gg, cg, N, cc, cs. `des_cluster_joint_covariance.py` owns the
project's three observed-redshift/four richness categories and the
initialization. Algorithms remain in the shared core notebook package.
All C work reuses existing covariance components; no data-vector C changes.

The model is massless, Limber, linear galaxy bias, zero IA/magnification/RSD,
fixed lognormal selection with selection_model=0, abundance-weighted
cluster windows, fixed NFW profiles and isotropic SSC. The Gaussian mean
model is the hybrid DES prescription: cc/cg from biased nonlinear power;
cs additionally contains the selected cluster's own halo profile.

For cNG alone, every density leg is approximated by b*delta_m, giving
the product of linear tracer biases times the full matter trispectrum.
This follows directly from multilinearity of a connected four-point
function. It is also the prescription in lighthouse's cluster covariance.
It is not the complete selected-halo model of Krause & Eifler Appendix A:
selected-cluster one-halo corrections remain omitted. Count cross
covariance contains SSC only. The non-SSC count-matter component already
exists separately, but integrating it with discrete cluster partners and
observed-catalog normalization remains open. Do not silently call those
missing terms zero in a purported complete discrete-halo calculation.

The output always records these omissions, including non-Limber, nonlinear
bias, tidal and environmental-selection responses. The exported forecast
is a complete matrix only within this stated approximation. Positivity
and numerical refinement cannot validate the omitted physical terms.

Sources directly checked: Krause & Eifler 1601.05779 Appendix A;
Takada & Hu 1302.6994 corrected Eq. 44; Schaan, Takada & Spergel
1406.3330 Eq. 35; To et al. 2008.10757; Park, Rozo & Krause 2004.07504;
DES 2503.13631 Sec. II.4. Local legacy code is a convention comparison,
not a substitute for those physical derivations.

## Shared SSC and localization

One radial rule supplies all count and two-point shell responses. The
selected own-profile contribution is J11/n at fixed reference n; subtract
the observed catalog response once as (U_A+U_B)*C_AB after angular
projection. U=chi^2*B/nbar for cluster density, and zero for shear.
Absolute counts instead respond as Omega*chi^2*B. Their common weighted
outer product supplies all SSC cross blocks, including cross-redshift
and cross-richness categories.

The mass reader works in batches of 16 active shells, limiting temporary
storage when boosts increase mass/k sampling. The profile response uses
twice the ordinary halo per-panel mass count over the supplied full cluster
mass interval. Angular spectra stream in 1024-multipole blocks. Neither
storage size is an accuracy control or changes a sum's order.

The project's existing Y operator acts on both matrix axes and the mean.
The last angular bin of each of 48 cs rows is exactly zero. Full arrays
retain those rows; valid_indices removes only their defined null modes,
not physical small-scale cuts or numerically negative modes. After A*C*A^T,
the assembler copies one triangle to enforce exact symmetry, matching the
ordinary SSC/cNG assembly. Roundoff from the two multiplication orders is
not an eigenvalue repair, and no diagonal variance is adjusted.

## Checks (2026-10-04)

External independent assembly used 704 radial nodes (11 panels x 64),
ell_max=10000, mask_ell_max=4096, angle_nquad=128, nwindow=4097,
16 NG multipole samples and 256 selected-mass nodes over 1e12--1e16.
The G+Poisson+SSC pilot's minimum correlation eigenvalues were
0.00741330 before Y and 1.79478e-5 after removal of the 48 Y null rows.
The count-SSC blocks agreed with the independently called count component
within 3.71e-16 in their natural variance units.

Adding the biased-tracer cNG baseline gives minima 0.00731075 and
1.78194e-5 respectively. Cholesky reconstruction residuals are below
1e-15 in correlation units. These do not certify omitted terms or accuracy.

The public full driver agrees with the separately assembled G/SSC/cNG
pilots to maximum errors 2.077e-15, 3.111e-16 and 1.183e-15 when each
entry is scaled by sqrt(C_total,ii*C_total,jj). Its Y total has minimum
correlation eigenvalue 1.78193938255e-5, and Cholesky residual 9.99e-16.
External evidence: covariance_reference/cluster_ssc_pilot.py,
cluster_cng_pilot.py, check_cluster_joint_forecast.py and their JSON/NPZ
results. These runs overlapped regression tests and are not benchmarks.

All 46 DES covariance tests pass. The joint test uses all 140 rows and
12 counts with five angular bins. It checks bitwise one/eight-thread
G/SSC/cNG/total and mean arrays, count Poisson normalization, nonzero
SSC crosses, the deliberately absent non-SSC crosses, count means against
the independent ordinary count predictor (2e-6 relative), two-sided Y
propagation, all 48 zero rows, total positivity and archive roundtrips.
The archive retains every numerical array, row positions, resolved model
choices and omissions without pickle. Earlier component tests retain
their independent NumPy/analytic and debug/sanitizer checks.

The first full-driver run exposed the mean operator's Fortran-contiguous
layout at a C-array boundary; an explicit contiguous copy fixes it. The
next exact-symmetry assertion exposed the A*C*A^T roundoff described
above. Copying the computed upper triangle establishes the same convention
as other covariance assemblers. Neither issue involved physical refreezing.

## Didactic and optimization review

After component/full-matrix tests, the manual review followed every stage:
all-pairs versus measured spectra; absolute counts versus normalized
windows; one-factor selection; mass and distance units; core versus real
source conventions; shared matter work; J11 versus J01; a single projected
catalog subtraction; count insertion; omitted cNG/count terms; and Y
before any scale selection. Loop overviews explain their physics and why
work is batched. No new SIMD intrinsic or C loop was introduced.

Potential performance work remains measured, not presumed: own-profile
response requests the general moment interface although it uses only J11;
profile table reads are dispatched per radial shell; the matrix transforms
make owned copies. Profile first on a quiet machine, then compare 1/2/4/8
workers and check accuracy before changing these paths. No speedup claim
or notebook stage time from contended runs is performance evidence.

The DES notebook executes headlessly through nbconvert, retaining its
galaxy/shear real/Fourier examples and adding joint angular boosts 1 and
2. Local Jupyter kernel sockets required sandbox escalation, which was
approved. All cells completed; the final two scientific plots were
extracted and visually inspected (correlation triangles and component
maps/histogram). All labels and the narrow count block remain readable.
The second run's plots are byte-identical to the inspected images. A
subsequent paragraph/comment formatting pass leaves the code AST unchanged.

Both 2764-entry Y totals are positive. Their minimum correlation
eigenvalues are 1.78193938305e-5 and 1.78220177945e-5. The maximum
generalized variance departure of boost 1 from boost 2 is 0.18190947;
the reference compared to itself differs by about 3.5e-11 from rounding.
No inference-accuracy claim follows from this teaching comparison.
The shared readmes, core README and all project README pages render with
working local links/anchors. The DES covariance-model note was updated;
unrelated quoted installation/usage blocks were preserved.

The galaxy/shear Fourier example remains separate; this joint cluster
generator is angular only. Notebook stage times are ordinary contended
run output, not controlled performance measurements.
