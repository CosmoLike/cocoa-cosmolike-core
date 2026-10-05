# Cluster-lensing localization of a joint covariance

`covariance/transform_cluster.py` propagates the supplied mean-model angular
operator across all selected rows and columns of a joint matrix. Explicit
joint-vector positions allow counts between two-point blocks. No large
block-diagonal transformation matrix is constructed. Two batched calls to
the existing SIMDe/OpenMP projection carry out the products, retaining each
sum's order. No data-vector C is changed and BLAS remains single-threaded.

Physics anchors checked directly: Park, Rozo & Krause (2021),
arXiv:2004.07504, Eqs. 9--12; DES (2025), arXiv:2503.13631, Sec. II.4.
The mean model's T=2S+SD supplies the same finite-difference convention.
The localization must include count and other-probe cross blocks. Use the
complete unmasked input grid and apply scale selection afterwards. The
last angular row is zero by definition and is not a negative-variance bug.
Selection factors are not inferred from this linear transformation.

## Checks (2026-10-04)

Six public tests in `des_cluster/tests/covariance/test_transform_cluster.py`
pass. Affine polynomials in ln(theta) have a closed exact integral and
test the mean operator at 5/7/20 bins. For 5/20 bins a dense independent
joint transformation checks all cross blocks, including three intervening
count/other entries and two cluster-lensing rows. Results are bitwise
identical at 1/2/4/8 threads. Input ownership, signed components, malformed
maps and nonfinite input guards are checked. Removing only the known zero
rows leaves the positive input example positive definite.

An untimed physical check applies the operator to the saved full DES
2800-entry Gaussian two-point pilot: 48 cluster-lensing rows, 20 angular
bins each. The resulting 48 deterministic zero rows are exact. The other
2752 entries have minimum correlation eigenvalue 1.79570e-5 and pass
Cholesky, with maximum correlation reconstruction residual 9.99e-16.
Maximum correlation asymmetry is 5.55e-16. No physical scale cuts or
eigenvalue repair were applied. Counts, SSC and cNG are not included in
this particular pilot; it is not full joint-model validation.
External reproduction: `test/covariance_reference/cluster_localization_pilot.py`
and its JSON report. No performance claim is made while project
regressions are running.

## Separate didactic review

Read the entire helper, tests and public explanation after the numerical
checks and before the next major ticket. The explanation derives why
one versus two transformations appear in different blocks; defines the
joint indexing and angular order; shows why the second product reads the
first product's result; and separates deterministic zero rows from bad
eigenvalues. Array rearrangements explain which axis the C projection
sums. Existing C lane-level documentation remains the implementation
reference; no new intrinsic or scalar fallback was added.
