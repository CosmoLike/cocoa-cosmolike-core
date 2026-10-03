# SSC/cNG survey projection: preliminary measurements

This is a Phase-0 diagnostic, **not a production covariance** and not the
full study acceptance gate. The external code in
`test/covariance_reference/` connects the tested halo, angular, response
and projection components without modifying a project covariance.

## Configuration and limits

The archived configuration has two overlapping lens bins, two source bins,
signed lens magnification, NLA, massless neutrinos, flat geometry, and a
12300 deg^2 polar-cap footprint. Densities are recorded in arcmin^-2 and
ellipticity dispersions per component. The CAMB arrays are shared inputs;
the reference contractions are independent NumPy computations.

The raw mask file is `inputs/cap_mask_cl.txt`, with L=0..4096 and
C_0=Omega^2/(4 pi). The subsequent [pair-geometry ticket](covariance_mask.md)
implements noise geometry from this same raw-mask convention. It finds that
the pilot's cutoff leaves 1.05e-5 pair-area error and up to 1.30e-6 SSC shell
error; the full survey gate must include a refined mask. This diagnostic
archive retains its original cutoff so the results remain reproducible.

The diagnostic uses 24 logarithmic band centers from ell=30 to 50000,
all ten field pairs, and no data-vector exclusions. The Gaussian mode
count uses each band's width at its center. **These are not exact band
averages.** Short spectra are Limber, with RSD off in every field pair.
The mean response uses density plus magnification for lens catalogs and
zero for sources. Deterministic NLA weights are held fixed when applying
the supplied matter response. The broad footprint's long-mode variance
uses the stated Limber approximation, not the full curved-sky BKS result.

For a unit-normalized observed-shear harmonic convention, signal receives
sqrt[(ell-1)(ell+2)/(ell(ell+1))] per source leg in addition to the core
harmonic transfer. White shape noise receives no such factor. Exact
spin operators and non-Limber spectra remain separate validation gates.

## Projection and independent checks

The cNG reference directly evaluates, for every pair of observable rows,

    integral dchi W_A W_B W_C W_D T / (Omega f_K^6).

The production weighted contraction evaluates the same shell as A T A^T
in two matrix contractions. SSC uses the supplied shell response Phi and
positive weights dchi sigma_b^2, retaining its factor form. The independent
reference uses NumPy contractions with a different operation order.
Maximum differences, divided by the geometric mean of covariance diagonals:

| Check | Maximum |
|---|---:|
| C versus NumPy SSC projection | 3.17e-15 |
| C versus NumPy cNG projection | 1.78e-14 |
| cNG after c/H0 to physical-Mpc conversion | 3.66e-15 |
| SSC after the same conversion | 1.04e-15 |

The units check rescales distances, measures, windows, powers, responses,
trispectra and the length-valued sigma_b^2 consistently. It does not merely
rescale a finished matrix. No eigenvalue is clipped and no negative term
is discarded.

With 64 radial nodes per panel, eight mass panels of 512 nodes, and twenty
angular panels of 256 nodes, correlation eigenvalue minima are:

| Contribution | Minimum eigenvalue |
|---|---:|
| Gaussian | 0.00642767 |
| Isotropic SSC | -2.42e-15 |
| Projected-tree SSC | -1.46e-15 |
| cNG | -57.0645 |
| G + projected-tree SSC + cNG | 0.00638893 |

The cNG cumulant alone is indefinite. The total passes this diagnostic's
PSD check; this is not evidence for all real-space bins or survey settings.
The refined run keeps the total minimum at 0.00638892.

## Refinement remains open

Doubling the radial, mass and angular counts together to 128, 1024 and 512
changes total-covariance generalized eigenvalues to
[0.99980428, 1.00020527]. Component changes divided by the diagonal scale
are 6.03e-6 (G), 2.02e-3 (isotropic SSC), 4.51e-3 (projected SSC), and
5.28e-4 (cNG). Relative changes of individual near-zero SSC entries can
be much larger. This **fails** the study's 1e-6 elementwise refinement
criterion; numerical settings and the full contract must not be frozen.

Separate real-CAMB angular tests at a=0.35,0.6,0.9,0.99, with multipoles
through 50000, show that 512 to 1024 nodes per graded angular panel changes
all P/B/T averages by at most 4.50e-7 using the stated small-value floor.
The largest k in this scan is about 1655 h/Mpc, where the supplied power
table is continued by its endpoint power law. This validates convergence
of that continuation, not the physical accuracy of an extrapolated model.

Small-step response derivatives also need their own convergence study:
the public log-power reader is piecewise linear, so its logarithmic slope
changes discontinuously at input knots. A smaller difference step does not
make the line-of-sight integrand smoother. The pilot deliberately exposes
this sensitivity rather than silently smoothing or changing the spectrum.

An isolation run doubles only the radial nodes. Comparing it with the run
that also doubles mass/angular resolution leaves changes of 4.52e-8 in
SSC and 7.68e-8 in cNG on the diagonal scale: radial sampling dominates
this diagnostic. An external experiment differentiating a cubic lnP(lnk)
interpolant through the same CAMB nodes reduces the projected-SSC radial
change from 4.51e-3 to 2.83e-4. This implicates the discontinuous supplied
slope, but does not meet the gate or justify a production interpolation
change. Neither experiment changes a production default.

Using that smooth interpolant consistently for every linear power in the
tree terms reduces the 64-to-128 radial change in cNG to 1.80e-5. Raising
the radial count from 128 to 256 then changes G by 3.16e-6, isotropic SSC
by 8.76e-6, projected SSC by 1.13e-5 and cNG by 6.49e-6 on the diagonal
scale. Total generalized eigenvalues are [0.99999463, 1.00000473]. This
is a promising numerical route, still outside production and still short
of the componentwise gate. It adds no change to the data-vector reader.
The experiment lives in `smooth_power_inputs.py`; its paired runs and
comparison are `survey_projection_smooth*` and `smooth_power*_refinement`.

## Model differences that cannot be ignored

All comparisons use the same positive total diagnostic covariance as the
reference. The eigenvalues below solve C_alternative v = lambda C_reference v.

| Alternative | Mean trace shift | Eigenvalue range |
|---|---:|---:|
| Published isotropic response versus projected tree | -3.4149e-4 | 0.91824–1.07560 |
| Old half-strength 1+3 versus full four partitions | +1.1765e-4 | 0.88773–1.12957 |

Both move some modes by more than 2%. The small mean trace change alone
would hide that. The corrected 1+3 multiplicity follows the labelled halo
partitions; the old value is retained only as an external comparison.
The two response choices remain explicit, with the published two-halo
slope distinguished from the study's projected linear slope.

For three specified parameter displacements, delta^T C^-1 delta is:

| Data-vector displacement | Projected/full 1+3 | Isotropic/full 1+3 | Projected/half 1+3 |
|---|---:|---:|---:|
| First lens bias +1% | 269.6417 | 267.8357 | 272.7715 |
| First source A1 +0.1 | 72.9225 | 72.9147 | 72.9326 |
| Both shear calibrations +0.005 | 24.2700 | 24.2379 | 24.2602 |

These are differences from the pinned theoretical vector, not chi-squared
against shipped data. They are harmonic center-of-band diagnostics and
cannot certify a production real-space delta-chi-squared threshold.

## Reproduction and didactic review

External adapters and references:
`survey_ng_inputs.py`, `projected_ng_reference.py`,
`check_realistic_ng_nodes.py`, `check_survey_projection.py`, and
`check_projected_ng_algebra.py`. The JSON/NPZ evidence is under
`covariance_reference/results/`, named `realistic_ng_*`,
`survey_projection_*`, `projected_ng_algebra`, `projected_refinement`,
and `survey_displacements`. The adapters read the compiled core; the
independent projection reference imports only NumPy.

After the projection comparisons and refinement run, a separate manual
red-eye pass checked the complete adapter and reference: row ordering,
array shapes, units, mean-window versus short-mode windows, signed terms,
and the distinction between a diagnostic and an exact band average.
Unused reference parameters were removed so the signature states the
actual mathematical inputs. This is a self-review, not a Fable review.
The failed refinement gate remains visible; no reference was refrozen.

The later smooth-power experiment was reviewed separately: its spline
interpolates lnP in lnk, uses endpoint secants as clamped derivatives, and
continues those slopes as power laws outside the input range. Redshift
interpolation remains linear. Every linear power within a tree diagram
uses this same interpolant, while the nonlinear target stays unchanged.
The class owns its one-row cache and does not monkey-patch the interface.
This external numerical experiment does not change production code.
