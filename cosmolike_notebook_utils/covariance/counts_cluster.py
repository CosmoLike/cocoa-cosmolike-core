"""Compute cluster count means, Poisson/SSC terms and count-matter cross covariance.

The count bins are exclusive observed categories: an object is assigned to
one richness/redshift bin. Their true-redshift or true-mass distributions
may overlap. Conditional Poisson sampling then gives diagonal count noise,
while the common large-scale density field correlates the bins through SSC.

The caller supplies the selected abundance and its response, including any
environment-dependent selection. This module does not choose that physical
model. Count-two-point SSC uses the same shell responses as the two-point
covariance. The separate non-SSC count-two-point term is computed by
count_matter_cross, for projected matter and linearly biased galaxy
partners only; an SSC cross block alone is not a complete joint cluster
covariance.
"""

import numpy as np


def count_matter_cross(interface, distance, dchi, pair_window, transfer,
                       linear_power, i11, moments):
    """Project the non-SSC cross covariance of counts and matter spectra.

    Schaan, Takada & Spergel (2014), arXiv:1406.3330, Eq. 35 gives
    integral dchi W_A W_B/f_K^2 [J02_i(k,k)+2 P_lin(k) I11(k) J11_i(k)].
    The first term places both matter points in the counted halo. The
    second correlates that halo with a different halo; either
    power-spectrum leg may belong to the counted halo, giving the factor
    two. The survey area in the absolute count cancels the inverse area
    of this local covariance. The SSC term, computed separately, does
    depend on the survey footprint.

    Arguments:
        interface = compiled project with covariance_project.
        distance, dchi = positive float [state], f_K and radial weights, L.
        pair_window = finite float [observable,state], W_A*W_B, L^-2.
            These are projected matter fields with fixed ensemble-mean
            normalization. Constant linear galaxy-bias factors can be
            included; discrete cluster legs and shared-object catalog
            noise require additional terms and are not covered.
            Observed-catalog-mean corrections are not supplied here.
        transfer = finite float [observable,k], product of source-leg spin
            factors in the intended harmonic convention (ones for scalars).
        linear_power, i11 = finite float [state,k], nonnegative P_lin in
            L^3 and the dimensionless biased one-profile moment of all
            halos, not only the selected ones. Each column follows one
            angular multipole, with k=(ell+1/2)/f_K per state.
        moments = output of covariance_cluster_moments on those same k
            samples. Its selected weights must include the count catalog's
            radial selection exactly once. J11 [state,count,k] is
            dimensionless; J02 [state,count,k*(k+1)/2] has units L^3.
    Returns:
        Dict of owned, dimensionless one_halo, two_halo and total arrays,
        each [count,observable,multipole]. total is only the non-SSC
        count-spectrum cross block. It is neither a full joint covariance
        nor a count-count covariance. Angular or bandpower operators can
        be applied to its final axis afterwards.
    Raises:
        ValueError for incompatible shapes, nonpositive distance or dchi,
        negative linear power or nonfinite inputs.
    """
    distance = np.asarray(a=distance, dtype=float)
    dchi = np.asarray(a=dchi, dtype=float)
    window = np.asarray(a=pair_window, dtype=float)
    transfer = np.asarray(a=transfer, dtype=float)
    power = np.asarray(a=linear_power, dtype=float)
    full_i11 = np.asarray(a=i11, dtype=float)
    j11 = np.asarray(a=moments['J11'], dtype=float)
    j02 = np.asarray(a=moments['J02'], dtype=float)
    if distance.ndim != 1 or distance.size == 0:
        raise ValueError("distance must be nonempty [state]")
    if (dchi.shape != distance.shape or np.any(distance <= 0.0)
            or np.any(dchi <= 0.0)):
        raise ValueError("distance and dchi must be positive matching vectors")
    nstate = distance.size
    if power.ndim != 2 or power.shape[0] != nstate or power.shape[1] == 0:
        raise ValueError("linear_power must have shape [state,k], with k nonempty")
    nk = power.shape[1]
    if full_i11.shape != power.shape or np.any(power < 0.0):
        raise ValueError("i11 must match nonnegative linear_power[state,k]")
    if (window.ndim != 2 or window.shape[0] == 0 or window.shape[1] != nstate
            or transfer.shape != (window.shape[0], nk)):
        raise ValueError("need pair_window[observable,state], transfer[observable,k]")
    if (j11.ndim != 3 or j11.shape[0] != nstate
            or j11.shape[1] == 0 or j11.shape[2] != nk):
        raise ValueError("moments['J11'] must have shape [state,count,k]")
    ncount = j11.shape[1]
    if j02.shape != (nstate, ncount, nk*(nk+1)//2):
        raise ValueError("moments['J02'] must match [state,count,k*(k+1)/2]")
    for values in (distance, dchi, window, transfer, power, full_i11, j11, j02):
        if not np.all(np.isfinite(values)):
            raise ValueError("count-spectrum inputs and selected moments must be finite")

    # The moment table keeps the triangular (K,Q) pairs. A single power
    # spectrum uses opposite vectors of the same magnitude, hence K=Q.
    first, second = np.triu_indices(n=nk)
    diagonal = np.flatnonzero(first == second)
    one_halo = np.take(a=j02, indices=diagonal, axis=-1)
    two_halo = 2.0*power[:, None, :]*full_i11[:, None, :]*j11
    measure = np.ascontiguousarray(dchi/distance**2)
    right = np.ascontiguousarray(window)
    result = {}

    # All angular samples use the same radial rule. Arrange count/mode as
    # independent C output rows, so the existing SIMDe/OpenMP contraction
    # integrates them together rather than starting a Python loop per pair.
    for name, kernel in (('one_halo', one_halo), ('two_halo', two_halo)):
        left = np.ascontiguousarray(kernel.transpose(1, 2, 0))
        left = left.reshape(ncount*nk, nstate)
        projected = interface.covariance_project(left=left, right=right,
                                                 weight=measure)
        projected = projected.reshape(ncount, nk, len(window))
        result[name] = np.ascontiguousarray(projected.transpose(0, 2, 1)*transfer)
    result['total'] = result['one_halo']+result['two_halo']
    return result


def count_statistics(interface, distance, dchi, density, derivative, area_sr,
                     background_variance, two_point_response=None):
    """Integrate supplied count-shell quantities and their shared-mode SSC.

    For a bin i, S_i=Omega*f_K^2*n_i and Phi_i=Omega*f_K^2*dn_i/d(delta_b).
    Mean counts are integral S_i dchi. SSC is the weighted outer product
    integral Phi_i Phi_j sigma_b^2 dchi. All count and two-point responses
    must describe the same background overdensity, mask and radial nodes.

    Arguments:
        interface = compiled project with covariance_counts_shell and
            covariance_project. It is supplied by the notebook, not imported.
        distance = positive float [nnode] transverse distances f_K, in L.
        dchi = positive float [nnode] quadrature weights, in the same L.
        density = nonnegative float [ncount,nnode] selected abundance, L^-3.
        derivative = finite float of the same shape, dn_i/d(delta_b), L^-3.
        area_sr = survey solid angle in steradians, inside (0,4*pi].
        background_variance = nonnegative float [nnode] sigma_b^2, in L.
            This is the long-mode Limber strength, not a dimensionless
            finite-shell variance. Recompute it when the mask changes.
        two_point_response = optional finite float [ndata,nnode], in L^-1,
            after the intended angular/bandpower transform and catalog-mean
            subtraction. None requests no count-two-point SSC block.
    Returns:
        Dict with mean [ncount] expected counts; poisson, ssc, total
        [ncount,ncount] in squared counts; cross_ssc [ncount,ndata] (zero
        columns if not requested); and shell_density, shell_response
        [ncount,nnode], in L^-1. total covers only the count-count
        Poisson+SSC model. No array is repaired to enforce positive
        eigenvalues. No files are written.
    Raises:
        ValueError for incompatible shapes or nonphysical integration inputs.
        Shared-object or weighted count catalogs require a different Poisson
        model and are outside this function's exclusive-bin contract.
    """
    distance = np.ascontiguousarray(distance, dtype=float)
    dchi = np.ascontiguousarray(dchi, dtype=float)
    variance = np.ascontiguousarray(background_variance, dtype=float)
    density = np.ascontiguousarray(density, dtype=float)
    derivative = np.ascontiguousarray(derivative, dtype=float)
    if distance.ndim != 1 or distance.size == 0:
        raise ValueError("distance must be a nonempty 1D array")
    for name, values in (("dchi", dchi), ("background_variance", variance)):
        if values.shape != distance.shape or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite and match distance.shape")
    if np.any(dchi <= 0.0) or np.any(variance < 0.0):
        raise ValueError("dchi must be positive and background_variance nonnegative")

    # Check the optional cross responses before requesting any integration.
    # Signed responses are physical: an observed-mean subtraction can make
    # a two-point response negative even when its power spectrum is positive.
    other = None
    if two_point_response is not None:
        other = np.ascontiguousarray(two_point_response, dtype=float)
        if (other.ndim != 2 or other.shape[0] == 0
                or other.shape[1] != distance.size
                or not np.all(np.isfinite(other))):
            raise ValueError("two_point_response must be finite [ndata,nnode]")

    shells = interface.covariance_counts_shell(
        distance=distance, density=density, derivative=derivative,
        area_sr=area_sr,
    )
    shell = shells["shell_density"]
    response = shells["shell_response"]

    # Each shell supplies expected objects per unit distance, dN_i/dchi,
    # not a normalized galaxy window. Integrating against a row of ones
    # counts all of them. The C contraction owns the ordered sums and
    # OpenMP work for every bin.
    unity = np.ones(shape=(1, distance.size), dtype=float)
    mean = interface.covariance_project(left=shell, right=unity, weight=dchi)
    mean = mean[:, 0]
    poisson = np.diag(mean)

    # Each shell adds its nonnegative weight dchi*sigma_b^2 times the outer
    # product of one response vector with itself, so count SSC is positive
    # semidefinite. Every cross-bin term is retained, including bins whose
    # true-redshift distributions overlap despite distinct labels.
    weight = np.ascontiguousarray(dchi*variance)
    ssc = interface.covariance_project(left=response, right=response, weight=weight)
    ssc = np.triu(ssc)+np.triu(ssc, k=1).T

    # Counts occupy the left index. Copy its transpose when inserting this
    # block on the opposite side of a joint matrix; do not recompute it
    # using a different selection, radial grid or background normalization.
    cross_ssc = np.empty(shape=(len(mean), 0), dtype=float)
    if other is not None:
        cross_ssc = interface.covariance_project(
            left=response, right=other, weight=weight,
        )

    return {
        "mean": mean,
        "poisson": poisson,
        "ssc": ssc,
        "total": poisson+ssc,
        "cross_ssc": cross_ssc,
        "shell_density": shell,
        "shell_response": response,
    }
