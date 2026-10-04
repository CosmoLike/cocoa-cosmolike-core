"""Count means, Poisson noise and SSC from supplied cluster abundances.

The count bins are exclusive observed categories: an object is assigned to
one richness/redshift bin. Their true-redshift or true-mass distributions
may overlap. Conditional Poisson sampling then gives diagonal count noise,
while the common large-scale density field correlates the bins through SSC.

The caller supplies the selected abundance and its response, including any
environment-dependent selection. This module does not choose that physical
model. Count-two-point SSC uses the same shell responses as the two-point
covariance. The non-SSC count-two-point term is a separate calculation;
an SSC cross block alone is not a complete joint cluster covariance.
"""

import numpy as np


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
        area_sr = survey solid angle, inside (0,4*pi].
        background_variance = nonnegative float [nnode] sigma_b^2, in L.
            This is the long-mode Limber strength, not a dimensionless
            finite-shell variance. Recompute it when the mask changes.
        two_point_response = optional finite float [ndata,nnode], in L^-1,
            after the intended angular/bandpower transform and catalog-mean
            subtraction. None requests no count-two-point SSC block.
    Returns:
        Dict with mean [ncount]; poisson, ssc, total [ncount,ncount];
        cross_ssc [ncount,ndata] (zero columns if not requested); and
        shell_density, shell_response [ncount,nnode], in L^-1.
        total refers ONLY to the count-count Poisson+SSC model. No array
        is repaired to enforce positive eigenvalues. No files are written.
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

    # Each shell supplies expected objects, not a normalized galaxy window.
    # Integrating against a row of ones counts all of them. The existing
    # C contraction owns the ordered sums and OpenMP work for every bin.
    unity = np.ones(shape=(1, distance.size), dtype=float)
    mean = interface.covariance_project(left=shell, right=unity, weight=dchi)
    mean = mean[:, 0]
    poisson = np.diag(mean)

    # A positive shell variance multiplies the outer product of the SAME
    # response vector. Every cross-bin term is retained, including bins
    # whose true-redshift distributions overlap despite distinct labels.
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
