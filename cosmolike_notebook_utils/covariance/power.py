"""Prepare dense power tables once, before the covariance C calculations."""

import numpy as np
from scipy.interpolate import CubicSpline


def refine_power_tables(tables, refinement=8):
    """Insert log-k samples into all three power tables without moving z nodes.

    Arguments:
        tables = CAMB arrays in the set_cosmology interchange format.
            log10k_2D is uniform; each flattened lnP array has redshift
            varying fastest, followed by wavenumber.
        refinement = positive integer number of intervals replacing each
            original interval. Eight turns 1,500 k nodes into 11,993.
    Returns:
        A new dictionary with dense linear, nonlinear and cb power arrays.
        Other inputs are shared unchanged. A factor of one returns a shallow
        copy. The supplied dictionary and arrays are never modified.

    Small interpolation errors in P can be amplified when the four-halo
    trispectrum subtracts large terms. At each fixed redshift, a natural
    cubic spline of ln(P) against log10(k) provides the inserted samples.
    Natural means that the spline's second derivative vanishes at each end.
    This preparation adds no new CAMB information or extrapolation range.

    The C kernels still use their ordinary fast linear lookups. All
    covariance components receive the same dense tables, rather than
    preparing a separate interpolation inside the four-halo calculation.
    Original samples are assigned explicitly to retain their exact values
    when the global boost divides each interval again.
    """
    if (isinstance(refinement, (bool, np.bool_))
            or not isinstance(refinement, (int, np.integer))
            or refinement < 1):
        raise ValueError("power refinement must be a positive integer")
    result = dict(tables)
    if refinement == 1:
        return result

    log10k = np.asarray(tables["log10k_2D"], dtype=float)
    if (log10k.ndim != 1 or log10k.size < 2
            or not np.all(np.isfinite(log10k))):
        raise ValueError("log10k_2D needs at least two finite samples")
    steps = np.diff(log10k)
    if steps[0] <= 0 or not np.allclose(steps, steps[0],
                                           rtol=1.e-10, atol=0.0):
        raise ValueError("log10k_2D must be increasing and uniform")
    nz = len(tables["z_2D"])
    nk = len(log10k)

    # Subdivide intervals, not the number of points: both endpoints count
    # only once. Copy the original anchors to avoid rounding them anew.
    dense = np.linspace(log10k[0], log10k[-1], refinement*(nk-1)+1)
    dense[::refinement] = log10k
    result["log10k_2D"] = dense

    # Each redshift row describes P(k) at one time. Splining along k only
    # leaves the original time sampling and the growth inputs unchanged.
    for name in ("lnP_linear", "lnP_nonlinear", "lnP_linear_cb"):
        values = np.asarray(tables[name], dtype=float)
        if values.shape != (nz*nk,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain nz*nk finite log powers")
        values = values.reshape((nz, nk), order="F")
        spline = CubicSpline(x=log10k, y=values, axis=1, bc_type="natural")
        refined = spline(dense)
        refined[:, ::refinement] = values
        result[name] = refined.ravel(order="F")
    return result
