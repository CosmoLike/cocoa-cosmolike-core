"""One covariance accuracy control with inspectable numerical settings.

The boost changes covariance integration only. It does not alter CAMB,
CosmoLike's data-vector tables, the physical covariance model, or the
survey footprint. Higher resolution reduces numerical errors only within
the specified model; convergence still needs to be measured.
"""

import numpy as np


def covariance_accuracy(accuracy_boost=1):
    """Resolve a single boost into the covariance integration controls.

    Arguments:
        accuracy_boost = 1, 2, 4 or 8. These increasing resolutions fit the
            supported C Gaussian-quadrature rules without rounding a
            requested value down. One is the inexpensive notebook pilot.
    Returns:
        dict with the boost and integer ell_max, mask_ell_max,
        radial_nquad, angle_nquad and nwindow, plus halo_mass_nquad,
        tree_nquad, tree_npanel, response_step and the ng_ell sample array.
        Passing resolved values to C does not modify data-vector accuracy.

    Why all five grow:
        ell_max retains more small-angle signal modes; mask_ell_max
        resolves finer footprint structure; radial_nquad improves the
        line-of-sight integral; angle_nquad resolves oscillations inside
        angular bins; nwindow refines the lensing-efficiency integral.
        Doubling intervals (nwindow-1) retains all old window-grid nodes.

    Boost 8 reaches 80,000 signal multipoles, 32,768 mask multipoles,
    512 radial nodes per panel, 1,024 angular nodes per bin and 32,769
    window nodes. This can be expensive. It is a numerical stress setting,
    not a promise of converged Fisher errors for an arbitrary survey.
    The same boost controls standalone halo preparation: more mass and
    angular nodes, an extra near-opposite angular panel per doubling, and
    a smaller centered derivative step. These changes require refinement
    checks against the underlying power-spectrum interpolation. They do
    not certify the full survey SSC/cNG model. The same boost increases
    ng_ell, the samples used to interpolate the matter trispectrum in
    ln(ell+1/2). Each doubling inserts midpoint samples without moving old
    ones. Extending the signal cutoff adds intervals to that same grid.
    Scientific Fourier-band endpoints remain fixed. Gaussian quadrature
    nodes are not interpolation-table nodes and do not follow this nesting.
    """
    if not isinstance(accuracy_boost, (int, np.integer)):
        raise ValueError("accuracy_boost must be one of the integers 1, 2, 4, 8")
    if accuracy_boost not in (1, 2, 4, 8):
        raise ValueError("accuracy_boost must be 1, 2, 4 or 8")
    boost = int(accuracy_boost)

    # The pilot has 15 logarithmic intervals from ell=2 to ell=10000.
    # Doubling the number of INTERVALS puts each old node at an even index.
    # Doubling a point count instead would displace all interior samples.
    span = np.log(10000.5)-np.log(2.5)
    required_span = np.log(10000*boost+0.5)-np.log(2.5)
    base_intervals = int(np.ceil(15*required_span/span))
    position = np.arange(base_intervals*boost+1)/(15*boost)
    ng_ell = np.exp(np.log(2.5)+span*position)-0.5

    # Fix the two physical anchors exactly, including on refined grids.
    # The upper table edge can lie beyond the signal cutoff: it brackets
    # the last requested mode without stretching any existing interval.
    ng_ell[0] = 2.0
    ng_ell[15*boost] = 10000.0
    return {
        "accuracy_boost": boost,
        "ell_max": 10000*boost,
        "mask_ell_max": 4096*boost,
        "ng_ell": ng_ell,
        "radial_nquad": 64*boost,
        "angle_nquad": 128*boost,
        "nwindow": 4096*boost+1,
        "halo_mass_nquad": 128*boost,
        "tree_nquad": 64*boost,
        "tree_npanel": 16+int(np.log2(boost)),
        "response_step": 0.01/boost,
    }


def non_gaussian_multipoles(samples, ell_max):
    """Retain only the supplied table nodes needed to bracket the signal.

    Arguments:
        samples = finite 1D array, uniform in ln(ell+1/2), starting at two
            and covering ell_max. covariance_accuracy supplies nested nodes.
        ell_max = last measured/integrated signal multipole, at least two.
    Returns:
        Owned float array containing the original nodes through the first
        node at or above ell_max. The endpoint is never moved to ell_max.
        Fourier bands can end below the real-space cutoff, so unused upper
        samples are removed before the expensive halo calculations.
    """
    grid = np.asarray(samples, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
            or grid[0] != 2.0 or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("ng_ell needs increasing finite samples starting at ell=2")
    if not np.isfinite(ell_max) or ell_max < 2 or grid[-1] < ell_max:
        raise ValueError("ng_ell must cover the signal range 2 <= ell <= ell_max")
    steps = np.diff(np.log(grid+0.5))
    if not np.allclose(steps, steps[0], rtol=1.e-12, atol=0.0):
        raise ValueError("ng_ell samples must be uniform in ln(ell+1/2)")
    count = max(2, np.count_nonzero(grid < ell_max)+1)
    return grid[:count].copy()
