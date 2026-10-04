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
        accuracy_boost = 1, 2, 4 or 8. These nested resolutions fit the
            supported C Gaussian-quadrature rules without rounding a
            requested value down. One is the inexpensive notebook pilot.
    Returns:
        dict with the boost and integer ell_max, mask_ell_max,
        radial_nquad, angle_nquad and nwindow, plus halo_mass_nquad,
        tree_nquad, tree_npanel and response_step. Passing resolved values
        to C does not modify the data-vector accuracy configuration.

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
    not supply the missing full survey SSC/cNG physics.
    """
    if not isinstance(accuracy_boost, (int, np.integer)):
        raise ValueError("accuracy_boost must be one of the integers 1, 2, 4, 8")
    if accuracy_boost not in (1, 2, 4, 8):
        raise ValueError("accuracy_boost must be 1, 2, 4 or 8")
    boost = int(accuracy_boost)
    return {
        "accuracy_boost": boost,
        "ell_max": 10000*boost,
        "mask_ell_max": 4096*boost,
        "radial_nquad": 64*boost,
        "angle_nquad": 128*boost,
        "nwindow": 4096*boost+1,
        "halo_mass_nquad": 128*boost,
        "tree_nquad": 64*boost,
        "tree_npanel": 16+int(np.log2(boost)),
        "response_step": 0.01/boost,
    }
