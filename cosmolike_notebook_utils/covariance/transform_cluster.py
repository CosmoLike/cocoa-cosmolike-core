"""Propagate the cluster-lensing localization through a joint covariance.

Tangential shear depends on mass inside the measured radius. The Y
statistic of Park, Rozo & Krause (2021), arXiv:2004.07504, is
Y(R)=Sigma(R)-Sigma(R_max): a linear combination of angular bins that
removes that interior contribution. If y=T*x, its covariance is
T*C*T^t. A joint analysis must also transform the cross covariances
with counts, galaxy clustering and other lensing bins. Transforming
only the shear diagonal blocks would describe inconsistent observables.

The caller supplies the angular matrix from the project's existing mean
prediction and the cluster-lensing positions in the full vector. This
module chooses no finite-difference rule, selection factor or scale cut.
"""

import numpy as np


def localize_covariance(interface, covariance, indices, operator):
    """Apply one angular localization to every supplied cluster-lensing row.

    Write the joint transformation as A: it equals the identity on counts
    and other probes, and equals operator (T below) on each selected
    angular row.
    This routine computes A*C*A^t without allocating the large, mostly
    zero A matrix. Each multiplication uses the existing SIMDe/OpenMP
    projection, batching all selected rows into the same call.

    Arguments:
        interface = compiled project exposing covariance_project.
        covariance = finite square float [ndata,ndata] matrix. It may be
            any component or their sum; signed entries are allowed.
        indices = integer [nrow,ntheta] positions in the joint vector.
            Each row gives one cluster/source/richness combination in
            increasing angular-bin order. All positions must be distinct.
            Count entries may lie between the two-point blocks.
        operator = finite float [ntheta,ntheta] angular matrix, e.g. the
            project's get_cluster_ytransform_matrix() for the same angular
            bins.
    Returns:
        Owned [ndata,ndata] transformed covariance. Other-other entries
        are unchanged; selected-other and selected-selected blocks receive
        the transformation on one and two sides respectively. The input
        arrays and interface state are unchanged.
    Raises:
        ValueError for incompatible shapes, repeated/out-of-range indices,
        noninteger indices or nonfinite numerical inputs.

    The final Y row is exactly zero in the project's convention, because
    it measures Sigma(R_max)-Sigma(R_max). Its zero covariance mode is
    retained. Apply the likelihood's scale selection only after
    transforming the full unmasked matrix; do not clip modes or drop input
    angles here.
    """
    # --- 1. INPUT CHECKS ---

    matrix = np.asarray(a=covariance, dtype=float)
    positions = np.asarray(a=indices)
    transform = np.asarray(a=operator, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")

    # indices holds one row per cluster-lensing combination and one column
    # per angular bin; dtype.kind "iu" admits signed or unsigned integers.
    if positions.ndim != 2 or 0 in positions.shape:
        raise ValueError("indices must be nonempty [nrow,ntheta]")
    if positions.dtype.kind not in "iu":
        raise ValueError("indices must contain integer joint-vector positions")
    ndata = matrix.shape[0]
    nrow, ntheta = positions.shape
    if transform.shape != (ntheta, ntheta):
        raise ValueError("operator must match [ntheta,ntheta] from indices")

    # Positions must be in range and distinct: fancy-index assignment keeps
    # only the last write to a repeated position, so a reused position
    # would silently lose one transformed row.
    if np.any(positions < 0) or np.any(positions >= ndata):
        raise ValueError("indices must be inside the supplied covariance")
    if np.unique(ar=positions).size != positions.size:
        raise ValueError("indices must not reuse a joint-vector position")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(transform)):
        raise ValueError("covariance and operator must be finite")

    # --- 2. LEFT MULTIPLICATION: A*C CHANGES THE SELECTED ROWS ---

    # Only ntheta terms contribute to one transformed value.
    # covariance_project sums along the last axis of both inputs, so put
    # the angular axis last: one C batch then sums it for every
    # (cluster row, other observable) pair. Every C worker owns complete
    # output sums; the result does not depend on the number of workers.
    # matrix[positions] picks whole rows, [nrow,ntheta,ndata]; the transpose
    # and reshape give [nrow*ndata,ntheta]. Unit weights leave the plain
    # sum over the angular index j.
    left = np.ascontiguousarray(transform)
    weight = np.ones(shape=ntheta, dtype=float)
    right = np.ascontiguousarray(matrix[positions].transpose(0, 2, 1))
    right = right.reshape(nrow*ndata, ntheta)
    projected = interface.covariance_project(left=left, right=right,
                                             weight=weight)
    result = matrix.copy()
    # projected[i, row*ndata+other] = sum_j T[i,j] C[positions[row,j], other];
    # reorder it to [row, i, other] to overwrite the selected rows.
    result[positions] = projected.reshape(ntheta, nrow, ndata).transpose(1, 0, 2)

    # --- 3. RIGHT MULTIPLICATION: (A*C)*A^t CHANGES THE SELECTED COLUMNS ---

    # The first multiplication changed all selected rows, including their
    # count cross blocks. The second must read that intermediate result:
    # selected-selected blocks need T on both sides. The columns for
    # unselected observables keep their one-sided transformation.
    # result[:, positions] takes every row and the selected columns,
    # [ndata,nrow,ntheta]; its angular axis is already last.
    right = np.ascontiguousarray(result[:, positions])
    right = right.reshape(ndata*nrow, ntheta)
    projected = interface.covariance_project(left=left, right=right,
                                             weight=weight)
    # projected[i, other*nrow+row] = sum_j T[i,j] (A*C)[other, positions[row,j]];
    # its transpose, reshaped to [other, row, i], fills the selected columns.
    result[:, positions] = projected.T.reshape(ndata, nrow, ntheta)
    return result
