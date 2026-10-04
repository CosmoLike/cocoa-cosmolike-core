"""Covariance positivity and numerical-refinement checks for notebooks.

No eigenvalue is clipped and no diagonal regularization is added. A failed
positivity check is information about the supplied matrix, not a request
to silently repair it. Covariance refinement is distinct from the
data-vector delta-chi-squared test.
"""

import numpy as np
from scipy.linalg import eigh


def _symmetric_matrix(matrix):
    """Validate a finite square matrix before using its symmetric eigenproblem."""
    values = np.asarray(a=matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("covariance must be a square 2D array")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("covariance must be nonempty and finite")
    scale = np.max(np.abs(values))
    difference = np.max(np.abs(values-values.T))
    if difference > 1.e-12*scale:
        raise ValueError(
            f"covariance is not symmetric: max difference {difference:g}, "
            f"matrix scale {scale:g}; check both matrix indices"
        )
    return values


def covariance_modes(matrix):
    """Report covariance and correlation eigenvalues without changing the input.

    Arguments:
        matrix: finite symmetric [ndata,ndata] total or component covariance.

    Returns:
        dict with diagonal/eigenvalue diagnostics. Correlation eigenvalues
        are available only when every diagonal is positive; otherwise that
        field is None. A negative individual cNG contribution need not be a
        failure, but a total covariance must give nonnegative variance to
        every linear combination. An invertible total must be positive
        definite. Raw largest eigenvalues do not rank cosmological information.
    """
    values = _symmetric_matrix(matrix=matrix)
    diagonal = np.diag(v=values)
    eigenvalues = np.linalg.eigvalsh(a=values)
    correlation_eigenvalues = None
    if np.all(diagonal > 0.0):
        normalization = np.sqrt(diagonal[:, None]*diagonal[None, :])
        correlation = values/normalization
        correlation_eigenvalues = np.linalg.eigvalsh(a=correlation)
    return {
        "minimum_diagonal": float(np.min(diagonal)),
        "positive_diagonal": bool(np.all(diagonal > 0.0)),
        "eigenvalues": eigenvalues,
        "correlation_eigenvalues": correlation_eigenvalues,
        "positive_definite": bool(eigenvalues[0] > 0.0),
    }


def compare_covariances(matrix, reference):
    """Measure variance changes in every direction relative to a reference.

    Arguments:
        matrix, reference: matching symmetric covariance matrices.
        The reference must be positive definite.

    Returns:
        dict with generalized eigenvalues and their maximum distance from
        one. For any vector v, these eigenvalues bound the ratio
        (v.T matrix v)/(v.T reference v). They avoid ranking modes solely
        by dimensionful covariance eigenvalues.

    This is a numerical refinement diagnostic. Parameter errors and Fisher
    Figures of Merit still require derivatives of the predicted data vector;
    use the shared Fisher helpers with those derivatives and each covariance.
    """
    values = _symmetric_matrix(matrix=matrix)
    baseline = _symmetric_matrix(matrix=reference)
    if values.shape != baseline.shape:
        raise ValueError("matrix and reference must have the same dimensions")
    eigenvalues = eigh(a=values, b=baseline, eigvals_only=True)
    return {
        "generalized_eigenvalues": eigenvalues,
        "maximum_fractional_variance_change": float(
            np.max(np.abs(eigenvalues-1.0))
        ),
    }

