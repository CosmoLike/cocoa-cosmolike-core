"""Independent line-of-sight projection of supplied numeric covariance inputs.

Three covariance terms are projected between data-vector rows: the
connected non-Gaussian (cNG) term, the super-sample covariance (SSC) and
the Gaussian term. A row is one field pair (i <= j, i-major, as
numpy.triu_indices orders them) at one multipole, with field pair as the
slow index and ell as the fast one.

The module imports only NumPy; it has no dependency on Cocoa or its
compiled libraries. cNG uses Takada & Hu (2013), Appendix A, with
dchi/f_K^6/Omega. SSC uses the functional derivative of the projected,
mean-normalized estimator. The functions evaluate supplied multipoles,
not real-space bins.
"""

import numpy as np


def project_ng(geometry, windows, trispectrum, area):
    """Return the five separately projected cNG contributions.

    Arguments:
        geometry [4,nnode] = a,chi,f_K,dchi in one common length unit.
        windows [nell,nfield,nnode] = complete short-mode field weights,
            in inverse length.
        trispectrum [5,nnode,nell,nell] has length^9; the five halo terms
            stay separate, in the supplied order.
        area = survey solid angle in steradians.
    Returns:
        dict with "cng" = float64 [5,nrow,nrow] and "pairs" = [npair,2]
        field IDs; rows order field pair, then ell, so nrow = npair*nell.
        No clipping occurs.

    This is a direct quadrature of the four field windows times T/f_K^6:
    each node adds dchi W_A W_B T W_C W_D/(area f_K^6), dimensionless.
    Catalog mean corrections enter SSC, not the connected trispectrum.
    """
    nfield = windows.shape[1]
    first, second = np.triu_indices(nfield)
    pair = windows[:, first, :]*windows[:, second, :]
    pair = np.transpose(pair, (1, 0, 2))  # [pair,ell,node]
    nrow = len(first)*windows.shape[0]
    result = np.zeros((5, nrow, nrow))
    for node, distance in enumerate(geometry[2]):
        measure = geometry[3, node]/(area*distance**6)
        for role in range(5):
            # ijkl means pair_i, ell_j, pair_k, ell_l. Only then flatten
            # into data-vector rows; the two multipole axes stay distinct.
            block = np.einsum("ij,jl,kl->ijkl", pair[:, :, node],
                              trispectrum[role, node], pair[:, :, node])
            result[role] += measure*block.reshape(nrow, nrow)
    return {"cng": result, "pairs": np.column_stack((first, second))}


def shell_responses(geometry, windows, signal, mean_windows, response,
                    sigma2):
    """Project supplied dimensional responses, preserving a factor form.

    Each row has the SSC response of ssc_reference.shell_response,
    Phi = W_A W_B (dP/d delta_b)/f_K^2 - (U_A+U_B) C_AB, and the
    covariance is F F^T with F = Phi*sqrt(dchi*s_b). The factor form keeps
    each matrix positive semidefinite; a negative dchi*s_b would give NaN
    through the square root.

    Arguments: geometry/windows as in project_ng; signal [nell,nfield,nfield]
        dimensionless spectra C_AB; catalog mean_windows U [nfield,nnode]
        in inverse length; response [nchoice,nnode,nell] dP/d(delta_b) in
        length^3; sigma2 [nnode] = shell weight s_b in length.
    Returns: SSC [nchoice,nrow,nrow] in field-pair-major row order.
    Means for source fields are zero; means for galaxy catalogs include
    their density and magnification kernels without the short-mode ell
    factors. Response choices remain separate, never blended.
    """
    first, second = np.triu_indices(windows.shape[1])
    pair = np.transpose(windows[:, first]*windows[:, second], (1, 0, 2))
    means = mean_windows[first]+mean_windows[second]
    angular = signal[:, first, second].T
    nrow = angular.size
    result = np.empty((len(response), nrow, nrow))
    for choice, dimensional in enumerate(response):
        shell = pair*dimensional.T[None, :, :]/geometry[2]**2
        shell -= means[:, None, :]*angular[:, :, None]
        factor = shell.reshape(nrow, -1)*np.sqrt(geometry[3]*sigma2)
        result[choice] = factor @ factor.T
    return result


def gaussian_selected(signal, noise, ell, area, mode_width=None):
    """Return the Gaussian covariance at selected ell, with independent noise.

    Rows (A,B) and (C,D) at one ell are linked by the Wick pairings
    [S_AC S_BD + S_AD S_BC]/modes, with S = signal + diag(noise) the total
    spectrum; different ell are uncorrelated, so each block between two
    field pairs is diagonal in ell.

    Arguments: signal [nell,nfield,nfield] dimensionless spectra; noise
        [nfield] white-noise power of each catalog, added to its auto
        spectrum; ell [nell] selected multipoles; area = survey solid angle
        in steradians; mode_width = None or [nell] multipoles per row.
    Returns: float64 [nrow,nrow] in field-pair-major row order.
    mode_width=None means one multipole, with (2ell+1) f_sky modes.
    Supplying widths is a center-of-band diagnostic only, not an exact
    band-averaged covariance. No real-space or final accuracy claim applies.
    """
    first, second = np.triu_indices(len(noise))
    total = signal+np.diag(noise)
    nell = len(ell)
    result = np.zeros((len(first)*nell, len(first)*nell))
    modes = (2*ell+1)*area/(4*np.pi)
    if mode_width is not None:
        modes = modes*mode_width
    for left, (a, b) in enumerate(zip(first, second)):
        for right, (c, d) in enumerate(zip(first, second)):
            values = (total[:, a, c]*total[:, b, d]
                      +total[:, a, d]*total[:, b, c])/modes
            row = left*nell+np.arange(nell)
            column = right*nell+np.arange(nell)
            result[row, column] = values
    return result
