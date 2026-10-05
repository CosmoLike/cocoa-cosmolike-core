"""Independent line-of-sight projection of supplied numeric covariance inputs.

Arrays have no dependency on Cocoa or its compiled libraries. cNG uses
Takada & Hu (2013), Appendix A, with dchi/f_K^6/Omega. SSC uses the
functional derivative of the projected, mean-normalized estimator.
This first reference evaluates supplied multipoles, not real-space bins.
"""

import numpy as np


def project_ng(geometry, windows, trispectrum, area):
    """Return the five separately projected cNG contributions.

    Arguments:
        geometry [4,nnode] = a,chi,f_K,dchi in one common length unit.
        windows [nell,nfield,nnode] = complete short-mode field weights.
        trispectrum [5,nnode,nell,nell] has length^9.
        area = survey solid angle in steradians.
    Returns:
        dict with cNG [5,nrow,nrow] and pair IDs; rows order field pair,
        then ell. No clipping occurs.

    This is a direct quadrature of the four field windows times T/f_K^6.
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

    Arguments: geometry/windows/signal as above, catalog mean_windows
        [nfield,nnode], response [nchoice,nnode,nell], sigma2 [nnode].
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
    """Gaussian covariance at selected ell, with independent catalog noise.

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
