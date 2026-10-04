"""Independent Gaussian covariance algebra for supplied angular spectra.

This module never imports a compiled project interface. It implements the
Gaussian four-point identity of Krause & Eifler (2017), arXiv:1601.05779,
Appendix A. Signal spectra and projection operators are supplied inputs:
these checks do not certify a cosmological spectrum or an angular kernel.

Noise powers use densities per steradian and ellipticity variance per
component. Array positions on the last axis are consecutive integer
multipoles. NumPy's widest float is used where available; on Apple arm64
it is still float64. Separate mpmath tests check extreme noise domination.
"""

import numpy as np


def harmonic_covariance(signal, noise, pairs, ell_min, fsky, include_nn):
    """Compute the covariance of every supplied pair against every other pair.

    Arguments:
        signal = symmetric [nfield, nfield, nell] angular signal spectra.
        noise = [nfield] nonnegative white-noise powers for independent fields.
        pairs = [npair, 2] integer field IDs, including cross-bin pairs.
        ell_min = nonnegative first integer multipole.
        fsky = survey area / (4 pi), strictly positive and at most one.
        include_nn = whether to retain the product of two noise powers.

    Returns:
        float64 [npair, npair, nell] harmonic covariance. Each final-axis
        entry is one ell, not a band average. No input is modified.
    """
    signal = np.asarray(signal, dtype=np.longdouble)
    noise = np.asarray(noise, dtype=np.longdouble)
    pairs = np.asarray(pairs, dtype=int)
    nfield = noise.size
    if signal.ndim != 3 or signal.shape[:2] != (nfield, nfield):
        raise ValueError("signal must have shape [len(noise), len(noise), nell]")
    if not 0.0 < fsky <= 1.0 or ell_min < 0 or signal.shape[2] < 1:
        raise ValueError("require 0 < fsky <= 1 and a nonempty integer ell grid")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [npair, 2]")
    if np.any(pairs < 0) or np.any(pairs >= nfield):
        raise ValueError("pairs contain an index outside the supplied field matrix")
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(noise)):
        raise ValueError("signal and noise must be finite")
    if np.any(noise < 0.0):
        raise ValueError("independent-field noise powers must be nonnegative")

    # Broadcasting adds the same diagonal noise matrix at every ell.
    # Its two field axes stay separate from the final multipole axis.
    noise_matrix = np.diag(noise)
    total = signal + noise_matrix[:, :, np.newaxis]
    ell = np.arange(signal.shape[2], dtype=np.longdouble) + ell_min
    modes = (2*ell + 1)*fsky
    result = np.empty((len(pairs), len(pairs), len(ell)), dtype=float)

    for left, (field_a, field_b) in enumerate(pairs):
        for right, (field_c, field_d) in enumerate(pairs):
            moment = total[field_a, field_c]*total[field_b, field_d]
            moment += total[field_a, field_d]*total[field_b, field_c]
            if not include_nn:
                noise_direct = (noise_matrix[field_a, field_c]
                                *noise_matrix[field_b, field_d])
                noise_exchanged = (noise_matrix[field_a, field_d]
                                   *noise_matrix[field_b, field_c])
                noise_moment = noise_direct + noise_exchanged
                moment -= noise_moment
            result[left, right] = moment/modes
    return result


def project_block(kernel_left, kernel_right, harmonic):
    """Project one diagonal harmonic covariance with two supplied operators.

    Arguments:
        kernel_left = [nleft, nell] normalized projection weights.
        kernel_right = [nright, nell] normalized projection weights.
        harmonic = [nell] harmonic covariance at the same integer nodes.

    Returns:
        float64 [nleft, nright] projected covariance, without modifying inputs.
        The contraction uses NumPy's independent reduction, with extra
        precision only on platforms where longdouble is wider than float64.
    """
    left = np.asarray(kernel_left, dtype=np.longdouble)
    right = np.asarray(kernel_right, dtype=np.longdouble)
    power = np.asarray(harmonic, dtype=np.longdouble)
    # i and j label observable rows; l labels the shared multipole axis.
    result = np.einsum("il,l,jl->ij", left, power, right, optimize=False)
    return np.asarray(result, dtype=float)


def spherical_annulus_kernel(edges_rad, ell_max):
    """Integrate scalar Legendre kernels over spherical angular bins.

    Arguments:
        edges_rad = strictly increasing 1D angular edges in [0, pi].
        ell_max = highest integer multipole, at least one.

    Returns:
        float64 [len(edges_rad)-1, ell_max+1], including the monopole.
        Each row is the solid-angle average of (2 ell+1) P_ell/(4 pi).

    The three-term polynomial recurrence and its exact antiderivative
    avoid copying the C data-vector kernels or using a flat-sky Bessel
    approximation. Including the monopole here permits an analytic
    completeness check; a data-vector operator may explicitly remove it.
    """
    edges = np.asarray(edges_rad, dtype=np.longdouble)
    if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges_rad must be a strictly increasing 1D array")
    if edges[0] < 0 or edges[-1] > np.pi or ell_max < 1:
        raise ValueError("require 0 <= angular edges <= pi and ell_max >= 1")
    cosine = np.cos(edges)
    polynomials = np.empty((ell_max+2, len(edges)), dtype=np.longdouble)
    polynomials[0] = 1
    polynomials[1] = cosine
    for ell in range(1, ell_max+1):
        polynomials[ell+1] = ((2*ell+1)*cosine*polynomials[ell]
                              - ell*polynomials[ell-1])/(ell+1)

    delta_cosine = cosine[:-1] - cosine[1:]
    kernels = np.empty((len(edges)-1, ell_max+1), dtype=np.longdouble)
    kernels[:, 0] = 1/(4*np.longdouble(np.pi))
    for ell in range(1, ell_max+1):
        primitive = polynomials[ell+1] - polynomials[ell-1]
        kernels[:, ell] = (primitive[:-1] - primitive[1:])/(4*np.pi*delta_cosine)
    return np.asarray(kernels, dtype=float)
