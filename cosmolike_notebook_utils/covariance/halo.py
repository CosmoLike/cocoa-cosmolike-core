"""Prepare physical inputs for the covariance-owned halo components.

The interface supplies halo fits and matter spectra. These helpers arrange
those inputs into triangular wavenumber pairs, angular quadratures and
power responses. The numerical integration remains in C with OpenMP/SIMDe.
The combined matter prescription here requires massless neutrinos: using
cb halo moments with total-matter response formulas needs a separate model
when neutrinos are massive. No such conversion is assumed here.
"""

import numpy as np

from .geometry import angular_rule
from .accuracy import covariance_accuracy


def halo_trispectrum(interface, a, k, lnm_edges, accuracy_boost, mnu):
    """Compute five halo trispectrum contributions for all unordered k pairs.

    Arguments:
        interface = initialized project interface.
        a = one scale factor inside the core's supported interval.
        k = positive [nk] wavenumbers in inverse c/H0 units.
        lnm_edges = increasing ln(M/[Msun/h]) panel edges.
        accuracy_boost = 1, 2, 4 or 8; refines mass and angular integration.
        mnu = neutrino mass of the initialized cosmology, eV; must be zero.
    Returns:
        dict with first/second k indices and terms [5,nk*(nk+1)/2].
        Term order is 1h, 2h(13), 2h(22), 3h, 4h; units are (c/H0)^9.
        Each pair appears once, including the diagonal.
    """
    if mnu != 0.0:
        raise ValueError("combined halo matter trispectrum requires mnu=0")
    wave = np.ascontiguousarray(k, dtype=float)
    if wave.ndim != 1 or len(wave) == 0 or np.any(wave <= 0):
        raise ValueError("k must be a nonempty positive 1D array")
    accuracy = covariance_accuracy(accuracy_boost=accuracy_boost)
    theta, weight, corner = angular_rule(
        nquad=accuracy["tree_nquad"], npanel=accuracy["tree_npanel"]
    )
    single, moments = interface.covariance_halo_moments(
        a=np.array([a], dtype=float),
        k=wave[None, :],
        lnm_edges=np.ascontiguousarray(lnm_edges, dtype=float),
        nquad=accuracy["halo_mass_nquad"],
    )
    first, second = np.triu_indices(n=len(wave))
    pairs = np.array([wave[first], wave[second]])
    linear = interface.covariance_power(a=a, k=wave, linear=True)
    pk = np.array([linear[first], linear[second]])

    # |K+Q|^2 = (K-Q)^2 + 2 K Q (1+cos(theta)). This form remains
    # accurate for nearly equal, nearly opposite vectors. All angles at
    # this redshift share a single batched power-spectrum read.
    magnitude = np.sqrt(
        (pairs[0, :, None]-pairs[1, :, None])**2
        +2.0*pairs[0, :, None]*pairs[1, :, None]*corner
    )
    internal = interface.covariance_power(
        a=a, k=magnitude, linear=True
    )
    angular = interface.covariance_tree_averages(
        k=pairs, pk=pk, corner=corner, weight=weight, ps=internal
    )
    terms = interface.covariance_halo_trispectrum(
        pk=pk,
        i11=np.array([single[0, first], single[0, second]]),
        moments=np.ascontiguousarray(moments[:, 0, :]),
        tree=angular,
    )
    return {"first": first, "second": second, "terms": terms}


def halo_power_response(interface, a, k, lnm_edges, accuracy_boost, mnu):
    """Compute the isotropic fractional-halo response transferred to Pdelta.

    Arguments:
        interface = initialized project interface.
        a = [na] scale factors; k = [na,nk] positive core wavenumbers.
        lnm_edges = mass panels as in halo_trispectrum.
        accuracy_boost = 1, 2, 4 or 8; raises mass resolution and decreases
            the centered finite-difference step in ln(k).
        mnu = initialized neutrino mass in eV; this prescription needs zero.
    Returns:
        [na,nk] dimensional dP/d(delta_b), in (c/H0)^3.

    The Takada--Hu isotropic prescription uses 47/21 for growth and 1/3
    for dilation of I11^2 P_linear. Divide the halo response by halo power,
    then multiply by the target nonlinear power. This is a specified halo
    approximation, not a calibrated nonlinear/tidal response. Refine both
    the mass rule and the finite-difference step before survey inference.
    """
    if mnu != 0.0:
        raise ValueError("combined halo matter response requires mnu=0")
    accuracy = covariance_accuracy(accuracy_boost=accuracy_boost)
    step = accuracy["response_step"]
    scale_factor = np.ascontiguousarray(a, dtype=float)
    wave = np.ascontiguousarray(k, dtype=float)
    if scale_factor.ndim != 1 or wave.ndim != 2:
        raise ValueError("a must be [na] and k must be [na,nk]")
    if wave.shape[0] != len(scale_factor) or np.any(wave <= 0):
        raise ValueError("k needs one positive-wavenumber row per a")

    # Treat each (a,k) point as one independent batch row with three
    # wavenumbers: the low, central and high samples. This avoids building
    # unused pair moments between different physical k values.
    shifted = wave.ravel()[:, None]*np.exp(np.array([-step, 0.0, step]))
    repeated_a = np.repeat(a=scale_factor, repeats=wave.shape[1])
    single, moments = interface.covariance_halo_moments(
        a=repeated_a,
        k=shifted,
        lnm_edges=np.ascontiguousarray(lnm_edges, dtype=float),
        nquad=accuracy["halo_mass_nquad"],
    )
    linear = np.empty_like(prototype=shifted)
    target = np.empty(shape=wave.size, dtype=float)
    for row, value in enumerate(scale_factor):
        rows = slice(row*wave.shape[1], (row+1)*wave.shape[1])
        linear[rows] = interface.covariance_power(
            a=float(value), k=shifted[rows].ravel(), linear=True
        ).reshape((wave.shape[1], 3))
        target[rows] = interface.covariance_power(
            a=float(value), k=wave[row], linear=False
        )

    two_halo = single**2*linear
    slope = np.log(two_halo[:, 2]/two_halo[:, 0])/(2.0*step)
    # Triangular pairs for three samples are 00,01,02,11,12,22.
    # Pair 11 (column 3) contains the central one-halo power and response.
    inputs = np.array([
        linear[:, 1],
        target,
        single[:, 1],
        moments[0, :, 3],
        moments[1, :, 3],
        slope,
    ])
    result = interface.covariance_halo_response(
        inputs=inputs,
        growth_coefficient=47.0/21.0,
        dilation_coefficient=1.0/3.0,
        fractional=True,
    )
    return result[1].reshape(wave.shape)
