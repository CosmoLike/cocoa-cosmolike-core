"""Prepare physical inputs for the covariance-owned halo components.

Two covariance terms use the halo model: the matter trispectrum of the
connected non-Gaussian part (cNG) and the response of the matter power to
a long density mode, used by the super-sample part (SSC). Both combine the
halo moments

    I_bp(k_1,...,k_p) = integral dlnM (dn/dlnM) b_b(M) (M/rho_cb)^p
                        u(k_1|M)...u(k_p|M),

with b_0 = 1, b_1 the linear halo bias, u(k|M) the halo profile normalized
to u(0|M) = 1, and rho_cb the mean density of cold dark matter plus baryons
(cb). The names I11, I02, I12, I13 and I04 give the bias order b and then
the profile count p. A low-mass completion fixes I11(0) = 1.

The interface supplies halo fits and matter spectra. These helpers arrange
those inputs into triangular wavenumber pairs, angular quadratures and
power responses. The numerical integration runs in C with OpenMP/SIMDe.
The combined matter prescription here requires massless neutrinos: using
cb halo moments with total-matter response formulas needs a separate model
when neutrinos are massive. No such conversion is assumed here.
"""

import numpy as np

from .geometry import angular_rule
from .accuracy import covariance_accuracy


def halo_mass_edges():
    """Return the default matter-halo integration panels in natural log mass.

    The range is 10^-40 to 10^17 Msun/h. Below 10^4, eleven four-decade
    panels supply partial integrals for Wynn epsilon extrapolation of I11.
    The C code extrapolates only when the first twelve edges match this
    tail, and never extrapolates the higher moments. The tail is a
    numerical continuation of the halo fits, not a calibrated model of
    halos at such small masses. A residual completion fixes I11(0) = 1.
    Above 10^4, two one-decade panels reach 10^6 and eight equal panels in
    ln(M) reach 10^17.

    Returns:
        Owned float array [22] of ln(M/[Msun/h]) edges for 21 panels.
        Integration accuracy selects the GSL rule inside each panel;
        neither accuracy control changes these physical mass boundaries.
    """
    tail = np.arange(start=-40.0, stop=4.0, step=4.0)
    lower = np.log(10.0)*np.concatenate((tail, [4.0, 5.0]))
    lower[0] = np.log(1.e-40)  # match the C reader's lower-domain boundary
    upper = np.linspace(start=np.log(1.e6), stop=np.log(1.e17), num=9)
    return np.concatenate((lower, upper))


def halo_trispectrum(interface, a, k, lnm_edges, accuracy_boost, mnu,
                     integration_accuracy=0):
    """Compute five halo trispectrum contributions for all unordered k pairs.

    Arguments:
        interface = initialized project interface.
        a = one scale factor inside the core's supported interval.
        k = positive [nk] wavenumbers in inverse c/H0 units.
        lnm_edges = increasing ln(M/[Msun/h]) panel edges, normally
            halo_mass_edges().
        accuracy_boost = 1, 2, 4 or 8; validated only, since the mass and
            angle rules used here do not depend on it.
        mnu = neutrino mass of the initialized cosmology, eV; must be zero.
        integration_accuracy = 0..4, selecting the precomputed GSL mass and
            angle rules and the 20+level angle panels.
    Returns:
        dict with first/second k indices and terms [5,nk*(nk+1)/2].
        Term order is 1h, 2h(13), 2h(22), 3h, 4h; units are (c/H0)^9.
        Each pair appears once, including the diagonal.
    Raises:
        ValueError for nonzero mnu, a k that is not a nonempty positive 1D
        array, or an invalid accuracy control.
    """
    if mnu != 0.0:
        raise ValueError("combined halo matter trispectrum requires mnu=0")
    wave = np.ascontiguousarray(k, dtype=float)
    if wave.ndim != 1 or len(wave) == 0 or np.any(wave <= 0):
        raise ValueError("k must be a nonempty positive 1D array")
    accuracy = covariance_accuracy(
        accuracy_boost=accuracy_boost, integration_accuracy=integration_accuracy,
    )
    theta, weight, corner = angular_rule(
        nquad=accuracy["tree_nquad"], npanel=accuracy["tree_npanel"],
        interface=interface,
    )
    single, moments = interface.covariance_halo_moments(
        a=np.array([a], dtype=float),
        k=wave[None, :],
        lnm_edges=np.ascontiguousarray(lnm_edges, dtype=float),
        nquad=accuracy["halo_mass_nquad"],
    )
    # np.triu_indices lists the pairs (i, j) with i <= j row by row: the
    # i-major upper-triangle order of the pair axis of the C moments.
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
    # Moment roles are I02, I12, I13(K,Q,Q), I13(K,K,Q) and I04. Axis 1 of
    # moments and axis 0 of single index the one requested scale factor.
    terms = interface.covariance_halo_trispectrum(
        pk=pk,
        i11=np.array([single[0, first], single[0, second]]),
        moments=np.ascontiguousarray(moments[:, 0, :]),
        tree=angular,
    )
    return {"first": first, "second": second, "terms": terms}


def halo_power_response(interface, a, k, lnm_edges, accuracy_boost, mnu,
                       integration_accuracy=0):
    """Compute the isotropic fractional-halo response transferred to Pdelta.

    Arguments:
        interface = initialized project interface.
        a = [na] scale factors; k = [na,nk] positive wavenumbers in inverse
            c/H0 units, one row per scale factor.
        lnm_edges = mass panels as in halo_trispectrum.
        accuracy_boost = 1, 2, 4 or 8; divides the centered derivative step.
        mnu = initialized neutrino mass in eV; this prescription needs zero.
        integration_accuracy = 0..4, selecting the precomputed mass rule.
    Returns:
        [na,nk] dimensional dP/d(delta_b), in (c/H0)^3. P is Pdelta, the
        configured nonlinear matter power; delta_b is the long density mode.
    Raises:
        ValueError for nonzero mnu, a not 1D, k not [na,nk], k <= 0, or an
        invalid accuracy control.

    The Takada--Hu isotropic prescription, with P_2h = I11^2 P_linear, is

        D_halo = (47/21 - (1/3) dlnP_2h/dlnk) P_2h + I12(k,k),
        P_halo = P_2h + I02(k,k).

    The constant 47/21 (growth_coefficient) collects growth (26/21), the
    reference-density term (2) and the -1 from dilating k^3. The dilation
    coefficient 1/3 multiplies the slope of P_2h, without the k^3 factor.
    The result is D_halo/P_halo times the target nonlinear power. This is a
    specified halo approximation, not a calibrated nonlinear/tidal
    response. Refine both the mass rule and the finite-difference step
    before survey inference.
    """
    if mnu != 0.0:
        raise ValueError("combined halo matter response requires mnu=0")
    accuracy = covariance_accuracy(
        accuracy_boost=accuracy_boost, integration_accuracy=integration_accuracy,
    )
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

    # P_2h = I11^2 P_lin at the low, central and high samples. Its slope
    # dlnP_2h/dlnk is the centered difference across 2*step in ln(k).
    two_halo = single**2*linear
    slope = np.log(two_halo[:, 2]/two_halo[:, 0])/(2.0*step)
    # Triangular pairs for three samples are 00,01,02,11,12,22. Pair 11
    # (column 3) holds the central one-halo power I02(k,k) in role 0 and
    # its response I12(k,k) in role 1. The rows below follow the C order
    # P_lin, P_target, I11, I02, I12, slope.
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
