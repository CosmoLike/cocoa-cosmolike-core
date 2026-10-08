"""Survey geometry and integration inputs shared by covariance notebooks.

noise_powers turns catalog densities into shot and shape noise, cap_mask
gives the angular power of an idealized circular footprint, and
angular_rule builds the angle quadrature for the tree-level terms of the
halo trispectrum.

Angles are radians, solid angles are steradians, and number densities are
per square arcminute at the input boundary. The C routines use densities
per steradian. None of these helpers selects a survey's numerical accuracy.
"""

import numpy as np


def noise_powers(lens_density, source_density, sigma_component):
    """Return independent-catalog white noise, with lenses before sources.

    Arguments:
        lens_density: 1D number densities per square arcminute.
        source_density: 1D effective shape densities in the same units.
        sigma_component: 1D rms ellipticity of one shear component, not the
            two-component total, for each source bin.

    Returns:
        float64 vector [nlens+nsource] of N_g=1/n_g followed by
        N_s=sigma_component^2/n_s, in steradians (n per steradian).
        Catalog overlap in redshift does not imply shared objects. These
        powers assume disjoint object catalogs and uniform noise per bin.

    Raises:
        ValueError unless both densities are nonempty 1D arrays,
        sigma_component matches source_density, and every value is finite
        and positive.
    """
    lens = np.asarray(a=lens_density, dtype=float)
    source = np.asarray(a=source_density, dtype=float)
    dispersion = np.asarray(a=sigma_component, dtype=float)
    if lens.ndim != 1 or source.ndim != 1 or dispersion.shape != source.shape:
        raise ValueError("supply 1D densities and one dispersion per source bin")
    if len(lens) == 0 or len(source) == 0:
        raise ValueError("at least one lens and one source density are required")
    for values in (lens, source, dispersion):
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("densities and component dispersions must be positive")
    arcmin2_per_sr = (180.0*60.0/np.pi)**2
    result = np.empty(shape=len(lens)+len(source), dtype=float)
    result[:len(lens)] = 1.0/(lens*arcmin2_per_sr)
    result[len(lens):] = dispersion**2/(source*arcmin2_per_sr)
    return result


def cap_mask(area_sr, ell_max):
    """Return the raw angular mask power of a circular spherical cap.

    Arguments:
        area_sr: cap area in (0,4*pi], measured in steradians.
        ell_max: last mask multipole, including L=0 and L=1.

    Returns:
        float64 [ell_max+1] C_L in sr^2, with C_0=area_sr^2/(4*pi).

    Raises:
        ValueError for an area that is not finite and inside (0,4*pi], or
        for a negative or non-integer ell_max.

    A cap centered on the pole has only m=0 spherical-harmonic
    coefficients. Integrating P_L over its polar extent gives those
    coefficients analytically: with x_c = 1-area_sr/(2*pi), the cosine of
    the cap radius, the integral of P_L from x_c to 1 is
    (P_{L-1}(x_c)-P_{L+1}(x_c))/(2L+1) for L >= 1, and C_L is pi times
    its square. Rotating the cap changes its coefficients but not C_L.
    This is a chosen idealized footprint, not a reconstruction of a real
    survey mask. It follows the raw-mask convention used by ssc_cov.c and
    mask_cov.c.
    """
    if not np.isfinite(area_sr) or not 0 < area_sr <= 4*np.pi:
        raise ValueError("area_sr must be finite and inside (0,4*pi]")
    if not isinstance(ell_max, (int, np.integer)) or ell_max < 0:
        raise ValueError("ell_max must be a nonnegative integer")
    boundary = 1.0-area_sr/(2.0*np.pi)
    polynomial = np.empty(shape=ell_max+2, dtype=float)
    polynomial[0] = 1.0
    polynomial[1] = boundary

    # Successive Legendre values share one boundary angle. The recurrence
    # builds all degrees in linear time without constructing polynomials.
    for ell in range(1, ell_max+1):
        polynomial[ell+1] = (
            (2*ell+1)*boundary*polynomial[ell]-ell*polynomial[ell-1]
        )/(ell+1)

    integral = np.empty(shape=ell_max+1, dtype=float)
    integral[0] = area_sr/(2.0*np.pi)
    for ell in range(1, ell_max+1):
        integral[ell] = (polynomial[ell-1]-polynomial[ell+1])/(2*ell+1)
    return np.pi*integral**2


def angular_rule(nquad, npanel, interface):
    """Build a planar dtheta/pi rule resolving nearly opposite wavevectors.

    Arguments:
        nquad: precomputed GSL size: 64, 96, 128, 256, 512 or 1024.
        npanel: number of panels, between 1 and 40.
        interface: project binding exposing covariance_integration_rule.

    Returns:
        theta, weight, corner: 1D float64 arrays, each length nquad*npanel.
        theta is in radians. Weights sum to one; corner=1+cos(theta) is
        evaluated without subtracting nearly equal numbers near theta=pi.

    Raises:
        ValueError for an unsupported nquad, npanel outside 1..40, or a
        node that rounds to theta=pi.

    When K approximately equals Q, the internal wavenumber |K+Q| changes
    rapidly near theta=pi. Panel edges are pi-pi/2^j for j=0..npanel-1,
    then pi: widths halve toward that endpoint, and the last two panels
    share the finest width pi/2^(npanel-1). At the cap of 40 panels this
    width, about 6e-12 rad, still spans about 1e4 rounding steps of pi.
    The last panel reaches pi, but Gaussian nodes never touch the singular
    endpoint. Callers refine both counts rather than treating them as a
    universal production setting.
    """
    if not isinstance(nquad, (int, np.integer)) or nquad not in (
        64, 96, 128, 256, 512, 1024
    ):
        raise ValueError("nquad must be a precomputed GSL rule of at least 64 nodes")
    if not isinstance(npanel, (int, np.integer)) or not 1 <= npanel <= 40:
        raise ValueError("npanel must be an integer between 1 and 40")
    nodes, weights = interface.covariance_integration_rule(nquad=nquad)
    gaps = np.pi*2.0**(-np.arange(npanel))
    edges = np.append(arr=np.pi-gaps, values=np.pi)
    theta = np.empty(shape=nquad*npanel, dtype=float)
    measure = np.empty_like(prototype=theta)
    corner = np.empty_like(prototype=theta)

    # Map the same Gaussian rule onto each panel. Keep the angle and its
    # positive measure together so every tree diagram uses identical nodes.
    for panel in range(npanel):
        lower = edges[panel]
        width = edges[panel+1]-lower
        rows = slice(panel*nquad, (panel+1)*nquad)
        theta[rows] = lower+width*(nodes+1.0)/2.0
        measure[rows] = weights*width/(2.0*np.pi)
        corner[rows] = 2.0*np.sin((np.pi-theta[rows])/2.0)**2
    if np.any(corner <= 0.0):
        raise ValueError("angular nodes round to pi; reduce npanel or nquad")
    return theta, measure, corner
