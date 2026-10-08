"""Direct spherical-cap geometry, independent of the harmonic mask expansion.

mask_cov.c computes the ordered pair area of the survey footprint in each
angular bin from the raw mask power spectrum C_L, normalized so that
C_0=area^2/(4*pi), and the bin-averaged Legendre operator. This module
computes the same area for a spherical-cap footprint from spherical
geometry alone, with no mask harmonics, in 60-digit mpmath arithmetic.
The ordered pair area is in sr^2: multiplied by two number densities per
steradian, it gives the expected number of unclustered pairs.
"""

import mpmath as mp


def cap_pair_area(area_sr, lower, upper):
    """Return the ordered pair area of a spherical cap in one separation bin.

    Two copies of the cap, of angular radius R, have centers theta apart.
    Their overlap is two spherical sectors minus two spherical triangles.
    Take the right spherical triangle formed by one center, the midpoint
    between the centers and one crossing point of the two cap edges.
    Napier's rules give sin(C/2)=sin(theta/2)/sin(R), with C/2 the angle
    at the crossing point, and cos(A)=cot(R)*tan(theta/2), with A the
    angle at the center. Girard's triangle area then gives
    overlap=2*pi-2*C-4*cos(R)*A. The code evaluates
    overlap=area-2*C+4*cos(R)*(pi/2-A), with C/2 and pi/2-A as arcsines
    that vanish at theta=0, so it starts from the cap area without
    cancellation. Around one point, the partners at separation theta fill
    a ring of measure 2*pi*sin(theta) dtheta, and integrating the
    ring-averaged pair indicator over the cap gives overlap(theta). The
    ordered pair area is therefore the bin integral of
    2*pi*sin(theta)*overlap(theta). mask_cov.c evaluates the same bin
    integral in harmonic form, with overlap(theta) replaced by
    sum_L (2L+1) C_L P_L(cos(theta)).

    Arguments:
        area_sr = cap solid angle in steradians, 0 < area_sr < 2*pi.
        lower, upper = bin edges in radians, 0 <= lower < upper < 2*R.
    Returns:
        float, the ordered pair area in sr^2, from a 60-digit quadrature.
    Raises:
        ValueError for a cap of a hemisphere or more, or a bin not wholly
        inside 0 <= theta < 2*R. The formula here does not extrapolate
        beyond the supported case.
    """
    with mp.workdps(60):
        area = mp.mpf(float(area_sr))
        lower = mp.mpf(float(lower))
        upper = mp.mpf(float(upper))
        if not 0 < area < 2*mp.pi:
            raise ValueError("direct cap reference requires 0 < area < 2*pi")
        cosine_radius = 1-area/(2*mp.pi)
        radius = mp.acos(cosine_radius)
        sine_radius = mp.sin(radius)
        if not 0 <= lower < upper < 2*radius:
            raise ValueError("require 0 <= theta_low < theta_high < 2*cap_radius")

        def integrand(theta):
            """Return 2*pi*sin(theta) times the cap overlap at separation theta."""
            first_angle = mp.asin(mp.sin(theta/2)/sine_radius)
            second_angle = mp.asin(cosine_radius*mp.tan(theta/2)/sine_radius)
            overlap = area-4*first_angle+4*cosine_radius*second_angle
            return 2*mp.pi*mp.sin(theta)*overlap

        return float(mp.quad(integrand, [lower, upper]))
