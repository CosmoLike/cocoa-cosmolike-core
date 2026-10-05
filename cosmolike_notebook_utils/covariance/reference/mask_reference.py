"""Direct spherical-cap geometry, independent of the harmonic mask expansion."""

import mpmath as mp


def cap_pair_area(area_sr, lower, upper):
    """Ordered pair area in a separation bin, from a spherical lens at 60 digits.

    The two equal caps have angular radius R, centers separated by theta.
    Their overlap is two spherical sectors minus two spherical triangles.
    The cosine rule gives sin(C/2)=sin(theta/2)/sin(R) and
    cos(A)=cot(R)*tan(theta/2). Girard's triangle area then gives
    overlap=2*pi-2*C-4*cos(R)*A. Rearranging around the cap area avoids
    cancellation at theta=0. Rotating the displacement around either center
    contributes 2*pi*sin(theta) dtheta to the ordered pair measure.

    The formula here is for a cap smaller than a hemisphere and bins wholly
    inside 0<=theta<2*R; it does not extrapolate beyond the supported case.
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
            first_angle = mp.asin(mp.sin(theta/2)/sine_radius)
            second_angle = mp.asin(cosine_radius*mp.tan(theta/2)/sine_radius)
            overlap = area-4*first_angle+4*cosine_radius*second_angle
            return 2*mp.pi*mp.sin(theta)*overlap

        return float(mp.quad(integrand, [lower, upper]))
