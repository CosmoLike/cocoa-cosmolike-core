"""Independent mask and SSC response algebra, using NumPy and supplied inputs.

No compiled project interface is imported. The long-mode Limber covariance
is Takada & Hu (2013), arXiv:1302.6994, Eq. 54. A supplied general radial
kernel permits correlations between shells; constructing that kernel with
non-Limber density/tidal responses remains a separate physics calculation.
"""

import numpy as np


def cap_mask(area_sr, ell_max):
    """Return raw spherical-cap C_L, with C_0=area_sr^2/(4*pi).

    Arguments: area_sr in (0,4*pi], ell_max >= 0.
    Returns: float64 [ell_max+1]; no C_0 or mask-variance normalization.
    The analytic integral of P_L over cos(theta) gives w_L0; all M!=0
    vanish when the cap is centered on the polar axis.
    """
    if not 0 < area_sr <= 4*np.pi or ell_max < 0:
        raise ValueError("require a positive sky area <= 4*pi and ell_max >= 0")
    edge = 1-area_sr/(2*np.pi)
    polynomial = np.empty(ell_max+2)
    polynomial[0] = 1.0
    polynomial[1] = edge
    for ell in range(1, ell_max+1):
        polynomial[ell+1] = ((2*ell+1)*edge*polynomial[ell]
                             -ell*polynomial[ell-1])/(ell+1)
    integral = np.empty(ell_max+1)
    integral[0] = area_sr/(2*np.pi)
    for ell in range(1, ell_max+1):
        integral[ell] = (polynomial[ell-1]-polynomial[ell+1])/(2*ell+1)
    return np.pi*integral**2


def mask_variance(mask_cl, area_sr, distance, power):
    """Return the Limber background strength in units of length.

    Arguments: raw mask_cl [nmask], area_sr, distance [nnode], and
        power [nnode,nmask] in length^3, at k=(L+1/2)/distance.
    Returns: [nnode] sigma_b^2, without a radial quadrature weight.
    """
    ell = np.arange(len(mask_cl))
    weights = (2*ell+1)*mask_cl/area_sr**2
    return np.einsum("pl,l->p", power, weights)/distance**2


def shell_response(distance, signal, pair_window, mean_window, response_power):
    """Differentiate the projected, survey-mean-normalized spectrum.

    Arguments: distance [nnode]; signal [nrow]; pair_window W_A*W_B,
        mean_window U_A+U_B, and response_power dP/d(delta_b), each
        [nrow,nnode]. Units respectively length, 1, length^-2,
        length^-1, length^3.
    Returns: Phi [nrow,nnode], in inverse length, without radial weights.
    """
    local = pair_window*response_power/distance**2
    return local-mean_window*signal[:, None]


def project_limber(response, radial_weights, sigma2):
    """Integrate Phi_i Phi_j dchi sigma_b^2 without a C-like loop order.

    Arguments: response [nrow,nnode], radial_weights and sigma2 [nnode].
    Returns: dimensionless covariance [nrow,nrow]. Signed Phi is allowed.
    """
    return np.einsum("ip,p,p,jp->ij", response, radial_weights, sigma2,
                     response, optimize=False)


def project_correlated(response, radial_weights, background):
    """Project a general dimensionless inter-shell covariance K(chi,chi').

    Arguments: response [nrow,nnode], radial_weights [nnode], background
        [nnode,nnode]. K includes no radial quadrature weights.
    Returns: covariance [nrow,nrow], using two radial integrals.
    For Limber K_pp = sigma_b^2(p)/dchi_p, which removes one integration.
    """
    weighted = response*radial_weights
    return weighted @ background @ weighted.T
