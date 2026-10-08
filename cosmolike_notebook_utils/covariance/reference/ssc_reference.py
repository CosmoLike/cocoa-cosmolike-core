"""Independent mask and SSC response algebra, using NumPy and supplied inputs.

Super-sample covariance (SSC) comes from density modes longer than the
survey: the survey-averaged background delta_b(chi) shifts every measured
spectrum together. The functions check the SSC steps of ssc_cov.c and the
projection that reuses gaussian_project_cov:

    cap_mask            raw mask spectrum C_L of a spherical cap
    mask_variance       shell weight s_b(chi), the sigma2 arguments below
    shell_response      response Phi_i(chi) of each spectrum row to delta_b
    project_limber      Cov_ij = int dchi s_b Phi_i Phi_j
    project_correlated  the same with a general inter-shell covariance

No compiled project interface is imported. The long-mode Limber covariance
is Takada & Hu (2013), arXiv:1302.6994, Eq. 54. A supplied general radial
kernel permits correlations between shells; constructing that kernel with
non-Limber density/tidal responses remains a separate physics calculation.
"""

import numpy as np


def cap_mask(area_sr, ell_max):
    """Return raw spherical-cap C_L, with C_0=area_sr^2/(4*pi).

    Arguments: area_sr in (0,4*pi], ell_max >= 0; other values raise
        ValueError.
    Returns: float64 [ell_max+1]; no C_0 or mask-variance normalization.
    The raw spectrum is C_L = sum_M |w_LM|^2/(2L+1) of the 0/1 footprint.
    All M!=0 vanish when the cap is centered on the polar axis, and
    w_L0 = 2*pi*sqrt((2L+1)/(4*pi))*J_L, with J_L the integral of P_L over
    cos(theta) from cos(R) to 1. The Legendre recurrence gives J_L in
    closed form, so C_L = pi*J_L^2.
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

    The shell weight s_b = sum_L (2L+1) C_L P((L+1/2)/f_K)/(area_sr*f_K)^2
    is the coefficient of the Dirac delta in the long-mode Limber
    covariance <delta_b(chi) delta_b(chi')> = s_b(chi) delta_D(chi-chi').
    The delta carries inverse length, so s_b is not the dimensionless
    variance of a finite shell.

    Arguments: raw mask_cl [nmask] with C_0=area_sr^2/(4*pi), area_sr in
        steradians, distance [nnode] = f_K, and power [nnode,nmask] in
        length^3, at k=(L+1/2)/distance.
    Returns: [nnode] s_b, without a radial quadrature weight.
    """
    ell = np.arange(len(mask_cl))
    weights = (2*ell+1)*mask_cl/area_sr**2
    return np.einsum("pl,l->p", power, weights)/distance**2


def shell_response(distance, signal, pair_window, mean_window, response_power):
    """Differentiate the projected, survey-mean-normalized spectrum.

    Phi_AB(chi) is defined by delta C_AB = int dchi Phi_AB delta_b(chi):

        Phi_AB = W_A W_B (dP/d delta_b)/f_K^2 - (U_A+U_B) C_AB.

    The first term is the change of the projected matter clustering. The
    second comes from the estimator's division by its catalog means, with
    delta nbar_A/nbar_A = int dchi U_A delta_b; U_A=0 for a field without
    a mean normalization.

    Arguments: distance [nnode] = f_K; signal [nrow] = C_AB of each row;
        pair_window W_A*W_B, mean_window U_A+U_B, and response_power
        dP/d(delta_b), each [nrow,nnode]. Units respectively length, 1,
        length^-2, length^-1, length^3.
    Returns: Phi [nrow,nnode], in inverse length, without radial weights.
    """
    local = pair_window*response_power/distance**2
    return local-mean_window*signal[:, None]


def project_limber(response, radial_weights, sigma2):
    """Integrate Phi_i Phi_j dchi s_b, the long-mode Limber SSC covariance.

    Arguments: response [nrow,nnode] in inverse length, radial_weights
        (dchi) and sigma2 (s_b) [nnode], both in length.
    Returns: dimensionless covariance [nrow,nrow]. Signed Phi is allowed.
    The contraction is NumPy's einsum, written independently of the C
    projection loop.
    """
    return np.einsum("ip,p,p,jp->ij", response, radial_weights, sigma2,
                     response, optimize=False)


def project_correlated(response, radial_weights, background):
    """Project a general dimensionless inter-shell covariance K(chi,chi').

    The result is sum_pq Phi_ip dchi_p K_pq dchi_q Phi_jq, with
    K_pq = <delta_b(chi_p) delta_b(chi_q)>.

    Arguments: response [nrow,nnode], radial_weights [nnode], background
        [nnode,nnode]. K includes no radial quadrature weights.
    Returns: covariance [nrow,nrow], using two radial integrals.
    For Limber K_pp = s_b(p)/dchi_p, the discrete form of
    s_b delta_D(chi-chi'), which removes one integration and gives
    project_limber.
    """
    weighted = response*radial_weights
    return weighted @ background @ weighted.T
