"""Independent angular operators for unit-normalized observed fields.

An angular operator maps an angular power spectrum to a correlation
function averaged over one angular bin: xi(bin) = sum_ell K_ell C_ell.
For unit-normalized observed fields, K_ell is the small Wigner rotation
matrix d^ell_{m m'}(theta) times (2 ell+1)/(4 pi), averaged over the bin:
d^ell_22 for xi+, d^ell_2,-2 for xi-, d^ell_20 for gamma_t and
d^ell_00 = P_ell for w. operators_cov.c builds the same kernels with its
own Jacobi recurrence.

SciPy evaluates the Jacobi polynomials at independently generated Gaussian
nodes. The C recurrence is not imported. Low-degree factorial Wigner sums
provide a separate check of the spin convention and the Jacobi identities.
"""

import math

import mpmath as mp
import numpy as np
from scipy.special import eval_jacobi, eval_legendre


def angular_operator(edges, multipoles, nquad):
    """Area-average w, gamma_t and xi+/xi- kernels; rows are +, -, gt, w.

    Arguments:
        edges = [nbin+1] increasing bin edges in radians.
        multipoles = [nell] integer degrees ell.
        nquad = Gauss-Legendre nodes per bin.
    Returns: float64 [4, nbin, nell], including (2 ell+1)/(4 pi). Rows are
        xi+ (d^ell_22), xi- (d^ell_2,-2), gamma_t (d^ell_20) and w (P_ell);
        the spin-2 rows are zero for ell < 2.
    Each bin average uses sin(theta) dtheta / [cos(theta_low)-cos(theta_high)].
    It is numerical quadrature, not an exact endpoint formula: refine
    nquad for wide bins and high ell.
    """
    nodes, weights = np.polynomial.legendre.leggauss(nquad)
    result = np.zeros((4, len(edges)-1, len(multipoles)))
    # zip pairs each lower edge with the next one; enumerate adds the bin
    # index as row.
    for row, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        theta = (lower+upper)/2+(upper-lower)*nodes/2
        # cos(lower)-cos(upper) as a product of sines, without cancellation
        # in a narrow bin.
        width = 2*np.sin((lower+upper)/2)*np.sin((upper-lower)/2)
        measure = weights*(upper-lower)/2*np.sin(theta)/width
        cosine = np.cos(theta)
        sine_half = np.sin(theta/2)
        cosine_half = np.cos(theta/2)
        for index, ell in enumerate(multipoles):
            normalization = (2*ell+1)/(4*np.pi)
            result[3, row, index] = normalization*np.dot(
                measure, eval_legendre(ell, cosine)
            )
            if ell < 2:
                continue
            # Small Wigner d^ell_22, d^ell_2,-2 and d^ell_20 in Jacobi form,
            # with half-angle powers and polynomials P^(a,b)_{ell-2}.
            plus = cosine_half**4*eval_jacobi(ell-2, 0, 4, cosine)
            minus = sine_half**4*eval_jacobi(ell-2, 4, 0, cosine)
            tangential = sine_half**2*cosine_half**2
            tangential *= np.sqrt((ell+2)*(ell+1)/(ell*(ell-1)))
            tangential *= eval_jacobi(ell-2, 2, 2, cosine)
            result[0, row, index] = normalization*np.dot(measure, plus)
            result[1, row, index] = normalization*np.dot(measure, minus)
            result[2, row, index] = normalization*np.dot(measure, tangential)
    return result


def wigner_factorial(ell, first, second, theta):
    """Return the small Wigner d^ell_{first,second}(theta) from its factorial sum.

    Factorials are exact Python integers; every other operation is mpmath,
    at the working precision set by the caller (60 digits in
    high_precision_bin). The even index differences used here remove
    the sign ambiguity between the two common rotation conventions.
    The alternating terms grow quickly with ell and cancel, so this
    explicit sum serves low-degree checks only.

    Arguments: ell = integer degree; first, second = integer orders, each
        of magnitude at most ell; theta = angle in radians.
    Returns: mpmath real number.
    """
    factorial = math.factorial
    normalization = mp.sqrt(
        factorial(ell+first)*factorial(ell-first)
        *factorial(ell+second)*factorial(ell-second)
    )
    cosine = mp.cos(theta/2)
    sine = mp.sin(theta/2)
    total = mp.mpf(0)
    for index in range(max(0, second-first), min(ell+second, ell-first)+1):
        denominator = (factorial(ell+second-index)*factorial(index)
                       *factorial(first-second+index)*factorial(ell-first-index))
        term = (-1)**(first-second+index)*normalization/denominator
        term *= cosine**(2*ell+second-first-2*index)
        term *= sine**(first-second+2*index)
        total += term
    return total


def high_precision_bin(lower, upper, ell, first, second):
    """Integrate the factorial rotation matrix at 60 digits for one bin.

    Arguments: lower, upper = bin edges in radians; ell, first, second as
        in wigner_factorial.
    Returns: float, (2 ell+1)/(4 pi) times the average of
        d^ell_{first,second} with weight sin(theta) dtheta /
        [cos(lower)-cos(upper)]: one angular_operator entry, computed
        without SciPy or the C recurrence.
    """
    with mp.workdps(60):
        lower = mp.mpf(float(lower))
        upper = mp.mpf(float(upper))
        width = 2*mp.sin((upper+lower)/2)*mp.sin((upper-lower)/2)
        integral = mp.quad(
            lambda theta: mp.sin(theta)*wigner_factorial(
                ell=ell, first=first, second=second, theta=theta
            ), [lower, upper],
        )
        return float((2*ell+1)*integral/(4*mp.pi*width))
