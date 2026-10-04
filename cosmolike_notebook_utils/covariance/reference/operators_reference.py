"""Independent angular operators for unit-normalized observed fields.

SciPy evaluates the Jacobi polynomials at independently generated Gaussian
nodes. The C recurrence is not imported. Low-degree factorial Wigner sums
provide a separate check of the spin convention and the Jacobi identities.
"""

import math

import mpmath as mp
import numpy as np
from scipy.special import eval_jacobi, eval_legendre


def angular_operator(edges, multipoles, nquad):
    """Area-average w, gamma_t and xi± kernels; rows are +, -, gt, w.

    edges are radians; multipoles are integer degrees. Each bin average
    uses sin(theta) dtheta / [cos(theta_low)-cos(theta_high)]. The result
    has shape [4, nbin, nell], including (2 ell+1)/(4 pi).
    """
    nodes, weights = np.polynomial.legendre.leggauss(nquad)
    result = np.zeros((4, len(edges)-1, len(multipoles)))
    for row, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        theta = (lower+upper)/2+(upper-lower)*nodes/2
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
    """Small Wigner d from its finite rotation-matrix factorial sum.

    All arithmetic is mpmath. The even index differences used here remove
    the sign ambiguity between the two common rotation conventions.
    This explicit sum is intended for low degrees, not the production loop.
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
    """Integrate the factorial rotation matrix at 60 digits for one bin."""
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
