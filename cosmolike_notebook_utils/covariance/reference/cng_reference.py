"""Independent tree-level cNG kernels and explicit Wick-contraction sums.

The connected non-Gaussian (cNG) covariance needs the tree-level
perturbation-theory power spectrum, bispectrum and trispectrum, averaged
over the in-plane angle theta between two wavevectors with weight
dtheta/pi (a planar average, not a 3D solid-angle one). The density
kernels F_n and velocity kernels G_n follow the Einstein-de Sitter (EdS)
recursion of Bernardeau et al. (2002), astro-ph/0112551,
Eqs. 43-44; the four-point tree diagrams are Takada & Hu (2013),
arXiv:1302.6994, Section III. No compiled project is imported.
The direct reference enumerates diagrams and six permutations rather than
using the reduced, cancellation-free angular formula of perturbation_cov.c.
reduced_averages is a NumPy form of that reduced formula; it is checked
against the explicit diagrams, since agreement with C alone would not be
an independent physics validation.
"""

from itertools import combinations, permutations

import numpy as np


def alpha(first, second):
    """Return the EdS mode coupling alpha(q1,q2) = (q1+q2).q1/|q1|^2.

    Both arguments are wavevectors; the first must not vanish.
    """
    return np.dot(first+second, first)/np.dot(first, first)


def beta(first, second):
    """Return the symmetric EdS beta(q1,q2) = |q1+q2|^2 q1.q2/(2|q1|^2|q2|^2).

    Neither wavevector may vanish.
    """
    total = first+second
    return (np.dot(total, total)*np.dot(first, second)
            /(2*np.dot(first, first)*np.dot(second, second)))


def second_order(first, second, velocity=False, symmetric=True):
    """Return F2 (density) or G2 (velocity) from the n=2 recursion.

    Arguments: two nonzero wavevectors; velocity selects G; symmetric
        averages the two input orders. Recursion uses unsymmetrized kernels.
    Returns: dimensionless scalar; no power spectrum is included.
    """
    coupling = alpha(first, second)
    if symmetric:
        coupling = 0.5*(coupling+alpha(second, first))
    # At n=2 the recursion denominator (2n+3)(n-1) is 7; F2 weights alpha
    # and beta by 2n+1=5 and 2, G2 by 3 and 2n=4.
    if velocity:
        return (3*coupling+4*beta(first, second))/7
    return (5*coupling+2*beta(first, second))/7


def third_order(vectors):
    """Symmetrize the full n=3 recursion over all six permutations.

    Arguments: three nonzero wavevectors, shape [3,dimension].
    Returns: symmetrized F3. An exactly opposite inner pair has zero
        contribution. For one order of the pair, the simple alpha/beta
        pole times an unsymmetrized F2/G2, which vanishes only linearly
        as the pair sum tends to zero, has a finite limit. The two orders,
        both present in the permutation average, add up to symmetrized
        F2/G2 that vanish quadratically, so their combined limit is zero.
        Both orders are omitted before any denominator is evaluated; no
        NaN is repaired.
    """
    total = 0.0
    # permutations yields the six orderings of the three vector indices.
    for order in permutations(range(3)):
        first, second, third = vectors[list(order)]
        right = second+third
        left = first+second
        value = 0.0
        # n=3 recursion with denominator (2n+3)(n-1) = 18 and 2n+1 = 7:
        # the m=1 term pairs G1(first)=1 with F2/G2(second, third), the
        # m=2 term pairs G2(first, second) with F1=G1=1 of third.
        if np.dot(right, right) != 0.0:
            f2 = second_order(first=second, second=third, symmetric=False)
            g2 = second_order(first=second, second=third, velocity=True,
                              symmetric=False)
            value += 7*alpha(first, right)*f2+2*beta(first, right)*g2
        if np.dot(left, left) != 0.0:
            g2 = second_order(first=first, second=second, velocity=True,
                              symmetric=False)
            value += g2*(7*alpha(left, third)+2*beta(left, third))
        total += value/18
    return total/6


def tree_bispectrum(vectors, power):
    """Sum the three tree-level bispectrum contractions for a closed triangle.

    Arguments: three wavevectors summing to zero; scalar callable power(k).
    Returns: B, with units of power squared.
    """
    result = 0.0
    for first, second in combinations(range(3), 2):
        pk = power(np.linalg.norm(vectors[first]))
        pq = power(np.linalg.norm(vectors[second]))
        result += 2*second_order(first=vectors[first], second=vectors[second])*pk*pq
    return result


def tree_trispectrum(vectors, power):
    """Enumerate the 4 third-order and 12 second-order Wick contractions.

    Arguments: four wavevectors summing to zero, scalar callable power(k).
    Returns: connected T, excluding the zero-internal-momentum channels
        whose finite survey-window limit belongs to SSC. Multiplicities
        are 3!=6 for a delta3 leg and 2*2=4 for two delta2 legs.
    """
    pk = np.empty(4)
    for index in range(4):
        pk[index] = power(np.linalg.norm(vectors[index]))
    result = 0.0
    for nonlinear in range(4):
        linear = []
        for index in range(4):
            if index != nonlinear:
                linear.append(index)
        result += 6*third_order(vectors=vectors[linear])*np.prod(pk[linear])

    # Two-delta2 diagrams: legs first and second are second order. Each
    # connects to one linear leg (left or right, in both assignments), and
    # the internal line carries first+left.
    for first, second in combinations(range(4), 2):
        linear = []
        for index in range(4):
            if index != first and index != second:
                linear.append(index)
        for left, right in permutations(linear):
            internal = vectors[first]+vectors[left]
            norm = np.linalg.norm(internal)
            if norm == 0.0:
                continue  # this exact zero-momentum channel belongs to SSC
            f_left = second_order(first=-vectors[left], second=internal)
            f_right = second_order(first=-vectors[right], second=-internal)
            result += 4*f_left*f_right*pk[left]*pk[right]*power(norm)
    return result


def angle_rule(nquad, npanel):
    """Return GL angles in (0,pi), with panels halving toward pi.

    GL is Gauss-Legendre. The panel edges are 0, pi/2, 3pi/4, ...,
    pi - pi/2^(npanel-1) and pi, so the last two panels have equal width;
    npanel=1 gives one panel over (0,pi).
    Arguments: positive nquad nodes per panel and npanel panels.
    Returns: theta, normalized dtheta/pi weights, and 1+cos(theta).
    The last quantity uses 2*sin((pi-theta)/2)^2 to retain accuracy in
    the squeezed corner where adding 1 to a cosine loses precision.
    """
    nodes, weights = np.polynomial.legendre.leggauss(nquad)
    gaps = np.pi*2.0**(-np.arange(npanel))
    edges = np.append(np.pi-gaps, np.pi)
    theta = []
    measure = []
    corner = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        width = upper-lower
        values = lower+width*(nodes+1)/2
        theta.extend(values)
        measure.extend(weights*width/(2*np.pi))
        corner.extend(2*np.sin((np.pi-values)/2)**2)
    return np.array(theta), np.array(measure), np.array(corner)


def paired_f3_average(paired_k, other_k):
    """Return the closed planar average of F3(k,-k,q); the order of K,Q matters.

    paired_k is K, the magnitude of the opposite pair (k,-k); other_k is
    Q. The average uses dtheta/pi in the plane. The two branches, for
    (Q/K)^2 <= 1 and > 1, meet at -4/63 when K=Q.
    """
    ratio2 = (other_k/paired_k)**2
    if ratio2 <= 1:
        return -ratio2*(9-ratio2)/126
    return -(21*ratio2-12+7/ratio2)/252


def reduced_averages(first_k, second_k, power, nquad=64, npanel=12):
    """Reduced planar P/B/T averages for comparison with explicit diagrams.

    Arguments: positive K,Q (first_k, second_k); scalar/vector callable
        power(k); GL rule sizes nquad and npanel, as in angle_rule.
    Returns: [<P(|k+q|)>, <B(k,q,-k-q)>, <T(k,-k,q,-q)>], averaged with
        dtheta/pi, in length^3, length^6 and length^9.
    This reduced algebra is checked against the explicit diagrams above.
    Agreement with C alone would not be an independent physics validation.
    """
    theta, weights, corner = angle_rule(nquad=nquad, npanel=npanel)
    pk = power(first_k)
    pq = power(second_k)
    # mu = cos(theta). s2 = |k+q|^2 = (K-Q)^2 + 2KQ(1+cos(theta)) avoids
    # subtracting nearly equal squares in the corner near theta=pi.
    mu = corner-1
    s2 = (first_k-second_k)**2+2*first_k*second_k*corner
    first_projection = (second_k-first_k)+first_k*corner
    second_projection = (first_k-second_k)+second_k*corner
    difference = (second_k-first_k)*(second_k+first_k)*(pq-pk)
    # bracket = F2(k,-k-q) P(K) + F2(q,-k-q) P(Q), with the two terms that
    # share the internal momentum k+q combined before evaluation.
    bracket = -(pk+pq)/28-mu*(first_k*pq/second_k+second_k*pk/first_k)/2
    bracket += ((2/7)*(first_projection**2*pq+second_projection**2*pk)
                -difference/4)/s2
    ps = power(np.sqrt(s2))
    average_p = np.dot(weights, ps)
    average_b = 12*pk*pq/7+2*np.dot(weights, ps*bracket)
    average_t = 12*paired_f3_average(paired_k=first_k, other_k=second_k)*pk**2*pq
    average_t += 12*paired_f3_average(paired_k=second_k, other_k=first_k)*pq**2*pk
    average_t += 8*np.dot(weights, ps*bracket**2)
    return np.array([average_p, average_b, average_t])


def direct_averages(first_k, second_k, power, nquad=96):
    """Integrate the explicit triangle and Wick sums on one single-panel GL rule.

    The rule is one Gauss-Legendre panel over (0,pi), without the graded
    panels of reduced_averages. This slower reference is intended for
    moderate k ratios, away from the very narrow high-k corner. Refinement
    is checked rather than assumed.

    Arguments: positive K,Q (first_k, second_k); scalar callable power(k);
        nquad nodes.
    Returns: [<P(|k+q|)>, <B(k,q,-k-q)>, <T(k,-k,q,-q)>], the same rows as
        reduced_averages.
    """
    theta, weights, unused = angle_rule(nquad=nquad, npanel=1)
    first = np.array([first_k, 0.0])
    result = np.zeros(3)
    for angle, weight in zip(theta, weights):
        second = second_k*np.array([np.cos(angle), np.sin(angle)])
        result[0] += weight*power(np.linalg.norm(first+second))
        result[1] += weight*tree_bispectrum(
            vectors=np.array([first, second, -first-second]), power=power
        )
        result[2] += weight*tree_trispectrum(
            vectors=np.array([first, -first, second, -second]), power=power
        )
    return result
