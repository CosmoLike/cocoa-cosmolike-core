"""Independent set-partition check of the halo trispectrum contributions.

The connected non-Gaussian (cNG) covariance needs the halo-model
trispectrum T(K,-K,Q,-Q). Its four density legs are grouped into one to
four halos: 1h, 2h(1+3), 2h(2+2), 3h and 4h. Different halos are linked
by tree-level perturbation theory: P between two halos, the tree
bispectrum among three and the tree trispectrum among four.

This reference starts with four labelled density legs and enumerates their
halo partitions. It uses the explicit tree diagrams in cng_reference,
not the five reduced expressions implemented by the production assembler
in non_gaussian_cov.c.
"""

from itertools import combinations

import numpy as np

def partition_terms(first_k, second_k, power, count, volume, bias,
                    profile_k, profile_q, tree_reference, nquad=128):
    """Average explicit halo partitions for a discrete mass population.

    At each in-plane angle theta between K and Q the legs are (K,-K,Q,-Q).
    Every labelled partition of the legs into halos is summed, and the sum
    is averaged over theta with the dtheta/pi weights of
    tree_reference.angle_rule on one panel, which suits moderate
    wavenumbers. A partition with exactly zero internal momentum is the
    long-mode channel that belongs to the super-sample covariance (SSC);
    it is omitted, as in the tree diagrams of cng_reference.

    Arguments:
        first_k, second_k = wavenumbers K and Q, positive, in 1/length.
        power = scalar callable P(k), in length^3.
        count = [nmass] halos per unit volume in each mass sample, the
            dn/dlnM dlnM of that node, in length^-3.
        volume = [nmass] M/rho of each sample, in length^3.
        bias = [nmass] linear halo bias.
        profile_k, profile_q = [nmass] Fourier profiles u(K|M), u(Q|M).
        tree_reference = the cng_reference module, supplied explicitly.
        nquad = Gauss-Legendre nodes over (0,pi).
    Returns: float64 [5], the 1h, 2h(1+3), 2h(2+2), 3h and 4h terms in
        length^9, in the term order of non_gaussian_cov.c.
    Discrete mass samples make the moment definition exact for this test;
    they do not specify a cosmological mass-function calibration.
    """
    profiles = np.array([profile_k, profile_k, profile_q, profile_q])

    def moment(legs, biased):
        """Integrate one halo's labelled legs over the supplied population."""
        integrand = count*volume**len(legs)
        if biased:
            integrand = integrand*bias
        for leg in legs:
            integrand = integrand*profiles[leg]
        return np.sum(integrand)

    theta, weights, unused = tree_reference.angle_rule(nquad=nquad, npanel=1)
    result = np.zeros(5)
    for angle, weight in zip(theta, weights):
        first = np.array([first_k, 0.0])
        second = second_k*np.array([np.cos(angle), np.sin(angle)])
        legs = np.array([first, -first, second, -second])
        terms = np.zeros(5)
        # 1h: all four legs in one halo, with no bias factor.
        terms[0] = moment(legs=[0, 1, 2, 3], biased=False)
        # 2h(1+3): one isolated leg and the other three in a second halo,
        # linked by P at the isolated leg's wavenumber.
        for isolated in range(4):
            triple = []
            for index in range(4):
                if index != isolated:
                    triple.append(index)
            terms[1] += (power(np.linalg.norm(legs[isolated]))
                         *moment(legs=[isolated], biased=True)
                         *moment(legs=triple, biased=True))
        # 2h(2+2): fix leg 0 in the first pair, giving exactly the three
        # unlabelled partitions {01|23}, {02|13}, {03|12}, without double
        # counting. {01|23} has zero internal momentum: the SSC channel.
        for partner in (1, 2, 3):
            left = [0, partner]
            right = []
            for index in range(4):
                if index not in left:
                    right.append(index)
            internal = legs[0]+legs[partner]
            norm = np.linalg.norm(internal)
            if norm != 0.0:
                terms[2] += (power(norm)*moment(legs=left, biased=True)
                             *moment(legs=right, biased=True))
        # 3h: combinations gives the six unordered leg pairs. A pair shares
        # one halo and each remaining leg has its own; the tree bispectrum
        # of (pair sum, single, single) links them. A zero pair sum is the
        # SSC channel again.
        for pair in combinations(range(4), 2):
            singles = []
            for index in range(4):
                if index not in pair:
                    singles.append(index)
            internal = legs[pair[0]]+legs[pair[1]]
            if np.dot(internal, internal) == 0.0:
                continue
            triangle = np.array([internal, legs[singles[0]], legs[singles[1]]])
            terms[3] += (moment(legs=pair, biased=True)
                         *moment(legs=[singles[0]], biased=True)
                         *moment(legs=[singles[1]], biased=True)
                         *tree_reference.tree_bispectrum(vectors=triangle, power=power))
        # 4h: one leg per halo, linked by the tree trispectrum.
        product = 1.0
        for leg in range(4):
            product *= moment(legs=[leg], biased=True)
        terms[4] = product*tree_reference.tree_trispectrum(vectors=legs, power=power)
        result += weight*terms
    return result
