"""Independent set-partition check of the halo trispectrum contributions.

This reference starts with four labelled density legs and enumerates their
halo partitions. It uses the explicit tree diagrams in cng_reference,
not the five reduced expressions implemented by the production assembler.
"""

from itertools import combinations

import numpy as np

def partition_terms(first_k, second_k, power, count, volume, bias,
                    profile_k, profile_q, tree_reference, nquad=128):
    """Average explicit halo partitions for a discrete mass population.

    Arguments: K,Q, scalar callable P; count, volume=M/rho, bias and two
        profiles, all [nmass]; explicit cng_reference module; node count nquad.
    Returns: five halo contributions, from one through four halos.
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
        terms[0] = moment(legs=[0, 1, 2, 3], biased=False)
        for isolated in range(4):
            triple = []
            for index in range(4):
                if index != isolated:
                    triple.append(index)
            terms[1] += (power(np.linalg.norm(legs[isolated]))
                         *moment(legs=[isolated], biased=True)
                         *moment(legs=triple, biased=True))
        # Fix leg 0 in the first pair, giving exactly the three unlabelled
        # partitions {01|23}, {02|13}, {03|12}, without double counting.
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
        product = 1.0
        for leg in range(4):
            product *= moment(legs=[leg], biased=True)
        terms[4] = product*tree_reference.tree_trispectrum(vectors=legs, power=power)
        result += weight*terms
    return result
