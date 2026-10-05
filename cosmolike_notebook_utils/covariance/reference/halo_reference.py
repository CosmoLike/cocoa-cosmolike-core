"""Independent contractions of supplied halo abundances and profiles.

The physical inputs are explicit arrays; this module never imports a
compiled interface. Tests can share physical samples with C while using
different integration and pair-assembly algorithms. This checks their
contraction, not the calibration or internal sampling of the halo readers.
"""

import numpy as np


def moments(mass, dlnmass, number_density, bias, profile, profile_min, rho_cb):
    """Return I11 and all-pairs I02/I12/I13/I04 from sampled physical inputs.

    Arguments:
        mass, dlnmass = [nmass] mass abscissae and dlnM quadrature weights.
        number_density, bias = [na,nmass] dn/dlnM and linear cb halo bias.
        profile = [na,nk,nmass] normalized NFW Fourier profiles.
        profile_min = [na,nk] profile at the lower mass cutoff.
        rho_cb = cb mean density in mass / length^3.
    Returns: I11 [na,nk] and five moments [5,na,npair], with i-major pairs.
    Only I11 receives the missing low-mass completion on the supplied rule.
    """
    volume = mass/rho_cb
    count = number_density*dlnmass
    one = np.einsum("am,am,m,akm->ak", count, bias, volume, profile)
    missing = 1-np.sum(count*bias*volume, axis=1)
    one += missing[:, None]*profile_min
    pair_rows = []
    for first in range(profile.shape[1]):
        for second in range(first, profile.shape[1]):
            left = profile[:, first]
            right = profile[:, second]
            product = left*right
            pair_rows.append(np.array([
                np.sum(count*volume**2*product, axis=1),
                np.sum(count*bias*volume**2*product, axis=1),
                np.sum(count*bias*volume**3*product*right, axis=1),
                np.sum(count*bias*volume**3*product*left, axis=1),
                np.sum(count*volume**4*product**2, axis=1),
            ]))
    return one, np.stack(pair_rows, axis=-1)
