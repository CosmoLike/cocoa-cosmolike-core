"""Independent contractions of supplied halo abundances and profiles.

halo_cov.c builds the halo-model mass moments of the cold dark matter
plus baryon (cb) field. A moment I<beta><mu> has mu profile legs and
beta (0 or 1) powers of the linear halo bias b:

    I<beta><mu>(k_1..k_mu) = int dlnM (dn/dlnM) b^beta (M/rho_cb)^mu
                             u(k_1|M) ... u(k_mu|M).

I11 is dimensionless; the pair moments I02, I12, I13 and I04 carry
length^3, length^3, length^6 and length^9.

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
        number_density, bias = [na,nmass] dn/dlnM in length^-3 and linear
            cb halo bias.
        profile = [na,nk,nmass] normalized NFW Fourier profiles u(k|M).
        profile_min = [na,nk] profile at the lower mass cutoff.
        rho_cb = cb mean density in mass / length^3, in the mass unit of
            the mass argument.
    Returns: I11 [na,nk] and five moments [5,na,npair], with i-major pairs.
        A pair (K,Q) takes the profile k indices i <= j, so
        npair = nk*(nk+1)/2. The rows are I02(K,Q), I12(K,Q), I13(K,Q,Q),
        I13(K,K,Q) and I04(K,K,Q,Q), the role order of halo_cov.h.
    Only I11 receives the missing low-mass completion on the supplied rule:
    the remainder 1 - sum (dn/dlnM) dlnM b M/rho_cb of the bias
    normalization takes the profile at the cutoff, so I11(k=0)=1.
    halo_cov.c applies the same completion when its mass panels use
    ordinary finite quadrature; the Wynn-extrapolated I11 tail of its
    default panels is outside this oracle.
    """
    volume = mass/rho_cb
    count = number_density*dlnmass
    # a labels scale factors, k wavenumbers and m mass nodes:
    # I11[a,k] = sum over m of count*bias*(M/rho_cb)*u(k|M).
    one = np.einsum("am,am,m,akm->ak", count, bias, volume, profile)
    # Over all masses count*bias*M/rho_cb sums to one; the unresolved mass
    # below the cutoff carries the remainder, with the cutoff profile.
    missing = 1-np.sum(count*bias*volume, axis=1)
    one += missing[:, None]*profile_min
    pair_rows = []
    for first in range(profile.shape[1]):
        for second in range(first, profile.shape[1]):
            left = profile[:, first]
            right = profile[:, second]
            product = left*right
            # Rows I02(K,Q), I12(K,Q), I13(K,Q,Q), I13(K,K,Q), I04(K,K,Q,Q),
            # with u(K|M) = left and u(Q|M) = right.
            pair_rows.append(np.array([
                np.sum(count*volume**2*product, axis=1),
                np.sum(count*bias*volume**2*product, axis=1),
                np.sum(count*bias*volume**3*product*right, axis=1),
                np.sum(count*bias*volume**3*product*left, axis=1),
                np.sum(count*volume**4*product**2, axis=1),
            ]))
    return one, np.stack(pair_rows, axis=-1)
