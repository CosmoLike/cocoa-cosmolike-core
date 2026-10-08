"""Independent NumPy Limber integration of supplied radial windows.

spectra_cov.c computes the Limber cross spectra of all lens and source
fields on one common radial rule,

    C_AB(ell) = int dchi W_A W_B P((ell+1/2)/f_K, a)/f_K^2,

with the spin, magnification and RSD factors applied to the windows.
This module never imports a project interface. The saved CAMB tables are
the shared physical inputs; the power interpolation and field-matrix
contraction below are independent of the C covariance implementation.
The windows and quadrature nodes are supplied explicitly, so agreement
tests their use, not the core's redshift-distribution or IA physics.
"""

import numpy as np


def lensing_efficiency(distance, source_distance, source_density,
                        redshift_weights):
    """Directly sum the lensing geometry against supplied source samples.

    g(chi) = sum over nodes of dz n(z) (1 - chi/chi'), the flat-space
    lensing efficiency, without the 1.5 Omega_m f_K/a prefactor of the
    lensing window. spectra_cov.c accumulates g = A - chi*B from two
    cumulative integrals; this direct sum does not use that decomposition.

    Arguments:
        distance = foreground distance in an arbitrary length unit.
        source_distance = [nnode] distances in the same unit, all >= distance.
        source_density = normalized n(z) at these quadrature nodes.
        redshift_weights = positive dz integration weights.
    Returns: dimensionless g.
    """
    return np.sum(redshift_weights*source_density*(1-distance/source_distance))


def power_at_nodes(archive, ell, geometry, linear):
    """Interpolate the supplied CAMB table at each Limber wavenumber.

    The Limber wavenumber is k=(ell+1/2)/f_K. ln P is interpolated
    bilinearly in (z, log10 k); outside the table the edge rectangle
    extrapolates linearly in both variables.

    Arguments:
        archive = mapping of the CAMB input arrays archived by survey_inputs:
            log10k_2D (k in h/Mpc), z_2D, and lnP_linear, lnP_nonlinear
            (ln P with P in (Mpc/h)^3).
        ell = [nell] multipoles.
        geometry = [4,nnode], rows a, chi, f_K, dchi in c/H0 units.
        linear = True selects the linear total-matter table, False the
            nonlinear one.
    Returns:
        [nell,nnode] P in (c/H0)^3; edge brackets give power-law k tails.
    """
    log10k = archive["log10k_2D"]
    redshift = archive["z_2D"]
    key = "lnP_nonlinear"
    if linear:
        key = "lnP_linear"
    # The saved input is flattened in Fortran order: consecutive z samples
    # at one k. Reshape before selecting the interpolation rectangles.
    logpower = archive[key].reshape((len(redshift), len(log10k)), order="F")
    distance_unit = 2997.92458  # c/H0 in Mpc/h
    query_z = 1.0/geometry[0]-1.0
    query_k = np.log10((ell[:, None]+0.5)/geometry[2]/distance_unit)
    iz = np.searchsorted(redshift, query_z)-1
    ik = np.searchsorted(log10k, query_k)-1
    iz = np.clip(iz, 0, len(redshift)-2)
    ik = np.clip(ik, 0, len(log10k)-2)
    fraction_z = (query_z-redshift[iz])/(redshift[iz+1]-redshift[iz])
    fraction_k = (query_k-log10k[ik])/(log10k[ik+1]-log10k[ik])
    low = (1-fraction_z)*logpower[iz, ik]+fraction_z*logpower[iz+1, ik]
    high = (1-fraction_z)*logpower[iz, ik+1]+fraction_z*logpower[iz+1, ik+1]
    result = np.exp((1-fraction_k)*low+fraction_k*high)
    return result/distance_unit**3


def limber_matrix(ell, geometry, windows, nlens, power, rsd=None,
                  distance_unit=1.0):
    """Integrate every field pair without pair exclusions or positivity cuts.

    Arguments:
        ell = [nell] multipoles >= 1.
        geometry = [4,nnode], a, chi, f_K, dchi in the supplied length unit;
            only f_K and dchi enter the integral.
        windows = [3,nfield,nnode], density, lensing, signed intrinsic terms,
            in inverse length.
        nlens = number of lens fields, which precede the source fields.
        power = [nell,nnode] matter power in the cube of that length unit.
        rsd = optional [nell,nlens,nnode] additional lens window.
        distance_unit = ratio of desired to supplied distance representation:
            multiply f_K,dchi by it, divide windows by it, multiply P by its
            cube. The output must be invariant under this conversion.
    Returns:
        [nell,nfield,nfield] dimensionless symmetric signal spectra.
    """
    nfield = windows.shape[1]
    nnode = windows.shape[2]
    fields = np.zeros((len(ell), nfield, nnode))
    # Core C_ell conventions, as in spectra_cov.c: a lens field is density
    # plus ell(ell+1)/(ell+1/2)^2 times its magnification window; a source
    # field is lensing plus signed IA, times the spin-2 factor
    # sqrt[(ell-1) ell (ell+1) (ell+2)]/(ell+1/2)^2.
    magnification = ell*(ell+1)/(ell+0.5)**2
    shear = np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+0.5)**2
    fields[:, :nlens] = windows[0, :nlens]
    fields[:, :nlens] += magnification[:, None, None]*windows[1, :nlens]
    fields[:, nlens:] = shear[:, None, None]*(windows[1, nlens:]
                                             +windows[2, nlens:])
    if rsd is not None:
        fields[:, :nlens] += rsd

    fields /= distance_unit
    measure = geometry[3]*distance_unit/(geometry[2]*distance_unit)**2
    dimensional_power = power*distance_unit**3
    # e labels ell, f/g label fields, and p labels radial nodes. NumPy
    # contracts all field pairs independently of the C triangular loop.
    return np.einsum("efp,egp,ep,p->efg", fields, fields,
                     dimensional_power, measure, optimize=False)
