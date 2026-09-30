"""Response of the set_cosmology tables to one cosmological parameter.

EXPERIMENTAL (future_port_unfinished/fisher). cosmo2d_fisher.c needs, for
each parameter X, the derivatives of the tables set_cosmology consumes:

  dlnP_NL/dX (k, z)   dchi/dX (z)   dlnG/dX (z)   dlnOmega_m/dX

This module computes them as central differences of
cosmolike_notebook_utils.get_camb_cosmology on the fixed fiducial grids,
so the result goes straight into ci.set_fisher_response:

    resp = get_camb_response("omegam", 0.005, point, **camb_kwargs)
    ci.set_fisher_response(ip=1, **resp)

Only the tables are differenced; every cosmolike integral is then
differentiated analytically (see fisher_derivatives.pdf).
"""

import numpy as np
from scipy.interpolate import interp1d


def get_camb_response(X, step, point, log=False, **camb_kwargs):
    """Central-difference response of the set_cosmology tables to X.

    Arguments:
      X      = name of a get_camb_cosmology cosmology argument: "omegam",
               "omegab", "H0", "ns", "As_1e9", "w", "w0pwa" or "mnu".
      step   = half step of the central difference, in X (or in ln X
               when log=True).
      point  = dict with the fiducial value of every get_camb_cosmology
               cosmology argument (the keys above).
      log    = True differentiates with respect to ln X (use it for
               As_1e9, so the derivative is dlnC/dlnA_s-ready).
      camb_kwargs = accuracy arguments forwarded unchanged to
               get_camb_cosmology (AccuracyBoost, kmax, k_per_logint,
               CAMBAccuracyBoost, CLAccuracyBoost, non_linear_emul); they
               must match the call that set the fiducial cosmology.

    Returns:
      dict with keys dlnOm_dX, z_chi, dchi_dX, z_G, dlnG_dX, log10k, z_P,
      dlnPNL_dX: the keyword arguments of ci.set_fisher_response (minus ip).
      dlnOm_dX follows Cocoa's sampling basis, where Omega_m is itself a
      sampled parameter: 1/Omega_m for X = omegam (1 for ln omegam), and
      0 for every other parameter (h included: cosmolike works in h units).
    """
    import cosmolike_notebook_utils as cnu

    if X not in point:
        raise KeyError(f"'{X}' is not a key of point: {sorted(point)}")
    x0 = point[X]
    if log:
        xp, xm = x0*np.exp(step), x0*np.exp(-step)
    else:
        xp, xm = x0 + step, x0 - step

    def run(value):
        p = dict(point)
        p[X] = value
        return cnu.get_camb_cosmology(**p, **camb_kwargs)

    (lk_p, zP, lnPL_p, lnPN_p, G_p, zG, zchi, chi_p, _, _) = run(xp)
    (lk_m, _,  lnPL_m, lnPN_m, G_m, _,  _,    chi_m, _, _) = run(xm)
    nz, nk = len(zP), len(lk_p)
    # (z, k) tables; the flat arrays are Fortran-ordered, as set_cosmology
    # consumes them
    PNp = lnPN_p.reshape(nz, nk, order="F")
    PNm = lnPN_m.reshape(nz, nk, order="F")
    # the tables' k grid is shifted to h/Mpc after evaluation, so only for
    # X = H0 do the perturbed grids differ from the fiducial one: bring
    # both onto the fiducial h/Mpc grid before differencing. The fiducial
    # grid pokes past the ends of the perturbed ones, and there the tables
    # must be extrapolated linearly in log10 k — the way p_nonlin
    # extrapolates — not clamped: a clamped edge node corrupts the edge
    # slope of the response, which cosmolike's linear extrapolation then
    # multiplies by hundreds of grid spacings at the high-k tail xi+ reads
    h0 = point["H0"]/100.0
    lk_fid = lk_p + np.log10((point["H0"] if X != "H0" else xp)/100.0) \
                  - np.log10(h0)
    if X == "H0":
        def regrid(lk_src, tab):
            return interp1d(lk_src, tab, axis=1, kind="linear",
                            fill_value="extrapolate",
                            assume_sorted=True)(lk_fid)
        PNp = regrid(lk_p, PNp)
        PNm = regrid(lk_m, PNm)
    dlnPNL = (PNp - PNm)/(2.0*step)
    dchi = (np.asarray(chi_p) - np.asarray(chi_m))/(2.0*step)
    dlnG = (np.log(G_p) - np.log(G_m))/(2.0*step)
    if X == "omegam":
        dlnOm = 1.0 if log else 1.0/x0
    else:
        dlnOm = 0.0
    return dict(dlnOm_dX=dlnOm,
                z_chi=np.asarray(zchi, dtype="float64"),
                dchi_dX=np.asarray(dchi, dtype="float64"),
                z_G=np.asarray(zG, dtype="float64"),
                dlnG_dX=np.asarray(dlnG, dtype="float64"),
                log10k=np.asarray(lk_fid, dtype="float64"),
                z_P=np.asarray(zP, dtype="float64"),
                dlnPNL_dX=np.asarray(dlnPNL.flatten(order="F"),
                                     dtype="float64"))
