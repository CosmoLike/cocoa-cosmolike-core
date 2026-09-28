"""How much noise does cosmolike itself add to finite-difference derivatives?

Along a SYNTHETIC path (every set_cosmology table moved exactly along the
CAMB response, so the input tables are perfectly smooth in the parameter),
finite differences of cosmolike's C_ss / xi_pm at Fisher-sized steps, with
the 3-point and the 5-point stencil (the notebook Fisher uses 5-point, h up
to 0.06), compared to the analytic derivative. A pipeline that adds noise
shows up as FD errors that do not scale as s^2 (3-pt) / s^4 (5-pt).
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import numpy as np

SCR = os.environ["FISHER_BUILD_DIR"]
ROOT = os.environ["ROOTDIR"]
sys.path.insert(0, SCR)
sys.path.insert(1, ROOT + "/external_modules/code/cosmolike_core/"
                "future_port_unfinished/fisher")
os.chdir(ROOT + "/projects/lsst_y1")
import cosmolike_lsst_y1_interface as ci
import cosmolike_lsst_y1_notebook_wrappers as nw
import cosmolike_notebook_utils as cnu
from fisher_camb_response import get_camb_response

nw.init_cosmolike()
POINT = dict(omegam=nw.omegam, omegab=nw.omegab, H0=nw.H0, ns=nw.ns,
             As_1e9=nw.As_1e9, w=nw.w, w0pwa=nw.w0pwa, mnu=nw.mnu)
CAMB = dict(AccuracyBoost=1.0, kmax=10.0, k_per_logint=10,
            CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0, non_linear_emul=2)
PARAMS = [("As_1e9", True, 0.02), ("omegam", False, 0.01), ("w", False, 0.04)]
ELL = np.array([30., 100., 300., 1000., 3000.])
STEPS = (1e-3, 3e-3, 1e-2, 3e-2, 6e-2)
iu = np.triu_indices(5)

nw.C_ss_tomo_limber(ell=ELL)
nw.xi()
(lk, z2, lnPL0, lnPN0, G0, z1, chi0) = cnu.get_camb_cosmology(**POINT, **CAMB)
lk, z2, z1 = map(np.asarray, (lk, z2, z1))
G0, chi0 = np.asarray(G0), np.asarray(chi0)


def at(r, t):
    """cosmolike's C_ss and xi_pm at synthetic parameter offset t."""
    ci.set_cosmology(omegam=POINT["omegam"]*(1 + t*r["dlnOm_dX"]),
                     H0=POINT["H0"], log10k_2D=lk, z_2D=z2,
                     lnP_linear=lnPL0, lnP_nonlinear=lnPN0 + t*r["dlnPNL_dX"],
                     G=G0*np.exp(t*r["dlnG_dX"]), z_1D=z1,
                     chi=chi0 + t*r["dchi_dX"])
    C = np.array(ci.C_ss_tomo_limber(l=ELL)[0])
    xp, xm = map(np.array, ci.xi_pm_tomo())
    return C, xp


resp = [get_camb_response(X, s, POINT, log=log, **CAMB)
        for X, log, s in PARAMS]
ci.set_cosmology(omegam=POINT["omegam"], H0=POINT["H0"], log10k_2D=lk,
                 z_2D=z2, lnP_linear=lnPL0, lnP_nonlinear=lnPN0, G=G0,
                 z_1D=z1, chi=chi0)
ci.reset_fisher_response()
for ip, r in enumerate(resp):
    ci.set_fisher_response(ip=ip, **r)
C0, dC = map(np.array, ci.dC_ss_dX_tomo_limber(l=ELL))
xip0, _, dxip, _ = map(np.array, ci.dxi_pm_dX_tomo())

print("max over ell x pairs of |FD/analytic - 1| (C_ss) and "
      "max|FD - analytic|/max|analytic| (xi+)")
for ip, ((X, log, _), r) in enumerate(zip(PARAMS, resp)):
    anC = dC[ip]
    anX = dxip[ip]
    print(f"d/d{'ln ' if log else ''}{X}:")
    for s in STEPS:
        v = {t: at(r, t) for t in (-2*s, -s, s, 2*s)}
        f3C = (v[s][0] - v[-s][0])/(2*s)
        f5C = (-v[2*s][0] + 8*v[s][0] - 8*v[-s][0] + v[-2*s][0])/(12*s)
        f3X = (v[s][1] - v[-s][1])/(2*s)
        f5X = (-v[2*s][1] + 8*v[s][1] - 8*v[-s][1] + v[-2*s][1])/(12*s)
        e3C = np.max(np.abs(f3C/anC - 1)[:, iu[0], iu[1]])
        e5C = np.max(np.abs(f5C/anC - 1)[:, iu[0], iu[1]])
        sc = np.max(np.abs(anX))
        e3X = np.max(np.abs(f3X - anX))/sc
        e5X = np.max(np.abs(f5X - anX))/sc
        print(f"  s={s:7.0e}   C_ss 3pt {e3C:.1e}  5pt {e5C:.1e}   "
              f"xi+ 3pt {e3X:.1e}  5pt {e5X:.1e}")
