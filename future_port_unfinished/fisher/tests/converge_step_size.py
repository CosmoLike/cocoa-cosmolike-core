"""Step-size convergence at high ell: analytic (responses at step s) vs
finite differences through cosmolike (step s), s = 0.0025 .. 0.02."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import numpy as np

# directory holding the lsst_y1 interface built with the fisher files
# (see ../README.md, "Trying the draft"); it must come first on sys.path
SCR = os.environ["FISHER_BUILD_DIR"]
ROOT = os.environ["ROOTDIR"]
sys.path.insert(0, SCR)
sys.path.insert(1, ROOT + "/external_modules/code/cosmolike_core/"
                "future_port_unfinished/fisher")
os.chdir(ROOT + "/projects/lsst_y1")
import cosmolike_lsst_y1_interface as ci
import cosmolike_lsst_y1_notebook_wrappers as nw
from fisher_camb_response import get_camb_response

nw.init_cosmolike()
print("non_linear_emul =", nw._CONFIG["non_linear_emul"])
POINT = dict(omegam=nw.omegam, omegab=nw.omegab, H0=nw.H0, ns=nw.ns,
             As_1e9=nw.As_1e9, w=nw.w, w0pwa=nw.w0pwa, mnu=nw.mnu)
CAMB = dict(AccuracyBoost=1.0, kmax=10.0, k_per_logint=10,
            CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
            non_linear_emul=nw._CONFIG["non_linear_emul"])
ELL = np.array([1000., 3000.])
for X, log, scale in (("As_1e9", True, 1.0), ("omegam", False, 0.5)):
    print(f"\n{X}: ell, pair | step: analytic  FD")
    rows = {}
    for s0 in (0.0025, 0.005, 0.01, 0.02):
        s = s0*scale
        x0 = POINT[X]
        xp, xm = (x0*np.exp(s), x0*np.exp(-s)) if log else (x0 + s, x0 - s)
        Cp = nw.C_ss_tomo_limber(ell=ELL, **{X: xp})[0]
        Cm = nw.C_ss_tomo_limber(ell=ELL, **{X: xm})[0]
        fd = (np.log(Cp) - np.log(Cm))/(2*s)
        nw.C_ss_tomo_limber(ell=ELL)          # fiducial state
        ci.reset_fisher_response()
        ci.set_fisher_response(ip=0, **get_camb_response(X, s, POINT,
                                                         log=log, **CAMB))
        C, dC = ci.dC_ss_dX_tomo_limber(l=ELL)
        an = np.array(dC)[0]/np.array(C)
        rows[s] = (an, fd)
    for li, ell in enumerate(ELL):
        for (i, j) in ((0, 0), (4, 4), (0, 4)):
            line = "  ".join(f"{s:.4f}: {rows[s][0][li,i,j]:+.5f} "
                             f"{rows[s][1][li,i,j]:+.5f}" for s in rows)
            print(f"  ell={ell:5.0f} ({i},{j}) | {line}")
