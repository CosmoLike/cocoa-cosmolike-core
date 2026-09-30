"""Chain-rule test of the cosmo2d_fisher draft with CAMB noise removed.

The perturbed cosmologies are SYNTHETIC: every set_cosmology table is the
fiducial one moved along the loaded response, exactly
  lnP_NL(+-) = lnP_NL0 +- s R,  chi(+-) = chi0 +- s chi_X,
  lnG(+-) = lnG0 +- s lnG_X,    Omega_m(+-) = Omega_m0 (1 +- s eps),
so the central difference of cosmolike's C_ss / xi_pm along that path
isolates the analytic chain rule from CAMB's run-to-run jitter.
"""
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
import cosmolike_notebook_utils as cnu
from fisher_camb_response import get_camb_response

nw.init_cosmolike()
POINT = dict(omegam=nw.omegam, omegab=nw.omegab, H0=nw.H0, ns=nw.ns,
             As_1e9=nw.As_1e9, w=nw.w, w0pwa=nw.w0pwa, mnu=nw.mnu)
CAMB = dict(AccuracyBoost=1.0, kmax=10.0, k_per_logint=10,
            CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0, non_linear_emul=2)
PARAMS = [("As_1e9", True, 0.02), ("omegam", False, 0.01), ("w", False, 0.04)]
ELL = np.array([30., 100., 300., 1000., 3000.])
iu = np.triu_indices(5)

# fiducial state (nuisances set by the wrapper) and fiducial tables
nw.C_ss_tomo_limber(ell=ELL)
nw.xi()
(lk, z2, lnPL0, lnPN0, G0, zG, z1, chi0, _, _) = cnu.get_camb_cosmology(**POINT, **CAMB)
lk = np.asarray(lk); z2 = np.asarray(z2); z1 = np.asarray(z1)


def set_tables(om, lnPN, G, chi):
    ci.set_cosmology(omegam=om, H0=POINT["H0"], log10k_2D=lk, z_2D=z2,
                     lnP_linear=lnPL0, lnP_nonlinear=lnPN, G=G, z_G=zG,
                     z_1D=z1, chi=chi)


resp = [get_camb_response(X, s, POINT, log=log, **CAMB)
        for X, log, s in PARAMS]
set_tables(POINT["omegam"], lnPN0, G0, chi0)
ci.reset_fisher_response()
for ip, r in enumerate(resp):
    ci.set_fisher_response(ip=ip, **r)
C, dC = map(np.array, ci.dC_ss_dX_tomo_limber(l=ELL))
xip, xim, dxip, dxim = map(np.array, ci.dxi_pm_dX_tomo())

for ip, ((X, log, _), r) in enumerate(zip(PARAMS, resp)):
    print(f"d/d{'ln ' if log else ''}{X}:")
    for s in (1e-3, 1e-4):
        out = {}
        for sign in (+1, -1):
            om = POINT["omegam"]*(1 + sign*s*r["dlnOm_dX"])
            set_tables(om,
                       lnPN0 + sign*s*r["dlnPNL_dX"],
                       np.asarray(G0)*np.exp(sign*s*r["dlnG_dX"]),
                       np.asarray(chi0) + sign*s*r["dchi_dX"])
            out[sign] = (np.array(ci.C_ss_tomo_limber(l=ELL)[0]),
                         np.array(ci.xi_pm_tomo()[0]),
                         np.array(ci.xi_pm_tomo()[1]))
        fdC = (np.log(out[1][0]) - np.log(out[-1][0]))/(2*s)
        anC = dC[ip]/C
        eC = np.abs(anC/fdC - 1)[:, iu[0], iu[1]]
        fdxp = (out[1][1] - out[-1][1])/(2*s)
        fdxm = (out[1][2] - out[-1][2])/(2*s)
        sc = np.max(np.abs(fdxp), axis=0)[iu]
        exp_ = (np.max(np.abs(dxip[ip] - fdxp)[:, iu[0], iu[1]], axis=0)/sc).max()
        sc = np.max(np.abs(fdxm), axis=0)[iu]
        exm_ = (np.max(np.abs(dxim[ip] - fdxm)[:, iu[0], iu[1]], axis=0)/sc).max()
        per_ell = ", ".join(f"{ELL[i]:.0f}: {eC[i].max():.1e}"
                            for i in range(len(ELL)))
        print(f"  s={s:.0e}  C_ss |an/FD-1| by ell: {per_ell}   "
              f"xi+ {exp_:.1e}  xi- {exm_:.1e}")
set_tables(POINT["omegam"], lnPN0, G0, chi0)
