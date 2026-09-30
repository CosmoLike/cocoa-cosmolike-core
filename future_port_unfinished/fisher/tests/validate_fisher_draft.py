"""Validate the cosmo2d_fisher draft (scratch lsst_y1 build) against
central finite differences through Cocoa's own pipeline.

For X in (ln As, Omega_m, w): responses from fisher_camb_response on the
fiducial grids -> ci.dC_ss_dX_tomo_limber / ci.dxi_pm_dX_tomo, versus
(C(X+h) - C(X-h))/2h with nw.C_ss_tomo_limber / nw.xi at two step sizes.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import time
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
assert ci.__file__.startswith(SCR), ci.__file__
import cosmolike_lsst_y1_notebook_wrappers as nw
from fisher_camb_response import get_camb_response

nw.init_cosmolike()
POINT = dict(omegam=nw.omegam, omegab=nw.omegab, H0=nw.H0, ns=nw.ns,
             As_1e9=nw.As_1e9, w=nw.w, w0pwa=nw.w0pwa, mnu=nw.mnu)
CAMB = dict(AccuracyBoost=1.0, kmax=10.0, k_per_logint=10,
            CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
            non_linear_emul=nw._CONFIG["non_linear_emul"])
# (name, log?, steps)  first step feeds the response, both feed FD
PARAMS = [("As_1e9", True, (0.01, 0.02)),
          ("omegam", False, (0.005, 0.01)),
          ("w", False, (0.02, 0.04))]
ELL = np.array([30., 100., 300., 1000., 3000.])
nb = 5
iu = np.triu_indices(nb)


def shifted(X, log, s):
    x0 = POINT[X]
    return (x0*np.exp(s), x0*np.exp(-s)) if log else (x0 + s, x0 - s)


# ---- finite differences through the cosmolike pipeline -------------------
fd_C, fd_xi = {}, {}
for X, log, steps in PARAMS:
    for s in steps:
        xp, xm = shifted(X, log, s)
        Cp = nw.C_ss_tomo_limber(ell=ELL, **{X: xp})[0]
        Cm = nw.C_ss_tomo_limber(ell=ELL, **{X: xm})[0]
        fd_C[(X, s)] = (np.log(Cp) - np.log(Cm))/(2*s)
        _, xpp, xmp = nw.xi(**{X: xp})
        _, xpm, xmm = nw.xi(**{X: xm})
        fd_xi[(X, s)] = ((xpp - xpm)/(2*s), (xmp - xmm)/(2*s))

# ---- analytic: fiducial state, responses, one call ------------------------
C0 = nw.C_ss_tomo_limber(ell=ELL)[0]
resp = {X: get_camb_response(X, steps[0], POINT, log=log, **CAMB)
        for X, log, steps in PARAMS}
ci.reset_fisher_response()
for ip, (X, _, _) in enumerate(PARAMS):
    ci.set_fisher_response(ip=ip, **resp[X])
t0 = time.perf_counter()
C, dC = ci.dC_ss_dX_tomo_limber(l=ELL)
t_an = time.perf_counter() - t0
t0 = time.perf_counter()
_ = ci.C_ss_tomo_limber(l=ELL)
t_c = time.perf_counter() - t0
C, dC = np.array(C), np.array(dC)
print(f"C_fisher vs likelihood C_ss: max rel diff = "
      f"{np.max(np.abs(C/C0 - 1)[:, iu[0], iu[1]]):.2e}")
print(f"timing: C + {len(PARAMS)} derivatives {1e3*t_an:.1f} ms "
      f"(plain C_ss {1e3*t_c:.1f} ms)\n")
print("C_ss: max over ell x pairs of |dlnC analytic/FD - 1|, and FD step "
      "sensitivity |FD(h)/FD(2h) - 1|")
for ip, (X, log, steps) in enumerate(PARAMS):
    an = dC[ip]/C
    f1, f2 = fd_C[(X, steps[0])], fd_C[(X, steps[1])]
    e1 = np.max(np.abs(an/f1 - 1)[:, iu[0], iu[1]])
    e12 = np.max(np.abs(f1/f2 - 1)[:, iu[0], iu[1]])
    print(f"  d/d{'ln ' if log else ''}{X:7s}: analytic vs FD(h) {e1:.2e}   "
          f"FD(h) vs FD(2h) {e12:.2e}")
    for li in (0, 2, 4):
        print(f"      ell={ELL[li]:5.0f} (0,0): analytic {an[li,0,0]:+.5f} "
              f" FD(h) {f1[li,0,0]:+.5f}  FD(2h) {f2[li,0,0]:+.5f}")

# ---- xi_pm ------------------------------------------------------------------
theta, xip0, xim0 = nw.xi()
for ip, (X, _, _) in enumerate(PARAMS):
    ci.set_fisher_response(ip=ip, **resp[X])
t0 = time.perf_counter()
xip, xim, dxip, dxim = ci.dxi_pm_dX_tomo()
t_xi = time.perf_counter() - t0
xip, xim, dxip, dxim = map(np.array, (xip, xim, dxip, dxim))
print(f"\nxi_fisher vs xi_pm_tomo: max rel diff xi+ "
      f"{np.max(np.abs(xip/xip0 - 1)[:, iu[0], iu[1]]):.2e}, xi- "
      f"{np.max(np.abs(xim/xim0 - 1)[:, iu[0], iu[1]]):.2e}")
print(f"timing: xi + {len(PARAMS)} derivatives {1e3*t_xi:.1f} ms")
print("xi_pm: max over theta x pairs of |dxi analytic - FD| / max|dxi FD| "
      "(per pair), and the same between FD(h) and FD(2h)")
for ip, (X, log, steps) in enumerate(PARAMS):
    row = []
    for pm, an in ((0, dxip[ip]), (1, dxim[ip])):
        f1 = fd_xi[(X, steps[0])][pm]
        f2 = fd_xi[(X, steps[1])][pm]
        scale = np.max(np.abs(f1), axis=0)[iu]
        e = np.max(np.abs(an - f1)[:, iu[0], iu[1]], axis=0)/scale
        e12 = np.max(np.abs(f1 - f2)[:, iu[0], iu[1]], axis=0)/scale
        row.append(f"xi{'+' if pm == 0 else '-'}: {e.max():.2e} "
                   f"(FD h vs 2h {e12.max():.2e})")
    print(f"  d/d{'ln ' if log else ''}{X:7s}: " + "   ".join(row))
