"""CAMB run-to-run jitter of the P tables versus CAMB accuracy.

jitter = max |lnP(X0) - (lnP(X0+s) + lnP(X0-s))/2| at fixed (k, z) for
X = ln A_s, s = 0.01. P_lin is exactly linear in A_s, so any nonzero value
is noise; for P_NL the genuine curvature term (s^2/2) d^2lnP/dlnAs^2 is
~1e-5, so anything well above that is noise too.
"""
import os
import sys
import numpy as np

ROOT = os.environ["ROOTDIR"]
sys.path.insert(0, ROOT + "/external_modules/code/cosmolike_core")
import cosmolike_notebook_utils as cnu

POINT = dict(omegam=0.3, omegab=0.04, H0=67.32, ns=0.96605, As_1e9=2.1,
             w=-0.9, w0pwa=-0.9, mnu=0.06)
s = 0.01
for lab, kw in (("AccuracyBoost=1 (default)", dict(AccuracyBoost=1.0)),
                ("AccuracyBoost=2", dict(AccuracyBoost=2.0)),
                ("AccuracyBoost=3", dict(AccuracyBoost=3.0)),
                ("CAMBAccuracyBoost=2", dict(CAMBAccuracyBoost=2.0)),
                ("k_per_logint=40", dict(k_per_logint=40))):
    base = dict(kmax=10.0, k_per_logint=10, non_linear_emul=2)
    base.update(kw)
    out = []
    for f in (1.0, np.exp(s), np.exp(-s)):
        (lk, z2, lnPL, lnPN, _, _, _, _, _, _) = cnu.get_camb_cosmology(
            **dict(POINT, As_1e9=POINT["As_1e9"]*f), **base)
        out.append((np.asarray(lnPL), np.asarray(lnPN)))
    nz, nk = len(z2), len(lk)
    sz = np.asarray(z2) < 1.5
    sk = (np.asarray(lk) >= -2) & (np.asarray(lk) <= 1.0)
    res = []
    for m, name in ((0, "P_lin"), (1, "P_NL")):
        d = (out[0][m] - 0.5*(out[1][m] + out[2][m])).reshape(nz, nk, order="F")
        res.append(f"{name} {np.abs(d[np.ix_(sz, sk)]).max():.1e}")
    print(f"{lab:26s}: max jitter (0.01<k<10 h/Mpc, z<1.5): " + ", ".join(res))
