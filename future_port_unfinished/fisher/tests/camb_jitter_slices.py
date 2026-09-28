"""Where CAMB's P_NL jitter lives, and what it does to one-sided derivatives.

jitter(k, z) = lnP(X0) - (lnP(X0+s) + lnP(X0-s))/2 for X = ln A_s. With
stock CAMB it is concentrated in whole redshift slices (~2.5e-3 across
every nonlinear k of a slice, ~1e-5 elsewhere): the signature of a jump in
halofit's nonlinear scale R_NL(z), see the Cocoa patch
cocoa_installation_libraries/camb_changes/camb/halofit.patch.
Run it once with stock CAMB and once with the patched CAMB first on
PYTHONPATH to compare.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.environ["ROOTDIR"] + "/external_modules/code/cosmolike_core")
import camb
import cosmolike_notebook_utils as cnu

print("CAMB from", os.path.dirname(camb.__file__))
POINT = dict(omegam=0.3, omegab=0.04, H0=67.32, ns=0.96605, As_1e9=2.1,
             w=-0.9, w0pwa=-0.9, mnu=0.06)
BASE = dict(AccuracyBoost=1.0, kmax=10.0, k_per_logint=10, non_linear_emul=2)
s = 0.01
out = []
for f in (0.0, 1.0, -1.0):
    p = dict(POINT, As_1e9=POINT["As_1e9"]*np.exp(f*s))
    (lk, z2, _, lnPN, _, _, _) = cnu.get_camb_cosmology(**p, **BASE)
    out.append(np.asarray(lnPN).reshape(len(z2), len(lk), order="F"))
lk, z2 = np.asarray(lk), np.asarray(z2)
d = out[0] - 0.5*(out[1] + out[2])
sk = (lk >= -2) & (lk <= 1.0)
print("max |jitter| per z slice (0.01 < k < 10 h/Mpc):")
print("  " + " ".join(f"{z:.2f}:{np.abs(d[j][sk]).max():.1e}"
                      for j, z in enumerate(z2) if z < 1.5 and j % 3 == 0))
fwd = (out[1] - out[0])/s
bwd = (out[0] - out[2])/s
sz = z2 < 1.5
print(f"max |forward - backward| dlnP_NL/dlnAs: "
      f"{np.abs(fwd - bwd)[np.ix_(sz, sk)].max():.2e}")
