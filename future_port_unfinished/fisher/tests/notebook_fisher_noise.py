"""Reproduce the lsst_y1 notebook Fisher (FoM(As, Omega_m), 17 parameters,
5-point stencil, relative step h) and split its step-size instability
into CAMB and cosmolike contributions.

  (a) notebook:        nw.get_dv at AccuracyBoost 1 (CAMB + cosmolike)
  (b) CAMB accurate:   same, CAMB AccuracyBoost 3, cosmolike at boost 1
  (c) cosmolike only:  cosmological parameters moved along SYNTHETIC
                       tables (fiducial + t * CAMB response, perfectly
                       smooth in t); nuisance parameters as in (a)

A flat FoM(h) means the derivatives do not depend on the step, i.e. no
noise; drift with h is the noise the notebook's "Test 2: Step Size" shows.
"""
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import time
import numpy as np

ROOT = os.environ["ROOTDIR"]
sys.path.insert(1, ROOT + "/external_modules/code/cosmolike_core/"
                "future_port_unfinished/fisher")
os.chdir(ROOT + "/projects/lsst_y1")
import cosmolike_lsst_y1_interface as ci
import cosmolike_lsst_y1_notebook_wrappers as nw
import cosmolike_notebook_utils as cnu
from fisher_camb_response import get_camb_response

nw.init_cosmolike(CLprobe="xi", with_data=True)
invcov = ci.get_inv_cov_masked()
CV = nw.fisher_fiducial_point()
EMUL = nw._CONFIG["non_linear_emul"]
HS = (0.01, 0.02, 0.03, 0.05, 0.08)

# ---- CAMB determinism: identical inputs, identical tables? ---------------
kw = dict(omegam=CV[4], omegab=CV[3], H0=CV[2], ns=CV[1], As_1e9=CV[0],
          w=-1.0, w0pwa=-1.0, mnu=nw.mnu, AccuracyBoost=1.0, kmax=5.0,
          k_per_logint=10, CAMBAccuracyBoost=1.0, CLAccuracyBoost=1.0,
          non_linear_emul=EMUL)
t0 = time.perf_counter()
a = cnu.get_camb_cosmology(**kw)
t_camb = time.perf_counter() - t0
b = cnu.get_camb_cosmology(**kw)
det = max(np.max(np.abs(np.asarray(x) - np.asarray(y))) for x, y in zip(a, b))
print(f"CAMB rerun at identical inputs: max |difference| = {det:.1e} "
      f"({t_camb:.1f} s per call)", flush=True)

# cache CAMB by its arguments (valid because CAMB is deterministic above):
# the notebook reruns CAMB at the fiducial for every nuisance evaluation
_camb = cnu.get_camb_cosmology
_cache = {}


def _cached(**k):
    key = tuple(sorted(k.items()))
    if key not in _cache:
        _cache[key] = _camb(**k)
    return _cache[key]


if det == 0.0:
    cnu.get_camb_cosmology = _cached


def make_dv(AB=1.0, camb_boost=1.0):
    """nw.get_dv with the CAMB boost exposed (get_dv pins it to 1)."""
    def dv(param, AccuracyBoost=None):
        nw._set_state(param[4], param[3], param[2], param[1], param[0],
                      -1.0, -1.0, AB, 5.0, 10, camb_boost, AB, 0, EMUL,
                      M=list(param[12:17]),
                      shear_photoz_bias=list(param[7:12]),
                      A1=[param[5], param[6], 0, 0, 0],
                      A2=nw.A2_FID, BTA=nw.BTA_FID)
        return np.array(ci.compute_data_vector_masked(), dtype=np.float64)
    return dv


def fom(D):
    F = cnu.add_gaussian_priors(F=D.T @ (invcov @ D),
                                priors=nw.FISHER_GAUSSIAN_PRIORS)
    return cnu.get_FoM(0, 4, F), np.sqrt(np.diag(np.linalg.inv(F)))[[0, 4]]


def fisher_D(dv, h, cosmo_dv=None):
    cols = []
    for p in range(len(CV)):
        f = cosmo_dv if (cosmo_dv is not None and p < 5) else dv
        cols.append(cnu.get_ddv(f, index=p, h=h, CV=CV, AccuracyBoost=1.0))
    return np.column_stack(cols)


# ---- (c) synthetic tables for the five cosmological parameters ------------
dv1 = make_dv(1.0, 1.0)
dv1(CV)                                   # fiducial state and nuisances
(lk, z2, lnPL0, lnPN0, G0, zG, z1, chi0) = cnu.get_camb_cosmology(**kw)
lk, z2, z1, G0, chi0 = map(np.asarray, (lk, z2, z1, G0, chi0))
POINT = {k: kw[k] for k in ("omegam", "omegab", "H0", "ns", "As_1e9", "w",
                            "w0pwa", "mnu")}
CAMBKW = {k: kw[k] for k in ("AccuracyBoost", "kmax", "k_per_logint",
                             "CAMBAccuracyBoost", "CLAccuracyBoost",
                             "non_linear_emul")}
NAMES = ["As_1e9", "ns", "H0", "omegab", "omegam"]
RESP = {i: get_camb_response(NAMES[i], 0.02*CV[i], POINT, **CAMBKW)
        for i in range(5)}


def synthetic_dv(param, AccuracyBoost=None):
    moved = [i for i in range(5) if param[i] != CV[i]]
    assert len(moved) == 1, moved
    i = moved[0]
    t = param[i] - CV[i]
    r = RESP[i]
    ci.set_cosmology(omegam=CV[4]*(1 + t*r["dlnOm_dX"]), H0=CV[2],
                     log10k_2D=lk, z_2D=z2, lnP_linear=lnPL0,
                     lnP_nonlinear=lnPN0 + t*r["dlnPNL_dX"],
                     G=G0*np.exp(t*r["dlnG_dX"]), z_G=zG, z_1D=z1,
                     chi=chi0 + t*r["dchi_dX"])
    return np.array(ci.compute_data_vector_masked(), dtype=np.float64)


# (d) is case (a) run with a CAMB whose halofit nonlinear-scale bisection
# tolerance is tightened (fortran/halofit.f90: 0.001 -> 1e-7), imported
# first through PYTHONPATH; the label is set by FISHER_CAMB_LABEL
cases = {"a": ("(a) notebook, AB=1 " + os.environ.get("FISHER_CAMB_LABEL", ""),
               dv1, None),
         "b": ("(b) CAMB boost 3, cosmolike AB=1", make_dv(1.0, 3.0), None),
         "c": ("(c) cosmolike only (synthetic)", dv1, synthetic_dv)}
for key in os.environ.get("FISHER_CASES", "abc"):
    lab, dv, cdv = cases[key]
    print(f"\n{lab}:  h -> FoM(As, Om)   sigma(As)  sigma(Om)", flush=True)
    for h in HS:
        t0 = time.perf_counter()
        dv1(CV)                           # restore fiducial nuisances
        f, sig = fom(fisher_D(dv, h, cdv))
        print(f"   h={h:.2f}: FoM {f:9.2f}   {sig[0]:.4e}  {sig[1]:.4e}"
              f"   ({time.perf_counter() - t0:.0f} s)", flush=True)
