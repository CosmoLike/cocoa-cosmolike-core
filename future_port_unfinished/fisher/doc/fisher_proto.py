"""Analytic Limber derivatives of C_ss from Python-side response inputs.

For a cosmological parameter X the chain rule needs only four inputs:
  R_X(k,z)  = dlnP_NL/dX at fixed (k [h/Mpc], z)      (2D table)
  chi_X(z)  = dchi/dX at fixed z  [Mpc/h]              (1D table)
  gam_X(z)  = dlnG/dX at fixed z (G the growth table)  (1D table)
  eps_X     = dlnOmega_m/dX (explicit Omega_m factors) (scalar)
Everything else is analytic: E = (c/H0)/chi'(z) and its response come from
cubic-spline derivatives of chi and chi_X; dlnP/dlnk from a bicubic spline.
The inputs are central differences of the CAMB tables on FIXED grids (what
the Python side of cosmolike would hand over); the truth is the central
difference of the full C_ss with the same step. Also reported: the best
table-only response models (no input), to show why the inputs are needed.

Writes fisher_proto_results.json and two figures for the PDF.
"""
import json
import numpy as np
import camb
from camb import model
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.integrate import solve_ivp, cumulative_trapezoid

H0 = 67.32; h = H0/100.0; OB = 0.04; MNU = 0.06; NS = 0.96605
CH0 = 2997.92458                     # c/H0 in Mpc/h
FID = dict(om=0.30, lnas=np.log(2.1e-9), w=-1.0)
A1, ETA, Z0 = 0.6, -1.5, 0.62        # NLA amplitude, redshift power law
C1RHO = 0.0134

ZT = np.linspace(0.0, 4.0, 161)      # P / growth table z nodes
LK = np.linspace(-3.0, 1.8, 481)     # log10 k [h/Mpc]
ZG = np.linspace(0.0, 4.0, 1601)     # chi table z nodes
ZC = np.linspace(0.02, 4.0, 1593)    # line-of-sight integration nodes
ELLS = np.array([20., 30., 50., 100., 200., 300., 500., 1000., 2000., 3000.])
PAIRS = ((0, 0), (1, 1), (0, 1))
K0 = 5e-4/h                          # growth-table scale (cosmolike: 5e-4/Mpc)


def run(om, lnas, w):
    omch2 = (om - OB)*h**2 - MNU*(3.046/3)**0.75/94.0708
    pars = camb.set_params(H0=H0, ombh2=OB*h**2, omch2=omch2, mnu=MNU,
                           omk=0, tau=0.06, As=np.exp(lnas), ns=NS,
                           halofit_version='takahashi', lmax=10,
                           num_massive_neutrinos=1, nnu=3.046,
                           k_per_logint=20, kmax=40.0)
    pars.set_dark_energy(w=w, wa=0.0, dark_energy_model='ppf')
    pars.NonLinear = model.NonLinear_both
    pars.set_matter_power(redshifts=list(ZT), kmax=40.0, silent=True)
    res = camb.get_results(pars)
    kw = dict(var1='delta_tot', var2='delta_tot', hubble_units=True,
              k_hunit=True, extrap_kmax=500.0)
    PL = res.get_matter_power_interpolator(nonlinear=False, **kw)
    PN = res.get_matter_power_interpolator(nonlinear=True, **kw)
    lnPL = np.log(PL.P(ZT, 10**LK))
    lnPN = np.log(PN.P(ZT, 10**LK))
    lnG = 0.5*np.log(PL.P(ZT, K0)).ravel()            # lnD up to a constant
    chi = res.comoving_radial_distance(ZG)*h
    zE = np.linspace(0.0, 60.0, 6001)
    E = res.hubble_parameter(zE)/H0
    return dict(lnPL=lnPL, lnPN=lnPN, lnG=lnG, chi=chi, zE=zE, E=E)


def nz_bins():
    out = []
    for zc in (0.5, 1.1):
        s = 0.12*(1 + zc)
        n = np.exp(-0.5*((ZC - zc)/s)**2)
        out.append(n/np.trapz(n, ZC))
    return out


NZ = nz_bins()


def lens_parts(chi, n, weight=None):
    """P(z) - chi Q(z) factorization: returns g, Q, and (optionally) the
    cumulative integral of n*weight from z to infinity."""
    def tail(f):
        return np.trapz(f, ZC) - cumulative_trapezoid(f, ZC, initial=0.0)
    Pc, Qc = tail(n), tail(n/chi)
    g = Pc - chi*Qc
    return g, Qc, (tail(n*weight) if weight is not None else None)


def state(c, om):
    """Geometry, growth, kernels, and P spline of one cosmology."""
    cs = CubicSpline(ZG, c['chi'])
    chi, chip = cs(ZC), cs(ZC, 1)
    E = CH0/chip                                  # H/H0 from the spline slope
    Gs = CubicSpline(ZT, c['lnG'])
    D = np.exp(Gs(ZC) - Gs(0.0))                  # growth, D(z=0) = 1
    C1 = A1*((1 + ZC)/(1 + Z0))**ETA*om*C1RHO/D   # NLA amplitude
    q = []
    for n in NZ:
        g = lens_parts(chi, n)[0]
        WK = 1.5*om*chi*(1 + ZC)*g/CH0**2
        WS = n*E/CH0
        q.append(WK - WS*C1)
    return dict(cs=cs, chi=chi, chip=chip, E=E, C1=C1, q=q,
                PN=RectBivariateSpline(ZT, LK, c['lnPN'], kx=3, ky=3),
                PL=RectBivariateSpline(ZT, LK, c['lnPL'], kx=3, ky=3))


def cls(c, om):
    s = state(c, om)
    out = {}
    for ell in ELLS:
        lk = np.log10((ell + 0.5)/s['chi'])
        w = (CH0/s['E'])/s['chi']**2*np.exp(s['PN'].ev(ZC, lk))
        for (i, j) in PAIRS:
            out[(ell, i, j)] = np.trapz(w*s['q'][i]*s['q'][j], ZC)
    return out


def inputs(cp, cm, step):
    """The four Python-side inputs: central differences on fixed grids."""
    return dict(R=(cp['lnPN'] - cm['lnPN'])/(2*step),
                chiX=(cp['chi'] - cm['chi'])/(2*step),
                lnGX=(cp['lnG'] - cm['lnG'])/(2*step))


def analytic(c, om, inp, epsX):
    """dC/dX by differentiating under the integral (fixed z nodes)."""
    s = state(c, om)
    chi, chip, E, C1 = s['chi'], s['chip'], s['E'], s['C1']
    xs = CubicSpline(ZG, inp['chiX'])
    chiX, chiXp = xs(ZC), xs(ZC, 1)               # cubic-spline derivative
    dlnchip = chiXp/chip                          # dln(dchi/dz)/dX
    Gx = CubicSpline(ZT, inp['lnGX'])
    gam = Gx(ZC) - Gx(0.0)                        # dlnD/dX, D(0) = 1
    dC1 = C1*(epsX - gam)
    dq = []
    for b, n in enumerate(NZ):
        g, Q, Rt = lens_parts(chi, n, weight=chiX/chi**2)
        gX = -chiX*Q + chi*Rt
        WK = 1.5*om*chi*(1 + ZC)*g/CH0**2
        dWK = WK*(epsX + chiX/chi) + 1.5*om*chi*(1 + ZC)*gX/CH0**2
        WS = n*E/CH0
        dWS = -WS*dlnchip                         # W_source ~ E = c/(H0 chi')
        dq.append(dWK - dWS*C1 - WS*dC1)
    RX = RectBivariateSpline(ZT, LK, inp['R'], kx=3, ky=3)
    dlnw = dlnchip - 2*chiX/chi                   # measure dchi/dz / chi^2
    out = {}
    for ell in ELLS:
        lk = np.log10((ell + 0.5)/chi)
        P = np.exp(s['PN'].ev(ZC, lk))
        nk = s['PN'].ev(ZC, lk, dy=1)/np.log(10.0)   # dlnP/dlnk (bicubic)
        w = (CH0/E)/chi**2*P
        lam = dlnw - nk*chiX/chi + RX.ev(ZC, lk)
        for (i, j) in PAIRS:
            qi, qj = s['q'][i], s['q'][j]
            C = np.trapz(w*qi*qj, ZC)
            dC = np.trapz(w*(lam*qi*qj + dq[i]*qj + qi*dq[j]), ZC)
            out[(ell, i, j)] = dC/C
    return out


# ---- table-only response models (no input), for comparison -----------------
def growth_sens(c, om):
    Es = CubicSpline(np.log(1/(1 + c['zE']))[::-1], c['E'][::-1])

    def rhs(x, y):
        a = np.exp(x); E = Es(x); E2 = E*E
        dE2 = a**-3 - 1.0
        ddlnE = 0.5*(-3*a**-3/E2 - 2*E*Es(x, 1)*dE2/E2**2)
        Om = om*a**-3/E2
        dOm = a**-3/E2 - om*a**-3*dE2/E2**2
        D, Dp, S, Sp = y
        return [Dp, -(2 + Es(x, 1)/E)*Dp + 1.5*Om*D,
                Sp, -(2 + Es(x, 1)/E)*Sp + 1.5*Om*S - ddlnE*Dp + 1.5*dOm*D]
    x0 = np.log(1/51.)
    sol = solve_ivp(rhs, (x0, 0.0), [np.exp(x0)]*2 + [0.0, 0.0],
                    dense_output=True, rtol=1e-10, atol=1e-13)
    D, _, S, _ = sol.sol(np.log(1/(1 + ZT)))
    return om*S/D


def table_only(c, om):
    PN = RectBivariateSpline(ZT, LK, c['lnPN'], kx=3, ky=3)
    PL = RectBivariateSpline(ZT, LK, c['lnPL'], kx=3, ky=3)
    Z, L = np.meshgrid(ZT, LK, indexing='ij')
    RAs = PN.ev(Z, L, dx=1)/PL.ev(Z, L, dx=1)
    nk = PN.ev(Z, L, dy=1)/np.log(10.0)
    gD = growth_sens(c, om)
    ROm = (RAs*(-2 + 2*gD[:, None] + NS + 3) - (3 + nk))/om
    return RAs, ROm


if __name__ == "__main__":
    steps = dict(lnas=0.01, om=0.005, w=0.02)
    eps = dict(lnas=0.0, om=1.0/FID['om'], w=0.0)
    fid = run(**FID)
    C0 = cls(fid, FID['om'])
    results = {}
    for X, dX in steps.items():
        pp, pm = dict(FID), dict(FID)
        pp[X] += dX; pm[X] -= dX
        cp, cm = run(**pp), run(**pm)
        Cp, Cm = cls(cp, pp['om']), cls(cm, pm['om'])
        an = analytic(fid, FID['om'], inputs(cp, cm, dX), eps[X])
        res = {}
        for key in C0:
            fd = (np.log(Cp[key]) - np.log(Cm[key]))/(2*dX)
            res[f"{key[0]:.0f}_{key[1]}{key[2]}"] = dict(fd=fd, an=an[key])
        # table-only response models (A_s and Omega_m only)
        if X in ("lnas", "om"):
            RAs, ROm = table_only(fid, FID['om'])
            inp = inputs(cp, cm, dX)
            inp['R'] = RAs if X == "lnas" else ROm
            tm = analytic(fid, FID['om'], inp, eps[X])
            for key in C0:
                res[f"{key[0]:.0f}_{key[1]}{key[2]}"]['table_only'] = tm[key]
        results[X] = res
        worst = max(abs(v['an']/v['fd'] - 1) for v in res.values())
        print(f"{X:5s}: worst |analytic/FD - 1| over ells x pairs = {worst:.2e}")
    json.dump(results, open("fisher_proto_results.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lab = dict(lnas=r"$\ln A_s$", om=r"$\Omega_m$", w=r"$w_0$")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a_, X in zip(ax, ("lnas", "om", "w")):
        for (i, j), mk in zip(PAIRS, ("o", "s", "^")):
            r = [results[X][f"{l:.0f}_{i}{j}"] for l in ELLS]
            a_.semilogx(ELLS, [v['an']/v['fd'] - 1 for v in r], mk + "-",
                        ms=4, label=f"bins ({i},{j})")
        a_.axhline(0, color="k", lw=0.6)
        a_.set_title(r"$d\ln C/d$" + lab[X])
        a_.set_xlabel(r"$\ell$")
    ax[0].set_ylabel("analytic / finite difference $-$ 1")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fisher_proto_exact.pdf")
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.4), sharey=True)
    for a_, X in zip(ax, ("lnas", "om")):
        for (i, j), mk in zip(PAIRS, ("o", "s", "^")):
            r = [results[X][f"{l:.0f}_{i}{j}"] for l in ELLS]
            a_.semilogx(ELLS, [v['table_only']/v['fd'] - 1 for v in r],
                        mk + "-", ms=4, label=f"bins ({i},{j})")
        a_.axhline(0, color="k", lw=0.6)
        a_.set_title(r"table-only response, $d\ln C/d$" + lab[X])
        a_.set_xlabel(r"$\ell$")
    ax[0].set_ylabel("model / finite difference $-$ 1")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fisher_proto_tableonly.pdf")
