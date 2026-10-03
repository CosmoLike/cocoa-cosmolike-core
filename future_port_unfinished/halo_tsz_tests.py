"""Tests of p_my and p_yy (future_port_unfinished/halo_tsz.c), kept with the
functions. They lived in projects/roman_real/tests: to restore, put the
constant and the method back into test_halo.py (the method belongs to the
class that holds test_p_gg_two_halo), add "p_my"/"p_yy" to SLOW_PROBES, to the
probe-point table (spectra) and to EVALUATORS (_spectrum(ci.p_my, x)), and
probe them in test_halo_cache_consistency.py halo_probes (with the gas steps
on the bound-gas slope and M0, GAS_DELTA = {1: 0.02, 2: 0.05}).
"""

# p_my^2 <= p_mm p_yy holds node by node (Cauchy-Schwarz on the mass
# integrals); bilinear interpolation of the three ln P tables with
# shared weights preserves it, up to rounding.
CAUCHY_SCHWARZ_RTOL = 1.0e-10


class _Restore:
    @slow
    def test_p_my_cauchy_schwarz(self, halo):
        """A cross spectrum is bounded by its autos: p_my^2 <= p_mm p_yy
        (Cauchy-Schwarz on each mass integral), and p_yy > 0."""
        ci = halo["ci"]
        k = np.asarray(PK_K, dtype=float)
        for a in P_2H_A:
            pmm = np.ravel(ci.p_mm(k=k, a=a))
            pmy = np.ravel(ci.p_my(k=k, a=a))
            pyy = np.ravel(ci.p_yy(k=k, a=a))
            assert np.all(pyy > 0), f"a={a}: p_yy = {pyy}"
            assert np.all(pmy**2 <= pmm*pyy*(1.0 + CAUCHY_SCHWARZ_RTOL)), (
                f"a={a}: p_my^2/(p_mm p_yy) = {pmy**2/(pmm*pyy)}")


# =============================================================================
# The gas-profile (u_KS) tests and settings of projects/roman_real/tests/
# test_halo.py, kept with the gas section (halo_tsz.c). To restore: put the
# constants and ks_u_reference back at module level, the three methods into
# TestPhysicsInvariants (the u_KS ones) and TestCacheConsistency
# (test_gas_round_trip), "u_KS" into the probe-point table and EVALUATORS,
# ci.set_nuisance_gas(gas=np.array(GAS_PARAMS, dtype=float)) into
# apply_halo_parameters and "gas": list(GAS_PARAMS) into frozen_meta (the
# stored frozen/halo_reference.json still carries both); test_hod_cell.py set
# the same gas parameters; test_halo_cache_consistency.py had a gas sector
# (GAS_FIDUCIAL = GAS_PARAMS, GAS_DELTA = {0: 0.02} probing u_KS at
# c = 2, 5, 10, rv = 3e-4).
# =============================================================================
# Gas (Compton-y) parameters in the structs.h layout. A representative
# physical point, not a fit: Gamma = 1.17 (the Komatsu-Seljak exponent
# 1/(Gamma - 1) needs Gamma > 1), beta = 0.6, lg M_0 = 14,
# eps1 = eps2 = 0 (unread by halo.c), alpha = 1 (bound gas at the
# virial temperature), A_star = 0.03, lg M_star = 12.5,
# sigma_star = 1.2, lg T_w = 6.5, f_H = 0.752 (the primordial hydrogen
# mass fraction).
GAS_PARAMS = (1.17, 0.6, 14.0, 0.0, 0.0, 1.0, 0.03, 12.5, 1.2, 6.5, 0.752)


U_KS_C = (2.0, 5.0, 10.0)       # inside [halo_uks_cmin, halo_uks_cmax]
U_KS_RV = 3.0e-4                # c/H0 (0.9 Mpc/h)
U_KS_K = np.logspace(0.0, 5.0, 6)
# u_KS against the real-axis integral (ks_u_reference): concentrations
# across the halo range and z = k r_v on both sides of the table switch
# at z = 3 (a table of u below, the contour factorization above).
U_KS_REF_C = (0.3, 1.0, 3.0, 5.0, 8.0, 12.0, 25.0)
U_KS_REF_Z = (1.0e-3, 0.5, 2.9, 3.1, 10.0, 50.0, 200.0, 800.0)

# u_KS <= u_KS(k -> 0) <= 1 holds node by node; allow the rounding of
# the table and of the Gauss-Legendre sums.
U_KS_BOUND_ATOL = 1.0e-8
# Measured 2026-09-29 against the gas study's reference at 20000 random
# (c, z): error <= 4.8e-6 of the local envelope of u, 1.8e-5 relative
# where |u| > 1e-2. Near the zeros of the ringing u only the absolute
# scale is meaningful, set by the plateau u(c, z -> 0).
U_KS_REF_RTOL = 1.0e-4
U_KS_REF_ATOL = 2.0e-5  # times u(c, z -> 0)
U_KS_K0 = 1.0e-3        # (c/H0)^-1; k rv/c ~ 1e-7, inside the table

GAS_GAMMA_STEP = 0.03

def ks_u_reference(c, z, gamma):
    """The KS pressure shape u = F/F0 on the real axis, independent of the
    complex-contour method halo.c uses above z = 3:

      F0 = int_0^c x^2 theta^q dx,  F = int_0^c x sin(y x)/y theta^p dx,
      theta = ln(1 + x)/x, p = gamma/(gamma - 1), q = 1/(gamma - 1),
      y = z/c.

    Composite Gauss-Legendre (24 nodes per panel), panels no wider than
    0.5 or a quarter period of sin(y x), so every oscillation is resolved.

    Arguments:
      c = concentration, z = k r_v, gamma = the polytropic index.

    Returns:
      u(c, z).
    """
    p, q = gamma/(gamma - 1.0), 1.0/(gamma - 1.0)
    y = z/c
    t, w = np.polynomial.legendre.leggauss(24)
    width = min(0.5, 0.5*np.pi/y)
    edges = np.linspace(0.0, c, max(1, int(np.ceil(c/width))) + 1)
    half = 0.5*np.diff(edges)
    x = (half[:, None]*(t + 1.0)[None, :] + edges[:-1, None]).ravel()
    wx = (half[:, None]*w[None, :]).ravel()
    th = np.log1p(x)/x
    return np.sum(wx*x*np.sin(y*x)/y*th**p)/np.sum(wx*x*x*th**q)



    """Write the pinned HOD (every lens bin) and gas parameters.

    ci.set_nuisance_gas(gas=np.array(GAS_PARAMS, dtype=float))

        "u_KS": {"c": floats(U_KS_C), "k": floats(U_KS_K),
                 "rv": [U_KS_RV]},

    "u_KS": lambda ci, x: [ci.u_KS(c=c, k=k, rv=rv)
                           for rv in x["rv"] for c in x["c"]
                           for k in x["k"]],

    and pins the HOD and gas parameters.

        "gas": list(GAS_PARAMS),

                    "settings (configuration, HOD, gas or defect list); "

    def test_u_KS_bounded(self, halo):
        """0 < u_KS(k) <= u_KS(k -> 0) <= 1: the pressure profile is a
        positive function (|sin z| <= z bounds its transform by the k = 0
        value), and the pressure-to-density normalization ln(1+x)/x <= 1
        keeps the k = 0 value at or below 1."""
        ci = halo["ci"]
        for c in U_KS_C:
            u0 = ci.u_KS(c=c, k=U_KS_K0, rv=U_KS_RV)
            assert 0.0 < u0 <= 1.0 + U_KS_BOUND_ATOL, f"u_KS(k->0) = {u0}"
            for k in U_KS_K:
                value = ci.u_KS(c=c, k=float(k), rv=U_KS_RV)
                assert value <= u0 + U_KS_BOUND_ATOL, (
                    f"u_KS(c={c}, k={k:.2e}) = {value} > u_KS(k->0)")

    def test_u_KS_matches_real_axis_integral(self, halo):
        """u_KS against ks_u_reference, the real-axis integral: checks the
        small-z table, the contour factorization above z = 3 and the
        switch between them."""
        ci = halo["ci"]
        gamma = GAS_PARAMS[0]
        for c in U_KS_REF_C:
            plateau = ks_u_reference(c, 1.0e-9, gamma)
            for z in U_KS_REF_Z:
                np.testing.assert_allclose(
                    ci.u_KS(c=c, k=z/U_KS_RV, rv=U_KS_RV),
                    ks_u_reference(c, z, gamma), rtol=U_KS_REF_RTOL,
                    atol=U_KS_REF_ATOL*plateau,
                    err_msg=f"u_KS(c={c}, z={z})")

    # ---- mass function and bias kernels ------------------------------------

    (cosmology.random, Ntable.random, nuisance.random_galaxy_bias,
    nuisance.random_gas). Each test

    def test_gas_round_trip(self, halo):
        """Gamma -> Gamma + GAS_GAMMA_STEP -> back, through the gas
        setter; the u_KS table rebuilds on nuisance.random_gas."""
        ci = halo["ci"]
        x = halo["inputs"]["u_KS"]
        before = np.array(evaluate_probe(ci, "u_KS", x))
        moved_gas = np.array(GAS_PARAMS, dtype=float)
        moved_gas[0] += GAS_GAMMA_STEP
        try:
            ci.set_nuisance_gas(gas=moved_gas)
            moved = np.array(evaluate_probe(ci, "u_KS", x))
        finally:
            apply_halo_parameters(halo)
        after = np.array(evaluate_probe(ci, "u_KS", x))
        assert _max_relative_change(moved, before) > CACHE_CHANGE_FLOOR
        assert np.array_equal(after, before)

halo bias (Tinker et al. 2010), halo concentrations (Bhattacharya et
al. 2013) and density profiles (NFW), the gas pressure profile
(Komatsu-Seljak), the HOD galaxy integrals, and the power spectra

cosmic-shear example at its fiducial point, with the HOD and gas
parameters pinned below:
