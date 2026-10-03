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
