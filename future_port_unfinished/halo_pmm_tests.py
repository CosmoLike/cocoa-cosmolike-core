"""Tests of p_mm (future_port_unfinished/halo_pmm.c), kept with the function.
They lived in projects/roman_real/tests: to restore, put the method back into
test_halo.py (the class that holds test_p_gm_two_halo_limit; it reads
P_2H_K_HMPC, P_2H_A, P_2H_RTOL and COVERH0 there), add "p_mm" to SLOW_PROBES,
to the probe-point table ("p_mm": spectra) and to EVALUATORS
(_spectrum(ci.p_mm, x)), and probe it in test_halo_cache_consistency.py
halo_probes (ci.p_mm(k=k, a=a) for a in PROBE_A, k in PROBE_K). The frozen
reference frozen/halo_reference.json still holds its p_mm values.
"""


class _Restore:
    @slow
    def test_p_mm_two_halo_limit(self, halo):
        """On large scales P_mm -> P_lin: the 2-halo term is
        I_m(k)^2 P_lin with I_m -> 1 (the HMx additive correction of
        2005.00009 App. A puts the matter of halos below M_min at M_min,
        so I_m(k -> 0) = bias_norm + (1 - bias_norm) = 1), and the
        1-halo term is small."""
        ci = halo["ci"]
        for a in P_2H_A:
            for k_h in P_2H_K_HMPC:
                k = k_h*COVERH0
                ratio = ci.p_mm(k=k, a=a)/ci.p_lin(k=k, a=a)
                assert abs(ratio - 1.0) < P_2H_RTOL, (
                    f"p_mm/p_lin(k={k_h} h/Mpc, a={a}) = {ratio}")

