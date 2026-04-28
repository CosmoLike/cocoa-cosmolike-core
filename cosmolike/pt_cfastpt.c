#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "cfastpt/utils_cfastpt.h"
#include "cfastpt/utils_complex_cfastpt.h"
#include "cfastpt/cfastpt.h"
#include "basics.h"
#include "cosmo3D.h"
#include "pt_cfastpt.h"
#include "structs.h"

#include "log.c/src/log.h"

void get_FPT_bias(void) 
{
  static uint64_t cache[MAX_SIZE_ARRAYS];

  if (fdiff2(cache[1], Ntable.random))
  {
    FPTbias.k_min     = 0.05;
    FPTbias.k_max     = 1.0e+6;
    FPTbias.k_cutoff  = 1.0e+4;
    FPTbias.N         = 1100 + 200 * Ntable.FPTboost;
    FPTbias.sigma4    = 0.0;
    if (FPTbias.tab != NULL) {
      free(FPTbias.tab);
    }
    FPTbias.tab = (double**) malloc2d(7, FPTbias.N);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    const double dlogk = (log(FPTbias.k_max) - log(FPTbias.k_min))/FPTbias.N;

    #pragma omp parallel for
    for (int i=0; i<FPTbias.N; i++) 
    {
      FPTbias.tab[6][i] = exp(log(FPTbias.k_min) + i*dlogk);
      FPTbias.tab[7][i] = p_lin(FPTbias.tab[6][i], 1.0);
    }

    double Pout[5][FPTbias.N];
    Pd1d2(FPTbias.tab[6], FPTbias.tab[7], FPTbias.N, Pout[0]);
    Pd2d2(FPTbias.tab[6], FPTbias.tab[7], FPTbias.N, Pout[1]);
    Pd1s2(FPTbias.tab[6], FPTbias.tab[7], FPTbias.N, Pout[2]);
    Pd2s2(FPTbias.tab[6], FPTbias.tab[7], FPTbias.N, Pout[3]);
    Ps2s2(FPTbias.tab[6], FPTbias.tab[7], FPTbias.N, Pout[4]);

    #pragma omp parallel for
    for (int i=0; i<FPTbias.N; i++) 
    {
      FPTbias.tab[0][i] = Pout[0][i]; // Pd1d2
      FPTbias.tab[1][i] = Pout[1][i]; // Pd2d2
      FPTbias.tab[2][i] = Pout[2][i]; // Pd1s2
      FPTbias.tab[3][i] = Pout[3][i]; // Pd2s2
      FPTbias.tab[4][i] = Pout[4][i]; // Ps2s2
      // (JX) Pd1p3: interpolated from precomputed table at a 
      //             mystery cosmology with sigma8 = 0.8
      double lnk = log(FPTbias.tab[6][i]);
      FPTbias.tab[5][i] = (lnk < tab_d1d3_lnkmin || lnk > tab_d1d3_lnkmax) ? 0.0 :
      interpol1d(tab_d1d3, tab_d1d3_Nk, tab_d1d3_lnkmin, tab_d1d3_lnkmax, tab_d1d3_dlnk, lnk);
    }
    FPTbias.sigma4 = FPTbias.tab[1][0]/2.; // JX: dirty fix for sigma4 term: P_{d2d2}(k->0) / 2
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}


static int Nmax_from_terms(int N, int (*terms)[5]) {
  int Nmax = 0;
  for (int i = 0; i < N; i++) {
    int l1 = terms[i][2], l2 = terms[i][3], l = terms[i][4];
    Nmax += (l1+l2 - abs(l1-l2) + 1) * (l1+l - abs(l1-l) + 1) * (l+l2 - abs(l-l2) + 1);
  }
  return Nmax;
}

void get_FPT_IA(void)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  if (fdiff2(cache[1], Ntable.random))
  {
    FPTIA.k_min    = 0.05;
    FPTIA.k_max    = 1.0e+6;
    FPTIA.k_cutoff = 1.0e+4;
    FPTIA.sigma4   = 0.0;
    FPTIA.N        = 1100 + 200 * Ntable.FPTboost;
    if (FPTIA.tab != NULL) {
      free(FPTIA.tab);
    }
    FPTIA.tab = (double**) malloc2d(12, FPTIA.N);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    const long Nk = FPTIA.N;
    double *k   = FPTIA.tab[10];
    double *Pin = FPTIA.tab[11];

    double lim[3];
    lim[0] = log(FPTIA.k_min);
    lim[1] = log(FPTIA.k_max);
    lim[2] = (lim[1] - lim[0]) / Nk;

    #pragma omp parallel for
    for (int i = 0; i < Nk; i++) {
      k[i]   = exp(lim[0] + i*lim[2]);
      Pin[i] = p_lin(k[i], 1.0);
    }

    // =====================================================================
    // MERGE ALL J_TABLE BLOCKS INTO ONE J_abJ1J2Jk_ar CALL
    //
    // Instead of calling IA_tt, IA_ta, IA_mix separately (3 calls to
    // J_abJ1J2Jk_ar with 3 separate extrapolation setups, 3 separate
    // alpha/beta FFT precomputations, and 3 separate OpenMP regions),
    // we expand all J_tables here, concatenate into one big array, and
    // make a single call. OpenMP then gets ~184 terms in one parallel
    // region for much better load balancing.
    //
    // Output mapping (8 groups → 8 output arrays):
    //   Group 0: IA_tt  E-mode   → FPTIA.tab[0]
    //   Group 1: IA_tt  B-mode   → FPTIA.tab[1]
    //   Group 2: IA_ta  deltaE1  → FPTIA.tab[2]
    //   Group 3: IA_ta  0E0E     → FPTIA.tab[4]
    //   Group 4: IA_ta  0B0B     → FPTIA.tab[5]
    //   Group 5: IA_mix A        → FPTIA.tab[6]
    //   Group 6: IA_mix D_EE     → FPTIA.tab[8]
    //   Group 7: IA_mix D_BB     → FPTIA.tab[9]
    //
    // The two direct convolution terms (IA_ta deltaE2, IA_mix B) don't
    // use J_abJ1J2Jk_ar and are computed separately below.
    // =====================================================================

    #define NGROUPS 8

    // --- Define all term tables ---

    // Group 0: IA_tt E-mode  {alpha, beta, l1, l2, l}
    int terms_tt_E[][5] = {
      {0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 4, 0, 0},
      {0, 0, 2, 2, 0}, {0, 0, 1, 1, 1}, {0, 0, 3, 1, 1},
      {0, 0, 0, 0, 2}, {0, 0, 2, 0, 2}, {0, 0, 2, 2, 2},
      {0, 0, 1, 1, 3}, {0, 0, 0, 0, 4},
    };
    double coeff_tt_E[] = {
      2*(16./81.),   2*(713./1134.), 2*(38./315.),
      2*(95./162.),  2*(-107./60.),  2*(-19./15.),
      2*(239./756.), 2*(11./9.),     2*(19./27.),
      2*(-7./10.),   2*(3./35.)
    };

    // Group 1: IA_tt B-mode
    int terms_tt_B[][5] = {
      {0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 4, 0, 0},
      {0, 0, 2, 2, 0}, {0, 0, 1, 1, 1}, {0, 0, 3, 1, 1},
      {0, 0, 0, 0, 2}, {0, 0, 2, 0, 2}, {0, 0, 2, 2, 2},
      {0, 0, 1, 1, 3},
    };
    double coeff_tt_B[] = {
      2*(-41./405.), 2*(-298./567.), 2*(-32./315.),
      2*(-40./81.),  2*(59./45.),    2*(16./15.),
      2*(-2./9.),    2*(-20./27.),   2*(-16./27.),
      2*(2./5.)
    };

    // Group 2: IA_ta deltaE1
    int terms_ta_dE1[][5] = {
      { 0,  0, 0, 2, 0}, { 0,  0, 0, 2, 2},
      { 1, -1, 0, 2, 1}, {-1,  1, 0, 2, 1},
    };
    double coeff_ta_dE1[] = {2*(17./21.), 2*(4./21.), 1., 1.};

    // Group 3: IA_ta 0E0E
    int terms_ta_0E0E[][5] = {
      {0, 0, 0, 0, 0}, {0, 0, 2, 0, 0},
      {0, 0, 2, 2, 0}, {0, 0, 0, 4, 0},
    };
    double coeff_ta_0E0E[] = {29./90., 5./63., 19./18., 19./35.};

    // Group 4: IA_ta 0B0B
    int terms_ta_0B0B[][5] = {
      {0, 0, 0, 0, 0}, {0, 0, 2, 0, 0},
      {0, 0, 2, 2, 0}, {0, 0, 0, 4, 0},
      {0, 0, 1, 1, 1},
    };
    double coeff_ta_0B0B[] = {2./45., -44./63., -8./9., -16./35., 2.};

    // Group 5: IA_mix A
    int terms_mix_A[][5] = {
      { 0,  0, 0, 0, 0}, { 0,  0, 2, 0, 0},
      { 0,  0, 0, 0, 2}, { 0,  0, 2, 0, 2},
      { 0,  0, 1, 1, 1}, { 0,  0, 1, 1, 3},
      { 0,  0, 0, 0, 4},
      { 1, -1, 0, 0, 1}, { 1, -1, 2, 0, 1},
      { 1, -1, 1, 1, 0}, { 1, -1, 1, 1, 2},
      { 1, -1, 0, 2, 1}, { 1, -1, 0, 0, 3},
    };
    double coeff_mix_A[] = {
      2*(-31./210.), 2*(-34./63.), 2*(-47./147.), 2*(-8./63.),
      2*(93./70.),   2*(6./35.),   2*(-8./245.),
      2*(-3./10.),   2*(-1./3.),   2*(1./2.),
      2*(1.),        2*(-1./3.),   2*(-1./5.)
    };

    // Group 6: IA_mix D_EE
    int terms_mix_DEE[][5] = {
      {0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 4, 0, 0},
      {0, 0, 0, 0, 2}, {0, 0, 2, 0, 2}, {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1}, {0, 0, 2, 2, 0},
    };
    double coeff_mix_DEE[] = {
      2*(-43./540.), 2*(-167./756.), 2*(-19./105.), 2*(1./18.),
      2*(-7./18.),   2*(11./20.),    2*(19./20.),   2*(-19./54.)
    };

    // Group 7: IA_mix D_BB
    int terms_mix_DBB[][5] = {
      {0, 0, 0, 0, 0}, {0, 0, 2, 0, 0}, {0, 0, 4, 0, 0},
      {0, 0, 0, 0, 2}, {0, 0, 2, 0, 2}, {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1}, {0, 0, 2, 2, 0},
    };
    double coeff_mix_DBB[] = {
      2*(13./135.), 2*(86./189.), 2*(16./105.), 2*(2./9.),
      2*(4./9.),    2*(-13./15.), 2*(-4./5.),   2*(8./27.)
    };

    // --- J_table expansion for each group ---
    #define NTERMS(arr) (int)(sizeof(arr) / sizeof(arr[0]))

    int N_in[] = {
      NTERMS(terms_tt_E),   NTERMS(terms_tt_B),
      NTERMS(terms_ta_dE1), NTERMS(terms_ta_0E0E), NTERMS(terms_ta_0B0B),
      NTERMS(terms_mix_A),  NTERMS(terms_mix_DEE),  NTERMS(terms_mix_DBB)
    };

    int Nmax[NGROUPS];
    Nmax[0] = Nmax_from_terms(N_in[0], terms_tt_E);
    Nmax[1] = Nmax_from_terms(N_in[1], terms_tt_B);
    Nmax[2] = Nmax_from_terms(N_in[2], terms_ta_dE1);
    Nmax[3] = Nmax_from_terms(N_in[3], terms_ta_0E0E);
    Nmax[4] = Nmax_from_terms(N_in[4], terms_ta_0B0B);
    Nmax[5] = Nmax_from_terms(N_in[5], terms_mix_A);
    Nmax[6] = Nmax_from_terms(N_in[6], terms_mix_DEE);
    Nmax[7] = Nmax_from_terms(N_in[7], terms_mix_DBB);

    // VLAs for J_table outputs
    int out0[Nmax[0]][5], out1[Nmax[1]][5], out2[Nmax[2]][5], out3[Nmax[3]][5];
    int out4[Nmax[4]][5], out5[Nmax[5]][5], out6[Nmax[6]][5], out7[Nmax[7]][5];
    double co0[Nmax[0]], co1[Nmax[1]], co2[Nmax[2]], co3[Nmax[3]];
    double co4[Nmax[4]], co5[Nmax[5]], co6[Nmax[6]], co7[Nmax[7]];
    int Nnew[NGROUPS];
    Nnew[0] = J_table(5, N_in[0], terms_tt_E,    coeff_tt_E,    out0, co0);
    Nnew[1] = J_table(5, N_in[1], terms_tt_B,    coeff_tt_B,    out1, co1);
    Nnew[2] = J_table(5, N_in[2], terms_ta_dE1,  coeff_ta_dE1,  out2, co2);
    Nnew[3] = J_table(5, N_in[3], terms_ta_0E0E, coeff_ta_0E0E, out3, co3);
    Nnew[4] = J_table(5, N_in[4], terms_ta_0B0B, coeff_ta_0B0B, out4, co4);
    Nnew[5] = J_table(5, N_in[5], terms_mix_A,   coeff_mix_A,   out5, co5);
    Nnew[6] = J_table(5, N_in[6], terms_mix_DEE, coeff_mix_DEE, out6, co6);
    Nnew[7] = J_table(5, N_in[7], terms_mix_DBB, coeff_mix_DBB, out7, co7);

    // --- Concatenate all groups into single arrays ---
    int Ntotal = 0;
    int starts[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) {
      starts[g] = Ntotal;
      Ntotal += Nnew[g];
    }

    int alpha_all[Ntotal], beta_all[Ntotal];
    int J1_all[Ntotal], J2_all[Ntotal], Jk_all[Ntotal];
    double coeff_all[Ntotal];

    // Pointers to the per-group J_table output arrays
    int (*out_ptrs[])[5] = {out0, out1, out2, out3, out4, out5, out6, out7};
    double *co_ptrs[]    = {co0,  co1,  co2,  co3,  co4,  co5,  co6,  co7};

    for (int g = 0; g < NGROUPS; g++) {
      int off = starts[g];
      for (int i = 0; i < Nnew[g]; i++) {
        alpha_all[off+i] = out_ptrs[g][i][0];
        beta_all[off+i]  = out_ptrs[g][i][1];
        J1_all[off+i]    = out_ptrs[g][i][2];
        J2_all[off+i]    = out_ptrs[g][i][3];
        Jk_all[off+i]    = out_ptrs[g][i][4];
        coeff_all[off+i] = co_ptrs[g][i];
      }
    }

    // --- Single J_abJ1J2Jk_ar call with all ~184 terms ---
    double *Fy_flat = malloc(sizeof(double) * Ntotal * Nk);
    double *Fy_ptrs[Ntotal];
    for (int i = 0; i < Ntotal; i++)
      Fy_ptrs[i] = Fy_flat + i * Nk;

    static const fastpt_config fpt_config = {
      .c_window_width = 0.65, .N_pad = 1500,
      .N_extrap_low = 500, .N_extrap_high = 500
    };
    J_abJ1J2Jk_ar(k, Pin, Nk, alpha_all, beta_all, J1_all, J2_all, Jk_all,
                   Ntotal, (fastpt_config*)&fpt_config, Fy_ptrs);

    // --- Accumulate results into the 8 output arrays ---
    // Group → output array mapping
    double *outputs[] = {
      FPTIA.tab[0], FPTIA.tab[1], FPTIA.tab[2], FPTIA.tab[4],
      FPTIA.tab[5], FPTIA.tab[6], FPTIA.tab[8], FPTIA.tab[9]
    };

    for (int g = 0; g < NGROUPS; g++) {
      memset(outputs[g], 0, sizeof(double) * Nk);
      for (int i = starts[g]; i < starts[g] + Nnew[g]; i++) {
        const double c = coeff_all[i];
        const double *row = Fy_flat + i * Nk;
        for (int j = 0; j < Nk; j++)
          outputs[g][j] += c * row[j];
      }
    }

    free(Fy_flat);

    #undef NTERMS
    #undef NGROUPS

    // =====================================================================
    // DIRECT CONVOLUTION TERMS (not part of J_abJ1J2Jk_ar)
    // =====================================================================

    // --- IA_ta deltaE2 term ---
    {
      double dL = log(k[1] / k[0]);
      long Ncut = floor(3. / dL);
      double exps[2*Nk-1], f[2*Nk-1];
      int i;

      for (i = 0; i < 2*Nk-1; i++)
        exps[i] = exp(-dL * (i - Nk + 1));

      for (i = 0; i < Nk-1-Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2;
        f[i] = r * (768./7 - 256/(7293.*r10) - 256/(3003.*r8)
                - 256/(1001.*r6) - 256/(231.*r4) - 256/(21.*r2));
      }
      for ( ; i < Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r, r6 = r4*r2, r7 = r6*r;
        f[i] = r * (30. + 146*r2 - 110*r4 + 30*r6
                + log(fabs(r-1.)/(r+1.)) * (15./r - 60.*r + 90*r3 - 60*r5 + 15*r7));
      }
      for (i = Nk; i < Nk-1+Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r, r6 = r4*r2, r7 = r6*r;
        f[i] = r * (30. + 146*r2 - 110*r4 + 30*r6
                + log(fabs(r-1.)/(r+1.)) * (15./r - 60.*r + 90*r3 - 60*r5 + 15*r7));
      }
      for ( ; i < 2*Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6, r14 = r8*r6;
        f[i] = r * (256*r2 - 256*r4 + (768*r6)/7.
                - (256*r8)/21. - (256*r10)/231.
                - (256*r12)/1001. - (256*r14)/3003.);
      }
      f[Nk-1] = 96.;

      double g[3*Nk-2];
      fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
      for (i = 0; i < Nk; i++) {
        double ki3 = k[i] * k[i] * k[i];
        FPTIA.tab[3][i] = 2. * ki3 / (896.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL;
      }
    }

    // --- IA_mix B term ---
    {
      double dL = log(k[1] / k[0]);
      long Ncut = floor(3. / dL);
      double exps[2*Nk-1], f[2*Nk-1];
      int i;

      for (i = 0; i < 2*Nk-1; i++)
        exps[i] = exp(-dL * (i - Nk + 1));

      for (i = 0; i < Nk-1-Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6;
        f[i] = r * (-16./147 - 16/(415701.*r12) - 32/(357357.*r10) - 16/(63063.*r8)
                - 64/(63063.*r6) - 16/(1617.*r4) + 32/(441.*r2)) / 2.;
      }
      for ( ; i < Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4;
        double rm1 = r2 - 1.;
        double rm1_2 = rm1*rm1, rm1_4 = rm1_2*rm1_2;
        f[i] = r * ((2. * r * (225. - 600.*r2 + 1198.*r4 - 600.*r6 + 225.*r8)
                + 225. * rm1_4 * (r2 + 1.) * log(fabs(r-1.)/(r+1.))) / (20160.*r3)
                - 29./315.*r2) / 2.;
      }
      for (i = Nk; i < Nk-1+Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4;
        double rm1 = r2 - 1.;
        double rm1_2 = rm1*rm1, rm1_4 = rm1_2*rm1_2;
        f[i] = r * ((2. * r * (225. - 600.*r2 + 1198.*r4 - 600.*r6 + 225.*r8)
                + 225. * rm1_4 * (r2 + 1.) * log(fabs(r-1.)/(r+1.))) / (20160.*r3)
                - 29./315.*r2) / 2.;
      }
      for ( ; i < 2*Nk-1; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6, r14 = r8*r6, r16 = r8*r8;
        f[i] = r * ((-16*r4)/147. + (32*r6)/441. - (16*r8)/1617.
                - (64*r10)/63063. - (16*r12)/63063. - (32*r14)/357357.
                - (16*r16)/415701.) / 2.;
      }
      f[Nk-1] = -1./42.;

      double g[3*Nk-2];
      fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
      for (i = 0; i < Nk; i++) {
        double ki3 = k[i] * k[i] * k[i];
        FPTIA.tab[7][i] = 4. * ki3 / (2.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL;
      }
    }

    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

