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

// ---------------------------------------------------------------------------
// get_FPT_bias: compute all 5 nonlinear galaxy bias spectra
// ---------------------------------------------------------------------------
//
// Computes the one-loop bias power spectra needed for nonlinear galaxy
// bias modeling: Pd1d2, Pd2d2, Pd1s2, Pd2s2, Ps2s2. These are the
// correlators between the density field (d), its square (d2), the tidal
// field squared (s2), and the third-order bias operator (p3).
//
// The combined version dispatches all 13 terms in a single J_abl_ar call,
// then accumulates the results with the appropriate coefficients:
//   Pd1d2(k) = sum_{terms 0,1,2} coeff[i] * Fy[i](k)
//   Pd2d2(k) = coeff[3] * Fy[3](k)
//   Pd1s2(k) = sum_{terms 4..8} coeff[i] * Fy[i](k)
//   ...etc
//
// FPTbias.tab layout (8 rows x FPTbias.N columns):
//   [0] Pd1d2   - density * density-squared correlator
//   [1] Pd2d2   - density-squared auto-correlator
//   [2] Pd1s2   - density * tidal-squared correlator
//   [3] Pd2s2   - density-squared * tidal-squared correlator
//   [4] Ps2s2   - tidal-squared auto-correlator
//   [5] Pd1p3   - density * third-order (from precomputed table, not FAST-PT)
//   [6] k       - wavenumber grid (log-spaced from k_min to k_max)
//   [7] P_lin   - linear power spectrum at z=0 (a=1)
//
// Caching: recomputed only when cosmology or Ntable settings change
// (tracked via cosmology.random and Ntable.random hash values).
// ---------------------------------------------------------------------------
void get_FPT_bias(void)
{
  // 13 terms -> 5 output spectra
  // ---------------------------------------------------------------
  // idx | alpha | beta | ell | output     | coeff
  // ----|-------|------|-----|------------|------------------
  //  0  |   0   |   0  |  0  | Pd1d2 [0]  | 2*(17/21)
  //  1  |   0   |   0  |  2  | Pd1d2 [0]  | 2*(4/21)
  //  2  |   1   |  -1  |  1  | Pd1d2 [0]  | 2
  //  3  |   0   |   0  |  0  | Pd2d2 [1]  | 2
  //  4  |   0   |   0  |  0  | Pd1s2 [2]  | 2*(8/315)
  //  5  |   0   |   0  |  2  | Pd1s2 [2]  | 2*(254/441)
  //  6  |   0   |   0  |  4  | Pd1s2 [2]  | 2*(16/245)
  //  7  |   1   |  -1  |  1  | Pd1s2 [2]  | 2*(4/15)
  //  8  |   1   |  -1  |  3  | Pd1s2 [2]  | 2*(2/5)
  //  9  |   0   |   0  |  2  | Pd2s2 [3]  | 2*(2/3)
  // 10  |   0   |   0  |  0  | Ps2s2 [4]  | 2*(4/45)
  // 11  |   0   |   0  |  2  | Ps2s2 [4]  | 2*(8/63)
  // 12  |   0   |   0  |  4  | Ps2s2 [4]  | 2*(8/35)
  // ---------------------------------------------------------------
  const int NTERMS = 13;
  const int NOUT   = 5;
  const int OUT_D2D2 = 1; // used for sigma4 below
 
  const int alpha_tab[13] = {0,0,1, 0, 0,0,0,1,1, 0, 0,0,0};
  const int beta_tab[13] = {0,0,-1, 0, 0,0,0,-1,-1, 0, 0,0,0};
  const int ell_tab[13] = {0,2,1, 0, 0,2,4,1,3, 2, 0,2,4};
  const int out_idx[13] = {0,0,0, 1, 2,2,2,2,2, 3, 4,4,4};
 
  const double coeff[13] = {
    2.*(17./21.), 2.*(4./21.), 2.,
    2.,
    2.*(8./315.), 2.*(254./441.), 2.*(16./245.), 2.*(4./15.), 2.*(2./5.),
    2.*(2./3.),
    2.*(4./45.), 2.*(8./63.), 2.*(8./35.)
  };
  static uint64_t cache[MAX_SIZE_ARRAYS];
 
  if (fdiff2(cache[1], Ntable.random))
  {
    FPTbias.k_min    = 0.05;
    FPTbias.k_max    = 1.0e+6;
    FPTbias.k_cutoff = 1.0e+4;
    FPTbias.N        = 1100 + 200 * Ntable.FPTboost;
    FPTbias.sigma4   = 0.0;
    if (FPTbias.tab != NULL)
    {
      free(FPTbias.tab);
    }
    FPTbias.tab = (double**) malloc2d(8, FPTbias.N);
  }
 
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random))
  {
    const long Nk = FPTbias.N;
    const double dlogk =
      (log(FPTbias.k_max) - log(FPTbias.k_min)) / Nk;
 
    // --- build k grid and linear P(k) ---
    #pragma omp parallel for
    for (int i = 0; i < Nk; i++) {
      FPTbias.tab[6][i] = exp(log(FPTbias.k_min) + i * dlogk);
      FPTbias.tab[7][i] = p_lin(FPTbias.tab[6][i], 1.0);
    }
 
    // single J_abl call with all 13 bias terms ---
    double **Fy = malloc(sizeof(double*) * NTERMS);
    for (int i = 0; i < NTERMS; i++) {
      Fy[i] = malloc(sizeof(double) * Nk);
    }
 
    int alpha_ar[13];
    int beta_ar[13];
    int ell_ar[13];
    int isP13type_ar[13];
    for (int i = 0; i < NTERMS; i++) {
      alpha_ar[i]     = alpha_tab[i];
      beta_ar[i]      = beta_tab[i];
      ell_ar[i]       = ell_tab[i];
      isP13type_ar[i] = 0;
    }
 
    fastpt_config config;
    config.nu             = -2.;
    config.c_window_width = 0.65;
    config.N_pad          = 1500;
    config.N_extrap_low   = 500;
    config.N_extrap_high  = 500;
 
    J_abl(FPTbias.tab[6], FPTbias.tab[7], Nk,
             alpha_ar, beta_ar, ell_ar, isP13type_ar,
             NTERMS, &config, Fy);
 
    // --- accumulate 13 Fy terms into 5 bias spectra ---
    for (int out = 0; out < NOUT; out++) {
      for (long j = 0; j < Nk; j++) {
        FPTbias.tab[out][j] = 0.;
      }
    }
    for (int i = 0; i < NTERMS; i++) {
      const int out = out_idx[i];
      const double c = coeff[i];
      #pragma omp parallel for
      for (long j = 0; j < Nk; j++) {
        FPTbias.tab[out][j] += c * Fy[i][j];
      }
    }
 
    for (int i = 0; i < NTERMS; i++) {
      free(Fy[i]);
    }
    free(Fy);
 
    // Pd1p3: interpolated from precomputed table
    #pragma omp parallel for
    for (int i = 0; i < Nk; i++)
    {
      const double lnk = log(FPTbias.tab[6][i]);
      FPTbias.tab[5][i] =
        (lnk < tab_d1d3_lnkmin || lnk > tab_d1d3_lnkmax) ? 0.0 :
        interpol1d(tab_d1d3, tab_d1d3_Nk, tab_d1d3_lnkmin,
                   tab_d1d3_lnkmax, tab_d1d3_dlnk, lnk);
    }
 
    FPTbias.sigma4 = FPTbias.tab[OUT_D2D2][0] / 2.;
 
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INTRINSIC ALIGMENT
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

enum {
  COL_ID_ALPHA = 0,
  COL_ID_BETA  = 1,
  COL_ID_L1    = 2,
  COL_ID_L2    = 3,
  COL_ID_L     = 4,
  NCOLS        = 5
};

// ---------------------------------------------------------------------------
// Nmax_from_terms: Upper bound on J_table output rows.
//
// Each input term (l1, l2, l) expands into a triple loop over (J1, J2, Jk),
// where each index is constrained by the triangle inequality for angular
// momentum coupling:
//   J1 ranges over |l  - l2| .. l  + l2  (couples l  and l2)
//   J2 ranges over |l1 - l|  .. l1 + l   (couples l1 and l)
//   Jk ranges over |l1 - l2| .. l1 + l2  (couples l1 and l2)
//
// The number of allowed values for each is (max - min + 1). Their product
// bounds how many (J1, J2, Jk) combinations each term can produce.
// Summing over all input terms gives the total upper bound.
// ---------------------------------------------------------------------------
static int Nmax_from_terms(int N, int (*terms)[NCOLS]) {
  int Nmax = 0;
  for (int i = 0; i < N; i++) {
    // assumed terms layout: {alpha, beta, l1, l2, l} → columns 0..4
    const int l1 = terms[i][COL_ID_L1];
    const int l2 = terms[i][COL_ID_L2];
    const int l  = terms[i][COL_ID_L];

    const int nJ1 = l  + l2 - abs(l  - l2) + 1;  // count of allowed J1
    const int nJ2 = l1 + l  - abs(l1 - l)  + 1;  // count of allowed J2
    const int nJk = l1 + l2 - abs(l1 - l2) + 1;  // count of allowed Jk

    Nmax += nJ1 * nJ2 * nJk;
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
    double *k   = FPTIA.tab[10];
    double *Pin = FPTIA.tab[11];

    double lim[3];
    lim[0] = log(FPTIA.k_min);
    lim[1] = log(FPTIA.k_max);
    lim[2] = (lim[1] - lim[0]) / FPTIA.N;

    #pragma omp parallel for
    for (int i = 0; i < FPTIA.N; i++) {
      k[i]   = exp(lim[0] + i*lim[2]);
      Pin[i] = p_lin(k[i], 1.0);
    }

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
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
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // Group 0: IA_tt E-mode, remind: {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_tt_E[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 2, 2, 2},
      {0, 0, 1, 1, 3},
      {0, 0, 0, 0, 4},
    };

    double coeff_tt_E[] = {
      2*(16./81.),   2*(713./1134.), 2*(38./315.),
      2*(95./162.),  2*(-107./60.),  2*(-19./15.),
      2*(239./756.), 2*(11./9.),     2*(19./27.),
      2*(-7./10.),   2*(3./35.)
    };

    // -------------------------------------------------------------------------
    // Group 1: IA_tt B-mode, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_tt_B[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 2, 2, 2},
      {0, 0, 1, 1, 3},
    };

    double coeff_tt_B[] = {
      2*(-41./405.), 2*(-298./567.), 2*(-32./315.),
      2*(-40./81.),  2*(59./45.),    2*(16./15.),
      2*(-2./9.),    2*(-20./27.),   2*(-16./27.),
      2*(2./5.)
    };

    // -------------------------------------------------------------------------
    // Group 2: IA_ta deltaE1, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_ta_dE1[][NCOLS] = {
      { 0,  0, 0, 2, 0},
      { 0,  0, 0, 2, 2},
      { 1, -1, 0, 2, 1},
      {-1,  1, 0, 2, 1},
    };

    double coeff_ta_dE1[] = {2*(17./21.), 2*(4./21.), 1., 1.};

    // -------------------------------------------------------------------------
    // Group 3: IA_ta 0E0E, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_ta_0E0E[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 0, 4, 0},
    };

    double coeff_ta_0E0E[] = {29./90., 5./63., 19./18., 19./35.};

    // -------------------------------------------------------------------------
    // Group 4: IA_ta 0B0B, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_ta_0B0B[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 2, 2, 0},
      {0, 0, 0, 4, 0},
      {0, 0, 1, 1, 1},
    };

    double coeff_ta_0B0B[] = {2./45., -44./63., -8./9., -16./35., 2.};

    // -------------------------------------------------------------------------
    // Group 5: IA_mix A, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_mix_A[][NCOLS] = {
      { 0,  0, 0, 0, 0},
      { 0,  0, 2, 0, 0},
      { 0,  0, 0, 0, 2},
      { 0,  0, 2, 0, 2},
      { 0,  0, 1, 1, 1},
      { 0,  0, 1, 1, 3},
      { 0,  0, 0, 0, 4},
      { 1, -1, 0, 0, 1},
      { 1, -1, 2, 0, 1},
      { 1, -1, 1, 1, 0},
      { 1, -1, 1, 1, 2},
      { 1, -1, 0, 2, 1},
      { 1, -1, 0, 0, 3},
    };

    double coeff_mix_A[] = {
      2*(-31./210.), 2*(-34./63.), 2*(-47./147.), 2*(-8./63.),
      2*(93./70.),   2*(6./35.),   2*(-8./245.),
      2*(-3./10.),   2*(-1./3.),   2*(1./2.),
      2*(1.),        2*(-1./3.),   2*(-1./5.)
    };

    // -------------------------------------------------------------------------
    // Group 6: IA_mix D_EE, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_mix_DEE[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 2, 2, 0},
    };

    double coeff_mix_DEE[] = {
      2*(-43./540.), 2*(-167./756.), 2*(-19./105.), 2*(1./18.),
      2*(-7./18.),   2*(11./20.),    2*(19./20.),   2*(-19./54.)
    };

    // -------------------------------------------------------------------------
    // Group 7: IA_mix D_BB, remind:  {alpha, beta, l1, l2, l}
    // -------------------------------------------------------------------------
    int terms_mix_DBB[][NCOLS] = {
      {0, 0, 0, 0, 0},
      {0, 0, 2, 0, 0},
      {0, 0, 4, 0, 0},
      {0, 0, 0, 0, 2},
      {0, 0, 2, 0, 2},
      {0, 0, 1, 1, 1},
      {0, 0, 3, 1, 1},
      {0, 0, 2, 2, 0},
    };

    double coeff_mix_DBB[] = {
      2*(13./135.), 2*(86./189.), 2*(16./105.), 2*(2./9.),
      2*(4./9.),    2*(-13./15.), 2*(-4./5.),   2*(8./27.)
    };

    // -------------------------------------------------------------------------
    // Number of input terms (rows) in each group's terms table
    // -------------------------------------------------------------------------
    enum {
      ID_TT_E    = 0,
      ID_TT_B    = 1,
      ID_TA_DE1  = 2,
      ID_TA_0E0E = 3,
      ID_TA_0B0B = 4,
      ID_MIX_A   = 5,
      ID_MIX_DEE = 6,
      ID_MIX_DBB = 7,
      NGROUPS    = 8
    };

    int Nrows[NGROUPS];
    Nrows[ID_TT_E]    = (int)(sizeof(terms_tt_E)    / sizeof(terms_tt_E[0]));
    Nrows[ID_TT_B]    = (int)(sizeof(terms_tt_B)    / sizeof(terms_tt_B[0]));
    Nrows[ID_TA_DE1]  = (int)(sizeof(terms_ta_dE1)  / sizeof(terms_ta_dE1[0]));
    Nrows[ID_TA_0E0E] = (int)(sizeof(terms_ta_0E0E) / sizeof(terms_ta_0E0E[0]));
    Nrows[ID_TA_0B0B] = (int)(sizeof(terms_ta_0B0B) / sizeof(terms_ta_0B0B[0]));
    Nrows[ID_MIX_A]   = (int)(sizeof(terms_mix_A)   / sizeof(terms_mix_A[0]));
    Nrows[ID_MIX_DEE] = (int)(sizeof(terms_mix_DEE) / sizeof(terms_mix_DEE[0]));
    Nrows[ID_MIX_DBB] = (int)(sizeof(terms_mix_DBB) / sizeof(terms_mix_DBB[0]));

    // -------------------------------------------------------------------------
    // Upper bound on J_table output rows per input term. Each term has three
    // angular momentum couplings (J1, J2, Jk) constrained by the triangle
    // inequality: e.g., J1 ranges from |l-l2| to l+l2, giving l+l2-|l-l2|+1
    // allowed values. The product of the three ranges bounds the output count.
    // -------------------------------------------------------------------------
    int Nmax[NGROUPS];
    Nmax[ID_TT_E]    = Nmax_from_terms(Nrows[ID_TT_E],    terms_tt_E);
    Nmax[ID_TT_B]    = Nmax_from_terms(Nrows[ID_TT_B],    terms_tt_B);
    Nmax[ID_TA_DE1]  = Nmax_from_terms(Nrows[ID_TA_DE1],  terms_ta_dE1);
    Nmax[ID_TA_0E0E] = Nmax_from_terms(Nrows[ID_TA_0E0E], terms_ta_0E0E);
    Nmax[ID_TA_0B0B] = Nmax_from_terms(Nrows[ID_TA_0B0B], terms_ta_0B0B);
    Nmax[ID_MIX_A]   = Nmax_from_terms(Nrows[ID_MIX_A],   terms_mix_A);
    Nmax[ID_MIX_DEE] = Nmax_from_terms(Nrows[ID_MIX_DEE], terms_mix_DEE);
    Nmax[ID_MIX_DBB] = Nmax_from_terms(Nrows[ID_MIX_DBB], terms_mix_DBB);

    // -------------------------------------------------------------------------
    // VLAs for J_table outputs
    // -------------------------------------------------------------------------
    int jterms_tt_E[Nmax[ID_TT_E]][NCOLS];
    int jterms_tt_B[Nmax[ID_TT_B]][NCOLS];
    int jterms_ta_dE1[Nmax[ID_TA_DE1]][NCOLS];
    int jterms_ta_0E0E[Nmax[ID_TA_0E0E]][NCOLS];
    int jterms_ta_0B0B[Nmax[ID_TA_0B0B]][NCOLS];
    int jterms_mix_A[Nmax[ID_MIX_A]][NCOLS];
    int jterms_mix_DEE[Nmax[ID_MIX_DEE]][NCOLS];
    int jterms_mix_DBB[Nmax[ID_MIX_DBB]][NCOLS];

    double jcoeff_tt_E[Nmax[ID_TT_E]];
    double jcoeff_tt_B[Nmax[ID_TT_B]];
    double jcoeff_ta_dE1[Nmax[ID_TA_DE1]];
    double jcoeff_ta_0E0E[Nmax[ID_TA_0E0E]];
    double jcoeff_ta_0B0B[Nmax[ID_TA_0B0B]];
    double jcoeff_mix_A[Nmax[ID_MIX_A]];
    double jcoeff_mix_DEE[Nmax[ID_MIX_DEE]];
    double jcoeff_mix_DBB[Nmax[ID_MIX_DBB]];
    
    // -------------------------------------------------------------------------
    // Expand each group's input terms via J_table into (alpha, beta, J1, J2, Jk) rows.
    // Njterms[g] is the actual number of output rows for group g (≤ Nmax[g]).
    // -------------------------------------------------------------------------
    int Njterms[NGROUPS];
    Njterms[ID_TT_E] = J_table(NCOLS, Nrows[ID_TT_E], terms_tt_E, coeff_tt_E, jterms_tt_E, jcoeff_tt_E);
    Njterms[ID_TT_B] = J_table(NCOLS, Nrows[ID_TT_B], terms_tt_B, coeff_tt_B, jterms_tt_B, jcoeff_tt_B);
    Njterms[ID_TA_DE1] = J_table(NCOLS, Nrows[ID_TA_DE1], terms_ta_dE1, coeff_ta_dE1, jterms_ta_dE1, jcoeff_ta_dE1);
    Njterms[ID_TA_0E0E] = J_table(NCOLS, Nrows[ID_TA_0E0E], terms_ta_0E0E, coeff_ta_0E0E, jterms_ta_0E0E, jcoeff_ta_0E0E);
    Njterms[ID_TA_0B0B] = J_table(NCOLS, Nrows[ID_TA_0B0B], terms_ta_0B0B, coeff_ta_0B0B, jterms_ta_0B0B, jcoeff_ta_0B0B);
    Njterms[ID_MIX_A] = J_table(NCOLS, Nrows[ID_MIX_A], terms_mix_A, coeff_mix_A, jterms_mix_A, jcoeff_mix_A);
    Njterms[ID_MIX_DEE] = J_table(NCOLS, Nrows[ID_MIX_DEE], terms_mix_DEE, coeff_mix_DEE, jterms_mix_DEE, jcoeff_mix_DEE);
    Njterms[ID_MIX_DBB] = J_table(NCOLS, Nrows[ID_MIX_DBB], terms_mix_DBB, coeff_mix_DBB, jterms_mix_DBB, jcoeff_mix_DBB);

    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // Concatenate all groups into single arrays
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    // Pointers to the per-group J_table output arrays
    int (*jterms_ptrs[])[NCOLS] = {
      jterms_tt_E,
      jterms_tt_B,
      jterms_ta_dE1,
      jterms_ta_0E0E,
      jterms_ta_0B0B,
      jterms_mix_A,
      jterms_mix_DEE,
      jterms_mix_DBB
    };
    double *jcoeff_ptrs[] = {
      jcoeff_tt_E,
      jcoeff_tt_B,
      jcoeff_ta_dE1,
      jcoeff_ta_0E0E,
      jcoeff_ta_0B0B,
      jcoeff_mix_A,
      jcoeff_mix_DEE,
      jcoeff_mix_DBB
    };

    // -----------------------------------------------------------------------
    // Concatenate all NGROUPS J_table outputs into flat arrays for a single
    // J_abJ1J2Jk_ar call. Each group g contributes Njterms[g] rows starting
    // at offset starts[g]. The flat arrays (alpha_all, beta_all, J1_all,
    // J2_all, Jk_all, coeff_all) are indexed from 0 to Ntotal-1.
    //
    // Note: after J_table expansion, the column meaning changed from the
    // input layout {alpha, beta, l1, l2, l} to {alpha, beta, J1, J2, Jk}.
    // Columns 0-1 (alpha, beta) are copied through unchanged by J_table.
    // Columns 2-4 were overwritten: l1->J1, l2->J2, l->Jk.
    // -----------------------------------------------------------------------
    int Ntotal = 0;
    int starts[NGROUPS];
    for (int g = 0; g < NGROUPS; g++) {
      starts[g] = Ntotal;
      Ntotal += Njterms[g];
    }

    enum {
      JCOL_ID_ALPHA = 0,
      JCOL_ID_BETA  = 1,
      JCOL_ID_J1    = 2,
      JCOL_ID_J2    = 3,
      JCOL_ID_JK    = 4
    };

    int alpha_all[Ntotal];
    int beta_all[Ntotal];
    int J1_all[Ntotal];
    int J2_all[Ntotal];
    int Jk_all[Ntotal];
    double coeff_all[Ntotal];
    for (int g=0; g<NGROUPS; g++) {
      int off = starts[g];
      for (int i = 0; i < Njterms[g]; i++) {
        alpha_all[off+i] = jterms_ptrs[g][i][JCOL_ID_ALPHA];
        beta_all[off+i]  = jterms_ptrs[g][i][JCOL_ID_BETA];
        J1_all[off+i]    = jterms_ptrs[g][i][JCOL_ID_J1];
        J2_all[off+i]    = jterms_ptrs[g][i][JCOL_ID_J2];
        Jk_all[off+i]    = jterms_ptrs[g][i][JCOL_ID_JK];
        coeff_all[off+i] = jcoeff_ptrs[g][i];
      }
    }

    // -----------------------------------------------------------------------
    // Single J_abJ1J2Jk call with all ~184 terms.
    //
    // J_abJ1J2Jk expects double **Fy where Fy[i] points to an array of
    // FPTIA.N doubles for term i's output. We allocate one contiguous block
    // (Fy_flat) and set up an array of pointers (Fy_ptrs) into it:
    //
    //   Fy_flat:  [--- term 0 ---][--- term 1 ---]...[--- term Ntotal-1 ---]
    //              ^               ^                   ^
    //   Fy_ptrs:  [0]             [1]                 [Ntotal-1]
    //
    // After the J_abJ1J2Jk_ar call, Fy_ptrs[i][j] = result for term i at k-point j.
    // -----------------------------------------------------------------------
    double *Fy_flat  = (double*) malloc(sizeof(double) * Ntotal * FPTIA.N);
    double **Fy_ptrs = (double**) malloc(sizeof(double*) * Ntotal);
    for (int i = 0; i < Ntotal; i++) {
      Fy_ptrs[i] = Fy_flat + i * FPTIA.N;
    }

    static const fastpt_config fpt_config = {
      .c_window_width = 0.65, 
      .N_pad = 1500,
      .N_extrap_low = 500, 
      .N_extrap_high = 500
    };

    J_abJ1J2Jk(k,         // input k grid, length N
               Pin,       // input power spectrum P(k)
               FPTIA.N,   // number of input k points (before padding)
               alpha_all, // biasing exponent 1 per term: nu1 = -2 - alpha[i]
               beta_all,  // biasing exponent 2 per term: nu2 = -2 - beta[i]
               J1_all,    // angular momentum coupling 1 per term (indexes g_m cache)
               J2_all,    // angular momentum coupling 2 per term (indexes g_m cache)
               Jk_all,    // angular momentum coupling 3 per term (indexes g_m cache)
               Ntotal,    // number of terms to compute
               &fpt_config, // padding/windowing config (N_pad, c_window_width, etc.)
               Fy_ptrs);  // output: Fy[i][j] = result for term i at k-point j


    // -----------------------------------------------------------------------
    // Accumulate J_abJ1J2Jk results into the 8 FPTIA output arrays.
    //
    // Each group's result is a weighted sum over its terms:
    //   output[j] = sum_i  coeff_all[i] * Fy[i][j]
    //
    // The mapping from group ID to FPTIA.tab index:
    //   ID_TT_E    → FPTIA.tab[0]   (IA_tt E-mode)
    //   ID_TT_B    → FPTIA.tab[1]   (IA_tt B-mode)
    //   ID_TA_DE1  → FPTIA.tab[2]   (IA_ta deltaE1)
    //   ID_TA_0E0E → FPTIA.tab[4]   (IA_ta 0E0E)
    //   ID_TA_0B0B → FPTIA.tab[5]   (IA_ta 0B0B)
    //   ID_MIX_A   → FPTIA.tab[6]   (IA_mix A)
    //   ID_MIX_DEE → FPTIA.tab[8]   (IA_mix D_EE)
    //   ID_MIX_DBB → FPTIA.tab[9]   (IA_mix D_BB)
    //
    // Note: FPTIA.tab[3] (deltaE2) and FPTIA.tab[7] (mix B) are computed
    // separately below via direct convolution, not J_abJ1J2Jk_ar.
    // -----------------------------------------------------------------------
    double *outputs[NGROUPS];
    outputs[ID_TT_E]    = FPTIA.tab[0];
    outputs[ID_TT_B]    = FPTIA.tab[1];
    outputs[ID_TA_DE1]  = FPTIA.tab[2];
    outputs[ID_TA_0E0E] = FPTIA.tab[4];
    outputs[ID_TA_0B0B] = FPTIA.tab[5];
    outputs[ID_MIX_A]   = FPTIA.tab[6];
    outputs[ID_MIX_DEE] = FPTIA.tab[8];
    outputs[ID_MIX_DBB] = FPTIA.tab[9];

    #pragma omp parallel for
    for (int g = 0; g < NGROUPS; g++) {
      // Zero the output array, then accumulate the weighted sum of all
      // terms belonging to this group (indices starts[g] .. starts[g]+Njterms[g]-1)
      memset(outputs[g], 0, sizeof(double) * FPTIA.N);
      for (int i = starts[g]; i < starts[g] + Njterms[g]; i++) {
        const double c = coeff_all[i];              // combined coefficient A*B
        const double *row = Fy_flat + i * FPTIA.N;  // term i's result array
        for (int j = 0; j < FPTIA.N; j++)
          outputs[g][j] += c * row[j];
      }
    }

    free(Fy_ptrs);
    free(Fy_flat);
    
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // Next section is gonna need the following definitions
    const double dL = log(k[1] / k[0]);  // log-spacing of k grid
    const long Ncut = floor(3. / dL);     // transition index from exact to asymptotic
    double exps[2*FPTIA.N-1];
    for (int i = 0; i < 2*FPTIA.N-1; i++) {
      // Precompute r = k'/k ratios for all convolution offsets
      exps[i] = exp(-dL * (i - FPTIA.N + 1));
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // IA_ta deltaE2 term: direct convolution (not via J_abJ1J2Jk).
    //
    // Computes P_deltaE2(k) = 2 * k^3 / (896 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
    //
    // The 1/(896 pi^2) prefactor comes from the angular integration of the delta-E2
    // perturbation theory kernel in the TATT model. See Blazek et al (2019)
    // 
    // The convolution kernel f(r) has three regimes:
    //   r << 1 (far below midpoint): asymptotic expansion in negative powers of r
    //   r ~ 1  (near midpoint):      exact closed-form with log(|r-1|/(r+1)) term
    //   r >> 1 (far above midpoint): asymptotic expansion in positive powers of r
    //
    // The cutoff Ncut = floor(3/dL) determines where to switch between the
    // exact formula and the asymptotic expansions. The exact formula has a
    // log singularity at r=1, so the midpoint f[FPTIA.N-1] is set analytically.
    //
    // r = exp(-dL*(i - FPTIA.N + 1)) maps array index i to the ratio k'/k.
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    {
      double f[2*FPTIA.N-1];
      int i;

      // Region 1: r << 1 (asymptotic expansion for small r)
      for (i = 0; i < FPTIA.N-1-Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2;
        f[i] = r * (768./7 - 256/(7293.*r10) - 256/(3003.*r8)
                - 256/(1001.*r6) - 256/(231.*r4) - 256/(21.*r2));
      }

       // Region 2: r ~ 1, below midpoint (exact closed-form with log term)
      for ( ; i < FPTIA.N-1; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r, r6 = r4*r2, r7 = r6*r;
        f[i] = r * (30. + 146*r2 - 110*r4 + 30*r6
                + log(fabs(r-1.)/(r+1.)) * (15./r - 60.*r + 90*r3 - 60*r5 + 15*r7));
      }

      // Region 3: r ~ 1, above midpoint (same exact formula, mirrored)
      for (i = FPTIA.N; i < FPTIA.N-1+Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r5 = r4*r, r6 = r4*r2, r7 = r6*r;
        f[i] = r * (30. + 146*r2 - 110*r4 + 30*r6
                + log(fabs(r-1.)/(r+1.)) * (15./r - 60.*r + 90*r3 - 60*r5 + 15*r7));
      }

       // Region 4: r >> 1 (asymptotic expansion for large r)
      for ( ; i < 2*FPTIA.N-1; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6, r14 = r8*r6;
        f[i] = r * (256*r2 - 256*r4 + (768*r6)/7.
                - (256*r8)/21. - (256*r10)/231.
                - (256*r12)/1001. - (256*r14)/3003.);
      }

      // Midpoint: r = 1 exactly (analytic limit of the closed-form expression)
      f[FPTIA.N-1] = 96.;

      // Convolve Pin with the kernel f, then extract and normalize
      double g[3*FPTIA.N-2];
      fftconvolve_real(Pin, f, FPTIA.N, 2*FPTIA.N-1, g);
      
      // P_deltaE2(k) = 2 * k^3 / (896 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
      for (i = 0; i < FPTIA.N; i++) {
        double ki3 = k[i] * k[i] * k[i];
        FPTIA.tab[3][i] = 2. * ki3 / (896.*M_PI*M_PI) * Pin[i] * g[FPTIA.N-1+i] * dL;
      }
    }

    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // IA_mix B term: direct convolution (not via J_abJ1J2Jk_ar).
    //
    // Computes P_B(k) = 4 * k^3 / (2 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
    //
    // The factor of 4 absorbs the rescaling that was previously done in a
    // separate loop (FPTIA.tab[7][i] *= 4).
    //
    // Same structure as the deltaE2 kernel but with a different convolution
    // kernel f(r) arising from the IA_mix B-mode perturbation theory integral.
    //
    // The kernel has three regimes:
    //   r << 1 (far below midpoint): asymptotic expansion in negative powers of r
    //   r ~ 1  (near midpoint):      exact closed-form with log(|r-1|/(r+1)) term
    //                                 and (r^2-1)^4 factor from the angular integral
    //   r >> 1 (far above midpoint): asymptotic expansion in positive powers of r
    //
    // The overall /2 factor in each region is part of the kernel normalization.
    //
    // r = exp(-dL*(i - FPTIA.N + 1)) maps array index i to the ratio k'/k.
    // Ncut = floor(3/dL) sets the transition between exact and asymptotic forms.
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    {
      double f[2*FPTIA.N-1];
      int i;

      // Region 1: r << 1 (asymptotic expansion for small r)
      for (i = 0; i < FPTIA.N-1-Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6;
        f[i] = r * (-16./147 - 16/(415701.*r12) - 32/(357357.*r10) - 16/(63063.*r8)
                - 64/(63063.*r6) - 16/(1617.*r4) + 32/(441.*r2)) / 2.;
      }

      // Region 2: r ~ 1, below midpoint (exact closed-form with log term)
      // The (r^2-1)^4 factor arises from the angular integration of the
      // IA_mix kernel; rm1_4 = (r^2 - 1)^4 is precomputed to avoid pow().
      for ( ; i < FPTIA.N-1; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4;
        double rm1 = r2 - 1.;
        double rm1_2 = rm1*rm1, rm1_4 = rm1_2*rm1_2;
        f[i] = r * ((2. * r * (225. - 600.*r2 + 1198.*r4 - 600.*r6 + 225.*r8)
                + 225. * rm1_4 * (r2 + 1.) * log(fabs(r-1.)/(r+1.))) / (20160.*r3)
                - 29./315.*r2) / 2.;
      }

       // Region 3: r ~ 1, above midpoint (same exact formula, mirrored)
      for (i = FPTIA.N; i < FPTIA.N-1+Ncut; i++) {
        double r = exps[i];
        double r2 = r*r, r3 = r2*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4;
        double rm1 = r2 - 1.;
        double rm1_2 = rm1*rm1, rm1_4 = rm1_2*rm1_2;
        f[i] = r * ((2. * r * (225. - 600.*r2 + 1198.*r4 - 600.*r6 + 225.*r8)
                + 225. * rm1_4 * (r2 + 1.) * log(fabs(r-1.)/(r+1.))) / (20160.*r3)
                - 29./315.*r2) / 2.;
      }

      // Region 4: r >> 1 (asymptotic expansion for large r)
      for ( ; i < 2*FPTIA.N-1; i++) {
        double r = exps[i];
        double r2 = r*r, r4 = r2*r2, r6 = r4*r2, r8 = r4*r4, r10 = r8*r2, r12 = r6*r6, r14 = r8*r6, r16 = r8*r8;
        f[i] = r * ((-16*r4)/147. + (32*r6)/441. - (16*r8)/1617.
                - (64*r10)/63063. - (16*r12)/63063. - (32*r14)/357357.
                - (16*r16)/415701.) / 2.;
      }

      // Midpoint: r = 1 exactly (analytic limit of the closed-form expression)
      f[FPTIA.N-1] = -1./42.;

      // Convolve Pin with the kernel f, then extract and normalize
      double g[3*FPTIA.N-2];
      fftconvolve_real(Pin, f, FPTIA.N, 2*FPTIA.N-1, g);
      
      // P_B(k) = 4 * k^3 / (2 * pi^2) * Pin(k) * [Pin ⊛ f](k) * dL
      // The factor of 4 is folded in here (was previously a separate
      // FPTIA.tab[7][i] *= 4 loop after IA_mix).
      for (i = 0; i < FPTIA.N; i++) {
        double ki3 = k[i] * k[i] * k[i];
        FPTIA.tab[7][i] = 4. * ki3 / (2.*M_PI*M_PI) * Pin[i] * g[FPTIA.N-1+i] * dL;
      }
    }

    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}

