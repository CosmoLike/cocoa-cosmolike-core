#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfastpt/cfastpt.h"
#include "basics.h"
#include "cosmo3D.h"
#include "pt_cfastpt.h"
#include "structs.h"

#include "log.c/src/log.h"

void get_FPT_bias(void) 
{
  static double cache[MAX_SIZE_ARRAYS];

  if (fdiff(cache[1], Ntable.random))
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
  if (fdiff(cache[0], cosmology.random) || fdiff(cache[1], Ntable.random))
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

void get_FPT_IA(void) 
{
  static double cache[MAX_SIZE_ARRAYS];

  if (fdiff(cache[1], Ntable.random))
  {
    FPTIA.k_min    = 0.05;
    FPTIA.k_max    = 1.0e+6;
    FPTIA.k_cutoff = 1.0e+4;
    FPTIA.sigma4   = 0.0; // Not relevant for IA, but set to zero.
    FPTIA.N        = 1100 + 200 * Ntable.FPTboost;

    if (FPTIA.tab != NULL) {
      free(FPTIA.tab);
    }
    FPTIA.tab = (double**) malloc2d(12, FPTIA.N);
  }
  if (fdiff(cache[0], cosmology.random) || fdiff(cache[1], Ntable.random))
  {
    double lim[3];
    lim[0] = log(FPTIA.k_min);
    lim[1] = log(FPTIA.k_max);
    lim[2] = (lim[1] - lim[0])/FPTIA.N;
    
    #pragma omp parallel for
    for (int i=0; i<FPTIA.N; i++) 
    {
      FPTIA.tab[10][i] = exp(lim[0] + i*lim[2]);
      FPTIA.tab[11][i] = p_lin(FPTIA.tab[10][i], 1.0);
    }

    IA_tt(FPTIA.tab[10], FPTIA.tab[11], FPTIA.N, FPTIA.tab[0], FPTIA.tab[1]);
    
    IA_ta(FPTIA.tab[10], FPTIA.tab[11], FPTIA.N, FPTIA.tab[2], 
      FPTIA.tab[3], FPTIA.tab[4], FPTIA.tab[5]);
    
    IA_mix(FPTIA.tab[10], FPTIA.tab[11], FPTIA.N, FPTIA.tab[6], 
      FPTIA.tab[7], FPTIA.tab[8], FPTIA.tab[9]);
    
    #pragma omp parallel for
    for (int i=0; i<FPTIA.N; i++) {
      FPTIA.tab[7][i] *= 4.;
    }
    
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
}
