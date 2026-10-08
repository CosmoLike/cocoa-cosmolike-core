// gsl lib
#include <gsl/gsl_interp2d.h>

#ifndef __COSMOLIKE_PARAMETERS_BARYONS_H
#define __COSMOLIKE_PARAMETERS_BARYONS_H
#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// The loaded baryonic-feedback scenario: the ratio
//
//   S(k, a) = P_hydro(k, a) / P_DMO(k, a)
//
// of a hydrodynamical simulation to its dark-matter-only twin, on a
// (log10 k, a) grid (scenario list and table conventions in baryons.c).
// PkRatio_baryons (cosmo3D.c) interpolates it, and p_nonlin multiplies the
// nonlinear matter power spectrum by the result.
//
//   is_Pk_bary - 1 once init_baryons or init_baryons_from_hdf5_file has
//                loaded a scenario; 0 after reset_bary_struct (no baryonic
//                contamination: PkRatio_baryons returns 1)
//   Na_bins    - number of scale-factor (snapshot) nodes
//   Nk_bins    - number of wavenumber nodes
//   a_bins     - a = 1/(1+z) of the snapshots, strictly increasing
//   logk_bins  - log10(k / (h/Mpc)), strictly increasing
//   log_PkR    - log10 S in gsl_interp2d layout, k index fastest:
//                log_PkR[ia*Nk_bins + ik] = log10 S at (logk_bins[ik],
//                a_bins[ia])
//   T          - interpolation type, gsl_interp2d_bilinear (a static GSL
//                object, never freed)
//   interp2d   - GSL interpolator over (logk_bins, a_bins, log_PkR)
// ---------------------------------------------------------------------------
typedef struct
{
  int is_Pk_bary;
  int Na_bins;
  int Nk_bins;
  double* a_bins;
  double* logk_bins;
  double* log_PkR;
  gsl_interp2d_type* T;
  gsl_interp2d* interp2d;
} barypara;

extern barypara bary;

// unload the scenario: free the tables and set is_Pk_bary = 0
void reset_bary_struct(void);

// load a scenario compiled into baryons.c by its label "name-tag"
// (e.g. "TNG100-1", "owls_AGN-2")
void init_baryons(const char* sim);

// load scenario tag of simulation group sim (e.g. "BAHAMAS", 2) from the
// HDF5 library file allsims
void init_baryons_from_hdf5_file(const char* sim, int tag, const char* allsims);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
