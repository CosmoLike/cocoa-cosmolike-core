#ifndef __COSMOLIKE_COSMO2D_SCUTS_H
#define __COSMOLIKE_COSMO2D_SCUTS_H
#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// Naming convention:
// ----------------------------------------------------------------------------
// c = cluster position ("c" as in "cluster")
// g = galaxy positions ("g" as in "galaxy")
// k = kappa CMB ("k" as in "kappa")
// s = kappa from source galaxies ("s" as in "shear")

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: important to determine scale cuts (2011.06469 eq 17)
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

double** dlnxi_dlnk_pm_tomo_nointerp(const double k);

double dlnxi_dlnk_pm_tomo(
    const double k,
    const int pm, 
    const int nt, 
    const int ni, 
    const int nj
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double dC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  );

double dlnC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double RF_xi_tomo_limber_nointerp(
    const double kmax,
    const int pm, 
    const int nt,
    const int ni, 
    const int nj, 
    const int init
  ); // compute RF_X = \int_{-infty}^{kmax} dlnk |dlnX_dlnk|

// Batch response function RF(kmax, l) = int_{-infty}^{ln kmax} |dlnC/dlnk|
// normalized by the full-line integral (2011.06469 eq 17), for every
// tomographic pair on a (ln kmax, ell) grid.
void RF_C_ss_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE,       // number of tomo shear power spectra
    double**** table       // output [2][NSIZE][nkmax][nl]: EE and BB
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD