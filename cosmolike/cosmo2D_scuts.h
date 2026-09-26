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

// dlnxi_pm/dlnk at one wavenumber for every tomographic pair and angular
// bin: Legendre sums of the batched dC_ss/dlnk over every integer
// multipole, normalized by xi_pm(theta).
double** dlnxi_dlnk_pm_tomo_nointerp(
    const double k    // wavenumber in (Mpc/h)^-1
  ); // returns [2][NSIZE*Ntheta] (caller frees): [0] = xi+, [1] = xi-

// Cached (in ln k) version of the function above; what the RF_xi
// integrals read. A k outside the Ntable.dCX_dlnk grid returns 0.
double dlnxi_dlnk_pm_tomo(
    const double k,   // wavenumber in (Mpc/h)^-1
    const int pm,     // 1 = xi_+, 0 = xi_-
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// Cached dC_ss/dlnk (2011.06469 eq 17) on a (ln k, ln l) table filled by
// dC_ss_dlnk_tomo_limber_work; a (k, l) outside the table returns 0.
double dC_ss_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int EE      // 1 = E-mode, 0 = B-mode
  );

// Cached dlnC_ss/dlnk = (dC_ss/dlnk)/C_ss on the same grid, filled by the
// normalized mode of dC_ss_dlnk_tomo_limber_work.
double dlnC_ss_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int EE      // 1 = E-mode, 0 = B-mode
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// Batch real-space response function RF(kmax, theta) =
// int_{-infty}^{ln kmax} |dlnxi_pm/dlnk| normalized by the full-line
// integral (2011.06469 eq 17), for every tomographic pair and angular
// bin on a ln kmax grid.
void RF_xi_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const int NSIZE,       // number of tomo shear power spectra
    double**** table       // output [2][NSIZE][nkmax][Ntheta]: xi+ and xi-
  );

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