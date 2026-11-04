#ifndef __COSMOLIKE_COSMO2D_H
#define __COSMOLIKE_COSMO2D_H
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
  #define RESTRICT __restrict__
#else
  #define RESTRICT restrict
#endif

// ----------------------------------------------------------------------------
// Naming convention:
// ----------------------------------------------------------------------------
// c = cluster position ("c" as in "cluster")
// g = galaxy positions ("g" as in "galaxy")
// k = kappa CMB ("k" as in "kappa")
// s = kappa from source galaxies ("s" as in "shear")

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Correlation Functions (real Space) - Full Sky - bin average
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ss in real space
double xi_pm_tomo(
    const int pm, 
    const int nt, 
    const int ni, 
    const int nj, 
    const int limber
  );

// gs in real space
double w_gammat_tomo(const int nt, const int ni, const int nj, const int limber);

double w_gg_tomo(const int nt, const int ni, const int nj, const int limber);

double w_gk_tomo(const int nt, const int ni, const int limber);

double w_ks_tomo(const int nt, const int ni, const int limber);

// CMB beam transfer function B_l (Gaussian approximation); zero outside
// the [cmb.lk_wxk[RANGE_MIN], cmb.lk_wxk[RANGE_MAX]] cross-correlation
// multipole range.
double beam_cmb(const int l);

// Precomputed HEALPix pixel window function at multipole l; zero for
// l >= cmb.healpixwin_ncls.
double w_pixel(const int l);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Limber Approximation (Angular Power Spectrum)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double C_ss_tomo_limber(
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  );

double C_gs_tomo_limber(const double l, const int ni, const int nj);

double C_gg_tomo_limber(const double l, const int ni, const int nj);

// runtime switch and reader of the HOD gate (cosmo2D.c include_HOD_GX)
void set_include_HOD_GX(const int flag);

int get_include_HOD_GX(void);

// runtime switch and reader of the halo-model IA gate (include_halo_IA)
void set_include_halo_IA(const int flag);

int get_include_halo_IA(void);

double C_ks_tomo_limber(const double l, const int ni);

double C_gk_tomo_limber(const double l, const int ni);

double C_kk_limber(const double l);

// ----------------------------------------------------------------------------
// Non-Interpolated Version (Will compute the Integral at every call)
// ----------------------------------------------------------------------------

// Point diagnostic on the batch engine: one C_ss_tomo_limber_nointerp_ells
// call at a single multipole, so it pays the whole-tomography batch cost
// per call — never loop it over (l, ni, nj). Kept as the exact
// per-multipole entry point a future non-Limber computation needs.
double C_ss_tomo_limber_nointerp(
    const double l,  // multipole moment
    const int ni,    // first source redshift bin
    const int nj,    // second source redshift bin
    const int EE     // 1 = E-mode, 0 = B-mode
  );

void C_ss_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of tomo shear power spectra
    double** out_EE,     // output EE [NSIZE][nell]
    double** out_BB      // output BB [NSIZE][nell]
  );

// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_ss_tomo_limber_nointerp_ells.
void C_ss_tomo_limber_nointerp_batch(
    const int lmin,
    const int lmax,
    const int NSIZE,
    double*** Cl
  );

// Interpolate ntab log-spaced C_l tables at the integer multipoles
// lmin..lmax-1 with shared index arithmetic and a SIMD (AVX2) gather
// fast path: the workhorse behind every C_XY_tomo_limber_fill. The
// tables hold tab[q][i] at l_i = exp(a + i/inv_dx) with n grid points.
void limber_fill_interp(
    const int ntab,                // number of tables (1 or 2)
    const double** RESTRICT tab,   // input tables [ntab][n]
    double** RESTRICT out,         // output arrays [ntab][>=lmax]
    const int lmin,                // first multipole (inclusive)
    const int lmax,                // last multipole (exclusive)
    const double* RESTRICT ln_ell, // log(l) array, indexed by l
    const double a,                // log(l_min) of the grid
    const double inv_dx,           // 1 / grid spacing in log(l)
    const int n                    // number of grid points
  );

// Legendre sums of the real-space functions, for every spectrum nz and
// theta bin i: w_vec[nz*ntheta + i] = sum_{l=lmin}^{lmax-1} Pl[i][l]*Cl[nz][l].
// Grouped (4 spectra x 4 theta bins per pass over l) so each C_l and
// kernel array is read far fewer times; bitwise equal to the one-sum-
// per-pass scalar reference. Details in cosmo2D.c.
// Call outside parallel regions.
void legendre_sums(
    const int NSIZE,   // number of spectra (rows of Cl)
    const int ntheta,  // number of theta bins (rows of Pl)
    const int lmin,    // first multipole of the sums
    const int lmax,    // one past the last multipole of the sums
    double** Pl,       // [ntheta][lmax] bin-averaged Legendre kernel
    double** Cl,       // [NSIZE][lmax] C_l at every integer l
    double* w_vec      // output [NSIZE*ntheta], indexed nz*ntheta + i
  );

// The xi+- version (2 pairs x 4 theta bins per pass over l):
//   xip[nz*ntheta + i] = sum_l Glp[i][l]*(Cl_EE[nz][l] + Cl_BB[nz][l])
//   xim[nz*ntheta + i] = sum_l Glm[i][l]*(Cl_EE[nz][l] - Cl_BB[nz][l])
// Call outside parallel regions.
void legendre_sums_xipm(
    const int NSIZE,   // number of shear pairs (rows of Cl_EE, Cl_BB)
    const int ntheta,  // number of theta bins (rows of Glp, Glm)
    const int lmin,    // first multipole of the sums
    const int lmax,    // one past the last multipole of the sums
    double** Glp,      // [ntheta][lmax] bin-averaged kernel of xi_+
    double** Glm,      // [ntheta][lmax] bin-averaged kernel of xi_-
    double** Cl_EE,    // [NSIZE][lmax] E-mode C_l at every integer l
    double** Cl_BB,    // [NSIZE][lmax] B-mode C_l at every integer l
    double* xip,       // output [NSIZE*ntheta], indexed nz*ntheta + i
    double* xim        // output [NSIZE*ntheta], indexed nz*ntheta + i
  );

// Batch computation of the scale-cut derivative dC_ss/dlnk (2011.06469
// eq 17) on a (ln k, ell) grid. Each (k, ell) maps onto the single Limber
// node chi(a) = (l + 1/2)/k; nodes outside the source support return 0.
// With normalize = 1 the output is instead dlnC_ss/dlnk = (dC/dlnk)/C_ss,
// with C_ss computed inside on the same thread team and the division
// fused into the fill loop (entries where either factor vanishes are 0);
// normalize = 0 gives the raw dC the real-space dlnxi machinery needs.
void dC_ss_dlnk_tomo_limber_work(
    const double* lnkx,  // ln k grid values (length nlnk), k in (Mpc/h)^-1
    const int nlnk,      // number of ln k grid values
    const double* lx,    // multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of tomo shear power spectra
    const int normalize, // 1: write dlnC = dC/C_ss; 0: write dC
    double**** table     // output [2][NSIZE][nlnk][nell]: EE and BB
  );

double C_gs_tomo_limber_nointerp(
    const double l,   // multipole moment
    const int nl,     // lens redshift bin index
    const int ns      // source bin index; (nl, ns) must be an enumerated pair
  ); // slow (whole-tomography batch per call) - use the batch version

void C_gs_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of ggl power spectra
    double** out         // output [NSIZE][nell]
  );

// use_linear_ps = 1: the linear Limber term C_gs_tomo subtracts
// ((D(a)/D(a_piv))^2 P_lin(k, a_piv), the per-lens-bin pivot of the
// FKEM split, no one-loop bias, IA through C1 only); 0: the full
// model.
void C_gs_tomo_limber_linpsopt_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of ggl power spectra
    const int use_linear_ps, // 1 = P_lin + linear kernels, 0 = full model
    double** out             // output [NSIZE][nell]
  );

// Both at once (either output may be NULL): one pass shares the nodes,
// the weights and the RSD kernel; bitwise two separate calls.
void C_gs_tomo_limber_nl_lin_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of ggl power spectra
    double** out,            // output [NSIZE][nell], full model (or NULL)
    double** out_lin         // output [NSIZE][nell], linear term (or NULL)
  );

// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_gs_tomo_limber_nointerp_ells.
void C_gs_tomo_limber_nointerp_batch(
    const int lmin,
    const int lmax,
    const int NSIZE,
    double** Cl
  );

// Single-ell point diagnostics on the gg batch engine; nj must equal ni
// (auto spectra only). Slow: whole-tomography batch cost per call.
double C_gg_tomo_limber_linpsopt_nointerp(const double l, const int ni,
  const int nj, const int use_linear_ps);

double C_gg_tomo_limber_nointerp(const double l, const int ni, const int nj);

// Batch galaxy-clustering C_l (auto spectra) at arbitrary multipole values.
// use_linear_ps = 1: the linear Limber term C_cl_tomo subtracts; 0: the
// full model.
void C_gg_tomo_limber_linpsopt_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of gg power spectra (= clustering_nbin)
    const int use_linear_ps, // 1 = P_lin, no one-loop bias; 0 = full model
    double** out             // output [NSIZE][nell]
  );

// Both at once (either output may be NULL): one pass shares the nodes,
// the radial weights and the RSD kernel; bitwise two separate calls.
void C_gg_tomo_limber_nl_lin_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of gg power spectra (= clustering_nbin)
    double** out,            // output [NSIZE][nell], full model (or NULL)
    double** out_lin         // output [NSIZE][nell], linear term (or NULL)
  );

void C_gg_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of gg power spectra (= clustering_nbin)
    double** out         // output [NSIZE][nell]
  );

// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_gg_tomo_limber_nointerp_ells.
void C_gg_tomo_limber_nointerp_batch(
    const int lmin,   // first multipole (inclusive)
    const int lmax,   // last multipole (exclusive)
    const int NSIZE,  // number of gg power spectra (= clustering_nbin)
    double** Cl       // output [NSIZE][>=lmax], indexed as Cl[nz][l]
  );

double C_gk_tomo_limber_nointerp(
    const double l,   // multipole moment
    const int ni      // lens redshift bin index
  ); // slow (whole-tomography batch per call) - use the batch version

void C_gk_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of lens tomographic bins (= clustering_nbin)
    double** out         // output [NSIZE][nell], indexed as out[nz][i]
  );

// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_gk_tomo_limber_nointerp_ells.
void C_gk_tomo_limber_nointerp_batch(
    const int lmin,
    const int lmax,
    const int NSIZE,
    double** Cl
  );

// CMB-lensing x shear C_l (the CMB is a single source plane, so one
// spectrum per source bin): the single-multipole point diagnostic, then
// the batch at arbitrary multipole values.
double C_ks_tomo_limber_nointerp(
    const double l,   // multipole moment
    const int ns      // source redshift bin index
  ); // slow (whole-tomography batch per call) - use the batch version

void C_ks_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of source tomographic bins (= shear_nbin)
    double** out         // output [NSIZE][nell], indexed as out[nz][i]
  );

// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_ks_tomo_limber_nointerp_ells.
void C_ks_tomo_limber_nointerp_batch(
    const int lmin,   // first multipole (inclusive)
    const int lmax,   // last multipole (exclusive)
    const int NSIZE,  // number of source tomographic bins (= shear_nbin)
    double** Cl       // output [NSIZE][>=lmax], indexed as Cl[nz][l]
  );

// Batch computation of the scale-cut derivative dC_ks/dlnk (2011.06469
// eq 17) on a (ln k, ell) grid. Each (k, ell) maps onto the single Limber
// node chi(a) = (l + 1/2)/k; a node outside a bin's source support gives
// 0 for that bin (the support differs per source bin, unlike ss).
// With normalize = 1 the output is instead dlnC_ks/dlnk = (dC/dlnk)/C_ks,
// with C_ks computed inside on the same thread team and the division
// fused into the fill loop (entries where either factor vanishes are 0);
// normalize = 0 gives the raw dC the real-space dlnw_ks machinery needs.
void dC_ks_dlnk_tomo_limber_work(
    const double* lnkx,  // ln k grid values (length nlnk), k in (Mpc/h)^-1
    const int nlnk,      // number of ln k grid values
    const double* lx,    // multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of source tomographic bins (= shear_nbin)
    const int normalize, // 1: write dlnC = dC/C_ks; 0: write dC
    double*** table      // output [NSIZE][nlnk][nell]
  );

double C_kk_limber_nointerp(const double l, const int init);

// ----------------------------------------------------------------------------
// Integrands 
// ----------------------------------------------------------------------------

double int_for_C_kk_limber(double a, void* params);

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Non Limber (Angular Power Spectrum)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

void C_cl_tomo(
    double* const* const Cl,
    double tol
  );

// Fourier-space data vectors: Limber C_gg at arbitrary multipoles plus the
// non-Limber correction of C_cl_tomo, interpolated between integers.
void C_gg_tomo_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of gg power spectra (= clustering_nbin)
    double** out         // output [NSIZE][nell]
  );

// Non-Limber galaxy-galaxy lensing below limits.LMAX_NOLIMBER, per
// lens-source pair; tol <= 0 disables the per-pair early exit.
void C_gs_tomo(
    double* const* const Cl, // output [ggl_Npowerspectra][>= LMAX_NOLIMBER]
    double tol               // Limber-convergence tolerance (typically 0.01)
  );

// Fourier-space data vectors: Limber C_gs at arbitrary multipoles plus the
// non-Limber correction, interpolated between integers, below LMAX_NOLIMBER.
void C_gs_tomo_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of ggl power spectra
    double** out         // output [NSIZE][nell]
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Non-Limber FFTLog machinery (C only), shared by the galaxy spectra of
// cosmo2D.c (C_cl_tomo, C_gs_tomo) and the cluster spectra of
// cosmo2D_cluster.c (w_cc, w_cg). The two phases, their memory layout and
// cache rules are documented at the definitions in cosmo2D.c. A C caller
// includes <complex.h> before this header, as cosmo2D.c does, so
// fftw_complex is the same type in every translation unit.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#ifndef __cplusplus
#include <fftw3.h>

// ---------------------------------------------------------------------------
// Configuration for one radial component in the FFTLog non-Limber calculation.
//
// Each radial component (galaxy density, RSD velocity, magnification) has
// its own config controlling the FFTLog bias parameter, windowing, and
// whether the transform computes C_l or its derivatives.
//
// Typical values used in C_cl_tomo and C_gs_tomo:
//   Galaxy density: nu=1.0,  c_window_width=0.25, derivative=0, N_pad=200
//   RSD velocity:   nu=1.01, c_window_width=0.25, derivative=2, N_pad=500
//   Magnification:  nu=1.0,  c_window_width=0.25, derivative=0, N_pad=500
// ---------------------------------------------------------------------------
typedef struct cfftlog_config
{
  double nu;              // bias parameter: input f(x) is divided by x^nu before FFT
                          // (must be > 0; nu=1 is the standard Hankel transform;
                          // slight offsets like 1.01 improve convergence for RSD)
  double c_window_width;  // fractional width of the Fourier-space tapering window
                          // (0 < c_window_width < 1; typically 0.25; controls how
                          // aggressively high-frequency ringing is suppressed)
  int derivative;         // order of the spherical Bessel derivative in the kernel:
                          //   0: j_l(kr)   - standard projection (density, magnification)
                          //   1: j_l'(kr)  - first derivative
                          //   2: j_l''(kr) - second derivative (RSD, which involves
                          //                  the second derivative of the velocity field)
  long N_pad;             // number of zero-padding elements added on each side of the
                          // input array before FFT (larger N_pad reduces edge effects
                          // but increases FFT cost; 200 for density, 500 for RSD/mag)
  long N_extrap_low;      // number of points for low-x power-law extrapolation (unused)
  long N_extrap_high;     // number of points for high-x power-law extrapolation (unused)
} cfftlog_config;

// Phase 1: ell-independent forward FFT of the radial functions
void cfftlog_ells_p1(
    double* const x,                        // chi grid, length Nx (log-spaced)
    double* const* const* const fx,         // input fx[SIZE1][SIZE2][Nx]
    int const Nx,                           // number of chi grid points
    cfftlog_config* const cfg,              // per component [SIZE2]
    fftw_complex* const* const toutfwd,     // out [SIZE1*SIZE2][Nmax/2+1]
    double* const* const eta_m,             // out [SIZE2][Nmax/2+1]
    int N[][3],                             // N_pad, Nx, FFT size per component
    int const Nmax,                         // max FFT size
    const int* const* const active,         // [SIZE1][SIZE2] or NULL (all)
    int const SIZE1,                        // radial rows
    int const SIZE2                         // radial components per row
  );

// Phase 2: ell-dependent inverse transforms of the multipoles [ks, ke)
void cfftlog_ells_p2(
    double* const x,                        // chi grid from phase 1
    int const Nx,                           // number of chi grid points
    cfftlog_config* const cfg,              // per component [SIZE2]
    int const LMAX,                         // multipole bound of y and Fy
    double* const* const* const y,          // out y[SIZE1][LMAX][Nx]
    double* const* const* const* const Fy,  // out Fy[SIZE1][SIZE2][LMAX][Nx]
    fftw_complex* const* const toutfwd,     // phase-1 coefficients
    double* const* const eta_m,             // phase-1 frequencies
    int N[][3],                             // as in phase 1
    int const Nmax,                         // as in phase 1
    int const ks,                           // first multipole (inclusive)
    int const ke,                           // last multipole (exclusive)
    const int* const converged,             // [SIZE1]; 1 = skip the row
    const int* const* const active,         // [SIZE1][SIZE2] or NULL (all)
    int const SIZE1,                        // radial rows
    int const SIZE2                         // radial components per row
  );
#endif // __cplusplus

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
